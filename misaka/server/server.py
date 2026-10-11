"""The experimental server: handshake, request routing and shutdown.

Translated from pi's ``packages/server/src/server.ts``. ``start()`` and ``close()`` return futures and do their
synchronous part when called, as a JavaScript async method does; ``void asyncMethod()`` is ``spawn`` (a task that
starts running at once and whose failure is observed). ``undefined`` is ``misaka.utils.js.UNDEFINED``.
"""

from __future__ import annotations

import asyncio
from typing import Any, Self

from misaka.chord.context import BACKGROUND_CONTEXT, TODO_CONTEXT, withAbortSignal
from misaka.chord.services.errors import RemoteServiceError
from misaka.chord.services.state_codec import createServiceStateEncoder
from misaka.chord.services.wire import (
    decodeServiceControlCall,
    parseServiceCall,
    parseServiceSubscriptionSnapshot,
)
from misaka.chord.types import Context
from misaka.protocol import (
    DEFAULT_MAX_FRAME_LENGTH,
    PROTOCOL_VERSION,
    CancelEnvelope,
    ClientHello,
    ClientMessage,
    ClientMessageDecoder,
    ProtocolError,
    ProtocolValidationError,
    RequestEnvelope,
    RpcTarget,
    ServerMessage,
    encodeServerMessage,
    isServerId,
    isSupportedProtocolVersion,
)
from misaka.utils.js import (
    UNDEFINED,
    AbortController,
    DOMException,
    aggregateError,
    allSettled,
    awaitValue,
    jsIsInteger,
    promiseAll,
    rejectedPromise,
    spawn,
    startTask,
)

from .connection import (
    ActiveRequest,
    ByteConnection,
    ByteConnectionHandler,
    ConnectionState,
    isTerminalConnection,
)
from .errors import INTERNAL_SERVER_ERROR_MESSAGE, ServerError, WrongServerError
from .listener import ServerListener
from .session_router import SessionRouter
from .types import ServerHost, ServerOptions

DEFAULT_HANDSHAKE_TIMEOUT_MS = 5_000
MAX_UINT32 = 0xFFFF_FFFF
MAX_TIMER_DELAY_MS = 2_147_483_647


class _Handler:
    """The handler an acceptor returns for one connection."""

    def __init__(self, server: Server, state: ConnectionState, connection: ByteConnection) -> None:
        self._server = server
        self._state = state
        self._connection = connection

    def onData(self, chunk: bytes) -> None:
        self._server._receive(self._state, chunk)

    def onClose(self) -> None:
        self._server._transportClosed(self._state)

    def onError(self, error: Exception) -> None:
        self._server._reportError(error)
        spawn(self._server._closeThenDisconnect(self._connection, self._state), self._server._background)


class _DrainingHandler:
    """The handler for a connection accepted while the server is closing."""

    def __init__(self, server: Server) -> None:
        self._server = server

    def onData(self, chunk: bytes) -> None:
        pass

    def onClose(self) -> None:
        pass

    def onError(self, error: Exception) -> None:
        self._server._reportError(error)


class _Presentation:
    """The presentation-scoped routing capabilities one connection's server services receive."""

    def __init__(self, server: Server, state: ConnectionState) -> None:
        self._server = server
        self._state = state

    async def attachSession(self, sessionId: str, context: Context) -> None:
        await self._server._sessions.attachClient(self._state, sessionId, context)

    def detachSession(self, context: Context) -> asyncio.Future[None]:
        return self._server._sessions.detachClient(self._state, context)

    def prepareSessionRemoval(self, sessionId: str, context: Context) -> asyncio.Future[None]:
        return startTask(self._server._sessions.removeSession(sessionId, context))


class Server:
    """``serverId`` is the stable logical identity; ``closed`` settles after shutdown, or fails when listener or
    routed-Session cleanup fails. Create a server inside a running event loop."""

    def __init__(self, host: ServerHost, options: ServerOptions) -> None:
        resolved = resolveOptions(options)
        self.host = host
        self.serverId: str = options["serverId"]
        self._listeners = options["listeners"]
        self._maxFrameLength: int = resolved["maxFrameLength"]
        self._handshakeTimeoutMs: int = resolved["handshakeTimeoutMs"]
        self._onConnectionCountChanged = options.get("onConnectionCountChanged")
        self._onError = options.get("onError")
        self._connections: set[ConnectionState] = set()
        self._closing = False
        self._closePromise: asyncio.Future[None] | None = None
        self._closedSettled = False
        self._startPromise: asyncio.Future[Self] | None = None
        self._started = False
        self._background: set[asyncio.Task[Any]] = set()
        self._sessions = SessionRouter(
            {
                "host": host,
                "serverId": self.serverId,
                "isClosing": lambda: self._closing,
                "publishAttachment": self._publishAttachment,
                "reportError": self._reportError,
            }
        )
        self._closed: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._closed.add_done_callback(_observe)

    @property
    def closed(self) -> asyncio.Future[None]:
        # MISAKA fork: a promise cannot be cancelled by whoever waits on it; a future can, so each reader gets its own.
        return asyncio.shield(self._closed)

    async def _publishAttachment(self, client: object, attachment: Any, context: Context) -> None:
        await self._sendMessage(client, {"type": "attachment", "attachment": attachment})  # type: ignore[arg-type]

    def start(self) -> asyncio.Future[Self]:
        if self._started:
            return rejectedPromise(RuntimeError("Server is already started"))
        if self._startPromise is not None:
            return rejectedPromise(RuntimeError("Server is already starting"))
        if self._closing:
            return rejectedPromise(RuntimeError("Server is closing or closed"))
        self._startPromise = startTask(self._startInternal())
        # MISAKA fork: shielded, because a caller that gives up waiting would otherwise abort the startup; a promise
        # cannot be cancelled by whoever waits on it.
        return asyncio.shield(self._startPromise)

    async def _startInternal(self) -> Self:
        started: list[ServerListener] = []
        try:
            for listener in self._listeners:
                await awaitValue(listener.start(self.accept))
                started.append(listener)
            self._started = True
            return self
        except Exception as error:
            self._closing = True
            cleanupErrors: list[Any] = []
            listenerResults = await allSettled([listener.close() for listener in started])
            for result in listenerResults:
                if result.status == "rejected":
                    cleanupErrors.append(result.reason)
            try:
                await self._closeServerState()
            except Exception as cleanupError:  # noqa: BLE001
                cleanupErrors.append(cleanupError)
            if cleanupErrors:
                failure = aggregateError([error, *cleanupErrors], "Server startup and cleanup failed")
                self._settleClosed(failure)
                raise failure from None
            self._settleClosed()
            raise
        finally:
            self._startPromise = None

    def accept(self, connection: ByteConnection) -> ByteConnectionHandler:
        if self._closing:
            spawn(self._closeConnection(connection), self._background)
            return _DrainingHandler(self)

        state = ConnectionState(connection, ClientMessageDecoder({"maxFrameLength": self._maxFrameLength}))

        def onHandshakeTimeout() -> None:
            spawn(self._failProtocol(state, {"code": "invalid_request", "message": "Handshake timeout"}), self._background)

        state.handshakeTimeout = asyncio.get_running_loop().call_later(self._handshakeTimeoutMs / 1000, onHandshakeTimeout)
        self._connections.add(state)
        self._notifyConnectionCountChanged()
        return _Handler(self, state, connection)

    def close(self) -> asyncio.Future[None]:
        if self._closePromise is None:
            self._closing = True
            self._closePromise = startTask(self._closeInternal())
        # MISAKA fork: shielded, because a caller that gives up waiting would otherwise abort the shutdown; a promise
        # cannot be cancelled by whoever waits on it.
        return asyncio.shield(self._closePromise)

    async def _closeInternal(self) -> None:
        starting = self._startPromise
        if starting is not None:
            try:
                await asyncio.shield(starting)
            except Exception:  # noqa: BLE001, S110
                pass
        errors: list[Any] = []
        listenerResults = await allSettled([listener.close() for listener in self._listeners])
        for result in listenerResults:
            if result.status == "rejected":
                errors.append(result.reason)
        try:
            await self._closeServerState()
        except Exception as error:  # noqa: BLE001
            errors.append(error)
        self._started = False
        if errors:
            failure = errors[0] if len(errors) == 1 and isinstance(errors[0], Exception) else aggregateError(errors, "Server shutdown failed")
            self._settleClosed(failure)
            raise failure
        self._settleClosed()

    def _receive(self, state: ConnectionState, chunk: bytes) -> None:
        if isTerminalConnection(state):
            return
        try:
            messages = state.decoder.push(chunk)
        except Exception as error:  # noqa: BLE001
            spawn(self._failProtocol(state, self._toProtocolError(error)), self._background)
            return
        for message in messages:
            if isTerminalConnection(state):
                return
            self._dispatchMessage(state, message)

    def _dispatchMessage(self, state: ConnectionState, message: ClientMessage) -> None:
        if state.stage == "awaitingHello":
            if message["type"] != "hello":
                spawn(
                    self._failProtocol(state, {"code": "invalid_request", "message": "The first client message must be hello"}),
                    self._background,
                )
                return
            state.stage = "handshaking"
            state.handshake = startTask(self._handshake(state, message))
            return

        if message["type"] == "hello":
            spawn(
                self._failProtocol(state, {"code": "invalid_request", "message": "hello may only be sent as the first message"}),
                self._background,
            )
            return

        if state.stage == "ready":
            if message["type"] == "cancel":
                self._handleCancel(state, message)
            else:
                spawn(self._handleRequest(state, message), self._background)  # type: ignore[arg-type]
            return
        if state.stage != "handshaking":
            return
        handshake = state.handshake
        if handshake is None:
            return
        spawn(self._afterHandshake(state, handshake, message), self._background)

    async def _handshake(self, state: ConnectionState, hello: ClientHello) -> None:
        try:
            await self._finishHandshake(state, hello)
        except Exception as error:  # noqa: BLE001
            await self._failProtocol(state, self._toProtocolError(error))

    async def _afterHandshake(self, state: ConnectionState, handshake: asyncio.Future[None], message: ClientMessage) -> None:
        await handshake
        if state.stage != "ready" or state.disconnected:
            return
        if message["type"] == "cancel":
            self._handleCancel(state, message)
        else:
            spawn(self._handleRequest(state, message), self._background)  # type: ignore[arg-type]

    async def _finishHandshake(self, state: ConnectionState, hello: ClientHello) -> None:
        if not isSupportedProtocolVersion(hello["version"]):
            await self._failProtocol(
                state,
                {
                    "code": "version",
                    "message": f"Unsupported protocol version {hello['version']}; expected {PROTOCOL_VERSION}",
                },
            )
            return

        if self._closing or state.disconnected or state.stage != "handshaking" or state.connection.closed:
            return
        services = await awaitValue(self.host.serverServices.attachClient(_Presentation(self, state), TODO_CONTEXT))
        if self._closing or state.disconnected or state.stage != "handshaking" or state.connection.closed:
            await awaitValue(services.release(TODO_CONTEXT))
            return
        state.serverServices = services
        sent = await self._sendMessage(state, {"type": "hello", "version": PROTOCOL_VERSION, "serverId": self.serverId})
        if sent and not state.disconnected and state.stage == "handshaking":
            state.stage = "ready"
            _clearTimeout(state)

    def _handleCancel(self, state: ConnectionState, envelope: CancelEnvelope) -> None:
        if envelope["target"]["serverId"] != self.serverId:
            return
        active = state.activeRequests.get(envelope["id"])
        if active is not None and sameTarget(active.target, envelope["target"]):
            active.controller.abort(DOMException("RPC request cancelled", "AbortError"))

    async def _handleRequest(self, state: ConnectionState, envelope: RequestEnvelope) -> None:
        requestId = envelope["id"]
        if requestId in state.activeRequests:
            await self._sendMessage(
                state,
                {
                    "type": "response",
                    "id": requestId,
                    "ok": False,
                    "error": {"code": "invalid_request", "message": "Request ID is already active"},
                },
            )
            return
        try:
            call = parseServiceCall(envelope["call"])
        except Exception:  # noqa: BLE001
            await self._sendMessage(
                state,
                {
                    "type": "response",
                    "id": requestId,
                    "ok": False,
                    "error": {"code": "invalid_request", "message": "Invalid service call"},
                },
            )
            return
        controller = AbortController()
        active = ActiveRequest(controller, envelope["target"])
        state.activeRequests[requestId] = active
        context = withAbortSignal(controller.signal, TODO_CONTEXT)
        control = decodeServiceControlCall(call)
        subscribing = control if control is not None and control["type"] == "subscribe" else None
        pendingUpdates: list[Any] = []
        subscriptionReady = subscribing is None
        installedSubscriptionEncoder = False
        responded = False

        async def publish(subscriptionId: str, update: Any, _context: Any = None) -> None:
            if subscribing is not None and subscriptionId == subscribing["subscriptionId"] and not subscriptionReady:
                pendingUpdates.append(update)
                return
            await self._sendServiceUpdate(state, subscriptionId, update)

        try:
            if envelope["target"]["serverId"] != self.serverId:
                raise WrongServerError
            if subscribing is not None and subscribing["subscriptionId"] in state.serviceStateEncoders:
                raise ProtocolValidationError(f"Duplicate service subscription {subscribing['subscriptionId']}")
            result: Any
            if "sessionId" in envelope["target"]:
                result = await self._sessions.executeServiceCall(call, envelope["target"], state, publish, context)
            elif state.serverServices is not None:
                result = await awaitValue(state.serverServices.invokeService(call, publish, context))
            else:
                raise ProtocolValidationError(f"Unknown service member {call['serviceId']}.{call['member']}")
            if subscribing is not None:
                if result is UNDEFINED:
                    raise ProtocolValidationError("Service subscription did not return a snapshot")
                stateEncoder = createServiceStateEncoder()
                result = stateEncoder.encodeSnapshot(parseServiceSubscriptionSnapshot(result))
                state.serviceStateEncoders[subscribing["subscriptionId"]] = stateEncoder
                installedSubscriptionEncoder = True
            elif control is not None and control["type"] == "unsubscribe":
                state.serviceStateEncoders.pop(control["subscriptionId"], None)
            await self._sendMessage(
                state,
                {"type": "response", "id": requestId, "ok": True}
                if result is UNDEFINED
                else {"type": "response", "id": requestId, "ok": True, "result": result},
            )
            responded = True
            if subscribing is not None:
                while pendingUpdates:
                    pending = pendingUpdates.pop(0)
                    await self._sendServiceUpdate(state, subscribing["subscriptionId"], pending)
                subscriptionReady = True
        except Exception as error:  # noqa: BLE001
            if subscribing is not None and installedSubscriptionEncoder and not responded:
                state.serviceStateEncoders.pop(subscribing["subscriptionId"], None)
            if responded:
                self._reportError(error)
                await self._closeConnection(state.connection)
                self._disconnect(state)
            else:
                await self._sendMessage(
                    state,
                    {
                        "type": "response",
                        "id": requestId,
                        "ok": False,
                        "error": (
                            {"code": "cancelled", "message": "RPC request cancelled"}
                            if controller.signal.aborted
                            else self._toProtocolError(error)
                        ),
                    },
                )
        finally:
            if state.activeRequests.get(requestId) is active:
                del state.activeRequests[requestId]

    def _transportClosed(self, connection: ConnectionState) -> None:
        if not connection.disconnected and connection.stage != "closing":
            try:
                connection.decoder.end()
            except Exception as error:  # noqa: BLE001
                self._reportError(error)
        self._disconnect(connection)

    def _disconnect(self, connection: ConnectionState) -> None:
        if connection.disconnected:
            return
        connection.disconnected = True
        connection.stage = "closed"
        _clearTimeout(connection)
        for active in list(connection.activeRequests.values()):
            active.controller.abort(Exception("Client disconnected"))
        connection.activeRequests.clear()
        connection.serviceStateEncoders.clear()
        if connection in self._connections:
            self._connections.discard(connection)
            self._notifyConnectionCountChanged()
        serverServices = connection.serverServices
        connection.serverServices = None
        spawn(self._releaseDisconnected(connection, serverServices), self._background)

    async def _releaseDisconnected(self, connection: ConnectionState, serverServices: Any) -> None:
        results = await allSettled(
            [
                startTask(self._sessions.disconnect(connection, TODO_CONTEXT)),
                serverServices.release(TODO_CONTEXT) if serverServices is not None else None,
            ]
        )
        for result in results:
            if result.status == "rejected":
                self._reportError(result.reason)

    async def _sendServiceUpdate(self, connection: ConnectionState, subscriptionId: str, update: Any) -> None:
        stateEncoder = connection.serviceStateEncoders.get(subscriptionId)
        if stateEncoder is None:
            return
        await self._sendMessage(
            connection,
            {"type": "service_update", "subscriptionId": subscriptionId, "update": stateEncoder.encodeUpdate(update)},
        )

    async def _sendMessage(self, connection: Any, message: ServerMessage) -> bool:
        if connection.disconnected or connection.connection.closed:
            return False
        try:
            frame = encodeServerMessage(message, {"maxFrameLength": self._maxFrameLength})
        except Exception as error:  # noqa: BLE001
            self._reportError(error)
            await self._closeConnection(connection.connection)
            self._disconnect(connection)
            return False
        try:
            await awaitValue(connection.connection.send(frame))
            return True
        except Exception as error:  # noqa: BLE001
            self._reportError(error)
            await self._closeConnection(connection.connection)
            self._disconnect(connection)
            return False

    async def _failProtocol(self, connection: ConnectionState, error: ProtocolError) -> None:
        if connection.disconnected or connection.stage == "closing" or connection.stage == "closed":
            return
        connection.stage = "closing"
        _clearTimeout(connection)
        message: ServerMessage = {"type": "hello_error", "error": error}
        finalFrame: bytes | None = None
        try:
            finalFrame = encodeServerMessage(message, {"maxFrameLength": self._maxFrameLength})
        except Exception as encodeError:  # noqa: BLE001
            self._reportError(encodeError)
        await self._closeConnection(connection.connection, finalFrame)
        self._disconnect(connection)

    async def _closeServerState(self) -> None:
        connections = list(self._connections)
        for connection in connections:
            connection.stage = "closing"
            _clearTimeout(connection)
        await promiseAll([self._closeConnection(connection.connection) for connection in connections])
        for connection in connections:
            self._disconnect(connection)
        cleanup = await allSettled([self._sessions.close(BACKGROUND_CONTEXT)])
        self._connections.clear()
        errors: list[Any] = [result.reason for result in cleanup if result.status == "rejected"]
        if len(errors) == 1:
            raise errors[0]
        if len(errors) > 1:
            raise aggregateError(errors, "Failed to close server Sessions")

    async def _closeConnection(self, connection: ByteConnection, finalChunk: bytes | None = None) -> None:
        try:
            await awaitValue(connection.close(finalChunk))
        except Exception as error:  # noqa: BLE001
            self._reportError(error)

    async def _closeThenDisconnect(self, connection: ByteConnection, state: ConnectionState) -> None:
        await self._closeConnection(connection)
        self._disconnect(state)

    def _toProtocolError(self, error: BaseException) -> ProtocolError:
        if isinstance(error, ServerError | RemoteServiceError):
            return {"code": error.code, "message": error.message}
        if isinstance(error, ProtocolValidationError):
            return {"code": "invalid_request", "message": error.message}
        self._reportError(error)
        return {"code": "internal_error", "message": INTERNAL_SERVER_ERROR_MESSAGE}

    def _notifyConnectionCountChanged(self) -> None:
        try:
            if self._onConnectionCountChanged is not None:
                self._onConnectionCountChanged(len(self._connections))
        except Exception as error:  # noqa: BLE001
            self._reportError(error)

    def _reportError(self, error: Any) -> None:
        try:
            if self._onError is not None:
                self._onError(error if isinstance(error, Exception) else Exception(str(error)))
        except Exception:  # noqa: BLE001, S110 - error observers cannot affect server state
            pass

    def _settleClosed(self, error: Exception | None = None) -> None:
        if self._closedSettled:
            return
        self._closedSettled = True
        if error is None:
            self._closed.set_result(None)
        else:
            self._closed.set_exception(error)


def _observe(future: asyncio.Future[Any]) -> None:
    """A failure of ``closed`` that nobody awaits is not an unhandled error."""
    if not future.cancelled():
        future.exception()


def _clearTimeout(state: ConnectionState) -> None:
    if state.handshakeTimeout is not None:
        state.handshakeTimeout.cancel()


def sameTarget(left: RpcTarget, right: RpcTarget) -> bool:
    if left["serverId"] != right["serverId"]:
        return False
    if "sessionId" not in left or "sessionId" not in right:
        return "sessionId" not in left and "sessionId" not in right
    return left["sessionId"] == right["sessionId"] and left["attachmentId"] == right["attachmentId"]  # type: ignore[typeddict-item]


def resolveOptions(options: ServerOptions) -> dict[str, int]:
    if not isinstance(options.get("listeners"), list | tuple):
        raise TypeError("Server listeners must be an array")
    if not isServerId(options.get("serverId")):
        raise TypeError("serverId must be a canonical lowercase UUIDv4")
    maxFrameLength = options.get("maxFrameLength")
    if maxFrameLength is None:
        maxFrameLength = DEFAULT_MAX_FRAME_LENGTH
    if not jsIsInteger(maxFrameLength) or maxFrameLength <= 0 or maxFrameLength > MAX_UINT32:
        raise TypeError(f"Server maxFrameLength must be an integer between 1 and {MAX_UINT32}")
    handshakeTimeoutMs = options.get("handshakeTimeoutMs")
    if handshakeTimeoutMs is None:
        handshakeTimeoutMs = DEFAULT_HANDSHAKE_TIMEOUT_MS
    if not jsIsInteger(handshakeTimeoutMs) or handshakeTimeoutMs <= 0 or handshakeTimeoutMs > MAX_TIMER_DELAY_MS:
        raise TypeError(f"Server handshakeTimeoutMs must be an integer between 1 and {MAX_TIMER_DELAY_MS}")
    return {"maxFrameLength": maxFrameLength, "handshakeTimeoutMs": handshakeTimeoutMs}


__all__ = ["Server"]
