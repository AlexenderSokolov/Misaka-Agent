"""One physical connection: handshake, framing and lifecycle.

Translated from pi's ``packages/client/src/connection.ts``. A promise held in a field is an ``asyncio`` future;
an async function that JavaScript runs at once is started with ``startTask``.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from typing import Any, NotRequired, TypedDict

from misaka.protocol import (
    DEFAULT_MAX_FRAME_LENGTH,
    PROTOCOL_VERSION,
    AttachmentEnvelope,
    ProtocolValidationError,
    ResponseEnvelope,
    ServerHello,
    ServerMessage,
    ServerMessageDecoder,
    ServiceEventEnvelope,
    encodeClientMessage,
)
from misaka.utils.js import (
    awaitValue,
    jsIsInteger,
    jsonStringify,
    rejectedPromise,
    startTask,
)

from .errors import DisconnectedError, ServerError, toDisconnectedError, toError
from .transport import ByteTransport, ByteTransportFactory
from .types import ConnectionState, ConnectionStateChange

MAX_UINT32 = 0xFFFF_FFFF


class _Lifecycle:
    """``disconnected``, ``connecting`` or ``connected``, with what each state carries."""

    __slots__ = ("decoder", "handshake", "id", "state", "transport")

    def __init__(
        self,
        state: str,
        id: int = 0,
        decoder: ServerMessageDecoder | None = None,
        transport: ByteTransport | None = None,
        handshake: asyncio.Future[Any] | None = None,
    ) -> None:
        self.state = state
        self.id = id
        self.decoder = decoder
        self.transport = transport
        self.handshake = handshake


class _Handlers:
    def __init__(self, connection: Connection, id: int) -> None:
        self._connection = connection
        self._id = id

    def onData(self, chunk: bytes) -> None:
        self._connection._handleData(self._id, chunk)

    def onClose(self) -> None:
        if self._connection._isCurrent(self._id):
            self._connection._handleClose()

    def onError(self, error: Exception) -> None:
        if self._connection._isCurrent(self._id):
            self._connection._failAndClose(toDisconnectedError(error))


class ConnectionOptions(TypedDict):
    transportFactory: ByteTransportFactory
    serverId: str
    maxFrameLength: NotRequired[int | None]
    onHandshake: Callable[[ServerHello], None]
    onMessage: Callable[[ResponseEnvelope | ServiceEventEnvelope | AttachmentEnvelope], None]
    onStateChange: Callable[[ConnectionStateChange], None]


class Connection:
    def __init__(self, options: ConnectionOptions) -> None:
        self._options = options
        maxFrameLength = options.get("maxFrameLength")
        self._maxFrameLength = maxFrameLength if maxFrameLength is not None else DEFAULT_MAX_FRAME_LENGTH
        if not jsIsInteger(self._maxFrameLength) or self._maxFrameLength <= 0 or self._maxFrameLength > MAX_UINT32:
            raise TypeError(f"Client maxFrameLength must be an integer between 1 and {MAX_UINT32}")
        self._lifecycle = _Lifecycle("disconnected")
        self._sequence = 0

    @property
    def state(self) -> ConnectionState:
        return self._lifecycle.state  # type: ignore[return-value]

    @property
    def maxFrameLength(self) -> int:
        return self._maxFrameLength

    def connect(self) -> asyncio.Future[dict[str, Any]]:
        if self._lifecycle.state != "disconnected":
            return rejectedPromise(DisconnectedError(f"Client is already {self._lifecycle.state}"))
        self._sequence += 1
        id_ = self._sequence
        handshake: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._lifecycle = _Lifecycle(
            "connecting",
            id_,
            ServerMessageDecoder({"maxFrameLength": self._maxFrameLength}),
            None,
            handshake,
        )
        self._options["onStateChange"]({"state": "connecting"})
        startTask(self._openTransport(id_, _Handlers(self, id_)))
        return handshake

    def disconnect(self, reason: str | Exception = "Client disconnected") -> None:
        if self._lifecycle.state == "disconnected":
            return
        self._failAndClose(DisconnectedError(reason) if isinstance(reason, str) else reason)

    def fail(self, error: Exception) -> None:
        self._failAndClose(error)

    def send(self, frame: bytes) -> None:
        lifecycle = self._lifecycle
        if lifecycle.state != "connected":
            raise DisconnectedError()
        assert lifecycle.transport is not None
        try:
            sending = lifecycle.transport.send(frame)
        except Exception as error:  # noqa: BLE001 - a transport that throws is disconnected
            self._failAndClose(toDisconnectedError(error))
            return
        if inspect.isawaitable(sending):
            # An awaitable starts when it is made, so calls reach the transport in invocation order.
            task = startTask(sending)

            def settled(done: asyncio.Future[Any]) -> None:
                if done.cancelled() or (error := done.exception()) is None:
                    return
                current = self._lifecycle
                if current.state != "disconnected" and current.transport is lifecycle.transport:
                    self._failAndClose(toDisconnectedError(error))

            task.add_done_callback(settled)

    async def _openTransport(self, id: int, handlers: _Handlers) -> None:
        try:
            # ``await`` takes a turn whatever it is given, so ``connect()`` returns while the state is still connecting.
            transport = await awaitValue(self._options["transportFactory"](handlers))  # type: ignore[arg-type]
        except Exception as error:  # noqa: BLE001
            if self._isCurrent(id):
                self._fail(toDisconnectedError(error))
            return
        lifecycle = self._lifecycle
        if lifecycle.state != "connecting" or lifecycle.id != id:
            transport.close()
            return
        self._lifecycle = _Lifecycle("connecting", lifecycle.id, lifecycle.decoder, transport, lifecycle.handshake)
        try:
            sending = transport.send(
                encodeClientMessage({"type": "hello", "version": PROTOCOL_VERSION}, {"maxFrameLength": self._maxFrameLength})
            )
            if inspect.isawaitable(sending):
                await sending
        except Exception as error:  # noqa: BLE001
            if self._isCurrent(id):
                self._failAndClose(toDisconnectedError(error))

    def _handleData(self, id: int, chunk: bytes) -> None:
        lifecycle = self._lifecycle
        if lifecycle.state == "disconnected" or lifecycle.id != id:
            return
        if lifecycle.state == "connecting" and lifecycle.transport is None:
            self._failAndClose(ProtocolValidationError("Received server data before the client hello was sent"))
            return
        assert lifecycle.decoder is not None
        try:
            messages = lifecycle.decoder.push(chunk)
        except Exception as error:  # noqa: BLE001
            self._failAndClose(toError(error))
            return
        for message in messages:
            if self._lifecycle.state == "disconnected":
                return
            self._handleMessage(message)

    def _handleMessage(self, message: ServerMessage) -> None:
        lifecycle = self._lifecycle
        if lifecycle.state == "connecting":
            if message["type"] == "hello_error":
                self._failAndClose(ServerError(message["error"]))
                return
            if message["type"] != "hello":
                self._failAndClose(ProtocolValidationError("Expected server hello as first message"))
                return
            if message["serverId"] != self._options["serverId"]:
                self._failAndClose(
                    ProtocolValidationError(
                        f"Connected server {jsonStringify(message['serverId'])} does not match {jsonStringify(self._options['serverId'])}"
                    )
                )
                return
            if lifecycle.transport is None:
                self._failAndClose(ProtocolValidationError("Received server hello before the client hello was sent"))
                return
            connected = _Lifecycle("connected", lifecycle.id, lifecycle.decoder, lifecycle.transport, lifecycle.handshake)
            self._lifecycle = connected
            try:
                self._options["onHandshake"](message)
            except Exception as error:  # noqa: BLE001
                if self._lifecycle is connected:
                    self._failAndClose(toError(error))
                return
            if self._lifecycle is not connected:
                return
            self._options["onStateChange"]({"state": "connected"})
            if self._lifecycle is not connected:
                return
            handshake = lifecycle.handshake
            self._lifecycle = _Lifecycle("connected", connected.id, connected.decoder, connected.transport, None)
            assert handshake is not None
            if not handshake.done():
                handshake.set_result(message)
            return
        if lifecycle.state != "connected":
            return
        if message["type"] == "hello" or message["type"] == "hello_error":
            self._failAndClose(ProtocolValidationError("Unexpected handshake message"))
            return
        self._options["onMessage"](message)

    def _handleClose(self) -> None:
        lifecycle = self._lifecycle
        if lifecycle.state == "disconnected":
            return
        error: Exception = DisconnectedError("Byte transport closed")
        try:
            assert lifecycle.decoder is not None
            lifecycle.decoder.end()
        except Exception as decoderError:  # noqa: BLE001
            error = toError(decoderError)
        self._fail(error)

    def _failAndClose(self, error: Exception) -> None:
        lifecycle = self._lifecycle
        transport = None if lifecycle.state == "disconnected" else lifecycle.transport
        self._fail(error)
        if transport is not None:
            transport.close()

    def _fail(self, error: Exception) -> None:
        lifecycle = self._lifecycle
        if lifecycle.state == "disconnected":
            return
        self._lifecycle = _Lifecycle("disconnected")
        if lifecycle.handshake is not None and not lifecycle.handshake.done():
            lifecycle.handshake.set_exception(error)
            # Nobody may be waiting for the handshake; a failure no one reads is not an unhandled error.
            lifecycle.handshake.exception()
        self._options["onStateChange"]({"state": "disconnected", "error": error})

    def _isCurrent(self, id: int) -> bool:
        return self._lifecycle.state != "disconnected" and self._lifecycle.id == id



__all__ = ["Connection"]
