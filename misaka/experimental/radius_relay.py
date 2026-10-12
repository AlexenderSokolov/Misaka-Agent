"""Translated from pi's ``coding-agent/src/experimental/radius-relay.ts``.

The Radius relay carries the experimental server's protocol over WebSocket connections to a Radius gateway: a server host
keeps one multiplexed connection open and turns every ``connection_open`` into a server byte connection, and a client
opens one raw connection to the gateway.

MISAKA fork:

- Upstream's WebSocket is undici's, with the browser API (``addEventListener``, events with ``data``, ``code``, ``reason`` and
  ``error``). A ``RadiusRelayWebSocket`` here has the same members, events are dicts, and binary frames are ``bytes``
  (JavaScript's ``ArrayBuffer`` and ``Uint8Array``). ``defaultWebSocketFactory`` adapts the ``websockets`` library to it.
- ``delay`` and the classes that call it take ``timers`` (``timers.py``), so tests can control the clock.
"""

from __future__ import annotations

import asyncio
import re
import urllib.parse
from collections.abc import Callable
from typing import Any

from misaka.experimental.timers import DEFAULT_TIMERS
from misaka.protocol import DEFAULT_MAX_FRAME_LENGTH
from misaka.utils.js import (
    UNDEFINED,
    AbortController,
    errorMessage,
    jsIsInteger,
    jsonParse,
    jsonStringify,
    jsTrim,
    rejectedPromise,
    startTask,
)

RADIUS_RELAY_HOST_SUBPROTOCOL = "pi-session-relay.host.v1"
RADIUS_RELAY_CLIENT_SUBPROTOCOL = "pi-session-relay.client.v1"

RELAY_DATA_HEADER_BYTES = 18
RELAY_DATA_FRAME_VERSION = 1
RELAY_DATA_FRAME_TYPE = 1
CONNECTION_ID_PATTERN = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}")
MAX_PENDING_BYTES = DEFAULT_MAX_FRAME_LENGTH * 4
DRAIN_THRESHOLD_BYTES = 1024 * 1024
HOST_RETRY_INITIAL_MS = 1_000
HOST_RETRY_MAX_MS = 30_000
MISSING_AUTH_RETRY_MS = 30_000
CLIENT_RETRY_INITIAL_MS = 1_000
CLIENT_RETRY_MAX_MS = 30_000
# A browser-style WebSocket permits callers to send only code 1000 or application codes from 3000 through 4999. RFC
# protocol codes such as 1002 and 1011 may be received but cannot be passed to close().
LOCAL_PROTOCOL_ERROR_CLOSE_CODE = 4000
LOCAL_TRANSPORT_ERROR_CLOSE_CODE = 4001


class RadiusRelayHost:
    """Maintain the experimental server's multiplexed, authenticated Radius host connection. ``options`` is
    ``{"serverId", "server" (with ``accept(connection)``), "auth", "webSocketFactory"?, "onStatus"?, "timers"?}``."""

    def __init__(self, options: dict[str, Any]) -> None:
        self._options = options
        self._timers = options.get("timers") or DEFAULT_TIMERS
        self._abortController = AbortController()
        self._connections: dict[str, dict[str, Any]] = {}
        self._socket: Any = None
        self._writer: OrderedWebSocketWriter | None = None
        self._loop: asyncio.Future[None] | None = None
        self._closed = False

    def start(self) -> None:
        if self._loop is not None:
            return
        self._loop = startTask(self._run())

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._abortController.abort()
        if self._writer is not None:
            self._writer.close()
        if self._socket is not None:
            closeWebSocket(self._socket, 1000, "Pi server stopped")
        self._dropConnections()
        if self._loop is not None:
            await self._loop

    def _status(self, status: dict[str, Any]) -> None:
        onStatus = self._options.get("onStatus")
        if onStatus is not None:
            onStatus(status)

    async def _run(self) -> None:
        retryMs = HOST_RETRY_INITIAL_MS
        signal = self._abortController.signal
        while not self._closed:
            try:
                auth = await self._options["auth"].resolve({"required": False, "signal": signal})
                if auth is None:
                    self._status({"status": "not_authenticated"})
                    await delay(MISSING_AUTH_RETRY_MS, signal, self._timers)
                    continue

                self._status({"status": "connecting"})
                socket = await openRadiusRelayWebSocket(
                    relayWebSocketUrl(auth["gateway"], self._options["serverId"]),
                    RADIUS_RELAY_HOST_SUBPROTOCOL,
                    auth["token"],
                    self._options.get("webSocketFactory") or defaultWebSocketFactory,
                    signal,
                )
                retryMs = HOST_RETRY_INITIAL_MS
                self._status({"status": "connected"})
                await self._serve(socket)
                if not self._closed:
                    raise RuntimeError("Radius relay host disconnected")
            except Exception as error:  # noqa: BLE001 - the host reconnects
                if self._closed or signal.aborted:
                    break
                self._status({"status": "retrying", "error": errorMessage(error)})
                try:
                    await delay(retryMs, signal, self._timers)
                except Exception:  # noqa: BLE001 - aborted while waiting
                    break
                retryMs = min(retryMs * 2, HOST_RETRY_MAX_MS)

    def _serve(self, socket: Any) -> asyncio.Future[None]:
        self._socket = socket
        self._writer = OrderedWebSocketWriter(socket, self._timers)
        done: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        state = {"terminal": False}

        def cleanup() -> None:
            socket.removeEventListener("message", onMessage)
            socket.removeEventListener("close", onClose)
            socket.removeEventListener("error", onError)
            if self._writer is not None:
                self._writer.close()
            if self._socket is socket:
                self._socket = None
                self._writer = None

        def finish(error: Exception | None = None) -> None:
            if state["terminal"]:
                return
            state["terminal"] = True
            cleanup()
            self._dropConnections(error)
            if error is not None:
                done.set_exception(error)
                done.exception()
            else:
                done.set_result(None)

        def onMessage(event: dict[str, Any]) -> None:
            try:
                self._handleHostMessage(event["data"])
            except Exception as error:  # noqa: BLE001 - a protocol error closes the connection
                closeWebSocket(socket, LOCAL_PROTOCOL_ERROR_CLOSE_CODE, "Radius relay protocol error")
                finish(toError(error))

        def onClose(event: dict[str, Any]) -> None:
            code, reason = event["code"], event.get("reason")
            finish(
                None
                if code == 1000
                else RuntimeError(f"Radius relay host closed ({code}{f': {reason}' if reason else ''})")
            )

        def onError(event: dict[str, Any]) -> None:
            finish(webSocketError(event))

        socket.addEventListener("message", onMessage)
        socket.addEventListener("close", onClose)
        socket.addEventListener("error", onError)
        return done

    def _handleHostMessage(self, value: Any) -> None:
        if isinstance(value, str):
            control = parseHostControlMessage(value)
            if control["type"] == "ping":
                self._reportSendFailure(self._sendControl({"version": 1, "type": "pong"}))
                return
            if control["type"] == "pong":
                return
            if control["type"] == "connection_open":
                self._openConnection(control["connection_id"])
                return
            if control["type"] == "connection_close":
                self._remoteCloseConnection(control["connection_id"])
                return
        if not isinstance(value, bytes | bytearray | memoryview):
            raise RuntimeError("Radius relay data message must be binary")  # noqa: TRY004
        frame = parseRelayDataFrame(bytes(value))
        if frame is None:
            raise RuntimeError("Invalid Radius relay data frame")
        active = self._connections.get(frame["connectionId"])
        if active is None:
            startTask(
                self._sendControl({"version": 1, "type": "connection_close", "connection_id": frame["connectionId"], "code": 1000})
            ).add_done_callback(_observe)
            return
        active["handler"].onData(frame["payload"])

    def _reportSendFailure(self, sending: asyncio.Future[None]) -> None:
        def report(future: asyncio.Future[None]) -> None:
            if future.cancelled() or future.exception() is None:
                return
            if self._socket is not None:
                closeWebSocket(self._socket, LOCAL_TRANSPORT_ERROR_CLOSE_CODE, "Radius relay send failed")

        sending.add_done_callback(report)

    def _openConnection(self, connectionId: str) -> None:
        if connectionId in self._connections:
            raise RuntimeError("Radius relay reused a connection ID")
        connection = RelayServerByteConnection(
            lambda chunk: self._sendData(connectionId, chunk),
            lambda finalChunk: self._serverCloseConnection(connectionId, finalChunk),
        )
        handler = self._options["server"].accept(connection)
        if connection.closed:
            startTask(
                self._sendControl({"version": 1, "type": "connection_close", "connection_id": connectionId, "code": 1012})
            ).add_done_callback(_observe)
        else:
            self._connections[connectionId] = {"connection": connection, "handler": handler}

    def _remoteCloseConnection(self, connectionId: str) -> None:
        active = self._connections.pop(connectionId, None)
        if active is None:
            return
        active["connection"].markClosed()
        active["handler"].onClose()

    async def _serverCloseConnection(self, connectionId: str, finalChunk: bytes | None = None) -> None:
        if self._connections.pop(connectionId, None) is None:
            return
        if finalChunk is not None:
            await self._requireWriter().send(encodeRelayDataFrame(connectionId, finalChunk))
        await self._sendControl({"version": 1, "type": "connection_close", "connection_id": connectionId, "code": 1000})

    def _sendData(self, connectionId: str, chunk: bytes) -> asyncio.Future[None]:
        if connectionId not in self._connections:
            return rejectedPromise(RuntimeError("Radius relay connection is closed"))
        return self._requireWriter().send(encodeRelayDataFrame(connectionId, chunk))

    def _sendControl(self, message: dict[str, Any]) -> asyncio.Future[None]:
        try:
            return self._requireWriter().send(jsonStringify(message))
        except Exception as error:  # noqa: BLE001 - returned as a rejection
            return rejectedPromise(error)

    def _requireWriter(self) -> OrderedWebSocketWriter:
        if self._writer is None:
            raise RuntimeError("Radius relay host is disconnected")
        return self._writer

    def _dropConnections(self, error: Exception | None = None) -> None:
        connections = list(self._connections.values())
        self._connections.clear()
        for active in connections:
            active["connection"].markClosed()
            if error is not None:
                active["handler"].onError(error)
            else:
                active["handler"].onClose()


def _observe(future: asyncio.Future[Any]) -> None:
    if not future.cancelled():
        future.exception()


def createRadiusClientTransportFactory(options: dict[str, Any]) -> Callable[[Any], Any]:
    """``options`` is ``{"serverId", "auth", "webSocketFactory"?, "timers"?}``; the factory takes the handlers and resolves to
    a connected byte transport."""

    async def factory(handlers: Any) -> Any:
        auth = await options["auth"].resolve({"required": True})
        if auth is None:
            raise RuntimeError("Radius authentication is required")
        socket = await openRadiusRelayWebSocket(
            relayWebSocketUrl(auth["gateway"], options["serverId"]),
            RADIUS_RELAY_CLIENT_SUBPROTOCOL,
            auth["token"],
            options.get("webSocketFactory") or defaultWebSocketFactory,
        )
        return RadiusClientByteTransport(socket, handlers, options.get("timers") or DEFAULT_TIMERS)

    return factory


class RadiusClientReconnect:
    """Reconnect one established Radius client and restore its last selected Session. ``client`` provides ``attachment``,
    ``connected``, ``connectionState``, ``disconnect``, ``onAttachmentChange``, ``onConnectionStateChange`` and
    ``reconnect``; ``reattach(sessionId)`` restores the Session."""

    def __init__(self, client: Any, reattach: Callable[[str], Any], timers: Any = None) -> None:
        self._client = client
        self._reattach = reattach
        self._timers = timers or DEFAULT_TIMERS
        self._abortController = AbortController()
        self._reconnecting: asyncio.Future[None] | None = None
        self._disposed = False
        attachment = client.attachment
        self._desiredSessionId: str | None = None if attachment is None else attachment["sessionId"]

        def onAttachment(attachment: Any) -> None:
            if attachment is not None:
                self._desiredSessionId = attachment["sessionId"]
            elif client.connected:
                self._desiredSessionId = None

        def onConnectionState(change: dict[str, Any]) -> None:
            if change["state"] == "disconnected" and not self._disposed:
                self._startReconnect()

        self._removeAttachmentListener = client.onAttachmentChange(onAttachment)
        self._removeConnectionListener = client.onConnectionStateChange(onConnectionState)

    async def dispose(self) -> None:
        if self._disposed:
            return
        self._disposed = True
        self._removeConnectionListener()
        self._removeAttachmentListener()
        self._abortController.abort()
        if self._client.connectionState != "disconnected":
            self._client.disconnect("Radius reconnect stopped")
        if self._reconnecting is not None:
            await self._reconnecting

    def _startReconnect(self) -> None:
        if self._reconnecting is not None:
            return

        async def run() -> None:
            try:
                await self._runReconnect()
            finally:
                self._reconnecting = None

        self._reconnecting = startTask(run())

    async def _runReconnect(self) -> None:
        retryMs = CLIENT_RETRY_INITIAL_MS
        while not self._disposed and not self._client.connected:
            try:
                await delay(retryMs, self._abortController.signal, self._timers)
                await self._client.reconnect()
                sessionId = self._desiredSessionId
                if sessionId is not None:
                    result = self._reattach(sessionId)
                    if asyncio.iscoroutine(result) or isinstance(result, asyncio.Future):
                        await result
                return
            except Exception as error:  # noqa: BLE001 - the next attempt waits longer
                if self._disposed or self._abortController.signal.aborted:
                    return
                if self._client.connected:
                    self._client.disconnect(errorMessage(error))
                retryMs = min(retryMs * 2, CLIENT_RETRY_MAX_MS)


class RelayServerByteConnection:
    def __init__(self, sendChunk: Callable[[bytes], Any], closeConnection: Callable[[bytes | None], Any]) -> None:
        self._sendChunk = sendChunk
        self._closeConnection = closeConnection
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    def send(self, chunk: bytes) -> asyncio.Future[None]:
        if self._closed:
            return rejectedPromise(RuntimeError("Radius relay connection is closed"))
        return startTask(_awaitable(self._sendChunk(chunk)))

    async def close(self, finalChunk: bytes | None = None) -> None:
        if self._closed:
            return
        self._closed = True
        await self._closeConnection(finalChunk)

    def markClosed(self) -> None:
        self._closed = True


async def _awaitable(value: Any) -> Any:
    if asyncio.iscoroutine(value) or isinstance(value, asyncio.Future):
        return await value
    return value


class RadiusClientByteTransport:
    def __init__(self, socket: Any, handlers: Any, timers: Any) -> None:
        self._socket = socket
        self._handlers = handlers
        self._writer = OrderedWebSocketWriter(socket, timers)
        self._closed = False

        def onMessage(event: dict[str, Any]) -> None:
            if self._closed:
                return
            data = event["data"]
            if not isinstance(data, bytes | bytearray | memoryview):
                self._fail(RuntimeError("Radius relay client received a non-binary message"))
                return
            self._handlers.onData(bytes(data))

        def onClose(event: dict[str, Any]) -> None:
            if self._markClosed():
                self._handlers.onClose()

        def onError(event: dict[str, Any]) -> None:
            self._fail(webSocketError(event))

        self._onMessage, self._onClose, self._onError = onMessage, onClose, onError
        socket.addEventListener("message", onMessage)
        socket.addEventListener("close", onClose)
        socket.addEventListener("error", onError)

    def send(self, chunk: bytes) -> asyncio.Future[None]:
        if not isinstance(chunk, bytes | bytearray | memoryview):
            return rejectedPromise(TypeError("Radius relay chunks must be bytes"))
        if self._closed:
            return rejectedPromise(RuntimeError("Radius relay client is closed"))
        return self._writer.send(bytes(chunk))

    def close(self) -> None:
        if not self._markClosed():
            return
        self._socket.close(1000, "Pi client closed")

    def _fail(self, error: Exception) -> None:
        if not self._markClosed():
            return
        closeWebSocket(self._socket, LOCAL_TRANSPORT_ERROR_CLOSE_CODE, "Radius relay transport error")
        self._handlers.onError(error)

    def _markClosed(self) -> bool:
        if self._closed:
            return False
        self._closed = True
        self._writer.close()
        self._socket.removeEventListener("message", self._onMessage)
        self._socket.removeEventListener("close", self._onClose)
        self._socket.removeEventListener("error", self._onError)
        return True


class OrderedWebSocketWriter:
    def __init__(self, socket: Any, timers: Any) -> None:
        self._socket = socket
        self._timers = timers
        self._tail: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._tail.set_result(None)
        self._pendingBytes = 0
        self._closed = False

    def send(self, value: str | bytes) -> asyncio.Future[None]:
        if self._closed:
            return rejectedPromise(RuntimeError("Radius relay WebSocket is closed"))
        byteLength = len(value.encode("utf-8")) if isinstance(value, str) else len(value)
        if self._pendingBytes + byteLength > MAX_PENDING_BYTES:
            return rejectedPromise(RuntimeError("Radius relay exceeded its pending byte limit"))
        self._pendingBytes += byteLength
        previous = self._tail

        async def operation() -> None:
            try:
                await previous
                if self._closed or self._socket.readyState != self._socket.OPEN:
                    raise RuntimeError("Radius relay WebSocket is closed")
                self._socket.send(value)
                while self._socket.bufferedAmount > DRAIN_THRESHOLD_BYTES:
                    if self._closed or self._socket.readyState != self._socket.OPEN:
                        raise RuntimeError("Radius relay WebSocket closed during write")
                    await delay(5, None, self._timers)
            finally:
                self._pendingBytes -= byteLength

        tracked = asyncio.ensure_future(operation())

        async def settle() -> None:
            try:
                await tracked
            except Exception:  # noqa: BLE001, S110 - `tracked.catch(() => {})`: the tail only marks that the write settled
                pass

        self._tail = asyncio.ensure_future(settle())
        return tracked

    def close(self) -> None:
        self._closed = True


def parseHostControlMessage(value: str) -> dict[str, Any]:
    try:
        parsed = jsonParse(value)
    except ValueError:
        raise RuntimeError("Invalid Radius relay control message") from None
    if not isinstance(parsed, dict):
        raise RuntimeError("Invalid Radius relay control message")  # noqa: TRY004
    if parsed.get("version") != 1 or isinstance(parsed.get("version"), bool):
        raise RuntimeError("Unsupported Radius relay control version")
    messageType = parsed.get("type")
    if messageType in ("ping", "pong"):
        return {"version": 1, "type": messageType}
    code = parsed.get("code", UNDEFINED)
    connectionId = parsed.get("connection_id")
    if (
        messageType in ("connection_open", "connection_close")
        and isinstance(connectionId, str)
        and CONNECTION_ID_PATTERN.fullmatch(connectionId) is not None
        and (code is UNDEFINED or (jsIsInteger(code) and 1000 <= code <= 4999))
    ):
        if messageType == "connection_open":
            return {"version": 1, "type": "connection_open", "connection_id": connectionId}
        return {
            "version": 1,
            "type": "connection_close",
            "connection_id": connectionId,
            **({} if code is UNDEFINED else {"code": int(code)}),
        }
    raise RuntimeError("Invalid Radius relay control message")


def encodeRelayDataFrame(connectionId: str, payload: bytes) -> bytes:
    if CONNECTION_ID_PATTERN.fullmatch(connectionId) is None:
        raise TypeError("Invalid Radius relay connection ID")
    return bytes([RELAY_DATA_FRAME_VERSION, RELAY_DATA_FRAME_TYPE]) + bytes.fromhex(connectionId.replace("-", "")) + bytes(payload)


def parseRelayDataFrame(frame: bytes) -> dict[str, Any] | None:
    if len(frame) < RELAY_DATA_HEADER_BYTES:
        return None
    if frame[0] != RELAY_DATA_FRAME_VERSION or frame[1] != RELAY_DATA_FRAME_TYPE:
        return None
    hexId = frame[2:RELAY_DATA_HEADER_BYTES].hex()
    connectionId = f"{hexId[:8]}-{hexId[8:12]}-{hexId[12:16]}-{hexId[16:20]}-{hexId[20:]}"
    if CONNECTION_ID_PATTERN.fullmatch(connectionId) is None:
        return None
    return {"connectionId": connectionId, "payload": bytes(frame[RELAY_DATA_HEADER_BYTES:])}


def relayWebSocketUrl(gateway: str, serverId: str) -> str:
    url = urllib.parse.urlsplit(urllib.parse.urljoin(gateway, f"/v1/session-relays/{serverId}/connect"))
    if url.scheme == "https":
        scheme = "wss"
    elif url.scheme == "http":
        scheme = "ws"
    else:
        raise RuntimeError(f"Unsupported Radius gateway protocol: {url.scheme}:")
    return urllib.parse.urlunsplit(url._replace(scheme=scheme))


class WebsocketsWebSocket:
    """A browser-style WebSocket over the ``websockets`` library: it connects in the background and reports ``open``,
    ``message``, ``error`` and ``close`` to its listeners."""

    OPEN = 1

    def __init__(self, url: str, protocol: str, authorization: str) -> None:
        self.binaryType = "blob"
        self.bufferedAmount = 0
        self.protocol = ""
        self.readyState = 0
        self._listeners: dict[str, list[tuple[Callable[[dict[str, Any]], None], bool]]] = {
            "open": [],
            "message": [],
            "close": [],
            "error": [],
        }
        self._connection: Any = None
        self._closeRequest: tuple[int, str] | None = None
        self._writes: set[asyncio.Task[Any]] = set()
        self._runner = asyncio.get_running_loop().create_task(self._run(url, protocol, authorization))

    def addEventListener(self, type: str, listener: Callable[[dict[str, Any]], None], options: dict[str, Any] | None = None) -> None:
        self._listeners[type].append((listener, bool(options and options.get("once"))))

    def removeEventListener(self, type: str, listener: Callable[[dict[str, Any]], None]) -> None:
        self._listeners[type] = [entry for entry in self._listeners[type] if entry[0] != listener]

    def _emit(self, type: str, event: dict[str, Any]) -> None:
        for entry in list(self._listeners[type]):
            if entry not in self._listeners[type]:
                continue
            if entry[1]:
                self._listeners[type].remove(entry)
            entry[0](event)

    async def _run(self, url: str, protocol: str, authorization: str) -> None:
        from websockets.asyncio.client import connect
        from websockets.exceptions import ConnectionClosed

        try:
            async with connect(
                url, subprotocols=[protocol], additional_headers={"authorization": authorization}, max_size=None
            ) as connection:
                if self._closeRequest is not None:
                    await connection.close(*self._closeRequest)
                self._connection = connection
                self.protocol = connection.subprotocol or ""
                self.readyState = 1
                self._emit("open", {})
                # MISAKA fork: ``websockets`` hands over the frames that arrived with the handshake without suspending,
                # so the listeners of whoever awaits the open event attach first; undici dispatches them a task later.
                await asyncio.sleep(0)
                try:
                    async for message in connection:
                        # A browser WebSocket drops what arrives once it is no longer OPEN.
                        if self.readyState == self.OPEN:
                            self._emit("message", {"data": message})
                except ConnectionClosed as closed:
                    # undici reports a connection that ended without a close frame as an error before it closes.
                    if closed.rcvd is None:
                        self.readyState = 3
                        self._emit("error", {"message": ""})
                self.readyState = 3
                code, reason = connection.close_code or 1006, connection.close_reason or ""
        except Exception as error:  # noqa: BLE001 - reported through the error and close events
            self.readyState = 3
            self._emit("error", {"error": error, "message": str(error)})
            code, reason = 1006, ""
        self._emit("close", {"code": code, "reason": reason})

    def send(self, data: str | bytes) -> None:
        if self.readyState != self.OPEN or self._connection is None:
            raise RuntimeError("socket is not open")
        size = len(data.encode("utf-8")) if isinstance(data, str) else len(data)
        self.bufferedAmount += size
        task = asyncio.get_running_loop().create_task(self._write(data, size))
        self._writes.add(task)
        task.add_done_callback(self._writes.discard)

    async def _write(self, data: str | bytes, size: int) -> None:
        try:
            await self._connection.send(data)
        except Exception as error:  # noqa: BLE001 - reported through the error event
            self._emit("error", {"error": error, "message": str(error)})
        finally:
            self.bufferedAmount -= size

    def close(self, code: int | None = None, reason: str | None = None) -> None:
        code = 1000 if code is None else code
        if code != 1000 and (code < 3000 or code > 4999):
            raise ValueError("Invalid close code")
        reason = reason or ""
        if self.readyState == 3:
            return
        if self._connection is None:
            # Closing while the handshake is pending: the runner closes once connected.
            self._closeRequest = (code, reason)
            return
        self.readyState = 2
        asyncio.get_running_loop().create_task(self._connection.close(code, reason))


def defaultWebSocketFactory(options: dict[str, str]) -> Any:
    return WebsocketsWebSocket(options["url"], options["protocol"], options["authorization"])


def openRadiusRelayWebSocket(
    url: str, protocol: str, token: str, factory: Callable[[dict[str, str]], Any], signal: Any = None
) -> asyncio.Future[Any]:
    if signal is not None:
        signal.throwIfAborted()
    loop = asyncio.get_running_loop()
    result: asyncio.Future[Any] = loop.create_future()
    socket = factory({"url": url, "protocol": protocol, "authorization": f"Bearer {token}"})
    socket.binaryType = "arraybuffer"
    state = {"settled": False}

    def cleanup() -> None:
        socket.removeEventListener("open", onOpen)
        socket.removeEventListener("close", onClose)
        socket.removeEventListener("error", onError)
        if signal is not None:
            signal.removeEventListener("abort", onAbort)

    def fail(error: Exception) -> None:
        if state["settled"]:
            return
        state["settled"] = True
        cleanup()
        closeWebSocket(socket, 1000, "Radius relay connection failed")
        result.set_exception(error)
        result.exception()

    def onOpen(event: dict[str, Any]) -> None:
        if state["settled"]:
            return
        if socket.protocol != protocol:
            fail(RuntimeError(f"Radius relay selected unexpected WebSocket protocol {jsonStringify(socket.protocol)}"))
            return
        state["settled"] = True
        cleanup()
        result.set_result(socket)

    def onClose(event: dict[str, Any]) -> None:
        fail(RuntimeError(f"Radius relay closed before connecting ({event['code']})"))

    def onError(event: dict[str, Any]) -> None:
        fail(webSocketError(event))

    def onAbort() -> None:
        fail(RuntimeError("Radius relay connection cancelled"))

    socket.addEventListener("open", onOpen, {"once": True})
    socket.addEventListener("close", onClose, {"once": True})
    socket.addEventListener("error", onError, {"once": True})
    if signal is not None:
        signal.addEventListener("abort", onAbort, True)
    return result


def closeWebSocket(socket: Any, code: int, reason: str) -> None:
    try:
        socket.close(code, reason)
    except Exception:  # noqa: BLE001, S110 - a WebSocket can reject close() while its opening handshake is still pending
        pass


def webSocketError(event: dict[str, Any]) -> Exception:
    error = event.get("error")
    if isinstance(error, Exception) and len(jsTrim(errorMessage(error))) > 0:
        return error
    return RuntimeError(jsTrim(event.get("message") or "") or "Radius WebSocket connection failed")


async def delay(milliseconds: float, signal: Any, timers: Any) -> None:
    if signal is not None:
        signal.throwIfAborted()
    done: asyncio.Future[None] = asyncio.get_running_loop().create_future()

    def cleanup() -> None:
        if signal is not None:
            signal.removeEventListener("abort", onAbort)

    def elapsed() -> None:
        cleanup()
        if not done.done():
            done.set_result(None)

    timer = timers.setTimeout(elapsed, milliseconds)

    def onAbort() -> None:
        timers.clearTimeout(timer)
        cleanup()
        try:
            signal.throwIfAborted()
        except Exception as error:  # noqa: BLE001 - the delay fails with the abort reason
            if not done.done():
                done.set_exception(error)
                done.exception()

    if signal is not None:
        signal.addEventListener("abort", onAbort, True)
    await done

# A RadiusRelayWebSocket is an object with ``binaryType``, ``bufferedAmount``, ``protocol``, ``readyState``, ``OPEN``,
# ``send(data)`` (``str`` or ``bytes``), ``close(code=None, reason=None)``, ``addEventListener(type, listener, options=None)``
# and ``removeEventListener(type, listener)``. Its events are ``open`` ({}), ``message`` ({"data"}), ``close`` ({"code",
# "reason"}) and ``error`` ({"error"?, "message"?}). A factory takes ``{"url", "protocol", "authorization"}``.


def toError(error: Any) -> Exception:
    return error if isinstance(error, Exception) else RuntimeError(str(error))
