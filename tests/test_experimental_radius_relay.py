"""pi coding-agent's ``test/experimental-radius-relay.test.ts``, plus the default WebSocket adapter against a local server."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest

from misaka.experimental.radius_auth import RadiusRelayAuthResolver
from misaka.experimental.radius_relay import (
    RadiusClientReconnect,
    RadiusRelayHost,
    createRadiusClientTransportFactory,
    defaultWebSocketFactory,
    encodeRelayDataFrame,
    parseRelayDataFrame,
    relayWebSocketUrl,
)
from tests.chat_support import waitFor
from tests.experimental_support import FakeTimers, flush

serverId = "00000000-0000-4000-8000-000000000001"
connectionId = "00000000-0000-4000-8000-000000000002"


@pytest.fixture(autouse=True)
def online(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MISAKA_OFFLINE", raising=False)


class FakeWebSocket:
    OPEN = 1

    def __init__(self) -> None:
        self.sent: list[str | bytes] = []
        self.onSend: Callable[[str | bytes], None] | None = None
        self.listeners: dict[str, list[Callable[[dict[str, Any]], None]]] = {}
        self.binaryType = "blob"
        self.bufferedAmount = 0
        self.protocol = ""
        self.readyState = 0

    def addEventListener(self, type: str, listener: Callable[[dict[str, Any]], None], options: Any = None) -> None:
        listeners = self.listeners.setdefault(type, [])
        if listener not in listeners:
            listeners.append(listener)

    def removeEventListener(self, type: str, listener: Callable[[dict[str, Any]], None]) -> None:
        listeners = self.listeners.get(type, [])
        if listener in listeners:
            listeners.remove(listener)

    def send(self, data: str | bytes) -> None:
        if self.readyState != self.OPEN:
            raise RuntimeError("socket is not open")
        self.sent.append(data)
        if self.onSend is not None:
            self.onSend(data)

    def close(self, code: int = 1000, reason: str = "") -> None:
        if code != 1000 and (code < 3000 or code > 4999):
            raise ValueError("Invalid close code")
        if self.readyState == 3:
            return
        self.readyState = 3
        self.emit("close", {"code": code, "reason": reason})

    def open(self, protocol: str) -> None:
        self.protocol = protocol
        self.readyState = self.OPEN
        self.emit("open", {})

    def message(self, data: Any) -> None:
        self.emit("message", {"data": data})

    def remoteClose(self, code: int = 1006, reason: str = "lost") -> None:
        self.readyState = 3
        self.emit("close", {"code": code, "reason": reason})

    def fail(self, error: Exception) -> None:
        self.emit("error", {"error": error, "message": str(error)})

    def abnormalClose(self, error: Exception) -> None:
        self.readyState = 3
        self.emit("error", {"error": error, "message": str(error)})
        self.emit("close", {"code": 1006, "reason": ""})

    def emit(self, type: str, event: dict[str, Any]) -> None:
        for listener in list(self.listeners.get(type, [])):
            listener(event)


def socketFactory() -> SimpleNamespace:
    sockets: list[SimpleNamespace] = []

    def factory(options: dict[str, str]) -> FakeWebSocket:
        socket = FakeWebSocket()
        sockets.append(SimpleNamespace(socket=socket, options=options))
        return socket

    return SimpleNamespace(sockets=sockets, factory=factory)


def test_matches_the_radius_multiplexing_envelope():
    payload = bytes([0, 1, 2, 255])
    parsed = parseRelayDataFrame(encodeRelayDataFrame(connectionId, payload))

    assert parsed is not None
    assert parsed["connectionId"] == connectionId
    assert parsed["payload"] == payload


def test_rejects_malformed_envelopes():
    assert parseRelayDataFrame(b"short") is None
    assert parseRelayDataFrame(bytes([2, 1]) + bytes(16)) is None
    assert parseRelayDataFrame(bytes([1, 2]) + bytes(16)) is None
    # A connection ID that is not a version 4 UUID is no connection ID.
    assert parseRelayDataFrame(bytes([1, 1]) + bytes(16)) is None
    with pytest.raises(TypeError, match="Invalid Radius relay connection ID"):
        encodeRelayDataFrame("not-an-id", b"")


def test_builds_the_secure_websocket_url_of_a_gateway():
    assert relayWebSocketUrl("https://gateway.example", serverId) == f"wss://gateway.example/v1/session-relays/{serverId}/connect"
    assert relayWebSocketUrl("http://localhost:8080/base", serverId) == f"ws://localhost:8080/v1/session-relays/{serverId}/connect"
    with pytest.raises(Exception, match="Unsupported Radius gateway protocol"):
        relayWebSocketUrl("ftp://gateway.example", serverId)


async def test_bridges_multiplexed_host_clients_into_independent_server_connections():
    webSockets = socketFactory()
    received: list[bytes] = []
    state: dict[str, Any] = {"accepted": None, "closed": 0}

    class Handler:
        def onData(self, chunk: bytes) -> None:
            received.append(chunk)

        def onClose(self) -> None:
            state["closed"] += 1

        def onError(self, error: Exception) -> None:
            raise AssertionError(error)

    class Server:
        def accept(self, connection: Any) -> Handler:
            state["accepted"] = connection
            return Handler()

    statuses: list[str] = []
    host = RadiusRelayHost(
        {
            "serverId": serverId,
            "server": Server(),
            "auth": RadiusRelayAuthResolver({"type": "token", "token": "secret"}),
            "webSocketFactory": webSockets.factory,
            "onStatus": lambda status: statuses.append(status["status"]),
        }
    )
    host.start()
    await waitFor(lambda: len(webSockets.sockets) == 1)
    socket, options = webSockets.sockets[0].socket, webSockets.sockets[0].options
    assert options["authorization"] == "Bearer secret"
    assert options["url"].endswith(f"/v1/session-relays/{serverId}/connect")
    socket.open(options["protocol"])
    await waitFor(lambda: "connected" in statuses)

    socket.message(json.dumps({"version": 1, "type": "connection_open", "connection_id": connectionId}))
    assert state["accepted"] is not None
    socket.message(encodeRelayDataFrame(connectionId, bytes([1, 2, 3])))
    assert received == [bytes([1, 2, 3])]

    await state["accepted"].send(bytes([4, 5, 6]))
    outbound = parseRelayDataFrame(socket.sent[-1])
    assert outbound is not None
    assert outbound["connectionId"] == connectionId
    assert outbound["payload"] == bytes([4, 5, 6])

    socket.message(json.dumps({"version": 1, "type": "connection_close", "connection_id": connectionId}))
    assert state["closed"] == 1
    await host.close()


async def test_closes_a_server_connection_with_its_final_chunk_and_answers_pings():
    webSockets = socketFactory()
    state: dict[str, Any] = {"accepted": None}

    class Server:
        def accept(self, connection: Any) -> Any:
            state["accepted"] = connection
            return SimpleNamespace(onData=lambda chunk: None, onClose=lambda: None, onError=lambda error: None)

    host = RadiusRelayHost(
        {
            "serverId": serverId,
            "server": Server(),
            "auth": RadiusRelayAuthResolver({"type": "token", "token": "secret"}),
            "webSocketFactory": webSockets.factory,
        }
    )
    host.start()
    await waitFor(lambda: len(webSockets.sockets) == 1)
    socket = webSockets.sockets[0].socket
    socket.open(webSockets.sockets[0].options["protocol"])
    await flush()

    socket.message(json.dumps({"version": 1, "type": "ping"}))
    await waitFor(lambda: any(isinstance(item, str) and json.loads(item)["type"] == "pong" for item in socket.sent))
    socket.message(json.dumps({"version": 1, "type": "connection_open", "connection_id": connectionId}))
    await state["accepted"].close(bytes([9]))
    assert state["accepted"].closed is True
    assert parseRelayDataFrame(socket.sent[-2])["payload"] == bytes([9])  # type: ignore[index]
    assert json.loads(socket.sent[-1]) == {"version": 1, "type": "connection_close", "connection_id": connectionId, "code": 1000}
    with pytest.raises(Exception, match="connection is closed"):
        await state["accepted"].send(b"late")
    await host.close()


async def test_a_protocol_error_closes_the_host_connection_and_reconnects():
    timers = FakeTimers()
    webSockets = socketFactory()
    host = RadiusRelayHost(
        {
            "serverId": serverId,
            "server": SimpleNamespace(accept=lambda connection: None),
            "auth": RadiusRelayAuthResolver({"type": "token", "token": "secret"}),
            "webSocketFactory": webSockets.factory,
            "timers": timers,
        }
    )
    host.start()
    await waitFor(lambda: len(webSockets.sockets) == 1)
    socket = webSockets.sockets[0].socket
    socket.open(webSockets.sockets[0].options["protocol"])
    await flush()
    socket.message("not json")
    await flush()
    assert socket.readyState == 3
    await timers.advanceAsync(1_000)
    await waitFor(lambda: len(webSockets.sockets) == 2)
    await host.close()


async def test_reconnects_the_server_host_after_the_relay_connection_drops():
    timers = FakeTimers()
    webSockets = socketFactory()
    statuses: list[str] = []
    host = RadiusRelayHost(
        {
            "serverId": serverId,
            "server": SimpleNamespace(accept=lambda connection: None),
            "auth": RadiusRelayAuthResolver({"type": "token", "token": "secret"}),
            "webSocketFactory": webSockets.factory,
            "onStatus": lambda status: statuses.append(status["status"]),
            "timers": timers,
        }
    )
    host.start()
    await waitFor(lambda: len(webSockets.sockets) == 1)
    webSockets.sockets[0].socket.open(webSockets.sockets[0].options["protocol"])
    await waitFor(lambda: "connected" in statuses)

    webSockets.sockets[0].socket.remoteClose()
    await timers.advanceAsync(1_000)
    await waitFor(lambda: len(webSockets.sockets) == 2)
    assert statuses[-3:] == ["connected", "retrying", "connecting"]
    await host.close()


async def test_reports_a_missing_credential_and_keeps_retrying():
    timers = FakeTimers()
    webSockets = socketFactory()
    statuses: list[str] = []

    class Auth:
        async def resolve(self, options: dict[str, Any]) -> None:
            return None

    host = RadiusRelayHost(
        {
            "serverId": serverId,
            "server": SimpleNamespace(accept=lambda connection: None),
            "auth": Auth(),
            "webSocketFactory": webSockets.factory,
            "onStatus": lambda status: statuses.append(status["status"]),
            "timers": timers,
        }
    )
    host.start()
    await waitFor(lambda: statuses == ["not_authenticated"])
    await timers.advanceAsync(30_000)
    await waitFor(lambda: statuses == ["not_authenticated", "not_authenticated"])
    assert webSockets.sockets == []
    await host.close()


async def test_uses_a_raw_authenticated_websocket_as_the_client_byte_transport():
    webSockets = socketFactory()
    received: list[bytes] = []
    events: list[str] = []
    handlers = SimpleNamespace(
        onData=received.append, onClose=lambda: events.append("close"), onError=lambda error: events.append("error")
    )
    transportPromise = asyncio.ensure_future(
        createRadiusClientTransportFactory(
            {
                "serverId": serverId,
                "auth": RadiusRelayAuthResolver({"type": "token", "token": "secret"}),
                "webSocketFactory": webSockets.factory,
            }
        )(handlers)
    )
    await waitFor(lambda: len(webSockets.sockets) == 1)
    socket, options = webSockets.sockets[0].socket, webSockets.sockets[0].options
    assert options["authorization"] == "Bearer secret"
    socket.open(options["protocol"])
    transport = await transportPromise

    await transport.send(bytes([1, 2, 3]))
    assert socket.sent[0] == bytes([1, 2, 3])
    socket.message(bytes([4, 5, 6]))
    assert received == [bytes([4, 5, 6])]
    socket.remoteClose()
    assert events == ["close"]


async def test_reports_a_text_frame_as_a_client_transport_error():
    webSockets = socketFactory()
    errors: list[Exception] = []
    handlers = SimpleNamespace(onData=lambda chunk: None, onClose=lambda: None, onError=errors.append)
    transportPromise = asyncio.ensure_future(
        createRadiusClientTransportFactory(
            {
                "serverId": serverId,
                "auth": RadiusRelayAuthResolver({"type": "token", "token": "secret"}),
                "webSocketFactory": webSockets.factory,
            }
        )(handlers)
    )
    await waitFor(lambda: len(webSockets.sockets) == 1)
    socket = webSockets.sockets[0].socket
    socket.open(webSockets.sockets[0].options["protocol"])
    await transportPromise
    socket.message("text")
    assert [str(error) for error in errors] == ["Radius relay client received a non-binary message"]
    assert socket.readyState == 3


async def test_reports_established_abnormal_closures_so_the_client_can_reconnect():
    webSockets = socketFactory()
    events: list[Any] = []
    handlers = SimpleNamespace(
        onData=lambda chunk: None, onClose=lambda: events.append("close"), onError=lambda error: events.append(error)
    )
    transportPromise = asyncio.ensure_future(
        createRadiusClientTransportFactory(
            {
                "serverId": serverId,
                "auth": RadiusRelayAuthResolver({"type": "token", "token": "secret"}),
                "webSocketFactory": webSockets.factory,
            }
        )(handlers)
    )
    await waitFor(lambda: len(webSockets.sockets) == 1)
    socket = webSockets.sockets[0].socket
    socket.open(webSockets.sockets[0].options["protocol"])
    await transportPromise

    socket.abnormalClose(RuntimeError("network lost"))
    assert len(events) == 1
    assert str(events[0]) == "network lost"


async def test_reports_a_useful_error_when_the_websocket_omits_failure_details():
    webSockets = socketFactory()
    opening = asyncio.ensure_future(
        createRadiusClientTransportFactory(
            {
                "serverId": serverId,
                "auth": RadiusRelayAuthResolver({"type": "token", "token": "secret"}),
                "webSocketFactory": webSockets.factory,
            }
        )(SimpleNamespace(onData=lambda chunk: None, onClose=lambda: None, onError=lambda error: None))
    )
    await waitFor(lambda: len(webSockets.sockets) == 1)
    webSockets.sockets[0].socket.fail(RuntimeError(""))
    with pytest.raises(Exception, match="Radius WebSocket connection failed"):
        await opening


async def test_rejects_an_unexpected_websocket_protocol():
    webSockets = socketFactory()
    opening = asyncio.ensure_future(
        createRadiusClientTransportFactory(
            {
                "serverId": serverId,
                "auth": RadiusRelayAuthResolver({"type": "token", "token": "secret"}),
                "webSocketFactory": webSockets.factory,
            }
        )(SimpleNamespace(onData=lambda chunk: None, onClose=lambda: None, onError=lambda error: None))
    )
    await waitFor(lambda: len(webSockets.sockets) == 1)
    webSockets.sockets[0].socket.open("something-else")
    with pytest.raises(Exception, match="unexpected WebSocket protocol"):
        await opening


async def test_reconnects_an_established_client_and_restores_its_selected_session():
    timers = FakeTimers()
    connectionListeners: list[Callable[[dict[str, str]], None]] = []
    attachmentListeners: list[Callable[[Any], None]] = []
    attempts = {"count": 0}
    reattached: list[str] = []

    class Client:
        connected = True
        connectionState = "connected"
        attachment: ClassVar[Any] = {"sessionId": "demo-1"}

        def onConnectionStateChange(self, listener: Callable[[dict[str, str]], None]) -> Callable[[], None]:
            connectionListeners.append(listener)
            return lambda: connectionListeners.remove(listener)

        def onAttachmentChange(self, listener: Callable[[Any], None]) -> Callable[[], None]:
            attachmentListeners.append(listener)
            return lambda: attachmentListeners.remove(listener)

        async def reconnect(self) -> dict[str, str]:
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise RuntimeError("temporary failure")
            self.connected = True
            self.connectionState = "connected"
            for listener in list(connectionListeners):  # noqa: PERF101 - a listener may remove itself
                listener({"state": "connected"})
            return {"serverId": serverId}

        def disconnect(self, reason: str = "") -> None:
            self.connected = False
            self.connectionState = "disconnected"
            for listener in list(connectionListeners):  # noqa: PERF101 - a listener may remove itself
                listener({"state": "disconnected"})

    client = Client()

    async def reattach(sessionId: str) -> None:
        reattached.append(sessionId)

    reconnect = RadiusClientReconnect(client, reattach, timers)
    client.disconnect()

    await timers.advanceAsync(1_000)
    assert attempts["count"] == 1
    await timers.advanceAsync(2_000)
    assert attempts["count"] == 2
    assert reattached == ["demo-1"]
    await reconnect.dispose()


async def test_the_default_websocket_adapter_talks_to_a_local_server():
    from websockets.asyncio.server import serve

    async def handler(connection: Any) -> None:
        async for message in connection:
            await connection.send(message if isinstance(message, bytes) else message.upper())

    async with serve(handler, "127.0.0.1", 0, subprotocols=["pi-session-relay.client.v1"]) as server:
        port = server.sockets[0].getsockname()[1]
        socket = defaultWebSocketFactory(
            {"url": f"ws://127.0.0.1:{port}", "protocol": "pi-session-relay.client.v1", "authorization": "Bearer x"}
        )
        events: list[tuple[str, Any]] = []
        for type in ("open", "message", "close", "error"):
            socket.addEventListener(type, lambda event, type=type: events.append((type, event)))
        await waitFor(lambda: [kind for kind, _event in events] == ["open"])
        assert socket.protocol == "pi-session-relay.client.v1"
        assert socket.readyState == 1
        socket.send(bytes([1, 2, 3]))
        socket.send("text")
        await waitFor(lambda: len([event for kind, event in events if kind == "message"]) == 2)
        assert [event["data"] for kind, event in events if kind == "message"] == [bytes([1, 2, 3]), "TEXT"]
        with pytest.raises(ValueError, match="Invalid close code"):
            socket.close(1002, "no")
        socket.close(1000, "done")
        await waitFor(lambda: any(kind == "close" for kind, _event in events))
        assert next(event for kind, event in events if kind == "close")["code"] == 1000
        assert socket.readyState == 3
        with pytest.raises(RuntimeError, match="socket is not open"):
            socket.send("late")


async def test_the_default_websocket_adapter_reports_a_refused_connection_as_error_and_close():
    socket = defaultWebSocketFactory(
        {"url": "ws://127.0.0.1:9", "protocol": "pi-session-relay.client.v1", "authorization": "Bearer x"}
    )
    events: list[str] = []
    for type in ("open", "close", "error"):
        socket.addEventListener(type, lambda event, type=type: events.append(type))
    await waitFor(lambda: "close" in events, 10_000)
    assert events == ["error", "close"]


async def test_the_default_adapter_delivers_a_frame_that_arrives_with_the_handshake_to_the_host():
    from websockets.asyncio.server import serve

    replies: list[str] = []

    async def handler(connection: Any) -> None:
        # Sent before the host could attach its listeners to the opened socket.
        await connection.send(json.dumps({"version": 1, "type": "ping"}))
        async for message in connection:
            replies.append(message)

    async with serve(handler, "127.0.0.1", 0, subprotocols=["pi-session-relay.host.v1"]) as server:
        port = server.sockets[0].getsockname()[1]

        async def resolve(_options: Any) -> dict[str, str]:
            return {"gateway": f"http://127.0.0.1:{port}", "token": "t"}

        host = RadiusRelayHost(
            {
                "serverId": serverId,
                "server": SimpleNamespace(accept=lambda connection: None),
                "auth": SimpleNamespace(resolve=resolve),
            }
        )
        host.start()
        try:
            await waitFor(lambda: len(replies) == 1)
            assert json.loads(replies[0]) == {"version": 1, "type": "pong"}
        finally:
            await host.close()


async def test_the_default_adapter_reports_a_connection_dropped_without_a_close_frame_as_error_then_close():
    from websockets.asyncio.server import serve

    async def handler(connection: Any) -> None:
        connection.transport.abort()

    async with serve(handler, "127.0.0.1", 0, subprotocols=["pi-session-relay.client.v1"]) as server:
        port = server.sockets[0].getsockname()[1]
        socket = defaultWebSocketFactory(
            {"url": f"ws://127.0.0.1:{port}", "protocol": "pi-session-relay.client.v1", "authorization": "Bearer x"}
        )
        events: list[str] = []
        for type in ("open", "message", "close", "error"):
            socket.addEventListener(type, lambda event, type=type: events.append(type))
        await waitFor(lambda: "close" in events, 10_000)
        assert events == ["open", "error", "close"]


async def test_the_default_adapter_drops_messages_once_it_is_closing():
    from websockets.asyncio.server import serve

    async def handler(connection: Any) -> None:
        async for _message in connection:
            await connection.send("one")
            await connection.send("two")

    async with serve(handler, "127.0.0.1", 0, subprotocols=["pi-session-relay.client.v1"]) as server:
        port = server.sockets[0].getsockname()[1]
        socket = defaultWebSocketFactory(
            {"url": f"ws://127.0.0.1:{port}", "protocol": "pi-session-relay.client.v1", "authorization": "Bearer x"}
        )
        events: list[tuple[str, Any]] = []
        for type in ("open", "message", "close", "error"):
            socket.addEventListener(type, lambda event, type=type: events.append((type, event)))
        await waitFor(lambda: [kind for kind, _event in events] == ["open"])
        socket.send("go")
        socket.close(1000, "done")
        await waitFor(lambda: any(kind == "close" for kind, _event in events))
        assert [kind for kind, _event in events if kind == "message"] == []


def test_a_host_control_message_is_parsed_like_json_parse_and_validated():
    from misaka.experimental.radius_relay import parseHostControlMessage

    def closeMessage(code: str) -> str:
        return f'{{"version":1,"type":"connection_close","connection_id":"{connectionId}"{code}}}'

    assert parseHostControlMessage(closeMessage("")) == {"version": 1, "type": "connection_close", "connection_id": connectionId}
    assert parseHostControlMessage(closeMessage(',"code":1000.0'))["code"] == 1000
    assert parseHostControlMessage(closeMessage(',"code":1e3'))["code"] == 1000
    for invalid in (',"code":null', ',"code":999', ',"code":5000', ',"code":1000.5', ',"code":true', ',"code":"1000"'):
        with pytest.raises(RuntimeError, match="Invalid Radius relay control message"):
            parseHostControlMessage(closeMessage(invalid))
    for constant in ("NaN", "Infinity", "-Infinity"):
        with pytest.raises(RuntimeError, match="Invalid Radius relay control message"):
            parseHostControlMessage(f'{{"version":1,"type":"ping","x":{constant}}}')
