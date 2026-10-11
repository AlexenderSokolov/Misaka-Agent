"""pi client's ``test/client.test.ts``."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from misaka.chord.context import BACKGROUND_CONTEXT
from misaka.client import (
    Client,
    ClientDisposedError,
    DisconnectedError,
    ServerError,
    connectClient,
    createClientServiceTransport,
)
from misaka.protocol import (
    PROTOCOL_VERSION,
    ProtocolValidationError,
    encodeCbor,
    encodeFrame,
    encodeServerMessage,
)
from misaka.utils.js import AbortController, startTask
from tests.chord_helpers import matches, waitFor
from tests.client_helpers import MemoryByteServer

serverId = "00000000-0000-4000-8000-000000000001"
serverTarget = {"serverId": serverId}


async def connect(server: MemoryByteServer, expectedServerId: str = serverId) -> Client:
    return await connectClient(
        {"serverId": expectedServerId, "transportFactory": lambda handlers: server.connect(handlers)}
    )


async def attachClient(client: Client, server: MemoryByteServer, sessionId: str) -> None:
    expectedMessages = len(server.messages) + 1
    attaching = client.request(
        serverTarget, {"serviceId": "pi.session-management", "member": "attach", "args": [sessionId]}
    )
    await server.waitForMessages(expectedMessages)
    request = server.messages[-1]
    if request["type"] != "request":
        raise RuntimeError("Missing attach request")
    server.send(
        {
            "type": "attachment",
            "attachment": {"serverId": serverId, "sessionId": sessionId, "attachmentId": f"attachment-{sessionId}"},
        }
    )
    server.send({"type": "response", "id": request["id"], "ok": True, "result": None})
    await attaching


def test_requires_a_canonical_uuidv4_server_identity():
    with pytest.raises(TypeError, match="serverId"):
        Client({"serverId": "invalid-server", "transportFactory": lambda handlers: asyncio.sleep(0)})


class TestClientServiceOperations:
    async def test_connects_only_to_the_expected_logical_server(self):
        matching = MemoryByteServer()
        client = await connect(matching)
        assert matches(client.hello, {"serverId": serverId})
        await client.dispose()

        wrong = MemoryByteServer("00000000-0000-4000-8000-000000000002")
        with pytest.raises(ProtocolValidationError):
            await connect(wrong)

    async def test_updates_attachment_state_from_out_of_band_server_routing(self):
        server = MemoryByteServer()
        client = await connect(server)
        changes: list[str | None] = []
        client.onAttachmentChange(lambda attachment: changes.append(None if attachment is None else attachment["sessionId"]))

        await attachClient(client, server, "session-1")
        assert matches(client.attachment, {"sessionId": "session-1", "attachmentId": "attachment-session-1"})
        assert matches(
            server.messages[1],
            {
                "type": "request",
                "target": serverTarget,
                "call": {"serviceId": "pi.session-management", "member": "attach", "args": ["session-1"]},
            },
        )

        server.send({"type": "attachment", "attachment": None})
        assert client.attachment is None
        assert changes == ["session-1", None]
        await client.dispose()

    async def test_buffers_service_updates_until_the_subscription_snapshot_arrives(self):
        server = MemoryByteServer()
        client = await connect(server)
        await attachClient(client, server, "session-1")
        target = client.attachment
        assert target is not None
        transport = createClientServiceTransport(client, lambda: client.attachment)
        updates: list[dict[str, Any]] = []
        opening = startTask(
            transport.subscribe(
                "pi.models", "singleton", lambda update, context: updates.append(update), BACKGROUND_CONTEXT
            )
        )
        await server.waitForMessages(3)
        assert matches(
            server.messages[2],
            {
                "type": "request",
                "target": target,
                "call": {
                    "serviceId": "$chord.service",
                    "member": "subscribe",
                    "args": ["service-1", "pi.models", "singleton"],
                },
            },
        )
        server.send(
            {
                "type": "service_update",
                "subscriptionId": "service-1",
                "update": {"type": "state", "member": "state", "sequence": 1, "ops": [["s", ["revision"], 1]]},
            }
        )
        await asyncio.sleep(0)
        assert updates == []
        server.send(
            {
                "type": "response",
                "id": "request-2",
                "ok": True,
                "result": {
                    "serviceId": "pi.models",
                    "mode": "singleton",
                    "instances": [
                        {"members": [{"name": "state", "kind": "state", "sequence": 0, "ops": [["r", {"revision": 0}]]}]}
                    ],
                },
            }
        )
        subscription = await opening
        assert updates == []
        server.send(
            {
                "type": "service_update",
                "subscriptionId": "closed-subscription",
                "update": {"type": "state", "member": "state", "sequence": 99, "ops": [["s", 99, 99]]},
            }
        )
        await asyncio.sleep(0)
        assert client.connected is True
        assert subscription.snapshot["instances"][0]["members"] == [
            {"name": "state", "kind": "state", "sequence": 0, "ops": [["r", {"revision": 0}]]}
        ]
        subscription.activate()
        await waitFor(lambda: [update["type"] for update in updates] == ["state"])
        server.send(
            {
                "type": "service_update",
                "subscriptionId": "service-1",
                "update": {"type": "state", "member": "state", "sequence": 2, "ops": [["s", ["revision"], 2]]},
            }
        )
        server.send(
            {
                "type": "service_update",
                "subscriptionId": "service-1",
                "update": {
                    "type": "state",
                    "member": "state",
                    "sequence": 3,
                    "ops": [["#", 0, ["revision"]], ["s", 0, 3]],
                },
            }
        )
        await waitFor(lambda: len(updates) == 3)
        assert updates[2]["ops"] == [["s", ["revision"], 3]]

        disposing = subscription.close(BACKGROUND_CONTEXT)
        await server.waitForMessages(4)
        assert matches(
            server.messages[3],
            {"call": {"serviceId": "$chord.service", "member": "unsubscribe", "args": ["service-1"]}},
        )
        server.send({"type": "response", "id": "request-3", "ok": True})
        await disposing
        await client.dispose()

    async def test_correlates_out_of_order_generic_service_responses(self):
        server = MemoryByteServer()
        client = await connect(server)
        first = client.request(serverTarget, {"serviceId": "test", "member": "first", "args": []})
        second = client.request(serverTarget, {"serviceId": "test", "member": "second", "args": []})
        await server.waitForMessages(3)
        server.send({"type": "response", "id": "request-2", "ok": True, "result": "second"})
        server.send({"type": "response", "id": "request-1", "ok": True, "result": "first"})
        assert await asyncio.gather(first, second) == ["first", "second"]
        await client.dispose()

    async def test_exposes_bounded_server_errors(self):
        server = MemoryByteServer()
        client = await connect(server)
        pending = client.request(serverTarget, {"serviceId": "test", "member": "missing", "args": []})
        await server.waitForMessages(2)
        server.send(
            {
                "type": "response",
                "id": "request-1",
                "ok": False,
                "error": {"code": "session_not_found", "message": "Unknown session"},
            }
        )
        with pytest.raises(ServerError) as info:
            await pending
        assert info.value.code == "session_not_found"
        await client.dispose()

    async def test_does_not_send_a_pre_aborted_untyped_rpc_request(self):
        server = MemoryByteServer()
        client = await connect(server)
        controller = AbortController()
        reason = Exception("already cancelled")
        controller.abort(reason)

        with pytest.raises(Exception, match="already cancelled") as info:
            await client.request(serverTarget, {"serviceId": "test", "member": "noop", "args": []}, controller.signal)
        assert info.value is reason
        assert len(server.messages) == 1
        await client.dispose()

    async def test_cancels_one_untyped_rpc_request_without_disconnecting(self):
        server = MemoryByteServer()
        client = await connect(server)
        controller = AbortController()
        reason = Exception("stop this request")
        pending = client.request(
            serverTarget,
            {"serviceId": "test", "member": "mutate", "args": [{"value": 42}]},
            controller.signal,
        )
        await server.waitForMessages(2)
        assert matches(
            server.messages[1],
            {
                "type": "request",
                "id": "request-1",
                "target": serverTarget,
                "call": {"serviceId": "test", "member": "mutate", "args": [{"value": 42}]},
            },
        )

        controller.abort(reason)
        with pytest.raises(Exception, match="stop this request") as info:
            await pending
        assert info.value is reason
        await server.waitForMessages(3)
        assert server.messages[2] == {"type": "cancel", "id": "request-1", "target": serverTarget}
        server.send(
            {
                "type": "response",
                "id": "request-1",
                "ok": False,
                "error": {"code": "cancelled", "message": "cancelled"},
            }
        )
        assert client.connected is True
        await client.dispose()

    async def test_rejects_pending_requests_after_disconnect_or_disposal(self):
        server = MemoryByteServer()
        client = await connect(server)
        pending = client.request(serverTarget, {"serviceId": "test", "member": "pending", "args": []})
        server.disconnect()
        with pytest.raises(DisconnectedError):
            await pending
        await client.dispose()
        with pytest.raises(ClientDisposedError):
            await client.request(serverTarget, {"serviceId": "test", "member": "disposed", "args": []})


class TestClientConnectionLifecycle:
    async def test_rejects_server_data_delivered_before_the_client_hello_is_sent(self):
        counts = {"close": 0, "send": 0}

        class Transport:
            async def send(self, chunk: bytes) -> None:
                counts["send"] += 1

            def close(self) -> None:
                counts["close"] += 1

        def transportFactory(handlers: Any) -> Transport:
            handlers.onData(encodeServerMessage({"type": "hello", "version": PROTOCOL_VERSION, "serverId": serverId}))
            return Transport()

        client = Client({"serverId": serverId, "transportFactory": transportFactory})

        with pytest.raises(ProtocolValidationError) as info:
            await client.connect()
        assert info.value.name == "ProtocolValidationError"
        assert str(info.value) == "Received server data before the client hello was sent"
        assert client.connectionState == "disconnected"
        assert counts["send"] == 0
        # MISAKA fork: vitest's ``rejects`` matcher settles a few microtasks after the rejection, by when the transport
        # (returned to the factory's ``await``, a turn later) is closed; here the turn is given.
        await asyncio.sleep(0)
        assert counts["close"] == 1

    async def test_rejects_typed_handshake_errors_and_closes_the_transport(self):
        holder: dict[str, Any] = {}
        counts = {"close": 0}

        class Transport:
            async def send(self, chunk: bytes) -> None:
                holder["handlers"].onData(
                    encodeServerMessage(
                        {
                            "type": "hello_error",
                            "error": {"code": "version", "message": "Unsupported protocol version"},
                        }
                    )
                )

            def close(self) -> None:
                counts["close"] += 1

        def transportFactory(createdHandlers: Any) -> Transport:
            holder["handlers"] = createdHandlers
            return Transport()

        client = Client({"serverId": serverId, "transportFactory": transportFactory})

        with pytest.raises(ServerError) as info:
            await client.connect()
        assert info.value.name == "ServerError"
        assert info.value.code == "version"
        assert info.value.message == "Unsupported protocol version"
        assert client.connectionState == "disconnected"
        assert counts["close"] == 1

    async def test_rejects_pending_requests_and_reconnects_through_a_fresh_transport(self):
        first = MemoryByteServer()
        second = MemoryByteServer()
        connection = 0

        def transportFactory(handlers: Any) -> Any:
            nonlocal connection
            server = first if connection == 0 else second
            connection += 1
            return server.connect(handlers)

        client = Client({"serverId": serverId, "transportFactory": transportFactory})
        states: list[str] = []
        client.onConnectionStateChange(lambda change: states.append(change["state"]))
        await client.connect()
        await attachClient(client, first, "session-1")
        target = client.attachment
        if target is None:
            raise RuntimeError("Missing attachment")
        pending = client.request(target, {"serviceId": "test.session", "member": "run", "args": []})
        await first.waitForMessages(3)
        assert matches(first.messages[2], {"call": {"serviceId": "test.session", "member": "run"}})
        first.disconnect()

        with pytest.raises(DisconnectedError):
            await pending
        assert matches(await client.reconnect(), {"serverId": serverId})
        assert connection == 2
        assert client.connected is True
        assert len(second.messages) == 1
        assert states == ["connecting", "connected", "disconnected", "connecting", "connected"]
        await client.dispose()

    async def test_reports_transport_failures_without_leaving_requests_pending(self):
        server = MemoryByteServer()
        client = await connect(server)
        pending = client.request(serverTarget, {"serviceId": "test", "member": "pending", "args": []})
        await server.waitForMessages(2)
        failure = Exception("read failed")
        server.error(failure)

        with pytest.raises(DisconnectedError) as info:
            await pending
        assert info.value.name == "DisconnectedError"
        assert info.value.message == "read failed"
        assert info.value.cause is failure
        assert client.connectionState == "disconnected"

    async def test_disconnects_on_invalid_or_truncated_server_framing(self):
        invalidServer = MemoryByteServer()
        invalidClient = await connect(invalidServer)
        invalidServer.sendRaw(encodeFrame(encodeCbor({"type": "response", "id": "unknown", "ok": True, "result": 1})))
        assert invalidClient.connectionState == "disconnected"

        truncatedServer = MemoryByteServer()
        truncatedClient = await connect(truncatedServer)
        pending = truncatedClient.request(serverTarget, {"serviceId": "test", "member": "pending", "args": []})
        await truncatedServer.waitForMessages(2)
        truncatedServer.sendRaw(bytes([0, 0, 0, 2, 1]))
        truncatedServer.disconnect()

        with pytest.raises(ProtocolValidationError, match="(?i)truncated") as info:
            await pending
        assert info.value.name == "ProtocolValidationError"
        assert truncatedClient.connectionState == "disconnected"

    async def test_disconnects_when_a_response_has_no_matching_request(self):
        server = MemoryByteServer()
        client = await connect(server)
        server.send({"type": "response", "id": "unknown-request", "ok": True, "result": []})

        assert client.connectionState == "disconnected"
        assert server.clientCloseCount == 1


class TestClientListenersAndTurns:
    async def test_connection_listeners_are_a_live_set(self):
        server = MemoryByteServer()
        client = Client({"serverId": serverId, "transportFactory": lambda handlers: server.connect(handlers)})
        calls: list[str] = []
        removers: dict[str, Any] = {}

        def second(change: dict[str, Any]) -> None:
            calls.append(f"b {change['state']}")

        def first(change: dict[str, Any]) -> None:
            calls.append(f"a {change['state']}")
            # A listener removed before its turn is not called; one added during the pass is.
            removers["b"]()
            client.onConnectionStateChange(late)

        def late(change: dict[str, Any]) -> None:
            calls.append(f"late {change['state']}")

        client.onConnectionStateChange(first)
        removers["b"] = client.onConnectionStateChange(second)
        # The same function registered again is held once.
        client.onConnectionStateChange(first)
        await client.connect()
        assert calls == ["a connecting", "late connecting", "a connected", "late connected"]
        await client.dispose()

    async def test_an_attachment_listener_is_registered_once(self):
        server = MemoryByteServer()
        client = await connect(server)
        seen: list[Any] = []
        listener = seen.append
        client.onAttachmentChange(listener)
        client.onAttachmentChange(listener)
        await attachClient(client, server, "session-1")
        assert len(seen) == 1
        await client.dispose()

    async def test_connect_returns_while_a_synchronous_transport_is_still_being_opened(self):
        server = MemoryByteServer()
        client = Client({"serverId": serverId, "transportFactory": lambda handlers: server.connect(handlers)})
        connecting = client.connect()
        assert client.connectionState == "connecting"
        assert server.messages == []
        client.disconnect()
        with pytest.raises(DisconnectedError):
            await connecting
        assert client.connectionState == "disconnected"
        await client.dispose()

    async def test_service_members_run_when_they_are_called(self):
        server = MemoryByteServer()
        client = await connect(server)
        await attachClient(client, server, "session-1")
        target = client.attachment
        assert target is not None
        before = len(server.messages)
        client.serviceCatalogue(target)
        client.request(target, {"serviceId": "test.service", "member": "run", "args": []})
        # Both frames are sent now, in call order, without being awaited.
        await server.waitForMessages(before + 2)
        assert server.messages[before]["call"]["serviceId"] == "$chord.service"
        assert server.messages[before]["call"]["member"] == "catalogue"
        assert server.messages[before + 1]["call"]["serviceId"] == "test.service"
        await client.dispose()

    async def test_abandoning_the_wait_for_a_subscription_dispose_keeps_the_delivery_chain(self):
        server = MemoryByteServer()
        client = await connect(server)
        await attachClient(client, server, "session-1")
        log: list[str] = []
        deliveries = [0]

        async def listener(update: dict[str, Any], context: Any) -> None:
            index = deliveries[0]
            deliveries[0] += 1
            log.append(f"start {index}")
            await asyncio.sleep(0.05)
            log.append(f"end {index}")

        transport = createClientServiceTransport(client, lambda: client.attachment)
        opening = startTask(transport.subscribe("pi.models", "singleton", listener, BACKGROUND_CONTEXT))
        await server.waitForMessages(3)
        server.send(
            {
                "type": "response",
                "id": "request-2",
                "ok": True,
                "result": {
                    "serviceId": "pi.models",
                    "mode": "singleton",
                    "instances": [
                        {"members": [{"name": "state", "kind": "state", "sequence": 0, "ops": [["r", {"revision": 0}]]}]}
                    ],
                },
            }
        )
        subscription = await opening
        subscription.activate()
        for sequence in (1, 2):
            server.send(
                {
                    "type": "service_update",
                    "subscriptionId": "service-1",
                    "update": {"type": "state", "member": "state", "sequence": sequence, "ops": [["s", ["revision"], sequence]]},
                }
            )
        await waitFor(lambda: "start 0" in log)
        closing = subscription.close(BACKGROUND_CONTEXT)
        await server.waitForMessages(4)
        server.send({"type": "response", "id": "request-3", "ok": True})
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(closing, 0.01)
        await waitFor(lambda: log == ["start 0", "end 0", "start 1", "end 1"])
        await client.dispose()
