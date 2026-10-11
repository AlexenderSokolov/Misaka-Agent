"""pi server's ``test/protocol.test.ts``."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest

from misaka.protocol import (
    PROTOCOL_VERSION,
    encodeCbor,
    encodeClientMessage,
    encodeFrame,
)
from misaka.server import Server
from misaka.server.testing import ProtocolTestClient, TestServerHost
from tests.chord_helpers import matches
from tests.server_helpers import connect as connectMemory

SERVER_ID = "00000000-0000-4000-8000-000000000001"


@pytest.fixture
async def holder() -> AsyncIterator[dict[str, Server]]:
    state: dict[str, Server] = {}
    yield state
    if "server" in state:
        await state["server"].close()


def connect(holder: dict[str, Server]) -> ProtocolTestClient:
    holder["server"] = Server(TestServerHost(), {"listeners": [], "serverId": SERVER_ID})
    return connectMemory(holder["server"], idempotentClose=False)


async def test_requires_hello_as_the_first_message(holder):
    client = connect(holder)
    await client.sendMessage(
        {
            "type": "request",
            "id": "request-1",
            "target": {"serverId": SERVER_ID},
            "call": {"serviceId": "pi.session-directory", "member": "list", "args": []},
        }
    )
    assert matches(
        await client.next(lambda message: message["type"] == "hello_error"),
        {"type": "hello_error", "error": {"code": "invalid_request"}},
    )
    await client.waitForClose()


async def test_rejects_unsupported_protocol_versions(holder):
    client = connect(holder)
    assert matches(
        await client.hello(PROTOCOL_VERSION + 1), {"type": "hello_error", "error": {"code": "version"}}
    )
    await client.waitForClose()


async def test_accepts_fragmented_hello_and_request_frames(holder):
    client = connect(holder)
    hello = encodeClientMessage({"type": "hello", "version": PROTOCOL_VERSION})
    helloResponse = client.next(lambda message: message["type"] == "hello")
    await client.sendFragmentedMessage({"type": "hello", "version": PROTOCOL_VERSION}, len(hello) // 2)
    assert matches(await helloResponse, {"type": "hello", "serverId": SERVER_ID})

    response = client.next(lambda message: message["type"] == "response")
    request = {
        "type": "request",
        "id": "request-1",
        "target": {"serverId": SERVER_ID},
        "call": {"serviceId": "pi.session-directory", "member": "list", "args": []},
    }
    frame = encodeClientMessage(request)
    await client.sendFragmentedMessage(request, len(frame) // 2)
    assert matches(await response, {"ok": False, "error": {"code": "internal_error"}})


@pytest.mark.parametrize(
    ("label", "data"),
    [
        ("malformed CBOR", encodeFrame(bytes([0xFF]))),
        ("schema-invalid CBOR", encodeFrame(encodeCbor({"type": "hello", "version": 1, "extra": True}))),
        ("oversized frame", bytes([1, 0, 0, 1])),
    ],
)
async def test_rejects_hostile_framed_input(holder, label, data):
    client = connect(holder)
    await client.sendBytes(data)
    assert matches(
        await client.next(lambda message: message["type"] == "hello_error"),
        {"type": "hello_error", "error": {"code": "invalid_request"}},
    )
    await client.waitForClose()


async def test_rejects_a_second_hello_after_completing_the_handshake(holder):
    client = connect(holder)
    await client.hello()
    await client.sendMessage({"type": "hello", "version": PROTOCOL_VERSION})
    message = await client.next(lambda message: message["type"] == "hello_error")
    assert matches(message, {"type": "hello_error", "error": {"code": "invalid_request"}})
    assert "first message" in message["error"]["message"]
    await client.waitForClose()


async def test_processes_a_hello_and_request_coalesced_in_one_byte_chunk(holder):
    client = connect(holder)
    hello = encodeClientMessage({"type": "hello", "version": PROTOCOL_VERSION})
    request = encodeClientMessage(
        {
            "type": "request",
            "id": "request-1",
            "target": {"serverId": SERVER_ID},
            "call": {"serviceId": "pi.session-directory", "member": "list", "args": []},
        }
    )

    await client.sendBytes(hello + request)
    assert matches(await client.next(lambda message: message["type"] == "hello"), {"type": "hello"})
    assert matches(
        await client.next(lambda message: message["type"] == "response"),
        {"type": "response", "id": "request-1", "ok": False, "error": {"code": "internal_error"}},
    )


async def test_reports_a_truncated_final_frame_when_the_peer_closes(holder):
    errors: list[Exception] = []
    holder["server"] = Server(
        TestServerHost(), {"listeners": [], "serverId": SERVER_ID, "onError": errors.append}
    )
    state: dict[str, Any] = {"closed": False}

    class Connection:
        @property
        def closed(self) -> bool:
            return state["closed"]

        async def send(self, chunk: bytes) -> None:
            pass

        def close(self, finalChunk: bytes | None = None) -> None:
            state["closed"] = True

    handler = holder["server"].accept(Connection())
    handler.onData(bytes([0, 0, 0, 2, 1]))
    handler.onClose()

    assert state["closed"] is False
    assert len(errors) == 1
    assert "truncated" in str(errors[0]).lower()
