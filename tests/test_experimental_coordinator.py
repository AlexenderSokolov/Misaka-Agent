"""The coordinator process's routing and public-connection handling, driven in-process over real Unix sockets."""

from __future__ import annotations

import asyncio
import json
import shutil
import sys
import tempfile
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

import pytest

from misaka.experimental import coordinator as coordinator_module
from misaka.experimental.coordinator import (
    COORDINATOR_PROTOCOL_VERSION,
    _CoordinatorProcess,
)
from tests.chat_support import waitFor

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="Unix sockets")


class Harness:
    def __init__(self, directory: str, coordinator: _CoordinatorProcess) -> None:
        self.directory = directory
        self.coordinator = coordinator
        self.writers: list[asyncio.StreamWriter] = []

    async def control(self, first: dict[str, Any]) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        reader, writer = await asyncio.open_unix_connection(self.coordinator.controlPath)
        self.writers.append(writer)
        writer.write((json.dumps(first) + "\n").encode())
        return reader, writer

    async def client(self) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        reader, writer = await asyncio.open_unix_connection(self.coordinator.publicPath)
        self.writers.append(writer)
        return reader, writer

    async def server(self, endpoint: str) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        reader, writer = await self.control(
            {
                "type": "register_server",
                "protocol": COORDINATOR_PROTOCOL_VERSION,
                "serverConnectionId": f"conn-{endpoint}",
                "endpoint": endpoint,
            }
        )
        assert (await readLine(reader))["type"] == "server_registered"
        return reader, writer

    async def peer(self, peerId: str) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        reader, writer = await self.control(
            {"type": "register_peer", "protocol": COORDINATOR_PROTOCOL_VERSION, "peerId": peerId}
        )
        assert (await readLine(reader))["type"] == "peer_registered"
        return reader, writer


async def readLine(reader: asyncio.StreamReader) -> dict[str, Any]:
    return json.loads(await asyncio.wait_for(reader.readline(), 5))


@pytest.fixture
async def harness() -> AsyncIterator[Harness]:
    directory = tempfile.mkdtemp(prefix="co-", dir="/tmp")
    coordinator = _CoordinatorProcess(f"{directory}/public.sock", f"{directory}/control.sock")
    await coordinator.main()
    context = Harness(directory, coordinator)
    yield context
    for writer in context.writers:
        writer.close()
    if not coordinator.shuttingDown:
        await asyncio.wait_for(coordinator.shutdownCoordinator(), 5)
    shutil.rmtree(directory, ignore_errors=True)


async def test_a_message_without_a_payload_is_forwarded_without_one(harness: Harness) -> None:
    serverReader, _serverWriter = await harness.server("endpoint")
    _peerReader, peerWriter = await harness.peer("w1")
    assert (await readLine(serverReader))["type"] == "peer_connected"
    peerWriter.write(b'{"type":"send","to":"server"}\n')
    assert await readLine(serverReader) == {"type": "message", "from": "w1"}
    peerWriter.write(b'{"type":"send","to":"server","payload":null}\n')
    assert await readLine(serverReader) == {"type": "message", "from": "w1", "payload": None}


async def test_a_broadcast_without_a_payload_is_forwarded_without_one(harness: Harness) -> None:
    _serverReader, serverWriter = await harness.server("endpoint")
    peerReader, _peerWriter = await harness.peer("w1")
    serverWriter.write(b'{"type":"broadcast"}\n')
    line = await readLine(peerReader)
    while line["type"] != "message":
        line = await readLine(peerReader)
    assert line == {"type": "message", "from": "server"}


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
async def test_a_line_with_a_non_json_number_ends_the_connection(harness: Harness, constant: str) -> None:
    serverReader, _serverWriter = await harness.server("endpoint")
    peerReader, peerWriter = await harness.peer("w1")
    assert (await readLine(serverReader))["type"] == "peer_connected"
    peerWriter.write(f'{{"type":"send","to":"server","payload":{constant}}}\n'.encode())
    assert await asyncio.wait_for(peerReader.read(), 5) == b""
    assert (await readLine(serverReader))["type"] == "peer_disconnected"


class Gate:
    """Holds the coordinator's upstream connect in flight until released."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.release = asyncio.Event()
        self.started = asyncio.Event()
        real: Callable[[str], Awaitable[Any]] = coordinator_module.connectSocket

        async def gated(path: str) -> Any:
            self.started.set()
            await self.release.wait()
            return await real(path)

        monkeypatch.setattr(coordinator_module, "connectSocket", gated)


async def listenUnix(directory: str) -> tuple[asyncio.Server, str]:
    path = f"{directory}/upstream.sock"
    server = await asyncio.start_unix_server(lambda reader, writer: writer.close(), path)
    return server, path


async def test_a_client_that_hangs_up_during_the_upstream_connect_leaves_no_entry(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    upstream, path = await listenUnix(harness.directory)
    gate = Gate(monkeypatch)
    try:
        await harness.server(path)
        _reader, writer = await harness.client()
        await gate.started.wait()
        assert len(harness.coordinator.publicConnections) == 1
        writer.close()
        await waitFor(lambda: len(harness.coordinator.publicConnections) == 0)
        gate.release.set()
        await waitFor(lambda: len(harness.coordinator.tasks) == 0)
        assert len(harness.coordinator.publicConnections) == 0
    finally:
        gate.release.set()
        upstream.close()


async def test_a_replacement_cuts_a_client_whose_upstream_is_still_connecting(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    upstream, path = await listenUnix(harness.directory)
    gate = Gate(monkeypatch)
    try:
        await harness.server(path)
        reader, _writer = await harness.client()
        await gate.started.wait()
        await harness.server(f"{path}-replacement")
        assert await asyncio.wait_for(reader.read(), 5) == b""
        gate.release.set()
        await waitFor(lambda: len(harness.coordinator.tasks) == 0)
        assert len(harness.coordinator.publicConnections) == 0
    finally:
        gate.release.set()
        upstream.close()


async def test_a_shutdown_cuts_a_client_whose_upstream_is_still_connecting(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    upstream, path = await listenUnix(harness.directory)
    gate = Gate(monkeypatch)
    try:
        await harness.server(path)
        reader, _writer = await harness.client()
        await gate.started.wait()
        await asyncio.wait_for(harness.coordinator.shutdownCoordinator(), 5)
        assert await asyncio.wait_for(reader.read(), 5) == b""
    finally:
        gate.release.set()
        upstream.close()
