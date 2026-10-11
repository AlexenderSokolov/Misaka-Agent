"""pi client's ``test/unix-transport.test.ts``."""

from __future__ import annotations

import asyncio
import errno
import shutil
import sys
import tempfile
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path

import pytest

from misaka.chord.services.wire import parseServiceCall
from misaka.client import Client
from misaka.client.unix import createUnixTransportFactory
from misaka.protocol import PROTOCOL_VERSION, ClientMessageDecoder, encodeServerMessage
from tests.chord_helpers import matches

serverId = "00000000-0000-4000-8000-000000000001"

posixOnly = pytest.mark.skipif(sys.platform == "win32", reason="Unix sockets are POSIX-only")


class UnixEnvironment:
    def __init__(self) -> None:
        self.tempDirectories: set[str] = set()
        self.servers: set[asyncio.AbstractServer] = set()
        self.sockets: set[asyncio.StreamWriter] = set()

    def makeSocketPath(self) -> str:
        directory = tempfile.mkdtemp(prefix="pi-client-transport-", dir="/tmp")
        self.tempDirectories.add(directory)
        return str(Path(directory) / "pi.sock")

    async def listen(self, path: str, handle: Callable[[asyncio.StreamReader, asyncio.StreamWriter], Awaitable[None]]) -> None:
        async def connection(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            self.sockets.add(writer)
            try:
                await handle(reader, writer)
            finally:
                writer.transport.abort()
                self.sockets.discard(writer)

        self.servers.add(await asyncio.start_unix_server(connection, path))

    async def close(self) -> None:
        for socket in list(self.sockets):
            socket.transport.abort()
        self.sockets.clear()
        for server in self.servers:
            server.close()
            await server.wait_closed()
        self.servers.clear()
        for directory in self.tempDirectories:
            shutil.rmtree(directory, ignore_errors=True)
        self.tempDirectories.clear()


@pytest.fixture
async def unix() -> AsyncIterator[UnixEnvironment]:
    environment = UnixEnvironment()
    yield environment
    await environment.close()


def test_rejects_invalid_unix_transport_options():
    with pytest.raises(TypeError, match="must not be empty"):
        createUnixTransportFactory({"path": ""})
    with pytest.raises(TypeError, match="positive"):
        createUnixTransportFactory({"path": "/tmp/pi.sock", "maxPendingBytes": 0})


@posixOnly
class TestCreateUnixTransportFactory:
    async def test_carries_a_complete_client_handshake_and_request_over_a_real_unix_socket(self, unix):
        path = unix.makeSocketPath()
        receivedMembers: list[str] = []

        async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            decoder = ClientMessageDecoder()
            while chunk := await reader.read(65536):
                for message in decoder.push(chunk):
                    if message["type"] == "hello":
                        frame = encodeServerMessage({"type": "hello", "version": PROTOCOL_VERSION, "serverId": serverId})
                        for byte in frame:
                            writer.write(bytes([byte]))
                        continue
                    if message["type"] == "cancel":
                        continue
                    call = parseServiceCall(message["call"])
                    receivedMembers.append(f"{call['serviceId']}.{call['member']}")
                    frame = encodeServerMessage({"type": "response", "id": message["id"], "ok": True, "result": []})
                    split = len(frame) // 2
                    writer.write(frame[:split])
                    writer.write(frame[split:])
                await writer.drain()

        await unix.listen(path, handle)
        client = Client({"serverId": serverId, "transportFactory": createUnixTransportFactory({"path": path})})

        try:
            assert matches(await client.connect(), {"serverId": serverId})
            assert await client.request({"serverId": serverId}, {"serviceId": "test.server", "member": "list", "args": []}) == []
            assert receivedMembers == ["test.server.list"]
        finally:
            await client.dispose()

    async def test_reports_truncated_final_frames_through_client(self, unix):
        path = unix.makeSocketPath()

        async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            decoder = ClientMessageDecoder()
            while chunk := await reader.read(65536):
                for message in decoder.push(chunk):
                    if message["type"] == "hello":
                        writer.write(encodeServerMessage({"type": "hello", "version": PROTOCOL_VERSION, "serverId": serverId}))
                    else:
                        writer.write(bytes([0, 0, 0, 2, 1]))
                        writer.close()
                        return

        await unix.listen(path, handle)
        client = Client({"serverId": serverId, "transportFactory": createUnixTransportFactory({"path": path})})

        try:
            await client.connect()
            with pytest.raises(Exception, match="(?i)truncated") as info:
                await client.request({"serverId": serverId}, {"serviceId": "test.server", "member": "list", "args": []})
            assert info.value.name == "ProtocolValidationError"  # type: ignore[attr-defined]
            assert client.connectionState == "disconnected"
        finally:
            await client.dispose()

    async def test_rejects_connection_attempts_to_missing_sockets(self, unix):
        path = unix.makeSocketPath()

        class Handlers:
            def onData(self, chunk: bytes) -> None:
                pass

            def onClose(self) -> None:
                pass

            def onError(self, error: Exception) -> None:
                pass

        with pytest.raises(OSError) as info:
            await createUnixTransportFactory({"path": path})(Handlers())  # type: ignore[misc]
        assert info.value.errno == errno.ENOENT
