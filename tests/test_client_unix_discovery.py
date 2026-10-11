"""pi client's ``test/unix.test.ts``: discovering local servers by probing their sockets."""

from __future__ import annotations

import asyncio
import errno
import os
import shutil
import stat
import sys
import tempfile
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from misaka.client.unix import discoverUnixServers
from misaka.server import Server
from misaka.server.testing import createTestServerServices
from misaka.server.transports.unix import createUnixListener
from tests.chord_helpers import waitFor

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="Unix sockets are POSIX-only")

FIXTURE = Path(__file__).parent / "fixtures" / "stale_socket_server.py"


class Environment:
    def __init__(self) -> None:
        self.tempDirectories: set[str] = set()
        self.servers: set[Server] = set()
        self.rawServers: set[asyncio.AbstractServer] = set()
        self.rawSockets: set[asyncio.StreamWriter] = set()
        self.children: set[asyncio.subprocess.Process] = set()

    def makeDirectory(self) -> str:
        directory = tempfile.mkdtemp(prefix="pc-", dir="/tmp")
        self.tempDirectories.add(directory)
        return directory

    async def startServer(self, directory: str, fileServerId: str, reportedServerId: str | None = None) -> Server:
        path = os.path.join(directory, f"{fileServerId}.sock")

        class Host:
            serverServices = createTestServerServices()

            async def resolveSession(self, sessionId: str, context: Any = None) -> Any:
                raise RuntimeError("unused")

            async def openSession(self, metadata: Any, context: Any = None) -> Any:
                raise RuntimeError("unused")

        server = Server(
            Host(), {"listeners": [createUnixListener({"path": path})], "serverId": reportedServerId or fileServerId}
        )
        self.servers.add(server)
        await server.start()
        return server

    async def startSilentSocket(self, path: str, connections: dict[str, int] | None = None) -> None:
        async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            self.rawSockets.add(writer)
            if connections is not None:
                connections["active"] += 1
                connections["maximum"] = max(connections["maximum"], connections["active"])
                connections["total"] += 1
            try:
                while await reader.read(65536):
                    pass
            finally:
                self.rawSockets.discard(writer)
                if connections is not None:
                    connections["active"] -= 1
                writer.transport.abort()

        self.rawServers.add(await asyncio.start_unix_server(handle, path))


def serverId(value: int) -> str:
    return f"00000000-0000-4000-8000-{value:012x}"


@pytest.fixture
async def unix() -> AsyncIterator[Environment]:
    environment = Environment()
    yield environment
    for child in environment.children:
        if child.returncode is None:
            child.kill()
        await child.wait()
    await asyncio.gather(*(server.close() for server in environment.servers))
    for socket in list(environment.rawSockets):
        socket.transport.abort()
    for rawServer in environment.rawServers:
        rawServer.close()
        await rawServer.wait_closed()
    for directory in environment.tempDirectories:
        shutil.rmtree(directory, ignore_errors=True)


class TestDiscoverUnixServers:
    async def test_returns_no_routes_when_the_server_directory_is_missing(self, unix):
        directory = os.path.join(unix.makeDirectory(), "missing")
        assert await discoverUnixServers({"directory": directory}) == []

    async def test_discovers_reachable_servers_in_server_id_order(self, unix):
        directory = unix.makeDirectory()
        first = serverId(1)
        second = serverId(2)
        await unix.startServer(directory, second)
        await unix.startServer(directory, first)

        assert await discoverUnixServers({"directory": directory}) == [
            {"serverId": first, "path": os.path.join(directory, f"{first}.sock")},
            {"serverId": second, "path": os.path.join(directory, f"{second}.sock")},
        ]

    async def test_ignores_malformed_entries_non_sockets_and_mismatched_servers(self, unix):
        directory = unix.makeDirectory()
        Path(directory, f"{serverId(1)}.sock").write_text("not a socket")
        Path(directory, "not-a-server.sock").write_text("ignored")
        os.mkdir(os.path.join(directory, f"{serverId(2)}.sock"))
        await unix.startServer(directory, serverId(3), serverId(4))

        assert await discoverUnixServers({"directory": directory}) == []

    async def test_ignores_stale_sockets_without_deleting_them(self, unix):
        directory = unix.makeDirectory()
        path = os.path.join(directory, f"{serverId(1)}.sock")
        child = await asyncio.create_subprocess_exec(
            sys.executable, str(FIXTURE), path, stdout=asyncio.subprocess.PIPE
        )
        unix.children.add(child)
        assert child.stdout is not None
        assert await child.stdout.readline() == b"listening\n"
        child.kill()
        await child.wait()
        unix.children.discard(child)

        assert await discoverUnixServers({"directory": directory}) == []
        assert stat.S_ISSOCK(os.lstat(path).st_mode)

    async def test_times_out_an_unresponsive_socket_without_deleting_it(self, unix):
        directory = unix.makeDirectory()
        path = os.path.join(directory, f"{serverId(1)}.sock")
        await unix.startSilentSocket(path)

        assert await discoverUnixServers({"directory": directory, "timeoutMs": 20}) == []
        assert stat.S_ISSOCK(os.lstat(path).st_mode)

    async def test_limits_concurrent_probes_to_16(self, unix):
        directory = unix.makeDirectory()
        connections = {"active": 0, "maximum": 0, "total": 0}
        for index in range(1, 21):
            await unix.startSilentSocket(os.path.join(directory, f"{serverId(index)}.sock"), connections)

        discovery = asyncio.ensure_future(discoverUnixServers({"directory": directory, "timeoutMs": 100}))
        await waitFor(lambda: connections["active"] == 16)
        assert connections["maximum"] == 16
        assert await discovery == []
        assert connections["total"] == 20

    async def test_ignores_an_endpoint_that_closes_before_its_handshake(self, unix):
        directory = unix.makeDirectory()
        path = os.path.join(directory, f"{serverId(1)}.sock")

        async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            writer.transport.abort()

        unix.rawServers.add(await asyncio.start_unix_server(handle, path))

        assert await discoverUnixServers({"directory": directory}) == []

    async def test_propagates_unexpected_filesystem_errors(self, unix):
        directory = unix.makeDirectory()
        file = os.path.join(directory, "not-a-directory")
        Path(file).write_text("content")

        with pytest.raises(OSError) as info:
            await discoverUnixServers({"directory": file})
        assert info.value.errno == errno.ENOTDIR
