"""pi server's ``test/unix.test.ts``."""

from __future__ import annotations

import asyncio
import os
import shutil
import stat
import sys
import tempfile
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from misaka.server import Server
from misaka.server.testing import (
    ProtocolTestClient,
    TestServerHost,
    connectUnixTestClient,
)
from misaka.server.transports.unix import createUnixServer, getUnixSocketPath
from tests.chord_helpers import matches

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="Unix sockets are POSIX-only")

SERVER_ID = "00000000-0000-4000-8000-000000000001"
FIXTURE = Path(__file__).parent / "fixtures" / "stale_socket_server.py"


class Environment:
    def __init__(self) -> None:
        self.servers: set[Server] = set()
        self.clients: set[ProtocolTestClient] = set()
        self.children: set[asyncio.subprocess.Process] = set()
        self.tempDirectories: set[str] = set()

    def makeSocketPath(self, nested: bool = False) -> str:
        directory = tempfile.mkdtemp(prefix="ps-", dir="/tmp")
        self.tempDirectories.add(directory)
        return os.path.join(directory, "p", "n", "server.sock") if nested else os.path.join(directory, "server.sock")

    def makeServer(self, path: str) -> Server:
        server = createUnixServer(TestServerHost(), {"path": path, "serverId": SERVER_ID})
        self.servers.add(server)
        return server

    async def connect(self, path: str) -> ProtocolTestClient:
        client = await connectUnixTestClient(path)
        self.clients.add(client)
        return client


@pytest.fixture
async def unix() -> AsyncIterator[Environment]:
    environment = Environment()
    yield environment
    for child in environment.children:
        if child.returncode is None:
            child.kill()
        await child.wait()
    environment.children.clear()
    await asyncio.gather(*(client.close() for client in environment.clients))
    environment.clients.clear()
    await asyncio.gather(*(server.close() for server in environment.servers))
    environment.servers.clear()
    for directory in environment.tempDirectories:
        shutil.rmtree(directory, ignore_errors=True)
    environment.tempDirectories.clear()


async def test_creates_an_in_memory_server_id_and_derives_its_explicit_unix_socket_path(unix):
    directory = tempfile.mkdtemp(prefix="pi-server-", dir="/tmp")
    unix.tempDirectories.add(directory)
    path = getUnixSocketPath(SERVER_ID, directory)

    assert path == os.path.join(directory, f"{SERVER_ID}.sock")
    first = createUnixServer(TestServerHost(), {"serverId": SERVER_ID, "path": path})
    unix.servers.add(first)
    await first.start()
    firstClient = await unix.connect(path)
    assert matches(await firstClient.hello(), {"serverId": SERVER_ID})
    await firstClient.close()
    unix.clients.discard(firstClient)
    await first.close()
    unix.servers.discard(first)

    replacement = createUnixServer(TestServerHost(), {"serverId": SERVER_ID, "path": path})
    unix.servers.add(replacement)
    await replacement.start()
    replacementClient = await unix.connect(path)
    assert matches(await replacementClient.hello(), {"serverId": SERVER_ID})


class TestUnixListenerFilesystemLifecycle:
    async def test_rejects_a_live_listener_without_unlinking_it(self, unix):
        path = unix.makeSocketPath()
        first = unix.makeServer(path)
        await first.start()
        firstIdentity = os.lstat(path)

        second = unix.makeServer(path)
        with pytest.raises(Exception, match="already running"):
            await second.start()
        currentIdentity = os.lstat(path)
        assert stat.S_ISSOCK(currentIdentity.st_mode)
        assert (currentIdentity.st_dev, currentIdentity.st_ino) == (firstIdentity.st_dev, firstIdentity.st_ino)

        client = await unix.connect(path)
        assert matches(await client.hello(), {"type": "hello"})

    async def test_never_unlinks_a_regular_file_at_the_configured_path(self, unix):
        path = unix.makeSocketPath()
        Path(path).write_text("do not remove")
        os.chmod(path, 0o640)
        server = unix.makeServer(path)
        with pytest.raises(Exception, match="non-socket"):
            await server.start()
        assert Path(path).read_text() == "do not remove"

    async def test_creates_nested_temp_parents_restricts_permissions_and_removes_its_own_socket(self, unix):
        path = unix.makeSocketPath(nested=True)
        server = unix.makeServer(path)
        await server.start()
        stats = os.lstat(path)
        assert stat.S_ISSOCK(stats.st_mode)
        assert stats.st_mode & 0o777 == 0o600
        assert os.listdir(os.path.dirname(path)) == ["server.sock"]

        await server.close()
        assert not os.path.lexists(path)

    async def test_does_not_remove_a_replacement_inode_during_shutdown(self, unix):
        path = unix.makeSocketPath()
        server = unix.makeServer(path)
        await server.start()
        os.unlink(path)
        Path(path).write_text("replacement")

        closing = server.close()
        assert Path(path).read_text() == "replacement"
        await closing
        assert Path(path).read_text() == "replacement"

    async def test_removes_a_genuinely_stale_socket_before_binding(self, unix):
        path = unix.makeSocketPath()
        child = await asyncio.create_subprocess_exec(
            sys.executable, str(FIXTURE), path, stdout=asyncio.subprocess.PIPE
        )
        unix.children.add(child)
        assert child.stdout is not None
        assert await child.stdout.readline() == b"listening\n"
        staleIdentity = os.lstat(path)
        assert stat.S_ISSOCK(staleIdentity.st_mode)
        child.kill()
        await child.wait()
        unix.children.discard(child)

        server = unix.makeServer(path)
        await server.start()
        liveIdentity = os.lstat(path)
        assert stat.S_ISSOCK(liveIdentity.st_mode)
        client = await unix.connect(path)
        assert matches(await client.hello(), {"type": "hello"})
