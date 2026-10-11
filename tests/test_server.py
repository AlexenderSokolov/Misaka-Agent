"""pi server's ``test/server.test.ts``."""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from collections.abc import AsyncIterator
from typing import Any

import pytest

from misaka.protocol import ServerMessageDecoder
from misaka.server import Server
from misaka.server.testing import TestServerHost
from misaka.server.transports.unix import createUnixServer
from tests.chord_helpers import matches

SERVER_ID = "00000000-0000-4000-8000-000000000001"


@pytest.fixture
def host() -> TestServerHost:
    return TestServerHost()


@pytest.fixture
async def environment() -> AsyncIterator[dict[str, Any]]:
    state: dict[str, Any] = {"server": None, "directory": None}
    yield state
    if state["server"] is not None:
        await state["server"].close()
    if state["directory"] is not None:
        import shutil

        shutil.rmtree(state["directory"], ignore_errors=True)


def makeSocketPath(environment: dict[str, Any]) -> str:
    environment["directory"] = tempfile.mkdtemp(prefix="pss-")
    return os.path.join(environment["directory"], "server.sock")


def test_requires_explicit_listeners_and_a_canonical_uuidv4_server_identity(host):
    for options, pattern in [
        ({}, "listeners"),
        ({"listeners": [], "serverId": ""}, "serverId"),
        ({"listeners": [], "serverId": "invalid-server"}, "serverId"),
    ]:
        with pytest.raises(TypeError, match=pattern):
            Server(host, options)  # type: ignore[arg-type]


@pytest.mark.skipif(sys.platform == "win32", reason="Unix sockets are POSIX-only")
async def test_rejects_concurrent_start_calls_without_leaking_the_unix_listener(host, environment):
    path = makeSocketPath(environment)
    environment["server"] = server = createUnixServer(host, {"path": path, "serverId": SERVER_ID})
    starting = server.start()
    with pytest.raises(Exception, match="starting"):
        await server.start()
    await starting
    await server.close()
    assert not os.path.lexists(path)


async def test_handshake_timeout_closes_with_a_final_hello_error_frame(host):
    closed: asyncio.Future[None] = asyncio.get_running_loop().create_future()

    class TimedOutConnection:
        def __init__(self) -> None:
            self.closed = False
            self.finalChunk: bytes | None = None

        async def send(self, chunk: bytes) -> None:
            raise RuntimeError("handshake timeout must use the terminal close frame")

        def close(self, finalChunk: bytes | None = None) -> None:
            self.finalChunk = finalChunk
            self.closed = True
            closed.set_result(None)

    core = Server(host, {"listeners": [], "serverId": SERVER_ID, "maxFrameLength": 1024, "handshakeTimeoutMs": 10})
    connection = TimedOutConnection()
    core.accept(connection)

    await closed
    assert connection.closed is True
    assert isinstance(connection.finalChunk, bytes)
    messages = ServerMessageDecoder().push(connection.finalChunk)
    assert matches(messages, [{"type": "hello_error", "error": {"code": "invalid_request"}}])
    await core.close()


def test_rejects_timeout_values_above_nodes_maximum_timer_delay(host):
    path = "/tmp/pi-server-timeout-test.sock"
    with pytest.raises(TypeError, match="handshakeTimeoutMs"):
        createUnixServer(host, {"path": path, "serverId": SERVER_ID, "handshakeTimeoutMs": 2_147_483_648})
    with pytest.raises(TypeError, match="gracefulCloseTimeoutMs"):
        createUnixServer(host, {"path": path, "serverId": SERVER_ID, "gracefulCloseTimeoutMs": 2_147_483_648})


async def test_rejects_pending_byte_limits_smaller_than_one_maximum_frame(host, environment):
    path = makeSocketPath(environment)
    with pytest.raises(TypeError, match="maxPendingBytes"):
        createUnixServer(
            host, {"path": path, "serverId": SERVER_ID, "maxFrameLength": 128, "maxPendingBytes": 131}
        )


async def test_rejects_close_and_closed_when_listener_shutdown_fails(host):
    failure = Exception("listener close failed")

    class Listener:
        async def start(self, accept: Any) -> None:
            pass

        async def close(self) -> None:
            raise failure

    core = Server(host, {"listeners": [Listener()], "serverId": SERVER_ID})
    await core.start()

    with pytest.raises(Exception, match="listener close failed") as info:
        await core.close()
    assert info.value is failure
    with pytest.raises(Exception, match="listener close failed") as closedInfo:
        await core.closed
    assert closedInfo.value is failure
