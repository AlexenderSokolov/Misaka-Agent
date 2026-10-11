"""Regression tests for what the adversarial review of the protocol, client and server ports found."""

from __future__ import annotations

import array
import asyncio
import os
import sys

import pytest

from misaka.protocol import (
    ClientMessageDecoder,
    FrameDecoder,
    decodeCbor,
    encodeCbor,
    encodeClientMessage,
)
from misaka.protocol.cbor import CborError
from misaka.server.testing import ProtocolTestClient, createTestServer
from misaka.utils.js import UNDEFINED
from tests.chat_support import waitFor


@pytest.mark.parametrize("empty", [None, UNDEFINED])
def test_an_absent_cbor_limit_keeps_the_default(empty):
    assert encodeCbor(1, {"maxDepth": empty}) == b"\x01"
    assert encodeCbor(1, {"maxByteLength": empty, "maxContainerLength": empty}) == b"\x01"
    assert decodeCbor(b"\x01", {"maxContainerLength": empty}) == 1


@pytest.mark.parametrize("empty", [None, UNDEFINED])
def test_an_absent_frame_limit_keeps_the_default(empty):
    FrameDecoder({"maxFrameLength": empty})
    ClientMessageDecoder({"maxFrameLength": empty})
    assert encodeClientMessage({"type": "hello", "version": 8}, {"maxFrameLength": empty})


def test_a_wide_memoryview_counts_bytes_against_the_limit():
    wide = memoryview(array.array("I", [0x01020304]))
    with pytest.raises(CborError, match="exceeds configured limit of 4"):
        encodeCbor(wide, {"maxByteLength": 4})
    assert len(encodeCbor(memoryview(bytes(3)), {"maxByteLength": 4})) == 4


def test_the_length_limit_is_judged_before_a_lone_surrogate_is():
    with pytest.raises(CborError, match="text string length exceeds configured limit of 2"):
        encodeCbor(chr(0xD800), {"maxByteLength": 2})
    with pytest.raises(CborError, match="valid Unicode scalar values"):
        encodeCbor(chr(0xD800))


class RecordingListener:
    def __init__(self, closeDelay: float = 0) -> None:
        self.accept = None
        self.closeDelay = closeDelay
        self.closeCount = 0

    async def start(self, accept):
        self.accept = accept

    async def close(self):
        await asyncio.sleep(self.closeDelay)
        self.closeCount += 1


async def test_a_synchronous_listener_leaves_the_start_state_consistent():
    server = createTestServer({"listeners": [RecordingListener()]}).server
    first = server.start()
    with pytest.raises(RuntimeError, match="already starting"):
        await server.start()
    await first
    with pytest.raises(RuntimeError, match="already started"):
        await server.start()
    await server.close()
    with pytest.raises(RuntimeError, match="closing or closed"):
        await server.start()


async def test_abandoning_the_wait_for_closed_does_not_break_close():
    server = createTestServer({"listeners": [RecordingListener()]}).server
    await server.start()
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(server.closed, 0.01)
    await server.close()
    await server.closed


async def test_abandoning_the_wait_for_close_does_not_abort_the_shutdown():
    listener = RecordingListener(closeDelay=0.05)
    server = createTestServer({"listeners": [listener]}).server
    await server.start()
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(server.close(), 0.01)
    await server.closed
    assert listener.closeCount == 1
    await server.close()


async def test_abandoning_the_wait_for_start_does_not_abort_the_startup():
    class SlowListener(RecordingListener):
        async def start(self, accept):
            await asyncio.sleep(0.05)
            self.accept = accept

    first, second = RecordingListener(), SlowListener()
    server = createTestServer({"listeners": [first, second]}).server
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(server.start(), 0.01)
    await waitFor(lambda: second.accept is not None)
    await server.close()
    assert first.closeCount == 1 and second.closeCount == 1


def test_the_test_server_passes_what_it_was_given_to_the_server():
    with pytest.raises(TypeError, match="serverId must be a canonical lowercase UUIDv4"):
        createTestServer({"listeners": [], "serverId": ""})
    with pytest.raises(TypeError, match="listeners"):
        createTestServer({})


async def test_message_waiters_settle_in_the_order_they_registered():
    settled: list[int] = []

    class Channel:
        async def send(self, chunk: bytes) -> None:
            pass

        async def close(self) -> None:
            pass

    client = ProtocolTestClient(Channel())
    from misaka.protocol import encodeServerMessage

    waiters = [client.next(lambda message, index=index: (settled.append(index), True)[1]) for index in range(8)]
    client.receive(encodeServerMessage({"type": "hello", "version": 8, "serverId": "00000000-0000-4000-8000-000000000001"}))
    await asyncio.gather(*waiters)
    assert settled == list(range(8))


@pytest.mark.skipif(sys.platform == "win32", reason="Unix sockets")
class TestUnix:
    async def test_a_bare_relative_socket_path_starts(self, tmp_path, monkeypatch):
        from misaka.server.testing import TestServerHost
        from misaka.server.transports.unix import createUnixServer

        monkeypatch.chdir(tmp_path)
        server = createUnixServer(
            TestServerHost(), {"path": "server.sock", "serverId": "00000000-0000-4000-8000-000000000001"}
        )
        await server.start()
        try:
            assert os.path.exists(tmp_path / "server.sock")
        finally:
            await server.close()

    async def test_a_second_bind_of_the_private_path_fails_instead_of_replacing_the_first(self, tmp_path, monkeypatch):
        import socket

        from misaka.server.transports.unix import listener as listenerModule
        from misaka.server.transports.unix.listener import UnixListener

        first = UnixListener({"path": str(tmp_path / "a.sock")})
        await first.start(lambda connection: None)
        try:
            ownedBind = listenerModule.getOwnedBindPath(str(tmp_path / "b.sock"))
            squatter = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            squatter.bind(ownedBind)
            squatter.listen()
            # The private path is taken by another socket: removeStaleSocket would remove a stale one, so it is held live.
            monkeypatch.setattr(listenerModule, "removeStaleSocket", lambda path: asyncio.sleep(0))
            reported: list[Exception] = []
            second = UnixListener({"path": str(tmp_path / "b.sock"), "onError": reported.append})
            with pytest.raises(OSError, match="already in use|Address already in use"):
                await second.start(lambda connection: None)
            # The failure to listen is reported, as the server's ``error`` event is.
            assert len(reported) == 1
            squatter.close()
        finally:
            await first.close()

    async def test_maximum_pending_bytes_must_be_a_safe_integer(self):
        from misaka.client.unix import validateUnixTransportOptions
        from misaka.server.transports.unix.listener import resolveUnixListenerOptions

        with pytest.raises(TypeError, match="safe integer"):
            resolveUnixListenerOptions({"path": "/tmp/x.sock", "maxPendingBytes": 2**53})
        with pytest.raises(TypeError, match="safe integer"):
            validateUnixTransportOptions({"path": "/tmp/x.sock", "maxPendingBytes": 2**53})

    async def test_a_frame_sent_before_the_transport_is_closed_is_not_written(self, tmp_path):
        from misaka.client.unix import connectUnixSocket
        from misaka.server.transports.unix.listener import UnixListener

        received: list[bytes] = []

        class Handler:
            def onData(self, chunk: bytes) -> None:
                received.append(chunk)

            def onClose(self) -> None:
                pass

            def onError(self, error: Exception) -> None:
                pass

        path = str(tmp_path / "s.sock")
        listener = UnixListener({"path": path})
        await listener.start(lambda connection: Handler())
        try:

            class Handlers:
                def onData(self, chunk: bytes) -> None:
                    pass

                def onClose(self) -> None:
                    pass

                def onError(self, error: Exception) -> None:
                    pass

            transport = await connectUnixSocket(path, 1024, Handlers())
            sending = transport.send(b"frame")
            transport.close()
            with pytest.raises(Exception, match="closed"):
                await sending
            await asyncio.sleep(0.05)
            assert received == []
        finally:
            await listener.close()

    async def test_a_peer_that_closes_right_after_connect_is_reported_as_a_closed_transport(self, tmp_path):
        from misaka.client.unix import probeUnixServer

        path = str(tmp_path / "00000000-0000-4000-8000-000000000001.sock")

        async def handler(reader, writer):
            writer.close()

        server = await asyncio.start_unix_server(handler, path)
        try:
            for _ in range(20):
                assert await probeUnixServer({"serverId": "00000000-0000-4000-8000-000000000001", "path": path}, 2000) is None
        finally:
            server.close()



async def test_a_caller_that_stops_waiting_does_not_cancel_the_queue_behind_it():
    """``runForClient`` chains ``previous.catch(() => {}).then(operation)``: nothing a caller does stops a later
    operation of the same client, nor the operation it stopped waiting for."""
    from misaka.server.session_router import SessionRouter

    router = SessionRouter({
        "host": None,
        "serverId": "00000000-0000-4000-8000-000000000001",
        "isClosing": lambda: False,
        "publishAttachment": None,
        "reportError": lambda error: None,
    })
    client, gate, ran = object(), asyncio.Event(), []

    async def first() -> str:
        await gate.wait()
        ran.append("first")
        return "one"

    async def second() -> str:
        ran.append("second")
        return "two"

    waiter = asyncio.ensure_future(router._runForClient(client, first))
    await asyncio.sleep(0.01)
    queued = router._runForClient(client, second)
    waiter.cancel()
    await asyncio.sleep(0.01)
    gate.set()
    assert await queued == "two"
    assert ran == ["first", "second"]
