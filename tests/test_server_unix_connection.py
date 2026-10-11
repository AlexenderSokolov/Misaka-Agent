"""pi server's ``test/unix-connection.test.ts``."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from misaka.protocol import ServerMessageDecoder, encodeServerMessage
from misaka.server.transports.unix.listener import UnixByteConnection
from tests.chord_helpers import waitFor


class ControlledSocket:
    """A socket whose write completes only when the test says so."""

    def __init__(self) -> None:
        self.destroyed = False
        self.writable = True
        self.writableLength = 0
        self.ended = False
        self.finalChunk: bytes | None = None
        self._writeCallback: Callable[[Exception | None], None] | None = None
        self._listeners: dict[str, list[tuple[Callable[..., None], bool]]] = {}

    def on(self, event: str, listener: Callable[..., None]) -> None:
        self._listeners.setdefault(event, []).append((listener, False))

    def once(self, event: str, listener: Callable[..., None]) -> None:
        self._listeners.setdefault(event, []).append((listener, True))

    def off(self, event: str, listener: Callable[..., None]) -> None:
        listeners = self._listeners.get(event, [])
        for index, (candidate, _once) in enumerate(listeners):
            if candidate == listener:
                del listeners[index]
                return

    def emit(self, event: str, *args: Any) -> None:
        for entry in list(self._listeners.get(event, [])):
            if entry[1]:
                self._listeners[event].remove(entry)
            entry[0](*args)

    def write(self, _chunk: bytes, callback: Callable[[Exception | None], None]) -> bool:
        self.writableLength = 1
        self._writeCallback = callback
        return False

    def end(self, chunk: bytes | None = None) -> None:
        self.ended = True
        self.finalChunk = None if chunk is None else bytes(chunk)

    def destroy(self) -> None:
        if self.destroyed:
            return
        self.destroyed = True
        self.writable = False
        self.emit("close")

    def completeWrite(self) -> None:
        callback = self._writeCallback
        if callback is None:
            raise RuntimeError("No pending write")
        self._writeCallback = None
        self.writableLength = 0
        callback(None)


async def test_queues_a_final_protocol_error_behind_pending_output_before_closing():
    socket = ControlledSocket()
    connection = UnixByteConnection(socket, 1_000, 64 * 1024)
    pendingWrite = connection.send(bytes([1, 2, 3]))
    await waitFor(lambda: socket.writableLength == 1)
    finalMessage = {"type": "hello_error", "error": {"code": "invalid_request", "message": "Protocol violation"}}
    closing = connection.close(encodeServerMessage(finalMessage))

    assert socket.ended is False
    assert socket.destroyed is False

    socket.completeWrite()
    await pendingWrite
    await waitFor(lambda: socket.ended)
    assert socket.finalChunk is not None
    assert ServerMessageDecoder().push(socket.finalChunk) == [finalMessage]

    connection.markClosed()
    await closing
