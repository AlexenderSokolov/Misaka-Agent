"""Unix-domain socket transport and local server discovery.

Translated from pi's ``packages/client/src/unix.ts``. Node's ``net.Socket`` events (``connect``, ``data``,
``end``, ``close``, ``error`` and ``drain``) are an ``asyncio.Protocol`` here: ``connection_made``,
``data_received``, ``eof_received``, ``connection_lost`` and ``resume_writing``. Unix sockets are POSIX-only,
as upstream says of Windows.
"""

from __future__ import annotations

import asyncio
import errno
import os
import stat
import sys
from collections.abc import Awaitable, Callable
from typing import Any, NotRequired, TypedDict

from misaka.protocol import (
    DEFAULT_MAX_FRAME_LENGTH,
    ProtocolValidationError,
    isServerId,
)
from misaka.utils.collation import localeSortKey
from misaka.utils.js import jsIsInteger, jsIsSafeInteger, rejectedPromise, startTask

from .client import Client
from .errors import DisconnectedError, ServerError
from .transport import ByteTransport, ByteTransportFactory, ByteTransportHandlers

DEFAULT_DISCOVERY_TIMEOUT_MS = 1_000
MAX_TIMER_DELAY_MS = 2_147_483_647
UNIX_SOCKET_SUFFIX = ".sock"
MAX_CONCURRENT_DISCOVERY_PROBES = 16


class UnixTransportOptions(TypedDict):
    path: str
    maxPendingBytes: NotRequired[int]


class UnixServerRoute(TypedDict):
    serverId: str
    path: str


class DiscoverUnixServersOptions(TypedDict):
    """``directory`` contains server-addressed Unix sockets; ``timeoutMs`` (default 1,000 ms) is the maximum time
    for each connection and handshake."""

    directory: str
    timeoutMs: NotRequired[int]


async def discoverUnixServers(options: DiscoverUnixServersOptions) -> list[UnixServerRoute]:
    """Discover reachable local servers by probing server-addressed Unix sockets."""
    if sys.platform == "win32":
        raise Exception("Unix transport is not supported on Windows")  # noqa: TRY002
    directory = options["directory"]
    timeoutMs = options.get("timeoutMs")
    if timeoutMs is None:
        timeoutMs = DEFAULT_DISCOVERY_TIMEOUT_MS
    if not jsIsInteger(timeoutMs) or timeoutMs <= 0 or timeoutMs > MAX_TIMER_DELAY_MS:
        raise TypeError(f"Unix discovery timeoutMs must be an integer between 1 and {MAX_TIMER_DELAY_MS}")

    try:
        names = await asyncio.to_thread(os.listdir, directory)
    except OSError as error:
        if isErrorCode(error, "ENOENT"):
            return []
        raise

    candidates: list[UnixServerRoute] = []
    for name in names:
        if not name.endswith(UNIX_SOCKET_SUFFIX):
            continue
        serverId = name[: -len(UNIX_SOCKET_SUFFIX)]
        if isServerId(serverId):
            candidates.append({"serverId": serverId, "path": os.path.normpath(os.path.join(directory, name))})
    routes: list[UnixServerRoute] = []
    nextIndex = 0
    failure: list[BaseException] = []
    workerCount = min(MAX_CONCURRENT_DISCOVERY_PROBES, len(candidates))

    async def worker() -> None:
        nonlocal nextIndex
        while not failure:
            if nextIndex >= len(candidates):
                return
            candidate = candidates[nextIndex]
            nextIndex += 1
            try:
                try:
                    if not stat.S_ISSOCK((await asyncio.to_thread(os.lstat, candidate["path"])).st_mode):
                        continue
                except OSError as error:
                    # A socket can disappear between readdir and lstat during normal server shutdown.
                    if isErrorCode(error, "ENOENT"):
                        continue
                    raise
                route = await probeUnixServer(candidate, timeoutMs)
                if route is not None:
                    routes.append(route)
            except Exception as error:  # noqa: BLE001
                if not failure:
                    failure.append(error)

    await asyncio.gather(*(worker() for _ in range(workerCount)))
    if failure:
        raise failure[0]
    routes.sort(key=lambda route: localeSortKey(route["serverId"]))
    return routes


def createUnixTransportFactory(options: UnixTransportOptions) -> ByteTransportFactory:
    """Creates fresh Unix-domain socket transports for Client connection attempts."""
    maxPendingBytes = validateUnixTransportOptions(options)
    return lambda handlers: connectUnixSocket(options["path"], maxPendingBytes, handlers)


def validateUnixTransportOptions(options: UnixTransportOptions) -> int:
    if len(options["path"]) == 0:
        raise TypeError("Unix transport path must not be empty")
    maxPendingBytes = options.get("maxPendingBytes")
    if maxPendingBytes is None:
        maxPendingBytes = DEFAULT_MAX_FRAME_LENGTH * 4
    if not jsIsSafeInteger(maxPendingBytes) or maxPendingBytes <= 0:
        raise TypeError("Unix transport maxPendingBytes must be a positive safe integer")
    if sys.platform == "win32":
        raise Exception("Unix transport is not supported on Windows")  # noqa: TRY002
    return maxPendingBytes


class _Socket(asyncio.Protocol):
    """The slice of a Node ``net.Socket`` this transport uses: it connects, reports data, end, close and error,
    writes with backpressure, and is destroyed."""

    def __init__(self) -> None:
        self.destroyed = False
        self.closed = False
        self.writable = False
        self.transport: asyncio.Transport | None = None
        self.connecting: asyncio.Future[Any] | None = None
        self.onData: Callable[[bytes], None] = lambda chunk: None
        self.onEnd: Callable[[], None] = lambda: None
        self.onClose: Callable[[], None] = lambda: None
        self.onError: Callable[[Exception], None] = lambda error: None
        self._paused = False
        self._drainWaiters: list[asyncio.Future[None]] = []
        self._closeWaiters: list[asyncio.Future[None]] = []

    def connection_made(self, transport: asyncio.BaseTransport) -> None:
        assert isinstance(transport, asyncio.Transport)
        self.transport = transport
        self.writable = True
        # MISAKA fork: Node starts reading after the ``connect`` event has been handled; asyncio reads from
        # ``connection_made``, so reading waits for ``startReading`` (the first write).
        transport.pause_reading()
        # Resume only once everything buffered is written, so a write that returns has reached the kernel.
        transport.set_write_buffer_limits(high=0)

    def data_received(self, data: bytes) -> None:
        self.onData(data)

    def eof_received(self) -> bool | None:
        self.onEnd()
        return False

    def connection_lost(self, exc: Exception | None) -> None:
        self.writable = False
        if exc is not None:
            self.onError(exc)
        self._markClosed()
        self.onClose()

    def pause_writing(self) -> None:
        self._paused = True

    def resume_writing(self) -> None:
        self._paused = False
        waiters, self._drainWaiters = self._drainWaiters, []
        for waiter in waiters:
            if not waiter.done():
                waiter.set_result(None)

    async def connect(self, path: str) -> None:
        loop = asyncio.get_running_loop()
        self.connecting = asyncio.ensure_future(loop.create_unix_connection(lambda: self, path))
        try:
            await asyncio.wait({self.connecting})
        except asyncio.CancelledError:
            self.connecting.cancel()
            self._markClosed()
            raise
        if self.connecting.cancelled():
            self._markClosed()
            raise Exception("Unix transport closed before connecting")  # noqa: TRY002
        if self.connecting.exception() is not None:
            # A socket whose connect failed is closed, as Node's is after its `error` event.
            self._markClosed()
        self.connecting.result()

    def write(self, chunk: bytes) -> bool:
        """Writes ``chunk`` and returns whether the socket accepts more without waiting for ``drain``."""
        assert self.transport is not None
        self.transport.write(chunk)
        return not self._paused

    def startReading(self) -> None:
        if self.transport is not None and not self.transport.is_closing():
            self.transport.resume_reading()

    def drained(self) -> asyncio.Future[None]:
        """Settles when the socket has written what it buffered, or fails when it closes first."""
        waiter: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._drainWaiters.append(waiter)
        return waiter

    def closedFuture(self) -> asyncio.Future[None]:
        waiter: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        if self.closed:
            waiter.set_result(None)
        else:
            self._closeWaiters.append(waiter)
        return waiter

    def destroy(self) -> None:
        if self.destroyed:
            return
        self.destroyed = True
        self.writable = False
        if self.transport is not None:
            self.transport.abort()
        elif self.connecting is not None:
            self.connecting.cancel()

    def _markClosed(self) -> None:
        self.closed = True
        self.writable = False
        waiters, self._drainWaiters = self._drainWaiters, []
        for drain in waiters:
            if not drain.done():
                drain.set_exception(Exception("Unix transport closed during write"))
                drain.exception()
        closing, self._closeWaiters = self._closeWaiters, []
        for waiter in closing:
            if not waiter.done():
                waiter.set_result(None)


async def connectUnixSocket(
    path: str,
    maxPendingBytes: int,
    handlers: ByteTransportHandlers,
    onSocket: Callable[[_Socket], None] | None = None,
) -> ByteTransport:
    socket = _Socket()
    if onSocket is not None:
        onSocket(socket)
    connected = False
    terminal = False

    def close() -> None:
        nonlocal terminal
        if terminal:
            return
        terminal = True
        socket.destroy()
        if connected:
            handlers.onClose()

    def markLocalClose() -> None:
        nonlocal terminal
        terminal = True

    def onData(chunk: bytes) -> None:
        if not terminal:
            handlers.onData(chunk)

    def onError(error: Exception) -> None:
        nonlocal terminal
        if terminal:
            return
        terminal = True
        socket.destroy()
        if connected:
            handlers.onError(error)

    socket.onData = onData
    socket.onEnd = close
    socket.onClose = close
    socket.onError = onError
    try:
        await socket.connect(path)
    except BaseException:
        terminal = True
        socket.destroy()
        raise
    if terminal:
        raise Exception("Unix transport closed before connecting")  # noqa: TRY002
    connected = True
    return UnixByteTransport(socket, maxPendingBytes, markLocalClose)


class UnixByteTransport:
    def __init__(self, socket: _Socket, maxPendingBytes: int, markLocalClose: Callable[[], None]) -> None:
        self._socket = socket
        self._maxPendingBytes = maxPendingBytes
        self._markLocalClose = markLocalClose
        self._closed = False
        self._pendingBytes = 0
        self._writeTail: asyncio.Future[None] | None = None

    def send(self, chunk: bytes) -> Awaitable[None]:
        if not isinstance(chunk, bytes | bytearray | memoryview):
            return rejectedPromise(TypeError("Unix transport chunks must be Uint8Array"))
        if self._closed:
            return rejectedPromise(Exception("Unix transport is closed"))
        if self._pendingBytes + len(chunk) > self._maxPendingBytes:
            return rejectedPromise(Exception("Unix transport exceeded its pending byte limit"))
        self._pendingBytes += len(chunk)
        data = bytes(chunk)
        tracked = startTask(self._enqueue(self._writeTail, data))
        self._writeTail = startTask(_swallow(tracked))
        return tracked

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._markLocalClose()
        self._socket.destroy()

    async def _enqueue(self, previous: asyncio.Future[None] | None, chunk: bytes) -> None:
        try:
            # `.then(write)` never writes inline.
            await asyncio.sleep(0)
            if previous is not None:
                await previous
            await self._write(chunk)
        finally:
            self._pendingBytes -= len(chunk)

    async def _write(self, chunk: bytes) -> None:
        socket = self._socket
        if self._closed or not socket.writable:
            raise Exception("Unix transport is closed")  # noqa: TRY002
        accepted = socket.write(chunk)
        socket.startReading()
        if not accepted:
            await socket.drained()


async def probeUnixServer(route: UnixServerRoute, timeoutMs: int) -> UnixServerRoute | None:
    maxPendingBytes = validateUnixTransportOptions({"path": route["path"]})
    socket: _Socket | None = None

    def created(createdSocket: _Socket) -> None:
        nonlocal socket
        socket = createdSocket

    client = Client(
        {
            "serverId": route["serverId"],
            "transportFactory": lambda handlers: connectUnixSocket(route["path"], maxPendingBytes, handlers, created),
        }
    )
    loop = asyncio.get_running_loop()
    timeout: asyncio.TimerHandle | None = None
    timedOut: asyncio.Future[None] = loop.create_future()
    first: asyncio.Future[asyncio.Future[Any]] = loop.create_future()

    def onTimeout() -> None:
        if socket is not None:
            socket.destroy()
        timedOut.set_exception(UnixDiscoveryTimeoutError())
        timedOut.exception()

    def settled(future: asyncio.Future[Any]) -> None:
        if not first.done():
            first.set_result(future)

    try:
        # `Promise.race`: whichever settles first decides.
        connecting = client.connect()
        timeout = loop.call_later(timeoutMs / 1000, onTimeout)
        connecting.add_done_callback(settled)
        timedOut.add_done_callback(settled)
        winner = await first
        if winner is timedOut:
            raise UnixDiscoveryTimeoutError
        await connecting
        return route
    except Exception as error:
        # Missing/refused sockets are stale or shutting down. Protocol failures mean
        # the endpoint is not the advertised server. Both are safe to omit.
        if (
            isinstance(error, UnixDiscoveryTimeoutError | ProtocolValidationError)
            or (isinstance(error, DisconnectedError) and error.cause is None)
            or (isinstance(error, ServerError) and error.code == "version")
            or isErrorCode(error, "ENOENT")
            or isErrorCode(error, "ECONNREFUSED")
            or isErrorCode(error, "ECONNRESET")
            or isErrorCode(error, "EPIPE")
            or isErrorCode(error, "ETIMEDOUT")
        ):
            return None
        raise
    finally:
        if timeout is not None:
            timeout.cancel()
        await client.dispose()
        activeSocket = socket
        if activeSocket is not None and not activeSocket.destroyed:
            activeSocket.destroy()
        if activeSocket is not None and not activeSocket.closed:
            await activeSocket.closedFuture()


class UnixDiscoveryTimeoutError(Exception):
    pass


def isErrorCode(error: BaseException | None, code: str) -> bool:
    current = error
    seen: set[int] = set()
    while isinstance(current, Exception) and id(current) not in seen:
        seen.add(id(current))
        if _codeOf(current) == code:
            return True
        current = current.__cause__
    return False


def _codeOf(error: Exception) -> Any:
    code = getattr(error, "code", None)
    if code is not None:
        return code
    number = getattr(error, "errno", None)
    return errno.errorcode.get(number) if isinstance(number, int) else None



async def _swallow(awaitable: Awaitable[Any]) -> None:
    try:
        await awaitable
    except Exception:  # noqa: BLE001, S110 - the write tail only orders the writes; each caller sees its own failure
        pass


__all__ = [
    "DiscoverUnixServersOptions",
    "UnixServerRoute",
    "UnixTransportOptions",
    "createUnixTransportFactory",
    "discoverUnixServers",
]
