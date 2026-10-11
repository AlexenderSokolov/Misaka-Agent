"""A Unix-domain socket listener for the experimental server.

Translated from pi's ``packages/server/src/transports/unix/listener.ts``. Node's ``net.Socket`` is ``NetSocket``
here, an ``asyncio.Protocol`` with the part of the socket API this file uses: ``on``/``once``/``off`` for the
``data``, ``error`` and ``close`` events, ``write(chunk, callback)``, ``end``, ``destroy``, ``destroyed``,
``writable`` and ``writableLength``. Unix sockets are POSIX-only: on Windows ``start()`` fails.
"""

from __future__ import annotations

import asyncio
import errno
import hashlib
import os
import socket
import stat
import sys
import uuid
from collections.abc import Callable
from typing import Any, TypedDict

from misaka.protocol import DEFAULT_MAX_FRAME_LENGTH
from misaka.utils.js import (
    awaitValue,
    jsIsInteger,
    jsIsSafeInteger,
    rejectedPromise,
    resolvedPromise,
    spawn,
    startTask,
)

from ...connection import ByteConnectionAcceptor
from ...listener import ServerListener
from .types import UnixListenerOptions

DEFAULT_SOCKET_MODE = 0o600
DEFAULT_GRACEFUL_CLOSE_TIMEOUT_MS = 5_000
MAX_UINT32 = 0xFFFF_FFFF
MAX_TIMER_DELAY_MS = 2_147_483_647
SOCKET_PROBE_TIMEOUT_MS = 1_000


class ResolvedUnixListenerOptions(TypedDict):
    path: str
    mode: int
    gracefulCloseTimeoutMs: int
    maxPendingBytes: int
    onError: Callable[[Exception], None] | None


class NetSocket(asyncio.Protocol):
    """The slice of a Node ``net.Socket`` this transport uses, over an asyncio transport."""

    def __init__(self) -> None:
        self.destroyed = False
        self.writable = False
        self._transport: asyncio.Transport | None = None
        self._listeners: dict[str, list[tuple[Callable[..., None], bool]]] = {}
        self._paused = False
        self._writeCallbacks: list[Callable[[Exception | None], None]] = []
        self._onConnect: Callable[[NetSocket], None] | None = None
        self._destroyError: Exception | None = None

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
        listeners = self._listeners.get(event, [])
        for entry in list(listeners):
            if entry not in listeners:
                continue
            if entry[1]:
                listeners.remove(entry)
            entry[0](*args)

    @property
    def writableLength(self) -> int:
        return self._transport.get_write_buffer_size() if self._transport is not None else 0

    def connection_made(self, transport: asyncio.BaseTransport) -> None:
        assert isinstance(transport, asyncio.Transport)
        self._transport = transport
        self.writable = True
        # Resume only once everything buffered is written, so a write callback means the data reached the kernel.
        transport.set_write_buffer_limits(high=0)
        if self._onConnect is not None:
            self._onConnect(self)

    def data_received(self, data: bytes) -> None:
        self.emit("data", data)

    def eof_received(self) -> bool | None:
        # The socket ends its own side when the peer ends: the transport closes after flushing.
        return False

    def connection_lost(self, exc: Exception | None) -> None:
        self.writable = False
        self.destroyed = True
        callbacks, self._writeCallbacks = self._writeCallbacks, []
        for callback in callbacks:
            callback(exc or ConnectionResetError("Socket closed before the write finished"))
        error = exc if exc is not None else self._destroyError
        if error is not None:
            self.emit("error", error)
        self.emit("close")

    def pause_writing(self) -> None:
        self._paused = True

    def resume_writing(self) -> None:
        self._paused = False
        callbacks, self._writeCallbacks = self._writeCallbacks, []
        for callback in callbacks:
            callback(None)

    def write(self, chunk: bytes, callback: Callable[[Exception | None], None] | None = None) -> bool:
        """Writes ``chunk``; ``callback`` runs once it reached the kernel or the socket failed. Returns whether the
        socket takes more without waiting."""
        transport = self._transport
        if transport is None or self.destroyed or not self.writable:
            if callback is not None:
                asyncio.get_running_loop().call_soon(callback, RuntimeError("Cannot write after the socket is closed"))
            return False
        transport.write(chunk)
        if callback is not None:
            if transport.get_write_buffer_size() == 0:
                asyncio.get_running_loop().call_soon(callback, None)
            else:
                self._writeCallbacks.append(callback)
        return not self._paused

    def end(self, chunk: bytes | None = None) -> None:
        """Writes ``chunk``, then half-closes once everything buffered is written."""
        transport = self._transport
        if transport is None or self.destroyed or not self.writable:
            return
        if chunk:
            transport.write(chunk)
        self.writable = False
        if transport.can_write_eof():
            transport.write_eof()

    def destroy(self, error: Exception | None = None) -> None:
        """``socket.destroy(error)``: closes at once; ``error``, when given, is emitted as ``error`` before ``close``."""
        if self.destroyed:
            return
        self.destroyed = True
        self.writable = False
        self._destroyError = error
        if self._transport is not None:
            self._transport.abort()


class UnixListener:
    def __init__(self, options: UnixListenerOptions) -> None:
        self._options = resolveUnixListenerOptions(options)
        self._path = self._options["path"]
        self._mode = self._options["mode"]
        self._connections: set[UnixByteConnection] = set()
        self._server: asyncio.Server | None = None
        self._socketIdentity: tuple[int, int] | None = None
        self._ownedBindPath: str | None = None
        self._closing = False
        self._closePromise: asyncio.Future[None] | None = None
        self._accept: ByteConnectionAcceptor | None = None

    async def start(self, accept: ByteConnectionAcceptor) -> None:
        if self._server is not None:
            raise RuntimeError("Unix listener is already started")
        if self._closing:
            raise RuntimeError("Unix listener is closing or closed")
        # MISAKA fork: asyncio has no Unix sockets on Windows, where Node listens on a named pipe instead.
        if sys.platform == "win32":
            raise RuntimeError("Unix listener is not supported on Windows")
        self._accept = accept

        ownedBindPath = getOwnedBindPath(self._path)
        await asyncio.to_thread(_makeDirectories, os.path.dirname(self._path) or ".", 0o700)
        await removeStaleSocket(self._path)
        await removeStaleSocket(ownedBindPath)
        self._ownedBindPath = ownedBindPath
        server: asyncio.Server | None = None
        try:
            kwargs: dict[str, Any] = {"cleanup_socket": False} if sys.version_info >= (3, 13) else {}
            # MISAKA fork: asyncio removes a socket file already at the path it is given, so the bind is made here, where
            # it fails with EADDRINUSE when another start bound the same private path first, as Node's ``listen`` does.
            # A failure to listen is reported to ``onError`` as the server's ``error`` event is; a failure to accept goes
            # to the event loop's exception handler.
            bound = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                bound.bind(ownedBindPath)
                server = await asyncio.get_running_loop().create_unix_server(self._createSocket, sock=bound, **kwargs)
            except Exception as error:
                bound.close()
                self._reportError(error)
                raise
            self._server = server
            stats = await asyncio.to_thread(os.lstat, ownedBindPath)
            if not stat.S_ISSOCK(stats.st_mode):
                raise RuntimeError(f"Unix listener path is not a socket after binding: {ownedBindPath}")
            self._socketIdentity = (stats.st_dev, stats.st_ino)
            await asyncio.to_thread(os.link, ownedBindPath, self._path)
            await setSocketMode(self._path, self._mode)
            await removePath(ownedBindPath)
            self._ownedBindPath = None
        except Exception:
            await self._closeServerAndCleanup(server)
            self._server = None
            raise

    def close(self) -> asyncio.Future[None]:
        if self._closePromise is not None:
            return self._closePromise
        self._closing = True
        self._closePromise = startTask(self._closeInternal())
        return self._closePromise

    def _createSocket(self) -> NetSocket:
        socket = NetSocket()
        socket._onConnect = self._acceptSocket
        return socket

    def _acceptSocket(self, socket: NetSocket) -> None:
        if self._closing:
            socket.destroy()
            return
        connection = UnixByteConnection(
            socket, self._options["gracefulCloseTimeoutMs"], self._options["maxPendingBytes"]
        )
        self._connections.add(connection)
        accept = self._accept
        if accept is None:
            socket.destroy()
            return
        handler = accept(connection)
        socket.on("data", lambda chunk: handler.onData(chunk))

        def onError(error: Exception) -> None:
            handler.onError(error)
            socket.destroy()

        def onClose() -> None:
            connection.markClosed()
            self._connections.discard(connection)
            handler.onClose()

        socket.on("error", onError)
        socket.once("close", onClose)

    async def _closeInternal(self) -> None:
        serverClosed = (
            startTask(self._closeServerAndCleanup(self._server))
            if self._server is not None
            else startTask(self._cleanupOwnedSocket())
        )
        await asyncio.gather(*(startTask(connection.close()) for connection in list(self._connections)))
        await serverClosed
        if self._ownedBindPath is not None:
            await removePath(self._ownedBindPath)
        self._ownedBindPath = None
        self._connections.clear()
        self._server = None

    async def _closeServerAndCleanup(self, server: asyncio.Server | None) -> None:
        try:
            if server is not None:
                await closeNetServer(server, self._reportError)
        finally:
            # Remove an unpublished startup bind path before the public route.
            if self._ownedBindPath is not None:
                await removePath(self._ownedBindPath)
            self._ownedBindPath = None
            await self._cleanupOwnedSocket()

    async def _cleanupOwnedSocket(self) -> None:
        identity = self._socketIdentity
        self._socketIdentity = None
        if identity is None:
            return
        try:
            current = await asyncio.to_thread(os.lstat, self._path)
        except OSError as error:
            if isErrorCode(error, "ENOENT"):
                return
            raise
        if not stat.S_ISSOCK(current.st_mode) or (current.st_dev, current.st_ino) != identity:
            return

        preserved = os.path.join(os.path.dirname(self._path), f"cleanup-{str(uuid.uuid4())[:6]}")
        try:
            await asyncio.to_thread(os.rename, self._path, preserved)
        except OSError as error:
            if isErrorCode(error, "ENOENT"):
                return
            raise
        moved = await asyncio.to_thread(os.lstat, preserved)
        if stat.S_ISSOCK(moved.st_mode) and (moved.st_dev, moved.st_ino) == identity:
            await removePath(preserved)
            return
        try:
            await asyncio.to_thread(os.lstat, self._path)
        except OSError as error:
            if isErrorCode(error, "ENOENT"):
                await asyncio.to_thread(os.rename, preserved, self._path)
            else:
                raise
        raise RuntimeError(f"Unix listener path changed during cleanup; preserved replacement at {preserved}")

    def _reportError(self, error: Any) -> None:
        try:
            onError = self._options["onError"]
            if onError is not None:
                onError(error if isinstance(error, Exception) else Exception(str(error)))
        except Exception:  # noqa: BLE001, S110 - error observers cannot affect listener state
            pass


class UnixByteConnection:
    """A byte connection over a socket. Exported only for transport-level verification."""

    def __init__(self, socket: Any, gracefulCloseTimeoutMs: int, maxPendingBytes: int) -> None:
        self._socket = socket
        self._gracefulCloseTimeoutMs = gracefulCloseTimeoutMs
        self._maxPendingBytes = maxPendingBytes
        self._pendingBytes = 0
        self._closedValue = False
        self._closing = False
        self._writeTail: asyncio.Future[None] = resolvedPromise(None)
        self._closePromise: asyncio.Future[None] | None = None
        self._resolveClose: Callable[[], None] | None = None
        self._background: set[asyncio.Task[Any]] = set()

    @property
    def closed(self) -> bool:
        return self._closedValue

    def send(self, chunk: bytes) -> asyncio.Future[None]:
        if not isinstance(chunk, bytes | bytearray | memoryview):
            return rejectedPromise(TypeError("Unix connection chunks must be Uint8Array"))
        if self._closedValue or self._closing:
            return rejectedPromise(RuntimeError("Unix connection is closed"))
        if self._pendingBytes + len(chunk) > self._maxPendingBytes:
            return rejectedPromise(RuntimeError("Unix connection exceeded its pending byte limit"))
        self._pendingBytes += len(chunk)
        data = bytes(chunk)
        tracked = startTask(self._enqueue(self._writeTail, data))
        self._writeTail = startTask(_swallow(tracked))
        return tracked

    def close(self, finalChunk: bytes | None = None) -> asyncio.Future[None]:
        if self._closedValue or self._socket.destroyed:
            self.markClosed()
            return resolvedPromise(None)
        if self._closePromise is not None:
            return self._closePromise
        self._closing = True
        finalBytes = bytes(finalChunk) if finalChunk is not None else None
        loop = asyncio.get_running_loop()
        closePromise: asyncio.Future[None] = loop.create_future()
        self._closePromise = closePromise

        def resolveClose() -> None:
            if not closePromise.done():
                closePromise.set_result(None)

        self._resolveClose = resolveClose

        def onTimeout() -> None:
            if not self._socket.destroyed:
                self._socket.destroy()
            self.markClosed()

        timer = loop.call_later(self._gracefulCloseTimeoutMs / 1000, onTimeout)
        self._socket.once("close", timer.cancel)
        spawn(self._endAfterWrites(finalBytes), self._background)
        return closePromise

    def markClosed(self) -> None:
        if self._closedValue:
            return
        self._closedValue = True
        self._closing = True
        if self._resolveClose is not None:
            self._resolveClose()
        self._resolveClose = None

    async def _endAfterWrites(self, finalBytes: bytes | None) -> None:
        await self._writeTail
        if self._socket.destroyed:
            self.markClosed()
            return
        try:
            if finalBytes:
                self._socket.end(finalBytes)
            else:
                self._socket.end()
        except Exception:  # noqa: BLE001
            self._socket.destroy()

    async def _enqueue(self, previous: asyncio.Future[None], chunk: bytes) -> None:
        try:
            # `.then(write)` never writes inline.
            await asyncio.sleep(0)
            await previous
            await self._write(chunk)
        finally:
            self._pendingBytes -= len(chunk)

    def _write(self, chunk: bytes) -> asyncio.Future[None]:
        if self._closedValue or self._closing or not self._socket.writable:
            return rejectedPromise(RuntimeError("Unix connection is closed"))
        future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        settled = False

        def finish(error: Exception | None = None) -> None:
            nonlocal settled
            if settled:
                return
            settled = True
            self._socket.off("close", onClose)
            if error is not None:
                future.set_exception(error)
            else:
                future.set_result(None)

        def onClose() -> None:
            finish(RuntimeError("Unix connection closed during write"))

        self._socket.once("close", onClose)
        try:
            self._socket.write(chunk, finish)
        except Exception as error:  # noqa: BLE001
            finish(error)
        return future


def getOwnedBindPath(path: str) -> str:
    suffix = hashlib.sha256(path.encode("utf-8")).hexdigest()[:8]
    return os.path.join(os.path.dirname(path), f"bind-{suffix}")


async def removeStaleSocket(path: str) -> None:
    try:
        original = await asyncio.to_thread(os.lstat, path)
    except OSError as error:
        if isErrorCode(error, "ENOENT"):
            return
        raise
    if not stat.S_ISSOCK(original.st_mode):
        raise RuntimeError(f"Refusing to remove non-socket Unix listener path: {path}")
    if await isSocketLive(path):
        raise RuntimeError(f"Unix listener is already running: {path}")

    preserved = os.path.join(os.path.dirname(path), f"stale-{str(uuid.uuid4())[:6]}")
    try:
        await asyncio.to_thread(os.rename, path, preserved)
    except OSError as error:
        if isErrorCode(error, "ENOENT"):
            return
        raise
    current = await asyncio.to_thread(os.lstat, preserved)
    if not stat.S_ISSOCK(current.st_mode) or current.st_dev != original.st_dev or current.st_ino != original.st_ino:
        try:
            await asyncio.to_thread(os.lstat, path)
        except OSError as error:
            if isErrorCode(error, "ENOENT"):
                await asyncio.to_thread(os.rename, preserved, path)
            else:
                raise
        raise RuntimeError(f"Unix listener path changed while checking for a stale socket: {path}")
    await removePath(preserved)


async def removePath(path: str) -> None:
    try:
        await asyncio.to_thread(os.unlink, path)
    except OSError as error:
        if not isErrorCode(error, "ENOENT"):
            raise


async def isSocketLive(path: str) -> bool:
    try:
        _reader, writer = await asyncio.wait_for(
            asyncio.open_unix_connection(path),  # type: ignore[attr-defined,unused-ignore]
            SOCKET_PROBE_TIMEOUT_MS / 1000,
        )
    except TimeoutError:
        return True
    except OSError as error:
        if errno.errorcode.get(error.errno or 0) in ("ECONNREFUSED", "ENOENT", "EPIPE", "ECONNRESET"):
            return False
        raise
    writer.transport.abort()
    return True


async def setSocketMode(path: str, mode: int) -> None:
    if sys.platform == "win32":
        return
    try:
        await asyncio.to_thread(os.chmod, path, mode)
    except OSError as error:
        if not isErrorCode(error, "ENOSYS") and not isErrorCode(error, "ENOTSUP"):
            raise


async def closeNetServer(server: asyncio.Server, reportError: Callable[[Exception], None]) -> None:
    if not server.is_serving():
        return
    server.close()
    try:
        await server.wait_closed()
    except Exception as error:  # noqa: BLE001
        reportError(error)


def isErrorCode(error: BaseException, code: str) -> bool:
    return isinstance(error, OSError) and errno.errorcode.get(error.errno or 0) == code


def createUnixListener(options: UnixListenerOptions) -> ServerListener:
    return UnixListener(options)  # type: ignore[return-value]


def resolveUnixListenerOptions(options: UnixListenerOptions) -> ResolvedUnixListenerOptions:
    if not options.get("path"):
        raise TypeError("Server Unix socket path must not be empty")
    mode = options.get("mode")
    if mode is None:
        mode = DEFAULT_SOCKET_MODE
    if not jsIsInteger(mode) or mode < 0 or mode > 0o777:
        raise TypeError("Server Unix socket mode must be an integer between 0 and 0o777")
    maxFrameLength = options.get("maxFrameLength")
    if maxFrameLength is None:
        maxFrameLength = DEFAULT_MAX_FRAME_LENGTH
    if not jsIsInteger(maxFrameLength) or maxFrameLength <= 0 or maxFrameLength > MAX_UINT32:
        raise TypeError(f"Server maxFrameLength must be an integer between 1 and {MAX_UINT32}")
    maxPendingBytes = options.get("maxPendingBytes")
    if maxPendingBytes is None:
        maxPendingBytes = maxFrameLength * 4
    if not jsIsSafeInteger(maxPendingBytes) or maxPendingBytes < maxFrameLength + 4:
        raise TypeError("Server maxPendingBytes must be a safe integer at least maxFrameLength + 4")
    gracefulCloseTimeoutMs = options.get("gracefulCloseTimeoutMs")
    if gracefulCloseTimeoutMs is None:
        gracefulCloseTimeoutMs = DEFAULT_GRACEFUL_CLOSE_TIMEOUT_MS
    if not jsIsInteger(gracefulCloseTimeoutMs) or gracefulCloseTimeoutMs <= 0 or gracefulCloseTimeoutMs > MAX_TIMER_DELAY_MS:
        raise TypeError(f"Server gracefulCloseTimeoutMs must be an integer between 1 and {MAX_TIMER_DELAY_MS}")
    return {
        "path": options["path"],
        "mode": mode,
        "maxPendingBytes": maxPendingBytes,
        "gracefulCloseTimeoutMs": gracefulCloseTimeoutMs,
        "onError": options.get("onError"),
    }


def _makeDirectories(path: str, mode: int) -> None:
    """``mkdir(path, { recursive: true, mode })``: every directory it creates gets ``mode``."""
    parent = os.path.dirname(path)
    if parent and parent != path and not os.path.isdir(parent):
        _makeDirectories(parent, mode)
    try:
        os.mkdir(path, mode)
    except FileExistsError:
        if not os.path.isdir(path):
            raise


async def _swallow(future: asyncio.Future[Any]) -> None:
    try:
        await awaitValue(future)
    except Exception:  # noqa: BLE001, S110 - the write tail only orders the writes
        pass


__all__ = ["NetSocket", "UnixByteConnection", "createUnixListener"]
