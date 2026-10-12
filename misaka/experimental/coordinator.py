"""Translated from pi's ``coding-agent/src/experimental/coordinator.ts``.

The coordinator is a small process that owns the stable public socket of a logical server and routes opaque messages
between the current server generation and the Session workers over a control socket. ``CoordinatorConnection`` is the
server-side endpoint of that router, and ``ensureCoordinator`` starts one when none runs.

MISAKA fork: Node's ``net.Socket`` is ``misaka.server.transports.unix.listener.NetSocket`` here, and the coordinator's
module-level state is held by ``_CoordinatorProcess``. Unix sockets are POSIX-only, as in pi.
"""

from __future__ import annotations

import asyncio
import codecs
import os
import signal
import stat
import sys
import time
import traceback
import uuid
from collections.abc import Callable
from typing import Any

from misaka.experimental import schema
from misaka.experimental.process import (
    MAX_CONTROL_LINE_BYTES,
    consumeInternalProcessRole,
    encodeControlLine,
    onceSignal,
    spawnInternalProcess,
)
from misaka.server.transports.unix.listener import NetSocket
from misaka.utils.js import eager, jsonParse, spawn

COORDINATOR_PROTOCOL_VERSION = 3
COORDINATOR_START_TIMEOUT_MS = 10_000
COORDINATOR_RETRY_MS = 10

CoordinatorMessageSchema = schema.union(
    schema.looseObject(
        {
            "type": schema.literal("server_registered"),
            "serverConnectionId": schema.string(),
            "peers": schema.array(schema.string()),
        }
    ),
    schema.looseObject({"type": schema.literal("server_replaced")}),
    schema.looseObject({"type": schema.literal("peer_connected"), "peerId": schema.string()}),
    schema.looseObject({"type": schema.literal("peer_disconnected"), "peerId": schema.string()}),
    schema.looseObject({"type": schema.literal("message"), "from": schema.string(), "payload": schema.UNKNOWN}),
)


class CoordinatorConnection:
    """The server-side endpoint of the coordinator's intentionally opaque message router. ``options`` is ``{"controlPath",
    "endpoint", "serverConnectionId"?}``."""

    def __init__(self, options: dict[str, Any]) -> None:
        self.serverConnectionId: str = (
            options["serverConnectionId"] if options.get("serverConnectionId") is not None else str(uuid.uuid4())
        )
        self.peerIds: set[str] = set()
        self.replaced: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._controlPath = options["controlPath"]
        self._endpoint = options["endpoint"]
        self._listeners: list[Callable[[dict[str, Any]], None]] = []
        self._socket: NetSocket | None = None
        self._registered = False
        self._closed = False
        self._replacedValue = False
        self._registeredFuture: asyncio.Future[None] | None = None

    @property
    def controlPath(self) -> str:
        return self._controlPath

    @property
    def wasReplaced(self) -> bool:
        return self._replacedValue

    def onEvent(self, listener: Callable[[dict[str, Any]], None]) -> Callable[[], None]:
        self._listeners.append(listener)

        def remove() -> None:
            if listener in self._listeners:
                self._listeners.remove(listener)

        return remove

    @eager
    async def connect(self) -> None:
        if self._socket is not None:
            raise RuntimeError("Coordinator server is already connected")
        socket = await connectSocket(self._controlPath)
        self._socket = socket
        attachJsonLineReader(socket, self._handleMessage)
        registered: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._registeredFuture = registered
        socket.once("close", lambda: self._disconnected(RuntimeError("Coordinator connection closed")))
        socket.once("error", lambda error: self._disconnected(error))
        await writeJsonLine(
            socket,
            {
                "type": "register_server",
                "protocol": COORDINATOR_PROTOCOL_VERSION,
                "serverConnectionId": self.serverConnectionId,
                "endpoint": self._endpoint,
            },
        )
        await registered

    def send(self, peerId: str, payload: Any) -> asyncio.Future[None]:
        return self._write({"type": "send", "to": peerId, "payload": payload})

    def broadcast(self, payload: Any) -> asyncio.Future[None]:
        return self._write({"type": "broadcast", "payload": payload})

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._socket is not None:
            self._socket.destroy()
        self._socket = None
        self.peerIds.clear()
        self._rejectRegistered(RuntimeError("Coordinator server closed"))

    @eager
    async def _write(self, message: Any) -> None:
        if not self._registered or self._socket is None or self._closed:
            raise RuntimeError("Coordinator server is not connected")
        await writeJsonLine(self._socket, message)

    def _resolveRegistered(self) -> None:
        registered, self._registeredFuture = self._registeredFuture, None
        if registered is not None and not registered.done():
            registered.set_result(None)

    def _rejectRegistered(self, error: BaseException) -> None:
        registered, self._registeredFuture = self._registeredFuture, None
        if registered is not None and not registered.done():
            registered.set_exception(error)
            # A caller that stopped waiting never retrieves this; upstream's rejection has no such side effect.
            registered.exception()

    def _handleMessage(self, value: Any) -> None:
        if not schema.Check(CoordinatorMessageSchema, value):
            self._destroySocket(RuntimeError("Coordinator sent an invalid message"))
            return
        message = value
        if message["type"] == "server_registered":
            if message["serverConnectionId"] != self.serverConnectionId:
                self._destroySocket(RuntimeError("Coordinator returned an invalid server registration"))
                return
            for peerId in message["peers"]:
                self.peerIds.add(peerId)
            self._registered = True
            self._resolveRegistered()
            return
        if message["type"] == "server_replaced":
            self._markReplaced()
            return
        if message["type"] == "peer_connected":
            self.peerIds.add(message["peerId"])
            self._emit({"type": "peer_connected", "peerId": message["peerId"]})
            return
        if message["type"] == "peer_disconnected":
            self.peerIds.discard(message["peerId"])
            self._emit({"type": "peer_disconnected", "peerId": message["peerId"]})
            return
        if message["type"] == "message":
            self._emit({"type": "message", "from": message["from"], "payload": message["payload"]})
            return
        self._destroySocket(RuntimeError("Coordinator sent an unsupported message"))

    def _destroySocket(self, error: Exception) -> None:
        if self._socket is not None:
            self._socket.destroy(error)

    def _emit(self, event: dict[str, Any]) -> None:
        for listener in list(self._listeners):
            listener(event)

    def _disconnected(self, error: BaseException) -> None:
        self._socket = None
        if self._closed:
            return
        self._rejectRegistered(error)
        self._markReplaced()

    def _markReplaced(self) -> None:
        if self._replacedValue:
            return
        self._replacedValue = True
        if not self.replaced.done():
            self.replaced.set_result(None)


class CoordinatorStartupLease:
    def __init__(self, close: Callable[[], None]) -> None:
        self.close = close


async def ensureCoordinator(publicPath: str, controlPath: str) -> CoordinatorStartupLease:
    existing = await tryConnect(controlPath)
    if existing is not None:
        return CoordinatorStartupLease(existing.destroy)
    child = spawnInternalProcess("coordinator", [publicPath, controlPath])
    deadline = time.time() * 1000 + COORDINATOR_START_TIMEOUT_MS
    while True:
        socket = await tryConnect(controlPath)
        if socket is not None:
            return CoordinatorStartupLease(socket.destroy)
        if child.exitCode is not None or child.signalCode is not None:
            raise RuntimeError("Coordinator exited during startup")
        if time.time() * 1000 >= deadline:
            raise RuntimeError("Timed out waiting for coordinator startup")
        await asyncio.sleep(COORDINATOR_RETRY_MS / 1000)

# A connection event is a dict: ``{"type": "peer_connected", "peerId"}``, ``{"type": "peer_disconnected", "peerId"}`` or
# ``{"type": "message", "from", "payload"}``.


async def connectSocket(path: str) -> NetSocket:
    _transport, socket = await asyncio.get_running_loop().create_unix_connection(NetSocket, path)
    return socket


async def tryConnect(path: str) -> NetSocket | None:
    try:
        return await connectSocket(path)
    except (FileNotFoundError, ConnectionRefusedError):
        # ENOENT or ECONNREFUSED.
        return None


def attachJsonLineReader(socket: NetSocket, onMessage: Callable[[Any], None]) -> None:
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    state = {"buffered": ""}

    def onData(chunk: bytes) -> None:
        state["buffered"] += decoder.decode(chunk)
        if len(state["buffered"].encode("utf-8", "surrogatepass")) > MAX_CONTROL_LINE_BYTES:
            socket.destroy(RuntimeError("Coordinator message is too large"))
            return
        while True:
            newline = state["buffered"].find("\n")
            if newline == -1:
                return
            line = state["buffered"][:newline]
            state["buffered"] = state["buffered"][newline + 1 :]
            try:
                value = jsonParse(line)
            except ValueError:
                socket.destroy(RuntimeError("Coordinator sent invalid JSON"))
                return
            onMessage(value)

    socket.on("data", onData)


@eager
async def writeJsonLine(socket: NetSocket, message: Any) -> None:
    done: asyncio.Future[None] = asyncio.get_running_loop().create_future()

    def callback(error: Exception | None) -> None:
        if done.done():
            return
        if error is not None:
            done.set_exception(error)
        else:
            done.set_result(None)

    socket.write(encodeControlLine(message).encode("utf-8"), callback)
    await done


# This process is intentionally a transport shim. It depends only on the standard library and never interprets Pi,
# session, worker, or lifecycle payloads.

EMPTY_STARTUP_GRACE_MS = 30_000
EMPTY_SHUTDOWN_GRACE_MS = 250


class _ServerPeer:
    def __init__(self, serverConnectionId: str, endpoint: str, socket: NetSocket) -> None:
        self.serverConnectionId = serverConnectionId
        self.endpoint = endpoint
        self.socket = socket


class _RoutedPeer:
    def __init__(self, peerId: str, socket: NetSocket) -> None:
        self.peerId = peerId
        self.socket = socket


class _CoordinatorProcess:
    def __init__(self, publicPath: str, controlPath: str) -> None:
        self.publicPath = publicPath
        self.controlPath = controlPath
        self.peers: dict[str, _RoutedPeer] = {}
        self.controlConnections: set[NetSocket] = set()
        self.publicConnections: dict[NetSocket, NetSocket | None] = {}
        self.currentServer: _ServerPeer | None = None
        self.shuttingDown = False
        self.emptyTimer: asyncio.TimerHandle | None = None
        self.publicServer: asyncio.Server | None = None
        self.controlServer: asyncio.Server | None = None
        self.finished: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self.tasks: set[asyncio.Task[Any]] = set()

    async def main(self) -> None:
        await removeStaleSocket(self.controlPath)
        await removeStaleSocket(self.publicPath)
        try:
            self.controlServer = await self.listen(self.acceptControlConnection, self.controlPath)
            await restrictSocket(self.controlPath)
            self.publicServer = await self.listen(self.acceptPublicConnection, self.publicPath)
            await restrictSocket(self.publicPath)
        except Exception:
            await asyncio.gather(cleanupSocket(self.controlPath), cleanupSocket(self.publicPath))
            raise
        self.scheduleEmptyShutdown(EMPTY_STARTUP_GRACE_MS)

    @staticmethod
    async def listen(accept: Callable[[NetSocket], None], path: str) -> asyncio.Server:
        def create() -> NetSocket:
            socket = NetSocket()
            socket._onConnect = accept
            return socket

        kwargs: dict[str, Any] = {"cleanup_socket": False} if sys.version_info >= (3, 13) else {}
        return await asyncio.get_running_loop().create_unix_server(create, path, **kwargs)

    def acceptControlConnection(self, socket: NetSocket) -> None:
        if self.shuttingDown:
            socket.destroy()
            return
        self.controlConnections.add(socket)
        registered: dict[str, Any] = {"server": None, "peer": None}

        def onMessage(message: Any) -> None:
            if not isinstance(message, dict) or not isinstance(message.get("type"), str):
                raise RuntimeError("Coordinator message must have a type")  # noqa: TRY004
            if registered["server"] is None and registered["peer"] is None:
                if message["type"] == "register_server":
                    registered["server"] = self.registerServer(socket, message)
                    return
                if message["type"] == "register_peer":
                    registered["peer"] = self.registerPeer(socket, message)
                    return
                raise RuntimeError("Coordinator connection did not register a role")
            if registered["server"] is not None:
                if registered["server"] is self.currentServer:
                    self.handleRoutedMessage("server", message)
                return
            self.handleRoutedMessage(registered["peer"].peerId, message)

        attachRoutedLineReader(socket, onMessage)

        def disconnect() -> None:
            self.controlConnections.discard(socket)
            server, peer = registered["server"], registered["peer"]
            if server is not None and self.currentServer is server:
                self.currentServer = None
                self.notifyPeers({"type": "server_disconnected", "serverConnectionId": server.serverConnectionId})
            if peer is not None and self.peers.get(peer.peerId) is peer:
                del self.peers[peer.peerId]
                if self.currentServer is not None:
                    self.writeRoutedLine(self.currentServer.socket, {"type": "peer_disconnected", "peerId": peer.peerId})
            self.checkEmpty()

        socket.once("close", disconnect)
        socket.once("error", lambda _error: socket.destroy())

    def registerServer(self, socket: NetSocket, message: dict[str, Any]) -> _ServerPeer:
        if message.get("protocol") != COORDINATOR_PROTOCOL_VERSION:
            raise RuntimeError("Unsupported coordinator protocol")
        serverConnectionId = message.get("serverConnectionId")
        if not isinstance(serverConnectionId, str) or len(serverConnectionId) == 0:
            raise RuntimeError("Coordinator serverConnectionId must be a string")
        endpoint = message.get("endpoint")
        if not isinstance(endpoint, str) or len(endpoint) == 0:
            raise RuntimeError("Coordinator endpoint must be a string")
        server = _ServerPeer(serverConnectionId, endpoint, socket)
        previous = self.currentServer
        self.currentServer = server
        self.cancelEmptyShutdown()
        self.writeRoutedLine(
            socket, {"type": "server_registered", "serverConnectionId": serverConnectionId, "peers": list(self.peers)}
        )
        if previous is not None and previous is not server:
            self.closePublicConnections()
            self.notifyPeers({"type": "server_disconnected", "serverConnectionId": previous.serverConnectionId})
            self.writeRoutedLine(previous.socket, {"type": "server_replaced"})
        self.notifyPeers({"type": "server_connected", "serverConnectionId": serverConnectionId})
        return server

    def registerPeer(self, socket: NetSocket, message: dict[str, Any]) -> _RoutedPeer:
        if message.get("protocol") != COORDINATOR_PROTOCOL_VERSION:
            raise RuntimeError("Unsupported coordinator protocol")
        peerId = message.get("peerId")
        if not isinstance(peerId, str) or len(peerId) == 0:
            raise RuntimeError("Coordinator peerId must be a string")
        if peerId == "server" or peerId in self.peers:
            raise RuntimeError(f"Coordinator peer is already connected: {peerId}")
        peer = _RoutedPeer(peerId, socket)
        self.peers[peerId] = peer
        self.cancelEmptyShutdown()
        self.writeRoutedLine(
            socket,
            {
                "type": "peer_registered",
                "peerId": peerId,
                **({} if self.currentServer is None else {"serverConnectionId": self.currentServer.serverConnectionId}),
            },
        )
        if self.currentServer is not None:
            self.writeRoutedLine(self.currentServer.socket, {"type": "peer_connected", "peerId": peerId})
        return peer

    def notifyPeers(self, message: Any) -> None:
        for peer in list(self.peers.values()):
            self.writeRoutedLine(peer.socket, message)

    def handleRoutedMessage(self, sender: str, message: dict[str, Any]) -> None:
        if message["type"] == "send":
            to = message.get("to")
            if not isinstance(to, str) or len(to) == 0:
                raise RuntimeError("Coordinator message target must be a string")
            if to == "server":
                target = None if self.currentServer is None else self.currentServer.socket
            else:
                peer = self.peers.get(to)
                target = None if peer is None else peer.socket
            if target is not None:
                self.writeRoutedLine(target, self._routedMessage(sender, message))
            return
        if message["type"] == "broadcast":
            if sender != "server":
                raise RuntimeError("Only the current server may broadcast")
            for peer in list(self.peers.values()):
                self.writeRoutedLine(peer.socket, self._routedMessage(sender, message))
            return
        raise RuntimeError(f"Unknown coordinator routing message: {message['type']}")

    @staticmethod
    def _routedMessage(sender: str, message: dict[str, Any]) -> dict[str, Any]:
        # ``payload: message.payload`` leaves the key out of the line when the sender sent none.
        routed: dict[str, Any] = {"type": "message", "from": sender}
        if "payload" in message:
            routed["payload"] = message["payload"]
        return routed

    def acceptPublicConnection(self, client: NetSocket) -> None:
        if self.shuttingDown or self.currentServer is None:
            client.destroy()
            return
        self.cancelEmptyShutdown()
        endpoint = self.currentServer.endpoint

        async def connectUpstream() -> NetSocket | None:
            try:
                return await connectSocket(endpoint)
            except OSError:
                return None

        # The client is tracked from the moment it is accepted, as the connecting upstream socket is upstream: a client
        # that hangs up, a replacement or a shutdown reaches it while the connect is in flight.
        self.publicConnections[client] = None
        spawn(self.pipePublicConnection(client, connectUpstream), self.tasks)

    async def pipePublicConnection(self, client: NetSocket, connectUpstream: Callable[[], Any]) -> None:
        upstreamBox: dict[str, NetSocket | None] = {"socket": None}
        finalized = False

        def finalize() -> None:
            nonlocal finalized
            if finalized:
                return
            finalized = True
            self.publicConnections.pop(client, None)
            self.checkEmpty()

        # Data that arrives before the upstream connection is open waits, as it does before a Node pipe is set up.
        pending: list[bytes] = []
        client.on("data", pending.append)
        client.once("error", lambda _error: upstreamBox["socket"] is not None and upstreamBox["socket"].destroy())
        client.once("close", lambda: (upstreamBox["socket"] is not None and upstreamBox["socket"].destroy(), finalize()))
        upstream = await connectUpstream()
        if upstream is None:
            client.destroy()
            finalize()
            return
        upstreamBox["socket"] = upstream
        if client.destroyed:
            upstream.destroy()
            finalize()
            return
        if client in self.publicConnections:
            self.publicConnections[client] = upstream
        client.off("data", pending.append)
        for chunk in pending:
            upstream.write(chunk)
        client.on("data", lambda chunk: upstream.write(chunk))
        upstream.on("data", lambda chunk: client.write(chunk))
        upstream.once("error", lambda _error: client.destroy())
        upstream.once("close", lambda: (client.destroy(), finalize()))

    def closePublicConnections(self) -> None:
        for client, upstream in list(self.publicConnections.items()):
            client.destroy()
            if upstream is not None:
                upstream.destroy()
        self.publicConnections.clear()

    def checkEmpty(self) -> None:
        if (
            self.shuttingDown
            or self.currentServer is not None
            or len(self.peers) > 0
            or len(self.publicConnections) > 0
            or len(self.controlConnections) > 0
        ):
            self.cancelEmptyShutdown()
            return
        self.scheduleEmptyShutdown(EMPTY_SHUTDOWN_GRACE_MS)

    def scheduleEmptyShutdown(self, delayMs: int) -> None:
        if self.emptyTimer is not None or self.shuttingDown:
            return

        def expire() -> None:
            self.emptyTimer = None
            if (
                self.currentServer is None
                and len(self.peers) == 0
                and len(self.publicConnections) == 0
                and len(self.controlConnections) == 0
            ):
                self.startShutdown()

        self.emptyTimer = asyncio.get_running_loop().call_later(delayMs / 1000, expire)

    def cancelEmptyShutdown(self) -> None:
        if self.emptyTimer is None:
            return
        self.emptyTimer.cancel()
        self.emptyTimer = None

    def startShutdown(self) -> None:
        task = asyncio.get_running_loop().create_task(self.shutdownCoordinator())
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

    async def shutdownCoordinator(self) -> None:
        if self.shuttingDown:
            return
        self.shuttingDown = True
        self.cancelEmptyShutdown()
        self.closePublicConnections()
        for connection in list(self.controlConnections):
            connection.destroy()
        await asyncio.gather(closeServer(self.publicServer), closeServer(self.controlServer))
        await asyncio.gather(cleanupSocket(self.publicPath), cleanupSocket(self.controlPath))
        if not self.finished.done():
            self.finished.set_result(None)

    @staticmethod
    def writeRoutedLine(socket: NetSocket, message: Any) -> None:
        if not socket.destroyed:
            socket.write(encodeControlLine(message).encode("utf-8"))


async def runCoordinatorProcess(args: list[str]) -> None:
    """Run the coordinator until it retires: when no server, peer or connection is left it shuts down by itself, and a
    ``SIGINT`` or ``SIGTERM`` stops it at once."""
    global _running
    if _running:
        raise RuntimeError("Coordinator process is already running")
    requestedPublicPath, requestedControlPath = (args + [None, None])[:2]
    if not requestedPublicPath or not requestedControlPath:
        raise RuntimeError("Coordinator requires public and control socket paths")
    _running = True
    coordinator = _CoordinatorProcess(requestedPublicPath, requestedControlPath)
    if sys.platform != "win32":
        onceSignal(signal.SIGINT, coordinator.startShutdown)
        onceSignal(signal.SIGTERM, coordinator.startShutdown)
    await coordinator.main()
    await coordinator.finished


def main() -> None:
    role = consumeInternalProcessRole()
    if role != "coordinator":
        raise RuntimeError("Coordinator entrypoint requires an internal coordinator invocation")
    try:
        asyncio.run(runCoordinatorProcess(sys.argv[1:]))
    except Exception:  # noqa: BLE001 - the process reports its failure and exits
        traceback.print_exc()
        sys.exit(1)


def attachRoutedLineReader(socket: NetSocket, onMessage: Callable[[Any], None]) -> None:
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    state = {"buffered": ""}

    def onData(chunk: bytes) -> None:
        state["buffered"] += decoder.decode(chunk)
        if len(state["buffered"].encode("utf-8", "surrogatepass")) > MAX_CONTROL_LINE_BYTES:
            socket.destroy(RuntimeError("Coordinator message is too large"))
            return
        while True:
            newline = state["buffered"].find("\n")
            if newline == -1:
                return
            line = state["buffered"][:newline]
            state["buffered"] = state["buffered"][newline + 1 :]
            try:
                onMessage(jsonParse(line))
            except Exception as error:  # noqa: BLE001 - a bad message ends the connection
                socket.destroy(error)
                return

    socket.on("data", onData)


async def closeServer(server: asyncio.Server | None) -> None:
    if server is None or not server.is_serving():
        return
    server.close()
    await server.wait_closed()


async def restrictSocket(path: str) -> None:
    if sys.platform != "win32":
        await asyncio.to_thread(os.chmod, path, 0o600)


def _isSocket(path: str) -> bool:
    return stat.S_ISSOCK(os.lstat(path).st_mode)


async def removeStaleSocket(path: str) -> None:
    try:
        isSocket = await asyncio.to_thread(_isSocket, path)
    except FileNotFoundError:
        return
    if not isSocket:
        raise RuntimeError(f"Coordinator path is not a socket: {path}")
    try:
        socket = await connectSocket(path)
    except (ConnectionRefusedError, FileNotFoundError):
        live = False
    else:
        socket.destroy()
        live = True
    if live:
        raise RuntimeError(f"Coordinator socket is already active: {path}")
    await cleanupSocket(path)


def _unlink(path: str) -> None:
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


async def cleanupSocket(path: str) -> None:
    if sys.platform == "win32":
        return
    await asyncio.to_thread(_unlink, path)


_running = False


if __name__ == "__main__":
    main()
