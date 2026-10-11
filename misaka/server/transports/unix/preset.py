"""Translated from pi's ``packages/server/src/transports/unix/preset.ts``."""

from __future__ import annotations

from ...server import Server
from ...types import ServerHost, ServerOptions
from .listener import createUnixListener
from .types import UnixListenerOptions, UnixServerOptions


def createUnixServer(host: ServerHost, options: UnixServerOptions) -> Server:
    """Compose Server with one Unix-domain socket listener."""
    listenerOptions: UnixListenerOptions = {"path": options["path"]}
    serverOptions: ServerOptions = {"listeners": [], "serverId": options["serverId"]}
    for key in ("mode", "maxFrameLength", "maxPendingBytes", "gracefulCloseTimeoutMs", "onError"):
        if key in options:
            listenerOptions[key] = options[key]  # type: ignore[literal-required]
    for key in ("maxFrameLength", "handshakeTimeoutMs", "onConnectionCountChanged", "onError"):
        if key in options:
            serverOptions[key] = options[key]  # type: ignore[literal-required]
    serverOptions["listeners"] = [createUnixListener(listenerOptions)]
    return Server(host, serverOptions)


__all__ = ["createUnixServer"]
