"""Translated from pi's ``packages/server/src/transports/unix/types.ts``."""

from __future__ import annotations

from collections.abc import Callable
from typing import NotRequired, TypedDict


class UnixListenerOptions(TypedDict):
    """``path`` is required. ``mode`` is the socket's filesystem permissions (default owner read/write only, 0o600).
    ``maxPendingBytes`` is the most framed bytes queued per connection before a slow peer is disconnected.
    ``maxFrameLength`` derives and validates ``maxPendingBytes`` and must match the server when customized."""

    path: str
    mode: NotRequired[int]
    maxPendingBytes: NotRequired[int]
    gracefulCloseTimeoutMs: NotRequired[int]
    maxFrameLength: NotRequired[int]
    onError: NotRequired[Callable[[Exception], None]]


class UnixServerOptions(TypedDict):
    """The server options except ``listeners``, plus the listener options."""

    serverId: str
    path: str
    handshakeTimeoutMs: NotRequired[int]
    onConnectionCountChanged: NotRequired[Callable[[int], None]]
    mode: NotRequired[int]
    maxPendingBytes: NotRequired[int]
    gracefulCloseTimeoutMs: NotRequired[int]
    maxFrameLength: NotRequired[int]
    onError: NotRequired[Callable[[Exception], None]]


__all__ = ["UnixListenerOptions", "UnixServerOptions"]
