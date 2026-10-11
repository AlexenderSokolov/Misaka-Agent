"""Byte connections and the per-connection state a server keeps.

Translated from pi's ``packages/server/src/connection.ts``. A connection's ``send`` returns an awaitable and
its ``close`` may return one; the handler an acceptor returns receives data, close and error events.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any, Literal, Protocol

from misaka.chord.services.state_codec import ServiceStateEncoder
from misaka.protocol import ClientMessageDecoder, RpcTarget

from .types import RoutedServerServiceAttachment


class ByteConnection(Protocol):
    """An established, authorized ordered byte connection."""

    @property
    def closed(self) -> bool: ...

    def send(self, chunk: bytes) -> Awaitable[None]: ...

    def close(self, finalChunk: bytes | None = None) -> Awaitable[None] | None: ...


class ByteConnectionHandler(Protocol):
    def onData(self, chunk: bytes) -> None: ...

    def onClose(self) -> None: ...

    def onError(self, error: Exception) -> None: ...


type ByteConnectionAcceptor = Callable[[ByteConnection], ByteConnectionHandler]

type ConnectionStage = Literal["awaitingHello", "handshaking", "ready", "closing", "closed"]


class ActiveRequest:
    __slots__ = ("controller", "target")

    def __init__(self, controller: Any, target: RpcTarget) -> None:
        self.controller = controller
        self.target = target


class ConnectionState:
    """Identity-hashed: the session router keys its maps by the connection object."""

    __slots__ = (
        "activeRequests",
        "connection",
        "decoder",
        "disconnected",
        "handshake",
        "handshakeTimeout",
        "serverServices",
        "serviceStateEncoders",
        "stage",
    )

    def __init__(self, connection: ByteConnection, decoder: ClientMessageDecoder) -> None:
        self.connection = connection
        self.decoder = decoder
        self.serviceStateEncoders: dict[str, ServiceStateEncoder] = {}
        self.stage: ConnectionStage = "awaitingHello"
        self.disconnected = False
        self.handshake: asyncio.Future[None] | None = None
        self.handshakeTimeout: asyncio.TimerHandle | None = None
        self.serverServices: RoutedServerServiceAttachment | None = None
        self.activeRequests: dict[str, ActiveRequest] = {}


def isTerminalConnection(state: ConnectionState) -> bool:
    return state.disconnected or state.stage == "closing" or state.stage == "closed"


__all__ = [
    "ActiveRequest",
    "ByteConnection",
    "ByteConnectionAcceptor",
    "ByteConnectionHandler",
    "ConnectionStage",
    "ConnectionState",
    "isTerminalConnection",
]
