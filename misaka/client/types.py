"""Translated from pi's ``packages/client/src/types.ts``."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, Literal, NotRequired, Protocol, TypedDict

from misaka.chord.types import ServiceSubscriptionSnapshot
from misaka.protocol import RpcTarget, SessionTarget

from .transport import ByteTransportFactory

type ConnectionState = Literal["disconnected", "connecting", "connected"]


class ConnectionStateChange(TypedDict):
    state: ConnectionState
    error: NotRequired[Exception]


type Unsubscribe = Callable[[], None]
type ListenerErrorHandler = Callable[[Exception], None]
type AttachmentChangeListener = Callable[[SessionTarget | None], None]


class ServiceSubscription(Protocol):
    @property
    def id(self) -> str: ...

    @property
    def target(self) -> RpcTarget: ...

    @property
    def snapshot(self) -> ServiceSubscriptionSnapshot: ...

    def start(self) -> None:
        """Begin ordered update delivery after the caller has installed the snapshot."""
        ...

    def dispose(self) -> Awaitable[None]: ...


class ClientOptions(TypedDict):
    """``transportFactory`` and ``serverId`` (the logical server identity expected at the physical endpoint)
    are required; ``maxFrameLength`` and ``onListenerError`` (which reports subscriber failures without letting
    them corrupt client state) are optional."""

    transportFactory: ByteTransportFactory
    serverId: str
    maxFrameLength: NotRequired[int]
    onListenerError: NotRequired[ListenerErrorHandler]


type ServiceResult = Any

__all__ = [
    "AttachmentChangeListener",
    "ClientOptions",
    "ConnectionState",
    "ConnectionStateChange",
    "ListenerErrorHandler",
    "ServiceResult",
    "ServiceSubscription",
    "Unsubscribe",
]
