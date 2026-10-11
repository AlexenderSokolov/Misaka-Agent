"""Server options and the application capabilities the server routes through.

Translated from pi's ``packages/server/src/types.ts``. A value that is ``MaybePromise`` may be a plain value or an
awaitable. ``undefined`` is ``misaka.utils.js.UNDEFINED`` and ``null`` is ``None``: a service call that returns
``UNDEFINED`` answers without a ``result``, one that returns ``None`` answers ``null``.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from typing import TYPE_CHECKING, Any, NotRequired, Protocol, TypedDict

from misaka.chord.types import Context, JsonValue, ServiceCall, ServiceProviderUpdate

if TYPE_CHECKING:
    from .listener import ServerListener

type MaybePromise[T] = T | Awaitable[T]
type PublishServiceUpdate = Callable[[str, ServiceProviderUpdate, Context], MaybePromise[None]]

# Minimal durable Session metadata the server needs for routing: a mapping with an ``id`` (or an object with an
# ``id`` attribute). Applications may extend it.
type SessionMetadata = Any


class ServerOptions(TypedDict):
    """``listeners`` and ``serverId`` (the stable logical server identity supplied by the installation or profile)
    are required."""

    listeners: Sequence[ServerListener]
    serverId: str
    maxFrameLength: NotRequired[int]
    handshakeTimeoutMs: NotRequired[int]
    onConnectionCountChanged: NotRequired[Callable[[int], None]]
    onError: NotRequired[Callable[[Exception], None]]


class RoutedSessionAttachment(Protocol):
    """One presentation connection's live capability for a hosted Session."""

    def invokeService(
        self, call: ServiceCall, publish: PublishServiceUpdate, context: Context
    ) -> Awaitable[JsonValue | Any]:
        """Route one contract-agnostic service operation to the attached Session endpoint."""
        ...

    def release(self, context: Context) -> MaybePromise[None]: ...


class RoutedServerPresentation(Protocol):
    """Presentation-scoped routing capabilities available to server service implementations."""

    def attachSession(self, sessionId: str, context: Context) -> Awaitable[None]: ...

    def detachSession(self, context: Context) -> Awaitable[None]: ...

    def prepareSessionRemoval(self, sessionId: str, context: Context) -> Awaitable[None]:
        """Release routed attachments and handles before the application deletes durable metadata."""
        ...


class RoutedServerServiceAttachment(Protocol):
    """One connection's server-scoped service endpoint."""

    def invokeService(
        self, call: ServiceCall, publish: PublishServiceUpdate, context: Context
    ) -> Awaitable[JsonValue | Any]: ...

    def release(self, context: Context) -> MaybePromise[None]: ...


class RoutedServerServiceHost(Protocol):
    def attachClient(
        self, presentation: RoutedServerPresentation, context: Context
    ) -> MaybePromise[RoutedServerServiceAttachment]: ...


class RoutedSessionHandle(Protocol):
    """A process-safe handle that acquires presentation-scoped Session capabilities. An optional ``terminated``
    future resolves with an error for unexpected termination, or ``None`` after an expected close."""

    def attachClient(self, context: Context) -> MaybePromise[RoutedSessionAttachment]: ...

    def close(self, context: Context) -> Awaitable[None]: ...


class ServerHost(Protocol):
    """Application capabilities used by server-wide management and Session routing."""

    @property
    def serverServices(self) -> RoutedServerServiceHost: ...

    def resolveSession(self, sessionId: str, context: Context) -> Awaitable[SessionMetadata]:
        """Resolve one durable Session ID or raise a bounded routing error."""
        ...

    def openSession(self, metadata: SessionMetadata, context: Context) -> Awaitable[RoutedSessionHandle]: ...


__all__ = [
    "MaybePromise",
    "PublishServiceUpdate",
    "RoutedServerPresentation",
    "RoutedServerServiceAttachment",
    "RoutedServerServiceHost",
    "RoutedSessionAttachment",
    "RoutedSessionHandle",
    "ServerHost",
    "ServerOptions",
    "SessionMetadata",
]
