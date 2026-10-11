"""Public types of chord.

Translated from pi's ``packages/chord/src/types.ts``. The TypeScript-level conditional types that check a
service contract (``JsonRepresentation``, ``RemoteServiceContract``) have no counterpart in Python and are
left out; the provider checks the same things at run time.

Shapes that cross a boundary (calls, updates, snapshots, catalogue entries) are plain ``dict`` values with
upstream's key names, as the operations are lists: they are the wire form. ``UNDEFINED``
(``misaka.utils.js``) stands for JavaScript's ``undefined``, ``None`` for ``null``. A listener may be a plain
function or a coroutine function.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, NotRequired, Protocol, TypedDict

from misaka.chord.delta import Draft, Op
from misaka.chord.json import JsonValue

if TYPE_CHECKING:
    from misaka.chord.services.provider import RemoteServiceProvider
    from misaka.utils.js import AbortSignal


class _Token:
    """The unique identity of a context key; its ``description`` names it in a context's text form."""

    __slots__ = ("description",)

    def __init__(self, description: str | None) -> None:
        self.description = description

    def __repr__(self) -> str:
        return f"Symbol({self.description or ''})"


@dataclass(frozen=True, slots=True)
class ContextKey[T]:
    """Typed identity for one value carried by a :class:`Context`."""

    token: _Token


class Context(ABC):
    """Immutable invocation-scoped values passed explicitly through operations."""

    @property
    @abstractmethod
    def abortSignal(self) -> AbortSignal | None:
        """The signal that cancels the invocation, or ``None``."""

    @abstractmethod
    def value[T](self, key: ContextKey[T]) -> T:
        """The value for ``key``, or ``UNDEFINED`` when the context holds none."""

    @abstractmethod
    def __str__(self) -> str: ...


class ReplicatedStateDelivery(TypedDict):
    kind: Literal["hydrate", "update"]
    sequence: int


type StateListener[T] = Callable[[T, Context, ReplicatedStateDelivery], Awaitable[None] | None]


class ReplicatedState[T](Protocol):
    @property
    def value(self) -> T:
        """Contract-immutable value, or ``UNDEFINED`` until hydration. It is not frozen and may share
        containers with an in-process provider; consumers must not mutate it. Later updates do not mutate
        previously returned values."""
        ...

    def subscribe(self, listener: StateListener[T]) -> Callable[[], None]:
        """Each subscription serializes callbacks, awaiting hydration before updates. Values are immutable
        and may share unchanged revision data. At most 100 deliveries wait behind the running callback;
        overflow keeps only the newest pending value/context/delivery, so update sequences may skip.
        Failures are reported in isolation and delivery continues. Unsubscribe discards pending work without
        aborting or joining a callback."""
        ...


class MutableReplicatedState[T](ReplicatedState[T], Protocol):
    def change(self, context: Context, mutate: Callable[[Draft[T]], None]) -> None:
        """Atomically publish one synchronous overlay mutation. Draft handles are unusable after the callback
        returns. Values placed through the draft are cloned by value; assigning ``UNDEFINED`` to an object
        key deletes it."""
        ...

    def replace(self, context: Context, value: T) -> None:
        """Atomically take immutable ownership of an alias-free strict-JSON replacement. The caller must not
        mutate the transferred root after this call."""
        ...


class ReplicatedStateSourceFrame[T](TypedDict):
    """One immutable authoritative revision committed after an attachment snapshot."""

    cursor: int  # Monotonic source cursor. The first frame after a snapshot is `snapshot.cursor + 1`.
    value: T  # The exact immutable value produced by this commit.
    ops: Sequence[Op]  # The exact immutable operation batch that produced `value` from the previous revision.
    context: Context


class ReplicatedStateSourceSnapshot[T](TypedDict):
    value: T
    cursor: int


class ReplicatedStateSourceAttachment[T](Protocol):
    @property
    def snapshot(self) -> ReplicatedStateSourceSnapshot[T]:
        """Fixed immutable snapshot captured at the atomic attachment boundary."""
        ...

    def activate(self, listener: Callable[[ReplicatedStateSourceFrame[T]], None]) -> None:
        """Install the sole listener and synchronously drain every buffered frame in source commit order.
        Single-use. After it begins, every new committed frame must also be delivered in order until
        disposal, including commits made reentrantly while a prior frame is being delivered."""
        ...

    def dispose(self) -> None:
        """Stop delivery and release source resources. Disposal must be idempotent."""
        ...


class ReplicatedStateSource[T](Protocol):
    """An authoritative immutable revision source.

    ``attach()`` must synchronously and atomically capture one snapshot and register the returned attachment
    to buffer every later committed frame. The snapshot must include every commit before that boundary;
    buffered frames must include every commit after it, with no overlap or gap. Snapshot values, frame values
    and operation batches are immutable and remain valid after delivery. Chord only publishes these
    references; it never applies or re-diffs them."""

    def attach(self) -> ReplicatedStateSourceAttachment[T]: ...


class ReplicatedStateSourceOptions(TypedDict, total=False):
    # Receives source-contract and publication-listener failures without throwing them into the source.
    onError: Callable[[Exception], None]


class AttachedReplicatedState[T](ReplicatedState[T], Protocol):
    """A synchronously hydrated publication-only state backed by one source attachment."""

    def dispose(self) -> None:
        """Idempotently release the source attachment. The last published value remains readable."""
        ...


type ServiceMode = Literal["singleton", "keyed"]


@dataclass(frozen=True, slots=True)
class Service[T]:
    """Stable identity for one shared service contract."""

    id: str
    # Process-local services accept unrestricted object contracts and are never published remotely.
    local: bool = False


class ServiceSpawner[T](Protocol):
    def spawn(self, key: str, implementation: T) -> Callable[[], None]: ...


class RemoteServices(Protocol):
    def use[T](self, service: Service[T]) -> T: ...

    def observe[T](self, service: Service[T], handler: Callable[[T, Context], Awaitable[None] | None]) -> Callable[[], None]: ...

    async def ready(self, context: Context) -> None:
        """Wait until every currently acquired service has installed its initial snapshot."""
        ...

    async def dispose(self, context: Context) -> None: ...


class ServiceCatalogueEntry(TypedDict):
    serviceId: str
    mode: ServiceMode


class ServiceInstanceAddress(TypedDict):
    key: str
    generation: int


class ServiceMemberSnapshot(TypedDict):
    """``{"name", "kind": "method"}`` or ``{"name", "kind": "state", "sequence", "ops"}``."""

    name: str
    kind: Literal["method", "state"]
    sequence: NotRequired[int]
    ops: NotRequired[Sequence[Op]]


class ServiceInstanceSnapshot(TypedDict):
    members: Sequence[ServiceMemberSnapshot]
    instance: NotRequired[ServiceInstanceAddress]


class ServiceSubscriptionSnapshot(TypedDict):
    serviceId: str
    mode: ServiceMode
    instances: Sequence[ServiceInstanceSnapshot]


class ServiceProviderUpdate(TypedDict):
    """One of ``state`` (``instance``?, ``member``, ``sequence``, ``ops``), ``reset`` (``snapshot``: the full
    subscription rebaseline after an overflow, every state a root replacement at its new sequence),
    ``unavailable``, ``replaced`` (``snapshot``), ``spawned`` (``instance``: a snapshot) and ``closed``
    (``instance``: an address), told apart by ``type``."""

    type: Literal["state", "reset", "unavailable", "replaced", "spawned", "closed"]
    instance: NotRequired[Any]
    member: NotRequired[str]
    sequence: NotRequired[int]
    ops: NotRequired[Sequence[Op]]
    snapshot: NotRequired[Any]


class ServiceCall(TypedDict):
    serviceId: str
    member: str
    # Borrowed immutable values. Chord validates but does not clone them.
    args: Sequence[JsonValue]
    instance: NotRequired[ServiceInstanceAddress]


class ServiceSubscription(Protocol):
    @property
    def snapshot(self) -> ServiceSubscriptionSnapshot:
        """Atomic baseline. Later updates buffer until activation, with a full reset on pending delivery 101."""
        ...

    def activate(self) -> None: ...

    def close(self, context: Context | None = None) -> Awaitable[None] | None: ...


class RemoteServiceTransport(Protocol):
    """Pluggable wire boundary consumed by a remote service binding.

    Implementations choose transport, framing, routing and envelope encoding. Values crossing this boundary
    must remain strict JSON. Chord does not clone values or require a particular application wire protocol;
    adapters own serialization and any isolation copies they require."""

    async def invoke(self, call: ServiceCall, context: Context) -> JsonValue: ...

    async def subscribe(
        self,
        serviceId: str,
        mode: ServiceMode,
        listener: Callable[[ServiceProviderUpdate, Context], None],
        context: Context,
    ) -> ServiceSubscription: ...


class RemoteServiceBindingOptions(TypedDict):
    """``services`` (a sequence of objects with an ``id``) and ``transport`` are required; ``bound``,
    ``onError`` and ``assertAccess`` are optional."""

    services: Sequence[Any]
    transport: RemoteServiceTransport
    bound: NotRequired[bool]
    onError: NotRequired[Callable[[Exception], None]]
    assertAccess: NotRequired[Callable[[], None]]


class RemoteServiceBinding(RemoteServices, Protocol):
    async def rebind(self, bound: bool, context: Context) -> None: ...


class FacetEnvironment(Protocol):
    def use[T](self, service: Service[T]) -> T:
        """Declare a hard dependency on one singleton service and return its stable handle."""
        ...

    def observe[T](self, service: Service[T], handler: Callable[[T, Context], Awaitable[None] | None]) -> None:
        """Declare a hard dependency on a keyed service and observe each live instance."""
        ...

    def provide[T](self, service: Service[T], implementation: T) -> None:
        """Declare and install this facet's singleton implementation of a service."""
        ...

    def provideMany[T](self, service: Service[T]) -> ServiceSpawner[T]:
        """Declare ownership of a multi-instance service and return its deferred spawning capability."""
        ...

    def replicatedState[T](self, initial: T) -> MutableReplicatedState[T]:
        """Create initialized mutable state by taking immutable ownership of an alias-free strict-JSON root.
        The caller must not mutate ``initial`` after this call."""
        ...

    def own(self, disposal: Callable[[], Awaitable[None] | None]) -> None:
        """Give the facet ownership of a resource cleanup function."""
        ...

    def onActivate(self, callback: Callable[[], Awaitable[None] | None]) -> None:
        """Register asynchronous initialization after dependencies are bound and ready."""
        ...

    def onDeactivate(self, callback: Callable[[], Awaitable[None] | None]) -> None:
        """Register final facet teardown."""
        ...


class Facet(Protocol):
    @property
    def id(self) -> str: ...

    def setup(self, env: FacetEnvironment) -> None: ...


class RemoteServiceSourceOpenOptions(TypedDict):
    services: Sequence[Any]
    assertAccess: Callable[[], None]
    onError: Callable[[Exception], None]


class RemoteServiceSource(Protocol):
    @property
    def acceptsUnavailableServices(self) -> bool:
        """Whether this currently unavailable source may provisionally own absent requirements."""
        ...

    async def catalogue(self, context: Context) -> Sequence[ServiceCatalogueEntry]: ...

    def open(self, options: RemoteServiceSourceOpenOptions) -> RemoteServices: ...


class FacetOptions(TypedDict):
    """``facets`` is required; ``serviceSources`` and ``onError`` are optional."""

    facets: Sequence[Facet]
    serviceSources: NotRequired[Sequence[RemoteServiceSource]]
    onError: NotRequired[Callable[[Exception], None]]


class FacetHost(Protocol):
    @property
    def services(self) -> RemoteServiceProvider: ...

    async def reload(self, facets: Sequence[Facet]) -> None:
        """Activate and replace facets with matching IDs without disconnecting consumer service handles."""
        ...

    async def dispose(self) -> None: ...


class LoadedFacets(Protocol):
    @property
    def facets(self) -> Sequence[Facet]: ...

    async def dispose(self) -> None: ...


class FacetLoader(Protocol):
    async def load(self) -> LoadedFacets: ...


__all__ = [
    "AttachedReplicatedState",
    "Context",
    "ContextKey",
    "Facet",
    "FacetEnvironment",
    "FacetHost",
    "FacetLoader",
    "FacetOptions",
    "JsonValue",
    "LoadedFacets",
    "MutableReplicatedState",
    "RemoteServiceBinding",
    "RemoteServiceBindingOptions",
    "RemoteServiceSource",
    "RemoteServiceSourceOpenOptions",
    "RemoteServiceTransport",
    "RemoteServices",
    "ReplicatedState",
    "ReplicatedStateDelivery",
    "ReplicatedStateSource",
    "ReplicatedStateSourceAttachment",
    "ReplicatedStateSourceFrame",
    "ReplicatedStateSourceOptions",
    "ReplicatedStateSourceSnapshot",
    "Service",
    "ServiceCall",
    "ServiceCatalogueEntry",
    "ServiceInstanceAddress",
    "ServiceInstanceSnapshot",
    "ServiceMemberSnapshot",
    "ServiceMode",
    "ServiceProviderUpdate",
    "ServiceSpawner",
    "ServiceSubscription",
    "ServiceSubscriptionSnapshot",
    "StateListener",
]
