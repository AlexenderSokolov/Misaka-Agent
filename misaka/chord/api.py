"""Chord's entry points.

Translated from pi's ``packages/chord/src/api.ts``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from typing import Any

from misaka.chord.facets.host import FacetKernel
from misaka.chord.facets.loader import disposeLoadedFacets
from misaka.chord.services.consumer import RemoteServiceBindingImpl
from misaka.chord.services.state import (
    MutableReplicatedStateImpl,
    attachReplicatedStateSource,
)
from misaka.chord.types import (
    AttachedReplicatedState,
    Facet,
    FacetHost,
    FacetLoader,
    FacetOptions,
    LoadedFacets,
    MutableReplicatedState,
    RemoteServiceBinding,
    RemoteServiceBindingOptions,
    ReplicatedStateSource,
    ReplicatedStateSourceOptions,
    Service,
)


class _FacetHostImpl:
    def __init__(self, kernel: FacetKernel) -> None:
        self._kernel = kernel

    @property
    def services(self) -> Any:
        return self._kernel.provider

    def reload(self, facets: Sequence[Facet]) -> asyncio.Future[None]:
        return self._kernel.reload(facets)  # type: ignore[no-any-return]

    def dispose(self) -> asyncio.Future[None]:
        return self._kernel.dispose()  # type: ignore[no-any-return]


async def createFacetHost(options: FacetOptions) -> FacetHost:
    """Create an active host for one complete set of facets."""
    kernel = FacetKernel(options)
    await kernel.activate()
    return _FacetHostImpl(kernel)  # type: ignore[return-value]


class _StaticLoadedFacets:
    def __init__(self, facets: tuple[Facet, ...]) -> None:
        self.facets = facets

    async def dispose(self) -> None:
        return None


class _StaticFacetLoader:
    def __init__(self, facets: tuple[Facet, ...]) -> None:
        self._facets = facets

    async def load(self) -> LoadedFacets:
        return _StaticLoadedFacets(self._facets)  # type: ignore[return-value]


def createStaticFacetLoader(facets: Sequence[Facet]) -> FacetLoader:
    return _StaticFacetLoader(tuple(facets))  # type: ignore[return-value]


class _CombinedLoadedFacets:
    def __init__(self, loaded: list[LoadedFacets]) -> None:
        self._loaded = loaded
        self.facets = tuple(facet for entry in loaded for facet in entry.facets)
        self._disposed = False

    async def dispose(self) -> None:
        if self._disposed:
            return
        self._disposed = True
        errors = await disposeLoadedFacets(list(reversed(self._loaded)))
        if len(errors) == 1:
            raise errors[0]
        if len(errors) > 1:
            raise ExceptionGroup("Failed to dispose loaded facets", errors)


class _CombinedFacetLoader:
    def __init__(self, loaders: Sequence[FacetLoader]) -> None:
        self._loaders = list(loaders)

    async def load(self) -> LoadedFacets:
        loaded: list[LoadedFacets] = []
        try:
            for loader in self._loaders:
                loaded.append(await loader.load())
        except Exception as error:
            cleanupErrors = await disposeLoadedFacets(list(reversed(loaded)))
            if cleanupErrors:
                raise ExceptionGroup("Facet loading and cleanup failed", [error, *cleanupErrors]) from None
            raise
        return _CombinedLoadedFacets(loaded)  # type: ignore[return-value]


def combineFacetLoaders(loaders: Sequence[FacetLoader]) -> FacetLoader:
    return _CombinedFacetLoader(loaders)  # type: ignore[return-value]


def defineFacet(facet: Facet) -> Facet:
    return facet


def defineService[T](id: str, options: Mapping[str, bool] | None = None) -> Service[T]:
    if len(id) == 0:
        raise TypeError("Service ID must not be empty")
    # TODO: check if the reserved namespace should be part of Chord.
    if id.startswith("$chord."):
        raise TypeError("Service IDs beginning with $chord. are reserved")
    local = options.get("local") if options is not None else None
    return Service(id, local if local is not None else False)


def createRemoteServiceBinding(options: RemoteServiceBindingOptions) -> RemoteServiceBinding:
    return RemoteServiceBindingImpl(options)  # type: ignore[return-value]


def replicatedState(
    initialOrSource: Any, options: ReplicatedStateSourceOptions | None = None
) -> MutableReplicatedState[Any] | AttachedReplicatedState[Any]:
    """Attach a publication-only state to a source (an object with ``attach()``), or create authoritative
    state by taking immutable ownership of an alias-free strict-JSON ``initial`` root. The caller must not
    mutate ``initial`` after this call."""
    if isReplicatedStateSource(initialOrSource):
        return attachReplicatedStateSource(initialOrSource, options)
    return MutableReplicatedStateImpl(initialOrSource)


def isReplicatedStateSource(value: Any) -> bool:
    return callable(getattr(value, "attach", None))


__all__ = [
    "ReplicatedStateSource",
    "combineFacetLoaders",
    "createFacetHost",
    "createRemoteServiceBinding",
    "createStaticFacetLoader",
    "defineFacet",
    "defineService",
    "replicatedState",
]
