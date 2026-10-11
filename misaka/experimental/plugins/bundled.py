"""Translated from pi's ``coding-agent/src/experimental/plugins/bundled.ts``.

MISAKA fork: pi resolves a bundle's external import of ``@earendil-works/pi-coding-agent/experimental/plugin`` to its own
plugin API module; a Python plugin simply imports ``misaka.experimental.plugin``, so no resolver is needed.
"""

from __future__ import annotations

from typing import Any

from misaka.chord import combineFacetLoaders
from misaka.chord.node import (
    createFacetBundleArtifactLoader,
    createFacetBundleLoader,
    readFacetBundleManifest,
)

PRESENTATION_FACET_BUNDLES_KEY = "presentationFacetBundles"


def createSessionPluginFacetLoader(manifestPaths: list[str] | tuple[str, ...]) -> Any:
    if len(manifestPaths) == 0:
        return None
    return combineFacetLoaders([createOptionalSessionFacetLoader(manifestPath) for manifestPath in manifestPaths])


class _EmptyLoadedFacets:
    facets: tuple[Any, ...] = ()

    async def dispose(self) -> None:
        return None


def createOptionalSessionFacetLoader(manifestPath: str) -> Any:
    loader = createFacetBundleLoader({"manifestPath": manifestPath, "entry": "session"})

    class OptionalLoader:
        async def load(self) -> Any:
            manifest = await readFacetBundleManifest(manifestPath)
            if "session" in manifest["entries"]:
                return await loader.load()
            return _EmptyLoadedFacets()

    return OptionalLoader()


def createPresentationFacetData(artifacts: list[dict[str, Any]]) -> dict[str, Any]:
    return {PRESENTATION_FACET_BUNDLES_KEY: [dict(artifact) for artifact in artifacts]}


def createPresentationFacetLoaders(data: Any) -> list[Any]:
    """Create local loaders only from artifacts selected and sent by the connected server."""
    if data is None or isinstance(data, list) or not isinstance(data, dict):
        raise RuntimeError("Invalid presentation plugin data")
    if PRESENTATION_FACET_BUNDLES_KEY not in data:
        return []
    artifacts = data[PRESENTATION_FACET_BUNDLES_KEY]
    if not isinstance(artifacts, list):
        raise RuntimeError("Invalid presentation plugin bundle list")  # noqa: TRY004
    return [createFacetBundleArtifactLoader({"artifact": artifact}) for artifact in artifacts]


__all__ = [
    "createPresentationFacetData",
    "createPresentationFacetLoaders",
    "createSessionPluginFacetLoader",
]
