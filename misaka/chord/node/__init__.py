"""Facet bundles: building plugin packages into archives, and loading them in another process (pi chord's ``node``).

MISAKA fork: see ``manifest.py``. ``bundleFacets`` and ``bundleFacetPackage`` build, ``readFacetBundleManifest``,
``readFacetBundleArtifact``, ``createFacetBundleLoader`` and ``createFacetBundleArtifactLoader`` read and load.
"""

from __future__ import annotations

from misaka.chord.node.bundle import bundleFacets
from misaka.chord.node.bundle_loader import (
    createFacetBundleArtifactLoader,
    createFacetBundleLoader,
    readFacetBundleArtifact,
    readFacetBundleManifest,
)
from misaka.chord.node.manifest import (
    FACET_BUNDLE_ARTIFACT_FORMAT,
    FACET_BUNDLE_ARTIFACT_FORMAT_VERSION,
    FACET_BUNDLE_FORMAT,
    FACET_BUNDLE_FORMAT_VERSION,
    FACET_BUNDLE_MANIFEST_FILE,
)
from misaka.chord.node.package import bundleFacetPackage

__all__ = [
    "FACET_BUNDLE_ARTIFACT_FORMAT",
    "FACET_BUNDLE_ARTIFACT_FORMAT_VERSION",
    "FACET_BUNDLE_FORMAT",
    "FACET_BUNDLE_FORMAT_VERSION",
    "FACET_BUNDLE_MANIFEST_FILE",
    "bundleFacetPackage",
    "bundleFacets",
    "createFacetBundleArtifactLoader",
    "createFacetBundleLoader",
    "readFacetBundleArtifact",
    "readFacetBundleManifest",
]
