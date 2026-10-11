"""The facet bundle format (pi chord's ``src/node/manifest.ts``).

MISAKA fork: pi bundles each facet entry of a plugin into one content-addressed CommonJS file with esbuild; Python has no
bundler, so a facet *bundle* here is a deterministic zip archive of the entry module and the files it imports relatively
(see ``bundle.py``). An entry of a manifest is the dict ``{"file", "integrity", "module"}``:

- ``file``: the content-addressed archive filename, relative to the manifest;
- ``integrity``: the SHA-256 subresource-integrity value of the archive (``sha256-<base64>``);
- ``module``: the slash-separated path of the entry module inside the archive.

A manifest is ``{"format", "formatVersion", "plugin": {"id", "version"?}, "entries": {name: entry}}``. An artifact is one
self-contained entry suitable for storage or transport to another host: ``{"format", "formatVersion", "plugin",
"entryName", "entry", "source"}`` where ``source`` is the archive, base64 encoded.

The format ids and versions are upstream's, kept deliberately: they name the chord bundle format a manifest belongs to.
A MISAKA archive and a pi CommonJS bundle are still not interchangeable; each host loads only its own.
"""

from __future__ import annotations

FACET_BUNDLE_FORMAT = "chord.facet-bundle"
FACET_BUNDLE_FORMAT_VERSION = 2
FACET_BUNDLE_MANIFEST_FILE = "chord-facets.json"
FACET_BUNDLE_ARTIFACT_FORMAT = "chord.facet-bundle-artifact"
FACET_BUNDLE_ARTIFACT_FORMAT_VERSION = 2

__all__ = [
    "FACET_BUNDLE_ARTIFACT_FORMAT",
    "FACET_BUNDLE_ARTIFACT_FORMAT_VERSION",
    "FACET_BUNDLE_FORMAT",
    "FACET_BUNDLE_FORMAT_VERSION",
    "FACET_BUNDLE_MANIFEST_FILE",
]
