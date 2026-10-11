"""Builds a plugin package into facet bundles (pi chord's ``src/node/package.ts``).

MISAKA fork: a package's metadata is its ``pyproject.toml`` (``[project]`` ``name`` and ``version``) where upstream reads
``package.json``. Its ``[tool.chord]`` table may hold ``facets``: a table mapping an entry name to a source path relative
to the package directory, or to ``false`` to drop an entry the application would otherwise find by convention. The
``external`` and ``sourceMap`` settings and the peer dependencies of a ``package.json`` have no counterpart: imports other
than relative ones are never bundled.

``bundleFacetPackage(options)`` takes ``{"packagePath", "outdir", "defaultFacets"?}`` (``packagePath`` is the package
directory or its ``pyproject.toml``) and returns the result of ``bundleFacets`` plus ``packageDirectory`` and
``packageMetadataPath``.
"""

from __future__ import annotations

import asyncio
import os
import tomllib
from typing import Any

from misaka.chord.node.bundle import bundleFacets

METADATA_FILE = "pyproject.toml"


async def bundleFacetPackage(options: dict[str, Any]) -> dict[str, Any]:
    """Build a plugin package using its metadata and application-provided facet conventions."""
    metadata = await asyncio.to_thread(readFacetPackageMetadata, options["packagePath"])
    entries = await asyncio.to_thread(resolveFacetEntries, metadata, options.get("defaultFacets") or {})
    result = await bundleFacets(
        {
            "plugin": {"id": metadata["name"], "version": metadata["version"]},
            "entries": entries,
            "outdir": options["outdir"],
            "workingDirectory": metadata["packageDirectory"],
        }
    )
    return {
        **result,
        "packageDirectory": metadata["packageDirectory"],
        "packageMetadataPath": metadata["packageMetadataPath"],
    }


def readFacetPackageMetadata(packagePath: str) -> dict[str, Any]:
    if len(packagePath) == 0:
        raise TypeError("Facet package path must not be empty")
    candidate = os.path.abspath(packagePath)
    try:
        isDirectory, isFile = os.path.isdir(candidate), os.path.isfile(candidate)
        os.stat(candidate)
    except OSError as error:
        raise RuntimeError(f"Could not access facet package {candidate}") from error
    if isDirectory:
        packageDirectory = os.path.realpath(candidate)
        packageMetadataPath = os.path.join(packageDirectory, METADATA_FILE)
    elif isFile and os.path.basename(candidate) == METADATA_FILE:
        packageMetadataPath = os.path.realpath(candidate)
        packageDirectory = os.path.dirname(packageMetadataPath)
    else:
        raise RuntimeError(f"Facet package path must name a directory or {METADATA_FILE}: {candidate}")

    try:
        with open(packageMetadataPath, "rb") as file:
            # The text is read leniently, as ``readFile(path, "utf8")`` does: invalid bytes become U+FFFD.
            parsed = tomllib.loads(file.read().decode("utf-8", errors="replace"))
    except (OSError, ValueError, RecursionError) as error:
        raise RuntimeError(f"Could not read facet package metadata {packageMetadataPath}") from error
    project = parsed.get("project")
    if not isinstance(project, dict):
        raise RuntimeError(f"Facet package metadata must have a [project] table: {packageMetadataPath}")  # noqa: TRY004
    if not isinstance(project.get("name"), str) or len(project["name"]) == 0:
        raise RuntimeError(f"Facet package must have a non-empty name: {packageMetadataPath}")
    if not isinstance(project.get("version"), str) or len(project["version"]) == 0:
        raise RuntimeError(f"Facet package must have a non-empty version: {packageMetadataPath}")
    tool = parsed.get("tool")
    if tool is not None and not isinstance(tool, dict):
        raise RuntimeError(f"Facet package chord configuration must be a table: {packageMetadataPath}")
    chord = parseChordConfiguration(None if tool is None else tool.get("chord"), packageMetadataPath)
    return {
        "packageDirectory": packageDirectory,
        "packageMetadataPath": packageMetadataPath,
        "name": project["name"],
        "version": project["version"],
        "configuredFacets": chord,
    }


def parseChordConfiguration(value: Any, packageMetadataPath: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise RuntimeError(f"Facet package chord configuration must be a table: {packageMetadataPath}")  # noqa: TRY004
    if any(key != "facets" for key in value):
        raise RuntimeError(f"Facet package chord configuration has an unknown field: {packageMetadataPath}")
    facets: dict[str, Any] = {}
    configured = value.get("facets")
    if configured is not None:
        if not isinstance(configured, dict):
            raise RuntimeError(f"Facet package chord.facets must be a table: {packageMetadataPath}")
        for name, source in configured.items():
            if len(name) == 0 or (not isinstance(source, str) and source is not False) or source == "":
                raise RuntimeError(f"Facet package has an invalid chord.facets entry: {packageMetadataPath}")
            facets[name] = source
    return facets


def resolveFacetEntries(metadata: dict[str, Any], defaultFacets: dict[str, str]) -> dict[str, str]:
    entries: dict[str, str] = {}
    for name, source in defaultFacets.items():
        validateFacetMapping(name, source, "default")
        path = resolvePackageEntry(metadata["packageDirectory"], source, name)
        try:
            os.stat(path)
        except FileNotFoundError:
            continue
        if not os.path.isfile(path):
            raise RuntimeError(f"Default facet entry {name} is not a file: {path}")
        canonicalPath = os.path.realpath(path)
        validateCanonicalPackageEntry(metadata["packageDirectory"], canonicalPath, name)
        entries[name] = canonicalPath
    for name, source in metadata["configuredFacets"].items():
        if source is False:
            entries.pop(name, None)
            continue
        validateFacetMapping(name, source, "configured")
        path = resolvePackageEntry(metadata["packageDirectory"], source, name)
        try:
            os.stat(path)
        except OSError as error:
            raise RuntimeError(f"Could not access configured facet entry {name}: {path}") from error
        if not os.path.isfile(path):
            raise RuntimeError(f"Configured facet entry {name} is not a file: {path}")
        canonicalPath = os.path.realpath(path)
        validateCanonicalPackageEntry(metadata["packageDirectory"], canonicalPath, name)
        entries[name] = canonicalPath
    if len(entries) == 0:
        raise RuntimeError(f"Facet package {metadata['name']} has no configured or conventional facet entries")
    return entries


def validateFacetMapping(name: str, source: str, kind: str) -> None:
    if len(name) == 0:
        raise RuntimeError(f"Facet package {kind} entry name must not be empty")
    if len(source) == 0:
        raise RuntimeError(f"Facet package {kind} entry {name} must have a source path")


def resolvePackageEntry(packageDirectory: str, source: str, name: str) -> str:
    if os.path.isabs(source):
        raise RuntimeError(f"Facet package entry {name} must be relative to the package directory")
    path = os.path.abspath(os.path.join(packageDirectory, source))
    try:
        relativePath = os.path.relpath(path, packageDirectory)
    except ValueError:
        # A path on another drive has no relative form.
        raise RuntimeError(f"Facet package entry {name} escapes the package directory") from None
    if (
        len(relativePath) == 0
        or relativePath == "."
        or relativePath == ".."
        or relativePath.startswith(f"..{os.sep}")
        or os.path.isabs(relativePath)
    ):
        raise RuntimeError(f"Facet package entry {name} escapes the package directory")
    return path


def validateCanonicalPackageEntry(packageDirectory: str, path: str, name: str) -> None:
    try:
        relativePath = os.path.relpath(path, packageDirectory)
    except ValueError:
        raise RuntimeError(f"Facet package entry {name} resolves outside the package directory") from None
    if relativePath == ".." or relativePath.startswith(f"..{os.sep}") or os.path.isabs(relativePath):
        raise RuntimeError(f"Facet package entry {name} resolves outside the package directory")
