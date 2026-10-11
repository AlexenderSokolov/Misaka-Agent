"""Reads and loads facet bundles (pi chord's ``src/node/bundle-loader.ts``).

MISAKA fork: see ``manifest.py``. Loading a bundle extracts its archive into a fresh directory and imports the entry module
as a submodule of a package that exists only for that load (``chord_facet_<id>``), so each load is an independent
generation: module state is not shared with an earlier load, and relative imports find the files of the same archive.
Disposing the loaded facets removes those modules and the directory.

A facet entry module exports its facets as ``default``: a facet (an object with an ``id`` and a ``setup``) or a sequence of
them, as the TypeScript entry has ``export default``.

Loader options are dicts: ``{"manifestPath", "entry", "verifyIntegrity"?, "temporaryDirectory"?}`` for
``createFacetBundleLoader`` and ``{"artifact", "temporaryDirectory"?}`` for ``createFacetBundleArtifactLoader``.
"""

from __future__ import annotations

import asyncio
import base64
import copy
import hashlib
import importlib.machinery
import importlib.util
import io
import json
import os
import shutil
import sys
import tempfile
import uuid
import zipfile
from typing import Any

from misaka.chord.node.manifest import (
    FACET_BUNDLE_ARTIFACT_FORMAT,
    FACET_BUNDLE_ARTIFACT_FORMAT_VERSION,
    FACET_BUNDLE_FORMAT,
    FACET_BUNDLE_FORMAT_VERSION,
    FACET_BUNDLE_MANIFEST_FILE,
)
from misaka.utils.js import UNDEFINED, errorMessage, jsonParse, jsString


def _readManifestFile(path: str) -> Any:
    # ``readFile(path, "utf8")`` replaces invalid bytes, and ``JSON.parse`` rejects ``NaN`` and ``Infinity``.
    with open(path, "rb") as file:
        return jsonParse(file.read().decode("utf-8", errors="replace"))


def _removeTree(path: str) -> None:
    """``rm(path, {recursive: true, force: true})``: only a missing path is not an error."""
    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        pass


async def readFacetBundleManifest(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Read and validate a versioned facet bundle manifest."""
    manifestPath = os.path.abspath(os.fspath(path))
    try:
        parsed = await asyncio.to_thread(_readManifestFile, manifestPath)
    except Exception as error:
        raise RuntimeError(f"Could not read facet bundle manifest {manifestPath}") from error
    return validateManifest(parsed, manifestPath)


def _readBytes(path: str) -> bytes:
    with open(path, "rb") as file:
        return file.read()


async def readFacetBundleArtifact(options: dict[str, Any]) -> dict[str, Any]:
    """Read and verify one transportable entry from a facet bundle on disk."""
    if len(options["entry"]) == 0:
        raise TypeError("Facet bundle entry name must not be empty")
    manifestPath = os.path.abspath(os.fspath(options["manifestPath"]))
    manifest = await readFacetBundleManifest(manifestPath)
    entry = manifest["entries"].get(options["entry"])
    if entry is None:
        raise RuntimeError(f"Facet bundle {manifest['plugin']['id']} has no entry named {options['entry']}")
    contents = await asyncio.to_thread(_readBytes, resolveBundleFile(manifestPath, entry["file"], "entry"))
    verifyContents(contents, entry)
    return {
        "format": FACET_BUNDLE_ARTIFACT_FORMAT,
        "formatVersion": FACET_BUNDLE_ARTIFACT_FORMAT_VERSION,
        "plugin": manifest["plugin"],
        "entryName": options["entry"],
        "entry": entry,
        "source": base64.b64encode(contents).decode("ascii"),
    }


class _LoadedFacets:
    def __init__(self, facets: tuple[Any, ...], cleanup: Any) -> None:
        self._facets = facets
        self._cleanup = cleanup
        self._disposed = False

    @property
    def facets(self) -> tuple[Any, ...]:
        return self._facets

    async def dispose(self) -> None:
        if self._disposed:
            return
        self._disposed = True
        self._facets = ()
        await self._cleanup()


class _ArtifactFacetLoader:
    def __init__(self, artifact: dict[str, Any], temporaryDirectory: str | None) -> None:
        self._artifact = artifact
        self._temporaryParent = os.path.abspath(temporaryDirectory or tempfile.gettempdir())

    async def load(self) -> Any:
        artifact = self._artifact
        await asyncio.to_thread(os.makedirs, self._temporaryParent, 0o777, True)
        directory = await asyncio.to_thread(tempfile.mkdtemp, prefix="chord-facet-", dir=self._temporaryParent)
        try:
            await asyncio.to_thread(materializeArtifact, directory, artifact)
            loaded = await createFacetBundleLoader(
                {
                    "manifestPath": os.path.join(directory, FACET_BUNDLE_MANIFEST_FILE),
                    "entry": artifact["entryName"],
                    # The generation is extracted inside the materialized directory, so nothing lands outside it.
                    "temporaryDirectory": directory,
                }
            ).load()
            disposed = False

            async def cleanup() -> None:
                nonlocal disposed
                if disposed:
                    return
                disposed = True
                errors: list[Exception] = []
                try:
                    await loaded.dispose()
                except Exception as error:  # noqa: BLE001 - collected
                    errors.append(error)
                try:
                    await asyncio.to_thread(_removeTree, directory)
                except Exception as error:  # noqa: BLE001 - collected
                    errors.append(error)
                if len(errors) == 1:
                    raise errors[0]
                if len(errors) > 1:
                    raise ExceptionGroup("Failed to dispose facet bundle artifact", errors)

            return _LoadedFacets(tuple(loaded.facets), cleanup)
        except BaseException as error:
            try:
                await asyncio.to_thread(_removeTree, directory)
            except Exception as cleanupError:  # noqa: BLE001
                raise ExceptionGroup(
                    "Facet bundle artifact loading and cleanup failed", [_asException(error), cleanupError]
                ) from None
            raise


def createFacetBundleArtifactLoader(options: dict[str, Any]) -> Any:
    """Materialize a transported artifact and create a fresh generation for each load."""
    return _ArtifactFacetLoader(validateArtifact(options["artifact"]), options.get("temporaryDirectory"))


class _BundleFacetLoader:
    def __init__(self, options: dict[str, Any]) -> None:
        self._entry = options["entry"]
        self._manifestPath = os.path.abspath(os.fspath(options["manifestPath"]))
        self._verifyIntegrity = options.get("verifyIntegrity") is not False
        self._temporaryDirectory = options.get("temporaryDirectory")

    async def load(self) -> Any:
        manifest = await readFacetBundleManifest(self._manifestPath)
        entry = manifest["entries"].get(self._entry)
        if entry is None:
            raise RuntimeError(f"Facet bundle {manifest['plugin']['id']} has no entry named {self._entry}")
        archivePath = resolveBundleFile(self._manifestPath, entry["file"], "entry")
        rootName = f"chord_facet_{uuid.uuid4().hex[:12]}"
        generation: str | None = None
        try:
            generation = await asyncio.to_thread(
                tempfile.mkdtemp, prefix="chord-facet-gen-", dir=self._temporaryDirectory or tempfile.gettempdir()
            )
            contents = await asyncio.to_thread(_readBytes, archivePath)
            if self._verifyIntegrity:
                verifyContents(contents, entry)
            await asyncio.to_thread(extractGeneration, contents, generation)
            # The module's code runs on the loop's thread, as the CommonJS module of pi's loader runs on the main one.
            facets = importEntry(generation, rootName, entry["module"], manifest["plugin"]["id"], self._entry)
        except BaseException as error:
            unloadGeneration(rootName, generation)
            if generation is not None:
                await asyncio.to_thread(_removeTree, generation)
            if isinstance(error, Exception):
                detail = f": {errorMessage(error)}"
                raise RuntimeError(  # noqa: TRY004
                    f"Could not load facet bundle entry {manifest['plugin']['id']}/{self._entry}{detail}"
                ) from error
            raise

        async def cleanup() -> None:
            unloadGeneration(rootName, generation)
            await asyncio.to_thread(_removeTree, generation)

        return _LoadedFacets(facets, cleanup)


def createFacetBundleLoader(options: dict[str, Any]) -> Any:
    """Create a reusable loader for one opaque entry in a facet bundle manifest."""
    if len(options["entry"]) == 0:
        raise TypeError("Facet bundle entry name must not be empty")
    return _BundleFacetLoader(options)


def _asException(error: BaseException) -> Exception:
    return error if isinstance(error, Exception) else RuntimeError(str(error))


def extractGeneration(contents: bytes, generation: str) -> None:
    """Extract the archive into the ``generation`` directory."""
    with zipfile.ZipFile(io.BytesIO(contents)) as archive:
        for member in archive.namelist():
            resolveInside(generation, member)
        archive.extractall(generation)


def importEntry(generation: str, rootName: str, modulePath: str, pluginId: str, entryName: str) -> tuple[Any, ...]:
    """Import the entry module of the archive extracted into ``generation`` under the package ``rootName``."""
    parts = modulePath.split("/")
    directories, filename = parts[:-1], parts[-1]
    packageName, packagePath = rootName, generation
    package = makePackage(packageName, packagePath)
    for directory in directories:
        packagePath = os.path.join(packagePath, directory)
        child = makePackage(f"{packageName}.{directory}", packagePath)
        setattr(package, directory, child)
        packageName, package = f"{packageName}.{directory}", child
    stem = filename.removesuffix(".py")
    fullname = f"{packageName}.{stem}"
    # A package initializer that imported the entry has run it already: it runs once, as in a plain import.
    module = sys.modules.get(fullname)
    if module is None:
        spec = importlib.util.spec_from_file_location(fullname, os.path.join(packagePath, filename))
        if spec is None or spec.loader is None:
            raise ImportError(f"Facet bundle entry {pluginId}/{entryName} is not a Python module")
        module = importlib.util.module_from_spec(spec)
        sys.modules[fullname] = module
        spec.loader.exec_module(module)
    return facetsFromModule(module, pluginId, entryName)


def makePackage(name: str, path: str) -> Any:
    """A package that stands for a directory of the extracted archive; its ``__init__.py`` runs when it has one. A
    package an earlier initializer already imported is that one."""
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    initializer = os.path.join(path, "__init__.py")
    if os.path.isfile(initializer):
        spec = importlib.util.spec_from_file_location(name, initializer, submodule_search_locations=[path])
        assert spec is not None
    else:
        spec = importlib.machinery.ModuleSpec(name, None, is_package=True)
        spec.submodule_search_locations = [path]
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    if spec.loader is not None:
        spec.loader.exec_module(module)
    return module


def unloadGeneration(rootName: str, directory: str | None = None) -> None:
    for name in [name for name in sys.modules if name == rootName or name.startswith(f"{rootName}.")]:
        del sys.modules[name]
    if directory is not None:
        # Importing through the generation's package paths cached a finder for each of its directories.
        for key in [key for key in sys.path_importer_cache if key == directory or key.startswith(directory + os.sep)]:
            del sys.path_importer_cache[key]


def resolveInside(directory: str, member: str) -> str:
    target = os.path.abspath(os.path.join(directory, member))
    if os.path.commonpath([directory, target]) != os.path.abspath(directory):
        raise ValueError(f"Facet bundle archive member {member} escapes the bundle")
    return target


def toFacetCandidates(exported: Any) -> list[Any]:
    return list(exported) if isinstance(exported, list | tuple) else [exported]


def facetsFromModule(imported: Any, pluginId: str, entryName: str) -> tuple[Any, ...]:
    exported = getattr(imported, "default", None)
    candidates = toFacetCandidates(exported)
    if len(candidates) == 0:
        raise RuntimeError(f"Facet bundle entry {pluginId}/{entryName} exported no facets")
    facets: list[Any] = []
    for candidate in candidates:
        facetId = getattr(candidate, "id", None)
        if not isinstance(facetId, str) or len(facetId) == 0:
            raise RuntimeError(f"Facet bundle entry {pluginId}/{entryName} has a facet with an invalid ID")
        if not callable(getattr(candidate, "setup", None)):
            raise RuntimeError(f"Facet bundle entry {pluginId}/{entryName} facet {facetId} has no setup function")  # noqa: TRY004
        facets.append(candidate)
    ids = [facet.id for facet in facets]
    if len(set(ids)) != len(ids):
        raise RuntimeError(f"Facet bundle entry {pluginId}/{entryName} exports duplicate facet IDs")
    return tuple(facets)


def resolveBundleFile(manifestPath: str, file: str, label: str) -> str:
    if len(file) == 0 or os.path.isabs(file) or os.path.basename(file) != file or file in (".", ".."):
        raise RuntimeError(f"Facet bundle {label} must be a filename relative to its manifest")
    return os.path.abspath(os.path.join(os.path.dirname(manifestPath), file))


def verifyContents(contents: bytes, entry: dict[str, Any]) -> None:
    expected = parseIntegrity(entry["integrity"])
    actual = base64.b64encode(hashlib.sha256(contents).digest()).decode("ascii")
    if actual != expected:
        raise RuntimeError(f"Facet bundle integrity check failed for {entry['file']}")


def parseIntegrity(integrity: str) -> str:
    prefix = "sha256-"
    if not integrity.startswith(prefix) or len(integrity) == len(prefix):
        raise RuntimeError("Facet bundle entry has an invalid SHA-256 integrity value")
    return integrity[len(prefix) :]


def verifySource(source: str, entry: dict[str, Any]) -> None:
    try:
        contents = base64.b64decode(source, validate=True)
    except ValueError as error:
        raise RuntimeError(f"Facet bundle artifact source for {entry['file']} is not base64") from error
    verifyContents(contents, entry)


def validateArtifact(value: Any) -> dict[str, Any]:
    if (
        not isinstance(value, dict)
        or value.get("format") != FACET_BUNDLE_ARTIFACT_FORMAT
        or value.get("formatVersion") != FACET_BUNDLE_ARTIFACT_FORMAT_VERSION
        or not isinstance(value.get("entryName"), str)
        or len(value["entryName"]) == 0
        or not isinstance(value.get("source"), str)
    ):
        raise RuntimeError("Invalid facet bundle artifact")
    manifest = validateManifest(
        {
            "format": FACET_BUNDLE_FORMAT,
            "formatVersion": FACET_BUNDLE_FORMAT_VERSION,
            "plugin": value.get("plugin"),
            "entries": {value["entryName"]: value.get("entry")},
        },
        "facet bundle artifact",
    )
    entry = manifest["entries"][value["entryName"]]
    verifySource(value["source"], entry)
    return {
        "format": FACET_BUNDLE_ARTIFACT_FORMAT,
        "formatVersion": FACET_BUNDLE_ARTIFACT_FORMAT_VERSION,
        "plugin": manifest["plugin"],
        "entryName": value["entryName"],
        "entry": entry,
        "source": value["source"],
    }


def materializeArtifact(directory: str, artifact: dict[str, Any]) -> None:
    manifest = {
        "format": FACET_BUNDLE_FORMAT,
        "formatVersion": FACET_BUNDLE_FORMAT_VERSION,
        "plugin": artifact["plugin"],
        "entries": {artifact["entryName"]: artifact["entry"]},
    }
    entryPath = os.path.join(directory, artifact["entry"]["file"])
    with open(entryPath, "wb") as file:
        file.write(base64.b64decode(artifact["source"]))
    with open(os.path.join(directory, FACET_BUNDLE_MANIFEST_FILE), "w", encoding="utf-8", newline="") as file:
        file.write(f"{json.dumps(manifest, indent=2, ensure_ascii=False)}\n")


def validateModulePath(modulePath: Any, path: str, name: str) -> None:
    parts = modulePath.split("/") if isinstance(modulePath, str) else []
    if (
        not isinstance(modulePath, str)
        or len(modulePath) == 0
        or "\\" in modulePath
        or modulePath.startswith("/")
        or any(part in ("", ".", "..") or ":" in part for part in parts)
        or not modulePath.endswith(".py")
    ):
        raise RuntimeError(f"Facet bundle entry {name} has an invalid module path in {path}")


def validateManifest(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("format") != FACET_BUNDLE_FORMAT:
        raise RuntimeError(f"Invalid facet bundle manifest format in {path}")
    if value.get("formatVersion") != FACET_BUNDLE_FORMAT_VERSION:
        raise RuntimeError(f"Unsupported facet bundle manifest version in {path}: {jsString(value.get('formatVersion', UNDEFINED))}")
    plugin = value.get("plugin")
    if not isinstance(plugin, dict) or not isinstance(plugin.get("id"), str) or len(plugin["id"]) == 0:
        raise RuntimeError(f"Facet bundle manifest has an invalid plugin identity in {path}")
    version = plugin.get("version", UNDEFINED)
    if version is not UNDEFINED and (not isinstance(version, str) or len(version) == 0):
        raise RuntimeError(f"Facet bundle manifest has an invalid plugin version in {path}")
    rawEntries = value.get("entries")
    if not isinstance(rawEntries, dict) or len(rawEntries) == 0:
        raise RuntimeError(f"Facet bundle manifest has no entries in {path}")
    entries: dict[str, Any] = {}
    for name, candidate in rawEntries.items():
        if len(name) == 0 or not isinstance(candidate, dict):
            raise RuntimeError(f"Facet bundle manifest has an invalid entry in {path}")
        if not isinstance(candidate.get("file"), str):
            raise RuntimeError(f"Facet bundle entry {name} has no file")  # noqa: TRY004
        resolveBundleFile(path, candidate["file"], f"entry {name}")
        if not isinstance(candidate.get("integrity"), str):
            raise RuntimeError(f"Facet bundle entry {name} has no integrity")  # noqa: TRY004
        parseIntegrity(candidate["integrity"])
        validateModulePath(candidate.get("module"), path, name)
        entries[name] = {"file": candidate["file"], "integrity": candidate["integrity"], "module": candidate["module"]}
    return copy.deepcopy(
        {
            "format": FACET_BUNDLE_FORMAT,
            "formatVersion": FACET_BUNDLE_FORMAT_VERSION,
            "plugin": {"id": plugin["id"], **({} if version is UNDEFINED else {"version": version})},
            "entries": entries,
        }
    )
