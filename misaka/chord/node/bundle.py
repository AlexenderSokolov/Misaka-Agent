"""Builds facet bundles (pi chord's ``src/node/bundle.ts``).

MISAKA fork: see ``manifest.py``. A facet entry is a Python module. Its bundle is a deterministic zip archive of that module
and of every file it imports relatively (``from . import x``, ``from .x import y``, up to the working directory), plus the
``__init__.py`` of each package directory on the way. Imports of anything else, such as ``misaka``, the standard library or
an installed package, are left to the loading application, as esbuild's ``external`` imports are upstream. Dynamic imports
are not followed.

``bundleFacets(options)`` takes ``{"plugin": {"id", "version"?}, "entries": {name: module path}, "outdir",
"workingDirectory"?}`` and returns ``{"manifest", "manifestPath"}``.
"""

from __future__ import annotations

import ast
import asyncio
import base64
import hashlib
import io
import json
import os
import shutil
import uuid
import zipfile
from typing import Any

from misaka.chord.node.manifest import (
    FACET_BUNDLE_FORMAT,
    FACET_BUNDLE_FORMAT_VERSION,
    FACET_BUNDLE_MANIFEST_FILE,
)
from misaka.utils.collation import localeSortKey


async def bundleFacets(options: dict[str, Any]) -> dict[str, Any]:
    """Bundle each opaque facet entry into an independent content-addressed archive."""
    return await asyncio.to_thread(_bundleFacets, options)


def _bundleFacets(options: dict[str, Any]) -> dict[str, Any]:
    validateOptions(options)
    workingDirectory = os.path.abspath(options.get("workingDirectory") or os.getcwd())
    outputDirectory = os.path.abspath(os.path.join(workingDirectory, options["outdir"]))
    outputParent = os.path.dirname(outputDirectory)
    os.makedirs(outputParent, exist_ok=True)
    temporaryDirectory = os.path.join(outputParent, f".{os.path.basename(outputDirectory)}.tmp-{uuid.uuid4()}")
    os.mkdir(temporaryDirectory)
    try:
        entries: dict[str, Any] = {}
        for entryName, source in sorted(options["entries"].items(), key=lambda item: localeSortKey(item[0])):
            entries[entryName] = bundleEntry(
                entryName, os.path.abspath(os.path.join(workingDirectory, source)), temporaryDirectory, workingDirectory
            )
        plugin: dict[str, Any] = {"id": options["plugin"]["id"]}
        if options["plugin"].get("version") is not None:
            plugin["version"] = options["plugin"]["version"]
        manifest = {
            "format": FACET_BUNDLE_FORMAT,
            "formatVersion": FACET_BUNDLE_FORMAT_VERSION,
            "plugin": plugin,
            "entries": entries,
        }
        with open(os.path.join(temporaryDirectory, FACET_BUNDLE_MANIFEST_FILE), "w", encoding="utf-8", newline="") as file:
            file.write(f"{json.dumps(manifest, indent=2, ensure_ascii=False)}\n")
        replaceDirectory(temporaryDirectory, outputDirectory)
        return {"manifest": manifest, "manifestPath": os.path.join(outputDirectory, FACET_BUNDLE_MANIFEST_FILE)}
    except BaseException:
        shutil.rmtree(temporaryDirectory, ignore_errors=True)
        raise


def bundleEntry(entryName: str, source: str, temporaryDirectory: str, workingDirectory: str) -> dict[str, Any]:
    entryPrefix = f"facet-{shortHash(entryName)}"
    try:
        files = importClosure(source, workingDirectory)
    except (SyntaxError, OSError, ValueError) as error:
        raise RuntimeError(f"Could not bundle facet entry {entryName}\n{error}") from error
    contents = archive(workingDirectory, files)
    digest = hashlib.sha256(contents)
    file = f"{entryPrefix}-{digest.hexdigest()[:16].upper()}.zip"
    target = os.path.join(temporaryDirectory, file)
    with open(target, "wb") as output:
        output.write(contents)
    return {
        "file": file,
        "integrity": f"sha256-{base64.b64encode(digest.digest()).decode('ascii')}",
        "module": relativePath(workingDirectory, source),
    }


def relativePath(root: str, path: str) -> str:
    return os.path.relpath(path, root).replace(os.sep, "/")


def importClosure(entry: str, root: str) -> list[str]:
    """``entry`` and every file it imports relatively, absolute paths in a stable order."""
    try:
        entryRelative = os.path.relpath(entry, root)
    except ValueError:
        # A path on another drive has no relative form.
        raise ValueError(f"{entry} is outside the working directory") from None
    if entryRelative == ".." or entryRelative.startswith(f"..{os.sep}") or os.path.isabs(entryRelative):
        raise ValueError(f"{entry} is outside the working directory")
    if not os.path.isfile(entry):
        raise FileNotFoundError(f"Facet entry source is not a file: {entry}")
    included: dict[str, None] = {}
    pending = [entry]
    while pending:
        path = pending.pop()
        if path in included:
            continue
        included[path] = None
        # The ``__init__.py`` of every package directory on the way runs when its package is imported.
        directory = os.path.dirname(path)
        while True:
            initializer = os.path.join(directory, "__init__.py")
            if os.path.isfile(initializer):
                pending.append(initializer)
            if os.path.normcase(directory) == os.path.normcase(root) or os.path.dirname(directory) == directory:
                break
            directory = os.path.dirname(directory)
        with open(path, "rb") as file:
            tree = ast.parse(file.read(), filename=path)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level > 0:
                pending.extend(relativeImportTargets(path, node, root))
    return sorted(included, key=lambda candidate: relativePath(root, candidate))


def relativeImportTargets(path: str, node: ast.ImportFrom, root: str) -> list[str]:
    base = os.path.dirname(path)
    for _ in range(node.level - 1):
        base = os.path.dirname(base)
    base = os.path.join(base, *(node.module.split(".") if node.module else []))
    candidates = [f"{base}.py", os.path.join(base, "__init__.py")] if node.module else []
    if os.path.isdir(base):
        for alias in node.names:
            if alias.name != "*":
                candidates.extend([os.path.join(base, f"{alias.name}.py"), os.path.join(base, alias.name, "__init__.py")])
    found: list[str] = []
    for candidate in candidates:
        relative = os.path.relpath(candidate, root)
        if relative == ".." or relative.startswith(f"..{os.sep}"):
            continue
        if os.path.isfile(candidate):
            found.append(os.path.abspath(candidate))
    return found


def archive(root: str, files: list[str]) -> bytes:
    """A zip archive of ``files`` that depends on their contents and relative paths only."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as output:
        for path in files:
            info = zipfile.ZipInfo(relativePath(root, path), date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            with open(path, "rb") as file:
                output.writestr(info, file.read(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    return buffer.getvalue()


def validateOptions(options: dict[str, Any]) -> None:
    if len(options["plugin"]["id"]) == 0:
        raise TypeError("Facet bundle plugin ID must not be empty")
    version = options["plugin"].get("version")
    if version is not None and len(version) == 0:
        raise TypeError("Facet bundle plugin version must not be empty")
    entries = list(options["entries"].items())
    if len(entries) == 0:
        raise TypeError("Facet bundle must contain at least one entry")
    for name, source in entries:
        if len(name) == 0:
            raise TypeError("Facet bundle entry name must not be empty")
        if len(source) == 0:
            raise TypeError(f"Facet bundle entry {name} must have a source path")


def replaceDirectory(temporaryDirectory: str, outputDirectory: str) -> None:
    backupDirectory = f"{outputDirectory}.old-{uuid.uuid4()}"
    movedExisting = False
    try:
        os.rename(outputDirectory, backupDirectory)
        movedExisting = True
    except FileNotFoundError:
        pass
    try:
        os.rename(temporaryDirectory, outputDirectory)
    except BaseException:
        if movedExisting:
            os.rename(backupDirectory, outputDirectory)
        raise
    if movedExisting:
        shutil.rmtree(backupDirectory, ignore_errors=True)


def shortHash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
