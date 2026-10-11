"""Translated from pi's ``coding-agent/src/experimental/plugins/package.ts``.

MISAKA fork: a plugin package is a directory with a ``pyproject.toml`` and the conventional entry modules ``src/session.py``
(the Session worker's facets) and ``src/tui.py`` (the presentation's facets); ``misaka.chord.node`` builds it (see there).
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
from types import SimpleNamespace
from typing import Any

from misaka.chord.node import (
    FACET_BUNDLE_MANIFEST_FILE,
    bundleFacetPackage,
    readFacetBundleArtifact,
)
from misaka.utils.js import jsonParse, jsonStringify, pathResolve

PLUGIN_PACKAGE_PROFILE_VERSION = 1
DEFAULT_PLUGIN_FACETS = {"session": "src/session.py", "tui": "src/tui.py"}


def _readText(path: str) -> str:
    # ``readFile(path, "utf8")`` replaces invalid bytes instead of failing.
    with open(path, encoding="utf-8", errors="replace", newline="") as file:
        return file.read()


def _writeProfile(path: str, text: str) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with open(descriptor, "w", encoding="utf-8", newline="") as file:
        file.write(text)


def _remove(path: str) -> None:
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


async def restoreServerPluginPackageProfile(
    directory: str, serverId: str, configuredPackagePaths: list[str] | tuple[str, ...] | None = None
) -> tuple[str, ...]:
    """Persist an explicit plugin package selection or restore it for a later server generation."""
    path = os.path.join(directory, f"plugin-packages-{serverId}.json")
    if configuredPackagePaths is None:
        return (await readPluginPackageProfile(path, False)) or ()
    packagePaths = normalizePluginPackagePaths(configuredPackagePaths)
    if len(packagePaths) == 0:
        await asyncio.to_thread(_remove, path)
    else:
        await writePluginPackageProfile(path, packagePaths)
    return packagePaths


def readSessionPluginPackageProfile(directory: str, serverId: str, sessionPath: str) -> Any:
    """Read the explicit plugin selection stored for one durable Session."""
    return readPluginPackageProfile(sessionPluginProfilePath(directory, serverId, sessionPath), True, sessionPath)


async def removeSessionPluginPackageProfile(directory: str, serverId: str, sessionPath: str) -> None:
    """Remove the plugin selection stored for one deleted Session."""
    await asyncio.to_thread(_remove, sessionPluginProfilePath(directory, serverId, sessionPath))


def writeSessionPluginPackageProfile(
    directory: str, serverId: str, sessionPath: str, packagePaths: list[str] | tuple[str, ...]
) -> Any:
    """Persist the plugin selection for one durable Session."""
    return writePluginPackageProfile(
        sessionPluginProfilePath(directory, serverId, sessionPath), normalizePluginPackagePaths(packagePaths), sessionPath
    )


def createServerPluginPackage(directory: str, serverId: str, packagePath: str) -> SimpleNamespace:
    """Create a serialized server-owned builder for one configured plugin package. Returns ``SimpleNamespace(manifestPath,
    build)``; ``build()`` resolves to the presentation (``tui``) artifacts, a list of zero or one."""
    normalizedPackagePath = pathResolve(packagePath)
    outdir = os.path.join(directory, "plugin-builds", serverId, pluginBuildDirectoryName(normalizedPackagePath))
    manifestPath = os.path.join(outdir, FACET_BUNDLE_MANIFEST_FILE)
    # ``Promise.resolve()``: no build has run yet.
    buildTail: asyncio.Future[Any] | None = None

    def build() -> asyncio.Future[Any]:
        nonlocal buildTail
        previous = buildTail

        async def run() -> list[dict[str, Any]]:
            if previous is not None:
                await previous
            result = await bundleFacetPackage(
                {"packagePath": normalizedPackagePath, "outdir": outdir, "defaultFacets": DEFAULT_PLUGIN_FACETS}
            )
            if "tui" not in result["manifest"]["entries"]:
                return []
            return [await readFacetBundleArtifact({"manifestPath": result["manifestPath"], "entry": "tui"})]

        operation = asyncio.ensure_future(run())

        async def settle() -> None:
            try:
                await operation
            except Exception:  # noqa: BLE001, S110 - the tail only marks that the build settled
                pass

        buildTail = asyncio.ensure_future(settle())
        return operation

    return SimpleNamespace(manifestPath=manifestPath, build=build)


async def readPluginPackageProfile(path: str, allowEmpty: bool, sessionPath: str | None = None) -> tuple[str, ...] | None:
    try:
        parsed = jsonParse(await asyncio.to_thread(_readText, path))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as error:
        raise RuntimeError(f"Could not read experimental plugin package profile {path}") from error
    if (
        not isinstance(parsed, dict)
        or any(key not in ("version", "packagePaths") and not (key == "sessionPath" and sessionPath is not None) for key in parsed)
        or "version" not in parsed
        or isinstance(parsed["version"], bool)
        or parsed["version"] != PLUGIN_PACKAGE_PROFILE_VERSION
        or ("sessionPath" in parsed if sessionPath is None else ("sessionPath" not in parsed or parsed["sessionPath"] != sessionPath))
        or "packagePaths" not in parsed
        or not isinstance(parsed["packagePaths"], list)
        or (not allowEmpty and len(parsed["packagePaths"]) == 0)
        or any(not isinstance(packagePath, str) or len(packagePath) == 0 for packagePath in parsed["packagePaths"])
    ):
        raise RuntimeError(f"Invalid experimental plugin package profile {path}")
    return normalizePluginPackagePaths(parsed["packagePaths"])


async def writePluginPackageProfile(path: str, packagePaths: tuple[str, ...], sessionPath: str | None = None) -> None:
    profile: dict[str, Any] = {"version": PLUGIN_PACKAGE_PROFILE_VERSION}
    if sessionPath is not None:
        profile["sessionPath"] = sessionPath
    profile["packagePaths"] = list(packagePaths)
    await asyncio.to_thread(_writeProfile, path, f"{jsonStringify(profile, 2)}\n")


def normalizePluginPackagePaths(packagePaths: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    normalized: list[str] = []
    for packagePath in packagePaths:
        if len(packagePath) == 0:
            raise RuntimeError("Plugin package path must not be empty")
        normalized.append(pathResolve(packagePath))
    if len(set(normalized)) != len(normalized):
        raise RuntimeError("Plugin package paths must be unique")
    return tuple(normalized)


def sessionPluginProfilePath(directory: str, serverId: str, sessionPath: str) -> str:
    digest = hashlib.sha256(sessionPath.encode("utf-8")).hexdigest()[:24]
    return os.path.join(directory, f"session-plugin-packages-{serverId}-{digest}.json")


def pluginBuildDirectoryName(packagePath: str) -> str:
    packageDirectory = os.path.dirname(packagePath) if os.path.basename(packagePath) == "pyproject.toml" else packagePath
    base = re.sub(r"[^a-zA-Z0-9._-]", "-", os.path.basename(packageDirectory)) or "plugin"
    label = base if base.endswith("-plugin") else f"{base}-plugin"
    digest = hashlib.sha256(packagePath.encode("utf-8")).hexdigest()[:12]
    return f"{label}-{digest}"
