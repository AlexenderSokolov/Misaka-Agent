"""Translated from pi's ``coding-agent/src/experimental/session-catalog.ts``.

A Session is a directory holding ``meta.json`` and the worker-owned ``session.sqlite``. The metadata is the plain dict
``{"id", "createdAt", "cwd", "path"}`` (upstream's ``SessionCatalogMetadata``): the Session's ID, creation time in
milliseconds, the working directory its agent runs in, and the Session directory that workers lock and own.
"""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import time
import uuid
from typing import Any

from misaka.utils.js import jsonParse, jsonStringify

METADATA_FILE = "meta.json"
STORAGE_FILE = "session.sqlite"
SESSION_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


def isSessionId(id: str) -> bool:
    return SESSION_ID.fullmatch(id) is not None


def sessionStoragePath(metadata: dict[str, Any]) -> str:
    """The durable storage file of a Session. Only the Session's worker opens it."""
    return os.path.join(metadata["path"], STORAGE_FILE)


async def listSessions(sessionDir: str) -> list[dict[str, Any]]:
    """Every Session in the directory. Entries without valid metadata are skipped."""
    try:
        names = await asyncio.to_thread(os.listdir, sessionDir)
    except FileNotFoundError:
        return []
    sessions = await asyncio.gather(*(readSession(sessionDir, name) for name in names if isSessionId(name)))
    return [metadata for metadata in sessions if metadata is not None]


def _readMetadataFile(path: str) -> Any:
    # ``readFile(path, "utf8")`` replaces invalid bytes, and ``JSON.parse`` rejects ``NaN`` and ``Infinity``.
    with open(path, "rb") as file:
        return jsonParse(file.read().decode("utf-8", errors="replace"))


async def readSession(sessionDir: str, id: str) -> dict[str, Any] | None:
    """One Session by ID, or ``None`` when it does not exist."""
    if not isSessionId(id):
        return None
    path = os.path.join(sessionDir, id)
    try:
        value = await asyncio.to_thread(_readMetadataFile, os.path.join(path, METADATA_FILE))
    except Exception:  # noqa: BLE001 - any unreadable or malformed entry is skipped
        return None
    if not isinstance(value, dict):
        return None
    createdAt, cwd = value.get("createdAt"), value.get("cwd")
    if not isinstance(createdAt, int | float) or isinstance(createdAt, bool) or not isinstance(cwd, str):
        return None
    return {"id": id, "createdAt": createdAt, "cwd": cwd, "path": path}


def _createSessionDirectory(sessionDir: str, path: str, id: str, metadata: dict[str, Any]) -> None:
    os.makedirs(sessionDir, exist_ok=True)
    try:
        os.mkdir(path)
    except FileExistsError:
        raise RuntimeError(f"Session {id} already exists") from None
    text = jsonStringify({"createdAt": metadata["createdAt"], "cwd": metadata["cwd"]}, "\t")
    with open(os.path.join(path, METADATA_FILE), "w", encoding="utf-8", newline="") as file:
        file.write(f"{text}\n")


async def createSession(sessionDir: str, options: dict[str, Any]) -> dict[str, Any]:
    """Create an empty Session. Its worker creates the storage on first open. ``options`` is ``{"id"?, "cwd"}``."""
    id = options["id"] if options.get("id") is not None else str(uuid.uuid4())
    if not isSessionId(id):
        raise RuntimeError(f"Invalid session ID: {id}")
    path = os.path.join(sessionDir, id)
    metadata = {"id": id, "createdAt": int(time.time() * 1000), "cwd": options["cwd"], "path": path}
    await asyncio.to_thread(_createSessionDirectory, sessionDir, path, id, metadata)
    return metadata


def _removeDirectory(path: str) -> None:
    """``rm(path, {recursive: true, force: true})``: a link (a symlink, or a junction on Windows) is unlinked rather than
    followed, a file is removed, and a missing path is not an error."""
    try:
        if os.path.islink(path) or os.path.isjunction(path):
            os.unlink(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)
    except FileNotFoundError:
        pass


async def deleteSession(metadata: dict[str, Any]) -> None:
    """Delete a Session directory. Its worker must be closed first."""
    await asyncio.to_thread(_removeDirectory, metadata["path"])
