"""Translated from pi's ``coding-agent/src/experimental/durable/sessions.ts``.

A session of the durable demo is one directory holding ``session.sqlite``, locked by the process that opened it.
``selectSession(cwd, continueSession)`` resolves to a ``SimpleNamespace`` of ``id``, ``directory``, ``database``, ``cwd``,
``created`` and ``release()`` (async): a new session for ``cwd``, or its newest one with ``continueSession``.

MISAKA fork: the sessions live in the home (``home.path("experimental_durable_sessions")``) where pi keeps them under
``<agent dir>/experimental/durable-sessions``.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
import time
import uuid
from types import SimpleNamespace

from misaka.config import home
from misaka.utils.js import pathResolve
from misaka.utils.lockfile import lock

SESSION_DIRECTORY = re.compile(r"\d{13}-[0-9a-f-]{36}")


def _newestSession(root: str) -> str | None:
    names = [entry.name for entry in os.scandir(root) if entry.is_dir() and SESSION_DIRECTORY.fullmatch(entry.name)]
    return max(names) if names else None


async def selectSession(cwdInput: str, continueSession: bool) -> SimpleNamespace:
    """A new session for ``cwdInput``, or its newest one with ``continueSession``."""
    # ``realpath`` rejects a directory that does not exist before anything is written.
    cwd = await asyncio.to_thread(os.path.realpath, pathResolve(cwdInput), strict=True)
    root = os.path.join(str(home.path("experimental_durable_sessions")), hashlib.sha256(cwd.encode("utf-8")).hexdigest()[:24])
    await asyncio.to_thread(os.makedirs, root, exist_ok=True)

    created = False
    if continueSession:
        newest = await asyncio.to_thread(_newestSession, root)
        if newest is None:
            raise RuntimeError(f"No durable session exists for {cwd}")
        directory = os.path.join(root, newest)
    else:
        directory = os.path.join(root, f"{str(int(time.time() * 1000)).zfill(13)}-{uuid.uuid4()}")
        await asyncio.to_thread(os.mkdir, directory)
        created = True

    try:
        # A lock left by a crashed process goes stale after 10 s; wait that long before giving up.
        release = await lock(directory, {"realpath": False, "retries": {"retries": 12, "minTimeout": 1000, "maxTimeout": 1000}})
    except Exception as error:
        raise RuntimeError(f"Session is already open in another process: {directory}") from error
    return SimpleNamespace(
        id=os.path.basename(directory),
        directory=directory,
        database=os.path.join(directory, "session.sqlite"),
        cwd=cwd,
        created=created,
        release=release,
    )
