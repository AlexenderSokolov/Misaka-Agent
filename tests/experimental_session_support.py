"""pi coding-agent's ``test/experimental-session-support.ts``."""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any

from misaka.chord.context import BACKGROUND_CONTEXT
from misaka.config import home
from misaka.durable import ROOT_CONVERSATION_ID, AgentDoc, createSession
from misaka.durable.storage.sqlite.node import openNodeSqliteStorage
from misaka.experimental.session_catalog import createSession as createCatalogSession
from misaka.experimental.session_catalog import readSession, sessionStoragePath


def makeShortDirectory(prefix: str) -> str:
    """A fresh directory with a short path: a Unix socket path is capped at 104 bytes on macOS, and pytest's own ``tmp_path``
    is already most of that. The experimental server puts sockets in the directory it is given."""
    return tempfile.mkdtemp(prefix=prefix, dir="/tmp" if os.path.isdir("/tmp") else None)


async def createExperimentalSessions(
    sessionsRoot: str, ids: list[str], cwd: str | None = None
) -> list[dict[str, Any]]:
    metadata: list[dict[str, Any]] = []
    for id in ids:
        metadata.append(await createCatalogSession(sessionsRoot, {"id": id, "cwd": os.getcwd() if cwd is None else cwd}))
    return metadata


def configureExperimentalWorkerModel() -> None:
    """Store an Anthropic API key in the home's credentials so a Session worker can select ``anthropic/claude-sonnet-4-5``.
    MISAKA fork: the credentials live in the home (``home.path("auth")``) where pi's tests write ``<agent dir>/auth.json``."""
    path = str(home.path("auth"))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with open(descriptor, "w", encoding="utf-8", newline="") as file:
        file.write(json.dumps({"anthropic": {"type": "api_key", "key": "test-key"}}))


async def readExperimentalSessionState(sessionsRoot: str, sessionId: str) -> dict[str, Any]:
    """The root conversation's model, read from the Session's durable storage while no worker owns it: ``{"model": ModelRef | None}``."""
    metadata = await readSession(sessionsRoot, sessionId)
    if metadata is None:
        raise RuntimeError(f"Expected Session {sessionId}")
    session = createSession(await openNodeSqliteStorage(sessionStoragePath(metadata)))
    try:
        agent = await session.snapshot(AgentDoc, ROOT_CONVERSATION_ID, BACKGROUND_CONTEXT)
        return {"model": None if agent is None else agent.get("model")}
    finally:
        await session.close(BACKGROUND_CONTEXT)
