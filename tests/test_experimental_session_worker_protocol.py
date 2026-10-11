"""The Session worker's control protocol, driven in-process against a fake coordinator on a real Unix socket.

The Session is the faux worker harness of ``tests/fixtures/faux_session_worker.py``.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import sys
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

import pytest

from misaka.experimental.session_worker import (
    SESSION_WORKER_CONTROL_ADDRESS_ENV,
    SESSION_WORKER_CONTROL_TOKEN_ENV,
    SESSION_WORKER_PEER_ID_ENV,
    SESSION_WORKER_SESSION_KEY_ENV,
    closeResources,
    connectControl,
    lifecycleDelay,
    runSessionWorkerWithHarness,
)
from tests.experimental_session_support import (
    configureExperimentalWorkerModel,
    createExperimentalSessions,
    makeShortDirectory,
)
from tests.fixtures.faux_session_worker import createHarness
from tests.fixtures.keyed_service import KeyedProbe

posix = pytest.mark.skipif(sys.platform == "win32", reason="Unix sockets")

SERVER = "gen-1"
ATTACHMENT = "a1"


class FakeCoordinator:
    """Accepts one worker connection, answers its registration, and then speaks as the server."""

    def __init__(self, directory: str, answerRegistration: bool = True) -> None:
        self.path = os.path.join(directory, "control.sock")
        self.answerRegistration = answerRegistration
        self.accepted: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self.reader: asyncio.StreamReader | None = None
        self.writer: asyncio.StreamWriter | None = None
        self.server: asyncio.Server | None = None

    async def start(self) -> None:
        self.server = await asyncio.start_unix_server(self._accept, self.path)

    async def _accept(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self.reader, self.writer = reader, writer
        if not self.answerRegistration:
            writer.close()
        else:
            registration = json.loads(await reader.readline())
            assert registration["type"] == "register_peer"
            self.write({"type": "peer_registered", "peerId": registration["peerId"], "serverConnectionId": SERVER})
        self.accepted.set_result(None)

    def write(self, *messages: dict[str, Any]) -> None:
        assert self.writer is not None
        self.writer.write(b"".join((json.dumps(message) + "\n").encode() for message in messages))

    def command(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {"type": "message", "from": "server", "payload": payload}

    async def event(self, kind: str) -> dict[str, Any]:
        """The next worker event of the given type."""
        await asyncio.wait_for(self.accepted, 15)
        assert self.reader is not None
        while True:
            line = await asyncio.wait_for(self.reader.readline(), 15)
            assert line, f"the worker closed before sending {kind}"
            message = json.loads(line)
            payload = message.get("payload")
            if isinstance(payload, dict) and payload.get("type") == kind:
                return payload  # type: ignore[no-any-return]

    async def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
        if self.server is not None:
            self.server.close()


@pytest.fixture
async def worker(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[SimpleNamespace]:
    configureExperimentalWorkerModel()
    directory = makeShortDirectory("psw-")
    sessionsRoot = os.path.join(directory, "sessions")
    [metadata] = await createExperimentalSessions(sessionsRoot, ["demo-1"], directory)
    coordinator = FakeCoordinator(directory)
    await coordinator.start()
    sessionKey = base64.urlsafe_b64encode(metadata["path"].encode()).decode().rstrip("=")
    monkeypatch.setenv(SESSION_WORKER_CONTROL_ADDRESS_ENV, coordinator.path)
    monkeypatch.setenv(SESSION_WORKER_CONTROL_TOKEN_ENV, "token")
    monkeypatch.setenv(SESSION_WORKER_SESSION_KEY_ENV, sessionKey)
    monkeypatch.setenv(SESSION_WORKER_PEER_ID_ENV, "worker-1")
    options = {
        "sessionDir": sessionsRoot,
        "metadata": metadata,
        "provider": "anthropic",
        "model": "claude-sonnet-4-5",
        "pluginManifestPaths": [],
    }
    running = asyncio.ensure_future(runSessionWorkerWithHarness([json.dumps(options)], createHarness))
    yield SimpleNamespace(coordinator=coordinator, running=running, metadata=metadata)
    if not running.done():
        coordinator.write(coordinator.command({"type": "shutdown"}))
        await asyncio.wait_for(running, 15)
    await coordinator.close()


@posix
async def test_a_cancel_queued_behind_an_operation_reaches_it(worker: SimpleNamespace) -> None:
    coordinator: FakeCoordinator = worker.coordinator
    await coordinator.event("worker_ready")
    coordinator.write(
        coordinator.command(
            {"type": "session_demand", "serverConnectionId": SERVER, "requestId": "d1", "attachmentId": ATTACHMENT, "attached": True}
        )
    )
    await coordinator.event("demand_applied")
    scope = {"serverConnectionId": SERVER, "attachmentId": ATTACHMENT}
    call = {"serviceId": KeyedProbe.id, "instance": {"key": "probe", "generation": 1}, "member": "wait", "args": []}
    # One write: the operation and its cancel reach the worker in the same chunk.
    coordinator.write(
        coordinator.command({"type": "operation", "requestId": "r1", "scope": scope, "call": call}),
        coordinator.command({"type": "operation_cancel", "requestId": "r1", "scope": scope}),
    )
    response = (await coordinator.event("operation_response"))["response"]
    assert response["type"] == "operation_error"
    assert response["message"] == "Service operation cancelled"
    coordinator.write(coordinator.command({"type": "shutdown"}))
    # A runtime without ``cleanup`` closes cleanly.
    assert await asyncio.wait_for(worker.running, 15) == 0


@posix
async def test_a_worker_whose_control_connection_closes_before_registration_exits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory = makeShortDirectory("psw-")
    coordinator = FakeCoordinator(directory, answerRegistration=False)
    await coordinator.start()
    monkeypatch.setenv(SESSION_WORKER_CONTROL_ADDRESS_ENV, coordinator.path)
    monkeypatch.setenv(SESSION_WORKER_CONTROL_TOKEN_ENV, "token")
    monkeypatch.setenv(SESSION_WORKER_SESSION_KEY_ENV, "a2V5")
    monkeypatch.setenv(SESSION_WORKER_PEER_ID_ENV, "worker-1")
    try:
        with pytest.raises(SystemExit) as caught:
            await asyncio.wait_for(connectControl(), 10)
        assert caught.value.code == 0
    finally:
        await coordinator.close()


async def test_a_runtime_without_cleanup_closes_cleanly() -> None:
    closed: list[str] = []

    async def close(context: Any) -> None:
        closed.append("harness")

    async def release() -> None:
        closed.append("release")

    runtime = SimpleNamespace(harness=SimpleNamespace(close=close))
    await closeResources({"runtime": runtime, "services": None, "releaseOwnership": release})
    assert closed == ["harness", "release"]


@pytest.mark.parametrize(
    ("value", "expected"),
    [("1000", 1000), ("1e3", 1000), ("", 0), (" ", 0), ("0x10", 16), ("10.0", 10), ("0", 0)],
)
def test_lifecycle_delay_reads_a_number(monkeypatch: pytest.MonkeyPatch, value: str, expected: int) -> None:
    monkeypatch.setenv("DELAY", value)
    assert lifecycleDelay("DELAY", 7) == expected


@pytest.mark.parametrize("value", ["1_000", "٣", "-1", "1.5", "abc", "Infinity", "9007199254740992"])
def test_lifecycle_delay_rejects_what_number_rejects(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("DELAY", value)
    with pytest.raises(RuntimeError, match="must be a non-negative safe integer"):
        lifecycleDelay("DELAY", 7)


def test_lifecycle_delay_falls_back_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DELAY", raising=False)
    assert lifecycleDelay("DELAY", 7) == 7
