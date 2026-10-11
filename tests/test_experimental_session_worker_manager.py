"""pi coding-agent's ``test/experimental-session-worker-manager.test.ts``."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest

from misaka.chord.context import BACKGROUND_CONTEXT
from misaka.experimental.session_worker_manager import (
    SessionPluginSelectionConflictError,
    SessionWorkerManager,
)
from tests.experimental_support import FakeTimers

metadata = {"id": "session-1", "createdAt": 1, "cwd": os.path.abspath("/tmp"), "path": os.path.abspath("/tmp/session-1")}


class FakeCoordinator:
    controlPath = "/tmp/control.sock"
    serverConnectionId = "server-generation-1"
    wasReplaced = False

    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []
        self._listeners: list[Callable[[dict[str, Any]], None]] = []
        self.onSend: Callable[[str, dict[str, Any]], None] | None = None

    def onEvent(self, listener: Callable[[dict[str, Any]], None]) -> Callable[[], None]:
        self._listeners.append(listener)
        return lambda: self._listeners.remove(listener) if listener in self._listeners else None

    async def send(self, peerId: str, payload: dict[str, Any]) -> None:
        self.sent.append({"peerId": peerId, "payload": payload})
        if self.onSend is not None:
            self.onSend(peerId, payload)

    async def broadcast(self, payload: dict[str, Any]) -> None:
        if payload["type"] != "discover_workers":
            return
        self.emit(
            {
                "type": "message",
                "from": "worker-1",
                "payload": {
                    "type": "worker_ready",
                    "token": "worker-token",
                    "sessionKey": metadata["path"],
                    "sessionId": metadata["id"],
                    "pid": 123,
                    "metadata": metadata,
                    "pluginManifestPaths": [],
                },
            }
        )

    def emit(self, event: dict[str, Any]) -> None:
        for listener in list(self._listeners):
            listener(event)


def later(callback: Callable[[], None]) -> None:
    asyncio.get_running_loop().call_soon(callback)


def respondToDemands(coordinator: FakeCoordinator, answer: Callable[[dict[str, Any]], bool] = lambda payload: True) -> None:
    def onSend(peerId: str, payload: dict[str, Any]) -> None:
        if payload["type"] != "session_demand" or not answer(payload):
            return
        later(
            lambda: coordinator.emit(
                {
                    "type": "message",
                    "from": peerId,
                    "payload": {
                        "type": "demand_applied",
                        "token": "worker-token",
                        "sessionKey": metadata["path"],
                        "requestId": payload["requestId"],
                        "attachmentId": payload["attachmentId"],
                        "attached": payload["attached"],
                    },
                }
            )
        )

    coordinator.onSend = onSend


def newManager(coordinator: FakeCoordinator, timers: FakeTimers, kills: list[tuple[int, str]] | None = None) -> SessionWorkerManager:
    return SessionWorkerManager(
        coordinator, "/tmp", None, None, timers, (lambda pid, name: kills.append((pid, name))) if kills is not None else None
    )


async def createAttachedWorker(timers: FakeTimers, kills: list[tuple[int, str]] | None = None) -> SimpleNamespace:
    coordinator = FakeCoordinator()
    workers = newManager(coordinator, timers, kills)
    await workers.discover({"worker-1"})
    handle = await workers.openSession(metadata, BACKGROUND_CONTEXT, [])
    respondToDemands(coordinator)
    attachment = await handle.attachClient(BACKGROUND_CONTEXT)
    return SimpleNamespace(
        coordinator=coordinator,
        workers=workers,
        handle=handle,
        attachment=attachment,
        release=lambda: attachment.release(BACKGROUND_CONTEXT),
    )


# ─── Session worker lifecycle failures ───────────────────────────────────────


async def test_adopts_a_discovered_worker_with_its_existing_session_plugin_selection():
    coordinator = FakeCoordinator()
    workers = newManager(coordinator, FakeTimers())
    await workers.discover({"worker-1"})
    assert {"peerId": "worker-1", "payload": {"type": "shutdown"}} not in coordinator.sent
    assert len(workers.workerPids) == 1
    workers.assertSessionPluginManifestPaths(metadata, [])
    workers.detach()


async def test_rejects_a_different_plugin_selection_for_an_active_session_without_stopping_it():
    attached = await createAttachedWorker(FakeTimers())
    with pytest.raises(SessionPluginSelectionConflictError, match="active with a different plugin selection"):
        attached.workers.assertSessionPluginManifestPaths(metadata, ["/tmp/plugin/chord-facets.json"])
    assert len(attached.workers.workerPids) == 1
    attached.workers.detach()


async def test_compensates_a_timed_out_attachment_before_rejecting_it():
    timers = FakeTimers()
    coordinator = FakeCoordinator()
    workers = newManager(coordinator, timers)
    await workers.discover({"worker-1"})
    handle = await workers.openSession(metadata, BACKGROUND_CONTEXT, [])
    demands: list[dict[str, Any]] = []
    respondToDemands(coordinator, lambda payload: (demands.append(payload), not payload["attached"])[1])

    attaching = handle.attachClient(BACKGROUND_CONTEXT)
    await timers.advanceAsync(5_000)
    with pytest.raises(Exception, match="timed out"):
        await attaching
    assert len(demands) == 2
    assert demands[0]["attached"] is True
    assert isinstance(demands[0]["attachmentId"], str)
    assert {"attachmentId": demands[1]["attachmentId"], "attached": False} == {
        "attachmentId": demands[0]["attachmentId"],
        "attached": demands[1]["attached"],
    }
    workers.detach()


async def test_kills_a_worker_when_timed_out_demand_cannot_be_reconciled():
    timers = FakeTimers()
    kills: list[tuple[int, str]] = []
    coordinator = FakeCoordinator()
    workers = newManager(coordinator, timers, kills)
    await workers.discover({"worker-1"})
    handle = await workers.openSession(metadata, BACKGROUND_CONTEXT, [])
    coordinator.onSend = None

    attaching = handle.attachClient(BACKGROUND_CONTEXT)
    await timers.advanceAsync(5_000)
    await timers.advanceAsync(5_000)
    await timers.advanceAsync(10_000)
    with pytest.raises(Exception, match="worker was terminated"):
        await attaching
    assert kills == [(123, "SIGKILL")]
    assert len(workers.workerPids) == 0
    workers.detach()


async def test_bounds_harness_driven_worker_shutdown():
    timers = FakeTimers()
    kills: list[tuple[int, str]] = []
    attached = await createAttachedWorker(timers, kills)
    await attached.release()
    attached.coordinator.onSend = None

    closing = attached.handle.close(BACKGROUND_CONTEXT)
    await timers.advanceAsync(10_000)
    await closing
    assert kills == [(123, "SIGKILL")]
    assert len(attached.workers.workerPids) == 0
    attached.workers.detach()


async def test_a_worker_that_exits_on_request_needs_no_kill():
    timers = FakeTimers()
    kills: list[tuple[int, str]] = []
    attached = await createAttachedWorker(timers, kills)
    await attached.release()

    def onSend(peerId: str, payload: dict[str, Any]) -> None:
        if payload["type"] == "shutdown":
            later(lambda: attached.coordinator.emit({"type": "peer_disconnected", "peerId": peerId}))

    attached.coordinator.onSend = onSend
    await attached.handle.close(BACKGROUND_CONTEXT)
    assert kills == []
    assert len(attached.workers.workerPids) == 0
    assert await attached.handle.terminated is None
    attached.workers.detach()


# ─── Session worker operations ───────────────────────────────────────────────

serviceCall = {"serviceId": "test.session", "member": "run", "args": ["Hello"]}


def answerOperations(coordinator: FakeCoordinator, response: Callable[[dict[str, Any]], dict[str, Any]], token: str = "worker-token") -> None:
    def onSend(peerId: str, payload: dict[str, Any]) -> None:
        if payload["type"] != "operation":
            return
        later(
            lambda: coordinator.emit(
                {
                    "type": "message",
                    "from": peerId,
                    "payload": {
                        "type": "operation_response",
                        "token": token,
                        "sessionKey": metadata["path"],
                        "response": response(payload),
                    },
                }
            )
        )

    coordinator.onSend = onSend


async def test_correlates_service_results_to_the_worker_generation_and_attachment():
    attached = await createAttachedWorker(FakeTimers())
    answerOperations(
        attached.coordinator,
        lambda payload: {
            "type": "operation_result",
            "requestId": payload["requestId"],
            "scope": payload["scope"],
            "result": {"accepted": True},
        },
    )
    assert await attached.attachment.invokeService(serviceCall, lambda *_: None, BACKGROUND_CONTEXT) == {"accepted": True}
    operation = next(entry["payload"] for entry in attached.coordinator.sent if entry["payload"]["type"] == "operation")
    assert operation["scope"]["serverConnectionId"] == "server-generation-1"
    assert isinstance(operation["scope"]["attachmentId"], str)
    assert operation["call"] == serviceCall
    attached.workers.detach()
    await attached.release()


async def test_reports_a_worker_error_with_its_code():
    from misaka.server import ServerError

    attached = await createAttachedWorker(FakeTimers())
    answerOperations(
        attached.coordinator,
        lambda payload: {
            "type": "operation_error",
            "requestId": payload["requestId"],
            "scope": payload["scope"],
            "code": "service_not_found",
            "message": "nope",
        },
    )
    with pytest.raises(ServerError) as raised:
        await attached.attachment.invokeService(serviceCall, lambda *_: None, BACKGROUND_CONTEXT)
    assert raised.value.code == "service_not_found"
    answerOperations(
        attached.coordinator,
        lambda payload: {"type": "operation_error", "requestId": payload["requestId"], "scope": payload["scope"], "message": "boom"},
    )
    with pytest.raises(Exception, match="Session worker operation failed: boom"):
        await attached.attachment.invokeService(serviceCall, lambda *_: None, BACKGROUND_CONTEXT)
    attached.workers.detach()
    await attached.release()


async def test_rejects_a_correlated_response_with_mismatched_worker_identity():
    attached = await createAttachedWorker(FakeTimers())
    answerOperations(
        attached.coordinator,
        lambda payload: {
            "type": "operation_result",
            "requestId": payload["requestId"],
            "scope": payload["scope"],
            "result": {"accepted": True},
        },
        token="wrong-token",
    )
    with pytest.raises(Exception, match="mismatched operation response"):
        await attached.attachment.invokeService(serviceCall, lambda *_: None, BACKGROUND_CONTEXT)
    attached.workers.detach()
    await attached.release()


async def test_rejects_a_null_request_scope():
    attached = await createAttachedWorker(FakeTimers())
    answerOperations(
        attached.coordinator,
        lambda payload: {
            "type": "operation_result",
            "requestId": payload["requestId"],
            "scope": None,
            "result": {"accepted": True},
        },
    )
    with pytest.raises(Exception, match="invalid operation response"):
        await attached.attachment.invokeService(serviceCall, lambda *_: None, BACKGROUND_CONTEXT)
    attached.workers.detach()
    await attached.release()


async def test_rejects_pending_service_calls_on_replacement_without_stopping_the_worker():
    attached = await createAttachedWorker(FakeTimers())
    attached.coordinator.onSend = None
    calling = attached.attachment.invokeService(serviceCall, lambda *_: None, BACKGROUND_CONTEXT)
    attached.workers.detach()

    with pytest.raises(Exception, match="replaced during a worker operation"):
        await calling
    assert "shutdown" not in [entry["payload"]["type"] for entry in attached.coordinator.sent]
    assert len(attached.workers.workerPids) == 0


async def test_cancels_a_call_whose_context_aborts():
    from misaka.chord.context import withCancel

    attached = await createAttachedWorker(FakeTimers())
    attached.coordinator.onSend = None
    cancellable = withCancel(BACKGROUND_CONTEXT)
    calling = attached.attachment.invokeService(serviceCall, lambda *_: None, cancellable.context)
    cancellable.cancel()
    with pytest.raises(Exception, match="aborted"):
        await calling
    cancelled = [entry["payload"] for entry in attached.coordinator.sent if entry["payload"]["type"] == "operation_cancel"]
    assert len(cancelled) == 1
    assert cancelled[0]["scope"]["serverConnectionId"] == "server-generation-1"
    attached.workers.detach()
    await attached.release()
