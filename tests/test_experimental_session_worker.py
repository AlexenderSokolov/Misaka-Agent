"""pi coding-agent's ``test/experimental-session-worker-lifecycle.test.ts``."""

from __future__ import annotations

from typing import Any

import pytest

from misaka.experimental.session_worker import WorkerLifecycle
from tests.experimental_support import FakeTimers

GENERATION = "generation-1"


def createLifecycle(initialDemandGraceMs: int = 100, orphanDemandGraceMs: int = 200) -> tuple[WorkerLifecycle, list[int], FakeTimers]:
    retired: list[int] = []
    timers = FakeTimers()
    lifecycle = WorkerLifecycle(
        {
            "initialServerConnectionId": GENERATION,
            "initialDemandGraceMs": initialDemandGraceMs,
            "orphanDemandGraceMs": orphanDemandGraceMs,
            "onRetire": lambda: retired.append(1),
            "timers": timers,
        }
    )
    return lifecycle, retired, timers


def test_retires_only_after_client_demand_and_harness_activity_are_both_gone():
    lifecycle, retired, _timers = createLifecycle()
    lifecycle.setDemand(GENERATION, "attachment-1", True)
    lifecycle.setHarnessActive(True)
    lifecycle.setDemand(GENERATION, "attachment-1", False)
    assert retired == []

    lifecycle.setHarnessActive(False)
    assert len(retired) == 1
    lifecycle.close()


def test_retains_the_worker_until_every_presentation_attachment_is_released():
    lifecycle, retired, _timers = createLifecycle()
    lifecycle.setDemand(GENERATION, "attachment-1", True)
    lifecycle.setDemand(GENERATION, "attachment-2", True)
    lifecycle.setDemand(GENERATION, "attachment-1", False)
    assert retired == []
    lifecycle.setDemand(GENERATION, "attachment-2", False)
    assert len(retired) == 1
    lifecycle.close()


def test_does_not_retire_on_repeated_activity_updates_while_the_harness_stays_active():
    lifecycle, retired, _timers = createLifecycle()
    lifecycle.setDemand(GENERATION, "attachment-1", True)
    lifecycle.setHarnessActive(True)
    lifecycle.setDemand(GENERATION, "attachment-1", False)
    lifecycle.setHarnessActive(True)
    assert retired == []
    lifecycle.setHarnessActive(False)
    assert len(retired) == 1
    lifecycle.close()


def test_an_idle_harness_update_does_not_retire_while_demand_remains():
    lifecycle, retired, _timers = createLifecycle()
    lifecycle.setDemand(GENERATION, "attachment-1", True)
    lifecycle.setHarnessActive(False)
    assert retired == []
    lifecycle.close()


def test_does_not_retire_while_a_demand_acknowledgement_holds_reconciliation():
    lifecycle, retired, _timers = createLifecycle()
    lifecycle.setDemand(GENERATION, "attachment-1", True)
    release = lifecycle.holdRetirement()
    lifecycle.setDemand(GENERATION, "attachment-1", False)
    assert retired == []
    release()
    assert len(retired) == 1
    lifecycle.close()


def test_holds_retirement_only_for_requests_from_the_active_attachment():
    lifecycle, retired, _timers = createLifecycle()
    lifecycle.setDemand(GENERATION, "attachment-1", True)
    release = lifecycle.beginRequest(GENERATION, "attachment-1")
    lifecycle.setDemand(GENERATION, "attachment-1", False)
    assert retired == []
    release()
    assert len(retired) == 1
    with pytest.raises(Exception, match="retiring"):
        lifecycle.beginRequest(GENERATION, "attachment-1")
    lifecycle.close()


def test_releasing_a_hold_twice_releases_it_once():
    lifecycle, retired, _timers = createLifecycle()
    lifecycle.setDemand(GENERATION, "attachment-1", True)
    first = lifecycle.holdRetirement()
    second = lifecycle.holdRetirement()
    lifecycle.setDemand(GENERATION, "attachment-1", False)
    first()
    first()
    assert retired == []
    second()
    assert len(retired) == 1
    lifecycle.close()


def test_rejects_requests_from_stale_generations_and_attachments():
    lifecycle, _retired, _timers = createLifecycle()
    lifecycle.setDemand(GENERATION, "attachment-1", True)
    with pytest.raises(Exception, match="stale server generation"):
        lifecycle.beginRequest("stale", "attachment-1")
    with pytest.raises(Exception, match="active attachment"):
        lifecycle.beginRequest(GENERATION, "wrong-attachment")
    lifecycle.close()


def test_retains_disconnected_generation_demand_for_the_orphan_grace():
    lifecycle, retired, timers = createLifecycle()
    lifecycle.setDemand(GENERATION, "attachment-1", True)
    lifecycle.serverDisconnected(GENERATION)

    timers.advance(199)
    assert retired == []
    timers.advance(1)
    assert len(retired) == 1
    lifecycle.close()


def test_allows_a_replacement_generation_to_retain_the_worker():
    lifecycle, retired, timers = createLifecycle()
    lifecycle.setDemand(GENERATION, "attachment-1", True)
    lifecycle.serverDisconnected(GENERATION)
    lifecycle.serverConnected("generation-2")
    lifecycle.setDemand("generation-2", "attachment-2", True)

    timers.advance(200)
    assert retired == []
    lifecycle.setDemand("generation-2", "attachment-2", False)
    assert len(retired) == 1
    lifecycle.close()


def test_a_reconnecting_generation_keeps_its_own_demand():
    lifecycle, retired, timers = createLifecycle()
    lifecycle.setDemand(GENERATION, "attachment-1", True)
    lifecycle.serverDisconnected(GENERATION)
    lifecycle.serverConnected(GENERATION)

    timers.advance(1000)
    assert retired == []
    lifecycle.beginRequest(GENERATION, "attachment-1")()
    lifecycle.close()


def test_retires_a_launched_worker_that_never_receives_initial_demand():
    lifecycle, retired, timers = createLifecycle()
    timers.advance(100)
    assert len(retired) == 1
    lifecycle.close()


def test_rejects_demand_after_retirement_has_won_the_race():
    lifecycle, _retired, timers = createLifecycle()
    timers.advance(100)
    with pytest.raises(Exception, match="retiring"):
        lifecycle.setDemand(GENERATION, "attachment-1", True)
    lifecycle.close()


def test_rejects_demand_from_a_stale_server_generation():
    lifecycle, _retired, _timers = createLifecycle()
    with pytest.raises(Exception, match="stale server generation"):
        lifecycle.setDemand("stale", "attachment-1", True)
    lifecycle.close()


def test_closing_cancels_every_timer():
    lifecycle, retired, timers = createLifecycle()
    lifecycle.setDemand(GENERATION, "attachment-1", True)
    lifecycle.serverDisconnected(GENERATION)
    lifecycle.close()
    timers.advance(10_000)
    assert retired == []
    assert timers.pending == 0


def test_the_worker_options_have_a_strict_schema():
    from misaka.experimental import schema
    from misaka.experimental.session_worker import SessionWorkerOptionsSchema

    metadata = {"id": "demo-1", "createdAt": 1, "cwd": "/work", "path": "/sessions/demo-1"}
    valid: dict[str, Any] = {"sessionDir": "/sessions", "metadata": metadata, "pluginManifestPaths": []}
    assert schema.Check(SessionWorkerOptionsSchema, valid)
    assert schema.Check(SessionWorkerOptionsSchema, {**valid, "provider": "anthropic", "model": "claude"})
    assert not schema.Check(SessionWorkerOptionsSchema, {**valid, "extra": 1})
    assert not schema.Check(SessionWorkerOptionsSchema, {**valid, "model": ""})
    assert not schema.Check(SessionWorkerOptionsSchema, {**valid, "metadata": {**metadata, "createdAt": "now"}})
    assert not schema.Check(SessionWorkerOptionsSchema, {"sessionDir": "/sessions", "metadata": metadata})
