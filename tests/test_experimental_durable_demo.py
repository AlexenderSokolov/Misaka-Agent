"""The durable demo (pi's ``coding-agent/src/experimental/durable``). pi has no tests for it; these open a session and drive the
controller without a model, and render views from a real durable conversation in the demo's TUI.
"""

from __future__ import annotations

import asyncio
import os
import re
from typing import Any

import pytest

from misaka.ai.providers.faux import fauxAssistantMessage
from misaka.chord.context import BACKGROUND_CONTEXT
from misaka.experimental.durable.runtime import agentOf, openDurable, titleOf
from misaka.experimental.durable.sessions import selectSession
from misaka.experimental.durable.tui import (
    DurableTui,
    contextTokens,
    describeTask,
    totalUsage,
)
from misaka.ui.tui.interactive.theme.theme import initTheme
from tests.chat_support import waitFor
from tests.experimental_durable_support import openFauxConversation


@pytest.fixture(autouse=True)
def theme():
    initTheme("dark")


def noticesOf(durable: Any) -> list[str]:
    return [notice["message"] for notice in durable.view.current()["notices"]]


async def test_selects_a_new_session_and_then_the_newest_one(tmp_path):
    first = await selectSession(str(tmp_path), False)
    try:
        assert first.created is True
        assert re.fullmatch(r"\d{13}-[0-9a-f-]{36}", first.id)
        assert first.database == os.path.join(first.directory, "session.sqlite")
        assert first.cwd == os.path.realpath(str(tmp_path))
    finally:
        await first.release()

    second = await selectSession(str(tmp_path), False)
    await second.release()
    resumed = await selectSession(str(tmp_path), True)
    try:
        assert resumed.created is False
        assert resumed.id == max(first.id, second.id)
    finally:
        await resumed.release()


async def test_continuing_without_a_session_fails(tmp_path):
    with pytest.raises(RuntimeError, match="No durable session exists for"):
        await selectSession(str(tmp_path), True)


async def test_opens_a_session_without_a_model_and_runs_controller_commands(tmp_path):
    durable = await openDurable({"cwd": str(tmp_path)})
    try:
        view = durable.view.current()
        assert view["session"]["cwd"] == os.path.realpath(str(tmp_path))
        assert [item["label"] for item in view["conversations"]] == ["main"]
        assert "No model configured; select one with /model." in noticesOf(durable)
        # The task panel starts open and /tasks hides it.
        assert view["tasks"] is not None
        await durable.controller.toggleTasks()
        assert durable.view.current()["tasks"] is None
        await durable.controller.toggleTasks()
        assert durable.view.current()["tasks"] is not None

        await durable.controller.cycleThinking()
        await durable.controller.setModel({"provider": "nobody", "modelId": "nothing"})
        await durable.controller.switchConversation(999)
        notices = noticesOf(durable)
        assert "No model selected" in notices
        assert "Unknown model: nobody/nothing" in notices
        assert "Conversation 999 does not exist" in notices

        seen: list[int] = []
        unsubscribe = durable.view.subscribe(lambda: seen.append(1))
        await durable.controller.toggleTasks()
        await waitFor(lambda: len(seen) > 0)
        unsubscribe()
    finally:
        await durable.close()


async def test_continues_the_newest_session_of_a_directory(tmp_path):
    first = await openDurable({"cwd": str(tmp_path)})
    sessionId = first.view.current()["session"]["id"]
    await first.close()

    again = await openDurable({"cwd": str(tmp_path), "continueSession": True})
    try:
        assert again.view.current()["session"]["id"] == sessionId
    finally:
        await again.close()


def test_titles_and_agent_documents():
    assert titleOf(None) == {}
    assert titleOf({"model": [{"role": "assistant", "content": []}]}) == {}
    assert titleOf({"model": [{"role": "user", "content": "  hello\n  world "}]}) == {"title": "hello world"}
    assert titleOf({"model": [{"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}]}) == {
        "title": "a b"
    }
    assert agentOf({"docs": {}}) == {}
    assert agentOf({"docs": {"pi.agent": {"thinkingLevel": "high"}}}) == {"thinkingLevel": "high"}


def test_titles_collapse_ecmascript_whitespace():
    # ``\s`` and ``trim()`` in JavaScript keep U+0085 and U+001C and drop U+FEFF; Python's are the other way round.
    assert titleOf({"model": [{"role": "user", "content": "\ufeffhello\u0085world\u001c"}]}) == {
        "title": "hello\u0085world\u001c"
    }
    assert titleOf({"model": [{"role": "user", "content": "a\u00a0\u2003b\u3000"}]}) == {"title": "a b"}


async def test_selecting_a_session_for_a_missing_directory_fails_before_writing(tmp_path):
    with pytest.raises(FileNotFoundError):
        await selectSession(str(tmp_path / "missing"), False)
    with pytest.raises(FileNotFoundError):
        await openDurable({"cwd": str(tmp_path / "missing")})


async def test_commands_are_queued_when_they_are_called_not_when_they_are_awaited(tmp_path):
    durable = await openDurable({"cwd": str(tmp_path)})
    try:
        first = durable.controller.setModel({"provider": "p", "modelId": "first"})
        second = durable.controller.setModel({"provider": "p", "modelId": "second"})
        await second
        await first
        assert noticesOf(durable)[-2:] == ["Unknown model: p/first", "Unknown model: p/second"]
        # A command nobody awaits still runs.
        durable.controller.toggleTasks()
        await waitFor(lambda: durable.view.current()["tasks"] is None)
    finally:
        await durable.close()


async def test_a_caller_that_stops_waiting_does_not_cancel_the_command_queue(tmp_path):
    durable = await openDurable({"cwd": str(tmp_path)})
    try:
        await durable.controller.toggleTasks()
        assert durable.view.current()["tasks"] is None
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(durable.controller.toggleTasks(), 1e-6)
        await durable.controller.cycleThinking()
        assert "No model selected" in noticesOf(durable)
        waiting = asyncio.ensure_future(durable.close())
        await asyncio.sleep(0)
        waiting.cancel()
        await asyncio.wait({waiting})
        await durable.close()
    finally:
        await durable.close()


def test_sums_usage_and_measures_the_context():
    usage = {"input": 10, "output": 5, "cacheRead": 2, "cacheWrite": 1, "cost": {"total": 0.25}}
    total = totalUsage({"models": {"a": usage, "b": usage}, "tools": {"c": usage}})
    assert (total["input"], total["output"], total["cacheRead"], total["cacheWrite"]) == (30, 15, 6, 3)
    assert total["cost"]["total"] == 0.75

    def answer(id: int, tokens: int, stopReason: str = "stop") -> dict[str, Any]:
        message = {
            "role": "assistant",
            "stopReason": stopReason,
            "usage": {"input": tokens, "output": 1, "cacheRead": 0, "cacheWrite": 0, "totalTokens": 0},
        }
        return {"id": id, "kind": "pi.assistant", "model": [message]}

    assert contextTokens([]) is None
    assert contextTokens([answer(1, 100), answer(2, 200)]) == 201
    assert contextTokens([answer(1, 100), answer(2, 200, "aborted")]) == 101
    # An answer older than the newest compaction measures the old context.
    assert contextTokens([answer(1, 100), {"id": 3, "kind": "pi.compaction"}]) is None


def test_describes_a_task():
    node = {
        "id": 7,
        "kind": "pi.tool",
        "state": {"status": "waiting", "on": [3, 4]},
        "background": True,
        "abortRequested": True,
        "conversations": [2],
    }
    assert describeTask(node) == "pi.tool #7: waiting on 3, 4 [background, aborting] owns conversation 2"
    running = {"id": 1, "kind": "pi.generation", "state": {"status": "running", "phase": "stream"}, "conversations": []}
    assert describeTask(running) == "pi.generation #1: running stream"
    completing = {"id": 2, "kind": "pi.tool", "state": {"status": "completing", "outcome": "done"}, "conversations": []}
    assert describeTask(completing) == "pi.tool #2: completing (done)"


async def test_renders_a_durable_view_in_the_demo_tui():
    durable = await openFauxConversation([fauxAssistantMessage("remote answer")])
    try:
        submission = await durable.conversation.submit({"type": "input", "content": "hello there"}, BACKGROUND_CONTEXT)
        assert (await submission.wait(BACKGROUND_CONTEXT))["status"] == "done"
        state = await durable.conversation.viewState(BACKGROUND_CONTEXT)
        graph = await durable.harness.taskGraph(BACKGROUND_CONTEXT)
        handlers = {name: (lambda *_: None) for name in ("submit", "followUp", "abort", "exit", "selectModel", "cycleThinking")}
        tui = DurableTui(os.getcwd(), handlers)
        tui.ui.stopped = True
        try:
            view = {
                "session": {"id": "s", "directory": os.getcwd(), "cwd": os.getcwd()},
                "conversation": state.value,
                "conversations": [
                    {"id": state.value["conversation"]["id"], "label": "main"},
                    {"id": 99, "label": "subagent 99", "title": "child task"},
                ],
                "models": [{"provider": "faux", "modelId": "faux-1", "name": "Faux", "contextWindow": 128000}],
                "notices": [{"id": 1, "level": "warning", "message": "a notice to read"}],
                "tasks": graph.value,
            }
            tui.apply(view)
            text = "\n".join(tui.ui.render(100))
            assert "hello there" in text
            assert "remote answer" in text
            assert "a notice to read" in text
            assert "Tasks (0 live, /tasks to hide)" in text
            assert "faux/faux-1" in text
            assert "main" in text
            assert "thinking:" in text
            # Hiding the panel and applying the same view again keeps the transcript.
            tui.apply({**view, "tasks": None})
            assert "Tasks (" not in "\n".join(tui.ui.render(100))
            assert "remote answer" in "\n".join(tui.ui.render(100))
        finally:
            tui.stop()
            graph.dispose()
            state.dispose()
    finally:
        await durable.close()

