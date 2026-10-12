"""Translated from pi's ``coding-agent/src/experimental/durable/runtime.ts``.

``openDurable(options)`` opens the durable demo's session: one process owns the model runtime, the durable Harness and its
SQLite storage. It resolves to ``SimpleNamespace(view, controller, settings, close)``.

Everything the TUI renders is plain data. A ``DurableView`` is the dict ``{"session": {"id", "directory", "cwd"},
"conversation" (a ``ConversationView``), "conversations": [{"id", "label", "title"?}], "models": [{"provider", "modelId",
"name", "contextWindow"}], "notices": [{"id", "level": "info" | "warning" | "error", "message"}], "tasks"?}``; ``tasks``
is the live task graph while the task panel is open. ``view`` provides ``current()`` and ``subscribe(listener)``;
``controller`` provides ``submit(text, whenBusy)``, ``compact(instructions)``, ``abort()``, ``cycleThinking()``,
``setModel(ref)``, ``toggleTasks()`` and ``switchConversation(id)``, all async. ``options`` is ``{"cwd"?, "continueSession"?}``.
"""

from __future__ import annotations

import asyncio
import os
from types import SimpleNamespace
from typing import Any

from misaka.ai.models import clampThinkingLevel, getSupportedThinkingLevels
from misaka.chord.context import BACKGROUND_CONTEXT
from misaka.core.model_runtime import ModelRuntime
from misaka.core.settings_manager import SettingsManager
from misaka.durable import ROOT_CONVERSATION_ID, Harness
from misaka.durable.storage.sqlite.node import openNodeSqliteStorage
from misaka.experimental.durable.harness_setup import (
    ExecutionEnvs,
    configureHarnessHttp,
    createCodingRegistry,
    createHarnessSettings,
    findInitialAgentModel,
)
from misaka.experimental.durable.sessions import selectSession
from misaka.experimental.durable.subagent import Subagent
from misaka.utils.js import (
    eager,
    errorMessage,
    jsCollapseWhitespace,
    jsonStringify,
    spawn,
    startTask,
)

context = BACKGROUND_CONTEXT


def agentOf(view: dict[str, Any]) -> dict[str, Any]:
    """The agent document of a view; absent while the conversation has none."""
    agent = view["docs"].get("pi.agent")
    return {} if agent is None else agent


async def firstInput(harness: Any, id: int) -> dict[str, str]:
    """A subagent's task: the oldest user message of its conversation. The main conversation needs no title."""
    if id == ROOT_CONVERSATION_ID:
        return {}
    conversation = await harness.conversation(id, context)
    first: dict[str, Any] | None = None
    cursor = None
    while True:
        page = await conversation.entries({}, 256, cursor, context)
        # A page is newest first, and a later page is older: the last user entry seen is the oldest.
        first = next((entry for entry in reversed(page["items"]) if entry["kind"] == "pi.user"), first)
        cursor = page.get("next")
        if cursor is None:
            break
    return titleOf(first)


def _resolved() -> asyncio.Future[None]:
    future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
    future.set_result(None)
    return future


def titleOf(entry: dict[str, Any] | None) -> dict[str, str]:
    """The text of a user entry, as a one-line title."""
    model = None if entry is None else entry.get("model")
    message = model[0] if model else None
    if message is None or message.get("role") != "user":
        return {}
    content = message["content"]
    text = content if isinstance(content, str) else " ".join(block["text"] for block in content if block["type"] == "text")
    return {"title": jsCollapseWhitespace(text).strip(" ")}


async def openDurable(options: dict[str, Any] | None = None) -> SimpleNamespace:
    options = options or {}
    cwdOption = options.get("cwd")
    location = await selectSession(os.getcwd() if cwdOption is None else cwdOption, options.get("continueSession") is True)
    envs = ExecutionEnvs(location.cwd)
    harness: Any = None
    background: set[asyncio.Task[Any]] = set()
    try:
        modelRuntime = await ModelRuntime.create()
        settingsManager = SettingsManager.create(location.cwd)
        configureHarnessHttp(settingsManager)
        settings = createHarnessSettings(settingsManager)
        registry = createCodingRegistry(settingsManager, location.cwd)
        registry.install(Subagent)

        pendingReports: list[Any] = []
        reporter: dict[str, Any] = {"report": pendingReports.append}
        harness = await Harness.open(
            await openNodeSqliteStorage(location.database),
            {
                "models": modelRuntime,
                "registry": registry,
                "settings": settings,
                "env": envs.env,
                "onReport": lambda error: reporter["report"](error),
            },
            context,
        )
        initial = await findInitialAgentModel(settingsManager, modelRuntime) if location.created else None
        agent: dict[str, Any] = {"cwd": location.cwd}
        if initial is not None and initial.get("model") is not None:
            agent["model"] = initial["model"]
        if initial is not None and initial.get("thinkingLevel") is not None:
            agent["thinkingLevel"] = initial["thinkingLevel"]
        root = await harness.root(context, {"agent": agent})

        def label(id: int) -> str:
            return "main" if id == root.id else f"subagent {id}"

        opened = harness
        summaries: list[dict[str, Any]] = []
        cursor = None
        while True:
            page = await opened.commit(lambda tx, cursor=cursor: tx.scanConversations({}, 256, cursor), context)
            for item in page["items"]:
                summaries.append({"id": item["id"], "label": label(item["id"]), **(await firstInput(opened, item["id"]))})
            cursor = page.get("next")
            if cursor is None:
                break
        current = root
        conversation = await root.viewState(context)

        def models() -> list[dict[str, Any]]:
            return [
                {"provider": model.provider, "modelId": model.id, "name": model.name, "contextWindow": model.contextWindow}
                for model in modelRuntime.getAvailableSnapshot()
            ]

        state: dict[str, Any] = {
            "session": {"id": location.id, "directory": location.directory, "cwd": location.cwd},
            "conversation": conversation.value,
            "conversations": summaries,
            "models": models(),
            "notices": [],
        }
        listeners: list[Any] = []
        notifying = False

        def notifyListeners() -> None:
            nonlocal notifying
            notifying = False
            for listener in listeners:
                listener()

        # Commit listeners and Chord frames call this on the Session line; rendering runs afterwards, once per burst.
        def update(patch: dict[str, Any]) -> None:
            nonlocal state, notifying
            state = {**state, **patch}
            if notifying:
                return
            notifying = True
            asyncio.get_running_loop().call_soon(notifyListeners)

        nextNotice = 1

        def notice(level: str, message: str) -> None:
            nonlocal nextNotice
            update({"notices": [*state["notices"], {"id": nextNotice, "level": level, "message": message}][-20:]})
            nextNotice += 1

        def fail(error: BaseException) -> None:
            notice("error", errorMessage(error))

        reporter["report"] = lambda error: notice("warning", errorMessage(error))
        for error in pendingReports:
            reporter["report"](error)
        unsubscribe = conversation.subscribe(lambda value, *_: update({"conversation": value}))

        # Subagents appear as their conversations are created. A commit listener only records; it calls no Session API.
        def onCommit(publication: dict[str, Any], *_: Any) -> None:
            conversations = state["conversations"]
            changed = False
            for change in publication["changes"]:
                if change["type"] == "conversation":
                    conversations = [*conversations, {"id": change["value"]["id"], "label": label(change["value"]["id"])}]
                    changed = True
                elif change["type"] == "entry" and change["value"]["kind"] == "pi.user":
                    id = change["value"]["conversationId"]
                    conversations = [
                        {**summary, **titleOf(change["value"])} if summary["id"] == id and "title" not in summary else summary
                        for summary in conversations
                    ]
                    changed = True
            if changed:
                update({"conversations": conversations})

        unsubscribeCommits = harness.subscribeCommits(onCommit)

        tasks: Any = None
        unsubscribeTasks: Any = lambda: None

        def closeTasks() -> None:
            nonlocal tasks, unsubscribeTasks
            unsubscribeTasks()
            if tasks is not None:
                tasks.dispose()
            tasks = None

        queue: asyncio.Future[None] = _resolved()

        # One at a time, so toggles, switches, and key presses apply in order.
        def command(operation: Any) -> asyncio.Future[None]:
            nonlocal queue
            previous = queue

            async def run() -> None:
                await previous
                try:
                    await operation()
                except Exception as error:  # noqa: BLE001 - shown as a notice
                    fail(error)

            queue = startTask(run())
            # Shielded: a caller that stops waiting must not cancel the queue every later command waits on.
            return asyncio.shield(queue)

        def watchAnswer(submission: Any) -> None:
            async def watch() -> None:
                try:
                    settled = await submission.wait(context)
                except Exception as error:  # noqa: BLE001 - shown as a notice
                    fail(error)
                    return
                if settled["status"] == "unanswered" and settled.get("reason") != "aborted":
                    detail = settled.get("detail")
                    notice("error", f"No answer: {settled['reason']}{'' if detail is None else f' {jsonStringify(detail)}'}")

            spawn(watch(), background)

        def agentModel() -> Any:
            ref = agentOf(state["conversation"]).get("model")
            model = None if ref is None else modelRuntime.getModel(ref["provider"], ref["modelId"])
            if model is None:
                raise RuntimeError("No model selected" if ref is None else "Current model is unavailable")
            return model

        # The members below are plain functions: a command is queued when it is called, not when it is first awaited.
        def submit(text: str, whenBusy: str) -> asyncio.Future[None]:
            async def operation() -> None:
                watchAnswer(await current.submit({"type": "input", "content": text, "whenBusy": whenBusy}, context))

            return command(operation)

        def compact(instructions: str | None) -> asyncio.Future[None]:
            async def report(id: int) -> None:
                try:
                    receipt = await opened.waitForTask(id, context)
                    outcome = receipt["state"]["outcome"]
                    if outcome["status"] == "completed":
                        result = outcome["result"]
                        entryId, submissionId = result.get("entryId"), result.get("submissionId")
                        status = None
                        if submissionId is not None:
                            submission = await opened.submission(submissionId, context)
                            status = None if submission is None else (await submission.status(context))["status"]
                        if entryId is not None or status == "done":
                            message = "Compacted."
                        elif status == "queued":
                            message = "Compaction summary queued; it is placed at the next turn boundary."
                        elif status == "unanswered":
                            message = "Compaction summary dropped: the context changed under it."
                        else:
                            message = "Nothing to compact: the context fits in the recent window that is kept verbatim."
                        notice("info", message)
                    elif outcome["status"] == "aborted":
                        notice("info", "Compaction aborted.")
                    else:
                        detail = (outcome.get("error") or {}).get("message") or outcome.get("reason") or ""
                        notice("error", f"Compaction {outcome['status']}: {detail}")
                except Exception as error:  # noqa: BLE001 - shown as a notice
                    fail(error)

            async def operation() -> None:
                id = await current.compact(instructions, context)
                # Report the outcome once it is known; the status line shows the compaction meanwhile.
                spawn(report(id), background)

            return command(operation)

        # Not queued: it waits until the conversation is idle.
        @eager
        async def abort() -> None:
            try:
                await current.abort(context)
            except Exception as error:  # noqa: BLE001 - shown as a notice
                fail(error)

        def cycleThinking() -> asyncio.Future[None]:
            async def operation() -> None:
                model = agentModel()
                if not model.reasoning:
                    raise RuntimeError("Current model does not support thinking")
                levels = getSupportedThinkingLevels(model)
                level = agentOf(state["conversation"]).get("thinkingLevel") or "off"
                index = levels.index(level) if level in levels else -1
                following = levels[(index + 1) % len(levels)] if levels else "off"
                await current.configure({"thinkingLevel": following}, context)

            return command(operation)

        def setModel(ref: dict[str, str]) -> asyncio.Future[None]:
            async def operation() -> None:
                model = modelRuntime.getModel(ref["provider"], ref["modelId"])
                if model is None:
                    raise RuntimeError(f"Unknown model: {ref['provider']}/{ref['modelId']}")
                thinking: Any = agentOf(state["conversation"]).get("thinkingLevel") or "off"
                await current.configure({"model": ref, "thinkingLevel": clampThinkingLevel(model, thinking)}, context)

            return command(operation)

        def toggleTasks() -> asyncio.Future[None]:
            async def operation() -> None:
                nonlocal tasks, unsubscribeTasks
                if tasks is not None:
                    closeTasks()
                    update({"tasks": None})
                    return
                graph = await opened.taskGraph(context)
                tasks = graph
                unsubscribeTasks = graph.subscribe(lambda value, *_: update({"tasks": value}))

            return command(operation)

        def switchConversation(id: int) -> asyncio.Future[None]:
            async def operation() -> None:
                nonlocal current, conversation, unsubscribe
                following = await opened.conversation(id, context)
                if following is None:
                    raise RuntimeError(f"Conversation {id} does not exist")
                followingState = await following.viewState(context)
                unsubscribe()
                conversation.dispose()
                current = following
                conversation = followingState
                unsubscribe = followingState.subscribe(lambda value, *_: update({"conversation": value}))

            return command(operation)

        controller = SimpleNamespace(
            submit=submit,
            compact=compact,
            abort=abort,
            cycleThinking=cycleThinking,
            setModel=setModel,
            toggleTasks=toggleTasks,
            switchConversation=switchConversation,
        )

        saved = agentOf(state["conversation"]).get("model")
        if saved is None:
            notice("warning", "No model configured; select one with /model.")
        elif modelRuntime.getModel(saved["provider"], saved["modelId"]) is None:
            notice("warning", f"Saved model is unavailable: {saved['provider']}/{saved['modelId']}")
        if initial is not None and initial.get("fallbackMessage") is not None:
            notice("info", initial["fallbackMessage"])
        # The task panel starts open; /tasks hides it.
        await controller.toggleTasks()
        # Recovered work from an interrupted turn continues now.
        harness.resume()

        closing: list[asyncio.Future[None]] = []

        def subscribe(listener: Any) -> Any:
            listeners.append(listener)
            return lambda: listeners.remove(listener) if listener in listeners else None

        def close() -> asyncio.Future[None]:
            if not closing:

                async def run() -> None:
                    unsubscribe()
                    unsubscribeCommits()
                    conversation.dispose()
                    closeTasks()
                    try:
                        # Close writes no outcome: a running turn resumes with --continue.
                        await opened.close(context)
                        await envs.cleanup(context)
                    finally:
                        await location.release()

                closing.append(startTask(run()))
            # Shielded: cancelling one caller of ``close()`` must not cancel the close the others share.
            return asyncio.shield(closing[0])

        return SimpleNamespace(
            view=SimpleNamespace(current=lambda: state, subscribe=subscribe),
            controller=controller,
            settings=settingsManager,
            close=close,
        )
    except BaseException:
        if harness is not None:
            try:
                await harness.close(context)
            except Exception:  # noqa: BLE001, S110 - `.catch(() => {})`
                pass
        try:
            await location.release()
        except Exception:  # noqa: BLE001, S110 - `.catch(() => {})`
            pass
        raise
