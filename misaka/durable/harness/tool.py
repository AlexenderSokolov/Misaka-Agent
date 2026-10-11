"""The built-in tool task: runs one tool call of an assistant entry.

Translated from pi's ``packages/durable/src/harness/tool.ts``. A tool is a ``dict`` (``name``, ``description``,
``parameters`` as a JSON schema, ``execute(args, api, context)``, optional ``replay``, ``executionMode``,
``prepareArguments(args)`` and ``outputLimits``), a result is a ``dict`` (``content``, ``isError``, ``details``,
``diagnostics``, ``usage``, ``control``) and the ``api`` a tool receives is a ``types.SimpleNamespace`` (spread it with
``SimpleNamespace(**{**vars(api), "name": value})``) with ``taskId``, ``conversationId``, ``callId``, ``registry``,
``env``, ``models``, ``agent``, ``output``, ``outputWindow``, ``diagnostic``, ``details``, ``commit``, ``memo``,
``createTask``, ``getTask``, ``waitForTask``, ``conversation``, ``snapshot``, ``snapshotAsOf`` and ``watchDoc``.
``details`` that is absent is ``UNDEFINED``. ``outputWindow`` is the tail this call's output keeps, for
``ShellExecOptions.window``, or ``None`` when the tool keeps the head; a wrapper that replaces ``output`` and transforms
text must also set it to ``None``, so skipped text cannot bypass its transform.
"""

from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any

from misaka.ai.types import Tool as AiTool
from misaka.ai.types import ToolCall as AiToolCall
from misaka.ai.utils.validation import validateToolArguments
from misaka.chord.context import Context, awaitWithContext
from misaka.chord.delta import overlap
from misaka.chord.delta.diff import strictEquals
from misaka.chord.json import copyJson
from misaka.utils.js import (
    UNDEFINED,
    awaitValue,
    eager,
    errorMessage,
    jsLength,
    jsonStringify,
    jsRound,
    jsSlice,
)

from ..entries import AssistantEntry, ToolResultEntry
from ..tasks import defineTask
from ..truncate import DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, utf8ByteLength
from ..types import ConversationId, Tx
from .json import assignJson
from .live import LiveDoc, clearProgress, finishSlot, toolSlot
from .output import (
    PROGRESS_BYTES_PER_SECOND,
    OutputBuffer,
    OutputLimits,
    Progress,
    boundOutput,
)
from .usage import recordUsage

COMPLETED: dict[str, Any] = {"status": "completed"}
"""How a tool task ends; the result entry is appended either way. ``failed`` (execution threw or was interrupted)
records cancellation intent for the conversations the call owns; a result with ``isError`` still completes."""


async def callPhase(task: dict[str, Any], runtime: Any, context: Context) -> None:
    call = await readCall(runtime, task["input"], context)
    tool = next((each for each in (await runtime.agent(context))["tools"] if each["name"] == call["name"]), None)
    if tool is None:
        error = harnessError("tool_unavailable", f"Tool {call['name']} is not available")
        await settle(runtime, call, COMPLETED, lambda slot: error, context)
        return
    prepared = prepare(tool, call["arguments"])
    checked = prepared if "error" in prepared else validate(tool, call, prepared["args"])
    if "error" in checked:
        await settle(runtime, call, COMPLETED, lambda slot: invalid(checked["error"]), context)
        return
    args = checked["args"]
    block: str | None = None

    async def beforeTool(hook: Any) -> None:
        nonlocal block, args
        if block is not None:
            return
        try:
            decision = await awaitValue(hook({**call, "arguments": args}, runtime, context))
            if decision is not None and decision.get("block") is not None:
                block = decision["block"]
            elif decision is not None and decision.get("arguments") is not None:
                args = decision["arguments"]
        except Exception as error:
            if runtime.signal.aborted:
                raise
            block = errorText(error)

    await runtime.hooks.each("beforeTool", beforeTool)
    if block is not None:
        blocked = harnessError("blocked", f"Tool call blocked: {block}")
        await settle(runtime, call, COMPLETED, lambda slot: blocked, context)
        return
    validated = validate(tool, call, args)
    if "error" in validated:
        await settle(runtime, call, COMPLETED, lambda slot: invalid(validated["error"]), context)
        return
    final = validated["args"]

    async def recordIntent(tx: Tx, current: dict[str, Any]) -> dict[str, Any]:
        slot = toolSlot(await tx.doc(LiveDoc, runtime.conversationId), runtime.taskId)
        if slot is not None:
            slot["status"] = "running"
        intent = {"phase": "execute", "arguments": final, "replay": tool["replay"] if tool.get("replay") is not None else "unsafe"}
        return {"status": "running", "checkpoint": intent}

    await runtime.commit(recordIntent, context)
    await run(runtime, call, tool, final, context)


async def executePhase(task: dict[str, Any], runtime: Any, context: Context) -> None:
    """Recovery after intent: rerun only when the stored and the current policy both say ``safe``."""
    args = task["state"]["checkpoint"]["arguments"]
    replay = task["state"]["checkpoint"]["replay"]
    call = await readCall(runtime, task["input"], context)
    tool = next((each for each in (await runtime.agent(context))["tools"] if each["name"] == call["name"]), None)
    if replay == "safe" and tool is not None and tool.get("replay") == "safe":
        # The rerun reports from scratch; clear what the interrupted attempt published.
        async def clear(tx: Tx, current: dict[str, Any]) -> None:
            slot = toolSlot(await tx.doc(LiveDoc, runtime.conversationId), runtime.taskId)
            if slot is not None:
                clearProgress(slot)

        await runtime.commit(clear, context)
        await run(runtime, call, tool, args, context)
        return
    message = f"Tool {call['name']} was interrupted and may have partially run"
    # ``failed`` records cancellation intent, so the call's owned conversations, left unsupervised, are aborted.
    ending = {"status": "failed", "message": message}
    await settle(runtime, call, ending, lambda slot: fromSlot(slot, "interrupted", message), context)


async def abortPhase(task: dict[str, Any], runtime: Any, context: Context) -> None:
    call = await readCall(runtime, task["input"], context)
    message = f"Tool {call['name']} was aborted"
    await settle(runtime, call, {"status": "aborted"}, lambda slot: fromSlot(slot, "aborted", message), context)


ToolTask = defineTask(
    {
        "name": "pi.tool",
        "version": 1,
        "initial": lambda input: {"phase": "call"},
        "phases": {"call": callPhase, "execute": executePhase},
        "abort": abortPhase,
    }
)
"""Built-in tool task: resolves the called tool among its phase agent's tools, validates, runs ``beforeTool``, records
intent, executes, runs ``afterTool``, and appends the result, all in one ``call`` handler so nothing separates
resolution from settlement. ``execute`` is reached only by recovery and applies the replay rule."""


async def readCall(runtime: Any, input: dict[str, Any], context: Context) -> dict[str, Any]:
    """The tool call ``callId`` of the assistant entry."""
    entry = await runtime.entry(AssistantEntry, input["assistant"], context)
    model = None if entry is None else entry.get("model")
    message = model[0] if model else None
    call = None
    if message is not None and message["role"] == "assistant":
        call = next(
            (content for content in message["content"] if content["type"] == "toolCall" and content["id"] == input["callId"]),
            None,
        )
    if call is None:
        raise RuntimeError(f"Entry {input['assistant']} has no tool call {input['callId']}")
    return call  # type: ignore[no-any-return]


def prepare(tool: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    """The call's arguments as repaired by the tool; a throwing repair makes them invalid. ``{"args": ...}`` or
    ``{"error": ...}``."""
    if tool.get("prepareArguments") is None:
        return {"args": args}
    try:
        return {"args": tool["prepareArguments"](args)}
    except Exception as error:  # noqa: BLE001 - a throwing repair makes the arguments invalid
        return {"error": errorText(error)}


def validate(tool: dict[str, Any], call: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    """Arguments validated and coerced against the implementation's schema."""
    try:
        declared = AiTool(name=tool["name"], description=tool["description"], parameters=tool["parameters"])
        return {"args": validateToolArguments(declared, AiToolCall.model_validate({**call, "arguments": args}))}
    except Exception as error:  # noqa: BLE001 - invalid arguments are reported to the model
        return {"error": errorText(error)}


def invalid(message: str) -> dict[str, Any]:
    return harnessError("invalid_arguments", message)


async def run(runtime: Any, call: dict[str, Any], tool: dict[str, Any], args: dict[str, Any], context: Context) -> None:
    """Execute with the resolved implementation, then settle its result."""
    outputLimits = tool.get("outputLimits") or {}
    limits: OutputLimits = {
        "maxBytes": outputLimits["maxBytes"] if outputLimits.get("maxBytes") is not None else DEFAULT_MAX_BYTES,
        "maxLines": outputLimits["maxLines"] if outputLimits.get("maxLines") is not None else DEFAULT_MAX_LINES,
        "retain": outputLimits["retain"] if outputLimits.get("retain") is not None else "head",
    }
    # What a running tool reported through its api: output, the last details, and diagnostics.
    reported: dict[str, Any] = {"output": OutputBuffer(limits), "limits": limits, "diagnostics": [], "details": UNDEFINED}
    progress = publishProgress(runtime, reported, context)
    ended = False

    def assertLive() -> None:
        if ended:
            raise RuntimeError(f"Tool call {call['id']} has settled")

    def output(chunk: Any, skipped: dict[str, Any] | None = None) -> None:
        assertLive()
        if reported["output"].push(chunk, skipped):
            progress.mark()

    def diagnostic(item: dict[str, Any]) -> None:
        assertLive()
        reported["diagnostics"].append(copyJson(item, {"omitUndefinedProperties": True}))
        progress.mark()

    @eager
    async def details(value: Any, detailsContext: Context) -> None:
        assertLive()
        signal = detailsContext.abortSignal
        if signal is not None:
            signal.throwIfAborted()
        reported["details"] = copyJson(value, {"omitUndefinedProperties": True})
        committed = progress.markAndWait()
        # Cancelling the wait leaves the update in place; the commit's own outcome stays observed.
        committed.add_done_callback(lambda done: None if done.cancelled() else done.exception())
        return await awaitWithContext(committed, detailsContext)  # type: ignore[no-any-return]

    @eager
    async def commit(change: Any, commitContext: Context) -> Any:
        result: Any = UNDEFINED

        async def apply(tx: Tx, current: dict[str, Any]) -> None:
            nonlocal result
            result = await awaitValue(change(tx))

        await runtime.commit(apply, commitContext)
        return result

    @eager
    async def createTask(task: dict[str, Any], input: Any, options: dict[str, Any], taskContext: Context) -> Any:
        id: Any = None

        async def apply(tx: Tx, current: dict[str, Any]) -> None:
            nonlocal id
            id = await tx.createTask(task, input, options)

        await runtime.commit(apply, taskContext)
        return id

    api = SimpleNamespace(
        taskId=runtime.taskId,
        conversationId=runtime.conversationId,
        callId=call["id"],
        registry=runtime.registry,
        agent=runtime.agent,
        models=runtime.models,
        output=output,
        outputWindow=(
            {
                "maxBytes": limits["maxBytes"],
                "maxLines": limits["maxLines"],
                "minIntervalMs": runtime.settings["progress"]["outputIntervalMs"],
                "bytesPerSecond": PROGRESS_BYTES_PER_SECOND,
            }
            if limits["retain"] == "tail"
            else None
        ),
        diagnostic=diagnostic,
        details=details,
        commit=commit,
        memo=runtime.memo,
        createTask=createTask,
        getTask=runtime.getTask,
        waitForTask=runtime.waitForTask,
        conversation=runtime.conversation,
        snapshot=runtime.snapshot,
        snapshotAsOf=runtime.snapshotAsOf,
        watchDoc=runtime.watchDoc,
    )

    result: dict[str, Any]
    ending = COMPLETED
    # Execution time of this attempt; a rerun after recovery measures only itself.
    durationMs: int | None = None
    try:
        # Built for this call, so a rerun after recovery gets the conversation's environment at that time.
        env = await runtime.env(context)
        startedAt = time.perf_counter()
        try:
            result = await awaitValue(tool["execute"](args, SimpleNamespace(**{**vars(api), "env": env}), context))
        finally:
            durationMs = jsRound((time.perf_counter() - startedAt) * 1000)
    except Exception as error:
        if runtime.signal.aborted:
            ended = True
            for waiter in await progress.stop():
                if not waiter.done():
                    waiter.set_exception(error)
            raise
        result = {"isError": True, "diagnostics": [toolDiagnostic("tool_error", errorText(error))]}
        # A throw, from ``execute()`` or from building the environment, ends the task ``failed``, which cancels what
        # the call owned; it no longer supervises it. The error text is already in the result entry.
        ending = {"status": "failed", "message": f"Tool {call['name']} threw"}
    ended = True
    reported["output"].end()
    # Details still waiting for a progress commit settle with the terminal commit, the final flush.
    pending = await progress.stop()
    try:
        settled = await finalResult(runtime, call, result, reported, context)
        await settle(runtime, call, ending, lambda slot: settled, context, durationMs)
    except Exception as error:
        for waiter in pending:
            if not waiter.done():
                waiter.set_exception(error)
        raise
    for waiter in pending:
        if not waiter.done():
            waiter.set_result(None)


def publishProgress(runtime: Any, reported: dict[str, Any], context: Context) -> Progress:
    """Throttled commits of what the tool reported into its ``pi.live.tools`` slot, each writing only what changed since
    the last one."""
    written: dict[str, Any] = {"text": "", "details": UNDEFINED, "diagnostics": 0}

    async def write() -> int:
        nonlocal written
        # Capture everything synchronously: the tool keeps reporting while the commit is in flight.
        snapshot = reported["output"].snapshot()
        current = {"text": snapshot["text"], "details": reported["details"], "diagnostics": len(reported["diagnostics"])}
        added = reported["diagnostics"][written["diagnostics"] : current["diagnostics"]]
        detailsChanged = not strictEquals(current["details"], written["details"])
        # What the commit writes, as Chord diffs the string: an append, a trim plus an append of what follows the shared
        # part, or the whole window when its bounded overlap search finds nothing.
        size = 0
        if snapshot["text"] != written["text"]:
            shared = (
                jsLength(written["text"])
                if snapshot["text"].startswith(written["text"])
                else overlap(written["text"], snapshot["text"], 65_536)
            )
            size += utf8ByteLength(jsSlice(snapshot["text"], shared))
        if detailsChanged:
            size += utf8ByteLength(jsonStringify(None if current["details"] is UNDEFINED else current["details"]))
        if len(added) > 0:
            size += utf8ByteLength(jsonStringify(added))

        async def apply(tx: Tx, task: dict[str, Any]) -> None:
            slot = toolSlot(await tx.doc(LiveDoc, runtime.conversationId), runtime.taskId)
            if slot is None:
                return
            # REMINDER: assign ``output`` as one string field. Chord then diffs it into an append, or a trim plus an
            # append for a sliding tail; replacing the slot object would record the whole window on every commit.
            existing = slot.get("output", "")
            if existing != snapshot["text"]:
                slot["output"] = snapshot["text"]
            if snapshot["droppedBytes"] > 0:
                slot["droppedBytes"] = snapshot["droppedBytes"]
            if snapshot["droppedLines"] > 0:
                slot["droppedLines"] = snapshot["droppedLines"]
            # Diff details leaf by leaf and append new diagnostics, so each commit writes only what changed.
            if detailsChanged and current["details"] is not UNDEFINED:
                assignJson(slot, "details", current["details"])
            if len(added) > 0:
                if "diagnostics" not in slot:
                    slot["diagnostics"] = []
                for item in added:
                    slot["diagnostics"].push(item)
            return

        await runtime.commit(apply, context)
        written = current
        return size

    def onError(error: BaseException) -> None:
        # Rejections after an abort mark or close are expected; the committed state stays consistent.
        if not runtime.signal.aborted:
            runtime.report(error)

    return Progress(write, onError, runtime.settings["progress"]["outputIntervalMs"])


async def finalResult(
    runtime: Any, call: dict[str, Any], result: dict[str, Any], reported: dict[str, Any], context: Context
) -> dict[str, Any]:
    """The settled result: the tool's result with the retained output and last details as fallbacks, its diagnostics
    after those reported through the api, ``afterTool`` applied, and explicit text bounded, with the Harness's
    truncation diagnostic last."""
    harness: list[dict[str, Any]] = []
    retained = reported["output"].snapshot() if result.get("content") is None else None
    content = (
        result["content"]
        if retained is None
        else ([] if retained["text"] == "" else [{"type": "text", "text": retained["text"]}])
    )
    details = result.get("details", UNDEFINED)
    if details is UNDEFINED:
        details = reported["details"]
    final: dict[str, Any] = {
        **result,
        "content": content,
        **({} if details is UNDEFINED else {"details": details}),
        "diagnostics": [*reported["diagnostics"], *(result.get("diagnostics") or [])],
    }

    async def afterTool(hook: Any) -> None:
        nonlocal final
        replaced = await awaitValue(hook(call, final, runtime, context))
        final = replaced if replaced is not None else final

    await runtime.hooks.each("afterTool", afterTool)
    # The retained output's truncation applies only while afterTool kept that content.
    if retained is not None and retained["droppedBytes"] > 0 and final.get("content") is content:
        harness.append(truncated(retained, reported["limits"]["retain"]))
    bounded = boundContent(final.get("content") or [], reported["limits"])
    if bounded["droppedBytes"] > 0:
        harness.append(truncated(bounded, reported["limits"]["retain"]))
    return {**final, "content": bounded["content"], "diagnostics": [*(final.get("diagnostics") or []), *harness]}


async def settle(
    runtime: Any, call: dict[str, Any], ending: dict[str, Any], build: Any, context: Context, durationMs: int | None = None
) -> None:
    """Commit the tool's terminal state: append its result entry, mark its slot done, and complete or end aborted with
    the entry ID. ``build`` receives the slot so interruption and abort can report the durable partial output."""

    async def apply(tx: Tx, current: dict[str, Any]) -> dict[str, Any]:
        slot = toolSlot(await tx.doc(LiveDoc, runtime.conversationId), runtime.taskId)
        result = build(slot)
        entry = await appendToolResult(tx, runtime.conversationId, call, result, runtime.now(), durationMs)
        if slot is not None:
            finishSlot(slot, entry["id"])
        entryId = entry["id"]
        if ending["status"] == "aborted":
            return {"status": "terminal", "outcome": {"status": "aborted", "result": {"entryId": entryId}}}
        if ending["status"] == "failed":
            error = {"message": ending["message"]}
            return {"status": "terminal", "outcome": {"status": "failed", "error": error, "result": {"entryId": entryId}}}
        # Tools build control objects freely; drop keys set to undefined so the task result is strict JSON.
        control = (
            {}
            if result.get("control") is None
            else {"control": copyJson(result["control"], {"omitUndefinedProperties": True})}
        )
        return {"status": "terminal", "outcome": {"status": "completed", "result": {"entryId": entryId, **control}}}

    await runtime.commit(apply, context)


def fromSlot(slot: Any, code: str, message: str) -> dict[str, Any]:
    """An error result from the slot's durable partial output, details, and diagnostics."""
    diagnostics = list(slot["diagnostics"]) if slot is not None and "diagnostics" in slot else []
    droppedBytes = slot["droppedBytes"] if slot is not None and "droppedBytes" in slot else 0
    if droppedBytes > 0:
        droppedLines = slot.get("droppedLines", 0)
        diagnostics.append(truncated({"droppedBytes": droppedBytes, "droppedLines": droppedLines}))
    diagnostics.append(toolDiagnostic(code, message))
    output = slot["output"] if slot is not None and "output" in slot else None
    return {
        "content": [] if output is None or output == "" else [{"type": "text", "text": output}],
        "isError": True,
        **({} if slot is None or "details" not in slot else {"details": slot["details"]}),
        "diagnostics": diagnostics,
    }


def harnessError(code: str, message: str) -> dict[str, Any]:
    """An error result the Harness writes itself: no content and one ``error`` diagnostic with ``code``."""
    return {"content": [], "isError": True, "diagnostics": [toolDiagnostic(code, message)]}


def toolDiagnostic(code: str, message: str) -> dict[str, Any]:
    return {"severity": "error", "code": code, "message": message}


def truncated(dropped: dict[str, Any], retain: str | None = None) -> dict[str, Any]:
    """The Harness's truncation diagnostic; ``retain`` is unknown when rebuilt from a slot after recovery."""
    kept = "" if retain is None else f" to its {'beginning' if retain == 'head' else 'end'}"
    return {
        "severity": "warn",
        "code": "truncated",
        "message": f"Output truncated{kept}: {dropped['droppedLines']} lines, {dropped['droppedBytes']} bytes dropped",
    }


async def appendToolResult(
    tx: Tx,
    conversationId: ConversationId,
    call: dict[str, Any],
    result: dict[str, Any],
    timestamp: int,
    durationMs: int | None = None,
) -> dict[str, Any]:
    """Append a ``pi.tool-result`` entry. The content ends with the rendered diagnostics, so the stored message is exactly
    what the model sees; ``data`` keeps the structured list. A result's usage is added to ``pi.usage`` in the same
    commit."""
    diagnostics = list(result.get("diagnostics") or [])
    content = list(result.get("content") or [])
    if len(diagnostics) > 0:
        content.append({"type": "text", "text": renderDiagnostics(diagnostics)})
    details = result.get("details", UNDEFINED)
    message = {
        "role": "toolResult",
        "toolCallId": call["id"],
        "toolName": call["name"],
        "content": content,
        **({} if details is UNDEFINED else {"details": details}),
        **({} if result.get("usage") is None else {"usage": result["usage"]}),
        "isError": result["isError"] if result.get("isError") is not None else False,
        **({} if durationMs is None else {"durationMs": durationMs}),
        "timestamp": timestamp,
    }
    if result.get("usage") is not None:
        await recordUsage(tx, conversationId, "tools", call["name"], result["usage"])
    return await tx.appendEntry(ToolResultEntry, conversationId, {"model": [message], "data": {"diagnostics": diagnostics}})  # type: ignore[no-any-return]


def renderDiagnostics(diagnostics: list[dict[str, Any]]) -> str:
    lines = "\n".join(f"[{diagnostic['severity']}] {diagnostic['message']}" for diagnostic in diagnostics)
    return f"<harness>\n{lines}\n</harness>"


def boundContent(content: list[dict[str, Any]], limits: OutputLimits) -> dict[str, Any]:
    """Bound the text of result content. When the joined text exceeds the limits, the text items are replaced by one
    bounded item at the position of the first (head) or last (tail) text item; other content is kept."""
    texts = [item for item in content if item["type"] == "text"]
    bounded = boundOutput("".join(item["text"] for item in texts), limits)
    if bounded["droppedBytes"] == 0:
        return {"content": content, "droppedBytes": 0, "droppedLines": 0}
    keep = texts[0] if limits["retain"] == "head" else texts[-1]
    result: list[dict[str, Any]] = []
    for item in content:
        if item["type"] != "text":
            result.append(item)
        elif item is keep:
            result.append({**item, "text": bounded["text"]})
    return {"content": result, "droppedBytes": bounded["droppedBytes"], "droppedLines": bounded["droppedLines"]}


def errorText(error: BaseException) -> str:
    return errorMessage(error)



__all__ = ["ToolTask", "appendToolResult", "harnessError"]
