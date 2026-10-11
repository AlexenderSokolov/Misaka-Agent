"""Agent events: the coding agent's session events, derived from the commits of one conversation.

Translated from pi's ``packages/durable/src/harness/events.ts`` (experimental, spec section 9.4). An event is a ``dict``
with a ``type`` and the upstream keys; the event types are ``snapshot``, ``run_start``, ``run_end``, ``turn_start``,
``turn_end``, ``message_start``, ``message_update`` (``usage`` and ``changes``), ``message_end``,
``tool_execution_start``, ``tool_execution_update``, ``tool_execution_end``, ``inbox_update``, ``submission``,
``auto_retry_start``, ``auto_retry_end``, ``deferred_poll``, ``entry_appended``, ``agent_changed``, ``usage_changed``,
``task_failed``, ``compaction_start`` and ``compaction_end``. A message change is ``text_start``, ``thinking_start`` or
``toolcall_start`` (``contentIndex``, ``block``), ``text_delta`` or ``thinking_delta`` (``contentIndex``, ``delta``),
``toolcall_delta`` (``contentIndex``, ``path``, ``delta``), ``block`` (``contentIndex``, ``block``) or ``message``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from misaka.chord.context import Context, abortError
from misaka.chord.delta.diff import strictEquals
from misaka.utils.js import UNDEFINED, eager

from ..session.observation import CommittedWatch
from ..types import CommitPublication, ConversationId
from .agent import AgentDoc
from .usage import UsageDoc
from .util import scanAll
from .view import conversationViews

PARTIAL_PATH = ["docs", "pi.live", "generation", "message"]


class _Observer:
    """The view observer of one event stream. A mount holds its observers in an identity set, so it is a plain class
    rather than a namespace, which compares by value."""

    def __init__(self, publication: Callable[..., None], closeSession: Callable[[], None]) -> None:
        self.publication = publication
        self.closeSession = closeSession


class AgentEventStream:
    """Serialized stream of one conversation's event batches, one per commit. ``snapshot`` is the ``snapshot`` event at
    attachment, ``closed`` settles when the stream ends."""

    def __init__(self, snapshot: dict[str, Any], watch: CommittedWatch) -> None:
        self.snapshot = snapshot
        self._watch = watch

    def start(self, listener: Any) -> None:
        self._watch.start(lambda events, ops, deliveryContext: listener(events, deliveryContext))

    def stop(self) -> Any:
        return self._watch.stop()

    @property
    def closed(self) -> Any:
        return self._watch.closed


def parts(view: dict[str, Any]) -> dict[str, Any]:
    """The typed parts of a view the events read."""
    docs = view["docs"]
    return {
        "live": docs.get("pi.live") or {},
        "inbox": docs.get("pi.inbox"),
        "agent": docs.get("pi.agent"),
        "usage": docs.get("pi.usage"),
    }


def snapshotOf(view: dict[str, Any]) -> dict[str, Any]:
    live, inbox, agent, usage = parts(view).values()
    return {
        "type": "snapshot",
        "entries": view["entries"],
        **({} if live.get("run") is None else {"run": {"inputs": live["run"]["inputs"]}}),
        **({} if live.get("generation") is None else {"generation": live["generation"]}),
        "tools": live.get("tools") or [],
        "compactions": live.get("compactions") or [],
        "inbox": queued(inbox),
        "agent": agent if agent is not None else AgentDoc["definition"]["initial"](),
        "usage": usage if usage is not None else UsageDoc["definition"]["initial"](),
    }


def queued(inbox: dict[str, Any] | None) -> list[dict[str, Any]]:
    return [{"id": item["id"], "mode": item["mode"]} for item in (inbox["items"] if inbox is not None else [])]


@eager
async def watchEvents(harness: Any, conversationId: ConversationId, context: Context) -> AgentEventStream:
    """Experimental: attach to one conversation's agent events (spec section 9.4). The snapshot and the registration for
    later commits are captured atomically on the Session line; overflow replaces undelivered batches with one
    snapshot."""
    holder: dict[str, Any] = {}

    async def create(initial: dict[str, Any], release: Any, storage: Any) -> Any:
        # Generations whose held outcome already ended their turn, read on the line with the snapshot.
        query = {"conversationId": conversationId, "kind": "pi.generation", "status": "completing"}
        completing = await scanAll(lambda cursor: storage.scanTasks(query, 100, cursor, context))
        held = {record["id"] for record in completing}
        current = initial
        holder["snapshot"] = snapshotOf(initial)
        # Batches are the watch's values; an overflow delivers a snapshot of the newest view instead.
        watch = CommittedWatch([], release, lambda: [snapshotOf(current)])
        holder["watch"] = watch

        def publication(
            before: dict[str, Any], after: dict[str, Any], ops: Any, committed: CommitPublication, commitContext: Context
        ) -> None:
            nonlocal current
            current = after
            events = translate(conversationId, before, after, ops, committed, held)
            if len(events) > 0:
                watch.advance(events, [], commitContext)

        return _Observer(publication, lambda: watch.closeSession())

    await conversationViews(harness).attach(conversationId, create, context)
    watch = holder["watch"]
    # Like a watch, the acquisition context governs the stream's lifetime.
    signal = context.abortSignal
    if signal is not None and signal.aborted:
        watch.cancel()
        raise abortError(signal)
    if signal is not None:
        watch.observeCancellation(signal)
    return AgentEventStream(holder["snapshot"], watch)


def resultOf(entries: list[dict[str, Any]], callId: str) -> dict[str, Any] | None:
    """The tool result for ``callId`` among ``entries``."""
    for entry in entries:
        model = entry.get("model")
        message = model[0] if model else None
        if message is not None and message["role"] == "toolResult" and message["toolCallId"] == callId:
            return entry
    return None


def translate(
    conversationId: ConversationId,
    before: dict[str, Any],
    after: dict[str, Any],
    viewOps: list[Any],
    publication: CommitPublication,
    held: set[int],
) -> list[dict[str, Any]]:
    """Every event one publication causes, in the order of spec section 9.4."""
    entries: list[dict[str, Any]] = []
    tasks: dict[int, dict[str, Any]] = {}
    submissions: list[dict[str, Any]] = []
    for change in publication["changes"]:
        if change["type"] == "entry" and change["value"]["conversationId"] == conversationId:
            entries.append(change["value"])
        if change["type"] == "task" and change["value"]["conversationId"] == conversationId:
            tasks[change["value"]["id"]] = change["value"]
        if change["type"] == "submission" and change["value"]["conversationId"] == conversationId:
            submissions.append(change["value"])
    if len(viewOps) == 0 and len(entries) == 0 and len(tasks) == 0 and len(submissions) == 0:
        return []
    # Entries are appended in ID order; submission records are published in the order the commit first touched them.
    submissions.sort(key=lambda record: record["id"])
    was = parts(before)
    now = parts(after)
    events: list[dict[str, Any]] = []

    # Progress: tool starts, the in-flight message, tool updates, retry and deferred state.
    slotsBefore = {slot["callId"]: slot for slot in was["live"].get("tools") or []}
    slots = now["live"].get("tools") or []
    for slot in slots:
        if slot["status"] != "running" or (slotsBefore.get(slot["callId"]) or {}).get("status") == "running":
            continue
        checkpoint = None if slot.get("taskId") is None else ((tasks.get(slot["taskId"]) or {}).get("state") or {}).get("checkpoint")
        args = checkpoint.get("arguments") if isinstance(checkpoint, dict) else None
        events.append(
            {
                "type": "tool_execution_start",
                "toolCallId": slot["callId"],
                "toolName": slot["name"],
                "args": args if args is not None else {},
            }
        )
    partialBefore = (was["live"].get("generation") or {}).get("message")
    partial = (now["live"].get("generation") or {}).get("message")
    if partial is not None and partialBefore is None:
        events.append({"type": "message_start", "message": partial})
    elif partial is not None and partial is not partialBefore:
        events.append(
            {"type": "message_update", "usage": partial["usage"], "changes": messageChanges(viewOps, partial)}
        )
    for index, slot in enumerate(slots):
        previous = slotsBefore.get(slot["callId"])
        if slot["status"] != "running" or previous is None or previous["status"] != "running":
            continue
        update = toolUpdate(viewOps, index, slot, previous)
        if update is None:
            continue
        events.append(
            {"type": "tool_execution_update", "toolCallId": slot["callId"], "toolName": slot["name"], **update}
        )
    generation = now["live"].get("generation")
    generationBefore = was["live"].get("generation")
    if generation is not None and generation.get("retry") is not None and (
        generationBefore is None or generationBefore.get("retry") is None
    ):
        events.append(
            {
                "type": "auto_retry_start",
                "attempt": generation["attempt"],
                "at": generation["retry"]["at"],
                "errorMessage": generation["retry"]["error"],
            }
        )
    if generationBefore is not None and generationBefore.get("retry") is not None and (
        generation is None or generation.get("retry") is None
    ):
        events.append({"type": "auto_retry_end", "attempt": generationBefore["attempt"]})
    if generation is not None and generation.get("deferred") is not None:
        pollBefore = ((generationBefore or {}).get("deferred") or {}).get("pollAt")
        if generation["deferred"]["pollAt"] != pollBefore:
            events.append({"type": "deferred_poll", "pollAt": generation["deferred"]["pollAt"]})

    # Tools that end in this commit: a slot that becomes done, one created done (a call not offered), or an unfinished
    # one that vanishes because its run ended. A done slot that vanishes ended earlier.
    toolEnds: list[dict[str, Any]] = []

    def endTool(callId: str, name: str, entryId: int | None) -> None:
        entry = next((candidate for candidate in entries if candidate["id"] == entryId), None)
        toolEnds.append(
            {
                "type": "tool_execution_end",
                "toolCallId": callId,
                "toolName": name,
                **({} if entry is None else {"entry": entry}),
            }
        )

    for previous in slotsBefore.values():
        if previous["status"] == "done":
            continue
        slot = next((candidate for candidate in slots if candidate["callId"] == previous["callId"]), None)
        if slot is not None and slot["status"] == "done":
            endTool(previous["callId"], previous["name"], slot.get("entry"))
        # A slot whose run ended in this commit may have had its result appended with it, as for unstarted calls.
        elif slot is None:
            result = resultOf(entries, previous["callId"])
            endTool(previous["callId"], previous["name"], None if result is None else result["id"])
    for slot in slots:
        if slot["status"] == "done" and slot["callId"] not in slotsBefore:
            endTool(slot["callId"], slot["name"], slot.get("entry"))

    # Entries in append order; a tool's end directly precedes its result's message, as in the coding agent.
    assistantAppended = False
    for entry in entries:
        events.extend(end for end in toolEnds if end.get("entry") is entry)
        model = entry.get("model")
        message = model[0] if model else None
        if message is None:
            events.append({"type": "entry_appended", "entry": entry})
            continue
        # A streamed answer already started with its first partial.
        streamed = message["role"] == "assistant" and partialBefore is not None and not assistantAppended
        if message["role"] == "assistant":
            assistantAppended = True
        if not streamed:
            events.append({"type": "message_start", "message": message})
        events.append({"type": "message_end", "entry": entry})
    # Ends without a result entry: a faulted or orphaned tool, or one whose run ended.
    events.extend(end for end in toolEnds if end.get("entry") is None)

    # Compaction ends, task failures, then turn and run ends.
    compactionsBefore = was["live"].get("compactions") or []
    compactions = now["live"].get("compactions") or []
    for status in compactionsBefore:
        if not any(candidate["taskId"] == status["taskId"] for candidate in compactions):
            events.append({"type": "compaction_end", "taskId": status["taskId"], "reason": status["reason"]})
    # A generation's turn ends when its outcome is committed: at a ``completing`` hold or at terminal, whichever comes
    # first, so a successor created at the hold starts after it.
    turnEnded = False
    for task in tasks.values():
        status = task["state"]["status"]
        if task["kind"] == "pi.generation" and status == "completing" and task["id"] not in held:
            held.add(task["id"])
            turnEnded = True
        if status != "terminal":
            continue
        if task["kind"] == "pi.generation":
            if task["id"] in held:
                held.discard(task["id"])
            else:
                turnEnded = True
        outcome = task["state"]["outcome"]
        if outcome["status"] in ("faulted", "orphaned"):
            failure = outcome["error"]["message"] if outcome["status"] == "faulted" else outcome["reason"]
            events.append({"type": "task_failed", "taskId": task["id"], "kind": task["kind"], "message": failure})
    if turnEnded:
        events.append({"type": "turn_end"})
    run = now["live"].get("run")
    runBefore = was["live"].get("run")
    runChanged = firstInput(run) != firstInput(runBefore)
    if runBefore is not None and runChanged:
        events.append({"type": "run_end", "inputs": runBefore["inputs"]})

    # Submissions, document state, then what began.
    for record in submissions:
        events.append({"type": "submission", "record": record})
    if now["inbox"] is not was["inbox"]:
        events.append({"type": "inbox_update", "items": queued(now["inbox"])})
    # A retired document reads as its initial value, as in a snapshot.
    if now["agent"] is not was["agent"]:
        events.append(
            {"type": "agent_changed", "agent": now["agent"] if now["agent"] is not None else AgentDoc["definition"]["initial"]()}
        )
    if now["usage"] is not was["usage"]:
        events.append(
            {"type": "usage_changed", "usage": now["usage"] if now["usage"] is not None else UsageDoc["definition"]["initial"]()}
        )
    for status in compactions:
        if not any(candidate["taskId"] == status["taskId"] for candidate in compactionsBefore):
            events.append(
                {
                    "type": "compaction_start",
                    "taskId": status["taskId"],
                    "reason": status["reason"],
                    "blocking": status["blocking"],
                }
            )
    if run is not None and runChanged:
        events.append({"type": "run_start", "inputs": run["inputs"]})
    if (
        run is not None
        and run["taskId"] != (None if runBefore is None else runBefore["taskId"])
        and (tasks.get(run["taskId"]) or {}).get("kind") == "pi.generation"
    ):
        events.append({"type": "turn_start"})
    return events


def firstInput(run: dict[str, Any] | None) -> Any:
    return run["inputs"][0] if run is not None and len(run["inputs"]) > 0 else None


def messageChanges(viewOps: list[Any], message: dict[str, Any]) -> list[dict[str, Any]]:
    """Translate the view operations on the in-flight message into message changes (spec section 9.4)."""
    changes: list[dict[str, Any]] = []
    # A block sent whole already holds every later change to it in this batch.
    whole: set[int] = set()
    for op in viewOps:
        # View operations never replace the root.
        path = op[1]
        if not startsWith(path, PARTIAL_PATH):
            # The whole message or generation was replaced.
            if startsWith(PARTIAL_PATH, path):
                return [{"type": "message", "message": message}]
            continue
        rest = path[len(PARTIAL_PATH) :]
        if len(rest) > 0 and rest[0] == "usage":
            continue
        if len(rest) == 0 or rest[0] != "content":
            return [{"type": "message", "message": message}]
        if len(rest) == 1:
            if op[0] != "p" or op[3] != 0:
                return [{"type": "message", "message": message}]
            for offset, block in enumerate(op[4]):
                kind = (
                    "text_start"
                    if block["type"] == "text"
                    else "thinking_start"
                    if block["type"] == "thinking"
                    else "toolcall_start"
                )
                changes.append({"type": kind, "contentIndex": op[2] + offset, "block": block})
            continue
        contentIndex = rest[1]
        field = rest[2] if len(rest) > 2 else None
        if contentIndex in whole:
            continue
        if op[0] == "a" and len(rest) == 3 and field in ("text", "thinking"):
            changes.append(
                {
                    "type": "text_delta" if field == "text" else "thinking_delta",
                    "contentIndex": contentIndex,
                    "delta": op[2],
                }
            )
        elif op[0] == "a" and field == "arguments":
            changes.append(
                {"type": "toolcall_delta", "contentIndex": contentIndex, "path": rest[3:], "delta": op[2]}
            )
        else:
            whole.add(contentIndex)
            changes.append({"type": "block", "contentIndex": contentIndex, "block": message["content"][contentIndex]})
    return changes


def toolUpdate(viewOps: list[Any], index: int, slot: dict[str, Any], previous: dict[str, Any]) -> dict[str, Any] | None:
    """Output, details, and diagnostics changes of a running slot, from the view operations on it."""
    outputPath = ["docs", "pi.live", "tools", index, "output"]
    trimStart = 0
    append = ""
    isSet = False
    for op in viewOps:
        if not startsWith(op[1], outputPath):
            continue
        if op[0] == "t":
            trimStart += op[2]
        elif op[0] == "a":
            append += op[2]
        else:
            isSet = True
    output: dict[str, Any] | None = None
    if isSet or (
        not strictEquals(slot.get("output", UNDEFINED), previous.get("output", UNDEFINED)) and trimStart == 0 and append == ""
    ):
        output = {"set": slot.get("output") if slot.get("output") is not None else ""}
    elif trimStart > 0 or append != "":
        output = {**({"trimStart": trimStart} if trimStart > 0 else {}), **({} if append == "" else {"append": append})}
    # A safe replay clears a running slot's progress: removed details send ``None``, removed diagnostics ``[]``.
    update: dict[str, Any] = {
        **({} if output is None else {"output": output}),
        **(
            {}
            if strictEquals(slot.get("details", UNDEFINED), previous.get("details", UNDEFINED))
            else {"details": slot.get("details")}
        ),
        **(
            {}
            if strictEquals(slot.get("diagnostics", UNDEFINED), previous.get("diagnostics", UNDEFINED))
            else {"diagnostics": slot["diagnostics"] if slot.get("diagnostics") is not None else []}
        ),
    }
    return None if len(update) == 0 else update


def startsWith(path: list[Any], prefix: list[Any]) -> bool:
    return len(prefix) <= len(path) and all(path[index] == segment for index, segment in enumerate(prefix))


__all__ = ["AgentEventStream", "watchEvents"]
