"""The built-in live conversation state: run control and presentation of the current generation and tool round.

Translated from pi's ``packages/durable/src/harness/live.ts``. The state is a plain ``dict`` (``run``, ``generation``,
``tools``, ``compactions``); inside a commit it is a draft, read by item and changed by item assignment and ``del``.
"""

from __future__ import annotations

from typing import Any, TypedDict

from misaka.utils.js import UNDEFINED

from ..documents import defineDoc
from ..types import TaskId, Tx


class ToolSlot(TypedDict, total=False):
    """Presentation of one tool call of the current round. ``taskId`` is absent for a call not started yet (sequential
    round) and for a call its request did not offer, which starts ``done`` with the ``entry`` generation wrote;
    ``output``, ``droppedBytes`` and ``droppedLines`` are the retained running output and what the bounds dropped,
    ``details`` the last ``details()`` value, ``diagnostics`` those recorded through ``api.diagnostic()``, and ``entry``
    the result entry once done (absent when the tool task faulted or was orphaned)."""

    callId: str
    name: str
    taskId: TaskId
    status: str  # "pending" | "running" | "done"
    output: str
    droppedBytes: int
    droppedLines: int
    details: Any
    diagnostics: list[dict[str, Any]]
    entry: int


class CompactionStatus(TypedDict, total=False):
    """Presentation of one live compaction task (spec section 8.7): ``taskId``, ``reason``, ``blocking`` (a generation
    waits for it), ``attempt``, and ``retry`` (``{"at", "error"}``, the durable backoff before the next attempt)."""

    taskId: TaskId
    reason: str
    blocking: bool
    attempt: int
    retry: dict[str, Any]


LiveDoc = defineDoc(
    {
        "kind": "pi.live",
        "version": 1,
        "scope": "conversation",
        "history": "latest",
        "fork": "initial",
        "initial": dict,
        # REMINDER: a complete base whenever nothing runs (spec section 8.2): no generation and no running tool slot.
        # That holds while idle, in the commit handing a generation over to its tool round, and between tools, so the
        # delta chain spans at most one generation or the overlapping execution of one round's tools. A slot holds
        # output only while running, so every base is small. Do not add a delta-count bound; the tool output
        # benchmark checks this rule. MISAKA fork: that benchmark (upstream's storage-benchmark.ts) is not ported.
        "checkpointWhen": lambda value, *_: "generation" not in value
        and not any(slot["status"] == "running" for slot in value.get("tools") or []),
    }
)

RUN_TASK_KINDS = frozenset({"pi.generation"})
"""Built-in task kinds that can own ``pi.live.run``."""
TOOL_TASK_KIND = "pi.tool"
COMPACTION_TASK_KIND = "pi.compaction"


def endRun(tx: Tx, live: Any, taskId: TaskId, settlement: dict[str, Any]) -> None:
    """End the run owned by ``taskId``: settle each of its inputs and remove ``run``. Always removes ``generation`` and
    ``tools``, whose presentation belongs to the ending run."""
    run = live.get("run", UNDEFINED)
    if run is not UNDEFINED and run["taskId"] == taskId:
        for id in run["inputs"]:
            tx.settleSubmission(id, settlement)
        del live["run"]
    if "generation" in live:
        del live["generation"]
    if "tools" in live:
        del live["tools"]


def addCompactionStatus(live: Any, status: dict[str, Any]) -> None:
    """Add the status of a compaction task created in this commit; statuses stay in task ID order."""
    if live.get("compactions", UNDEFINED) is UNDEFINED:
        live["compactions"] = []
    live["compactions"].push(status)


def compactionStatus(live: Any, taskId: TaskId) -> Any:
    """The status of compaction task ``taskId``, if listed."""
    statuses = live.get("compactions", UNDEFINED)
    if statuses is UNDEFINED:
        return None
    for status in statuses:
        if status["taskId"] == taskId:
            return status
    return None


def removeCompactionStatus(live: Any, taskId: TaskId) -> None:
    """Remove the status of compaction task ``taskId``, and the list once empty."""
    statuses = live.get("compactions", UNDEFINED)
    if statuses is UNDEFINED:
        return
    index = next((at for at, status in enumerate(statuses) if status["taskId"] == taskId), -1)
    if index >= 0:
        statuses.splice(index, 1)
    if len(statuses) == 0:
        del live["compactions"]


def toolSlot(live: Any, taskId: TaskId) -> Any:
    """The slot of tool task ``taskId`` in the current round, if the round still lists it."""
    slots = live.get("tools", UNDEFINED)
    if slots is UNDEFINED:
        return None
    for slot in slots:
        if slot.get("taskId", UNDEFINED) == taskId:
            return slot
    return None


def finishSlot(slot: Any, entry: int | None) -> None:
    """Mark a slot done: the result entry, if any, now carries its running output, details, and diagnostics."""
    slot["status"] = "done"
    if entry is not None:
        slot["entry"] = entry
    clearProgress(slot)


def clearProgress(slot: Any) -> None:
    """Remove what a tool published while running; its result entry or a rerun replaces it."""
    for key in ("output", "droppedBytes", "droppedLines", "details", "diagnostics"):
        if key in slot:
            del slot[key]


async def settleSchedulerOutcome(tx: Any, record: dict[str, Any], outcome: dict[str, Any]) -> None:
    """Harness cleanup for a terminal outcome the scheduler writes itself (``faulted`` or ``orphaned``). A run task ends
    its run; a tool task's slot is marked done without an entry, and context derivation synthesizes the missing result;
    a compaction task's status is removed. Ignores other kinds so it never creates ``pi.live`` elsewhere. The scheduler
    calls this without knowing task kinds; the Harness passes it in (spec section 5.4).

    REMINDER: a committed generation partial becomes an aborted assistant entry here, exactly as in the generation
    abort handler, so the transcript keeps what the model produced and ``pi.usage`` counts its spend. The scheduler's
    commit has no task scope, so that entry has no ``byTaskId``. Faults come from task bugs or malformed provider data
    (a non-JSON value in a response), or a commit the Storage rejected without effect; an uncertain storage failure
    poisons the Session instead and writes no outcome."""
    # Imported here: the generation module imports this one.
    from .generation import convertPartial

    if record["kind"] == TOOL_TASK_KIND:
        slot = toolSlot(await tx.doc(LiveDoc, record["conversationId"]), record["id"])
        if slot is not None:
            finishSlot(slot, None)
        return
    if record["kind"] == COMPACTION_TASK_KIND:
        removeCompactionStatus(await tx.doc(LiveDoc, record["conversationId"]), record["id"])
        return
    if record["kind"] not in RUN_TASK_KINDS:
        return
    live = await tx.doc(LiveDoc, record["conversationId"])
    run = live.get("run", UNDEFINED)
    if run is UNDEFINED or run["taskId"] != record["id"]:
        return
    await convertPartial(tx, live, record["conversationId"])
    endRun(
        tx,
        live,
        record["id"],
        {"status": "unanswered", "reason": "faulted", "detail": outcome["error"]["message"]}
        if outcome["status"] == "faulted"
        else {"status": "unanswered", "reason": outcome["reason"]},
    )



__all__ = [
    "CompactionStatus",
    "LiveDoc",
    "ToolSlot",
    "addCompactionStatus",
    "clearProgress",
    "compactionStatus",
    "endRun",
    "finishSlot",
    "removeCompactionStatus",
    "settleSchedulerOutcome",
    "toolSlot",
]
