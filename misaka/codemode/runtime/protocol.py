"""Messages between the host and the worker that drives one VM. Tool arguments, results, and values
cross as JSON strings: the worker passes them into and out of the VM as strings and never builds
structured values itself.

Translated from pi's ``packages/codemode/src/runtime/protocol.ts``.

MISAKA fork: no compiled wasm module and no shared interrupt flag travel with the worker data. The
VM is a V8 isolate the worker owns in this process, and a runaway script is stopped by cancelling the
call that runs it (see ``engine.py``).
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from ..types import CodemodeOutputItem


class WorkerTool(TypedDict):
    name: str
    # The identifier the script uses.
    jsName: str
    # Listed in `ALL_TOOLS`.
    description: str


class WorkerGlobal(TypedDict):
    name: str
    spread: bool


class WorkerData(TypedDict):
    code: str
    tools: list[WorkerTool]
    globals: list[WorkerGlobal]
    memoryLimitBytes: int | None
    # Snapshot for `load()`: key to JSON text.
    store: dict[str, str]


# JSON-encoded `{ name?, message, stack? }` of an error thrown by the script.
type ScriptErrorJson = str


class CallMessage(TypedDict):
    type: Literal["call"]
    id: int
    target: Literal["tool", "global"]
    name: str
    args: str | None


class OutputMessage(TypedDict):
    type: Literal["output"]
    item: CodemodeOutputItem


class DoneOkMessage(TypedDict):
    type: Literal["done"]
    ok: Literal[True]
    value: str | None
    # A JSON array of `[key, json]` for `store()` and `[key]` for deletions.
    writes: str


class DoneErrorMessage(TypedDict):
    type: Literal["done"]
    ok: Literal[False]
    error: ScriptErrorJson


class CrashMessage(TypedDict):
    """The VM failed outside the script's control, for example running out of memory."""

    type: Literal["crash"]
    message: str


type WorkerToHostMessage = CallMessage | OutputMessage | DoneOkMessage | DoneErrorMessage | CrashMessage


class HostToWorkerMessage(TypedDict):
    type: Literal["result"]
    id: int
    ok: bool
    # The JSON result when `ok`, otherwise the error message.
    payload: str | None


def isWorkerToHostMessage(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    type = value.get("type")
    return type in ("call", "output", "done", "crash")


def isHostToWorkerMessage(value: Any) -> bool:
    return isinstance(value, dict) and value.get("type") == "result"


__all__ = [
    "CallMessage",
    "CrashMessage",
    "DoneErrorMessage",
    "DoneOkMessage",
    "HostToWorkerMessage",
    "OutputMessage",
    "ScriptErrorJson",
    "WorkerData",
    "WorkerGlobal",
    "WorkerToHostMessage",
    "WorkerTool",
    "isHostToWorkerMessage",
    "isWorkerToHostMessage",
]
