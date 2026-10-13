"""Messages between the host (the event loop) and the worker. Tool arguments, results, and values cross as JSON
strings: the worker passes them into and out of the QuickJS VM as strings and never builds structured values itself.

Translated from pi's ``packages/codemode/src/runtime/protocol.ts``. MISAKA fork: the interrupt flag is a
``threading.Event`` where upstream shares an ``Int32Array`` over a ``SharedArrayBuffer``.
"""

from __future__ import annotations

import threading
from typing import Any, Literal, TypedDict

from ..types import CodemodeOutputItem
from ..wasm import CodemodeWasmModule


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
    # Compiled `quickjs.wasm`. The worker instantiates it; the compiled code is shared.
    wasm: CodemodeWasmModule
    memoryLimitBytes: int | None
    # Snapshot for `load()`: key to JSON text.
    store: dict[str, str]
    # Set by the host before terminating the worker. The VM's interrupt handler polls it, because a thread spinning
    # in wasm cannot be stopped from outside.
    interrupt: threading.Event


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
    """The VM failed outside the script's control, for example a wasm trap."""

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
