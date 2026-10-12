"""Translated from pi's ``packages/codemode/src/types.ts``.

MISAKA fork: scripts run on V8 through ``mini-racer`` rather than QuickJS in a wasm worker, so
the ``wasm`` and ``workerUrl`` sandbox options have no counterpart.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, NotRequired, TypedDict


class CodemodeToolContext(TypedDict):
    # Aborted when the script finishes (including unawaited calls), the execution times out, the
    # caller aborts, or the sandbox is closed.
    signal: Any


# A JSON Schema document. Only used to render declarations; values are not validated against it.
type CodemodeJsonSchema = dict[str, Any] | bool


# MISAKA fork: keyword-only, so the members keep upstream's order with the required `execute` last.
@dataclass(slots=True, kw_only=True)
class CodemodeTool:
    # The script calls tools as `tools.<id>(args)`, where `<id>` is the name with characters that
    # are not valid in identifiers replaced by `_` (see `toCodemodeIdentifier`), and also as
    # `tools["<name>"](args)`. Globals are called as `<name>(args)` and must be identifiers, or
    # `<namespace>.<member>`, which groups them into a frozen namespace object.
    name: str
    # Shown as a doc comment in `renderDeclarations`, and listed in `ALL_TOOLS` for tools.
    description: str | None = None
    # Schema of the single argument. Rendered as the parameter type; `unknown` when omitted.
    inputSchema: CodemodeJsonSchema | None = None
    # Schema of the resolved value. Rendered as the promise type; `unknown` when omitted.
    outputSchema: CodemodeJsonSchema | None = None
    # Globals only: `execute` receives all call arguments as a list instead of the first one.
    spread: bool | None = None
    # Globals only: TypeScript parameter list and return type for `renderDeclarations`, for
    # example `(type: string, id?: string): Promise<Model[]>`. Replaces the rendering from the
    # schemas.
    signature: str | None = None
    # `args` is whatever the script passed, after a JSON round trip. The return value must be
    # JSON-serializable; a raised error surfaces in the script as an `Error` with the same
    # message. May return an awaitable.
    execute: Callable[[Any, CodemodeToolContext], Any]


class CodemodeTextOutputItem(TypedDict):
    type: Literal["text"]
    text: str
    console: NotRequired[Literal[True]]


class CodemodeImageOutputItem(TypedDict):
    type: Literal["image"]
    # base64
    data: str
    mimeType: str


# One item of the script's output, in the order the script produced it: `text()` and `console.*`
# produce text items, with `console: True` for `console.*`, and `image()` image items. `data` is base64.
type CodemodeOutputItem = CodemodeTextOutputItem | CodemodeImageOutputItem

type CodemodeCallStatus = Literal["ok", "error", "cancelled"]


class CodemodeCall(TypedDict):
    name: str
    status: CodemodeCallStatus
    durationMs: float


# "script": the script threw or failed to parse; `name` and `stack` come from the script's error.
# "timeout": the overall deadline expired; the VM was terminated.
# "aborted": the caller's signal fired or the sandbox was closed; the VM was terminated.
# "sandbox": the VM failed outside the script's control (for example out of memory).
type CodemodeErrorKind = Literal["script", "timeout", "aborted", "sandbox"]


class CodemodeError(TypedDict):
    kind: CodemodeErrorKind
    name: NotRequired[str]
    message: str
    stack: NotRequired[str]


class CodemodeStoreWrites(TypedDict):
    """Keys the script changed with ``store()``. Only successful executions report writes."""

    set: dict[str, Any]
    # Keys stored as `undefined`.
    delete: list[str]


class CodemodeSuccess(TypedDict):
    ok: Literal[True]
    # `exit()` completes with `value: UNDEFINED`.
    value: Any
    output: list[CodemodeOutputItem]
    calls: list[CodemodeCall]
    storeWrites: CodemodeStoreWrites


class CodemodeFailure(TypedDict):
    ok: Literal[False]
    error: CodemodeError
    # Kept for failed executions too, up to the failure.
    output: list[CodemodeOutputItem]
    calls: list[CodemodeCall]


type CodemodeResult = CodemodeSuccess | CodemodeFailure


class CodemodeSandboxOptions(TypedDict, total=False):
    tools: Sequence[CodemodeTool]
    # Functions exposed as top-level identifiers instead of on `tools`, for host helpers such as
    # attaching an image to the result. They behave like tools (JSON round trip, promise result)
    # but are not recorded in `result.calls`. Names must be identifiers and may not shadow the
    # built-in globals (`tools`, `ALL_TOOLS`, `console`, `text`, `image`, `exit`, `store`, `load`).
    globals: Sequence[CodemodeTool]
    # Overall deadline per execution, including time spent in tools. `math.inf` disables the
    # deadline; the execution then only ends when the script settles or is aborted.
    # Default: 300000.
    timeoutMs: float
    # Maximum memory the VM may allocate. Default: no limit beyond the engine's own.
    memoryLimitBytes: int


class CodemodeExecuteOptions(TypedDict, total=False):
    signal: Any
    # Overrides the sandbox default for this execution.
    timeoutMs: float
    # Values the script reads with `load(key)`. Must be JSON-serializable. The script's own
    # `store()` calls come back as `result["storeWrites"]`; persisting them is up to the caller.
    store: Mapping[str, Any]


__all__ = [
    "CodemodeCall",
    "CodemodeCallStatus",
    "CodemodeError",
    "CodemodeErrorKind",
    "CodemodeExecuteOptions",
    "CodemodeFailure",
    "CodemodeImageOutputItem",
    "CodemodeJsonSchema",
    "CodemodeOutputItem",
    "CodemodeResult",
    "CodemodeSandboxOptions",
    "CodemodeStoreWrites",
    "CodemodeSuccess",
    "CodemodeTextOutputItem",
    "CodemodeTool",
    "CodemodeToolContext",
]
