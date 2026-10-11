"""Translated from pi's ``packages/codemode/src/runtime/host.ts``.

MISAKA fork: the worker is an in-process task around a V8 isolate (``worker.py``), so there is no
wasm module to load and no worker file to resolve; ``terminate()`` cancels the call that runs the
script. Results are plain dicts, as the TypeScript objects are.
"""

from __future__ import annotations

import asyncio
import inspect
import math
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from misaka.utils.js import (
    UNDEFINED,
    AbortController,
    jsonParse,
    jsonStringify,
    jsString,
    onAbort,
    spawn,
)
from misaka.utils.values import read_field

from ..identifier import toCodemodeIdentifier
from ..types import (
    CodemodeCall,
    CodemodeCallStatus,
    CodemodeError,
    CodemodeExecuteOptions,
    CodemodeOutputItem,
    CodemodeResult,
    CodemodeSandboxOptions,
    CodemodeStoreWrites,
    CodemodeTool,
)
from .protocol import (
    HostToWorkerMessage,
    WorkerData,
    isWorkerToHostMessage,
)
from .worker import Worker

DEFAULT_TIMEOUT_MS = 300_000
IDENTIFIER = re.compile(r"^[A-Za-z_$][A-Za-z0-9_$]*\Z")
RESERVED_GLOBALS: frozenset[str] = frozenset({
    "tools",
    "ALL_TOOLS",
    "console",
    "text",
    "image",
    "exit",
    "globalThis",
    "store",
    "load",
})


def _now() -> float:
    """``performance.now()``, in milliseconds."""
    return time.perf_counter() * 1000


def errorMessage(error: Any) -> str:
    if isinstance(error, BaseException):
        return read_field(error, "message", None) or str(error)
    return jsString(error)


def _stringify(value: Any) -> str | None:
    """``JSON.stringify(value)``: ``None`` where it gives ``undefined`` (``undefined`` itself and values
    that have no JSON)."""
    if value is UNDEFINED:
        return None
    try:
        return jsonStringify(value)
    except (TypeError, ValueError):
        return None


def serializeStore(store: Mapping[str, Any] | None) -> dict[str, str]:
    serialized: dict[str, str] = {}
    for key, value in (store or {}).items():
        json = _stringify(value)
        if json is not None:
            serialized[key] = json
    return serialized


class _BridgeError(Exception):
    """The prelude that serializes values for the host runs in the same VM as the script, so a script
    that patches built-ins (for example ``Array.prototype.toJSON``) can make it send malformed data.
    Payloads that do not decode raise this error, and the execution fails as a sandbox error."""


def _parseBridgeJson(json: str, what: str) -> Any:
    try:
        return jsonParse(json)
    except RecursionError:
        # MISAKA fork: Python's decoder gives up on values nested a few thousand levels deep, which V8's
        # decodes; that is the host's limit, not a broken bridge.
        raise
    except Exception:  # noqa: BLE001 - upstream `catch {}`
        raise _BridgeError(f"{what} is not valid JSON") from None


def parseStoreWrites(json: str) -> CodemodeStoreWrites:
    entries = _parseBridgeJson(json, "store writes")
    if not isinstance(entries, list):
        raise _BridgeError("store writes are not an array")
    writes: CodemodeStoreWrites = {"set": {}, "delete": []}
    for entry in entries:
        if (
            not isinstance(entry, list)
            or len(entry) == 0
            or not isinstance(entry[0], str)
            or (not isinstance(entry[1], str) if len(entry) == 2 else len(entry) != 1)
        ):
            raise _BridgeError("store writes contain a malformed entry")
        key = entry[0]
        if len(entry) == 1:
            writes["delete"].append(key)
        else:
            writes["set"][key] = _parseBridgeJson(entry[1], f"store value for {jsonStringify(key)}")
    return writes


def _parseScriptError(json: str) -> CodemodeError:
    parsed = _parseBridgeJson(json, "script error")
    # `typeof parsed !== "object"`: an array is an object too, and has none of the fields.
    if not isinstance(parsed, (dict, list)):
        raise _BridgeError("script error is not an object")
    fields = parsed if isinstance(parsed, dict) else {}
    name, message, stack = (fields.get(key, UNDEFINED) for key in ("name", "message", "stack"))
    if (
        not isinstance(message, str)
        or (name is not UNDEFINED and not isinstance(name, str))
        or (stack is not UNDEFINED and not isinstance(stack, str))
    ):
        raise _BridgeError("script error is malformed")
    error: CodemodeError = {"kind": "script", "message": message}
    if isinstance(name, str):
        error["name"] = name
    if isinstance(stack, str):
        error["stack"] = stack
    return error


@dataclass(slots=True)
class _PendingCall:
    record: CodemodeCall | None
    startedAt: float
    controller: AbortController


@dataclass(slots=True)
class _ExecutionOptions:
    code: str
    tools: Mapping[str, CodemodeTool]
    globals: Mapping[str, CodemodeTool]
    timeoutMs: float
    signal: Any
    memoryLimitBytes: int | None
    store: dict[str, str]


class Execution:
    """One script run in its own VM. A fresh VM per run keeps termination simple: a runaway script,
    including one that only spins the microtask queue, is killed with ``terminate()`` and cannot
    poison a later run."""

    def __init__(self, options: _ExecutionOptions) -> None:
        loop = asyncio.get_running_loop()
        self.promise: asyncio.Future[CodemodeResult] = loop.create_future()
        self._worker: Worker | None = None
        self._tools = options.tools
        self._globals = options.globals
        self._signal = options.signal
        self._timer: asyncio.TimerHandle | None = None
        self._removeAbortListener: Callable[[], None] = lambda: None
        self._output: list[CodemodeOutputItem] = []
        self._calls: list[CodemodeCall] = []
        self._pending: dict[int, _PendingCall] = {}
        self._finished = False
        self._tasks: set[asyncio.Task[Any]] = set()
        # MISAKA fork: cancelling the task that awaits the result cancels this future, which has no
        # counterpart for a promise. The run is then abandoned: stop it, and release the isolate.
        self.promise.add_done_callback(self._onSettled)

        if math.isfinite(options.timeoutMs):
            self._timer = loop.call_later(
                options.timeoutMs / 1000,
                lambda: self._finish({
                    "kind": "timeout",
                    "message": f"Execution timed out after {jsString(options.timeoutMs)} ms",
                }),
            )

        if options.signal is not None:
            if read_field(options.signal, "aborted"):
                self._onAbort()
            else:
                self._removeAbortListener = onAbort(options.signal, self._onAbort)

        self._start(options)

    def abort(self, message: str) -> asyncio.Future[CodemodeResult]:
        self._finish({"kind": "aborted", "message": message})
        return self.promise

    def _onSettled(self, future: asyncio.Future[CodemodeResult]) -> None:
        if future.cancelled():
            self._finish({"kind": "aborted", "message": "Execution cancelled"})

    def _start(self, options: _ExecutionOptions) -> None:
        if self._finished:
            return
        workerData: WorkerData = {
            "code": options.code,
            "tools": [
                {
                    "name": tool.name,
                    "jsName": toCodemodeIdentifier(tool.name),
                    "description": tool.description if tool.description is not None else "",
                }
                for tool in options.tools.values()
            ],
            "globals": [{"name": global_.name, "spread": global_.spread is True} for global_ in options.globals.values()],
            "memoryLimitBytes": options.memoryLimitBytes,
            "store": options.store,
        }
        try:
            worker = Worker(workerData, self._onMessage)
            worker.start()
        except Exception as error:  # noqa: BLE001 - upstream `catch (error)` around `new Worker`
            self._finish({"kind": "sandbox", "message": f"Failed to start worker: {errorMessage(error)}"})
            return
        self._worker = worker

    def _onAbort(self) -> None:
        reason = read_field(self._signal, "reason") if self._signal is not None else None
        self._finish({
            "kind": "aborted",
            "message": errorMessage(reason) if isinstance(reason, BaseException) else "Execution aborted",
        })

    def _post(self, message: HostToWorkerMessage) -> None:
        if self._worker is not None:
            self._worker.post(message)

    def _onMessage(self, message: Any) -> None:
        # An exception here would be lost in the worker's task and leave the execution unsettled.
        try:
            self._handleMessage(message)
        except Exception as error:  # noqa: BLE001 - upstream `catch (error)`: the run's failure
            self._finish({
                "kind": "sandbox",
                "message": (
                    f"Sandbox bridge broken: {error}. The script may have modified built-ins such as a prototype's toJSON."
                    if isinstance(error, _BridgeError)
                    else f"Sandbox host failed: {errorMessage(error)}"
                ),
            })

    def _handleMessage(self, message: Any) -> None:
        if self._finished:
            return
        if not isWorkerToHostMessage(message):
            raise _BridgeError("unknown message from the worker")
        match message["type"]:
            case "output":
                self._output.append(message["item"])
            case "call":
                if message["id"] in self._pending:
                    raise _BridgeError(f"duplicate call id {message['id']}")
                # Started eagerly, so the call is pending before the next message of the batch.
                spawn(self._handleCall(message), self._tasks)
            case "done":
                self._handleDone(message)
            case "crash":
                self._finish({"kind": "sandbox", "message": message["message"]})

    def _handleDone(self, message: Any) -> None:
        # Decode everything before finish(), which must not throw once it starts.
        if not message["ok"]:
            self._finish(_parseScriptError(message["error"]))
            return
        value = UNDEFINED if message["value"] is None else _parseBridgeJson(message["value"], "return value")
        self._finish(None, value, parseStoreWrites(message["writes"]))

    async def _handleCall(self, message: Any) -> None:
        id, name = message["id"], message["name"]
        isTool = message["target"] == "tool"
        record: CodemodeCall | None = {"name": name, "status": "cancelled", "durationMs": 0} if isTool else None
        if record is not None:
            self._calls.append(record)
        pending = _PendingCall(record=record, startedAt=_now(), controller=AbortController())
        self._pending[id] = pending

        status: CodemodeCallStatus
        reply: HostToWorkerMessage
        try:
            tool = (self._tools if isTool else self._globals).get(name)
            if tool is None:
                raise RuntimeError(f'Unknown {"tool" if isTool else "global"} "{name}"')
            # `undefined` is `UNDEFINED` and `null` is `None`, as everywhere in the port.
            args = UNDEFINED if message["args"] is None else jsonParse(message["args"])
            value = tool.execute(args, {"signal": pending.controller.signal})
            # `await tool.execute(...)`: anything awaitable is awaited.
            if inspect.isawaitable(value):
                value = await value
            reply = {
                "type": "result",
                "id": id,
                "ok": True,
                "payload": None if value is UNDEFINED else jsonStringify(value),
            }
            status = "ok"
        except Exception as error:  # noqa: BLE001 - the script sees the error as a rejection
            reply = {"type": "result", "id": id, "ok": False, "payload": errorMessage(error)}
            status = "error"
        except asyncio.CancelledError:
            # MISAKA fork: a tool that ends in a cancellation nobody asked this call for (it awaited a
            # future another component cancelled) rejects like any other failure; unanswered, the
            # script's promise would never settle. A cancellation of this task passes through.
            task = asyncio.current_task()
            if self._finished or (task is not None and task.cancelling() > 0):
                raise
            reply = {"type": "result", "id": id, "ok": False, "payload": "Tool call was cancelled"}
            status = "error"

        # Already cancelled by finish(): the record keeps "cancelled" and the worker is gone or going.
        if self._pending.pop(id, None) is None:
            return
        if record is not None:
            record["status"] = status
            record["durationMs"] = _now() - pending.startedAt
        self._post(reply)

    def _finish(
        self, error: CodemodeError | None, value: Any = UNDEFINED, writes: CodemodeStoreWrites | None = None
    ) -> None:
        if self._finished:
            return
        self._finished = True
        if self._timer is not None:
            self._timer.cancel()
        self._removeAbortListener()

        now = _now()
        for pending in self._pending.values():
            if pending.record is not None:
                pending.record["durationMs"] = now - pending.startedAt
            pending.controller.abort()
        self._pending.clear()

        result: CodemodeResult
        if error is not None:
            result = {"ok": False, "error": error, "output": self._output, "calls": self._calls}
        else:
            result = {
                "ok": True,
                "value": value,
                "output": self._output,
                "calls": self._calls,
                "storeWrites": {"set": {}, "delete": []} if writes is None else writes,
            }
        worker = self._worker
        if worker is None:
            self._resolveResult(result)
            return

        async def terminate() -> None:
            try:
                leftover = await worker.terminate()
                # MISAKA fork: what the script printed during the evaluation a timeout or an abort cut
                # off; upstream's bridge had posted it already. A script that finished or failed by
                # itself has had its output relayed in order.
                if error is not None and error["kind"] in ("timeout", "aborted"):
                    self._output.extend(leftover)
            except Exception:  # noqa: BLE001, S110 - upstream `.catch(() => undefined)`
                pass
            self._resolveResult(result)

        spawn(terminate(), self._tasks)

    def _resolveResult(self, result: CodemodeResult) -> None:
        if not self.promise.done():
            self.promise.set_result(result)


class CodemodeSandbox:
    """Runs JavaScript in a V8 isolate of its own. The script sees ``tools.<name>(args)`` for every
    registered tool, ``ALL_TOOLS``, the output helpers ``text``, ``image``, ``exit``, and
    ``console.*``, ``store``/``load``, and the configured globals; nothing else (no timers, ``fetch``,
    ``process``, ``require``, modules). The built-ins are frozen before the script runs.

    Each ``execute()`` gets its own VM; the sandbox only holds the tool table and defaults.
    ``close()`` aborts in-flight executions.
    """

    def __init__(self, options: CodemodeSandboxOptions | None = None) -> None:
        options = options or {}
        self._toolsByName: dict[str, CodemodeTool] = {}
        self._globalsByName: dict[str, CodemodeTool] = {}
        self._timeoutMs: float = options.get("timeoutMs", DEFAULT_TIMEOUT_MS)
        self._memoryLimitBytes = options.get("memoryLimitBytes")
        self._running: set[Execution] = set()
        self._closed = False
        for tool in options.get("tools") or []:
            self.registerTool(tool)
        namespaces: dict[str, None] = {}
        for global_ in options.get("globals") or []:
            parts = global_.name.split(".")
            if len(parts) > 2 or not all(IDENTIFIER.match(part) for part in parts) or parts[0] in RESERVED_GLOBALS:
                raise ValueError(f'Invalid global name "{global_.name}"')
            if global_.name in self._globalsByName:
                raise ValueError(f'Global "{global_.name}" is already registered')
            if len(parts) == 2:
                namespaces[parts[0]] = None
            self._globalsByName[global_.name] = global_
        for name in namespaces:
            if name in self._globalsByName:
                raise ValueError(f'Global "{name}" conflicts with the namespace "{name}"')

    def registerTool(self, tool: CodemodeTool) -> None:
        """Raises if a tool with the same name is already registered."""
        if tool.name in self._toolsByName:
            raise ValueError(f'Tool "{tool.name}" is already registered')
        self._toolsByName[tool.name] = tool

    def unregisterTool(self, name: str) -> bool:
        return self._toolsByName.pop(name, None) is not None

    @property
    def tools(self) -> list[CodemodeTool]:
        return list(self._toolsByName.values())

    @property
    def globals(self) -> list[CodemodeTool]:
        return list(self._globalsByName.values())

    def execute(self, code: str, options: CodemodeExecuteOptions | None = None) -> asyncio.Future[CodemodeResult]:
        """``code`` is an async function body: ``return`` and top-level ``await`` work. The execution
        starts at once and the returned future never fails for script failures; those come back as
        ``{"ok": False}``. The script can use ``store(key, value)`` and ``load(key)`` on
        ``options["store"]``. A script fails once its output passes ``MAX_OUTPUT_CHARS`` characters of
        text and base64 image data or ``MAX_OUTPUT_ITEMS`` items, even if it catches the error. The
        future fails once the sandbox is closed."""
        if self._closed:
            rejected: asyncio.Future[CodemodeResult] = asyncio.get_running_loop().create_future()
            rejected.set_exception(RuntimeError("Sandbox is closed"))
            return rejected
        options = options or {}
        execution = Execution(_ExecutionOptions(
            code=code,
            tools=dict(self._toolsByName),
            globals=self._globalsByName,
            timeoutMs=options.get("timeoutMs", self._timeoutMs),
            signal=options.get("signal"),
            memoryLimitBytes=self._memoryLimitBytes,
            store=serializeStore(options.get("store")),
        ))
        self._running.add(execution)
        execution.promise.add_done_callback(lambda _future: self._running.discard(execution))
        return execution.promise

    async def close(self) -> None:
        """Aborts in-flight executions (they resolve with ``kind: "aborted"``) and rejects new ones."""
        self._closed = True
        await asyncio.gather(*(execution.abort("Sandbox closed") for execution in list(self._running)))


__all__ = ["DEFAULT_TIMEOUT_MS", "CodemodeSandbox", "Execution"]
