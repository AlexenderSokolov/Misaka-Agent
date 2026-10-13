"""Worker thread entry. One worker runs one script inside a fresh QuickJS VM (a separate wasm instance), relays tool
calls and output to the host, and reports the result. The host terminates the worker when the script settles, times
out, or is aborted; the worker exists so that a spinning script never blocks the host thread.

Translated from pi's ``packages/codemode/src/runtime/worker.ts``.

MISAKA fork: Python has no ``worker_threads``. ``Worker`` stands in for Node's: it runs ``main`` on a thread of its
own, delivers what the thread posts to the host's event loop as "message", "error" and "exit" events, and queues the
host's messages for the thread, which handles them once ``main`` has returned, as a worker's event loop does. wasm runs
without the GIL, so a spinning script leaves the event loop free. ``terminate()`` cannot stop a thread: the host sets
the interrupt flag first (upstream does too, for Bun), and the thread then ends.
"""

from __future__ import annotations

import asyncio
import contextlib
import queue
import threading
from collections.abc import Callable
from typing import Any

from misaka.utils.js import jsonStringify, jsTrimEnd

from ..quickjs_wasi import MAX_STACK_SIZE, JSException, JSValueHandle, Memory, QuickJS
from ..types import CodemodeOutputItem
from .prelude_source import PRELUDE_SOURCE
from .protocol import WorkerData, WorkerToHostMessage, isHostToWorkerMessage

# Stack of a worker thread: the wasm stack limit (`quickjs_wasi.MAX_WASM_STACK`) and the host frames below and above
# it (host callbacks run on top of wasm frames). Node's workers default to 4 MB as well.
WORKER_STACK_BYTES = 4 * 1024 * 1024

_stackLock = threading.Lock()
_TERMINATE = object()


class _ParentPort:
    """``parentPort`` inside the worker thread."""

    def __init__(self, worker: Worker) -> None:
        self._worker = worker
        self.listener: Callable[[Any], None] | None = None
        # The worker's VM, which the thread disposes when it ends; upstream's ends with its worker.
        self.vm: QuickJS | None = None

    def postMessage(self, message: WorkerToHostMessage) -> None:
        self._worker._emit("message", message)

    def on(self, _event: str, listener: Callable[[Any], None]) -> None:
        self.listener = listener


def _errorText(error: BaseException) -> str:
    """``${error.name}: ${error.message}``. A wasm trap's text is its cause; the backtrace above it is wasmtime's."""
    name = error.name if isinstance(error, JSException) else type(error).__name__
    lines = [line.strip() for line in str(error).splitlines() if line.strip()]
    return f"{name}: {lines[-1] if lines else ''}"


def crash(parentPort: _ParentPort, error: BaseException) -> None:
    parentPort.postMessage({"type": "crash", "message": _errorText(error)})


def discardOutput(memory: Memory) -> dict[str, Callable[..., int]]:
    """QuickJS writes engine diagnostics to fd 1 and 2, which the default shim forwards to the host's stdout and
    stderr. That output belongs to the host application (for example a TUI), so it is discarded. Reporting every byte
    as written keeps libc from retrying."""

    def fd_write(_fd: int, iovsPtr: int, iovsLen: int, nwrittenPtr: int) -> int:
        written = 0
        for i in range(iovsLen):
            written += memory.getUint32(iovsPtr + i * 8 + 4)
        memory.setUint32(nwrittenPtr, written)
        return 0

    return {"fd_write": fd_write}


def describeException(error: JSException) -> str:
    head = f"{error.name}: {error.message}" if error.message else error.name
    stack = jsTrimEnd(error.stack) if error.stack is not None else None
    return jsonStringify({"name": error.name, "message": error.message, "stack": f"{head}\n{stack}" if stack else head})


def main(data: WorkerData, parentPort: _ParentPort) -> None:
    def post(message: WorkerToHostMessage) -> None:
        parentPort.postMessage(message)

    interrupt = data["interrupt"]
    vm = QuickJS.create(
        wasm=data["wasm"],
        memoryLimit=data["memoryLimitBytes"],
        # Without a guard, deep recursion overflows the wasm stack and traps instead of throwing a
        # catchable RangeError.
        maxStackSize=MAX_STACK_SIZE,
        interruptHandler=interrupt.is_set,
        wasi=discardOutput,
    )
    parentPort.vm = vm

    # Called from the prelude with primitives only. A missing argument is `None` (`undefined` upstream).
    # MISAKA fork: each argument is read out of the VM once; a read costs a few calls into wasm.
    def bridgeCallback(kind: JSValueHandle, a: Any = None, b: Any = None, c: Any = None) -> JSValueHandle:
        match kindText := kind.toString():
            case "call" | "global":
                post({
                    "type": "call",
                    "id": int(a.toNumber()),
                    "target": "tool" if kindText == "call" else "global",
                    "name": b.toString(),
                    "args": None if c is None or c.isUndefined else c.toString(),
                })
            case "output":
                match a.toString():
                    case "image":
                        item: CodemodeOutputItem = {"type": "image", "data": b.toString(), "mimeType": c.toString()}
                    case "console":
                        item = {"type": "text", "text": b.toString(), "console": True}
                    case _:
                        item = {"type": "text", "text": b.toString()}
                post({"type": "output", "item": item})
            case "done":
                if a.toBoolean():
                    post({
                        "type": "done",
                        "ok": True,
                        "value": None if b is None or b.isUndefined else b.toString(),
                        "writes": c.toString(),
                    })
                else:
                    post({"type": "done", "ok": False, "error": b.toString()})
        return vm.undefined

    bridge = vm.newFunction("bridge", bridgeCallback)

    # The VM lives until the host terminates the worker, so these handles are never disposed.
    api = vm.withScope(
        lambda scope: scope.escape(
            vm.callFunction(
                vm.evalCode(PRELUDE_SOURCE, "codemode-prelude.js"),
                vm.undefined,
                bridge,
                vm.newString(jsonStringify(data["tools"])),
                vm.newString(jsonStringify(data["globals"])),
                vm.newString(jsonStringify(data["store"])),
            )
        )
    )
    settle = api.getProp("settle")
    run = api.getProp("run")
    stalled = api.getProp("stalled")

    def drain() -> None:
        """Run queued jobs, then fail a script that waits on nothing that can ever resume it."""
        vm.executePendingJobs()
        vm.callFunction(stalled, api).dispose()

    def onMessage(message: Any) -> None:
        if not isHostToWorkerMessage(message):
            return
        try:
            vm.withScope(
                lambda _scope: vm.callFunction(
                    settle,
                    api,
                    vm.newNumber(message["id"]),
                    vm.true if message["ok"] else vm.false,
                    vm.undefined if message["payload"] is None else vm.newString(message["payload"]),
                )
            )
            drain()
        except Exception as error:  # noqa: BLE001 - upstream `catch (error) { crash(error) }`
            crash(parentPort, error)

    parentPort.on("message", onMessage)

    # The prefix shares the first line with the script so reported line numbers
    # match the script as written.
    try:
        fn = vm.evalCode(f"(async (tools, console) => {{{data['code']}\n}})", "codemode.js")
    except JSException as error:
        post({"type": "done", "ok": False, "error": describeException(error)})
        return
    vm.callFunction(run, api, fn).dispose()
    fn.dispose()
    drain()


class Worker:
    """Node's ``worker_threads.Worker`` running ``main`` with ``workerData``. Construct it on the event loop; listeners
    added with ``on()`` right after construction see every event."""

    def __init__(self, workerData: WorkerData) -> None:
        self._loop = asyncio.get_running_loop()
        self._listeners: dict[str, list[Callable[[Any], None]]] = {}
        self._inbox: queue.SimpleQueue[Any] = queue.SimpleQueue()
        self._outbox: list[tuple[str, Any]] = []
        self._outboxLock = threading.Lock()
        self._flushScheduled = False
        self._exited: asyncio.Future[int] = self._loop.create_future()
        self._thread = threading.Thread(target=self._run, args=(workerData,), name="codemode-worker", daemon=True)
        # `threading.stack_size()` applies to every thread started after it, so it is set only around this start.
        with _stackLock:
            previous = threading.stack_size()
            try:
                threading.stack_size(max(previous, WORKER_STACK_BYTES))
            except (ValueError, RuntimeError):  # a platform that cannot size thread stacks keeps its default
                previous = None
            try:
                self._thread.start()
            finally:
                if previous is not None:
                    threading.stack_size(previous)

    def on(self, event: str, listener: Callable[[Any], None]) -> None:
        self._listeners.setdefault(event, []).append(listener)

    def postMessage(self, message: Any) -> None:
        self._inbox.put(message)

    async def terminate(self) -> int:
        """Ends the worker once it is idle; resolves with its exit code. The host sets the interrupt flag first."""
        self._inbox.put(_TERMINATE)
        return await asyncio.shield(self._exited)

    def _emit(self, event: str, payload: Any) -> None:
        """From the worker thread: deliver ``event`` on the host's event loop. Events queue in order and the loop
        takes them in batches, woken once per batch rather than once per event."""
        with self._outboxLock:
            self._outbox.append((event, payload))
            if self._flushScheduled:
                return
            self._flushScheduled = True
        with contextlib.suppress(RuntimeError):  # the loop is closed: nobody is listening any more
            self._loop.call_soon_threadsafe(self._flush)

    def _flush(self) -> None:
        with self._outboxLock:
            events, self._outbox = self._outbox, []
            self._flushScheduled = False
        for event, payload in events:
            if event == "exit" and not self._exited.done():
                self._exited.set_result(payload)
            for listener in list(self._listeners.get(event, ())):
                listener(payload)

    def _run(self, workerData: WorkerData) -> None:
        parentPort = _ParentPort(self)
        try:
            try:
                main(workerData, parentPort)
            except Exception as error:  # noqa: BLE001 - upstream `main(workerData).catch(crash)`
                crash(parentPort, error)
            while (message := self._inbox.get()) is not _TERMINATE:
                if parentPort.listener is not None:
                    parentPort.listener(message)
        except Exception as error:  # noqa: BLE001 - the worker's "error" event
            self._emit("error", error)
        finally:
            parentPort.listener = None
            if parentPort.vm is not None:
                parentPort.vm.dispose()
            # A terminated worker exits with code 1, as Node's does.
            self._emit("exit", 1)


__all__ = ["WORKER_STACK_BYTES", "Worker", "describeException", "discardOutput", "main"]
