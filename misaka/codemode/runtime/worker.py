"""Drives one script inside a fresh VM, relays tool calls and output to the host, and reports the
result. The host terminates the worker when the script settles, times out, or is aborted.

Translated from pi's ``packages/codemode/src/runtime/worker.ts``.

MISAKA fork: upstream's worker is a thread with its own message port and a QuickJS instance whose
``bridge`` host function posts to the main thread as the script runs. Here the worker is a task on
the host's event loop that owns a V8 isolate (``engine.py``). The prelude's ``bridge`` queues
``[kind, a, b, c]`` tuples inside the VM, and the worker takes the queue after every call into the
VM, so no callback ever runs while the script is spinning and every call stays cancellable. Messages
keep the order the script produced them, as with the port. A "done" also notifies the worker, which
then ends the run at once, as upstream's host terminates the worker when the script reports done,
even if the script keeps spinning (for example in a loop that catches the output limit's error). Code
after an ``await`` runs outside any evaluation and cannot be stopped that way; a script that spins
there after reporting done ends as a sandbox failure (see ``_onDone``).

The bridge and the prelude's API live in the closure of a host object under a non-enumerable
global, which the worker's own evaluations call. The object is reachable from scripts, as every
global is, and the prelude freezes it with the other built-ins; what it offers a script is only
what the script could do anyway: answer its own calls, or end itself.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable
from typing import Any

from misaka.utils.js import jsonParse, jsonStringify, toWellFormed

from ..types import CodemodeOutputItem
from . import engine
from .prelude_source import PRELUDE_SOURCE
from .protocol import (
    HostToWorkerMessage,
    WorkerData,
    WorkerToHostMessage,
    isHostToWorkerMessage,
)

HOST = "__misaka_codemode__"

_HOST_SOURCE = f"""((notify) => {{
  const queue = [];
  let api;
  let fn;
  let installed = false;
  let ran = false;
  const bridge = (kind, a, b, c) => {{
    queue.push([kind, a, b, c]);
    if (kind === "done") notify();
  }};
  const host = {{
    take() {{
      return JSON.stringify(queue.splice(0));
    }},
    install(prelude, toolsJson, globalsJson, storeJson) {{
      if (installed) return;
      installed = true;
      api = prelude(bridge, toolsJson, globalsJson, storeJson);
    }},
    compile(script) {{
      fn = script;
    }},
    run() {{
      if (ran) return;
      ran = true;
      return api.run(fn);
    }},
    settle(id, ok, payload) {{
      return api.settle(id, ok, payload);
    }},
    stalled() {{
      return api.stalled();
    }},
  }};
  Object.defineProperty(globalThis, "{HOST}", {{ value: host, enumerable: false, writable: false, configurable: false }});
}})"""


# How long `terminate()` waits for the output a stopped script left in its queue. Short: upstream's host has the
# output already and resolves at once; an isolate still running code after an `await` never answers.
_QUEUED_OUTPUT_TIMEOUT_SECONDS = 0.25
# How long a script that reported done may keep the isolate busy before the run is stopped (see `_onDone`).
_DONE_GRACE_SECONDS = 0.25


def _hostCode(expression: str) -> str:
    # Named like the prelude, so its frames leave the script's stacks with the prelude's.
    return f"{expression}\n//# sourceURL={engine.PRELUDE_FILE_NAME}"


def _outputItem(kind: Any, data: Any, mimeType: Any) -> CodemodeOutputItem:
    if kind == "image":
        return {"type": "image", "data": str(data), "mimeType": str(mimeType)}
    if kind == "console":
        return {"type": "text", "text": str(data), "console": True}
    return {"type": "text", "text": str(data)}


class Worker:
    def __init__(self, data: WorkerData, post: Callable[[WorkerToHostMessage], None]) -> None:
        self._data = data
        self._post = post
        self._inbox: asyncio.Queue[HostToWorkerMessage] = asyncio.Queue()
        self._vm: engine.Vm | None = None
        self._task: asyncio.Task[None] | None = None
        # The evaluation that may run script code, which a "done" ends.
        self._running: asyncio.Future[Any] | None = None
        self._doneRelayed = False

    def start(self) -> None:
        self._task = asyncio.get_running_loop().create_task(self._main())

    def post(self, message: HostToWorkerMessage) -> None:
        self._inbox.put_nowait(message)

    async def terminate(self) -> list[CodemodeOutputItem]:
        """Stop the script wherever it is and release the isolate. Answers the output the script produced
        since the queue was last relayed, which the host keeps when it ends a run (a timeout or an
        abort); upstream's bridge posts each item as the script makes it."""
        task = self._task
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        output: list[CodemodeOutputItem] = []
        vm = self._vm
        if vm is not None:
            # Bounded, as a script that keeps the microtask queue busy would otherwise hold this call.
            with contextlib.suppress(Exception):
                output = await asyncio.wait_for(self._queuedOutput(vm), _QUEUED_OUTPUT_TIMEOUT_SECONDS)
            vm.close()
            self._vm = None
        return output

    async def _queuedOutput(self, vm: engine.Vm) -> list[CodemodeOutputItem]:
        queued = jsonParse(await vm.run(_hostCode(f"{HOST}.take()")))
        return [_outputItem(a, b, c) for kind, a, b, c in queued if kind == "output"]

    def _crash(self, error: BaseException) -> None:
        self._post({"type": "crash", "message": f"{type(error).__name__}: {error}"})

    async def _main(self) -> None:
        try:
            await self._run()
        except asyncio.CancelledError:
            raise
        except engine.OutOfMemoryError:
            vm = self._vm
            if vm is not None:
                with contextlib.suppress(Exception):
                    for item in await asyncio.wait_for(self._queuedOutput(vm), _QUEUED_OUTPUT_TIMEOUT_SECONDS):
                        self._post({"type": "output", "item": item})
            self._post({"type": "crash", "message": "out of memory"})
        except Exception as error:  # noqa: BLE001 - upstream `main(workerData).catch(crash)`
            self._crash(error)

    def _onDone(self) -> None:
        """The prelude reported the script done; a script still spinning is stopped here."""
        if self._running is not None:
            self._running.cancel()
            return
        # MISAKA fork: code after an `await` runs once the evaluation that resumed it has returned, outside any
        # evaluation, and V8 stops it only with the isolate. A done that is not relayed shortly means the script
        # keeps spinning past it (upstream's host terminates the worker then); the run ends as a sandbox failure,
        # since what it reported is still in the busy isolate.
        asyncio.get_running_loop().call_later(_DONE_GRACE_SECONDS, self._stopSpinning)

    def _stopSpinning(self) -> None:
        if not self._doneRelayed:
            self._post({"type": "crash", "message": "the script kept running after it finished"})

    async def _runScript(self, vm: engine.Vm, code: str, fileName: str | None = None) -> None:
        """Run an evaluation that may run script code. A "done" cancels it, which terminates the script
        wherever it is; the worker then relays the queue as after any call."""
        running = self._running = asyncio.ensure_future(vm.run(code, fileName))
        try:
            await running
        except asyncio.CancelledError:
            current = asyncio.current_task()
            if not running.cancelled() or (current is not None and current.cancelling() > 0):
                raise
        finally:
            self._running = None

    async def _run(self) -> None:
        data = self._data
        vm = self._vm = engine.Vm(data["memoryLimitBytes"])
        with vm.notifier(self._onDone) as notify:
            vm.evaluate(_HOST_SOURCE, engine.PRELUDE_FILE_NAME)(notify)
            await self._runVm(vm)

    async def _runVm(self, vm: engine.Vm) -> None:
        data = self._data
        # The prelude receives the bridge and builds the script's globals; it runs no script code.
        await vm.run(
            f"{HOST}.install(({PRELUDE_SOURCE}), {jsonStringify(jsonStringify(data['tools']))}, "
            f"{jsonStringify(jsonStringify(data['globals']))}, {jsonStringify(jsonStringify(data['store']))})"
            f"\n//# sourceURL={engine.PRELUDE_FILE_NAME}"
        )

        async def pump() -> None:
            for kind, a, b, c in jsonParse(await vm.run(_hostCode(f"{HOST}.take()"))):
                self._bridge(kind, a, b, c)

        async def drain() -> None:
            """Relay what the script produced, then fail a script that waits on nothing that can ever
            resume it."""
            await pump()
            # A script V8 terminated for memory waits on nothing either; report what ended it.
            vm.assertWithinMemory()
            await vm.run(_hostCode(f"{HOST}.stalled()"))
            await pump()

        # The prefix shares the first line with the script so reported line numbers match the script
        # as written. Passed as an argument, so V8 does not name the function after a property.
        # MISAKA fork: compiled through `vm.run`, as a script can close the wrapper and run code at the
        # top level of this evaluation, which must stay cancellable; a lone surrogate in the code
        # becomes U+FFFD, as upstream's UTF-8 encoding of the code does.
        try:
            await self._runScript(
                vm,
                f"{HOST}.compile(async (tools, console) => {{{toWellFormed(data['code'])}\n}})",
                engine.SCRIPT_FILE_NAME,
            )
        except engine.ScriptSyntaxError as error:
            self._post({"type": "done", "ok": False, "error": jsonStringify(error.describe())})
            return
        await self._runScript(vm, _hostCode(f"{HOST}.run()"))
        await drain()

        while True:
            message = await self._inbox.get()
            if not isHostToWorkerMessage(message):
                continue
            payload = message["payload"]
            await self._runScript(vm, _hostCode(
                f"{HOST}.settle({jsonStringify(message['id'])}, {'true' if message['ok'] else 'false'}, "
                f"{'undefined' if payload is None else jsonStringify(payload)})"
            ))
            await drain()

    def _bridge(self, kind: str, a: Any, b: Any, c: Any) -> None:
        """One queued ``bridge(kind, a, b, c)`` call; `null` stands for `undefined`."""
        match kind:
            case "call" | "global":
                self._post({
                    "type": "call",
                    "id": int(a),
                    "target": "tool" if kind == "call" else "global",
                    "name": str(b),
                    "args": None if c is None else str(c),
                })
            case "output":
                self._post({"type": "output", "item": _outputItem(a, b, c)})
            case "done":
                self._doneRelayed = True
                if a:
                    self._post({"type": "done", "ok": True, "value": None if b is None else str(b), "writes": str(c)})
                else:
                    self._post({"type": "done", "ok": False, "error": str(b)})


__all__ = ["HOST", "Worker"]
