"""MISAKA fork: the JavaScript engine the sandbox runs scripts on.

pi runs QuickJS compiled to wasm (``quickjs-wasi``) inside a ``worker_threads`` worker; ``wasm.ts``
loads the module and ``worker.ts`` drives the VM. misaka runs V8 through ``mini-racer``, the one
Python engine package with wheels for every platform misaka supports (Windows x64 and ARM64, macOS
Intel and Apple silicon, glibc and musl Linux). Everything that differs between the two engines is
kept here, so ``worker.py`` and ``host.py`` can follow upstream:

- One V8 isolate per execution, created on the running event loop and closed with it.
- Script code only ever runs inside :meth:`Vm.run`, an awaited, cancellable evaluation; cancelling it
  terminates the script, which is what the shared interrupt flag did for QuickJS. mini-racer runs
  the microtasks a call leaves behind after the call has returned, on the isolate's own thread, so
  while a script may be running nothing else may touch the isolate synchronously: :meth:`Vm.evaluate`
  (synchronous) is only for our own definitions on the idle isolate. The script is compiled through
  :meth:`Vm.run` too, as its source can close the wrapper function and run code at the top level.
- The script is evaluated with ``//# sourceURL=codemode.js`` (QuickJS took a file name), and
  ``Error.prepareStackTrace`` renders stacks as QuickJS does, frames only and ``at fn (file:line)``,
  so the prelude's error text works unchanged; frames outside named sources are dropped.
- mini-racer installs timers and a console that QuickJS never had; they are removed before the
  prelude runs, with V8's ``WebAssembly`` global. The timer names stay as non-writable ``undefined``
  globals, since mini-racer's own are not configurable.
- QuickJS (``quickjs-wasi``) has ``atob``, ``btoa``, ``queueMicrotask``, ``performance``,
  ``DOMException`` and ``InternalError``, which V8 alone has not; the shim defines them to behave as
  QuickJS's do, messages included. V8's ``Intl``, ``Atomics`` and ``Temporal`` are left in place: a
  script that uses them works here and would fail upstream.
- Running out of memory terminates the script instead of throwing inside it (V8 has no catchable
  out-of-memory error); the worker reports it as a sandbox failure.
- The prelude freezes the built-ins before the script runs, ``Error.prepareStackTrace`` included, so
  a script cannot change how its stacks are rendered.
- :meth:`Vm.notifier` gives the VM a one-way call into Python that arrives while a script keeps the
  isolate busy, which upstream's worker gets from posting to its port: the worker uses it to end a
  run as soon as the prelude reports the script done, as upstream's host terminates the worker then.
  It relies on mini-racer's private ``_register_js_notification`` (the public ``wrap_py_function``
  converts its arguments by calling into the busy isolate and deadlocks), so mini-racer is pinned to
  0.14.x.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import Any

SCRIPT_FILE_NAME = "codemode.js"
PRELUDE_FILE_NAME = "codemode-prelude.js"

_SHIM_SOURCE = """(() => {
  for (const name of ["setTimeout", "clearTimeout", "setInterval", "clearInterval", "__timer_manager"]) {
    if (name in globalThis) Object.defineProperty(globalThis, name, { value: undefined, writable: false, configurable: false });
  }
  delete globalThis.console;
  delete globalThis.WebAssembly;
  const define = (name, value) =>
    Object.defineProperty(globalThis, name, { value, writable: true, enumerable: false, configurable: true });
  const codes = {
    IndexSizeError: 1, HierarchyRequestError: 3, WrongDocumentError: 4, InvalidCharacterError: 5,
    NoModificationAllowedError: 7, NotFoundError: 8, NotSupportedError: 9, InvalidStateError: 11, SyntaxError: 12,
    InvalidModificationError: 13, NamespaceError: 14, InvalidAccessError: 15, TypeMismatchError: 17,
    SecurityError: 18, NetworkError: 19, AbortError: 20, URLMismatchError: 21, QuotaExceededError: 22,
    TimeoutError: 23, InvalidNodeTypeError: 24, DataCloneError: 25,
  };
  class DOMException extends Error {
    #name;
    constructor(message = "", name = "Error") {
      super(String(message));
      this.#name = String(name);
    }
    get name() { return this.#name; }
    get code() { return codes[this.#name] ?? 0; }
  }
  Object.defineProperty(DOMException.prototype, Symbol.toStringTag, { value: "DOMException", configurable: true });
  define("DOMException", DOMException);
  class InternalError extends Error {}
  Object.defineProperty(InternalError.prototype, "name", { value: "InternalError", writable: true, configurable: true });
  define("InternalError", InternalError);
  const alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  define("btoa", function btoa(data) {
    const text = String(data);
    let out = "";
    for (let i = 0; i < text.length; i += 3) {
      const codes3 = [text.charCodeAt(i), text.charCodeAt(i + 1), text.charCodeAt(i + 2)];
      for (const code of codes3) {
        if (code > 0xff) throw new DOMException("String contains an invalid character", "InvalidCharacterError");
      }
      const [a, b, c] = codes3;
      out += alphabet[a >> 2] + alphabet[((a & 3) << 4) | ((b || 0) >> 4)];
      out += Number.isNaN(b) ? "=" : alphabet[((b & 15) << 2) | ((c || 0) >> 6)];
      out += Number.isNaN(c) ? "=" : alphabet[c & 63];
    }
    return out;
  });
  define("atob", function atob(data) {
    let text = String(data).replace(/[\\t\\n\\f\\r ]/g, "");
    if (text.length % 4 === 0) text = text.replace(/==?$/, "");
    if (text.length % 4 === 1 || /[^A-Za-z0-9+/]/.test(text)) {
      throw new DOMException("The string to be decoded is not correctly encoded", "InvalidCharacterError");
    }
    let out = "", bits = 0, buffer = 0;
    for (const char of text) {
      buffer = (buffer << 6) | alphabet.indexOf(char);
      bits += 6;
      if (bits >= 8) {
        bits -= 8;
        out += String.fromCharCode((buffer >> bits) & 0xff);
      }
    }
    return out;
  });
  define("queueMicrotask", function queueMicrotask(callback) {
    if (typeof callback !== "function") throw new TypeError("not a function");
    Promise.resolve().then(() => callback());
  });
  const timeOrigin = Date.now();
  define("performance", { now: () => Date.now() - timeOrigin, timeOrigin });
  Error.prepareStackTrace = (error, callSites) =>
    callSites
      .filter((site) => site.getScriptNameOrSourceURL())
      .map((site) => "    at " + (site.getFunctionName() || "<anonymous>") + " (" + site.getScriptNameOrSourceURL() + ":" + site.getLineNumber() + ")")
      .join("\\n");
})()"""

# `<anonymous>:2: SyntaxError: Unexpected token ';'`, then the source line and a caret.
_PARSE_ERROR_HEAD = re.compile(r"^[^\n]*?:(\d+): ([A-Za-z]*Error): ([^\n]*)")


class EngineUnavailableError(RuntimeError):
    """``mini-racer`` could not be imported or could not start an isolate."""


class OutOfMemoryError(RuntimeError):
    """The isolate hit its memory limit; V8 terminated the script."""


class ScriptSyntaxError(Exception):
    """A script (or the prelude) did not parse, or threw at the top level while it was compiled (its source
    can close the wrapper function). ``describe()`` is the `{ name, message, stack }` the worker reports, in
    the shape QuickJS reported it."""

    def __init__(self, name: str, message: str, line: int | None, fileName: str) -> None:
        super().__init__(f"{name}: {message}")
        self.name = name
        self.message = message
        self.line = line
        self.fileName = fileName

    def describe(self) -> dict[str, str]:
        head = f"{self.name}: {self.message}" if self.message else self.name
        stack = f"{head}\n    at {self.fileName}:{self.line}" if self.line is not None else head
        return {"name": self.name, "message": self.message, "stack": stack}


def _syntaxError(error: Exception, fileName: str) -> ScriptSyntaxError:
    match = _PARSE_ERROR_HEAD.match(str(error))
    if match is None:
        return ScriptSyntaxError("SyntaxError", str(error).strip(), None, fileName)
    return ScriptSyntaxError(match.group(2), match.group(3), int(match.group(1)), fileName)


def _module() -> Any:
    try:
        import py_mini_racer
    except ImportError as error:
        raise EngineUnavailableError(f"The codemode sandbox needs the mini-racer package: {error}") from error
    return py_mini_racer


class Vm:
    """One isolate. Create it on the running event loop and ``close()`` it when done."""

    def __init__(self, memoryLimitBytes: int | None) -> None:
        racer = _module()
        self._parseError = racer.JSParseException
        self._outOfMemory = racer.JSOOMException
        self._evalError = racer.JSEvalException
        # Termination and timeouts end a run; they are not the script's exception (`_exc` is where 0.14 keeps the first).
        self._stopped = (racer._exc.JSTerminatedException, racer.JSTimeoutException)
        try:
            self._racer = racer.MiniRacer()
        except Exception as error:
            raise EngineUnavailableError(f"Failed to start the JavaScript engine: {error}") from error
        if memoryLimitBytes is not None:
            self._racer.set_hard_memory_limit(memoryLimitBytes)
        self._racer.eval(_SHIM_SOURCE)

    def evaluate(self, source: str, fileName: str) -> Any:
        """Evaluate ``source`` on the idle isolate (a definition of ours; no script code may run in it)
        and return the value. A parse error raises :class:`ScriptSyntaxError`."""
        try:
            return self._racer.eval(f"{source}\n//# sourceURL={fileName}")
        except self._parseError as error:
            raise _syntaxError(error, fileName) from None

    async def run(self, code: str, fileName: str | None = None) -> Any:
        """Evaluate ``code``, which may run script code. Cancelling the awaiting task terminates the
        script; the result is the value of the last expression. With a ``fileName`` the code is a
        script's source, and a parse error or an exception it throws raises :class:`ScriptSyntaxError`."""
        try:
            return await self._racer.eval_cancelable(code if fileName is None else f"{code}\n//# sourceURL={fileName}")
        except self._outOfMemory as error:
            raise OutOfMemoryError(str(error)) from error
        except self._evalError as error:
            if fileName is None or isinstance(error, self._stopped):
                raise
            raise _syntaxError(error, fileName) from None

    def assertWithinMemory(self) -> None:
        """Raise :class:`OutOfMemoryError` when the isolate reached its memory limit. V8 terminates a script that
        does so while promise jobs run after an evaluation, and that evaluation still returns normally."""
        if self._racer.was_hard_memory_limit_reached():
            raise OutOfMemoryError("The isolate reached its memory limit")

    def notifier(self, callback: Callable[[], None]) -> AbstractContextManager[Any]:
        """A JavaScript function, for the duration of the ``with`` block, that calls ``callback`` on the
        event loop. The call returns in the VM at once and arrives also while a script keeps the isolate
        busy."""
        return self._racer._ctx._register_js_notification(lambda _handle: callback())

    def close(self) -> None:
        """Terminate whatever is running and release the isolate."""
        self._racer.close()


__all__ = [
    "PRELUDE_FILE_NAME",
    "SCRIPT_FILE_NAME",
    "EngineUnavailableError",
    "OutOfMemoryError",
    "ScriptSyntaxError",
    "Vm",
]
