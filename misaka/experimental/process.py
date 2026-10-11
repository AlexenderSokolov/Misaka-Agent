"""Translated from pi's ``coding-agent/src/experimental/process.ts``.

MISAKA fork: pi launches its internal processes (coordinator, server, Session worker) through Node, from TypeScript source
or a bundle; here each is a Python module run with ``python -m``. ``spawnInternalProcess`` therefore takes ``entry``
(a module name) where upstream takes ``entryUrl``, and ``isDirectInternalProcessEntry`` is not needed: an entry module checks
``__name__ == "__main__"``. A child is an :class:`InternalProcess`, which offers the parts of Node's ``ChildProcess`` the
callers use: ``pid``, ``exitCode``, ``signalCode``, ``once("exit" | "error", listener)`` and ``kill(signalName)``.
"""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import sys
import threading
from collections.abc import Callable
from typing import Any, Literal

from misaka.utils.js import jsonStringify

INTERNAL_PROCESS_ENV = "__PI_INTERNAL_SPAWN"

type InternalProcessRole = Literal["coordinator", "server", "session-worker"]

_ROLES = ("coordinator", "server", "session-worker")
_DEFAULT_ENTRIES: dict[str, str] = {
    "coordinator": "misaka.experimental.coordinator",
    "server": "misaka.experimental.server",
    "session-worker": "misaka.experimental.session_worker",
}

MAX_CONTROL_LINE_BYTES = 128 * 1024 * 1024


class InternalProcess:
    """A spawned internal process. ``exitCode`` and ``signalCode`` are ``None`` while it runs; once it ended, one of them is
    set (the exit code, or the name of the signal that ended it)."""

    def __init__(self, popen: subprocess.Popen[bytes] | None, error: OSError | None, loop: asyncio.AbstractEventLoop) -> None:
        self._popen = popen
        self._loop = loop
        self.pid: int | None = None if popen is None else popen.pid
        self.exitCode: int | None = None
        self.signalCode: str | None = None
        self._listeners: dict[str, list[Callable[..., None]]] = {"exit": [], "error": []}
        if popen is not None:
            # The wait blocks, so a thread does it and reports back to the loop.
            threading.Thread(target=self._wait, args=(popen,), daemon=True).start()
        else:
            assert error is not None
            loop.call_soon(self._emit, "error", error)

    def once(self, event: str, listener: Callable[..., None]) -> None:
        self._listeners[event].append(listener)

    def unref(self) -> None:
        """Node's ``child.unref()``: the thread that waits is a daemon, so a child never keeps this process alive."""

    def kill(self, signalName: str = "SIGTERM") -> bool:
        if self._popen is None or self.exitCode is not None or self.signalCode is not None:
            return False
        try:
            if sys.platform == "win32":
                self._popen.kill()
            else:
                os.kill(self._popen.pid, getattr(signal, signalName))
        except ProcessLookupError:
            return False
        return True

    def _wait(self, popen: subprocess.Popen[bytes]) -> None:
        code = popen.wait()
        try:
            self._loop.call_soon_threadsafe(self._exited, code)
        except RuntimeError:
            # The loop is closed: nobody is left to tell.
            pass

    def _exited(self, code: int) -> None:
        if code < 0 and sys.platform != "win32":
            self.signalCode = signal.Signals(-code).name
        else:
            self.exitCode = code
        self._emit("exit", self.exitCode, self.signalCode)

    def _emit(self, event: str, *args: Any) -> None:
        listeners, self._listeners[event] = self._listeners[event], []
        for listener in listeners:
            listener(*args)


def getInternalProcessRole() -> str | None:
    """Read and validate an internal process role without consuming it."""
    role = os.environ.get(INTERNAL_PROCESS_ENV)
    if role is None:
        return None
    if role in _ROLES:
        return role
    raise RuntimeError(f"Unsupported internal process role: {role}")


def consumeInternalProcessRole() -> str | None:
    """Read, validate, and remove the role so descendants do not inherit it."""
    role = getInternalProcessRole()
    os.environ.pop(INTERNAL_PROCESS_ENV, None)
    return role


def spawnInternalProcess(role: str, args: list[str], options: dict[str, Any] | None = None) -> InternalProcess:
    """Spawn a detached Pi-owned process. ``options`` is ``{"entry"?: module name, "env"?: {...}}``."""
    options = options or {}
    entry = options.get("entry") or _DEFAULT_ENTRIES[role]
    environment = {**os.environ, **(options.get("env") or {}), INTERNAL_PROCESS_ENV: role}
    loop = asyncio.get_running_loop()
    arguments = [sys.executable, "-m", entry, *args]
    try:
        if sys.platform == "win32":
            popen = subprocess.Popen(
                arguments,
                cwd=os.getcwd(),
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW,
            )
        else:
            popen = subprocess.Popen(
                arguments,
                cwd=os.getcwd(),
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
    except OSError as error:
        # Node reports a failed spawn through the child's ``error`` event.
        return InternalProcess(None, error, loop)
    child = InternalProcess(popen, None, loop)
    child.unref()
    return child


async def terminateInternalProcess(child: InternalProcess) -> None:
    """Force a spawned internal process to exit and wait until it can no longer take ownership."""
    if child.pid is None or child.exitCode is not None or child.signalCode is not None:
        return
    terminated: asyncio.Future[None] = asyncio.get_running_loop().create_future()

    def done(*_: Any) -> None:
        if not terminated.done():
            terminated.set_result(None)

    child.once("exit", done)
    child.once("error", done)
    child.kill("SIGKILL")
    await terminated


def onceSignal(signalNumber: int, callback: Callable[[], None]) -> Callable[[], None]:
    """``process.once(signal, callback)`` (POSIX only): ``callback`` runs for the first delivery of the signal. The handler is
    then gone and the signal's default action applies again, so a second signal ends a process whose graceful shutdown is
    stuck. Returns the ``process.off`` that removes the handler."""
    loop = asyncio.get_running_loop()

    def off() -> None:
        loop.remove_signal_handler(signalNumber)

    def handler() -> None:
        off()
        # ``signal-exit``: the default action then ends the process after the held locks are removed.
        from misaka.utils.lockfile import hookSignals

        hookSignals()
        callback()

    loop.add_signal_handler(signalNumber, handler)  # windows-footgun: ok - only called on POSIX
    return off


def encodeControlLine(message: Any) -> str:
    line = f"{jsonStringify(message)}\n"
    if len(line.encode("utf-8", "surrogatepass")) > MAX_CONTROL_LINE_BYTES:
        raise RuntimeError("Internal control message is too large")
    return line
