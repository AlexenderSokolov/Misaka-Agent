"""Keyed instance lifetime and the cancellable tasks observing those instances.

Translated from pi's ``packages/chord/src/services/instances.ts``. An observer's handler may be a plain
function or a coroutine function; a returned awaitable is awaited on the running event loop.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable, Iterator
from typing import Any, Protocol, TypedDict

from misaka.chord.context import BACKGROUND_CONTEXT, withCancel
from misaka.chord.types import Context
from misaka.utils.js import liveValues, startTask


class InstanceDirectoryEntry(Protocol):
    @property
    def key(self) -> str: ...

    @property
    def generation(self) -> int: ...

    @property
    def service(self) -> Any: ...

    def deactivate(self) -> None: ...


class _Observer:
    __slots__ = ("closed", "handler", "tasks")

    def __init__(self, handler: Callable[[Any, Context], Any]) -> None:
        self.handler = handler
        self.tasks: dict[Any, Callable[..., None]] = {}
        self.closed = False


class InstanceDirectoryOptions(TypedDict):
    """The options object ``InstanceDirectory`` takes."""

    ready: bool
    onError: Callable[[Exception], None]


class InstanceDirectory[TEntry: InstanceDirectoryEntry]:
    """Owns keyed instance lifetime and the cancellable tasks observing those instances."""

    def __init__(self, options: InstanceDirectoryOptions) -> None:
        self._entries: dict[str, TEntry] = {}
        self._observers: list[_Observer] = []
        self._ready = options["ready"]
        self._reportError = options["onError"]
        self._disposed = False

    @property
    def observerCount(self) -> int:
        return len(self._observers)

    def values(self) -> Iterator[TEntry]:
        return iter(self._entries.values())

    def get(self, key: str) -> TEntry | None:
        return self._entries.get(key)

    def insert(self, entry: TEntry) -> None:
        self._assertActive()
        if entry.key in self._entries:
            raise RuntimeError(f"Keyed service already has a live instance with key {entry.key}")
        self._entries[entry.key] = entry
        if self._ready:
            self._startAll(entry)

    def replace(self, entry: TEntry) -> None:
        self._assertActive()
        previous = self._entries.get(entry.key)
        if previous is not None:
            if previous.generation == entry.generation:
                raise RuntimeError("Keyed service repeated a live generation")
            self._remove(previous)
        self._entries[entry.key] = entry
        if self._ready:
            self._startAll(entry)

    def remove(self, entry: TEntry) -> None:
        if self._entries.get(entry.key) is not entry:
            return
        self._remove(entry)

    def ready(self) -> None:
        self._assertActive()
        if self._ready:
            return
        self._ready = True
        for entry in liveValues(self._entries):
            self._startAll(entry)

    def reset(self) -> None:
        if self._disposed:
            return
        self._ready = False
        for entry in liveValues(self._entries):
            self._remove(entry)

    def observe(self, handler: Callable[[Any, Context], Any]) -> Callable[[], None]:
        self._assertActive()
        observer = _Observer(handler)
        self._observers.append(observer)
        if self._ready:
            for entry in liveValues(self._entries):
                self._start(observer, entry)

        def stop() -> None:
            if observer.closed:
                return
            observer.closed = True
            for cancel in list(observer.tasks.values()):
                cancel()
            observer.tasks.clear()
            if observer in self._observers:
                self._observers.remove(observer)

        return stop

    def dispose(self) -> None:
        if self._disposed:
            return
        self._disposed = True
        for observer in self._observers:
            observer.closed = True
            for cancel in list(observer.tasks.values()):
                cancel()
            observer.tasks.clear()
        self._observers.clear()
        for entry in list(self._entries.values()):
            entry.deactivate()
        self._entries.clear()

    def _remove(self, entry: TEntry) -> None:
        if self._entries.get(entry.key) is not entry:
            return
        del self._entries[entry.key]
        entry.deactivate()
        for observer in list(self._observers):
            cancel = observer.tasks.pop(entry, None)
            if cancel is not None:
                cancel()

    def _startAll(self, entry: TEntry) -> None:
        for observer in list(self._observers):
            self._start(observer, entry)

    def _start(self, observer: _Observer, entry: TEntry) -> None:
        if observer.closed or entry in observer.tasks:
            return
        cancelable = withCancel(BACKGROUND_CONTEXT)
        context = cancelable.context
        observer.tasks[entry] = cancelable.cancel

        def report(error: BaseException) -> None:
            signal = context.abortSignal
            if not (signal is not None and signal.aborted):
                self._reportError(toError(error))

        try:
            result = observer.handler(entry.service, context)
            if inspect.isawaitable(result):
                task = startTask(result)

                def settled(done: asyncio.Future[Any]) -> None:
                    if not done.cancelled() and (error := done.exception()) is not None:
                        report(error)

                task.add_done_callback(settled)
        except Exception as error:  # noqa: BLE001 - a handler's failure is reported unless the observation was cancelled
            report(error)

    def _assertActive(self) -> None:
        if self._disposed:
            raise RuntimeError("Keyed service directory is disposed")


def toError(error: BaseException) -> Exception:
    return error if isinstance(error, Exception) else Exception(str(error))


__all__ = ["InstanceDirectory", "InstanceDirectoryEntry"]
