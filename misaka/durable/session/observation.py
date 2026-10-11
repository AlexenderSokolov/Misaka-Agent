"""Session-to-Chord bridges: a replicated-state source and a serialized watch.

Translated from pi's ``packages/durable/src/session/observation.ts``. ``queueMicrotask`` is ``loop.call_soon`` and
``void asyncMethod()`` is a task the watch keeps alive until it finishes.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from misaka.chord.context import Context, withoutAbortSignal
from misaka.chord.delta import Op
from misaka.utils.js import queueMicrotask, spawn

type ObservedDocumentValue = dict[str, Any] | None

MAX_PENDING_WATCH_FRAMES = 100
"""Maximum exact committed frames retained behind one unavailable watch listener."""

RETIREMENT_OPERATIONS: Sequence[Op] = (["r", None],)
"""Canonical terminal update for a retired document incarnation."""


class CommittedStateSource:
    """Session-to-Chord bridge owned one-to-one by one attached state: a document, or a conversation view. A ``None``
    value retires it."""

    def __init__(self, value: Any, release: Callable[[], None]) -> None:
        self._attachments: set[SessionSourceAttachment] = set()
        self._release: Callable[[], None] | None = release
        self._value: Any = value
        self._cursor = 0
        self._retired = False
        self._closed = False

    def attach(self) -> SessionSourceAttachment:
        if self._closed:
            raise RuntimeError("State source is closed")

        def release() -> None:
            self._attachments.discard(attachment)
            if len(self._attachments) == 0:
                self._finishDisposal()

        attachment = SessionSourceAttachment({"value": self._value, "cursor": self._cursor}, release)
        self._attachments.add(attachment)
        return attachment

    def advance(self, value: Any, ops: Sequence[Op], context: Context) -> None:
        if self._closed or self._retired:
            return
        self._value = value
        self._cursor += 1
        if value is None:
            self._retired = True
        frame = {"cursor": self._cursor, "value": value, "ops": ops, "context": context}
        for attachment in list(self._attachments):
            attachment.publish(frame)

    def closeSession(self) -> None:
        if self._closed:
            return
        for attachment in list(self._attachments):
            attachment.dispose()
        self._finishDisposal()

    def _finishDisposal(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._value = None
        release = self._release
        self._release = None
        if release is not None:
            release()


class SessionSourceAttachment:
    def __init__(self, snapshot: dict[str, Any], release: Callable[[], None]) -> None:
        self.snapshot = snapshot
        self._release: Callable[[], None] | None = release
        self._frames: list[dict[str, Any]] = []
        self._listener: Callable[[dict[str, Any]], None] | None = None
        self._activated = False
        self._disposed = False
        self._scheduled = False
        self._delivering = False

    def activate(self, listener: Callable[[dict[str, Any]], None]) -> None:
        if self._activated:
            raise RuntimeError("State attachment is already active")
        if self._disposed:
            raise RuntimeError("State attachment is disposed")
        self._activated = True
        self._listener = listener
        self._drain()

    def publish(self, frame: dict[str, Any]) -> None:
        if self._disposed:
            return
        self._frames.append(frame)
        if not self._activated or self._delivering or self._scheduled:
            return
        self._scheduled = True

        def deliver() -> None:
            self._scheduled = False
            if self._disposed:
                return
            try:
                self._drain()
            except Exception:  # noqa: BLE001
                # Chord's frame listener contains source-contract failures. Isolate an
                # unexpected direct listener failure to this attachment as well.
                self.dispose()

        queueMicrotask(deliver)

    def dispose(self) -> None:
        if self._disposed:
            return
        self._disposed = True
        self._frames.clear()
        self._listener = None
        release = self._release
        self._release = None
        if release is not None:
            release()

    def _drain(self) -> None:
        listener = self._listener
        if listener is None or self._delivering or self._disposed:
            return
        self._delivering = True
        try:
            while not self._disposed:
                if not self._frames:
                    break
                frame = self._frames.pop(0)
                listener(frame)
        finally:
            self._delivering = False


class CommittedWatch:
    """Serialized exact-frame watch bound to one document incarnation or conversation view. A ``None`` value retires
    it. ``replace`` gives the value an overflow delivers; by default the newest value."""

    def __init__(self, value: Any, detach: Callable[[], None], replace: Callable[[], Any] | None = None) -> None:
        self._value = value
        self._detach = detach
        self._replace = replace
        self._closedPromise: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._pending: list[dict[str, Any]] = []
        self._listener: Callable[[Any, Sequence[Op], Context], Awaitable[None]] | None = None
        self._started = False
        self._scheduled = False
        self._running = False
        self._detached = False
        self._retired = False
        self._end: dict[str, Any] | None = None
        self._resolved = False
        self._cancellationSignal: Any = None
        self._cancellationListener: Callable[[], None] | None = None
        self._tasks: set[asyncio.Task[Any]] = set()

    @property
    def value(self) -> Any:
        return self._value

    @property
    def closed(self) -> asyncio.Future[dict[str, Any]]:
        return self._closedPromise

    def start(self, listener: Callable[[Any, Sequence[Op], Context], Awaitable[None]]) -> None:
        if self._started:
            raise RuntimeError("Watch is already started")
        if self._end is not None:
            raise RuntimeError("Watch is stopped")
        self._started = True
        self._listener = listener
        if self._pending:
            self._schedule()

    def stop(self) -> asyncio.Future[dict[str, Any]]:
        self._terminate({"reason": "stopped"})
        return self._closedPromise

    def observeCancellation(self, signal: Any) -> None:
        if self._cancellationSignal is not None:
            raise RuntimeError("Watch cancellation is already installed")
        if self._end is not None:
            return
        self._cancellationSignal = signal
        self._cancellationListener = self.cancel
        signal.addEventListener("abort", self._cancellationListener, once=True)
        if signal.aborted:
            self.cancel()

    def cancel(self) -> None:
        self._terminate({"reason": "cancelled"})

    def closeSession(self) -> None:
        self._terminate({"reason": "session_closed"})

    def advance(self, value: Any, ops: Sequence[Op], context: Context) -> None:
        if self._end is not None or self._retired:
            return
        if value is None:
            self._retired = True
        if len(self._pending) >= MAX_PENDING_WATCH_FRAMES:
            self._pending.clear()
            replacement = self._replace() if self._replace is not None else None
            if replacement is None:
                replacement = value
            self._pending.append({"value": replacement, "ops": [["r", replacement]], "context": context})
        else:
            self._pending.append({"value": value, "ops": ops, "context": context})
        if self._started:
            self._schedule()

    def _schedule(self) -> None:
        if self._scheduled or self._running or self._end is not None:
            return
        self._scheduled = True

        def run() -> None:
            self._scheduled = False
            spawn(self._drain(), self._tasks)

        queueMicrotask(run)

    async def _drain(self) -> None:
        if self._running or self._end is not None or not self._started:
            self._finishIfReady()
            return
        self._running = True
        try:
            while self._end is None:
                if not self._pending:
                    break
                frame = self._pending.pop(0)
                self._value = frame["value"]
                deliveryContext = withoutAbortSignal(frame["context"])
                try:
                    assert self._listener is not None
                    await self._listener(frame["value"], frame["ops"], deliveryContext)
                except Exception as error:  # noqa: BLE001
                    if self._end is None:
                        self._terminate({"reason": "listener_error", "error": error})
                    break
                if frame["value"] is None:
                    self._terminate({"reason": "retired"})
                    break
        finally:
            self._running = False
            if self._end is None and self._pending:
                self._schedule()
            self._finishIfReady()

    def _terminate(self, end: dict[str, Any]) -> None:
        if self._end is not None:
            return
        self._end = end
        self._detachNow()
        self._pending.clear()
        self._finishIfReady()

    def _detachNow(self) -> None:
        if self._detached:
            return
        self._detached = True
        self._detach()

    def _finishIfReady(self) -> None:
        if self._resolved or self._end is None:
            return
        self._resolved = True
        if self._cancellationSignal is not None and self._cancellationListener is not None:
            self._cancellationSignal.removeEventListener("abort", self._cancellationListener)
        if not self._closedPromise.done():
            self._closedPromise.set_result(self._end)


__all__ = [
    "MAX_PENDING_WATCH_FRAMES",
    "RETIREMENT_OPERATIONS",
    "CommittedStateSource",
    "CommittedWatch",
    "ObservedDocumentValue",
    "SessionSourceAttachment",
]
