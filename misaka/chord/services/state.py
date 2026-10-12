"""Replicated state: publication order, subscriptions, and the cold replica a consumer reads.

Translated from pi's ``packages/chord/src/services/state.ts``.

MISAKA forks, all forced by the language: a listener may be a plain function or a coroutine function (a
returned awaitable is awaited on the running event loop, where upstream awaits a ``PromiseLike``);
``AggregateError`` is an ``ExceptionGroup``; ``queueMicrotask(() => { throw error })`` reports on the running
loop through ``loop.call_soon``, so asyncio's exception handler sees it, and through ``sys.excepthook``
without one; ``undefined`` is ``UNDEFINED``.
"""

from __future__ import annotations

import asyncio
import inspect
import sys
from collections.abc import Callable, Sequence
from types import SimpleNamespace
from typing import Any

from misaka.chord.context import BACKGROUND_CONTEXT
from misaka.chord.delta import Op, applyImmutable, isBase, track
from misaka.chord.delta.revision_validator import JsonRevisionValidator
from misaka.chord.types import (
    AttachedReplicatedState,
    Context,
    MutableReplicatedState,
    ReplicatedState,
    ReplicatedStateDelivery,
    ReplicatedStateSource,
    ReplicatedStateSourceAttachment,
    ReplicatedStateSourceFrame,
    ReplicatedStateSourceOptions,
    StateListener,
)
from misaka.utils.js import UNDEFINED, jsIsInteger, startTask

from .state_internals import registerReplicatedStateInternals

_MAX_SAFE_INTEGER = 2**53 - 1


class StateDelivery[T]:
    __slots__ = ("context", "delivery", "value")

    def __init__(self, value: T, context: Context, delivery: ReplicatedStateDelivery) -> None:
        self.value = value
        self.context = context
        self.delivery = delivery


class StateSubscriber[T]:
    """One public subscription, independent of producer and other subscriber progress."""

    def __init__(self, listener: StateListener[T], reportError: Callable[[Exception], None]) -> None:
        self._listener = listener
        self._reportError = reportError
        self._pending: list[StateDelivery[T]] = []
        self._running = False
        self._started = False
        self._closed = False

    def push(self, frame: StateDelivery[T]) -> None:
        if self._closed:
            return
        if len(self._pending) == 100:
            # A cold replica can queue updates reentrantly before this subscriber's first hydration starts.
            hydration = None if self._started else self._pending[0]
            self._pending.clear()
            if hydration is not None:
                self._pending.append(hydration)
        self._pending.append(frame)

    def drain(self) -> None:
        if self._running or self._closed:
            return
        self._running = True
        while self._pending:
            frame = self._pending.pop(0)
            self._started = True
            result: Any = None
            try:
                result = self._listener(frame.value, frame.context, frame.delivery)
                if inspect.isawaitable(result):
                    task = startTask(result)
                    task.add_done_callback(self._settled)
                    return
            except Exception as error:  # noqa: BLE001 - a listener's failure is reported and delivery continues
                if inspect.iscoroutine(result):
                    result.close()
                self._report(error)
        self._running = False

    def _settled(self, task: asyncio.Future[Any]) -> None:
        if not task.cancelled() and (error := task.exception()) is not None:
            self._report(error)
        self._resume()

    def clear(self) -> None:
        self._pending.clear()

    def close(self) -> None:
        self._closed = True
        self.clear()

    def _resume(self) -> None:
        self._running = False
        self.drain()

    def _report(self, error: BaseException) -> None:
        try:
            self._reportError(toError(error))
        except Exception as reportError:  # noqa: BLE001 - a reporter that fails goes to the loop's handler
            reportErrorAsync(toError(reportError))

type SourceListener = Callable[[Sequence[Op], int, Context], None]


class Publication[T]:
    __slots__ = ("context", "ops", "sequence", "value")

    def __init__(self, value: T, ops: Sequence[Op], sequence: int, context: Context) -> None:
        self.value = value
        self.ops = ops
        self.sequence = sequence
        self.context = context


class ReplicatedStatePublisher[T]:
    """Maintains local publication order independently of how revisions are produced."""

    def __init__(self, initial: T, reportError: Callable[[Exception], None] | None = None) -> None:
        self._listeners: dict[StateSubscriber[T], int] = {}
        self._reportError = reportError if reportError is not None else reportErrorAsync
        self._sourceListeners: list[SourceListener] = []
        self._publications: list[Publication[T]] = []
        self._value = initial
        self._sequence = 0
        self._delivering = False

    @property
    def value(self) -> T:
        return self._value

    def snapshot(self) -> dict[str, Any]:
        return {"value": self._value, "sequence": self._sequence}

    def subscribe(self, listener: StateListener[T]) -> Callable[[], None]:
        value, sequence = self._value, self._sequence
        subscriber = StateSubscriber(listener, self._reportError)
        self._listeners[subscriber] = sequence
        subscriber.push(StateDelivery(value, serviceDeliveryContext(), {"kind": "hydrate", "sequence": sequence}))
        subscriber.drain()

        def unsubscribe() -> None:
            subscriber.close()
            self._listeners.pop(subscriber, None)

        return unsubscribe

    def subscribeSource(self, listener: SourceListener) -> Callable[[], None]:
        self._sourceListeners.append(listener)

        def remove() -> None:
            if listener in self._sourceListeners:
                self._sourceListeners.remove(listener)

        return remove

    def publish(self, value: T, ops: Sequence[Op], context: Context) -> list[Exception]:
        """Publish an already-prepared immutable revision and return isolated listener failures."""
        self._value = value
        self._sequence += 1
        self._publications.append(Publication(value, ops, self._sequence, context))
        if self._delivering:
            return []

        self._delivering = True
        errors: list[Exception] = []
        try:
            while self._publications:
                publication = self._publications.pop(0)
                for listener in list(self._sourceListeners):
                    try:
                        listener(publication.ops, publication.sequence, publication.context)
                    except Exception as error:  # noqa: BLE001 - collected, as upstream collects what a listener throws
                        errors.append(error)
                delivery: ReplicatedStateDelivery = {"kind": "update", "sequence": publication.sequence}
                for subscriber, hydratedSequence in list(self._listeners.items()):
                    if publication.sequence <= hydratedSequence:
                        continue
                    subscriber.push(StateDelivery(publication.value, publication.context, delivery))
                    subscriber.drain()
        finally:
            self._delivering = False
        return errors


class MutableReplicatedStateImpl[T](MutableReplicatedState[T]):
    def __init__(self, initial: T) -> None:
        self._tracker = track(initial)
        self._publisher: ReplicatedStatePublisher[T] = ReplicatedStatePublisher(self._tracker.value)
        self._changing = False
        registerReplicatedStateInternals(
            self,
            SimpleNamespace(snapshot=self._publisher.snapshot, subscribe=self._publisher.subscribeSource),  # type: ignore[arg-type]
        )

    @property
    def value(self) -> T:
        return self._tracker.value

    def change(self, context: Context, mutate: Callable[[Any], Any]) -> None:
        if self._changing:
            raise RuntimeError("Replicated state cannot be changed reentrantly from a change callback")
        self._changing = True
        try:
            change = self._tracker.beginChange()
            try:
                outcome = mutate(change.state)
                if inspect.isawaitable(outcome):
                    discardAwaitable(outcome)
                    raise TypeError("Replicated state change callbacks must be synchronous")
                prepared = change.prepare()
            except BaseException:
                change.abort()
                raise
        finally:
            self._changing = False

        self._tracker.adopt(prepared)
        if len(prepared.ops) == 0:
            return
        throwCollectedErrors(
            self._publisher.publish(prepared.value, prepared.ops, context), "Replicated state listeners failed"
        )

    def replace(self, context: Context, value: T) -> None:
        if self._changing:
            raise RuntimeError("Replicated state cannot be replaced from a change callback")
        prepared = self._tracker.prepareReplace(value)
        self._tracker.adopt(prepared)
        if len(prepared.ops) == 0:
            return
        throwCollectedErrors(
            self._publisher.publish(prepared.value, prepared.ops, context), "Replicated state listeners failed"
        )

    def subscribe(self, listener: StateListener[T]) -> Callable[[], None]:
        return self._publisher.subscribe(listener)


class AttachedReplicatedStateImpl[T](AttachedReplicatedState[T]):
    def __init__(self, attachment: ReplicatedStateSourceAttachment[T], options: ReplicatedStateSourceOptions) -> None:
        snapshot = attachment.snapshot
        assertCursor(snapshot["cursor"], "snapshot")
        self._attachment = attachment
        self._cursor = snapshot["cursor"]
        self._reportError = options.get("onError") or reportErrorAsync
        self._disposed = False
        self._publisher: ReplicatedStatePublisher[T] = ReplicatedStatePublisher(snapshot["value"], self._report)
        registerReplicatedStateInternals(
            self,
            SimpleNamespace(snapshot=self._publisher.snapshot, subscribe=self._publisher.subscribeSource),  # type: ignore[arg-type]
        )

    @property
    def value(self) -> T:
        return self._publisher.value

    def subscribe(self, listener: StateListener[T]) -> Callable[[], None]:
        return self._publisher.subscribe(listener)

    def activate(self) -> None:
        self._attachment.activate(self._receive)

    def dispose(self) -> None:
        if self._disposed:
            return
        self._disposed = True
        self._attachment.dispose()

    def _receive(self, frame: ReplicatedStateSourceFrame[T]) -> None:
        if self._disposed:
            return
        try:
            assertCursor(frame["cursor"], "frame")
            expected = self._cursor + 1
            if frame["cursor"] != expected:
                raise RuntimeError(
                    f"Replicated state source cursor has a gap: expected {expected}, received {frame['cursor']}"
                )
            self._cursor = frame["cursor"]
            errors = self._publisher.publish(frame["value"], frame["ops"], frame["context"])
            if len(errors) == 1:
                self._report(errors[0])
            elif len(errors) > 1:
                self._report(ExceptionGroup("Replicated state listeners failed", errors))
        except Exception as error:  # noqa: BLE001 - a source that breaks its contract fails the state
            self._fail(toError(error))

    def _fail(self, error: Exception) -> None:
        if self._disposed:
            return
        self._disposed = True
        try:
            self._attachment.dispose()
        except Exception as disposeError:  # noqa: BLE001
            self._report(ExceptionGroup("Replicated state source contract failed", [error, disposeError]))
            return
        self._report(error)

    def _report(self, error: BaseException) -> None:
        try:
            self._reportError(toError(error))
        except Exception as reportError:  # noqa: BLE001
            reportErrorAsync(toError(reportError))


def attachReplicatedStateSource[T](
    source: ReplicatedStateSource[T], options: ReplicatedStateSourceOptions | None = None
) -> AttachedReplicatedState[T]:
    """Attach a publication-only replicated state to one authoritative immutable source stream."""
    attachment = source.attach()
    try:
        state = AttachedReplicatedStateImpl(attachment, options or {})
        state.activate()
        return state
    except Exception as error:
        try:
            attachment.dispose()
        except Exception as disposeError:  # noqa: BLE001
            raise ExceptionGroup("Failed to attach replicated state source", [error, disposeError]) from None
        raise


class ReplicatedStateReplica[T](ReplicatedState[T]):
    """A cold read-only state used by service consumers until a complete snapshot arrives."""

    def __init__(self, reportError: Callable[[Exception], None]) -> None:
        self._listeners: list[StateSubscriber[T]] = []
        self._reportError = reportError
        self._validator = JsonRevisionValidator()
        self._value: Any = UNDEFINED
        self._sequence: int | None = None

    @property
    def value(self) -> T:
        return self._value

    def subscribe(self, listener: StateListener[T]) -> Callable[[], None]:
        subscriber = StateSubscriber(listener, self._reportError)
        self._listeners.append(subscriber)
        if self._value is not UNDEFINED:
            assert self._sequence is not None
            subscriber.push(
                StateDelivery(self._value, serviceDeliveryContext(), {"kind": "hydrate", "sequence": self._sequence})
            )
            subscriber.drain()

        def unsubscribe() -> None:
            subscriber.close()
            if subscriber in self._listeners:
                self._listeners.remove(subscriber)

        return unsubscribe

    def hydrate(self, sequence: int, ops: Sequence[Op], context: Context) -> None:
        try:
            if not isBase(ops):
                raise RuntimeError("Replicated state snapshot is not a base operation batch")
            next_ = self._validator.validate(applyImmutable(UNDEFINED, ops))
        except BaseException:
            self.clear()
            raise
        self._sequence = sequence
        self._value = next_
        self._deliverAll(context, {"kind": "hydrate", "sequence": sequence})

    def update(self, sequence: int, ops: Sequence[Op], context: Context) -> None:
        if self._sequence is None or self._value is UNDEFINED:
            raise RuntimeError("Replicated state received an update before hydration")
        if sequence != self._sequence + 1:
            self.clear()
            raise RuntimeError("Replicated state update sequence has a gap")
        try:
            next_ = self._validator.validate(applyImmutable(self._value, ops))
        except BaseException:
            self.clear()
            raise
        self._sequence = sequence
        self._value = next_
        self._deliverAll(context, {"kind": "update", "sequence": sequence})

    def clear(self) -> None:
        self._value = UNDEFINED
        self._sequence = None
        for subscriber in self._listeners:
            subscriber.clear()

    def _deliverAll(self, context: Context, delivery: ReplicatedStateDelivery) -> None:
        if self._value is UNDEFINED:
            return
        frame = StateDelivery(self._value, context, delivery)
        subscribers = list(self._listeners)
        # Enqueue for everyone before user code can publish another revision reentrantly.
        for subscriber in subscribers:
            subscriber.push(frame)
        for subscriber in subscribers:
            subscriber.drain()


def serviceDeliveryContext() -> Context:
    """Context for synthetic service deliveries without a caller."""
    return BACKGROUND_CONTEXT


def assertCursor(cursor: Any, kind: str) -> None:
    if not jsIsInteger(cursor) or abs(cursor) > _MAX_SAFE_INTEGER:
        raise TypeError(f"Replicated state source {kind} cursor must be a safe integer")


def discardAwaitable(value: Any) -> None:
    """``void Promise.resolve(value).catch(() => undefined)``: nothing will wait for the outcome."""
    if inspect.iscoroutine(value):
        value.close()
        return
    if isinstance(value, asyncio.Future):
        value.add_done_callback(lambda done: done.cancelled() or done.exception())


def throwCollectedErrors(errors: Sequence[Exception], message: str) -> None:
    if len(errors) == 1:
        raise errors[0]
    if len(errors) > 1:
        raise ExceptionGroup(message, list(errors))


def reportErrorAsync(error: Exception) -> None:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        sys.excepthook(type(error), error, error.__traceback__)
        return
    loop.call_soon(_raise, error)


def _raise(error: Exception) -> None:
    raise error


def toError(error: BaseException) -> Exception:
    return error if isinstance(error, Exception) else Exception(str(error))


__all__ = [
    "AttachedReplicatedStateImpl",
    "MutableReplicatedStateImpl",
    "ReplicatedStatePublisher",
    "ReplicatedStateReplica",
    "attachReplicatedStateSource",
    "serviceDeliveryContext",
]
