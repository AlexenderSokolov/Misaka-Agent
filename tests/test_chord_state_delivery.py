"""pi chord's ``test/state-delivery.test.ts``.

A listener returning an awaitable (a coroutine or a future) is awaited, as one returning a promise is. A
coroutine listener runs at once, up to its first suspension, as a JavaScript async function does.
"""

from __future__ import annotations

import asyncio

import pytest

from misaka.chord import replicatedState
from misaka.chord.context import BACKGROUND_CONTEXT, withCancel
from misaka.chord.services.state import ReplicatedStateReplica
from misaka.chord.services.state_internals import getReplicatedStateInternals


class Deferred:
    def __init__(self):
        self.promise = asyncio.get_running_loop().create_future()

    def resolve(self):
        self.promise.set_result(None)

    def reject(self, error):
        self.promise.set_exception(error)


async def waitFor(predicate, timeout=2.0):
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() > deadline:
            raise AssertionError("condition not met in time")
        await asyncio.sleep(0.005)


def fixture(kind, onError=lambda error: None):
    if kind == "mutable":
        mutable = replicatedState({"value": 0})
        return mutable, lambda value, context: mutable.replace(context, value)
    if kind == "attached":
        holder = {}

        class Attachment:
            def __init__(self):
                self.snapshot = {"value": {"value": 0}, "cursor": 0}

            def activate(self, listener):
                holder["receive"] = listener

            def dispose(self):
                pass

        class Source:
            def attach(self):
                return Attachment()

        state = replicatedState(Source(), {"onError": onError})
        return state, lambda value, context: holder["receive"](
            {"value": value, "cursor": value["value"], "ops": [["s", ["value"], value["value"]]], "context": context}
        )
    replica = ReplicatedStateReplica(onError)
    replica.hydrate(0, [["r", {"value": 0}]], BACKGROUND_CONTEXT)
    return replica, lambda value, context: replica.update(value["value"], [["s", ["value"], value["value"]]], context)


def exactSubscribe(state, listener):
    internals = getReplicatedStateInternals(state)
    if internals is not None:
        internals.subscribe(listener)


KINDS = ["mutable", "attached", "replica"]


@pytest.mark.parametrize("kind", KINDS)
async def test_awaits_hydration_and_each_update_independently_of_other_subscribers_and_exact_listeners(kind):
    state, publish = fixture(kind)
    hydration = Deferred()
    update = Deferred()
    events = []
    exact = []
    fast = []
    exactSubscribe(state, lambda _ops, sequence, _context: exact.append(sequence))

    async def slow(value, _context, _delivery):
        events.append(f"start:{value['value']}")
        if value["value"] == 0:
            await hydration.promise
        if value["value"] == 1:
            await update.promise
        events.append(f"end:{value['value']}")

    state.subscribe(slow)
    state.subscribe(lambda value, _context, _delivery: fast.append(value["value"]))
    publish({"value": 1}, BACKGROUND_CONTEXT)
    publish({"value": 2}, BACKGROUND_CONTEXT)
    assert events == ["start:0"]
    assert fast == [0, 1, 2]
    assert state.value == {"value": 2}
    if kind != "replica":
        assert exact == [1, 2]
    hydration.resolve()
    await waitFor(lambda: events == ["start:0", "end:0", "start:1"])
    update.resolve()
    await waitFor(lambda: events == ["start:0", "end:0", "start:1", "end:1", "start:2", "end:2"])


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("count", [100, 101, 102, 201, 202])
async def test_bounds_pending_deliveries_without_changing_exact_publication(kind, count):
    state, publish = fixture(kind)
    hydration = Deferred()
    received = []
    exact = []
    exactSubscribe(state, lambda _ops, sequence, _context: exact.append(sequence))

    def listener(value, _context, _delivery):
        received.append(value["value"])
        if value["value"] == 0:
            return hydration.promise
        return None

    state.subscribe(listener)
    for value in range(1, count + 1):
        publish({"value": value}, BACKGROUND_CONTEXT)
    assert received == [0]
    if kind != "replica":
        assert exact == list(range(1, count + 1))
    hydration.resolve()
    await hydration.promise
    await asyncio.sleep(0)
    first = (count - 1) // 100 * 100 + 1
    assert received == [0, *range(first, count + 1)]


@pytest.mark.parametrize("kind", KINDS)
async def test_excludes_a_running_update_from_overflow_and_retains_exact_value_context_delivery(kind):
    state, publish = fixture(kind)
    gate = Deferred()
    context = withCancel(BACKGROUND_CONTEXT).context
    received = []

    def listener(value, listenerContext, delivery):
        received.append({"value": value, "context": listenerContext, "delivery": delivery})
        if value["value"] == 1:
            return gate.promise
        return None

    state.subscribe(listener)
    for value in range(1, 102):
        publish({"value": value}, BACKGROUND_CONTEXT)
    newest = {"value": 102}
    publish(newest, context)
    adopted = state.value
    publish({"value": 103}, BACKGROUND_CONTEXT)
    assert [entry["value"]["value"] for entry in received] == [0, 1]
    gate.resolve()
    await gate.promise
    await asyncio.sleep(0)
    assert [entry["value"]["value"] for entry in received] == [0, 1, 102, 103]
    assert received[2]["value"] is adopted
    assert received[2]["context"] is context
    assert received[2]["delivery"] == {"kind": "update", "sequence": 102}


@pytest.mark.parametrize("kind", KINDS)
def test_serializes_reentrant_hydration_and_update_callbacks(kind):
    state, publish = fixture(kind)
    events = []

    def listener(value, _context, _delivery):
        events.append(f"start:{value['value']}")
        if value["value"] < 2:
            publish({"value": value["value"] + 1}, BACKGROUND_CONTEXT)
        events.append(f"end:{value['value']}")

    state.subscribe(listener)
    assert events == ["start:0", "end:0", "start:1", "end:1", "start:2", "end:2"]


@pytest.mark.parametrize("kind", KINDS)
async def test_treats_two_subscriptions_of_the_same_callback_independently(kind):
    state, publish = fixture(kind)
    gate = Deferred()
    received = []

    def listener(value, _context, _delivery):
        received.append(value["value"])
        if value["value"] == 0:
            return gate.promise
        return None

    stopFirst = state.subscribe(listener)
    stopSecond = state.subscribe(listener)
    publish({"value": 1}, BACKGROUND_CONTEXT)
    stopFirst()
    stopFirst()
    gate.resolve()
    await gate.promise
    await asyncio.sleep(0)
    assert received == [0, 0, 1]
    stopSecond()


@pytest.mark.parametrize("kind", KINDS)
async def test_unsubscribe_drops_queued_callbacks_without_joining_or_aborting_the_running_callback(kind):
    state, publish = fixture(kind)
    gate = Deferred()
    context = withCancel(BACKGROUND_CONTEXT).context
    received = []
    completed = [False]

    async def listener(value, _context, _delivery):
        received.append(value["value"])
        if value["value"] == 1:
            await gate.promise
            completed[0] = True

    stop = state.subscribe(listener)
    await asyncio.sleep(0)
    publish({"value": 1}, context)
    publish({"value": 2}, context)
    assert stop() is None
    publish({"value": 3}, context)
    assert context.abortSignal.aborted is False
    assert completed[0] is False
    gate.resolve()
    await waitFor(lambda: completed[0])
    assert received == [0, 1]


# -- listener errors -----------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["attached", "replica"])
def test_isolates_a_synchronous_hydration_failure_without_removing_the_subscription(kind):
    errors = []
    state, publish = fixture(kind, errors.append)
    received = []
    failure = RuntimeError("sync hydration")

    def listener(value, _context, _delivery):
        received.append(value["value"])
        if value["value"] == 0:
            raise failure

    state.subscribe(listener)
    publish({"value": 1}, BACKGROUND_CONTEXT)
    assert errors == [failure]
    assert received == [0, 1]


@pytest.mark.parametrize("kind", ["attached", "replica"])
async def test_observes_hydration_rejection_synchronous_throw_and_update_rejection_while_continuing_delivery(kind):
    errors = []
    state, publish = fixture(kind, errors.append)
    gate = Deferred()
    received = []
    fast = []

    def listener(value, _context, _delivery):
        received.append(value["value"])
        if value["value"] == 0:
            return gate.promise
        if value["value"] == 1:
            raise RuntimeError("sync update")
        if value["value"] == 2:
            future = asyncio.get_running_loop().create_future()
            future.set_exception(RuntimeError("async update"))
            return future
        return None

    state.subscribe(listener)
    state.subscribe(lambda value, _context, _delivery: fast.append(value["value"]))
    for value in range(1, 4):
        publish({"value": value}, BACKGROUND_CONTEXT)
    gate.reject(RuntimeError("async hydration"))
    await waitFor(lambda: received == [0, 1, 2, 3])
    assert [str(error) for error in errors] == ["async hydration", "sync update", "async update"]
    assert fast == [0, 1, 2, 3]


@pytest.mark.parametrize("kind", ["attached", "replica"])
async def test_still_observes_a_callback_rejection_after_unsubscribe(kind):
    errors = []
    state, publish = fixture(kind, errors.append)
    gate = Deferred()
    received = []

    def listener(value, _context, _delivery):
        received.append(value["value"])
        return gate.promise

    stop = state.subscribe(listener)
    publish({"value": 1}, BACKGROUND_CONTEXT)
    stop()
    failure = RuntimeError("stopped callback")
    gate.reject(failure)
    await waitFor(lambda: errors == [failure])
    assert received == [0]


async def test_mutable_state_reports_rejected_callbacks_through_its_default_error_reporter():
    loop = asyncio.get_running_loop()
    reports = []
    previous = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: reports.append(context))
    try:
        state = replicatedState({"value": 0})
        failure = RuntimeError("async mutable hydration")
        received = []

        def listener(value, _context, _delivery):
            received.append(value["value"])
            if value["value"] == 0:
                future = loop.create_future()
                future.set_exception(failure)
                return future
            return None

        state.subscribe(listener)
        state.replace(BACKGROUND_CONTEXT, {"value": 1})
        await asyncio.sleep(0.01)
        assert received == [0, 1]
        assert len(reports) == 1
        assert reports[0]["exception"] is failure
    finally:
        loop.set_exception_handler(previous)


async def test_replica_disconnect_drops_obsolete_pending_work_but_waits_for_the_running_callback_before_rehydration():
    replica = ReplicatedStateReplica(lambda error: None)
    gate = Deferred()
    received = []

    def listener(_value, _context, delivery):
        received.append(dict(delivery))
        if delivery["sequence"] == 0:
            return gate.promise
        return None

    replica.subscribe(listener)
    replica.hydrate(0, [["r", {"value": 0}]], BACKGROUND_CONTEXT)
    replica.update(1, [["s", ["value"], 1]], BACKGROUND_CONTEXT)
    replica.clear()
    replica.hydrate(50, [["r", {"value": 50}]], BACKGROUND_CONTEXT)
    replica.update(51, [["s", ["value"], 51]], BACKGROUND_CONTEXT)
    assert received == [{"kind": "hydrate", "sequence": 0}]
    gate.resolve()
    await gate.promise
    await asyncio.sleep(0)
    assert received == [
        {"kind": "hydrate", "sequence": 0},
        {"kind": "hydrate", "sequence": 50},
        {"kind": "update", "sequence": 51},
    ]


def test_cold_replicas_hydrate_all_listeners_before_their_reentrant_updates():
    replica = ReplicatedStateReplica(lambda error: None)
    second = []

    def first(_value, _context, delivery):
        if delivery["kind"] == "hydrate":
            replica.update(1, [["s", ["value"], 1]], BACKGROUND_CONTEXT)

    replica.subscribe(first)
    replica.subscribe(lambda _value, _context, delivery: second.append(dict(delivery)))
    replica.hydrate(0, [["r", {"value": 0}]], BACKGROUND_CONTEXT)
    assert second == [{"kind": "hydrate", "sequence": 0}, {"kind": "update", "sequence": 1}]
