"""pi chord's ``test/service-delivery.test.ts``."""

from __future__ import annotations

import pytest

from misaka.chord import (
    RemoteServiceProvider,
    createRemoteServiceBinding,
    createServiceStateDecoder,
    createServiceStateEncoder,
    defineService,
    parseWireServiceProviderUpdate,
    parseWireServiceSubscriptionSnapshot,
    replicatedState,
)
from misaka.chord.context import BACKGROUND_CONTEXT, withCancel
from misaka.chord.services.state_internals import getReplicatedStateInternals
from tests.chord_helpers import matches

Counter = defineService("test.delivery-counter")
Pair = defineService("test.delivery-pair")


class WireSubscription:
    def __init__(self, subscription, snapshot, beforeActivate):
        self._subscription = subscription
        self.snapshot = snapshot
        self._beforeActivate = beforeActivate

    def activate(self):
        self._beforeActivate()
        self._subscription.activate()

    def close(self, context=None):
        return self._subscription.close()


class WireTransport:
    def __init__(self, provider, beforeActivate, updates):
        self._provider = provider
        self._beforeActivate = beforeActivate
        self._updates = updates

    def invoke(self, call, context):
        return self._provider.invoke(call, context)

    async def subscribe(self, serviceId, mode, listener, context):
        encoder = createServiceStateEncoder()
        decoder = createServiceStateDecoder()

        def onUpdate(update, updateContext):
            decoded = decoder.decodeUpdate(parseWireServiceProviderUpdate(encoder.encodeUpdate(update)))
            self._updates.append(decoded)
            listener(decoded, updateContext)

        subscription = self._provider.subscribe(serviceId, mode, onUpdate)
        snapshot = decoder.decodeSnapshot(
            parseWireServiceSubscriptionSnapshot(encoder.encodeSnapshot(subscription.snapshot))
        )
        return WireSubscription(subscription, snapshot, self._beforeActivate)


def setValue(state, value):
    state.change(BACKGROUND_CONTEXT, lambda draft: draft.__setitem__("value", value))


# -- provider delivery queues --------------------------------------------------------------------


def test_appends_reentrant_updates_to_the_activation_fifo():
    provider = RemoteServiceProvider([Counter])
    state = replicatedState({"value": 0})
    provider.provide(Counter, {"state": state})
    received = []

    def listener(update, _context):
        if update["type"] != "state":
            return
        received.append(update["sequence"])
        if update["sequence"] == 1:
            state.replace(BACKGROUND_CONTEXT, {"value": 3})

    subscription = provider.subscribe(Counter.id, "singleton", listener)
    state.replace(BACKGROUND_CONTEXT, {"value": 1})
    state.replace(BACKGROUND_CONTEXT, {"value": 2})
    subscription.activate()
    subscription.activate()
    assert received == [1, 2, 3]
    provider.dispose()


@pytest.mark.parametrize("count", [100, 101, 102, 201, 202])
def test_bounds_pending_updates_with_explicit_root_resets(count):
    provider = RemoteServiceProvider([Counter])
    state = replicatedState({"value": 0})
    provider.provide(Counter, {"state": state})
    updates = []
    contexts = []

    def listener(update, context):
        updates.append(update)
        contexts.append(context)

    subscription = provider.subscribe(Counter.id, "singleton", listener)
    context = withCancel(BACKGROUND_CONTEXT).context
    for value in range(1, count + 1):
        state.replace(context, {"value": value})
    assert updates == []
    assert matches(subscription.snapshot["instances"][0]["members"][0], {"sequence": 0})
    subscription.activate()
    first = (count - 1) // 100 * 100 + 1
    assert len(updates) == count - first + 1
    if count > 100:
        assert updates[0] == {
            "type": "reset",
            "snapshot": {
                "serviceId": Counter.id,
                "mode": "singleton",
                "instances": [
                    {"members": [{"name": "state", "kind": "state", "sequence": first, "ops": [["r", {"value": first}]]}]}
                ],
            },
        }
    assert all(received is context for received in contexts)
    state.replace(context, {"value": count + 1})
    assert matches(updates[-1], {"type": "state", "sequence": count + 1})
    provider.dispose()


def test_does_not_compact_the_running_delivery_when_activation_overflows_reentrantly():
    provider = RemoteServiceProvider([Counter])
    state = replicatedState({"value": 0})
    provider.provide(Counter, {"state": state})
    updates = []

    def listener(update, _context):
        updates.append(update)
        if update["type"] == "state" and update["sequence"] == 1:
            for value in range(3, 104):
                state.replace(BACKGROUND_CONTEXT, {"value": value})

    subscription = provider.subscribe(Counter.id, "singleton", listener)
    state.replace(BACKGROUND_CONTEXT, {"value": 1})
    state.replace(BACKGROUND_CONTEXT, {"value": 2})
    subscription.activate()
    assert len(updates) == 3
    assert matches(updates[0], {"type": "state", "sequence": 1})
    assert matches(
        updates[1],
        {"type": "reset", "snapshot": {"instances": [{"members": [{"sequence": 102, "ops": [["r", {"value": 102}]]}]}]}},
    )
    assert matches(updates[2], {"type": "state", "sequence": 103})
    provider.dispose()


def test_suppresses_notifications_already_covered_by_an_overflow_snapshot():
    provider = RemoteServiceProvider([Counter])
    state = replicatedState({"value": 0})

    def onSequence(_ops, sequence, _context):
        if sequence == 101:
            state.replace(BACKGROUND_CONTEXT, {"value": 102})
            state.replace(BACKGROUND_CONTEXT, {"value": 103})

    getReplicatedStateInternals(state).subscribe(onSequence)
    provider.provide(Counter, {"state": state})
    updates = []
    subscription = provider.subscribe(Counter.id, "singleton", lambda update, _context: updates.append(update))
    for value in range(1, 102):
        state.replace(BACKGROUND_CONTEXT, {"value": value})
    subscription.activate()
    assert len(updates) == 1
    assert matches(
        updates[0],
        {"type": "reset", "snapshot": {"instances": [{"members": [{"sequence": 103, "ops": [["r", {"value": 103}]]}]}]}},
    )
    state.replace(BACKGROUND_CONTEXT, {"value": 104})
    assert matches(updates[1], {"type": "state", "sequence": 104})
    provider.dispose()


def test_preserves_lifecycle_ordering_across_subscribers_during_reentrant_publication():
    provider = RemoteServiceProvider([Counter])
    provider.provide(Counter, {"state": replicatedState({"value": 0})})
    first = []
    second = []

    def onFirst(update, _context):
        first.append(update)
        if len(first) == 1:
            provider.replace(Counter, {"state": replicatedState({"value": 2})})

    provider.subscribe(Counter.id, "singleton", onFirst).activate()
    provider.subscribe(Counter.id, "singleton", lambda update, _context: second.append(update)).activate()
    provider.replace(Counter, {"state": replicatedState({"value": 1})})
    assert first == second
    assert len(second) == 2
    assert matches(second[0], {"type": "replaced", "snapshot": {"members": [{"ops": [["r", {"value": 1}]]}]}})
    provider.dispose()


def test_close_during_a_callback_prevents_later_buffered_callbacks():
    provider = RemoteServiceProvider([Counter])
    state = replicatedState({"value": 0})
    provider.provide(Counter, {"state": state})
    received = []
    holder = []

    def listener(update, _context):
        received.append(update)
        holder[0].close()

    holder.append(provider.subscribe(Counter.id, "singleton", listener))
    state.replace(BACKGROUND_CONTEXT, {"value": 1})
    state.replace(BACKGROUND_CONTEXT, {"value": 2})
    holder[0].activate()
    assert len(received) == 1
    provider.dispose()


def test_drains_terminal_updates_when_disposed_reentrantly():
    provider = RemoteServiceProvider([Counter])
    state = replicatedState({"value": 0})
    provider.provide(Counter, {"state": state})
    received = []

    def listener(update, _context):
        received.append(update)
        if update["type"] == "state":
            provider.dispose()

    subscription = provider.subscribe(Counter.id, "singleton", listener)
    state.replace(BACKGROUND_CONTEXT, {"value": 1})
    subscription.activate()
    assert [update["type"] for update in received] == ["state", "unavailable"]


async def test_rebaselines_every_member_through_wire_codecs_then_resumes_contiguous_deltas():
    provider = RemoteServiceProvider([Pair])
    left = replicatedState({"value": 0})
    right = replicatedState({"value": 0})
    provider.provide(Pair, {"left": left, "right": right})
    errors = []
    updates = []

    def beforeActivate():
        for value in range(1, 102):
            setValue(left, value)
            setValue(right, value)

    namespace = createRemoteServiceBinding({
        "services": [Pair],
        "transport": WireTransport(provider, beforeActivate, updates),
        "onError": errors.append,
    })
    pair = namespace.use(Pair)
    await namespace.ready(BACKGROUND_CONTEXT)
    assert [update["type"] for update in updates] == ["reset", "state"]
    assert pair.left.value == {"value": 101}
    assert pair.right.value == {"value": 101}
    for value in range(102, 105):
        setValue(left, value)
        setValue(right, value)
    assert pair.left.value == {"value": 104}
    assert pair.right.value == {"value": 104}
    assert errors == []
    await namespace.dispose(BACKGROUND_CONTEXT)
    provider.dispose()


async def test_an_overflow_reset_can_make_a_singleton_unavailable_before_a_later_replacement():
    from misaka.utils.js import UNDEFINED

    provider = RemoteServiceProvider([Counter])
    provider.provide(Counter, {"state": replicatedState({"value": 0})})
    updates = []
    errors = []

    def beforeActivate():
        for value in range(1, 101):
            provider.replace(Counter, {"state": replicatedState({"value": value})})
        provider.withdraw(Counter)

    namespace = createRemoteServiceBinding({
        "services": [Counter],
        "transport": WireTransport(provider, beforeActivate, updates),
        "onError": errors.append,
    })
    counter = namespace.use(Counter)
    await namespace.ready(BACKGROUND_CONTEXT)
    assert updates == [{"type": "reset", "snapshot": {"serviceId": Counter.id, "mode": "singleton", "instances": []}}]
    assert counter.state.value is UNDEFINED
    replacement = replicatedState({"value": 200})
    provider.replace(Counter, {"state": replacement})
    setValue(replacement, 201)
    assert namespace.use(Counter) is counter
    assert counter.state.value == {"value": 201}
    assert errors == []
    await namespace.dispose(BACKGROUND_CONTEXT)
    provider.dispose()


async def test_keyed_resets_retain_live_generations_and_reconcile_closed_and_reused_keys():
    provider = RemoteServiceProvider([{"service": Counter, "mode": "keyed"}])
    retained = replicatedState({"value": 0})
    provider.spawn(Counter, "retained", {"state": retained})
    closeOld = provider.spawn(Counter, "reused", {"state": replicatedState({"value": 0})})
    removed = provider.spawn(Counter, "removed", {"state": replicatedState({"value": 0})})
    updates = []
    errors = []
    namespace = createRemoteServiceBinding({
        "services": [Counter],
        "transport": WireTransport(provider, lambda: None, updates),
        "onError": errors.append,
    })
    observed = []
    namespace.observe(Counter, lambda service, context: observed.append({"service": service, "context": context}))
    await namespace.ready(BACKGROUND_CONTEXT)
    assert len(observed) == 3
    stable = observed[1]  # Provider snapshots are sorted by key.
    old = observed[2]
    replacement = replicatedState({"value": 500})

    def onStable(value, _context, _delivery):
        if value["value"] != 1:
            return
        closeOld()
        removed()
        provider.spawn(Counter, "reused", {"state": replacement})
        retained.replace(BACKGROUND_CONTEXT, {"value": 102})
        for nextValue in range(501, 602):
            replacement.replace(BACKGROUND_CONTEXT, {"value": nextValue})

    stable["service"].state.subscribe(onStable)
    retained.replace(BACKGROUND_CONTEXT, {"value": 1})
    assert any(update["type"] == "reset" for update in updates)
    assert len(observed) == 4
    assert stable["context"].abortSignal.aborted is False
    assert stable["service"].state.value == {"value": 102}
    assert old["context"].abortSignal.aborted is True
    assert observed[0]["context"].abortSignal.aborted is True
    assert observed[3]["service"].state.value == {"value": 601}
    setValue(replacement, 602)
    assert observed[3]["service"].state.value == {"value": 602}
    assert errors == []
    await namespace.dispose(BACKGROUND_CONTEXT)
    provider.dispose()
