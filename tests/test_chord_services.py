"""pi chord's ``test/services.test.ts``."""

from __future__ import annotations

import asyncio
import datetime
from typing import ClassVar

import pytest

from misaka.chord import (
    RemoteServiceProvider,
    createRemoteServiceBinding,
    defineService,
    replicatedState,
)
from misaka.chord.context import BACKGROUND_CONTEXT
from misaka.utils.js import UNDEFINED
from tests.chord_helpers import createLoopbackServiceTransport, matches, waitFor

Models = defineService("test.models")
QuestionDialogs = defineService("test.question-dialog")
Echo = defineService("test.echo")
Timeline = defineService("test.timeline")


async def noop(*_args):
    return None


def setField(state, key, value):
    state.change(BACKGROUND_CONTEXT, lambda draft: draft.__setitem__(key, value))


def modelsImplementation(revision=0, select=noop):
    return {"state": replicatedState({"selected": None, "revision": revision}), "select": select}


def test_defines_remote_and_local_services():
    assert defineService("test.json-passthrough").local is False
    assert defineService("test.local-non-json", {"local": True}).local is True


def test_marks_services_remotable_by_default_and_reserves_chord_service_ids():
    local = defineService("test.local", {"local": True})
    assert Models.local is False
    assert local.local is True
    with pytest.raises(TypeError, match=r"Service IDs beginning with \$chord\. are reserved"):
        defineService("$chord.internal", {"local": True})
    with pytest.raises(TypeError, match="cannot be published remotely"):
        RemoteServiceProvider([local])


def test_tracks_mutable_source_state_while_publishing_immutable_revisions():
    initial = {"selected": None, "revision": 0}
    state = replicatedState(initial)
    delivered = []
    deliveries = []

    def listener(value, _context, delivery):
        delivered.append(value)
        deliveries.append(delivery["kind"])

    unsubscribe = state.subscribe(listener)
    assert state.value is initial
    assert delivered[-1] is state.value
    hydrated = delivered[-1]

    def mutate(draft):
        draft["selected"] = {"provider": "test", "modelId": "one"}
        draft["revision"] = 1

    state.change(BACKGROUND_CONTEXT, mutate)
    assert state.value == {"selected": {"provider": "test", "modelId": "one"}, "revision": 1}
    assert state.value is not initial
    assert delivered[-1] is state.value
    assert hydrated == {"selected": None, "revision": 0}
    assert deliveries == ["hydrate", "update"]
    unsubscribe()


def test_does_not_publish_when_a_transaction_restores_the_prior_value():
    state = replicatedState({"value": 1})
    deliveries = []
    state.subscribe(lambda _value, _context, delivery: deliveries.append(dict(delivery)))

    def mutate(draft):
        draft["value"] = 2
        draft["value"] = 1

    state.change(BACKGROUND_CONTEXT, mutate)
    assert state.value == {"value": 1}
    assert deliveries == [{"kind": "hydrate", "sequence": 0}]


def test_hydrates_a_new_subscriber_from_the_latest_atomic_revision():
    state = replicatedState({"entries": [{"id": "one"}]})
    first = []
    state.subscribe(lambda value, _context, _delivery: first.append(value))
    state.change(BACKGROUND_CONTEXT, lambda draft: draft["entries"].push({"id": "two"}))

    second = []
    state.subscribe(lambda value, _context, _delivery: second.append(value))

    assert first == [{"entries": [{"id": "one"}]}, {"entries": [{"id": "one"}, {"id": "two"}]}]
    assert second == [{"entries": [{"id": "one"}, {"id": "two"}]}]


async def test_does_not_defensively_clone_method_arguments_or_results():
    provider = RemoteServiceProvider([Echo])
    received = []
    response = {"value": "response"}

    async def echoImpl(payload, _context):
        received.append(payload)
        return response

    provider.provide(Echo, {"echo": echoImpl})
    namespace = createRemoteServiceBinding({"services": [Echo], "transport": createLoopbackServiceTransport(provider)})
    echo = namespace.use(Echo)
    await namespace.ready(BACKGROUND_CONTEXT)
    request = {"value": "request"}

    assert await echo.echo(request, BACKGROUND_CONTEXT) is response
    assert received[0] is request

    await namespace.dispose(BACKGROUND_CONTEXT)
    provider.dispose()


async def test_provides_and_consumes_one_singleton_with_replicated_state():
    provider = RemoteServiceProvider([Models])
    assert provider.catalogue == [{"serviceId": Models.id, "mode": "singleton"}]
    initialState = {"selected": None, "revision": 0}
    state = replicatedState(initialState)
    publishedState = []

    async def select(model, context):
        def mutate(draft):
            draft["selected"] = model
            draft["revision"] += 1

        state.change(context, mutate)
        publishedState.append(state.value)

    provider.provide(Models, {"state": state, "select": select})
    errors = []
    namespace = createRemoteServiceBinding({
        "services": [Models],
        "transport": createLoopbackServiceTransport(provider),
        "onError": errors.append,
    })

    first = namespace.use(Models)
    second = namespace.use(Models)
    assert first is second
    assert first.state.value is UNDEFINED
    await namespace.ready(BACKGROUND_CONTEXT)
    assert first.state.value == initialState

    updates = []
    unsubscribe = second.state.subscribe(lambda value, _context, _delivery: updates.append(value))
    await first.select({"provider": "test", "modelId": "one"}, BACKGROUND_CONTEXT)
    assert first.state.value == publishedState[0]
    assert first.state.value == {"selected": {"provider": "test", "modelId": "one"}, "revision": 1}
    assert updates == [
        {"selected": None, "revision": 0},
        {"selected": {"provider": "test", "modelId": "one"}, "revision": 1},
    ]
    assert errors == []

    lateNamespace = createRemoteServiceBinding({
        "services": [Models],
        "transport": createLoopbackServiceTransport(provider),
    })
    lateModels = lateNamespace.use(Models)
    await lateNamespace.ready(BACKGROUND_CONTEXT)
    assert lateModels.state.value["revision"] == 1

    unsubscribe()
    await asyncio.gather(namespace.dispose(BACKGROUND_CONTEXT), lateNamespace.dispose(BACKGROUND_CONTEXT))
    provider.dispose()


async def test_publishes_compact_tracked_operations_through_the_remote_provider():
    provider = RemoteServiceProvider([Timeline])
    initial = {"entries": [{"id": "one"}], "retained": {"value": 1}}
    source = replicatedState(initial)
    provider.provide(Timeline, {"state": source})
    updates = []
    raw = provider.subscribe(Timeline.id, "singleton", lambda update, _context: updates.append(update))
    assert raw.snapshot["instances"][0]["members"] == [
        {"name": "state", "kind": "state", "sequence": 0, "ops": [["r", initial]]}
    ]
    raw.activate()

    namespace = createRemoteServiceBinding({
        "services": [Timeline],
        "transport": createLoopbackServiceTransport(provider),
    })
    timeline = namespace.use(Timeline)
    await namespace.ready(BACKGROUND_CONTEXT)
    previous = timeline.state.value
    nextValue = {"entries": [{"id": "one"}, {"id": "two"}], "retained": initial["retained"]}
    source.change(BACKGROUND_CONTEXT, lambda draft: draft["entries"].push({"id": "two"}))

    assert {
        "type": "state",
        "member": "state",
        "sequence": 1,
        "ops": [["p", ["entries"], 1, 0, [{"id": "two"}]]],
    } in updates
    assert previous == {"entries": [{"id": "one"}], "retained": {"value": 1}}
    assert timeline.state.value == nextValue

    source.change(BACKGROUND_CONTEXT, lambda draft: draft["entries"].push({"id": "three"}))
    late = provider.subscribe(Timeline.id, "singleton", lambda _update, _context: None)
    assert updates[-1] == {
        "type": "state",
        "member": "state",
        "sequence": 2,
        "ops": [["p", ["entries"], 2, 0, [{"id": "three"}]]],
    }
    assert late.snapshot["instances"][0]["members"] == [
        {
            "name": "state",
            "kind": "state",
            "sequence": 2,
            "ops": [["r", {"entries": [{"id": "one"}, {"id": "two"}, {"id": "three"}], "retained": {"value": 1}}]],
        }
    ]
    late.close()
    raw.close()
    await namespace.dispose(BACKGROUND_CONTEXT)
    provider.dispose()


def test_publishes_authoritative_source_references_through_services_without_re_diffing():
    initial = {"entries": [{"id": "one"}], "retained": {"value": 1}}
    holder = {"disposed": False}

    class Attachment:
        def __init__(self):
            self.snapshot = {"value": initial, "cursor": 20}

        def activate(self, listener):
            holder["publish"] = listener

        def dispose(self):
            holder["disposed"] = True

    class Source:
        def attach(self):
            return Attachment()

    state = replicatedState(Source())
    provider = RemoteServiceProvider([Timeline])
    provider.provide(Timeline, {"state": state})
    updates = []
    subscription = provider.subscribe(Timeline.id, "singleton", lambda update, _context: updates.append(update))
    snapshotMember = subscription.snapshot["instances"][0]["members"][0]
    assert snapshotMember["kind"] == "state"
    assert snapshotMember["ops"][0][1] is initial
    subscription.activate()

    nextValue = {"entries": [{"id": "one"}, {"id": "two"}], "retained": initial["retained"]}
    ops = [["p", ["entries"], 1, 0, [{"id": "two"}]]]
    holder["publish"]({"cursor": 21, "value": nextValue, "ops": ops, "context": BACKGROUND_CONTEXT})
    assert len(updates) == 1
    assert updates[0]["type"] == "state"
    assert updates[0]["ops"] is ops

    subscription.close()
    provider.dispose()
    state.dispose()
    assert holder["disposed"] is True


async def test_keeps_singleton_facades_stable_when_their_provider_is_replaced():
    provider = RemoteServiceProvider([Models])
    provider.provide(Models, modelsImplementation(1))
    namespace = createRemoteServiceBinding({"services": [Models], "transport": createLoopbackServiceTransport(provider)})
    models = namespace.use(Models)
    state = models.state
    select = models.select
    await namespace.ready(BACKGROUND_CONTEXT)
    assert state.value["revision"] == 1

    provider.withdraw(Models)
    assert state.value is UNDEFINED
    with pytest.raises(Exception, match="no provider") as raised:
        await select({"provider": "test", "modelId": "unavailable"}, BACKGROUND_CONTEXT)
    assert raised.value.code == "service_not_found"

    replacementCalls = []

    async def replacementSelect(*args):
        replacementCalls.append(args)

    provider.replace(Models, modelsImplementation(2, replacementSelect))

    assert namespace.use(Models) is models
    assert models.state is state
    assert state.value["revision"] == 2
    await select({"provider": "test", "modelId": "replacement"}, BACKGROUND_CONTEXT)
    assert len(replacementCalls) == 1
    with pytest.raises(Exception, match="replacement must preserve its member shape"):
        provider.replace(Models, {"select": noop})
    with pytest.raises(Exception, match="replacement must preserve its member shape"):
        provider.replace(Models, {"state": noop, "select": noop})
    assert state.value["revision"] == 2

    await namespace.dispose(BACKGROUND_CONTEXT)
    provider.dispose()


def test_delivers_active_subscriber_updates_before_reporting_listener_failures():
    failure = RuntimeError("listener failed")
    provider = RemoteServiceProvider([Models])
    provider.provide(Models, modelsImplementation(1))
    delivered = [0]

    def failingListener(_update, _context):
        raise failure

    def succeedingListener(_update, _context):
        delivered[0] += 1

    failing = provider.subscribe(Models.id, "singleton", failingListener)
    succeeding = provider.subscribe(Models.id, "singleton", succeedingListener)
    failing.activate()
    succeeding.activate()

    with pytest.raises(RuntimeError) as raised:
        provider.replace(Models, modelsImplementation(2))
    assert raised.value is failure
    assert delivered[0] == 1

    failing.close()
    succeeding.close()
    provider.dispose()


def test_does_not_replay_a_queued_revision_already_covered_by_a_new_subscription_snapshot():
    provider = RemoteServiceProvider([Models])
    state = replicatedState({"selected": None, "revision": 0})
    provider.provide(Models, {"state": state, "select": noop})
    lateUpdates = []
    late = []

    def onFirst(update, _context):
        if update["type"] != "state" or update["sequence"] != 1:
            return
        setField(state, "revision", 2)
        late.append(
            provider.subscribe(
                Models.id,
                "singleton",
                lambda nextUpdate, _context: nextUpdate["type"] == "state" and lateUpdates.append(nextUpdate["sequence"]),
            )
        )
        assert {
            "name": "state",
            "kind": "state",
            "sequence": 2,
            "ops": [["r", {"selected": None, "revision": 2}]],
        } in late[0].snapshot["instances"][0]["members"]
        late[0].activate()

    first = provider.subscribe(Models.id, "singleton", onFirst)
    first.activate()
    setField(state, "revision", 1)
    assert lateUpdates == []
    setField(state, "revision", 3)
    assert lateUpdates == [3]
    late[0].close()
    first.close()
    provider.dispose()


def test_replays_every_buffered_update_before_reporting_listener_failures():
    failure = RuntimeError("listener failed")
    provider = RemoteServiceProvider([Models])
    state = replicatedState({"selected": None, "revision": 0})
    provider.provide(Models, {"state": state, "select": noop})
    delivered = [0]

    def listener(_update, _context):
        delivered[0] += 1
        raise failure

    subscription = provider.subscribe(Models.id, "singleton", listener)
    setField(state, "revision", 1)
    setField(state, "revision", 2)

    with pytest.raises(ExceptionGroup, match="Failed to activate remote service subscription"):
        subscription.activate()
    assert delivered[0] == 2
    subscription.close()
    provider.dispose()


async def test_clears_retained_facades_when_providers_and_bindings_are_disposed():
    provider = RemoteServiceProvider([Models])
    provider.provide(Models, modelsImplementation(1))
    namespace = createRemoteServiceBinding({"services": [Models], "transport": createLoopbackServiceTransport(provider)})
    models = namespace.use(Models)
    state = models.state
    await namespace.ready(BACKGROUND_CONTEXT)
    assert state.value["revision"] == 1

    provider.dispose()
    assert state.value is UNDEFINED
    with pytest.raises(Exception, match="Remote service provider is disposed"):
        await models.select({"provider": "test", "modelId": "one"}, BACKGROUND_CONTEXT)

    await namespace.dispose(BACKGROUND_CONTEXT)
    with pytest.raises(Exception, match="Remote service binding is disposed"):
        state.value  # noqa: B018


async def test_applies_provider_disposal_buffered_while_subscriptions_are_starting():
    provider = RemoteServiceProvider([Models, {"service": QuestionDialogs, "mode": "keyed"}])
    provider.provide(Models, modelsImplementation(1))

    async def submit(*_args):
        return {"accepted": True}

    provider.spawn(QuestionDialogs, "pending", {"request": replicatedState({"question": "Pending?"}), "submit": submit})
    namespace = createRemoteServiceBinding({
        "services": [Models, QuestionDialogs],
        "transport": createLoopbackServiceTransport(provider),
    })
    models = namespace.use(Models)
    observed = []
    namespace.observe(QuestionDialogs, lambda service, _context: observed.append(service))

    provider.dispose()
    await namespace.ready(BACKGROUND_CONTEXT)
    assert models.state.value is UNDEFINED
    assert observed == []

    await namespace.dispose(BACKGROUND_CONTEXT)


async def test_keeps_deferred_service_handles_inaccessible_until_host_activation():
    provider = RemoteServiceProvider([Models])
    provider.provide(Models, modelsImplementation(0))
    active = [False]

    def assertAccess():
        if not active[0]:
            raise RuntimeError("Service handles are not active")

    namespace = createRemoteServiceBinding({
        "services": [Models],
        "transport": createLoopbackServiceTransport(provider),
        "bound": False,
        "assertAccess": assertAccess,
    })
    models = namespace.use(Models)

    with pytest.raises(RuntimeError, match="Service handles are not active"):
        models.state.value  # noqa: B018
    with pytest.raises(RuntimeError, match="Service handles are not active"):
        models.state.subscribe(lambda *_args: None)
    with pytest.raises(RuntimeError, match="Service handles are not active"):
        models.select({"provider": "test", "modelId": "one"}, BACKGROUND_CONTEXT)

    await namespace.rebind(True, BACKGROUND_CONTEXT)
    active[0] = True
    await namespace.ready(BACKGROUND_CONTEXT)
    assert models.state.value["revision"] == 0

    await namespace.dispose(BACKGROUND_CONTEXT)
    provider.dispose()


async def test_rejects_namespace_readiness_when_initial_hydration_fails():
    failure = RuntimeError("initial hydration failed")
    errors = []

    class FailingTransport:
        async def invoke(self, call, context):
            raise RuntimeError("unexpected invocation")

        async def subscribe(self, serviceId, mode, listener, context):
            raise failure

    namespace = createRemoteServiceBinding({
        "services": [Models],
        "transport": FailingTransport(),
        "onError": errors.append,
    })
    models = namespace.use(Models)

    with pytest.raises(RuntimeError) as raised:
        await namespace.ready(BACKGROUND_CONTEXT)
    assert raised.value is failure
    assert models.state.value is UNDEFINED
    assert errors == [failure]
    await namespace.dispose(BACKGROUND_CONTEXT)


async def test_buffers_state_updates_that_race_subscription_hydration():
    provider = RemoteServiceProvider([Models])
    state = replicatedState({"selected": None, "revision": 0})
    provider.provide(Models, {"state": state, "select": noop})

    class RacingTransport:
        def invoke(self, call, context):
            return provider.invoke(call, context)

        async def subscribe(self, serviceId, mode, listener, context):
            subscription = provider.subscribe(serviceId, mode, listener)
            setField(state, "revision", 1)

            class Raced:
                snapshot = subscription.snapshot

                def activate(self):
                    subscription.activate()

                def close(self, context=None):
                    return subscription.close()

            return Raced()

    namespace = createRemoteServiceBinding({"services": [Models], "transport": RacingTransport()})
    models = namespace.use(Models)
    revisions = []
    models.state.subscribe(lambda value, _context, _delivery: revisions.append(value["revision"]))

    await waitFor(lambda: revisions == [0, 1])
    assert models.state.value["revision"] == 1
    await namespace.dispose(BACKGROUND_CONTEXT)
    provider.dispose()


@pytest.mark.parametrize(("kind", "sequence"), [("duplicate", 0), ("gap", 2)])
async def test_clears_replicated_state_after_a_bad_operation_sequence(kind, sequence):
    holder = {}
    errors = []

    class ScriptedTransport:
        async def invoke(self, call, context):
            raise RuntimeError("unexpected invocation")

        async def subscribe(self, serviceId, mode, listener, context):
            holder["send"] = lambda update: listener(update, BACKGROUND_CONTEXT)

            class Scripted:
                snapshot: ClassVar = {
                    "serviceId": Models.id,
                    "mode": "singleton",
                    "instances": [
                        {
                            "members": [
                                {"name": "select", "kind": "method"},
                                {"name": "state", "kind": "state", "sequence": 0, "ops": [["r", {"selected": None, "revision": 0}]]},
                            ]
                        }
                    ],
                }

                def activate(self):
                    pass

                def close(self, context=None):
                    pass

            return Scripted()

    namespace = createRemoteServiceBinding({
        "services": [Models],
        "transport": ScriptedTransport(),
        "onError": errors.append,
    })
    models = namespace.use(Models)
    await namespace.ready(BACKGROUND_CONTEXT)
    assert models.state.value["revision"] == 0

    holder["send"]({
        "type": "state",
        "member": "state",
        "sequence": sequence,
        "ops": [["r", {"selected": None, "revision": sequence}]],
    })
    assert models.state.value is UNDEFINED
    assert len(errors) == 1
    assert "sequence has a gap" in str(errors[0])
    await namespace.dispose(BACKGROUND_CONTEXT)


async def test_hydrates_cold_replicated_state_replicas_and_replaces_them_across_rebinds():
    provider = RemoteServiceProvider([Models])
    state = replicatedState({"selected": None, "revision": 0})
    provider.provide(Models, {"state": state, "select": noop})
    namespace = createRemoteServiceBinding({
        "services": [Models],
        "transport": createLoopbackServiceTransport(provider),
        "bound": False,
    })
    models = namespace.use(Models)
    revisions = []
    models.state.subscribe(lambda value, _context, _delivery: revisions.append(value["revision"]))
    assert models.state.value is UNDEFINED
    assert revisions == []

    setField(state, "revision", 1)
    await namespace.rebind(True, BACKGROUND_CONTEXT)
    assert models.state.value["revision"] == 1
    assert revisions == [1]
    setField(state, "revision", 2)
    assert revisions == [1, 2]

    await namespace.rebind(False, BACKGROUND_CONTEXT)
    assert models.state.value is UNDEFINED
    setField(state, "revision", 3)
    assert revisions == [1, 2]
    await namespace.rebind(True, BACKGROUND_CONTEXT)
    assert models.state.value["revision"] == 3
    assert revisions == [1, 2, 3]

    await namespace.dispose(BACKGROUND_CONTEXT)
    provider.dispose()


async def test_hydrates_keyed_state_before_observe_handlers_and_fences_reused_keys():
    provider = RemoteServiceProvider([{"service": QuestionDialogs, "mode": "keyed"}])
    assert provider.catalogue == [{"serviceId": QuestionDialogs.id, "mode": "keyed"}]
    errors = []
    namespace = createRemoteServiceBinding({
        "services": [QuestionDialogs],
        "transport": createLoopbackServiceTransport(provider),
        "onError": errors.append,
    })
    observed = []

    def onObserved(service, context):
        observed.append({"question": service.request.value, "service": service, "context": context})

    stop = namespace.observe(QuestionDialogs, onObserved)
    await waitFor(lambda: errors == [])

    firstRequest = replicatedState({"question": "First?"})
    submitCalls = []

    async def firstSubmit(*args):
        submitCalls.append(args)
        return {"accepted": True}

    closeFirst = provider.spawn(QuestionDialogs, "invocation-1", {"request": firstRequest, "submit": firstSubmit})
    await waitFor(lambda: len(observed) == 1)
    assert matches(observed[0], {"question": {"question": "First?"}})

    firstService = observed[0]["service"]
    setField(firstRequest, "question", "Updated?")
    assert firstService.request.value == {"question": "Updated?"}
    assert await firstService.submit("yes", BACKGROUND_CONTEXT) == {"accepted": True}
    assert submitCalls[0][0] == "yes"
    assert submitCalls[0][1].abortSignal is None

    retainedFirstSubmit = firstService.submit
    closeFirst()
    assert observed[0]["context"].abortSignal.aborted is True
    with pytest.raises(Exception, match="observation is closed"):
        firstService.request.value  # noqa: B018
    with pytest.raises(Exception, match="observation is closed"):
        retainedFirstSubmit("late", BACKGROUND_CONTEXT)

    secondRequest = replicatedState({"question": "Again?"})

    async def secondSubmit(*_args):
        return {"accepted": False}

    closeSecond = provider.spawn(QuestionDialogs, "invocation-1", {"request": secondRequest, "submit": secondSubmit})
    await waitFor(lambda: len(observed) == 2)
    assert matches(observed[1], {"question": {"question": "Again?"}})
    assert observed[1]["service"] is not firstService
    assert errors == []

    secondService = observed[1]["service"]
    retainedSecondSubmit = secondService.submit
    stop()
    assert observed[1]["context"].abortSignal.aborted is True
    with pytest.raises(Exception, match="observation is closed"):
        secondService.request.value  # noqa: B018
    with pytest.raises(Exception, match="observation is closed"):
        retainedSecondSubmit("late", BACKGROUND_CONTEXT)
    closeSecond()
    await namespace.dispose(BACKGROUND_CONTEXT)
    provider.dispose()


def test_rejects_mode_mixing_and_unsupported_members():
    provider = RemoteServiceProvider([Models, {"service": QuestionDialogs, "mode": "keyed"}])
    provider.provide(Models, modelsImplementation(0))
    with pytest.raises(Exception, match="singleton"):
        provider.spawn(Models, "wrong", {})

    async def submit(*_args):
        return {"accepted": True}

    with pytest.raises(TypeError, match="not remotely exposable"):
        provider.spawn(
            QuestionDialogs,
            "invalid",
            {"request": datetime.datetime.now(), "submit": submit},
        )
    provider.dispose()
