"""Regression tests for what the adversarial review of the chord port found."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from misaka.chord import (
    RemoteServiceProvider,
    createFacetHost,
    createRemoteServiceBinding,
    defineFacet,
    defineService,
    replicatedState,
)
from misaka.chord.context import BACKGROUND_CONTEXT
from misaka.chord.delta import apply, track
from misaka.chord.json import isJsonValue
from misaka.utils.js import UNDEFINED
from tests.chord_helpers import createLoopbackServiceTransport, waitFor

Models = defineService("test.review.models")
Dialogs = defineService("test.review.dialogs")
Shared = defineService("test.review.shared", local=True)


def test_the_default_sort_compares_utf16_code_units():
    tracker = track({"a": ["\U0001f600", "！", ""]})
    change = tracker.beginChange()
    change.state["a"].sort()
    prepared = change.prepare()
    # An astral character is a surrogate pair (0xD83D...), which sorts before U+E000 and U+FF01.
    assert prepared.value["a"] == ["\U0001f600", "", "！"]
    assert prepared.ops == [["m", ["a"], [0, 2, 1]]]
    change.abort()


def test_object_deletions_come_out_in_the_order_they_were_made():
    tracker = track({"alpha": 1, "beta": 2, "gamma": 3, "delta": 4, "x": {"k": 1}})
    change = tracker.beginChange()
    for key in ("alpha", "beta", "gamma", "delta", "x"):
        change.state[key] = UNDEFINED
    prepared = change.prepare()
    assert prepared.ops == [["d", [key]] for key in ("alpha", "beta", "gamma", "delta", "x")]
    change.abort()


def test_a_numeric_segment_names_the_key_javascript_would():
    assert apply({"a": {}}, [["s", ["a", 1e21], 1]]) == {"a": {"1e+21": 1}}
    assert apply({"a": {"1e+21": "x"}}, [["a", ["a", 1e21], "y"]]) == {"a": {"1e+21": "xy"}}
    with pytest.raises(Exception, match="unsafe path segment: 1e\\+21"):
        apply({"a": []}, [["s", ["a", 1e21], 1]])


def test_delta_errors_print_values_as_string_does():
    with pytest.raises(Exception, match="unsafe path segment: null"):
        apply({"a": {}}, [["s", ["a", None], 1]])
    with pytest.raises(Exception, match="unsafe path segment: true"):
        apply({"a": {}}, [["s", ["a", True], 1]])
    with pytest.raises(Exception, match="unknown op verb: null"):
        apply({"a": {}}, [[None, ["a"]]])
    with pytest.raises(Exception, match="unknown op verb: 1,2"):
        apply({"a": {}}, [[[1, 2], ["a"], 1]])


def test_a_draft_of_json_is_json():
    tracker = track({"nested": {"rows": [{"n": 1}]}, "text": "a"})
    change = tracker.beginChange()
    change.state["nested"]["rows"].push({"n": 2})
    assert isJsonValue(change.state) is True
    assert isJsonValue(change.state["nested"]["rows"]) is True
    change.abort()


def facet(id_, setup):
    return defineFacet(SimpleNamespace(id=id_, setup=setup))


async def test_facets_activate_in_the_order_they_were_given():
    trace = []

    def setupProvider(env):
        env.provide(Shared, {"read": lambda *_: 1})
        env.onActivate(lambda: trace.append("a"))

    def consumer(name):
        def setup(env):
            env.use(Shared)
            env.onActivate(lambda: trace.append(name))

        return setup

    names = ["b", "c", "d", "e", "f", "g"]
    host = await createFacetHost(
        {"facets": [facet("a", setupProvider), *(facet(name, consumer(name)) for name in names)]}
    )
    assert trace == ["a", *names]
    await host.dispose()


async def test_overlapping_host_operations_are_rejected_by_the_phase_guards():
    def setup(env):
        env.provide(Shared, {"read": lambda *_: 1})

    host = await createFacetHost({"facets": [facet("a", setup)]})
    results = await asyncio.gather(host.reload([facet("a", setup)]), host.dispose(), return_exceptions=True)
    assert results[0] is None
    assert isinstance(results[1], RuntimeError)
    assert "cannot be disposed while reloading" in str(results[1])
    await host.dispose()

    host = await createFacetHost({"facets": [facet("a", setup)]})
    results = await asyncio.gather(
        host.reload([facet("a", setup)]), host.reload([facet("a", setup)]), return_exceptions=True
    )
    assert results[0] is None
    assert "cannot reload while reloading" in str(results[1])
    await host.dispose()


async def test_a_function_is_not_a_service_implementation():
    def setup(env):
        env.provide(Shared, lambda: 1)

    with pytest.raises(Exception, match="implementation must be an object"):
        await createFacetHost({"facets": [facet("a", setup)]})


class SlowTransport:
    def __init__(self, inner, delay=0.05):
        self._inner = inner
        self._delay = delay

    def invoke(self, call, context):
        return self._inner.invoke(call, context)

    async def subscribe(self, serviceId, mode, listener, context):
        await asyncio.sleep(self._delay)
        return await self._inner.subscribe(serviceId, mode, listener, context)


async def test_abandoning_the_wait_for_readiness_does_not_cancel_the_binding_start():
    provider = RemoteServiceProvider([{"service": Models, "mode": "singleton"}])
    provider.provide(Models, {"state": replicatedState({"revision": 7}), "select": lambda *_: None})
    errors = []
    namespace = createRemoteServiceBinding(
        {
            "services": [Models],
            "transport": SlowTransport(createLoopbackServiceTransport(provider)),
            "onError": errors.append,
        }
    )
    models = namespace.use(Models)
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(namespace.ready(BACKGROUND_CONTEXT), 0.01)
    await namespace.ready(BACKGROUND_CONTEXT)
    assert models.state.value == {"revision": 7}
    await namespace.dispose(BACKGROUND_CONTEXT)
    provider.dispose()
    assert errors == []


async def test_disposing_a_binding_closes_its_keyed_bindings_at_once():
    provider = RemoteServiceProvider([{"service": Dialogs, "mode": "keyed"}])
    provider.spawn(Dialogs, "pending", {"request": replicatedState({"question": "?"}), "submit": lambda *_: None})
    namespace = createRemoteServiceBinding(
        {"services": [Dialogs], "transport": createLoopbackServiceTransport(provider), "onError": lambda error: None}
    )
    observed = []
    namespace.observe(Dialogs, lambda service, context: observed.append(service))
    await namespace.dispose(BACKGROUND_CONTEXT)
    await asyncio.sleep(0.05)
    assert observed == []
    provider.dispose()


async def test_a_provider_disposes_when_a_subscriber_closes_another_instance():
    provider = RemoteServiceProvider([{"service": Dialogs, "mode": "keyed"}])
    implementation = {"request": replicatedState({"question": "?"}), "submit": lambda *_: None}
    holder = {}
    holder["closeA"] = provider.spawn(Dialogs, "a", implementation)
    holder["closeB"] = provider.spawn(
        Dialogs, "b", {"request": replicatedState({"question": "?"}), "submit": lambda *_: None}
    )
    transport = createLoopbackServiceTransport(provider)
    seen = []

    def listener(update, context):
        seen.append(update["type"])
        if update["type"] == "closed" and len(seen) < 6:
            holder["closeB"]()

    subscription = await transport.subscribe(Dialogs.id, "keyed", listener, BACKGROUND_CONTEXT)
    subscription.activate()
    await waitFor(lambda: True)
    provider.dispose()
