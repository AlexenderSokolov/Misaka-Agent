"""pi chord's ``test/service-wire.test.ts``."""

from __future__ import annotations

import pytest

from misaka.chord import (
    RemoteServiceProvider,
    createRemoteServiceEndpoint,
    createServiceCatalogueCall,
    createServiceStateDecoder,
    createServiceStateEncoder,
    createServiceSubscribeCall,
    createServiceUnsubscribeCall,
    decodeServiceControlCall,
    defineService,
    parseServiceCall,
    parseServiceCatalogue,
    parseServiceProviderUpdate,
    parseServiceSubscriptionSnapshot,
    parseWireServiceProviderUpdate,
    parseWireServiceSubscriptionSnapshot,
    replicatedState,
)
from misaka.chord.context import BACKGROUND_CONTEXT


def matches(actual, expected) -> bool:
    """``toMatchObject``: every expected key is present and matches, recursively."""
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(key in actual and matches(actual[key], value) for key, value in expected.items())
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(matches(a, e) for a, e in zip(actual, expected, strict=True))
    return actual == expected


def test_encodes_control_calls_and_validates_service_values():
    assert decodeServiceControlCall(createServiceCatalogueCall()) == {"type": "catalogue"}
    catalogue = [{"serviceId": "pi.models", "mode": "singleton"}, {"serviceId": "pi.dialogs", "mode": "keyed"}]
    assert parseServiceCatalogue(catalogue) == catalogue
    subscribe = createServiceSubscribeCall("subscription-1", "pi.models", "singleton")
    assert decodeServiceControlCall(subscribe) == {
        "type": "subscribe",
        "subscriptionId": "subscription-1",
        "serviceId": "pi.models",
        "mode": "singleton",
    }
    assert decodeServiceControlCall(createServiceUnsubscribeCall("subscription-1")) == {
        "type": "unsubscribe",
        "subscriptionId": "subscription-1",
    }
    call = parseServiceCall({
        "serviceId": "pi.question-dialog",
        "instance": {"key": "invocation-1", "generation": 2},
        "member": "submit",
        "args": [{"outcome": "selected", "index": 0}],
    })
    assert call["member"] == "submit"


def test_rejects_malformed_service_values():
    with pytest.raises(TypeError, match="Invalid service call"):
        parseServiceCall({"serviceId": "pi.models", "member": "list", "args": [], "extra": True})
    with pytest.raises(TypeError, match="Invalid service catalogue"):
        parseServiceCatalogue([{"serviceId": "pi.models", "mode": "unknown"}])
    with pytest.raises(TypeError, match="Invalid service state update"):
        parseServiceProviderUpdate({"type": "state", "member": "state", "sequence": 0, "ops": []})
    with pytest.raises(Exception):  # noqa: B017
        parseWireServiceProviderUpdate({"type": "state", "member": "state", "sequence": 1, "ops": [["?", 0]]})


def test_validates_decoded_and_wire_snapshots_and_updates():
    snapshot = {
        "serviceId": "pi.models",
        "mode": "singleton",
        "instances": [{"members": [{"name": "state", "kind": "state", "sequence": 0, "ops": [["r", {"revision": 1}]]}]}],
    }
    assert parseServiceSubscriptionSnapshot(snapshot) is snapshot
    encoder = createServiceStateEncoder()
    wireSnapshot = encoder.encodeSnapshot(snapshot)
    assert parseWireServiceSubscriptionSnapshot(wireSnapshot) is wireSnapshot
    update = {"type": "state", "member": "state", "sequence": 1, "ops": [["s", ["revision"], 2]]}
    assert parseServiceProviderUpdate(update) is update
    wireUpdate = encoder.encodeUpdate(update)
    assert parseWireServiceProviderUpdate(wireUpdate) is wireUpdate


def test_keeps_one_operation_codec_pair_for_one_subscription_state():
    enc = createServiceStateEncoder()
    dec = createServiceStateDecoder()
    snapshot = {
        "serviceId": "pi.models",
        "mode": "singleton",
        "instances": [{"members": [{"name": "state", "kind": "state", "sequence": 0, "ops": [["r", {"revision": 0}]]}]}],
    }
    assert dec.decodeSnapshot(enc.encodeSnapshot(snapshot)) == snapshot

    first = {"type": "state", "member": "state", "sequence": 1, "ops": [["s", ["revision"], 1]]}
    second = {"type": "state", "member": "state", "sequence": 2, "ops": [["s", ["revision"], 2]]}
    firstWire = enc.encodeUpdate(first)
    secondWire = enc.encodeUpdate(second)
    assert matches(firstWire, {"ops": [["s", ["revision"], 1]]})
    assert matches(secondWire, {"ops": [["#", 0, ["revision"]], ["s", 0, 2]]})
    assert dec.decodeUpdate(firstWire) == first
    assert dec.decodeUpdate(secondWire) == second


def test_validates_explicit_resets_and_restarts_path_dictionaries_at_the_new_baseline():
    enc = createServiceStateEncoder()
    dec = createServiceStateDecoder()
    snapshot = {
        "serviceId": "pi.states",
        "mode": "singleton",
        "instances": [{"members": [{"name": "state", "kind": "state", "sequence": 0, "ops": [["r", {"before": 0}]]}]}],
    }
    dec.decodeSnapshot(enc.encodeSnapshot(snapshot))
    for sequence in range(1, 3):
        dec.decodeUpdate(
            enc.encodeUpdate({"type": "state", "member": "state", "sequence": sequence, "ops": [["s", ["before"], sequence]]})
        )
    reset = {
        "type": "reset",
        "snapshot": {
            **snapshot,
            "instances": [{"members": [{"name": "state", "kind": "state", "sequence": 103, "ops": [["r", {"after": 103}]]}]}],
        },
    }
    assert parseServiceProviderUpdate(reset) is reset
    wireReset = enc.encodeUpdate(reset)
    assert parseWireServiceProviderUpdate(wireReset) is wireReset
    assert dec.decodeUpdate(wireReset) == reset
    for sequence in range(104, 107):
        update = {"type": "state", "member": "state", "sequence": sequence, "ops": [["s", ["after"], sequence]]}
        assert dec.decodeUpdate(enc.encodeUpdate(update)) == update
    for parse in (parseServiceProviderUpdate, parseWireServiceProviderUpdate):
        with pytest.raises(Exception):  # noqa: B017
            parse({**reset, "extra": True})
        with pytest.raises(Exception):  # noqa: B017
            parse({"type": "reset", "snapshot": {}})
        with pytest.raises(Exception, match="full root replacements"):
            parse({
                "type": "reset",
                "snapshot": {
                    **snapshot,
                    "instances": [
                        {"members": [{"name": "state", "kind": "state", "sequence": 103, "ops": [["s", ["before"], 103]]}]}
                    ],
                },
            })


def test_isolates_operation_dictionaries_between_states_and_subscriptions():
    snapshot = {
        "serviceId": "pi.states",
        "mode": "singleton",
        "instances": [
            {
                "members": [
                    {"name": "left", "kind": "state", "sequence": 0, "ops": [["r", {"revision": 0}]]},
                    {"name": "right", "kind": "state", "sequence": 0, "ops": [["r", {"revision": 0}]]},
                ]
            }
        ],
    }
    firstEncoder, firstDecoder = createServiceStateEncoder(), createServiceStateDecoder()
    secondEncoder, secondDecoder = createServiceStateEncoder(), createServiceStateDecoder()
    firstDecoder.decodeSnapshot(firstEncoder.encodeSnapshot(snapshot))
    secondDecoder.decodeSnapshot(secondEncoder.encodeSnapshot(snapshot))

    def update(member, sequence, revision):
        return {"type": "state", "member": member, "sequence": sequence, "ops": [["s", ["revision"], revision]]}

    firstLeft = firstEncoder.encodeUpdate(update("left", 1, 1))
    firstRight = firstEncoder.encodeUpdate(update("right", 1, 1))
    secondLeft = firstEncoder.encodeUpdate(update("left", 2, 2))
    secondRight = firstEncoder.encodeUpdate(update("right", 2, 2))
    assert matches(firstLeft, {"ops": [["s", ["revision"], 1]]})
    assert matches(firstRight, {"ops": [["s", ["revision"], 1]]})
    assert matches(secondLeft, {"ops": [["#", 0, ["revision"]], ["s", 0, 2]]})
    assert matches(secondRight, {"ops": [["#", 0, ["revision"]], ["s", 0, 2]]})
    assert firstDecoder.decodeUpdate(firstLeft) == update("left", 1, 1)
    assert firstDecoder.decodeUpdate(firstRight) == update("right", 1, 1)
    assert firstDecoder.decodeUpdate(secondLeft) == update("left", 2, 2)
    assert firstDecoder.decodeUpdate(secondRight) == update("right", 2, 2)

    independentLeft = secondEncoder.encodeUpdate(update("left", 1, 1))
    assert matches(independentLeft, {"ops": [["s", ["revision"], 1]]})
    assert secondDecoder.decodeUpdate(independentLeft) == update("left", 1, 1)

    leftBase = {"type": "state", "member": "left", "sequence": 3, "ops": [["r", {"revision": 3}]]}
    assert firstDecoder.decodeUpdate(firstEncoder.encodeUpdate(leftBase)) == leftBase
    thirdRight = firstEncoder.encodeUpdate(update("right", 3, 3))
    assert matches(thirdRight, {"ops": [["s", 0, 3]]})
    assert firstDecoder.decodeUpdate(thirdRight) == update("right", 3, 3)


def test_creates_and_removes_keyed_instance_codecs_with_their_lifecycle():
    enc = createServiceStateEncoder()
    dec = createServiceStateDecoder()
    snapshot = {"serviceId": "pi.dialogs", "mode": "keyed", "instances": []}
    assert dec.decodeSnapshot(enc.encodeSnapshot(snapshot)) == snapshot
    address = {"key": "dialog-1", "generation": 1}
    spawned = {
        "type": "spawned",
        "instance": {
            "instance": address,
            "members": [{"name": "request", "kind": "state", "sequence": 0, "ops": [["r", {"value": 0}]]}],
        },
    }
    assert dec.decodeUpdate(enc.encodeUpdate(spawned)) == spawned
    update = {"type": "state", "instance": address, "member": "request", "sequence": 1, "ops": [["s", ["value"], 1]]}
    assert dec.decodeUpdate(enc.encodeUpdate(update)) == update
    closed = {"type": "closed", "instance": address}
    assert dec.decodeUpdate(enc.encodeUpdate(closed)) == closed
    with pytest.raises(Exception, match="Unknown service state"):
        enc.encodeUpdate({**update, "sequence": 2})


Counter = defineService("test.counter")


async def test_remote_service_endpoints_publish_and_clean_up_provider_subscriptions():
    provider = RemoteServiceProvider([Counter])
    state = replicatedState({"value": 0})
    provider.provide(Counter, {"state": state})
    endpoint = createRemoteServiceEndpoint(provider)
    updates = []

    def publish(_subscriptionId, update, _context):
        updates.append(update)

    assert await endpoint.invoke(createServiceCatalogueCall(), publish, BACKGROUND_CONTEXT) == [
        {"serviceId": Counter.id, "mode": "singleton"}
    ]
    subscribed = await endpoint.invoke(
        createServiceSubscribeCall("subscription-1", Counter.id, "singleton"), publish, BACKGROUND_CONTEXT
    )
    assert matches(subscribed, {"serviceId": Counter.id, "mode": "singleton"})

    state.change(BACKGROUND_CONTEXT, lambda draft: draft.__setitem__("value", 1))
    assert updates == [{"type": "state", "member": "state", "sequence": 1, "ops": [["s", ["value"], 1]]}]
    endpoint.dispose()
    state.change(BACKGROUND_CONTEXT, lambda draft: draft.__setitem__("value", 2))
    assert len(updates) == 1
    provider.dispose()
