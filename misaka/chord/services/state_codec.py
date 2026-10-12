"""Stateful operation codecs for the replicated states of one service subscription.

Translated from pi's ``packages/chord/src/services/state-codec.ts``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from misaka.chord.delta import Decoder, Encoder, decoder, encoder
from misaka.chord.types import (
    ServiceInstanceAddress,
    ServiceProviderUpdate,
    ServiceSubscriptionSnapshot,
)
from misaka.utils.js import jsonStringify

from .wire import WireServiceProviderUpdate, WireServiceSubscriptionSnapshot


class ServiceStateEncoder:
    """Stateful operation encoders for every replicated state in one service subscription."""

    def __init__(self) -> None:
        self._codecs: StateCodecRegistry[Encoder] = StateCodecRegistry(encoder)

    def encodeSnapshot(self, snapshot: ServiceSubscriptionSnapshot) -> WireServiceSubscriptionSnapshot:
        self._codecs.reset()
        return {**snapshot, "instances": [encodeInstance(instance, self._codecs) for instance in snapshot["instances"]]}

    def encodeUpdate(self, update: ServiceProviderUpdate) -> WireServiceProviderUpdate:
        codecs = self._codecs
        match update["type"]:
            case "reset":
                codecs.reset()
                snapshot = update["snapshot"]
                return {
                    **update,
                    "snapshot": {
                        **snapshot,
                        "instances": [encodeInstance(instance, codecs) for instance in snapshot["instances"]],
                    },
                }
            case "state":
                return {**update, "ops": codecs.get(update.get("instance"), update["member"]).encode(update["ops"])}
            case "replaced":
                codecs.reset()
                return {**update, "snapshot": encodeInstance(update["snapshot"], codecs)}
            case "spawned":
                return {**update, "instance": encodeInstance(update["instance"], codecs)}
            case "unavailable":
                codecs.reset()
                return update  # type: ignore[return-value]
            case "closed":
                codecs.removeInstance(update["instance"])
                return update  # type: ignore[return-value]
            case _:
                raise TypeError("Invalid service provider update")


class ServiceStateDecoder:
    """Stateful operation decoders for every replicated state in one service subscription."""

    def __init__(self) -> None:
        self._codecs: StateCodecRegistry[Decoder] = StateCodecRegistry(decoder)

    def decodeSnapshot(self, snapshot: WireServiceSubscriptionSnapshot) -> Any:
        self._codecs.reset()
        return {**snapshot, "instances": [decodeInstance(instance, self._codecs) for instance in snapshot["instances"]]}  # type: ignore[typeddict-item, return-value]

    def decodeUpdate(self, update: WireServiceProviderUpdate) -> Any:
        codecs = self._codecs
        match update["type"]:
            case "reset":
                codecs.reset()
                snapshot = update["snapshot"]
                return {  # type: ignore[return-value]
                    **update,
                    "snapshot": {
                        **snapshot,
                        "instances": [decodeInstance(instance, codecs) for instance in snapshot["instances"]],
                    },
                }
            case "state":
                return {**update, "ops": codecs.get(update.get("instance"), update["member"]).decode(update["ops"])}  # type: ignore[return-value]
            case "replaced":
                codecs.reset()
                return {**update, "snapshot": decodeInstance(update["snapshot"], codecs)}  # type: ignore[return-value]
            case "spawned":
                return {**update, "instance": decodeInstance(update["instance"], codecs)}  # type: ignore[return-value]
            case "unavailable":
                codecs.reset()
                return update  # type: ignore[return-value]
            case "closed":
                codecs.removeInstance(update["instance"])
                return update  # type: ignore[return-value]
            case _:
                raise TypeError("Invalid service provider update")


class StateCodecRegistry[C]:
    def __init__(self, create: Callable[[], C]) -> None:
        self._create = create
        # (instance, codec) by state key, in insertion order.
        self._entries: dict[str, tuple[ServiceInstanceAddress | None, C]] = {}

    def reset(self) -> None:
        self._entries.clear()

    def add(self, instance: ServiceInstanceAddress | None, member: str) -> C:
        key = stateKey(instance, member)
        if key in self._entries:
            raise RuntimeError(f"Duplicate service state {describeState(instance, member)}")
        codec = self._create()
        self._entries[key] = (instance, codec)
        return codec

    def get(self, instance: ServiceInstanceAddress | None, member: str) -> C:
        entry = self._entries.get(stateKey(instance, member))
        if entry is None:
            raise RuntimeError(f"Unknown service state {describeState(instance, member)}")
        return entry[1]

    def removeInstance(self, instance: ServiceInstanceAddress) -> None:
        for key, (address, _codec) in list(self._entries.items()):
            if sameAddress(address, instance):
                del self._entries[key]


def createServiceStateEncoder() -> ServiceStateEncoder:
    return ServiceStateEncoder()


def createServiceStateDecoder() -> ServiceStateDecoder:
    return ServiceStateDecoder()


def encodeInstance(instance: Any, codecs: StateCodecRegistry[Encoder]) -> dict[str, Any]:
    return {
        **instance,
        "members": [
            {**member, "ops": codecs.add(instance.get("instance"), member["name"]).encode(member["ops"])}
            if member["kind"] == "state"
            else member
            for member in instance["members"]
        ],
    }


def decodeInstance(instance: Any, codecs: StateCodecRegistry[Decoder]) -> dict[str, Any]:
    return {
        **instance,
        "members": [
            {**member, "ops": codecs.add(instance.get("instance"), member["name"]).decode(member["ops"])}
            if member["kind"] == "state"
            else member
            for member in instance["members"]
        ],
    }


def stateKey(instance: ServiceInstanceAddress | None, member: str) -> str:
    return jsonStringify([
        None if instance is None else instance["key"],
        None if instance is None else instance["generation"],
        member,
    ])


def sameAddress(left: ServiceInstanceAddress | None, right: ServiceInstanceAddress) -> bool:
    return left is not None and left["key"] == right["key"] and left["generation"] == right["generation"]


def describeState(instance: ServiceInstanceAddress | None, member: str) -> str:
    return member if instance is None else f"{instance['key']}@{instance['generation']}.{member}"


__all__ = [
    "ServiceStateDecoder",
    "ServiceStateEncoder",
    "createServiceStateDecoder",
    "createServiceStateEncoder",
]
