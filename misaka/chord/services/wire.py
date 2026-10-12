"""Service control calls and the parsers for what crosses a service boundary.

Translated from pi's ``packages/chord/src/services/wire.ts``. A call, an update and a snapshot are plain
``dict`` values with upstream's key names; the ``Wire*`` shapes carry operations in their wire form.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from misaka.chord.delta import assertValidOp, assertValidWireOp
from misaka.chord.types import (
    ServiceCall,
    ServiceCatalogueEntry,
    ServiceMode,
    ServiceProviderUpdate,
    ServiceSubscriptionSnapshot,
)
from misaka.utils.js import UNDEFINED, jsIsInteger

type WireServiceMemberSnapshot = dict[str, Any]
type WireServiceInstanceSnapshot = dict[str, Any]
type WireServiceSubscriptionSnapshot = dict[str, Any]
type WireServiceProviderUpdate = dict[str, Any]

SERVICE_CONTROL_ID = "$chord.service"
SERVICE_CATALOGUE_MEMBER = "catalogue"
SERVICE_SUBSCRIBE_MEMBER = "subscribe"
SERVICE_UNSUBSCRIBE_MEMBER = "unsubscribe"
type ServiceControlCall = dict[str, Any]


def createServiceCatalogueCall() -> ServiceCall:
    return {"serviceId": SERVICE_CONTROL_ID, "member": SERVICE_CATALOGUE_MEMBER, "args": []}


def createServiceSubscribeCall(subscriptionId: str, serviceId: str, mode: ServiceMode) -> ServiceCall:
    return {
        "serviceId": SERVICE_CONTROL_ID,
        "member": SERVICE_SUBSCRIBE_MEMBER,
        "args": [subscriptionId, serviceId, mode],
    }


def createServiceUnsubscribeCall(subscriptionId: str) -> ServiceCall:
    return {"serviceId": SERVICE_CONTROL_ID, "member": SERVICE_UNSUBSCRIBE_MEMBER, "args": [subscriptionId]}


def decodeServiceControlCall(call: ServiceCall) -> ServiceControlCall | None:
    if call["serviceId"] != SERVICE_CONTROL_ID or call.get("instance", UNDEFINED) is not UNDEFINED:
        return None
    args = call["args"]
    if call["member"] == SERVICE_CATALOGUE_MEMBER and len(args) == 0:
        return {"type": "catalogue"}
    if (
        call["member"] == SERVICE_SUBSCRIBE_MEMBER
        and len(args) == 3
        and isId(args[0])
        and isId(args[1])
        and (args[2] == "singleton" or args[2] == "keyed")
    ):
        return {"type": "subscribe", "subscriptionId": args[0], "serviceId": args[1], "mode": args[2]}
    if call["member"] == SERVICE_UNSUBSCRIBE_MEMBER and len(args) == 1 and isId(args[0]):
        return {"type": "unsubscribe", "subscriptionId": args[0]}
    return None


def parseServiceCall(value: Any) -> ServiceCall:
    call = record(value, "service call")
    assertKeys(call, ["serviceId", "member", "args"], ["instance"], "service call")
    if not isId(call["serviceId"]) or not isId(call["member"]) or not isinstance(call["args"], list):
        raise TypeError("Invalid service call")
    if call.get("instance", UNDEFINED) is not UNDEFINED:
        assertAddress(call["instance"])
    return value


def parseServiceCatalogue(value: Any) -> list[ServiceCatalogueEntry]:
    if not isinstance(value, list):
        raise TypeError("Invalid service catalogue")
    ids: set[str] = set()
    for candidate in value:
        entry = record(candidate, "service catalogue entry")
        assertKeys(entry, ["serviceId", "mode"], [], "service catalogue entry")
        if not isId(entry["serviceId"]) or not isMode(entry["mode"]) or entry["serviceId"] in ids:
            raise TypeError("Invalid service catalogue")
        ids.add(entry["serviceId"])
    return value


def parseServiceSubscriptionSnapshot(value: Any) -> ServiceSubscriptionSnapshot:
    assertSubscriptionSnapshot(value, assertValidOp)
    return value


def parseWireServiceSubscriptionSnapshot(value: Any) -> WireServiceSubscriptionSnapshot:
    assertSubscriptionSnapshot(value, assertValidWireOp)
    return value


def parseServiceProviderUpdate(value: Any) -> ServiceProviderUpdate:
    assertProviderUpdate(value, assertValidOp)
    return value


def parseWireServiceProviderUpdate(value: Any) -> WireServiceProviderUpdate:
    assertProviderUpdate(value, assertValidWireOp)
    return value


def assertSubscriptionSnapshot(value: Any, assertOp: Callable[[Any], None]) -> None:
    snapshot = record(value, "service subscription snapshot")
    assertKeys(snapshot, ["serviceId", "mode", "instances"], [], "service subscription snapshot")
    if not isId(snapshot["serviceId"]) or not isMode(snapshot["mode"]) or not isinstance(snapshot["instances"], list):
        raise TypeError("Invalid service subscription snapshot")
    for instance in snapshot["instances"]:
        assertInstance(instance, assertOp)


def assertProviderUpdate(value: Any, assertOp: Callable[[Any], None]) -> None:
    update = record(value, "service provider update")
    match update.get("type", UNDEFINED):
        case "state":
            assertKeys(update, ["type", "member", "sequence", "ops"], ["instance"], "state update")
            if not isId(update["member"]) or not isInteger(update["sequence"], 1) or not isinstance(update["ops"], list):
                raise TypeError("Invalid service state update")
            if update.get("instance", UNDEFINED) is not UNDEFINED:
                assertAddress(update["instance"])
            for op in update["ops"]:
                assertOp(op)
        case "reset":
            assertKeys(update, ["type", "snapshot"], [], "reset update")
            assertSubscriptionSnapshot(update["snapshot"], assertOp)
            for instance in update["snapshot"]["instances"]:
                for member in instance["members"]:
                    if member["kind"] == "state" and (len(member["ops"]) != 1 or member["ops"][0][0] != "r"):
                        raise TypeError("Service reset must contain full root replacements")
        case "unavailable":
            assertKeys(update, ["type"], [], "unavailable update")
        case "replaced":
            assertKeys(update, ["type", "snapshot"], [], "replacement update")
            assertInstance(update["snapshot"], assertOp)
        case "spawned":
            assertKeys(update, ["type", "instance"], [], "spawn update")
            assertInstance(update["instance"], assertOp)
        case "closed":
            assertKeys(update, ["type", "instance"], [], "close update")
            assertAddress(update["instance"])
        case _:
            raise TypeError("Invalid service provider update")


def assertInstance(value: Any, assertOp: Callable[[Any], None]) -> None:
    instance = record(value, "service instance snapshot")
    assertKeys(instance, ["members"], ["instance"], "service instance snapshot")
    if instance.get("instance", UNDEFINED) is not UNDEFINED:
        assertAddress(instance["instance"])
    if not isinstance(instance["members"], list):
        raise TypeError("Invalid service instance snapshot")
    for candidate in instance["members"]:
        member = record(candidate, "service member snapshot")
        kind = member.get("kind", UNDEFINED)
        if kind == "method":
            assertKeys(member, ["name", "kind"], [], "service method snapshot")
            if not isId(member["name"]):
                raise TypeError("Invalid service method snapshot")
            continue
        if kind == "state":
            assertKeys(member, ["name", "kind", "sequence", "ops"], [], "service state snapshot")
            if not isId(member["name"]) or not isInteger(member["sequence"], 0) or not isinstance(member["ops"], list):
                raise TypeError("Invalid service state snapshot")
            for op in member["ops"]:
                assertOp(op)
            continue
        raise TypeError("Invalid service member snapshot")


def assertAddress(value: Any) -> None:
    address = record(value, "service instance address")
    assertKeys(address, ["key", "generation"], [], "service instance address")
    if not isId(address["key"]) or not isInteger(address["generation"], 1):
        raise TypeError("Invalid service instance address")


def record(value: Any, description: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"Invalid {description}")
    return value


def assertKeys(value: dict[str, Any], required: list[str], optional: list[str], description: str) -> None:
    allowed = {*required, *optional}
    if any(key not in value for key in required) or any(key not in allowed for key in value):
        raise TypeError(f"Invalid {description}")


def isId(value: Any) -> bool:
    return isinstance(value, str) and len(value) > 0


def isMode(value: Any) -> bool:
    return value == "singleton" or value == "keyed"


def isInteger(value: Any, minimum: int) -> bool:
    return jsIsInteger(value) and value >= minimum


__all__ = [
    "ServiceControlCall",
    "WireServiceInstanceSnapshot",
    "WireServiceMemberSnapshot",
    "WireServiceProviderUpdate",
    "WireServiceSubscriptionSnapshot",
    "createServiceCatalogueCall",
    "createServiceSubscribeCall",
    "createServiceUnsubscribeCall",
    "decodeServiceControlCall",
    "parseServiceCall",
    "parseServiceCatalogue",
    "parseServiceProviderUpdate",
    "parseServiceSubscriptionSnapshot",
    "parseWireServiceProviderUpdate",
    "parseWireServiceSubscriptionSnapshot",
]
