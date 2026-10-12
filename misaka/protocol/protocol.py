"""Routed envelopes of the experimental pi protocol, version 8.

Translated from pi's ``packages/protocol/src/protocol.ts``. Upstream describes the envelopes as TypeBox
schemas; the ``*Schema`` values below are the JSON Schema those TypeBox calls produce, under upstream's names.

MISAKA fork: TypeBox's ``Check(schema, value)`` becomes the hand-written ``check*`` functions, one per schema:
an object has exactly its properties (a ``dict`` with no other key), ``Optional`` ones may be absent, and an
opaque payload is anything (the codec then requires the whole message to be strict JSON). A message is a plain
``dict``. tests/test_protocol_schemas.py holds the checks to the schemas.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any, Final, Literal, NotRequired, TypedDict

from misaka.utils.js import jsIsInteger

PROTOCOL_VERSION: Final = 8

IdSchema: Final[dict[str, Any]] = {"type": "string", "minLength": 1}
OpaqueJsonValueSchema: Final[dict[str, Any]] = {}


def StrictObject(properties: dict[str, Any], optional: Sequence[str] = ()) -> dict[str, Any]:
    """``Type.Object(properties, { additionalProperties: false })``; ``optional`` names the properties upstream
    wraps in ``Type.Optional``."""
    return {
        "type": "object",
        "properties": properties,
        "required": [name for name in properties if name not in optional],
        "additionalProperties": False,
    }


def _literal(value: Any) -> dict[str, Any]:
    """``Type.Literal(value)``."""
    kind = "boolean" if isinstance(value, bool) else "string" if isinstance(value, str) else "number"
    return {"const": value, "type": kind}


ServerIdSchema: Final[dict[str, Any]] = {
    "type": "string",
    "pattern": "^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
}

type ServerId = str


def isServerId(value: Any) -> bool:
    return isinstance(value, str) and _SERVER_ID.fullmatch(value) is not None


def _isId(value: Any) -> bool:
    return isinstance(value, str) and len(value) >= 1


def _isStrictObject(value: Any, required: Sequence[str], optional: Sequence[str] = ()) -> bool:
    if not isinstance(value, dict):
        return False
    return all(key in value for key in required) and all(key in required or key in optional for key in value)


def _checkProtocolError(value: Any) -> bool:
    return _isStrictObject(value, ["code", "message"]) and _isId(value["code"]) and isinstance(value["message"], str)


def _checkServerTarget(value: Any) -> bool:
    return _isStrictObject(value, ["serverId"]) and isServerId(value["serverId"])


def _checkSessionTarget(value: Any) -> bool:
    return (
        _isStrictObject(value, ["serverId", "sessionId", "attachmentId"])
        and isServerId(value["serverId"])
        and _isId(value["sessionId"])
        and _isId(value["attachmentId"])
    )


def _checkRpcTarget(value: Any) -> bool:
    return _checkServerTarget(value) or _checkSessionTarget(value)


def _checkClientHello(value: Any) -> bool:
    return (
        _isStrictObject(value, ["type", "version"])
        and value["type"] == "hello"
        and jsIsInteger(value["version"])
        and value["version"] >= 0
    )


def _checkRequest(value: Any) -> bool:
    return (
        _isStrictObject(value, ["type", "id", "target", "call"])
        and value["type"] == "request"
        and _isId(value["id"])
        and _checkRpcTarget(value["target"])
    )


def _checkCancel(value: Any) -> bool:
    return (
        _isStrictObject(value, ["type", "id", "target"])
        and value["type"] == "cancel"
        and _isId(value["id"])
        and _checkRpcTarget(value["target"])
    )


def checkClientMessage(value: Any) -> bool:
    """Whether ``value`` is a client hello, request or cancel envelope."""
    return _checkClientHello(value) or _checkRequest(value) or _checkCancel(value)


def _checkServerHello(value: Any) -> bool:
    return (
        _isStrictObject(value, ["type", "version", "serverId"])
        and value["type"] == "hello"
        and not isinstance(value["version"], bool)
        and value["version"] == PROTOCOL_VERSION
        and isServerId(value["serverId"])
    )


def _checkServerHelloError(value: Any) -> bool:
    return (
        _isStrictObject(value, ["type", "error"]) and value["type"] == "hello_error" and _checkProtocolError(value["error"])
    )


def _checkResponse(value: Any) -> bool:
    if not isinstance(value, dict) or value.get("type") != "response":
        return False
    if value.get("ok") is True:
        return _isStrictObject(value, ["type", "id", "ok"], ["result"]) and _isId(value["id"])
    if value.get("ok") is False:
        return _isStrictObject(value, ["type", "id", "ok", "error"]) and _isId(value["id"]) and _checkProtocolError(value["error"])
    return False


def _checkServiceEvent(value: Any) -> bool:
    return (
        _isStrictObject(value, ["type", "subscriptionId", "update"])
        and value["type"] == "service_update"
        and _isId(value["subscriptionId"])
    )


def _checkAttachment(value: Any) -> bool:
    return (
        _isStrictObject(value, ["type", "attachment"])
        and value["type"] == "attachment"
        and (value["attachment"] is None or _checkSessionTarget(value["attachment"]))
    )


def checkServerMessage(value: Any) -> bool:
    """Whether ``value`` is a server hello, hello error, response, service update or attachment envelope."""
    return (
        _checkServerHello(value)
        or _checkServerHelloError(value)
        or _checkResponse(value)
        or _checkServiceEvent(value)
        or _checkAttachment(value)
    )

ProtocolErrorSchema = StrictObject({"code": IdSchema, "message": {"type": "string"}})
type ProtocolErrorCode = str


class ProtocolError(TypedDict):
    code: str
    message: str

# Must be the first frame sent by a client.
ClientHelloSchema = StrictObject({"type": _literal("hello"), "version": {"type": "integer", "minimum": 0}})


class ClientHello(TypedDict):
    """Must be the first frame sent by a client."""

    type: Literal["hello"]
    version: int


class ServerTarget(TypedDict):
    """A server-wide call, fenced to one logical server."""

    serverId: str

# A server-wide call, fenced to one logical server.
ServerTargetSchema = StrictObject({"serverId": ServerIdSchema})
# A session call, fenced to one logical server, durable session, and live attachment.
SessionTargetSchema = StrictObject({"serverId": ServerIdSchema, "sessionId": IdSchema, "attachmentId": IdSchema})


class SessionTarget(TypedDict):
    """A session call, fenced to one logical server, durable session, and live attachment."""

    serverId: str
    sessionId: str
    attachmentId: str
RpcTargetSchema: Final[dict[str, Any]] = {"anyOf": [ServerTargetSchema, SessionTargetSchema]}


type RpcTarget = ServerTarget | SessionTarget

RequestEnvelopeSchema = StrictObject({
    "type": _literal("request"),
    "id": IdSchema,
    "target": RpcTargetSchema,
    "call": OpaqueJsonValueSchema,
})
CancelEnvelopeSchema = StrictObject({"type": _literal("cancel"), "id": IdSchema, "target": RpcTargetSchema})


class RequestEnvelope(TypedDict):
    type: Literal["request"]
    id: str
    target: RpcTarget
    call: Any


class CancelEnvelope(TypedDict):
    type: Literal["cancel"]
    id: str
    target: RpcTarget
ClientMessageSchema: Final[dict[str, Any]] = {"anyOf": [ClientHelloSchema, RequestEnvelopeSchema, CancelEnvelopeSchema]}


type ClientMessage = ClientHello | RequestEnvelope | CancelEnvelope

ServerHelloSchema = StrictObject({
    "type": _literal("hello"),
    "version": _literal(PROTOCOL_VERSION),
    "serverId": ServerIdSchema,
})
ServerHelloErrorSchema = StrictObject({"type": _literal("hello_error"), "error": ProtocolErrorSchema})
ResponseEnvelopeSchema: Final[dict[str, Any]] = {
    "anyOf": [
        StrictObject(
            {"type": _literal("response"), "id": IdSchema, "ok": _literal(True), "result": OpaqueJsonValueSchema},
            optional=["result"],
        ),
        StrictObject({"type": _literal("response"), "id": IdSchema, "ok": _literal(False), "error": ProtocolErrorSchema}),
    ]
}
ServiceEventEnvelopeSchema = StrictObject({
    "type": _literal("service_update"),
    "subscriptionId": IdSchema,
    "update": OpaqueJsonValueSchema,
})
# Out-of-band update to this presentation's selected Session route.
AttachmentEnvelopeSchema = StrictObject({
    "type": _literal("attachment"),
    "attachment": {"anyOf": [SessionTargetSchema, {"type": "null"}]},
})
ServerMessageSchema: Final[dict[str, Any]] = {
    "anyOf": [
        ServerHelloSchema,
        ServerHelloErrorSchema,
        ResponseEnvelopeSchema,
        ServiceEventEnvelopeSchema,
        AttachmentEnvelopeSchema,
    ]
}

_SERVER_ID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}")


class ServerHello(TypedDict):
    type: Literal["hello"]
    version: Literal[8]
    serverId: str


class ServerHelloError(TypedDict):
    type: Literal["hello_error"]
    error: ProtocolError


class ResponseEnvelope(TypedDict):
    """``ok`` true carries an optional ``result``; ``ok`` false carries an ``error``."""

    type: Literal["response"]
    id: str
    ok: bool
    result: NotRequired[Any]
    error: NotRequired[ProtocolError]


class ServiceEventEnvelope(TypedDict):
    type: Literal["service_update"]
    subscriptionId: str
    update: Any


class AttachmentEnvelope(TypedDict):
    """Out-of-band update to this presentation's selected Session route."""

    type: Literal["attachment"]
    attachment: SessionTarget | None


type ServerMessage = ServerHello | ServerHelloError | ResponseEnvelope | ServiceEventEnvelope | AttachmentEnvelope


__all__ = [
    "PROTOCOL_VERSION",
    "AttachmentEnvelope",
    "CancelEnvelope",
    "ClientHello",
    "ClientMessage",
    "ClientMessageSchema",
    "ProtocolError",
    "ProtocolErrorCode",
    "RequestEnvelope",
    "ResponseEnvelope",
    "RpcTarget",
    "ServerHello",
    "ServerHelloError",
    "ServerId",
    "ServerMessage",
    "ServerMessageSchema",
    "ServerTarget",
    "ServiceEventEnvelope",
    "SessionTarget",
    "checkClientMessage",
    "checkServerMessage",
    "isServerId",
]
