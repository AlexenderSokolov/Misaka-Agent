"""The protocol's exported JSON Schemas (upstream's TypeBox schemas) and the hand-written checks that stand in for
TypeBox's ``Check`` agree on every envelope shape (misaka/protocol/protocol.py)."""

from __future__ import annotations

import jsonschema
import pytest

from misaka.protocol.protocol import (
    ClientMessageSchema,
    ServerMessageSchema,
    checkClientMessage,
    checkServerMessage,
)

SERVER = "0123abcd-0123-4abc-8def-0123456789ab"
SESSION = {"serverId": SERVER, "sessionId": "s", "attachmentId": "a"}

CLIENT_MESSAGES = [
    {"type": "hello", "version": 8},
    {"type": "hello", "version": 0},
    {"type": "hello", "version": -1},
    {"type": "hello", "version": 1.5},
    {"type": "hello", "version": True},
    {"type": "hello", "version": 8, "extra": 1},
    {"type": "request", "id": "1", "target": {"serverId": SERVER}, "call": {"anything": [1, 2]}},
    {"type": "request", "id": "1", "target": SESSION, "call": None},
    {"type": "request", "id": "", "target": SESSION, "call": None},
    {"type": "request", "id": "1", "target": {"serverId": "not-a-server"}, "call": None},
    {"type": "request", "id": "1", "target": {**SESSION, "extra": 1}, "call": None},
    {"type": "request", "id": "1", "target": SESSION},
    {"type": "cancel", "id": "1", "target": SESSION},
    {"type": "cancel", "id": "1", "target": {"serverId": SERVER}, "call": 1},
    {"type": "other"},
    "hello",
    None,
    [],
]

SERVER_MESSAGES = [
    {"type": "hello", "version": 8, "serverId": SERVER},
    {"type": "hello", "version": 7, "serverId": SERVER},
    {"type": "hello", "version": 8, "serverId": "nope"},
    {"type": "hello_error", "error": {"code": "bad", "message": "no"}},
    {"type": "hello_error", "error": {"code": "", "message": "no"}},
    {"type": "hello_error", "error": {"code": "bad", "message": "no", "more": 1}},
    {"type": "response", "id": "1", "ok": True},
    {"type": "response", "id": "1", "ok": True, "result": [1]},
    {"type": "response", "id": "1", "ok": True, "error": {"code": "x", "message": ""}},
    {"type": "response", "id": "1", "ok": False, "error": {"code": "x", "message": ""}},
    {"type": "response", "id": "1", "ok": False},
    {"type": "response", "id": "1", "ok": 1},
    {"type": "service_update", "subscriptionId": "s", "update": {"x": 1}},
    {"type": "service_update", "subscriptionId": "", "update": {}},
    {"type": "attachment", "attachment": None},
    {"type": "attachment", "attachment": SESSION},
    {"type": "attachment", "attachment": {"serverId": SERVER}},
    {"type": "attachment"},
    {},
]


def _valid(schema, value) -> bool:
    return jsonschema.Draft202012Validator(schema).is_valid(value)


@pytest.mark.parametrize("message", CLIENT_MESSAGES)
def test_client_checks_agree_with_the_client_message_schema(message):
    assert checkClientMessage(message) == _valid(ClientMessageSchema, message)


@pytest.mark.parametrize("message", SERVER_MESSAGES)
def test_server_checks_agree_with_the_server_message_schema(message):
    assert checkServerMessage(message) == _valid(ServerMessageSchema, message)
