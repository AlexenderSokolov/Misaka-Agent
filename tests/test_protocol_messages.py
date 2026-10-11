"""pi protocol's ``test/protocol.test.ts``."""

from __future__ import annotations

import math

import pytest

from misaka.protocol import (
    PROTOCOL_VERSION,
    ClientMessageDecoder,
    FrameDecoder,
    ProtocolValidationError,
    ServerMessageDecoder,
    decodeCbor,
    encodeCbor,
    encodeClientMessage,
    encodeFrame,
    encodeServerMessage,
    isSupportedProtocolVersion,
    parseClientMessage,
    parseServerMessage,
)
from misaka.utils.js import UNDEFINED
from tests.chord_helpers import matches

SERVER_ID = "00000000-0000-4000-8000-000000000001"
clientHello = {"type": "hello", "version": PROTOCOL_VERSION}
serverHello = {"type": "hello", "version": PROTOCOL_VERSION, "serverId": SERVER_ID}


def test_negotiates_protocol_version_8():
    assert PROTOCOL_VERSION == 8
    assert isSupportedProtocolVersion(8) is True
    assert isSupportedProtocolVersion(7) is False
    assert isSupportedProtocolVersion(8.5) is False


@pytest.mark.parametrize("version", [0, PROTOCOL_VERSION, PROTOCOL_VERSION + 1])
def test_accepts_integer_client_hello_versions_for_negotiation(version):
    assert parseClientMessage({**clientHello, "version": version}) == {**clientHello, "version": version}


@pytest.mark.parametrize(
    "message",
    [
        {"type": "hello", "version": str(PROTOCOL_VERSION)},
        {"type": "hello", "version": PROTOCOL_VERSION + 0.5},
        {"type": "hello", "version": PROTOCOL_VERSION, "extra": True},
    ],
)
def test_rejects_an_invalid_client_hello(message):
    with pytest.raises(ProtocolValidationError):
        parseClientMessage(message)


@pytest.mark.parametrize(
    "serverId",
    [
        "",
        "server-1",
        "00000000-0000-7000-8000-000000000001",
        "00000000-0000-4000-7000-000000000001",
        "00000000-0000-4000-8000-00000000000A",
    ],
)
def test_rejects_non_canonical_uuid_v4_server_ids(serverId):
    with pytest.raises(ProtocolValidationError):
        parseClientMessage({
            "type": "request",
            "id": "request-1",
            "target": {"serverId": serverId},
            "call": {"serviceId": "pi.models", "member": "list", "args": []},
        })


def test_keeps_routed_request_and_event_payloads_opaque():
    request = {
        "type": "request",
        "id": "request-1",
        "target": {"serverId": SERVER_ID, "sessionId": "session-1", "attachmentId": "attachment-1"},
        "call": {
            "serviceId": "application.custom",
            "instance": {"key": "instance-1", "generation": 2},
            "member": "invoke",
            "args": [{"arbitrary": True}, ["opaque"]],
        },
    }
    assert parseClientMessage(request) == request
    assert matches(
        parseClientMessage({**request, "call": {"arbitrary": "strict JSON whose service meaning belongs to Chord"}}),
        {"call": {"arbitrary": "strict JSON whose service meaning belongs to Chord"}},
    )
    assert matches(
        parseServerMessage({
            "type": "service_update",
            "subscriptionId": "subscription-1",
            "update": {"applicationDefined": True},
        }),
        {"update": {"applicationDefined": True}},
    )


def test_rejects_non_json_opaque_payloads():
    request = {
        "type": "request",
        "id": "request-1",
        "target": {"serverId": SERVER_ID},
        "call": {"serviceId": "application.custom", "member": "invoke", "args": []},
    }
    cyclic: dict = {}
    cyclic["self"] = cyclic
    for label, value in (
        ("byte array", bytes([1])),
        ("non-finite number", math.nan),
        ("undefined property", {"value": UNDEFINED}),
        ("cycle", cyclic),
    ):
        with pytest.raises(ProtocolValidationError):
            parseClientMessage({**request, "call": {**request["call"], "args": [value]}})
        with pytest.raises(ProtocolValidationError):
            parseServerMessage({"type": "response", "id": "request-1", "ok": True, "result": value})
        del label


def test_validates_request_cancellation_envelopes():
    cancel = {"type": "cancel", "id": "request-1", "target": {"serverId": SERVER_ID}}
    assert parseClientMessage(cancel) == cancel
    with pytest.raises(ProtocolValidationError):
        parseClientMessage({**cancel, "id": ""})
    with pytest.raises(ProtocolValidationError):
        parseClientMessage({**cancel, "extra": True})


def test_validates_attachment_route_updates():
    attached = {
        "type": "attachment",
        "attachment": {"serverId": SERVER_ID, "sessionId": "session-1", "attachmentId": "attachment-1"},
    }
    detached = {"type": "attachment", "attachment": None}
    assert parseServerMessage(attached) == attached
    assert parseServerMessage(detached) == detached
    with pytest.raises(ProtocolValidationError):
        parseServerMessage({**attached, "attachment": {"sessionId": "session-1"}})


@pytest.mark.parametrize(
    "message",
    [
        {
            "type": "request",
            "id": "",
            "target": {"serverId": SERVER_ID},
            "call": {"serviceId": "pi.models", "member": "list", "args": []},
        },
        {
            "type": "request",
            "id": "request-1",
            "target": {"serverId": SERVER_ID},
            "call": {"serviceId": "pi.models", "member": "list", "args": []},
            "extra": True,
        },
    ],
    ids=["empty request id", "extra envelope field"],
)
def test_rejects_malformed_request_boundaries(message):
    with pytest.raises(ProtocolValidationError):
        parseClientMessage(message)


def test_accepts_a_successful_void_response_without_a_result_field():
    assert parseServerMessage({"type": "response", "id": "request-1", "ok": True}) == {
        "type": "response",
        "id": "request-1",
        "ok": True,
    }


@pytest.mark.parametrize(
    "message",
    [
        {**serverHello, "serverId": "server-1"},
        {"type": "response", "id": "request-1", "ok": True, "result": [], "extra": True},
        {"type": "response", "id": "request-1", "ok": False, "error": {"code": "", "message": "bad"}},
    ],
    ids=["invalid server id", "extra response field", "empty error code"],
)
def test_rejects_malformed_server_boundaries(message):
    with pytest.raises(ProtocolValidationError):
        parseServerMessage(message)


@pytest.mark.parametrize("code", ["wrong_server", "cancelled", "service_not_found", "application_error"])
def test_accepts_the_opaque_error_code(code):
    message = {"type": "response", "id": "request-1", "ok": False, "error": {"code": code, "message": "safe"}}
    assert parseServerMessage(message) == message


def test_rejects_unknown_messages_and_fields():
    with pytest.raises(ProtocolValidationError):
        parseServerMessage({**serverHello, "snapshot": {}})
    with pytest.raises(ProtocolValidationError):
        parseServerMessage({"type": "unknown", "event": {}})


def test_does_not_parse_json_strings_as_messages():
    import json

    with pytest.raises(ProtocolValidationError):
        parseClientMessage(json.dumps(clientHello))
    with pytest.raises(ProtocolValidationError):
        parseServerMessage(json.dumps(serverHello))


# -- validated framed protocol APIs --------------------------------------------------------------


def test_encodes_complete_client_and_server_frames():
    clientFrames = FrameDecoder().push(encodeClientMessage(clientHello))
    assert parseClientMessage(decodeCbor(clientFrames[0])) == clientHello
    serverFrames = FrameDecoder().push(encodeServerMessage(serverHello))
    assert parseServerMessage(decodeCbor(serverFrames[0])) == serverHello


def test_enforces_outbound_frame_limits():
    with pytest.raises(ProtocolValidationError):
        encodeClientMessage(clientHello, {"maxFrameLength": 8})
    with pytest.raises(ProtocolValidationError):
        encodeServerMessage(serverHello, {"maxFrameLength": 8})


def test_incrementally_decodes_fragmented_and_coalesced_client_messages():
    request = {
        "type": "request",
        "id": "request-1",
        "target": {"serverId": SERVER_ID},
        "call": {"serviceId": "pi.session-directory", "member": "list", "args": []},
    }
    wire = encodeClientMessage(clientHello) + encodeClientMessage(request)

    for split in range(len(wire) + 1):
        decoder = ClientMessageDecoder()
        messages = [*decoder.push(wire[:split]), *decoder.push(wire[split:])]
        decoder.end()
        assert messages == [clientHello, request]


def test_incrementally_decodes_fragmented_and_coalesced_server_messages():
    response = {"type": "response", "id": "request-1", "ok": True, "result": []}
    first = encodeServerMessage(serverHello)
    second = encodeServerMessage(response)
    wire = first + second

    split = len(first) + len(second) // 2
    decoder = ServerMessageDecoder()
    assert decoder.push(wire[:split]) == [serverHello]
    assert decoder.push(wire[split:]) == [response]
    decoder.end()


@pytest.mark.parametrize(
    "wire",
    [
        encodeFrame(b""),
        encodeFrame(bytes([0xFF])),
        encodeFrame(encodeCbor({"type": "hello", "version": 1, "extra": True})),
    ],
    ids=["empty CBOR payload", "malformed CBOR", "schema-invalid CBOR"],
)
def test_rejects_invalid_framed_input(wire):
    decoder = ClientMessageDecoder()
    with pytest.raises(ProtocolValidationError):
        decoder.push(wire)
    with pytest.raises(ProtocolValidationError, match="(?i)failed"):
        decoder.push(encodeClientMessage(clientHello))


def test_rejects_truncated_and_oversized_framing():
    truncated = ServerMessageDecoder()
    assert truncated.push(bytes([0, 0, 0, 2, 1])) == []
    with pytest.raises(ProtocolValidationError):
        truncated.end()
    oversized = ClientMessageDecoder({"maxFrameLength": 3})
    with pytest.raises(ProtocolValidationError):
        oversized.push(bytes([0, 0, 0, 4]))
