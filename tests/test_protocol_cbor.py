"""pi protocol's ``test/cbor/cbor.test.ts`` and ``test/framing.test.ts``."""

from __future__ import annotations

import collections
import datetime
import math

import pytest

from misaka.protocol import (
    DEFAULT_MAX_CBOR_BYTE_LENGTH,
    DEFAULT_MAX_CBOR_CONTAINER_LENGTH,
    DEFAULT_MAX_CBOR_DEPTH,
    DEFAULT_MAX_FRAME_LENGTH,
    CborError,
    FrameDecoder,
    FrameError,
    decodeCbor,
    encodeCbor,
    encodeFrame,
)
from misaka.utils.js import UNDEFINED, RangeError

MAX_SAFE = 2**53 - 1

knownVectors = [
    (None, "f6"),
    (False, "f4"),
    (True, "f5"),
    (0, "00"),
    (1, "01"),
    (10, "0a"),
    (23, "17"),
    (24, "1818"),
    (25, "1819"),
    (100, "1864"),
    (1000, "1903e8"),
    (1_000_000, "1a000f4240"),
    (1_000_000_000_000, "1b000000e8d4a51000"),
    (MAX_SAFE, "1b001fffffffffffff"),
    (-1, "20"),
    (-10, "29"),
    (-24, "37"),
    (-25, "3818"),
    (-100, "3863"),
    (-1000, "3903e7"),
    (-1_000_000, "3a000f423f"),
    (-MAX_SAFE, "3b001ffffffffffffe"),
    (1.1, "fb3ff199999999999a"),
    (-0.0, "fb8000000000000000"),
    (bytes([1, 2, 3, 4]), "4401020304"),
    ("", "60"),
    ("IETF", "6449455446"),
    ("ü", "62c3bc"),
    ("水", "63e6b0b4"),
    (chr(0x10151), "64f0908591"),
    ([], "80"),
    ([1, 2, 3], "83010203"),
    ([1, [2, 3], [4, 5]], "8301820203820405"),
    ({"a": 1, "b": [2, 3]}, "a26161016162820203"),
]


def _shape(value):
    """The value with its types: ``True`` is not ``1`` and ``1.0`` is not ``1``."""
    if isinstance(value, list):
        return [_shape(item) for item in value]
    if isinstance(value, dict):
        return {key: _shape(item) for key, item in value.items()}
    return (type(value).__name__, value)


@pytest.mark.parametrize(("value", "wire"), knownVectors, ids=[str(index) for index in range(len(knownVectors))])
def test_encodes_and_decodes_rfc_8949_vectors(value, wire):
    assert encodeCbor(value).hex() == wire
    decoded = decodeCbor(bytes.fromhex(wire))
    if isinstance(value, float) and value == 0:
        assert isinstance(decoded, float) and math.copysign(1, decoded) < 0
    else:
        # ``toEqual`` tells ``true`` from ``1`` and ``-0`` from ``0``; ``==`` does not.
        assert decoded == value
        assert _shape(decoded) == _shape(value)


def test_encodes_integral_floats_as_integers():
    assert encodeCbor(2.0).hex() == "02"
    assert encodeCbor(-3.0).hex() == "22"
    assert decodeCbor(bytes.fromhex("fb4000000000000000")) == 2


def test_omits_undefined_object_properties_without_omitting_falsey_values():
    value = {"omitted": UNDEFINED, "zero": 0, "empty": "", "no": False, "nil": None}
    assert decodeCbor(encodeCbor(value)) == {"zero": 0, "empty": "", "no": False, "nil": None}


def test_preserves_a_leading_unicode_bom_and_treats_proto_as_data():
    assert decodeCbor(bytes.fromhex("63efbbbf")) == "﻿"
    value = {"__proto__": "safe"}
    decoded = decodeCbor(encodeCbor(value))
    assert decoded == value
    assert type(decoded) is dict
    assert "__proto__" in decoded


@pytest.mark.parametrize(
    "value",
    [
        UNDEFINED,
        [UNDEFINED],
        math.nan,
        math.inf,
        -math.inf,
        MAX_SAFE + 1,
        -MAX_SAFE - 1,
        10**30,
        lambda: None,
        datetime.datetime(1970, 1, 1),
        collections.OrderedDict(),
        {1, 2},
        (1, 2),
    ],
    ids=[
        "top-level undefined",
        "undefined array element",
        "NaN",
        "positive infinity",
        "negative infinity",
        "unsafe positive integer",
        "unsafe negative integer",
        "bigint",
        "function",
        "Date",
        "Map",
        "set",
        "tuple",
    ],
)
def test_rejects_unsupported_encoder_values(value):
    with pytest.raises(CborError):
        encodeCbor(value)


def test_rejects_maps_with_non_string_keys():
    with pytest.raises(CborError):
        encodeCbor({"valid": True, 1: False})


def test_rejects_lossy_strings_cycles_and_excessive_encoder_depth():
    with pytest.raises(CborError, match="(?i)Unicode"):
        encodeCbor(chr(0xD800))

    cyclic: list = []
    cyclic.append(cyclic)
    with pytest.raises(CborError, match="(?i)cycles"):
        encodeCbor(cyclic)

    tooDeep = None
    for _depth in range(DEFAULT_MAX_CBOR_DEPTH + 1):
        tooDeep = [tooDeep]
    with pytest.raises(CborError, match="(?i)depth"):
        encodeCbor(tooDeep)


@pytest.mark.parametrize(
    "wire",
    [
        "",
        "18",
        "1c",
        "5f",
        "7f",
        "9f",
        "bf",
        "c000",
        "f7",
        "e0",
        "ff",
        "f93c00",
        "fa3f800000",
        "fb7ff0000000000000",
        "fb7ff8000000000000",
        "fb3ff00000",
        "44010203",
        "636162",
        "8201",
        "a16161",
        "0000",
        "a10102",
        "a2616101616102",
        "61ff",
        "62c080",
        "63eda080",
        "1b0020000000000000",
        "3b001fffffffffffff",
        "fb4340000000000000",
    ],
    ids=[
        "empty input",
        "truncated integer",
        "reserved additional information",
        "indefinite byte string",
        "indefinite text string",
        "indefinite array",
        "indefinite map",
        "tag",
        "undefined",
        "unsupported simple value",
        "break outside an indefinite item",
        "float16",
        "float32",
        "positive infinity",
        "NaN",
        "truncated float64",
        "truncated byte string",
        "truncated text string",
        "truncated array",
        "truncated map",
        "trailing data",
        "non-string map key",
        "duplicate map key",
        "invalid UTF-8 byte",
        "overlong UTF-8",
        "UTF-8 surrogate",
        "unsafe positive integer",
        "unsafe negative integer",
        "unsafe integer encoded as float64",
    ],
)
def test_rejects_invalid_decoder_input(wire):
    with pytest.raises(CborError):
        decodeCbor(bytes.fromhex(wire))


def test_enforces_depth_and_declared_length_limits_before_traversing_values():
    tooDeep = bytes([0x81] * (DEFAULT_MAX_CBOR_DEPTH + 1)) + bytes([0xF6])
    with pytest.raises(CborError, match="(?i)depth"):
        decodeCbor(tooDeep)

    for prefix, limit in (
        ("5a", DEFAULT_MAX_CBOR_BYTE_LENGTH),
        ("7a", DEFAULT_MAX_CBOR_BYTE_LENGTH),
        ("9a", DEFAULT_MAX_CBOR_CONTAINER_LENGTH),
        ("ba", DEFAULT_MAX_CBOR_CONTAINER_LENGTH),
    ):
        with pytest.raises(CborError, match="(?i)limit"):
            decodeCbor(bytes.fromhex(f"{prefix}{limit + 1:08x}"))


def test_supports_stricter_caller_provided_limits():
    with pytest.raises(CborError, match="(?i)limit"):
        decodeCbor(bytes.fromhex("83010203"), {"maxContainerLength": 2})
    with pytest.raises(CborError, match="(?i)limit"):
        decodeCbor(bytes.fromhex("626162"), {"maxByteLength": 2})
    with pytest.raises(CborError, match="(?i)limit"):
        encodeCbor([1, 2, 3], {"maxContainerLength": 2})
    with pytest.raises(CborError, match="(?i)limit"):
        encodeCbor("ab", {"maxByteLength": 2})


def test_rejects_invalid_limits_and_input_types():
    with pytest.raises(RangeError):
        encodeCbor(1, {"maxDepth": 513})
    with pytest.raises(TypeError):
        decodeCbor("not bytes")  # type: ignore[arg-type]


# -- binary framing ------------------------------------------------------------------------------


def test_prefixes_payloads_with_a_four_byte_big_endian_length():
    assert encodeFrame(bytes([0xAA, 0xBB, 0xCC])) == bytes([0, 0, 0, 3, 0xAA, 0xBB, 0xCC])
    assert encodeFrame(b"") == bytes([0, 0, 0, 0])


def test_decodes_fragmented_coalesced_and_empty_frames_in_order():
    wire = encodeFrame(bytes([1, 2, 3])) + encodeFrame(b"") + encodeFrame(bytes([4]))
    decoder = FrameDecoder()
    frames = []
    for byte in wire:
        frames.extend(decoder.push(bytes([byte])))
    decoder.end()
    assert frames == [bytes([1, 2, 3]), b"", bytes([4])]

    coalesced = FrameDecoder()
    assert coalesced.push(wire) == frames
    coalesced.end()


def test_assembles_payloads_spanning_multiple_blocks():
    payload = bytes(index % 251 for index in range(70_000))
    wire = encodeFrame(payload)
    decoder = FrameDecoder()
    frames = [*decoder.push(wire[:101]), *decoder.push(wire[101:65_541]), *decoder.push(wire[65_541:])]
    decoder.end()
    assert frames == [payload]


def test_handles_every_split_point_across_a_frame():
    wire = encodeFrame(bytes([10, 20, 30, 40]))
    for split in range(len(wire) + 1):
        decoder = FrameDecoder()
        frames = [*decoder.push(wire[:split]), *decoder.push(wire[split:])]
        decoder.end()
        assert frames == [bytes([10, 20, 30, 40])]


def test_copies_payload_bytes_instead_of_retaining_or_aliasing_input_chunks():
    chunk = bytearray(encodeFrame(bytes([1, 2, 3])))
    decoder = FrameDecoder()
    frames = decoder.push(chunk)
    chunk[:] = bytes([9] * len(chunk))
    assert frames == [bytes([1, 2, 3])]


def test_accepts_empty_chunks_and_a_clean_empty_stream():
    decoder = FrameDecoder()
    assert decoder.push(b"") == []
    decoder.end()


@pytest.mark.parametrize("wire", [bytes([0, 0, 0]), bytes([0, 0, 0, 2, 1])], ids=["partial header", "partial payload"])
def test_rejects_a_truncated_stream_at_end(wire):
    decoder = FrameDecoder()
    assert decoder.push(wire) == []
    with pytest.raises(FrameError):
        decoder.end()


def test_rejects_an_oversized_declared_length_as_soon_as_its_header_is_complete():
    decoder = FrameDecoder({"maxFrameLength": 3})
    with pytest.raises(FrameError, match="(?i)limit"):
        decoder.push(bytes([0, 0, 0, 4]))
    with pytest.raises(FrameError, match="(?i)failed"):
        decoder.push(bytes([1]))


def test_accepts_a_frame_exactly_at_the_configured_maximum():
    decoder = FrameDecoder({"maxFrameLength": 3})
    assert decoder.push(encodeFrame(bytes([1, 2, 3]))) == [bytes([1, 2, 3])]
    decoder.end()


def test_cannot_be_pushed_after_end():
    decoder = FrameDecoder()
    decoder.end()
    with pytest.raises(FrameError, match="(?i)ended"):
        decoder.push(b"")
    with pytest.raises(FrameError, match="(?i)ended"):
        decoder.end()


@pytest.mark.parametrize("maxFrameLength", [-1, 1.5, math.nan, DEFAULT_MAX_FRAME_LENGTH * 1_000])
def test_rejects_invalid_maximum_frame_length(maxFrameLength):
    with pytest.raises(RangeError):
        FrameDecoder({"maxFrameLength": maxFrameLength})
