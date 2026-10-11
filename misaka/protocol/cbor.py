"""The protocol's strict, definite-length RFC 8949 subset.

Translated from pi's ``packages/protocol/src/cbor/{options,encoder,decoder,index}.ts``.

A value is ``None``, a ``bool``, an ``int`` or finite ``float``, a ``str``, a byte string (``bytes``,
``bytearray`` or ``memoryview``), a ``list`` or a ``dict`` with ``str`` keys. JavaScript has one number type, so
an integral ``float`` encodes as an integer, as ``2.0`` is ``2`` upstream (negative zero stays a float), and an
integral float64 item decodes to an ``int``. A dict value of ``UNDEFINED`` is left out, as an ``undefined``
property is.
"""

from __future__ import annotations

import math
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, TypedDict

from misaka.utils.js import (
    UNDEFINED,
    RangeError,
    jsIsInteger,
    nullishCoalesce,
    toWellFormed,
)

UINT32_BASE = 0x1_0000_0000
MAX_UINT32 = 0xFFFF_FFFF
_MAX_CONFIGURED_DEPTH = 512
_MAX_SAFE_INTEGER = 2**53 - 1

# Safe defaults for untrusted protocol payloads.
DEFAULT_MAX_CBOR_BYTE_LENGTH = 16 * 1024 * 1024
DEFAULT_MAX_CBOR_CONTAINER_LENGTH = 1_000_000
DEFAULT_MAX_CBOR_DEPTH = 64


class CborOptions(TypedDict, total=False):
    # Maximum encoded input/output bytes and maximum byte/text string length.
    maxByteLength: int
    # Maximum number of elements in an array or entries in a map.
    maxContainerLength: int
    # Maximum recursive item depth.
    maxDepth: int


@dataclass(frozen=True, slots=True)
class ResolvedCborOptions:
    maxByteLength: int
    maxContainerLength: int
    maxDepth: int


class CborError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.name = "CborError"
        self.message = message


def resolveLimit(name: str, value: Any, maximum: int) -> int:
    if not jsIsInteger(value) or value < 0 or value > maximum:
        raise RangeError(f"{name} must be an integer between 0 and {maximum}")
    return int(value)


def resolveOptions(options: Mapping[str, Any] | None) -> ResolvedCborOptions:
    options = options or {}
    return ResolvedCborOptions(
        maxByteLength=resolveLimit(
            "maxByteLength", nullishCoalesce(options.get("maxByteLength"), DEFAULT_MAX_CBOR_BYTE_LENGTH), MAX_UINT32
        ),
        maxContainerLength=resolveLimit(
            "maxContainerLength",
            nullishCoalesce(options.get("maxContainerLength"), DEFAULT_MAX_CBOR_CONTAINER_LENGTH),
            MAX_UINT32,
        ),
        maxDepth=resolveLimit(
            "maxDepth", nullishCoalesce(options.get("maxDepth"), DEFAULT_MAX_CBOR_DEPTH), _MAX_CONFIGURED_DEPTH
        ),
    )


def _typeName(value: Any) -> str:
    """``typeof value``, for the values CBOR does not hold."""
    if value is UNDEFINED:
        return "undefined"
    if callable(value):
        return "function"
    return "object"


# ─── Encoder ─────────────────────────────────────────────────────────────────


class CborWriter:
    def __init__(self, maxByteLength: int) -> None:
        self._buffer = bytearray()
        self._maxByteLength = maxByteLength

    def write(self, data: bytes | bytearray | memoryview) -> None:
        # A memoryview of wider items counts bytes, not items.
        size = data.nbytes if isinstance(data, memoryview) else len(data)
        if len(self._buffer) + size > self._maxByteLength:
            raise CborError(f"CBOR byte length exceeds configured limit of {self._maxByteLength}")
        self._buffer += data

    def writeByte(self, value: int) -> None:
        self.write(bytes((value,)))

    def finish(self) -> bytes:
        return bytes(self._buffer)


def writeArgument(writer: CborWriter, majorType: int, value: int) -> None:
    prefix = majorType << 5
    if value < 24:
        writer.writeByte(prefix | value)
    elif value <= 0xFF:
        writer.write(bytes((prefix | 24, value)))
    elif value <= 0xFFFF:
        writer.write(bytes((prefix | 25,)) + value.to_bytes(2, "big"))
    elif value <= MAX_UINT32:
        writer.write(bytes((prefix | 26,)) + value.to_bytes(4, "big"))
    else:
        writer.write(bytes((prefix | 27,)) + value.to_bytes(8, "big"))


def encodeText(writer: CborWriter, value: str, options: ResolvedCborOptions) -> None:
    # ``TextEncoder`` replaces a lone surrogate with U+FFFD: the length is checked on that, then the round trip.
    wellFormed = True
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError:
        wellFormed = False
        encoded = toWellFormed(value).encode("utf-8")
    if len(encoded) > options.maxByteLength:
        raise CborError(f"CBOR text string length exceeds configured limit of {options.maxByteLength}")
    if not wellFormed:
        raise CborError("CBOR text strings must contain valid Unicode scalar values")
    writeArgument(writer, 3, len(encoded))
    writer.write(encoded)


def encodeValue(writer: CborWriter, value: Any, options: ResolvedCborOptions, depth: int, ancestors: set[int]) -> None:
    if depth > options.maxDepth:
        raise CborError(f"CBOR nesting depth exceeds configured limit of {options.maxDepth}")

    if value is None:
        writer.writeByte(0xF6)
        return
    if isinstance(value, bool):
        writer.writeByte(0xF5 if value else 0xF4)
        return
    if isinstance(value, int | float):
        if isinstance(value, float):
            if not math.isfinite(value):
                raise CborError("CBOR numbers must be finite")
            if not value.is_integer() or (value == 0 and math.copysign(1, value) < 0):
                writer.write(b"\xfb" + struct.pack(">d", value))
                return
            value = int(value)
        if abs(value) > _MAX_SAFE_INTEGER:
            raise CborError("CBOR integers must be safe JavaScript integers")
        if value >= 0:
            writeArgument(writer, 0, value)
        else:
            writeArgument(writer, 1, -1 - value)
        return
    if isinstance(value, str):
        encodeText(writer, value, options)
        return
    if isinstance(value, bytes | bytearray | memoryview):
        length = value.nbytes if isinstance(value, memoryview) else len(value)
        if length > options.maxByteLength:
            raise CborError(f"CBOR byte string length exceeds configured limit of {options.maxByteLength}")
        writeArgument(writer, 2, length)
        writer.write(value)
        return
    if isinstance(value, list):
        if id(value) in ancestors:
            raise CborError("CBOR values must not contain cycles")
        if len(value) > options.maxContainerLength:
            raise CborError(f"CBOR array length exceeds configured limit of {options.maxContainerLength}")
        ancestors.add(id(value))
        try:
            writeArgument(writer, 4, len(value))
            for item in value:
                if item is UNDEFINED:
                    raise CborError("CBOR arrays must not contain holes or undefined values")
                encodeValue(writer, item, options, depth + 1, ancestors)
        finally:
            ancestors.discard(id(value))
        return
    if type(value) is dict:
        if id(value) in ancestors:
            raise CborError("CBOR values must not contain cycles")
        for key in value:
            if not isinstance(key, str):
                raise CborError("CBOR map keys must be strings")
        entries = [(key, entry) for key, entry in value.items() if entry is not UNDEFINED]
        if len(entries) > options.maxContainerLength:
            raise CborError(f"CBOR map length exceeds configured limit of {options.maxContainerLength}")
        ancestors.add(id(value))
        try:
            writeArgument(writer, 5, len(entries))
            for key, entry in entries:
                encodeText(writer, key, options)
                encodeValue(writer, entry, options, depth + 1, ancestors)
        finally:
            ancestors.discard(id(value))
        return

    raise CborError(f"Unsupported CBOR value type: {_typeName(value)}")


def encodeCbor(value: Any, options: Mapping[str, Any] | None = None) -> bytes:
    """Encodes the protocol's strict, definite-length RFC 8949 subset."""
    resolved = resolveOptions(options)
    writer = CborWriter(resolved.maxByteLength)
    encodeValue(writer, value, resolved, 0, set())
    return writer.finish()


# ─── Decoder ─────────────────────────────────────────────────────────────────


class CborReader:
    def __init__(self, data: bytes | bytearray | memoryview, options: ResolvedCborOptions) -> None:
        self._bytes = memoryview(data).cast("B")
        self._offset = 0
        self._options = options

    def decode(self) -> Any:
        value = self._readItem(0)
        if self._offset != len(self._bytes):
            raise CborError("CBOR payload contains trailing data")
        return value

    def _readItem(self, depth: int) -> Any:
        if depth > self._options.maxDepth:
            raise CborError(f"CBOR nesting depth exceeds configured limit of {self._options.maxDepth}")
        initial = self._readByte()
        majorType = initial >> 5
        additionalInformation = initial & 0x1F

        if majorType == 0:
            return self._readArgument(additionalInformation)
        if majorType == 1:
            value = -1 - self._readArgument(additionalInformation)
            if abs(value) > _MAX_SAFE_INTEGER:
                raise CborError("Decoded CBOR integer is outside the safe range")
            return value
        if majorType == 2:
            length = self._readLength(additionalInformation, "byte string", self._options.maxByteLength)
            return bytes(self._readBytes(length))
        if majorType == 3:
            length = self._readLength(additionalInformation, "text string", self._options.maxByteLength)
            encoded = self._readBytes(length)
            try:
                return str(encoded, "utf-8")
            except UnicodeDecodeError:
                raise CborError("CBOR text string contains invalid UTF-8") from None
        if majorType == 4:
            length = self._readLength(additionalInformation, "array", self._options.maxContainerLength)
            return [self._readItem(depth + 1) for _ in range(length)]
        if majorType == 5:
            length = self._readLength(additionalInformation, "map", self._options.maxContainerLength)
            result: dict[str, Any] = {}
            for _ in range(length):
                key = self._readItem(depth + 1)
                if not isinstance(key, str):
                    raise CborError("CBOR map keys must be strings")
                if key in result:
                    raise CborError("CBOR map contains a duplicate key")
                result[key] = self._readItem(depth + 1)
            return result
        if majorType == 6:
            raise CborError("CBOR tags are not supported")
        return self._readSimple(additionalInformation)

    def _readSimple(self, additionalInformation: int) -> Any:
        if additionalInformation == 20:
            return False
        if additionalInformation == 21:
            return True
        if additionalInformation == 22:
            return None
        if additionalInformation == 27:
            (value,) = struct.unpack(">d", self._readBytes(8))
            if not math.isfinite(value):
                raise CborError("Decoded CBOR number must be finite")
            if value.is_integer():
                if abs(value) > _MAX_SAFE_INTEGER:
                    raise CborError("Decoded CBOR integer is outside the safe range")
                if not (value == 0 and math.copysign(1, value) < 0):
                    return int(value)
            return value
        if additionalInformation == 31:
            raise CborError("CBOR break marker is not supported")
        raise CborError("Unsupported CBOR simple value or floating-point width")

    def _readLength(self, additionalInformation: int, kind: str, limit: int) -> int:
        if additionalInformation == 31:
            raise CborError(f"Indefinite-length CBOR {kind}s are not supported")
        length = self._readArgument(additionalInformation)
        if length > limit:
            raise CborError(f"CBOR {kind} length exceeds configured limit of {limit}")
        return length

    def _readArgument(self, additionalInformation: int) -> int:
        if additionalInformation < 24:
            return additionalInformation
        if additionalInformation == 24:
            return self._readByte()
        if additionalInformation == 25:
            return int.from_bytes(self._readBytes(2), "big")
        if additionalInformation == 26:
            return int.from_bytes(self._readBytes(4), "big")
        if additionalInformation == 27:
            high = self._readArgument(26)
            low = self._readArgument(26)
            if high > 0x1F_FFFF:
                raise CborError("Decoded CBOR integer or length is outside the safe range")
            return high * UINT32_BASE + low
        if additionalInformation == 31:
            raise CborError("Indefinite-length CBOR items are not supported")
        raise CborError("Malformed CBOR additional information")

    def _readByte(self) -> int:
        if self._offset >= len(self._bytes):
            raise CborError("Truncated CBOR payload")
        value = self._bytes[self._offset]
        self._offset += 1
        return value

    def _readBytes(self, length: int) -> memoryview:
        if length > len(self._bytes) - self._offset:
            raise CborError("Truncated CBOR payload")
        value = self._bytes[self._offset : self._offset + length]
        self._offset += length
        return value


def decodeCbor(data: bytes | bytearray | memoryview, options: Mapping[str, Any] | None = None) -> Any:
    """Decodes exactly one item from the protocol's strict RFC 8949 subset."""
    if not isinstance(data, bytes | bytearray | memoryview):
        raise TypeError("CBOR input must be a Uint8Array")
    resolved = resolveOptions(options)
    length = data.nbytes if isinstance(data, memoryview) else len(data)
    if length > resolved.maxByteLength:
        raise CborError(f"CBOR byte length exceeds configured limit of {resolved.maxByteLength}")
    return CborReader(data, resolved).decode()


__all__ = [
    "DEFAULT_MAX_CBOR_BYTE_LENGTH",
    "DEFAULT_MAX_CBOR_CONTAINER_LENGTH",
    "DEFAULT_MAX_CBOR_DEPTH",
    "CborError",
    "CborOptions",
    "decodeCbor",
    "encodeCbor",
]
