"""Validation, encoding and incremental decoding of framed protocol messages.

Translated from pi's ``packages/protocol/src/codec.ts``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from misaka.chord.json import isJsonValue
from misaka.utils.js import jsLength, jsSlice, nullishCoalesce

from .cbor import decodeCbor, encodeCbor
from .framing import DEFAULT_MAX_FRAME_LENGTH, FrameDecoder, encodeFrame
from .protocol import (
    PROTOCOL_VERSION,
    ClientMessage,
    ServerMessage,
    checkClientMessage,
    checkServerMessage,
)


class ProtocolValidationError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.name = "ProtocolValidationError"
        self.message = message


def parseClientMessage(value: Any) -> ClientMessage:
    if not checkClientMessage(value) or not isJsonValue(value):
        raise ProtocolValidationError("Invalid client protocol message")
    return value


def parseServerMessage(value: Any) -> ServerMessage:
    if not checkServerMessage(value) or not isJsonValue(value):
        raise ProtocolValidationError("Invalid server protocol message")
    return value


def _boundedErrorMessage(error: Any) -> str:
    if not isinstance(error, Exception):
        return "Unknown codec error"
    message = str(error)
    return message if jsLength(message) <= 500 else f"{jsSlice(message, 0, 497)}..."


def _encodeProtocolMessage(
    value: Any, parse: Callable[[Any], Any], kind: str, options: Mapping[str, Any] | None
) -> bytes:
    validated = parse(value)
    try:
        maxFrameLength = nullishCoalesce((options or {}).get("maxFrameLength"), DEFAULT_MAX_FRAME_LENGTH)
        return encodeFrame(encodeCbor(validated, {"maxByteLength": maxFrameLength}))
    except ProtocolValidationError:
        raise
    except Exception as error:  # noqa: BLE001 - reported with the bounded message
        raise ProtocolValidationError(f"Unable to encode {kind} protocol message: {_boundedErrorMessage(error)}") from None


def encodeClientMessage(message: ClientMessage, options: Mapping[str, Any] | None = None) -> bytes:
    """Validates and encodes one complete length-prefixed client message."""
    return _encodeProtocolMessage(message, parseClientMessage, "client", options)


def encodeServerMessage(message: ServerMessage, options: Mapping[str, Any] | None = None) -> bytes:
    """Validates and encodes one complete length-prefixed server message."""
    return _encodeProtocolMessage(message, parseServerMessage, "server", options)


class ValidatedMessageDecoder:
    def __init__(self, kind: str, parse: Callable[[Any], Any], options: Mapping[str, Any] | None = None) -> None:
        self._failed = False
        self._frames = FrameDecoder(options)
        self._kind = kind
        self._maxFrameLength = nullishCoalesce((options or {}).get("maxFrameLength"), DEFAULT_MAX_FRAME_LENGTH)
        self._parse = parse

    def push(self, chunk: bytes | bytearray | memoryview) -> list[Any]:
        if self._failed:
            raise ProtocolValidationError(f"{self._kind} message decoder has failed")
        try:
            messages = []
            for frame in self._frames.push(chunk):
                messages.append(self._parse(decodeCbor(frame, {"maxByteLength": self._maxFrameLength})))
            return messages
        except Exception as error:
            self._failed = True
            if isinstance(error, ProtocolValidationError):
                raise
            raise ProtocolValidationError(f"Invalid {self._kind} protocol frame: {_boundedErrorMessage(error)}") from None

    def end(self) -> None:
        if self._failed:
            raise ProtocolValidationError(f"{self._kind} message decoder has failed")
        try:
            self._frames.end()
        except Exception as error:  # noqa: BLE001
            self._failed = True
            raise ProtocolValidationError(f"Invalid {self._kind} protocol framing: {_boundedErrorMessage(error)}") from None


class ClientMessageDecoder:
    """Incrementally decodes and validates framed client messages."""

    def __init__(self, options: Mapping[str, Any] | None = None) -> None:
        self._decoder = ValidatedMessageDecoder("client", parseClientMessage, options)

    def push(self, chunk: bytes | bytearray | memoryview) -> list[ClientMessage]:
        return self._decoder.push(chunk)

    def end(self) -> None:
        self._decoder.end()


class ServerMessageDecoder:
    """Incrementally decodes and validates framed server messages."""

    def __init__(self, options: Mapping[str, Any] | None = None) -> None:
        self._decoder = ValidatedMessageDecoder("server", parseServerMessage, options)

    def push(self, chunk: bytes | bytearray | memoryview) -> list[ServerMessage]:
        return self._decoder.push(chunk)

    def end(self) -> None:
        self._decoder.end()


def isSupportedProtocolVersion(version: Any) -> bool:
    return not isinstance(version, bool) and isinstance(version, int | float) and float(version).is_integer() and version == PROTOCOL_VERSION


__all__ = [
    "ClientMessageDecoder",
    "ProtocolValidationError",
    "ServerMessageDecoder",
    "encodeClientMessage",
    "encodeServerMessage",
    "isSupportedProtocolVersion",
    "parseClientMessage",
    "parseServerMessage",
]
