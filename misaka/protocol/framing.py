"""Length-prefixed framing for byte streams.

Translated from pi's ``packages/protocol/src/framing.ts``. A payload is ``bytes``, ``bytearray`` or
``memoryview`` where upstream takes a ``Uint8Array``; decoded frames are ``bytes``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from misaka.utils.js import RangeError, jsIsInteger, nullishCoalesce

FRAME_HEADER_LENGTH = 4
MAX_UINT32 = 0xFFFF_FFFF

# Default upper bound for one framed CBOR payload.
DEFAULT_MAX_FRAME_LENGTH = 16 * 1024 * 1024


class FrameError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.name = "FrameError"
        self.message = message


def resolveMaxFrameLength(options: Mapping[str, Any] | None) -> int:
    value = nullishCoalesce((options or {}).get("maxFrameLength"), DEFAULT_MAX_FRAME_LENGTH)
    if not jsIsInteger(value) or value < 0 or value > MAX_UINT32:
        raise RangeError(f"maxFrameLength must be an integer between 0 and {MAX_UINT32}")
    return int(value)


def encodeFrame(payload: bytes | bytearray | memoryview) -> bytes:
    """Prefixes a payload with its unsigned 32-bit big-endian byte length."""
    if not isinstance(payload, bytes | bytearray | memoryview):
        raise TypeError("Frame payload must be a Uint8Array")
    length = payload.nbytes if isinstance(payload, memoryview) else len(payload)
    if length > MAX_UINT32:
        raise RangeError("Frame payload exceeds the unsigned 32-bit length limit")
    return length.to_bytes(FRAME_HEADER_LENGTH, "big") + bytes(payload)


class FrameDecoder:
    """Incrementally splits arbitrary byte chunks into length-prefixed payloads.

    MISAKA fork: one growing ``bytearray`` stands in for upstream's fixed-size payload blocks; both allocate as
    bytes arrive, never the announced frame length up front."""

    def __init__(self, options: Mapping[str, Any] | None = None) -> None:
        self._maxFrameLength = resolveMaxFrameLength(options)
        self._header = bytearray()
        self._payload = bytearray()
        self._expectedPayloadLength: int | None = None
        self._state = "open"

    def push(self, chunk: bytes | bytearray | memoryview) -> list[bytes]:
        if self._state == "ended":
            raise FrameError("Frame decoder has ended")
        if self._state == "failed":
            raise FrameError("Frame decoder has failed")
        if not isinstance(chunk, bytes | bytearray | memoryview):
            raise TypeError("Frame chunk must be a Uint8Array")

        data = memoryview(chunk).cast("B")
        frames: list[bytes] = []
        offset = 0
        while offset < len(data):
            if self._expectedPayloadLength is None:
                take = min(FRAME_HEADER_LENGTH - len(self._header), len(data) - offset)
                self._header += data[offset : offset + take]
                offset += take
                if len(self._header) < FRAME_HEADER_LENGTH:
                    continue
                frameLength = int.from_bytes(self._header, "big")
                self._header.clear()
                if frameLength > self._maxFrameLength:
                    self._fail(f"Frame length {frameLength} exceeds configured limit of {self._maxFrameLength}")
                if frameLength == 0:
                    frames.append(b"")
                    continue
                self._expectedPayloadLength = frameLength
                self._payload = bytearray()

            expected = self._expectedPayloadLength
            if expected is None:
                continue
            take = min(expected - len(self._payload), len(data) - offset)
            self._payload += data[offset : offset + take]
            offset += take
            if len(self._payload) == expected:
                frames.append(bytes(self._payload))
                self._payload = bytearray()
                self._expectedPayloadLength = None
        return frames

    def end(self) -> None:
        if self._state == "ended":
            raise FrameError("Frame decoder has ended")
        if self._state == "failed":
            raise FrameError("Frame decoder has failed")
        if len(self._header) != 0 or self._expectedPayloadLength is not None:
            self._fail("Truncated frame at end of stream")
        self._state = "ended"

    def _fail(self, message: str) -> None:
        self._state = "failed"
        self._header.clear()
        self._payload = bytearray()
        self._expectedPayloadLength = None
        raise FrameError(message)


__all__ = ["DEFAULT_MAX_FRAME_LENGTH", "FrameDecoder", "FrameError", "encodeFrame"]
