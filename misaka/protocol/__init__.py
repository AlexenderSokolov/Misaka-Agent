"""Runtime-neutral routed envelopes, CBOR encoding, and byte-stream framing for the experimental Pi protocol.

Translated from pi's ``packages/protocol/src/index.ts``. Protocol version 8: see ``protocol.py`` for the
envelopes, ``cbor.py`` for the encoding and ``framing.py`` for the four-byte length prefix.
"""

from __future__ import annotations

from .cbor import (
    DEFAULT_MAX_CBOR_BYTE_LENGTH,
    DEFAULT_MAX_CBOR_CONTAINER_LENGTH,
    DEFAULT_MAX_CBOR_DEPTH,
    CborError,
    CborOptions,
    decodeCbor,
    encodeCbor,
)
from .codec import (
    ClientMessageDecoder,
    ProtocolValidationError,
    ServerMessageDecoder,
    encodeClientMessage,
    encodeServerMessage,
    isSupportedProtocolVersion,
    parseClientMessage,
    parseServerMessage,
)
from .framing import DEFAULT_MAX_FRAME_LENGTH, FrameDecoder, FrameError, encodeFrame
from .protocol import (
    PROTOCOL_VERSION,
    AttachmentEnvelope,
    CancelEnvelope,
    ClientHello,
    ClientMessage,
    ProtocolError,
    ProtocolErrorCode,
    RequestEnvelope,
    ResponseEnvelope,
    RpcTarget,
    ServerHello,
    ServerHelloError,
    ServerId,
    ServerMessage,
    ServiceEventEnvelope,
    SessionTarget,
    isServerId,
)

__all__ = [
    "DEFAULT_MAX_CBOR_BYTE_LENGTH",
    "DEFAULT_MAX_CBOR_CONTAINER_LENGTH",
    "DEFAULT_MAX_CBOR_DEPTH",
    "DEFAULT_MAX_FRAME_LENGTH",
    "PROTOCOL_VERSION",
    "AttachmentEnvelope",
    "CancelEnvelope",
    "CborError",
    "CborOptions",
    "ClientHello",
    "ClientMessage",
    "ClientMessageDecoder",
    "FrameDecoder",
    "FrameError",
    "ProtocolError",
    "ProtocolErrorCode",
    "ProtocolValidationError",
    "RequestEnvelope",
    "ResponseEnvelope",
    "RpcTarget",
    "ServerHello",
    "ServerHelloError",
    "ServerId",
    "ServerMessage",
    "ServerMessageDecoder",
    "ServiceEventEnvelope",
    "SessionTarget",
    "decodeCbor",
    "encodeCbor",
    "encodeClientMessage",
    "encodeFrame",
    "encodeServerMessage",
    "isServerId",
    "isSupportedProtocolVersion",
    "parseClientMessage",
    "parseServerMessage",
]
