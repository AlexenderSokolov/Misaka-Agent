"""Experimental local server that routes clients to application-hosted durable Sessions.

Translated from pi's ``packages/server/src/index.ts``. The Unix-domain-socket transport lives in
``misaka.server.transports.unix`` and the test doubles in ``misaka.server.testing``.
"""

from __future__ import annotations

from .errors import (
    INTERNAL_SERVER_ERROR_MESSAGE,
    ServerDrainingError,
    ServerError,
    ServerOperationErrorCode,
    SessionAmbiguousError,
    SessionNotAttachedError,
    SessionNotFoundError,
    WrongServerError,
)
from .listener import ServerListener
from .server import Server
from .types import (
    MaybePromise,
    RoutedServerPresentation,
    RoutedServerServiceAttachment,
    RoutedServerServiceHost,
    RoutedSessionAttachment,
    RoutedSessionHandle,
    ServerHost,
    ServerOptions,
    SessionMetadata,
)

__all__ = [
    "INTERNAL_SERVER_ERROR_MESSAGE",
    "MaybePromise",
    "RoutedServerPresentation",
    "RoutedServerServiceAttachment",
    "RoutedServerServiceHost",
    "RoutedSessionAttachment",
    "RoutedSessionHandle",
    "Server",
    "ServerDrainingError",
    "ServerError",
    "ServerHost",
    "ServerListener",
    "ServerOperationErrorCode",
    "ServerOptions",
    "SessionAmbiguousError",
    "SessionMetadata",
    "SessionNotAttachedError",
    "SessionNotFoundError",
    "WrongServerError",
]
