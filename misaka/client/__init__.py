"""Transport-neutral client for the experimental Pi service protocol.

Translated from pi's ``packages/client/src/index.ts``. ``Client.connect(options)`` is ``connectClient(options)``
here: ``connect`` is the instance method that connects an existing client, as it is upstream.
"""

from __future__ import annotations

from .client import Client, connectClient, createClientServiceTransport
from .errors import ClientDisposedError, DisconnectedError, ServerError
from .transport import ByteTransport, ByteTransportFactory, ByteTransportHandlers
from .types import (
    AttachmentChangeListener,
    ClientOptions,
    ConnectionState,
    ConnectionStateChange,
    ListenerErrorHandler,
    ServiceSubscription,
    Unsubscribe,
)

__all__ = [
    "AttachmentChangeListener",
    "ByteTransport",
    "ByteTransportFactory",
    "ByteTransportHandlers",
    "Client",
    "ClientDisposedError",
    "ClientOptions",
    "ConnectionState",
    "ConnectionStateChange",
    "DisconnectedError",
    "ListenerErrorHandler",
    "ServerError",
    "ServiceSubscription",
    "Unsubscribe",
    "connectClient",
    "createClientServiceTransport",
]
