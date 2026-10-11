"""Byte transports.

Translated from pi's ``packages/client/src/transport.ts``. A transport is any object with ``send(chunk)``
(an awaitable; calls are delivered in invocation order) and ``close()`` (repeated calls are harmless); the
handlers are an object with ``onData(chunk)``, ``onClose()`` and ``onError(error)``. A factory is a function
(or coroutine function) taking the handlers and returning a connected, authenticated transport; exactly one
terminal handler is expected.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Protocol


class ByteTransport(Protocol):
    def send(self, chunk: bytes) -> Awaitable[None]:
        """Sends one byte chunk. Calls must be delivered in invocation order."""
        ...

    def close(self) -> None:
        """Closes the transport. Implementations must make repeated calls harmless."""
        ...


class ByteTransportHandlers(Protocol):
    def onData(self, chunk: bytes) -> None:
        """Delivers an arbitrary inbound byte chunk."""
        ...

    def onClose(self) -> None:
        """Reports an orderly terminal close."""
        ...

    def onError(self, error: Exception) -> None:
        """Reports a terminal transport failure."""
        ...


type ByteTransportFactory = Callable[[ByteTransportHandlers], ByteTransport | Awaitable[ByteTransport]]

__all__ = ["ByteTransport", "ByteTransportFactory", "ByteTransportHandlers"]
