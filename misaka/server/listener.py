"""Translated from pi's ``packages/server/src/listener.ts``."""

from __future__ import annotations

from collections.abc import Awaitable
from typing import Protocol

from .connection import ByteConnectionAcceptor


class ServerListener(Protocol):
    """Supplies established byte connections after any required transport authentication."""

    def start(self, accept: ByteConnectionAcceptor) -> Awaitable[None]:
        """Starts listening and passes authorized connections to ``accept``."""
        ...

    def close(self) -> Awaitable[None]: ...


__all__ = ["ServerListener"]
