"""The in-memory wire the pi server tests connect a ``ProtocolTestClient`` to a ``Server`` through.

pi's conformance and protocol tests each define this ``connect`` function; they differ in one line: whether closing
the channel twice is a no-op (``idempotentClose``).
"""

from __future__ import annotations

from typing import Any

from misaka.server import Server
from misaka.server.testing import ProtocolTestClient


def connect(server: Server, idempotentClose: bool = True) -> ProtocolTestClient:
    state: dict[str, Any] = {"closed": False, "handler": None, "client": None}

    class Connection:
        @property
        def closed(self) -> bool:
            return state["closed"]

        async def send(self, chunk: bytes) -> None:
            state["client"].receive(chunk)

        def close(self, finalChunk: bytes | None = None) -> None:
            if finalChunk:
                state["client"].receive(finalChunk)
            state["closed"] = True
            state["client"].markClosed()

    class Channel:
        async def send(self, chunk: bytes) -> None:
            state["handler"].onData(chunk)

        async def sendFragmented(self, chunk: bytes, splitAt: int) -> None:
            state["handler"].onData(chunk[:splitAt])
            state["handler"].onData(chunk[splitAt:])

        async def close(self) -> None:
            if idempotentClose and state["closed"]:
                return
            state["closed"] = True
            state["handler"].onClose()
            state["client"].markClosed()

    state["client"] = ProtocolTestClient(Channel())
    state["handler"] = server.accept(Connection())
    return state["client"]
