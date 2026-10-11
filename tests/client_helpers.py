"""pi client's ``test/support.ts``: an in-memory byte server a ``Client`` can connect to."""

from __future__ import annotations

import asyncio
from typing import Any

from misaka.protocol import PROTOCOL_VERSION, ClientMessageDecoder, encodeServerMessage

__all__ = ["MemoryByteServer"]


class MemoryByteServer:
    def __init__(self, serverId: str = "00000000-0000-4000-8000-000000000001") -> None:
        self.messages: list[dict[str, Any]] = []
        self.serverId = serverId
        self.clientCloseCount = 0
        self.handlers: Any = None
        self._decoder = ClientMessageDecoder()
        self._messageWaiters: list[tuple[int, asyncio.Future[None]]] = []

    def connect(self, handlers: Any) -> Any:
        self.handlers = handlers
        self._decoder = ClientMessageDecoder()
        server = self
        state = {"closed": False}

        class Transport:
            async def send(self, chunk: bytes) -> None:
                for message in server._decoder.push(chunk):
                    server.messages.append(message)
                    server._resolveMessageWaiters()
                    if message["type"] == "hello":
                        server.send({"type": "hello", "version": PROTOCOL_VERSION, "serverId": server.serverId})

            def close(self) -> None:
                if state["closed"]:
                    return
                state["closed"] = True
                server.clientCloseCount += 1
                if server.handlers is handlers:
                    server.handlers = None

        return Transport()

    def waitForMessages(self, count: int) -> asyncio.Future[None]:
        future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        if len(self.messages) >= count:
            future.set_result(None)
        else:
            self._messageWaiters.append((count, future))
        return future

    def send(self, message: dict[str, Any]) -> None:
        if self.handlers is None:
            raise RuntimeError("No client connection")
        self.handlers.onData(encodeServerMessage(message))

    def sendRaw(self, chunk: bytes) -> None:
        if self.handlers is None:
            raise RuntimeError("No client connection")
        self.handlers.onData(chunk)

    def disconnect(self) -> None:
        handlers = self.handlers
        self.handlers = None
        if handlers is not None:
            handlers.onClose()

    def error(self, error: Exception) -> None:
        handlers = self.handlers
        self.handlers = None
        if handlers is not None:
            handlers.onError(error)

    def _resolveMessageWaiters(self) -> None:
        for index in range(len(self._messageWaiters) - 1, -1, -1):
            count, future = self._messageWaiters[index]
            if len(self.messages) < count:
                continue
            del self._messageWaiters[index]
            if not future.done():
                future.set_result(None)
