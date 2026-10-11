"""A protocol-level client for server tests.

Translated from pi's ``packages/server/src/testing/client.ts``. The client reads server messages through a wire
channel, keeps them, and lets a test wait for the next message that satisfies a predicate. Create one inside a
running event loop.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Callable
from typing import Any, Protocol

from misaka.protocol import PROTOCOL_VERSION, ServerMessageDecoder, encodeClientMessage
from misaka.utils.js import rejectedPromise, resolvedPromise, startTask

from .host import Deferred


class _MessageWaiter:
    __slots__ = ("future", "predicate")

    def __init__(self, predicate: Callable[[Any], bool], future: asyncio.Future[Any]) -> None:
        self.predicate = predicate
        self.future = future


class WireChannel(Protocol):
    def send(self, chunk: bytes) -> Any: ...

    def sendFragmented(self, chunk: bytes, splitAt: int) -> Any: ...

    def close(self) -> Any: ...


class ProtocolTestClient:
    def __init__(self, channel: WireChannel) -> None:
        self.messages: list[Any] = []
        self._channel = channel
        self._decoder = ServerMessageDecoder()
        # A ``Set``: waiters settle in the order they were registered.
        self._waiters: dict[_MessageWaiter, None] = {}
        self._closedDeferred = Deferred()
        self._requestSequence = 0
        self._attachment: dict[str, str] | None = None
        self._closedValue = False

    @property
    def closed(self) -> bool:
        return self._closedValue

    def hello(self, version: int = PROTOCOL_VERSION) -> asyncio.Future[Any]:
        response = self.next(lambda message: message["type"] == "hello" or message["type"] == "hello_error")
        self.sendMessage({"type": "hello", "version": version})
        return response

    def requestService(self, target: Any, call: Any, id: str | None = None) -> asyncio.Future[Any]:
        if id is None:
            self._requestSequence += 1
            id = f"request-{self._requestSequence}"
        return startTask(self._requestService(target, call, id))

    async def _requestService(self, target: Any, call: Any, id: str) -> Any:
        response = self.next(lambda message: message["type"] == "response" and message["id"] == id)
        await self.sendMessage({"type": "request", "id": id, "target": target, "call": call})
        return await response

    def attach(self, serverId: str, sessionId: str) -> asyncio.Future[Any]:
        return self.requestService(
            {"serverId": serverId},
            {"serviceId": "pi.session-management", "member": "attach", "args": [sessionId]},
        )

    def requestSessionService(
        self, serverId: str, sessionId: str, call: Any, id: str | None = None
    ) -> asyncio.Future[Any]:
        attachment = self._attachment
        target: Any = (
            {"serverId": serverId, "sessionId": sessionId, "attachmentId": "missing-attachment"}
            if attachment is None or attachment["sessionId"] != sessionId
            else {"serverId": serverId, **attachment}
        )
        return self.requestService(target, call, id)

    def sendMessage(self, message: Any) -> asyncio.Future[None]:
        return startTask(self._channel.send(encodeClientMessage(message)))

    def sendBytes(self, chunk: bytes) -> asyncio.Future[None]:
        return startTask(self._channel.send(chunk))

    def sendFragmentedMessage(self, message: Any, splitAt: int) -> asyncio.Future[None]:
        return startTask(self._channel.sendFragmented(encodeClientMessage(message), splitAt))

    def next(self, predicate: Callable[[Any], bool]) -> asyncio.Future[Any]:
        return self.nextFrom(0, predicate)

    def nextFrom(self, index: int, predicate: Callable[[Any], bool]) -> asyncio.Future[Any]:
        existing = next((message for message in self.messages[index:] if predicate(message)), None)
        if existing is not None:
            return resolvedPromise(existing)
        if self._closedValue:
            return rejectedPromise(RuntimeError("Wire client is closed"))
        future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        self._waiters[_MessageWaiter(predicate, future)] = None
        return future

    def waitForClose(self) -> asyncio.Future[Any]:
        return resolvedPromise(None) if self._closedValue else self._closedDeferred.promise

    def close(self) -> asyncio.Future[Any]:
        return startTask(self._channel.close())

    def receive(self, chunk: bytes) -> None:
        try:
            for message in self._decoder.push(chunk):
                if message["type"] == "attachment":
                    attachment = message["attachment"]
                    self._attachment = (
                        None
                        if attachment is None
                        else {"sessionId": attachment["sessionId"], "attachmentId": attachment["attachmentId"]}
                    )
                self.messages.append(message)
                for waiter in list(self._waiters):
                    if not waiter.predicate(message):
                        continue
                    self._waiters.pop(waiter, None)
                    if not waiter.future.done():
                        waiter.future.set_result(message)
        except Exception as error:  # noqa: BLE001
            self.fail(error)

    def markClosed(self) -> None:
        if self._closedValue:
            return
        self._closedValue = True
        self._closedDeferred.resolve(None)
        self.fail(RuntimeError("Wire connection closed"))

    def fail(self, error: Exception) -> None:
        for waiter in list(self._waiters):
            if not waiter.future.done():
                waiter.future.set_exception(error)
        self._waiters.clear()


class _UnixChannel:
    def __init__(self, writer: asyncio.StreamWriter) -> None:
        self._writer = writer
        self.closedFuture: asyncio.Future[None] = asyncio.get_running_loop().create_future()

    async def send(self, chunk: bytes) -> None:
        self._writer.write(chunk)
        await self._writer.drain()

    async def sendFragmented(self, chunk: bytes, splitAt: int) -> None:
        await self.send(chunk[:splitAt])
        await self.send(chunk[splitAt:])

    async def close(self) -> None:
        if self.closedFuture.done():
            return
        self._writer.transport.abort()
        await self.closedFuture


async def connectUnixTestClient(path: str) -> ProtocolTestClient:
    # MISAKA fork: asyncio has no Unix sockets on Windows, where Node connects to a named pipe instead.
    if sys.platform == "win32":
        raise RuntimeError("Unix transport is not supported on Windows")
    reader, writer = await asyncio.open_unix_connection(path)
    channel = _UnixChannel(writer)
    client = ProtocolTestClient(channel)

    async def pump() -> None:
        try:
            while chunk := await reader.read(65536):
                client.receive(chunk)
        except Exception as error:  # noqa: BLE001
            client.fail(error)
        finally:
            writer.transport.abort()
            client.markClosed()
            if not channel.closedFuture.done():
                channel.closedFuture.set_result(None)

    client._pump = asyncio.ensure_future(pump())  # type: ignore[attr-defined]
    return client


__all__ = ["ProtocolTestClient", "WireChannel", "connectUnixTestClient"]
