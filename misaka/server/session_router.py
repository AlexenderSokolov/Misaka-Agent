"""Routes presentation connections to application-hosted Sessions.

Translated from pi's ``packages/server/src/session-router.ts``. A JavaScript ``async`` function runs when it is
called and a ``.then`` callback never runs inline; here an eager ``startTask`` stands in for the first and a
``sleep(0)`` for the second, so the order in which the router's operations interleave is the one upstream has.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Awaitable, Callable
from typing import Any, TypedDict, cast

from misaka.chord.context import BACKGROUND_CONTEXT
from misaka.chord.types import Context, JsonValue, ServiceCall
from misaka.protocol import RpcTarget, SessionTarget
from misaka.utils.js import (
    aggregateError,
    allSettled,
    awaitValue,
    rejectedPromise,
    resolvedPromise,
    spawn,
    startTask,
)

from .errors import ServerDrainingError, SessionNotAttachedError
from .types import (
    PublishServiceUpdate,
    RoutedSessionAttachment,
    RoutedSessionHandle,
    ServerHost,
    SessionMetadata,
)


class SessionCleanupError(ExceptionGroup):
    pass


class _ClientAttachment:
    __slots__ = ("acquiring", "client", "id", "lease", "operations", "releasing", "session")

    def __init__(self, id: str, client: object, session: _HostedSession) -> None:
        self.id = id
        self.client = client
        self.session = session
        self.operations: set[asyncio.Future[Any]] = set()
        self.acquiring: asyncio.Future[RoutedSessionAttachment] | None = None
        self.lease: RoutedSessionAttachment | None = None
        self.releasing: asyncio.Future[None] | None = None


class _HostedSession:
    __slots__ = ("attachments", "handle", "id")

    def __init__(self, id: str, handle: RoutedSessionHandle) -> None:
        self.id = id
        self.handle = handle
        self.attachments: set[_ClientAttachment] = set()


class SessionRouterOptions(TypedDict):
    host: ServerHost
    serverId: str
    isClosing: Callable[[], bool]
    publishAttachment: Callable[[object, SessionTarget | None, Context], Awaitable[None]]
    reportError: Callable[[Any], None]


def _sessionId(metadata: SessionMetadata) -> str:
    return metadata["id"] if isinstance(metadata, dict) else metadata.id


class SessionRouter:
    def __init__(self, options: SessionRouterOptions) -> None:
        self._options = options
        self._hostedSessions: dict[str, _HostedSession] = {}
        self._openingSessions: dict[str, asyncio.Future[_HostedSession]] = {}
        self._attachmentsByClient: dict[object, _ClientAttachment] = {}
        self._disconnectedClients: set[object] = set()
        self._clientOperations: dict[object, asyncio.Future[None]] = {}
        self._closePromise: asyncio.Future[None] | None = None
        self._watchers: set[asyncio.Task[Any]] = set()

    async def executeServiceCall(
        self,
        call: ServiceCall,
        target: RpcTarget,
        client: object,
        publish: PublishServiceUpdate,
        context: Context,
    ) -> JsonValue | Any:
        admitted = await self._runForClient(client, lambda: self._startServiceCall(client, target, call, publish, context))
        return await admitted["result"]

    def attachClient(self, client: object, sessionId: str, context: Context) -> asyncio.Future[None]:
        if self._options["isClosing"]():
            return rejectedPromise(ServerDrainingError())
        return self._runForClient(client, lambda: self._attachClientNow(client, sessionId, context))

    def detachClient(self, client: object, context: Context) -> asyncio.Future[None]:
        async def detach() -> None:
            attachment = self._attachmentsByClient.get(client)
            if attachment is not None:
                await self._releaseAttachment(attachment, context)

        return self._runForClient(client, detach)

    async def removeSession(self, sessionId: str, context: Context) -> None:
        if self._options["isClosing"]():
            raise ServerDrainingError()
        hosted = self._hostedSessions.get(sessionId)
        if hosted is None:
            return
        errors: list[Any] = []
        releases = await allSettled([self._releaseAttachment(attachment, context) for attachment in list(hosted.attachments)])
        for result in releases:
            if result.status == "rejected":
                errors.append(result.reason)
        try:
            await awaitValue(hosted.handle.close(context))
        except Exception as error:  # noqa: BLE001
            errors.append(error)
        if self._hostedSessions.get(sessionId) is hosted:
            del self._hostedSessions[sessionId]
        if len(errors) == 1:
            raise errors[0]
        if len(errors) > 1:
            raise aggregateError(errors, f"Failed to close Session {sessionId}")

    async def disconnect(self, client: object, context: Context) -> None:
        self._disconnectedClients.add(client)
        try:

            async def release() -> None:
                attachment = self._attachmentsByClient.get(client)
                if attachment is not None:
                    await self._releaseAttachment(attachment, context, False)

            await self._runForClient(client, release)
        finally:
            self._disconnectedClients.discard(client)

    def close(self, context: Context) -> asyncio.Future[None]:
        if self._closePromise is None:
            self._closePromise = startTask(self._closeInternal(context))
        return self._closePromise

    async def _closeInternal(self, context: Context) -> None:
        operationPromises = list(self._clientOperations.values())
        openingPromises = list(self._openingSessions.values())
        operationResults, openingResults = await asyncio.gather(
            startTask(allSettled(operationPromises)), startTask(allSettled(openingPromises))
        )
        closeErrors: list[Any] = []
        for result in [*operationResults, *openingResults]:
            if result.status != "rejected":
                continue
            self._options["reportError"](result.reason)
            if isinstance(result.reason, SessionCleanupError):
                closeErrors.append(result.reason)
        attachmentResults = await allSettled(
            [
                self._releaseAttachment(attachment, context)
                for session in list(self._hostedSessions.values())
                for attachment in list(session.attachments)
            ]
        )
        for result in attachmentResults:
            if result.status == "rejected":
                closeErrors.append(result.reason)
        hosted = list(self._hostedSessions.values())
        closeResults = await allSettled([session.handle.close(context) for session in hosted])
        for result, session in zip(closeResults, hosted, strict=True):
            if result.status == "fulfilled":
                if self._hostedSessions.get(session.id) is session:
                    del self._hostedSessions[session.id]
                continue
            self._options["reportError"](result.reason)
            closeErrors.append(result.reason)
        self._attachmentsByClient.clear()
        self._clientOperations.clear()
        if closeErrors:
            raise aggregateError(closeErrors, "Failed to close routed Sessions")

    def _runForClient(self, client: object, operation: Callable[[], Awaitable[Any]]) -> asyncio.Future[Any]:
        previous = self._clientOperations.get(client)
        result = startTask(self._runAfter(previous, operation))
        tail = startTask(_settle(result))
        self._clientOperations[client] = tail

        def forget(_done: asyncio.Future[None]) -> None:
            if self._clientOperations.get(client) is tail:
                del self._clientOperations[client]

        tail.add_done_callback(forget)
        # MISAKA fork: a promise has no cancellation, so a caller that stops waiting leaves the operation running and
        # the client's later operations queued behind it, as upstream's do.
        return asyncio.shield(result)

    @staticmethod
    async def _runAfter(previous: asyncio.Future[None] | None, operation: Callable[[], Awaitable[Any]]) -> Any:
        # `previous.catch(() => {}).then(operation)`: never inline, and after `previous` settles however it settles.
        await asyncio.sleep(0)
        if previous is not None:
            await asyncio.wait([previous])
        return await operation()

    async def _attachClientNow(self, client: object, sessionId: str, context: Context) -> None:
        if self._options["isClosing"]() or client in self._disconnectedClients:
            raise ServerDrainingError()
        current = self._attachmentsByClient.get(client)
        if current is not None and current.session.id == sessionId:
            return
        hosted = await self._acquire(sessionId, context)
        if self._options["isClosing"]() or client in self._disconnectedClients:
            raise ServerDrainingError()
        if current is not None:
            await self._releaseAttachment(current, context, False)
        attachment = _ClientAttachment(str(uuid.uuid4()), client, hosted)
        hosted.attachments.add(attachment)
        try:
            acquiring = resolvedPromise(hosted.handle.attachClient(context))
            attachment.acquiring = acquiring
            attachment.lease = await acquiring
        except Exception:
            hosted.attachments.discard(attachment)
            raise
        if (
            self._hostedSessions.get(hosted.id) is not hosted
            or attachment not in hosted.attachments
            or client in self._disconnectedClients
            or self._options["isClosing"]()
        ):
            await self._releaseAttachment(attachment, context)
            raise ServerDrainingError()
        self._attachmentsByClient[client] = attachment
        await self._options["publishAttachment"](
            client,
            {"serverId": self._options["serverId"], "sessionId": sessionId, "attachmentId": attachment.id},
            context,
        )

    async def _startServiceCall(
        self,
        client: object,
        target: RpcTarget,
        call: ServiceCall,
        publish: PublishServiceUpdate,
        context: Context,
    ) -> dict[str, asyncio.Future[Any]]:
        attachment = self._requireAttachment(client, target)
        assert attachment.lease is not None
        result = resolvedPromise(
            attachment.lease.invokeService(
                call,
                lambda subscriptionId, update, updateContext: publish(subscriptionId, update, updateContext),
                context,
            )
        )
        self._trackOperation(attachment, result)
        return {"result": result}

    @staticmethod
    def _trackOperation(attachment: _ClientAttachment, result: asyncio.Future[Any]) -> None:
        attachment.operations.add(result)

        def remove(done: asyncio.Future[Any]) -> None:
            attachment.operations.discard(result)
            if not done.cancelled():
                done.exception()  # a failure is the caller's to see; settling it here keeps it from going unhandled

        result.add_done_callback(remove)

    def _requireAttachment(self, client: object, target: RpcTarget) -> _ClientAttachment:
        if self._options["isClosing"]() or client in self._disconnectedClients:
            raise ServerDrainingError()
        if "sessionId" not in target:
            raise SessionNotAttachedError()
        route = cast("SessionTarget", target)
        attachment = self._attachmentsByClient.get(client)
        if attachment is None or attachment.session.id != route["sessionId"] or attachment.id != route["attachmentId"]:
            raise SessionNotAttachedError()
        return attachment

    def _releaseAttachment(self, attachment: _ClientAttachment, context: Context, publish: bool = True) -> asyncio.Future[None]:
        if attachment.releasing is None:
            attachment.releasing = startTask(self._release(attachment, context, publish))
        return attachment.releasing

    async def _release(self, attachment: _ClientAttachment, context: Context, publish: bool) -> None:
        errors: list[Any] = []
        try:
            await allSettled(list(attachment.operations))
            try:
                lease = attachment.lease
                if lease is None and attachment.acquiring is not None:
                    lease = await attachment.acquiring
                if lease is not None:
                    await awaitValue(lease.release(context))
            except Exception as error:  # noqa: BLE001
                errors.append(error)
            if len(errors) == 1:
                raise errors[0]
            if len(errors) > 1:
                raise aggregateError(errors, "Failed to release Session attachment")
        finally:
            await self._clearAttachment(attachment, context, publish)

    async def _clearAttachment(self, attachment: _ClientAttachment, context: Context, publish: bool) -> None:
        attachment.session.attachments.discard(attachment)
        if self._attachmentsByClient.get(attachment.client) is attachment:
            del self._attachmentsByClient[attachment.client]
            if publish:
                await self._options["publishAttachment"](attachment.client, None, context)

    async def _acquire(self, sessionId: str, context: Context) -> _HostedSession:
        existing = self._hostedSessions.get(sessionId)
        if existing is not None:
            return existing
        opening = self._openingSessions.get(sessionId)
        if opening is not None:
            return await opening
        pending = startTask(self._open(sessionId, context))
        self._openingSessions[sessionId] = pending
        try:
            return await pending
        finally:
            if self._openingSessions.get(sessionId) is pending:
                del self._openingSessions[sessionId]

    async def _open(self, sessionId: str, context: Context) -> _HostedSession:
        host = self._options["host"]
        metadata = await awaitValue(host.resolveSession(sessionId, context))
        handle = await awaitValue(host.openSession(metadata, context))
        if self._options["isClosing"]():
            try:
                await awaitValue(handle.close(context))
            except Exception as error:  # noqa: BLE001
                self._options["reportError"](error)
                raise SessionCleanupError(
                    "Failed to close routed Session acquired while draining", [ServerDrainingError(), error]
                )
            raise ServerDrainingError()
        hosted = _HostedSession(_sessionId(metadata), handle)
        self._hostedSessions[hosted.id] = hosted
        terminated = getattr(handle, "terminated", None)
        if terminated is not None:
            spawn(self._watchTermination(hosted, terminated), self._watchers)
        return hosted

    async def _watchTermination(self, hosted: _HostedSession, terminated: Awaitable[Exception | None]) -> None:
        try:
            error = await terminated
        except Exception as failure:  # noqa: BLE001
            self._invalidate(hosted, failure)
            return
        self._invalidate(hosted, error)

    def _invalidate(self, hosted: _HostedSession, error: Exception | None) -> None:
        if self._hostedSessions.get(hosted.id) is not hosted:
            return
        del self._hostedSessions[hosted.id]
        for attachment in list(hosted.attachments):
            spawn(self._releaseAndReport(attachment), self._watchers)
        if error is not None:
            self._options["reportError"](error)

    async def _releaseAndReport(self, attachment: _ClientAttachment) -> None:
        try:
            await self._releaseAttachment(attachment, BACKGROUND_CONTEXT)
        except Exception as releaseError:  # noqa: BLE001
            self._options["reportError"](releaseError)


async def _settle(future: asyncio.Future[Any]) -> None:
    """A future that finishes when ``future`` settles, whatever it settles with (a cancellation included)."""
    await asyncio.wait([future])
    if not future.cancelled():
        future.exception()


__all__ = ["SessionCleanupError", "SessionRouter", "SessionRouterOptions"]
