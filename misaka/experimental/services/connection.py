"""Translated from pi's ``coding-agent/src/experimental/services/connection.ts``.

A *service source* binds the services a presentation consumes to one client connection: ``createServerServiceSource`` for
the server-scoped services and ``createSessionServiceSource`` for those of the selected Session, which follow the
client's attachment. A source is a ``RemoteServiceSource`` of the facet host (``acceptsUnavailableServices``,
``catalogue(context)`` and ``open(options)``) and also provides the connection or attachment state
(``connection`` or ``attachment``, replicated states) and ``dispose(context)``; the Session source adds
``whenAttached(sessionId, context)`` and ``whenDetached(context)``.

The server connection state is ``{"status": "connecting", "attempt"}``, ``{"status": "connected", "since"}`` or
``{"status": "disconnected", "since", "reason", "retryAt"}``; the attachment state is ``{"status": "detached"}`` or
``{"status": "attaching" | "attached" | "degraded", "sessionId"}``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from misaka.chord import Context, createRemoteServiceBinding, replicatedState
from misaka.chord.context import BACKGROUND_CONTEXT
from misaka.client import createClientServiceTransport
from misaka.utils.js import (
    allSettled,
    errorMessage,
    promiseAll,
    resolvedPromise,
    startTask,
)


def toError(error: BaseException) -> Exception:
    return error if isinstance(error, Exception) else RuntimeError(str(error))


class RoutedServiceBinding:
    def __init__(self, options: dict[str, Any]) -> None:
        self._services = createRemoteServiceBinding(
            {
                "services": options["services"],
                "transport": options["transport"],
                "bound": False,
                "assertAccess": options["assertAccess"],
                "onError": options["onError"],
            }
        )
        self._getBound: Callable[[], bool] = options["getBound"]
        self._onActivate: Callable[[Context], Any] = options.get("onActivate") or (lambda context: resolvedPromise(None))
        self._remove: Callable[[], None] = options["remove"]
        self._activated = False
        self._activationComplete = False

    def use(self, service: Any) -> Any:
        return self._services.use(service)

    def observe(self, service: Any, handler: Callable[..., Any]) -> Callable[[], None]:
        return self._services.observe(service, handler)

    async def ready(self, context: Context) -> None:
        if not self._activated:
            self._activated = True
            await self._services.rebind(self._getBound(), context)
        await self.serviceReady(context)
        if not self._activationComplete:
            await self._onActivate(context)
            self._activationComplete = True

    def serviceReady(self, context: Context) -> Any:
        return self._services.ready(context)

    def updateBound(self, bound: bool, context: Context) -> Any:
        if self._activated:
            return self._services.rebind(bound, context)
        return _done()

    async def dispose(self, context: Context) -> None:
        self._remove()
        await self._services.dispose(context)


def _done() -> asyncio.Future[None]:
    future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
    future.set_result(None)
    return future


class ServerServiceSourceImpl:
    acceptsUnavailableServices = False

    def __init__(self, client: Any, options: dict[str, Any]) -> None:
        self._client = client
        self._transport = createClientServiceTransport(client, lambda: {"serverId": client.serverId})
        self._onError: Callable[[Exception], None] = options.get("onError") or (lambda error: None)
        self._bindings: list[RoutedServiceBinding] = []
        self._transition: asyncio.Future[None] = _done()
        self._connectionAttempt = 1 if client.connectionState == "connecting" else 0
        self._disposed = False
        connectionState = replicatedState(toServerConnectionState(client, self._connectionAttempt))
        self.connection = connectionState

        def onConnectionState(change: dict[str, Any]) -> None:
            state, error = change["state"], change.get("error")
            if state == "connecting":
                self._connectionAttempt += 1
            publishReplacement(
                connectionState, toServerConnectionState(client, self._connectionAttempt, error), BACKGROUND_CONTEXT
            )
            previous = self._transition

            async def transition() -> None:
                try:
                    await previous
                    results = await allSettled(
                        [binding.updateBound(state == "connected", BACKGROUND_CONTEXT) for binding in list(self._bindings)]
                    )
                    failures = [result.reason for result in results if result.status == "rejected"]
                    if len(failures) > 0:
                        raise ExceptionGroup("", [toError(failure) for failure in failures if failure is not None])
                except Exception as transitionError:  # noqa: BLE001 - reported
                    self._onError(toError(transitionError))

            self._transition = asyncio.ensure_future(transition())

        self._removeConnectionListener = client.onConnectionStateChange(onConnectionState)

    def catalogue(self, context: Context) -> Any:
        return self._client.serviceCatalogue({"serverId": self._client.serverId}, context.abortSignal)

    def open(self, options: dict[str, Any]) -> RoutedServiceBinding:
        if self._disposed:
            raise RuntimeError("Server service source is disposed")
        holder: dict[str, RoutedServiceBinding] = {}
        binding = RoutedServiceBinding(
            {
                "services": options["services"],
                "transport": self._transport,
                "getBound": lambda: self._client.connected,
                "assertAccess": options["assertAccess"],
                "onError": options["onError"],
                "remove": lambda: self._discard(holder["binding"]),
            }
        )
        holder["binding"] = binding
        self._bindings.append(binding)
        return binding

    def _discard(self, binding: RoutedServiceBinding) -> None:
        if binding in self._bindings:
            self._bindings.remove(binding)

    async def dispose(self, context: Context) -> None:
        if self._disposed:
            return
        self._disposed = True
        self._removeConnectionListener()
        await self._transition
        bindings = list(self._bindings)
        self._bindings.clear()
        results = await allSettled([binding.dispose(context) for binding in bindings])
        throwFailures(results, "Failed to dispose server service source")


class SessionServiceSourceImpl:
    def __init__(self, client: Any, options: dict[str, Any]) -> None:
        self._client = client
        self._transport = createClientServiceTransport(client, lambda: client.attachment)
        self._onError: Callable[[Exception], None] = options.get("onError") or (lambda error: None)
        self._bindings: list[RoutedServiceBinding] = []
        self._transitions: set[asyncio.Future[None]] = set()
        self._catalogueValue: list[Any] | None = None
        self._attachmentRevision = 0
        self._disposed = False
        attachment = client.attachment
        self._attachmentState = replicatedState(
            {"status": "detached"} if attachment is None else {"status": "attaching", "sessionId": attachment["sessionId"]}
        )
        self.attachment = self._attachmentState

        def onAttachment(attachment: Any) -> None:
            self._attachmentRevision += 1
            revision = self._attachmentRevision
            if attachment is not None:
                publishReplacement(
                    self._attachmentState, {"status": "attaching", "sessionId": attachment["sessionId"]}, BACKGROUND_CONTEXT
                )
                loading = startTask(self._client.serviceCatalogue(attachment))

                def loaded(future: asyncio.Future[Any]) -> None:
                    if future.cancelled():
                        return
                    error = future.exception()
                    if error is not None:
                        self._onError(toError(error))
                        return
                    if self._attachmentRevision == revision and sameAttachment(self._client.attachment, attachment):
                        self._catalogueValue = future.result()

                loading.add_done_callback(loaded)
            transition = startTask(self._rebind(attachment is not None, BACKGROUND_CONTEXT))
            self._transitions.add(transition)

            def settled(future: asyncio.Future[None]) -> None:
                self._transitions.discard(future)
                if future.cancelled():
                    return
                error = future.exception()
                if self._attachmentRevision != revision or not sameAttachment(self._client.attachment, attachment):
                    return
                if error is None:
                    publishReplacement(
                        self._attachmentState,
                        {"status": "detached"} if attachment is None else {"status": "attached", "sessionId": attachment["sessionId"]},
                        BACKGROUND_CONTEXT,
                    )
                    return
                publishReplacement(
                    self._attachmentState,
                    {"status": "detached"} if attachment is None else {"status": "degraded", "sessionId": attachment["sessionId"]},
                    BACKGROUND_CONTEXT,
                )
                self._onError(toError(error))

            transition.add_done_callback(settled)

        self._removeAttachmentListener = client.onAttachmentChange(onAttachment)

    @property
    def acceptsUnavailableServices(self) -> bool:
        return self._client.attachment is None and self._catalogueValue is None

    async def catalogue(self, context: Context) -> list[Any]:
        target = self._client.attachment
        if target is None:
            return self._catalogueValue or []
        catalogue = await self._client.serviceCatalogue(target, context.abortSignal)
        self._catalogueValue = catalogue
        return catalogue  # type: ignore[no-any-return]

    def open(self, options: dict[str, Any]) -> RoutedServiceBinding:
        if self._disposed:
            raise RuntimeError("Session service source is disposed")
        holder: dict[str, RoutedServiceBinding] = {}

        async def onActivate(context: Context) -> None:
            attachment = self._client.attachment
            if attachment is None:
                await self._whenDetached(self._attachmentRevision, context)
            else:
                await self._whenAttached(attachment, self._attachmentRevision, context)

        binding = RoutedServiceBinding(
            {
                "services": options["services"],
                "transport": self._transport,
                "getBound": lambda: self._client.attachment is not None,
                "onActivate": onActivate,
                "assertAccess": options["assertAccess"],
                "onError": options["onError"],
                "remove": lambda: self._discard(holder["binding"]),
            }
        )
        holder["binding"] = binding
        self._bindings.append(binding)
        return binding

    def _discard(self, binding: RoutedServiceBinding) -> None:
        if binding in self._bindings:
            self._bindings.remove(binding)

    async def whenAttached(self, sessionId: str, context: Context) -> None:
        attachment = self._client.attachment
        if attachment is None or attachment["sessionId"] != sessionId:
            raise RuntimeError(f"Session {sessionId} is not the current attachment")
        await self._whenAttached(attachment, self._attachmentRevision, context)

    async def whenDetached(self, context: Context) -> None:
        if self._client.attachment is not None:
            raise RuntimeError("A Session is still attached")
        await self._whenDetached(self._attachmentRevision, context)

    async def dispose(self, context: Context) -> None:
        if self._disposed:
            return
        self._disposed = True
        self._removeAttachmentListener()
        await allSettled(list(self._transitions))
        bindings = list(self._bindings)
        self._bindings.clear()
        results = await allSettled([binding.dispose(context) for binding in bindings])
        throwFailures(results, "Failed to dispose Session service source")

    async def _rebind(self, bound: bool, context: Context) -> None:
        results = await allSettled([binding.updateBound(bound, context) for binding in list(self._bindings)])
        throwFailures(results, "Failed to rebind Session services")

    async def _whenAttached(self, attachment: dict[str, Any], revision: int, context: Context) -> None:
        try:
            await promiseAll([binding.serviceReady(context) for binding in list(self._bindings)])
        except BaseException:
            if self._attachmentRevision == revision and sameAttachment(self._client.attachment, attachment):
                publishReplacement(self._attachmentState, {"status": "degraded", "sessionId": attachment["sessionId"]}, context)
            raise
        if self._attachmentRevision != revision or not sameAttachment(self._client.attachment, attachment):
            raise RuntimeError(f"Session {attachment['sessionId']} was replaced while attaching")
        state = self._attachmentState.value
        if state["status"] != "attached" or state.get("sessionId") != attachment["sessionId"]:
            publishReplacement(self._attachmentState, {"status": "attached", "sessionId": attachment["sessionId"]}, context)

    async def _whenDetached(self, revision: int, context: Context) -> None:
        await promiseAll([binding.serviceReady(context) for binding in list(self._bindings)])
        if self._attachmentRevision != revision or self._client.attachment is not None:
            raise RuntimeError("The Session attachment changed while detaching")
        if self._attachmentState.value["status"] != "detached":
            publishReplacement(self._attachmentState, {"status": "detached"}, context)


def createServerServiceSource(client: Any, options: dict[str, Any] | None = None) -> ServerServiceSourceImpl:
    """Create the server-scoped remote service source for one presentation client."""
    return ServerServiceSourceImpl(client, options or {})


def createSessionServiceSource(client: Any, options: dict[str, Any] | None = None) -> SessionServiceSourceImpl:
    """Create the selected-Session remote service source for one presentation client."""
    return SessionServiceSourceImpl(client, options or {})


def publishReplacement(state: Any, value: Any, context: Context) -> None:
    state.replace(context, value)


def toServerConnectionState(client: Any, attempt: int, error: BaseException | None = None) -> dict[str, Any]:
    since = datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")
    if client.connectionState == "connecting":
        return {"status": "connecting", "attempt": attempt}
    if client.connectionState == "connected":
        return {"status": "connected", "since": since}
    return {
        "status": "disconnected",
        "since": since,
        "reason": errorMessage(error) if error is not None else "Client is disconnected",
        "retryAt": None,
    }


def sameAttachment(left: dict[str, Any] | None, right: dict[str, Any] | None) -> bool:
    if left is None or right is None:
        return left is right
    return (
        left["serverId"] == right["serverId"]
        and left["sessionId"] == right["sessionId"]
        and left["attachmentId"] == right["attachmentId"]
    )


def throwFailures(results: list[Any], message: str) -> None:
    errors = [toError(result.reason) for result in results if result.status == "rejected"]
    if len(errors) == 1:
        raise errors[0]
    if len(errors) > 1:
        raise ExceptionGroup(message, errors)
