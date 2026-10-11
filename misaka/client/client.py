"""The transport-neutral client of the experimental Pi service protocol.

Translated from pi's ``packages/client/src/client.ts``. ``request()`` answers an ``asyncio`` future after doing,
at once, what upstream's does before it returns its promise: registering the request and sending the frame.
``signal`` is an ``AbortSignal`` (``misaka.utils.js``).
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from typing import Any, Self

from misaka.chord.context import BACKGROUND_CONTEXT
from misaka.chord.services.state_codec import (
    ServiceStateDecoder,
    createServiceStateDecoder,
)
from misaka.chord.services.wire import (
    createServiceCatalogueCall,
    createServiceSubscribeCall,
    createServiceUnsubscribeCall,
    parseServiceCall,
    parseServiceCatalogue,
    parseWireServiceProviderUpdate,
    parseWireServiceSubscriptionSnapshot,
)
from misaka.chord.types import (
    RemoteServiceTransport,
    ServiceCall,
    ServiceCatalogueEntry,
    ServiceMode,
    ServiceProviderUpdate,
    ServiceSubscriptionSnapshot,
)
from misaka.protocol import (
    AttachmentEnvelope,
    ProtocolValidationError,
    ResponseEnvelope,
    RpcTarget,
    ServerHello,
    ServiceEventEnvelope,
    SessionTarget,
    encodeClientMessage,
    isServerId,
)
from misaka.utils.js import (
    DOMException,
    eager,
    liveKeys,
    rejectedPromise,
    resolvedPromise,
    startTask,
)

from .connection import Connection
from .errors import ClientDisposedError, DisconnectedError, ServerError, toError
from .types import (
    AttachmentChangeListener,
    ClientOptions,
    ConnectionState,
    ConnectionStateChange,
    ServiceResult,
    Unsubscribe,
)


class _PendingRequest:
    __slots__ = ("cleanup", "reject", "resolve")

    def __init__(
        self,
        resolve: Callable[[Any], None],
        reject: Callable[[Exception], None],
        cleanup: Callable[[], None],
    ) -> None:
        self.resolve = resolve
        self.reject = reject
        self.cleanup = cleanup


class _ActiveServiceListener:
    def __init__(
        self,
        target: RpcTarget,
        listener: Callable[[ServiceProviderUpdate], Awaitable[None] | None],
        decoder: ServiceStateDecoder,
    ) -> None:
        self.target = target
        self.listener = listener
        self.decoder = decoder
        self.queuedWireUpdates: list[Any] = []
        self.queued: list[ServiceProviderUpdate] = []
        self.deliveryTail: asyncio.Future[None] = resolvedPromise(None)
        self.hydrated = False
        self.ready = False


class _Subscription:
    def __init__(
        self,
        client: Client,
        subscriptionId: str,
        active: _ActiveServiceListener,
        target: RpcTarget,
        snapshot: ServiceSubscriptionSnapshot,
    ) -> None:
        self._client = client
        self._active = active
        self._disposed = False
        self.id = subscriptionId
        self.target = target
        self.snapshot = snapshot

    def start(self) -> None:
        """Begin ordered update delivery after the caller has installed the snapshot."""
        active = self._active
        if self._disposed or active.ready:
            return
        active.ready = True
        updates, active.queued = active.queued, []
        for update in updates:
            self._client._deliverServiceUpdate(active, update)

    @eager
    async def dispose(self) -> None:
        if self._disposed:
            return
        self._disposed = True
        client = self._client
        if client._serviceListeners.get(self.id) is self._active:
            del client._serviceListeners[self.id]
        try:
            if client.connected and client._targetIsCurrent(self.target):
                await client._request(self.target, createServiceUnsubscribeCall(self.id))
            # Shielded: a caller that stops waiting must not cancel the delivery chain every later update waits on.
            await asyncio.shield(self._active.deliveryTail)
        finally:
            self._active.queuedWireUpdates.clear()
            self._active.queued.clear()


class Client:
    def __init__(self, options: ClientOptions) -> None:
        if not isServerId(options["serverId"]):
            raise TypeError("serverId must be a canonical lowercase UUIDv4")
        self._options = options
        self._pendingRequests: dict[str, _PendingRequest] = {}
        # ``Set``s: a listener registered twice is held once.
        self._connectionStateListeners: dict[Callable[[ConnectionStateChange], None], None] = {}
        self._attachmentListeners: dict[AttachmentChangeListener, None] = {}
        self._serviceListeners: dict[str, _ActiveServiceListener] = {}
        self._requestSequence = 0
        self._serviceSubscriptionSequence = 0
        self._hello: ServerHello | None = None
        self._attachment: SessionTarget | None = None
        self._disposed = False
        self._disposePromise: asyncio.Future[None] | None = None
        self._connection = Connection(
            transportFactory=options["transportFactory"],
            serverId=options["serverId"],
            maxFrameLength=options.get("maxFrameLength"),
            onHandshake=self._onHandshake,
            onMessage=self._handleMessage,
            onStateChange=self._handleConnectionStateChange,
        )

    def _onHandshake(self, hello: ServerHello) -> None:
        self._hello = hello

    @property
    def disposed(self) -> bool:
        return self._disposed

    @property
    def connectionState(self) -> ConnectionState:
        return self._connection.state

    @property
    def connected(self) -> bool:
        return self._connection.state == "connected"

    @property
    def serverId(self) -> str:
        return self._options["serverId"]

    @property
    def hello(self) -> ServerHello | None:
        return self._hello

    @property
    def attachment(self) -> SessionTarget | None:
        return self._attachment

    def connect(self) -> asyncio.Future[ServerHello]:
        if self._disposed:
            return rejectedPromise(ClientDisposedError())
        self._hello = None
        return self._connection.connect()  # type: ignore[return-value]

    def reconnect(self) -> asyncio.Future[ServerHello]:
        return self.connect()

    def disconnect(self, reason: str = "Client disconnected") -> None:
        self._connection.disconnect(reason)

    def onConnectionStateChange(self, listener: Callable[[ConnectionStateChange], None]) -> Unsubscribe:
        self._assertNotDisposed()
        self._connectionStateListeners[listener] = None
        return lambda: self._connectionStateListeners.pop(listener, None)

    def onAttachmentChange(self, listener: AttachmentChangeListener) -> Unsubscribe:
        self._assertNotDisposed()
        self._attachmentListeners[listener] = None
        return lambda: self._attachmentListeners.pop(listener, None)

    def request(self, target: RpcTarget, call: ServiceCall, signal: Any = None) -> asyncio.Future[ServiceResult]:
        """Invoke one low-level protocol call against an explicit routed target."""
        return self._request(target, call, signal)

    @eager
    async def serviceCatalogue(self, target: RpcTarget, signal: Any = None) -> list[ServiceCatalogueEntry]:
        result = await self._request(target, createServiceCatalogueCall(), signal)
        try:
            return parseServiceCatalogue(result)
        except Exception as error:  # noqa: BLE001
            validationError = ProtocolValidationError(str(error) or "Invalid service catalogue")
            self._connection.fail(validationError)
            raise validationError from None

    @eager
    async def subscribeService(
        self,
        target: RpcTarget,
        serviceId: str,
        mode: ServiceMode,
        listener: Callable[[ServiceProviderUpdate], Awaitable[None] | None],
        signal: Any = None,
    ) -> _Subscription:
        self._serviceSubscriptionSequence += 1
        subscriptionId = f"service-{self._serviceSubscriptionSequence}"
        active = _ActiveServiceListener(target, listener, createServiceStateDecoder())
        self._serviceListeners[subscriptionId] = active

        def transform(result: Any) -> ServiceSubscriptionSnapshot:
            decoded = active.decoder.decodeSnapshot(parseWireServiceSubscriptionSnapshot(result))
            active.hydrated = True
            queuedWire, active.queuedWireUpdates = active.queuedWireUpdates, []
            for update in queuedWire:
                active.queued.append(active.decoder.decodeUpdate(parseWireServiceProviderUpdate(update)))
            return decoded

        try:
            snapshot = await self._request(
                target, createServiceSubscribeCall(subscriptionId, serviceId, mode), signal, transform
            )
        except BaseException:
            if self._serviceListeners.get(subscriptionId) is active:
                del self._serviceListeners[subscriptionId]
            raise
        if self._serviceListeners.get(subscriptionId) is not active:
            raise DisconnectedError()
        return _Subscription(self, subscriptionId, active, target, snapshot)

    def _request(
        self,
        target: RpcTarget,
        call: ServiceCall,
        signal: Any = None,
        transform: Callable[[Any], Any] | None = None,
    ) -> asyncio.Future[Any]:
        if self._disposed:
            return rejectedPromise(ClientDisposedError())
        if not self.connected:
            return rejectedPromise(DisconnectedError())
        if signal is not None and signal.aborted:
            return rejectedPromise(abortError(signal))
        self._requestSequence += 1
        id_ = f"request-{self._requestSequence}"
        promise: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        state = {"sent": False, "aborted": False}

        def resolve(value: Any) -> None:
            if not promise.done():
                promise.set_result(value)

        def reject(error: Exception) -> None:
            if not promise.done():
                promise.set_exception(error)
                # A request nobody awaits any more (aborted, disconnected) is not an unhandled error.
                promise.exception()

        def sendCancel() -> None:
            if not state["sent"] or not self.connected:
                return
            try:
                self._connection.send(
                    encodeClientMessage(
                        {"type": "cancel", "id": id_, "target": target},
                        {"maxFrameLength": self._connection.maxFrameLength},
                    )
                )
            except Exception as error:  # noqa: BLE001
                self._connection.fail(toError(error))

        def onAbort() -> None:
            if state["aborted"]:
                return
            state["aborted"] = True
            reject(abortError(signal))
            sendCancel()

        if signal is not None:
            signal.addEventListener("abort", onAbort, once=True)

        def onResult(result: Any) -> None:
            try:
                resolve(result if transform is None else transform(result))
            except Exception as error:  # noqa: BLE001
                validationError = ProtocolValidationError(str(error) or "Invalid service operation stream")
                self._connection.fail(validationError)
                reject(validationError)

        def cleanup() -> None:
            if signal is not None:
                signal.removeEventListener("abort", onAbort)

        self._pendingRequests[id_] = _PendingRequest(onResult, reject, cleanup)
        try:
            frame = encodeClientMessage(
                {"type": "request", "id": id_, "target": target, "call": parseServiceCall(call)},
                {"maxFrameLength": self._connection.maxFrameLength},
            )
        except Exception as error:  # noqa: BLE001
            pending = self._takePendingRequest(id_)
            if pending is not None:
                pending.reject(toError(error))
            return promise
        self._connection.send(frame)
        state["sent"] = True
        if state["aborted"]:
            sendCancel()
        return promise

    def _handleMessage(self, message: ResponseEnvelope | ServiceEventEnvelope | AttachmentEnvelope) -> None:
        if message["type"] == "attachment":
            attachment = message["attachment"]
            if attachment is not None and attachment["serverId"] != self._options["serverId"]:
                self._connection.fail(ProtocolValidationError("Attachment update belongs to another server"))
                return
            self._setAttachment(attachment)
            return
        if message["type"] == "service_update":
            active = self._serviceListeners.get(message["subscriptionId"])
            if active is None:
                return
            if not active.hydrated:
                active.queuedWireUpdates.append(message["update"])
                return
            try:
                update = active.decoder.decodeUpdate(parseWireServiceProviderUpdate(message["update"]))
            except Exception as error:  # noqa: BLE001
                self._connection.fail(ProtocolValidationError(str(error) or "Invalid service operation stream"))
                return
            if active.ready:
                self._deliverServiceUpdate(active, update)
            else:
                active.queued.append(update)
            return
        pending = self._takePendingRequest(message["id"])
        if pending is None:
            self._connection.fail(ProtocolValidationError("Response has no matching request"))
            return
        if not message["ok"]:
            pending.reject(ServerError(message["error"]))
            return
        pending.resolve(message.get("result"))

    def _handleConnectionStateChange(self, change: ConnectionStateChange) -> None:
        if change["state"] == "disconnected":
            self._hello = None
            self._setAttachment(None)
            self._rejectPendingRequests(change.get("error") or DisconnectedError())
            self._serviceListeners.clear()
        for listener in liveKeys(self._connectionStateListeners):
            try:
                listener(change)
            except Exception as error:  # noqa: BLE001
                self._reportListenerError(error)

    def _takePendingRequest(self, id: str) -> _PendingRequest | None:
        request = self._pendingRequests.pop(id, None)
        if request is not None:
            request.cleanup()
        return request

    def _rejectPendingRequests(self, error: Exception) -> None:
        requests = list(self._pendingRequests.values())
        self._pendingRequests.clear()
        for request in requests:
            request.cleanup()
            request.reject(error)

    def dispose(self) -> asyncio.Future[None]:
        if self._disposePromise is not None:
            return self._disposePromise
        self._disposed = True
        self._disposePromise = resolvedPromise(None)
        error = ClientDisposedError()
        self._rejectPendingRequests(error)
        self._connection.disconnect(error)
        self._hello = None
        self._setAttachment(None)
        self._connectionStateListeners.clear()
        self._attachmentListeners.clear()
        self._serviceListeners.clear()
        return self._disposePromise

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.dispose()

    def _setAttachment(self, attachment: SessionTarget | None) -> None:
        previous = self._attachment
        if _sameTarget(previous, attachment):
            return
        self._attachment = attachment
        for listener in liveKeys(self._attachmentListeners):
            try:
                listener(attachment)
            except Exception as error:  # noqa: BLE001
                self._reportListenerError(error)

    def _deliverServiceUpdate(self, active: _ActiveServiceListener, update: ServiceProviderUpdate) -> None:
        previous = active.deliveryTail

        async def deliver() -> None:
            # `.then(...)` runs the listener on a later turn, never inside the call that queued it.
            await asyncio.sleep(0)
            await previous
            try:
                result = active.listener(update)
                if inspect.isawaitable(result):
                    await result
            except Exception as error:  # noqa: BLE001
                self._reportListenerError(error)

        active.deliveryTail = startTask(deliver())

    def _targetIsCurrent(self, target: RpcTarget) -> bool:
        if "sessionId" not in target:
            return self._hello is not None and self._hello["serverId"] == target["serverId"]
        return _sameTarget(self._attachment, target)  # type: ignore[arg-type]

    def _assertNotDisposed(self) -> None:
        if self._disposed:
            raise ClientDisposedError()

    def _reportListenerError(self, error: BaseException) -> None:
        onListenerError = self._options.get("onListenerError")
        if onListenerError is None:
            return
        try:
            onListenerError(toError(error))
        except Exception:  # noqa: BLE001, S110 - diagnostics cannot affect protocol or transport state
            pass


@eager
async def connectClient(options: ClientOptions) -> Client:
    """``Client.connect(options)``: a connected client, or the error that stopped it."""
    client = Client(options)
    try:
        await client.connect()
        return client
    except BaseException:
        await client.dispose()
        raise


def _sameTarget(left: SessionTarget | None, right: SessionTarget | None) -> bool:
    if left is None or right is None:
        return left is right
    return (
        left["serverId"] == right["serverId"]
        and left["sessionId"] == right["sessionId"]
        and left["attachmentId"] == right["attachmentId"]
    )



def abortError(signal: Any) -> Exception:
    reason = getattr(signal, "reason", None)
    return reason if isinstance(reason, Exception) else DOMException("The operation was aborted", "AbortError")


class _ClientServiceTransport:
    def __init__(self, client: Client, getTarget: Callable[[], RpcTarget | None]) -> None:
        self._client = client
        self._getTarget = getTarget

    def _target(self) -> RpcTarget:
        resolved = self._getTarget()
        if resolved is None:
            raise RuntimeError("Remote service target is unavailable")
        return resolved

    @eager
    async def invoke(self, call: ServiceCall, context: Any) -> Any:
        return await self._client.request(self._target(), call, context.abortSignal)

    @eager
    async def subscribe(self, serviceId: str, mode: ServiceMode, listener: Any, context: Any) -> Any:
        subscription = await self._client.subscribeService(
            self._target(), serviceId, mode, lambda update: listener(update, BACKGROUND_CONTEXT), context.abortSignal
        )
        return _TransportSubscription(subscription)


class _TransportSubscription:
    def __init__(self, subscription: _Subscription) -> None:
        self._subscription = subscription
        self.snapshot = subscription.snapshot

    def activate(self) -> None:
        self._subscription.start()

    def close(self, context: Any = None) -> Awaitable[None]:
        # An async function runs at once, so the unsubscribe request is sent before this returns.
        return startTask(self._subscription.dispose())


def createClientServiceTransport(client: Client, getTarget: Callable[[], RpcTarget | None]) -> RemoteServiceTransport:
    """Adapts a lazily resolved routed client target to a Chord service transport."""
    return _ClientServiceTransport(client, getTarget)  # type: ignore[return-value]


__all__ = ["Client", "connectClient", "createClientServiceTransport"]
