"""Translated from pi's ``coding-agent/src/experimental/session-worker-manager.ts``.

``SessionWorkerManager`` is the session and process bookkeeping of one replaceable server process: it launches Session
workers, finds the ones a previous server generation left running, routes service operations to them through the
coordinator, and stops them.

MISAKA fork: the manager takes ``timers`` (``timers.py``) and ``kill`` (``kill(pid, signalName)``, ``os.kill`` by default) so
tests can control the clock and the signals, where pi's tests replace ``setTimeout`` and ``process.kill``; it starts
workers through ``misaka.experimental.process.spawnInternalProcess``, looked up when it launches, so a test can pick the
worker module.
"""

from __future__ import annotations

import asyncio
import base64
import inspect
import os
import signal
import uuid
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

from misaka.chord import (
    Context,
    createServiceUnsubscribeCall,
    decodeServiceControlCall,
    parseServiceProviderUpdate,
)
from misaka.chord.context import BACKGROUND_CONTEXT, TODO_CONTEXT
from misaka.experimental import process as processRuntime
from misaka.experimental import schema
from misaka.experimental.session_worker import (
    SESSION_WORKER_CONTROL_ADDRESS_ENV,
    SESSION_WORKER_CONTROL_TOKEN_ENV,
    SESSION_WORKER_PEER_ID_ENV,
    SESSION_WORKER_SESSION_KEY_ENV,
    SessionWorkerEventSchema,
)
from misaka.experimental.timers import DEFAULT_TIMERS
from misaka.server import ServerError
from misaka.utils.js import (
    UNDEFINED,
    DOMException,
    eager,
    jsonStringify,
    rejectedPromise,
    startTask,
)

WORKER_STARTUP_TIMEOUT_MS = 15_000
WORKER_SHUTDOWN_TIMEOUT_MS = 10_000
WORKER_DISCOVERY_TIMEOUT_MS = 5_000
WORKER_DEMAND_TIMEOUT_MS = 5_000


class SessionPluginSelectionConflictError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.name = "SessionPluginSelectionConflictError"


def _observe(future: asyncio.Future[Any]) -> None:
    if not future.cancelled():
        future.exception()


def _settle(future: asyncio.Future[Any], error: BaseException | None = None, value: Any = None) -> None:
    """Resolve or reject a promise stand-in once; a rejection nobody awaits leaves no "never retrieved" noise."""
    if future.done():
        return
    if error is None:
        future.set_result(value)
    else:
        future.set_exception(error)
        future.exception()


class WorkerRecord:
    def __init__(self, peerId: str, metadata: dict[str, Any], pid: int, token: str, pluginManifestPaths: tuple[str, ...]) -> None:
        self.peerId = peerId
        self.metadata = metadata
        self.pid = pid
        self.token = token
        self.pluginManifestPaths = pluginManifestPaths
        self.terminated: asyncio.Future[Exception | None] = asyncio.get_running_loop().create_future()
        self.attachmentIds: set[str] = set()
        self.expectedStop = False
        self.stopPromise: asyncio.Future[None] | None = None
        self.stopping = False

    def resolveTerminated(self, error: Exception | None) -> None:
        _settle(self.terminated, value=error)


class PendingDemand:
    def __init__(self, attachmentId: str, attached: bool, requestId: str, timer: Any, worker: WorkerRecord) -> None:
        self.attachmentId = attachmentId
        self.attached = attached
        self.requestId = requestId
        self.timer = timer
        self.worker = worker
        self.applied: asyncio.Future[None] = asyncio.get_running_loop().create_future()

    def resolve(self) -> None:
        _settle(self.applied)

    def reject(self, error: BaseException) -> None:
        _settle(self.applied, error)


class PendingWorkerOperation:
    def __init__(self, worker: WorkerRecord, scope: dict[str, str], cleanup: Callable[[], None]) -> None:
        self.worker = worker
        self.scope = scope
        self.cleanup = cleanup
        self.result: asyncio.Future[Any] = asyncio.get_running_loop().create_future()

    def resolve(self, result: Any) -> None:
        _settle(self.result, value=result)

    def reject(self, error: BaseException) -> None:
        _settle(self.result, error)


class WorkerServiceSubscription:
    def __init__(self, worker: WorkerRecord, scope: dict[str, str], listener: Callable[..., Any]) -> None:
        self.worker = worker
        self.scope = scope
        self.listener = listener
        self.deliveryTail: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        self.deliveryTail.set_result(None)


class PendingLaunch:
    def __init__(
        self, sessionKey: str, peerId: str, token: str, pluginManifestPaths: tuple[str, ...], child: Any, timer: Any
    ) -> None:
        self.sessionKey = sessionKey
        self.peerId = peerId
        self.token = token
        self.pluginManifestPaths = pluginManifestPaths
        self.child = child
        self.timer = timer
        self.promise: asyncio.Future[WorkerRecord] = asyncio.get_running_loop().create_future()

    def resolve(self, worker: WorkerRecord) -> None:
        _settle(self.promise, value=worker)

    def reject(self, error: BaseException) -> None:
        _settle(self.promise, error)


def _defaultKill(pid: int, signalName: str) -> None:
    os.kill(pid, getattr(signal, signalName))


class SessionWorkerManager:
    """Session and process bookkeeping owned by one replaceable server process. ``coordinator`` provides ``controlPath``,
    ``serverConnectionId``, ``wasReplaced``, ``onEvent``, ``send`` and ``broadcast`` (a ``CoordinatorConnection``);
    ``model`` is ``{"provider"?, "model"}`` or ``None``."""

    def __init__(
        self,
        coordinator: Any,
        sessionDir: str,
        model: dict[str, str] | None = None,
        onWorkerCountChanged: Callable[[int], None] | None = None,
        timers: Any = None,
        kill: Callable[[int, str], None] | None = None,
    ) -> None:
        self.workerPids: dict[str, int] = {}
        self._coordinator = coordinator
        self._sessionDir = sessionDir
        self._model = model
        self._timers = timers or DEFAULT_TIMERS
        self._kill = kill or _defaultKill
        self._workersBySession: dict[str, WorkerRecord] = {}
        self._workersByPeer: dict[str, WorkerRecord] = {}
        self._pending: dict[str, PendingLaunch] = {}
        self._pendingDemand: dict[str, PendingDemand] = {}
        self._pendingOperations: dict[str, PendingWorkerOperation] = {}
        self._serviceSubscriptions: dict[str, WorkerServiceSubscription] = {}
        self._onWorkerCountChanged = onWorkerCountChanged
        self._discoveryPeers: set[str] | None = None
        self._resolveDiscovery: Callable[[], None] | None = None
        self._detached = False
        self._shuttingDown = False
        self._removeListener = coordinator.onEvent(self._handleCoordinatorEvent)

    @property
    def trackedSessions(self) -> list[dict[str, Any]]:
        return [worker.metadata for worker in self._workersBySession.values()]

    def assertSessionPluginManifestPaths(self, metadata: dict[str, Any], manifestPaths: list[str] | tuple[str, ...]) -> None:
        existing = self._workersBySession.get(metadata["path"])
        if existing is not None and not sameStrings(existing.pluginManifestPaths, manifestPaths):
            raise SessionPluginSelectionConflictError(
                f"Session {metadata['id']} is active with a different plugin selection"
            )
        pending = self._pending.get(metadata["path"])
        if pending is not None and not sameStrings(pending.pluginManifestPaths, manifestPaths):
            raise SessionPluginSelectionConflictError(
                f"Session {metadata['id']} is starting with a different plugin selection"
            )

    @eager
    async def discover(self, peerIds: set[str]) -> None:
        if self._detached:
            return
        undiscovered = {
            peerId for peerId in peerIds if peerId not in self._workersByPeer and self._pendingPeer(peerId) is None
        }
        if len(undiscovered) == 0:
            return
        self._discoveryPeers = undiscovered
        discovered: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._resolveDiscovery = lambda: _settle(discovered)
        await self._coordinator.broadcast({"type": "discover_workers"})
        timer = self._timers.setTimeout(lambda: _settle(discovered), WORKER_DISCOVERY_TIMEOUT_MS)
        await discovered
        self._timers.clearTimeout(timer)
        self._discoveryPeers = None
        self._resolveDiscovery = None

    @eager
    async def openSession(
        self, metadata: dict[str, Any], context: Context, pluginManifestPaths: list[str] | tuple[str, ...]
    ) -> SimpleNamespace:
        if self._detached or self._shuttingDown:
            raise RuntimeError("Experimental server is shutting down")
        self.assertSessionPluginManifestPaths(metadata, pluginManifestPaths)
        existing = self._workersBySession.get(metadata["path"])
        if existing is not None:
            return self._routedHandle(existing)
        pending = self._pending.get(metadata["path"])
        if pending is not None:
            return self._routedHandle(await pending.promise)
        return self._routedHandle(await self._launch(metadata, context, pluginManifestPaths))

    @eager
    async def closeSession(self, metadata: dict[str, Any], context: Context) -> None:
        worker = self._workersBySession.get(metadata["path"])
        if worker is None:
            pending = self._pending.get(metadata["path"])
            worker = None if pending is None else await pending.promise
        if worker is not None:
            await self._stopWorker(worker, context)

    def _routedHandle(self, worker: WorkerRecord) -> SimpleNamespace:
        return SimpleNamespace(
            terminated=worker.terminated,
            attachClient=lambda context: self._attachClient(worker, context),
            close=lambda context: self._stopWorker(worker, context),
        )

    @eager
    async def _attachClient(self, worker: WorkerRecord, context: Context) -> SimpleNamespace:
        if self._detached or self._shuttingDown or worker.stopping:
            raise RuntimeError("Experimental Session worker is stopping")
        if self._workersByPeer.get(worker.peerId) is not worker:
            raise RuntimeError("Experimental Session worker is no longer available")
        attachmentId = str(uuid.uuid4())
        worker.attachmentIds.add(attachmentId)
        try:
            await self._applyDemand(worker, attachmentId, True, True, context)
        except BaseException:
            worker.attachmentIds.discard(attachmentId)
            raise
        scope = self._operationScope(worker, attachmentId)
        released = False

        def invokeService(call: dict[str, Any], publish: Callable[..., Any], serviceContext: Context) -> Any:
            return self._invokeService(worker, scope, call, publish, serviceContext)

        async def release(releaseContext: Context | None = None, *_: Any) -> None:
            nonlocal released
            if released:
                return
            released = True
            if self._detached or attachmentId not in worker.attachmentIds:
                return
            try:
                await self._applyDemand(worker, attachmentId, False, True, releaseContext)
            except Exception:
                if not self._detached and not self._coordinator.wasReplaced:
                    raise
            finally:
                worker.attachmentIds.discard(attachmentId)
                self._removeServiceSubscriptions(lambda entry: entry.worker is worker and sameScope(entry.scope, scope))

        return SimpleNamespace(invokeService=invokeService, release=release)

    @eager
    async def _invokeService(
        self,
        worker: WorkerRecord,
        scope: dict[str, str],
        call: dict[str, Any],
        publish: Callable[..., Any],
        context: Context,
    ) -> Any:
        control = decodeServiceControlCall(call)
        addedSubscriptionKey: str | None = None
        if control is not None and control["type"] == "subscribe":
            key = scopedServiceSubscriptionKey(scope, control["subscriptionId"])
            if key in self._serviceSubscriptions:
                raise RuntimeError("Service subscription ID is already active")
            subscriptionId = control["subscriptionId"]
            self._serviceSubscriptions[key] = WorkerServiceSubscription(
                worker, scope, lambda update, updateContext: publish(subscriptionId, update, updateContext)
            )
            addedSubscriptionKey = key
        try:
            response = await self._invoke(worker, scope, call, context)
        except BaseException:
            if addedSubscriptionKey is not None and control is not None and control["type"] == "subscribe":
                self._serviceSubscriptions.pop(addedSubscriptionKey, None)
                _ignoreFailure(
                    self._invoke(worker, scope, createServiceUnsubscribeCall(control["subscriptionId"]), BACKGROUND_CONTEXT)
                )
            raise
        if control is not None and control["type"] == "unsubscribe":
            self._serviceSubscriptions.pop(scopedServiceSubscriptionKey(scope, control["subscriptionId"]), None)
        return response

    async def _applyDemand(
        self, worker: WorkerRecord, attachmentId: str, attached: bool, compensateOnTimeout: bool, _context: Context | None
    ) -> None:
        if worker.stopping or self._workersByPeer.get(worker.peerId) is not worker:
            raise RuntimeError("Experimental Session worker is stopping")
        requestId = str(uuid.uuid4())

        def expire() -> None:
            if compensateOnTimeout:
                _ignoreFailure(self._reconcileDemandTimeout(requestId))
            else:
                self._rejectDemand(requestId, RuntimeError("Session worker demand update timed out"))

        timer = self._timers.setTimeout(expire, WORKER_DEMAND_TIMEOUT_MS)
        pending = PendingDemand(attachmentId, attached, requestId, timer, worker)
        self._pendingDemand[requestId] = pending
        try:
            await startTask(
                self._coordinator.send(
                    worker.peerId,
                    {
                        "type": "session_demand",
                        "serverConnectionId": self._coordinator.serverConnectionId,
                        "requestId": requestId,
                        "attachmentId": attachmentId,
                        "attached": attached,
                    },
                )
            )
        except Exception as error:  # noqa: BLE001 - the demand fails with it
            self._rejectDemand(requestId, error)
        await pending.applied

    def _invoke(self, worker: WorkerRecord, scope: dict[str, str], call: dict[str, Any], context: Context) -> asyncio.Future[Any]:
        """Worker operations have no wall-clock timeout: completion, disconnect, replacement, or shutdown settles them."""
        try:
            self._operationScope(worker, scope["attachmentId"])
        except Exception as error:  # noqa: BLE001 - returned as a rejection
            return rejectedPromise(error)
        abortSignal = context.abortSignal
        if abortSignal is not None and abortSignal.aborted:
            return rejectedPromise(abortError(abortSignal))
        requestId = str(uuid.uuid4())

        def onAbort() -> None:
            if abortSignal is None or requestId not in self._pendingOperations:
                return
            _ignoreFailure(
                self._coordinator.send(worker.peerId, {"type": "operation_cancel", "requestId": requestId, "scope": scope})
            )
            self._rejectOperation(requestId, abortError(abortSignal))

        if abortSignal is not None:
            abortSignal.addEventListener("abort", onAbort, True)

        def cleanup() -> None:
            if abortSignal is not None:
                abortSignal.removeEventListener("abort", onAbort)

        operation = PendingWorkerOperation(worker, scope, cleanup)
        self._pendingOperations[requestId] = operation
        sent = startTask(
            self._coordinator.send(worker.peerId, {"type": "operation", "requestId": requestId, "scope": scope, "call": call})
        )
        sent.add_done_callback(lambda future: self._operationSendFinished(requestId, future))
        return operation.result

    def _operationSendFinished(self, requestId: str, future: asyncio.Future[Any]) -> None:
        if future.cancelled():
            return
        error = future.exception()
        if error is not None:
            self._rejectOperation(requestId, error)

    def _operationScope(self, worker: WorkerRecord, attachmentId: str) -> dict[str, str]:
        if self._detached or self._shuttingDown or worker.stopping:
            raise RuntimeError("Experimental Session worker is stopping")
        if self._workersByPeer.get(worker.peerId) is not worker or attachmentId not in worker.attachmentIds:
            raise RuntimeError("Experimental Session worker has no active attachment")
        return {"serverConnectionId": self._coordinator.serverConnectionId, "attachmentId": attachmentId}

    def _stopWorker(self, worker: WorkerRecord, _context: Context | None) -> asyncio.Future[None]:
        if self._detached or self._workersByPeer.get(worker.peerId) is not worker:
            done: asyncio.Future[None] = asyncio.get_running_loop().create_future()
            done.set_result(None)
            return done
        if worker.stopPromise is None:
            worker.stopPromise = startTask(self._stopWorkerInternal(worker))
        return worker.stopPromise

    async def _stopWorkerInternal(self, worker: WorkerRecord) -> None:
        self._rejectWorkerOperations(worker, RuntimeError("Session worker is stopping"))
        worker.stopping = True
        worker.expectedStop = True
        _ignoreFailure(self._coordinator.send(worker.peerId, {"type": "shutdown"}))
        timedOut: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
        timer = self._timers.setTimeout(lambda: _settle(timedOut, value=True), WORKER_SHUTDOWN_TIMEOUT_MS)
        terminated: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
        worker.terminated.add_done_callback(lambda _future: _settle(terminated, value=False))
        try:
            await asyncio.wait({terminated, timedOut}, return_when=asyncio.FIRST_COMPLETED)
            if timedOut.done() and not terminated.done():
                try:
                    self._kill(worker.pid, "SIGKILL")
                except ProcessLookupError:
                    pass
                self._removeWorker(worker, None)
                await worker.terminated
        finally:
            self._timers.clearTimeout(timer)

    @eager
    async def shutdown(self) -> None:
        if self._detached or self._shuttingDown:
            return
        self._shuttingDown = True
        pendingWorkers = list(self._pending.values())
        for pending in pendingWorkers:
            _ignoreFailure(self._coordinator.send(pending.peerId, {"type": "shutdown"}))

        async def waitForExit(pending: PendingLaunch) -> None:
            if pending.child.exitCode is not None or pending.child.signalCode is not None:
                return
            exited: asyncio.Future[None] = asyncio.get_running_loop().create_future()
            pending.child.once("exit", lambda *_: _settle(exited))
            await exited

        pendingFinished = asyncio.ensure_future(asyncio.gather(*(waitForExit(pending) for pending in pendingWorkers)))
        pendingTimedOut: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
        timer = self._timers.setTimeout(lambda: _settle(pendingTimedOut, value=True), WORKER_SHUTDOWN_TIMEOUT_MS)

        async def stopPending() -> None:
            await asyncio.wait({pendingFinished, pendingTimedOut}, return_when=asyncio.FIRST_COMPLETED)
            if pendingTimedOut.done() and not pendingFinished.done():
                for pending in pendingWorkers:
                    pending.child.kill("SIGKILL")
                await pendingFinished
            self._timers.clearTimeout(timer)

        await asyncio.gather(
            stopPending(), *(self._stopWorker(worker, BACKGROUND_CONTEXT) for worker in list(self._workersBySession.values()))
        )
        self._detachState()

    def detach(self) -> None:
        """Forget workers without stopping them when this server is replaced."""
        if self._detached:
            return
        self._detached = True
        for pending in list(self._pending.values()):
            self._timers.clearTimeout(pending.timer)
            pending.reject(RuntimeError("Experimental server was replaced"))
        for requestId in list(self._pendingOperations):
            self._rejectOperation(requestId, RuntimeError("Experimental server was replaced during a worker operation"))
        self._detachState()

    def _launch(
        self, metadata: dict[str, Any], _context: Context, pluginManifestPaths: list[str] | tuple[str, ...]
    ) -> asyncio.Future[WorkerRecord]:
        sessionKey = metadata["path"]
        peerId = f"worker-{uuid.uuid4()}"
        token = str(uuid.uuid4())
        try:
            options: dict[str, Any] = {
                "sessionDir": self._sessionDir,
                "metadata": {
                    "id": metadata["id"],
                    "createdAt": metadata["createdAt"],
                    "cwd": metadata["cwd"],
                    "path": metadata["path"],
                },
                "pluginManifestPaths": list(pluginManifestPaths),
                **(self._model or {}),
            }
            child = processRuntime.spawnInternalProcess(
                "session-worker",
                [jsonStringify(options)],
                {
                    "env": {
                        SESSION_WORKER_CONTROL_ADDRESS_ENV: self._coordinator.controlPath,
                        SESSION_WORKER_CONTROL_TOKEN_ENV: token,
                        SESSION_WORKER_SESSION_KEY_ENV: base64.urlsafe_b64encode(sessionKey.encode("utf-8")).decode("ascii").rstrip("="),
                        SESSION_WORKER_PEER_ID_ENV: peerId,
                    }
                },
            )
        except Exception as error:  # noqa: BLE001 - returned as a rejection
            return rejectedPromise(error)
        timer = self._timers.setTimeout(
            lambda: self._failPending(sessionKey, RuntimeError("Session worker startup timed out")), WORKER_STARTUP_TIMEOUT_MS
        )
        pending = PendingLaunch(sessionKey, peerId, token, tuple(pluginManifestPaths), child, timer)
        self._pending[sessionKey] = pending
        self._notifyWorkerCountChanged()
        child.once("error", lambda error: self._failPending(sessionKey, error))
        child.once("exit", lambda code, signalName: self._childExited(pending, code, signalName))
        return pending.promise

    def _handleCoordinatorEvent(self, event: dict[str, Any]) -> None:
        if self._detached:
            return
        if event["type"] == "peer_disconnected":
            self._markDiscovered(event["peerId"])
            worker = self._workersByPeer.get(event["peerId"])
            if worker is not None:
                self._removeWorker(
                    worker,
                    None if worker.expectedStop else RuntimeError(f"Session worker {worker.metadata['id']} disconnected unexpectedly"),
                )
            pending = self._pendingPeer(event["peerId"])
            if pending is not None:
                self._failPending(pending.sessionKey, RuntimeError("Session worker disconnected during startup"))
            return
        if event["type"] != "message":
            return
        if not schema.Check(SessionWorkerEventSchema, event["payload"]):
            payload = event["payload"]
            if isinstance(payload, dict) and payload.get("type") == "operation_response":
                worker = self._workersByPeer.get(event["from"])
                if worker is not None:
                    self._rejectWorkerOperations(worker, RuntimeError("Session worker returned an invalid operation response"))
            return
        message = event["payload"]
        if message["type"] == "worker_failed":
            pending = self._pending.get(message["sessionKey"])
            if pending is not None and pending.peerId == event["from"] and pending.token == message["token"]:
                self._failPending(message["sessionKey"], RuntimeError(f"Session worker failed: {message['message']}"))
            return
        if message["type"] in ("demand_applied", "demand_rejected"):
            pendingDemand = self._pendingDemand.get(message["requestId"])
            if (
                pendingDemand is None
                or pendingDemand.worker.peerId != event["from"]
                or pendingDemand.worker.token != message["token"]
                or pendingDemand.worker.metadata["path"] != message["sessionKey"]
                or (
                    message["type"] == "demand_applied"
                    and (pendingDemand.attachmentId != message["attachmentId"] or pendingDemand.attached != message["attached"])
                )
            ):
                return
            del self._pendingDemand[message["requestId"]]
            self._timers.clearTimeout(pendingDemand.timer)
            if message["type"] == "demand_applied":
                pendingDemand.resolve()
            else:
                pendingDemand.reject(RuntimeError(f"Session worker rejected demand: {message['message']}"))
            return
        if message["type"] == "operation_response":
            self._handleOperationResponse(event["from"], message["token"], message["sessionKey"], message["response"])
            return
        if message["type"] == "service_update":
            self._handleServiceEvent(
                event["from"], message["token"], message["sessionKey"], message["scope"], message["subscriptionId"], message["update"]
            )
            return
        self._recordReadyWorker(event["from"], message)

    def _handleOperationResponse(self, peerId: str, token: str, sessionKey: str, response: dict[str, Any]) -> None:
        pending = self._pendingOperations.get(response["requestId"])
        if pending is None:
            return
        if (
            pending.worker.peerId != peerId
            or pending.worker.token != token
            or pending.worker.metadata["path"] != sessionKey
            or response["scope"]["serverConnectionId"] != pending.scope["serverConnectionId"]
            or response["scope"]["attachmentId"] != pending.scope["attachmentId"]
        ):
            self._rejectOperation(response["requestId"], RuntimeError("Session worker returned a mismatched operation response"))
            return
        del self._pendingOperations[response["requestId"]]
        pending.cleanup()
        if response["type"] == "operation_error":
            pending.reject(
                RuntimeError(f"Session worker operation failed: {response['message']}")
                if response.get("code") is None
                else ServerError(response["code"], response["message"])
            )
        else:
            pending.resolve(response.get("result", UNDEFINED))

    def _handleServiceEvent(
        self, peerId: str, token: str, sessionKey: str, scope: dict[str, str], subscriptionId: str, update: Any
    ) -> None:
        entry = self._serviceSubscriptions.get(scopedServiceSubscriptionKey(scope, subscriptionId))
        if (
            entry is None
            or entry.worker.peerId != peerId
            or entry.worker.token != token
            or entry.worker.metadata["path"] != sessionKey
            or not sameScope(entry.scope, scope)
        ):
            return
        try:
            parsed = parseServiceProviderUpdate(update)
        except Exception:  # noqa: BLE001 - an invalid update is dropped
            return
        previous = entry.deliveryTail

        async def deliver() -> None:
            try:
                await previous
                result = entry.listener(parsed, TODO_CONTEXT)
                if inspect.isawaitable(result):
                    await result
            except Exception:  # noqa: BLE001, S110 - `.catch(() => {})`: delivery continues
                pass

        entry.deliveryTail = asyncio.ensure_future(deliver())

    def _recordReadyWorker(self, peerId: str, message: dict[str, Any]) -> None:
        if (
            message["sessionKey"] != message["metadata"]["path"]
            or message["sessionId"] != message["metadata"]["id"]
            or not os.path.isabs(message["metadata"]["cwd"])
            or not os.path.isabs(message["metadata"]["path"])
        ):
            return
        self._markDiscovered(peerId)
        pending = self._pending.get(message["sessionKey"])
        if pending is not None and pending.peerId == peerId and not sameStrings(message["pluginManifestPaths"], pending.pluginManifestPaths):
            self._failPending(message["sessionKey"], RuntimeError("Session worker started with stale plugin packages"))
            return
        existing = self._workersBySession.get(message["sessionKey"])
        if existing is not None and existing.peerId != peerId:
            _ignoreFailure(self._coordinator.send(peerId, {"type": "shutdown"}))
            return
        if pending is not None and (
            pending.peerId != peerId or pending.token != message["token"] or pending.child.pid != message["pid"]
        ):
            return
        if existing is not None:
            return
        worker = WorkerRecord(peerId, message["metadata"], message["pid"], message["token"], tuple(message["pluginManifestPaths"]))
        self._workersBySession[message["sessionKey"]] = worker
        self._workersByPeer[peerId] = worker
        self.workerPids[message["sessionId"]] = message["pid"]
        if pending is not None:
            del self._pending[message["sessionKey"]]
            self._timers.clearTimeout(pending.timer)
            pending.resolve(worker)
        self._notifyWorkerCountChanged()

    def _childExited(self, pending: PendingLaunch, code: int | None, signalName: str | None) -> None:
        if self._pending.get(pending.sessionKey) is pending:
            self._failPending(
                pending.sessionKey,
                RuntimeError(f"Session worker exited before readiness ({signalName if signalName is not None else (code if code is not None else 'unknown')})"),
            )
            return
        worker = self._workersByPeer.get(pending.peerId)
        if worker is not None:
            self._removeWorker(
                worker,
                None
                if worker.expectedStop
                else RuntimeError(f"Session worker {worker.metadata['id']} exited unexpectedly ({signalName if signalName is not None else (code if code is not None else 'unknown')})"),
            )

    def _failPending(self, sessionKey: str, error: BaseException) -> None:
        pending = self._pending.get(sessionKey)
        if pending is None:
            return
        del self._pending[sessionKey]
        self._timers.clearTimeout(pending.timer)
        self._notifyWorkerCountChanged()
        if pending.child.exitCode is None and pending.child.signalCode is None:
            pending.child.kill("SIGKILL")
        pending.reject(error)

    def _rejectDemand(self, requestId: str, error: BaseException) -> None:
        pending = self._pendingDemand.get(requestId)
        if pending is None:
            return
        del self._pendingDemand[requestId]
        self._timers.clearTimeout(pending.timer)
        pending.reject(error)

    def _rejectOperation(self, requestId: str, error: BaseException) -> None:
        pending = self._pendingOperations.get(requestId)
        if pending is None:
            return
        del self._pendingOperations[requestId]
        pending.cleanup()
        pending.reject(error)

    def _rejectWorkerOperations(self, worker: WorkerRecord, error: BaseException) -> None:
        for requestId, pending in list(self._pendingOperations.items()):
            if pending.worker is worker:
                self._rejectOperation(requestId, error)

    @eager
    async def _reconcileDemandTimeout(self, requestId: str) -> None:
        pending = self._pendingDemand.get(requestId)
        if pending is None:
            return
        del self._pendingDemand[requestId]
        self._timers.clearTimeout(pending.timer)
        timeoutError = RuntimeError("Session worker demand update timed out")
        try:
            await self._applyDemand(pending.worker, pending.attachmentId, False, False, BACKGROUND_CONTEXT)
            if pending.attached:
                pending.reject(timeoutError)
            else:
                pending.resolve()
        except Exception as cleanupError:  # noqa: BLE001 - the worker is stopped
            try:
                await self._stopWorker(pending.worker, BACKGROUND_CONTEXT)
            except Exception as stopError:  # noqa: BLE001
                pending.reject(
                    ExceptionGroup(
                        "Session worker demand reconciliation and termination failed", [timeoutError, cleanupError, stopError]
                    )
                )
                return
            pending.reject(
                ExceptionGroup(
                    "Session worker demand reconciliation failed; worker was terminated", [timeoutError, cleanupError]
                )
            )

    def _removeWorker(self, worker: WorkerRecord, error: Exception | None) -> None:
        if self._workersByPeer.get(worker.peerId) is not worker:
            return
        self._rejectWorkerOperations(worker, RuntimeError("Session worker disconnected during an operation"))
        self._removeServiceSubscriptions(lambda entry: entry.worker is worker)
        for pending in list(self._pendingDemand.values()):
            if pending.worker is worker:
                self._rejectDemand(pending.requestId, RuntimeError("Session worker disconnected during demand update"))
        del self._workersByPeer[worker.peerId]
        self._workersBySession.pop(worker.metadata["path"], None)
        if self.workerPids.get(worker.metadata["id"]) == worker.pid:
            del self.workerPids[worker.metadata["id"]]
        worker.resolveTerminated(error)
        self._notifyWorkerCountChanged()

    def _removeServiceSubscriptions(self, matches: Callable[[WorkerServiceSubscription], bool]) -> None:
        for key, entry in list(self._serviceSubscriptions.items()):
            if matches(entry):
                del self._serviceSubscriptions[key]

    def _notifyWorkerCountChanged(self) -> None:
        if self._onWorkerCountChanged is not None:
            self._onWorkerCountChanged(len(self._workersBySession) + len(self._pending))

    def _pendingPeer(self, peerId: str) -> PendingLaunch | None:
        return next((pending for pending in self._pending.values() if pending.peerId == peerId), None)

    def _markDiscovered(self, peerId: str) -> None:
        if self._discoveryPeers is None or peerId not in self._discoveryPeers:
            return
        self._discoveryPeers.discard(peerId)
        if len(self._discoveryPeers) != 0:
            return
        if self._resolveDiscovery is not None:
            self._resolveDiscovery()

    def _detachState(self) -> None:
        self._removeListener()
        for requestId in list(self._pendingOperations):
            self._rejectOperation(requestId, RuntimeError("Experimental server detached during a worker operation"))
        for pending in list(self._pendingDemand.values()):
            del self._pendingDemand[pending.requestId]
            self._timers.clearTimeout(pending.timer)
            pending.resolve()
        self._pending.clear()
        self._serviceSubscriptions.clear()
        self._workersByPeer.clear()
        self._workersBySession.clear()
        self.workerPids.clear()
        self._discoveryPeers = None
        if self._resolveDiscovery is not None:
            self._resolveDiscovery()
        self._resolveDiscovery = None


def _ignoreFailure(awaitable: Any) -> asyncio.Future[Any]:
    """``void promise.catch(() => {})``: start ``awaitable`` and leave its failure unobserved."""
    future = startTask(awaitable)
    future.add_done_callback(_observe)
    return future


def sameStrings(left: list[str] | tuple[str, ...], right: list[str] | tuple[str, ...]) -> bool:
    return len(left) == len(right) and all(value == right[index] for index, value in enumerate(left))


def sameScope(left: dict[str, str], right: dict[str, str]) -> bool:
    return left["serverConnectionId"] == right["serverConnectionId"] and left["attachmentId"] == right["attachmentId"]


def scopedServiceSubscriptionKey(scope: dict[str, str], subscriptionId: str) -> str:
    return f"{scope['serverConnectionId']}\0{scope['attachmentId']}\0{subscriptionId}"


def abortError(abortSignal: Any) -> BaseException:
    reason = abortSignal.reason
    return reason if isinstance(reason, Exception) else DOMException("The operation was aborted", "AbortError")
