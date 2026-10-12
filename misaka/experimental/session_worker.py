"""Translated from pi's ``coding-agent/src/experimental/session-worker.ts``.

A Session worker is a process that owns one Session's directory: it locks it, opens the durable Harness over the Session's
storage, serves the Session services to the presentations the server routes to it, and retires when no client and no
live task needs it. It talks to its server through the coordinator over a control socket of JSON lines.

MISAKA fork: Node's ``net.Socket`` is ``NetSocket`` (``misaka.server.transports.unix.listener``); ``typebox`` schemas are
JSON Schemas checked with ``jsonschema`` (``schema.py``); the process stays alive until ``closeAndExit`` has closed its
resources, where upstream's event loop is kept alive by the control socket, so ``runSessionWorkerWithHarness`` resolves to the
process exit code; and ``WorkerLifecycle`` takes ``timers`` (``timers.py``) so tests can control the clock.
"""

from __future__ import annotations

import asyncio
import base64
import codecs
import os
import signal
import sys
import traceback
from collections.abc import AsyncIterable, AsyncIterator, Callable
from typing import Any

from misaka.chord import (
    REMOTE_SERVICE_ERROR_CODES,
    RemoteServiceError,
    isJsonValue,
    parseServiceProviderUpdate,
)
from misaka.chord.context import BACKGROUND_CONTEXT, TODO_CONTEXT, withCancel
from misaka.core.model_runtime import ModelRuntime
from misaka.core.settings_manager import SettingsManager
from misaka.durable import ROOT_CONVERSATION_ID, Harness
from misaka.durable.storage.sqlite.node import openNodeSqliteStorage
from misaka.experimental import schema
from misaka.experimental.coordinator import (
    COORDINATOR_PROTOCOL_VERSION,
    connectSocket,
    writeJsonLine,
)
from misaka.experimental.durable.harness_setup import (
    ExecutionEnvs,
    configureHarnessHttp,
    createCodingRegistry,
    createHarnessSettings,
    findInitialAgentModel,
)
from misaka.experimental.plugins.bundled import createSessionPluginFacetLoader
from misaka.experimental.process import (
    MAX_CONTROL_LINE_BYTES,
    consumeInternalProcessRole,
    onceSignal,
)
from misaka.experimental.services.worker import createSessionWorkerServices
from misaka.experimental.session_catalog import sessionStoragePath
from misaka.experimental.timers import DEFAULT_TIMERS
from misaka.utils import lockfile
from misaka.utils.js import (
    UNDEFINED,
    DOMException,
    errorMessage,
    jsIsSafeInteger,
    jsNumber,
    jsonParse,
    spawn,
)

ServiceCallSchema = schema.strictObject(
    {
        "serviceId": schema.string(1),
        "instance": schema.strictObject({"key": schema.string(1), "generation": schema.integer(1)}),
        "member": schema.string(1),
        "args": schema.array(schema.UNKNOWN),
    },
    optional=("instance",),
)
RemoteServiceErrorCodeSchema = {"type": "string", "pattern": f"^(?:{'|'.join(REMOTE_SERVICE_ERROR_CODES)})$"}

SESSION_WORKER_CONTROL_ADDRESS_ENV = "PI_SESSION_WORKER_CONTROL_ADDRESS"
SESSION_WORKER_CONTROL_TOKEN_ENV = "PI_SESSION_WORKER_CONTROL_TOKEN"
SESSION_WORKER_SESSION_KEY_ENV = "PI_SESSION_WORKER_SESSION_KEY_BASE64"
SESSION_WORKER_PEER_ID_ENV = "PI_SESSION_WORKER_PEER_ID"

SessionWorkerMetadataSchema = schema.strictObject(
    {"id": schema.string(1), "createdAt": schema.NUMBER, "cwd": schema.string(), "path": schema.string()}
)

SessionWorkerOptionsSchema = schema.strictObject(
    {
        "sessionDir": schema.string(1),
        "metadata": SessionWorkerMetadataSchema,
        "provider": schema.string(1),
        "model": schema.string(1),
        "pluginManifestPaths": schema.array(schema.string(1)),
    },
    optional=("provider", "model"),
)

WorkerOperationScopeSchema = schema.strictObject({"serverConnectionId": schema.string(), "attachmentId": schema.string()})

WorkerOperationRequestSchema = schema.strictObject(
    {
        "type": schema.literal("operation"),
        "requestId": schema.string(1),
        "scope": WorkerOperationScopeSchema,
        "call": ServiceCallSchema,
    }
)

WorkerOperationResponseSchema = schema.union(
    schema.strictObject(
        {
            "type": schema.literal("operation_result"),
            "requestId": schema.string(1),
            "scope": WorkerOperationScopeSchema,
            "result": schema.UNKNOWN,
        },
        optional=("result",),
    ),
    schema.strictObject(
        {
            "type": schema.literal("operation_error"),
            "requestId": schema.string(1),
            "scope": WorkerOperationScopeSchema,
            "code": RemoteServiceErrorCodeSchema,
            "message": schema.string(),
        },
        optional=("code",),
    ),
)

SessionWorkerCommandSchema = schema.union(
    schema.looseObject({"type": schema.literal("shutdown")}),
    schema.looseObject({"type": schema.literal("discover_workers")}),
    schema.looseObject(
        {
            "type": schema.literal("session_demand"),
            "serverConnectionId": schema.string(),
            "requestId": schema.string(),
            "attachmentId": schema.string(),
            "attached": schema.BOOLEAN,
        }
    ),
    WorkerOperationRequestSchema,
    schema.strictObject(
        {"type": schema.literal("operation_cancel"), "requestId": schema.string(1), "scope": WorkerOperationScopeSchema}
    ),
)

SessionWorkerEventSchema = schema.union(
    schema.looseObject(
        {
            "type": schema.literal("worker_ready"),
            "token": schema.string(),
            "sessionKey": schema.string(),
            "sessionId": schema.string(),
            "pid": schema.integer(1),
            "metadata": SessionWorkerMetadataSchema,
            "pluginManifestPaths": schema.array(schema.string(1)),
        }
    ),
    schema.looseObject(
        {
            "type": schema.literal("worker_failed"),
            "token": schema.string(),
            "sessionKey": schema.string(),
            "message": schema.string(),
        }
    ),
    schema.looseObject(
        {
            "type": schema.literal("demand_applied"),
            "token": schema.string(),
            "sessionKey": schema.string(),
            "requestId": schema.string(),
            "attachmentId": schema.string(),
            "attached": schema.BOOLEAN,
        }
    ),
    schema.looseObject(
        {
            "type": schema.literal("demand_rejected"),
            "token": schema.string(),
            "sessionKey": schema.string(),
            "requestId": schema.string(),
            "message": schema.string(),
        }
    ),
    schema.looseObject(
        {
            "type": schema.literal("operation_response"),
            "token": schema.string(),
            "sessionKey": schema.string(),
            "response": WorkerOperationResponseSchema,
        }
    ),
    schema.looseObject(
        {
            "type": schema.literal("service_update"),
            "token": schema.string(),
            "sessionKey": schema.string(),
            "scope": WorkerOperationScopeSchema,
            "subscriptionId": schema.string(1),
            "update": schema.UNKNOWN,
        }
    ),
)


class WorkerLifecycle:
    """Worker-local reconciliation of server-generation demand and Harness activity. ``options`` is ``{"initialServerConnectionId"?,
    "initialDemandGraceMs", "orphanDemandGraceMs", "onRetire", "timers"?}``."""

    def __init__(self, options: dict[str, Any]) -> None:
        self._timers = options.get("timers") or DEFAULT_TIMERS
        self._initialDemandGraceMs = options["initialDemandGraceMs"]
        self._orphanDemandGraceMs = options["orphanDemandGraceMs"]
        self._onRetire: Callable[[], None] = options["onRetire"]
        self._demands: dict[str, dict[str, Any]] = {}
        self._harnessActive = False
        self._currentServerConnectionId: str | None = options.get("initialServerConnectionId")
        self._demandInitialized = False
        self._retirementHolds = 0
        self._retiring = False
        self._initialTimer: Any = self._timers.setTimeout(self._initialExpired, self._initialDemandGraceMs)

    def _initialExpired(self) -> None:
        self._initialTimer = None
        self._demandInitialized = True
        self._reconcile()

    def serverConnected(self, serverConnectionId: str) -> None:
        self._currentServerConnectionId = serverConnectionId
        for demand in self._demands.values():
            if demand["serverConnectionId"] != serverConnectionId or demand.get("timer") is None:
                continue
            self._timers.clearTimeout(demand["timer"])
            del demand["timer"]

    def serverDisconnected(self, serverConnectionId: str) -> None:
        if self._currentServerConnectionId == serverConnectionId:
            self._currentServerConnectionId = None
        for key, demand in list(self._demands.items()):
            if demand["serverConnectionId"] != serverConnectionId or demand.get("timer") is not None:
                continue

            def orphan(key: str = key, demand: dict[str, Any] = demand) -> None:
                if self._demands.get(key) is not demand:
                    return
                del self._demands[key]
                self._reconcile()

            demand["timer"] = self._timers.setTimeout(orphan, self._orphanDemandGraceMs)

    def beginRequest(self, serverConnectionId: str, attachmentId: str) -> Callable[[], None]:
        if self._retiring:
            raise RuntimeError("Session worker is retiring")
        if serverConnectionId != self._currentServerConnectionId:
            raise RuntimeError("Session worker received a request from a stale server generation")
        demand = self._demands.get(demandKey(serverConnectionId, attachmentId))
        if demand is None or demand.get("timer") is not None:
            raise RuntimeError("Session worker request does not match the active attachment")
        return self.holdRetirement()

    def holdRetirement(self) -> Callable[[], None]:
        self._retirementHolds += 1
        released = False

        def release() -> None:
            nonlocal released
            if released:
                return
            released = True
            self._retirementHolds -= 1
            self._reconcile()

        return release

    def setDemand(self, serverConnectionId: str, attachmentId: str, attached: bool) -> None:
        if self._retiring:
            raise RuntimeError("Session worker is retiring")
        if serverConnectionId != self._currentServerConnectionId:
            raise RuntimeError("Session worker received demand from a stale server generation")
        self._demandInitialized = True
        if self._initialTimer is not None:
            self._timers.clearTimeout(self._initialTimer)
            self._initialTimer = None
        key = demandKey(serverConnectionId, attachmentId)
        previous = self._demands.get(key)
        if previous is not None and previous.get("timer") is not None:
            self._timers.clearTimeout(previous["timer"])
        if attached:
            self._demands[key] = {"serverConnectionId": serverConnectionId, "attachmentId": attachmentId}
        else:
            self._demands.pop(key, None)
        self._reconcile()

    def setHarnessActive(self, active: bool) -> None:
        """Live Harness tasks, such as runs and compactions, hold the worker until they end."""
        self._harnessActive = active
        if not active:
            self._reconcile()

    def close(self) -> None:
        if self._initialTimer is not None:
            self._timers.clearTimeout(self._initialTimer)
        for demand in self._demands.values():
            if demand.get("timer") is not None:
                self._timers.clearTimeout(demand["timer"])
        self._demands.clear()

    def _reconcile(self) -> None:
        if (
            self._retiring
            or not self._demandInitialized
            or self._retirementHolds != 0
            or self._harnessActive
            or len(self._demands) != 0
        ):
            return
        self._retiring = True
        self._onRetire()


DEFAULT_INITIAL_DEMAND_GRACE_MS = 10_000
DEFAULT_ORPHAN_DEMAND_GRACE_MS = 30_000
SESSION_WORKER_INITIAL_DEMAND_GRACE_ENV = "__PI_SESSION_WORKER_INITIAL_DEMAND_GRACE_MS"
SESSION_WORKER_ORPHAN_DEMAND_GRACE_ENV = "__PI_SESSION_WORKER_ORPHAN_DEMAND_GRACE_MS"

CoordinatorInputSchema = schema.union(
    schema.looseObject(
        {"type": schema.literal("peer_registered"), "peerId": schema.string(), "serverConnectionId": schema.string()},
        optional=("serverConnectionId",),
    ),
    schema.looseObject({"type": schema.literal("server_connected"), "serverConnectionId": schema.string()}),
    schema.looseObject({"type": schema.literal("server_disconnected"), "serverConnectionId": schema.string()}),
    schema.looseObject({"type": schema.literal("message"), "from": schema.literal("server"), "payload": schema.UNKNOWN}),
)


class WorkerControl:
    def __init__(self, initialServerConnectionId: str | None, messages: AsyncIterable[Any], socket: Any) -> None:
        self.initialServerConnectionId = initialServerConnectionId
        self.messages = messages
        self.socket = socket

    def send(self, event: dict[str, Any]) -> Any:
        return writeJsonLine(self.socket, {"type": "send", "to": "server", "payload": event})


_failureControl: dict[str, WorkerControl | None] = {"control": None}


async def connectControl() -> WorkerControl:
    address = os.environ.get(SESSION_WORKER_CONTROL_ADDRESS_ENV)
    token = os.environ.get(SESSION_WORKER_CONTROL_TOKEN_ENV)
    encodedSessionKey = os.environ.get(SESSION_WORKER_SESSION_KEY_ENV)
    if not address or not token or not encodedSessionKey:
        raise RuntimeError("Session worker requires a control address")
    peerId = os.environ.get(SESSION_WORKER_PEER_ID_ENV)
    if not peerId:
        raise RuntimeError("Session worker requires a peer ID")
    socket = await connectSocket(address)
    messages = createJsonLineMessages(socket)
    # MISAKA fork: until the worker is registered nothing but this socket keeps Node's event loop alive, so a coordinator
    # that drops the connection ends the process by itself; asyncio.run would wait on the lost reply forever.
    closed: asyncio.Future[None] = asyncio.get_running_loop().create_future()
    socket.once("close", lambda: closed.done() or closed.set_result(None))
    await writeJsonLine(socket, {"type": "register_peer", "protocol": COORDINATOR_PROTOCOL_VERSION, "peerId": peerId})
    reply = asyncio.ensure_future(anext(aiter(messages)))
    await asyncio.wait({reply, closed}, return_when=asyncio.FIRST_COMPLETED)
    if not reply.done():
        reply.cancel()
        raise SystemExit(0)
    registered = reply.result()
    if not schema.Check(CoordinatorInputSchema, registered) or registered["type"] != "peer_registered":
        raise RuntimeError("Coordinator rejected the session worker registration")
    return WorkerControl(registered.get("serverConnectionId"), messages, socket)


async def readCommands(control: WorkerControl, handlers: dict[str, Any]) -> None:
    async for value in control.messages:
        if not schema.Check(CoordinatorInputSchema, value):
            control.socket.destroy(RuntimeError("Coordinator sent an invalid worker message"))
            return
        message = value
        if message["type"] == "server_connected":
            handlers["onServerConnected"](message["serverConnectionId"])
            continue
        if message["type"] == "server_disconnected":
            handlers["onServerDisconnected"](message["serverConnectionId"])
            continue
        if message["type"] != "message" or not schema.Check(SessionWorkerCommandSchema, message["payload"]):
            continue
        command = message["payload"]
        if command["type"] == "shutdown":
            handlers["onShutdown"]()
        elif command["type"] == "discover_workers":
            handlers["onDiscovery"]()
        elif command["type"] == "session_demand":
            await handlers["onDemand"](command)
        elif command["type"] == "operation_cancel":
            handlers["onOperationCancel"](command)
        else:
            handlers["onOperation"](command)


def createJsonLineMessages(socket: Any) -> AsyncIterable[Any]:
    queued: list[Any] = []
    waiters: list[Callable[[Any], None]] = []
    buffered = ""
    # ``socket.setEncoding("utf8")``: a character split across chunks waits for the rest of its bytes.
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

    def onData(chunk: bytes) -> None:
        nonlocal buffered
        buffered += decoder.decode(chunk)
        if len(buffered.encode("utf-8", "surrogatepass")) > MAX_CONTROL_LINE_BYTES:
            socket.destroy(RuntimeError("Session worker control message is too large"))
            return
        while True:
            newline = buffered.find("\n")
            if newline == -1:
                return
            line = buffered[:newline]
            buffered = buffered[newline + 1 :]
            try:
                value = jsonParse(line)
                waiter = waiters.pop(0) if waiters else None
                if waiter:
                    waiter(value)
                else:
                    queued.append(value)
            except Exception:  # noqa: BLE001 - upstream's bare catch
                socket.destroy(RuntimeError("Session worker received invalid control JSON"))
                return

    socket.on("data", onData)

    async def iterate() -> AsyncIterator[Any]:
        while True:
            # ``queued.shift() ?? await ...``: a queued ``null`` is dropped too.
            value = queued.pop(0) if queued else None
            if value is not None:
                yield value
                continue
            future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
            waiters.append(future.set_result)
            try:
                value = await future
            except asyncio.CancelledError:
                # MISAKA fork: a cancelled `next()` withdraws its waiter; a promise cannot be cancelled.
                if future.set_result in waiters:
                    waiters.remove(future.set_result)
                raise
            yield value

    class Messages:
        def __aiter__(self) -> AsyncIterator[Any]:
            return iterate()

    return Messages()


def toWorkerServiceUpdate(update: dict[str, Any]) -> dict[str, Any]:
    if not isJsonValue(update):
        raise RuntimeError("Service produced a non-JSON update")
    return parseServiceProviderUpdate(update)  # type: ignore[no-any-return]


def demandKey(serverConnectionId: str, attachmentId: str) -> str:
    return f"{serverConnectionId}\0{attachmentId}"


def sameScope(left: dict[str, str], right: dict[str, str]) -> bool:
    return left["serverConnectionId"] == right["serverConnectionId"] and left["attachmentId"] == right["attachmentId"]


def lifecycleDelay(name: str, fallback: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return fallback
    parsed = jsNumber(value)
    if not jsIsSafeInteger(parsed) or parsed < 0:
        raise RuntimeError(f"{name} must be a non-negative safe integer")
    return int(parsed)


async def closeResources(resources: dict[str, Any]) -> None:
    errors: list[Exception] = []
    services, runtime = resources.get("services"), resources.get("runtime")
    try:
        if services is not None:
            await services.dispose()
    except Exception as error:  # noqa: BLE001 - collected
        errors.append(error)
    try:
        if runtime is not None:
            await runtime.harness.close(TODO_CONTEXT)
    except Exception as error:  # noqa: BLE001 - collected
        errors.append(error)
    try:
        cleanup = None if runtime is None else getattr(runtime, "cleanup", None)
        if cleanup is not None:
            await cleanup(TODO_CONTEXT)
    except Exception as error:  # noqa: BLE001 - collected
        errors.append(error)
    try:
        await resources["releaseOwnership"]()
    except Exception as error:  # noqa: BLE001 - collected
        errors.append(error)
    if len(errors) == 1:
        raise errors[0]
    if len(errors) > 1:
        raise ExceptionGroup("Session worker cleanup failed", errors)


def decodeSessionKey(encoded: str) -> str:
    """``Buffer.from(encoded, "base64url").toString()``."""
    return base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4)).decode("utf-8")


async def run(options: dict[str, Any], createHarness: Callable[[str, dict[str, Any]], Any]) -> asyncio.Future[int]:
    """Open the Session and serve it. Resolves, once the worker announced itself, to a future of the process exit code."""
    metadata = options["metadata"]
    sessionId = metadata["id"]
    control = await connectControl()
    token = os.environ[SESSION_WORKER_CONTROL_TOKEN_ENV]
    sessionKey = decodeSessionKey(os.environ[SESSION_WORKER_SESSION_KEY_ENV])
    _failureControl["control"] = control
    pluginManifestPaths = options["pluginManifestPaths"]
    releaseOwnership = await lockfile.lock(
        metadata["path"],
        {
            "realpath": True,
            "stale": 2_000,
            "update": 1_000,
            "retries": {"retries": 320, "factor": 1, "minTimeout": 25, "maxTimeout": 25, "maxRetryTime": 8_000},
        },
    )
    runtime: Any = None
    services: Any = None
    taskGraph: Any = None
    try:
        runtime = await createHarness(sessionStoragePath(metadata), options)
        taskGraph = await runtime.harness.taskGraph(TODO_CONTEXT)
        services = await createSessionWorkerServices(
            {
                "harness": runtime.harness,
                "conversation": runtime.conversation,
                "modelRuntime": getattr(runtime, "modelRuntime", None),
                "settingsManager": getattr(runtime, "settingsManager", None),
                "facetLoader": getattr(runtime, "facetLoader", None),
                "publish": lambda scope, subscriptionId, update: control.send(
                    {
                        "type": "service_update",
                        "token": token,
                        "sessionKey": sessionKey,
                        "scope": scope,
                        "subscriptionId": subscriptionId,
                        "update": toWorkerServiceUpdate(update),
                    }
                ),
            }
        )
    except Exception as error:
        if taskGraph is not None:
            taskGraph.dispose()
        try:
            await closeResources({"runtime": runtime, "services": services, "releaseOwnership": releaseOwnership})
        except Exception as cleanupError:  # noqa: BLE001
            raise ExceptionGroup("Session worker startup and cleanup failed", [error, cleanupError]) from None
        raise

    loop = asyncio.get_running_loop()
    exited: asyncio.Future[int] = loop.create_future()
    activeRequests: dict[str, dict[str, Any]] = {}
    harness = runtime.harness
    activity = taskGraph
    state: dict[str, Any] = {"lifecycle": None, "stopActivity": lambda: None, "closing": None}
    tasks: set[asyncio.Task[Any]] = set()

    def spawnTask(coroutine: Any) -> asyncio.Task[Any]:
        # ``void asyncFunction()``: the call runs up to its first suspension before the next command is read.
        return spawn(coroutine, tasks)

    def close() -> asyncio.Future[None]:
        if state["closing"] is not None:
            return state["closing"]  # type: ignore[no-any-return]
        if state["lifecycle"] is not None:
            state["lifecycle"].close()
        state["stopActivity"]()
        activity.dispose()
        services.removeSubscriptions(lambda _scope: True)
        for request in activeRequests.values():
            request["cancel"](RuntimeError("Session worker is closing"))
        activeRequests.clear()
        state["closing"] = asyncio.ensure_future(
            closeResources({"runtime": runtime, "services": services, "releaseOwnership": releaseOwnership})
        )
        return state["closing"]  # type: ignore[no-any-return]

    def closeAndExit() -> None:
        async def finish() -> None:
            try:
                await close()
            except Exception:  # noqa: BLE001 - the process reports its failure and exits
                traceback.print_exc()
                if not exited.done():
                    exited.set_result(1)
                return
            if not exited.done():
                exited.set_result(0)

        spawnTask(finish())

    lifecycle = WorkerLifecycle(
        {
            "initialServerConnectionId": control.initialServerConnectionId,
            "initialDemandGraceMs": lifecycleDelay(SESSION_WORKER_INITIAL_DEMAND_GRACE_ENV, DEFAULT_INITIAL_DEMAND_GRACE_MS),
            "orphanDemandGraceMs": lifecycleDelay(SESSION_WORKER_ORPHAN_DEMAND_GRACE_ENV, DEFAULT_ORPHAN_DEMAND_GRACE_MS),
            "onRetire": closeAndExit,
        }
    )
    state["lifecycle"] = lifecycle
    # Every live task, background work included, keeps the worker and its Session open.
    state["stopActivity"] = activity.subscribe(
        lambda graph, *_: lifecycle.setHarnessActive(len(graph["tasks"]) > 0)
    )
    # Recovered work from an interrupted turn continues now.
    harness.resume()

    async def handleOperation(request: dict[str, Any]) -> None:
        releaseRequest: Callable[[], None] = lambda: None
        cancellable = withCancel(BACKGROUND_CONTEXT)
        try:
            releaseRequest = lifecycle.beginRequest(request["scope"]["serverConnectionId"], request["scope"]["attachmentId"])
            activeRequests[request["requestId"]] = {"scope": request["scope"], "cancel": cancellable.cancel}
            result = await services.invoke(request["call"], request["scope"], cancellable.context)
            if result is not UNDEFINED and not isJsonValue(result):
                raise RuntimeError("Service produced a non-JSON result")
            response: dict[str, Any] = {
                "type": "operation_result",
                "requestId": request["requestId"],
                "scope": request["scope"],
            }
            if result is not UNDEFINED:
                response["result"] = result
            await control.send({"type": "operation_response", "token": token, "sessionKey": sessionKey, "response": response})
        except Exception as error:  # noqa: BLE001 - the failure is the response
            code = None
            if isinstance(error, RemoteServiceError):
                code = error.code
            else:
                candidate = getattr(error, "code", None)
                if isinstance(candidate, str) and schema.Check(RemoteServiceErrorCodeSchema, candidate):
                    code = candidate
            response = {
                "type": "operation_error",
                "requestId": request["requestId"],
                "scope": request["scope"],
                **({} if code is None else {"code": code}),
                "message": errorMessage(error),
            }
            await control.send({"type": "operation_response", "token": token, "sessionKey": sessionKey, "response": response})
        finally:
            active = activeRequests.get(request["requestId"])
            if active is not None and active["cancel"] is cancellable.cancel:
                del activeRequests[request["requestId"]]
            releaseRequest()

    ready = False

    def announce() -> None:
        if not ready:
            return

        async def send() -> None:
            try:
                await control.send(readyEvent())
            except Exception:  # noqa: BLE001 - a worker that cannot announce itself closes
                closeAndExit()

        spawnTask(send())

    def readyEvent() -> dict[str, Any]:
        return {
            "type": "worker_ready",
            "token": token,
            "sessionKey": sessionKey,
            "sessionId": sessionId,
            "pid": os.getpid(),
            "metadata": metadata,
            "pluginManifestPaths": list(pluginManifestPaths),
        }

    async def onDemand(command: dict[str, Any]) -> None:
        releaseRetirement = lifecycle.holdRetirement()
        try:
            try:
                if not command["attached"]:

                    def matches(scope: dict[str, str]) -> bool:
                        return (
                            scope["serverConnectionId"] == command["serverConnectionId"]
                            and scope["attachmentId"] == command["attachmentId"]
                        )

                    services.removeSubscriptions(matches)
                lifecycle.setDemand(command["serverConnectionId"], command["attachmentId"], command["attached"])
            except Exception as error:  # noqa: BLE001 - the rejection is the response
                await control.send(
                    {
                        "type": "demand_rejected",
                        "token": token,
                        "sessionKey": sessionKey,
                        "requestId": command["requestId"],
                        "message": errorMessage(error),
                    }
                )
                return
            await control.send(
                {
                    "type": "demand_applied",
                    "token": token,
                    "sessionKey": sessionKey,
                    "requestId": command["requestId"],
                    "attachmentId": command["attachmentId"],
                    "attached": command["attached"],
                }
            )
        finally:
            releaseRetirement()

    def onOperation(request: dict[str, Any]) -> None:
        async def operate() -> None:
            try:
                await handleOperation(request)
            except Exception:  # noqa: BLE001 - a worker that cannot answer closes
                closeAndExit()

        spawnTask(operate())

    def onOperationCancel(command: dict[str, Any]) -> None:
        active = activeRequests.get(command["requestId"])
        if active is not None and sameScope(active["scope"], command["scope"]):
            active["cancel"](DOMException("Service operation cancelled", "AbortError"))

    def onServerDisconnected(serverConnectionId: str) -> None:
        def matches(scope: dict[str, str]) -> bool:
            return scope["serverConnectionId"] == serverConnectionId

        services.removeSubscriptions(matches)
        for request in list(activeRequests.values()):
            if matches(request["scope"]):
                request["cancel"](RuntimeError("Server disconnected"))
        lifecycle.serverDisconnected(serverConnectionId)

    async def commands() -> None:
        try:
            await readCommands(
                control,
                {
                    "onShutdown": closeAndExit,
                    "onDiscovery": announce,
                    "onDemand": onDemand,
                    "onOperation": onOperation,
                    "onOperationCancel": onOperationCancel,
                    "onServerConnected": lifecycle.serverConnected,
                    "onServerDisconnected": onServerDisconnected,
                },
            )
        except Exception:  # noqa: BLE001 - a worker that cannot read its commands closes
            closeAndExit()

    spawnTask(commands())
    control.socket.once("close", closeAndExit)
    control.socket.once("error", lambda _error: closeAndExit())
    if sys.platform != "win32":
        onceSignal(signal.SIGTERM, closeAndExit)
        onceSignal(signal.SIGINT, closeAndExit)

    try:
        ready = True
        await control.send(readyEvent())
    except Exception as error:
        try:
            await close()
        except Exception as cleanupError:  # noqa: BLE001
            raise ExceptionGroup("Session worker readiness and cleanup failed", [error, cleanupError]) from None
        raise
    return exited


async def runSessionWorkerWithHarness(args: list[str], createHarness: Callable[[str, dict[str, Any]], Any]) -> int:
    """Run a Session worker over the Harness ``createHarness(databasePath, options)`` opens (a ``SimpleNamespace`` of
    ``harness``, ``conversation`` and optionally ``modelRuntime``, ``settingsManager``, ``facetLoader`` and ``cleanup``).
    Resolves to the process exit code once the worker retired."""
    try:
        if len(args) != 1:
            raise RuntimeError("Session worker requires one options argument")
        try:
            options = jsonParse(args[0])
        except ValueError as error:
            raise RuntimeError("Session worker received invalid options") from error
        if (
            not schema.Check(SessionWorkerOptionsSchema, options)
            or not os.path.isabs(options["sessionDir"])
            or not os.path.isabs(options["metadata"]["cwd"])
            or not os.path.isabs(options["metadata"]["path"])
            or (options.get("provider") is not None and options.get("model") is None)
        ):
            raise RuntimeError("Session worker received invalid options")
        exited = await run(options, createHarness)
        return await exited
    except Exception as error:
        token = os.environ.get(SESSION_WORKER_CONTROL_TOKEN_ENV)
        encodedSessionKey = os.environ.get(SESSION_WORKER_SESSION_KEY_ENV)
        control = _failureControl["control"]
        if token and encodedSessionKey and control is not None:
            try:
                await control.send(
                    {"type": "worker_failed", "token": token, "sessionKey": decodeSessionKey(encodedSessionKey), "message": errorMessage(error)}
                )
            except Exception:  # noqa: BLE001, S110 - the worker is failing already
                pass
        raise


async def createCodingAgentHarness(databasePath: str, options: dict[str, Any]) -> Any:
    from types import SimpleNamespace

    cwd = options["metadata"]["cwd"]
    modelRuntime = await ModelRuntime.create()
    settingsManager = SettingsManager.create(cwd)
    configureHarnessHttp(settingsManager)
    envs = ExecutionEnvs(cwd)
    harness = None
    try:
        harness = await Harness.open(
            await openNodeSqliteStorage(databasePath),
            {
                "models": modelRuntime,
                "registry": createCodingRegistry(settingsManager, cwd),
                "settings": createHarnessSettings(settingsManager),
                "env": envs.env,
                "onReport": lambda error: traceback.print_exception(error),
            },
            TODO_CONTEXT,
        )
        # The initial model applies only when the root conversation is created; later starts keep its durable model.
        created = (await harness.conversation(ROOT_CONVERSATION_ID, TODO_CONTEXT)) is None
        initial = (
            await findInitialAgentModel(
                settingsManager,
                modelRuntime,
                None
                if options.get("model") is None
                else {**({} if options.get("provider") is None else {"provider": options["provider"]}), "model": options["model"]},
            )
            if created
            else None
        )
        agent: dict[str, Any] = {"cwd": cwd}
        if initial is not None and initial.get("model") is not None:
            agent["model"] = initial["model"]
        if initial is not None and initial.get("thinkingLevel") is not None:
            agent["thinkingLevel"] = initial["thinkingLevel"]
        conversation = await harness.root(TODO_CONTEXT, {"agent": agent})
        return SimpleNamespace(
            harness=harness,
            conversation=conversation,
            modelRuntime=modelRuntime,
            settingsManager=settingsManager,
            facetLoader=createSessionPluginFacetLoader(options["pluginManifestPaths"]),
            cleanup=lambda context: envs.cleanup(context),
        )
    except Exception as error:
        try:
            if harness is not None:
                await harness.close(TODO_CONTEXT)
            await envs.cleanup(TODO_CONTEXT)
        except Exception as cleanupError:  # noqa: BLE001
            raise ExceptionGroup("Session worker Harness startup and cleanup failed", [error, cleanupError]) from None
        raise


def runSessionWorkerProcess(args: list[str]) -> Any:
    return runSessionWorkerWithHarness(args, createCodingAgentHarness)


def main() -> None:
    role = consumeInternalProcessRole()
    if role != "session-worker":
        raise RuntimeError("Session worker entrypoint requires an internal session-worker invocation")
    try:
        code = asyncio.run(runSessionWorkerProcess(sys.argv[1:]))
    except Exception:  # noqa: BLE001 - the worker has reported its failure
        sys.exit(1)
    sys.exit(code)


if __name__ == "__main__":
    main()
