"""Translated from pi's ``coding-agent/src/experimental/server.ts``.

The experimental server: it routes clients to durable Sessions, each owned by a Session worker process, behind a stable
endpoint that a coordinator process holds so the server can be replaced while its Sessions keep running.

MISAKA fork:

- The default directories live in the MISAKA home (``home.path("experimental_server")`` and
  ``home.path("experimental_sessions")``) where upstream uses ``~/.pi/server`` and ``<agent dir>/experimental/sessions``.
- Options and results are plain dicts with upstream's key names. ``activateServer`` resolves to ``{"client", "route"}``;
  ``acquireServerProfile`` to a ``SimpleNamespace(serverId, release)``; ``startServer`` to a ``SimpleNamespace`` of
  ``serverId``, ``sessionDir``, ``socketPath``, ``server``, ``workerPids``, ``closed`` and ``close()``.
- ``ServerLifetime`` takes ``timers`` (``timers.py``) so tests can control the clock. A timer never keeps the process alive
  here, so upstream's ``unref()`` is not needed.
- The internal server process is ``python -m misaka.experimental.server``; there is no direct-entry check.
"""

from __future__ import annotations

import asyncio
import math
import os
import signal
import stat
import sys
import time
import traceback
import uuid
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

from misaka.chord import Context
from misaka.chord.context import BACKGROUND_CONTEXT
from misaka.client import Client, DisconnectedError
from misaka.client import ServerError as ClientServerError
from misaka.client.unix import createUnixTransportFactory, isErrorCode
from misaka.config import home
from misaka.experimental.coordinator import CoordinatorConnection, ensureCoordinator
from misaka.experimental.plugins.bundled import createPresentationFacetData
from misaka.experimental.plugins.package import (
    createServerPluginPackage,
    normalizePluginPackagePaths,
    readSessionPluginPackageProfile,
    removeSessionPluginPackageProfile,
    restoreServerPluginPackageProfile,
    writeSessionPluginPackageProfile,
)
from misaka.experimental.process import (
    consumeInternalProcessRole,
    onceSignal,
    spawnInternalProcess,
    terminateInternalProcess,
)
from misaka.experimental.radius_auth import RadiusRelayAuthResolver
from misaka.experimental.radius_relay import RadiusRelayHost
from misaka.experimental.services.server import createExperimentalServerServices
from misaka.experimental.session_catalog import (
    createSession as createCatalogSession,
)
from misaka.experimental.session_catalog import (
    deleteSession,
    readSession,
)
from misaka.experimental.session_catalog import (
    listSessions as listCatalogSessions,
)
from misaka.experimental.session_worker_manager import (
    SessionPluginSelectionConflictError,
    SessionWorkerManager,
)
from misaka.experimental.timers import DEFAULT_TIMERS
from misaka.protocol import isServerId
from misaka.server import ServerError as RoutedServerError
from misaka.server import SessionNotFoundError
from misaka.server.transports.unix import createUnixServer, getUnixSocketPath
from misaka.utils.collation import localeSortKey
from misaka.utils.js import (
    aggregateError,
    allSettled,
    errorMessage,
    jsonParse,
    jsonStringify,
    jsTrim,
    promiseAll,
    spawn,
    startTask,
)
from misaka.utils.lockfile import lock
from misaka.utils.paths import resolvePath

ENV_SERVER_DIR = "PI_SERVER_DIR"
ENV_SERVER_ID = "PI_SERVER_ID"

_background: set[asyncio.Task[Any]] = set()


def resolveServerDirectory(directory: str | None = None) -> str:
    chosen = directory if directory is not None else os.environ.get(ENV_SERVER_DIR)
    return resolvePath(chosen if chosen is not None else str(home.path("experimental_server")))


def _ensurePrivateDirectory(directory: str) -> None:
    os.makedirs(directory, mode=0o700, exist_ok=True)
    stats = os.lstat(directory)
    if not stat.S_ISDIR(stats.st_mode):
        raise RuntimeError(f"Unix socket directory is not a directory: {directory}")
    if stats.st_uid != os.getuid():  # type: ignore[attr-defined,unused-ignore]  # windows-footgun: ok - hasattr-guarded
        raise RuntimeError(f"Unix socket directory is not owned by the current user: {directory}")
    os.chmod(directory, 0o700)


async def ensurePrivateServerDirectory(directory: str) -> None:
    if not hasattr(os, "getuid"):
        raise RuntimeError("Unix socket directory requires a POSIX user ID")
    await asyncio.to_thread(_ensurePrivateDirectory, directory)


def resolveSessionDirectory(sessionDir: str | None = None) -> str:
    return resolvePath(sessionDir if sessionDir is not None else str(home.path("experimental_sessions")))


LOCK_STALE_MS = 30_000
LOCK_RETRY_MS = 25
LOCK_WAIT_MS = 30_000
DEFAULT_SERVER_ID_FILE = "default-server-id"


def _readText(path: str) -> str:
    # ``readFile(path, "utf8")`` replaces invalid bytes instead of failing.
    with open(path, encoding="utf-8", errors="replace", newline="") as file:
        return file.read()


def _createText(path: str, text: str) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with open(descriptor, "w", encoding="utf-8", newline="") as file:
        file.write(text)


async def acquireServerProfile(directory: str, requestedServerId: str | None = None) -> SimpleNamespace:
    """Lock one logical server ID in a shared experimental server directory. Resolves to
    ``SimpleNamespace(serverId, release)``; ``release()`` is async."""
    await asyncio.to_thread(os.makedirs, directory, 0o700, True)
    if requestedServerId is not None:
        if not isServerId(requestedServerId):
            raise RuntimeError(f"Invalid experimental server ID: {requestedServerId}")
        serverId = requestedServerId
    else:
        path = os.path.join(directory, DEFAULT_SERVER_ID_FILE)
        try:
            value = jsTrim(await asyncio.to_thread(_readText, path))
        except FileNotFoundError:
            candidate = str(uuid.uuid4())
            try:
                await asyncio.to_thread(_createText, path, candidate)
                serverId = candidate
            except FileExistsError:
                value = jsTrim(await asyncio.to_thread(_readText, path))
                if not isServerId(value):
                    raise RuntimeError(f"Invalid default experimental server identity in {path}") from None
                serverId = value
        else:
            if not isServerId(value):
                raise RuntimeError(f"Invalid default experimental server identity in {path}")
            serverId = value

    release = await lock(
        os.path.join(directory, f"launcher-{serverId}"),
        {
            "realpath": False,
            "stale": LOCK_STALE_MS,
            "update": LOCK_STALE_MS / 3,
            "retries": {
                "retries": math.ceil(LOCK_WAIT_MS / LOCK_RETRY_MS),
                "factor": 1,
                "minTimeout": LOCK_RETRY_MS,
                "maxTimeout": LOCK_RETRY_MS,
                "maxRetryTime": LOCK_WAIT_MS,
            },
        },
    )
    return SimpleNamespace(serverId=serverId, release=release)


ACTIVATION_TIMEOUT_MS = 10_000
ACTIVATION_RETRY_MS = 10


def _now() -> float:
    """``Date.now()``: milliseconds."""
    return time.time() * 1000


async def activateServer(options: dict[str, Any]) -> dict[str, Any]:
    """Ensure the selected logical server is reachable, launching the current Pi installation if needed. ``options`` is
    ``{"directory", "requestedServerId"?, "sessionDir", "provider"?, "model"?}``; the result is ``{"client", "route"}``."""
    provider, model = options.get("provider"), options.get("model")
    if provider is not None and model is None:
        raise RuntimeError("Server model provider requires a model")
    await ensurePrivateServerDirectory(options["directory"])
    profile = await acquireServerProfile(options["directory"], options.get("requestedServerId"))
    serverId = profile.serverId
    await profile.release()
    route = {"serverId": serverId, "path": getUnixSocketPath(serverId, options["directory"])}
    release = await acquireServerActivation(options["directory"], serverId)
    try:
        existing = await connect(route)
        if existing is not None:
            # Another activator won the race, so startup-only selections can no longer be applied.
            if model is not None:
                await existing.dispose()
                raise RuntimeError("Model selection is only valid when automatically activating a new server")
            return {"client": existing, "route": route}
        modelArgs = (
            []
            if model is None
            else [jsonStringify({**({} if provider is None else {"provider": provider}), "model": model})]
        )
        child = spawnInternalProcess("server", [options["directory"], serverId, options["sessionDir"], *modelArgs])
        spawnError: list[BaseException] = []
        child.once("error", lambda error: spawnError.append(error))
        try:
            deadline = _now() + ACTIVATION_TIMEOUT_MS
            while True:
                client = await connect(route)
                if client is not None:
                    return {"client": client, "route": route}
                if len(spawnError) > 0:
                    raise RuntimeError("Failed to automatically activate server") from spawnError[-1]
                if child.exitCode is not None or child.signalCode is not None:
                    raise RuntimeError("Automatically activated server exited during startup")
                if _now() >= deadline:
                    raise RuntimeError("Timed out waiting for automatically activated server")
                await asyncio.sleep(ACTIVATION_RETRY_MS / 1000)
        except BaseException:
            await terminateInternalProcess(child)
            raise
    finally:
        await release()


def acquireServerActivation(directory: str, serverId: str) -> Any:
    """Lock the activation of one logical server. Resolves to the async function that releases the lock."""
    return lock(
        os.path.join(directory, f"activation-{serverId}"),
        {
            "realpath": False,
            "stale": ACTIVATION_TIMEOUT_MS * 2,
            "update": ACTIVATION_TIMEOUT_MS,
            "retries": {
                "retries": math.ceil(ACTIVATION_TIMEOUT_MS / 25),
                "factor": 1,
                "minTimeout": 25,
                "maxTimeout": 25,
                "maxRetryTime": ACTIVATION_TIMEOUT_MS,
            },
        },
    )


_UNREACHABLE_CODES = ("ENOENT", "ECONNREFUSED", "ECONNRESET", "EPIPE", "ETIMEDOUT")


async def connect(route: dict[str, str]) -> Client | None:
    client = Client({"serverId": route["serverId"], "transportFactory": createUnixTransportFactory({"path": route["path"]})})
    try:
        await client.connect()
        return client
    except BaseException as error:
        await client.dispose()
        if isinstance(error, DisconnectedError) or (isinstance(error, ClientServerError) and error.code == "version"):
            return None
        if any(isErrorCode(error, code) for code in _UNREACHABLE_CODES):
            return None
        raise


AUTO_SERVER_STARTUP_GRACE_MS = 10_000
AUTO_SERVER_IDLE_GRACE_MS = 1_000


class ServerLifetime:
    """Reconcile operator, startup, client, and worker holds for one server generation."""

    def __init__(self, keepAlive: bool, timers: Any = None) -> None:
        self._timers = timers or DEFAULT_TIMERS
        self._keepAlive = keepAlive
        self._connectionCount = 0
        self._workerCount = 0
        self._startupHeld = not keepAlive
        self._startupTimer: Any = None
        self._retirementTimer: Any = None
        self._retire: Callable[[], None] | None = None
        self._stopped = False

    def start(self, retire: Callable[[], None]) -> None:
        self._retire = retire
        if self._startupHeld:

            def release() -> None:
                self._startupTimer = None
                self._startupHeld = False
                self._reconcile()

            self._startupTimer = self._timers.setTimeout(release, AUTO_SERVER_STARTUP_GRACE_MS)
        self._reconcile()

    def setConnectionCount(self, count: int) -> None:
        self._connectionCount = count
        if count > 0 and self._startupHeld:
            self._startupHeld = False
            if self._startupTimer is not None:
                self._timers.clearTimeout(self._startupTimer)
            self._startupTimer = None
        self._reconcile()

    def setWorkerCount(self, count: int) -> None:
        self._workerCount = count
        self._reconcile()

    def stop(self) -> None:
        self._stopped = True
        if self._startupTimer is not None:
            self._timers.clearTimeout(self._startupTimer)
        if self._retirementTimer is not None:
            self._timers.clearTimeout(self._retirementTimer)
        self._startupTimer = None
        self._retirementTimer = None

    def _reconcile(self) -> None:
        if (
            self._stopped
            or self._keepAlive
            or self._startupHeld
            or self._connectionCount != 0
            or self._workerCount != 0
        ):
            if self._retirementTimer is not None:
                self._timers.clearTimeout(self._retirementTimer)
            self._retirementTimer = None
            return
        retire = self._retire
        if self._retirementTimer is not None or retire is None:
            return

        def fire() -> None:
            self._retirementTimer = None
            if not self._stopped and not self._startupHeld and self._connectionCount == 0 and self._workerCount == 0:
                retire()

        self._retirementTimer = self._timers.setTimeout(fire, AUTO_SERVER_IDLE_GRACE_MS)


def _retire(runtime: SimpleNamespace) -> None:
    spawn(_quietly(runtime.close()), _background)


def _observe(future: asyncio.Future[Any]) -> None:
    if not future.cancelled():
        future.exception()


async def _quietly(awaitable: Any) -> None:
    try:
        await awaitable
    except Exception:  # noqa: BLE001, S110 - `.catch(() => {})`
        pass


async def startServerBackend(
    options: dict[str, Any],
    workers: SessionWorkerManager,
    onConnectionCountChanged: Callable[[int], None] | None = None,
) -> SimpleNamespace:
    """``options`` is ``{"path", "serverId", "sessionDir"?, "resolveSessionPlugins", "removeSessionPlugins",
    "reloadPresentationFacetBundles"}``."""
    serverId = options["serverId"]
    sessionDir = resolveSessionDirectory(options.get("sessionDir"))

    async def listSessions() -> list[dict[str, Any]]:
        sessions = {metadata["path"]: metadata for metadata in await listCatalogSessions(sessionDir)}
        for metadata in workers.trackedSessions:
            sessions[metadata["path"]] = metadata
        return list(sessions.values())

    async def resolveSession(sessionId: str, _context: Context) -> dict[str, Any]:
        metadata = next((candidate for candidate in workers.trackedSessions if candidate["id"] == sessionId), None)
        if metadata is None:
            metadata = await readSession(sessionDir, sessionId)
        if metadata is None:
            raise SessionNotFoundError(f"Unknown session: {sessionId}")
        return metadata

    def createSession(createOptions: dict[str, Any]) -> Any:
        return createCatalogSession(sessionDir, {**createOptions, "cwd": os.getcwd()})

    def summarize(metadata: dict[str, Any]) -> dict[str, Any]:
        return {"serverId": serverId, "sessionId": metadata["id"], "createdAt": metadata["createdAt"]}

    async def list_(*_: Any) -> list[dict[str, Any]]:
        summaries = [summarize(metadata) for metadata in await listSessions()]
        return sorted(summaries, key=lambda summary: (localeSortKey(summary["sessionId"]), summary["createdAt"]))

    async def create(createOptions: dict[str, Any], *_: Any) -> dict[str, Any]:
        return summarize(await createSession(createOptions))

    async def remove(sessionId: str, context: Context) -> None:
        metadata = await resolveSession(sessionId, context)
        await workers.closeSession(metadata, context)
        await deleteSession(metadata)
        await options["removeSessionPlugins"](metadata)

    async def prepareSessionPlugins(sessionId: str, packagePaths: Any, context: Context) -> dict[str, Any]:
        metadata = await resolveSession(sessionId, context)
        try:
            selected = await options["resolveSessionPlugins"](metadata, packagePaths, context)
            workers.assertSessionPluginManifestPaths(metadata, selected["manifestPaths"])
        except SessionPluginSelectionConflictError as error:
            raise RoutedServerError("service_invalid_value", errorMessage(error)) from None
        return {
            "packagePaths": selected["packagePaths"],
            "presentationPlugins": createPresentationFacetData(selected["presentationArtifacts"]),
        }

    async def reloadPresentationPlugins(packagePaths: Any, *_: Any) -> Any:
        return createPresentationFacetData(await options["reloadPresentationFacetBundles"](packagePaths))

    serverServices = await createExperimentalServerServices(
        {
            "list": list_,
            "create": create,
            "remove": remove,
            "prepareSessionPlugins": prepareSessionPlugins,
            "reloadPresentationPlugins": reloadPresentationPlugins,
        }
    )

    async def openSession(metadata: dict[str, Any], context: Context) -> Any:
        selected = await options["resolveSessionPlugins"](metadata, None, context)
        return await workers.openSession(metadata, context, selected["manifestPaths"])

    host = SimpleNamespace(serverServices=serverServices.host, resolveSession=resolveSession, openSession=openSession)
    socketPath = options["path"]

    def closeCatalog() -> Any:
        return serverServices.dispose()

    serverOptions: dict[str, Any] = {"serverId": serverId, "path": socketPath, "mode": 0o600}
    if onConnectionCountChanged is not None:
        serverOptions["onConnectionCountChanged"] = onConnectionCountChanged
    server = createUnixServer(host, serverOptions)  # type: ignore[arg-type]
    try:
        await server.start()
    except Exception as error:
        cleanup = await allSettled([startTask(server.close()), startTask(closeCatalog())])
        cleanupErrors = [result.reason for result in cleanup if result.status == "rejected"]
        if len(cleanupErrors) > 0:
            raise aggregateError([error, *cleanupErrors], "Experimental server startup and cleanup failed") from None
        raise

    async def runClosed() -> None:
        try:
            await server.closed
        except Exception as serverError:
            try:
                await closeCatalog()
            except Exception as repoError:  # noqa: BLE001 - reported together with the server's
                raise aggregateError([serverError, repoError], "Server and repository shutdown failed") from None
            raise
        await closeCatalog()

    closed = startTask(runClosed())
    closed.add_done_callback(_observe)
    closeHolder: list[asyncio.Future[None]] = []

    def close() -> asyncio.Future[None]:
        if not closeHolder:

            async def run() -> None:
                try:
                    await server.close()
                except Exception:  # noqa: BLE001, S110 - `closed` reports the failure
                    pass
                await closed

            closeHolder.append(startTask(run()))
        return closeHolder[0]

    return SimpleNamespace(
        serverId=serverId,
        sessionDir=sessionDir,
        socketPath=socketPath,
        server=server,
        workerPids=workers.workerPids,
        closed=closed,
        refreshSessions=lambda: serverServices.refresh(BACKGROUND_CONTEXT),
        close=close,
    )


async def startServer(options: dict[str, Any] | None = None) -> SimpleNamespace:
    """Start a replaceable experimental server behind the stable coordinator endpoint. ``options`` is ``{"directory"?,
    "serverId"?, "sessionDir"?, "provider"?, "model"?, "keepAlive"?, "relayAuth"?, "pluginPackages"?, "onRelayStatus"?}``:
    the server profile and socket directory (default ``PI_SERVER_DIR`` or the home's ``run/server``); the logical service ID
    (default ``PI_SERVER_ID`` or the directory's ``default-server-id``); the durable Session directory; an optional provider
    and model for newly started Session workers; whether to hold the server open without client or Session demand
    (default true); an explicit Radius credential (stored Radius auth otherwise); and explicit plugin packages (``None``
    restores the logical server profile, an empty list clears it)."""
    options = options or {}
    provider, model = options.get("provider"), options.get("model")
    if provider is not None and model is None:
        raise RuntimeError("Server model provider requires a model")
    workerModel = (
        None if model is None else {**({} if provider is None else {"provider": provider}), "model": model}
    )
    directory = resolveServerDirectory(options.get("directory"))
    requestedServerId = options.get("serverId")
    profile = await acquireServerProfile(
        directory, requestedServerId if requestedServerId is not None else os.environ.get(ENV_SERVER_ID)
    )
    serverId, release = profile.serverId, profile.release
    keepAlive = options.get("keepAlive")
    lifetime = ServerLifetime(True if keepAlive is None else keepAlive)
    backend: SimpleNamespace | None = None
    coordinator: CoordinatorConnection | None = None
    startupLease: Any = None
    workers: SessionWorkerManager | None = None
    relay: RadiusRelayHost | None = None
    released = False
    try:
        await ensurePrivateServerDirectory(directory)
        pluginPackagePaths = await restoreServerPluginPackageProfile(directory, serverId, options.get("pluginPackages"))
        pluginPackages: dict[str, Any] = {}

        def getPluginPackage(packagePath: str) -> Any:
            plugin = pluginPackages.get(packagePath)
            if plugin is None:
                plugin = createServerPluginPackage(directory, serverId, packagePath)
                pluginPackages[packagePath] = plugin
            return plugin

        async def buildPluginSelection(packagePaths: Any) -> dict[str, Any]:
            normalizedPackagePaths = normalizePluginPackagePaths(packagePaths)
            plugins = [getPluginPackage(packagePath) for packagePath in normalizedPackagePaths]
            built = [artifact for artifacts in await promiseAll([plugin.build() for plugin in plugins]) for artifact in artifacts]
            return {
                "packagePaths": normalizedPackagePaths,
                "manifestPaths": [plugin.manifestPath for plugin in plugins],
                "presentationArtifacts": built,
            }

        defaultPluginSelection = await buildPluginSelection(pluginPackagePaths)
        sessionPluginSelections: dict[str, dict[str, Any]] = {}

        async def resolveSessionPlugins(
            metadata: dict[str, Any], requestedPackagePaths: Any, _context: Context
        ) -> dict[str, Any]:
            if requestedPackagePaths is not None:
                normalizedPackagePaths = normalizePluginPackagePaths(requestedPackagePaths)
                current = sessionPluginSelections.get(metadata["path"])
                if current is not None and sameStrings(current["packagePaths"], normalizedPackagePaths):
                    return current
                requestedManifestPaths = [
                    getPluginPackage(packagePath).manifestPath for packagePath in normalizedPackagePaths
                ]
                if workers is not None:
                    workers.assertSessionPluginManifestPaths(metadata, requestedManifestPaths)
                candidate = await buildPluginSelection(normalizedPackagePaths)
                if workers is not None:
                    workers.assertSessionPluginManifestPaths(metadata, candidate["manifestPaths"])
                await writeSessionPluginPackageProfile(directory, serverId, metadata["path"], candidate["packagePaths"])
                sessionPluginSelections[metadata["path"]] = candidate
                return candidate
            cached = sessionPluginSelections.get(metadata["path"])
            if cached is not None:
                return cached
            storedPackagePaths = await readSessionPluginPackageProfile(directory, serverId, metadata["path"])
            selected = defaultPluginSelection if storedPackagePaths is None else await buildPluginSelection(storedPackagePaths)
            if storedPackagePaths is None:
                await writeSessionPluginPackageProfile(directory, serverId, metadata["path"], selected["packagePaths"])
            sessionPluginSelections[metadata["path"]] = selected
            return selected

        async def removeSessionPlugins(metadata: dict[str, Any]) -> None:
            sessionPluginSelections.pop(metadata["path"], None)
            await removeSessionPluginPackageProfile(directory, serverId, metadata["path"])

        async def reloadPresentationFacetBundles(packagePaths: Any) -> Any:
            nonlocal defaultPluginSelection
            reloaded = await buildPluginSelection(packagePaths)
            if sameStrings(defaultPluginSelection["packagePaths"], reloaded["packagePaths"]):
                defaultPluginSelection = reloaded
            for sessionPath, selected in list(sessionPluginSelections.items()):
                if sameStrings(selected["packagePaths"], reloaded["packagePaths"]):
                    sessionPluginSelections[sessionPath] = reloaded
            return reloaded["presentationArtifacts"]

        socketPath = getUnixSocketPath(serverId, directory)
        controlPath = os.path.join(directory, f"control-{serverId}.sock")
        serverNonce = str(uuid.uuid4()).replace("-", "")[:12]
        serverPath = os.path.join(directory, f"server-{serverId}-{serverNonce}.sock")
        startupLease = await ensureCoordinator(socketPath, controlPath)
        coordinator = CoordinatorConnection({"controlPath": controlPath, "endpoint": serverPath})
        sessionDir = resolveSessionDirectory(options.get("sessionDir"))
        workers = SessionWorkerManager(coordinator, sessionDir, workerModel, lambda count: lifetime.setWorkerCount(count))
        backend = await startServerBackend(
            {
                "path": serverPath,
                "serverId": serverId,
                "sessionDir": options.get("sessionDir"),
                "resolveSessionPlugins": resolveSessionPlugins,
                "removeSessionPlugins": removeSessionPlugins,
                "reloadPresentationFacetBundles": reloadPresentationFacetBundles,
            },
            workers,
            lambda count: lifetime.setConnectionCount(count),
        )
        await coordinator.connect()
        startupLease.close()
        startupLease = None
        await workers.discover(coordinator.peerIds)
        await backend.refreshSessions()
        relayOptions: dict[str, Any] = {
            "serverId": serverId,
            "server": backend.server,
            "auth": RadiusRelayAuthResolver(options.get("relayAuth")),
        }
        if options.get("onRelayStatus") is not None:
            relayOptions["onStatus"] = options["onRelayStatus"]
        relay = RadiusRelayHost(relayOptions)
        relay.start()

        activeBackend = backend
        activeCoordinator = coordinator
        activeWorkers = workers
        activeRelay = relay

        async def watchReplaced() -> None:
            try:
                # Shielded: closing the server cancels this watcher, not the coordinator's shared future.
                await asyncio.shield(activeCoordinator.replaced)
                lifetime.stop()
                activeWorkers.detach()
                await activeRelay.close()
                await activeBackend.close()
            finally:
                activeCoordinator.close()

        watcher = spawn(_quietly(watchReplaced()), _background)
        closePromise: list[asyncio.Future[None]] = []

        async def closedAfterRelay() -> None:
            try:
                await activeBackend.closed
            finally:
                await activeRelay.close()

        closed = startTask(closedAfterRelay())
        closed.add_done_callback(_observe)

        def close() -> asyncio.Future[None]:
            lifetime.stop()
            if not closePromise:

                async def run() -> None:
                    try:
                        await activeRelay.close()
                        await activeBackend.close()
                    finally:
                        try:
                            if activeCoordinator.wasReplaced:
                                activeWorkers.detach()
                            else:
                                await activeWorkers.shutdown()
                        finally:
                            activeCoordinator.close()
                            # MISAKA fork: a closed coordinator never settles ``replaced``, and asyncio holds this task
                            # strongly (and the generation with it) until it is cancelled.
                            if not activeCoordinator.replaced.done():
                                watcher.cancel()

                closePromise.append(startTask(run()))
            return closePromise[0]

        runtime = SimpleNamespace(
            serverId=serverId,
            sessionDir=activeBackend.sessionDir,
            socketPath=socketPath,
            server=activeBackend.server,
            workerPids=activeWorkers.workerPids,
            closed=closed,
            close=close,
        )
        lifetime.start(lambda: _retire(runtime))
        released = True
        await release()
        return runtime
    except BaseException as error:
        lifetime.stop()
        if startupLease is not None:
            startupLease.close()
        if coordinator is not None and coordinator.wasReplaced and workers is not None:
            workers.detach()
        replaced = coordinator is not None and coordinator.wasReplaced
        pending: list[Any] = [
            None if relay is None else startTask(relay.close()),
            None if backend is None else backend.close(),
            None if replaced or workers is None else workers.shutdown(),
        ]
        if coordinator is not None:
            coordinator.close()
        pending.append(None if released else startTask(release()))
        cleanup = await allSettled(pending)
        cleanupErrors = [result.reason for result in cleanup if result.status == "rejected"]
        if len(cleanupErrors) > 0:
            raise aggregateError([error, *cleanupErrors], "Server runtime startup and cleanup failed") from None
        raise


async def startForegroundServer(options: dict[str, Any] | None = None) -> SimpleNamespace:
    """Start an operator-held server while serializing against automatic cold activation. ``options`` is
    ``startServer``'s without ``keepAlive``."""
    options = options or {}
    directory = resolveServerDirectory(options.get("directory"))
    await ensurePrivateServerDirectory(directory)
    requestedServerId = options.get("serverId")
    profile = await acquireServerProfile(
        directory, requestedServerId if requestedServerId is not None else os.environ.get(ENV_SERVER_ID)
    )
    serverId = profile.serverId
    await profile.release()
    release = await acquireServerActivation(directory, serverId)
    try:
        return await startServer({**options, "directory": directory, "serverId": serverId, "keepAlive": True})
    finally:
        await release()


def parseServerModelOptions(value: str | None) -> dict[str, str] | None:
    if value is None:
        return None
    try:
        parsed = jsonParse(value)
    except ValueError as error:
        raise RuntimeError("Internal server received invalid model options") from error
    if not isinstance(parsed, dict):
        raise RuntimeError("Internal server received invalid model options")  # noqa: TRY004
    model = parsed.get("model")
    provider = parsed.get("provider")
    if (
        any(key not in ("provider", "model") for key in parsed)
        or not isinstance(model, str)
        or len(model) == 0
        or ("provider" in parsed and (not isinstance(provider, str) or len(provider) == 0))
    ):
        raise RuntimeError("Internal server received invalid model options")
    return {"model": model} if "provider" not in parsed else {"provider": provider, "model": model}  # type: ignore[dict-item]


async def runServerProcess(args: list[str]) -> None:
    """Run an automatically activated server until its client and Session demand disappears."""
    directory, serverId, sessionDir, serializedModel = (list(args) + [None] * 4)[:4]  # type: ignore[list-item]
    if len(args) > 4:
        raise RuntimeError("Internal server received unexpected arguments")
    if not directory or not os.path.isabs(directory):
        raise RuntimeError("Internal server requires an absolute server directory")
    if not isServerId(serverId):
        raise RuntimeError("Internal server requires a canonical server ID")
    if not sessionDir or not os.path.isabs(sessionDir):
        raise RuntimeError("Internal server requires an absolute Session directory")
    workerModel = parseServerModelOptions(serializedModel)

    runtime = await startServer(
        {"directory": directory, "serverId": serverId, "sessionDir": sessionDir, "keepAlive": False, **(workerModel or {})}
    )

    def close() -> None:
        spawn(_quietly(runtime.close()), _background)

    handled: list[Callable[[], None]] = []
    if sys.platform != "win32":
        for signalNumber in (signal.SIGINT, signal.SIGTERM):
            handled.append(onceSignal(signalNumber, close))
    try:
        await runtime.closed
    finally:
        for off in handled:
            off()
        await runtime.close()


def sameStrings(left: Any, right: Any) -> bool:
    return len(left) == len(right) and all(value == right[index] for index, value in enumerate(left))


def main() -> None:
    role = consumeInternalProcessRole()
    if role != "server":
        raise RuntimeError("Server entrypoint requires an internal server invocation")
    try:
        asyncio.run(runServerProcess(sys.argv[1:]))
    except Exception:  # noqa: BLE001 - the process reports its failure and exits
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
