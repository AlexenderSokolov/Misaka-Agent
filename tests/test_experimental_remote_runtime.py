"""pi coding-agent's ``test/experimental-remote-runtime.test.ts``.

The server, its coordinator and its Session workers are real processes, so these tests need Unix sockets (``skipif`` on Windows).

MISAKA fork:

- The default Session directory and credentials are rows of the per-test home (``MISAKA_HOME``) where pi's tests point
  ``PI_CODING_AGENT_DIR`` at a temporary agent directory.
- The faux Session worker is ``tests.fixtures.faux_session_worker``, selected by wrapping ``spawnInternalProcess`` with an ``entry``.
- Plugin packages are Python (``pyproject.toml`` and ``src/session.py``).
- A worker process that the server stopped tracking is awaited (``waitFor``) before it is expected to be gone, where pi
  asserts at once: the Python worker finishes interpreter shutdown after its control socket closed.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import signal
import stat
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from misaka.chord import createFacetHost, defineFacet, defineService
from misaka.chord.context import BACKGROUND_CONTEXT
from misaka.client import ServerError, connectClient
from misaka.client.unix import createUnixTransportFactory
from misaka.config import home
from misaka.experimental import client_runtime
from misaka.experimental import process as processRuntime
from misaka.experimental.client import runClient
from misaka.experimental.client_runtime import (
    activateBuiltinClientServices,
    openClientRuntime,
)
from misaka.experimental.plugins.bundled import createPresentationFacetLoaders
from misaka.experimental.server import startServer
from misaka.experimental.services.agent_controller import AgentController
from misaka.experimental.services.connection import createSessionServiceSource
from misaka.experimental.services.models import Models
from misaka.experimental.services.plugins import PresentationPlugins, SessionPlugins
from misaka.experimental.services.sessions import SessionDirectory, SessionManagement
from misaka.experimental.services.transcript import Transcript
from tests.chat_support import waitFor
from tests.experimental_service_binding import (
    createServerServiceBinding,
    createSessionServiceBinding,
)
from tests.experimental_session_support import (
    configureExperimentalWorkerModel,
    createExperimentalSessions,
    makeShortDirectory,
    readExperimentalSessionState,
)
from tests.fixtures.keyed_service import KeyedProbe

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="the experimental server listens on a Unix socket")

REPOSITORY = str(Path(__file__).resolve().parents[1])
EXAMPLE_PLUGIN = os.path.join(REPOSITORY, "examples", "plugins", "pi-example-plugin")
ExampleFacetService = defineService("pi.example-plugin.greeting")
SecondPluginService = defineService("test.second-plugin")
sessionWorkerModel = {"provider": "anthropic", "model": "claude-sonnet-4-5"}
realSpawnInternalProcess = processRuntime.spawnInternalProcess

SECOND_SESSION_SOURCE = (
    "from types import SimpleNamespace\n"
    "\n"
    "from misaka.chord import defineFacet, defineService\n"
    "\n"
    'Service = defineService("test.second-plugin")\n'
    "\n"
    "\n"
    "async def read(*_):\n"
    '    return "second"\n'
    "\n"
    "\n"
    "def setup(env):\n"
    '    env.provide(Service, {"read": read})\n'
    "\n"
    "\n"
    'default = defineFacet(SimpleNamespace(id="second-session-plugin", setup=setup))\n'
)


class RemoteEnvironment:
    def __init__(self) -> None:
        self.servers: list[Any] = []
        self.clients: set[Any] = set()
        self.directories: set[str] = set()

    def shortDirectory(self, prefix: str) -> str:
        directory = makeShortDirectory(prefix)
        self.directories.add(directory)
        return directory

    async def makeServer(self) -> tuple[str, Any]:
        directory = self.shortDirectory("pes-")
        runtime = await startServer({**sessionWorkerModel, "directory": directory})
        self.servers.append(runtime)
        return directory, runtime

    async def connect(self, runtime: Any) -> Any:
        client = await connectClient(
            {
                "serverId": runtime.serverId,
                "transportFactory": createUnixTransportFactory({"path": runtime.socketPath}),
            }
        )
        self.clients.add(client)
        return client

    async def attachSession(self, client: Any, sessionId: str) -> None:
        services = createServerServiceBinding(client, {"services": [SessionManagement]})
        try:
            await services.ready(BACKGROUND_CONTEXT)
            await services.use(SessionManagement).attach(sessionId, BACKGROUND_CONTEXT)
        finally:
            await services.dispose(BACKGROUND_CONTEXT)

    async def attachClient(self, runtime: Any, sessionId: str) -> Any:
        client = await self.connect(runtime)
        await self.attachSession(client, sessionId)
        return client


@pytest.fixture(autouse=True)
async def remote():
    state = RemoteEnvironment()
    configureExperimentalWorkerModel()
    await createExperimentalSessions(str(home.path("experimental_sessions")), ["demo-1", "demo-2"])
    yield state
    # A rejected dispose or close fails the test, as it does in pi's ``afterEach``; every cleanup still runs first.
    results = await asyncio.gather(*(client.dispose() for client in list(state.clients)), return_exceptions=True)
    state.clients.clear()
    results += await asyncio.gather(*(server.close() for server in list(state.servers)), return_exceptions=True)
    state.servers.clear()
    for directory in state.directories:
        shutil.rmtree(directory, ignore_errors=True)
    errors = [result for result in results if isinstance(result, Exception)]
    if errors:
        raise ExceptionGroup("Remote runtime teardown failed", errors)


@pytest.fixture
def fauxWorker(monkeypatch):
    """Run Session workers as the faux worker module."""

    def spawn(role: str, args: list[str], options: dict[str, Any] | None = None) -> Any:
        if role == "session-worker":
            options = {
                **(options or {}),
                "entry": "tests.fixtures.faux_session_worker",
                "env": {**((options or {}).get("env") or {}), "PYTHONPATH": REPOSITORY},
            }
        return realSpawnInternalProcess(role, args, options)

    monkeypatch.setattr(processRuntime, "spawnInternalProcess", spawn)


def pathExists(path: str) -> bool:
    return os.path.lexists(path)


def processExists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except PermissionError:
        return True
    except ProcessLookupError:
        return False


def sessionIds(state: Any) -> list[str]:
    return [session["sessionId"] for session in state.value["sessions"]]


async def test_uses_pi_server_dir_and_pi_server_id(remote, monkeypatch):
    directory = remote.shortDirectory("pi-server-dir-")
    serverId = "00000000-0000-4000-8000-000000000001"
    monkeypatch.setenv("PI_SERVER_DIR", directory)
    monkeypatch.setenv("PI_SERVER_ID", serverId)
    runtime = await startServer()
    remote.servers.append(runtime)

    assert runtime.serverId == serverId
    assert runtime.socketPath == os.path.join(directory, f"{serverId}.sock")
    assert stat.S_IMODE(os.lstat(directory).st_mode) == 0o700
    publicSocket = os.lstat(runtime.socketPath)
    controlSocket = os.lstat(os.path.join(directory, f"control-{runtime.serverId}.sock"))
    assert stat.S_ISSOCK(publicSocket.st_mode)
    assert stat.S_IMODE(publicSocket.st_mode) == 0o600
    assert stat.S_ISSOCK(controlSocket.st_mode)
    assert stat.S_IMODE(controlSocket.st_mode) == 0o600
    entries = os.listdir(directory)
    assert len(entries) == 3
    assert all(not entry.startswith(".") for entry in entries)
    assert f"{serverId}.sock" in entries
    assert f"control-{serverId}.sock" in entries
    assert any(re.fullmatch(rf"server-{serverId}-[0-9a-f]{{12}}\.sock", entry) for entry in entries)
    result = await runClient({"command": "client"})
    assert result["kind"] == "list"
    assert [session["sessionId"] for session in result["sessions"]] == ["demo-1", "demo-2"]


async def test_rejects_a_provider_without_a_model(remote):
    directory = remote.shortDirectory("pes-")
    with pytest.raises(RuntimeError, match="provider requires a model"):
        await startServer({"directory": directory, "provider": "anthropic"})


async def test_preserves_an_existing_session_model_when_the_server_default_changes(remote):
    Path(home.path("settings")).write_bytes(
        json.dumps({"defaultProvider": "anthropic", "defaultModel": "claude-opus-4-6"}).encode("utf-8")
    )
    directory = remote.shortDirectory("pes-")
    first = await startServer({"directory": directory})
    remote.servers.append(first)
    firstClient = await remote.attachClient(first, "demo-1")
    await firstClient.dispose()
    remote.clients.discard(firstClient)
    await waitFor(lambda: "demo-1" not in first.workerPids, 10_000)
    await first.close()

    second = await startServer({**sessionWorkerModel, "directory": directory})
    remote.servers.append(second)
    secondClient = await remote.attachClient(second, "demo-1")
    await secondClient.dispose()
    remote.clients.discard(secondClient)
    await waitFor(lambda: "demo-1" not in second.workerPids, 10_000)
    state = await readExperimentalSessionState(second.sessionDir, "demo-1")
    assert state["model"] == {"provider": "anthropic", "modelId": "claude-opus-4-6"}


async def test_rejects_model_options_when_discovery_selects_an_existing_server(remote):
    directory, _runtime = await remote.makeServer()
    with pytest.raises(RuntimeError, match="Model selection is only valid when automatically activating a new server"):
        await runClient({"command": "client", "model": "anthropic/claude-opus-4-6"}, {"directory": directory})


async def test_rechecks_an_auto_discovered_server_after_a_version_mismatch(remote, monkeypatch):
    directory, runtime = await remote.makeServer()
    calls: list[int] = []

    async def connect(options: dict[str, Any]) -> Any:
        calls.append(1)
        if len(calls) == 1:
            raise ServerError({"code": "version", "message": "stale server"})
        return await connectClient(options)

    monkeypatch.setattr(client_runtime, "connectClient", connect)
    clientRuntime = await openClientRuntime({"command": "client"}, {"directory": directory})
    try:
        assert [server.route["serverId"] for server in clientRuntime.servers] == [runtime.serverId]
    finally:
        await clientRuntime.dispose()


async def test_serializes_concurrent_cold_activation_and_retires_after_both_clients_leave(remote, monkeypatch):
    directory = remote.shortDirectory("pi-auto-server-")
    serverId = "00000000-0000-4000-8000-000000000001"
    monkeypatch.setenv("PI_SERVER_DIR", directory)
    monkeypatch.setenv("PI_SERVER_ID", serverId)

    results = await asyncio.gather(runClient({"command": "client"}), runClient({"command": "client"}))
    expected = {
        "kind": "list",
        "sessions": [{"serverId": serverId, "sessionId": "demo-1"}, {"serverId": serverId, "sessionId": "demo-2"}],
    }
    assert list(results) == [expected, expected]
    assert pathExists(os.path.join(directory, f"{serverId}.sock"))
    await waitFor(lambda: not pathExists(os.path.join(directory, f"{serverId}.sock")), 10_000)
    await waitFor(lambda: not pathExists(os.path.join(directory, f"control-{serverId}.sock")), 10_000)


async def test_passes_client_plugin_packages_to_a_cold_server_and_restores_them_for_its_next_generation(
    remote, monkeypatch
):
    directory = remote.shortDirectory("pi-auto-plugin-")
    serverId = "00000000-0000-4000-8000-000000000001"
    monkeypatch.setenv("PI_SERVER_DIR", directory)
    monkeypatch.setenv("PI_SERVER_ID", serverId)

    first = await openClientRuntime({"command": "client", **sessionWorkerModel})
    try:
        activated = await activateBuiltinClientServices(first.servers[0])
        presentationPlugins = await activated.plugins.prepareSession(
            {"sessionId": "demo-1", "packagePaths": [EXAMPLE_PLUGIN]}, BACKGROUND_CONTEXT
        )
        await activated.management.attach("demo-1", BACKGROUND_CONTEXT)
        loaded = await createPresentationFacetLoaders(presentationPlugins)[0].load()
        assert [facet.id for facet in loaded.facets] == ["pi-example-plugin/tui"]
        await loaded.dispose()
    finally:
        await first.dispose()
    await waitFor(lambda: not pathExists(os.path.join(directory, f"{serverId}.sock")), 10_000)

    second = await openClientRuntime({"command": "client", **sessionWorkerModel})
    try:
        activated = await activateBuiltinClientServices(second.servers[0])
        presentationPlugins = await activated.plugins.prepareSession(
            {"sessionId": "demo-1", "packagePaths": None}, BACKGROUND_CONTEXT
        )
        await activated.management.attach("demo-1", BACKGROUND_CONTEXT)
        assert len(createPresentationFacetLoaders(presentationPlugins)) == 1
    finally:
        await second.dispose()


async def test_retires_a_cold_server_after_its_only_session_attachment_disconnects(remote, monkeypatch):
    directory = remote.shortDirectory("pi-auto-session-")
    serverId = "00000000-0000-4000-8000-000000000001"
    monkeypatch.setenv("PI_SERVER_DIR", directory)
    monkeypatch.setenv("PI_SERVER_ID", serverId)

    assert await runClient({"command": "client", "sessionId": "demo-1", **sessionWorkerModel}) == {
        "kind": "attached",
        "serverId": serverId,
        "sessionId": "demo-1",
    }
    await waitFor(lambda: not pathExists(os.path.join(directory, f"{serverId}.sock")), 10_000)
    await waitFor(lambda: not pathExists(os.path.join(directory, f"control-{serverId}.sock")), 10_000)


async def test_runs_and_discovers_multiple_logical_servers_from_one_directory(remote):
    directory = remote.shortDirectory("pi-multi-server-")
    firstId = "00000000-0000-4000-8000-000000000001"
    secondId = "00000000-0000-4000-8000-000000000002"
    first, second = await asyncio.gather(
        startServer({"directory": directory, "serverId": firstId}),
        startServer({"directory": directory, "serverId": secondId}),
    )
    remote.servers.append(first)
    remote.servers.append(second)

    assert stat.S_ISSOCK(os.lstat(os.path.join(directory, f"control-{firstId}.sock")).st_mode)
    assert stat.S_ISSOCK(os.lstat(os.path.join(directory, f"control-{secondId}.sock")).st_mode)
    assert await runClient({"command": "client"}, {"directory": directory}) == {
        "kind": "list",
        "sessions": [
            {"serverId": firstId, "sessionId": "demo-1"},
            {"serverId": firstId, "sessionId": "demo-2"},
            {"serverId": secondId, "sessionId": "demo-1"},
            {"serverId": secondId, "sessionId": "demo-2"},
        ],
    }

    await first.close()
    await waitFor(lambda: not pathExists(first.socketPath))
    assert await runClient({"command": "client"}, {"directory": directory}) == {
        "kind": "list",
        "sessions": [{"serverId": secondId, "sessionId": "demo-1"}, {"serverId": secondId, "sessionId": "demo-2"}],
    }


async def test_hydrates_and_mutates_server_session_services_across_framed_clients(remote):
    _directory, runtime = await remote.makeServer()
    firstClient = await remote.connect(runtime)
    secondClient = await remote.connect(runtime)
    errors: list[Exception] = []
    firstServices = createServerServiceBinding(
        firstClient, {"services": [SessionDirectory, SessionManagement], "onError": errors.append}
    )
    secondServices = createServerServiceBinding(
        secondClient, {"services": [SessionDirectory, SessionManagement], "onError": errors.append}
    )
    assert firstServices.connection.value["status"] == "connected"
    assert secondServices.connection.value["status"] == "connected"
    assert await firstServices.catalogue(BACKGROUND_CONTEXT) == [
        {"serviceId": SessionDirectory.id, "mode": "singleton"},
        {"serviceId": SessionManagement.id, "mode": "singleton"},
        {"serviceId": PresentationPlugins.id, "mode": "singleton"},
    ]
    firstDirectory = firstServices.use(SessionDirectory)
    secondDirectory = secondServices.use(SessionDirectory)
    firstManagement = firstServices.use(SessionManagement)
    secondManagement = secondServices.use(SessionManagement)

    await asyncio.gather(firstServices.ready(BACKGROUND_CONTEXT), secondServices.ready(BACKGROUND_CONTEXT))
    assert sessionIds(firstDirectory.state) == ["demo-1", "demo-2"]
    assert secondDirectory.state.value == firstDirectory.state.value
    await firstManagement.create({"id": "demo-3"}, BACKGROUND_CONTEXT)
    await waitFor(
        lambda: "demo-3" in sessionIds(firstDirectory.state) and secondDirectory.state.value == firstDirectory.state.value
    )

    await asyncio.gather(
        firstManagement.attach("demo-1", BACKGROUND_CONTEXT), secondManagement.attach("demo-1", BACKGROUND_CONTEXT)
    )
    assert firstClient.attachment["sessionId"] == "demo-1"
    assert secondClient.attachment["sessionId"] == "demo-1"
    await firstManagement.remove("demo-1", BACKGROUND_CONTEXT)
    await waitFor(
        lambda: firstClient.attachment is None
        and secondClient.attachment is None
        and "demo-1" not in sessionIds(firstDirectory.state)
        and secondDirectory.state.value == firstDirectory.state.value
    )
    await secondManagement.detach(BACKGROUND_CONTEXT)
    assert errors == []

    await asyncio.gather(firstServices.dispose(BACKGROUND_CONTEXT), secondServices.dispose(BACKGROUND_CONTEXT))


async def test_hydrates_and_updates_the_models_service_across_concurrent_framed_clients(remote):
    _directory, runtime = await remote.makeServer()
    firstClient = await remote.attachClient(runtime, "demo-1")
    workerPid = runtime.workerPids.get("demo-1")
    assert isinstance(workerPid, int)
    secondClient = await remote.attachClient(runtime, "demo-1")
    assert runtime.workerPids.get("demo-1") == workerPid
    errors: list[Exception] = []
    firstServices = createSessionServiceBinding(firstClient, {"services": [Models], "onError": errors.append})
    secondServices = createSessionServiceBinding(secondClient, {"services": [Models], "onError": errors.append})
    firstModels = firstServices.use(Models)
    secondModels = secondServices.use(Models)

    await asyncio.gather(firstServices.ready(BACKGROUND_CONTEXT), secondServices.ready(BACKGROUND_CONTEXT))
    assert firstServices.attachment.value == {"status": "attached", "sessionId": "demo-1"}
    assert secondServices.attachment.value == {"status": "attached", "sessionId": "demo-1"}
    assert firstModels.state.value["configuration"]["model"] == {"provider": "anthropic", "modelId": "claude-sonnet-4-5"}
    assert secondModels.state.value == firstModels.state.value
    previousThinking = firstModels.state.value["configuration"]["thinkingLevel"]
    await firstModels.cycleThinking(BACKGROUND_CONTEXT)
    await waitFor(
        lambda: firstModels.state.value["configuration"]["thinkingLevel"] != previousThinking
        and secondModels.state.value == firstModels.state.value
    )
    assert errors == []

    await asyncio.gather(firstServices.dispose(BACKGROUND_CONTEXT), secondServices.dispose(BACKGROUND_CONTEXT))


async def test_loads_conventional_session_facets_from_multiple_configured_plugin_packages(remote):
    directory = remote.shortDirectory("pes-plugin-")
    secondPackagePath = os.path.join(directory, "second-plugin")
    os.makedirs(os.path.join(secondPackagePath, "src"))
    Path(secondPackagePath, "pyproject.toml").write_bytes(b'[project]\nname = "second-session-plugin"\nversion = "1.0.0"\n')
    Path(secondPackagePath, "src", "session.py").write_bytes(SECOND_SESSION_SOURCE.encode("utf-8"))
    runtime = await startServer({**sessionWorkerModel, "directory": directory})
    remote.servers.append(runtime)
    clientRuntime = await openClientRuntime({"command": "client"}, {"directory": directory})
    activated = await activateBuiltinClientServices(clientRuntime.servers[0])
    await activated.plugins.prepareSession(
        {"sessionId": "demo-1", "packagePaths": [EXAMPLE_PLUGIN, secondPackagePath]}, BACKGROUND_CONTEXT
    )
    await activated.management.attach("demo-1", BACKGROUND_CONTEXT)
    services = createSessionServiceBinding(
        clientRuntime.servers[0].client, {"services": [ExampleFacetService, SecondPluginService]}
    )
    try:
        await services.ready(BACKGROUND_CONTEXT)
        greeting = await services.use(ExampleFacetService).greet({"name": "Armin"}, BACKGROUND_CONTEXT)
        assert greeting["workerActivations"] == 1
        assert await services.use(SecondPluginService).read(BACKGROUND_CONTEXT) == "second"
    finally:
        await services.dispose(BACKGROUND_CONTEXT)
        await clientRuntime.dispose()


async def test_uses_the_most_recently_selected_model_for_a_new_session(remote):
    directory = remote.shortDirectory("pes-model-default-")
    runtime = await startServer({"directory": directory})
    remote.servers.append(runtime)
    firstClient = await remote.attachClient(runtime, "demo-1")
    firstServices = createSessionServiceBinding(firstClient, {"services": [Models]})
    firstModels = firstServices.use(Models)
    await firstServices.ready(BACKGROUND_CONTEXT)
    await firstModels.select({"provider": "anthropic", "modelId": "claude-opus-4-6"}, BACKGROUND_CONTEXT)
    await firstServices.dispose(BACKGROUND_CONTEXT)
    await firstClient.dispose()
    remote.clients.discard(firstClient)
    await waitFor(lambda: "demo-1" not in runtime.workerPids, 10_000)

    secondClient = await remote.attachClient(runtime, "demo-2")
    secondServices = createSessionServiceBinding(secondClient, {"services": [Models]})
    secondModels = secondServices.use(Models)
    await secondServices.ready(BACKGROUND_CONTEXT)
    assert secondModels.state.value["configuration"]["model"] == {"provider": "anthropic", "modelId": "claude-opus-4-6"}
    await secondServices.dispose(BACKGROUND_CONTEXT)


async def test_composes_management_attachment_with_session_service_hydration(remote):
    _directory, runtime = await remote.makeServer()
    clientRuntime = await openClientRuntime(
        {"command": "client", "connect": {"transport": "unix", "path": runtime.socketPath}}
    )
    try:
        server = await activateBuiltinClientServices(clientRuntime.servers[0])
        await server.plugins.prepareSession({"sessionId": "demo-1", "packagePaths": None}, BACKGROUND_CONTEXT)
        await server.management.attach("demo-1", BACKGROUND_CONTEXT)

        assert server.session.attachment.value == {"status": "attached", "sessionId": "demo-1"}
        assert server.models.state.value["configuration"]["model"] == {
            "provider": "anthropic",
            "modelId": "claude-sonnet-4-5",
        }
    finally:
        await clientRuntime.dispose()


async def test_fences_superseded_attachment_hydration_by_attachment_generation(remote, monkeypatch):
    _directory, runtime = await remote.makeServer()
    client = await remote.connect(runtime)
    errors: list[Exception] = []
    services = createSessionServiceBinding(client, {"services": [Models], "onError": errors.append})
    models = services.use(Models)
    await services.ready(BACKGROUND_CONTEXT)
    states: list[dict[str, Any]] = []
    removeStateListener = services.attachment.subscribe(lambda state, *_: states.append(state))
    delayed: asyncio.Future[None] = asyncio.get_running_loop().create_future()
    subscribeService = client.subscribeService

    async def delayedSubscribe(target: Any, serviceId: str, mode: str, listener: Any, signal: Any = None) -> Any:
        if target.get("sessionId") == "demo-2" and serviceId == Models.id:
            await delayed
        return await subscribeService(target, serviceId, mode, listener, signal)

    monkeypatch.setattr(client, "subscribeService", delayedSubscribe)

    try:
        await remote.attachSession(client, "demo-2")
        assert services.attachment.value == {"status": "attaching", "sessionId": "demo-2"}
        await remote.attachSession(client, "demo-1")
        await services.whenAttached("demo-1", BACKGROUND_CONTEXT)
        assert services.attachment.value == {"status": "attached", "sessionId": "demo-1"}
        assert models.state.value["configuration"]["model"] == {"provider": "anthropic", "modelId": "claude-sonnet-4-5"}
    finally:
        delayed.set_result(None)
        removeStateListener()
        await services.dispose(BACKGROUND_CONTEXT)

    attaching = [state == {"status": "attaching", "sessionId": "demo-1"} for state in states]
    assert True in attaching
    latestAttach = len(attaching) - 1 - attaching[::-1].index(True)
    assert {"status": "attached", "sessionId": "demo-2"} not in states[latestAttach:]
    assert {"status": "degraded", "sessionId": "demo-2"} not in states[latestAttach:]
    assert errors == []


async def test_observes_keyed_service_instances_and_fences_replacement_generations_over_framed_transport(
    remote, fauxWorker
):
    _directory, runtime = await remote.makeServer()
    client = await remote.attachClient(runtime, "demo-1")
    errors: list[Exception] = []
    services = createSessionServiceSource(client, {"onError": errors.append})
    assert {"serviceId": KeyedProbe.id, "mode": "keyed"} in await services.catalogue(BACKGROUND_CONTEXT)
    observed: list[SimpleNamespace] = []

    def valueOf(service: Any) -> Any:
        value = service.state.value
        return value["value"] if isinstance(value, dict) else None

    def setup(env: Any) -> None:
        env.observe(KeyedProbe, lambda service, *_: observed.append(SimpleNamespace(service=service, value=valueOf(service))))

    consumer = defineFacet(SimpleNamespace(id="@test/keyed-probe-consumer", setup=setup))
    facetHost = await createFacetHost({"facets": [consumer], "serviceSources": [services]})

    assert services.attachment.value == {"status": "attached", "sessionId": "demo-1"}
    await waitFor(lambda: len(observed) == 1)
    assert observed[0].value == "first"
    staleReplace = observed[0].service.replace
    assert await staleReplace("second", BACKGROUND_CONTEXT) is None
    await waitFor(lambda: len(observed) == 2)
    assert observed[1].value == "second"
    with pytest.raises(Exception, match="observation is closed"):
        staleReplace("late", BACKGROUND_CONTEXT)

    replacedReplace = observed[1].service.replace
    assert await remote.attachSession(client, "demo-2") is None
    await services.whenAttached("demo-2", BACKGROUND_CONTEXT)
    assert len(observed) == 3
    assert services.attachment.value == {"status": "attached", "sessionId": "demo-2"}
    assert observed[2].value == "first"
    with pytest.raises(Exception, match="observation is closed"):
        replacedReplace("late", BACKGROUND_CONTEXT)
    assert errors == []

    await facetHost.dispose()
    assert await services.dispose(BACKGROUND_CONTEXT) is None


async def test_answers_a_client_prompt_through_the_worker_owned_service_provider(remote, fauxWorker):
    directory, _runtime = await remote.makeServer()

    result = await runClient({"command": "client", "sessionId": "demo-1", "prompt": "question"}, {"directory": directory})

    assert result["kind"] == "prompted"
    assert result["text"] == "deterministic remote answer"


async def test_replicates_the_transcript_after_consecutive_prompts(remote, fauxWorker):
    _directory, runtime = await remote.makeServer()
    client = await remote.attachClient(runtime, "demo-1")
    services = createSessionServiceBinding(client, {"services": [AgentController, SessionPlugins, Transcript]})
    controller = services.use(AgentController)
    transcript = services.use(Transcript)
    await services.ready(BACKGROUND_CONTEXT)
    await services.use(SessionPlugins).reload(BACKGROUND_CONTEXT)
    try:
        for message in ("first question", "second question"):
            response = await controller.prompt({"message": message, "images": None}, BACKGROUND_CONTEXT)
            assert response["accepted"] is True
            assert isinstance(response["operationId"], str)
            assert await controller.waitForPrompt(response["operationId"], BACKGROUND_CONTEXT) == {
                "status": "done",
                "text": "deterministic remote answer",
                "reason": None,
            }
            expected = 1 if message == "first question" else 2
            await waitFor(
                lambda expected=expected: len(
                    [entry for entry in transcript.state.value["entries"] if entry["kind"] == "pi.assistant"]
                )
                == expected
            )
    finally:
        await services.dispose(BACKGROUND_CONTEXT)


async def test_stops_an_idle_session_worker_after_its_client_disconnects(remote):
    _directory, runtime = await remote.makeServer()
    client = await remote.attachClient(runtime, "demo-1")
    pid = runtime.workerPids.get("demo-1")
    assert isinstance(pid, int)

    await client.dispose()
    remote.clients.discard(client)
    await waitFor(lambda: "demo-1" not in runtime.workerPids, 10_000)
    await waitFor(lambda: not processExists(pid), 10_000)


async def test_starts_one_process_per_attached_session_and_stops_them_during_shutdown(remote):
    _directory, runtime = await remote.makeServer()
    await asyncio.gather(remote.attachClient(runtime, "demo-1"), remote.attachClient(runtime, "demo-2"))

    pids = list(runtime.workerPids.values())
    assert len(pids) == 2
    assert len(set(pids)) == 2
    for pid in pids:
        assert processExists(pid)

    await runtime.close()
    assert len(runtime.workerPids) == 0
    for pid in pids:
        await waitFor(lambda pid=pid: not processExists(pid), 10_000)


async def test_server_runtime_replaces_an_exited_worker_on_the_next_attach(remote):
    directory = remote.shortDirectory("pew-")
    runtime = await startServer({**sessionWorkerModel, "directory": directory})
    remote.servers.append(runtime)
    client = await remote.attachClient(runtime, "demo-1")
    firstPid = runtime.workerPids.get("demo-1")
    assert isinstance(firstPid, int)

    os.kill(firstPid, signal.SIGKILL)
    await waitFor(lambda: "demo-1" not in runtime.workerPids, 10_000)
    await remote.attachSession(client, "demo-1")
    replacementPid = runtime.workerPids.get("demo-1")
    assert isinstance(replacementPid, int)
    assert replacementPid != firstPid


async def test_discovers_workers_after_replacing_the_server(remote):
    firstDirectory = remote.shortDirectory("per-")
    first = await startServer({**sessionWorkerModel, "directory": firstDirectory})
    remote.servers.append(first)
    await remote.attachClient(first, "demo-1")
    firstWorkerPid = first.workerPids.get("demo-1")
    assert isinstance(firstWorkerPid, int)

    replacement = await startServer({**sessionWorkerModel, "directory": firstDirectory})
    remote.servers.append(replacement)
    await first.closed

    assert replacement.serverId == first.serverId
    assert replacement.workerPids.get("demo-1") == firstWorkerPid
    await waitFor(lambda: len(first.workerPids) == 0)
    assert processExists(firstWorkerPid)

    result = await runClient({"command": "client"}, {"directory": firstDirectory})
    assert result["kind"] == "list"
    assert result["sessions"] == [
        {"serverId": first.serverId, "sessionId": "demo-1"},
        {"serverId": first.serverId, "sessionId": "demo-2"},
    ]
    await remote.attachClient(replacement, "demo-1")
    assert replacement.workerPids.get("demo-1") == firstWorkerPid
    await remote.attachClient(replacement, "demo-2")
    assert isinstance(replacement.workerPids.get("demo-2"), int)
    assert replacement.workerPids.get("demo-2") != firstWorkerPid


async def test_retires_an_unclaimed_idle_worker_after_replacement_demand_expires(remote, monkeypatch):
    directory = remote.shortDirectory("pi-orphan-worker-")
    monkeypatch.setenv("__PI_SESSION_WORKER_ORPHAN_DEMAND_GRACE_MS", "50")
    first = await startServer({**sessionWorkerModel, "directory": directory})
    remote.servers.append(first)
    await remote.attachClient(first, "demo-1")
    workerPid = first.workerPids.get("demo-1")
    assert isinstance(workerPid, int)

    replacement = await startServer({**sessionWorkerModel, "directory": directory})
    remote.servers.append(replacement)
    await first.closed
    assert replacement.workerPids.get("demo-1") == workerPid

    await waitFor(lambda: "demo-1" not in replacement.workerPids, 10_000)
    await waitFor(lambda: not processExists(workerPid), 10_000)


async def test_restores_tracked_sessions_that_are_outside_the_replacement_catalog(remote):
    directory = remote.shortDirectory("pet-")
    emptySessionDir = remote.shortDirectory("pet-sessions-")
    first = await startServer({**sessionWorkerModel, "directory": directory})
    remote.servers.append(first)
    await remote.attachClient(first, "demo-1")
    workerPid = first.workerPids.get("demo-1")

    replacement = await startServer({**sessionWorkerModel, "directory": directory, "sessionDir": emptySessionDir})
    remote.servers.append(replacement)
    await first.closed

    assert await runClient({"command": "client"}, {"directory": directory}) == {
        "kind": "list",
        "sessions": [{"serverId": first.serverId, "sessionId": "demo-1"}],
    }
    await remote.attachClient(replacement, "demo-1")
    assert replacement.workerPids.get("demo-1") == workerPid


async def test_reports_missing_and_ambiguous_session_selections(remote):
    sharedDirectory = remote.shortDirectory("ped-")
    firstShared = await startServer(
        {**sessionWorkerModel, "directory": sharedDirectory, "serverId": "00000000-0000-4000-8000-000000000001"}
    )
    secondShared = await startServer(
        {**sessionWorkerModel, "directory": sharedDirectory, "serverId": "00000000-0000-4000-8000-000000000002"}
    )
    remote.servers.append(firstShared)
    remote.servers.append(secondShared)

    with pytest.raises(RuntimeError, match="No discovered server contains session missing"):
        await runClient({"command": "client", "sessionId": "missing"}, {"directory": sharedDirectory})
    with pytest.raises(RuntimeError, match="Session demo-1 is available from more than one server"):
        await runClient({"command": "client", "sessionId": "demo-1"}, {"directory": sharedDirectory})


async def test_rejects_a_duplicate_session_id_within_one_session_directory():
    with pytest.raises(RuntimeError, match="Session demo-1 already exists"):
        await createExperimentalSessions(
            str(home.path("experimental_sessions")), ["demo-1"], os.path.join(str(home.path("agent")), "other-cwd")
        )


async def test_closing_a_server_leaves_no_replaced_watcher_behind(remote):
    from misaka.experimental import server as serverModule

    before = len(serverModule._background)
    _directory, runtime = await remote.makeServer()
    await runtime.close()
    await waitFor(lambda: len(serverModule._background) == before)
