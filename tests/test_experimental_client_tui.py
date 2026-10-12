"""pi coding-agent's ``test/experimental-client-tui.test.ts``.

MISAKA fork: the reloadable plugin is a real package built by ``createServerPluginPackage`` (pi's test hand-writes a CommonJS
artifact), and the UI is the regular ``TUI`` (pi's is ``TuiMainScreen``).
"""

from __future__ import annotations

import asyncio
import os
import shutil
from types import SimpleNamespace
from typing import Any

import pytest

from misaka.ai.providers.faux import fauxAssistantMessage
from misaka.chord import (
    RemoteServiceProvider,
    createRemoteServiceBinding,
    replicatedState,
)
from misaka.chord.context import BACKGROUND_CONTEXT
from misaka.chord.services.loopback import createLoopbackServiceTransport
from misaka.experimental.client_tui import ExperimentalClientTui
from misaka.experimental.plugins.bundled import createPresentationFacetData
from misaka.experimental.plugins.package import createServerPluginPackage
from misaka.experimental.services.agent_controller import AgentController
from misaka.experimental.services.agent_controller_provider import createAgentController
from misaka.experimental.services.models import Models
from misaka.experimental.services.plugins import PresentationPlugins, SessionPlugins
from misaka.experimental.services.sessions import SessionDirectory, SessionManagement
from misaka.experimental.services.transcript import Transcript
from misaka.modes.interactive.theme.theme import initTheme
from misaka.ui.tui import ProcessTerminal, TuiMainScreen
from tests.chat_support import waitFor
from tests.experimental_durable_support import openFauxConversation
from tests.experimental_session_support import makeShortDirectory
from tests.experimental_support import flush

serverId = "00000000-0000-4000-8000-000000000001"

RELOADABLE_FACET_SOURCE = (
    "from types import SimpleNamespace\n"
    "\n"
    "from misaka.chord import defineFacet\n"
    "from misaka.experimental.services.models import Models\n"
    "\n"
    "\n"
    "def setup(env):\n"
    "    env.use(Models)\n"
    "\n"
    "\n"
    'default = defineFacet(SimpleNamespace(id="test-tui-facet", setup=setup))\n'
)


@pytest.fixture(autouse=True)
def theme():
    initTheme("dark")


def session(sessionId: str, createdAt: int) -> dict[str, Any]:
    return {"serverId": serverId, "sessionId": sessionId, "createdAt": createdAt}


def publishReplacement(state: Any, value: Any) -> None:
    state.replace(BACKGROUND_CONTEXT, value)


def writeReloadablePackage(directory: str) -> str:
    packagePath = os.path.join(directory, "test-tui-plugin")
    os.makedirs(os.path.join(packagePath, "src"))
    with open(os.path.join(packagePath, "pyproject.toml"), "w", encoding="utf-8", newline="\n") as file:
        file.write('[project]\nname = "test-tui-plugin"\nversion = "1.0.0"\n')
    with open(os.path.join(packagePath, "src", "tui.py"), "w", encoding="utf-8", newline="\n") as file:
        file.write(RELOADABLE_FACET_SOURCE)
    return packagePath


async def buildReloadData(directory: str) -> dict[str, Any]:
    packagePath = writeReloadablePackage(directory)
    return createPresentationFacetData(await createServerPluginPackage(directory, serverId, packagePath).build())


@pytest.mark.parametrize(
    ("kind", "command", "sessionId", "creates"),
    [
        ("new", {"command": "client"}, "two", 1),
        ("continued", {"command": "client", "continue": True}, "one", 0),
        ("plugin-selected", {"command": "client", "pluginPackages": ["./example-plugin"]}, "two", 1),
    ],
)
async def test_opens_a_session_directly_and_exercises_the_full_lifecycle_only_for_a_new_session(
    kind, command, sessionId, creates
):
    directoryState = replicatedState({"revision": 1, "sessions": [session("one", 1)]})
    attachment = replicatedState({"status": "detached"})
    connectionState = replicatedState({"status": "connected", "since": "now"})
    modelsState = replicatedState(
        {
            "catalog": {
                "revision": 1,
                "availableModels": [
                    {"provider": "test", "modelId": "one", "name": "Model One", "reasoning": False},
                    {"provider": "test", "modelId": "two", "name": "Model Two", "reasoning": True},
                ],
            },
            "configuration": {"model": {"provider": "test", "modelId": "one"}, "thinkingLevel": "off"},
            "refresh": {"status": "idle"},
        }
    )
    createCalls: list[Any] = []
    selectCalls: list[Any] = []
    selectThinkingCalls: list[Any] = []

    async def create(*args: Any) -> dict[str, Any]:
        createCalls.append(args)
        created = session("two", 2)

        def update(draft: Any) -> None:
            draft["revision"] = 2
            draft["sessions"].push(created)

        directoryState.change(BACKGROUND_CONTEXT, update)
        return created

    async def select(model: Any, *args: Any) -> None:
        selectCalls.append((model, args))

        def update(draft: Any) -> None:
            draft["configuration"]["model"] = model

        modelsState.change(BACKGROUND_CONTEXT, update)

    async def selectThinking(thinkingLevel: str, *args: Any) -> None:
        selectThinkingCalls.append((thinkingLevel, args))

        def update(draft: Any) -> None:
            draft["configuration"]["thinkingLevel"] = thinkingLevel

        modelsState.change(BACKGROUND_CONTEXT, update)

    promptFinished: asyncio.Future[None] = asyncio.get_running_loop().create_future()
    prompts: list[str] = []

    async def respond(context: Any, *_: Any) -> Any:
        last = context.messages[-1] if context.messages else None
        if last is not None and last.role == "user":
            prompts.append(last.content if isinstance(last.content, str) else str(last.content))
        await promptFinished
        return fauxAssistantMessage("remote answer")

    durable = await openFauxConversation([respond])
    transcriptState = await durable.conversation.viewState(BACKGROUND_CONTEXT)

    pluginDirectory = makeShortDirectory("pi-tui-plugin-")
    reloadData = await buildReloadData(pluginDirectory)
    prepareCalls: list[Any] = []
    reloadPresentationCalls: list[Any] = []
    reloadSessionCalls: list[Any] = []

    async def prepareSession(request: Any, *args: Any) -> Any:
        prepareCalls.append((request, args))
        return reloadData

    async def reloadPresentation(*args: Any) -> Any:
        reloadPresentationCalls.append(args)
        return reloadData

    async def reloadSession(*args: Any) -> None:
        reloadSessionCalls.append(args)

    serverProvider = RemoteServiceProvider([SessionDirectory, SessionManagement, PresentationPlugins])
    serverProvider.provide(SessionDirectory, {"state": directoryState})
    serverProvider.provide(PresentationPlugins, {"prepareSession": prepareSession, "reload": reloadPresentation})

    async def removeSession(*_: Any) -> None:
        return None

    async def attachSession(attachedSessionId: str, *_: Any) -> None:
        publishReplacement(attachment, {"status": "attaching", "sessionId": attachedSessionId})

    async def detachSession(*_: Any) -> None:
        publishReplacement(attachment, {"status": "detached"})

    serverProvider.provide(
        SessionManagement, {"create": create, "remove": removeSession, "attach": attachSession, "detach": detachSession}
    )
    sessionProvider = RemoteServiceProvider([Models, AgentController, SessionPlugins, Transcript])
    sessionProvider.provide(SessionPlugins, {"reload": reloadSession})

    async def cycleThinking(*_: Any) -> None:
        return None

    async def getThinkingLevels(*_: Any) -> list[str]:
        return ["off", "high"]

    async def refresh(*_: Any) -> None:
        return None

    sessionProvider.provide(
        Models,
        {
            "state": modelsState,
            "cycleThinking": cycleThinking,
            "getThinkingLevels": getThinkingLevels,
            "refresh": refresh,
            "select": select,
            "selectThinking": selectThinking,
        },
    )
    sessionProvider.provide(AgentController, createAgentController(durable.harness, durable.conversation))
    sessionProvider.provide(Transcript, {"state": transcriptState})

    serverNamespace = createRemoteServiceBinding(
        {
            "services": [SessionDirectory, SessionManagement, PresentationPlugins],
            "transport": createLoopbackServiceTransport(serverProvider),
            "bound": False,
        }
    )

    async def serverReady(*_: Any) -> None:
        await serverNamespace.rebind(True, BACKGROUND_CONTEXT)
        await serverNamespace.ready(BACKGROUND_CONTEXT)

    async def noDispose(*_: Any) -> None:
        return None

    async def serverCatalogue(*_: Any) -> Any:
        return serverProvider.catalogue

    serverServices = SimpleNamespace(
        acceptsUnavailableServices=False,
        connection=connectionState,
        catalogue=serverCatalogue,
        open=lambda *_: SimpleNamespace(
            use=serverNamespace.use, observe=serverNamespace.observe, ready=serverReady, dispose=noDispose
        ),
    )
    sessionNamespace = createRemoteServiceBinding(
        {
            "services": [Models, AgentController, SessionPlugins, Transcript],
            "transport": createLoopbackServiceTransport(sessionProvider),
            "bound": False,
        }
    )

    async def sessionCatalogue(*_: Any) -> list[Any]:
        return []

    async def whenAttached(attachedSessionId: str, *_: Any) -> None:
        await sessionNamespace.rebind(True, BACKGROUND_CONTEXT)
        await sessionNamespace.ready(BACKGROUND_CONTEXT)
        publishReplacement(attachment, {"status": "attached", "sessionId": attachedSessionId})

    async def whenDetached(*_: Any) -> None:
        await sessionNamespace.rebind(False, BACKGROUND_CONTEXT)
        publishReplacement(attachment, {"status": "detached"})

    sessionServices = SimpleNamespace(
        acceptsUnavailableServices=True,
        attachment=attachment,
        catalogue=sessionCatalogue,
        open=lambda *_: SimpleNamespace(
            use=sessionNamespace.use,
            observe=sessionNamespace.observe,
            ready=sessionNamespace.ready,
            dispose=noDispose,
        ),
        whenAttached=whenAttached,
        whenDetached=whenDetached,
    )
    server = SimpleNamespace(serverId=serverId, radius=True, server=serverServices, session=sessionServices)
    state = {"finished": False, "renders": 0}

    def requestRender() -> None:
        state["renders"] += 1

    def finish() -> None:
        state["finished"] = True

    ui = TuiMainScreen(ProcessTerminal())
    ui.stopped = True
    component = await ExperimentalClientTui.create(
        {"command": command, "ui": ui, "servers": [server], "requestRender": requestRender, "finish": finish}
    )

    def rendered() -> str:
        return "\n".join(component.render(80))

    try:
        assert len(createCalls) == creates
        assert len(prepareCalls) == 1
        request, args = prepareCalls[0]
        assert request == {
            "sessionId": sessionId,
            "packagePaths": [os.path.abspath(path) for path in command["pluginPackages"]]
            if "pluginPackages" in command
            else None,
        }
        assert len(args) == 1
        assert attachment.value == {"status": "attached", "sessionId": sessionId}
        assert selectCalls == []
        assert f"Server: {serverId}" in rendered()
        assert f"Session: {sessionId}" in rendered()
        assert "faux/faux-1" in rendered()
        assert "Experimental Sessions" not in rendered()
        assert "Experimental Models" not in rendered()

        # Startup selection is the only behavior specific to continue and plugin-selected Sessions.
        if kind != "new":
            return

        component.handleInput("hello")
        component.handleInput("\r")
        await waitFor(lambda: prompts == ["hello"])
        await waitFor(lambda: "Working..." in rendered())
        promptFinished.set_result(None)
        await waitFor(lambda: "remote answer" in rendered())
        assert "hello" in rendered()
        assert "Working..." not in rendered()

        component.handleInput("/reload")
        component.handleInput("\x1b")
        component.handleInput("\r")
        await waitFor(lambda: len(reloadPresentationCalls) == 1 and len(reloadSessionCalls) == 1)
        await waitFor(lambda: "Reloaded plugins." in rendered())

        publishReplacement(attachment, {"status": "detached"})
        publishReplacement(
            connectionState, {"status": "disconnected", "since": "later", "reason": "network lost", "retryAt": None}
        )
        await waitFor(lambda: "retrying" in rendered())
        # The recovery step that closes the lane runs a turn later and clears the status container, as in pi.
        await flush()
        assert "retrying" not in rendered()
        component.handleInput("\x03")
        assert state["finished"] is True
        state["finished"] = False
        component.handleInput("\x04")
        assert state["finished"] is True
        state["finished"] = False
        publishReplacement(connectionState, {"status": "connecting", "attempt": 1})
        publishReplacement(connectionState, {"status": "connected", "since": "reconnected"})
        publishReplacement(attachment, {"status": "attached", "sessionId": sessionId})
        await waitFor(lambda: "Reattaching" not in rendered())
        # MISAKA fork: pi's ``vi.waitFor`` settles a microtask later, after the queued recovery step ran; ``waitFor`` returns at
        # once when its condition already holds, so the loop is given its turns here.
        await flush()

        component.handleInput("/model")
        component.handleInput("\x1b")
        component.handleInput("\r")
        await waitFor(lambda: "Select model:" in rendered())
        component.handleInput("\x1b[B")
        component.handleInput("\r")
        await waitFor(lambda: len(selectCalls) == 1)
        assert selectCalls[0][0] == {"provider": "test", "modelId": "two"}
        assert modelsState.value["configuration"]["model"] == {"provider": "test", "modelId": "two"}
        await waitFor(lambda: "Select model:" not in rendered())

        component.handleInput("/thinking")
        component.handleInput("\x1b")
        component.handleInput("\r")
        await waitFor(lambda: "Select thinking level:" in rendered())
        component.handleInput("\x1b[B")
        component.handleInput("\r")
        await waitFor(lambda: len(selectThinkingCalls) == 1)
        assert selectThinkingCalls[0][0] == "high"
        assert modelsState.value["configuration"]["thinkingLevel"] == "high"

        component.handleInput("\x03")
        await waitFor(lambda: state["finished"])

        await component.close()
        rendersAfterClose = state["renders"]

        def removeSessions(draft: Any) -> None:
            draft["revision"] = 3
            draft["sessions"] = []

        directoryState.change(BACKGROUND_CONTEXT, removeSessions)
        publishReplacement(attachment, {"status": "detached"})

        def refreshing(draft: Any) -> None:
            draft["refresh"] = {"status": "refreshing"}

        modelsState.change(BACKGROUND_CONTEXT, refreshing)
        assert state["renders"] == rendersAfterClose
    finally:
        await component.close()
        await serverNamespace.dispose(BACKGROUND_CONTEXT)
        await sessionNamespace.dispose(BACKGROUND_CONTEXT)
        serverProvider.dispose()
        sessionProvider.dispose()
        transcriptState.dispose()
        await durable.close()
        shutil.rmtree(pluginDirectory, ignore_errors=True)
