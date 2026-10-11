"""Translated from pi's ``coding-agent/src/experimental/client-tui.ts``.

``ExperimentalClientTui`` is the service-only presentation of the experimental client: a replicated main-lane snapshot drives
a chat view, and slash commands, the model selector and the plugin reload run through services. ``ClientTuiServer`` is a
``SimpleNamespace(serverId, radius, server, session)`` (the two service sources of ``services/connection.py``).
``ExperimentalClientTui.create(options)`` takes ``{"command", "ui", "servers", "facetLoader"?, "requestRender", "finish"}``.

MISAKA fork:

- pi's interactive mode has a fullscreen layout (``createChatViewport``, ``tuiMode: "fullscreen"``) and the experimental
  client runs in it; MISAKA has only the regular mode, in which the component renders itself, so there is no ``layoutRoot``
  and ``runClientTui`` adds the component to the TUI directly.
- MISAKA's ``InteractiveThemeController`` does not follow the terminal's colors (D11, see ``theme_controller.py``), so
  ``runClientTui`` has no ``disableAutoSync()``/``dispose()`` calls.
- A slash command argument completion is a ``{"value", "label", "description"?}`` mapping; the autocomplete provider takes
  ``AutocompleteItem`` objects.
"""

from __future__ import annotations

import asyncio
import inspect
import os
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

from misaka.chord import combineFacetLoaders, createFacetHost, defineFacet
from misaka.chord.context import BACKGROUND_CONTEXT
from misaka.config.engine import get_agent_dir
from misaka.core.keybindings import KeybindingsManager
from misaka.core.resource_loader import DefaultResourceLoader
from misaka.core.settings_manager import SettingsManager
from misaka.experimental.client_runtime import openClientRuntime
from misaka.experimental.client_tui_chat import ExperimentalChatView, liveOf
from misaka.experimental.plugins.bundled import createPresentationFacetLoaders
from misaka.experimental.services.agent_controller import AgentController
from misaka.experimental.services.plugins import PresentationPlugins
from misaka.experimental.services.presentation_ui import PresentationUI
from misaka.experimental.services.sessions import SessionDirectory, SessionManagement
from misaka.experimental.services.slash_commands import SlashCommands
from misaka.experimental.services.slash_commands_provider import (
    createBuiltInSlashCommandsFacet,
    createSlashCommandsRuntimeFacet,
)
from misaka.experimental.services.transcript import Transcript
from misaka.ui.tui import (
    CombinedAutocompleteProvider,
    Container,
    EditorOptions,
    SelectItem,
    SelectList,
    SelectListTheme,
    SlashCommand,
    Text,
    setKeybindings,
)
from misaka.ui.tui.autocomplete import AutocompleteItem
from misaka.ui.tui.interactive.components.custom_editor import CustomEditor
from misaka.ui.tui.interactive.theme.theme import (
    getEditorTheme,
    setRegisteredThemes,
    stopThemeWatcher,
    theme,
)
from misaka.ui.tui.interactive.theme.theme_controller import InteractiveThemeController
from misaka.ui.tui.interactive.tui_renderer import (
    InteractiveTuiOptions,
    createInteractiveTui,
)
from misaka.utils.collation import localeSortKey
from misaka.utils.js import (
    UNDEFINED,
    aggregateError,
    allSettled,
    errorMessage,
    jsTrim,
    pathResolve,
    promiseAll,
    spawn,
    startTask,
)

selectTheme = SelectListTheme(
    selectedPrefix=lambda text: theme.fg("accent", text),
    selectedText=lambda text: theme.fg("accent", text),
    description=lambda text: theme.fg("muted", text),
    scrollInfo=lambda text: theme.fg("dim", text),
    noMatch=lambda text: theme.fg("warning", text),
)


def message(error: BaseException) -> str:
    return errorMessage(error)


async def _awaited(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


def _valueOf(state: Any) -> Any:
    """A replicated state's value, or ``None`` before its first one."""
    value = state.value
    return None if value is UNDEFINED else value


class ExperimentalClientTui:
    """Service-only presentation driven by a replicated main-lane snapshot."""

    def __init__(self, ui: Any, requestRender: Callable[[], None], finish: Callable[[], None], loadedFacets: Any) -> None:
        self._ui = ui
        self._requestRender = requestRender
        self._finish = finish
        self._documentContainer = Container()
        self._sessionHeading = Text("", 1, 0)
        self._pendingMessagesContainer = Container()
        self._statusContainer = Container()
        self._editorContainer = Container()
        self._footerComponent = Text("", 1, 0)
        self._sharedFacets = loadedFacets
        self._keybindings = KeybindingsManager.create()
        self._presentationFacets: Any = None
        self._facetHost: Any = None
        self._facetReloadTail: asyncio.Future[None] = _resolved()
        self._session: SimpleNamespace | None = None
        self._slashCommands: Any = None
        self._controller: Any = None
        self._selectList: SelectList | None = None
        self._selection: dict[str, Any] | None = None
        self._screen = "chat"
        self._selectedServerId: str | None = None
        self._sessionId: str | None = None
        self._status = "Starting Session…"
        self._busy = False
        self._closed = False
        self._closePromise: asyncio.Future[None] | None = None
        self._recoveryTransition: asyncio.Future[None] = _resolved()
        self._laneUnsubscribe: Callable[[], None] | None = None
        self._chatView: ExperimentalChatView | None = None
        self._background: set[asyncio.Task[Any]] = set()
        setKeybindings(self._keybindings)
        self._chatInput = CustomEditor(ui, getEditorTheme(), self._keybindings, EditorOptions(paddingX=1))
        self._chatInput.onSubmit = lambda text: spawn(self._runPrompt(text), self._background)
        self._chatInput.onEscape = self._interrupt
        self._chatInput.onCtrlD = finish
        self._chatInput.onAction("app.clear", finish)
        self._chatInput.onAction("app.model.select", lambda: spawn(self._executeSlashCommand("model", ""), self._background))

        def followUp() -> None:
            text = jsTrim(self._chatInput.getText())
            if len(text) == 0:
                return
            self._chatInput.setText("")
            spawn(self._queueFollowUp(text), self._background)

        self._chatInput.onAction("app.message.followUp", followUp)
        self._editorContainer.addChild(self._chatInput)
        self._rebuild()

    @classmethod
    async def create(cls, options: dict[str, Any]) -> ExperimentalClientTui:
        prepared = await prepareClientSession(options["command"], options["servers"])
        facetLoader = options.get("facetLoader")
        loadedFacets = await combineFacetLoaders([] if facetLoader is None else [facetLoader]).load()
        component = cls(options["ui"], options["requestRender"], options["finish"], loadedFacets)
        try:
            await component._start(prepared)
            await component._openPreparedSession(prepared)
            return component
        except BaseException as error:
            try:
                await component.close()
            except Exception as cleanupError:  # noqa: BLE001 - reported together with the startup error
                raise aggregateError([error, cleanupError], "Experimental TUI startup and cleanup failed") from None
            raise

    def render(self, width: int) -> list[str]:
        return [
            *self._documentContainer.render(width),
            *self._pendingMessagesContainer.render(width),
            *self._statusContainer.render(width),
            *self._editorContainer.render(width),
            *self._footerComponent.render(width),
        ]

    def handleInput(self, data: str) -> None:
        if self._busy:
            if self._keybindings.matches(data, "app.clear") or (
                len(self._chatInput.getText()) == 0 and self._keybindings.matches(data, "app.exit")
            ):
                self._finish()
            return
        if self._screen == "chat":
            self._chatInput.handleInput(data)
            self._requestRender()
            return
        if self._selectList is not None:
            self._selectList.handleInput(data)

    def invalidate(self) -> None:
        for container in (
            self._documentContainer,
            self._pendingMessagesContainer,
            self._statusContainer,
            self._editorContainer,
            self._footerComponent,
        ):
            container.invalidate()

    def dispose(self) -> None:
        spawn(_quietly(self.close()), self._background)

    def refreshTheme(self) -> None:
        view = self._conversationView()
        if view is not None and self._chatView is not None:
            self._chatView.refreshTheme(view)
        self._rebuild()

    def showError(self, error: str) -> None:
        self._status = f"Error: {error}"
        self._rebuild()

    def close(self) -> asyncio.Future[None]:
        if self._closePromise is None:
            self._closePromise = startTask(self._close())
        return self._closePromise

    async def _start(self, prepared: SimpleNamespace) -> None:
        server = prepared.server
        presentationFacets = await combineFacetLoaders(createPresentationFacetLoaders(prepared.presentationPlugins)).load()
        self._presentationFacets = presentationFacets
        facetHost: Any = None

        def reloadPresentationPlugins(data: Any) -> asyncio.Future[None]:
            nonlocal presentationFacets
            previous = self._facetReloadTail

            async def run() -> None:
                nonlocal presentationFacets
                await previous
                candidate = await combineFacetLoaders(createPresentationFacetLoaders(data)).load()
                try:
                    await facetHost.reload(candidate.facets)
                except Exception as error:
                    try:
                        await candidate.dispose()
                    except Exception as cleanupError:  # noqa: BLE001 - reported together with the reload error
                        raise aggregateError([error, cleanupError], "TUI plugin reload and cleanup failed") from None
                    raise
                retired = presentationFacets
                presentationFacets = candidate
                self._presentationFacets = candidate
                await retired.dispose()

            operation = startTask(run())
            self._facetReloadTail = startTask(_quietly(operation))
            return operation

        def setup(env: Any) -> None:
            env.provide(
                PresentationUI,
                {
                    "select": lambda title, items, selectedValue=None, *_: self._select(
                        title, [dict(item) for item in items], selectedValue
                    ),
                    "showStatus": lambda status, *_: self._showStatus(status),
                },
            )
            commands = env.use(SlashCommands)
            controller = env.use(AgentController)
            transcript = env.use(Transcript)
            sessionFeature = SimpleNamespace(serverId=server.serverId, session=server.session, transcript=transcript)

            def activate() -> None:
                if self._session is not None or self._slashCommands is not None or self._controller is not None:
                    raise RuntimeError("Presentation services are already active")
                self._session = sessionFeature
                self._slashCommands = commands
                self._controller = controller

                def release() -> None:
                    if self._session is sessionFeature:
                        self._session = None
                    if self._slashCommands is commands:
                        self._slashCommands = None
                    if self._controller is controller:
                        self._controller = None

                env.own(release)
                env.own(commands.subscribe(lambda *_: self._updateAutocomplete()))
                if server.radius:
                    env.own(
                        server.server.connection.subscribe(
                            lambda state, *_: self._handleConnectionState(server.serverId, state)
                        )
                    )
                    env.own(
                        server.session.attachment.subscribe(lambda state, *_: self._handleAttachmentState(sessionFeature, state))
                    )

            env.onActivate(activate)

        presentationBridgeFacet = defineFacet(SimpleNamespace(id="@pi/presentation-bridge", setup=setup))
        facetHost = await createFacetHost(
            {
                "facets": [
                    createSlashCommandsRuntimeFacet(),
                    presentationBridgeFacet,
                    createBuiltInSlashCommandsFacet({"reloadPresentationPlugins": reloadPresentationPlugins}),
                    *self._sharedFacets.facets,
                    *presentationFacets.facets,
                ],
                "serviceSources": [server.server, server.session],
            }
        )
        self._facetHost = facetHost

    async def _openPreparedSession(self, prepared: SimpleNamespace) -> None:
        feature = self._session
        if feature is None:
            raise RuntimeError(f"No Session service is available for {prepared.server.serverId}")
        await feature.session.whenAttached(prepared.summary["sessionId"], BACKGROUND_CONTEXT)
        self._selectedServerId = feature.serverId
        self._sessionId = prepared.summary["sessionId"]
        self._updateAutocomplete()
        await self._openLane(feature)
        self._screen = "chat"
        self._status = ""
        self._rebuild()

    async def _close(self) -> None:
        self._closed = True
        self._completeSelection(None)
        errors: list[BaseException] = []
        try:
            await self._recoveryTransition
            await self._closeLane()
            await self._facetReloadTail
        except Exception as error:  # noqa: BLE001 - collected with the other disposal failures
            errors.append(error)
        if self._facetHost is not None:
            try:
                await self._facetHost.dispose()
            except Exception as error:  # noqa: BLE001 - collected with the other disposal failures
                errors.append(error)
            self._facetHost = None
        generations = [generation for generation in (self._presentationFacets, self._sharedFacets) if generation is not None]
        self._presentationFacets = None
        results = await allSettled([generation.dispose() for generation in generations])
        errors.extend(result.reason for result in results if result.status == "rejected" and result.reason is not None)
        if len(errors) == 1:
            raise errors[0]
        if len(errors) > 1:
            raise aggregateError(errors, "Failed to dispose experimental TUI facets")

    def _showStatus(self, status: str) -> None:
        self._status = status
        self._rebuild()

    def _rebuild(self) -> None:
        self._sessionHeading.setText(
            ""
            if self._sessionId is None or self._selectedServerId is None
            else theme.fg("dim", f"Server: {self._selectedServerId}\nSession: {self._sessionId}")
        )
        self._statusContainer.clear()
        if len(self._status) > 0:
            self._statusContainer.addChild(Text(theme.fg("dim", self._status), 1, 0))
        if self._chatView is not None:
            self._statusContainer.addChild(self._chatView.status)
        self._footerComponent.setText(theme.fg("dim", self._footer()))
        self._editorContainer.clear()
        if self._screen == "select" and self._selection is not None:
            self._chatInput.focused = False
            selector = Container()
            selector.addChild(Text(theme.bold(self._selection["title"]), 1, 1))
            items = [
                SelectItem(value=item["value"], label=item["label"], description=item.get("description"))
                for item in self._selection["items"]
            ]
            self._selectList = SelectList(items, min(max(len(items), 1), 12), selectTheme)
            selectedIndex = next(
                (index for index, item in enumerate(items) if item.value == self._selection.get("selectedValue")), -1
            )
            if selectedIndex >= 0:
                self._selectList.setSelectedIndex(selectedIndex)
            self._selectList.onSelect = lambda item: self._completeSelection(item.value)
            self._selectList.onCancel = lambda: self._completeSelection(None)
            selector.addChild(self._selectList)
            self._editorContainer.addChild(selector)
        else:
            self._selectList = None
            self._chatInput.focused = not self._busy
            self._editorContainer.addChild(self._chatInput)
        self.invalidate()
        self._requestRender()

    def _select(self, title: str, items: list[dict[str, Any]], selectedValue: str | None = None) -> asyncio.Future[str | None]:
        if self._selection is not None:
            raise RuntimeError("A slash command selector is already active")
        selected: asyncio.Future[str | None] = asyncio.get_running_loop().create_future()
        self._selection = {"title": title, "items": items, "selectedValue": selectedValue, "resolve": selected}
        self._screen = "select"
        self._rebuild()
        return selected

    def _completeSelection(self, value: str | None) -> None:
        selection = self._selection
        if selection is None:
            return
        self._selection = None
        self._screen = "chat"
        if not selection["resolve"].done():
            selection["resolve"].set_result(value)
        if not self._closed:
            self._rebuild()

    def _updateAutocomplete(self) -> None:
        selected = self._selectedSlashCommands()
        commands = [] if selected is None else selected.list()
        self._chatInput.setAutocompleteProvider(
            CombinedAutocompleteProvider([_slashCommand(command) for command in commands], os.getcwd())
        )
        self._requestRender()

    def _selectedSlashCommands(self) -> Any:
        return self._slashCommands

    def _handleConnectionState(self, serverId: str, state: dict[str, Any]) -> None:
        if self._closed or self._selectedServerId != serverId:
            return
        if state["status"] == "connected":
            if self._laneUnsubscribe is None:
                self._busy = True
                self._status = "Reattaching Session…"
                self._rebuild()
            return
        self._busy = True
        self._status = "Reconnecting to Radius…" if state["status"] == "connecting" else "Radius disconnected; retrying…"
        self._queueRecovery(self._closeLane)
        self._rebuild()

    def _handleAttachmentState(self, feature: SimpleNamespace, state: dict[str, Any]) -> None:
        if self._closed or self._selectedServerId != feature.serverId or self._sessionId is None:
            return
        if state["status"] == "attached" and state["sessionId"] == self._sessionId:

            async def reattach() -> None:
                if self._laneUnsubscribe is None:
                    await self._openLane(feature)
                self._busy = False
                self._status = ""
                self._rebuild()

            self._queueRecovery(reattach)
            return
        if state["status"] == "attaching" and state["sessionId"] == self._sessionId:
            self._busy = True
            self._status = "Reattaching Session…"
            self._rebuild()

    def _queueRecovery(self, operation: Callable[[], Any]) -> None:
        previous = self._recoveryTransition

        async def run() -> None:
            await previous
            try:
                if not self._closed:
                    await operation()
            except Exception as error:  # noqa: BLE001 - shown as the status
                if self._closed:
                    return
                self._busy = True
                self._status = f"Reconnect error: {message(error)}"
                self._rebuild()

        # ``promise.then(...)``: the step runs a turn after it is queued, never inline.
        self._recoveryTransition = asyncio.ensure_future(run())

    async def _openLane(self, feature: SimpleNamespace) -> None:
        await self._closeLane()
        view = ExperimentalChatView(self._ui, os.getcwd())
        self._chatView = view
        self._documentContainer.addChild(self._sessionHeading)
        self._documentContainer.addChild(view.transcript)
        self._pendingMessagesContainer.addChild(view.pendingMessages)

        def onTranscript(value: Any, *_: Any) -> None:
            view.apply(value)
            self._rebuild()

        self._laneUnsubscribe = feature.transcript.state.subscribe(onTranscript)
        if feature.transcript.state.value is UNDEFINED:
            await self._closeLane()
            raise RuntimeError("Transcript has no initialized view")

    async def _closeLane(self) -> None:
        if self._laneUnsubscribe is not None:
            self._laneUnsubscribe()
        self._laneUnsubscribe = None
        if self._chatView is not None:
            self._chatView.dispose()
        self._chatView = None
        self._documentContainer.clear()
        self._pendingMessagesContainer.clear()
        self._statusContainer.clear()

    async def _runPrompt(self, messageText: str) -> None:
        prompt = jsTrim(messageText)
        if len(prompt) == 0:
            return
        if prompt.startswith("/"):
            separator = prompt.find(" ")
            name = prompt[1:] if separator == -1 else prompt[1:separator]
            args = "" if separator == -1 else jsTrim(prompt[separator + 1 :])
            await self._executeSlashCommand(name, args)
            return
        self._chatInput.setText("")
        try:
            await self._submitPrompt(prompt)
        except Exception as error:  # noqa: BLE001 - shown as the status
            self._status = f"Error: {message(error)}"
            self._rebuild()

    async def _executeSlashCommand(self, name: str, args: str) -> None:
        selected = self._selectedSlashCommands()
        commands = [] if selected is None else selected.list()
        command = next((candidate for candidate in commands if candidate["name"] == name), None)
        self._chatInput.setText("")
        if command is None:
            self._status = f"Unknown slash command: /{name}"
            self._rebuild()
            return
        try:
            result = await _awaited(command["run"](args, BACKGROUND_CONTEXT))
            if result is not None:
                if "entryId" in result:
                    self._reportQueue(result)
                else:
                    self._reportOperation(result)
        except Exception as error:  # noqa: BLE001 - shown as the status
            self._status = f"Error: {message(error)}"
            self._rebuild()

    async def _submitPrompt(self, prompt: str) -> None:
        controller = self._selectedController()
        if controller is None:
            raise RuntimeError("No Session AgentController service is available")
        view = self._conversationView()
        running = view is not None and liveOf(view).get("run") is not None
        self._status = "Queueing steering message…" if running else "Running turn…"
        self._rebuild()
        if running:
            self._reportQueue(await controller.steer({"message": prompt, "images": None}, BACKGROUND_CONTEXT))
        else:
            self._reportOperation(await controller.prompt({"message": prompt, "images": None}, BACKGROUND_CONTEXT))

    async def _queueFollowUp(self, text: str) -> None:
        controller = self._selectedController()
        if controller is None:
            return
        try:
            self._status = "Queueing follow-up…"
            self._rebuild()
            self._reportQueue(await controller.followUp({"message": text, "images": None}, BACKGROUND_CONTEXT))
        except Exception as error:  # noqa: BLE001 - shown as the status
            self._status = f"Error: {message(error)}"
            self._rebuild()

    def _reportOperation(self, response: dict[str, Any]) -> None:
        self._status = "" if response["accepted"] else f"Operation rejected: {response['error']['message']}"
        self._rebuild()

    def _reportQueue(self, response: dict[str, Any]) -> None:
        self._status = (
            f"Queued {response['entryId']}." if response["accepted"] else f"Message rejected: {response['error']['message']}"
        )
        self._rebuild()

    def _interrupt(self) -> None:
        view = self._conversationView()
        controller = self._selectedController()
        if view is None or liveOf(view).get("run") is None or controller is None:
            return
        self._status = "Aborting…"
        self._rebuild()

        async def abort() -> None:
            try:
                await controller.abort(BACKGROUND_CONTEXT)
            except Exception as error:  # noqa: BLE001 - shown as the status
                self._status = f"Error: {message(error)}"
                self._rebuild()
                return
            if self._status != "Aborting…":
                return
            self._status = ""
            self._rebuild()

        spawn(abort(), self._background)

    def _selectedController(self) -> Any:
        return self._controller

    def _conversationView(self) -> dict[str, Any] | None:
        return None if self._session is None else _valueOf(self._session.transcript.state)

    def _footer(self) -> str:
        view = self._conversationView()
        if not view:
            return "/model · /thinking · /compact · /reload"
        agent = view["docs"].get("pi.agent") or {}
        agentModel = agent.get("model")
        model = "no model" if agentModel is None else f"{agentModel['provider']}/{agentModel['modelId']}"
        thinkingLevel = agent.get("thinkingLevel")
        return (
            f"{model} · thinking:{'off' if thinkingLevel is None else thinkingLevel} · {len(view['entries'])} entries"
            " · /model · /thinking · /compact · /reload"
        )


def _resolved() -> asyncio.Future[None]:
    future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
    future.set_result(None)
    return future


async def _quietly(awaitable: Any) -> None:
    try:
        await awaitable
    except Exception:  # noqa: BLE001, S110 - `.catch(() => {})`
        pass


def _slashCommand(command: Any) -> SlashCommand:
    getCompletions = command.get("getArgumentCompletions")

    async def getArgumentCompletions(prefix: str) -> list[AutocompleteItem] | None:
        items = await _awaited(getCompletions(prefix))
        if items is None:
            return None
        return [AutocompleteItem(value=item["value"], label=item["label"], description=item.get("description")) for item in items]

    return SlashCommand(
        name=command["name"],
        description=command.get("description"),
        argumentHint=command.get("argumentHint"),
        getArgumentCompletions=None if getCompletions is None else getArgumentCompletions,
    )


async def prepareClientSession(command: dict[str, Any], servers: list[SimpleNamespace]) -> SimpleNamespace:
    opened = [
        SimpleNamespace(
            server=server,
            services=server.server.open(
                {
                    "services": [SessionDirectory, SessionManagement, PresentationPlugins],
                    "assertAccess": lambda: None,
                    "onError": lambda error: None,
                }
            ),
        )
        for server in servers
    ]
    try:
        features = [
            SimpleNamespace(
                server=entry.server,
                directory=entry.services.use(SessionDirectory),
                management=entry.services.use(SessionManagement),
                plugins=entry.services.use(PresentationPlugins),
            )
            for entry in opened
        ]
        await promiseAll([entry.services.ready(BACKGROUND_CONTEXT) for entry in opened])

        def sessionsOf(feature: SimpleNamespace) -> list[dict[str, Any]]:
            value = _valueOf(feature.directory.state)
            return [] if value is None else value["sessions"]

        selected: SimpleNamespace | None = None
        sessionId = command.get("sessionId")
        if sessionId is not None:
            matches = [
                SimpleNamespace(server=feature.server, management=feature.management, plugins=feature.plugins, summary=summary)
                for feature in features
                for summary in sessionsOf(feature)
                if summary["sessionId"] == sessionId
            ]
            if len(matches) > 1:
                raise RuntimeError(f"Session {sessionId} is available from more than one server")
            selected = matches[0] if matches else None
            if selected is None:
                connect = command.get("connect")
                if connect is not None and connect["transport"] == "radius":
                    raise RuntimeError(f"Remote server does not contain Session {sessionId}")
                feature = requireSingleServer(features)
                selected = SimpleNamespace(
                    server=feature.server,
                    management=feature.management,
                    plugins=feature.plugins,
                    summary=await feature.management.create({"id": sessionId}, BACKGROUND_CONTEXT),
                )
        elif command.get("continue") is True or command.get("resume") is True:
            candidates = sorted(
                (
                    SimpleNamespace(
                        server=feature.server, management=feature.management, plugins=feature.plugins, summary=summary
                    )
                    for feature in features
                    for summary in sessionsOf(feature)
                ),
                key=lambda candidate: (
                    -candidate.summary["createdAt"],
                    localeSortKey(candidate.summary["serverId"]),
                    localeSortKey(candidate.summary["sessionId"]),
                ),
            )
            selected = candidates[0] if candidates else None
        if selected is None:
            feature = requireSingleServer(features)
            selected = SimpleNamespace(
                server=feature.server,
                management=feature.management,
                plugins=feature.plugins,
                summary=await feature.management.create({}, BACKGROUND_CONTEXT),
            )
        pluginPackages = command.get("pluginPackages")
        presentationPlugins = await selected.plugins.prepareSession(
            {
                "sessionId": selected.summary["sessionId"],
                "packagePaths": None if pluginPackages is None else [pathResolve(path) for path in pluginPackages],
            },
            BACKGROUND_CONTEXT,
        )
        await selected.management.attach(selected.summary["sessionId"], BACKGROUND_CONTEXT)
        await selected.server.session.whenAttached(selected.summary["sessionId"], BACKGROUND_CONTEXT)
        return SimpleNamespace(server=selected.server, summary=selected.summary, presentationPlugins=presentationPlugins)
    finally:
        await allSettled([entry.services.dispose(BACKGROUND_CONTEXT) for entry in opened])


async def runClientTui(command: dict[str, Any], options: dict[str, Any] | None = None) -> None:
    """Run the interactive experimental client. ``options`` is ``{"directory"?, "facetLoader"?}``."""
    options = options or {}
    cwd = os.getcwd()
    agentDir = get_agent_dir()
    settingsManager = SettingsManager.create(cwd, agentDir)
    resourceLoader = DefaultResourceLoader(
        {
            "cwd": cwd,
            "agentDir": agentDir,
            "settingsManager": settingsManager,
            "noExtensions": True,
            # MISAKA fork: no `noSkills: true`; this DefaultResourceLoader has no such option.
            "noPromptTemplates": True,
            "noContextFiles": True,
        }
    )
    await resourceLoader.reload()
    setRegisteredThemes(resourceLoader.getThemes()["themes"])
    runtime = await openClientRuntime(command, options)
    # MISAKA fork (D1): the main-screen renderer (pi opens the fullscreen one).
    tui = createInteractiveTui(
        InteractiveTuiOptions(showHardwareCursor=settingsManager.getShowHardwareCursor(), logDirectory=agentDir)
    )
    tui.setClearOnShrink(settingsManager.getClearOnShrink())
    component: ExperimentalClientTui | None = None
    tuiStarted = False
    themeController = InteractiveThemeController(
        tui,
        getSettingsManager=lambda: settingsManager,
        showError=lambda error: component.showError(error) if component is not None else None,
        onChanged=lambda: component.refreshTheme() if component is not None else None,
    )
    loop = asyncio.get_running_loop()
    finished: asyncio.Future[None] = loop.create_future()

    def finishNow() -> None:
        nonlocal tuiStarted
        if tuiStarted:
            tui.stop()
            tuiStarted = False
        if not finished.done():
            finished.set_result(None)

    def finish() -> None:
        loop.call_soon_threadsafe(finishNow)

    try:
        component = await ExperimentalClientTui.create(
            {
                "command": command,
                "ui": tui,
                "servers": [
                    SimpleNamespace(
                        serverId=server.route["serverId"],
                        radius=server.route["transport"] == "radius",
                        server=server.server,
                        session=server.session,
                    )
                    for server in runtime.servers
                ],
                "facetLoader": options.get("facetLoader"),
                "requestRender": lambda: tui.requestRender(),
                "finish": finish,
            }
        )
        tui.addChild(component)
        tui.setFocus(component)
        tuiStarted = True
        tui.start()
        themeController.applyFromSettings()
        await finished
    finally:
        stopThemeWatcher()
        if tuiStarted:
            tui.stop()
        if component is not None:
            await component.close()
        await runtime.dispose()


def requireSingleServer(features: list[SimpleNamespace]) -> SimpleNamespace:
    if len(features) != 1:
        raise RuntimeError("Starting a Session requires exactly one server")
    return features[0]

