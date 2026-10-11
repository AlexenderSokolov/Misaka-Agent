"""Translated from pi's ``coding-agent/src/experimental/services/slash-commands-provider.ts``.

A slash command is a mapping (see ``slash_commands.py``); the registry keeps an immutable copy of each registered one.
"""

from __future__ import annotations

import builtins
import re
from collections.abc import Callable
from types import MappingProxyType, SimpleNamespace
from typing import Any

from misaka.chord import Context, defineFacet
from misaka.experimental.services.agent_controller import AgentController
from misaka.experimental.services.models import Models
from misaka.experimental.services.plugins import PresentationPlugins, SessionPlugins
from misaka.experimental.services.presentation_ui import PresentationUI
from misaka.experimental.services.slash_commands import SlashCommands
from misaka.utils.js import UNDEFINED

THINKING_DESCRIPTIONS: dict[str, str] = {
    "off": "No reasoning",
    "minimal": "Very brief reasoning",
    "low": "Light reasoning",
    "medium": "Moderate reasoning",
    "high": "Deep reasoning",
    "xhigh": "Extra-high reasoning",
    "max": "Maximum reasoning",
}

_COMMAND_NAME = re.compile(r"[a-z0-9][a-z0-9:-]*")


class _RegisteredSlashCommand:
    def __init__(self, command: Any) -> None:
        self.command = command
        self.closed = False


class SlashCommandRegistry:
    def __init__(self) -> None:
        self._commands: dict[str, list[_RegisteredSlashCommand]] = {}
        # A JavaScript ``Set``: insertion ordered, one entry per listener.
        self._listeners: dict[Callable[[list[Any]], None], None] = {}

    def register(self, command: dict[str, Any]) -> Callable[[], None]:
        self._validate(command)
        if command["name"] in self._commands:
            raise RuntimeError(f"Slash command /{command['name']} is already registered")
        return self._add(command)

    def replace(self, command: dict[str, Any]) -> Callable[[], None]:
        """Stage a same-name replacement while the previous facet generation retires."""
        self._validate(command)
        return self._add(command)

    def list(self) -> builtins.list[Any]:
        return [entries[0].command for entries in self._commands.values()]

    def subscribe(self, listener: Callable[[list[Any]], None]) -> Callable[[], None]:
        self._listeners[listener] = None
        listener(self.list())

        def unsubscribe() -> None:
            self._listeners.pop(listener, None)

        return unsubscribe

    def _add(self, command: dict[str, Any]) -> Callable[[], None]:
        entry = _RegisteredSlashCommand(MappingProxyType(dict(command)))
        entries = self._commands.setdefault(command["name"], [])
        entries.append(entry)
        if len(entries) == 1:
            self._publish()

        def close() -> None:
            if entry.closed:
                return
            entry.closed = True
            if entries[0] is not entry:
                return
            while len(entries) > 0 and entries[0].closed:
                entries.pop(0)
            if len(entries) == 0:
                del self._commands[command["name"]]
            self._publish()

        return close

    def _validate(self, command: dict[str, Any]) -> None:
        if _COMMAND_NAME.fullmatch(command["name"]) is None:
            raise TypeError(f"Invalid slash command name: {command['name']}")

    def _publish(self) -> None:
        commands = self.list()
        for listener in list(self._listeners):
            listener(commands)


def createSlashCommandsRuntimeFacet(registry: SlashCommandRegistry | None = None) -> Any:
    registry = registry if registry is not None else SlashCommandRegistry()

    def setup(env: Any) -> None:
        env.provide(SlashCommands, registry)

    return defineFacet(SimpleNamespace(id="@pi/slash-commands-runtime", setup=setup))


def createBuiltInSlashCommandsFacet(options: dict[str, Any]) -> Any:
    """``options`` is ``{"reloadPresentationPlugins": async function(data)}``."""

    def setup(env: Any) -> None:
        commands = env.use(SlashCommands)
        models = env.use(Models)
        controller = env.use(AgentController)
        ui = env.use(PresentationUI)
        presentationPlugins = env.use(PresentationPlugins)
        sessionPlugins = env.use(SessionPlugins)

        async def reload(_args: str, context: Context) -> None:
            ui.showStatus("Reloading plugins…", context)
            data = await presentationPlugins.reload(context)
            await sessionPlugins.reload(context)
            await options["reloadPresentationPlugins"](data)
            ui.showStatus("Reloaded plugins.", context)

        def activate() -> None:
            env.own(commands.replace(modelCommand(models, ui)))
            env.own(commands.replace(thinkingCommand(models, ui)))
            env.own(commands.replace(compactCommand(controller, ui)))
            env.own(commands.replace({"name": "reload", "description": "Reload server-selected plugins", "run": reload}))

        env.onActivate(activate)

    return defineFacet(SimpleNamespace(id="@pi/slash-commands-builtin", setup=setup))


def modelCommand(models: Any, ui: Any) -> dict[str, Any]:
    def getArgumentCompletions(prefix: str) -> list[dict[str, str]]:
        normalized = prefix.lower()
        state = models.state.value
        available = [] if state is UNDEFINED or state is None else state["catalog"]["availableModels"]
        return [
            {"value": f"{model['provider']}/{model['modelId']}", "label": model["modelId"], "description": model["provider"]}
            for model in available
            if normalized in f"{model['provider']}/{model['modelId']} {model['name']}".lower()
        ]

    async def run(args: str, context: Context) -> None:
        state = models.state.value
        if state is UNDEFINED or state is None:
            raise RuntimeError("Models service is not ready")
        selected = exactModel(state["catalog"]["availableModels"], args)
        if len(args) > 0 and selected is None:
            raise RuntimeError(f"Unknown model: {args}")
        if selected is None:
            configured = state["configuration"]["model"]
            value = await ui.select(
                "Select model:",
                [
                    {
                        "value": f"{model['provider']}/{model['modelId']}",
                        "label": f"{model['name']} (selected)"
                        if configured is not None
                        and configured["provider"] == model["provider"]
                        and configured["modelId"] == model["modelId"]
                        else model["name"],
                        "description": f"{model['provider']}/{model['modelId']}",
                    }
                    for model in state["catalog"]["availableModels"]
                ],
                None if configured is None else f"{configured['provider']}/{configured['modelId']}",
                context,
            )
            if value is None:
                return
            selected = exactModel(state["catalog"]["availableModels"], value)
            if selected is None:
                raise RuntimeError(f"Unknown model: {value}")
        await models.select({"provider": selected["provider"], "modelId": selected["modelId"]}, context)
        ui.showStatus(f"Selected {selected['provider']}/{selected['modelId']}.", context)
        return

    return {
        "name": "model",
        "description": "Select model",
        "argumentHint": "<provider/model>",
        "getArgumentCompletions": getArgumentCompletions,
        "run": run,
    }


def thinkingCommand(models: Any, ui: Any) -> dict[str, Any]:
    async def run(args: str, context: Context) -> None:
        levels = await models.getThinkingLevels(context)
        selected = next((level for level in levels if level == args.lower()), None)
        if len(args) > 0 and selected is None:
            raise RuntimeError(f'Unknown thinking level "{args}". Available levels: {", ".join(levels)}.')
        if selected is None:
            state = models.state.value
            current = None if state is UNDEFINED or state is None else state["configuration"]["thinkingLevel"]
            value = await ui.select(
                "Select thinking level:",
                [
                    {
                        "value": level,
                        "label": f"{level} (selected)" if current == level else level,
                        "description": THINKING_DESCRIPTIONS[level],
                    }
                    for level in levels
                ],
                current,
                context,
            )
            if value is None:
                return
            selected = next((level for level in levels if level == value), None)
            if selected is None:
                raise RuntimeError(f"Unknown thinking level: {value}")
        await models.selectThinking(selected, context)
        ui.showStatus(f"Thinking level: {selected}.", context)
        return

    return {"name": "thinking", "description": "Set thinking level", "argumentHint": "<level>", "run": run}


def compactCommand(controller: Any, ui: Any) -> dict[str, Any]:
    def run(args: str, context: Context) -> Any:
        ui.showStatus("Compacting…", context)
        return controller.compact({"customInstructions": None if len(args) == 0 else args}, context)

    return {
        "name": "compact",
        "description": "Manually compact the session context",
        "argumentHint": "<instructions>",
        "run": run,
    }


def exactModel(models: list[dict[str, Any]], query: str) -> dict[str, Any] | None:
    if len(query) == 0:
        return None
    normalized = query.lower()
    matches = [
        model
        for model in models
        if f"{model['provider']}/{model['modelId']}".lower() == normalized or model["modelId"].lower() == normalized
    ]
    return matches[0] if len(matches) == 1 else None
