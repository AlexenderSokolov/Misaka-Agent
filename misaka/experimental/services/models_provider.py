"""Translated from pi's ``coding-agent/src/experimental/services/models-provider.ts``.

``modelRuntime`` and ``settingsManager`` are optional (``None``); the agent document is a dict
(``{"model"?: {"provider", "modelId"}, "thinkingLevel"?}``) read from the replicated ``DocumentState``.
"""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

from misaka.ai.models import (
    ModelsRefreshOptions,
    clampThinkingLevel,
    getSupportedThinkingLevels,
)
from misaka.chord import Context, defineFacet
from misaka.chord.context import BACKGROUND_CONTEXT
from misaka.durable import AgentDoc
from misaka.experimental.services.models import Models
from misaka.utils.js import errorMessage


def createModelsService(
    conversation: Any,
    agent: Any,
    modelRuntime: Any,
    settingsManager: Any,
    createState: Callable[[dict[str, Any]], Any],
) -> SimpleNamespace:
    """The Models service over one conversation. ``agent`` is the conversation's replicated ``pi.agent`` document: its
    configuration follows every change, also those made by other clients. Returns ``SimpleNamespace(service,
    activate, syncConfiguration)``: ``activate(context)`` publishes the catalog, and ``syncConfiguration(context)``
    publishes the agent document's model and thinking level when they changed."""
    catalogRevision = 0

    def configurationOf(value: dict[str, Any] | None) -> dict[str, Any]:
        model = (value or {}).get("model")
        return {
            "model": None if model is None else {"provider": model["provider"], "modelId": model["modelId"]},
            "thinkingLevel": (value or {}).get("thinkingLevel") or "off",
        }

    state = createState(
        {"catalog": {"revision": 0, "availableModels": []}, "configuration": configurationOf(agent.value), "refresh": {"status": "idle"}}
    )

    def selectedModel() -> Any:
        ref = (agent.value or {}).get("model")
        return None if ref is None or modelRuntime is None else modelRuntime.getModel(ref["provider"], ref["modelId"])

    def readThinkingLevels() -> list[str]:
        selected = selectedModel()
        return ["off"] if selected is None else getSupportedThinkingLevels(selected)

    def readCatalog() -> dict[str, Any]:
        nonlocal catalogRevision
        selected = selectedModel()
        available = [] if modelRuntime is None else modelRuntime.getAvailableSnapshot()
        catalog = available if selected is None or includesModel(available, selected) else [*available, selected]
        catalogRevision += 1
        return {
            "revision": catalogRevision,
            "availableModels": [
                {"provider": model.provider, "modelId": model.id, "name": model.name, "reasoning": model.reasoning}
                for model in catalog
            ],
        }

    def currentThinkingLevel() -> str:
        return (agent.value or {}).get("thinkingLevel") or "off"

    async def cycleThinking(context: Context, *_: Any) -> None:
        levels = readThinkingLevels()
        current = currentThinkingLevel()
        index = levels.index(current) if current in levels else -1
        following = levels[(index + 1) % len(levels)] if len(levels) > 0 else "off"
        await conversation.configure({"thinkingLevel": following}, context)

    async def getThinkingLevels(*_: Any) -> list[str]:
        return readThinkingLevels()

    async def refresh(context: Context, *_: Any) -> None:
        def start(draft: Any) -> None:
            draft["refresh"] = {"status": "refreshing"}

        state.change(context, start)
        errors: dict[str, str] = {}
        if modelRuntime is not None:
            result = await modelRuntime.refresh(ModelsRefreshOptions(signal=context.abortSignal))
            errors = {id: errorMessage(error) for id, error in result.errors.items()}
        catalog = readCatalog()

        def finish(draft: Any) -> None:
            draft["catalog"] = catalog
            draft["refresh"] = {"status": "done"} if len(errors) == 0 else {"status": "warning", "errors": errors}

        state.change(context, finish)

    async def select(model: dict[str, str], context: Context, *_: Any) -> None:
        selected = None if modelRuntime is None else modelRuntime.getModel(model["provider"], model["modelId"])
        if selected is None:
            raise RuntimeError(f"Unknown model: {model['provider']}/{model['modelId']}")
        thinkingLevel = clampThinkingLevel(selected, currentThinkingLevel())
        await conversation.configure(
            {"model": {"provider": selected.provider, "modelId": selected.id}, "thinkingLevel": thinkingLevel}, context
        )
        if settingsManager is not None:
            settingsManager.setDefaultModelAndProvider(selected.provider, selected.id)
            await settingsManager.flush()

    async def selectThinking(level: str, context: Context, *_: Any) -> None:
        levels = readThinkingLevels()
        if level not in levels:
            raise RuntimeError(f"Thinking level {level} is unavailable; choose one of: {', '.join(levels)}")
        await conversation.configure({"thinkingLevel": level}, context)

    service = {
        "state": state,
        "cycleThinking": cycleThinking,
        "getThinkingLevels": getThinkingLevels,
        "refresh": refresh,
        "select": select,
        "selectThinking": selectThinking,
    }

    async def activate(context: Context) -> None:
        catalog = readCatalog()

        def update(draft: Any) -> None:
            draft["catalog"] = catalog
            draft["refresh"] = {"status": "idle"}

        state.change(context, update)

    def syncConfiguration(context: Context) -> None:
        following = configurationOf(agent.value)
        current = state.value["configuration"]
        currentModel, followingModel = current["model"] or {}, following["model"] or {}
        if (
            currentModel.get("provider") == followingModel.get("provider")
            and currentModel.get("modelId") == followingModel.get("modelId")
            and current["thinkingLevel"] == following["thinkingLevel"]
        ):
            return

        def update(draft: Any) -> None:
            draft["configuration"] = following

        state.change(context, update)

    return SimpleNamespace(service=service, activate=activate, syncConfiguration=syncConfiguration)


async def createModelsServiceFacet(options: dict[str, Any]) -> Any:
    """Acquire the conversation's agent document, then build the facet that owns it. ``options`` is ``{"harness",
    "conversation", "modelRuntime", "settingsManager"?, "context"}``."""
    agent = await options["harness"].documentState(AgentDoc, options["conversation"].id, options["context"])
    if agent is None:
        raise RuntimeError(f"Conversation {options['conversation'].id} has no agent document")

    def setup(env: Any) -> None:
        env.own(lambda: agent.dispose())
        runtime = createModelsService(
            options["conversation"], agent, options.get("modelRuntime"), options.get("settingsManager"), env.replicatedState
        )
        env.provide(Models, runtime.service)

        async def activate() -> None:
            await runtime.activate(BACKGROUND_CONTEXT)
            env.own(agent.subscribe(lambda _value, context, *_: runtime.syncConfiguration(context)))

        env.onActivate(activate)

    return defineFacet(SimpleNamespace(id="@pi/models", setup=setup))


def includesModel(models: Any, selected: Any) -> bool:
    return any(model.provider == selected.provider and model.id == selected.id for model in models)
