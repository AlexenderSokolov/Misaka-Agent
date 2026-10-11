"""Translated from pi's ``coding-agent/src/experimental/durable/harness-setup.ts``."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

from misaka.chord import Context
from misaka.core.http_dispatcher import applyHttpProxySettings, configureHttpDispatcher
from misaka.core.model_resolver import findInitialModel, resolveCliModel
from misaka.durable import createRegistry
from misaka.durable.env.node import NodeExecutionEnv
from misaka.durable.tools import CodingTools
from misaka.experimental.durable.prompt import createPiPrompt


def configureHarnessHttp(settingsManager: Any) -> None:
    """pi's HTTP setup: the proxy and the idle timeout from the settings."""
    applyHttpProxySettings(settingsManager.getGlobalSettings().get("httpProxy"))
    configureHttpDispatcher(settingsManager.getHttpIdleTimeoutMs())


class _HarnessSettings(Mapping[str, Any]):
    """Harness settings read at every use from pi's settings as loaded at startup, like the getters of upstream's object."""

    def __init__(self, settingsManager: Any) -> None:
        self._settingsManager = settingsManager

    def _stream(self) -> dict[str, Any]:
        provider = self._settingsManager.getProviderRetrySettings()
        idle = self._settingsManager.getHttpIdleTimeoutMs()
        timeoutMs = provider.get("timeoutMs")
        stream: dict[str, Any] = {
            "timeoutMs": timeoutMs if timeoutMs is not None else (2147483647 if idle == 0 else idle),
            "maxRetryDelayMs": provider.get("maxRetryDelayMs"),
        }
        if provider.get("maxRetries") is not None:
            stream["maxRetries"] = provider["maxRetries"]
        return stream

    def __getitem__(self, key: str) -> Any:
        settings = self._settingsManager
        if key == "stream":
            return self._stream()
        if key == "compaction":
            return settings.getCompactionSettings()
        if key == "retry":
            return settings.getRetrySettings()
        if key == "steeringMode":
            return settings.getSteeringMode()
        if key == "followUpMode":
            return settings.getFollowUpMode()
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return iter(("stream", "compaction", "retry", "steeringMode", "followUpMode"))

    def __len__(self) -> int:
        return 5


def createHarnessSettings(settingsManager: Any) -> Mapping[str, Any]:
    """Harness settings read at every use from pi's settings as loaded at startup."""
    return _HarnessSettings(settingsManager)


def createCodingRegistry(settingsManager: Any, cwd: str) -> Any:
    """A registry with pi's coding tools and system prompt."""
    registry = createRegistry()
    registry.install(CodingTools)
    registry.install(createPiPrompt(settingsManager, cwd))
    return registry


class ExecutionEnvs:
    """One execution environment per directory, shared by every conversation in it."""

    def __init__(self, defaultCwd: str) -> None:
        self._defaultCwd = defaultCwd
        self._envs: dict[str, NodeExecutionEnv] = {}

    def env(self, target: dict[str, Any], *_: Any) -> NodeExecutionEnv:
        cwd = target.get("cwd", self._defaultCwd)
        env = self._envs.get(cwd)
        if env is None:
            env = NodeExecutionEnv({"cwd": cwd})
            self._envs[cwd] = env
        return env

    async def cleanup(self, context: Context) -> None:
        envs = list(self._envs.values())
        self._envs.clear()
        for env in envs:
            await env.cleanup(context)


async def findInitialAgentModel(settingsManager: Any, modelRuntime: Any, cli: dict[str, Any] | None = None) -> dict[str, Any]:
    """The model a new root conversation starts with: an explicit ``--provider``/``--model``, or pi's default resolution.
    ``cli`` is ``{"provider"?, "model"}``. Returns ``{"model"?: {"provider", "modelId"}, "thinkingLevel"?,
    "fallbackMessage"?}``."""
    if cli is not None:
        resolved = resolveCliModel({"cliProvider": cli.get("provider"), "cliModel": cli["model"], "modelRuntime": modelRuntime})
        if resolved.error is not None or resolved.model is None:
            raise RuntimeError(f"Could not resolve model: {resolved.error or cli['model']}")
        result: dict[str, Any] = {"model": {"provider": resolved.model.provider, "modelId": resolved.model.id}}
        if resolved.thinkingLevel is not None:
            result["thinkingLevel"] = resolved.thinkingLevel
        return result
    initial = await findInitialModel(
        {
            "scopedModels": [],
            "isContinuing": False,
            "defaultProvider": settingsManager.getDefaultProvider(),
            "defaultModelId": settingsManager.getDefaultModel(),
            "defaultThinkingLevel": settingsManager.getDefaultThinkingLevel(),
            "modelRuntime": modelRuntime,
        }
    )
    found: dict[str, Any] = {}
    if initial.model is not None:
        found["model"] = {"provider": initial.model.provider, "modelId": initial.model.id}
        found["thinkingLevel"] = initial.thinkingLevel
    if initial.fallbackMessage is not None:
        found["fallbackMessage"] = initial.fallbackMessage
    return found
