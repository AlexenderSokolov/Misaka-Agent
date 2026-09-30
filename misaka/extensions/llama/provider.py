"""Native llama.cpp provider, including Pi timeline item 66 autoload semantics."""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from misaka.ai.auth.types import (
    ApiKeyAuth,
    ApiKeyCredential,
    AuthCheck,
    AuthResult,
    ModelAuth,
    ProviderAuth,
    SecretPrompt,
    TextPrompt,
)
from misaka.ai.models_runtime import ModelsPublication, RefreshModelsContext
from misaka.ai.models_store import ModelsStoreEntry
from misaka.ai.stream import stream, stream_simple
from misaka.ai.types import Model, ModelCost, OpenAICompletionsCompat
from misaka.utils.values import signal_aborted

from .client import (
    LlamaClient,
    LlamaModelInfo,
    LlamaServerProps,
    llamaInferenceUrl,
    normalizeLlamaServerUrl,
)

LLAMA_PROVIDER_ID = "llama.cpp"
DEFAULT_LLAMA_SERVER_URL = "http://127.0.0.1:8080"


def _credential_server_url(credential: ApiKeyCredential | None) -> str | None:
    value = (
        (credential.env or {}).get("LLAMA_BASE_URL") if credential is not None else None
    )
    return (
        normalizeLlamaServerUrl(value)
        if isinstance(value, str) and value.strip()
        else None
    )


async def _resolve_server_url(
    ctx: Any, credential: ApiKeyCredential | None
) -> str | None:
    configured = _credential_server_url(credential)
    if configured is None:
        ambient = await ctx.env("LLAMA_BASE_URL")
        configured = ambient.strip() if isinstance(ambient, str) else None
    return normalizeLlamaServerUrl(configured) if configured else None


def modelIsSelectable(model: LlamaModelInfo, routerAutoload: bool) -> bool:
    if model.status.value in {"loaded", "sleeping"}:
        return True
    return (
        routerAutoload
        and model.status.value == "unloaded"
        and not model.status.failed
        and model.source == "preset"
    )


async def _router_autoload_enabled(
    client: LlamaClient, catalog: Sequence[LlamaModelInfo], signal: Any
) -> bool:
    if not any(
        model.status.value == "unloaded" and model.source == "preset"
        for model in catalog
    ):
        return False
    try:
        return (await client.props(signal=signal)).models_autoload is True
    except Exception:  # noqa: BLE001 - old/erroring servers fail closed
        return False


def configuredContextWindow(model: LlamaModelInfo) -> int | None:
    args = model.status.args or []
    for index in range(len(args) - 1):
        flag = args[index]
        if flag not in {"--ctx-size", "-c", "-ctx"}:
            continue
        try:
            context_window = int(args[index + 1])
        except ValueError:
            continue
        if context_window > 0:
            return context_window
    return None


def contextWindowOf(model: LlamaModelInfo, cachedContextWindow: int | None = None) -> int:
    runtime_context_window = model.meta.n_ctx if model.meta is not None else None
    if runtime_context_window and runtime_context_window > 0:
        return runtime_context_window
    configured_context = configuredContextWindow(model)
    if configured_context:
        return configured_context
    if cachedContextWindow and cachedContextWindow > 0:
        return cachedContextWindow
    training_context_window = model.meta.n_ctx_train if model.meta is not None else None
    return training_context_window if training_context_window and training_context_window > 0 else 128_000


def _to_model(
    model: LlamaModelInfo,
    serverUrl: str,
    props: LlamaServerProps | None = None,
    cachedContextWindow: int | None = None,
) -> Model:
    context_window = contextWindowOf(model, cachedContextWindow)
    reasoning = props is not None and props.chat_template is not None and "enable_thinking" in props.chat_template
    modalities = model.architecture.input_modalities if model.architecture else None
    return Model(
        id=model.id,
        name=model.id,
        api="openai-completions",
        provider=LLAMA_PROVIDER_ID,
        baseUrl=llamaInferenceUrl(serverUrl),
        reasoning=reasoning,
        thinkingLevelMap=(
            {"off": "off", "minimal": None, "low": None, "medium": "medium", "high": None, "xhigh": None}
            if reasoning
            else None
        ),
        input=["text", "image"] if modalities and "image" in modalities else ["text"],
        cost=ModelCost(input=0, output=0, cacheRead=0, cacheWrite=0),
        contextWindow=context_window,
        maxTokens=context_window,
        compat=OpenAICompletionsCompat(
            supportsStore=False,
            supportsDeveloperRole=False,
            supportsReasoningEffort=False,
            supportsUsageInStreaming=True,
            supportsStrictMode=False,
            maxTokensField="max_tokens",
            thinkingFormat="qwen-chat-template" if reasoning else None,
        ),
    )


class _LlamaProvider:
    id = LLAMA_PROVIDER_ID
    name = "llama.cpp"
    baseUrl = llamaInferenceUrl(DEFAULT_LLAMA_SERVER_URL)

    def __init__(self) -> None:
        self._models: list[Model] = []

        async def login(interaction: Any) -> ApiKeyCredential:
            entered_url = await interaction.prompt(
                TextPrompt(
                    message="llama.cpp server URL",
                    placeholder=os.environ.get(
                        "LLAMA_BASE_URL", DEFAULT_LLAMA_SERVER_URL
                    ),
                )
            )
            server_url = normalizeLlamaServerUrl(
                entered_url.strip()
                or os.environ.get("LLAMA_BASE_URL")
                or DEFAULT_LLAMA_SERVER_URL
            )
            api_key = (
                await interaction.prompt(SecretPrompt(message="API key (optional)"))
            ).strip()
            await LlamaClient(server_url, api_key or None).list(
                signal=interaction.signal
            )
            return ApiKeyCredential(
                key=api_key or None, env={"LLAMA_BASE_URL": server_url}
            )

        async def check(
            *, ctx: Any, credential: ApiKeyCredential | None, signal: Any
        ) -> AuthCheck | None:
            server_url = await _resolve_server_url(ctx, credential)
            if not server_url:
                return None
            return AuthCheck(
                type="api_key",
                source="stored credential" if credential else "LLAMA_BASE_URL",
            )

        async def resolve(
            *, ctx: Any, credential: ApiKeyCredential | None, signal: Any
        ) -> AuthResult | None:
            server_url = await _resolve_server_url(ctx, credential)
            if not server_url:
                return None
            api_key = credential.key if credential is not None else None
            if api_key is None:
                api_key = await ctx.env("LLAMA_API_KEY")
            if api_key is None:
                api_key = "local"
            return AuthResult(
                auth=ModelAuth(apiKey=api_key, baseUrl=llamaInferenceUrl(server_url)),
                env={
                    **((credential.env or {}) if credential is not None else {}),
                    "LLAMA_BASE_URL": server_url,
                },
                source="stored credential" if credential else "LLAMA_BASE_URL",
            )

        self.auth = ProviderAuth(
            apiKey=ApiKeyAuth(
                name="llama.cpp server", login=login, check=check, resolve=resolve
            )
        )

    def getModels(self) -> list[Model]:
        return list(self._models)

    def setCatalog(
        self,
        catalog: Sequence[LlamaModelInfo],
        serverUrl: str,
        *,
        routerAutoload: bool = False,
    ) -> None:
        self._models = [
            _to_model(model, serverUrl)
            for model in catalog
            if modelIsSelectable(model, routerAutoload)
        ]

    async def refreshModels(self, context: RefreshModelsContext) -> None:
        cached_context_windows: dict[str, int] = {}
        if context.stored is not None:
            restored = [
                model
                for model in context.stored.models
                if model.provider == LLAMA_PROVIDER_ID
                and model.api == "openai-completions"
            ]
            for model in restored:
                cached_context_windows[model.id] = model.contextWindow

            def apply_restored() -> None:
                self._models = restored

            if not await context.publish(ModelsPublication(update=apply_restored)):
                return

        if (
            not context.allowNetwork
            or signal_aborted(context.signal)
            or not isinstance(context.credential, ApiKeyCredential)
        ):
            return
        server_url = _credential_server_url(context.credential)
        if server_url is None:
            return
        client = LlamaClient(server_url, context.credential.key)
        catalog = await client.list(signal=context.signal)
        if signal_aborted(context.signal):
            return
        router_autoload = await _router_autoload_enabled(
            client, catalog, context.signal
        )
        if signal_aborted(context.signal):
            return
        selectable = [model for model in catalog if modelIsSelectable(model, router_autoload)]

        async def to_model(model: LlamaModelInfo) -> Model:
            cached_context_window = cached_context_windows.get(model.id)
            # Only loaded models expose their template without side effects. Unloaded autoload presets
            # would need to be loaded, while querying sleeping models may wake them. Those models remain
            # unclassified until they are loaded or woken and a later catalog refresh discovers them.
            if model.status.value != "loaded":
                return _to_model(model, server_url, None, cached_context_window)
            props = await client.props(model=model.id, signal=context.signal)
            return _to_model(model, server_url, props, cached_context_window)

        refreshed = list(await asyncio.gather(*(to_model(model) for model in selectable)))
        if signal_aborted(context.signal):
            return

        def apply_refreshed() -> None:
            self._models = refreshed

        await context.publish(
            ModelsPublication(
                persist=ModelsStoreEntry(
                    models=refreshed, checkedAt=int(time.time() * 1000)
                ),
                persist_is_set=True,
                update=apply_refreshed,
            )
        )

    def stream(self, model: Model, context: Any, options: Any = None):
        return stream(model, context, options)

    def streamSimple(self, model: Model, context: Any, options: Any = None):
        return stream_simple(model, context, options)


@dataclass(slots=True)
class LlamaProviderController:
    provider: _LlamaProvider

    def setCatalog(
        self,
        models: Sequence[LlamaModelInfo],
        serverUrl: str,
        *,
        routerAutoload: bool = False,
    ) -> None:
        self.provider.setCatalog(models, serverUrl, routerAutoload=routerAutoload)


def createLlamaProvider() -> LlamaProviderController:
    return LlamaProviderController(provider=_LlamaProvider())


__all__ = [
    "DEFAULT_LLAMA_SERVER_URL",
    "LLAMA_PROVIDER_ID",
    "LlamaProviderController",
    "createLlamaProvider",
    "modelIsSelectable",
]
