"""Api-dispatch ``stream()``/``complete()`` with env API key injection (pi ``compat.ts``)."""

from __future__ import annotations

from functools import cache
from typing import Any

from misaka.ai.api_registry import get_api_provider
from misaka.ai.env_api_keys import get_env_api_key
from misaka.ai.providers import register_builtins as _register_builtins
from misaka.ai.types import (
    AssistantMessage,
    Context,
    Model,
    ProviderStreamOptions,
    SimpleStreamOptions,
    StreamOptions,
)
from misaka.ai.utils.event_stream import AssistantMessageEventStream
from misaka.ai.utils.transcript import normalize_context

AMBIENT_AUTH_MARKER = "<authenticated>"


@cache
def _compat_models() -> Any:
    # MISAKA fork: pi builds this at import (`const compatModels = builtinModels()`). The
    # provider definitions import most of the ai package, which imports this module first,
    # so here it is built on first use.
    from misaka.ai.provider_definitions import builtinModels

    return builtinModels()


def _option(options: Any, name: str) -> Any:
    if options is None:
        return None
    if isinstance(options, dict):
        return options.get(name)
    return getattr(options, name, None)


def _has_explicit_api_key(api_key: Any) -> bool:
    return isinstance(api_key, str) and len(api_key.strip()) > 0


def _with_env_api_key(model: Model, options: Any, default_type: type) -> Any:
    if _has_explicit_api_key(_option(options, "apiKey")):
        return options
    api_key = get_env_api_key(model.provider, _option(options, "env"))
    if not api_key or api_key == AMBIENT_AUTH_MARKER:
        return options
    if options is None:
        return default_type(apiKey=api_key)
    if isinstance(options, dict):
        return {**options, "apiKey": api_key}
    return options.model_copy(update={"apiKey": api_key})


def _has_resolved_cloudflare_auth(options: Any) -> bool:
    headers = _option(options, "headers")
    return _has_explicit_api_key(_option(options, "apiKey")) or (
        isinstance(headers, dict) and isinstance(headers.get("cf-aig-authorization"), str)
    )


def _get_builtin_provider_for_model(model: Model) -> Any | None:
    if get_api_provider(model.api) is not _register_builtins.get_builtin_api_provider_instance(model.api):
        return None
    provider = _compat_models().getProvider(model.provider)
    return provider if provider is not None and any(
        candidate.api == model.api for candidate in provider.getModels()
    ) else None


def _resolve_api_provider(api: str):
    provider = get_api_provider(api)
    if provider is None:
        raise RuntimeError(f"No API provider registered for api: {api}")
    return provider


def stream(model: Model, context: Context, options: ProviderStreamOptions | None = None) -> AssistantMessageEventStream:
    transcript = normalize_context(context)
    builtin_provider = _get_builtin_provider_for_model(model)
    if builtin_provider is not None:
        if model.provider.startswith("cloudflare-") and not _has_resolved_cloudflare_auth(options):
            return _compat_models().stream(model, transcript, options)
        return builtin_provider.stream(model, transcript, _with_env_api_key(model, options, ProviderStreamOptions))
    provider = _resolve_api_provider(model.api)
    return provider.stream(model, transcript, _with_env_api_key(model, options, StreamOptions))


async def complete(model: Model, context: Context, options: ProviderStreamOptions | None = None) -> AssistantMessage:
    return await stream(model, context, options).result()


def stream_simple(
    model: Model,
    context: Context,
    options: SimpleStreamOptions | None = None,
) -> AssistantMessageEventStream:
    transcript = normalize_context(context)
    builtin_provider = _get_builtin_provider_for_model(model)
    if builtin_provider is not None:
        if model.provider.startswith("cloudflare-") and not _has_resolved_cloudflare_auth(options):
            return _compat_models().streamSimple(model, transcript, options)
        return builtin_provider.streamSimple(model, transcript, _with_env_api_key(model, options, SimpleStreamOptions))
    provider = _resolve_api_provider(model.api)
    return provider.streamSimple(model, transcript, _with_env_api_key(model, options, SimpleStreamOptions))


async def complete_simple(
    model: Model,
    context: Context,
    options: SimpleStreamOptions | None = None,
) -> AssistantMessage:
    return await stream_simple(model, context, options).result()


streamSimple = stream_simple
completeSimple = complete_simple
__all__ = [
    "complete",
    "completeSimple",
    "complete_simple",
    "stream",
    "streamSimple",
    "stream_simple",
]
