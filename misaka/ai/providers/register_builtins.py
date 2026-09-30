"""Lazy registration for built-in AI providers."""

from __future__ import annotations

import importlib
from collections.abc import AsyncIterable, Awaitable, Callable
from dataclasses import dataclass
from functools import cache
from typing import Any

from misaka.ai.api_lazy import lazy_api
from misaka.ai.api_registry import (
    ApiProvider,
    clear_api_providers,
    get_api_provider,
    register_api_provider,
)
from misaka.ai.types import (
    Context,
    Model,
    SimpleStreamOptions,
    StreamOptions,
)

ProviderStreamCallable = Callable[[Model, Context, StreamOptions | None], AsyncIterable[Any]]
ProviderSimpleStreamCallable = Callable[[Model, Context, SimpleStreamOptions | None], AsyncIterable[Any]]


@dataclass(slots=True)
class LazyProviderModule:
    stream: ProviderStreamCallable
    streamSimple: ProviderSimpleStreamCallable


_bedrock_provider_module_override: LazyProviderModule | None = None


def _error_message(error: Exception) -> str:
    message = getattr(error, "message", None)
    return message if isinstance(message, str) else str(error)


def _create_lazy_stream(load_module: Callable[[], Awaitable[LazyProviderModule]]) -> ProviderStreamCallable:
    """The provider's ``stream``, deferred behind its module import."""
    return lazy_api(load_module).stream


def _create_lazy_simple_stream(load_module: Callable[[], Awaitable[LazyProviderModule]]) -> ProviderSimpleStreamCallable:
    """The provider's ``streamSimple``, deferred behind its module import."""
    return lazy_api(load_module).streamSimple


async def _load_provider_module(
    cache_key: str,
    module_name: str,
    stream_name: str,
    stream_simple_name: str,
) -> LazyProviderModule:
    return _import_provider_module(cache_key, module_name, stream_name, stream_simple_name)


@cache
def _import_provider_module(
    cache_key: str, module_name: str, stream_name: str, stream_simple_name: str,
) -> LazyProviderModule:
    # importlib owns the import lock; cached values work across auxiliary worker loops.
    module = importlib.import_module(module_name)
    return LazyProviderModule(
        stream=getattr(module, stream_name), streamSimple=getattr(module, stream_simple_name),
    )


async def _load_anthropic_provider_module() -> LazyProviderModule:
    return await _load_provider_module(
        "anthropic-messages",
        "misaka.ai.providers.anthropic",
        "streamAnthropic",
        "streamSimpleAnthropic",
    )


async def _load_azure_openai_responses_provider_module() -> LazyProviderModule:
    return await _load_provider_module(
        "azure-openai-responses",
        "misaka.ai.providers.azure_openai_responses",
        "streamAzureOpenAIResponses",
        "streamSimpleAzureOpenAIResponses",
    )


async def _load_google_provider_module() -> LazyProviderModule:
    return await _load_provider_module(
        "google-generative-ai",
        "misaka.ai.providers.google",
        "streamGoogle",
        "streamSimpleGoogle",
    )


async def _load_google_vertex_provider_module() -> LazyProviderModule:
    return await _load_provider_module(
        "google-vertex",
        "misaka.ai.providers.google_vertex",
        "streamGoogleVertex",
        "streamSimpleGoogleVertex",
    )


async def _load_pi_messages_provider_module() -> LazyProviderModule:
    return await _load_provider_module(
        "pi-messages",
        "misaka.ai.providers.pi_messages",
        "stream_pi_messages",
        "stream_simple_pi_messages",
    )


async def _load_mistral_provider_module() -> LazyProviderModule:
    return await _load_provider_module(
        "mistral-conversations",
        "misaka.ai.providers.mistral",
        "streamMistral",
        "streamSimpleMistral",
    )


async def _load_openai_codex_responses_provider_module() -> LazyProviderModule:
    return await _load_provider_module(
        "openai-codex-responses",
        "misaka.ai.providers.openai_codex_responses",
        "streamOpenAICodexResponses",
        "streamSimpleOpenAICodexResponses",
    )


async def _load_openai_completions_provider_module() -> LazyProviderModule:
    return await _load_provider_module(
        "openai-completions",
        "misaka.ai.providers.openai_completions",
        "streamOpenAICompletions",
        "streamSimpleOpenAICompletions",
    )


async def _load_openai_responses_provider_module() -> LazyProviderModule:
    return await _load_provider_module(
        "openai-responses",
        "misaka.ai.providers.openai_responses",
        "streamOpenAIResponses",
        "streamSimpleOpenAIResponses",
    )


def set_bedrock_provider_module(module: LazyProviderModule | None) -> None:
    """Replace the dynamically imported bedrock implementation.

    Upstream exports this for its standalone-binary build, where the variable-specifier
    import cannot be bundled. Here it is the only writer of the override the loader below
    reads -- without it that branch was unreachable and the module-level variable dead.
    """
    global _bedrock_provider_module_override
    _bedrock_provider_module_override = module


async def _load_bedrock_provider_module() -> LazyProviderModule:
    if _bedrock_provider_module_override is not None:
        return _bedrock_provider_module_override
    return await _load_provider_module(
        "bedrock-converse-stream",
        "misaka.ai.providers.amazon_bedrock",
        "streamBedrock",
        "streamSimpleBedrock",
    )


stream_anthropic = _create_lazy_stream(_load_anthropic_provider_module)
stream_simple_anthropic = _create_lazy_simple_stream(_load_anthropic_provider_module)
stream_azure_openai_responses = _create_lazy_stream(_load_azure_openai_responses_provider_module)
stream_simple_azure_openai_responses = _create_lazy_simple_stream(_load_azure_openai_responses_provider_module)
stream_google = _create_lazy_stream(_load_google_provider_module)
stream_simple_google = _create_lazy_simple_stream(_load_google_provider_module)
stream_google_vertex = _create_lazy_stream(_load_google_vertex_provider_module)
stream_simple_google_vertex = _create_lazy_simple_stream(_load_google_vertex_provider_module)
stream_pi_messages = _create_lazy_stream(_load_pi_messages_provider_module)
stream_simple_pi_messages = _create_lazy_simple_stream(_load_pi_messages_provider_module)
stream_mistral = _create_lazy_stream(_load_mistral_provider_module)
stream_simple_mistral = _create_lazy_simple_stream(_load_mistral_provider_module)
stream_openai_codex_responses = _create_lazy_stream(_load_openai_codex_responses_provider_module)
stream_simple_openai_codex_responses = _create_lazy_simple_stream(_load_openai_codex_responses_provider_module)
stream_openai_completions = _create_lazy_stream(_load_openai_completions_provider_module)
stream_simple_openai_completions = _create_lazy_simple_stream(_load_openai_completions_provider_module)
stream_openai_responses = _create_lazy_stream(_load_openai_responses_provider_module)
stream_simple_openai_responses = _create_lazy_simple_stream(_load_openai_responses_provider_module)
_stream_bedrock_lazy = _create_lazy_stream(_load_bedrock_provider_module)
_stream_simple_bedrock_lazy = _create_lazy_simple_stream(_load_bedrock_provider_module)


_BUILTIN_APIS = (
    ("anthropic-messages", stream_anthropic, stream_simple_anthropic),
    ("openai-completions", stream_openai_completions, stream_simple_openai_completions),
    ("openai-responses", stream_openai_responses, stream_simple_openai_responses),
    ("openai-codex-responses", stream_openai_codex_responses, stream_simple_openai_codex_responses),
    ("azure-openai-responses", stream_azure_openai_responses, stream_simple_azure_openai_responses),
    ("google-generative-ai", stream_google, stream_simple_google),
    ("google-vertex", stream_google_vertex, stream_simple_google_vertex),
    ("mistral-conversations", stream_mistral, stream_simple_mistral),
    ("bedrock-converse-stream", _stream_bedrock_lazy, _stream_simple_bedrock_lazy),
    # pi's own wire protocol rather than a vendor's: the whole context goes out as-is and
    # comes back as pi's event stream. The Radius gateway speaks it, and so does anything a
    # models.json custom provider points at with `"api": "pi-messages"`.
    ("pi-messages", stream_pi_messages, stream_simple_pi_messages),
)

_builtin_api_provider_instances: dict[str, ApiProvider | None] = {}


def register_built_in_api_providers() -> None:
    """Registers the builtin API implementations into the api-registry without
    clobbering existing entries: this module may load after a test or extension has
    already registered an override for a builtin api id.
    """
    for api, stream, stream_simple in _BUILTIN_APIS:
        if get_api_provider(api) is None:
            register_api_provider(ApiProvider(api=api, stream=stream, streamSimple=stream_simple))
        _builtin_api_provider_instances[api] = get_api_provider(api)


def get_builtin_api_provider_instance(api: str) -> ApiProvider | None:
    """The registry entry the builtin registration left for ``api``; the stream facade only
    routes through the builtin provider definitions while that entry is still in place."""
    return _builtin_api_provider_instances.get(api)


def reset_api_providers() -> None:
    clear_api_providers()
    _builtin_api_provider_instances.clear()
    register_built_in_api_providers()


register_built_in_api_providers()

streamAnthropic = stream_anthropic
streamSimpleAnthropic = stream_simple_anthropic
streamAzureOpenAIResponses = stream_azure_openai_responses
streamSimpleAzureOpenAIResponses = stream_simple_azure_openai_responses
streamGoogle = stream_google
streamSimpleGoogle = stream_simple_google
streamGoogleVertex = stream_google_vertex
streamSimpleGoogleVertex = stream_simple_google_vertex
streamMistral = stream_mistral
streamSimpleMistral = stream_simple_mistral
streamOpenAICodexResponses = stream_openai_codex_responses
streamSimpleOpenAICodexResponses = stream_simple_openai_codex_responses
streamOpenAICompletions = stream_openai_completions
streamSimpleOpenAICompletions = stream_simple_openai_completions
streamOpenAIResponses = stream_openai_responses
streamSimpleOpenAIResponses = stream_simple_openai_responses

__all__ = [
    "streamAnthropic",
    "streamAzureOpenAIResponses",
    "streamGoogle",
    "streamGoogleVertex",
    "streamMistral",
    "streamOpenAICodexResponses",
    "streamOpenAICompletions",
    "streamOpenAIResponses",
    "streamSimpleAnthropic",
    "streamSimpleAzureOpenAIResponses",
    "streamSimpleGoogle",
    "streamSimpleGoogleVertex",
    "streamSimpleMistral",
    "streamSimpleOpenAICodexResponses",
    "streamSimpleOpenAICompletions",
    "streamSimpleOpenAIResponses",
]
