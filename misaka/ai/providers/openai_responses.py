"""OpenAI Responses provider adapter (pi ``api/openai-responses.ts``)."""

from __future__ import annotations

import contextlib
import json
import time
from collections.abc import AsyncIterator, Mapping
from typing import Any, Literal, TypedDict

try:
    from openai import AsyncOpenAI, omit
except ImportError:  # optional extra: misaka[openai]
    AsyncOpenAI = omit = None

from misaka.ai.models import clamp_thinking_level
from misaka.ai.providers._common import (
    _await_maybe_with_signal,
    _close_stream,
    _empty_usage,
    _iterate_async_iterable,
    _option,
    resolve_cache_retention,
    split_sdk_headers,
    split_sdk_params,
)
from misaka.ai.providers.constrained_sampling import (
    create_grammar_tool_input_properties,
)
from misaka.ai.providers.github_copilot_headers import (
    build_copilot_dynamic_headers,
    has_copilot_vision_input,
)
from misaka.ai.providers.openai_prompt_cache import clamp_openai_prompt_cache_key
from misaka.ai.providers.openai_responses_shared import (
    convert_responses_messages,
    convert_responses_tools,
    process_responses_stream,
)
from misaka.ai.providers.sdk import require
from misaka.ai.providers.simple_options import build_base_options
from misaka.ai.types import (
    AssistantMessage,
    CacheRetention,
    DoneEvent,
    ErrorEvent,
    Model,
    SimpleStreamOptions,
    StartEvent,
    StreamOptions,
    TranscriptContext,
    Usage,
)
from misaka.ai.utils.error_body import format_provider_error, normalize_provider_error
from misaka.ai.utils.event_stream import AssistantMessageEventStream, spawn_stream_task
from misaka.ai.utils.headers import headers_to_record
from misaka.ai.utils.provider_retry import retry_provider_request
from misaka.ai.utils.transcript import (
    get_declared_tools,
    resolve_transcript,
    resolve_transcript_tools,
)
from misaka.ai.utils.user_agent import get_misaka_user_agent
from misaka.utils.values import maybe_await, read_field, signal_aborted

OPENAI_TOOL_CALL_PROVIDERS = {"openai", "openai-codex", "opencode"}
# OpenAI Responses rejects max_output_tokens below 16: https://github.com/earendil-works/pi/issues/6265
OPENAI_RESPONSES_MIN_OUTPUT_TOKENS = 16
CHATGPT_USAGE_URL = "https://chatgpt.com/settings/usage"


def is_chatgpt_sign_in(model: Model, api_key: str | None) -> bool:
    """OpenAI API keys start with `sk-`; a different credential sent directly to OpenAI
    is a Sign in with ChatGPT access token.
    """
    return (
        model.provider == "openai"
        and model.baseUrl == "https://api.openai.com/v1"
        and api_key is not None
        and not api_key.startswith("sk-")
    )


def has_header(headers: Mapping[str, Any] | None, name: str) -> bool:
    if not headers:
        return False
    expected = name.lower()
    return any(
        str(key).lower() == expected and value is not None and len(str(value).strip()) > 0
        for key, value in headers.items()
    )


def get_client_api_key(provider: str, api_key: str | None, headers: Mapping[str, Any] | None) -> str:
    if api_key:
        return api_key
    if has_header(headers, "authorization") or has_header(headers, "cf-aig-authorization"):
        return "unused"
    raise RuntimeError(f"No API key for provider: {provider}")


def detect_session_affinity_format(model: Model) -> str:
    return "openrouter" if model.provider == "openrouter" or "openrouter.ai" in model.baseUrl else "openai"


def get_compat(model: Model) -> dict[str, Any]:
    compat = model.compat
    return {
        "supportsDeveloperRole": read_field(compat, "supportsDeveloperRole", True),
        "supportsMidConvoSystemMessages": read_field(compat, "supportsMidConvoSystemMessages", False),
        "sessionAffinityFormat": read_field(compat, "sessionAffinityFormat", detect_session_affinity_format(model)),
        "supportsLongCacheRetention": read_field(compat, "supportsLongCacheRetention", True),
        "supportsStrictMode": read_field(compat, "supportsStrictMode", False),
        "supportsOpenAIGrammarTools": read_field(compat, "supportsOpenAIGrammarTools", False),
        "supportsAdditionalTools": read_field(compat, "supportsAdditionalTools", False),
        "supportsToolSearch": read_field(compat, "supportsToolSearch", False),
        "supportsExplicitPromptCacheMode": read_field(compat, "supportsExplicitPromptCacheMode", False),
        "supportsMaxOutputTokens": read_field(compat, "supportsMaxOutputTokens", True),
    }


def get_prompt_cache_retention(compat: Mapping[str, Any], cache_retention: CacheRetention) -> str | None:
    return (
        "24h"
        if cache_retention == "long"
        and compat["supportsLongCacheRetention"]
        and not compat["supportsExplicitPromptCacheMode"]
        else None
    )


def get_prompt_cache_options(compat: Mapping[str, Any], cache_retention: CacheRetention) -> dict[str, str] | None:
    if not compat["supportsExplicitPromptCacheMode"]:
        return None
    if cache_retention == "none":
        return {"mode": "explicit"}
    if cache_retention == "long" and compat["supportsLongCacheRetention"]:
        return {"ttl": "30m"}
    return None


class OpenAIResponsesOptions(TypedDict, total=False):
    apiKey: str
    headers: dict[str, str | None]
    env: dict[str, str]
    signal: Any
    sessionId: str
    cacheRetention: str
    onPayload: Any
    onResponse: Any
    onProviderStreamEvent: Any
    timeoutMs: int
    maxRetries: int
    maxRetryDelayMs: int
    maxTokens: int
    temperature: float
    reasoningEffort: Literal["minimal", "low", "medium", "high", "xhigh", "max"]
    reasoningSummary: Literal["auto", "detailed", "concise"] | None
    serviceTier: Literal["auto", "default", "flex", "scale", "priority", "fast"]
    toolChoice: Any


def stream_openai_responses(
    model: Model,
    context: TranscriptContext,
    options: StreamOptions | dict[str, Any] | None = None,
) -> AssistantMessageEventStream:
    stream = AssistantMessageEventStream()
    normalized_context = resolve_transcript(context, get_compat(model)["supportsMidConvoSystemMessages"])

    async def run() -> None:
        output = AssistantMessage(
            content=[],
            api=model.api,
            provider=model.provider,
            model=model.id,
            usage=_empty_usage(),
            stopReason="pending",
            timestamp=time.time_ns() // 1_000_000,
        )

        try:
            compat = get_compat(model)
            grammar_tool_input_properties = create_grammar_tool_input_properties(
                get_declared_tools(normalized_context.messages),
                compat["supportsOpenAIGrammarTools"],
            )
            client = _option(options, "client")
            if client is None:
                # Create OpenAI client
                api_key = get_client_api_key(model.provider, _option(options, "apiKey"), _option(options, "headers"))
                cache_retention = resolve_cache_retention(_option(options, "cacheRetention"), _option(options, "env"))
                cache_session_id = None if cache_retention == "none" else _option(options, "sessionId")
                client, request_headers = create_client(
                    model,
                    normalized_context,
                    api_key,
                    _option(options, "headers"),
                    cache_session_id,
                )
            else:
                request_headers = {}
            params = build_params(model, normalized_context, options, compat, grammar_tool_input_properties)
            on_payload = _option(options, "onPayload")
            if callable(on_payload):
                next_params = await maybe_await(on_payload(params, model))
                if next_params is not None:
                    params = next_params
            openai_stream = await retry_provider_request(
                lambda: _create_responses_stream(client, params, options, model, request_headers),
                max_retries=_option(options, "maxRetries") or 0,
                max_retry_delay_ms=_option(options, "maxRetryDelayMs"),
                signal=_option(options, "signal"),
            )
            stream.push(StartEvent(partial=output))

            # MISAKA fork: the event iterator is closed with the reply (GitHub issue #6).
            events = _iterate_stream(openai_stream, _option(options, "signal"))
            async with contextlib.aclosing(events):
                await process_responses_stream(
                    events,
                    output,
                    stream,
                    model,
                    {
                        "onProviderStreamEvent": _option(options, "onProviderStreamEvent"),
                        "serviceTier": _option(options, "serviceTier"),
                        "grammarToolInputProperties": grammar_tool_input_properties,
                        "applyServiceTierPricing": lambda usage, service_tier: apply_service_tier_pricing(
                            usage, service_tier, model
                        ),
                    },
                )

            if signal_aborted(_option(options, "signal")):
                raise RuntimeError("Request was aborted")

            if output.stopReason == "pending":
                raise RuntimeError("OpenAI Responses stream ended without a stop reason")
            if output.stopReason in {"aborted", "error"}:
                raise RuntimeError(output.errorMessage or "An unknown error occurred")

            stream.push(DoneEvent(reason=output.stopReason, message=output))
        except Exception as error:  # noqa: BLE001
            # MISAKA fork: streaming scratch buffers live beside the blocks
            # (`openai_responses_shared._StreamingToolCallScratch`), so there is nothing to strip here.
            output.stopReason = "aborted" if signal_aborted(_option(options, "signal")) else "error"
            error_message = format_provider_error(
                normalize_provider_error(error),
                f"{'OpenAI' if model.provider == 'openai' else model.provider} API error",
            )
            # Sign in with ChatGPT shares the subscription's usage limit with other apps.
            output.errorMessage = (
                f"{error_message}\nCheck your ChatGPT usage: {CHATGPT_USAGE_URL}"
                if "subscription_sharing_usage_limit_exceeded" in error_message
                else error_message
            )
            stream.push(ErrorEvent(reason=output.stopReason, error=output), cause=error)
        finally:
            stream.end()

    spawn_stream_task(run(), stream=stream)
    return stream


def stream_simple_openai_responses(
    model: Model,
    context: TranscriptContext,
    options: SimpleStreamOptions | None = None,
) -> AssistantMessageEventStream:
    get_client_api_key(model.provider, _option(options, "apiKey"), _option(options, "headers"))

    base = {
        **build_base_options(model, context, options, _option(options, "apiKey")).model_dump(),
        "toolChoice": _option(options, "toolChoice"),
    }
    clamped_reasoning = clamp_thinking_level(model, options.reasoning) if options and options.reasoning else None
    reasoning_effort = None if clamped_reasoning == "off" else clamped_reasoning

    return stream_openai_responses(model, context, {**base, "reasoningEffort": reasoning_effort})


def create_client(
    model: Model,
    context: TranscriptContext,
    api_key: str,
    options_headers: Mapping[str, Any] | None = None,
    session_id: str | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Returns the client and (MISAKA fork, see ``split_sdk_headers``) the headers every
    request has to carry."""
    compat = get_compat(model)
    headers: dict[str, Any] = {"User-Agent": get_misaka_user_agent(), **(model.headers or {})}
    if model.provider == "github-copilot":
        has_images = has_copilot_vision_input(context.messages)
        copilot_headers = build_copilot_dynamic_headers(messages=context.messages, hasImages=has_images)
        headers.update(copilot_headers)

    if session_id:
        if compat["sessionAffinityFormat"] == "openrouter":
            headers["x-session-id"] = session_id
        else:
            if compat["sessionAffinityFormat"] == "openai":
                headers["session_id"] = session_id
            headers["x-client-request-id"] = session_id

    # Merge options headers last so they can override defaults
    if options_headers:
        headers.update(options_headers)

    default_headers, request_headers = split_sdk_headers(headers, omit)
    client = require(AsyncOpenAI, "openai")(
        api_key=api_key,
        base_url=model.baseUrl,
        default_headers=default_headers,
    )
    return client, request_headers


def build_params(
    model: Model,
    context: TranscriptContext,
    options: Any = None,
    compat: Mapping[str, Any] | None = None,
    grammar_tool_input_properties: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    compat = compat or get_compat(model)
    if grammar_tool_input_properties is None:
        grammar_tool_input_properties = create_grammar_tool_input_properties(
            get_declared_tools(context.messages), compat["supportsOpenAIGrammarTools"]
        )
    transcript_tools = resolve_transcript_tools(
        context.messages,
        compat["supportsAdditionalTools"] or compat["supportsToolSearch"],
    )
    messages = convert_responses_messages(
        model,
        context,
        OPENAI_TOOL_CALL_PROVIDERS,
        {
            "grammarToolInputProperties": grammar_tool_input_properties,
            "supportsMidConvoSystemMessages": compat["supportsMidConvoSystemMessages"],
            "supportsAdditionalTools": compat["supportsAdditionalTools"],
            "supportsToolSearch": compat["supportsToolSearch"],
            "toolOptions": {
                "supportsStrictMode": compat["supportsStrictMode"],
                "supportsOpenAIGrammarTools": compat["supportsOpenAIGrammarTools"],
            },
        },
    )

    cache_retention = resolve_cache_retention(_option(options, "cacheRetention"), _option(options, "env"))
    # Sign in with ChatGPT rejects these request fields.
    omit_unsupported_fields = is_chatgpt_sign_in(model, _option(options, "apiKey"))
    params: dict[str, Any] = {
        "model": model.id,
        "input": messages,
        "stream": True,
        "prompt_cache_key": (
            None if cache_retention == "none" else clamp_openai_prompt_cache_key(_option(options, "sessionId"))
        ),
        "prompt_cache_retention": None if omit_unsupported_fields else get_prompt_cache_retention(compat, cache_retention),
        "prompt_cache_options": None if omit_unsupported_fields else get_prompt_cache_options(compat, cache_retention),
        "store": False,
    }

    if _option(options, "maxTokens") and compat["supportsMaxOutputTokens"] and not omit_unsupported_fields:
        params["max_output_tokens"] = max(_option(options, "maxTokens"), OPENAI_RESPONSES_MIN_OUTPUT_TOKENS)

    if _option(options, "temperature") is not None and not omit_unsupported_fields:
        params["temperature"] = _option(options, "temperature")

    if _option(options, "serviceTier") is not None:
        params["service_tier"] = _option(options, "serviceTier")

    if len(transcript_tools.requestTools) > 0:
        params["tools"] = convert_responses_tools(
            transcript_tools.requestTools,
            {
                "supportsStrictMode": compat["supportsStrictMode"],
                "supportsOpenAIGrammarTools": compat["supportsOpenAIGrammarTools"],
            },
        )

    if _option(options, "toolChoice") is not None:
        params["tool_choice"] = _option(options, "toolChoice")

    if model.reasoning:
        reasoning_effort = _option(options, "reasoningEffort")
        reasoning_summary = _option(options, "reasoningSummary")
        thinking_level_map = model.thinkingLevelMap or {}
        if reasoning_effort or reasoning_summary:
            mapped_effort = thinking_level_map.get(reasoning_effort) if reasoning_effort else None
            effort = (reasoning_effort if mapped_effort is None else mapped_effort) if reasoning_effort else "medium"
            params["reasoning"] = {"effort": effort, "summary": reasoning_summary or "auto"}
            params["include"] = ["reasoning.encrypted_content"]
        elif model.provider != "github-copilot" and (
            model.thinkingLevelMap is None or thinking_level_map.get("off", ...) is not None
        ):
            off_value = thinking_level_map.get("off")
            params["reasoning"] = {"effort": "none" if off_value is None else off_value}
        if model.provider == "xai":
            params["include"] = ["reasoning.encrypted_content"]

    # Last so custom keys override the named request fields. Per-request keys override model defaults.
    params.update(model.samplingParams or {})
    params.update(_option(options, "samplingParams") or {})

    return params


def get_service_tier_cost_multiplier(model: Model, service_tier: str | None) -> float:
    if service_tier == "flex":
        return 0.5
    if service_tier in {"priority", "fast"}:
        return 2.5 if model.id == "gpt-5.5" else 2
    return 1


def apply_service_tier_pricing(usage: Usage, service_tier: str | None, model: Model) -> None:
    multiplier = get_service_tier_cost_multiplier(model, service_tier)
    if multiplier == 1:
        return

    usage.cost.input *= multiplier
    usage.cost.output *= multiplier
    usage.cost.cacheRead *= multiplier
    usage.cost.cacheWrite *= multiplier
    usage.cost.total = usage.cost.input + usage.cost.output + usage.cost.cacheRead + usage.cost.cacheWrite


# --- MISAKA: Python SDK plumbing ---------------------------------------------------------


async def _create_responses_stream(
    client: Any, params: dict[str, Any], options: Any, model: Model, request_headers: dict[str, Any] | None = None
) -> Any:
    """``client.responses.create(params, {signal, timeout, maxRetries: 0}).withResponse()``.

    The SDK's own retry is always off: ``utils/provider_retry`` decides what is worth retrying.
    """
    signal = _option(options, "signal")
    request_options: dict[str, Any] = {"max_retries": 0}
    if _option(options, "timeoutMs") is not None:
        request_options["timeout"] = _option(options, "timeoutMs") / 1000
    request_client = client.with_options(**request_options) if hasattr(client, "with_options") else client
    responses = request_client.responses
    call_options = {"extra_headers": dict(request_headers)} if request_headers else {}

    if hasattr(responses, "with_raw_response"):
        raw_response = await _await_maybe_with_signal(
            responses.with_raw_response.create(**split_sdk_params(responses.create, params), **call_options),
            signal,
        )
        on_response = _option(options, "onResponse")
        if callable(on_response):
            await maybe_await(
                on_response(
                    {
                        "status": raw_response.http_response.status_code,
                        "headers": headers_to_record(raw_response.http_response.headers),
                    },
                    model,
                )
            )
        return await _await_maybe_with_signal(raw_response.parse(), signal)

    return await _await_maybe_with_signal(
        responses.create(**split_sdk_params(responses.create, params), **call_options), signal
    )


async def _iterate_stream(stream_obj: Any, signal: Any = None) -> AsyncIterator[dict[str, Any]]:
    # MISAKA fork: one abort task for the whole stream (_iterate_async_iterable), not one per SSE
    # item: a task per item cost several loop turns per event, which a busy TUI turned into a stream
    # that fell minutes behind the provider (GitHub issue #6). Closed with this generator, so its
    # abort task goes with it (a turn that ended in an error left a pending signal.wait() behind
    # until the cyclic GC destroyed it).
    async with contextlib.aclosing(
            _iterate_async_iterable(stream_obj, signal, on_abort=lambda: _close_stream(stream_obj))) as items:
        async for event in items:
            if hasattr(event, "model_dump"):
                yield event.model_dump()
            elif isinstance(event, dict):
                yield event
            else:
                yield json.loads(json.dumps(event, default=lambda value: value.__dict__))


streamOpenAIResponses = stream_openai_responses
streamSimpleOpenAIResponses = stream_simple_openai_responses
applyServiceTierPricing = apply_service_tier_pricing

__all__ = [
    "OpenAIResponsesOptions",
    "streamOpenAIResponses",
    "streamSimpleOpenAIResponses",
]
