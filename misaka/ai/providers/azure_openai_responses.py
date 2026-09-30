"""Azure OpenAI Responses provider adapter (pi ``api/azure-openai-responses.ts``)."""

from __future__ import annotations

import contextlib
import json
import time
from collections.abc import AsyncIterable, Mapping
from typing import Any, Literal, TypedDict
from urllib.parse import urlparse, urlunparse

try:
    from openai import AsyncAzureOpenAI, omit
except ImportError:  # optional extra: misaka[openai]
    AsyncAzureOpenAI = omit = None

from misaka.ai.models import clamp_thinking_level
from misaka.ai.providers._common import (
    _await_maybe_with_signal,
    _close_stream,
    _empty_usage,
    _iterate_async_iterable,
    _option,
    split_sdk_headers,
    split_sdk_params,
)
from misaka.ai.providers.constrained_sampling import (
    create_grammar_tool_input_properties,
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
    DoneEvent,
    ErrorEvent,
    Model,
    SimpleStreamOptions,
    StartEvent,
    StreamOptions,
    TranscriptContext,
)
from misaka.ai.utils.error_body import format_provider_error, normalize_provider_error
from misaka.ai.utils.event_stream import AssistantMessageEventStream, spawn_stream_task
from misaka.ai.utils.headers import headers_to_record
from misaka.ai.utils.provider_env import get_provider_env_value
from misaka.ai.utils.provider_retry import retry_provider_request
from misaka.ai.utils.transcript import (
    get_declared_tools,
    resolve_transcript,
    resolve_transcript_tools,
)
from misaka.ai.utils.user_agent import get_misaka_user_agent
from misaka.utils.values import maybe_await, read_field, signal_aborted

DEFAULT_AZURE_API_VERSION = "v1"
AZURE_TOOL_CALL_PROVIDERS = {"openai", "openai-codex", "opencode", "azure-openai-responses"}
# OpenAI Responses rejects max_output_tokens below 16: https://github.com/earendil-works/pi/issues/6265
OPENAI_RESPONSES_MIN_OUTPUT_TOKENS = 16


def parse_deployment_name_map(value: str | None) -> dict[str, str]:
    mapping: dict[str, str] = {}
    if not value:
        return mapping
    for entry in value.split(","):
        trimmed = entry.strip()
        if not trimmed:
            continue
        model_id, _, deployment_name = trimmed.partition("=")
        if not model_id or not deployment_name:
            continue
        mapping[model_id.strip()] = deployment_name.strip()
    return mapping


def resolve_deployment_name(model: Model, options: Any = None) -> str:
    if _option(options, "azureDeploymentName"):
        return _option(options, "azureDeploymentName")
    mapped_deployment = parse_deployment_name_map(
        get_provider_env_value("AZURE_OPENAI_DEPLOYMENT_NAME_MAP", _option(options, "env"))
    ).get(model.id)
    return mapped_deployment or model.id


def format_azure_openai_error(error: Any) -> str:
    return format_provider_error(normalize_provider_error(error), "Azure OpenAI API error")


class AzureOpenAIResponsesOptions(TypedDict, total=False):
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
    reasoningEffort: Literal["minimal", "low", "medium", "high", "xhigh", "max"]
    toolChoice: Any
    reasoningSummary: Literal["auto", "detailed", "concise"] | None
    azureApiVersion: str
    azureResourceName: str
    azureBaseUrl: str
    azureDeploymentName: str


def _compat(model: Model, name: str, default: Any) -> Any:
    return read_field(model.compat, name, default)


def stream_azure_openai_responses(
    model: Model,
    context: TranscriptContext,
    options: StreamOptions | dict[str, Any] | None = None,
) -> AssistantMessageEventStream:
    stream = AssistantMessageEventStream()
    normalized_context = resolve_transcript(context, _compat(model, "supportsMidConvoSystemMessages", None))

    async def run() -> None:
        deployment_name = resolve_deployment_name(model, options)

        output = AssistantMessage(
            content=[],
            api="azure-openai-responses",
            provider=model.provider,
            model=model.id,
            usage=_empty_usage(),
            stopReason="pending",
            timestamp=time.time_ns() // 1_000_000,
        )

        try:
            # Create Azure OpenAI client
            client = _option(options, "client")
            if client is None:
                api_key = _option(options, "apiKey")
                if not api_key:
                    raise RuntimeError(f"No API key for provider: {model.provider}")
                client, request_headers = create_client(model, api_key, options)
            else:
                request_headers = {}
            grammar_tool_input_properties = create_grammar_tool_input_properties(
                get_declared_tools(normalized_context.messages),
                _compat(model, "supportsOpenAIGrammarTools", False),
            )
            params = build_params(model, normalized_context, options, deployment_name, grammar_tool_input_properties)
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
                        "grammarToolInputProperties": grammar_tool_input_properties,
                    },
                )

            if signal_aborted(_option(options, "signal")):
                raise RuntimeError("Request was aborted")

            if output.stopReason == "pending":
                raise RuntimeError("Azure OpenAI Responses stream ended without a stop reason")
            if output.stopReason in {"aborted", "error"}:
                raise RuntimeError(output.errorMessage or "An unknown error occurred")

            stream.push(DoneEvent(reason=output.stopReason, message=output))
        except Exception as error:  # noqa: BLE001
            output.stopReason = "aborted" if signal_aborted(_option(options, "signal")) else "error"
            output.errorMessage = format_azure_openai_error(error)
            stream.push(ErrorEvent(reason=output.stopReason, error=output), cause=error)
        finally:
            stream.end()

    spawn_stream_task(run(), stream=stream)
    return stream


def stream_simple_azure_openai_responses(
    model: Model,
    context: TranscriptContext,
    options: SimpleStreamOptions | None = None,
) -> AssistantMessageEventStream:
    api_key = _option(options, "apiKey")
    if not api_key:
        raise RuntimeError(f"No API key for provider: {model.provider}")

    base = {
        **build_base_options(model, context, options, api_key).model_dump(),
        "toolChoice": _option(options, "toolChoice"),
    }
    clamped_reasoning = clamp_thinking_level(model, options.reasoning) if options and options.reasoning else None
    reasoning_effort = None if clamped_reasoning == "off" else clamped_reasoning

    return stream_azure_openai_responses(model, context, {**base, "reasoningEffort": reasoning_effort})


def normalize_azure_base_url(base_url: str) -> str:
    trimmed = base_url.strip().rstrip("/")
    parsed = urlparse(trimmed)
    try:
        hostname = parsed.hostname
        _ = parsed.port
    except ValueError as error:
        raise RuntimeError(f"Invalid Azure OpenAI base URL: {base_url}") from error
    if not parsed.scheme or not parsed.netloc or hostname is None:
        raise RuntimeError(f"Invalid Azure OpenAI base URL: {base_url}")

    is_azure_host = hostname.endswith((".openai.azure.com", ".cognitiveservices.azure.com", ".ai.azure.com"))
    normalized_path = parsed.path.rstrip("/")

    # Ensure Azure hosts have /openai/v1 as base path so the AzureOpenAI SDK
    # can append /deployments/<model>/... and ?api-version=v1 correctly.
    if is_azure_host and normalized_path in {"", "/", "/openai", "/openai/v1/responses"}:
        parsed = parsed._replace(path="/openai/v1", query="")

    return urlunparse(parsed).rstrip("/")


def build_default_base_url(resource_name: str) -> str:
    return f"https://{resource_name}.openai.azure.com/openai/v1"


def resolve_azure_config(model: Model, options: Any = None) -> dict[str, str]:
    env = _option(options, "env")
    api_version = (
        _option(options, "azureApiVersion")
        or get_provider_env_value("AZURE_OPENAI_API_VERSION", env)
        or DEFAULT_AZURE_API_VERSION
    )

    base_url = (_option(options, "azureBaseUrl") or "").strip() or (
        get_provider_env_value("AZURE_OPENAI_BASE_URL", env) or ""
    ).strip() or None
    resource_name = _option(options, "azureResourceName") or get_provider_env_value("AZURE_OPENAI_RESOURCE_NAME", env)

    resolved_base_url = base_url

    if not resolved_base_url and resource_name:
        resolved_base_url = build_default_base_url(resource_name)

    if not resolved_base_url and model.baseUrl:
        resolved_base_url = model.baseUrl

    if not resolved_base_url:
        raise RuntimeError(
            "Azure OpenAI base URL is required. Set AZURE_OPENAI_BASE_URL or AZURE_OPENAI_RESOURCE_NAME, or pass azureBaseUrl, azureResourceName, or model.baseUrl."
        )

    return {"baseUrl": normalize_azure_base_url(resolved_base_url), "apiVersion": api_version}


def create_client(model: Model, api_key: str, options: Any = None) -> tuple[Any, dict[str, Any]]:
    """Returns the client and (MISAKA fork, see ``split_sdk_headers``) the headers every
    request has to carry."""
    headers: dict[str, Any] = {"User-Agent": get_misaka_user_agent(), **(model.headers or {})}

    if _option(options, "headers"):
        headers.update(_option(options, "headers"))

    config = resolve_azure_config(model, options)

    default_headers, request_headers = split_sdk_headers(headers, omit)
    client = require(AsyncAzureOpenAI, "openai")(
        api_key=api_key,
        api_version=config["apiVersion"],
        default_headers=default_headers,
        base_url=config["baseUrl"],
    )
    return client, request_headers


def build_params(
    model: Model,
    context: TranscriptContext,
    options: Any,
    deployment_name: str,
    grammar_tool_input_properties: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    if grammar_tool_input_properties is None:
        grammar_tool_input_properties = create_grammar_tool_input_properties(
            get_declared_tools(context.messages), _compat(model, "supportsOpenAIGrammarTools", False)
        )
    supports_additional_tools = _compat(model, "supportsAdditionalTools", False)
    supports_tool_search = _compat(model, "supportsToolSearch", False)
    transcript_tools = resolve_transcript_tools(context.messages, supports_additional_tools or supports_tool_search)
    messages = convert_responses_messages(
        model,
        context,
        AZURE_TOOL_CALL_PROVIDERS,
        {
            "grammarToolInputProperties": grammar_tool_input_properties,
            "supportsMidConvoSystemMessages": _compat(model, "supportsMidConvoSystemMessages", False),
            "supportsAdditionalTools": supports_additional_tools,
            "supportsToolSearch": supports_tool_search,
            "toolOptions": {
                "supportsStrictMode": _compat(model, "supportsStrictMode", True),
                "supportsOpenAIGrammarTools": _compat(model, "supportsOpenAIGrammarTools", False),
            },
        },
    )

    params: dict[str, Any] = {
        "model": deployment_name,
        "input": messages,
        "stream": True,
        "prompt_cache_key": clamp_openai_prompt_cache_key(_option(options, "sessionId")),
        "store": False,
    }

    if _option(options, "maxTokens"):
        params["max_output_tokens"] = max(_option(options, "maxTokens"), OPENAI_RESPONSES_MIN_OUTPUT_TOKENS)

    if _option(options, "temperature") is not None:
        params["temperature"] = _option(options, "temperature")

    if len(transcript_tools.requestTools) > 0:
        params["tools"] = convert_responses_tools(
            transcript_tools.requestTools,
            {
                "supportsStrictMode": _compat(model, "supportsStrictMode", True),
                "supportsOpenAIGrammarTools": _compat(model, "supportsOpenAIGrammarTools", False),
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
        elif model.thinkingLevelMap is None or thinking_level_map.get("off", ...) is not None:
            off_value = thinking_level_map.get("off")
            params["reasoning"] = {"effort": "none" if off_value is None else off_value}

    # Last so custom keys override the named request fields. Per-request keys override model defaults.
    params.update(model.samplingParams or {})
    params.update(_option(options, "samplingParams") or {})

    return params


# --- MISAKA: Python SDK plumbing ---------------------------------------------------------


async def _create_responses_stream(
    client: Any, params: dict[str, Any], options: Any, model: Model, request_headers: dict[str, Any] | None = None
) -> Any:
    """``client.responses.create(params, {signal, timeout, maxRetries: 0}).withResponse()``."""
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


async def _iterate_stream(stream_obj: Any, signal: Any = None) -> AsyncIterable[dict[str, Any]]:
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


streamAzureOpenAIResponses = stream_azure_openai_responses
streamSimpleAzureOpenAIResponses = stream_simple_azure_openai_responses

__all__ = [
    "AzureOpenAIResponsesOptions",
    "streamAzureOpenAIResponses",
    "streamSimpleAzureOpenAIResponses",
]
