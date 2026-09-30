"""Google Vertex AI provider adapter."""

from __future__ import annotations

import json
import re
import time
import warnings
from collections.abc import Mapping
from typing import Any, Literal, TypedDict
from urllib.parse import urlparse

try:
    from google.genai import Client as GoogleGenAI
except ImportError:  # optional extra: misaka[google]
    GoogleGenAI = None

from misaka.ai.auth.types import ProviderEnv
from misaka.ai.models import clamp_thinking_level
from misaka.ai.providers._common import (
    _close_stream,
    _iterate_async_iterable,
    _prepare_sdk_params,
)
from misaka.ai.providers.google import (
    _apply_usage_metadata,
    _build_tool_call,
    _coalesce_attr,
    _custom_budget,
    _empty_usage,
    _finish_current_block,
    _nested_option,
    _option,
    _raw_finish_reason,
    maybe_await,
)
from misaka.ai.providers.google_shared import (
    chunks_from,
    convert_messages,
    convert_tools,
    get_disabled_google_thinking_config,
    is_thinking_part,
    map_stop_reason,
    open_content_stream,
    resolve_google_function_calling_mode,
    resolve_google_thinking_level,
    retain_thought_signature,
    retry_google_request,
    supports_google_strict_tool_sampling,
    to_google_sdk_thinking_level,
    to_google_thinking_level,
    uses_google_thinking_level,
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
    TextContent,
    TextDeltaEvent,
    TextStartEvent,
    ThinkingBudgets,
    ThinkingContent,
    ThinkingDeltaEvent,
    ThinkingStartEvent,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
    TranscriptContext,
)
from misaka.ai.utils.abort import race_with_abort_signal
from misaka.ai.utils.error_body import format_provider_error, normalize_provider_error
from misaka.ai.utils.event_stream import AssistantMessageEventStream, spawn_stream_task
from misaka.ai.utils.headers import provider_headers_to_record
from misaka.ai.utils.provider_env import get_provider_env_value
from misaka.ai.utils.sanitize_unicode import sanitize_surrogates
from misaka.ai.utils.text import get_system_message_text
from misaka.ai.utils.transcript import (
    collapse_system_messages,
    get_current_tools,
    get_initial_system_message,
)
from misaka.ai.utils.user_agent import get_misaka_user_agent
from misaka.utils.values import signal_aborted

ClampedThinkingLevel = Literal["minimal", "low", "medium", "high"]


class GoogleVertexOptions(TypedDict, total=False):
    apiKey: str
    headers: dict[str, str]
    signal: Any
    sessionId: str
    cacheRetention: str
    onPayload: Any
    onResponse: Any
    onProviderStreamEvent: Any
    timeoutMs: int
    maxRetries: int
    toolChoice: Literal["auto", "none", "any"]
    thinking: dict[str, Any]
    project: str
    location: str

API_VERSION = "v1"
GCP_VERTEX_CREDENTIALS_MARKER = "gcp-vertex-credentials"
_API_VERSION_PATTERN = re.compile(r"^v\d+(?:beta\d*)?$")
_PLACEHOLDER_API_KEY_PATTERN = re.compile(r"^<[^>]+>$")
# MISAKA fork: the scope google-auth-library applies for @google/genai by default; the
# Python SDK only applies it on its own ADC path, so a key file loaded here names it.
_CLOUD_PLATFORM_SCOPE = "https://www.googleapis.com/auth/cloud-platform"


def stream_google_vertex(
    model: Model,
    context: TranscriptContext,
    options: StreamOptions | Mapping[str, Any] | None = None,
) -> AssistantMessageEventStream:
    stream = AssistantMessageEventStream()
    normalized_context = collapse_system_messages(context)

    async def run() -> None:
        output = AssistantMessage(
            content=[],
            api="google-vertex",
            provider=model.provider,
            model=model.id,
            usage=_empty_usage(),
            stopReason="pending",
            timestamp=time.time_ns() // 1_000_000,
        )
        signal = _option(options, "signal")
        # MISAKA fork: a caller may hand in a ready SDK client (the tests do).
        client: Any = _option(options, "client")

        try:
            if client is None:
                api_key = resolve_api_key(options)
                # Create the client using either a Vertex API key, if provided, or ADC with project and location
                client = (
                    create_client_with_api_key(model, api_key, _option(options, "headers"))
                    if api_key
                    else create_client(
                        model,
                        resolve_project(options),
                        resolve_location(options),
                        _option(options, "headers"),
                        _option(options, "env"),
                    )
                )
            params = build_params(model, normalized_context, options)
            on_payload = _option(options, "onPayload")
            if callable(on_payload):
                next_params = await maybe_await(on_payload(params, model))
                if next_params is not None:
                    params = next_params

            # MISAKA fork: pi hands the SDK the abort signal as ``config.abortSignal``; the
            # Python SDK takes none (``_prepare_sdk_params`` drops it), so the wait for the
            # stream races the signal instead.
            google_stream, first = await race_with_abort_signal(retry_google_request(
                lambda: open_content_stream(client, _prepare_sdk_params(params)),
                options,
            ), signal)

            stream.push(StartEvent(partial=output))
            current_block: TextContent | ThinkingContent | None = None
            on_provider_stream_event = _option(options, "onProviderStreamEvent")
            async for chunk in _iterate_async_iterable(chunks_from(first, google_stream), signal,
                                                       on_abort=lambda: _close_stream(google_stream)):
                if callable(on_provider_stream_event):
                    await maybe_await(on_provider_stream_event(chunk, model))
                if signal_aborted(signal):
                    raise RuntimeError("Request was aborted")
                # Vertex uses the same @google/genai GenerateContentResponse type as Gemini.
                # responseId is documented there as an output-only identifier for each response.
                output.responseId = output.responseId or _coalesce_attr(chunk, "response_id", "responseId")
                candidates = _coalesce_attr(chunk, "candidates") or []
                candidate = candidates[0] if candidates else None
                parts = _coalesce_attr(_coalesce_attr(candidate, "content"), "parts") or []

                for part in parts:
                    part_text = _coalesce_attr(part, "text")
                    if part_text is not None:
                        is_thinking = is_thinking_part(part)
                        if (
                            current_block is None
                            or (is_thinking and current_block.type != "thinking")
                            or (not is_thinking and current_block.type != "text")
                        ):
                            _finish_current_block(current_block, output, stream)
                            if is_thinking:
                                current_block = ThinkingContent(thinking="")
                                output.content.append(current_block)
                                stream.push(ThinkingStartEvent(contentIndex=len(output.content) - 1, partial=output))
                            else:
                                current_block = TextContent(text="")
                                output.content.append(current_block)
                                stream.push(TextStartEvent(contentIndex=len(output.content) - 1, partial=output))
                        if current_block.type == "thinking":
                            current_block.thinking += part_text
                            current_block.thinkingSignature = retain_thought_signature(
                                current_block.thinkingSignature,
                                _coalesce_attr(part, "thought_signature", "thoughtSignature"),
                            )
                            stream.push(
                                ThinkingDeltaEvent(
                                    contentIndex=len(output.content) - 1,
                                    delta=part_text,
                                    partial=output,
                                )
                            )
                        else:
                            current_block.text += part_text
                            current_block.textSignature = retain_thought_signature(
                                current_block.textSignature,
                                _coalesce_attr(part, "thought_signature", "thoughtSignature"),
                            )
                            stream.push(
                                TextDeltaEvent(
                                    contentIndex=len(output.content) - 1,
                                    delta=part_text,
                                    partial=output,
                                )
                            )

                    function_call = _coalesce_attr(part, "function_call", "functionCall")
                    if function_call is not None:
                        _finish_current_block(current_block, output, stream)
                        current_block = None

                        tool_call = _build_tool_call(function_call, part, output)
                        output.content.append(tool_call)
                        content_index = len(output.content) - 1
                        stream.push(ToolCallStartEvent(contentIndex=content_index, partial=output))
                        stream.push(
                            ToolCallDeltaEvent(
                                contentIndex=content_index,
                                delta=json.dumps(tool_call.arguments),
                                partial=output,
                            )
                        )
                        stream.push(ToolCallEndEvent(contentIndex=content_index, toolCall=tool_call, partial=output))

                finish_reason = _coalesce_attr(candidate, "finish_reason", "finishReason")
                if finish_reason:
                    output.rawStopReason = _raw_finish_reason(finish_reason)
                    output.stopReason = map_stop_reason(finish_reason)
                    if any(block.type == "toolCall" for block in output.content) and output.stopReason == "stop":
                        output.stopReason = "toolUse"

                usage_metadata = _coalesce_attr(chunk, "usage_metadata", "usageMetadata")
                if usage_metadata is not None:
                    _apply_usage_metadata(output, model, usage_metadata)

            _finish_current_block(current_block, output, stream)

            if signal_aborted(signal):
                raise RuntimeError("Request was aborted")

            if output.stopReason == "pending":
                raise RuntimeError("Google Vertex stream ended without a finish reason")
            if output.stopReason in {"aborted", "error"}:
                raise RuntimeError(
                    f"Provider stopped with: {output.rawStopReason}"
                    if output.rawStopReason
                    else "An unknown error occurred"
                )

            stream.push(DoneEvent(reason=output.stopReason, message=output))
        except Exception as error:  # noqa: BLE001
            output.stopReason = "aborted" if signal_aborted(signal) else "error"
            output.errorMessage = format_provider_error(normalize_provider_error(error))
            stream.push(ErrorEvent(reason=output.stopReason, error=output), cause=error)
        finally:
            aio_client = getattr(client, "aio", None)
            close = getattr(aio_client, "aclose", None) if aio_client is not None else None
            if callable(close):
                try:
                    await maybe_await(close())
                except Exception:  # noqa: BLE001, S110 - closing the SDK client is best-effort after the stream ended
                    pass
            stream.end()

    spawn_stream_task(run(), stream=stream)
    return stream


def stream_simple_google_vertex(
    model: Model,
    context: TranscriptContext,
    options: SimpleStreamOptions | None = None,
) -> AssistantMessageEventStream:
    base = {
        **build_base_options(model, context, options, None).model_dump(),
        "toolChoice": _option(options, "toolChoice"),
    }
    reasoning = _option(options, "reasoning")
    if not reasoning:
        return stream_google_vertex(model, context, {**base, "thinking": {"enabled": False}})

    clamped_reasoning = clamp_thinking_level(model, reasoning)
    if clamped_reasoning == "off":
        return stream_google_vertex(model, context, {**base, "thinking": {"enabled": False}})
    resolved_level: ClampedThinkingLevel = resolve_google_thinking_level(model, clamped_reasoning)

    if uses_google_thinking_level(model):
        return stream_google_vertex(
            model,
            context,
            {
                **base,
                "thinking": {
                    "enabled": True,
                    "level": to_google_thinking_level(resolved_level),
                },
            },
        )

    return stream_google_vertex(
        model,
        context,
        {
            **base,
            "thinking": {
                "enabled": True,
                "budgetTokens": get_google_budget(model, resolved_level, _option(options, "thinkingBudgets")),
            },
        },
    )


def create_client(
    model: Model,
    project: str,
    location: str,
    options_headers: Mapping[str, str | None] | None = None,
    env: ProviderEnv | None = None,
) -> GoogleGenAI:
    google_auth_options = build_google_auth_options(env)
    return require(GoogleGenAI, "google-genai")(
        vertexai=True,
        project=project,
        location=location,
        **({"credentials": _load_google_credentials(google_auth_options)} if google_auth_options else {}),
        http_options=_with_api_version(build_http_options(model, options_headers)),
    )


def create_client_with_api_key(
    model: Model,
    api_key: str,
    options_headers: Mapping[str, str | None] | None = None,
) -> GoogleGenAI:
    return require(GoogleGenAI, "google-genai")(
        vertexai=True,
        api_key=api_key,
        http_options=_with_api_version(build_http_options(model, options_headers)),
    )


def _with_api_version(http_options: dict[str, Any] | None) -> dict[str, Any]:
    """MISAKA fork: @google/genai takes ``apiVersion`` beside ``httpOptions`` and lets
    ``httpOptions.apiVersion`` override it; the Python SDK reads it from the HTTP options
    alone, so the default goes in there unless the options already set one."""
    return {"apiVersion": API_VERSION, **(http_options or {})}


def build_http_options(
    model: Model,
    options_headers: Mapping[str, str | None] | None = None,
) -> dict[str, Any] | None:
    http_options: dict[str, Any] = {}
    base_url = resolve_custom_base_url(model.baseUrl)
    if base_url:
        http_options["baseUrl"] = base_url
        http_options["baseUrlResourceScope"] = "COLLECTION"
        if base_url_includes_api_version(base_url):
            http_options["apiVersion"] = ""

    headers = provider_headers_to_record(
        {"User-Agent": get_misaka_user_agent(), **(model.headers or {}), **dict(options_headers or {})}
    )
    if headers:
        http_options["headers"] = headers

    return http_options if len(http_options) > 0 else None


def resolve_custom_base_url(base_url: str) -> str | None:
    trimmed = base_url.strip()
    if not trimmed or "{location}" in trimmed:
        return None
    return trimmed


def base_url_includes_api_version(base_url: str) -> bool:
    try:
        parsed = urlparse(base_url)
        return any(_API_VERSION_PATTERN.fullmatch(part) for part in parsed.path.split("/") if part)
    except Exception:  # noqa: BLE001
        return bool(re.search(r"(?:^|/)v\d+(?:beta\d*)?(?:/|$)", base_url))


def build_google_auth_options(env: ProviderEnv | None = None) -> dict[str, str] | None:
    key_filename = get_provider_env_value("GOOGLE_APPLICATION_CREDENTIALS", env)
    return {"keyFilename": key_filename} if key_filename else None


def _load_google_credentials(google_auth_options: Mapping[str, str]) -> Any:
    """MISAKA fork: @google/genai takes ``googleAuthOptions`` and loads the key file itself;
    the Python SDK takes loaded ``credentials``. The loader's deprecation warning is
    silenced the way google-auth silences it for ``GOOGLE_APPLICATION_CREDENTIALS``."""
    import google.auth

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        credentials, _project_id = google.auth.load_credentials_from_file(
            google_auth_options["keyFilename"], scopes=[_CLOUD_PLATFORM_SCOPE]
        )
    return credentials


def resolve_api_key(options: StreamOptions | Mapping[str, Any] | None = None) -> str | None:
    api_key = (_option(options, "apiKey") or "").strip()
    if not api_key or api_key == GCP_VERTEX_CREDENTIALS_MARKER or is_placeholder_api_key(api_key):
        return None
    return api_key


def is_placeholder_api_key(api_key: str) -> bool:
    return _PLACEHOLDER_API_KEY_PATTERN.fullmatch(api_key) is not None


def resolve_project(options: StreamOptions | Mapping[str, Any] | None = None) -> str:
    project = (
        _option(options, "project")
        or get_provider_env_value("GOOGLE_CLOUD_PROJECT", _option(options, "env"))
        or get_provider_env_value("GCLOUD_PROJECT", _option(options, "env"))
    )
    if not project:
        raise RuntimeError(
            "Vertex AI requires a project ID. Set GOOGLE_CLOUD_PROJECT/GCLOUD_PROJECT or pass project in options."
        )
    return project


def resolve_location(options: StreamOptions | Mapping[str, Any] | None = None) -> str:
    location = _option(options, "location") or get_provider_env_value("GOOGLE_CLOUD_LOCATION", _option(options, "env"))
    if not location:
        raise RuntimeError("Vertex AI requires a location. Set GOOGLE_CLOUD_LOCATION or pass location in options.")
    return location


def build_params(
    model: Model,
    context: TranscriptContext,
    options: StreamOptions | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    contents = convert_messages(model, context)
    initial_system_message = get_initial_system_message(context.messages)
    current_tools = get_current_tools(context.messages)

    generation_config: dict[str, Any] = {}
    if _option(options, "temperature") is not None:
        generation_config["temperature"] = _option(options, "temperature")
    if _option(options, "maxTokens") is not None:
        generation_config["maxOutputTokens"] = _option(options, "maxTokens")

    supports_strict_mode = supports_google_strict_tool_sampling(model.id)
    function_calling_mode = (
        resolve_google_function_calling_mode(current_tools, _option(options, "toolChoice"), supports_strict_mode)
        if len(current_tools) > 0
        else None
    )
    system_instruction = get_system_message_text(initial_system_message) if initial_system_message else ""
    config: dict[str, Any] = dict(generation_config)
    if system_instruction:
        config["systemInstruction"] = sanitize_surrogates(system_instruction)
    if len(current_tools) > 0:
        config["tools"] = convert_tools(current_tools, False, supports_strict_mode)
    if function_calling_mode is not None:
        config["toolConfig"] = {"functionCallingConfig": {"mode": function_calling_mode}}

    thinking = _option(options, "thinking")
    if _nested_option(thinking, "enabled") and model.reasoning:
        thinking_config: dict[str, Any] = {"includeThoughts": True}
        if _nested_option(thinking, "level") is not None:
            thinking_config["thinkingLevel"] = to_google_sdk_thinking_level(_nested_option(thinking, "level"))
        elif _nested_option(thinking, "budgetTokens") is not None:
            thinking_config["thinkingBudget"] = _nested_option(thinking, "budgetTokens")
        config["thinkingConfig"] = thinking_config
    elif model.reasoning and thinking is not None and not _nested_option(thinking, "enabled"):
        config["thinkingConfig"] = get_disabled_google_thinking_config(model)

    signal = _option(options, "signal")
    if signal is not None:
        if signal_aborted(signal):
            raise RuntimeError("Request aborted")
        config["abortSignal"] = signal

    return {
        "model": model.id,
        "contents": contents,
        "config": config,
    }


def get_google_budget(
    model: Model,
    level: ClampedThinkingLevel,
    custom_budgets: ThinkingBudgets | Mapping[str, int | None] | None = None,
) -> int:
    custom_budget = _custom_budget(custom_budgets, level)
    if custom_budget is not None:
        return custom_budget

    if "2.5-pro" in model.id:
        budgets = {"minimal": 128, "low": 2048, "medium": 8192, "high": 32768}
        return budgets[level]

    if "2.5-flash" in model.id:
        budgets = {"minimal": 128, "low": 2048, "medium": 8192, "high": 24576}
        return budgets[level]

    return -1


streamGoogleVertex = stream_google_vertex
streamSimpleGoogleVertex = stream_simple_google_vertex
__all__ = [
    "GoogleVertexOptions",
    "streamGoogleVertex",
    "streamSimpleGoogleVertex",
]
