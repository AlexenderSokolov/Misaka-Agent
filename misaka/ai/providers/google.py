"""Google Generative AI provider adapter."""

from __future__ import annotations

import itertools
import json
import time
from collections.abc import Mapping
from typing import Any, Literal, TypedDict

try:
    from google.genai import Client as GoogleGenAI
except ImportError:  # optional extra: misaka[google]
    GoogleGenAI = None

from misaka.ai.models import calculate_cost, clamp_thinking_level
from misaka.ai.providers._common import (
    _close_stream,
    _empty_usage,
    _iterate_async_iterable,
    _option,
    _prepare_sdk_params,
)
from misaka.ai.providers.google_shared import (
    chunks_from,
    coerce_thought_signature,
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
    TextEndEvent,
    TextStartEvent,
    ThinkingBudgets,
    ThinkingContent,
    ThinkingDeltaEvent,
    ThinkingEndEvent,
    ThinkingStartEvent,
    ToolCall,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
    TranscriptContext,
    Usage,
    UsageCost,
)
from misaka.ai.utils.abort import race_with_abort_signal
from misaka.ai.utils.error_body import format_provider_error, normalize_provider_error
from misaka.ai.utils.event_stream import AssistantMessageEventStream, spawn_stream_task
from misaka.ai.utils.headers import provider_headers_to_record
from misaka.ai.utils.sanitize_unicode import sanitize_surrogates
from misaka.ai.utils.text import get_system_message_text
from misaka.ai.utils.transcript import (
    collapse_system_messages,
    get_current_tools,
    get_initial_system_message,
)
from misaka.ai.utils.user_agent import get_misaka_user_agent
from misaka.utils.values import maybe_await, signal_aborted

ClampedThinkingLevel = Literal["minimal", "low", "medium", "high"]


class GoogleOptions(TypedDict, total=False):
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

# Counter for generating unique tool call IDs
_TOOL_CALL_COUNTER = itertools.count(1)


def _nested_option(options: Any, name: str, default: Any = None) -> Any:
    if options is None:
        return default
    if isinstance(options, Mapping):
        return options.get(name, default)
    return getattr(options, name, default)


def _coalesce_attr(obj: Any, *names: str) -> Any:
    for name in names:
        value = getattr(obj, name, None)
        if value is not None:
            return value
    return None


def _finish_current_block(
    current_block: TextContent | ThinkingContent | None,
    output: AssistantMessage,
    stream: AssistantMessageEventStream,
) -> None:
    if current_block is None:
        return

    content_index = len(output.content) - 1
    if current_block.type == "text":
        stream.push(TextEndEvent(contentIndex=content_index, content=current_block.text, partial=output))
    else:
        stream.push(ThinkingEndEvent(contentIndex=content_index, content=current_block.thinking, partial=output))


def _build_tool_call(function_call: Any, part: Any, output: AssistantMessage) -> ToolCall:
    name = _coalesce_attr(function_call, "name") or ""
    raw_args = _coalesce_attr(function_call, "args") or {}
    if isinstance(raw_args, Mapping):
        arguments = dict(raw_args)
    elif hasattr(raw_args, "model_dump"):
        arguments = raw_args.model_dump()
    else:
        arguments = {}

    # Generate unique ID if not provided or if it's a duplicate
    provided_id = _coalesce_attr(function_call, "id")
    needs_new_id = not provided_id or any(
        isinstance(block, ToolCall) and block.id == provided_id
        for block in output.content
    )
    tool_call_id = f"{name}_{int(time.time() * 1000)}_{next(_TOOL_CALL_COUNTER)}" if needs_new_id else provided_id

    thought_signature = coerce_thought_signature(
        _coalesce_attr(part, "thought_signature", "thoughtSignature")
    )
    return ToolCall(
        id=tool_call_id,
        name=name,
        arguments=arguments,
        thoughtSignature=thought_signature or None,
    )


def _raw_finish_reason(finish_reason: Any) -> str:
    """The wire string of a finish reason: the Python SDK hands back its ``FinishReason``
    enum where upstream's gets the string itself."""
    return str(getattr(finish_reason, "value", finish_reason))


def _apply_usage_metadata(output: AssistantMessage, model: Model, usage_metadata: Any) -> None:
    prompt_token_count = int(_coalesce_attr(usage_metadata, "prompt_token_count", "promptTokenCount") or 0)
    cached_content_token_count = int(
        _coalesce_attr(usage_metadata, "cached_content_token_count", "cachedContentTokenCount") or 0
    )
    candidates_token_count = int(_coalesce_attr(usage_metadata, "candidates_token_count", "candidatesTokenCount") or 0)
    thoughts_token_count = int(_coalesce_attr(usage_metadata, "thoughts_token_count", "thoughtsTokenCount") or 0)
    output.usage = Usage(
        input=prompt_token_count - cached_content_token_count,
        output=candidates_token_count + thoughts_token_count,
        cacheRead=cached_content_token_count,
        cacheWrite=0,
        reasoning=thoughts_token_count,
        totalTokens=int(_coalesce_attr(usage_metadata, "total_token_count", "totalTokenCount") or 0),
        cost=UsageCost(input=0, output=0, cacheRead=0, cacheWrite=0, total=0),
    )
    calculate_cost(model, output.usage)


def stream_google(
    model: Model,
    context: TranscriptContext,
    options: StreamOptions | Mapping[str, Any] | None = None,
) -> AssistantMessageEventStream:
    stream = AssistantMessageEventStream()
    normalized_context = collapse_system_messages(context)

    async def run() -> None:
        output = AssistantMessage(
            content=[],
            api="google-generative-ai",
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
                api_key = _option(options, "apiKey")
                if not api_key:
                    raise RuntimeError(f"No API key for provider: {model.provider}")
                client = create_client(model, api_key, _option(options, "headers"))
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
                # @google/genai documents GenerateContentResponse.responseId as an output-only field
                # used to identify each response. Keep the first non-empty one from the stream.
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
                raise RuntimeError("Google stream ended without a finish reason")
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


def stream_simple_google(
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
    reasoning = _option(options, "reasoning")
    if not reasoning:
        return stream_google(model, context, {**base, "thinking": {"enabled": False}})

    clamped_reasoning = clamp_thinking_level(model, reasoning)
    if clamped_reasoning == "off":
        return stream_google(model, context, {**base, "thinking": {"enabled": False}})
    resolved_level: ClampedThinkingLevel = resolve_google_thinking_level(model, clamped_reasoning)

    if uses_google_thinking_level(model):
        return stream_google(
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

    return stream_google(
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
    api_key: str | None = None,
    options_headers: Mapping[str, str | None] | None = None,
) -> GoogleGenAI:
    http_options: dict[str, Any] = {}
    if model.baseUrl:
        http_options["baseUrl"] = model.baseUrl
        http_options["apiVersion"] = ""  # baseUrl already includes version path, don't append
    headers = provider_headers_to_record(
        {"User-Agent": get_misaka_user_agent(), **(model.headers or {}), **dict(options_headers or {})}
    )
    if headers:
        http_options["headers"] = headers

    return require(GoogleGenAI, "google-genai")(
        api_key=api_key,
        http_options=http_options if len(http_options) > 0 else None,
    )


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


def _custom_budget(custom_budgets: ThinkingBudgets | Mapping[str, int | None] | None, level: str) -> int | None:
    if custom_budgets is None:
        return None
    if isinstance(custom_budgets, Mapping):
        value = custom_budgets.get(level)
        return int(value) if value is not None else None
    value = getattr(custom_budgets, level, None)
    return int(value) if value is not None else None


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

    if "2.5-flash-lite" in model.id:
        budgets = {"minimal": 512, "low": 2048, "medium": 8192, "high": 24576}
        return budgets[level]

    if "2.5-flash" in model.id:
        budgets = {"minimal": 128, "low": 2048, "medium": 8192, "high": 24576}
        return budgets[level]

    return -1


streamGoogle = stream_google
streamSimpleGoogle = stream_simple_google

__all__ = [
    "GoogleOptions",
    "streamGoogle",
    "streamSimpleGoogle",
]
