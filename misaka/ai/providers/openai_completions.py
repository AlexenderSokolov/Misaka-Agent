"""OpenAI Chat Completions provider adapter (pi ``api/openai-completions.ts``)."""

from __future__ import annotations

import contextlib
import json
import re
import time
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from typing import Any, Literal, TypedDict

try:
    from openai import AsyncOpenAI, omit
except ImportError:  # optional extra: misaka[openai]
    AsyncOpenAI = omit = None

from misaka.ai.models import calculate_cost, clamp_thinking_level
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
    GrammarToolInputJsonBuffer,
    append_grammar_tool_input_json_delta,
    create_grammar_tool_input_properties,
    get_grammar_tool_input,
    get_json_schema_tool_parameters,
    resolve_grammar_constrained_sampling,
    resolve_json_schema_strict_sampling,
)
from misaka.ai.providers.github_copilot_headers import (
    build_copilot_dynamic_headers,
    has_copilot_vision_input,
)
from misaka.ai.providers.openai_prompt_cache import clamp_openai_prompt_cache_key
from misaka.ai.providers.sdk import require
from misaka.ai.providers.simple_options import (
    build_base_options,
    clamp_thinking_budget_to_answer_room,
    thinking_budget_for_level,
)
from misaka.ai.providers.transform_messages import transform_messages
from misaka.ai.types import (
    AssistantMessage,
    CacheRetention,
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
    ThinkingContent,
    ThinkingDeltaEvent,
    ThinkingEndEvent,
    ThinkingStartEvent,
    Tool,
    ToolCall,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
    ToolResultMessage,
    TranscriptContext,
    Usage,
    UsageCost,
)
from misaka.ai.utils.error_body import format_provider_error, normalize_provider_error
from misaka.ai.utils.event_stream import AssistantMessageEventStream, spawn_stream_task
from misaka.ai.utils.hash import short_hash
from misaka.ai.utils.headers import headers_to_record
from misaka.ai.utils.json_parse import StreamingArgs
from misaka.ai.utils.provider_retry import retry_provider_request
from misaka.ai.utils.sanitize_unicode import sanitize_surrogates
from misaka.ai.utils.text import get_system_message_text, render_system_message_update
from misaka.ai.utils.transcript import (
    get_declared_tools,
    resolve_transcript,
    resolve_transcript_tools,
)
from misaka.ai.utils.user_agent import get_misaka_user_agent
from misaka.utils.values import maybe_await, read_field, signal_aborted


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


def has_tool_history(messages: list[Any]) -> bool:
    """Check if conversation messages contain tool calls or tool results.

    This is needed because Anthropic (via proxy) requires the tools param
    to be present when messages include tool_calls or tool role messages.
    """
    for msg in messages:
        if msg.role == "toolResult":
            return True
        if msg.role == "assistant" and any(block.type == "toolCall" for block in msg.content):
            return True
    return False


def _is_reasoning_detail_object(detail: Any) -> bool:
    return isinstance(detail, Mapping)


def _has_valid_common_reasoning_detail_fields(candidate: Mapping[str, Any]) -> bool:
    return (
        (candidate.get("id") is None or isinstance(candidate.get("id"), str))
        and ("format" not in candidate or isinstance(candidate.get("format"), str))
        and ("index" not in candidate or isinstance(candidate.get("index"), (int, float)))
    )


def is_openai_reasoning_detail(detail: Any) -> bool:
    if not _is_reasoning_detail_object(detail) or not _has_valid_common_reasoning_detail_fields(detail):
        return False
    kind = detail.get("type")
    if kind == "reasoning.summary":
        return isinstance(detail.get("summary"), str)
    if kind == "reasoning.encrypted":
        return isinstance(detail.get("data"), str)
    if kind == "reasoning.text":
        return isinstance(detail.get("text"), str) and (
            detail.get("signature") is None or isinstance(detail.get("signature"), str)
        )
    return False


class OpenAICompletionsToolChoiceFunction(TypedDict):
    name: str


class OpenAICompletionsToolChoiceObject(TypedDict):
    type: Literal["function"]
    function: OpenAICompletionsToolChoiceFunction


class OpenAICompletionsOptions(TypedDict, total=False):
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
    toolChoice: Literal["auto", "none", "required"] | OpenAICompletionsToolChoiceObject
    reasoningEffort: Literal["minimal", "low", "medium", "high", "xhigh", "max"]
    # Token budgets per thinking level. Used when `compat.thinkingTokenBudgetField` or
    # `compat.supportsThinkingTokenBudget` is set, or by `{ "$var": "thinking.budget" }`.
    thinkingBudgets: Any


def parse_openai_reasoning_details(signature: str | None) -> list[dict[str, Any]] | None:
    if not signature:
        return None
    try:
        parsed = json.loads(signature)
    except ValueError:
        return None
    return parsed if isinstance(parsed, list) and parsed and all(map(is_openai_reasoning_detail, parsed)) else None


def parse_legacy_encrypted_reasoning_detail(signature: str | None) -> dict[str, Any] | None:
    if not signature:
        return None
    try:
        parsed = json.loads(signature)
    except ValueError:
        return None
    return (
        parsed
        if is_openai_reasoning_detail(parsed)
        and parsed.get("type") == "reasoning.encrypted"
        and isinstance(parsed.get("id"), str)
        and len(parsed["id"]) > 0
        and len(parsed["data"]) > 0
        else None
    )


def _fill_missing_common_reasoning_detail_fields(target: dict[str, Any], source: Mapping[str, Any]) -> None:
    if target.get("id") is None:
        target["id"] = source.get("id")
    if not target.get("format"):
        target["format"] = source.get("format")
    if target.get("index") is None:
        target["index"] = source.get("index")
    for key in ("id", "format", "index"):
        if target.get(key) is None:
            target.pop(key, None)


def append_openai_reasoning_detail(details: list[dict[str, Any]], detail: Mapping[str, Any]) -> None:
    last_detail = details[-1] if details else None
    if detail.get("type") == "reasoning.text" and last_detail is not None and last_detail.get("type") == "reasoning.text":
        last_detail["text"] += detail["text"]
        if not last_detail.get("signature"):
            last_detail["signature"] = detail.get("signature")
            if last_detail["signature"] is None:
                del last_detail["signature"]
        _fill_missing_common_reasoning_detail_fields(last_detail, detail)
        return
    if (
        detail.get("type") == "reasoning.summary"
        and last_detail is not None
        and last_detail.get("type") == "reasoning.summary"
    ):
        last_detail["summary"] += detail["summary"]
        _fill_missing_common_reasoning_detail_fields(last_detail, detail)
        return
    details.append(dict(detail))


OPENAI_COMPLETIONS_REASONING_FIELDS = ("reasoning", "reasoning_content", "reasoning_text")


def is_openai_completions_reasoning_field(field: str | None) -> bool:
    return field in OPENAI_COMPLETIONS_REASONING_FIELDS


@dataclass
class _CustomToolCallInput:
    property: str
    jsonBuffer: GrammarToolInputJsonBuffer


@dataclass
class _StreamingToolCallScratch:
    """MISAKA fork: pi widens the block with ``partialArgs``/``customInput``/``streamIndex`` and
    deletes them when the call finishes; pydantic blocks forbid extra fields, so they live
    here, beside the block. ``partialArgs`` is a ``StreamingArgs`` so the live view does not
    re-parse the whole buffer on every delta."""

    partialArgs: StreamingArgs | None = None
    customInput: _CustomToolCallInput | None = None
    streamIndex: int | None = None


def stream_openai_completions(
    model: Model,
    context: TranscriptContext,
    options: StreamOptions | dict[str, Any] | None = None,
) -> AssistantMessageEventStream:
    stream = AssistantMessageEventStream()
    normalized_context = resolve_transcript(context, get_compat(model).get("supportsMidConvoSystemMessages"))

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

        chunks = None
        # `reasoning_details` are replay metadata, not user-visible stream deltas.
        # Keep them in memory during streaming and serialize once when the block is finalized.
        streamed_reasoning_details: list[dict[str, Any]] | None = None

        def apply_streamed_reasoning_details(block: ThinkingContent) -> None:
            if streamed_reasoning_details is not None:
                block.thinkingSignature = json.dumps(streamed_reasoning_details, separators=(",", ":"))

        try:
            compat = get_compat(model)
            grammar_tool_input_properties = create_grammar_tool_input_properties(
                get_declared_tools(normalized_context.messages),
                bool(compat.get("supportsOpenAIGrammarTools")),
            )
            cache_retention = resolve_cache_retention(_option(options, "cacheRetention"), _option(options, "env"))
            client = _option(options, "client")
            if client is None:
                api_key = get_client_api_key(model.provider, _option(options, "apiKey"), _option(options, "headers"))
                cache_session_id = None if cache_retention == "none" else _option(options, "sessionId")
                client, request_headers = create_client(
                    model,
                    normalized_context,
                    api_key,
                    _option(options, "headers"),
                    cache_session_id,
                    compat,
                )
            else:
                request_headers = {}
            params = build_params(
                model,
                normalized_context,
                options,
                compat,
                cache_retention,
                grammar_tool_input_properties,
            )
            on_payload = _option(options, "onPayload")
            if callable(on_payload):
                next_params = await maybe_await(on_payload(params, model))
                if next_params is not None:
                    params = next_params
            openai_stream = await retry_provider_request(
                lambda: _create_completion_stream(client, params, options, model, request_headers),
                max_retries=_option(options, "maxRetries") or 0,
                max_retry_delay_ms=_option(options, "maxRetryDelayMs"),
                signal=_option(options, "signal"),
            )
            stream.push(StartEvent(partial=output))

            text_block: TextContent | None = None
            thinking_block: ThinkingContent | None = None
            has_finish_reason = False
            tool_call_blocks_by_index: dict[int, ToolCall] = {}
            tool_call_blocks_by_id: dict[str, ToolCall] = {}
            scratch_by_block: dict[int, _StreamingToolCallScratch] = {}
            blocks = output.content

            def get_content_index(block: Any) -> int:
                return next((i for i, candidate in enumerate(blocks) if candidate is block), -1)

            def scratch_of(block: ToolCall) -> _StreamingToolCallScratch:
                return scratch_by_block.setdefault(id(block), _StreamingToolCallScratch())

            def get_custom_tool_call_input(block: ToolCall) -> str:
                custom_input = scratch_of(block).customInput
                if custom_input is None:
                    return ""
                value = block.arguments.get(custom_input.property)
                return value if isinstance(value, str) else ""

            def append_custom_tool_call_input(block: ToolCall, next_input: str, close: bool) -> str | None:
                custom_input = scratch_of(block).customInput
                if custom_input is None:
                    return None
                delta = append_grammar_tool_input_json_delta(
                    custom_input.jsonBuffer, custom_input.property, next_input, close
                )
                block.arguments = {custom_input.property: next_input}
                return delta

            def finish_block(block: Any) -> None:
                content_index = get_content_index(block)
                if content_index == -1:
                    return
                if block.type == "text":
                    stream.push(TextEndEvent(contentIndex=content_index, content=block.text, partial=output))
                elif block.type == "thinking":
                    apply_streamed_reasoning_details(block)
                    stream.push(ThinkingEndEvent(contentIndex=content_index, content=block.thinking, partial=output))
                elif block.type == "toolCall":
                    scratch = scratch_of(block)
                    if scratch.customInput is not None:
                        delta = append_custom_tool_call_input(block, get_custom_tool_call_input(block), True)
                        if delta is not None:
                            stream.push(ToolCallDeltaEvent(contentIndex=content_index, delta=delta, partial=output))
                    elif scratch.partialArgs is not None and scratch.partialArgs.raw:
                        # MISAKA fork: final arguments go through `finish_into`, which records a
                        # malformed object as `argumentsError` instead of running a repaired prefix.
                        scratch.partialArgs.finish_into(block)
                    else:
                        block.arguments = {}
                    # Finalize in-place and strip the scratch buffers so replay only
                    # carries parsed arguments.
                    scratch_by_block.pop(id(block), None)
                    stream.push(ToolCallEndEvent(contentIndex=content_index, toolCall=block, partial=output))

            def ensure_text_block() -> TextContent:
                nonlocal text_block
                if text_block is None:
                    text_block = TextContent(text="")
                    blocks.append(text_block)
                    stream.push(TextStartEvent(contentIndex=get_content_index(text_block), partial=output))
                return text_block

            def ensure_thinking_block(thinking_signature: str) -> ThinkingContent:
                nonlocal thinking_block
                if thinking_block is None:
                    thinking_block = ThinkingContent(thinking="", thinkingSignature=thinking_signature)
                    blocks.append(thinking_block)
                    stream.push(ThinkingStartEvent(contentIndex=get_content_index(thinking_block), partial=output))
                return thinking_block

            def ensure_tool_call_block(tool_call: Mapping[str, Any]) -> ToolCall:
                stream_index = tool_call.get("index") if isinstance(tool_call.get("index"), int) else None
                function = tool_call.get("function") if isinstance(tool_call.get("function"), Mapping) else None
                custom = tool_call.get("custom") if isinstance(tool_call.get("custom"), Mapping) else None
                name = (function or {}).get("name") or (custom or {}).get("name") or ""
                tool_call_id = tool_call.get("id")
                block = tool_call_blocks_by_index.get(stream_index) if stream_index is not None else None
                if block is None and tool_call_id:
                    block = tool_call_blocks_by_id.get(tool_call_id)
                if block is None:
                    # Note: the "input" fallback here should/must not be taken.  in case the LLM makes up
                    # a tool we don't knwo about, we at least have a place to stash our stuff.
                    custom_input_property = (
                        (grammar_tool_input_properties.get(name) or "input") if custom and not function else None
                    )
                    has_custom_input = custom_input_property is not None
                    block = ToolCall(
                        id=tool_call_id or "",
                        name=name,
                        arguments={custom_input_property: ""} if has_custom_input else {},
                    )
                    scratch_by_block[id(block)] = _StreamingToolCallScratch(
                        partialArgs=None if has_custom_input else StreamingArgs(),
                        customInput=(
                            _CustomToolCallInput(property=custom_input_property, jsonBuffer=GrammarToolInputJsonBuffer())
                            if has_custom_input
                            else None
                        ),
                        streamIndex=stream_index,
                    )
                    if stream_index is not None:
                        tool_call_blocks_by_index[stream_index] = block
                    if tool_call_id:
                        tool_call_blocks_by_id[tool_call_id] = block
                    blocks.append(block)
                    stream.push(ToolCallStartEvent(contentIndex=get_content_index(block), partial=output))
                scratch = scratch_of(block)
                if stream_index is not None and scratch.streamIndex is None:
                    scratch.streamIndex = stream_index
                    tool_call_blocks_by_index[stream_index] = block
                if tool_call_id:
                    tool_call_blocks_by_id[tool_call_id] = block
                if not block.name and name:
                    block.name = name
                if custom and not function and scratch.customInput is None:
                    custom_input_property = grammar_tool_input_properties.get(block.name) or "input"
                    block.arguments = {custom_input_property: ""}
                    scratch.customInput = _CustomToolCallInput(
                        property=custom_input_property, jsonBuffer=GrammarToolInputJsonBuffer()
                    )
                    scratch.partialArgs = None
                return block

            on_provider_stream_event = _option(options, "onProviderStreamEvent")
            # MISAKA fork: kept so the finally below closes it with the reply (GitHub issue #6).
            chunks = _iterate_stream(openai_stream, _option(options, "signal"))
            async for chunk in chunks:
                if callable(on_provider_stream_event):
                    await maybe_await(on_provider_stream_event(chunk, model))
                if not isinstance(chunk, Mapping):
                    continue

                # OpenAI documents ChatCompletionChunk.id as the unique chat completion identifier,
                # and each chunk in a streamed completion carries the same id.
                output.responseId = output.responseId or chunk.get("id")
                chunk_model = chunk.get("model")
                if isinstance(chunk_model, str) and len(chunk_model) > 0 and chunk_model != model.id:
                    output.responseModel = output.responseModel or chunk_model
                if chunk.get("usage"):
                    output.usage = parse_chunk_usage(chunk["usage"], model)

                choices = chunk.get("choices")
                choice = choices[0] if isinstance(choices, list) and choices else None
                if not isinstance(choice, Mapping):
                    continue

                # Fallback: some providers (e.g., Moonshot) return usage
                # in choice.usage instead of the standard chunk.usage
                if not chunk.get("usage") and choice.get("usage"):
                    output.usage = parse_chunk_usage(choice["usage"], model)

                if choice.get("finish_reason"):
                    output.rawStopReason = choice["finish_reason"]
                    finish_reason_result = map_stop_reason(choice["finish_reason"])
                    output.stopReason = finish_reason_result["stopReason"]
                    if finish_reason_result.get("errorMessage"):
                        output.errorMessage = finish_reason_result["errorMessage"]
                    has_finish_reason = True

                delta = choice.get("delta")
                if isinstance(delta, Mapping):
                    content = delta.get("content")
                    if isinstance(content, str) and len(content) > 0:
                        block = ensure_text_block()
                        block.text += content
                        stream.push(TextDeltaEvent(contentIndex=get_content_index(block), delta=content, partial=output))

                    # Some endpoints return reasoning in reasoning_content (llama.cpp),
                    # or reasoning (other openai compatible endpoints)
                    # Use the first non-empty reasoning field to avoid duplication
                    # (e.g., chutes.ai returns both reasoning_content and reasoning with same content)
                    found_reasoning_field = next(
                        (
                            field
                            for field in ("reasoning_content", "reasoning", "reasoning_text")
                            if isinstance(delta.get(field), str) and len(delta[field]) > 0
                        ),
                        None,
                    )

                    if found_reasoning_field:
                        reasoning_delta = delta[found_reasoning_field]
                        thinking_signature = (
                            "reasoning_content"
                            if model.provider == "opencode-go" and found_reasoning_field == "reasoning"
                            else found_reasoning_field
                        )
                        block = ensure_thinking_block(thinking_signature)
                        block.thinking += reasoning_delta
                        stream.push(ThinkingDeltaEvent(
                            contentIndex=get_content_index(block), delta=reasoning_delta, partial=output
                        ))

                    if delta.get("tool_calls"):
                        for tool_call in delta["tool_calls"]:
                            if not isinstance(tool_call, Mapping):
                                continue
                            block = ensure_tool_call_block(tool_call)
                            if not block.id and tool_call.get("id"):
                                block.id = tool_call["id"]
                                tool_call_blocks_by_id[tool_call["id"]] = block
                            function = tool_call.get("function") if isinstance(tool_call.get("function"), Mapping) else {}
                            custom = tool_call.get("custom") if isinstance(tool_call.get("custom"), Mapping) else {}
                            name = function.get("name") or custom.get("name")
                            if not block.name and name:
                                block.name = name

                            tool_delta = ""
                            scratch = scratch_of(block)
                            if function.get("arguments"):
                                tool_delta = function["arguments"]
                                if scratch.partialArgs is None:
                                    scratch.partialArgs = StreamingArgs()
                                scratch.partialArgs.append(tool_delta)
                                block.arguments = scratch.partialArgs.arguments
                            elif custom.get("input"):
                                next_input = get_custom_tool_call_input(block) + custom["input"]
                                tool_delta = append_custom_tool_call_input(block, next_input, False) or ""
                            stream.push(ToolCallDeltaEvent(
                                contentIndex=get_content_index(block), delta=tool_delta, partial=output
                            ))

                    reasoning_details = delta.get("reasoning_details")
                    if isinstance(reasoning_details, list):
                        for detail in reasoning_details:
                            if not is_openai_reasoning_detail(detail):
                                continue
                            ensure_thinking_block("")
                            if streamed_reasoning_details is None:
                                streamed_reasoning_details = []
                            # Keep provider replay data in the existing signature slot. OpenRouter streams
                            # reasoning_details as deltas: consecutive text/summary deltas are merged into
                            # logical entries, while encrypted entries remain opaque and discrete.
                            append_openai_reasoning_detail(streamed_reasoning_details, detail)

            for block in list(blocks):
                finish_block(block)
            if signal_aborted(_option(options, "signal")):
                raise RuntimeError("Request was aborted")

            if output.stopReason == "aborted":
                raise RuntimeError("Request was aborted")
            if not has_finish_reason and not compat["supportsFinishReason"]:
                output.stopReason = "toolUse" if any(block.type == "toolCall" for block in output.content) else "stop"
            if output.stopReason == "error":
                raise RuntimeError(output.errorMessage or "Provider returned an error stop reason")
            if (compat["supportsFinishReason"] and not has_finish_reason) or output.stopReason == "pending":
                raise RuntimeError("Stream ended without finish_reason")

            stream.push(DoneEvent(reason=output.stopReason, message=output))
        except Exception as error:  # noqa: BLE001
            for block in output.content:
                if block.type == "thinking":
                    apply_streamed_reasoning_details(block)
            output.stopReason = "aborted" if signal_aborted(_option(options, "signal")) else "error"
            output.errorMessage = format_provider_error(normalize_provider_error(error))
            # Some providers via OpenRouter give additional information in this field.
            # normalize_provider_error already stringifies the parsed body (error.error)
            # into errorMessage, so only append the raw metadata when it is not already
            # present to avoid double-printing it.
            error_payload = getattr(error, "error", None)
            metadata = read_field(error_payload, "metadata") if error_payload is not None else None
            raw_metadata = read_field(metadata, "raw") if metadata is not None else None
            if raw_metadata and str(raw_metadata) not in output.errorMessage:
                output.errorMessage += f"\n{raw_metadata}"
            stream.push(ErrorEvent(reason=output.stopReason, error=output), cause=error)
        finally:
            if chunks is not None:          # closed now, its abort task with it, not when GC finds it
                with contextlib.suppress(Exception):
                    await chunks.aclose()
            stream.end()

    spawn_stream_task(run(), stream=stream)
    return stream


def stream_simple_openai_completions(
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

    return stream_openai_completions(
        model,
        context,
        {
            **base,
            "reasoningEffort": reasoning_effort,
            "thinkingBudgets": _option(options, "thinkingBudgets"),
        },
    )


def create_client(
    model: Model,
    context: TranscriptContext,
    api_key: str,
    options_headers: Mapping[str, Any] | None = None,
    session_id: str | None = None,
    compat: Mapping[str, Any] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Returns the client and (MISAKA fork, see ``split_sdk_headers``) the headers every
    request has to carry."""
    compat = compat or get_compat(model)
    headers: dict[str, Any] = {"User-Agent": get_misaka_user_agent(), **(model.headers or {})}
    if model.provider == "github-copilot":
        has_images = has_copilot_vision_input(context.messages)
        copilot_headers = build_copilot_dynamic_headers(messages=context.messages, hasImages=has_images)
        headers.update(copilot_headers)

    if session_id and compat.get("sendSessionAffinityHeaders"):
        if compat.get("sessionAffinityFormat") == "openrouter":
            headers["x-session-id"] = session_id
        else:
            if compat.get("sessionAffinityFormat") == "openai":
                headers["session_id"] = session_id
            headers["x-client-request-id"] = session_id
            headers["x-session-affinity"] = session_id

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
    cache_retention: CacheRetention | None = None,
    grammar_tool_input_properties: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    compat = compat or get_compat(model)
    if cache_retention is None:
        cache_retention = resolve_cache_retention(_option(options, "cacheRetention"), _option(options, "env"))
    if grammar_tool_input_properties is None:
        grammar_tool_input_properties = create_grammar_tool_input_properties(
            get_declared_tools(context.messages), bool(compat.get("supportsOpenAIGrammarTools"))
        )
    transcript_tools = resolve_transcript_tools(
        context.messages,
        compat.get("supportsMidConvoSystemMessages") is True and compat.get("supportsMidConvoToolAdditions") is True,
    )
    messages = convert_messages(model, context, compat, {"grammarToolInputProperties": grammar_tool_input_properties})
    cache_control = get_compat_cache_control(compat, cache_retention)

    params: dict[str, Any] = {
        "model": model.id,
        "messages": messages,
        "stream": True,
        "prompt_cache_key": (
            clamp_openai_prompt_cache_key(_option(options, "sessionId"))
            if ("api.openai.com" in model.baseUrl and cache_retention != "none")
            or (cache_retention == "long" and compat.get("supportsLongCacheRetention"))
            else None
        ),
        "prompt_cache_retention": (
            "24h" if cache_retention == "long" and compat.get("supportsLongCacheRetention") else None
        ),
    }

    if compat.get("supportsUsageInStreaming") is not False:
        params["stream_options"] = {"include_usage": True}

    if compat.get("supportsStore"):
        params["store"] = False

    if _option(options, "maxTokens"):
        if compat.get("maxTokensField") == "max_tokens":
            # Deprecated by OpenAI, but some OpenAI-compatible providers only accept max_tokens.
            params["max_tokens"] = _option(options, "maxTokens")
        else:
            params["max_completion_tokens"] = _option(options, "maxTokens")

    if _option(options, "temperature") is not None:
        params["temperature"] = _option(options, "temperature")

    if len(transcript_tools.requestTools) > 0:
        params["tools"] = convert_tools(transcript_tools.requestTools, compat)
        if compat.get("zaiToolStream"):
            params["tool_stream"] = True
    elif has_tool_history(context.messages):
        # Anthropic (via LiteLLM/proxy) requires tools param when conversation has tool_calls/tool_results
        params["tools"] = []

    if cache_control:
        apply_anthropic_cache_control(messages, params.get("tools"), cache_control)

    if _option(options, "toolChoice"):
        params["tool_choice"] = _option(options, "toolChoice")

    if compat.get("vllmPriority") is not None:
        params["priority"] = compat["vllmPriority"]

    reasoning_effort = _option(options, "reasoningEffort")
    thinking_level_map = model.thinkingLevelMap or {}
    thinking_token_budget_field = resolve_thinking_token_budget_field(compat)
    thinking_budget = resolve_clamped_thinking_budget(model, options, params)

    if compat.get("thinkingFormat") == "zai" and model.reasoning:
        params["thinking"] = {"type": "enabled", "clear_thinking": False} if reasoning_effort else {"type": "disabled"}
        if reasoning_effort and compat.get("supportsReasoningEffort"):
            effort = thinking_level_map.get(reasoning_effort, reasoning_effort)
            if isinstance(effort, str):
                params["reasoning_effort"] = effort
    elif compat.get("thinkingFormat") == "qwen" and model.reasoning:
        params["enable_thinking"] = bool(reasoning_effort)
        if reasoning_effort and compat.get("supportsReasoningEffort"):
            effort = _coalesce(thinking_level_map.get(reasoning_effort), reasoning_effort)
            if isinstance(effort, str):
                params["reasoning_effort"] = effort
    elif compat.get("thinkingFormat") == "qwen-chat-template" and model.reasoning:
        params["chat_template_kwargs"] = {
            "enable_thinking": bool(reasoning_effort),
            "preserve_thinking": True,
        }
    elif compat.get("thinkingFormat") == "chat-template" and model.reasoning:
        chat_template_kwargs = build_chat_template_values(
            model, options, compat.get("chatTemplateKwargs") or {}, thinking_budget
        )
        if chat_template_kwargs:
            params["chat_template_kwargs"] = chat_template_kwargs
    elif compat.get("thinkingFormat") == "baseten" and model.reasoning:
        chat_template_args = build_chat_template_values(
            model, options, compat.get("chatTemplateArgs") or {}, thinking_budget
        )
        if chat_template_args:
            params["chat_template_args"] = chat_template_args
        if compat.get("supportsReasoningEffort"):
            key = reasoning_effort if reasoning_effort else "off"
            mapped_effort = thinking_level_map.get(key, _UNDEFINED)
            effort = reasoning_effort if mapped_effort is _UNDEFINED else mapped_effort
            if isinstance(effort, str):
                params["reasoning_effort"] = effort
    elif compat.get("thinkingFormat") == "deepseek" and model.reasoning:
        if reasoning_effort:
            params["thinking"] = {"type": "enabled"}
        elif _thinking_off(model) is not None:
            params["thinking"] = {"type": "disabled"}
        if reasoning_effort and compat.get("supportsReasoningEffort"):
            params["reasoning_effort"] = _coalesce(thinking_level_map.get(reasoning_effort), reasoning_effort)
    elif compat.get("thinkingFormat") == "openrouter" and model.reasoning:
        # OpenRouter normalizes reasoning across providers via a nested reasoning object.
        if reasoning_effort:
            params["reasoning"] = {"effort": _coalesce(thinking_level_map.get(reasoning_effort), reasoning_effort)}
        elif _thinking_off(model) is not None:
            params["reasoning"] = {"effort": _coalesce(thinking_level_map.get("off"), "none")}
    elif compat.get("thinkingFormat") == "ant-ling" and model.reasoning and reasoning_effort:
        effort = thinking_level_map.get(reasoning_effort)
        if isinstance(effort, str):
            params["reasoning"] = {"effort": effort}
    elif compat.get("thinkingFormat") == "together" and model.reasoning:
        params["reasoning"] = {"enabled": bool(reasoning_effort)}
        if reasoning_effort and compat.get("supportsReasoningEffort"):
            params["reasoning_effort"] = _coalesce(thinking_level_map.get(reasoning_effort), reasoning_effort)
    elif compat.get("thinkingFormat") == "string-thinking" and model.reasoning:
        if reasoning_effort:
            params["thinking"] = _coalesce(thinking_level_map.get(reasoning_effort), reasoning_effort)
        elif _thinking_off(model) is not None:
            params["thinking"] = _coalesce(thinking_level_map.get("off"), "none")
    elif reasoning_effort and model.reasoning and compat.get("supportsReasoningEffort"):
        # OpenAI-style reasoning_effort
        params["reasoning_effort"] = _coalesce(thinking_level_map.get(reasoning_effort), reasoning_effort)
    elif not reasoning_effort and model.reasoning and compat.get("supportsReasoningEffort"):
        off_value = thinking_level_map.get("off")
        if isinstance(off_value, str):
            params["reasoning_effort"] = off_value

    # Cap reasoning with a top-level budget field. Independent of thinkingFormat: the
    # same server can serve zai, qwen or chat-template models. Reasoning and the answer
    # share max_tokens here, so an uncapped reasoning phase can consume the whole
    # response and leave no answer and no tool call.
    if thinking_token_budget_field and thinking_budget is not None:
        params[thinking_token_budget_field] = thinking_budget

    # OpenRouter provider routing preferences
    open_router_routing = getattr(model.compat, "openRouterRouting", None)
    if open_router_routing is not None:
        params["provider"] = _dump_model(open_router_routing)

    # Vercel AI Gateway provider routing preferences
    vercel_gateway_routing = getattr(model.compat, "vercelGatewayRouting", None)
    if vercel_gateway_routing is not None:
        routing = _dump_model(vercel_gateway_routing)
        if routing.get("only") or routing.get("order"):
            gateway_options: dict[str, list[str]] = {}
            if routing.get("only"):
                gateway_options["only"] = routing["only"]
            if routing.get("order"):
                gateway_options["order"] = routing["order"]
            params["providerOptions"] = {"gateway": gateway_options}

    # Last so custom keys override the named request fields. Per-request keys override model defaults.
    params.update(model.samplingParams or {})
    params.update(_option(options, "samplingParams") or {})

    return params


_UNDEFINED = object()


def _coalesce(value: Any, fallback: Any) -> Any:
    """``value ?? fallback``."""
    return fallback if value is None else value


def _thinking_off(model: Model) -> Any:
    """``model.thinkingLevelMap?.off``, with a missing key or map as ``_UNDEFINED`` so the
    callers' ``!== null`` test tells an explicit null from an absent entry."""
    if model.thinkingLevelMap is None or "off" not in model.thinkingLevelMap:
        return _UNDEFINED
    return model.thinkingLevelMap["off"]


def _dump_model(value: Any) -> Any:
    return value.model_dump(exclude_none=True) if hasattr(value, "model_dump") else value


def resolve_thinking_token_budget_field(compat: Mapping[str, Any]) -> str | None:
    if compat.get("thinkingTokenBudgetField"):
        return compat["thinkingTokenBudgetField"]
    if compat.get("supportsThinkingTokenBudget"):
        return "thinking_token_budget"
    return None


def resolve_clamped_thinking_budget(model: Model, options: Any, params: Mapping[str, Any]) -> int | None:
    reasoning_effort = _option(options, "reasoningEffort")
    if not reasoning_effort or not model.reasoning:
        return None
    ceiling = _coalesce(params.get("max_tokens"), _coalesce(params.get("max_completion_tokens"), model.maxTokens))
    budget = clamp_thinking_budget_to_answer_room(
        thinking_budget_for_level(reasoning_effort, _option(options, "thinkingBudgets")),
        ceiling,
    )
    return budget if budget > 0 else None


def build_chat_template_values(
    model: Model,
    options: Any,
    values: Mapping[str, Any],
    thinking_budget: int | None = None,
) -> dict[str, Any] | None:
    resolved_values: dict[str, Any] = {}

    for key, value in values.items():
        resolved = resolve_chat_template_kwarg_value(model, options, value, thinking_budget)
        if resolved is not _UNDEFINED:
            resolved_values[key] = resolved

    return resolved_values or None


def resolve_chat_template_kwarg_value(
    model: Model,
    options: Any,
    value: Any,
    thinking_budget: int | None = None,
) -> Any:
    """``_UNDEFINED`` for "do not send this key" -- distinct from ``None``, which a template
    may legitimately want."""
    if not isinstance(value, Mapping):
        return value

    reasoning_effort = _option(options, "reasoningEffort")
    if not reasoning_effort and value.get("omitWhenOff"):
        return _UNDEFINED
    if value.get("$var") == "thinking.enabled":
        return bool(reasoning_effort)
    if value.get("$var") == "thinking.budget":
        return _UNDEFINED if thinking_budget is None else thinking_budget

    thinking_level_map = model.thinkingLevelMap or {}
    key = reasoning_effort if reasoning_effort else "off"
    mapped_value = thinking_level_map.get(key, _UNDEFINED)
    if mapped_value is _UNDEFINED:
        return reasoning_effort if reasoning_effort else _UNDEFINED
    return mapped_value if isinstance(mapped_value, str) else _UNDEFINED


def get_compat_cache_control(compat: Mapping[str, Any], cache_retention: CacheRetention) -> dict[str, Any] | None:
    if compat.get("cacheControlFormat") != "anthropic" or cache_retention == "none":
        return None

    ttl = "1h" if cache_retention == "long" and compat.get("supportsLongCacheRetention") else None
    return {"type": "ephemeral", **({"ttl": ttl} if ttl else {})}


def apply_anthropic_cache_control(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    cache_control: dict[str, Any],
) -> None:
    add_cache_control_to_system_prompt(messages, cache_control)
    add_cache_control_to_last_tool(tools, cache_control)
    add_cache_control_to_last_conversation_message(messages, cache_control)


def add_cache_control_to_system_prompt(messages: list[dict[str, Any]], cache_control: dict[str, Any]) -> None:
    for message in messages:
        if message.get("role") in {"system", "developer"}:
            add_cache_control_to_text_content(message, cache_control)
            return


def add_cache_control_to_last_conversation_message(messages: list[dict[str, Any]], cache_control: dict[str, Any]) -> None:
    for message in reversed(messages):
        if message.get("role") in {"user", "assistant", "tool"} and add_cache_control_to_text_content(
            message, cache_control
        ):
            return


def add_cache_control_to_last_tool(tools: list[dict[str, Any]] | None, cache_control: dict[str, Any]) -> None:
    if not tools:
        return
    tools[-1]["cache_control"] = cache_control


def add_cache_control_to_text_content(message: dict[str, Any], cache_control: dict[str, Any]) -> bool:
    content = message.get("content")
    if isinstance(content, str):
        if len(content) == 0:
            return False
        message["content"] = [{"type": "text", "text": content, "cache_control": cache_control}]
        return True

    if not isinstance(content, list):
        return False

    for part in reversed(content):
        if isinstance(part, dict) and part.get("type") == "text":
            part["cache_control"] = cache_control
            return True

    return False


_TOOL_CALL_ID_DISALLOWED = re.compile(r"[^a-zA-Z0-9_-]")


def convert_messages(
    model: Model,
    context: TranscriptContext,
    compat: Mapping[str, Any],
    options: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    normalized_context = resolve_transcript(context, compat.get("supportsMidConvoSystemMessages"))
    params: list[dict[str, Any]] = []
    grammar_tool_input_properties = (options or {}).get("grammarToolInputProperties") or {}

    def normalize_tool_call_id(tool_call_id: str, _target_model: Any = None, _source: Any = None) -> str:
        # Handle pipe-separated IDs from OpenAI Responses API
        # Format: {call_id}|{id} where {id} can be 400+ chars with special chars (+, /, =)
        # These come from providers like github-copilot, openai-codex, opencode
        # Extract just the call_id part and normalize it
        # Multiple tool calls in the same turn can share call_id but differ by item_id.
        # Preserve item-level uniqueness when replaying into Chat Completions, which
        # requires distinct tool call ids.
        if "|" in tool_call_id:
            # Sanitize to allowed chars and truncate to 40 chars (OpenAI limit)
            separator_index = tool_call_id.index("|")
            call_id = _TOOL_CALL_ID_DISALLOWED.sub("_", tool_call_id[:separator_index])
            item_id = _TOOL_CALL_ID_DISALLOWED.sub("_", tool_call_id[separator_index + 1 :])
            combined_id = f"{call_id}_{item_id}" if len(item_id) > 0 else call_id
            if len(combined_id) <= 40:
                return combined_id
            hash_value = short_hash(tool_call_id)[:8]
            prefix = call_id[: max(1, 40 - len(hash_value) - 1)]
            return f"{prefix}_{hash_value}"

        if model.provider == "openai":
            return tool_call_id[:40] if len(tool_call_id) > 40 else tool_call_id
        return tool_call_id

    transformed_messages = transform_messages(normalized_context.messages, model, normalize_tool_call_id)
    transcript_tools = resolve_transcript_tools(
        normalized_context.messages,
        compat.get("supportsMidConvoSystemMessages") is True and compat.get("supportsMidConvoToolAdditions") is True,
    )
    instruction_role = "developer" if model.reasoning and compat.get("supportsDeveloperRole") else "system"

    last_role: str | None = None

    i = 0
    while i < len(transformed_messages):
        msg = transformed_messages[i]
        # Some providers don't allow user messages directly after tool results
        # Insert a synthetic assistant message to bridge the gap
        if compat.get("requiresAssistantAfterToolResult") and last_role == "toolResult" and msg.role == "user":
            params.append({"role": "assistant", "content": "I have processed the tool results."})

        if msg.role == "system":
            added_tools = (msg.toolsAdded or []) if i > 0 and transcript_tools.anchorsAdditions else []
            if len(added_tools) > 0:
                kimi_tool_message = {"role": "system", "tools": convert_tools(added_tools, compat)}
                params.append(kimi_tool_message)
            text = get_system_message_text(msg) if i == 0 else render_system_message_update(msg)
            if len(text) > 0:
                params.append({"role": instruction_role, "content": sanitize_surrogates(text)})
        elif msg.role == "user":
            if isinstance(msg.content, str):
                params.append({"role": "user", "content": sanitize_surrogates(msg.content)})
            else:
                content = [
                    {"type": "text", "text": sanitize_surrogates(item.text)}
                    if item.type == "text"
                    else {"type": "image_url", "image_url": {"url": f"data:{item.mimeType};base64,{item.data}"}}
                    for item in msg.content
                    if item.type != "text" or len(item.text) > 0
                ]
                if len(content) == 0:
                    i += 1
                    continue
                params.append({"role": "user", "content": content})
        elif msg.role == "assistant":
            # Some providers don't accept null content, use empty string instead
            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": "" if compat.get("requiresAssistantAfterToolResult") else None,
            }

            assistant_text_parts = [
                {"type": "text", "text": sanitize_surrogates(block.text)}
                for block in msg.content
                if block.type == "text" and len(block.text.strip()) > 0
            ]
            assistant_text = "".join(part["text"] for part in assistant_text_parts)

            thinking_blocks = [block for block in msg.content if block.type == "thinking"]
            tool_calls = [block for block in msg.content if block.type == "toolCall"]
            signed_reasoning_details = next(
                (
                    details
                    for details in (parse_openai_reasoning_details(block.thinkingSignature) for block in thinking_blocks)
                    if details is not None
                ),
                None,
            )
            legacy_reasoning_details = [
                detail
                for detail in (
                    parse_legacy_encrypted_reasoning_detail(tool_call.thoughtSignature) for tool_call in tool_calls
                )
                if detail is not None
            ]
            preserved_reasoning_details = signed_reasoning_details or (legacy_reasoning_details or None)

            non_empty_thinking_blocks = [block for block in thinking_blocks if len(block.thinking.strip()) > 0]
            if len(non_empty_thinking_blocks) > 0:
                if compat.get("requiresThinkingAsText"):
                    # Convert thinking blocks to plain text (no tags to avoid model mimicking them)
                    thinking_text = "\n\n".join(sanitize_surrogates(block.thinking) for block in non_empty_thinking_blocks)
                    assistant_msg["content"] = [{"type": "text", "text": thinking_text}, *assistant_text_parts]
                else:
                    # Always send assistant content as a plain string (OpenAI Chat Completions
                    # API standard format). Sending as an array of {type:"text", text:"..."}
                    # objects is non-standard and causes some models (e.g. DeepSeek V3.2 via
                    # NVIDIA NIM) to mirror the content-block structure literally in their
                    # output, producing recursive nesting like [{'type':'text','text':'[{...}]'}].
                    if len(assistant_text) > 0:
                        assistant_msg["content"] = assistant_text

                    # reasoning_details is the structured alternative to a raw reasoning field.
                    if not preserved_reasoning_details:
                        # Use the signature from the first thinking block if available (for llama.cpp server + gpt-oss)
                        signature = non_empty_thinking_blocks[0].thinkingSignature
                        if model.provider == "opencode-go" and signature == "reasoning":
                            signature = "reasoning_content"
                        if signature and is_openai_completions_reasoning_field(signature):
                            assistant_msg[signature] = "\n".join(block.thinking for block in non_empty_thinking_blocks)
            elif len(assistant_text) > 0:
                # Always send assistant content as a plain string (see above).
                assistant_msg["content"] = assistant_text

            if len(tool_calls) > 0:
                assistant_msg["tool_calls"] = []
                for tool_call in tool_calls:
                    custom_input_property = grammar_tool_input_properties.get(tool_call.name)
                    if custom_input_property is not None:
                        assistant_msg["tool_calls"].append({
                            "id": tool_call.id,
                            "type": "custom",
                            "custom": {
                                "name": tool_call.name,
                                "input": sanitize_surrogates(
                                    get_grammar_tool_input(tool_call.name, tool_call.arguments, custom_input_property)
                                ),
                            },
                        })
                    else:
                        assistant_msg["tool_calls"].append({
                            "id": tool_call.id,
                            "type": "function",
                            "function": {
                                "name": tool_call.name,
                                "arguments": json.dumps(tool_call.arguments, separators=(",", ":"), ensure_ascii=False),
                            },
                        })
            if preserved_reasoning_details:
                assistant_msg["reasoning_details"] = preserved_reasoning_details
            if (
                compat.get("requiresReasoningContentOnAssistantMessages")
                and model.reasoning
                and "reasoning_content" not in assistant_msg
            ):
                assistant_msg["reasoning_content"] = ""
            # Skip assistant messages that have no content and no tool calls.
            # Some providers require "either content or tool_calls, but not none".
            # Other providers also don't accept empty assistant messages.
            # This handles aborted assistant responses that got no content.
            content = assistant_msg["content"]
            has_content = content is not None and len(content) > 0
            if not has_content and "tool_calls" not in assistant_msg:
                i += 1
                continue
            params.append(assistant_msg)
        elif msg.role == "toolResult":
            image_blocks: list[dict[str, Any]] = []
            j = i

            while j < len(transformed_messages) and transformed_messages[j].role == "toolResult":
                tool_msg: ToolResultMessage = transformed_messages[j]

                # Extract text and image content
                text_result = "\n".join(block.text for block in tool_msg.content if block.type == "text")
                has_images = any(block.type == "image" for block in tool_msg.content)

                # Always send tool result with text (or placeholder if only images)
                has_text = len(text_result) > 0
                tool_result_text = text_result if has_text else "(see attached image)" if has_images else "(no tool output)"
                # Some providers require the 'name' field in tool results
                tool_result_msg: dict[str, Any] = {
                    "role": "tool",
                    "content": sanitize_surrogates(tool_result_text),
                    "tool_call_id": tool_msg.toolCallId,
                }
                if compat.get("requiresToolResultName") and tool_msg.toolName:
                    tool_result_msg["name"] = tool_msg.toolName
                params.append(tool_result_msg)

                if has_images and "image" in model.input:
                    for block in tool_msg.content:
                        if block.type == "image":
                            image_blocks.append({
                                "type": "image_url",
                                "image_url": {"url": f"data:{block.mimeType};base64,{block.data}"},
                            })
                j += 1

            i = j - 1

            if len(image_blocks) > 0:
                if compat.get("requiresAssistantAfterToolResult"):
                    params.append({"role": "assistant", "content": "I have processed the tool results."})

                params.append({
                    "role": "user",
                    "content": [{"type": "text", "text": "Attached image(s) from tool result:"}, *image_blocks],
                })
                last_role = "user"
            else:
                last_role = "toolResult"

            i += 1
            continue

        last_role = msg.role
        i += 1

    return params


def convert_tools(tools: list[Tool], compat: Mapping[str, Any]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for tool in tools:
        grammar = resolve_grammar_constrained_sampling(tool, bool(compat.get("supportsOpenAIGrammarTools")))
        if grammar:
            converted.append({
                "type": "custom",
                "custom": {
                    "name": tool.name,
                    "description": tool.description,
                    "format": {
                        "type": "grammar",
                        "grammar": {"syntax": grammar.format, "definition": grammar.definition},
                    },
                },
            })
            continue

        strict = resolve_json_schema_strict_sampling(tool, compat.get("supportsStrictMode") is not False)
        converted.append({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": get_json_schema_tool_parameters(tool, strict),
                # Only include strict if provider supports it. Some reject unknown fields.
                **({"strict": _coalesce(strict, False)} if compat.get("supportsStrictMode") is not False else {}),
            },
        })
    return converted


def parse_chunk_usage(raw_usage: Mapping[str, Any], model: Model) -> Usage:
    prompt_tokens = raw_usage.get("prompt_tokens") or 0
    prompt_details = raw_usage.get("prompt_tokens_details") or {}
    cache_read_tokens = _coalesce(
        prompt_details.get("cached_tokens"),
        _coalesce(raw_usage.get("prompt_cache_hit_tokens"), _coalesce(raw_usage.get("cached_tokens"), 0)),
    )
    cache_write_tokens = prompt_details.get("cache_write_tokens") or 0

    # Follow documented OpenAI/OpenRouter semantics: cached_tokens is cache-read
    # tokens (hits). Providers disagree on placement: OpenAI/OpenRouter use
    # prompt_tokens_details.cached_tokens, DeepSeek uses prompt_cache_hit_tokens,
    # and Kimi documents top-level usage.cached_tokens on the final usage chunk.
    # OpenAI does not document or emit cache_write_tokens, but
    # OpenRouter-compatible providers can include it as a separate write count.
    # Do not subtract writes from cached_tokens, otherwise spec-compliant
    # providers are under-reported.
    input_tokens = max(0, prompt_tokens - cache_read_tokens - cache_write_tokens)
    # OpenAI completion_tokens already includes reasoning_tokens.
    output_tokens = raw_usage.get("completion_tokens") or 0
    usage = Usage(
        input=input_tokens,
        output=output_tokens,
        cacheRead=cache_read_tokens,
        cacheWrite=cache_write_tokens,
        reasoning=(raw_usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0,
        totalTokens=input_tokens + output_tokens + cache_read_tokens + cache_write_tokens,
        cost=UsageCost(input=0, output=0, cacheRead=0, cacheWrite=0, total=0),
    )
    calculate_cost(model, usage)
    return usage


def map_stop_reason(reason: str | None) -> dict[str, str]:
    if reason is None:
        return {"stopReason": "stop"}
    if reason in {"stop", "end"}:
        return {"stopReason": "stop"}
    if reason == "length":
        return {"stopReason": "length"}
    if reason in {"function_call", "tool_calls"}:
        return {"stopReason": "toolUse"}
    if reason == "content_filter":
        return {"stopReason": "error", "errorMessage": "Provider finish_reason: content_filter"}
    if reason == "network_error":
        return {"stopReason": "error", "errorMessage": "Provider finish_reason: network_error"}
    return {"stopReason": "error", "errorMessage": f"Provider finish_reason: {reason}"}


def detect_compat(model: Model) -> dict[str, Any]:
    """Auto-detect compatibility settings from provider name and baseUrl.

    Used as the base when model.compat is not set; explicit model.compat entries override
    these detected values.
    """
    provider = model.provider
    base_url = model.baseUrl

    is_zai = provider in {"zai", "zai-coding-cn"} or "api.z.ai" in base_url or "open.bigmodel.cn" in base_url
    is_together = provider == "together" or "api.together.ai" in base_url or "api.together.xyz" in base_url
    is_moonshot = provider in {"moonshotai", "moonshotai-cn"} or "api.moonshot." in base_url
    is_open_router = provider == "openrouter" or "openrouter.ai" in base_url
    is_cloudflare_workers_ai = provider == "cloudflare-workers-ai" or "api.cloudflare.com" in base_url
    is_cloudflare_ai_gateway = provider == "cloudflare-ai-gateway" or "gateway.ai.cloudflare.com" in base_url
    is_nvidia = provider == "nvidia" or "integrate.api.nvidia.com" in base_url
    is_ant_ling = provider == "ant-ling" or "api.ant-ling.com" in base_url
    is_cerebras = provider == "cerebras" or "cerebras.ai" in base_url
    is_deep_seek = provider == "deepseek" or "deepseek.com" in base_url.lower()

    is_non_standard = (
        is_nvidia
        or is_cerebras
        or provider == "xai"
        or "api.x.ai" in base_url
        or is_together
        or "chutes.ai" in base_url
        or is_deep_seek
        or is_zai
        or is_moonshot
        or provider == "opencode"
        or "opencode.ai" in base_url
        or is_cloudflare_workers_ai
        or is_cloudflare_ai_gateway
        or is_ant_ling
    )

    use_max_tokens = (
        "chutes.ai" in base_url
        or is_deep_seek
        or is_moonshot
        or is_cloudflare_ai_gateway
        or is_together
        or is_nvidia
        or is_ant_ling
        or is_zai
    )

    is_grok = provider == "xai" or "api.x.ai" in base_url
    is_open_router_developer_role_model = is_open_router and model.id.startswith(("anthropic/", "openai/"))
    cache_control_format = "anthropic" if provider == "openrouter" and model.id.startswith("anthropic/") else None

    return {
        "supportsStore": not is_non_standard,
        "supportsDeveloperRole": is_open_router_developer_role_model or (not is_non_standard and not is_open_router),
        "supportsReasoningEffort": not (
            is_grok or is_zai or is_moonshot or is_together or is_cloudflare_ai_gateway or is_nvidia or is_ant_ling
        ),
        "supportsUsageInStreaming": True,
        "supportsFinishReason": True,
        "maxTokensField": "max_tokens" if use_max_tokens else "max_completion_tokens",
        "requiresToolResultName": False,
        "requiresAssistantAfterToolResult": False,
        "requiresThinkingAsText": False,
        "requiresReasoningContentOnAssistantMessages": is_deep_seek,
        "thinkingFormat": (
            "deepseek"
            if is_deep_seek
            else "zai"
            if is_zai
            else "together"
            if is_together
            else "ant-ling"
            if is_ant_ling
            else "openrouter"
            if is_open_router
            else "openai"
        ),
        "openRouterRouting": {},
        "vercelGatewayRouting": {},
        "chatTemplateKwargs": {},
        "chatTemplateArgs": {},
        "zaiToolStream": False,
        "supportsThinkingTokenBudget": False,
        "thinkingTokenBudgetField": None,
        # OpenAI compatibility alone does not imply strict JSON-schema tool support.
        "supportsStrictMode": False,
        "supportsOpenAIGrammarTools": False,
        "supportsMidConvoSystemMessages": False,
        "supportsMidConvoToolAdditions": False,
        "cacheControlFormat": cache_control_format,
        "sendSessionAffinityHeaders": is_open_router,
        "sessionAffinityFormat": "openrouter" if is_open_router else "openai",
        "supportsLongCacheRetention": not (
            is_together or is_cloudflare_workers_ai or is_cloudflare_ai_gateway or is_nvidia or is_ant_ling
        ),
    }


def get_compat(model: Model) -> dict[str, Any]:
    """Get resolved compatibility settings for a model.

    Auto-detects from provider/URL then overrides with explicit model.compat.
    """
    detected = detect_compat(model)
    compat = model.compat
    if compat is None:
        return detected

    return {
        "supportsStore": read_field(compat, "supportsStore", detected["supportsStore"]),
        "supportsDeveloperRole": read_field(compat, "supportsDeveloperRole", detected["supportsDeveloperRole"]),
        "supportsReasoningEffort": read_field(compat, "supportsReasoningEffort", detected["supportsReasoningEffort"]),
        "supportsUsageInStreaming": read_field(compat, "supportsUsageInStreaming", detected["supportsUsageInStreaming"]),
        "supportsFinishReason": read_field(compat, "supportsFinishReason", detected["supportsFinishReason"]),
        "maxTokensField": read_field(compat, "maxTokensField", detected["maxTokensField"]),
        "requiresToolResultName": read_field(compat, "requiresToolResultName", detected["requiresToolResultName"]),
        "requiresAssistantAfterToolResult": read_field(
            compat, "requiresAssistantAfterToolResult", detected["requiresAssistantAfterToolResult"]
        ),
        "requiresThinkingAsText": read_field(compat, "requiresThinkingAsText", detected["requiresThinkingAsText"]),
        "requiresReasoningContentOnAssistantMessages": read_field(
            compat,
            "requiresReasoningContentOnAssistantMessages",
            detected["requiresReasoningContentOnAssistantMessages"],
        ),
        "thinkingFormat": read_field(compat, "thinkingFormat", detected["thinkingFormat"]),
        "openRouterRouting": read_field(compat, "openRouterRouting", {}),
        "vercelGatewayRouting": read_field(compat, "vercelGatewayRouting", detected["vercelGatewayRouting"]),
        "chatTemplateKwargs": read_field(compat, "chatTemplateKwargs", detected["chatTemplateKwargs"]),
        "chatTemplateArgs": read_field(compat, "chatTemplateArgs", detected["chatTemplateArgs"]),
        "zaiToolStream": read_field(compat, "zaiToolStream", detected["zaiToolStream"]),
        "supportsThinkingTokenBudget": read_field(
            compat, "supportsThinkingTokenBudget", detected["supportsThinkingTokenBudget"]
        ),
        "thinkingTokenBudgetField": read_field(compat, "thinkingTokenBudgetField", detected["thinkingTokenBudgetField"]),
        "supportsStrictMode": read_field(compat, "supportsStrictMode", detected["supportsStrictMode"]),
        "supportsOpenAIGrammarTools": read_field(
            compat, "supportsOpenAIGrammarTools", detected["supportsOpenAIGrammarTools"]
        ),
        "supportsMidConvoSystemMessages": read_field(
            compat, "supportsMidConvoSystemMessages", detected["supportsMidConvoSystemMessages"]
        ),
        "supportsMidConvoToolAdditions": read_field(
            compat, "supportsMidConvoToolAdditions", detected["supportsMidConvoToolAdditions"]
        ),
        "cacheControlFormat": read_field(compat, "cacheControlFormat", detected["cacheControlFormat"]),
        "sendSessionAffinityHeaders": read_field(
            compat, "sendSessionAffinityHeaders", detected["sendSessionAffinityHeaders"]
        ),
        "sessionAffinityFormat": read_field(compat, "sessionAffinityFormat", detected["sessionAffinityFormat"]),
        "supportsLongCacheRetention": read_field(
            compat, "supportsLongCacheRetention", detected["supportsLongCacheRetention"]
        ),
        "vllmPriority": read_field(compat, "vllmPriority", None),
    }


# --- MISAKA: Python SDK plumbing ---------------------------------------------------------


async def _create_completion_stream(
    client: Any, params: dict[str, Any], options: Any, model: Model, request_headers: dict[str, Any] | None = None
) -> Any:
    """``client.chat.completions.create(params, {signal, timeout, maxRetries: 0}).withResponse()``.

    The SDK's own retry is always off: ``utils/provider_retry`` decides what is worth retrying.
    """
    signal = _option(options, "signal")
    request_options: dict[str, Any] = {"max_retries": 0}
    if _option(options, "timeoutMs") is not None:
        request_options["timeout"] = _option(options, "timeoutMs") / 1000
    request_client = client.with_options(**request_options) if hasattr(client, "with_options") else client
    completions = request_client.chat.completions
    call_options = {"extra_headers": dict(request_headers)} if request_headers else {}

    if hasattr(completions, "with_raw_response"):
        raw_response = await _await_maybe_with_signal(
            completions.with_raw_response.create(**split_sdk_params(completions.create, params), **call_options),
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
        completions.create(**split_sdk_params(completions.create, params), **call_options), signal
    )


async def _iterate_stream(stream_obj: Any, signal: Any = None) -> AsyncIterator[dict[str, Any]]:
    # MISAKA fork: one abort task for the whole stream (_iterate_async_iterable), not one per SSE
    # item: a task per item cost several loop turns per event, which a busy TUI turned into a stream
    # that fell minutes behind the provider (GitHub issue #6). Closed with this generator, so its
    # abort task goes with it (a turn that ended in an error left a pending signal.wait() behind
    # until the cyclic GC destroyed it).
    async with contextlib.aclosing(
            _iterate_async_iterable(stream_obj, signal, on_abort=lambda: _close_stream(stream_obj))) as items:
        async for chunk in items:
            if hasattr(chunk, "model_dump"):
                dumped = chunk.model_dump()
                if isinstance(dumped, dict):
                    yield dumped
                    continue
            if isinstance(chunk, dict):
                yield chunk
                continue
            yield json.loads(json.dumps(chunk, default=lambda value: value.__dict__))


streamOpenAICompletions = stream_openai_completions
streamSimpleOpenAICompletions = stream_simple_openai_completions
__all__ = [
    "OpenAICompletionsOptions",
    "apply_anthropic_cache_control",
    "build_params",
    "convert_messages",
    "convert_tools",
    "create_client",
    "detect_compat",
    "get_compat",
    "get_compat_cache_control",
    "has_tool_history",
    "map_stop_reason",
    "parse_chunk_usage",
    "streamOpenAICompletions",
    "streamSimpleOpenAICompletions",
    "stream_openai_completions",
    "stream_simple_openai_completions",
]
