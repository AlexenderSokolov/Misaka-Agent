"""Anthropic Messages provider adapter (pi ``api/anthropic-messages.ts``)."""

from __future__ import annotations

import codecs
import json
import re
import time
from collections.abc import AsyncIterator, Iterable, Mapping
from typing import Any, Literal, NotRequired, TypedDict

try:
    from anthropic import AsyncAnthropic, omit
except ImportError:  # optional extra: misaka[anthropic]
    AsyncAnthropic = omit = None

from misaka.ai.models import calculate_cost
from misaka.ai.providers._common import (
    _await_maybe_with_signal,
    _close_stream,
    _empty_usage,
    _iterate_async_iterable,
    _option,
    resolve_cache_retention,
)
from misaka.ai.providers.constrained_sampling import (
    get_json_schema_tool_parameters,
    resolve_json_schema_strict_sampling,
)
from misaka.ai.providers.github_copilot_headers import (
    build_copilot_dynamic_headers,
    has_copilot_vision_input,
)
from misaka.ai.providers.sdk import require
from misaka.ai.providers.simple_options import (
    adjust_max_tokens_for_thinking,
    build_base_options,
    clamp_thinking_budget_to_answer_room,
)
from misaka.ai.providers.transform_messages import transform_messages
from misaka.ai.types import (
    AssistantMessage,
    CacheRetention,
    DoneEvent,
    ErrorEvent,
    ImageContent,
    Model,
    SimpleStreamOptions,
    StartEvent,
    StopReason,
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
)
from misaka.ai.utils.diagnostics import (
    AssistantMessageDiagnostic,
    append_assistant_message_diagnostic,
    create_assistant_message_diagnostic,
)
from misaka.ai.utils.estimate import clamp_max_tokens_to_context
from misaka.ai.utils.event_stream import AssistantMessageEventStream, spawn_stream_task
from misaka.ai.utils.headers import headers_to_record
from misaka.ai.utils.json_parse import StreamingArgs, parse_json_with_repair
from misaka.ai.utils.provider_retry import retry_provider_request
from misaka.ai.utils.sanitize_unicode import sanitize_surrogates
from misaka.ai.utils.text import get_system_message_text, render_system_message_update
from misaka.ai.utils.transcript import (
    get_current_tools,
    get_declared_tools,
    get_initial_system_message,
    has_tool_redefinitions,
    resolve_transcript,
)
from misaka.ai.utils.user_agent import get_misaka_user_agent
from misaka.utils.values import maybe_await, signal_aborted


def get_cache_control(
    model: Model, cache_retention: CacheRetention | None = None, env: Any = None
) -> dict[str, Any]:
    retention = resolve_cache_retention(cache_retention, env)
    if retention == "none":
        return {"retention": retention}
    ttl = "1h" if retention == "long" and get_anthropic_compat(model)["supportsLongCacheRetention"] else None
    return {
        "retention": retention,
        "cacheControl": {"type": "ephemeral", **({"ttl": ttl} if ttl else {})},
    }


# Stealth mode: Mimic Claude Code's tool naming exactly
CLAUDE_CODE_VERSION = "2.1.280"

# Claude Code 2.x tool names (canonical casing)
# Source: https://cchistory.mariozechner.at/data/prompts-2.1.11.md
# To update: https://github.com/badlogic/cchistory
_CLAUDE_CODE_TOOLS = (
    "Read",
    "Write",
    "Edit",
    "Bash",
    "Grep",
    "Glob",
    "AskUserQuestion",
    "EnterPlanMode",
    "ExitPlanMode",
    "KillShell",
    "NotebookEdit",
    "Skill",
    "Task",
    "TaskOutput",
    "TodoWrite",
    "WebFetch",
    "WebSearch",
)

_CC_TOOL_LOOKUP = {name.lower(): name for name in _CLAUDE_CODE_TOOLS}


def to_claude_code_name(name: str) -> str:
    """Convert tool name to CC canonical casing if it matches (case-insensitive)."""
    return _CC_TOOL_LOOKUP.get(name.lower(), name)


def from_claude_code_name(name: str, tools: Iterable[Tool] | None = None) -> str:
    if tools:
        lower_name = name.lower()
        for tool in tools:
            if tool.name.lower() == lower_name:
                return tool.name
    return name


def convert_content_blocks(content: list[TextContent | ImageContent]) -> str | list[dict[str, Any]]:
    """Convert content blocks to Anthropic API format."""
    # If only text blocks, return as concatenated string for simplicity
    has_images = any(block.type == "image" for block in content)
    if not has_images:
        return sanitize_surrogates("\n".join(block.text for block in content if block.type == "text"))

    # If we have images, convert to content block array
    blocks: list[dict[str, Any]] = []
    for block in content:
        if block.type == "text":
            blocks.append({"type": "text", "text": sanitize_surrogates(block.text)})
        else:
            blocks.append({
                "type": "image",
                "source": {"type": "base64", "media_type": block.mimeType, "data": block.data},
            })

    # If only images (no text), add placeholder text block
    has_text = any(block["type"] == "text" for block in blocks)
    if not has_text:
        blocks.insert(0, {"type": "text", "text": "(see attached image)"})

    return blocks


AnthropicEffort = Literal["low", "medium", "high", "xhigh", "max"]

AnthropicThinkingDisplay = Literal["summarized", "omitted"]

FINE_GRAINED_TOOL_STREAMING_BETA = "fine-grained-tool-streaming-2025-05-14"
INTERLEAVED_THINKING_BETA = "interleaved-thinking-2025-05-14"
SERVER_SIDE_FALLBACK_BETA = "server-side-fallback-2026-07-01"
MID_CONVERSATION_OUTPUT_CONFIG_BETA = "mid-conversation-output-config-2026-07-01"
THINKING_BINDING_CONTROLS_BETA = "thinking-binding-controls-2026-08-01"
MID_CONVERSATION_TOOL_CHANGES_BETA = "mid-conversation-tool-changes-2026-07-01"

# Stable deferred tool declared whenever native tool changes are in use. Anthropic adds
# hidden prompt scaffolding as soon as any tool has `defer_loading`; declaring this
# placeholder from the first request keeps that scaffolding in the cached prefix, so the
# first real late tool does not invalidate the cache (measured: full miss without it).
# It is never activated and the model cannot see it.
DEFERRED_TOOL_PLACEHOLDER: dict[str, Any] = {
    "name": "__pi_deferred_placeholder__",
    "description": "Reserved placeholder. Never available. Never call this.",
    "input_schema": {"type": "object", "properties": {}, "required": []},
    "defer_loading": True,
}


def _compat_value(model: Model, name: str) -> Any:
    return getattr(getattr(model, "compat", None), name, None)


def _default(value: Any, default: Any) -> Any:
    """``value ?? default``: only an absent field takes the default, never a false one."""
    return default if value is None else value


def should_use_server_side_fallback_beta(model: Model) -> bool:
    return len(_compat_value(model, "allowedFallbackModels") or []) > 0


def get_anthropic_compat(model: Model) -> dict[str, Any]:
    is_open_router = model.provider == "openrouter" or "openrouter.ai" in model.baseUrl
    return {
        "supportsEagerToolInputStreaming": _default(_compat_value(model, "supportsEagerToolInputStreaming"), True),
        "supportsLongCacheRetention": _default(_compat_value(model, "supportsLongCacheRetention"), True),
        "sendSessionAffinityHeaders": _default(_compat_value(model, "sendSessionAffinityHeaders"), is_open_router),
        "sessionAffinityFormat": _default(
            _compat_value(model, "sessionAffinityFormat"), "openrouter" if is_open_router else None
        ),
        "supportsCacheControlOnTools": _default(_compat_value(model, "supportsCacheControlOnTools"), True),
        "supportsTemperature": _default(_compat_value(model, "supportsTemperature"), True),
        "allowEmptySignature": _default(_compat_value(model, "allowEmptySignature"), False),
        "supportsStrictTools": _default(_compat_value(model, "supportsStrictTools"), False),
        "supportsMidConvoSystemMessages": _default(_compat_value(model, "supportsMidConvoSystemMessages"), False),
        "supportsMidConvoToolChanges": _default(_compat_value(model, "supportsMidConvoToolChanges"), False),
    }


class AnthropicOptions(TypedDict, total=False):
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
    # Enable extended thinking. For adaptive thinking models the model decides when/how
    # much to think; for older models budget-based thinking uses `thinkingBudgetTokens`.
    thinkingEnabled: bool
    # Token budget for extended thinking (older models only). Default: 1024.
    thinkingBudgetTokens: int
    # Effort level for adaptive thinking models. Ignored for older models.
    effort: AnthropicEffort
    # "summarized" (default when thinking is enabled) or "omitted".
    thinkingDisplay: AnthropicThinkingDisplay
    # Whether to request the interleaved thinking beta header for non-adaptive thinking
    # models. Default: true.
    interleavedThinking: bool
    # String values map to Anthropic's built-in choices; `{"type": "tool", "name": ...}`
    # forces a specific tool.
    toolChoice: str | dict[str, str]
    # Pre-built Anthropic client instance. When provided, skips internal client construction.
    client: Any


def merge_headers(*header_sources: Mapping[str, Any] | None) -> dict[str, Any]:
    """Later sources win; a ``None`` value is kept, meaning "do not send this header".

    MISAKA fork: names are matched case-insensitively. Upstream merges with `Object.assign`
    and relies on the `Headers` object downstream to collapse `User-Agent` and `user-agent`
    into one; a dict handed to httpx collapses nothing, so the OAuth path -- which overrides
    the client string with `claude-cli/...` -- would send both identities.
    """
    merged: dict[str, Any] = {}
    for headers in header_sources:
        if not headers:
            continue
        for key, value in headers.items():
            name = str(key)
            lowered = name.lower()
            for existing in [k for k in merged if k.lower() == lowered]:
                del merged[existing]
            merged[name] = value
    return merged


def merge_client_headers(*header_sources: Mapping[str, Any] | None) -> dict[str, Any]:
    return merge_headers({"User-Agent": get_misaka_user_agent()}, *header_sources)


def has_header(headers: Mapping[str, Any] | None, name: str) -> bool:
    if not headers:
        return False
    expected = name.lower()
    return any(
        str(key).lower() == expected and value is not None and len(str(value).strip()) > 0
        for key, value in headers.items()
    )


def assert_request_auth(provider: str, api_key: str | None, headers: Mapping[str, Any] | None) -> None:
    if api_key:
        return
    if (
        has_header(headers, "authorization")
        or has_header(headers, "x-api-key")
        or has_header(headers, "cf-aig-authorization")
    ):
        return
    raise RuntimeError(f"No API key for provider: {provider}")


class ServerSentEvent(dict[str, Any]):
    event: str | None
    data: str
    raw: list[str]


ANTHROPIC_MESSAGE_EVENTS = frozenset(
    {
        "message_start",
        "message_delta",
        "message_stop",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
    }
)


def _flush_sse_event(state: dict[str, Any]) -> ServerSentEvent | None:
    if not state["event"] and not state["data"]:
        return None

    event = ServerSentEvent(event=state["event"], data="\n".join(state["data"]), raw=list(state["raw"]))
    state["event"] = None
    state["data"] = []
    state["raw"] = []
    return event


def _decode_sse_line(line: str, state: dict[str, Any]) -> ServerSentEvent | None:
    if line == "":
        return _flush_sse_event(state)

    state["raw"].append(line)
    if line.startswith(":"):
        return None

    delimiter_index = line.find(":")
    field_name = line if delimiter_index == -1 else line[:delimiter_index]
    value = "" if delimiter_index == -1 else line[delimiter_index + 1 :]
    value = value.removeprefix(" ")

    if field_name == "event":
        state["event"] = value
    elif field_name == "data":
        state["data"].append(value)
    return None


def _next_line_break_index(text: str) -> int:
    carriage_return = text.find("\r")
    newline = text.find("\n")
    if carriage_return == -1:
        return newline
    if newline == -1:
        return carriage_return
    return min(carriage_return, newline)


def _consume_line(text: str) -> tuple[str, str] | None:
    line_break_index = _next_line_break_index(text)
    if line_break_index == -1:
        return None

    next_index = line_break_index + 1
    if text[line_break_index] == "\r" and next_index < len(text) and text[next_index] == "\n":
        next_index += 1
    return text[:line_break_index], text[next_index:]


async def _iter_response_lines(source: Any, signal: Any = None) -> AsyncIterator[str]:
    """MISAKA fork: pi reads one `ReadableStream`; a Python response may be an SDK raw-response
    wrapper, an httpx response or a plain iterable, so each surface is read here."""
    # Unwrap SDK raw-response wrappers (e.g. anthropic LegacyAPIResponse), which expose
    # the streaming httpx.Response as `.http_response` and none of the iteration surfaces.
    # The wrapped response comes from AsyncAnthropic, so its sync iter_lines() would raise
    # ("Attempted to call a sync iterator on an async stream") -- go straight to aiter_lines.
    http_response = getattr(source, "http_response", None)
    if http_response is not None and not hasattr(source, "iter_lines") and not hasattr(source, "aiter_lines"):
        if hasattr(http_response, "aiter_lines"):
            async for line in _iterate_async_iterable(
                http_response.aiter_lines(), signal, on_abort=lambda: _close_stream(http_response)
            ):
                yield line.decode("utf-8") if isinstance(line, bytes) else str(line)
            return
        source = http_response
    if hasattr(source, "iter_lines"):
        lines = source.iter_lines()
        if hasattr(lines, "__aiter__"):
            async for line in _iterate_async_iterable(lines, signal, on_abort=lambda: _close_stream(source)):
                yield line.decode("utf-8") if isinstance(line, bytes) else str(line)
            return
        for line in lines:
            if signal_aborted(signal):
                raise RuntimeError("Request was aborted")
            yield line.decode("utf-8") if isinstance(line, bytes) else str(line)
        return

    if hasattr(source, "aiter_lines"):
        async for line in _iterate_async_iterable(source.aiter_lines(), signal, on_abort=lambda: _close_stream(source)):
            yield line.decode("utf-8") if isinstance(line, bytes) else str(line)
        return

    body = getattr(source, "body", None)
    if body is None and hasattr(source, "__aiter__"):
        body = source
    if body is None:
        raise RuntimeError("Attempted to iterate over an Anthropic response with no body")

    # pi decodes the body with ``TextDecoder.decode(value, {stream: true})``, which holds back a
    # multi-byte character split across chunks and replaces invalid bytes instead of throwing.
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    buffer = ""
    if hasattr(body, "__aiter__"):
        async for chunk in _iterate_async_iterable(body, signal, on_abort=lambda: _close_stream(source)):
            buffer += decoder.decode(chunk) if isinstance(chunk, bytes) else str(chunk)
            consumed = _consume_line(buffer)
            while consumed is not None:
                line, buffer = consumed
                yield line
                consumed = _consume_line(buffer)
    else:
        for chunk in body:
            if signal_aborted(signal):
                raise RuntimeError("Request was aborted")
            buffer += decoder.decode(chunk) if isinstance(chunk, bytes) else str(chunk)
            consumed = _consume_line(buffer)
            while consumed is not None:
                line, buffer = consumed
                yield line
                consumed = _consume_line(buffer)

    buffer += decoder.decode(b"", final=True)
    if buffer:
        consumed = _consume_line(buffer)
        while consumed is not None:
            line, buffer = consumed
            yield line
            consumed = _consume_line(buffer)
        if buffer:
            yield buffer


async def iterate_sse_messages(source: Any, signal: Any = None) -> AsyncIterator[ServerSentEvent]:
    state: dict[str, Any] = {"event": None, "data": [], "raw": []}
    async for line in _iter_response_lines(source, signal):
        if signal_aborted(signal):
            raise RuntimeError("Request was aborted")
        event = _decode_sse_line(line, state)
        if event is not None:
            yield event

    trailing_event = _flush_sse_event(state)
    if trailing_event is not None:
        yield trailing_event


async def iterate_anthropic_events(source: Any, signal: Any = None) -> AsyncIterator[dict[str, Any]]:
    saw_message_start = False
    saw_message_end = False

    async for sse in iterate_sse_messages(source, signal):
        event_name = sse.get("event")
        if event_name == "error":
            raise RuntimeError(sse["data"])

        if event_name not in ANTHROPIC_MESSAGE_EVENTS:
            continue

        try:
            event = parse_json_with_repair(sse["data"])
            if not isinstance(event, dict):
                raise ValueError("parsed payload was not an object")  # noqa: TRY004 - same report as a parse failure
        except Exception as error:
            raw_text = "\\n".join(sse["raw"])
            raise RuntimeError(
                f"Could not parse Anthropic SSE event {event_name}: {error}; data={sse['data']}; raw={raw_text}"
            ) from error
        if event.get("type") == "message_start":
            saw_message_start = True
        elif event.get("type") == "message_stop":
            saw_message_end = True
        yield event

    if saw_message_start and not saw_message_end:
        raise RuntimeError("Anthropic stream ended before message_stop")


def stream_anthropic(
    model: Model,
    context: TranscriptContext,
    options: StreamOptions | dict[str, Any] | None = None,
) -> AssistantMessageEventStream:
    stream = AssistantMessageEventStream()
    normalized_context = resolve_transcript(context, get_anthropic_compat(model)["supportsMidConvoSystemMessages"])
    current_tools = get_current_tools(normalized_context.messages)

    async def run() -> None:
        provider_thinking_level = (
            (_option(options, "effort") or "high") if _compat_value(model, "supportsMidConvoEffort") else None
        )
        output = AssistantMessage(
            content=[],
            api=model.api,
            provider=model.provider,
            model=model.id,
            providerThinkingLevel=provider_thinking_level,
            usage=_empty_usage(),
            stopReason="pending",
            timestamp=time.time_ns() // 1_000_000,
        )
        raw_response: Any = None
        owned_client: Any = None

        try:
            usage_model = model
            input_transformations: list[Any] | None = None
            request_headers: dict[str, Any] = {}

            client = _option(options, "client")
            if client is not None:
                is_oauth = False
            else:
                api_key = _option(options, "apiKey")
                assert_request_auth(model.provider, api_key, _option(options, "headers"))

                copilot_dynamic_headers: dict[str, str] | None = None
                if model.provider == "github-copilot":
                    has_images = has_copilot_vision_input(normalized_context.messages)
                    copilot_dynamic_headers = build_copilot_dynamic_headers(
                        messages=normalized_context.messages,
                        hasImages=has_images,
                    )

                cache_retention = resolve_cache_retention(_option(options, "cacheRetention"), _option(options, "env"))
                cache_session_id = None if cache_retention == "none" else _option(options, "sessionId")

                client, is_oauth, request_headers = create_client(
                    model,
                    api_key,
                    _option(options, "headers"),
                    copilot_dynamic_headers,
                    cache_session_id,
                )
                owned_client = client
            params = build_params(model, normalized_context, is_oauth, options)
            on_payload = _option(options, "onPayload")
            if callable(on_payload):
                next_params = await maybe_await(on_payload(params, model))
                if next_params is not None:
                    params = {**next_params, "stream": True}
            raw_response = await retry_provider_request(
                lambda: _create_raw_response(client, params, options, request_headers),
                max_retries=_option(options, "maxRetries") or 0,
                max_retry_delay_ms=_option(options, "maxRetryDelayMs"),
                signal=_option(options, "signal"),
            )
            await _emit_response_metadata(raw_response, options, model)
            stream.push(StartEvent(partial=output))

            # MISAKA fork: pi keeps the provider block index and the tool's partial JSON on
            # the block itself; pydantic blocks forbid extra fields, so both live beside it.
            provider_indexes: dict[int, int] = {}
            tool_partial_json: dict[int, StreamingArgs] = {}
            # Deltas whose block index was never started: a compatible endpoint's framing
            # fault. Counted so a tool call that ends empty can say where its input went.
            dropped_deltas: dict[str, int] = {}

            if hasattr(raw_response, "http_response") or hasattr(raw_response, "iter_lines") or hasattr(raw_response, "aiter_lines"):
                event_iter: AsyncIterator[dict[str, Any]] = iterate_anthropic_events(raw_response, _option(options, "signal"))
            else:
                event_iter = _iter_event_objects(raw_response, _option(options, "signal"))

            on_provider_stream_event = _option(options, "onProviderStreamEvent")
            async for event in event_iter:
                if callable(on_provider_stream_event):
                    await maybe_await(on_provider_stream_event(event, model))
                event_type = event.get("type")
                if event_type == "message_start":
                    message = event.get("message") if isinstance(event.get("message"), Mapping) else {}
                    output.responseId = message.get("id")
                    transformations = message.get("input_transformations")
                    if isinstance(transformations, list):
                        input_transformations = transformations
                    response_model = message.get("model")
                    if isinstance(response_model, str) and response_model and response_model != model.id:
                        output.responseModel = response_model
                    fallback_cost = (
                        None
                        if not isinstance(response_model, str) or response_model == model.id
                        else next(
                            (
                                fallback.cost
                                for fallback in _compat_value(model, "allowedFallbackModels") or []
                                if fallback.provider == model.provider and fallback.model == response_model
                            ),
                            None,
                        )
                    )
                    usage_model = (
                        model.model_copy(update={"id": response_model, "cost": fallback_cost}) if fallback_cost else model
                    )
                    usage = message.get("usage") if isinstance(message.get("usage"), Mapping) else {}
                    # Capture initial token usage from message_start event
                    # This ensures we have input token counts even if the stream is aborted early
                    output.usage.input = int(usage.get("input_tokens") or 0)
                    output.usage.output = int(usage.get("output_tokens") or 0)
                    output.usage.cacheRead = int(usage.get("cache_read_input_tokens") or 0)
                    output.usage.cacheWrite = int(usage.get("cache_creation_input_tokens") or 0)
                    cache_creation = usage.get("cache_creation")
                    output.usage.cacheWrite1h = int(
                        (cache_creation.get("ephemeral_1h_input_tokens") if isinstance(cache_creation, Mapping) else 0)
                        or 0
                    )
                    # Anthropic doesn't provide total_tokens, compute from components
                    output.usage.totalTokens = (
                        output.usage.input + output.usage.output + output.usage.cacheRead + output.usage.cacheWrite
                    )
                    calculate_cost(usage_model, output.usage)
                elif event_type == "content_block_start":
                    content_block = event.get("content_block")
                    provider_index = event.get("index")
                    if not isinstance(content_block, Mapping) or not isinstance(provider_index, int):
                        continue
                    block_type = content_block.get("type")
                    if block_type == "fallback":
                        if len(output.content) > 0:
                            raise RuntimeError("Anthropic performed an unsupported mid-output model fallback")
                        continue
                    if block_type == "text":
                        output.content.append(TextContent(text=str(content_block.get("text") or "")))
                        provider_indexes[provider_index] = len(output.content) - 1
                        stream.push(TextStartEvent(contentIndex=len(output.content) - 1, partial=output))
                    elif block_type == "thinking":
                        output.content.append(ThinkingContent(
                            thinking=str(content_block.get("thinking") or ""),
                            thinkingSignature=str(content_block.get("signature") or ""),
                        ))
                        provider_indexes[provider_index] = len(output.content) - 1
                        stream.push(ThinkingStartEvent(contentIndex=len(output.content) - 1, partial=output))
                    elif block_type == "redacted_thinking":
                        output.content.append(ThinkingContent(
                            thinking="[Reasoning redacted]",
                            thinkingSignature=str(content_block.get("data") or ""),
                            redacted=True,
                        ))
                        provider_indexes[provider_index] = len(output.content) - 1
                        stream.push(ThinkingStartEvent(contentIndex=len(output.content) - 1, partial=output))
                    elif block_type == "tool_use":
                        initial_arguments = content_block.get("input")
                        name = str(content_block.get("name") or "")
                        output.content.append(ToolCall(
                            id=str(content_block.get("id") or ""),
                            name=from_claude_code_name(name, current_tools) if is_oauth else name,
                            arguments=initial_arguments if isinstance(initial_arguments, dict) else {},
                        ))
                        provider_indexes[provider_index] = len(output.content) - 1
                        tool_partial_json[provider_index] = StreamingArgs()
                        stream.push(ToolCallStartEvent(contentIndex=len(output.content) - 1, partial=output))
                elif event_type == "content_block_delta":
                    provider_index = event.get("index")
                    delta = event.get("delta")
                    if not isinstance(provider_index, int) or not isinstance(delta, Mapping):
                        continue
                    index = provider_indexes.get(provider_index)
                    if index is None:
                        kind = str(delta.get("type") or "?")
                        dropped_deltas[kind] = dropped_deltas.get(kind, 0) + 1
                        continue
                    block = output.content[index]
                    delta_type = delta.get("type")
                    if delta_type == "text_delta" and isinstance(block, TextContent):
                        text = str(delta.get("text") or "")
                        block.text += text
                        stream.push(TextDeltaEvent(contentIndex=index, delta=text, partial=output))
                    elif delta_type == "thinking_delta" and isinstance(block, ThinkingContent):
                        thinking = str(delta.get("thinking") or "")
                        block.thinking += thinking
                        stream.push(ThinkingDeltaEvent(contentIndex=index, delta=thinking, partial=output))
                    elif delta_type == "input_json_delta" and isinstance(block, ToolCall):
                        partial_json = str(delta.get("partial_json") or "")
                        accumulated = tool_partial_json.setdefault(provider_index, StreamingArgs())
                        accumulated.append(partial_json)
                        block.arguments = accumulated.arguments
                        stream.push(ToolCallDeltaEvent(contentIndex=index, delta=partial_json, partial=output))
                    elif delta_type == "signature_delta" and isinstance(block, ThinkingContent):
                        block.thinkingSignature = block.thinkingSignature or ""
                        block.thinkingSignature += str(delta.get("signature") or "")
                elif event_type == "content_block_stop":
                    provider_index = event.get("index")
                    index = provider_indexes.get(provider_index) if isinstance(provider_index, int) else None
                    if index is None:
                        continue
                    block = output.content[index]
                    if isinstance(block, TextContent):
                        stream.push(TextEndEvent(contentIndex=index, content=block.text, partial=output))
                    elif isinstance(block, ThinkingContent):
                        stream.push(ThinkingEndEvent(contentIndex=index, content=block.thinking, partial=output))
                    elif isinstance(block, ToolCall):
                        # MISAKA fork: final arguments go through `finish_into`, which records a
                        # malformed object as `argumentsError`; and only streamed deltas replace
                        # what content_block_start inlined -- compatible endpoints (proxies,
                        # gateways) put the whole input in the start block and send no
                        # input_json_delta, where pi's parse of the empty buffer hands the tool {}.
                        accumulated = tool_partial_json.get(provider_index)
                        if accumulated is not None and accumulated.raw:
                            accumulated.finish_into(block)
                        _note_empty_tool_arguments(output, block, accumulated, dropped_deltas)
                        # Finalize in-place and strip the scratch buffer so replay only
                        # carries parsed arguments.
                        tool_partial_json.pop(provider_index, None)
                        stream.push(ToolCallEndEvent(contentIndex=index, toolCall=block, partial=output))
                elif event_type == "message_delta":
                    transformations = event.get("input_transformations")
                    if isinstance(transformations, list):
                        input_transformations = transformations
                    delta = event.get("delta") if isinstance(event.get("delta"), Mapping) else {}
                    stop_reason = delta.get("stop_reason")
                    if stop_reason:
                        output.rawStopReason = stop_reason
                        stop_reason_result = map_stop_reason(stop_reason, delta.get("stop_details"))
                        output.stopReason = stop_reason_result["stopReason"]
                        if stop_reason_result.get("errorMessage"):
                            output.errorMessage = stop_reason_result["errorMessage"]
                    # Only update usage fields if present (not null).
                    # Preserves input_tokens from message_start when proxies omit it in message_delta.
                    usage = event.get("usage")
                    if isinstance(usage, Mapping):
                        if usage.get("input_tokens") is not None:
                            output.usage.input = int(usage["input_tokens"])
                        if usage.get("output_tokens") is not None:
                            output.usage.output = int(usage["output_tokens"])
                        if usage.get("cache_read_input_tokens") is not None:
                            output.usage.cacheRead = int(usage["cache_read_input_tokens"])
                        if usage.get("cache_creation_input_tokens") is not None:
                            output.usage.cacheWrite = int(usage["cache_creation_input_tokens"])
                        # Vercel AI Gateway includes the TTL breakdown in deltas, though the SDK only types it on message_start.
                        cache_creation = usage.get("cache_creation")
                        if isinstance(cache_creation, Mapping) and cache_creation.get("ephemeral_1h_input_tokens") is not None:
                            output.usage.cacheWrite1h = int(cache_creation["ephemeral_1h_input_tokens"])
                        # Anthropic reports reasoning tokens as a subset of output tokens.
                        output_details = usage.get("output_tokens_details")
                        thinking_tokens = (
                            output_details.get("thinking_tokens") if isinstance(output_details, Mapping) else None
                        )
                        if thinking_tokens is not None:
                            output.usage.reasoning = int(thinking_tokens)
                    # Anthropic doesn't provide total_tokens, compute from components
                    output.usage.totalTokens = (
                        output.usage.input + output.usage.output + output.usage.cacheRead + output.usage.cacheWrite
                    )
                    calculate_cost(usage_model, output.usage)

            # MISAKA fork: a compatible endpoint may omit a block-stop event. Finalize every
            # raw buffer before dispatch; a streaming preview alone is never executable.
            for provider_index, accumulated in tool_partial_json.items():
                index = provider_indexes.get(provider_index)
                if index is not None and accumulated.raw:
                    block = output.content[index]
                    if isinstance(block, ToolCall):
                        accumulated.finish_into(block)

            if signal_aborted(_option(options, "signal")):
                raise RuntimeError("Request was aborted")

            if output.stopReason == "pending":
                raise RuntimeError("Anthropic stream ended without a stop reason")
            if output.stopReason in {"aborted", "error"}:
                raise RuntimeError(output.errorMessage or "An unknown error occurred")
            if input_transformations:
                append_assistant_message_diagnostic(output, AssistantMessageDiagnostic(
                    type="anthropic_input_transformations",
                    timestamp=time.time_ns() // 1_000_000,
                    details={
                        "transformations": [
                            {
                                key: transformation.get(key)
                                for key in ("type", "path", "reason")
                                if isinstance(transformation, Mapping) and transformation.get(key) is not None
                            }
                            for transformation in input_transformations
                        ]
                    },
                ))

            stream.push(DoneEvent(reason=output.stopReason, message=output))
        except Exception as error:  # noqa: BLE001 - every failure becomes an error event on the stream
            output.stopReason = "aborted" if signal_aborted(_option(options, "signal")) else "error"
            output.errorMessage = str(error) if isinstance(error, Exception) else json.dumps(error, default=str)
            stream.push(ErrorEvent(reason=output.stopReason, error=output), cause=error)
        finally:
            await _close_stream(raw_response)
            if owned_client is not None:
                # MISAKA fork: closed while the loop that made it is still running. Left to
                # the garbage collector, httpx closes the pool from its finaliser: in a one-shot
                # helper loop (`platform.session.run_coro`) that loop is gone by then, and every
                # call printed "Event loop is closed" through asyncio's default handler.
                try:
                    await owned_client.close()
                except Exception:  # noqa: BLE001, S110 - closing is best-effort; the stream has already ended
                    pass
            stream.end()

    spawn_stream_task(run(), stream=stream)
    return stream


def map_thinking_level_to_effort(model: Model, level: str | None) -> AnthropicEffort:
    """Map ThinkingLevel to Anthropic effort levels for adaptive thinking.

    Note: effort "max" is available on all adaptive-thinking Claude models, while native
    "xhigh" is only available on Opus 4.7/4.8, Sonnet 5, and Fable 5.
    """
    mapped = model.thinkingLevelMap.get(level) if level and model.thinkingLevelMap else None
    if isinstance(mapped, str):
        return mapped  # type: ignore[return-value]

    if level in {"minimal", "low"}:
        return "low"
    if level == "medium":
        return "medium"
    return "high"


def stream_simple_anthropic(
    model: Model,
    context: TranscriptContext,
    options: SimpleStreamOptions | None = None,
) -> AssistantMessageEventStream:
    assert_request_auth(model.provider, _option(options, "apiKey"), _option(options, "headers"))

    base = {
        **build_base_options(model, context, options, _option(options, "apiKey")).model_dump(),
        "toolChoice": _option(options, "toolChoice"),
    }
    if not options or not options.reasoning:
        return stream_anthropic(model, context, {**base, "thinkingEnabled": False})

    # For models with adaptive thinking: use an effort level.
    # For older models: use budget-based thinking.
    if _compat_value(model, "forceAdaptiveThinking") is True:
        effort = map_thinking_level_to_effort(model, options.reasoning)
        return stream_anthropic(model, context, {**base, "thinkingEnabled": True, "effort": effort})

    # Undefined means the caller did not request an output cap; let the helper use the model cap.
    # Do not coerce to 0 here, or the thinking budget would become the entire max_tokens value.
    adjusted = adjust_max_tokens_for_thinking(
        base["maxTokens"],
        model.maxTokens,
        options.reasoning,
        options.thinkingBudgets,
    )

    max_tokens = clamp_max_tokens_to_context(model, context, adjusted.maxTokens)

    return stream_anthropic(
        model,
        context,
        {
            **base,
            "maxTokens": max_tokens,
            "thinkingEnabled": True,
            "thinkingBudgetTokens": clamp_thinking_budget_to_answer_room(adjusted.thinkingBudget, max_tokens),
        },
    )


def is_oauth_token(api_key: str | None) -> bool:
    return bool(api_key) and "sk-ant-oat" in api_key


_CANONICAL_AUTH_HEADERS = {"x-api-key": "X-Api-Key", "authorization": "Authorization"}


def _sdk_client(
    *, api_key: str | None, auth_token: str | None, base_url: str, headers: Mapping[str, Any]
) -> tuple[Any, dict[str, Any]]:
    """Build the SDK client and the per-request headers that carry every suppression.

    MISAKA fork, three Python SDK differences from the TS one pi builds on:
    - A ``None`` header value is pi's "do not send this header"; the Python SDK takes
      ``omit`` for that, and only honours it per request, not in ``default_headers``.
    - ``AsyncAnthropic`` reads ``ANTHROPIC_API_KEY``/``ANTHROPIC_AUTH_TOKEN`` (and credential
      files) when neither ``api_key`` nor ``auth_token`` is given, where pi's
      ``apiKey: null`` means none. Header-owned auth therefore passes a placeholder key and
      suppresses the header it would mint.
    - Its request-time auth check needs an ``X-Api-Key`` or ``Authorization`` header, or one
      of them omitted per request.
    """
    request_headers: dict[str, Any] = {}
    default_headers: dict[str, Any] = {}
    for name, value in headers.items():
        if value is None:
            request_headers[_CANONICAL_AUTH_HEADERS.get(name.lower(), name)] = omit
        else:
            default_headers[name] = value
    if api_key is None and auth_token is None:
        api_key = "header-auth-placeholder"
        request_headers["X-Api-Key"] = omit
    client = require(AsyncAnthropic, "anthropic")(
        api_key=api_key,
        auth_token=auth_token,
        base_url=base_url,
        default_headers=default_headers,
    )
    return client, request_headers


def create_client(
    model: Model,
    api_key: str | None,
    options_headers: Mapping[str, Any] | None = None,
    dynamic_headers: Mapping[str, str] | None = None,
    session_id: str | None = None,
) -> tuple[Any, bool, dict[str, Any]]:
    """Returns the client, whether it authenticates with an OAuth token, and (MISAKA fork, see
    ``_sdk_client``) the headers every request has to carry."""
    # Copilot: Bearer auth.
    if model.provider == "github-copilot":
        client, request_headers = _sdk_client(
            api_key=None,
            auth_token=api_key,
            base_url=model.baseUrl,
            headers=merge_client_headers(
                {
                    "accept": "application/json",
                    "anthropic-dangerous-direct-browser-access": "true",
                },
                model.headers,
                dynamic_headers,
                options_headers,
            ),
        )
        return client, False, request_headers

    # OAuth: Bearer auth, Claude Code identity headers
    if api_key and is_oauth_token(api_key):
        client, request_headers = _sdk_client(
            api_key=None,
            auth_token=api_key,
            base_url=model.baseUrl,
            headers=merge_client_headers(
                {
                    "accept": "application/json",
                    "anthropic-dangerous-direct-browser-access": "true",
                    "user-agent": f"claude-cli/{CLAUDE_CODE_VERSION}",
                    "x-app": "cli",
                },
                model.headers,
                options_headers,
            ),
        )
        return client, True, request_headers

    # API key or header-owned auth.
    compat = get_anthropic_compat(model)
    session_affinity_headers: dict[str, str] = {}
    if session_id and compat["sendSessionAffinityHeaders"]:
        header = "x-session-id" if compat["sessionAffinityFormat"] == "openrouter" else "x-session-affinity"
        session_affinity_headers[header] = session_id
    default_headers = merge_client_headers(
        {
            "accept": "application/json",
            "anthropic-dangerous-direct-browser-access": "true",
        },
        session_affinity_headers,
        model.headers,
        options_headers,
    )
    client, request_headers = _sdk_client(
        api_key=api_key or None, auth_token=None, base_url=model.baseUrl, headers=default_headers
    )
    return client, False, request_headers


def get_beta_features(
    model: Model,
    context: TranscriptContext,
    is_oauth: bool,
    native_tool_changes: bool,
    options: Any = None,
) -> list[str]:
    configured_features: Any = ...
    for headers in (model.headers, _option(options, "headers")):
        for name, value in (headers or {}).items():
            if str(name).lower() == "anthropic-beta":
                configured_features = value
    if configured_features is None:
        return []
    if configured_features is not ...:
        return list(dict.fromkeys(
            feature.strip() for feature in str(configured_features).split(",") if feature.strip()
        ))

    features: list[str] = []
    if is_oauth:
        features.extend(("claude-code-20250219", "oauth-2025-04-20"))
    if should_use_fine_grained_tool_streaming_beta(model, context):
        features.append(FINE_GRAINED_TOOL_STREAMING_BETA)
    if (
        model.reasoning
        and _option(options, "thinkingEnabled") is True
        and _option(options, "interleavedThinking", True)
        and _compat_value(model, "forceAdaptiveThinking") is not True
    ):
        features.append(INTERLEAVED_THINKING_BETA)
    if should_use_server_side_fallback_beta(model):
        features.append(SERVER_SIDE_FALLBACK_BETA)
    if _compat_value(model, "supportsMidConvoEffort") is True:
        features.extend((MID_CONVERSATION_OUTPUT_CONFIG_BETA, THINKING_BINDING_CONTROLS_BETA))
    if native_tool_changes:
        features.append(MID_CONVERSATION_TOOL_CHANGES_BETA)
    return list(dict.fromkeys(features))


def build_params(
    model: Model,
    context: TranscriptContext,
    is_oauth: bool,
    options: Any = None,
) -> dict[str, Any]:
    cache_control = get_cache_control(model, _option(options, "cacheRetention"), _option(options, "env")).get(
        "cacheControl"
    )
    compat = get_anthropic_compat(model)
    initial_system_message = get_initial_system_message(context.messages)
    initial_system_text = get_system_message_text(initial_system_message) if initial_system_message else ""
    transformed_messages = transform_messages(context.messages, model, normalize_tool_call_id)
    conversation_messages = transformed_messages[1:] if initial_system_message else transformed_messages
    # Native tool changes reference tools by name, so a redefined name cannot be expressed,
    # and Anthropic rejects a tool list where every tool is deferred, so there must be an
    # initial active tool to anchor the deferred ones. Otherwise the current tool list is sent.
    initial_tools = (initial_system_message.toolsAdded if initial_system_message else None) or []
    native_tool_changes = bool(
        compat["supportsMidConvoSystemMessages"]
        and compat["supportsMidConvoToolChanges"]
        and len(initial_tools) > 0
        and not has_tool_redefinitions(context.messages)
    )
    managed_effort = _compat_value(model, "supportsMidConvoEffort") is True
    converted = convert_messages(
        conversation_messages,
        is_oauth,
        cache_control,
        bool(compat["allowEmptySignature"]),
        model.provider if managed_effort else None,
        native_tool_changes,
    )
    active_effort = _option(options, "effort") or "high"
    beta_features = get_beta_features(model, context, is_oauth, native_tool_changes, options)
    params: dict[str, Any] = {
        "model": model.id,
        "messages": insert_thinking_level_messages(converted, active_effort) if managed_effort else converted["messages"],
        "max_tokens": _option(options, "maxTokens", model.maxTokens),
        "stream": True,
        **({"betas": beta_features} if beta_features else {}),
    }

    # For OAuth tokens, we MUST include Claude Code identity
    if is_oauth:
        params["system"] = [
            {
                "type": "text",
                "text": "You are Claude Code, Anthropic's official CLI for Claude.",
                **({"cache_control": cache_control} if cache_control else {}),
            }
        ]
        if initial_system_text:
            params["system"].append({
                "type": "text",
                "text": sanitize_surrogates(initial_system_text),
                **({"cache_control": cache_control} if cache_control else {}),
            })
    elif initial_system_text:
        # Add cache control to system prompt for non-OAuth tokens
        params["system"] = [
            {
                "type": "text",
                "text": sanitize_surrogates(initial_system_text),
                **({"cache_control": cache_control} if cache_control else {}),
            }
        ]

    # Temperature is incompatible with extended thinking and unsupported on Claude Opus 4.7+.
    if (
        _option(options, "temperature") is not None
        and not _option(options, "thinkingEnabled")
        and not managed_effort
        and compat["supportsTemperature"]
    ):
        params["temperature"] = _option(options, "temperature")

    tool_cache_control = cache_control if compat["supportsCacheControlOnTools"] else None
    if native_tool_changes:
        # Initial tools stay active with the cache breakpoint on the last one. Every later
        # declaration is deferred and only surfaced by its `tool_addition` block; removed
        # tools stay declared and are withdrawn by `tool_removal`. The request-level list
        # therefore only grows, keeping the cached prefix intact across tool changes.
        initial_names = {tool.name for tool in initial_tools}
        later_tools = [tool for tool in get_declared_tools(context.messages) if tool.name not in initial_names]
        params["tools"] = [
            *convert_tools(
                initial_tools,
                is_oauth,
                bool(compat["supportsEagerToolInputStreaming"]),
                bool(compat["supportsStrictTools"]),
                tool_cache_control,
            ),
            DEFERRED_TOOL_PLACEHOLDER,
            *(
                {**tool, "defer_loading": True}
                for tool in convert_tools(
                    later_tools,
                    is_oauth,
                    bool(compat["supportsEagerToolInputStreaming"]),
                    bool(compat["supportsStrictTools"]),
                )
            ),
        ]
    else:
        tools = get_current_tools(context.messages)
        if len(tools) > 0:
            params["tools"] = convert_tools(
                tools,
                is_oauth,
                bool(compat["supportsEagerToolInputStreaming"]),
                bool(compat["supportsStrictTools"]),
                tool_cache_control,
            )

    # Managed effort models always use adaptive thinking so prefix mismatches can
    # be dropped instead of surfacing as persistent 400 responses.
    if managed_effort:
        params["thinking"] = {
            "type": "adaptive",
            "display": _option(options, "thinkingDisplay", "summarized"),
            "block_binding": {"prefix_mismatch_behavior": "drop_block"},
        }
        params["output_config"] = {"effort": "high"}
    elif model.reasoning:
        if _option(options, "thinkingEnabled"):
            # Default to "summarized" so Opus 4.7 and Mythos Preview behave like
            # older Claude 4 models (whose API default is also "summarized").
            display = _option(options, "thinkingDisplay", "summarized")
            if _compat_value(model, "forceAdaptiveThinking") is True:
                # Adaptive thinking: Claude decides when and how much to think.
                params["thinking"] = {"type": "adaptive", "display": display}
                if _option(options, "effort"):
                    params["output_config"] = {"effort": _option(options, "effort")}
            else:
                # Budget-based thinking for older models
                params["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": _option(options, "thinkingBudgetTokens") or 1024,
                    "display": display,
                }
        elif _option(options, "thinkingEnabled") is False and (
            model.thinkingLevelMap is None or model.thinkingLevelMap.get("off", ...) is not None
        ):
            params["thinking"] = {"type": "disabled"}

    metadata = _option(options, "metadata")
    if isinstance(metadata, Mapping) and isinstance(metadata.get("user_id"), str):
        params["metadata"] = {"user_id": metadata["user_id"]}

    tool_choice = _option(options, "toolChoice")
    if tool_choice:
        params["tool_choice"] = {"type": tool_choice} if isinstance(tool_choice, str) else tool_choice

    allowed_fallback_models = _compat_value(model, "allowedFallbackModels")
    if allowed_fallback_models:
        params["fallbacks"] = [{"model": fallback.model} for fallback in allowed_fallback_models]

    return params


_TOOL_CALL_ID_DISALLOWED = re.compile(r"[^a-zA-Z0-9_-]")


def normalize_tool_call_id(tool_call_id: str, _target_model: Model | None = None, _source: Any = None) -> str:
    """Normalize tool call IDs to match Anthropic's required pattern and length.

    ``transform_messages`` calls this with three arguments; pi's declares one and JavaScript
    discards the rest. The extra parameters are accepted and ignored.
    """
    return _TOOL_CALL_ID_DISALLOWED.sub("_", tool_call_id)[:64]


def convert_tool_result(message: ToolResultMessage) -> dict[str, Any]:
    return {
        "type": "tool_result",
        "tool_use_id": message.toolCallId,
        "content": convert_content_blocks(message.content),
        "is_error": message.isError,
    }


class ConvertedAnthropicMessages(TypedDict):
    messages: list[dict[str, Any]]
    assistantLevels: dict[int, AnthropicEffort]


def convert_messages(
    transformed_messages: list[Any],
    is_oauth: bool,
    cache_control: dict[str, Any] | None = None,
    allow_empty_signature: bool = False,
    managed_provider: str | None = None,
    native_tool_changes: bool = False,
) -> ConvertedAnthropicMessages:
    params: list[dict[str, Any]] = []
    assistant_levels: dict[int, AnthropicEffort] = {}
    # Later system messages are held back and emitted directly before the next assistant
    # message (or at the end of the transcript). Anthropic requires `tool_result` blocks to
    # immediately follow their `tool_use`, so a system message between them is rejected; this
    # also mirrors where the managed-effort system messages are inserted. As a result an
    # update placed before a user message in the transcript lands after it on the wire.
    pending_system_messages: list[dict[str, Any]] = []

    def flush_pending_system_messages() -> None:
        params.extend(pending_system_messages)
        pending_system_messages.clear()

    i = 0
    while i < len(transformed_messages):
        msg = transformed_messages[i]

        if msg.role == "system":
            # Later system messages only reach this point when the model accepts them natively;
            # otherwise the transcript was collapsed into the leading message before conversion.
            text = render_system_message_update(msg)
            blocks: list[dict[str, Any]] = []
            if len(text) > 0:
                blocks.append({"type": "text", "text": sanitize_surrogates(text)})
            if native_tool_changes:
                for tool in msg.toolsRemoved or []:
                    blocks.append({
                        "type": "tool_removal",
                        "tool": {"type": "tool_reference", "name": to_claude_code_name(tool.name) if is_oauth else tool.name},
                    })
                for tool in msg.toolsAdded or []:
                    blocks.append({
                        "type": "tool_addition",
                        "tool": {"type": "tool_reference", "name": to_claude_code_name(tool.name) if is_oauth else tool.name},
                    })
            if len(blocks) > 0:
                pending_system_messages.append({"role": "system", "content": blocks})
        elif msg.role == "user":
            if isinstance(msg.content, str):
                if len(msg.content.strip()) > 0:
                    params.append({"role": "user", "content": sanitize_surrogates(msg.content)})
            else:
                blocks = [
                    {"type": "text", "text": sanitize_surrogates(item.text)}
                    if item.type == "text"
                    else {
                        "type": "image",
                        "source": {"type": "base64", "media_type": item.mimeType, "data": item.data},
                    }
                    for item in msg.content
                ]
                filtered_blocks = [
                    block for block in blocks if block["type"] != "text" or len(block["text"].strip()) > 0
                ]
                if len(filtered_blocks) > 0:
                    params.append({"role": "user", "content": filtered_blocks})
        elif msg.role == "assistant":
            flush_pending_system_messages()
            blocks = []

            for block in msg.content:
                if block.type == "text":
                    if len(block.text.strip()) == 0:
                        continue
                    blocks.append({"type": "text", "text": sanitize_surrogates(block.text)})
                elif block.type == "thinking":
                    # Redacted thinking: pass the opaque payload back as redacted_thinking
                    if block.redacted:
                        blocks.append({"type": "redacted_thinking", "data": block.thinkingSignature})
                        continue
                    thinking_signature = block.thinkingSignature
                    has_thinking_signature = bool(thinking_signature) and len(thinking_signature.strip()) > 0
                    if len(block.thinking.strip()) == 0 and not has_thinking_signature:
                        continue
                    # If thinking signature is missing/empty (e.g., from aborted stream),
                    # convert to plain text for Anthropic. Some compatible providers emit
                    # and accept empty signatures, so let marked models preserve the block.
                    if not has_thinking_signature:
                        blocks.append(
                            {"type": "thinking", "thinking": sanitize_surrogates(block.thinking), "signature": ""}
                            if allow_empty_signature
                            else {"type": "text", "text": sanitize_surrogates(block.thinking)}
                        )
                    else:
                        blocks.append({
                            "type": "thinking",
                            "thinking": sanitize_surrogates(block.thinking),
                            "signature": thinking_signature,
                        })
                elif block.type == "toolCall":
                    blocks.append({
                        "type": "tool_use",
                        "id": block.id,
                        "name": to_claude_code_name(block.name) if is_oauth else block.name,
                        "input": block.arguments or {},
                    })
            if len(blocks) == 0:
                i += 1
                continue
            message_index = len(params)
            params.append({"role": "assistant", "content": blocks})
            if (
                managed_provider is not None
                and msg.api == "anthropic-messages"
                and msg.provider == managed_provider
                and is_anthropic_effort(msg.providerThinkingLevel)
            ):
                assistant_levels[message_index] = msg.providerThinkingLevel
        elif msg.role == "toolResult":
            # Collect all consecutive toolResult messages, needed for z.ai Anthropic endpoint.
            tool_results: list[dict[str, Any]] = []
            j = i
            while j < len(transformed_messages) and transformed_messages[j].role == "toolResult":
                tool_results.append(convert_tool_result(transformed_messages[j]))
                j += 1

            # Skip the messages we've already processed.
            i = j - 1

            params.append({"role": "user", "content": tool_results})
        i += 1

    flush_pending_system_messages()

    # Add cache_control to the last user or system message to cache conversation history
    if cache_control and len(params) > 0:
        last_message = params[-1]
        if last_message["role"] in {"user", "system"}:
            if isinstance(last_message["content"], list):
                last_block = last_message["content"][-1] if last_message["content"] else None
                if last_block is not None and last_block["type"] in {
                    "text", "image", "tool_result", "tool_addition", "tool_removal",
                }:
                    if len(last_message["content"]) == 1 and last_block["type"] == "text":
                        _apply_stable_prefix_cache_marker(last_message, last_block["text"], cache_control)
                    else:
                        last_block["cache_control"] = cache_control
            elif isinstance(last_message["content"], str):
                _apply_stable_prefix_cache_marker(last_message, last_message["content"], cache_control)

    return {"messages": params, "assistantLevels": assistant_levels}


def _apply_stable_prefix_cache_marker(message: dict[str, Any], text: str, cache_control: dict[str, Any]) -> None:
    """MISAKA fork: pi marks the whole last text. A user message whose builder declared a stable
    prefix (the research scaffold) carries the breakpoint on that prefix instead, and its
    volatile tail rides unmarked, so the next request still hits the cached prefix."""
    from misaka.utils.prompt_cache_wire import _apply_cache_marker

    text_message = {"role": message["role"], "content": text}
    _apply_cache_marker(text_message, cache_control, native_anthropic=True)
    message["content"] = text_message["content"]


def is_anthropic_effort(value: Any) -> bool:
    return value in ("low", "medium", "high", "xhigh", "max")


def insert_thinking_level_messages(
    converted: ConvertedAnthropicMessages,
    active_effort: str,
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for index, message in enumerate(converted["messages"]):
        historical_effort = converted["assistantLevels"].get(index)
        if historical_effort is not None:
            messages.append({"role": "system", "content": [], "output_config": {"effort": historical_effort}})
        messages.append(message)
    messages.append({"role": "system", "content": [], "output_config": {"effort": active_effort}})
    return messages


def should_use_fine_grained_tool_streaming_beta(model: Model, context: TranscriptContext) -> bool:
    return len(get_current_tools(context.messages)) > 0 and not get_anthropic_compat(model)[
        "supportsEagerToolInputStreaming"
    ]


_ROOT_COMBINATORS = frozenset({"oneOf", "allOf", "anyOf"})


def convert_tools(
    tools: list[Tool],
    is_oauth: bool,
    supports_eager_tool_input_streaming: bool,
    supports_strict_tools: bool,
    cache_control: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if not tools:
        return []

    converted: list[dict[str, Any]] = []
    for index, tool in enumerate(tools):
        # MISAKA fork: Anthropic rejects root combinators even without strict sampling, so a
        # schema that has them is never marked strict; and the non-strict schema keeps its
        # `$defs` and other keywords (pi sends only type/properties/required), because
        # MISAKA's tools reference `$defs` from their properties.
        parameters = get_json_schema_tool_parameters(tool, None)
        strict = resolve_json_schema_strict_sampling(
            tool, supports_strict_tools and not _ROOT_COMBINATORS.intersection(parameters)
        )
        if strict is True:
            parameters = get_json_schema_tool_parameters(tool, strict)
        input_schema = {
            **{key: value for key, value in parameters.items() if key not in _ROOT_COMBINATORS},
            "type": "object",
            "properties": parameters.get("properties", {}),
            "required": parameters.get("required", []),
        }

        converted.append({
            "name": to_claude_code_name(tool.name) if is_oauth else tool.name,
            "description": tool.description,
            **({"eager_input_streaming": True} if supports_eager_tool_input_streaming else {}),
            **({"strict": True} if strict is True else {}),
            "input_schema": input_schema,
            **({"cache_control": cache_control} if cache_control and index == len(tools) - 1 else {}),
        })
    return converted


class _StopReasonResult(TypedDict):
    stopReason: StopReason
    errorMessage: NotRequired[str]


def map_stop_reason(reason: str, stop_details: Mapping[str, Any] | None = None) -> _StopReasonResult:
    if reason == "end_turn":
        return {"stopReason": "stop"}
    if reason == "max_tokens":
        return {"stopReason": "length"}
    if reason == "tool_use":
        return {"stopReason": "toolUse"}
    if reason == "refusal":
        explanation = stop_details.get("explanation") if isinstance(stop_details, Mapping) else None
        return {
            "stopReason": "error",
            "errorMessage": explanation or "The model refused to complete the request",
        }
    if reason == "pause_turn":  # Stop is good enough -> resubmit
        return {"stopReason": "stop"}
    if reason == "stop_sequence":
        return {"stopReason": "stop"}  # We don't supply stop sequences, so this should never happen
    if reason == "sensitive":  # Content flagged by safety filters (not yet in SDK types)
        return {"stopReason": "error", "errorMessage": "Provider stopped with: sensitive"}
    # Handle unknown stop reasons gracefully (API may add new values)
    raise RuntimeError(f"Unhandled stop reason: {reason}")


# --- MISAKA: Python SDK plumbing ---------------------------------------------------------


async def _create_raw_response(
    client: Any, params: dict[str, Any], options: Any = None, request_headers: dict[str, Any] | None = None
) -> Any:
    """``client.beta.messages.create(params, {signal, timeout, maxRetries: 0}).asResponse()``.

    The SDK's own retry is always off: ``utils/provider_retry`` decides what is worth retrying.
    """
    signal = _option(options, "signal")
    request_options: dict[str, Any] = {"max_retries": 0}
    if _option(options, "timeoutMs") is not None:
        request_options["timeout"] = _option(options, "timeoutMs") / 1000
    request_client = client.with_options(**request_options) if hasattr(client, "with_options") else client
    call_options: dict[str, Any] = {"extra_headers": dict(request_headers)} if request_headers else {}

    beta = getattr(request_client, "beta", None)
    messages = getattr(beta, "messages", None) or getattr(request_client, "messages", None)
    if hasattr(messages, "with_raw_response"):
        return await _await_maybe_with_signal(messages.with_raw_response.create(**params, **call_options), signal)

    created = await _await_maybe_with_signal(messages.create(**params, **call_options), signal)
    if hasattr(created, "asResponse"):
        return await _await_maybe_with_signal(created.asResponse(), signal)
    return created


async def _emit_response_metadata(response: Any, options: Any, model: Model) -> None:
    on_response = _option(options, "onResponse")
    if not callable(on_response):
        return

    if hasattr(response, "http_response"):
        http_response = response.http_response
        await maybe_await(
            on_response(
                {"status": http_response.status_code, "headers": headers_to_record(http_response.headers)},
                model,
            )
        )
        return

    status = getattr(response, "status_code", None)
    if status is None:
        status = getattr(response, "status", None)
    headers = getattr(response, "headers", None)
    if isinstance(status, int) and headers is not None:
        await maybe_await(on_response({"status": status, "headers": headers_to_record(headers)}, model))


async def _iter_event_objects(stream_like: Any, signal: Any = None) -> AsyncIterator[dict[str, Any]]:
    """Events from an injected client that streams SDK event objects instead of SSE bytes."""
    async for event in _iterate_async_iterable(stream_like, signal, on_abort=lambda: _close_stream(stream_like)):
        if hasattr(event, "model_dump"):
            dumped = event.model_dump()
            if isinstance(dumped, dict):
                yield dumped
                continue
        if isinstance(event, dict):
            yield event
            continue
        try:
            yield json.loads(json.dumps(event, default=lambda value: value.__dict__))
        except Exception as error:
            raise RuntimeError(f"Could not serialize Anthropic stream event: {error}") from error


def _note_empty_tool_arguments(output: AssistantMessage, block: ToolCall, accumulated: StreamingArgs | None,
                               dropped_deltas: dict[str, int]) -> None:
    """MISAKA addition: a tool call that ends with ``{}`` although its input was streamed is a
    silent failure with two possible causes -- the streamed buffer did not parse, or its
    deltas were framed under a block that never started -- and the tool's validation error
    ("received {}") shows neither. The message keeps a diagnostic with what actually arrived.
    A tool call that genuinely takes no arguments records nothing."""
    if block.arguments != {}:
        return
    raw = accumulated.raw if accumulated is not None else ""
    if raw.strip() not in ("", "{}"):
        kind, why = "tool_arguments_unparsed", f"{len(raw)} streamed characters did not parse as a JSON object"
    elif dropped_deltas.get("input_json_delta"):
        kind, why = "tool_arguments_missing", "input_json_delta events arrived for a block that was never started"
    else:
        return
    append_assistant_message_diagnostic(output, create_assistant_message_diagnostic(
        kind, RuntimeError(f"tool {block.name}: {why}"),
        {"tool": block.name, "toolCallId": block.id, "rawLength": len(raw),
         "rawHead": raw[:200], "rawTail": raw[-200:] if len(raw) > 200 else "",
         "droppedDeltas": dict(dropped_deltas)}))


streamAnthropic = stream_anthropic
streamSimpleAnthropic = stream_simple_anthropic
__all__ = [
    "AnthropicEffort",
    "AnthropicOptions",
    "AnthropicThinkingDisplay",
    "streamAnthropic",
    "streamSimpleAnthropic",
]
