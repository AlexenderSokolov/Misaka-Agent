"""Mistral Conversations provider adapter."""

from __future__ import annotations

import json
import math
import re
import time
from collections.abc import AsyncIterable, AsyncIterator, Mapping
from typing import Any, Literal, TypedDict, cast
from urllib.parse import urlsplit, urlunsplit

import httpx

from misaka.ai.models import calculate_cost, clamp_thinking_level
from misaka.ai.providers._common import (
    _await_with_signal,
    _empty_usage,
    _option,
    safe_json_stringify,
)
from misaka.ai.providers.constrained_sampling import (
    get_json_schema_tool_parameters,
    resolve_json_schema_strict_sampling,
)
from misaka.ai.providers.simple_options import build_base_options
from misaka.ai.providers.transform_messages import transform_messages
from misaka.ai.types import (
    AssistantMessage,
    DoneEvent,
    ErrorEvent,
    MessageValue,
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
    TranscriptContext,
)
from misaka.ai.utils.abort import (
    TimeoutAbortSignal,
    abort_reason,
    combine_abort_signals,
)
from misaka.ai.utils.event_stream import AssistantMessageEventStream, spawn_stream_task
from misaka.ai.utils.hash import short_hash
from misaka.ai.utils.headers import apply_provider_headers, headers_to_record
from misaka.ai.utils.json_parse import StreamingArgs
from misaka.ai.utils.sanitize_unicode import sanitize_surrogates
from misaka.ai.utils.text import get_system_message_text, render_system_message_update
from misaka.ai.utils.transcript import get_current_tools, resolve_transcript
from misaka.ai.utils.user_agent import get_misaka_user_agent
from misaka.utils.values import maybe_await, signal_aborted

MISTRAL_TOOL_CALL_ID_LENGTH = 9
MAX_MISTRAL_ERROR_BODY_CHARS = 4000
_MISTRAL_ID_DISALLOWED = re.compile(r"[^a-zA-Z0-9]")

# Provider-specific options for the Mistral API.
MistralReasoningEffort = Literal["none", "low", "medium", "high", "max"]


class MistralToolChoiceFunction(TypedDict):
    name: str


class MistralToolChoiceObject(TypedDict):
    type: Literal["function"]
    function: MistralToolChoiceFunction


class MistralOptions(TypedDict, total=False):
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
    toolChoice: Literal["auto", "none", "any", "required"] | MistralToolChoiceObject
    promptMode: Literal["reasoning"]
    reasoningEffort: MistralReasoningEffort


def stream_mistral(
    model: Model,
    context: TranscriptContext,
    options: StreamOptions | Mapping[str, Any] | None = None,
) -> AssistantMessageEventStream:
    """Stream responses from the native Mistral Chat Completions endpoint."""
    stream = AssistantMessageEventStream()
    normalized_context = resolve_transcript(
        context, getattr(getattr(model, "compat", None), "supportsMidConvoSystemMessages", None)
    )

    async def run() -> None:
        output = create_output(model)

        try:
            api_key = _option(options, "apiKey")
            if not api_key:
                raise RuntimeError(f"No API key for provider: {model.provider}")

            normalize_mistral_tool_call_id = create_mistral_tool_call_id_normalizer()
            transformed_messages = transform_messages(
                normalized_context.messages,
                model,
                lambda tool_call_id, _target_model, _source: normalize_mistral_tool_call_id(tool_call_id),
            )

            payload = build_chat_payload(model, normalized_context, transformed_messages, options)
            on_payload = _option(options, "onPayload")
            if callable(on_payload):
                next_payload = await maybe_await(on_payload(payload, model))
                if next_payload is not None:
                    payload = next_payload
            mistral_stream = await request_mistral_stream(model, payload, api_key, options)
            stream.push(StartEvent(partial=output))
            try:
                await consume_chat_stream(model, output, stream, mistral_stream, _option(options, "onProviderStreamEvent"))
            finally:
                # MISAKA fork: an async generator left mid-iteration is not released until the
                # loop finalizes it; closing it here releases the response straight away.
                await mistral_stream.aclose()

            if signal_aborted(_option(options, "signal")):
                raise RuntimeError("Request was aborted")

            if output.stopReason == "pending":
                raise RuntimeError("Mistral stream ended without a finish reason")
            if output.stopReason in {"aborted", "error"}:
                raise RuntimeError(output.errorMessage or "An unknown error occurred")

            stream.push(DoneEvent(reason=output.stopReason, message=output))
        except Exception as error:  # noqa: BLE001
            output.stopReason = "aborted" if signal_aborted(_option(options, "signal")) else "error"
            output.errorMessage = format_mistral_error(error)
            stream.push(ErrorEvent(reason=output.stopReason, error=output), cause=error)
        finally:
            stream.end()

    spawn_stream_task(run(), stream=stream)
    return stream


def stream_simple_mistral(
    model: Model,
    context: TranscriptContext,
    options: SimpleStreamOptions | None = None,
) -> AssistantMessageEventStream:
    """Maps provider-agnostic `SimpleStreamOptions` to Mistral options."""
    api_key = _option(options, "apiKey")
    if not api_key:
        raise RuntimeError(f"No API key for provider: {model.provider}")

    base = {
        **build_base_options(model, context, options, api_key).model_dump(),
        "toolChoice": _option(options, "toolChoice"),
    }
    clamped_reasoning = clamp_thinking_level(model, options.reasoning) if options and options.reasoning else None
    reasoning = None if clamped_reasoning == "off" else clamped_reasoning
    # Models with a thinking level map use `reasoning_effort`; other reasoning models use `prompt_mode`.
    effort_map = model.thinkingLevelMap if model.reasoning else None
    reasoning_effort = None
    if effort_map is not None:
        if reasoning:
            mapped = effort_map.get(reasoning)
            reasoning_effort = "high" if mapped is None else mapped
        else:
            reasoning_effort = effort_map.get("off")

    return stream_mistral(
        model,
        context,
        {
            **base,
            "promptMode": "reasoning" if model.reasoning and effort_map is None and reasoning else None,
            "reasoningEffort": reasoning_effort,
        },
    )


def create_output(model: Model) -> AssistantMessage:
    return AssistantMessage(
        content=[],
        api=model.api,
        provider=model.provider,
        model=model.id,
        usage=_empty_usage(),
        stopReason="pending",
        timestamp=time.time_ns() // 1_000_000,
    )


def create_mistral_tool_call_id_normalizer() -> Any:
    id_map: dict[str, str] = {}
    reverse_map: dict[str, str] = {}

    def normalize(tool_call_id: str) -> str:
        existing = id_map.get(tool_call_id)
        if existing:
            return existing

        attempt = 0
        while True:
            candidate = derive_mistral_tool_call_id(tool_call_id, attempt)
            owner = reverse_map.get(candidate)
            if not owner or owner == tool_call_id:
                id_map[tool_call_id] = candidate
                reverse_map[candidate] = tool_call_id
                return candidate
            attempt += 1

    return normalize


def derive_mistral_tool_call_id(tool_call_id: str, attempt: int) -> str:
    normalized = _MISTRAL_ID_DISALLOWED.sub("", tool_call_id)
    if attempt == 0 and len(normalized) == MISTRAL_TOOL_CALL_ID_LENGTH:
        return normalized
    seed_base = normalized or tool_call_id
    seed = seed_base if attempt == 0 else f"{seed_base}:{attempt}"
    return _MISTRAL_ID_DISALLOWED.sub("", short_hash(seed))[:MISTRAL_TOOL_CALL_ID_LENGTH]


def format_mistral_error(error: Any) -> str:
    if isinstance(error, Exception):
        status_code = getattr(error, "statusCode", None)
        status_code = status_code if isinstance(status_code, int) and not isinstance(status_code, bool) else None
        body = getattr(error, "body", None)
        body_text = body.strip() if isinstance(body, str) else None
        if status_code is not None and body_text:
            return f"Mistral API error ({status_code}): {truncate_error_text(body_text, MAX_MISTRAL_ERROR_BODY_CHARS)}"
        if status_code is not None:
            return f"Mistral API error ({status_code}): {error}"
        return str(error)
    return safe_json_stringify(error)


def truncate_error_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}... [truncated {len(text) - max_chars} chars]"


async def request_mistral_stream(
    model: Model,
    payload: Mapping[str, Any],
    api_key: str,
    options: StreamOptions | Mapping[str, Any] | None = None,
) -> AsyncIterator[dict[str, Any]]:
    base_url = urlsplit(model.baseUrl)
    url = urlunsplit((base_url.scheme, base_url.netloc, f"{base_url.path.rstrip('/')}/v1/chat/completions", "", ""))
    headers = build_mistral_headers(model, api_key, options)
    timeout_ms = _option(options, "timeoutMs")
    timeout_signal = TimeoutAbortSignal(timeout_ms if timeout_ms is not None else 60_000)
    signal = combine_abort_signals(_option(options, "signal"), timeout_signal)
    # MISAKA fork: `fetch` has no timeouts of its own, the signal above is the only bound;
    # httpx's defaults would add a five-second one.
    client = httpx.AsyncClient(timeout=httpx.Timeout(None))
    try:
        request = client.build_request(
            "POST",
            url,
            headers=headers,
            # ensure_ascii escapes a lone surrogate the way JSON.stringify does.
            content=json.dumps(to_mistral_wire_payload(payload), separators=(",", ":")),
        )
        response = await _await_or_throw_reason(client.send(request, stream=True), signal)

        on_response = _option(options, "onResponse")
        if callable(on_response):
            await maybe_await(on_response({"status": response.status_code, "headers": headers_to_record(response.headers)}, model))

        if not response.is_success:
            body = (await _await_or_throw_reason(response.aread(), signal)).decode("utf-8", errors="replace")
            await response.aclose()
            raise MistralHttpError(response.status_code, body, response.reason_phrase)
    except BaseException:
        await client.aclose()
        raise

    return read_mistral_events(response, signal, client)


class MistralHttpError(Exception):
    def __init__(self, status_code: int, body: str, status_text: str) -> None:
        super().__init__(status_text or f"Request failed with status {status_code}")
        self.name = "MistralHttpError"
        self.statusCode = status_code
        self.body = body


def build_mistral_headers(
    model: Model, api_key: str, options: StreamOptions | Mapping[str, Any] | None = None
) -> dict[str, str]:
    headers: dict[str, str] = {
        "User-Agent": get_misaka_user_agent(),
        "accept": "text/event-stream",
        "authorization": f"Bearer {api_key}",
        "content-type": "application/json",
    }
    apply_provider_headers(headers, model.headers)
    apply_provider_headers(headers, _option(options, "headers"))

    has_explicit_affinity = has_mistral_header_override(model.headers, "x-affinity") or has_mistral_header_override(
        _option(options, "headers"), "x-affinity"
    )
    if should_use_prompt_caching(options) and not has_explicit_affinity:
        apply_provider_headers(headers, {"x-affinity": _option(options, "sessionId")})

    return headers


def has_mistral_header_override(overrides: Mapping[str, Any] | None, target: str) -> bool:
    return bool(overrides) and any(str(name).lower() == target for name in overrides)


_MISTRAL_WIRE_KEYS = (
    ("topP", "top_p"),
    ("maxTokens", "max_tokens"),
    ("randomSeed", "random_seed"),
    ("responseFormat", "response_format"),
    ("toolChoice", "tool_choice"),
    ("presencePenalty", "presence_penalty"),
    ("frequencyPenalty", "frequency_penalty"),
    ("parallelToolCalls", "parallel_tool_calls"),
    ("reasoningEffort", "reasoning_effort"),
    ("promptMode", "prompt_mode"),
    ("promptCacheKey", "prompt_cache_key"),
    ("safePrompt", "safe_prompt"),
)
_MISTRAL_WIRE_CHUNK_KEYS = (
    ("imageUrl", "image_url"),
    ("documentUrl", "document_url"),
    ("documentName", "document_name"),
    ("fileId", "file_id"),
    ("referenceIds", "reference_ids"),
    ("inputAudio", "input_audio"),
)


def to_mistral_wire_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    wire_payload: dict[str, Any] = dict(payload)
    for source, target in _MISTRAL_WIRE_KEYS:
        remap_mistral_property(wire_payload, source, target)
    wire_payload["messages"] = [to_mistral_wire_message(message) for message in payload["messages"]]

    response_format = wire_payload.get("response_format")
    if is_mistral_record(response_format):
        wire_response_format = dict(response_format)
        remap_mistral_property(wire_response_format, "jsonSchema", "json_schema")
        json_schema = wire_response_format.get("json_schema")
        if is_mistral_record(json_schema):
            wire_json_schema = dict(json_schema)
            remap_mistral_property(wire_json_schema, "schemaDefinition", "schema")
            wire_response_format["json_schema"] = wire_json_schema
        wire_payload["response_format"] = wire_response_format

    return wire_payload


def to_mistral_wire_message(message: Mapping[str, Any]) -> dict[str, Any]:
    wire_message: dict[str, Any] = dict(message)
    remap_mistral_property(wire_message, "toolCalls", "tool_calls")
    remap_mistral_property(wire_message, "toolCallId", "tool_call_id")
    if isinstance(message.get("content"), list):
        wire_message["content"] = [to_mistral_wire_content_chunk(chunk) for chunk in message["content"]]
    return wire_message


def to_mistral_wire_content_chunk(chunk: Mapping[str, Any]) -> dict[str, Any]:
    wire_chunk: dict[str, Any] = dict(chunk)
    for source, target in _MISTRAL_WIRE_CHUNK_KEYS:
        remap_mistral_property(wire_chunk, source, target)
    return wire_chunk


def remap_mistral_property(record: dict[str, Any], source: str, target: str) -> None:
    if source not in record:
        return
    record[target] = record.pop(source)


def is_mistral_record(value: Any) -> bool:
    return isinstance(value, Mapping)


MISTRAL_STREAM_DONE = object()
_MISTRAL_EVENT_BOUNDARY = re.compile(r"\r\n\r\n|\r\n\r|\r\n\n|\r\r\n|\n\r\n|\r\r|\n\r|\n\n")
_MISTRAL_LINE_BREAK = re.compile(r"\r\n|\r|\n")


async def read_mistral_events(
    response: httpx.Response, signal: Any, client: httpx.AsyncClient | None = None
) -> AsyncIterator[dict[str, Any]]:
    buffer = ""
    reader = response.aiter_text().__aiter__()

    try:
        while True:
            if signal_aborted(signal):
                raise abort_reason(signal)
            try:
                value = await _await_or_throw_reason(reader.__anext__(), signal)
                done = False
            except StopAsyncIteration:
                value = ""
                done = True
            if signal_aborted(signal):
                raise abort_reason(signal)
            buffer += value

            boundary = find_mistral_event_boundary(buffer)
            while boundary is not None:
                event = parse_mistral_event(buffer[: boundary[0]])
                buffer = buffer[boundary[0] + boundary[1] :]
                if event is MISTRAL_STREAM_DONE:
                    return
                if event is not None:
                    yield cast(dict[str, Any], event)
                boundary = find_mistral_event_boundary(buffer)

            if done:
                break

        if buffer.strip():
            event = parse_mistral_event(buffer)
            if event is not MISTRAL_STREAM_DONE and event is not None:
                yield cast(dict[str, Any], event)
    finally:
        await response.aclose()
        if client is not None:
            await client.aclose()


def find_mistral_event_boundary(buffer: str) -> tuple[int, int] | None:
    match = _MISTRAL_EVENT_BOUNDARY.search(buffer)
    return None if match is None else (match.start(), len(match.group(0)))


def parse_mistral_event(raw: str) -> dict[str, Any] | object | None:
    data = "\n".join(
        line[5:].lstrip() for line in _MISTRAL_LINE_BREAK.split(raw) if line.startswith("data:")
    ).strip()
    if not data:
        return None
    if data == "[DONE]":
        return MISTRAL_STREAM_DONE

    parsed = json.loads(data)
    if not is_mistral_record(parsed) or not isinstance(parsed.get("choices"), list):
        raise RuntimeError("Invalid Mistral streaming event")
    return {"data": parsed}


async def _await_or_throw_reason(awaitable: Any, signal: Any) -> Any:
    """Await under ``signal``, raising its reason when it fires -- what a fetch or a read
    rejects with once its signal aborts."""
    try:
        return await _await_with_signal(awaitable, signal)
    except RuntimeError:
        if signal_aborted(signal):
            raise abort_reason(signal) from None
        raise


def build_chat_payload(
    model: Model,
    context: TranscriptContext,
    messages: list[MessageValue],
    options: StreamOptions | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model.id,
        "stream": True,
        "messages": to_chat_messages(messages, "image" in model.input),
    }

    current_tools = get_current_tools(context.messages)
    if len(current_tools) > 0:
        payload["tools"] = to_function_tools(current_tools)
    if _option(options, "temperature") is not None:
        payload["temperature"] = _option(options, "temperature")
    if _option(options, "maxTokens") is not None:
        payload["maxTokens"] = _option(options, "maxTokens")
    if _option(options, "toolChoice"):
        payload["toolChoice"] = map_tool_choice(_option(options, "toolChoice"))
    if _option(options, "promptMode"):
        payload["promptMode"] = _option(options, "promptMode")
    if _option(options, "reasoningEffort"):
        payload["reasoningEffort"] = _option(options, "reasoningEffort")
    if should_use_prompt_caching(options):
        payload["promptCacheKey"] = _option(options, "sessionId")

    return payload


def should_use_prompt_caching(options: Any) -> bool:
    return _option(options, "cacheRetention") != "none" and bool(_option(options, "sessionId"))


def get_mistral_cached_prompt_tokens(usage: Mapping[str, Any], prompt_tokens: int) -> int:
    raw_cached_tokens = next(
        (
            value
            for value in (
                (usage.get("promptTokensDetails") or {}).get("cachedTokens"),
                (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
                (usage.get("promptTokenDetails") or {}).get("cachedTokens"),
                (usage.get("prompt_token_details") or {}).get("cached_tokens"),
                usage.get("numCachedTokens"),
                usage.get("num_cached_tokens"),
            )
            if value is not None
        ),
        0,
    )
    cached_tokens = (
        raw_cached_tokens
        if isinstance(raw_cached_tokens, (int, float))
        and not isinstance(raw_cached_tokens, bool)
        and math.isfinite(raw_cached_tokens)
        else 0
    )
    return int(min(prompt_tokens, max(0, cached_tokens)))


async def consume_chat_stream(
    model: Model,
    output: AssistantMessage,
    stream: AssistantMessageEventStream,
    mistral_stream: AsyncIterable[dict[str, Any]],
    on_provider_stream_event: Any = None,
) -> None:
    current_block: TextContent | ThinkingContent | None = None
    tool_blocks_by_key: dict[str | int, int] = {}
    # MISAKA fork: the arguments accumulate in StreamingArgs beside the block, not in a
    # `partialArgs` field on it, and the final parse flags malformed JSON.
    partial_args_by_index: dict[int, StreamingArgs] = {}

    def block_index() -> int:
        return len(output.content) - 1

    def finish_current_block(block: TextContent | ThinkingContent | None) -> None:
        if block is None:
            return
        if block.type == "text":
            stream.push(TextEndEvent(contentIndex=block_index(), content=block.text, partial=output))
            return
        if block.type == "thinking":
            stream.push(ThinkingEndEvent(contentIndex=block_index(), content=block.thinking, partial=output))

    async for event in mistral_stream:
        chunk = event["data"]
        if callable(on_provider_stream_event):
            await maybe_await(on_provider_stream_event(chunk, model))
        # Mistral's streamed CompletionChunk carries an id field. Keep the first non-empty one,
        # mirroring how OpenAI-style streaming exposes a stable response identifier per stream.
        output.responseId = output.responseId or chunk.get("id")

        usage = chunk.get("usage")
        if usage:
            prompt_tokens = usage.get("prompt_tokens") or 0
            cached_prompt_tokens = get_mistral_cached_prompt_tokens(usage, prompt_tokens)

            output.usage.input = max(0, prompt_tokens - cached_prompt_tokens)
            output.usage.output = usage.get("completion_tokens") or 0
            output.usage.cacheRead = cached_prompt_tokens
            output.usage.cacheWrite = 0
            output.usage.totalTokens = usage.get("total_tokens") or (
                output.usage.input + output.usage.output + output.usage.cacheRead + output.usage.cacheWrite
            )
            calculate_cost(model, output.usage)

        choices = chunk.get("choices") or []
        choice = choices[0] if choices else None
        if not choice:
            continue

        if choice.get("finish_reason"):
            output.rawStopReason = choice["finish_reason"]
            stop_reason_result = map_chat_stop_reason(choice["finish_reason"])
            output.stopReason = cast(StopReason, stop_reason_result["stopReason"])
            if stop_reason_result.get("errorMessage"):
                output.errorMessage = stop_reason_result["errorMessage"]

        delta = choice.get("delta") or {}
        if delta.get("content") is not None:
            content_items = [delta["content"]] if isinstance(delta["content"], str) else delta["content"]
            for item in content_items:
                if isinstance(item, str):
                    text_delta = sanitize_surrogates(item)
                    # GLM models on Mistral send empty content deltas around thinking and tool calls.
                    # Opening a block for them splits thinking into multiple blocks, which Mistral rejects on replay.
                    if not text_delta:
                        continue
                    if current_block is None or current_block.type != "text":
                        finish_current_block(current_block)
                        current_block = TextContent(text="")
                        output.content.append(current_block)
                        stream.push(TextStartEvent(contentIndex=block_index(), partial=output))
                    current_block.text += text_delta
                    stream.push(TextDeltaEvent(contentIndex=block_index(), delta=text_delta, partial=output))
                    continue

                if item.get("type") == "thinking":
                    delta_text = "".join(
                        text for text in (part.get("text") or "" for part in item.get("thinking") or []) if len(text) > 0
                    )
                    thinking_delta = sanitize_surrogates(delta_text)
                    if not thinking_delta:
                        continue
                    if current_block is None or current_block.type != "thinking":
                        finish_current_block(current_block)
                        current_block = ThinkingContent(thinking="")
                        output.content.append(current_block)
                        stream.push(ThinkingStartEvent(contentIndex=block_index(), partial=output))
                    current_block.thinking += thinking_delta
                    stream.push(ThinkingDeltaEvent(contentIndex=block_index(), delta=thinking_delta, partial=output))
                    continue

                if item.get("type") == "text":
                    text_delta = sanitize_surrogates(item.get("text") or "")
                    if not text_delta:
                        continue
                    if current_block is None or current_block.type != "text":
                        finish_current_block(current_block)
                        current_block = TextContent(text="")
                        output.content.append(current_block)
                        stream.push(TextStartEvent(contentIndex=block_index(), partial=output))
                    current_block.text += text_delta
                    stream.push(TextDeltaEvent(contentIndex=block_index(), delta=text_delta, partial=output))

        tool_calls = delta.get("tool_calls") or []
        for tool_call in tool_calls:
            if current_block is not None:
                finish_current_block(current_block)
                current_block = None
            call_id = (
                tool_call["id"]
                if tool_call.get("id") and tool_call["id"] != "null"
                else derive_mistral_tool_call_id(f"toolcall:{tool_call.get('index') or 0}", 0)
            )
            key = tool_call["index"] if tool_call.get("index") is not None else call_id
            existing_index = tool_blocks_by_key.get(key)
            block: ToolCall | None = None

            if existing_index is not None:
                existing = output.content[existing_index]
                if existing.type == "toolCall":
                    block = existing

            if block is None:
                block = ToolCall(id=call_id, name=tool_call["function"]["name"], arguments={})
                output.content.append(block)
                tool_blocks_by_key[key] = len(output.content) - 1
                stream.push(ToolCallStartEvent(contentIndex=len(output.content) - 1, partial=output))

            arguments = tool_call["function"].get("arguments")
            args_delta = (
                arguments
                if isinstance(arguments, str)
                else json.dumps(arguments or {}, separators=(",", ":"), ensure_ascii=False)
            )
            content_index = tool_blocks_by_key[key]
            partial_args = partial_args_by_index.setdefault(content_index, StreamingArgs())
            partial_args.append(args_delta)
            block.arguments = partial_args.arguments
            stream.push(ToolCallDeltaEvent(contentIndex=content_index, delta=args_delta, partial=output))

    finish_current_block(current_block)
    for index in tool_blocks_by_key.values():
        block = output.content[index]
        if block.type != "toolCall":
            continue
        # Finalize in-place so replay only carries parsed arguments. Only a buffer that
        # received something may overwrite what the tool call was born with.
        partial_args = partial_args_by_index.get(index)
        if partial_args is not None and partial_args.raw:
            partial_args.finish_into(block)
        stream.push(ToolCallEndEvent(contentIndex=index, toolCall=block, partial=output))


def to_function_tools(tools: list[Tool]) -> list[dict[str, Any]]:
    def entry(tool: Tool) -> dict[str, Any]:
        strict = resolve_json_schema_strict_sampling(tool, True)
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": strip_symbol_keys(get_json_schema_tool_parameters(tool, strict)),
                "strict": strict if strict is not None else False,
            },
        }

    return [entry(tool) for tool in tools]


def strip_symbol_keys(value: Any) -> Any:
    if isinstance(value, list):
        return [strip_symbol_keys(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): strip_symbol_keys(entry) for key, entry in value.items()}
    return value


def to_chat_messages(messages: list[MessageValue], supports_images: bool) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []

    for index, message in enumerate(messages):
        if message.role == "system":
            text = get_system_message_text(message) if index == 0 else render_system_message_update(message)
            if len(text) > 0:
                result.append({"role": "system", "content": sanitize_surrogates(text)})
            continue

        if message.role == "user":
            if isinstance(message.content, str):
                result.append({"role": "user", "content": sanitize_surrogates(message.content)})
                continue
            had_images = any(item.type == "image" for item in message.content)
            content: list[dict[str, Any]] = [
                {"type": "text", "text": sanitize_surrogates(item.text)}
                if item.type == "text"
                else {"type": "image_url", "imageUrl": f"data:{item.mimeType};base64,{item.data}"}
                for item in message.content
                if item.type == "text" or supports_images
            ]
            if len(content) > 0:
                result.append({"role": "user", "content": content})
                continue
            if had_images and not supports_images:
                result.append({"role": "user", "content": "(image omitted: model does not support images)"})
            continue

        if message.role == "assistant":
            content_parts: list[dict[str, Any]] = []
            tool_calls: list[dict[str, Any]] = []

            for block in message.content:
                if block.type == "text":
                    if len(block.text.strip()) > 0:
                        content_parts.append({"type": "text", "text": sanitize_surrogates(block.text)})
                    continue
                if block.type == "thinking":
                    if len(block.thinking.strip()) > 0:
                        content_parts.append(
                            {
                                "type": "thinking",
                                "thinking": [{"type": "text", "text": sanitize_surrogates(block.thinking)}],
                            }
                        )
                    continue
                tool_calls.append(
                    {
                        "id": block.id,
                        "type": "function",
                        "function": {
                            "name": block.name,
                            "arguments": json.dumps(block.arguments or {}, separators=(",", ":"), ensure_ascii=False),
                        },
                        "index": 0,
                    }
                )

            assistant_message: dict[str, Any] = {"role": "assistant", "prefix": False}
            if len(content_parts) > 0:
                assistant_message["content"] = content_parts
            if len(tool_calls) > 0:
                assistant_message["toolCalls"] = tool_calls
            if len(content_parts) > 0 or len(tool_calls) > 0:
                result.append(assistant_message)
            continue

        tool_content: list[dict[str, Any]] = []
        text_result = "\n".join(sanitize_surrogates(part.text) for part in message.content if part.type == "text")
        has_images = any(part.type == "image" for part in message.content)
        tool_text = build_tool_result_text(text_result, has_images, supports_images, message.isError)
        tool_content.append({"type": "text", "text": tool_text})
        for part in message.content:
            if not supports_images:
                continue
            if part.type != "image":
                continue
            tool_content.append({"type": "image_url", "imageUrl": f"data:{part.mimeType};base64,{part.data}"})
        result.append(
            {
                "role": "tool",
                "toolCallId": message.toolCallId,
                "name": message.toolName,
                "content": tool_content,
            }
        )

    return result


def build_tool_result_text(text: str, has_images: bool, supports_images: bool, is_error: bool) -> str:
    trimmed = text.strip()
    error_prefix = "[tool error] " if is_error else ""

    if len(trimmed) > 0:
        image_suffix = "\n[tool image omitted: model does not support images]" if has_images and not supports_images else ""
        return f"{error_prefix}{trimmed}{image_suffix}"

    if has_images:
        if supports_images:
            return "[tool error] (see attached image)" if is_error else "(see attached image)"
        return (
            "[tool error] (image omitted: model does not support images)"
            if is_error
            else "(image omitted: model does not support images)"
        )

    return "[tool error] (no tool output)" if is_error else "(no tool output)"


def map_tool_choice(choice: Any) -> Any:
    if not choice:
        return None
    if choice in ("auto", "none", "any", "required"):
        return choice
    return {
        "type": "function",
        "function": {"name": choice["function"]["name"]},
    }


def map_chat_stop_reason(reason: str | None) -> dict[str, str]:
    if reason is None:
        return {"stopReason": "stop"}
    if reason == "stop":
        return {"stopReason": "stop"}
    if reason in ("length", "model_length"):
        return {"stopReason": "length"}
    if reason == "tool_calls":
        return {"stopReason": "toolUse"}
    if reason == "error":
        return {"stopReason": "error", "errorMessage": "Provider stopped with: error"}
    return {"stopReason": "error", "errorMessage": f"Provider stopped with: {reason}"}


streamMistral = stream_mistral
streamSimpleMistral = stream_simple_mistral
__all__ = [
    "MistralOptions",
    "streamMistral",
    "streamSimpleMistral",
]
