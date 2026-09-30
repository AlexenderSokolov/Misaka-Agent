"""Shared OpenAI Responses message conversion and stream processing."""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, TypedDict

from misaka.ai.models import calculate_cost
from misaka.ai.providers.constrained_sampling import (
    GrammarToolInputJsonBuffer,
    append_grammar_tool_input_json_delta,
    get_grammar_tool_input,
    get_json_schema_tool_parameters,
    resolve_grammar_constrained_sampling,
    resolve_json_schema_strict_sampling,
)
from misaka.ai.providers.transform_messages import transform_messages
from misaka.ai.types import (
    AssistantMessage,
    Model,
    TextContent,
    TextDeltaEvent,
    TextEndEvent,
    TextSignatureV1,
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
    Usage,
    UsageCost,
)
from misaka.ai.utils.event_stream import AssistantMessageEventStream
from misaka.ai.utils.hash import short_hash
from misaka.ai.utils.json_parse import StreamingArgs
from misaka.ai.utils.sanitize_unicode import sanitize_surrogates
from misaka.ai.utils.text import get_system_message_text, render_system_message_update
from misaka.ai.utils.transcript import resolve_transcript, resolve_transcript_tools
from misaka.utils.values import maybe_await

_TOOL_CALL_ID_PART_PATTERN = re.compile(r"[^a-zA-Z0-9_-]")


class OpenAIResponsesStreamOptions(TypedDict, total=False):
    onProviderStreamEvent: Any
    serviceTier: Any
    resolveServiceTier: Any
    applyServiceTierPricing: Any
    # Tool name -> the property a grammar tool's free-text output is encoded into. Used to
    # rebuild `custom_tool_call` items into ordinary tool calls with one string argument.
    grammarToolInputProperties: Mapping[str, str]


class ConvertResponsesMessagesOptions(TypedDict, total=False):
    includeSystemPrompt: bool
    # Tool name -> the property its grammar output is encoded into. A tool listed here is
    # a grammar tool: it is replayed as a `custom_tool_call` carrying free text rather
    # than a `function_call` carrying JSON arguments.
    grammarToolInputProperties: Mapping[str, str]
    supportsMidConvoSystemMessages: bool
    # How this endpoint takes tools a later system message adds: a developer
    # `additional_tools` item, or a pair of `tool_search_call` / `tool_search_output` items.
    supportsAdditionalTools: bool
    supportsToolSearch: bool
    toolOptions: ConvertResponsesToolsOptions


class ConvertResponsesToolsOptions(TypedDict, total=False):
    strict: bool | None
    supportsStrictMode: bool
    supportsOpenAIGrammarTools: bool
    toolSearchResult: bool


def encode_text_signature_v1(text_id: str, phase: TextSignatureV1 | str | None = None) -> str:
    payload: dict[str, Any] = {"v": 1, "id": text_id}
    if phase in {"commentary", "final_answer"}:
        payload["phase"] = phase
    return json.dumps(payload, separators=(",", ":"))


def parse_text_signature(signature: str | None) -> dict[str, str] | None:
    if not signature:
        return None
    if signature.startswith("{"):
        try:
            parsed = json.loads(signature)
        except ValueError:
            parsed = None
        if isinstance(parsed, dict) and parsed.get("v") == 1 and isinstance(parsed.get("id"), str):
            phase = parsed.get("phase")
            if phase in {"commentary", "final_answer"}:
                return {"id": parsed["id"], "phase": phase}
            return {"id": parsed["id"]}
    return {"id": signature}


def convert_tool_result_output(model: Model, content: list[Any]) -> str | list[dict[str, Any]]:
    text_result = "\n".join(block.text for block in content if block.type == "text")
    images = [block for block in content if block.type == "image"]
    has_text = len(text_result) > 0

    if len(images) == 0 or "image" not in model.input:
        return sanitize_surrogates(
            text_result if has_text else "(see attached image)" if len(images) > 0 else "(no tool output)"
        )

    output: list[dict[str, Any]] = []
    if has_text:
        output.append({"type": "input_text", "text": sanitize_surrogates(text_result)})
    for image in images:
        output.append({
            "type": "input_image",
            "detail": "auto",
            "image_url": f"data:{image.mimeType};base64,{image.data}",
        })
    return output


def _normalize_id_part(part: str) -> str:
    sanitized = _TOOL_CALL_ID_PART_PATTERN.sub("_", part)
    normalized = sanitized[:64] if len(sanitized) > 64 else sanitized
    return normalized.rstrip("_")


def _build_foreign_responses_item_id(item_id: str) -> str:
    normalized = f"fc_{short_hash(item_id)}"
    return normalized[:64] if len(normalized) > 64 else normalized


def _create_normalize_tool_call_id(model: Model, allowed_tool_call_providers: set[str] | frozenset[str]):
    def normalize_tool_call_id(tool_call_id: str, _target_model: Model, source: AssistantMessage) -> str:
        if model.provider not in allowed_tool_call_providers:
            return _normalize_id_part(tool_call_id)
        if "|" not in tool_call_id:
            return _normalize_id_part(tool_call_id)

        parts = tool_call_id.split("|")
        call_id = parts[0]
        item_id = parts[1] if len(parts) > 1 else ""
        normalized_call_id = _normalize_id_part(call_id)
        is_foreign_tool_call = source.provider != model.provider or source.api != model.api
        normalized_item_id = (
            _build_foreign_responses_item_id(item_id) if is_foreign_tool_call else _normalize_id_part(item_id)
        )
        if not normalized_item_id.startswith("fc_"):
            normalized_item_id = _normalize_id_part(f"fc_{normalized_item_id}")
        return f"{normalized_call_id}|{normalized_item_id}"

    return normalize_tool_call_id


def convert_responses_messages(
    model: Model,
    context: TranscriptContext,
    allowed_tool_call_providers: set[str] | frozenset[str],
    options: ConvertResponsesMessagesOptions | None = None,
) -> list[dict[str, Any]]:
    opts = options or {}
    normalized_context = resolve_transcript(context, opts.get("supportsMidConvoSystemMessages"))
    messages: list[dict[str, Any]] = []
    tool_options: ConvertResponsesToolsOptions = opts.get("toolOptions") or {}
    transformed_messages = transform_messages(
        normalized_context.messages,
        model,
        _create_normalize_tool_call_id(model, allowed_tool_call_providers),
    )
    transcript_tools = resolve_transcript_tools(
        normalized_context.messages,
        bool(opts.get("supportsAdditionalTools", False)) or bool(opts.get("supportsToolSearch", False)),
    )

    def append_system_tool_additions(message: Any, seed: str) -> None:
        tools = (message.toolsAdded or []) if transcript_tools.anchorsAdditions else []
        if len(tools) == 0:
            return
        if opts.get("supportsAdditionalTools"):
            messages.append({
                "type": "additional_tools",
                "role": "developer",
                "tools": convert_responses_tools(tools, tool_options),
            })
            return
        if not opts.get("supportsToolSearch"):
            return
        names = [tool.name for tool in tools]
        call_id = f"pi_tool_load_{short_hash(seed + ':' + ','.join(names))}"
        messages.append({
            "type": "tool_search_call",
            "call_id": call_id,
            "execution": "client",
            "status": "completed",
            "arguments": {"query": " ".join(names), "limit": len(names)},
        })
        messages.append({
            "type": "tool_search_output",
            "call_id": call_id,
            "execution": "client",
            "status": "completed",
            "tools": convert_responses_tools(tools, {**tool_options, "toolSearchResult": True}),
        })

    include_initial_system_message = True if options is None else options.get("includeSystemPrompt", True)
    grammar_tool_input_properties: Mapping[str, str] = (
        {} if options is None else options.get("grammarToolInputProperties") or {}
    )
    compat = model.compat
    instruction_role = (
        "developer" if model.reasoning and getattr(compat, "supportsDeveloperRole", None) is not False else "system"
    )

    msg_index = 0
    for source_index, message in enumerate(transformed_messages):
        is_leading_system_message = source_index == 0 and message.role == "system"
        if message.role == "system":
            if not is_leading_system_message:
                append_system_tool_additions(message, f"system:{msg_index}")
            if not is_leading_system_message or include_initial_system_message:
                text = get_system_message_text(message) if is_leading_system_message else render_system_message_update(message)
                if len(text) > 0:
                    messages.append({"role": instruction_role, "content": sanitize_surrogates(text)})
        elif message.role == "user":
            if isinstance(message.content, str):
                messages.append(
                    {
                        "role": "user",
                        "content": [{"type": "input_text", "text": sanitize_surrogates(message.content)}],
                    }
                )
            else:
                content: list[dict[str, Any]] = []
                for item in message.content:
                    if item.type == "text":
                        content.append({"type": "input_text", "text": sanitize_surrogates(item.text)})
                    else:
                        content.append(
                            {
                                "type": "input_image",
                                "detail": "auto",
                                "image_url": f"data:{item.mimeType};base64,{item.data}",
                            }
                        )
                if content:
                    messages.append({"role": "user", "content": content})
        elif message.role == "assistant":
            output: list[dict[str, Any]] = []
            assistant_message = message
            is_same_provider_and_api = (
                assistant_message.provider == model.provider and assistant_message.api == model.api
            )
            is_same_model = is_same_provider_and_api and assistant_message.model == model.id
            is_different_model = is_same_provider_and_api and assistant_message.model != model.id
            text_block_index = 0

            for block in assistant_message.content:
                if block.type == "thinking":
                    if block.thinkingSignature:
                        output.append(json.loads(block.thinkingSignature))
                elif block.type == "text":
                    parsed_signature = parse_text_signature(block.textSignature)
                    fallback_message_id = (
                        f"msg_pi_{msg_index}" if text_block_index == 0 else f"msg_pi_{msg_index}_{text_block_index}"
                    )
                    text_block_index += 1
                    # OpenAI requires id to be max 64 characters
                    msg_id = parsed_signature["id"] if parsed_signature else None
                    if not msg_id:
                        msg_id = fallback_message_id
                    elif len(msg_id) > 64:
                        msg_id = f"msg_{short_hash(msg_id)}"
                    message_item: dict[str, Any] = {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": sanitize_surrogates(block.text), "annotations": []}],
                        "status": "completed",
                        "id": msg_id,
                    }
                    phase = parsed_signature.get("phase") if parsed_signature else None
                    if phase:
                        message_item["phase"] = phase
                    output.append(message_item)
                elif block.type == "toolCall":
                    parts = block.id.split("|")
                    call_id = parts[0]
                    item_id_raw = parts[1] if len(parts) > 1 else ""
                    item_id: str | None = item_id_raw or None
                    custom_input_property = grammar_tool_input_properties.get(block.name)
                    # A `function_call` item id must be an `fc_` one. Replaying a
                    # custom-tool call as a function call carries a `ctc_` id, which the
                    # endpoint rejects, so it is dropped along with the cross-model case.
                    if (is_different_model and item_id and item_id.startswith("fc_")) or (
                        custom_input_property is None and not (item_id or "").startswith("fc_")
                    ):
                        item_id = None
                    namespace = (
                        {"namespace": block.namespace} if is_same_model and block.namespace is not None else {}
                    )
                    # MISAKA fork: pi's `id: undefined` leaves the key out of the JSON; a Python
                    # `None` would be sent as `null`.
                    item_id_field = {} if item_id is None else {"id": item_id}
                    if custom_input_property is not None:
                        output.append(
                            {
                                "type": "custom_tool_call",
                                **item_id_field,
                                "call_id": call_id,
                                "name": block.name,
                                "input": sanitize_surrogates(
                                    get_grammar_tool_input(
                                        block.name, block.arguments, custom_input_property
                                    )
                                ),
                                **namespace,
                            }
                        )
                    else:
                        output.append(
                            {
                                "type": "function_call",
                                **item_id_field,
                                "call_id": call_id,
                                "name": block.name,
                                "arguments": json.dumps(block.arguments, separators=(",", ":"), ensure_ascii=False),
                                **namespace,
                            }
                        )
            if output:
                messages.extend(output)
        elif message.role == "toolResult":
            call_id = message.toolCallId.split("|", 1)[0]
            output_value = convert_tool_result_output(model, message.content)

            # A grammar tool's result goes back on the custom-tool channel it came from.
            messages.append(
                {
                    "type": (
                        "custom_tool_call_output"
                        if message.toolName in grammar_tool_input_properties
                        else "function_call_output"
                    ),
                    "call_id": call_id,
                    "output": output_value,
                }
            )
        if not is_leading_system_message:
            msg_index += 1

    return messages


def convert_responses_tools(
    tools: Iterable[Tool],
    options: ConvertResponsesToolsOptions | None = None,
) -> list[dict[str, Any]]:
    options = options or {}
    default_strict = options.get("strict", False)
    supports_strict_mode = options.get("supportsStrictMode", True)
    supports_grammar_tools = options.get("supportsOpenAIGrammarTools", False)
    tool_search_result = bool(options.get("toolSearchResult"))

    converted: list[dict[str, Any]] = []
    for tool in tools:
        grammar = resolve_grammar_constrained_sampling(tool, bool(supports_grammar_tools))
        if grammar:
            # A grammar tool is not a function tool: the model emits free text that the
            # grammar constrains, and it comes back as a `custom_tool_call`.
            converted.append(
                {
                    "type": "custom",
                    "name": tool.name,
                    "description": tool.description,
                    "format": {
                        "type": "grammar",
                        "syntax": grammar.format,
                        "definition": grammar.definition,
                    },
                    **({"defer_loading": True} if tool_search_result else {}),
                }
            )
            continue

        constrained = resolve_json_schema_strict_sampling(tool, bool(supports_strict_mode))
        strict = default_strict if constrained is None else constrained
        function_tool: dict[str, Any] = {
            "type": "function",
            "name": tool.name,
            "description": tool.description,
            "parameters": get_json_schema_tool_parameters(tool, strict is True),
            **({"defer_loading": True} if tool_search_result else {}),
        }
        if supports_strict_mode:
            function_tool["strict"] = strict
        converted.append(function_tool)
    return converted


# =============================================================================
# Stream processing
# =============================================================================


@dataclass
class _CustomToolCallInput:
    property: str
    jsonBuffer: GrammarToolInputJsonBuffer


@dataclass
class _StreamingToolCallScratch:
    """MISAKA fork: upstream's ``StreamingToolCall`` widens the block itself with the
    scratch fields ``partialJson`` and ``customInput`` and deletes them once the call is
    finished. Pydantic blocks forbid extra fields, so the scratch rides beside the block,
    keyed by content index; ``None`` stands for a deleted field. ``partialJson`` is a
    ``StreamingArgs`` so the live view does not re-parse the whole buffer on every delta.
    """

    partialJson: StreamingArgs | None = None
    customInput: _CustomToolCallInput | None = None


def _get_custom_tool_call_input(block: ToolCall, scratch: _StreamingToolCallScratch) -> str:
    prop = scratch.customInput.property if scratch.customInput is not None else None
    if prop is None:
        return ""
    value = block.arguments.get(prop)
    return value if isinstance(value, str) else ""


def _append_custom_tool_call_input(
    block: ToolCall, scratch: _StreamingToolCallScratch, next_input: str, close: bool
) -> str | None:
    custom_input = scratch.customInput
    if custom_input is None:
        return None
    delta = append_grammar_tool_input_json_delta(custom_input.jsonBuffer, custom_input.property, next_input, close)
    block.arguments = {custom_input.property: next_input}
    return delta


@dataclass
class _ResponsesOutputSlot:
    type: Literal["thinking", "text", "toolCall"]
    block: Any
    contentIndex: int


async def process_responses_stream(
    openai_stream: AsyncIterable[dict[str, Any]],
    output: AssistantMessage,
    stream: AssistantMessageEventStream,
    model: Model,
    options: OpenAIResponsesStreamOptions | None = None,
) -> None:
    saw_terminal_response_event = False
    output_slots: dict[Any, _ResponsesOutputSlot] = {}
    reasoning_blocks_by_id: dict[str, ThinkingContent] = {}
    streaming_tool_calls: dict[int, _StreamingToolCallScratch] = {}
    grammar_tool_input_properties: Mapping[str, str] = (
        {} if options is None else options.get("grammarToolInputProperties") or {}
    )
    on_provider_stream_event = None if options is None else options.get("onProviderStreamEvent")

    def apply_message_phase_stop_reason(item: dict[str, Any]) -> None:
        if item.get("type") == "message" and item.get("phase") == "final_answer":
            output.stopReason = "stop"

    def get_slot(output_index: Any, slot_type: str) -> _ResponsesOutputSlot | None:
        slot = output_slots.get(output_index)
        return slot if slot is not None and slot.type == slot_type else None

    def push_tool_call_delta(slot: _ResponsesOutputSlot, delta: str | None) -> None:
        if delta is None:
            return
        stream.push(ToolCallDeltaEvent(contentIndex=slot.contentIndex, delta=delta, partial=output))

    def create_slot(output_index: Any, item: dict[str, Any]) -> _ResponsesOutputSlot | None:
        item_type = item.get("type")
        if item_type == "reasoning":
            block = ThinkingContent(thinking="")
            output.content.append(block)
            slot = _ResponsesOutputSlot(type="thinking", block=block, contentIndex=len(output.content) - 1)
            output_slots[output_index] = slot
            stream.push(ThinkingStartEvent(contentIndex=slot.contentIndex, partial=output))
            return slot
        if item_type == "message":
            apply_message_phase_stop_reason(item)
            block = TextContent(text="")
            output.content.append(block)
            slot = _ResponsesOutputSlot(type="text", block=block, contentIndex=len(output.content) - 1)
            output_slots[output_index] = slot
            stream.push(TextStartEvent(contentIndex=slot.contentIndex, partial=output))
            return slot
        if item_type == "function_call":
            block = ToolCall(
                id=f"{item.get('call_id', '')}|{item.get('id', '')}",
                name=item.get("name", ""),
                arguments={},
                namespace=item.get("namespace"),
            )
            output.content.append(block)
            slot = _ResponsesOutputSlot(type="toolCall", block=block, contentIndex=len(output.content) - 1)
            streaming_tool_calls[slot.contentIndex] = _StreamingToolCallScratch(
                partialJson=StreamingArgs(item.get("arguments") or "")
            )
            output_slots[output_index] = slot
            stream.push(ToolCallStartEvent(contentIndex=slot.contentIndex, partial=output))
            return slot
        if item_type == "custom_tool_call":
            input_property = grammar_tool_input_properties.get(item.get("name", "")) or "input"
            input_value = item.get("input") or ""
            block = ToolCall(
                id=f"{item.get('call_id', '')}|{item.get('id', '')}",
                name=item.get("name", ""),
                arguments={input_property: input_value},
                namespace=item.get("namespace"),
            )
            output.content.append(block)
            slot = _ResponsesOutputSlot(type="toolCall", block=block, contentIndex=len(output.content) - 1)
            streaming_tool_calls[slot.contentIndex] = _StreamingToolCallScratch(
                customInput=_CustomToolCallInput(property=input_property, jsonBuffer=GrammarToolInputJsonBuffer())
            )
            output_slots[output_index] = slot
            stream.push(ToolCallStartEvent(contentIndex=slot.contentIndex, partial=output))
            return slot
        return None

    def get_or_create_slot(output_index: Any, item: dict[str, Any]) -> _ResponsesOutputSlot | None:
        slot = output_slots.get(output_index)
        return slot if slot is not None else create_slot(output_index, item)

    # Azure OpenAI can omit reasoning.encrypted_content from response.output_item.done
    # and provide it only in response.completed.response.output. Backfill the
    # persisted reasoning signature from the terminal response to keep store:false
    # multi-turn replay stateless. See https://github.com/earendil-works/pi/issues/6409.
    def backfill_reasoning_signatures(response_output: list[Any]) -> None:
        for item in response_output:
            if not isinstance(item, dict) or item.get("type") != "reasoning" or not item.get("encrypted_content"):
                continue
            block = reasoning_blocks_by_id.get(item.get("id"))
            if block is None or not block.thinkingSignature:
                continue

            stored_item = json.loads(block.thinkingSignature)
            if stored_item.get("encrypted_content"):
                continue
            block.thinkingSignature = json.dumps(
                {**stored_item, "encrypted_content": item["encrypted_content"]}, separators=(",", ":")
            )

    def finalize_response(response: Any) -> None:
        nonlocal saw_terminal_response_event
        saw_terminal_response_event = True
        response = response if isinstance(response, dict) else {}
        backfill_reasoning_signatures(response.get("output") or [])
        if response.get("id"):
            output.responseId = response["id"]
        usage = response.get("usage")
        if usage:
            input_details = usage.get("input_tokens_details") or {}
            cached_tokens = input_details.get("cached_tokens") or 0
            cache_write_tokens = input_details.get("cache_write_tokens") or 0
            output.usage = Usage(
                # OpenAI includes cached and cache-write tokens in input_tokens, so subtract both.
                input=max(0, (usage.get("input_tokens") or 0) - cached_tokens - cache_write_tokens),
                output=usage.get("output_tokens") or 0,
                cacheRead=cached_tokens,
                cacheWrite=cache_write_tokens,
                reasoning=(usage.get("output_tokens_details") or {}).get("reasoning_tokens") or 0,
                totalTokens=usage.get("total_tokens") or 0,
                cost=UsageCost(input=0, output=0, cacheRead=0, cacheWrite=0, total=0),
            )
        calculate_cost(model, output.usage)
        if options and options.get("applyServiceTierPricing"):
            resolve_service_tier = options.get("resolveServiceTier")
            service_tier = (
                resolve_service_tier(response.get("service_tier"), options.get("serviceTier"))
                if resolve_service_tier
                else response.get("service_tier") or options.get("serviceTier")
            )
            options["applyServiceTierPricing"](output.usage, service_tier)
        # Map status to stop reason. For incomplete responses, retain the provider's
        # specific reason so max-output truncation and content filtering stay distinct.
        status = response.get("status")
        incomplete_details = response.get("incomplete_details")
        reason = incomplete_details.get("reason") if isinstance(incomplete_details, dict) else None
        incomplete_reason = reason if isinstance(reason, str) else None
        output.rawStopReason = f"{status}.{incomplete_reason}" if incomplete_reason else status
        mapped_stop = _map_stop_reason(status, incomplete_reason)
        output.stopReason = mapped_stop["stopReason"]
        output.errorMessage = mapped_stop.get("errorMessage")
        if any(block.type == "toolCall" for block in output.content) and output.stopReason == "stop":
            output.stopReason = "toolUse"

    async for event in openai_stream:
        if callable(on_provider_stream_event):
            await maybe_await(on_provider_stream_event(event, model))
        event_type = event.get("type")
        output_index = event.get("output_index")
        if event_type == "response.created":
            response = event.get("response")
            if isinstance(response, dict) and isinstance(response.get("id"), str):
                output.responseId = response["id"]
        elif event_type == "response.output_item.added":
            item = event.get("item")
            if isinstance(item, dict):
                create_slot(output_index, item)
        elif event_type == "response.reasoning_summary_text.delta":
            slot = get_slot(output_index, "thinking")
            if slot is None:
                continue
            slot.block.thinking += event.get("delta", "")
            stream.push(ThinkingDeltaEvent(contentIndex=slot.contentIndex, delta=event.get("delta", ""), partial=output))
        elif event_type == "response.reasoning_summary_part.done":
            slot = get_slot(output_index, "thinking")
            if slot is None:
                continue
            slot.block.thinking += "\n\n"
            stream.push(ThinkingDeltaEvent(contentIndex=slot.contentIndex, delta="\n\n", partial=output))
        elif event_type == "response.reasoning_text.delta":
            slot = get_slot(output_index, "thinking")
            if slot is None:
                continue
            slot.block.thinking += event.get("delta", "")
            stream.push(ThinkingDeltaEvent(contentIndex=slot.contentIndex, delta=event.get("delta", ""), partial=output))
        elif event_type == "response.output_text.delta" or event_type == "response.refusal.delta":
            slot = get_slot(output_index, "text")
            if slot is None:
                continue
            slot.block.text += event.get("delta", "")
            stream.push(TextDeltaEvent(contentIndex=slot.contentIndex, delta=event.get("delta", ""), partial=output))
        elif event_type == "response.function_call_arguments.delta":
            slot = get_slot(output_index, "toolCall")
            scratch = streaming_tool_calls.get(slot.contentIndex) if slot is not None else None
            if slot is None or scratch is None or scratch.partialJson is None:
                continue
            scratch.partialJson.append(event.get("delta", ""))
            slot.block.arguments = scratch.partialJson.arguments
            push_tool_call_delta(slot, event.get("delta", ""))
        elif event_type == "response.function_call_arguments.done":
            slot = get_slot(output_index, "toolCall")
            scratch = streaming_tool_calls.get(slot.contentIndex) if slot is not None else None
            if slot is None or scratch is None or scratch.partialJson is None:
                continue
            previous_partial_json = scratch.partialJson.raw
            arguments = event.get("arguments", "")
            scratch.partialJson = StreamingArgs(arguments)
            slot.block.arguments = scratch.partialJson.arguments

            if arguments.startswith(previous_partial_json):
                delta = arguments[len(previous_partial_json) :]
                if len(delta) > 0:
                    push_tool_call_delta(slot, delta)
        elif event_type == "response.custom_tool_call_input.delta":
            slot = get_slot(output_index, "toolCall")
            scratch = streaming_tool_calls.get(slot.contentIndex) if slot is not None else None
            if slot is None or scratch is None or scratch.customInput is None:
                continue
            push_tool_call_delta(
                slot,
                _append_custom_tool_call_input(
                    slot.block, scratch, _get_custom_tool_call_input(slot.block, scratch) + event.get("delta", ""), False
                ),
            )
        elif event_type == "response.custom_tool_call_input.done":
            slot = get_slot(output_index, "toolCall")
            scratch = streaming_tool_calls.get(slot.contentIndex) if slot is not None else None
            if slot is None or scratch is None or scratch.customInput is None:
                continue
            push_tool_call_delta(slot, _append_custom_tool_call_input(slot.block, scratch, event.get("input", ""), True))
        elif event_type == "response.output_item.done":
            item = event.get("item")
            if not isinstance(item, dict):
                continue
            apply_message_phase_stop_reason(item)
            slot = get_or_create_slot(output_index, item)
            scratch = streaming_tool_calls.get(slot.contentIndex) if slot is not None else None
            item_type = item.get("type")

            if item_type == "reasoning" and slot is not None and slot.type == "thinking":
                summary_text = "\n\n".join(s.get("text", "") for s in item.get("summary") or [] if isinstance(s, dict))
                content_text = "\n\n".join(c.get("text", "") for c in item.get("content") or [] if isinstance(c, dict))
                slot.block.thinking = summary_text or content_text or slot.block.thinking
                slot.block.thinkingSignature = json.dumps(item, separators=(",", ":"))
                reasoning_blocks_by_id[item.get("id")] = slot.block
                stream.push(ThinkingEndEvent(contentIndex=slot.contentIndex, content=slot.block.thinking, partial=output))
                output_slots.pop(output_index, None)
            elif item_type == "message" and slot is not None and slot.type == "text":
                slot.block.text = "".join(
                    c.get("text", "") if c.get("type") == "output_text" else c.get("refusal", "")
                    for c in item.get("content") or []
                    if isinstance(c, dict)
                )
                slot.block.textSignature = encode_text_signature_v1(item.get("id", ""), item.get("phase"))
                stream.push(TextEndEvent(contentIndex=slot.contentIndex, content=slot.block.text, partial=output))
                output_slots.pop(output_index, None)
            elif (
                item_type == "function_call"
                and slot is not None
                and slot.type == "toolCall"
                and scratch is not None
                and scratch.partialJson is not None
            ):
                # MISAKA fork: the final arguments go through `finish_into`, which records a
                # malformed object as `argumentsError` instead of running a repaired prefix.
                item_arguments = item.get("arguments")
                (StreamingArgs(item_arguments) if item_arguments else scratch.partialJson).finish_into(slot.block)
                if item.get("namespace") is not None:
                    slot.block.namespace = item["namespace"]
                # Finalize in-place and strip the scratch buffer so replay only
                # carries parsed arguments.
                scratch.partialJson = None
                stream.push(ToolCallEndEvent(contentIndex=slot.contentIndex, toolCall=slot.block, partial=output))
                output_slots.pop(output_index, None)
            elif (
                item_type == "custom_tool_call"
                and slot is not None
                and slot.type == "toolCall"
                and scratch is not None
                and scratch.customInput is not None
            ):
                final_input = item.get("input")
                push_tool_call_delta(
                    slot,
                    _append_custom_tool_call_input(
                        slot.block,
                        scratch,
                        final_input if final_input is not None else _get_custom_tool_call_input(slot.block, scratch),
                        True,
                    ),
                )
                if item.get("namespace") is not None:
                    slot.block.namespace = item["namespace"]
                scratch.customInput = None
                stream.push(ToolCallEndEvent(contentIndex=slot.contentIndex, toolCall=slot.block, partial=output))
                output_slots.pop(output_index, None)
        elif event_type in {"response.completed", "response.incomplete"}:
            finalize_response(event.get("response"))
        elif event_type == "error":
            raise RuntimeError(f"Error Code {event.get('code')}: {event.get('message')}")
        elif event_type == "response.failed":
            saw_terminal_response_event = True
            response = event.get("response") if isinstance(event.get("response"), dict) else {}
            output.rawStopReason = response.get("status")
            error = response.get("error")
            details = response.get("incomplete_details")
            if isinstance(error, dict):
                msg = f"{error.get('code') or 'unknown'}: {error.get('message') or 'no message'}"
            elif isinstance(details, dict) and details.get("reason"):
                msg = f"incomplete: {details['reason']}"
            else:
                msg = "Unknown error (no error details in response)"
            raise RuntimeError(msg)

    if not saw_terminal_response_event:
        raise RuntimeError("OpenAI Responses stream ended before a terminal response event")
    # The agent runs every tool call in the final message. Refuse to hand over calls whose
    # output_item.done never arrived: their arguments may be cut off or mixed up, e.g. when a
    # non-compliant server omits output_index. Finished calls have their scratch buffers removed.
    if output.stopReason == "toolUse":
        for index, block in enumerate(output.content):
            if block.type != "toolCall":
                continue
            scratch = streaming_tool_calls.get(index)
            if scratch is not None and (scratch.partialJson is not None or scratch.customInput is not None):
                raise RuntimeError(
                    f"OpenAI Responses stream completed with an unfinished tool call: {block.name} ({block.id})"
                )


def _map_stop_reason(status: str | None, incomplete_reason: str | None = None) -> dict[str, Any]:
    if not status:
        return {"stopReason": "stop"}
    if status == "completed":
        return {"stopReason": "stop"}
    if status == "incomplete":
        if incomplete_reason == "max_output_tokens":
            return {"stopReason": "length"}
        return {
            "stopReason": "error",
            "errorMessage": f"Response incomplete: {incomplete_reason}"
            if incomplete_reason
            else "Response incomplete without a provider reason",
        }
    if status in {"failed", "cancelled"}:
        return {"stopReason": "error"}
    # These two are wonky ...
    if status in {"in_progress", "queued"}:
        return {"stopReason": "stop"}
    raise RuntimeError(f"Unhandled stop reason: {status}")


__all__ = [
    "ConvertResponsesMessagesOptions",
    "ConvertResponsesToolsOptions",
    "OpenAIResponsesStreamOptions",
    "convert_responses_messages",
    "convert_responses_tools",
    "encode_text_signature_v1",
    "parse_text_signature",
    "process_responses_stream",
]
