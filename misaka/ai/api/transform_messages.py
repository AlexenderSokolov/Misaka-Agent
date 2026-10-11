"""Translated from pi's ``packages/ai/src/api/transform-messages.ts`` (v1.1.0).

MISAKA fork: messages are pydantic models, so an object spread is ``model_copy(update=...)`` and deleting
``thoughtSignature`` sets it to ``None`` (an absent field).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from misaka.ai.types import (
    AssistantMessage,
    ImageContent,
    MessageValue,
    Model,
    TextContent,
    ToolCall,
    ToolResultMessage,
)
from misaka.utils.js import jsTrim

_NON_VISION_USER_IMAGE_PLACEHOLDER = "(image omitted: model does not support images)"
_NON_VISION_TOOL_IMAGE_PLACEHOLDER = "(tool image omitted: model does not support images)"


def _replaceImagesWithPlaceholder(content: list[TextContent | ImageContent], placeholder: str) -> list[TextContent]:
    result: list[TextContent] = []
    previousWasPlaceholder = False

    for block in content:
        if block.type == "image":
            if not previousWasPlaceholder:
                result.append(TextContent(type="text", text=placeholder))
            previousWasPlaceholder = True
            continue

        result.append(block)
        previousWasPlaceholder = block.text == placeholder

    return result


def _downgradeUnsupportedImages(messages: list[MessageValue], model: Model) -> list[MessageValue]:
    if "image" in model.input:
        return messages

    def downgrade(msg: Any) -> Any:
        if msg.role == "user" and isinstance(msg.content, list):
            return msg.model_copy(
                update={"content": _replaceImagesWithPlaceholder(msg.content, _NON_VISION_USER_IMAGE_PLACEHOLDER)}
            )

        if msg.role == "toolResult":
            return msg.model_copy(
                update={"content": _replaceImagesWithPlaceholder(msg.content, _NON_VISION_TOOL_IMAGE_PLACEHOLDER)}
            )

        return msg

    return [downgrade(msg) for msg in messages]


def transformMessages(
    messages: list[MessageValue],
    model: Model,
    normalizeToolCallId: Callable[[str, Model, AssistantMessage], str] | None = None,
) -> list[MessageValue]:
    """Normalize tool call ID for cross-provider compatibility.
    OpenAI Responses API generates IDs that are 450+ chars with special characters like `|`.
    Anthropic APIs require IDs matching ^[a-zA-Z0-9_-]+$ (max 64 chars)."""
    # Build a map of original tool call IDs to normalized IDs
    toolCallIdMap: dict[str, str] = {}
    # Normalize null/undefined content from untyped callers (custom tools, hand-built
    # histories, old session files) so downstream code can rely on the type contract.
    normalizedMessages = [
        msg.model_copy(update={"content": []}) if getattr(msg, "content", None) is None else msg for msg in messages
    ]
    imageAwareMessages = _downgradeUnsupportedImages(normalizedMessages, model)

    # First pass: transform messages (unsupported image downgrade, thinking blocks, tool call ID normalization)
    def transform(msg: Any) -> Any:
        # System and user messages pass through unchanged
        if msg.role == "system" or msg.role == "user":
            return msg

        # Handle toolResult messages - normalize toolCallId if we have a mapping
        if msg.role == "toolResult":
            normalizedId = toolCallIdMap.get(msg.toolCallId)
            if normalizedId and normalizedId != msg.toolCallId:
                return msg.model_copy(update={"toolCallId": normalizedId})
            return msg

        # Assistant messages need transformation check
        if msg.role == "assistant":
            assistantMsg: AssistantMessage = msg
            isSameModel = (
                assistantMsg.provider == model.provider
                and assistantMsg.api == model.api
                and assistantMsg.model == model.id
            )

            def transformBlock(block: Any) -> list[Any]:
                if block.type == "thinking":
                    # Redacted thinking is opaque encrypted content, only valid for the same model.
                    # Drop it for cross-model to avoid API errors.
                    if block.redacted:
                        return [block] if isSameModel else []
                    # For same model: keep thinking blocks with signatures (needed for replay)
                    # even if the thinking text is empty (OpenAI encrypted reasoning)
                    if isSameModel and block.thinkingSignature:
                        return [block]
                    # Skip empty thinking blocks, convert others to plain text
                    if not block.thinking or jsTrim(block.thinking) == "":
                        return []
                    if isSameModel:
                        return [block]
                    return [TextContent(type="text", text=block.thinking)]

                if block.type == "text":
                    if isSameModel:
                        return [block]
                    return [TextContent(type="text", text=block.text)]

                if block.type == "toolCall":
                    toolCall: ToolCall = block
                    normalizedToolCall = toolCall

                    if not isSameModel and toolCall.thoughtSignature:
                        normalizedToolCall = toolCall.model_copy(update={"thoughtSignature": None})

                    if not isSameModel and normalizeToolCallId:
                        normalizedId = normalizeToolCallId(toolCall.id, model, assistantMsg)
                        if normalizedId != toolCall.id:
                            toolCallIdMap[toolCall.id] = normalizedId
                            normalizedToolCall = normalizedToolCall.model_copy(update={"id": normalizedId})

                    return [normalizedToolCall]

                return [block]

            transformedContent = [item for block in assistantMsg.content for item in transformBlock(block)]

            return assistantMsg.model_copy(update={"content": transformedContent})
        return msg

    transformed = [transform(msg) for msg in imageAwareMessages]

    # Second pass: insert synthetic empty tool results for orphaned tool calls
    # This preserves thinking signatures and satisfies API requirements
    result: list[MessageValue] = []
    pendingToolCalls: list[ToolCall] = []
    existingToolResultIds: set[str] = set()
    # System messages are transparent to tool-call accounting: one that lands between a tool
    # call and its results is held back and emitted after the results (synthetic ones
    # included), so it never causes a duplicate result for a call that is answered later.
    heldSystemMessages: list[MessageValue] = []

    def closePendingToolCalls() -> None:
        nonlocal pendingToolCalls, existingToolResultIds
        if len(pendingToolCalls) > 0:
            for tc in pendingToolCalls:
                if tc.id not in existingToolResultIds:
                    result.append(
                        ToolResultMessage(
                            role="toolResult",
                            toolCallId=tc.id,
                            toolName=tc.name,
                            content=[TextContent(type="text", text="No result provided")],
                            isError=True,
                            timestamp=time.time_ns() // 1_000_000,
                        )
                    )
            pendingToolCalls = []
            existingToolResultIds = set()
        result.extend(heldSystemMessages)
        heldSystemMessages.clear()

    for msg in transformed:
        if msg.role == "assistant":
            # If we have pending orphaned tool calls from a previous assistant, insert synthetic results now
            closePendingToolCalls()

            # Skip errored/aborted assistant messages entirely.
            # These are incomplete turns that shouldn't be replayed:
            # - May have partial content (reasoning without message, incomplete tool calls)
            # - Replaying them can cause API errors (e.g., OpenAI "reasoning without following item")
            # - The model should retry from the last valid state
            assistantMsg = msg
            if assistantMsg.stopReason == "error" or assistantMsg.stopReason == "aborted":
                continue

            # Track tool calls from this assistant message
            toolCalls = [b for b in assistantMsg.content if b.type == "toolCall"]
            if len(toolCalls) > 0:
                pendingToolCalls = toolCalls
                existingToolResultIds = set()

            result.append(msg)
        elif msg.role == "toolResult":
            existingToolResultIds.add(msg.toolCallId)
            result.append(msg)
        elif msg.role == "system":
            if len(pendingToolCalls) > 0:
                heldSystemMessages.append(msg)
            else:
                result.append(msg)
        elif msg.role == "user":
            # A new user turn interrupts tool flow - insert synthetic results for orphaned calls
            closePendingToolCalls()
            result.append(msg)
        else:
            result.append(msg)

    # If the conversation ends with unresolved tool calls, synthesize results now.
    closePendingToolCalls()

    return result


__all__ = ["transformMessages"]
