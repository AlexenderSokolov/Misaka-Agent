"""Translated from pi's ``packages/ai/src/utils/assistant-message-frame.ts`` (v1.1.0).

MISAKA fork: frames are plain dicts with the upstream keys; ``partial``, ``content`` and ``toolCall`` hold the pi-ai
models, as the upstream objects hold the typed values. ``structuredClone`` is a deep model copy (or ``copy.deepcopy``
for arguments); lengths and slices count UTF-16 code units as JavaScript strings do.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal

from misaka.ai.types import (
    AssistantMessage,
    AssistantMessageEvent,
    TextContent,
    ThinkingContent,
    ToolCall,
)
from misaka.ai.utils.json_parse import parseStreamingJson
from misaka.utils.js import (
    UNDEFINED,
    jsIsSafeInteger,
    jsLength,
    jsonStringify,
    jsSlice,
    jsString,
)

# Compact, replayable assistant-message progress. Terminal settlement is
# intentionally excluded and must be persisted separately.
#
#   {type: "start", partial}
#   {type: "text_start", contentIndex, content}
#   {type: "text_delta", contentIndex, delta}
#   {type: "text_end", contentIndex, content, textSignature?}
#   {type: "thinking_start", contentIndex, content}
#   {type: "thinking_delta", contentIndex, delta}
#   {type: "thinking_end", contentIndex, content, thinkingSignature?, redacted?}
#   {type: "toolcall_start", contentIndex, toolCall}
#   {type: "toolcall_checkpoint", contentIndex, json}
#   {type: "toolcall_delta", contentIndex, delta}
#   {type: "toolcall_end", contentIndex, id, name, arguments, thoughtSignature?, namespace?}
type AssistantMessageFrame = dict[str, Any]


@dataclass(slots=True)
class _EncoderTextBlockState:
    kind: Literal["text", "thinking"]
    coveredChars: int
    deltaChars: int


@dataclass(slots=True)
class _EncoderToolCallBlockState:
    kind: Literal["toolCall"]
    caughtUp: bool
    catchupJson: str
    snapshotArguments: str


type _EncoderBlockState = _EncoderTextBlockState | _EncoderToolCallBlockState


@dataclass(slots=True)
class _ReducerBlockState:
    kind: Literal["text", "thinking", "toolCall"]
    ended: bool
    json: str = ""


def _cloneTextContent(content: TextContent) -> TextContent:
    return TextContent(
        type="text",
        text=content.text,
        **({} if content.textSignature is None else {"textSignature": content.textSignature}),
    )


def _cloneThinkingContent(content: ThinkingContent) -> ThinkingContent:
    return ThinkingContent(
        type="thinking",
        thinking=content.thinking,
        **({} if content.thinkingSignature is None else {"thinkingSignature": content.thinkingSignature}),
        **({} if content.redacted is None else {"redacted": content.redacted}),
    )


def _cloneToolCall(toolCall: ToolCall) -> ToolCall:
    return ToolCall(
        type="toolCall",
        id=toolCall.id,
        name=toolCall.name,
        arguments=copy.deepcopy(toolCall.arguments),
        **({} if toolCall.thoughtSignature is None else {"thoughtSignature": toolCall.thoughtSignature}),
        **({} if toolCall.namespace is None else {"namespace": toolCall.namespace}),
    )


def _cloneStartMessage(message: AssistantMessage) -> AssistantMessage:
    return AssistantMessage(
        role="assistant",
        content=[],
        api=message.api,
        provider=message.provider,
        model=message.model,
        **({} if message.responseModel is None else {"responseModel": message.responseModel}),
        **({} if message.responseId is None else {"responseId": message.responseId}),
        **({} if message.providerThinkingLevel is None else {"providerThinkingLevel": message.providerThinkingLevel}),
        **(
            {}
            if message.diagnostics is None
            else {"diagnostics": [diagnostic.model_copy(deep=True) for diagnostic in message.diagnostics]}
        ),
        usage=message.usage.model_copy(deep=True),
        stopReason="pending",
        timestamp=message.timestamp,
    )


def _assertContentIndex(contentIndex: Any) -> None:
    if not jsIsSafeInteger(contentIndex) or contentIndex < 0:
        raise RuntimeError(f"Invalid assistant message frame contentIndex: {jsString(contentIndex)}")


def _eventBlock(event: Any) -> Any:
    _assertContentIndex(event.contentIndex)
    content = event.partial.content
    block = content[event.contentIndex] if event.contentIndex < len(content) else None
    if not block:
        raise RuntimeError(f"{event.type} event has no content block at index {event.contentIndex}")
    return block


def _serializedArguments(argumentsValue: Any) -> str:
    if argumentsValue is UNDEFINED or callable(argumentsValue):
        raise RuntimeError("Tool-call arguments are not JSON-serializable")
    return jsonStringify(argumentsValue)


_EMPTY_PARSED_TOOL_ARGUMENTS = _serializedArguments(parseStreamingJson(""))


def _objectIs(left: Any, right: Any) -> bool:
    """``Object.is`` on JSON values: ``NaN`` is itself, ``0`` and ``-0`` differ, ``true`` is not ``1``."""
    if isinstance(left, bool) or isinstance(right, bool):
        return isinstance(left, bool) and isinstance(right, bool) and left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        if isinstance(left, float) and isinstance(right, float) and math.isnan(left) and math.isnan(right):
            return True
        if left == 0 and right == 0:
            return math.copysign(1, left) == math.copysign(1, right)
        return left == right
    return type(left) is type(right) and left == right


def _isJsonPrefix(snapshot: Any, current: Any) -> bool:
    if isinstance(snapshot, str):
        return isinstance(current, str) and current.startswith(snapshot)
    if isinstance(snapshot, list):
        return (
            isinstance(current, list)
            and len(snapshot) <= len(current)
            and all(_isJsonPrefix(value, current[index]) for index, value in enumerate(snapshot))
        )
    if not isinstance(snapshot, dict):
        return _objectIs(snapshot, current)
    if not isinstance(current, dict):
        return False
    currentRecord = current
    return all(key in currentRecord and _isJsonPrefix(value, currentRecord[key]) for key, value in snapshot.items())


class AssistantMessageFrameEncoder:
    """Encodes one assistant stream. `partial` remains a shared live accumulator;
    the encoder uses per-block offsets to avoid replaying deltas already visible
    when an older queued event is consumed."""

    def __init__(self) -> None:
        self._started = False
        self._terminal = False
        self._blocks: dict[int, _EncoderBlockState] = {}

    def encode(self, event: AssistantMessageEvent) -> AssistantMessageFrame | None:
        if self._terminal:
            raise RuntimeError(f"Assistant message event {event.type} follows a terminal event")

        match event.type:
            case "start":
                if self._started:
                    raise RuntimeError("Assistant message stream contains more than one start event")
                self._started = True
                return {"type": "start", "partial": _cloneStartMessage(event.partial)}
            case "done":
                if not self._started:
                    raise RuntimeError("Assistant message done event appears before start")
                self._terminal = True
                return None
            case "error":
                self._terminal = True
                return None

        if not self._started:
            raise RuntimeError(f"Assistant message {event.type} event appears before start")

        match event.type:
            case "text_start":
                content = _eventBlock(event)
                if content.type != "text":
                    raise RuntimeError(f"text_start event points to {content.type} block at index {event.contentIndex}")
                self._startBlock(
                    event.contentIndex,
                    _EncoderTextBlockState(kind="text", coveredChars=jsLength(content.text), deltaChars=0),
                )
                return {"type": "text_start", "contentIndex": event.contentIndex, "content": _cloneTextContent(content)}
            case "text_delta":
                return self._encodeTextDelta(event.contentIndex, event.delta, "text")
            case "text_end":
                content = _eventBlock(event)
                if content.type != "text":
                    raise RuntimeError(f"text_end event points to {content.type} block at index {event.contentIndex}")
                self._endBlock(event.contentIndex, "text")
                return {
                    "type": "text_end",
                    "contentIndex": event.contentIndex,
                    "content": event.content,
                    **({} if content.textSignature is None else {"textSignature": content.textSignature}),
                }
            case "thinking_start":
                content = _eventBlock(event)
                if content.type != "thinking":
                    raise RuntimeError(
                        f"thinking_start event points to {content.type} block at index {event.contentIndex}"
                    )
                self._startBlock(
                    event.contentIndex,
                    _EncoderTextBlockState(kind="thinking", coveredChars=jsLength(content.thinking), deltaChars=0),
                )
                return {
                    "type": "thinking_start",
                    "contentIndex": event.contentIndex,
                    "content": _cloneThinkingContent(content),
                }
            case "thinking_delta":
                return self._encodeTextDelta(event.contentIndex, event.delta, "thinking")
            case "thinking_end":
                content = _eventBlock(event)
                if content.type != "thinking":
                    raise RuntimeError(
                        f"thinking_end event points to {content.type} block at index {event.contentIndex}"
                    )
                self._endBlock(event.contentIndex, "thinking")
                return {
                    "type": "thinking_end",
                    "contentIndex": event.contentIndex,
                    "content": event.content,
                    **({} if content.thinkingSignature is None else {"thinkingSignature": content.thinkingSignature}),
                    **({} if content.redacted is None else {"redacted": content.redacted}),
                }
            case "toolcall_start":
                content = _eventBlock(event)
                if content.type != "toolCall":
                    raise RuntimeError(
                        f"toolcall_start event points to {content.type} block at index {event.contentIndex}"
                    )
                snapshotArguments = _serializedArguments(content.arguments)
                caughtUp = snapshotArguments == _EMPTY_PARSED_TOOL_ARGUMENTS
                self._startBlock(
                    event.contentIndex,
                    _EncoderToolCallBlockState(
                        kind="toolCall",
                        caughtUp=caughtUp,
                        catchupJson="",
                        snapshotArguments="" if caughtUp else snapshotArguments,
                    ),
                )
                return {"type": "toolcall_start", "contentIndex": event.contentIndex, "toolCall": _cloneToolCall(content)}
            case "toolcall_delta":
                state = self._block(event.contentIndex, "toolCall")
                if not isinstance(state, _EncoderToolCallBlockState):
                    raise RuntimeError("Unreachable tool-call encoder state")  # noqa: TRY004
                if state.caughtUp:
                    return (
                        None
                        if jsLength(event.delta) == 0
                        else {"type": "toolcall_delta", "contentIndex": event.contentIndex, "delta": event.delta}
                    )
                state.catchupJson += event.delta
                argumentsValue = parseStreamingJson(state.catchupJson)
                if _serializedArguments(argumentsValue) != state.snapshotArguments:
                    # Legacy grammar calls include the initial input in toolcall_start, but their
                    # JSON delta stream still begins at an empty input. Its parsed arguments can
                    # therefore extend, rather than exactly reproduce, the start snapshot.
                    snapshotArguments = parseStreamingJson(state.snapshotArguments)
                    if not _isJsonPrefix(snapshotArguments, argumentsValue):
                        return None
                state.caughtUp = True
                state.snapshotArguments = ""
                json = state.catchupJson
                state.catchupJson = ""
                return (
                    None
                    if jsLength(json) == 0
                    else {"type": "toolcall_checkpoint", "contentIndex": event.contentIndex, "json": json}
                )
            case "toolcall_end":
                content = _eventBlock(event)
                if content.type != "toolCall":
                    raise RuntimeError(
                        f"toolcall_end event points to {content.type} block at index {event.contentIndex}"
                    )
                if event.toolCall.type != "toolCall":
                    raise RuntimeError(f"toolcall_end event has invalid tool call at index {event.contentIndex}")
                self._endBlock(event.contentIndex, "toolCall")
                return {
                    "type": "toolcall_end",
                    "contentIndex": event.contentIndex,
                    "id": event.toolCall.id,
                    "name": event.toolCall.name,
                    "arguments": copy.deepcopy(event.toolCall.arguments),
                    **(
                        {}
                        if event.toolCall.thoughtSignature is None
                        else {"thoughtSignature": event.toolCall.thoughtSignature}
                    ),
                    **({} if event.toolCall.namespace is None else {"namespace": event.toolCall.namespace}),
                }
        return None

    def _startBlock(self, contentIndex: int, state: _EncoderBlockState) -> None:
        _assertContentIndex(contentIndex)
        if contentIndex in self._blocks:
            raise RuntimeError(f"Assistant message block {contentIndex} starts more than once")
        self._blocks[contentIndex] = state

    def _block(self, contentIndex: int, kind: str) -> _EncoderBlockState:
        _assertContentIndex(contentIndex)
        state = self._blocks.get(contentIndex)
        if state is None:
            raise RuntimeError(f"Assistant message {kind} block {contentIndex} has not started")
        if state.kind != kind:
            raise RuntimeError(f"Assistant message block {contentIndex} is {state.kind}, not {kind}")
        return state

    def _endBlock(self, contentIndex: int, kind: str) -> None:
        self._block(contentIndex, kind)
        del self._blocks[contentIndex]

    def _encodeTextDelta(
        self,
        contentIndex: int,
        delta: str,
        kind: Literal["text", "thinking"],
    ) -> AssistantMessageFrame | None:
        state = self._block(contentIndex, kind)
        if not isinstance(state, _EncoderTextBlockState):
            raise RuntimeError("Unreachable text encoder state")  # noqa: TRY004
        deltaStart = state.deltaChars
        state.deltaChars += jsLength(delta)
        covered = max(0, state.coveredChars - deltaStart)
        if covered >= jsLength(delta):
            return None
        uncovered = delta if covered == 0 else jsSlice(delta, covered)
        return (
            {"type": "text_delta", "contentIndex": contentIndex, "delta": uncovered}
            if kind == "text"
            else {"type": "thinking_delta", "contentIndex": contentIndex, "delta": uncovered}
        )


def _appendBlock(
    message: AssistantMessage,
    states: dict[int, _ReducerBlockState],
    contentIndex: int,
    block: TextContent | ThinkingContent | ToolCall,
    state: _ReducerBlockState,
) -> None:
    _assertContentIndex(contentIndex)
    if contentIndex != len(message.content):
        reason = "already exists" if contentIndex < len(message.content) else "would leave a gap"
        raise RuntimeError(f"Cannot start assistant message block at index {contentIndex}: {reason}")
    message.content.append(block.model_copy(deep=True))
    states[contentIndex] = state


def _activeBlock(
    message: AssistantMessage,
    states: dict[int, _ReducerBlockState],
    contentIndex: int,
    expectedKind: str,
    frameType: str,
) -> tuple[Any, _ReducerBlockState]:
    _assertContentIndex(contentIndex)
    state = states.get(contentIndex)
    block = message.content[contentIndex] if contentIndex < len(message.content) else None
    if not state or not block:
        raise RuntimeError(f"{frameType} frame has no started block at index {contentIndex}")
    if state.kind != expectedKind or block.type != expectedKind:
        raise RuntimeError(
            f"{frameType} frame expected {expectedKind} block at index {contentIndex}, found {block.type}"
        )
    if state.ended:
        raise RuntimeError(f"{frameType} frame follows the end of block at index {contentIndex}")
    return block, state


def reduceAssistantMessageFrames(frames: Iterable[AssistantMessageFrame]) -> AssistantMessage | None:
    """Replay compact frames without mutating them. Returns `undefined` when the
    iterable contains no start frame."""
    message: AssistantMessage | None = None
    frameBeforeStart: str | None = None
    states: dict[int, _ReducerBlockState] = {}

    for frame in frames:
        if frame["type"] == "start":
            if message:
                raise RuntimeError("Assistant message frame sequence contains more than one start frame")
            if frameBeforeStart is not None:
                raise RuntimeError(f"{frameBeforeStart} frame appears before the start frame")
            message = frame["partial"].model_copy(deep=True)
            continue
        if not message:
            if frameBeforeStart is None:
                frameBeforeStart = frame["type"]
            continue

        match frame["type"]:
            case "text_start":
                if frame["content"].type != "text":
                    raise RuntimeError(f"text_start frame contains {frame['content'].type} content")
                _appendBlock(message, states, frame["contentIndex"], frame["content"], _ReducerBlockState("text", False))
            case "text_delta":
                block, _state = _activeBlock(message, states, frame["contentIndex"], "text", frame["type"])
                if block.type != "text":
                    raise RuntimeError("Unreachable text frame state")
                block.text += frame["delta"]
            case "text_end":
                block, state = _activeBlock(message, states, frame["contentIndex"], "text", frame["type"])
                if block.type != "text":
                    raise RuntimeError("Unreachable text frame state")
                block.text = frame["content"]
                block.textSignature = None
                if frame.get("textSignature") is not None:
                    block.textSignature = frame["textSignature"]
                state.ended = True
            case "thinking_start":
                if frame["content"].type != "thinking":
                    raise RuntimeError(f"thinking_start frame contains {frame['content'].type} content")
                _appendBlock(
                    message, states, frame["contentIndex"], frame["content"], _ReducerBlockState("thinking", False)
                )
            case "thinking_delta":
                block, _state = _activeBlock(message, states, frame["contentIndex"], "thinking", frame["type"])
                if block.type != "thinking":
                    raise RuntimeError("Unreachable thinking frame state")
                block.thinking += frame["delta"]
            case "thinking_end":
                block, state = _activeBlock(message, states, frame["contentIndex"], "thinking", frame["type"])
                if block.type != "thinking":
                    raise RuntimeError("Unreachable thinking frame state")
                block.thinking = frame["content"]
                block.thinkingSignature = None
                block.redacted = None
                if frame.get("thinkingSignature") is not None:
                    block.thinkingSignature = frame["thinkingSignature"]
                if frame.get("redacted") is not None:
                    block.redacted = frame["redacted"]
                state.ended = True
            case "toolcall_start":
                if frame["toolCall"].type != "toolCall":
                    raise RuntimeError(f"toolcall_start frame contains {frame['toolCall'].type} content")
                _appendBlock(
                    message, states, frame["contentIndex"], frame["toolCall"], _ReducerBlockState("toolCall", False, "")
                )
            case "toolcall_checkpoint":
                block, state = _activeBlock(message, states, frame["contentIndex"], "toolCall", frame["type"])
                if block.type != "toolCall" or state.kind != "toolCall":
                    raise RuntimeError("Unreachable tool-call checkpoint state")
                state.json = frame["json"]
                block.arguments = parseStreamingJson(frame["json"])
            case "toolcall_delta":
                block, state = _activeBlock(message, states, frame["contentIndex"], "toolCall", frame["type"])
                if block.type != "toolCall" or state.kind != "toolCall":
                    raise RuntimeError("Unreachable tool-call frame state")
                state.json += frame["delta"]
            case "toolcall_end":
                block, state = _activeBlock(message, states, frame["contentIndex"], "toolCall", frame["type"])
                if block.type != "toolCall":
                    raise RuntimeError("Unreachable tool-call frame state")
                block.id = frame["id"]
                block.name = frame["name"]
                block.arguments = copy.deepcopy(frame["arguments"])
                block.thoughtSignature = None
                block.namespace = None
                if frame.get("thoughtSignature") is not None:
                    block.thoughtSignature = frame["thoughtSignature"]
                if frame.get("namespace") is not None:
                    block.namespace = frame["namespace"]
                state.ended = True

    if not message:
        return None
    for contentIndex, state in states.items():
        if state.kind != "toolCall" or state.ended or jsLength(state.json) == 0:
            continue
        block = message.content[contentIndex] if contentIndex < len(message.content) else None
        if block is None or block.type != "toolCall":
            raise RuntimeError("Unreachable tool-call frame state")
        block.arguments = parseStreamingJson(state.json)

    return message


__all__ = ["AssistantMessageFrame", "AssistantMessageFrameEncoder", "reduceAssistantMessageFrames"]
