"""Translated from pi's ``packages/ai/test/assistant-message-frame.test.ts`` (v1.1.0)."""

from __future__ import annotations

from typing import Any

import pytest

from misaka.ai.api.openai_responses_shared import processResponsesStream
from misaka.ai.types import (
    AssistantMessage,
    DoneEvent,
    ErrorEvent,
    Model,
    ModelCost,
    StartEvent,
    TextContent,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
    ThinkingContent,
    ThinkingEndEvent,
    ThinkingStartEvent,
    ToolCall,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
    Usage,
    UsageCost,
)
from misaka.ai.utils.assistant_message_frame import (
    AssistantMessageFrameEncoder,
    reduceAssistantMessageFrames,
)
from misaka.ai.utils.diagnostics import AssistantMessageDiagnostic
from misaka.ai.utils.event_stream import AssistantMessageEventStream


def seed() -> AssistantMessage:
    return AssistantMessage(
        role="assistant",
        content=[],
        api="test-api",
        provider="test-provider",
        model="test-model",
        usage=Usage(
            input=0,
            output=0,
            cacheRead=0,
            cacheWrite=0,
            totalTokens=0,
            cost=UsageCost(input=0, output=0, cacheRead=0, cacheWrite=0, total=0),
        ),
        stopReason="pending",
        timestamp=1,
    )


def frame(encoder: AssistantMessageFrameEncoder, event: Any) -> dict[str, Any]:
    converted = encoder.encode(event)
    if not converted:
        raise RuntimeError(f"Expected {event.type} event to produce a frame")
    return converted


def dumped(content: list[Any] | None) -> list[dict[str, Any]]:
    """`toEqual` on content blocks: the blocks as plain objects, absent fields left out."""
    return [block.model_dump(exclude_none=True) for block in content or []]


def test_uses_authoritative_text_end_content_and_signature():
    partial = seed()
    encoder = AssistantMessageFrameEncoder()
    frames = [frame(encoder, StartEvent(partial=partial))]
    partial.content.append(TextContent(text="Hello "))
    frames.append(frame(encoder, TextStartEvent(contentIndex=0, partial=partial)))
    partial.content[0] = TextContent(text="Hello world", textSignature="sig-text")
    frames.extend(
        [
            frame(encoder, TextDeltaEvent(contentIndex=0, delta="incorrect", partial=partial)),
            frame(encoder, TextEndEvent(contentIndex=0, content="Hello world", partial=partial)),
        ]
    )

    assert frames[-1] == {
        "type": "text_end",
        "contentIndex": 0,
        "content": "Hello world",
        "textSignature": "sig-text",
    }
    reduced = reduceAssistantMessageFrames(frames)
    assert dumped(reduced.content if reduced else None) == [
        {"type": "text", "text": "Hello world", "textSignature": "sig-text"}
    ]


def test_preserves_provider_thinking_level_from_the_stream_start():
    partial = seed()
    partial.providerThinkingLevel = "high"
    encoder = AssistantMessageFrameEncoder()
    start = frame(encoder, StartEvent(partial=partial))

    assert start["type"] == "start"
    assert start["partial"].providerThinkingLevel == "high"
    reduced = reduceAssistantMessageFrames([start])
    assert reduced is not None and reduced.providerThinkingLevel == "high"


def test_preserves_initial_and_final_thinking_metadata_including_redaction():
    partial = seed()
    encoder = AssistantMessageFrameEncoder()
    frames = [frame(encoder, StartEvent(partial=partial))]
    partial.content.append(
        ThinkingContent(thinking="[redacted]", thinkingSignature="encrypted-start", redacted=True)
    )
    frames.append(frame(encoder, ThinkingStartEvent(contentIndex=0, partial=partial)))
    partial.content[0] = ThinkingContent(thinking="[redacted]", thinkingSignature="encrypted-final", redacted=True)
    frames.append(frame(encoder, ThinkingEndEvent(contentIndex=0, content="[redacted]", partial=partial)))

    assert frames[-1] == {
        "type": "thinking_end",
        "contentIndex": 0,
        "content": "[redacted]",
        "thinkingSignature": "encrypted-final",
        "redacted": True,
    }
    reduced = reduceAssistantMessageFrames(frames)
    assert reduced is not None
    assert reduced.content[0].model_dump(exclude_none=True) == {
        "type": "thinking",
        "thinking": "[redacted]",
        "thinkingSignature": "encrypted-final",
        "redacted": True,
    }


def test_parses_unfinished_tool_json_once_and_uses_authoritative_completed_arguments():
    initialFrames: list[dict[str, Any]] = [
        {"type": "start", "partial": seed()},
        {
            "type": "toolcall_start",
            "contentIndex": 0,
            "toolCall": ToolCall(id="initial-id", name="write", arguments={}),
        },
        {"type": "toolcall_delta", "contentIndex": 0, "delta": '{"path":"READ'},
    ]

    reduced = reduceAssistantMessageFrames(initialFrames)
    assert reduced is not None
    assert reduced.content[0].type == "toolCall"
    assert reduced.content[0].arguments == {"path": "READ"}

    completeFrames = [
        *initialFrames,
        {"type": "toolcall_delta", "contentIndex": 0, "delta": 'ME.md","lines":[1,2]}'},
        {
            "type": "toolcall_end",
            "contentIndex": 0,
            "id": "final-id",
            "name": "write_file",
            "arguments": {"path": "final.md", "lines": [3]},
            "thoughtSignature": "thought",
            "namespace": "files",
        },
    ]
    reduced = reduceAssistantMessageFrames(completeFrames)
    assert reduced is not None
    assert reduced.content[0].model_dump(exclude_none=True) == {
        "type": "toolCall",
        "id": "final-id",
        "name": "write_file",
        "arguments": {"path": "final.md", "lines": [3]},
        "thoughtSignature": "thought",
        "namespace": "files",
    }


async def test_round_trips_openai_responses_content_supplied_only_by_authoritative_end_events():
    output = seed()
    output.api = "openai-responses"
    output.provider = "openai"
    model = Model(
        id=output.model,
        name="Test",
        api="openai-responses",
        provider="openai",
        baseUrl="https://api.openai.com/v1",
        reasoning=False,
        input=["text"],
        cost=ModelCost(input=0, output=0, cacheRead=0, cacheWrite=0),
        contextWindow=1000,
        maxTokens=100,
    )
    events: list[dict[str, Any]] = [
        {
            "type": "response.output_item.added",
            "sequence_number": 0,
            "output_index": 0,
            "item": {"type": "message", "id": "msg", "role": "assistant", "status": "in_progress", "content": []},
        },
        {
            "type": "response.output_item.done",
            "sequence_number": 1,
            "output_index": 0,
            "item": {
                "type": "message",
                "id": "msg",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "final text", "annotations": []}],
            },
        },
        {
            "type": "response.output_item.added",
            "sequence_number": 2,
            "output_index": 1,
            "item": {"type": "function_call", "id": "fc", "call_id": "call", "name": "lookup", "arguments": ""},
        },
        {
            "type": "response.output_item.done",
            "sequence_number": 3,
            "output_index": 1,
            "item": {
                "type": "function_call",
                "id": "fc",
                "call_id": "call",
                "name": "lookup",
                "arguments": '{"query":"pi"}',
            },
        },
        {
            "type": "response.completed",
            "sequence_number": 4,
            "response": {"id": "response", "status": "completed", "output": []},
        },
    ]

    async def source():
        for event in events:
            yield event

    encoder = AssistantMessageFrameEncoder()
    frames = [frame(encoder, StartEvent(partial=output))]
    stream = AssistantMessageEventStream()
    push = stream.push

    def pushing(event: Any, **kwargs: Any) -> None:
        converted = encoder.encode(event)
        if converted:
            frames.append(converted)
        push(event, **kwargs)

    stream.push = pushing  # type: ignore[method-assign]
    await processResponsesStream(source(), output, stream, model)

    reduced = reduceAssistantMessageFrames(frames)
    assert dumped(reduced.content if reduced else None) == dumped(output.content)


def test_reconciles_queued_text_events_against_one_advanced_live_partial_without_duplicate_content():
    partial = seed()
    events: list[Any] = [StartEvent(partial=partial)]
    text = TextContent(text="")
    partial.content.append(text)
    events.append(TextStartEvent(contentIndex=0, partial=partial))
    for delta in ["Hel", "lo", " ", "world"]:
        text.text += delta
        events.append(TextDeltaEvent(contentIndex=0, delta=delta, partial=partial))

    encoder = AssistantMessageFrameEncoder()
    frames = [encoded for event in events if (encoded := encoder.encode(event)) is not None]

    assert [item["type"] for item in frames] == ["start", "text_start"]
    assert frames[0]["type"] == "start"
    assert frames[0]["partial"].content == []
    assert frames[0]["partial"].stopReason == "pending"
    reduced = reduceAssistantMessageFrames(frames)
    assert dumped(reduced.content if reduced else None) == [{"type": "text", "text": "Hello world"}]


def test_trims_only_the_covered_prefix_when_a_start_snapshot_lands_inside_a_delta():
    partial = seed()
    encoder = AssistantMessageFrameEncoder()
    frames = [frame(encoder, StartEvent(partial=partial))]
    text = TextContent(text="Hel")
    partial.content.append(text)
    frames.append(frame(encoder, TextStartEvent(contentIndex=0, partial=partial)))
    assert encoder.encode(TextDeltaEvent(contentIndex=0, delta="He", partial=partial)) is None
    remainder = encoder.encode(TextDeltaEvent(contentIndex=0, delta="llo", partial=partial))
    if remainder is None:
        raise RuntimeError("Expected uncovered text delta")
    frames.append(remainder)

    assert remainder == {"type": "text_delta", "contentIndex": 0, "delta": "lo"}
    reduced = reduceAssistantMessageFrames(frames)
    assert dumped(reduced.content if reduced else None) == [{"type": "text", "text": "Hello"}]


def test_checkpoints_queued_tool_json_without_replaying_covered_deltas():
    partial = seed()
    toolCall = ToolCall(id="call", name="write", arguments={})
    events: list[Any] = [StartEvent(partial=partial)]
    partial.content.append(toolCall)
    events.append(ToolCallStartEvent(contentIndex=0, partial=partial))
    toolCall.arguments = {"path": "README.md"}
    events.extend(
        [
            ToolCallDeltaEvent(contentIndex=0, delta='{"path":"READ', partial=partial),
            ToolCallDeltaEvent(contentIndex=0, delta='ME.md"}', partial=partial),
        ]
    )

    encoder = AssistantMessageFrameEncoder()
    frames = [encoded for event in events if (encoded := encoder.encode(event)) is not None]
    assert [item["type"] for item in frames] == ["start", "toolcall_start", "toolcall_checkpoint"]
    assert frames[-1] == {"type": "toolcall_checkpoint", "contentIndex": 0, "json": '{"path":"README.md"}'}
    reduced = reduceAssistantMessageFrames(frames)
    assert dumped(reduced.content if reduced else None) == [
        {"type": "toolCall", "id": "call", "name": "write", "arguments": {"path": "README.md"}}
    ]


def test_resumes_legacy_grammar_tool_json_from_initial_arguments():
    partial = seed()
    encoder = AssistantMessageFrameEncoder()
    frames = [frame(encoder, StartEvent(partial=partial))]
    toolCall = ToolCall(id="call", name="bash", arguments={"input": "a"})
    partial.content.append(toolCall)
    frames.append(frame(encoder, ToolCallStartEvent(contentIndex=0, partial=partial)))
    toolCall.arguments = {"input": "ab"}
    frames.append(frame(encoder, ToolCallDeltaEvent(contentIndex=0, delta='{"input":"ab', partial=partial)))
    toolCall.arguments = {"input": "abc"}
    frames.append(frame(encoder, ToolCallDeltaEvent(contentIndex=0, delta='c"}', partial=partial)))

    assert frames[2:] == [
        {"type": "toolcall_checkpoint", "contentIndex": 0, "json": '{"input":"ab'},
        {"type": "toolcall_delta", "contentIndex": 0, "delta": 'c"}'},
    ]
    reduced = reduceAssistantMessageFrames(frames)
    assert dumped(reduced.content if reduced else None) == [
        {"type": "toolCall", "id": "call", "name": "bash", "arguments": {"input": "abc"}}
    ]


def test_streams_tool_json_compactly_from_an_empty_argument_start():
    partial = seed()
    encoder = AssistantMessageFrameEncoder()
    frames = [frame(encoder, StartEvent(partial=partial))]
    toolCall = ToolCall(id="call", name="bash", arguments={})
    partial.content.append(toolCall)
    frames.append(frame(encoder, ToolCallStartEvent(contentIndex=0, partial=partial)))
    toolCall.arguments = {"command": "ls -la /tmp"}
    frames.append(
        frame(encoder, ToolCallDeltaEvent(contentIndex=0, delta='{"command":"ls -la /tmp"}', partial=partial))
    )

    assert frames[-1] == {"type": "toolcall_delta", "contentIndex": 0, "delta": '{"command":"ls -la /tmp"}'}
    reduced = reduceAssistantMessageFrames(frames)
    assert reduced is not None
    assert reduced.content[0].type == "toolCall"
    assert reduced.content[0].arguments == {"command": "ls -la /tmp"}


def test_accepts_a_pre_generation_error_but_rejects_success_or_updates_before_start():
    failed = seed()
    failed.stopReason = "error"
    failed.errorMessage = "setup failed"
    assert AssistantMessageFrameEncoder().encode(ErrorEvent(reason="error", error=failed)) is None

    completed = seed()
    completed.stopReason = "stop"
    with pytest.raises(RuntimeError, match="done event appears before start"):
        AssistantMessageFrameEncoder().encode(DoneEvent(reason="stop", message=completed))
    with pytest.raises(RuntimeError, match="text_delta event appears before start"):
        AssistantMessageFrameEncoder().encode(TextDeltaEvent(contentIndex=0, delta="x", partial=seed()))


def test_treats_end_signature_metadata_including_absence_as_authoritative():
    frames: list[dict[str, Any]] = [
        {"type": "start", "partial": seed()},
        {"type": "text_start", "contentIndex": 0, "content": TextContent(text="", textSignature="stale-text")},
        {"type": "text_end", "contentIndex": 0, "content": ""},
        {
            "type": "thinking_start",
            "contentIndex": 1,
            "content": ThinkingContent(thinking="", thinkingSignature="stale-thinking", redacted=True),
        },
        {"type": "thinking_end", "contentIndex": 1, "content": "", "thinkingSignature": "", "redacted": False},
        {
            "type": "toolcall_start",
            "contentIndex": 2,
            "toolCall": ToolCall(
                id="call", name="read", arguments={}, thoughtSignature="stale-tool", namespace="stale-namespace"
            ),
        },
        {"type": "toolcall_end", "contentIndex": 2, "id": "call", "name": "read", "arguments": {}},
    ]

    reduced = reduceAssistantMessageFrames(frames)
    assert dumped(reduced.content if reduced else None) == [
        {"type": "text", "text": ""},
        {"type": "thinking", "thinking": "", "thinkingSignature": "", "redacted": False},
        {"type": "toolCall", "id": "call", "name": "read", "arguments": {}},
    ]


def test_stores_authoritative_final_arguments_in_toolcall_end_frames():
    partial = seed()
    toolCall = ToolCall(
        id="call-1", name="read", arguments={"path": "README.md"}, thoughtSignature="thought", namespace="files"
    )
    partial.content.append(toolCall)

    encoder = AssistantMessageFrameEncoder()
    frame(encoder, StartEvent(partial=partial))
    frame(encoder, ToolCallStartEvent(contentIndex=0, partial=partial))
    end = frame(encoder, ToolCallEndEvent(contentIndex=0, toolCall=toolCall, partial=partial))
    assert end == {
        "type": "toolcall_end",
        "contentIndex": 0,
        "id": "call-1",
        "name": "read",
        "arguments": {"path": "README.md"},
        "thoughtSignature": "thought",
        "namespace": "files",
    }


def test_whitelists_public_block_fields_from_provider_shaped_partials():
    # MISAKA fork: the blocks are strict models; adapter scratch rides on them as plain attributes.
    partial = seed()
    text = TextContent(text="visible", textSignature="text-sig")
    object.__setattr__(text, "index", 4)
    thinking = ThinkingContent(thinking="reasoning", thinkingSignature="thinking-sig", redacted=False)
    object.__setattr__(thinking, "index", 5)
    toolCall = ToolCall(id="call", name="run", arguments={"value": 1}, thoughtSignature="tool-sig", namespace="tools")
    object.__setattr__(toolCall, "partialJson", '{"value":')
    object.__setattr__(toolCall, "streamIndex", 6)
    partial.content.extend([text, thinking, toolCall])
    object.__setattr__(partial, "outputIndex", 3)

    encoder = AssistantMessageFrameEncoder()
    start = frame(encoder, StartEvent(partial=partial))
    textStart = frame(encoder, TextStartEvent(contentIndex=0, partial=partial))
    thinkingStart = frame(encoder, ThinkingStartEvent(contentIndex=1, partial=partial))
    toolStart = frame(encoder, ToolCallStartEvent(contentIndex=2, partial=partial))

    assert start["type"] == "start" and start["partial"].content == []
    assert not hasattr(start["partial"], "outputIndex")
    assert not hasattr(textStart["content"], "index")
    assert not hasattr(thinkingStart["content"], "index")
    assert not hasattr(toolStart["toolCall"], "partialJson")
    assert not hasattr(toolStart["toolCall"], "streamIndex")


def test_supports_interleaved_streams_by_content_index():
    frames: list[dict[str, Any]] = [
        {"type": "start", "partial": seed()},
        {"type": "text_start", "contentIndex": 0, "content": TextContent(text="")},
        {"type": "toolcall_start", "contentIndex": 1, "toolCall": ToolCall(id="call", name="lookup", arguments={})},
        {"type": "thinking_start", "contentIndex": 2, "content": ThinkingContent(thinking="")},
        {"type": "text_delta", "contentIndex": 0, "delta": "answer"},
        {"type": "toolcall_delta", "contentIndex": 1, "delta": '{"query":"pi"}'},
        {"type": "thinking_delta", "contentIndex": 2, "delta": "check"},
        {"type": "toolcall_end", "contentIndex": 1, "id": "call", "name": "lookup", "arguments": {"query": "pi"}},
        {"type": "text_end", "contentIndex": 0, "content": "answer"},
        {"type": "thinking_end", "contentIndex": 2, "content": "check"},
    ]

    reduced = reduceAssistantMessageFrames(frames)
    assert dumped(reduced.content if reduced else None) == [
        {"type": "text", "text": "answer"},
        {"type": "toolCall", "id": "call", "name": "lookup", "arguments": {"query": "pi"}},
        {"type": "thinking", "thinking": "check"},
    ]


def test_snapshots_mutable_event_data_and_keeps_reduction_pure():
    partial = seed()
    partial.diagnostics = [AssistantMessageDiagnostic(type="test", timestamp=2, details={"value": "original"})]
    encoder = AssistantMessageFrameEncoder()
    start = frame(encoder, StartEvent(partial=partial))
    partial.diagnostics[0].details["value"] = "mutated"  # type: ignore[index]
    partial.usage.cost.total = 99

    partial.content.append(ToolCall(id="call", name="run", arguments={"nested": {"value": "original"}}))
    toolStart = frame(encoder, ToolCallStartEvent(contentIndex=0, partial=partial))
    sourceTool = partial.content[0]
    if sourceTool.type != "toolCall":
        raise RuntimeError("Expected source tool call")
    sourceTool.arguments["nested"]["value"] = "mutated"

    reduced = reduceAssistantMessageFrames([start, toolStart])
    assert reduced is not None
    assert reduced.diagnostics[0].details["value"] == "original"  # type: ignore[index]
    assert reduced.usage.cost.total == 0
    assert reduced.content[0].arguments == {"nested": {"value": "original"}}

    if reduced.content[0].type != "toolCall":
        raise RuntimeError("Expected reduced tool call")
    reduced.content[0].arguments["nested"] = "changed-output"
    assert toolStart["type"] == "toolcall_start" and toolStart["toolCall"].arguments["nested"] == {"value": "original"}


def test_omits_terminal_events_because_settlement_is_separate():
    message = seed()
    completed = AssistantMessageFrameEncoder()
    completed.encode(StartEvent(partial=message))
    message.stopReason = "stop"
    assert completed.encode(DoneEvent(reason="stop", message=message)) is None
    message.stopReason = "error"
    message.errorMessage = "failed"
    assert AssistantMessageFrameEncoder().encode(ErrorEvent(reason="error", error=message)) is None


def test_returns_undefined_when_there_is_no_start_frame():
    assert reduceAssistantMessageFrames([]) is None
    assert reduceAssistantMessageFrames([{"type": "text_delta", "contentIndex": 0, "delta": "x"}]) is None


def test_rejects_frames_before_start_wrong_block_kinds_duplicate_ends_and_index_gaps():
    with pytest.raises(RuntimeError, match="before the start frame"):
        reduceAssistantMessageFrames(
            [
                {"type": "text_delta", "contentIndex": 0, "delta": "x"},
                {"type": "start", "partial": seed()},
            ]
        )
    with pytest.raises(RuntimeError, match="expected text block"):
        reduceAssistantMessageFrames(
            [
                {"type": "start", "partial": seed()},
                {"type": "toolcall_start", "contentIndex": 0, "toolCall": ToolCall(id="call", name="run", arguments={})},
                {"type": "text_delta", "contentIndex": 0, "delta": "wrong"},
            ]
        )
    with pytest.raises(RuntimeError, match="follows the end"):
        reduceAssistantMessageFrames(
            [
                {"type": "start", "partial": seed()},
                {"type": "text_start", "contentIndex": 0, "content": TextContent(text="")},
                {"type": "text_end", "contentIndex": 0, "content": ""},
                {"type": "text_end", "contentIndex": 0, "content": ""},
            ]
        )
    with pytest.raises(RuntimeError, match="would leave a gap"):
        reduceAssistantMessageFrames(
            [
                {"type": "start", "partial": seed()},
                {"type": "text_start", "contentIndex": 1, "content": TextContent(text="")},
            ]
        )


def test_rejects_conversion_events_whose_content_index_points_to_the_wrong_block_kind():
    partial = seed()
    encoder = AssistantMessageFrameEncoder()
    encoder.encode(StartEvent(partial=partial))
    partial.content.append(ThinkingContent(thinking=""))
    with pytest.raises(RuntimeError, match="text_start event points to thinking block"):
        encoder.encode(TextStartEvent(contentIndex=0, partial=partial))
