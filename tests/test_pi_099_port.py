"""The pi 0.87.1 -> 0.99.1 port (2026-09-30). Each test is an upstream test case from
`packages/ai/test/`, named after the file it comes from."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import httpx
import openai
import pytest

from misaka.ai.models import get_model
from misaka.ai.providers import anthropic, mistral, openai_completions, openai_responses
from misaka.ai.providers.openai_responses_shared import process_responses_stream
from misaka.ai.stream import complete_simple, stream
from misaka.ai.types import (
    AssistantMessage,
    Context,
    Model,
    ProviderStreamOptions,
    SimpleStreamOptions,
    UserMessage,
)
from misaka.ai.utils.event_stream import AssistantMessageEventStream
from misaka.ai.utils.transcript import normalize_context

CONTEXT = Context(messages=[UserMessage(content="hi", timestamp=0)])


def _model(api: str, **overrides) -> Model:
    return Model.model_validate(
        {
            "id": "fixture",
            "name": "fixture",
            "api": api,
            "provider": "fixture",
            "baseUrl": "http://127.0.0.1:9",
            "reasoning": False,
            "input": ["text"],
            "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
            "contextWindow": 100_000,
            "maxTokens": 8192,
            **overrides,
        }
    )


def _output(model: Model) -> AssistantMessage:
    return AssistantMessage.model_validate(
        {
            "content": [],
            "api": model.api,
            "provider": model.provider,
            "model": model.id,
            "usage": {
                "input": 0,
                "output": 0,
                "cacheRead": 0,
                "cacheWrite": 0,
                "totalTokens": 0,
                "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0},
            },
            "stopReason": "stop",
            "timestamp": 0,
        }
    )


class _PayloadCaptured(Exception):
    pass


async def _capture_payload(model: Model, **options) -> dict:
    captured: dict = {}

    def on_payload(payload, _model):
        captured.update(payload)
        raise _PayloadCaptured

    await stream(model, CONTEXT, ProviderStreamOptions(apiKey="fake-key", onPayload=on_payload, **options)).result()
    return captured


# --- sampling-options.test.ts ------------------------------------------------------------


# Model defaults must apply to direct stream()/complete() calls, not only streamSimple() (#9506)
@pytest.mark.parametrize(
    ("api", "base_url"),
    [
        ("openai-completions", "http://127.0.0.1:9"),
        ("openai-responses", "http://127.0.0.1:9"),
        ("azure-openai-responses", "https://fixture.openai.azure.com/openai/v1"),
    ],
)
async def test_model_sampling_params_apply_with_request_keys_taking_precedence(api, base_url):
    model = _model(api, baseUrl=base_url, samplingParams={"top_p": 0.95, "min_p": 0.05})

    payload = await _capture_payload(model, samplingParams={"top_p": 0.5})

    assert payload["extra_body"]["top_p"] == 0.5
    assert payload["extra_body"]["min_p"] == 0.05


# --- mistral-reasoning-mode.test.ts --------------------------------------------------------

NONE_HIGH_LEVELS = {
    "off": "none",
    "minimal": None,
    "low": None,
    "medium": None,
    "high": "high",
    "xhigh": None,
    "max": None,
}
GLM_5_2_LEVELS = {**NONE_HIGH_LEVELS, "max": "max"}
GLM_5_3_LEVELS = {**NONE_HIGH_LEVELS, "off": None, "low": "low", "max": "max"}


def _mistral_payload(monkeypatch, model_id: str, reasoning: bool, levels=None, level=None) -> dict:
    model = _model(
        "mistral-conversations",
        id=model_id,
        provider="mistral",
        reasoning=reasoning,
        **({"thinkingLevelMap": levels} if levels else {}),
    )
    captured: dict = {}

    def capture(_model, context, options):
        captured["payload"] = mistral.build_chat_payload(_model, context, [], options)
        return AssistantMessageEventStream()

    monkeypatch.setattr(mistral, "stream_mistral", capture)
    mistral.stream_simple_mistral(model, normalize_context(CONTEXT), SimpleStreamOptions(apiKey="fake", reasoning=level))
    return captured["payload"]


def test_mistral_uses_prompt_mode_for_reasoning_models_without_a_thinking_level_map(monkeypatch):
    payload = _mistral_payload(monkeypatch, "magistral-medium-latest", True, level="medium")

    assert payload["promptMode"] == "reasoning"
    assert "reasoningEffort" not in payload


def test_mistral_omits_reasoning_controls_for_magistral_when_thinking_is_off(monkeypatch):
    payload = _mistral_payload(monkeypatch, "magistral-medium-latest", True)

    assert "reasoningEffort" not in payload
    assert "promptMode" not in payload


# Regression for #8700 and #9375: Medium and GLM-5.2 ignore Magistral's prompt_mode.
@pytest.mark.parametrize("model_id", ["mistral-small-2603", "mistral-medium-latest", "zai-glm-5-2"])
def test_mistral_effort_map_models(monkeypatch, model_id):
    levels = GLM_5_2_LEVELS if model_id == "zai-glm-5-2" else NONE_HIGH_LEVELS

    enabled = _mistral_payload(monkeypatch, model_id, True, levels, "high")
    assert enabled["reasoningEffort"] == "high"
    assert "promptMode" not in enabled
    # Clamps unsupported levels to a supported effort.
    assert _mistral_payload(monkeypatch, model_id, True, levels, "low")["reasoningEffort"] == "high"
    # Sends reasoning_effort none when thinking is off.
    assert _mistral_payload(monkeypatch, model_id, True, levels)["reasoningEffort"] == "none"


# Regression for #9678: requested levels must reach Mistral-hosted GLM models.
def test_mistral_sends_max_for_glm_5_2(monkeypatch):
    assert _mistral_payload(monkeypatch, "zai-glm-5-2", True, GLM_5_2_LEVELS, "max")["reasoningEffort"] == "max"


@pytest.mark.parametrize(("level", "expected"), [("low", "low"), ("high", "high"), ("max", "max"), ("medium", "high")])
def test_mistral_glm_5_3_levels(monkeypatch, level, expected):
    assert _mistral_payload(monkeypatch, "zai-glm-5-3", True, GLM_5_3_LEVELS, level)["reasoningEffort"] == expected


# Regression for #8700: reasoning controls must respect the model's reasoning capability.
def test_mistral_omits_reasoning_controls_for_non_reasoning_models(monkeypatch):
    payload = _mistral_payload(monkeypatch, "mistral-medium-2604", False, level="medium")

    assert "reasoningEffort" not in payload
    assert "promptMode" not in payload


async def test_mistral_skips_empty_content_deltas():
    # GLM models on Mistral send empty content deltas around thinking and tool calls.
    model = _model("mistral-conversations", provider="mistral")

    async def events():
        for content in [[{"type": "thinking", "thinking": [{"text": "a"}]}], "", [{"type": "text", "text": ""}],
                        [{"type": "thinking", "thinking": [{"text": "b"}]}]]:
            yield {"choices": [{"delta": {"content": content}}]}
        yield {"choices": [{"delta": {}, "finish_reason": "stop"}]}

    output = _output(model)
    assert await mistral.consume_chat_stream(model, output, AssistantMessageEventStream(), events())

    assert [block.type for block in output.content] == ["thinking"]
    assert output.content[0].thinking == "ab"


# --- openai-responses-terminal-event.test.ts ----------------------------------------------


def _responses_model() -> Model:
    return _model("openai-responses", provider="openai", baseUrl="https://api.openai.com/v1")


async def _events(events):
    for event in events:
        yield event


async def test_responses_rejects_completed_streams_whose_tool_call_never_received_output_item_done():
    model = _responses_model()
    events = [
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "bash", "arguments": ""},
        },
        {
            "type": "response.function_call_arguments.delta",
            "output_index": 0,
            "item_id": "fc_1",
            "delta": '{"command":"rm -rf /tmp/build',
        },
        {"type": "response.completed", "response": {"id": "resp_unfinished", "status": "completed"}},
    ]

    with pytest.raises(RuntimeError, match=r"unfinished tool call: bash \(call_1\|fc_1\)"):
        await process_responses_stream(_events(events), _output(model), AssistantMessageEventStream(), model)


# https://github.com/earendil-works/pi/issues/9974
async def test_responses_rejects_parallel_tool_calls_without_output_index():
    # llama.cpp omits output_index from every event and sends both done events after all deltas.
    model = _responses_model()

    def call(n):
        return {"type": "function_call", "id": f"fc_{n}", "call_id": f"call_{n}", "name": "bash"}

    events = [
        {"type": "response.output_item.added", "item": {**call("a"), "arguments": ""}},
        {"type": "response.function_call_arguments.delta", "item_id": "fc_a", "delta": '{"command":"echo a"}'},
        {"type": "response.output_item.added", "item": {**call("b"), "arguments": ""}},
        {"type": "response.function_call_arguments.delta", "item_id": "fc_b", "delta": '{"command":"echo b"}'},
        {"type": "response.output_item.done", "item": {**call("a"), "arguments": '{"command":"echo a"}'}},
        {"type": "response.output_item.done", "item": {**call("b"), "arguments": '{"command":"echo b"}'}},
        {"type": "response.completed", "response": {"id": "resp_no_output_index", "status": "completed"}},
    ]

    with pytest.raises(RuntimeError, match=r"unfinished tool call: bash \(call_a\|fc_a\)"):
        await process_responses_stream(_events(events), _output(model), AssistantMessageEventStream(), model)


async def test_responses_forwards_parsed_provider_stream_events_in_order(monkeypatch):
    model = _responses_model()
    events = [
        {"type": "response.created", "response": {"id": "resp_1"}},
        {"type": "response.output_item.added", "output_index": 0, "item": {"type": "reasoning", "id": "rs_1"}},
        {"type": "response.reasoning_text.delta", "output_index": 0, "delta": "thinking"},
    ]
    monkeypatch.setattr(openai_responses, "_create_responses_stream", AsyncMock(return_value=_events(events)))
    provider_events, event_models = [], []

    async def on_provider_stream_event(event, event_model):
        provider_events.append(event)
        event_models.append(event_model)

    await openai_responses.stream_openai_responses(
        model, normalize_context(CONTEXT), {"apiKey": "test", "onProviderStreamEvent": on_provider_stream_event}
    ).result()

    assert [event["type"] for event in provider_events] == [
        "response.created",
        "response.output_item.added",
        "response.reasoning_text.delta",
    ]
    assert event_models == [model, model, model]


@pytest.mark.parametrize(
    ("status", "details", "stop_reason", "raw", "error"),
    [
        ("incomplete", {"reason": "max_output_tokens"}, "length", "incomplete.max_output_tokens", None),
        ("incomplete", {"reason": "content_filter"}, "error", "incomplete.content_filter", "Response incomplete: content_filter"),
        ("incomplete", None, "error", "incomplete", "Response incomplete without a provider reason"),
    ],
)
async def test_responses_incomplete_keeps_the_provider_reason(status, details, stop_reason, raw, error):
    model = _responses_model()
    events = [{"type": "response.incomplete", "response": {"status": status, "incomplete_details": details}}]
    output = _output(model)

    await process_responses_stream(_events(events), output, AssistantMessageEventStream(), model)

    assert (output.stopReason, output.rawStopReason, output.errorMessage) == (stop_reason, raw, error)


async def test_responses_usage_counts_cache_writes_and_reasoning():
    model = _responses_model()
    usage = {
        "input_tokens": 100,
        "output_tokens": 30,
        "total_tokens": 130,
        "input_tokens_details": {"cached_tokens": 20, "cache_write_tokens": 10},
        "output_tokens_details": {"reasoning_tokens": 12},
    }
    output = _output(model)

    await process_responses_stream(
        _events([{"type": "response.completed", "response": {"status": "completed", "usage": usage}}]),
        output,
        AssistantMessageEventStream(),
        model,
    )

    assert (output.usage.input, output.usage.cacheRead, output.usage.cacheWrite) == (70, 20, 10)
    assert (output.usage.output, output.usage.reasoning, output.usage.totalTokens) == (30, 12, 130)


# --- openai-responses-chatgpt-sign-in.test.ts ---------------------------------------------


def _sign_in_params(api_key: str, model: Model | None = None) -> dict:
    options = {"apiKey": api_key, "maxTokens": 1000, "temperature": 0.5, "cacheRetention": "long"}
    return openai_responses.build_params(model or _responses_model(), normalize_context(CONTEXT), options)


def test_chatgpt_sign_in_omits_request_fields_that_token_sharing_rejects():
    payload = _sign_in_params("chatgpt-access-token")

    assert "max_output_tokens" not in payload
    assert "temperature" not in payload
    assert payload["prompt_cache_retention"] is None


def test_chatgpt_sign_in_omits_prompt_cache_options_on_explicit_cache_models():
    model = Model.model_validate(
        {**_responses_model().model_dump(), "compat": {"supportsExplicitPromptCacheMode": True}}
    )

    assert _sign_in_params("chatgpt-access-token", model)["prompt_cache_options"] is None
    assert _sign_in_params("sk-proj-test", model)["prompt_cache_options"] == {"ttl": "30m"}


@pytest.mark.parametrize(
    ("api_key", "base_url"),
    [("sk-proj-test", "https://api.openai.com/v1"), ("gateway-key", "https://gateway.example.com/v1")],
)
def test_chatgpt_sign_in_keeps_those_fields_for_api_keys_and_other_endpoints(api_key, base_url):
    model = _responses_model().model_copy(update={"baseUrl": base_url})
    payload = _sign_in_params(api_key, model)

    assert payload["max_output_tokens"] == 1000
    assert payload["temperature"] == 0.5
    assert payload["prompt_cache_retention"] == "24h"


# --- openai-responses-usage-limit.test.ts -------------------------------------------------

USAGE_LIMIT_ERROR = {"code": "subscription_sharing_usage_limit_exceeded", "message": "Usage limit reached."}


async def _error_message(monkeypatch, create) -> str | None:
    monkeypatch.setattr(openai_responses, "_create_responses_stream", create)
    result = await openai_responses.stream_openai_responses(
        _responses_model(), normalize_context(CONTEXT), {"apiKey": "test"}
    ).result()
    assert result.stopReason == "error"
    return result.errorMessage


async def test_usage_limit_links_to_chatgpt_usage_when_the_request_is_rejected(monkeypatch):
    body = {"error": {**USAGE_LIMIT_ERROR, "type": "rate_limit_error"}}
    response = httpx.Response(429, json=body, request=httpx.Request("POST", "https://api.openai.com/v1/responses"))
    rejected = openai.RateLimitError(f"Error code: 429 - {body}", response=response, body=body)

    error_message = await _error_message(monkeypatch, AsyncMock(side_effect=rejected))

    assert "subscription_sharing_usage_limit_exceeded" in error_message
    assert "Check your ChatGPT usage: https://chatgpt.com/settings/usage" in error_message


async def test_usage_limit_links_to_chatgpt_usage_when_the_stream_fails(monkeypatch):
    event = {"type": "response.failed", "response": {"id": "resp_failed", "status": "failed", "error": USAGE_LIMIT_ERROR}}

    error_message = await _error_message(monkeypatch, AsyncMock(return_value=_events([event])))

    assert "subscription_sharing_usage_limit_exceeded: Usage limit reached." in error_message
    assert "Check your ChatGPT usage: https://chatgpt.com/settings/usage" in error_message


# --- openai-completions-provider-stream-event.test.ts -------------------------------------


# Regression test for #9784.
async def test_completions_exposes_provider_chunks_including_openrouter_metadata(monkeypatch):
    first_chunk = {
        "id": "chatcmpl-1",
        "model": "anthropic/claude-sonnet-4.6",
        "choices": [{"index": 0, "delta": {"content": "hello"}}],
    }
    final_chunk = {
        "id": "chatcmpl-1",
        "model": "anthropic/claude-sonnet-4.6",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12, "cost": 0.0012, "is_byok": False},
        "openrouter_metadata": {"strategy": "direct", "region": "iad"},
    }
    monkeypatch.setattr(
        openai_completions, "_create_completion_stream", AsyncMock(return_value=_events([first_chunk, final_chunk]))
    )
    events: list = []
    model = _model(
        "openai-completions", id="openrouter/auto", provider="openrouter", baseUrl="https://openrouter.ai/api/v1"
    )

    message = await complete_simple(model, CONTEXT, SimpleStreamOptions(apiKey="test", onProviderStreamEvent=lambda data, _m: events.append(data)))

    assert [block.model_dump(exclude_none=True) for block in message.content] == [{"type": "text", "text": "hello"}]
    assert events == [first_chunk, final_chunk]


# --- anthropic-cache-write-1h-cost.test.ts ------------------------------------------------


# Regression for #9210: Vercel AI Gateway sends cache usage in message_delta, not message_start.
async def test_anthropic_prices_1h_cache_writes_reported_only_in_message_delta(monkeypatch):
    model = get_model("vercel-ai-gateway", "anthropic/claude-haiku-4.5")

    async def incoming():
        yield {"type": "message_start", "message": {"id": "msg_test", "usage": {"input_tokens": 0, "output_tokens": 0}}}
        yield {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {
                "input_tokens": 3,
                "output_tokens": 4,
                "cache_creation_input_tokens": 6535,
                "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 6535},
            },
        }
        yield {"type": "message_stop"}

    monkeypatch.setattr(anthropic, "_create_raw_response", AsyncMock(return_value=incoming()))
    result = await anthropic.stream_anthropic(model, normalize_context(CONTEXT), {"client": object()}).result()

    assert result.usage.cacheWrite == 6535
    assert result.usage.cacheWrite1h == 6535
    assert result.usage.cost.cacheWrite == pytest.approx((6535 * model.cost.input * 2) / 1_000_000, abs=1e-10)


async def test_anthropic_forwards_provider_stream_events(monkeypatch):
    model = get_model("anthropic", "claude-haiku-4-5")
    raw = [
        {"type": "message_start", "message": {"id": "msg_1", "usage": {"input_tokens": 1, "output_tokens": 0}}},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 1}},
        {"type": "message_stop"},
    ]
    monkeypatch.setattr(anthropic, "_create_raw_response", AsyncMock(return_value=_events(raw)))
    seen: list = []

    await anthropic.stream_anthropic(
        model,
        normalize_context(CONTEXT),
        {"client": object(), "onProviderStreamEvent": lambda event, _m: seen.append(json.loads(json.dumps(event)))},
    ).result()

    assert [event["type"] for event in seen] == ["message_start", "message_delta", "message_stop"]
