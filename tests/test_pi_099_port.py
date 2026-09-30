"""The pi 0.87.1 -> 0.99.1 port (2026-09-30). Each test is an upstream test case from
`packages/ai/test/` or `packages/coding-agent/test/`, named after the file it comes from."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
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
    TextContent,
    ToolCall,
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

    assert payload["top_p"] == 0.5
    assert payload["min_p"] == 0.05


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
            yield {"data": {"choices": [{"delta": {"content": content}}]}}
        yield {"data": {"choices": [{"delta": {}, "finish_reason": "stop"}]}}

    output = _output(model)
    await mistral.consume_chat_stream(model, output, AssistantMessageEventStream(), events())

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


# === B2: coding-agent kernel ===============================================================

# --- session-manager/file-operations.test.ts ----------------------------------------------


def _roles(path) -> list[str]:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    return [row["message"]["role"] if row["type"] == "message" else row["type"] for row in rows]


def test_session_file_is_not_created_for_setup_entries_only(tmp_path):
    from misaka.core.session_manager import SessionManager

    session = SessionManager.create(str(tmp_path), str(tmp_path))
    session.appendModelChange("anthropic", "claude-sonnet-4-5")
    session.appendThinkingLevelChange("off")

    assert not os.path.exists(session.getSessionFile())


# #10000: the first prompt must survive a first turn that never produces an assistant message
def test_session_file_is_created_when_the_first_user_message_is_appended(tmp_path):
    from misaka.core.session_manager import SessionManager

    session = SessionManager.create(str(tmp_path), str(tmp_path))
    session.appendModelChange("anthropic", "claude-sonnet-4-5")
    session.appendMessage(UserMessage(content="first question", timestamp=0))

    file = session.getSessionFile()
    assert _roles(file) == ["session", "model_change", "user"]
    assert len(SessionManager.open(file, str(tmp_path)).buildSessionContext().messages) == 1


def test_session_file_appends_later_entries_without_rewriting(tmp_path):
    from misaka.core.session_manager import SessionManager

    session = SessionManager.create(str(tmp_path), str(tmp_path))
    session.appendMessage(UserMessage(content="first question", timestamp=0))
    session.appendCustomEntry("preset-state", {"name": "plan"})
    session.appendMessage(_output(_model("openai-completions")).model_copy(update={"content": []}))

    assert _roles(session.getSessionFile()) == ["session", "user", "custom", "assistant"]
    assert session.getEntryCount() == 3


# --- default-tools-setting.test.ts / settings-manager.test.ts ------------------------------


def _settings(global_settings: dict, project_settings: dict | None = None):
    from misaka.core.settings_manager import InMemorySettingsStorage, SettingsManager

    storage = InMemorySettingsStorage()
    storage.global_value = json.dumps(global_settings)
    if project_settings is not None:
        storage.project_value = json.dumps(project_settings)
    return SettingsManager.fromStorage(storage, {"projectTrusted": True})


@pytest.mark.parametrize(
    ("global_tools", "project_tools", "expected"),
    [
        (None, None, None),
        (["+todo", "-write"], None, ["read", "bash", "edit", "office", "grep", "find", "ls", "todo"]),
        (["read", "bash"], ["+edit"], ["read", "bash", "edit"]),
        (["read", "bash"], ["grep"], ["grep"]),
        (["read"], ["-read", "+ls"], ["ls"]),
        ([], None, []),
    ],
)
def test_default_tools_modifiers_merge_across_layers(global_tools, project_tools, expected):
    settings = _settings(
        {} if global_tools is None else {"defaultTools": global_tools},
        None if project_tools is None else {"defaultTools": project_tools},
    )

    assert settings.getDefaultTools() == expected


# --- clipboard-image.test.ts --------------------------------------------------------------


@pytest.fixture
def x11_clipboard(monkeypatch):
    from misaka.utils import clipboard_image

    calls: list[list[str]] = []
    native_reads: list[bool] = []

    class Native:
        async def get_image_binary(self):
            native_reads.append(True)

    monkeypatch.setattr(clipboard_image, "_get_native_clipboard", lambda: Native())
    monkeypatch.setattr(clipboard_image, "is_wsl", lambda env=None: False)

    def install(answer):
        def run_command(command, args, *, timeout_ms=None, env=None):
            calls.append(args)
            return answer(args)

        monkeypatch.setattr(clipboard_image, "run_command", run_command)

    return clipboard_image, install, calls, native_reads


# Regression test for #9786.
async def test_x11_does_not_probe_image_types_when_targets_fails(x11_clipboard):
    clipboard_image, install, calls, native_reads = x11_clipboard
    command_result = clipboard_image._CommandResult
    install(lambda args: command_result(stdout=b"", ok=False) if "TARGETS" in args else command_result(stdout=b"hello", ok=True))

    assert await clipboard_image.read_clipboard_image({"platform": "linux", "env": {"DISPLAY": ":0"}}) is None
    assert calls == [["-selection", "clipboard", "-t", "TARGETS", "-o"]]
    assert native_reads == [True]


async def test_x11_does_not_probe_unadvertised_image_types(x11_clipboard):
    clipboard_image, install, calls, native_reads = x11_clipboard
    command_result = clipboard_image._CommandResult

    def answer(args):
        if "TARGETS" in args:
            return command_result(stdout=b"image/png\n", ok=True)
        if "image/png" in args:
            return command_result(stdout=b"", ok=False)
        return command_result(stdout=b"hello", ok=True)

    install(answer)

    assert await clipboard_image.read_clipboard_image({"platform": "linux", "env": {"DISPLAY": ":0"}}) is None
    assert [args[3] for args in calls] == ["TARGETS", "image/png"]
    assert native_reads == [True]


async def test_an_empty_wayland_clipboard_does_not_fall_through_to_x11(x11_clipboard):
    clipboard_image, install, calls, native_reads = x11_clipboard
    command_result = clipboard_image._CommandResult
    install(lambda args: command_result(stdout=b"text/plain\n", ok=True))

    assert await clipboard_image.read_clipboard_image({"platform": "linux", "env": {"WAYLAND_DISPLAY": "w"}}) is None
    assert calls == [["--list-types"]]
    assert native_reads == []


# --- llama-extension.test.ts --------------------------------------------------------------


async def test_llama_preserves_cached_context_for_unloaded_autoload_presets(monkeypatch):
    from misaka.ai.auth.types import ApiKeyCredential
    from misaka.ai.models_runtime import RefreshModelsContext
    from misaka.extensions.llama import client as llama_client
    from misaka.extensions.llama.provider import createLlamaProvider

    state = {"loaded": True, "args": None}

    async def request(self, path, *, method="GET", body=None, signal=None):
        if path == "/models":
            if state["loaded"]:
                model = {"id": "qwen", "status": {"value": "loaded"}, "source": "preset",
                         "meta": {"n_ctx": 65536, "n_ctx_train": 128000}}
            else:
                status = {"value": "unloaded", **({"args": state["args"]} if state["args"] else {})}
                model = {"id": "qwen", "status": status, "source": "preset", "meta": {"n_ctx_train": 128000}}
            return {"data": [model]}
        if path == "/props?model=qwen&autoload=false":
            return {}
        if path == "/props":
            return {"role": "router", "models_autoload": True}
        raise AssertionError(path)

    monkeypatch.setattr(llama_client.LlamaClient, "_request", request)
    stored: dict = {}

    async def publish(publication):
        if publication.persist_is_set:
            stored["entry"] = publication.persist
        if publication.update:
            publication.update()
        return True

    credential = ApiKeyCredential(key="local", env={"LLAMA_BASE_URL": "http://127.0.0.1:9"})

    def context():
        return RefreshModelsContext(
            credential=credential, stored=stored.get("entry"), publish=publish, allowNetwork=True, signal=None
        )

    def windows():
        return [model.contextWindow for model in stored["entry"].models]

    first = createLlamaProvider()
    await first.provider.refreshModels(context())
    assert windows() == [65536]

    state["loaded"] = False
    second = createLlamaProvider()
    await second.provider.refreshModels(context())
    assert [(m.id, m.contextWindow) for m in second.provider.getModels()] == [("qwen", 65536)]
    assert windows() == [65536]

    state["args"] = ["llama-server", "--ctx-size", "32768"]
    await second.provider.refreshModels(context())
    assert windows() == [32768]


async def test_llama_detects_reasoning_from_the_chat_template(monkeypatch):
    from misaka.ai.auth.types import ApiKeyCredential
    from misaka.ai.models_runtime import RefreshModelsContext
    from misaka.extensions.llama import client as llama_client
    from misaka.extensions.llama.provider import createLlamaProvider

    async def request(self, path, *, method="GET", body=None, signal=None):
        if path == "/models":
            return {"data": [{"id": "qwen", "status": {"value": "loaded"}, "meta": {"n_ctx": 4096}}]}
        if path == "/props?model=qwen&autoload=false":
            return {"chat_template": "{% if enable_thinking %}...{% endif %}"}
        raise AssertionError(path)

    monkeypatch.setattr(llama_client.LlamaClient, "_request", request)

    async def publish(publication):
        if publication.update:
            publication.update()
        return True

    controller = createLlamaProvider()
    await controller.provider.refreshModels(RefreshModelsContext(
        credential=ApiKeyCredential(key="local", env={"LLAMA_BASE_URL": "http://127.0.0.1:9"}),
        stored=None, publish=publish, allowNetwork=True, signal=None,
    ))

    [model] = controller.provider.getModels()
    assert model.reasoning and model.thinkingLevelMap["medium"] == "medium"
    assert model.compat.thinkingFormat == "qwen-chat-template"


# --- footer-width.test.ts -----------------------------------------------------------------


def test_footer_updates_cached_usage_totals_after_an_entry_is_appended():
    from types import SimpleNamespace

    from misaka.ui.tui.interactive.components.footer import FooterComponent

    usage = {"input": 10, "output": 1, "cacheRead": 0, "cacheWrite": 0, "cost": {"total": 0.5}}
    entries: list = [{"type": "message", "message": {"role": "assistant", "usage": usage}}]
    manager = SimpleNamespace(
        getEntries=lambda: entries,
        getEntryCount=lambda: len(entries),
        getSessionId=lambda: "test-session",
        getLeafId=lambda: None,
        getCwd=lambda: "/tmp",
        getSessionName=lambda: None,
    )
    session = SimpleNamespace(
        sessionManager=manager,
        model=None,
        state=SimpleNamespace(model=None, thinkingLevel="off"),
        getContextUsage=lambda: None,
        modelRegistry=SimpleNamespace(isUsingSubscription=lambda _model: False),
    )
    footer = FooterComponent(session, SimpleNamespace(getGitBranch=lambda: None, getExtensionStatuses=dict,
                                                      getAvailableProviderCount=lambda: 1))

    assert footer.getSessionStats().usageTotals.cost == 0.5
    assert footer.getSessionStats() is footer.getSessionStats()
    entries.append({"type": "message", "message": {"role": "assistant", "usage": usage}})
    entries.append({"type": "usage", "usage": usage})
    assert footer.getSessionStats().usageTotals.cost == 1.5


# --- utils/shell.ts sanitizeBinaryOutput --------------------------------------------------


def test_sanitize_binary_output_keeps_tabs_newlines_and_astral_characters():
    from misaka.utils.shell import sanitize_binary_output

    assert sanitize_binary_output("a\x00b\tc\nd\re\x1b[0m￺\U0001f600\x7f") == "ab\tc\nd\re[0m\U0001f600\x7f"


# --- agent-session.ts prompt dispositions -------------------------------------------------


def test_prompt_disposition_types_match_upstream():
    import typing

    from misaka.core.agent_session import PromptDisposition, QueuedInputDisposition

    assert set(typing.get_args(QueuedInputDisposition)) == {"handled", "queued"}
    assert set(typing.get_args(PromptDisposition)) == {"handled", "queued", "started"}


# === B1b: ai/ layer aligned file by file with 0.99.1 =========================================

_ANTHROPIC_SSE = (
    'event: message_start\ndata: {"type":"message_start","message":{"id":"msg_1","model":"MODEL",'
    '"usage":{"input_tokens":3,"output_tokens":0}}}\n\n'
    'event: content_block_start\ndata: {"type":"content_block_start","index":0,'
    '"content_block":{"type":"text","text":""}}\n\n'
    'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,'
    '"delta":{"type":"text_delta","text":"hi"}}\n\n'
    'event: content_block_stop\ndata: {"type":"content_block_stop","index":0}\n\n'
    'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn"},'
    '"usage":{"output_tokens":1}}\n\n'
    'event: message_stop\ndata: {"type":"message_stop"}\n\n'
)


@pytest.fixture
def anthropic_wire(monkeypatch):
    """Real SDK client, fake HTTP: every request the adapter sends is recorded."""
    requests: list[httpx.Request] = []
    responses: list[httpx.Response] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if responses:
            return responses.pop(0)
        body = _ANTHROPIC_SSE.replace("MODEL", json.loads(request.content)["model"])
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

    original = anthropic.create_client

    def create_client(*args, **kwargs):
        client, is_oauth, request_headers = original(*args, **kwargs)
        client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url=client.base_url)
        return client, is_oauth, request_headers

    monkeypatch.setattr(anthropic, "create_client", create_client)
    return requests, responses


async def test_anthropic_fallback_models_go_through_the_beta_messages_api(anthropic_wire):
    requests, _ = anthropic_wire
    model = get_model("anthropic", "claude-fable-5")

    result = await anthropic.stream_anthropic(model, normalize_context(CONTEXT), {"apiKey": "sk-ant-test"}).result()

    assert result.stopReason == "stop", result.errorMessage
    [request] = requests
    body = json.loads(request.content)
    assert [fallback["model"] for fallback in body["fallbacks"]] == [
        fallback.model for fallback in model.compat.allowedFallbackModels
    ]
    assert anthropic.SERVER_SIDE_FALLBACK_BETA in request.headers["anthropic-beta"]
    assert request.headers["x-api-key"] == "sk-ant-test"


async def test_anthropic_sdk_does_not_retry_on_its_own(anthropic_wire):
    requests, responses = anthropic_wire
    responses.append(httpx.Response(500, json={"type": "error", "error": {"type": "api_error", "message": "boom"}}))

    result = await anthropic.stream_anthropic(
        get_model("anthropic", "claude-haiku-4-5"), normalize_context(CONTEXT), {"apiKey": "sk-ant-test"}
    ).result()

    assert result.stopReason == "error"
    assert len(requests) == 1


async def test_anthropic_header_owned_auth_honours_null_headers(anthropic_wire, monkeypatch):
    requests, _ = anthropic_wire
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-ambient")
    headers = {"cf-aig-authorization": "Bearer gateway", "Authorization": None, "x-api-key": None}

    result = await anthropic.stream_anthropic(
        get_model("anthropic", "claude-haiku-4-5"), normalize_context(CONTEXT), {"headers": headers}
    ).result()

    assert result.stopReason == "stop", result.errorMessage
    [request] = requests
    assert request.headers["cf-aig-authorization"] == "Bearer gateway"
    assert "x-api-key" not in request.headers and "authorization" not in request.headers


def test_anthropic_refuses_a_request_without_any_auth():
    with pytest.raises(RuntimeError, match="No API key for provider: anthropic"):
        anthropic.stream_simple_anthropic(get_model("anthropic", "claude-haiku-4-5"), normalize_context(CONTEXT))


def test_anthropic_configured_beta_header_replaces_the_computed_betas():
    model = get_model("anthropic", "claude-fable-5")

    configured = anthropic.build_params(model, normalize_context(CONTEXT), False, {"headers": {"anthropic-beta": "a, b,a"}})
    suppressed = anthropic.build_params(model, normalize_context(CONTEXT), False, {"headers": {"Anthropic-Beta": None}})

    assert configured["betas"] == ["a", "b"]
    assert "betas" not in suppressed


def test_anthropic_tool_call_ids_keep_only_ascii_word_characters():
    assert anthropic.normalize_tool_call_id("call|é1-x") == "call__1-x"


async def test_stream_facade_injects_the_environment_api_key(anthropic_wire, monkeypatch):
    requests, _ = anthropic_wire
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-from-env")

    result = await complete_simple(get_model("anthropic", "claude-haiku-4-5"), CONTEXT)

    assert result.stopReason == "stop", result.errorMessage
    assert requests[0].headers["x-api-key"] == "sk-ant-from-env"


def test_env_api_key_reads_scoped_env_first(monkeypatch):
    from misaka.ai.env_api_keys import find_env_keys, get_env_api_key

    monkeypatch.setenv("NVIDIA_API_KEY", "ambient")
    assert get_env_api_key("nvidia") == "ambient"
    assert get_env_api_key("nvidia", {"NVIDIA_API_KEY": "scoped"}) == "scoped"
    assert find_env_keys("typesafe", {"TYPESAFE_API_KEY": "k"}) == ["TYPESAFE_API_KEY"]
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "token")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_OAUTH_TOKEN", raising=False)
    assert get_env_api_key("anthropic") is None


# --- B1b-2: OpenAI completions / responses / azure --------------------------------------------


def _openai_wire(monkeypatch, module, sse_body: str):
    import httpx2  # the OpenAI SDK's own HTTP client

    requests: list = []

    def handler(request):
        requests.append(request)
        return httpx2.Response(200, text=sse_body, headers={"content-type": "text/event-stream"})

    original = module.create_client

    def create_client(*args, **kwargs):
        client, request_headers = original(*args, **kwargs)
        client._client = httpx2.AsyncClient(transport=httpx2.MockTransport(handler), base_url=str(client.base_url))
        return client, request_headers

    monkeypatch.setattr(module, "create_client", create_client)
    return requests


_COMPLETIONS_SSE = (
    'data: {"id":"c1","object":"chat.completion.chunk","created":0,"model":"m","choices":[{"index":0,'
    '"delta":{"content":"hi"},"finish_reason":null}]}\n\n'
    'data: {"id":"c1","object":"chat.completion.chunk","created":0,"model":"m","choices":[{"index":0,'
    '"delta":{},"finish_reason":"stop"}]}\n\n'
    "data: [DONE]\n\n"
)


async def test_completions_sends_the_request_body_with_upstream_top_level_fields(monkeypatch):
    requests = _openai_wire(monkeypatch, openai_completions, _COMPLETIONS_SSE)
    model = _model(
        "openai-completions",
        reasoning=True,
        compat={"thinkingFormat": "qwen-chat-template"},
        samplingParams={"top_k": 20},
    )

    result = await openai_completions.stream_openai_completions(
        model,
        normalize_context(CONTEXT),
        {"apiKey": "k", "reasoningEffort": "high", "headers": {"Authorization": None, "x-extra": "1"}},
    ).result()

    assert result.stopReason == "stop", result.errorMessage
    assert result.rawStopReason == "stop"
    [request] = requests
    body = json.loads(request.content)
    assert body["chat_template_kwargs"] == {"enable_thinking": True, "preserve_thinking": True}
    assert body["top_k"] == 20
    assert None not in body.values()
    assert "authorization" not in request.headers and request.headers["x-extra"] == "1"


async def test_completions_header_owned_auth_uses_the_unused_key(monkeypatch):
    requests = _openai_wire(monkeypatch, openai_completions, _COMPLETIONS_SSE)

    result = await openai_completions.stream_openai_completions(
        _model("openai-completions"), normalize_context(CONTEXT), {"headers": {"cf-aig-authorization": "Bearer gw"}}
    ).result()

    assert result.stopReason == "stop", result.errorMessage
    assert requests[0].headers["cf-aig-authorization"] == "Bearer gw"
    with pytest.raises(RuntimeError, match="No API key for provider: fixture"):
        openai_completions.stream_simple_openai_completions(_model("openai-completions"), normalize_context(CONTEXT))


def test_completions_usage_reads_kimi_top_level_cached_tokens_and_reasoning():
    usage = openai_completions.parse_chunk_usage(
        {"prompt_tokens": 100, "completion_tokens": 20, "cached_tokens": 30,
         "completion_tokens_details": {"reasoning_tokens": 7}},
        _model("openai-completions"),
    )

    assert (usage.input, usage.cacheRead, usage.output, usage.reasoning) == (70, 30, 20, 7)


def test_completions_replays_pipe_ids_with_their_item_ids():
    from misaka.ai.types import ToolResultMessage

    model = _model("openai-completions", provider="openai")
    source = _output(_model("openai-responses", provider="openai")).model_copy(update={
        "content": [
            ToolCall(id="call_1|fc_a", name="t", arguments={}),
            ToolCall(id="call_1|fc_b", name="t", arguments={}),
            ToolCall(id="call_2|" + "x" * 60, name="t", arguments={}),
        ],
        "stopReason": "toolUse",
    })
    results = [
        ToolResultMessage(toolCallId=call.id, toolName="t", content=[], isError=False, timestamp=0)
        for call in source.content
    ]
    messages = openai_completions.convert_messages(
        model, normalize_context(Context(messages=[UserMessage(content="q", timestamp=0), source, *results])),
        openai_completions.get_compat(model),
    )

    ids = [call["id"] for call in messages[1]["tool_calls"]]
    assert ids[:2] == ["call_1_fc_a", "call_1_fc_b"]
    assert len(ids[2]) <= 40 and ids[2].startswith("call_2_")
    assert [m["content"] for m in messages[2:]] == ["(no tool output)"] * 3


def test_completions_cache_breakpoint_lands_on_a_trailing_tool_result():
    from misaka.ai.types import ToolResultMessage

    model = _model("openai-completions", provider="openrouter", id="anthropic/claude-x", baseUrl="https://openrouter.ai/api/v1")
    call = ToolCall(id="call_1", name="t", arguments={})
    source = _output(model).model_copy(update={"content": [call], "stopReason": "toolUse"})
    context = Context(messages=[
        UserMessage(content="q", timestamp=0),
        source,
        ToolResultMessage(toolCallId="call_1", toolName="t", content=[TextContent(text="r")], isError=False, timestamp=0),
    ])

    params = openai_completions.build_params(model, normalize_context(context), {"cacheRetention": "short"})

    assert params["messages"][-1]["role"] == "tool"
    assert params["messages"][-1]["content"][0]["cache_control"] == {"type": "ephemeral"}


def test_completions_detects_ant_ling():
    compat = openai_completions.detect_compat(_model("openai-completions", provider="ant-ling"))

    assert (compat["thinkingFormat"], compat["maxTokensField"], compat["supportsReasoningEffort"]) == (
        "ant-ling", "max_tokens", False,
    )


async def test_completions_stream_without_finish_reason_is_an_error(monkeypatch):
    async def chunks():
        yield {"id": "c1", "choices": [{"index": 0, "delta": {"content": "half"}}]}

    monkeypatch.setattr(openai_completions, "_create_completion_stream", AsyncMock(return_value=chunks()))
    result = await openai_completions.stream_openai_completions(
        _model("openai-completions"), normalize_context(CONTEXT), {"client": object()}
    ).result()

    assert result.stopReason == "error"
    assert result.errorMessage == "Stream ended without finish_reason"


def test_responses_fallback_message_ids_and_empty_tool_output():
    from misaka.ai.providers.openai_responses_shared import convert_responses_messages
    from misaka.ai.types import ToolResultMessage

    model = _responses_model()
    assistant = _output(model).model_copy(update={
        "content": [TextContent(text="one"), TextContent(text="two"), ToolCall(id="call_1", name="t", arguments={})],
        "stopReason": "toolUse",
    })
    context = Context(messages=[
        UserMessage(content="q", timestamp=0),
        assistant,
        ToolResultMessage(toolCallId="call_1", toolName="t", content=[], isError=False, timestamp=0),
    ])

    items = convert_responses_messages(model, normalize_context(context), {"openai"})

    assert [item.get("id") for item in items if item.get("type") == "message"] == ["msg_pi_1", "msg_pi_1_1"]
    function_call = next(item for item in items if item.get("type") == "function_call")
    assert "id" not in function_call
    assert items[-1]["output"] == "(no tool output)"


def test_responses_requires_a_key_or_an_auth_header():
    with pytest.raises(RuntimeError, match="No API key for provider: openai"):
        openai_responses.stream_simple_openai_responses(_responses_model(), normalize_context(CONTEXT))
    assert openai_responses.get_client_api_key("openai", None, {"authorization": "Bearer x"}) == "unused"


def test_azure_reads_its_configuration_from_the_request_env():
    from misaka.ai.providers import azure_openai_responses as azure

    config = azure.resolve_azure_config(
        _model("azure-openai-responses", baseUrl=""),
        {"env": {"AZURE_OPENAI_RESOURCE_NAME": "res", "AZURE_OPENAI_API_VERSION": "2025-01-01"}},
    )

    assert config == {"baseUrl": "https://res.openai.azure.com/openai/v1", "apiVersion": "2025-01-01"}
    assert azure.resolve_deployment_name(
        _model("azure-openai-responses", id="gpt"), {"env": {"AZURE_OPENAI_DEPLOYMENT_NAME_MAP": "gpt=dep"}}
    ) == "dep"


# --- B1b-3: OpenAI Codex ------------------------------------------------------------------


def _codex_token(account_id: str = "acct_1") -> str:
    import base64

    payload = base64.urlsafe_b64encode(
        json.dumps({"https://api.openai.com/auth": {"chatgpt_account_id": account_id}}).encode()
    ).decode().rstrip("=")
    return f"header.{payload}.sig"


def _codex_model() -> Model:
    return _model(
        "openai-codex-responses", provider="openai-codex", baseUrl="https://chatgpt.com/backend-api", reasoning=True
    )


def _codex_sse(*events: dict) -> str:
    return "".join(f"data: {json.dumps(event)}\n\n" for event in events)


@pytest.fixture
def codex_wire(monkeypatch):
    from misaka.ai.providers import openai_codex_responses as codex

    requests: list[httpx.Request] = []
    responses: list[httpx.Response] = []
    real_client = httpx.AsyncClient

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return responses.pop(0)

    monkeypatch.setattr(
        codex.httpx, "AsyncClient", lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs)
    )
    return codex, requests, responses


async def test_codex_incomplete_response_is_an_error_with_its_reason(codex_wire):
    codex, requests, responses = codex_wire
    responses.append(httpx.Response(200, text=_codex_sse(
        {"type": "response.created", "response": {"id": "resp_1"}},
        {"type": "response.incomplete", "response": {
            "id": "resp_1", "status": "incomplete", "incomplete_details": {"reason": "content_filter"},
            "end_turn": True,
        }},
    ), headers={"content-type": "text/event-stream"}))

    result = await codex.stream_openai_codex_responses(
        _codex_model(), normalize_context(CONTEXT),
        {"apiKey": _codex_token(), "transport": "sse", "sessionId": "session-1", "toolChoice": "none"},
    ).result()

    assert result.stopReason == "error"
    assert result.errorMessage == "Response incomplete: content_filter"
    assert result.endTurn is True
    [request] = requests
    assert request.headers["session-id"] == "session-1"
    assert request.headers["chatgpt-account-id"] == "acct_1"
    assert "session_id" not in request.headers
    body = json.loads(request.content) if request.headers.get("content-encoding") != "zstd" else None
    if body is not None:
        assert body["tool_choice"] == "none" and body["prompt_cache_key"] == "session-1"


async def test_codex_does_not_retry_by_default_and_caps_retry_after(codex_wire):
    codex, requests, responses = codex_wire
    responses.append(httpx.Response(500, json={"error": {"message": "boom"}}))

    result = await codex.stream_openai_codex_responses(
        _codex_model(), normalize_context(CONTEXT), {"apiKey": _codex_token(), "transport": "sse"}
    ).result()

    assert result.stopReason == "error" and len(requests) == 1
    assert "boom" in result.errorMessage

    responses.append(httpx.Response(429, json={"error": {"message": "slow down"}}, headers={"retry-after": "120"}))
    result = await codex.stream_openai_codex_responses(
        _codex_model(), normalize_context(CONTEXT),
        {"apiKey": _codex_token(), "transport": "sse", "maxRetries": 2, "maxRetryDelayMs": 1000},
    ).result()

    assert result.stopReason == "error"
    assert result.errorMessage == "Server requested 120s retry delay (max: 1s)"


async def test_codex_error_events_read_the_nested_code():
    from misaka.ai.providers import openai_codex_responses as codex

    async def events():
        yield {"type": "error", "error": {"code": "previous_response_not_found", "message": "gone"}}

    with pytest.raises(codex.CodexApiError) as caught:
        async for _event in codex.map_codex_events(events(), _output(_codex_model()), _codex_model()):
            pass

    assert caught.value.code == "previous_response_not_found"
    assert str(caught.value) == "Codex error: gone"
    assert codex.is_previous_response_not_found_error(caught.value)


def test_codex_terminal_rate_limits_are_not_retried():
    from misaka.ai.providers import openai_codex_responses as codex

    assert codex.is_retryable_error(429, "rate limit") is True
    assert codex.is_retryable_error(429, "insufficient_quota: add billing") is False


async def test_codex_sse_parses_a_final_frame_without_a_blank_line():
    from misaka.ai.providers import openai_codex_responses as codex

    response = httpx.Response(200, text='data: {"type":"a"}\n\ndata: {"type":"b"}')

    assert [event["type"] async for event in codex.parse_sse(response)] == ["a", "b"]


def test_codex_requires_an_api_key_and_honours_null_headers():
    from misaka.ai.providers import openai_codex_responses as codex

    with pytest.raises(RuntimeError, match="No API key for provider: openai-codex"):
        codex.stream_simple_openai_codex_responses(_codex_model(), normalize_context(CONTEXT))
    headers = codex.build_sse_headers({"X-Model": "m"}, {"x-model": None, "X-Extra": "1"}, "acct", "tok", "sid")
    assert {name.lower() for name in headers} >= {"authorization", "session-id", "x-client-request-id", "x-extra"}
    assert "x-model" not in {name.lower() for name in headers}


# --- B1b-4: Google / Vertex / google_shared -----------------------------------------------


def _gemini_model(api: str = "google-generative-ai", **overrides) -> Model:
    return _model(
        api,
        **{
            "id": "gemini-2.5-flash",
            "provider": "google" if api == "google-generative-ai" else "google-vertex",
            "baseUrl": "https://generativelanguage.googleapis.com/v1beta",
            "reasoning": True,
            **overrides,
        },
    )


@pytest.fixture
def gemini_wire(monkeypatch):
    from misaka.ai.providers import google

    requests: list[httpx.Request] = []
    chunks: list[dict] = []
    real_client = google.GoogleGenAI

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        body = "".join(f"data: {json.dumps(chunk)}\r\n\r\n" for chunk in chunks)
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

    def client(**kwargs):
        transport = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        return real_client(**{**kwargs, "http_options": {**(kwargs["http_options"] or {}), "httpx_async_client": transport}})

    monkeypatch.setattr(google, "GoogleGenAI", client)
    return google, requests, chunks


async def test_google_error_finish_reason_reports_the_raw_reason_and_upstream_usage(gemini_wire):
    google, requests, chunks = gemini_wire
    chunks += [
        {"candidates": [{"content": {"role": "model", "parts": [{"text": "hel"}]}}], "responseId": "r1"},
        {
            "candidates": [{"content": {"role": "model", "parts": [{"text": "lo"}]}, "finishReason": "SAFETY"}],
            "usageMetadata": {
                "promptTokenCount": 100, "cachedContentTokenCount": 30, "candidatesTokenCount": 20,
                "thoughtsTokenCount": 5, "totalTokenCount": 125,
            },
        },
    ]

    result = await google.stream_google(
        _gemini_model(), normalize_context(CONTEXT), {"apiKey": "k", "headers": {"X-Extra": None}}
    ).result()

    assert (result.stopReason, result.rawStopReason) == ("error", "SAFETY")
    assert result.errorMessage == "Provider stopped with: SAFETY"
    assert (result.usage.input, result.usage.output, result.usage.cacheRead, result.usage.reasoning) == (70, 25, 30, 5)
    assert result.responseId == "r1"
    [request] = requests
    assert "x-extra" not in request.headers and "toolConfig" not in json.loads(request.content)


async def test_google_stream_without_finish_reason_is_an_error(gemini_wire):
    google, _requests, chunks = gemini_wire
    chunks.append({"candidates": [{"content": {"role": "model", "parts": [{"text": "half"}]}}]})

    result = await google.stream_google(_gemini_model(), normalize_context(CONTEXT), {"apiKey": "k"}).result()

    assert result.stopReason == "error"
    assert result.errorMessage == "Google stream ended without a finish reason"


async def test_google_tool_calls_turn_only_a_stop_into_tool_use(gemini_wire):
    google, _requests, chunks = gemini_wire
    call = {"functionCall": {"name": "t", "args": {"a": 1}}}
    chunks.append({"candidates": [{"content": {"role": "model", "parts": [call]}, "finishReason": "MAX_TOKENS"}]})

    result = await google.stream_google(_gemini_model(), normalize_context(CONTEXT), {"apiKey": "k"}).result()

    assert result.stopReason == "length" and result.content[0].arguments == {"a": 1}


async def test_google_requires_the_api_key_in_options(monkeypatch):
    from misaka.ai.providers import google

    monkeypatch.setenv("GEMINI_API_KEY", "from-env")
    result = await google.stream_google(_gemini_model(), normalize_context(CONTEXT)).result()

    assert result.errorMessage == "No API key for provider: google"
    with pytest.raises(RuntimeError, match="No API key for provider: google"):
        google.stream_simple_google(_gemini_model(), normalize_context(CONTEXT))


def test_google_simple_options_carry_tool_choice(monkeypatch):
    from misaka.ai.providers import google
    from misaka.ai.types import Tool

    seen: dict = {}
    monkeypatch.setattr(google, "stream_google", lambda _model, _context, options: seen.update(options))
    tool = Tool(name="t", description="d", parameters={"type": "object", "properties": {}})
    context = normalize_context(Context(messages=[UserMessage(content="q", timestamp=0)], tools=[tool]))

    google.stream_simple_google(_gemini_model(), context, SimpleStreamOptions(apiKey="k", toolChoice="none"))

    assert seen["toolChoice"] == "none"
    params = google.build_params(_gemini_model(), context, seen)
    assert params["config"]["toolConfig"] == {"functionCallingConfig": {"mode": "NONE"}}
    assert "toolConfig" not in google.build_params(_gemini_model(), context, {})["config"]


def test_gemini_replay_keeps_empty_blocks_that_carry_a_signature():
    from misaka.ai.providers.google_shared import convert_messages
    from misaka.ai.types import ThinkingContent, ToolResultMessage

    model = _gemini_model(id="gemini-3-pro-preview")
    same = _output(model).model_copy(update={"content": [
        TextContent(text="", textSignature="c2ln"),
        ThinkingContent(thinking=" ", thinkingSignature="c2ln"),
        TextContent(text="", textSignature=None),
    ]})
    other = _output(_gemini_model(id="gemini-2.5-pro")).model_copy(update={"content": [
        ThinkingContent(thinking="", thinkingSignature="c2ln"),
        TextContent(text="kept"),
        ToolCall(id="call:é1", name="t", arguments={}),
    ], "stopReason": "toolUse"})
    context = Context(messages=[
        UserMessage(content="q", timestamp=0), same, UserMessage(content="r", timestamp=0), other,
        ToolResultMessage(toolCallId="call:é1", toolName="t", content=[], isError=False, timestamp=0),
    ])

    contents = convert_messages(model, normalize_context(context))

    assert contents[1]["parts"] == [
        {"text": "", "thoughtSignature": "c2ln"},
        {"thought": True, "text": " ", "thoughtSignature": "c2ln"},
    ]
    assert contents[3]["parts"][0] == {"text": "kept"}
    assert contents[3]["parts"][1]["functionCall"]["id"] == "call__1"
    assert contents[4]["parts"][0]["functionResponse"]["id"] == "call__1"


def test_vertex_reads_its_setup_from_the_request_env(tmp_path):
    from misaka.ai.providers import google_vertex as vertex

    key_file = tmp_path / "adc.json"
    key_file.write_text(json.dumps(
        {"type": "authorized_user", "client_id": "x", "client_secret": "y", "refresh_token": "z"}
    ))
    env = {"GOOGLE_CLOUD_PROJECT": "proj", "GOOGLE_CLOUD_LOCATION": "europe-west4",
           "GOOGLE_APPLICATION_CREDENTIALS": str(key_file)}
    options = {"env": env, "apiKey": "<authenticated>"}
    model = _gemini_model("google-vertex", id="gemini-2.5-flash-lite",
                          baseUrl="https://{location}-aiplatform.googleapis.com")

    assert vertex.resolve_api_key(options) is None
    assert vertex.resolve_api_key({"apiKey": "gcp-vertex-credentials"}) is None
    assert (vertex.resolve_project(options), vertex.resolve_location(options)) == ("proj", "europe-west4")
    client = vertex.create_client(model, "proj", "europe-west4", None, env)
    assert client._api_client._credentials is not None
    assert client._api_client._http_options.api_version == "v1"
    assert vertex.get_google_budget(model, "minimal") == 128


def test_provider_headers_merge_case_insensitively_and_null_removes():
    from misaka.ai.utils.headers import provider_headers_to_record

    assert provider_headers_to_record(
        {"User-Agent": "a", "X-Gone": "1"}, {"user-agent": "b", "x-gone": None}, None
    ) == {"user-agent": "b"}
    assert provider_headers_to_record({"X": None}) is None


# --- B1b-5: Mistral over raw HTTP ---------------------------------------------------------


def _mistral_model(**overrides) -> Model:
    return _model("mistral-conversations", **{"provider": "mistral", "baseUrl": "https://api.mistral.ai", **overrides})


@pytest.fixture
def mistral_wire(monkeypatch):
    requests: list[httpx.Request] = []
    responses: list[httpx.Response] = []
    real_client = httpx.AsyncClient

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return responses.pop(0)

    monkeypatch.setattr(
        mistral.httpx, "AsyncClient", lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs)
    )
    return requests, responses


def _mistral_sse(*chunks: dict, done: bool = True, separator: str = "\r\n\r\n") -> str:
    frames = [f"data: {json.dumps(chunk)}" for chunk in chunks] + (["data: [DONE]"] if done else [])
    return separator.join(frames)


async def test_mistral_posts_the_wire_payload_and_parses_the_event_stream(mistral_wire):
    from misaka.ai.types import ToolResultMessage

    requests, responses = mistral_wire
    call = {"index": 0, "id": "abcDEF123", "function": {"name": "t", "arguments": '{"a":'}}
    responses.append(httpx.Response(200, text=_mistral_sse(
        {"id": "r1", "choices": [{"delta": {"content": "hi"}}]},
        {"choices": [{"delta": {"tool_calls": [call]}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"name": "t", "arguments": "1}"}}]}}]},
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}],
         "usage": {"prompt_tokens": 10, "completion_tokens": 4, "prompt_tokens_details": {"cached_tokens": 3}}},
        done=False, separator="\n\n",
    ), headers={"content-type": "text/event-stream"}))
    seen: list[dict] = []
    history = _output(_mistral_model()).model_copy(update={
        "content": [ToolCall(id="call_1", name="t", arguments={})], "stopReason": "toolUse",
    })
    context = Context(messages=[
        UserMessage(content="q", timestamp=0), history,
        ToolResultMessage(toolCallId="call_1", toolName="t", content=[], isError=False, timestamp=0),
    ])

    result = await mistral.stream_mistral(_mistral_model(), normalize_context(context), {
        "apiKey": "k", "sessionId": "s1", "maxTokens": 5, "headers": {"X-Drop": None},
        "toolChoice": {"type": "function", "function": {"name": "t"}},
        "onResponse": lambda response, _model: seen.append(response),
    }).result()

    assert result.stopReason == "toolUse", result.errorMessage
    assert result.rawStopReason == "tool_calls" and result.responseId == "r1"
    assert [block.type for block in result.content] == ["text", "toolCall"]
    assert result.content[1].arguments == {"a": 1}
    assert (result.usage.input, result.usage.cacheRead, result.usage.output) == (7, 3, 4)
    [request] = requests
    assert str(request.url) == "https://api.mistral.ai/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer k" and request.headers["x-affinity"] == "s1"
    assert "x-drop" not in request.headers
    body = json.loads(request.content)
    assert (body["max_tokens"], body["prompt_cache_key"]) == (5, "s1")
    assert body["tool_choice"] == {"type": "function", "function": {"name": "t"}}
    assert body["messages"][1]["tool_calls"][0]["index"] == 0 and body["messages"][1]["prefix"] is False
    assert body["messages"][2]["tool_call_id"] == body["messages"][1]["tool_calls"][0]["id"]
    assert seen[0]["status"] == 200


async def test_mistral_reports_http_errors_and_missing_finish_reasons(mistral_wire):
    _requests, responses = mistral_wire
    responses.append(httpx.Response(400, text='{"message":"bad"}'))
    result = await mistral.stream_mistral(_mistral_model(), normalize_context(CONTEXT), {"apiKey": "k"}).result()
    assert result.errorMessage == 'Mistral API error (400): {"message":"bad"}'

    responses.append(httpx.Response(200, text=_mistral_sse({"choices": [{"delta": {"content": "half"}}]})))
    result = await mistral.stream_mistral(_mistral_model(), normalize_context(CONTEXT), {"apiKey": "k"}).result()
    assert result.errorMessage == "Mistral stream ended without a finish reason"

    responses.append(httpx.Response(200, text='data: {"object":"x"}\n\n'))
    result = await mistral.stream_mistral(_mistral_model(), normalize_context(CONTEXT), {"apiKey": "k"}).result()
    assert result.errorMessage == "Invalid Mistral streaming event"


async def test_mistral_timeout_bounds_the_whole_stream(mistral_wire):
    _requests, responses = mistral_wire

    class Hanging(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'data: {"choices":[{"delta":{"content":"a"}}]}\n\n'
            await asyncio.Event().wait()

    responses.append(httpx.Response(200, stream=Hanging()))
    result = await mistral.stream_mistral(
        _mistral_model(), normalize_context(CONTEXT), {"apiKey": "k", "timeoutMs": 50}
    ).result()

    assert (result.stopReason, result.errorMessage) == ("error", "The operation was aborted due to timeout")


def test_mistral_requires_an_api_key_in_options(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "from-env")
    with pytest.raises(RuntimeError, match="No API key for provider: mistral"):
        mistral.stream_simple_mistral(_mistral_model(), normalize_context(CONTEXT))


async def test_abort_reason_of_a_combined_signal_is_the_first_input_that_fired():
    from misaka.ai.utils.abort import (
        AbortController,
        TimeoutAbortSignal,
        abort_reason,
        combine_abort_signals,
    )

    user = AbortController()
    timeout = TimeoutAbortSignal(0)
    combined = combine_abort_signals(user, timeout)
    assert isinstance(abort_reason(combined), TimeoutError)
    user.abort()
    assert str(abort_reason(combined)) == "Request was aborted"
    await asyncio.wait_for(TimeoutAbortSignal(10).wait(), 1)
