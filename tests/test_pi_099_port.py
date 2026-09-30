"""The pi 0.87.1 -> 0.99.1 port (2026-09-30). Each test is an upstream test case from
`packages/ai/test/` or `packages/coding-agent/test/`, named after the file it comes from."""

from __future__ import annotations

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
