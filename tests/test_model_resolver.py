"""Ported from pi's ``packages/coding-agent/test/model-resolver.test.ts`` (v1.1.0). ``console.warn`` spies read
stderr; the registry stand-ins are namespaces with the methods each case calls (``getAvailable`` is awaited)."""

from __future__ import annotations

import shutil
import tempfile
from types import SimpleNamespace
from typing import Any

import pytest

from misaka.agent import Agent
from misaka.ai.compat import getModel, streamSimple
from misaka.ai.providers.all import getBuiltinModels, getBuiltinProviders
from misaka.ai.types import Model
from misaka.core.agent_session import AgentSession, AgentSessionConfig
from misaka.core.auth_storage import AuthStorage
from misaka.core.model_resolver import (
    defaultModelPerProvider,
    findInitialModel,
    parseModelPattern,
    resolveCliModel,
    resolveModelScope,
    resolveModelScopeWithDiagnostics,
)
from misaka.core.session_manager import SessionManager
from misaka.core.settings_manager import SettingsManager
from tests.model_runtime_test_utils import createModelRegistry, getModelRuntime
from tests.suite_harness import createTestResourceLoader


def _model(**fields: Any) -> Model:
    return Model.model_validate(fields)


# Mock models for testing
mockModels = [
    _model(
        id="claude-sonnet-4-5",
        name="Claude Sonnet 4.5",
        api="anthropic-messages",
        provider="anthropic",
        baseUrl="https://api.anthropic.com",
        reasoning=True,
        input=["text", "image"],
        cost={"input": 3, "output": 15, "cacheRead": 0.3, "cacheWrite": 3.75},
        contextWindow=200000,
        maxTokens=8192,
    ),
    _model(
        id="gpt-4o",
        name="GPT-4o",
        api="anthropic-messages",  # Using same type for simplicity
        provider="openai",
        baseUrl="https://api.openai.com",
        reasoning=False,
        input=["text", "image"],
        cost={"input": 5, "output": 15, "cacheRead": 0.5, "cacheWrite": 5},
        contextWindow=128000,
        maxTokens=4096,
    ),
]

# Mock OpenRouter models with colons in IDs
mockOpenRouterModels = [
    _model(
        id="qwen/qwen3-coder:exacto",
        name="Qwen3 Coder Exacto",
        api="anthropic-messages",
        provider="openrouter",
        baseUrl="https://openrouter.ai/api/v1",
        reasoning=True,
        input=["text"],
        cost={"input": 1, "output": 2, "cacheRead": 0.1, "cacheWrite": 1},
        contextWindow=128000,
        maxTokens=8192,
    ),
    _model(
        id="openai/gpt-4o:extended",
        name="GPT-4o Extended",
        api="anthropic-messages",
        provider="openrouter",
        baseUrl="https://openrouter.ai/api/v1",
        reasoning=False,
        input=["text", "image"],
        cost={"input": 5, "output": 15, "cacheRead": 0.5, "cacheWrite": 5},
        contextWindow=128000,
        maxTokens=4096,
    ),
]

allModels = [*mockModels, *mockOpenRouterModels]

THINKING_LEVELS = ["off", "minimal", "low", "medium", "high", "xhigh", "max"]


def _registry(**methods: Any) -> SimpleNamespace:
    return SimpleNamespace(**methods)


def _available(models: list[Model]):
    async def getAvailable(*_args: Any) -> list[Model]:
        return models

    return getAvailable


# parseModelPattern -- simple patterns without colons


def test_exact_match_returns_model_with_undefined_thinking_level():
    result = parseModelPattern("claude-sonnet-4-5", allModels)
    assert result.model.id == "claude-sonnet-4-5"
    assert result.thinkingLevel is None
    assert result.warning is None


def test_partial_match_returns_best_model_with_undefined_thinking_level():
    result = parseModelPattern("sonnet", allModels)
    assert result.model.id == "claude-sonnet-4-5"
    assert result.thinkingLevel is None
    assert result.warning is None


def test_no_match_returns_undefined_model_and_thinking_level():
    result = parseModelPattern("nonexistent", allModels)
    assert result.model is None
    assert result.thinkingLevel is None
    assert result.warning is None


# parseModelPattern -- patterns with valid thinking levels


def test_sonnet_high_returns_sonnet_with_high_thinking_level():
    result = parseModelPattern("sonnet:high", allModels)
    assert result.model.id == "claude-sonnet-4-5"
    assert result.thinkingLevel == "high"
    assert result.warning is None


def test_gpt_4o_medium_returns_gpt_4o_with_medium_thinking_level():
    result = parseModelPattern("gpt-4o:medium", allModels)
    assert result.model.id == "gpt-4o"
    assert result.thinkingLevel == "medium"
    assert result.warning is None


def test_all_valid_thinking_levels_work():
    for level in THINKING_LEVELS:
        result = parseModelPattern(f"sonnet:{level}", allModels)
        assert result.model.id == "claude-sonnet-4-5"
        assert result.thinkingLevel == level
        assert result.warning is None


# parseModelPattern -- patterns with invalid thinking levels


def test_sonnet_random_returns_sonnet_with_undefined_thinking_level_and_warning():
    result = parseModelPattern("sonnet:random", allModels)
    assert result.model.id == "claude-sonnet-4-5"
    assert result.thinkingLevel is None
    assert "Invalid thinking level" in result.warning
    assert "random" in result.warning


def test_gpt_4o_invalid_returns_gpt_4o_with_undefined_thinking_level_and_warning():
    result = parseModelPattern("gpt-4o:invalid", allModels)
    assert result.model.id == "gpt-4o"
    assert result.thinkingLevel is None
    assert "Invalid thinking level" in result.warning


# parseModelPattern -- OpenRouter models with colons in IDs


def test_qwen3_coder_exacto_matches_the_model_with_undefined_thinking_level():
    result = parseModelPattern("qwen/qwen3-coder:exacto", allModels)
    assert result.model.id == "qwen/qwen3-coder:exacto"
    assert result.thinkingLevel is None
    assert result.warning is None


def test_openrouter_qwen_qwen3_coder_exacto_matches_with_provider_prefix():
    result = parseModelPattern("openrouter/qwen/qwen3-coder:exacto", allModels)
    assert result.model.id == "qwen/qwen3-coder:exacto"
    assert result.model.provider == "openrouter"
    assert result.thinkingLevel is None
    assert result.warning is None


def test_qwen3_coder_exacto_high_matches_model_with_high_thinking_level():
    result = parseModelPattern("qwen/qwen3-coder:exacto:high", allModels)
    assert result.model.id == "qwen/qwen3-coder:exacto"
    assert result.thinkingLevel == "high"
    assert result.warning is None


def test_openrouter_qwen_qwen3_coder_exacto_high_matches_with_provider_and_thinking_level():
    result = parseModelPattern("openrouter/qwen/qwen3-coder:exacto:high", allModels)
    assert result.model.id == "qwen/qwen3-coder:exacto"
    assert result.model.provider == "openrouter"
    assert result.thinkingLevel == "high"
    assert result.warning is None


def test_gpt_4o_extended_matches_the_extended_model_with_undefined_thinking_level():
    result = parseModelPattern("openai/gpt-4o:extended", allModels)
    assert result.model.id == "openai/gpt-4o:extended"
    assert result.thinkingLevel is None
    assert result.warning is None


# parseModelPattern -- invalid thinking levels with OpenRouter models


def test_qwen3_coder_exacto_random_returns_model_with_undefined_thinking_level_and_warning():
    result = parseModelPattern("qwen/qwen3-coder:exacto:random", allModels)
    assert result.model.id == "qwen/qwen3-coder:exacto"
    assert result.thinkingLevel is None
    assert "Invalid thinking level" in result.warning
    assert "random" in result.warning


def test_qwen3_coder_exacto_high_random_returns_model_with_undefined_thinking_level_and_warning():
    result = parseModelPattern("qwen/qwen3-coder:exacto:high:random", allModels)
    assert result.model.id == "qwen/qwen3-coder:exacto"
    assert result.thinkingLevel is None
    assert "Invalid thinking level" in result.warning
    assert "random" in result.warning


# parseModelPattern -- edge cases


def test_empty_pattern_matches_via_partial_matching():
    # Empty string is included in all model IDs, so partial matching finds a match
    result = parseModelPattern("", allModels)
    assert result.model is not None
    assert result.thinkingLevel is None


def test_pattern_ending_with_colon_treats_empty_suffix_as_invalid():
    result = parseModelPattern("sonnet:", allModels)
    # Empty string after colon is not a valid thinking level
    # So it tries to match "sonnet:" which won't match, then tries "sonnet"
    assert result.model.id == "claude-sonnet-4-5"
    assert "Invalid thinking level" in result.warning


# resolveModelScopeWithDiagnostics


async def test_returns_scoped_models_and_structured_diagnostics_without_writing_console_warnings(capsys):
    registry = _registry(getAvailable=_available(allModels))

    result = await resolveModelScopeWithDiagnostics(["sonnet:high", "gpt-4o:invalid", "missing"], registry)

    assert [scoped.model.id for scoped in result.scopedModels] == ["claude-sonnet-4-5", "gpt-4o"]
    assert result.scopedModels[0].thinkingLevel == "high"
    assert result.scopedModels[1].thinkingLevel is None
    assert [(d.type, d.message, d.code, d.pattern) for d in result.diagnostics] == [
        (
            "warning",
            'Invalid thinking level "invalid" in pattern "gpt-4o:invalid". Using default instead.',
            "invalid-thinking-level",
            "gpt-4o:invalid",
        ),
        ("warning", 'No models match pattern "missing"', "no-match", "missing"),
    ]
    assert capsys.readouterr().err == ""


async def test_resolve_model_scope_preserves_cli_warning_output(capsys):
    registry = _registry(getAvailable=_available(allModels))

    scopedModels = await resolveModelScope(["missing"], registry)

    assert scopedModels == []
    warnings = capsys.readouterr().err.splitlines()
    assert len(warnings) == 1
    assert 'Warning: No models match pattern "missing"' in warnings[0]


def _bracketedModel() -> Model:
    return _model(
        id="bracketed-model[1m]",
        name="Bracketed Model",
        api="anthropic-messages",
        provider="custom",
        baseUrl="https://example.invalid",
        reasoning=True,
        input=["text"],
        cost={"input": 1, "output": 2, "cacheRead": 0.1, "cacheWrite": 1},
        contextWindow=128000,
        maxTokens=8192,
    )


async def test_resolves_bracketed_model_ids_as_exact_references_before_glob_matching():
    registry = _registry(getAvailable=_available([*allModels, _bracketedModel()]))

    result = await resolveModelScopeWithDiagnostics(["custom/bracketed-model[1m]"], registry)

    assert [scoped.model.id for scoped in result.scopedModels] == ["bracketed-model[1m]"]
    assert result.diagnostics == []


async def test_resolves_bracketed_model_ids_with_thinking_levels_as_exact_references_before_glob_matching():
    registry = _registry(getAvailable=_available([*allModels, _bracketedModel()]))

    result = await resolveModelScopeWithDiagnostics(["custom/bracketed-model[1m]:high"], registry)

    assert [scoped.model.id for scoped in result.scopedModels] == ["bracketed-model[1m]"]
    assert result.scopedModels[0].thinkingLevel == "high"
    assert result.diagnostics == []


# resolveCliModel


def test_resolves_model_provider_id_without_provider():
    registry = _registry(getModels=lambda: allModels)

    result = resolveCliModel({"cliModel": "openai/gpt-4o", "modelRuntime": registry})

    assert result.error is None
    assert result.model.provider == "openai"
    assert result.model.id == "gpt-4o"


def test_resolves_fuzzy_patterns_within_an_explicit_provider():
    registry = _registry(getModels=lambda: allModels)

    result = resolveCliModel({"cliProvider": "openai", "cliModel": "4o", "modelRuntime": registry})

    assert result.error is None
    assert result.model.provider == "openai"
    assert result.model.id == "gpt-4o"


def test_supports_model_pattern_thinking_without_explicit_thinking():
    registry = _registry(getModels=lambda: allModels)

    result = resolveCliModel({"cliModel": "sonnet:high", "modelRuntime": registry})

    assert result.error is None
    assert result.model.id == "claude-sonnet-4-5"
    assert result.thinkingLevel == "high"


def test_prefers_exact_model_id_match_over_provider_inference_openrouter_style_ids():
    registry = _registry(getModels=lambda: allModels)

    result = resolveCliModel({"cliModel": "openai/gpt-4o:extended", "modelRuntime": registry})

    assert result.error is None
    assert result.model.provider == "openrouter"
    assert result.model.id == "openai/gpt-4o:extended"


def test_does_not_strip_invalid_suffix_as_thinking_level_in_model_treat_as_raw_id():
    registry = _registry(getModels=lambda: allModels)

    result = resolveCliModel({"cliProvider": "openai", "cliModel": "gpt-4o:extended", "modelRuntime": registry})

    assert result.error is None
    assert result.model.provider == "openai"
    assert result.model.id == "gpt-4o:extended"


def test_allows_custom_model_ids_for_explicit_providers_without_double_prefixing():
    registry = _registry(getModels=lambda: allModels)

    result = resolveCliModel(
        {"cliProvider": "openrouter", "cliModel": "openrouter/openai/ghost-model", "modelRuntime": registry}
    )

    assert result.error is None
    assert result.model.provider == "openrouter"
    assert result.model.id == "openai/ghost-model"


def test_returns_a_clear_error_when_there_are_no_models():
    registry = _registry(getModels=list)

    result = resolveCliModel({"cliProvider": "openai", "cliModel": "gpt-4o", "modelRuntime": registry})

    assert result.model is None
    assert "No models available" in result.error


def _gpt56Sol(provider: str) -> Model:
    return mockModels[1].model_copy(update={"id": "gpt-5.6-sol", "name": "GPT 5.6 Sol", "provider": provider})


def test_prefers_the_sole_authenticated_provider_for_an_ambiguous_bare_exact_model_id():
    registry = _registry(
        getModels=lambda: [_gpt56Sol("azure"), _gpt56Sol("openai-codex")],
        hasConfiguredAuth=lambda provider: provider == "openai-codex",
    )

    result = resolveCliModel({"cliModel": "gpt-5.6-sol", "modelRuntime": registry})

    assert result.error is None
    assert result.model.provider == "openai-codex"
    assert result.model.id == "gpt-5.6-sol"


def test_requires_an_explicit_provider_for_an_ambiguous_bare_exact_model_id_without_a_unique_authenticated_provider():
    registry = _registry(
        getModels=lambda: [_gpt56Sol("azure"), _gpt56Sol("openai-codex")],
        hasConfiguredAuth=lambda _provider: False,
    )

    result = resolveCliModel({"cliModel": "gpt-5.6-sol", "modelRuntime": registry})

    assert result.model is None
    assert 'Model "gpt-5.6-sol" is ambiguous across providers' in result.error
    assert "azure/gpt-5.6-sol" in result.error
    assert "openai-codex/gpt-5.6-sol" in result.error
    assert "Use --provider or provider/model" in result.error


def test_prefers_provider_model_split_over_gateway_model_with_matching_id():
    # When a user writes "zai/glm-5", and both a zai provider model (id: "glm-5")
    # and a gateway model (id: "zai/glm-5") exist, prefer the zai provider model.
    zaiModel = _model(
        id="glm-5",
        name="GLM-5",
        api="anthropic-messages",
        provider="zai",
        baseUrl="https://open.bigmodel.cn/api/paas/v4",
        reasoning=True,
        input=["text"],
        cost={"input": 1, "output": 2, "cacheRead": 0.1, "cacheWrite": 1},
        contextWindow=128000,
        maxTokens=8192,
    )
    gatewayModel = _model(
        id="zai/glm-5",
        name="GLM-5",
        api="anthropic-messages",
        provider="vercel-ai-gateway",
        baseUrl="https://ai-gateway.vercel.sh",
        reasoning=True,
        input=["text"],
        cost={"input": 1, "output": 2, "cacheRead": 0.1, "cacheWrite": 1},
        contextWindow=128000,
        maxTokens=8192,
    )
    registry = _registry(
        getModels=lambda: [*allModels, zaiModel, gatewayModel],
        hasConfiguredAuth=lambda _provider: True,
    )

    result = resolveCliModel({"cliModel": "zai/glm-5", "modelRuntime": registry})

    assert result.error is None
    assert result.model.provider == "zai"
    assert result.model.id == "glm-5"


def test_prefers_an_authenticated_exact_raw_model_id_over_an_unauthenticated_inferred_provider():
    commandcodeModel = _model(
        id="xiaomi/mimo-v2.5-pro",
        name="Xiaomi MiMo via Commandcode",
        api="anthropic-messages",
        provider="commandcode",
        baseUrl="https://example.invalid",
        reasoning=False,
        input=["text"],
        cost={"input": 1, "output": 2, "cacheRead": 0.1, "cacheWrite": 1},
        contextWindow=128000,
        maxTokens=8192,
    )
    xiaomiModel = _model(
        id="mimo-v2.5-pro",
        name="Xiaomi MiMo",
        api="anthropic-messages",
        provider="xiaomi",
        baseUrl="https://api.xiaomimimo.com",
        reasoning=False,
        input=["text"],
        cost={"input": 1, "output": 2, "cacheRead": 0.1, "cacheWrite": 1},
        contextWindow=128000,
        maxTokens=8192,
    )
    registry = _registry(
        getModels=lambda: [*allModels, commandcodeModel, xiaomiModel],
        hasConfiguredAuth=lambda provider: provider == "commandcode",
    )

    result = resolveCliModel({"cliModel": "xiaomi/mimo-v2.5-pro", "modelRuntime": registry})

    assert result.error is None
    assert result.model.provider == "commandcode"
    assert result.model.id == "xiaomi/mimo-v2.5-pro"


def test_resolves_provider_prefixed_fuzzy_patterns_openrouter_qwen_to_openrouter_model():
    registry = _registry(getModels=lambda: allModels)

    result = resolveCliModel({"cliModel": "openrouter/qwen", "modelRuntime": registry})

    assert result.error is None
    assert result.model.provider == "openrouter"
    assert result.model.id == "qwen/qwen3-coder:exacto"


# resolveCliModel -- custom model fallback with :thinking suffix (#5552)

# Models for a provider that has registered models but the specific model ID
# is not in the registry (triggers buildFallbackModel path).
neuralwattModel = _model(
    id="some-base-model",
    name="Some Base Model",
    api="anthropic-messages",
    provider="neuralwatt",
    baseUrl="https://api.neuralwatt.com",
    reasoning=False,
    input=["text"],
    cost={"input": 1, "output": 2, "cacheRead": 0.1, "cacheWrite": 1},
    contextWindow=128000,
    maxTokens=8192,
)

modelsWithNeuralwatt = [*allModels, neuralwattModel]


def test_strips_thinking_suffix_from_custom_model_id_in_fallback_path():
    registry = _registry(getModels=lambda: modelsWithNeuralwatt)

    result = resolveCliModel({"cliModel": "neuralwatt/zai-org/GLM-5.1-FP8:high", "modelRuntime": registry})

    assert result.error is None
    assert result.model.provider == "neuralwatt"
    # The :high suffix must NOT leak into the model id sent to the API
    assert result.model.id == "zai-org/GLM-5.1-FP8"
    assert result.model.reasoning is True
    assert result.thinkingLevel == "high"


def test_custom_model_without_thinking_suffix_works_normally_in_fallback_path():
    registry = _registry(getModels=lambda: modelsWithNeuralwatt)

    result = resolveCliModel({"cliModel": "neuralwatt/zai-org/GLM-5.1-FP8", "modelRuntime": registry})

    assert result.error is None
    assert result.model.provider == "neuralwatt"
    assert result.model.id == "zai-org/GLM-5.1-FP8"
    assert result.thinkingLevel is None


def test_all_valid_thinking_levels_work_in_fallback_path():
    registry = _registry(getModels=lambda: modelsWithNeuralwatt)

    for level in THINKING_LEVELS:
        result = resolveCliModel({"cliModel": f"neuralwatt/zai-org/GLM-5.1-FP8:{level}", "modelRuntime": registry})

        assert result.error is None
        assert result.model.id == "zai-org/GLM-5.1-FP8"
        assert result.thinkingLevel == level


def test_invalid_thinking_suffix_on_custom_model_is_treated_as_part_of_model_id():
    registry = _registry(getModels=lambda: modelsWithNeuralwatt)

    result = resolveCliModel({"cliModel": "neuralwatt/zai-org/GLM-5.1-FP8:banana", "modelRuntime": registry})

    assert result.error is None
    assert result.model.provider == "neuralwatt"
    # Invalid suffix stays in the id (it's not a thinking level)
    assert result.model.id == "zai-org/GLM-5.1-FP8:banana"
    assert result.thinkingLevel is None


def test_explicit_provider_with_custom_model_thinking_strips_suffix_correctly():
    registry = _registry(getModels=lambda: modelsWithNeuralwatt)

    result = resolveCliModel(
        {"cliProvider": "neuralwatt", "cliModel": "zai-org/GLM-5.1-FP8:high", "modelRuntime": registry}
    )

    assert result.error is None
    assert result.model.provider == "neuralwatt"
    assert result.model.id == "zai-org/GLM-5.1-FP8"
    assert result.thinkingLevel == "high"


def test_with_explicit_thinking_suffix_is_kept_as_part_of_model_id():
    registry = _registry(getModels=lambda: modelsWithNeuralwatt)

    result = resolveCliModel(
        {"cliModel": "neuralwatt/zai-org/GLM-5.1-FP8:high", "cliThinking": "medium", "modelRuntime": registry}
    )

    assert result.error is None
    assert result.model.provider == "neuralwatt"
    # :high is kept as part of the model id since --thinking was explicit
    assert result.model.id == "zai-org/GLM-5.1-FP8:high"
    assert result.thinkingLevel is None


# default model selection


def test_openai_defaults_track_current_models():
    assert defaultModelPerProvider["openai"] == "gpt-5.5"
    assert defaultModelPerProvider["openai-codex"] == "gpt-6.1-sol"


def test_zai_minimax_cerebras_and_ant_ling_defaults_track_current_models():
    assert defaultModelPerProvider["zai"] == "glm-5.3"
    assert defaultModelPerProvider["zai-coding-cn"] == "glm-5.3"
    assert defaultModelPerProvider["minimax"] == "MiniMax-M2.7"
    assert defaultModelPerProvider["minimax-cn"] == "MiniMax-M2.7"
    assert defaultModelPerProvider["cerebras"] == "gpt-oss-120b"
    assert defaultModelPerProvider["ant-ling"] == "Ring-2.6-1T"


def test_built_in_chat_providers_have_defaults_in_their_generated_catalogs():
    for provider in getBuiltinProviders():
        chatModels = getBuiltinModels(provider)
        defaultId = defaultModelPerProvider.get(provider)
        if len(chatModels) == 0:
            assert defaultId is None, f"{provider} has no chat models and should have no chat default"
            continue
        assert any(model.id == defaultId for model in chatModels), (
            f"{provider} default {defaultId} should exist in its generated catalog"
        )


def test_ai_gateway_default_tracks_current_model():
    assert defaultModelPerProvider["vercel-ai-gateway"] == "zai/glm-5.1"


def test_xai_default_tracks_current_model():
    assert defaultModelPerProvider["xai"] == "grok-4.7"


def test_qwen_token_plan_individual_default_tracks_current_model():
    assert defaultModelPerProvider["qwen-token-plan-individual"] == "qwen3.8-max"


async def test_find_initial_model_accepts_explicit_provider_custom_model_ids():
    registry = _registry(getModels=lambda: allModels)

    result = await findInitialModel({
        "cliProvider": "openrouter",
        "cliModel": "openrouter/openai/ghost-model",
        "scopedModels": [],
        "isContinuing": False,
        "modelRuntime": registry,
    })

    assert result.model.provider == "openrouter"
    assert result.model.id == "openai/ghost-model"


async def test_find_initial_model_selects_ai_gateway_default_when_available():
    aiGatewayModel = _model(
        id="anthropic/claude-opus-4-6",
        name="Claude Opus 4.6",
        api="anthropic-messages",
        provider="vercel-ai-gateway",
        baseUrl="https://ai-gateway.vercel.sh",
        reasoning=True,
        input=["text", "image"],
        cost={"input": 5, "output": 15, "cacheRead": 0.5, "cacheWrite": 5},
        contextWindow=200000,
        maxTokens=8192,
    )

    registry = _registry(getAvailableSnapshot=lambda: [aiGatewayModel])

    result = await findInitialModel({"scopedModels": [], "isContinuing": False, "modelRuntime": registry})

    assert result.model.provider == "vercel-ai-gateway"
    assert result.model.id == "anthropic/claude-opus-4-6"


async def test_find_initial_model_ignores_an_unauthenticated_saved_default():
    savedDeepSeekModel = _model(
        id="deepseek-v4-flash",
        name="DeepSeek V4 Flash",
        api="anthropic-messages",
        provider="deepseek",
        baseUrl="https://api.deepseek.com",
        reasoning=True,
        input=["text"],
        cost={"input": 1, "output": 2, "cacheRead": 0.1, "cacheWrite": 1},
        contextWindow=128000,
        maxTokens=8192,
    )
    localDeepSeekModel = savedDeepSeekModel.model_copy(
        update={"provider": "spark-two", "baseUrl": "http://spark-two:8000/v1"}
    )
    registry = _registry(
        getModel=lambda provider, modelId: (
            savedDeepSeekModel
            if provider == savedDeepSeekModel.provider and modelId == savedDeepSeekModel.id
            else None
        ),
        hasConfiguredAuth=lambda provider: provider == "spark-two",
        getAvailableSnapshot=lambda: [localDeepSeekModel],
    )

    result = await findInitialModel({
        "scopedModels": [],
        "isContinuing": False,
        "defaultProvider": "deepseek",
        "defaultModelId": "deepseek-v4-flash",
        "modelRuntime": registry,
    })

    assert result.model.provider == "spark-two"
    assert result.model.id == "deepseek-v4-flash"


# default model selection -- persisted default model scoping

sonnet = getModel("anthropic", "claude-sonnet-4-5")
opus = getModel("anthropic", "claude-opus-4-8")


@pytest.fixture
def tempDirs():
    dirs: list[str] = []
    yield dirs
    for directory in dirs:
        shutil.rmtree(directory, ignore_errors=True)


async def _createSession(tempDirs: list[str], *, scoped: bool, persistedScope: list[str] | None = None):
    tempDir = tempfile.mkdtemp(prefix="pi-default-scope-")
    tempDirs.append(tempDir)

    settingsManager = SettingsManager.create(tempDir, tempDir)
    if persistedScope:
        settingsManager.setEnabledModels(persistedScope)

    authStorage = AuthStorage.inMemory({"anthropic": {"type": "api_key", "key": "test-key"}})
    modelRuntime = getModelRuntime(await createModelRegistry(authStorage, f"{tempDir}/models.json"))
    agent = Agent(initialState={"model": sonnet, "systemPrompt": "test", "tools": []}, streamFn=streamSimple)
    session = AgentSession(AgentSessionConfig(
        agent=agent,
        sessionManager=SessionManager.inMemory(tempDir),
        settingsManager=settingsManager,
        cwd=tempDir,
        modelRuntime=modelRuntime,
        resourceLoader=createTestResourceLoader(),
        scopedModels=[{"model": sonnet}] if scoped else [],
    ))

    return session, settingsManager


def _references(session: AgentSession) -> list[str]:
    return [f"{scoped['model'].provider}/{scoped['model'].id}" for scoped in session.scopedModels]


async def test_adds_a_persisted_default_to_an_existing_scoped_model_list(tempDirs):
    session, settingsManager = await _createSession(
        tempDirs, scoped=True, persistedScope=[f"{sonnet.provider}/{sonnet.id}"]
    )

    await session.setModel(opus, {"persist": True})

    assert settingsManager.getDefaultProvider() == opus.provider
    assert settingsManager.getDefaultModel() == opus.id
    assert _references(session) == [f"{sonnet.provider}/{sonnet.id}", f"{opus.provider}/{opus.id}"]
    assert settingsManager.getEnabledModels() == [f"{sonnet.provider}/{sonnet.id}", f"{opus.provider}/{opus.id}"]


async def test_does_not_create_a_scoped_model_list_when_all_models_are_available(tempDirs):
    session, settingsManager = await _createSession(tempDirs, scoped=False)

    await session.setModel(opus, {"persist": True})

    assert session.scopedModels == []
    assert settingsManager.getEnabledModels() is None


async def test_keeps_session_only_model_changes_out_of_scope(tempDirs):
    session, settingsManager = await _createSession(
        tempDirs, scoped=True, persistedScope=[f"{sonnet.provider}/{sonnet.id}"]
    )

    await session.setModel(opus, {"persist": False})

    assert _references(session) == [f"{sonnet.provider}/{sonnet.id}"]
    assert settingsManager.getEnabledModels() == [f"{sonnet.provider}/{sonnet.id}"]
