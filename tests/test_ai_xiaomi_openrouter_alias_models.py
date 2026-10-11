"""Translated from pi's ``packages/ai/test/xiaomi-models.test.ts`` and ``openrouter-cache-control-models.test.ts``
(v1.1.0)."""

from __future__ import annotations

import pytest

from misaka.ai.compat import getModel, getModels
from misaka.utils.values import read_field

XIAOMI_PROVIDERS = ["xiaomi", "xiaomi-token-plan-cn", "xiaomi-token-plan-ams", "xiaomi-token-plan-sgp"]
DEPRECATED_MODEL_IDS = ["mimo-v2-flash", "mimo-v2-omni", "mimo-v2-pro"]
REPLACEMENT_MODEL_IDS = ["mimo-v2.5", "mimo-v2.5-pro"]


# --- Xiaomi MiMo models -------------------------------------------------------------------------


@pytest.mark.parametrize("provider", XIAOMI_PROVIDERS)
def test_omits_deprecated_models(provider: str):
    modelIds = [model.id for model in getModels(provider)]
    for modelId in DEPRECATED_MODEL_IDS:
        assert modelId not in modelIds


@pytest.mark.parametrize("provider", XIAOMI_PROVIDERS)
def test_keeps_replacement_models(provider: str):
    modelIds = [model.id for model in getModels(provider)]
    for modelId in REPLACEMENT_MODEL_IDS:
        assert modelId in modelIds


# --- OpenRouter Anthropic latest alias metadata ---------------------------------------------------

OPENROUTER_ANTHROPIC_LATEST_MODEL_IDS = [
    "~anthropic/claude-fable-latest",
    "~anthropic/claude-haiku-latest",
    "~anthropic/claude-opus-latest",
    "~anthropic/claude-sonnet-latest",
]


@pytest.mark.parametrize("modelId", OPENROUTER_ANTHROPIC_LATEST_MODEL_IDS)
def test_keeps_completions_cache_control(modelId: str):
    model = getModel("openrouter", modelId)
    assert model.api == "openai-completions"
    assert read_field(model.compat, "cacheControlFormat") == "anthropic"
