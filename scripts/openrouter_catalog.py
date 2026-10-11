"""Translated from pi's ``packages/ai/scripts/openrouter-catalog.ts`` (v1.1.0).

An OpenRouter list item is the API's JSON object (``id``, ``name``, ``supported_parameters``, ``architecture``,
``pricing`` with prompt-length or time-of-day ``overrides``, ``top_provider``, ``context_length``, ``reasoning``); the
catalog is ``{"chat": [...], "images": [...], "classifiers": [...]}`` of model records. A property the source lacks is
``UNDEFINED``, which ``jsonStringify`` leaves out as ``JSON.stringify`` does.
"""

from __future__ import annotations

import re
from typing import Any

from openrouter_reasoning_options import getOpenRouterThinkingLevelMap

from misaka.utils.js import UNDEFINED, parseFloat, toFixed


def roundCost(value: float) -> float:
    return float(toFixed(value, 6))


def modalities(values: list[Any] | None) -> list[str]:
    result: list[str] = []
    for value in values or []:
        if (value == "text" or value == "image") and value not in result:
            result.append(value)
    return result


def perMillion(value: str | None, fallback: float) -> float:
    """Convert pricing from $/token to $/million tokens."""
    return roundCost(parseFloat(value) * 1_000_000) if value else fallback


def cost(model: dict[str, Any]) -> dict[str, Any]:
    pricing = model.get("pricing") or {}
    base: dict[str, Any] = {
        "input": perMillion(pricing.get("prompt"), 0),
        "output": perMillion(pricing.get("completion"), 0),
        "cacheRead": perMillion(pricing.get("input_cache_read"), 0),
        "cacheWrite": perMillion(pricing.get("input_cache_write"), 0),
    }
    # Prompt-length overrides become request-wide tiers. Time-of-day overrides are skipped
    # because ModelCost cannot express them.
    tiers: list[dict[str, Any]] = []
    for override in pricing.get("overrides") or []:
        if (
            "min_prompt_tokens" not in override
            or "utc_start" in override
            or "utc_end" in override
            or "utc_days" in override
        ):
            continue
        tiers.append(
            {
                "inputTokensAbove": override["min_prompt_tokens"],
                "input": perMillion(override.get("prompt"), base["input"]),
                "output": perMillion(override.get("completion"), base["output"]),
                "cacheRead": perMillion(override.get("input_cache_read"), base["cacheRead"]),
                "cacheWrite": perMillion(override.get("input_cache_write"), base["cacheWrite"]),
            }
        )
    return {**base, "tiers": tiers} if len(tiers) > 0 else base


def _architecture(model: dict[str, Any]) -> dict[str, Any]:
    architecture = model.get("architecture")
    return architecture if isinstance(architecture, dict) else {}


def _contextWindow(model: dict[str, Any]) -> Any:
    topProvider = model.get("top_provider") or {}
    return topProvider.get("context_length") or model.get("context_length") or 4096


def buildOpenRouterCatalog(
    listed: list[dict[str, Any]],
    imageListed: list[dict[str, Any]],
    decisionListed: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Build the OpenRouter catalog from the default listing and the ``output_modalities=image`` and
    ``output_modalities=decisions`` listings. The default listing omits image-only and decision models, so those come
    from the other listings. An upstream model may appear in several results; it then gets separate entries per
    operation."""
    chat: list[dict[str, Any]] = []

    for model in listed:
        # Only include models that support tools
        if "tools" not in (model.get("supported_parameters") or []):
            continue
        # Parse input modalities
        input = ["text"]
        modality = _architecture(model).get("modality")
        if isinstance(modality, str) and "image" in modality:
            input.append("image")

        thinkingLevelMap = getOpenRouterThinkingLevelMap(model.get("reasoning"))
        useAnthropicMessages = re.match(r"^anthropic/", model["id"]) is not None and not model["id"].endswith(":batch")
        chat.append(
            {
                "type": "chat",
                "id": model["id"],
                "name": model.get("name", UNDEFINED),
                "api": "anthropic-messages" if useAnthropicMessages else "openai-completions",
                "baseUrl": "https://openrouter.ai/api" if useAnthropicMessages else "https://openrouter.ai/api/v1",
                "provider": "openrouter",
                "reasoning": "reasoning" in (model.get("supported_parameters") or []) or False,
                **({"thinkingLevelMap": thinkingLevelMap} if thinkingLevelMap else {}),
                "input": input,
                "cost": cost(model),
                "contextWindow": _contextWindow(model),
                "maxTokens": (model.get("top_provider") or {}).get("max_completion_tokens") or 4096,
            }
        )

    images: list[dict[str, Any]] = []
    for model in imageListed:
        if any(entry["id"] == model["id"] for entry in images):
            continue
        output = modalities(_architecture(model).get("output_modalities"))
        if "image" not in output:
            continue
        input = modalities(_architecture(model).get("input_modalities"))
        images.append(
            {
                "type": "image",
                "id": model["id"],
                "name": model.get("name", UNDEFINED),
                "api": "openrouter-images",
                "provider": "openrouter",
                "baseUrl": "https://openrouter.ai/api/v1",
                "input": input if len(input) > 0 else ["text"],
                "output": output,
                "cost": cost(model),
            }
        )

    # Decision models such as TypeSafe's Jev are served through OpenRouter's
    # TypeSafe-compatible System One endpoint.
    classifiers: list[dict[str, Any]] = []
    for model in decisionListed:
        if any(entry["id"] == model["id"] for entry in classifiers):
            continue
        if "decisions" not in (_architecture(model).get("output_modalities") or []):
            continue
        input = modalities(_architecture(model).get("input_modalities"))
        classifiers.append(
            {
                "type": "classifier",
                "id": model["id"],
                "name": model.get("name", UNDEFINED),
                "api": "typesafe-system-one",
                "provider": "openrouter",
                "baseUrl": "https://openrouter.ai/api/v1",
                "input": input if len(input) > 0 else ["text"],
                "cost": cost(model),
                "contextWindow": _contextWindow(model),
            }
        )

    return {"chat": chat, "images": images, "classifiers": classifiers}


__all__ = ["buildOpenRouterCatalog", "cost", "modalities", "perMillion", "roundCost"]
