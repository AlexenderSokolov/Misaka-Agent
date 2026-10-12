"""Translated from pi's ``packages/coding-agent/src/core/model-resolver.ts`` (v1.1.0).

Model resolution, scoping, and initial selection.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from typing import Any, Literal

from wcmatch import glob

from misaka.ai.models import modelsAreEqual
from misaka.ai.types import Model
from misaka.cli.args import isValidThinkingLevel
from misaka.core.defaults import DEFAULT_THINKING_LEVEL
from misaka.utils.collation import localeSortKey
from misaka.utils.js import jsTrim

type ResolvedThinkingLevel = Literal["off", "minimal", "low", "medium", "high", "xhigh", "max"]

# Default chat model IDs for providers with built-in chat models.
defaultModelPerProvider: dict[str, str] = {
    "amazon-bedrock": "us.anthropic.claude-opus-4-6-v1",
    "ant-ling": "Ring-2.6-1T",
    "anthropic": "claude-opus-4-8",
    "openai": "gpt-5.5",
    "azure": "gpt-5.4",
    "openai-codex": "gpt-6.1-sol",
    "radius": "balanced",
    "nvidia": "nvidia/nemotron-3-ultra-550b-a55b",
    "deepseek": "deepseek-v4-pro",
    "google": "gemini-3.1-pro-preview",
    "google-vertex": "gemini-3.1-pro-preview",
    "github-copilot": "gpt-5.4",
    "openrouter": "moonshotai/kimi-k2.6",
    "vercel-ai-gateway": "zai/glm-5.1",
    "xai": "grok-4.7",
    "groq": "openai/gpt-oss-120b",
    "cerebras": "gpt-oss-120b",
    "zai": "glm-5.3",
    "zai-coding-cn": "glm-5.3",
    "mistral": "devstral-medium-latest",
    "minimax": "MiniMax-M2.7",
    "minimax-cn": "MiniMax-M2.7",
    "moonshotai": "kimi-k2.6",
    "moonshotai-cn": "kimi-k2.6",
    "huggingface": "moonshotai/Kimi-K2.6",
    "fireworks": "accounts/fireworks/models/kimi-k3",
    "together": "moonshotai/Kimi-K3",
    "baseten": "zai-org/GLM-5.2",
    "opencode": "kimi-k2.6",
    "opencode-go": "kimi-k3",
    "kimi-coding": "kimi-for-coding",
    "meta": "muse-spark-1.3",
    "cloudflare-workers-ai": "@cf/moonshotai/kimi-k2.6",
    "cloudflare-ai-gateway": "workers-ai/@cf/moonshotai/kimi-k2.6",
    "qwen-token-plan": "qwen3.7-max",
    "qwen-token-plan-cn": "qwen3.7-max",
    "qwen-token-plan-individual": "qwen3.8-max",
    "xiaomi": "mimo-v2.5-pro",
    "xiaomi-token-plan-cn": "mimo-v2.5-pro",
    "xiaomi-token-plan-ams": "mimo-v2.5-pro",
    "xiaomi-token-plan-sgp": "mimo-v2.5-pro",
}


@dataclass(slots=True)
class ScopedModel:
    model: Model
    # Thinking level if explicitly specified in pattern (e.g., "model:high"), undefined otherwise
    thinkingLevel: ResolvedThinkingLevel | None = None


def _chalk(code: str, message: str) -> str:
    return f"\x1b[{code}m{message}\x1b[0m"


def _dim(message: str) -> str:
    return _chalk("2", message)


def _yellow(message: str) -> str:
    return _chalk("33", message)


def _red(message: str) -> str:
    return _chalk("31", message)


# `minimatch(path, pattern, { nocase })`: globstar, braces and extglobs on, as minimatch has them by default.
_MINIMATCH_FLAGS = glob.GLOBSTAR | glob.BRACE | glob.EXTMATCH | glob.FORCEUNIX


def _minimatch(path: str, pattern: str, *, nocase: bool = False) -> bool:
    return glob.globmatch(path, pattern, flags=_MINIMATCH_FLAGS | (glob.IGNORECASE if nocase else 0))


def isAlias(id: str) -> bool:
    """Helper to check if a model ID looks like an alias (no date suffix).
    Dates are typically in format: -20241022 or -20250929"""
    # Check if ID ends with -latest
    if id.endswith("-latest"):
        return True

    # Check if ID ends with a date pattern (-YYYYMMDD)
    return re.search(r"-\d{8}$", id) is None


def findExactModelReferenceMatch(modelReference: str, availableModels: list[Model]) -> Model | None:
    """Find an exact model reference match.
    Supports either a bare model id or a canonical provider/modelId reference.
    When matching by bare id, ambiguous matches across providers are rejected."""
    trimmedReference = jsTrim(modelReference)
    if not trimmedReference:
        return None

    normalizedReference = trimmedReference.lower()

    canonicalMatches = [
        model for model in availableModels if f"{model.provider}/{model.id}".lower() == normalizedReference
    ]
    if len(canonicalMatches) == 1:
        return canonicalMatches[0]
    if len(canonicalMatches) > 1:
        return None

    slashIndex = trimmedReference.find("/")
    if slashIndex != -1:
        provider = jsTrim(trimmedReference[:slashIndex])
        modelId = jsTrim(trimmedReference[slashIndex + 1 :])
        if provider and modelId:
            providerMatches = [
                model
                for model in availableModels
                if model.provider.lower() == provider.lower() and model.id.lower() == modelId.lower()
            ]
            if len(providerMatches) == 1:
                return providerMatches[0]
            if len(providerMatches) > 1:
                return None

    idMatches = [model for model in availableModels if model.id.lower() == normalizedReference]
    return idMatches[0] if len(idMatches) == 1 else None


def tryMatchModel(modelPattern: str, availableModels: list[Model]) -> Model | None:
    """Try to match a pattern to a model from the available models list.
    Returns the matched model or undefined if no match found."""
    exactMatch = findExactModelReferenceMatch(modelPattern, availableModels)
    if exactMatch is not None:
        return exactMatch

    # No exact match - fall back to partial matching
    matches = [
        m
        for m in availableModels
        if modelPattern.lower() in m.id.lower() or (m.name is not None and modelPattern.lower() in m.name.lower())
    ]

    if len(matches) == 0:
        return None

    # Separate into aliases and dated versions
    aliases = [m for m in matches if isAlias(m.id)]
    datedVersions = [m for m in matches if not isAlias(m.id)]

    if len(aliases) > 0:
        # Prefer alias - if multiple aliases, pick the one that sorts highest
        aliases.sort(key=lambda m: localeSortKey(m.id), reverse=True)
        return aliases[0]
    # No alias found, pick latest dated version
    datedVersions.sort(key=lambda m: localeSortKey(m.id), reverse=True)
    return datedVersions[0]


@dataclass(slots=True)
class ParsedModelResult:
    model: Model | None
    # Thinking level if explicitly specified in pattern, undefined otherwise
    thinkingLevel: ResolvedThinkingLevel | None
    warning: str | None


def buildFallbackModel(provider: str, modelId: str, availableModels: list[Model]) -> Model | None:
    providerModels = [m for m in availableModels if m.provider == provider]
    if len(providerModels) == 0:
        return None

    defaultId = defaultModelPerProvider.get(provider)
    baseModel = (
        next((m for m in providerModels if m.id == defaultId), providerModels[0]) if defaultId else providerModels[0]
    )

    return baseModel.model_copy(update={"id": modelId, "name": modelId})


def parseModelPattern(
    pattern: str,
    availableModels: list[Model],
    options: dict[str, Any] | None = None,
) -> ParsedModelResult:
    """Parse a pattern to extract model and thinking level.
    Handles models with colons in their IDs (e.g., OpenRouter's :exacto suffix).

    Algorithm:
    1. Try to match full pattern as a model
    2. If found, return it with "off" thinking level
    3. If not found and has colons, split on last colon:
       - If suffix is valid thinking level, use it and recurse on prefix
       - If suffix is invalid, warn and recurse on prefix with "off"

    @internal Exported for testing
    """
    # Try exact match first
    exactMatch = tryMatchModel(pattern, availableModels)
    if exactMatch is not None:
        return ParsedModelResult(model=exactMatch, thinkingLevel=None, warning=None)

    # No match - try splitting on last colon if present
    lastColonIndex = pattern.rfind(":")
    if lastColonIndex == -1:
        # No colons, pattern simply doesn't match any model
        return ParsedModelResult(model=None, thinkingLevel=None, warning=None)

    prefix = pattern[:lastColonIndex]
    suffix = pattern[lastColonIndex + 1 :]

    if isValidThinkingLevel(suffix):
        # Valid thinking level - recurse on prefix and use this level
        result = parseModelPattern(prefix, availableModels, options)
        if result.model is not None:
            # Only use this thinking level if no warning from inner recursion
            return ParsedModelResult(
                model=result.model,
                thinkingLevel=None if result.warning else suffix,  # type: ignore[arg-type]
                warning=result.warning,
            )
        return result
    # Invalid suffix
    allowFallback = (options or {}).get("allowInvalidThinkingLevelFallback", True)
    if not allowFallback:
        # In strict mode (CLI --model parsing), treat it as part of the model id and fail.
        # This avoids accidentally resolving to a different model.
        return ParsedModelResult(model=None, thinkingLevel=None, warning=None)

    # Scope mode: recurse on prefix and warn
    result = parseModelPattern(prefix, availableModels, options)
    if result.model is not None:
        return ParsedModelResult(
            model=result.model,
            thinkingLevel=None,
            warning=f'Invalid thinking level "{suffix}" in pattern "{pattern}". Using default instead.',
        )
    return result


# Resolve model patterns to actual Model objects with optional thinking levels
# Format: "pattern:level" where :level is optional
# For each pattern, finds all matching models and picks the best version:
# 1. Prefer alias (e.g., claude-sonnet-4-5) over dated versions (claude-sonnet-4-5-20250929)
# 2. If no alias, pick the latest dated version
#
# Supports models with colons in their IDs (e.g., OpenRouter's model:exacto).
# The algorithm tries to match the full pattern first, then progressively
# strips colon-suffixes to find a match.
@dataclass(slots=True)
class ModelScopeDiagnostic:
    code: Literal["no-match", "invalid-thinking-level"]
    message: str
    pattern: str
    type: Literal["warning"] = "warning"


@dataclass(slots=True)
class ResolveModelScopeResult:
    scopedModels: list[ScopedModel]
    diagnostics: list[ModelScopeDiagnostic]


def resolveModelScopeFromModels(patterns: list[str], models: list[Model]) -> ResolveModelScopeResult:
    availableModels = list(models)
    scopedModels: list[ScopedModel] = []
    diagnostics: list[ModelScopeDiagnostic] = []

    for pattern in patterns:
        # Check if pattern contains glob characters
        if "*" in pattern or "?" in pattern or "[" in pattern:
            # Extract optional thinking level suffix (e.g., "provider/*:high")
            colonIdx = pattern.rfind(":")
            globPattern = pattern
            thinkingLevel: ResolvedThinkingLevel | None = None

            if colonIdx != -1:
                suffix = pattern[colonIdx + 1 :]
                if isValidThinkingLevel(suffix):
                    thinkingLevel = suffix  # type: ignore[assignment]
                    globPattern = pattern[:colonIdx]

            exactMatch = findExactModelReferenceMatch(globPattern, availableModels)
            if exactMatch is not None:
                if not any(modelsAreEqual(sm.model, exactMatch) for sm in scopedModels):
                    scopedModels.append(ScopedModel(model=exactMatch, thinkingLevel=thinkingLevel))
                continue

            # Match against "provider/modelId" format OR just model ID
            # This allows "*sonnet*" to match without requiring "anthropic/*sonnet*"
            matchingModels = [
                m
                for m in availableModels
                if _minimatch(f"{m.provider}/{m.id}", globPattern, nocase=True)
                or _minimatch(m.id, globPattern, nocase=True)
            ]

            if len(matchingModels) == 0:
                diagnostics.append(
                    ModelScopeDiagnostic(code="no-match", message=f'No models match pattern "{pattern}"', pattern=pattern)
                )
                continue

            for model in matchingModels:
                if not any(modelsAreEqual(sm.model, model) for sm in scopedModels):
                    scopedModels.append(ScopedModel(model=model, thinkingLevel=thinkingLevel))
            continue

        result = parseModelPattern(pattern, availableModels)

        if result.warning:
            diagnostics.append(
                ModelScopeDiagnostic(code="invalid-thinking-level", message=result.warning, pattern=pattern)
            )

        if result.model is None:
            diagnostics.append(
                ModelScopeDiagnostic(code="no-match", message=f'No models match pattern "{pattern}"', pattern=pattern)
            )
            continue

        # Avoid duplicates
        if not any(modelsAreEqual(sm.model, result.model) for sm in scopedModels):
            scopedModels.append(ScopedModel(model=result.model, thinkingLevel=result.thinkingLevel))

    return ResolveModelScopeResult(scopedModels=scopedModels, diagnostics=diagnostics)


async def resolveModelScopeWithDiagnostics(
    patterns: list[str], modelRuntime: Any, options: Any = None
) -> ResolveModelScopeResult:
    return resolveModelScopeFromModels(patterns, list(await modelRuntime.getAvailable(None, options)))


async def resolveModelScope(patterns: list[str], modelRuntime: Any, options: Any = None) -> list[ScopedModel]:
    result = await resolveModelScopeWithDiagnostics(patterns, modelRuntime, options)
    for diagnostic in result.diagnostics:
        print(_yellow(f"Warning: {diagnostic.message}"), file=sys.stderr)
    return result.scopedModels


@dataclass(slots=True)
class ResolveCliModelResult:
    model: Model | None
    warning: str | None
    # Error message suitable for CLI display.
    # When set, model will be undefined.
    error: str | None
    thinkingLevel: ResolvedThinkingLevel | None = None


def resolveCliModel(options: dict[str, Any]) -> ResolveCliModelResult:
    """Resolve a single model from CLI flags.

    Supports:
    - --provider <provider> --model <pattern>
    - --model <provider>/<pattern>
    - Fuzzy matching (same rules as model scoping: exact id, then partial id/name)

    Note: This does not apply the thinking level by itself, but it may *parse* and
    return a thinking level from "<pattern>:<thinking>" so the caller can apply it.
    """
    cliProvider = options.get("cliProvider")
    cliModel = options.get("cliModel")
    cliThinking = options.get("cliThinking")
    modelRuntime = options["modelRuntime"]

    if not cliModel:
        return ResolveCliModelResult(model=None, warning=None, error=None)

    # Important: use *all* models here, not just models with pre-configured auth.
    # This allows "--api-key" to be used for first-time setup.
    availableModels = list(modelRuntime.getModels())
    if len(availableModels) == 0:
        return ResolveCliModelResult(
            model=None,
            warning=None,
            error="No models available. Check your installation or add models to models.json.",
        )

    # Build canonical provider lookup (case-insensitive)
    providerMap: dict[str, str] = {}
    for m in availableModels:
        providerMap[m.provider.lower()] = m.provider

    provider = providerMap.get(cliProvider.lower()) if cliProvider else None
    if cliProvider and not provider:
        return ResolveCliModelResult(
            model=None,
            warning=None,
            error=f'Unknown provider "{cliProvider}". Use --list-models to see available providers/models.',
        )

    # If no explicit --provider, try to interpret "provider/model" format first.
    # When the prefix before the first slash matches a known provider, prefer that
    # interpretation over matching models whose IDs literally contain slashes
    # (e.g. "zai/glm-5" should resolve to provider=zai, model=glm-5, not to a
    # vercel-ai-gateway model with id "zai/glm-5").
    pattern = cliModel
    inferredProvider = False

    if not provider:
        slashIndex = cliModel.find("/")
        if slashIndex != -1:
            maybeProvider = cliModel[:slashIndex]
            canonical = providerMap.get(maybeProvider.lower())
            if canonical:
                provider = canonical
                pattern = cliModel[slashIndex + 1 :]
                inferredProvider = True

    # If no provider was inferred from the slash, try exact matches without provider inference.
    # This handles models whose IDs naturally contain slashes (e.g. OpenRouter-style IDs).
    # Bare exact IDs can exist in multiple providers, so do not choose by catalog order.
    # Prefer the sole authenticated provider when there is one; otherwise require an
    # explicit provider to avoid silently selecting an unusable provider.
    if not provider:
        lower = cliModel.lower()
        exactMatches = [
            m for m in availableModels if m.id.lower() == lower or f"{m.provider}/{m.id}".lower() == lower
        ]
        if len(exactMatches) == 1:
            return ResolveCliModelResult(model=exactMatches[0], warning=None, thinkingLevel=None, error=None)
        if len(exactMatches) > 1:
            authenticatedExactMatches = [m for m in exactMatches if modelRuntime.hasConfiguredAuth(m.provider)]
            if len(authenticatedExactMatches) == 1:
                return ResolveCliModelResult(
                    model=authenticatedExactMatches[0],
                    warning=None,
                    thinkingLevel=None,
                    error=None,
                )

            matches = ", ".join(sorted((f"{m.provider}/{m.id}" for m in exactMatches), key=localeSortKey))
            authHint = (
                "No matching provider is authenticated."
                if len(authenticatedExactMatches) == 0
                else "More than one matching provider is authenticated."
            )
            return ResolveCliModelResult(
                model=None,
                warning=None,
                thinkingLevel=None,
                error=f'Model "{cliModel}" is ambiguous across providers: {matches}. {authHint} Use --provider or provider/model.',
            )

    if cliProvider and provider:
        # If both were provided, tolerate --model <provider>/<pattern> by stripping the provider prefix
        prefix = f"{provider}/"
        if cliModel.lower().startswith(prefix.lower()):
            pattern = cliModel[len(prefix) :]

    candidates = [m for m in availableModels if m.provider == provider] if provider else availableModels
    parsed = parseModelPattern(pattern, candidates, {"allowInvalidThinkingLevelFallback": False})
    model, thinkingLevel, warning = parsed.model, parsed.thinkingLevel, parsed.warning

    if model is not None:
        # If provider inference matched an unauthenticated provider/model pair, prefer
        # one exact raw model-id match that is authenticated. This keeps
        # "provider/model" syntax preferred when usable, but handles models whose
        # literal id starts with a known provider name (for example
        # commandcode model id "xiaomi/mimo-v2.5-pro").
        if inferredProvider:
            rawExactMatches = [
                m for m in availableModels if m.id.lower() == cliModel.lower() and not modelsAreEqual(m, model)
            ]
            if len(rawExactMatches) > 0 and not modelRuntime.hasConfiguredAuth(model.provider):
                authenticatedRawMatches = [m for m in rawExactMatches if modelRuntime.hasConfiguredAuth(m.provider)]
                if len(authenticatedRawMatches) == 1:
                    return ResolveCliModelResult(
                        model=authenticatedRawMatches[0],
                        thinkingLevel=None,
                        warning=None,
                        error=None,
                    )
        return ResolveCliModelResult(model=model, thinkingLevel=thinkingLevel, warning=warning, error=None)

    # If we inferred a provider from the slash but found no match within that provider,
    # fall back to matching the full input as a raw model id across all models.
    # This handles OpenRouter-style IDs like "openai/gpt-4o:extended" where "openai"
    # looks like a provider but the full string is actually a model id on openrouter.
    if inferredProvider:
        lower = cliModel.lower()
        exact = next(
            (m for m in availableModels if m.id.lower() == lower or f"{m.provider}/{m.id}".lower() == lower),
            None,
        )
        if exact is not None:
            return ResolveCliModelResult(model=exact, warning=None, thinkingLevel=None, error=None)
        # Also try parseModelPattern on the full input against all models
        fallback = parseModelPattern(cliModel, availableModels, {"allowInvalidThinkingLevelFallback": False})
        if fallback.model is not None:
            return ResolveCliModelResult(
                model=fallback.model,
                thinkingLevel=fallback.thinkingLevel,
                warning=fallback.warning,
                error=None,
            )

    if provider:
        # Parse thinking level suffix from the pattern before building the fallback model,
        # but only when --thinking is not explicitly provided.
        # e.g. "zai-org/GLM-5.1-FP8:high" → modelId="zai-org/GLM-5.1-FP8", fallbackThinking="high"
        fallbackPattern = pattern
        fallbackThinking: ResolvedThinkingLevel | None = None
        if not cliThinking:
            lastColon = pattern.rfind(":")
            if lastColon != -1:
                suffix = pattern[lastColon + 1 :]
                if isValidThinkingLevel(suffix):
                    fallbackPattern = pattern[:lastColon]
                    fallbackThinking = suffix  # type: ignore[assignment]

        fallbackModel = buildFallbackModel(provider, fallbackPattern, availableModels)
        if fallbackModel is not None:
            requestedThinking = cliThinking or fallbackThinking
            model = (
                fallbackModel.model_copy(update={"reasoning": True})
                if requestedThinking and requestedThinking != "off"
                else fallbackModel
            )
            fallbackWarning = (
                f'{warning} Model "{fallbackPattern}" not found for provider "{provider}". Using custom model id.'
                if warning
                else f'Model "{fallbackPattern}" not found for provider "{provider}". Using custom model id.'
            )
            return ResolveCliModelResult(model=model, thinkingLevel=fallbackThinking, warning=fallbackWarning, error=None)

    display = f"{provider}/{pattern}" if provider else cliModel
    return ResolveCliModelResult(
        model=None,
        thinkingLevel=None,
        warning=warning,
        error=f'Model "{display}" not found. Use --list-models to see available models.',
    )


@dataclass(slots=True)
class InitialModelResult:
    model: Model | None
    thinkingLevel: ResolvedThinkingLevel
    fallbackMessage: str | None


async def findInitialModel(options: dict[str, Any]) -> InitialModelResult:
    """Find the initial model to use based on priority:
    1. CLI args (provider + model)
    2. First model from scoped models (if not continuing/resuming)
    3. Restored from session (if continuing/resuming)
    4. Saved default from settings
    5. First available model with valid API key
    """
    cliProvider = options.get("cliProvider")
    cliModel = options.get("cliModel")
    scopedModels: list[ScopedModel] = options["scopedModels"]
    isContinuing = options["isContinuing"]
    defaultProvider = options.get("defaultProvider")
    defaultModelId = options.get("defaultModelId")
    defaultThinkingLevel = options.get("defaultThinkingLevel")
    modelThinkingLevels: dict[str, ResolvedThinkingLevel] | None = options.get("modelThinkingLevels")
    modelRuntime = options["modelRuntime"]

    model: Model | None = None
    thinkingLevel: ResolvedThinkingLevel = DEFAULT_THINKING_LEVEL

    # 1. CLI args take priority
    if cliProvider and cliModel:
        resolved = resolveCliModel({"cliProvider": cliProvider, "cliModel": cliModel, "modelRuntime": modelRuntime})
        if resolved.error:
            print(_red(resolved.error), file=sys.stderr)
            raise SystemExit(1)
        if resolved.model is not None:
            return InitialModelResult(model=resolved.model, thinkingLevel=DEFAULT_THINKING_LEVEL, fallbackMessage=None)

    # 2. Use first model from scoped models (skip if continuing/resuming)
    if len(scopedModels) > 0 and not isContinuing:
        scopedModel = scopedModels[0]
        perModel = (modelThinkingLevels or {}).get(f"{scopedModel.model.provider}/{scopedModel.model.id}")
        return InitialModelResult(
            model=scopedModel.model,
            thinkingLevel=next(
                (level for level in (scopedModel.thinkingLevel, perModel, defaultThinkingLevel) if level is not None),
                DEFAULT_THINKING_LEVEL,
            ),
            fallbackMessage=None,
        )

    # 3. Try saved default from settings if auth is configured.
    if defaultProvider and defaultModelId:
        found = modelRuntime.getModel(defaultProvider, defaultModelId)
        if found is not None and modelRuntime.hasConfiguredAuth(found.provider):
            model = found
            perModel = (modelThinkingLevels or {}).get(f"{defaultProvider}/{defaultModelId}")
            if perModel:
                thinkingLevel = perModel
            elif defaultThinkingLevel:
                thinkingLevel = defaultThinkingLevel
            return InitialModelResult(model=model, thinkingLevel=thinkingLevel, fallbackMessage=None)

    # 4. Try first available model with valid API key
    availableModels = list(modelRuntime.getAvailableSnapshot())

    if len(availableModels) > 0:
        # Try to find a default model from known providers
        for provider, defaultId in defaultModelPerProvider.items():
            match = next((m for m in availableModels if m.provider == provider and m.id == defaultId), None)
            if match is not None:
                return InitialModelResult(model=match, thinkingLevel=DEFAULT_THINKING_LEVEL, fallbackMessage=None)

        # If no default found, use first available
        return InitialModelResult(model=availableModels[0], thinkingLevel=DEFAULT_THINKING_LEVEL, fallbackMessage=None)

    # 5. No model found
    return InitialModelResult(model=None, thinkingLevel=DEFAULT_THINKING_LEVEL, fallbackMessage=None)


async def restoreModelFromSession(
    savedProvider: str,
    savedModelId: str,
    currentModel: Model | None,
    shouldPrintMessages: bool,
    modelRuntime: Any,
) -> dict[str, Any]:
    """Restore model from session, with fallback to available models. Resolves ``{model, fallbackMessage}``."""
    restoredModel = modelRuntime.getModel(savedProvider, savedModelId)

    # Check if restored model exists and still has auth configured
    hasConfiguredAuth = modelRuntime.hasConfiguredAuth(restoredModel.provider) if restoredModel is not None else False

    if restoredModel is not None and hasConfiguredAuth:
        if shouldPrintMessages:
            print(_dim(f"Restored model: {savedProvider}/{savedModelId}"))
        return {"model": restoredModel, "fallbackMessage": None}

    # Model not found or no API key - fall back
    reason = "model no longer exists" if restoredModel is None else "no auth configured"

    if shouldPrintMessages:
        print(_yellow(f"Warning: Could not restore model {savedProvider}/{savedModelId} ({reason})."), file=sys.stderr)

    # If we already have a model, use it as fallback
    if currentModel is not None:
        if shouldPrintMessages:
            print(_dim(f"Falling back to: {currentModel.provider}/{currentModel.id}"))
        return {
            "model": currentModel,
            "fallbackMessage": f"Could not restore model {savedProvider}/{savedModelId} ({reason}). Using {currentModel.provider}/{currentModel.id}.",
        }

    # Try to find any available model
    availableModels = list(modelRuntime.getAvailableSnapshot())

    if len(availableModels) > 0:
        # Try to find a default model from known providers
        fallbackModel: Model | None = None
        for provider, defaultId in defaultModelPerProvider.items():
            match = next((m for m in availableModels if m.provider == provider and m.id == defaultId), None)
            if match is not None:
                fallbackModel = match
                break

        # If no default found, use first available
        if fallbackModel is None:
            fallbackModel = availableModels[0]

        if shouldPrintMessages:
            print(_dim(f"Falling back to: {fallbackModel.provider}/{fallbackModel.id}"))

        return {
            "model": fallbackModel,
            "fallbackMessage": f"Could not restore model {savedProvider}/{savedModelId} ({reason}). Using {fallbackModel.provider}/{fallbackModel.id}.",
        }

    # No models available
    return {"model": None, "fallbackMessage": None}


__all__ = [
    "InitialModelResult",
    "ModelScopeDiagnostic",
    "ParsedModelResult",
    "ResolveCliModelResult",
    "ResolveModelScopeResult",
    "ScopedModel",
    "defaultModelPerProvider",
    "findExactModelReferenceMatch",
    "findInitialModel",
    "parseModelPattern",
    "resolveCliModel",
    "resolveModelScope",
    "resolveModelScopeFromModels",
    "resolveModelScopeWithDiagnostics",
    "restoreModelFromSession",
]
