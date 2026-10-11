"""Immutable, credential-blind models.json snapshot.

Translated from pi's ``coding-agent/src/core/model-config.ts``. The schema is TypeBox's
there: object schemas that allow extra keys, ``Type.Optional`` fields that reject an
explicit ``null``, values that are never coerced, and a compat block that is a union of
three object schemas. The pydantic models below keep those properties (``_ConfigModel``);
any schema error discards the whole file, as upstream's does.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated, Any, ClassVar, Literal

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    ValidationError,
    ValidationInfo,
    field_validator,
)

from misaka.utils.json import stripJsonComments
from misaka.utils.paths import normalizePath


class _ConfigModel(BaseModel):
    """A TypeBox object schema: extra keys are allowed (and not kept), ``Type.Optional``
    fields may be absent but not ``null`` unless the field lists ``null`` itself, and a
    value of the wrong type is an error rather than converted (strict).

    MISAKA fork: frozen stands in for upstream's ``deepFreeze`` (nested mappings stay
    mutable in Python; nothing downstream writes to them).
    """

    model_config = ConfigDict(extra="ignore", frozen=True, strict=True)

    # Field names whose TypeBox type includes `Type.Null()`.
    _nullable: ClassVar[frozenset[str]] = frozenset()

    @field_validator("*", mode="before")
    @classmethod
    def _reject_null(cls, value: Any, info: ValidationInfo) -> Any:
        if value is None and info.field_name not in cls._nullable:
            raise ValueError("Expected a value, not null")
        return value


def _integral(value: Any) -> Any:
    # `Type.Integer` is `Number.isInteger`: JSON's `2.0` is the integer 2 in JavaScript.
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


_Integer = Annotated[int, BeforeValidator(_integral)]


class _PercentileCutoffsSchema(_ConfigModel):
    p50: float | None = None
    p75: float | None = None
    p90: float | None = None
    p99: float | None = None


class _OpenRouterRoutingSortSchema(_ConfigModel):
    _nullable: ClassVar[frozenset[str]] = frozenset({"partition"})

    by: str | None = None
    partition: str | None = None


class _OpenRouterRoutingMaxPriceSchema(_ConfigModel):
    prompt: float | str | None = None
    completion: float | str | None = None
    image: float | str | None = None
    audio: float | str | None = None
    request: float | str | None = None


class _OpenRouterRoutingSchema(_ConfigModel):
    allow_fallbacks: bool | None = None
    require_parameters: bool | None = None
    data_collection: Literal["deny", "allow"] | None = None
    zdr: bool | None = None
    enforce_distillable_text: bool | None = None
    order: list[str] | None = None
    only: list[str] | None = None
    ignore: list[str] | None = None
    quantizations: list[str] | None = None
    sort: str | _OpenRouterRoutingSortSchema | None = None
    max_price: _OpenRouterRoutingMaxPriceSchema | None = None
    preferred_min_throughput: float | _PercentileCutoffsSchema | None = None
    preferred_max_latency: float | _PercentileCutoffsSchema | None = None


class _VercelGatewayRoutingSchema(_ConfigModel):
    only: list[str] | None = None
    order: list[str] | None = None


class _ThinkingLevelMapSchema(_ConfigModel):
    _nullable: ClassVar[frozenset[str]] = frozenset({"off", "minimal", "low", "medium", "high", "xhigh", "max"})

    off: str | None = None
    minimal: str | None = None
    low: str | None = None
    medium: str | None = None
    high: str | None = None
    xhigh: str | None = None
    max: str | None = None


type _SamplingParamsSchema = dict[str, Any]


class _SamplingParamsByThinkingLevelSchema(_ConfigModel):
    off: _SamplingParamsSchema | None = None
    minimal: _SamplingParamsSchema | None = None
    low: _SamplingParamsSchema | None = None
    medium: _SamplingParamsSchema | None = None
    high: _SamplingParamsSchema | None = None
    xhigh: _SamplingParamsSchema | None = None
    max: _SamplingParamsSchema | None = None


class _ChatTemplateKwargVariableSchema(_ConfigModel):
    var: Literal["thinking.enabled", "thinking.effort"] = Field(alias="$var")
    omitWhenOff: bool | None = None


_ChatTemplateKwargSchema = str | float | bool | None | _ChatTemplateKwargVariableSchema


class _OpenAICompletionsCompatSchema(_ConfigModel):
    supportsStore: bool | None = None
    supportsDeveloperRole: bool | None = None
    supportsReasoningEffort: bool | None = None
    supportsUsageInStreaming: bool | None = None
    supportsFinishReason: bool | None = None
    maxTokensField: Literal["max_completion_tokens", "max_tokens"] | None = None
    requiresToolResultName: bool | None = None
    requiresAssistantAfterToolResult: bool | None = None
    requiresThinkingAsText: bool | None = None
    requiresReasoningContentOnAssistantMessages: bool | None = None
    thinkingFormat: Literal[
        "openai",
        "openrouter",
        "together",
        "baseten",
        "deepseek",
        "zai",
        "qwen",
        "chat-template",
        "qwen-chat-template",
        "string-thinking",
        "ant-ling",
    ] | None = None
    chatTemplateKwargs: dict[str, _ChatTemplateKwargSchema] | None = None
    chatTemplateArgs: dict[str, _ChatTemplateKwargSchema] | None = None
    cacheControlFormat: Literal["anthropic"] | None = None
    openRouterRouting: _OpenRouterRoutingSchema | None = None
    vercelGatewayRouting: _VercelGatewayRoutingSchema | None = None
    supportsOpenAIGrammarTools: bool | None = None
    supportsStrictMode: bool | None = None
    sendSessionAffinityHeaders: bool | None = None
    sessionAffinityFormat: Literal["openai", "openai-nosession", "openrouter"] | None = None
    supportsLongCacheRetention: bool | None = None
    vllmPriority: float | None = None


class _OpenAIResponsesCompatSchema(_ConfigModel):
    supportsDeveloperRole: bool | None = None
    sessionAffinityFormat: Literal["openai", "openai-nosession", "openrouter"] | None = None
    supportsLongCacheRetention: bool | None = None
    supportsStrictMode: bool | None = None
    supportsOpenAIGrammarTools: bool | None = None
    supportsMaxOutputTokens: bool | None = None


class _ModelCostTierSchema(_ConfigModel):
    inputTokensAbove: float
    input: float
    output: float
    cacheRead: float
    cacheWrite: float


class _ModelCostSchema(_ConfigModel):
    input: float
    output: float
    cacheRead: float
    cacheWrite: float
    tiers: list[_ModelCostTierSchema] | None = None


class _PartialModelCostSchema(_ConfigModel):
    input: float | None = None
    output: float | None = None
    cacheRead: float | None = None
    cacheWrite: float | None = None
    tiers: list[_ModelCostTierSchema] | None = None


class _ModelPromptCacheSchema(_ConfigModel):
    short: float | None = Field(default=None, gt=0)
    long: float | None = Field(default=None, gt=0)


class _ImageResizeSchema(_ConfigModel):
    maxWidth: _Integer | None = Field(default=None, ge=1)
    maxHeight: _Integer | None = Field(default=None, ge=1)
    maxBytes: _Integer | None = Field(default=None, ge=1)
    jpegQuality: _Integer | None = Field(default=None, ge=1, le=100)


class _ModelImageInputLimitsSchema(_ConfigModel):
    resize: _ImageResizeSchema | None = None
    maxPerMessage: _Integer | None = Field(default=None, ge=1)
    maxPerRequest: _Integer | None = Field(default=None, ge=1)


class _ModelInputLimitsSchema(_ConfigModel):
    maxRequestBytes: _Integer | None = Field(default=None, ge=1)
    images: _ModelImageInputLimitsSchema | None = None


class _AllowedFallbackModelSchema(_ConfigModel):
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    cost: _ModelCostSchema


class _AnthropicMessagesCompatSchema(_ConfigModel):
    supportsEagerToolInputStreaming: bool | None = None
    supportsLongCacheRetention: bool | None = None
    sendSessionAffinityHeaders: bool | None = None
    supportsCacheControlOnTools: bool | None = None
    supportsTemperature: bool | None = None
    forceAdaptiveThinking: bool | None = None
    allowEmptySignature: bool | None = None
    supportsStrictTools: bool | None = None
    supportsMidConvoEffort: bool | None = None
    allowedFallbackModels: list[_AllowedFallbackModelSchema] | None = Field(default=None, max_length=3)


_COMPAT_UNION: tuple[type[BaseModel], ...] = (
    _OpenAICompletionsCompatSchema,
    _OpenAIResponsesCompatSchema,
    _AnthropicMessagesCompatSchema,
)


def _check_compat_block(value: Any) -> Any:
    """``ProviderCompatSchema``: valid when any arm of the union accepts it. The block is
    kept as written; the composer builds the runtime compat model from it."""
    if value is None:
        raise ValueError("Expected a value, not null")
    for schema in _COMPAT_UNION:
        try:
            schema.model_validate(value)
        except ValidationError:
            continue
        return value
    raise ValueError("Expected union value")


_CompatBlock = Annotated[dict[str, Any] | None, BeforeValidator(_check_compat_block)]


class _ModelDefinitionSchema(_ConfigModel):
    id: str = Field(min_length=1)
    name: str | None = Field(default=None, min_length=1)
    api: str | None = Field(default=None, min_length=1)
    baseUrl: str | None = Field(default=None, min_length=1)
    reasoning: bool | None = None
    thinkingLevelMap: _ThinkingLevelMapSchema | None = None
    input: list[Literal["text", "image"]] | None = None
    inputLimits: _ModelInputLimitsSchema | None = None
    cost: _ModelCostSchema | None = None
    promptCache: _ModelPromptCacheSchema | None = None
    contextWindow: float | None = None
    maxTokens: float | None = None
    samplingParams: _SamplingParamsSchema | None = None
    samplingParamsByThinkingLevel: _SamplingParamsByThinkingLevelSchema | None = None
    headers: dict[str, str] | None = None
    compat: _CompatBlock = None


class _ModelOverrideSchema(_ConfigModel):
    name: str | None = Field(default=None, min_length=1)
    reasoning: bool | None = None
    thinkingLevelMap: _ThinkingLevelMapSchema | None = None
    input: list[Literal["text", "image"]] | None = None
    inputLimits: _ModelInputLimitsSchema | None = None
    cost: _PartialModelCostSchema | None = None
    promptCache: _ModelPromptCacheSchema | None = None
    contextWindow: float | None = None
    maxTokens: float | None = None
    samplingParams: _SamplingParamsSchema | None = None
    samplingParamsByThinkingLevel: _SamplingParamsByThinkingLevelSchema | None = None
    headers: dict[str, str] | None = None
    compat: _CompatBlock = None


class _ProviderConfigSchema(_ConfigModel):
    name: str | None = Field(default=None, min_length=1)
    baseUrl: str | None = Field(default=None, min_length=1)
    apiKey: str | None = Field(default=None, min_length=1)
    api: str | None = Field(default=None, min_length=1)
    oauth: Literal["radius"] | None = None
    headers: dict[str, str] | None = None
    compat: _CompatBlock = None
    authHeader: bool | None = None
    models: list[_ModelDefinitionSchema] | None = None
    modelOverrides: dict[str, _ModelOverrideSchema] | None = None


class _ModelsConfigSchema(_ConfigModel):
    providers: dict[str, _ProviderConfigSchema]


ModelsJsonModel = _ModelDefinitionSchema
ModelsJsonModelOverride = _ModelOverrideSchema
ModelsJsonProvider = _ProviderConfigSchema


def _formatValidationPath(location: tuple[Any, ...]) -> str:
    path = ".".join(str(part) for part in location if part != "")
    return path or "root"


def _schemaErrors(error: ValidationError) -> str:
    return "\n".join(
        f"  - {_formatValidationPath(tuple(item.get('loc', ())))}: {item['msg']}"
        for item in error.errors(include_url=False)
    ) or "Unknown schema error"


class ModelConfig:
    """One immutable load of models.json."""

    def __init__(self, providers: dict[str, ModelsJsonProvider], error: str | None = None) -> None:
        # Private by convention: build one with `ModelConfig.load`.
        self._providers = providers
        self._error = error

    @classmethod
    async def load(cls, modelsJsonPath: str | None) -> ModelConfig:
        if not modelsJsonPath:
            return cls({})
        path = normalizePath(modelsJsonPath)
        try:
            content = await asyncio.to_thread(Path(path).read_text, encoding="utf-8", errors="replace")
        except FileNotFoundError:
            return cls({})
        except OSError as error:
            return cls({}, f"Failed to load models.json: {error}\n\nFile: {path}")

        try:
            parsed = json.loads(stripJsonComments(content.removeprefix("﻿")))
        except ValueError as error:
            return cls({}, f"Failed to parse models.json: {error}\n\nFile: {path}")

        try:
            config = _ModelsConfigSchema.model_validate(parsed)
        except ValidationError as error:
            return cls({}, f"Invalid models.json schema:\n{_schemaErrors(error)}\n\nFile: {path}")

        return cls(dict(config.providers))

    def getProvider(self, providerId: str) -> ModelsJsonProvider | None:
        return self._providers.get(providerId)

    def getProviderIds(self) -> list[str]:
        return list(self._providers)

    def getError(self) -> str | None:
        return self._error


__all__ = [
    "ModelConfig",
    "ModelsJsonModel",
    "ModelsJsonModelOverride",
    "ModelsJsonProvider",
]
