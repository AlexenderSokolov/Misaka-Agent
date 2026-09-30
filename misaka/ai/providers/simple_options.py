"""Shared helpers for mapping simple reasoning options to provider options."""

from __future__ import annotations

from dataclasses import dataclass

from misaka.ai.types import (
    Model,
    SimpleStreamOptions,
    StreamOptions,
    ThinkingBudgets,
    ThinkingLevel,
    TranscriptContext,
)
from misaka.ai.utils.estimate import clamp_max_tokens_to_context


def build_base_options(
    model: Model,
    context: TranscriptContext,
    options: SimpleStreamOptions | None = None,
    api_key: str | None = None,
) -> StreamOptions:
    # The builders merge `model.samplingParams` under these (api/simple-options.ts:29).
    sampling = options.samplingParams if options is not None else None
    # `?? model.maxTokens`, then clamped to what the window can still hold
    # (api/simple-options.ts:34). Passing the caller's value straight through meant a
    # request with no explicit maxTokens sent none at all -- and one near a full window
    # asked for more output than the provider could grant, failing instead of shrinking.
    requested_max = options.maxTokens if options is not None and options.maxTokens is not None else model.maxTokens
    max_tokens = clamp_max_tokens_to_context(model, context, requested_max)
    if options is None:
        return StreamOptions(apiKey=api_key or None, samplingParams=sampling, maxTokens=max_tokens)

    return StreamOptions(
        samplingParams=sampling,
        temperature=options.temperature,
        maxTokens=max_tokens,
        signal=options.signal,
        apiKey=api_key or options.apiKey,
        env=options.env,
        transport=options.transport,
        cacheRetention=options.cacheRetention,
        sessionId=options.sessionId,
        headers=options.headers,
        onPayload=options.onPayload,
        onResponse=options.onResponse,
        onProviderStreamEvent=options.onProviderStreamEvent,
        timeoutMs=options.timeoutMs,
        maxRetries=options.maxRetries,
        maxRetryDelayMs=options.maxRetryDelayMs,
        metadata=options.metadata,
    )


def clamp_reasoning(effort: ThinkingLevel | None) -> ThinkingLevel | None:
    """Fold the two top levels onto a key the thinking-budget table actually has.

    This is a *lookup* clamp, not a wire restriction. The budget table below stops at
    `high`, so `budgets[level]` would raise on `xhigh` or `max`; both borrow high's budget
    instead. Upstream calls its `clampReasoning` in exactly the same two places and for
    the same reason (`api/simple-options.ts:70`, `api/bedrock-converse-stream.ts:550`).

    What this deliberately does *not* do is clamp the request path: `reasoningEffort`
    carries `xhigh`/`max` through to the provider.
    """
    return "high" if effort in ("xhigh", "max") else effort


MIN_ANSWER_TOKENS = 1024
"""Tokens always left for the answer when a thinking budget shares the response ceiling."""


def clamp_thinking_budget_to_answer_room(thinking_budget: int, ceiling: int) -> int:
    """Cap a thinking budget so at least MIN_ANSWER_TOKENS remain under a shared ceiling."""
    return min(thinking_budget, max(0, ceiling - MIN_ANSWER_TOKENS))


@dataclass(frozen=True, slots=True)
class AdjustedThinkingTokens:
    maxTokens: int
    thinkingBudget: int


DEFAULT_THINKING_BUDGETS = ThinkingBudgets(minimal=1024, low=2048, medium=8192, high=16384)


def thinking_budget_for_level(
    reasoning_level: ThinkingLevel, custom_budgets: ThinkingBudgets | None = None
) -> int:
    """The token budget one thinking level asks for, before any clamping.

    Its own function, as upstream's ``thinkingBudgetForLevel`` is: two callers need it --
    the Anthropic-style adjustment below and openai-completions' ``thinking_token_budget``
    field -- and the table has to be the same one for both.
    """
    budgets = DEFAULT_THINKING_BUDGETS.model_dump()
    if custom_budgets is not None:
        budgets.update({key: value for key, value in custom_budgets.model_dump().items() if value is not None})
    level = clamp_reasoning(reasoning_level)
    if level is None:
        raise ValueError("reasoning_level must not be None")
    budget = budgets[level]
    if budget is None:
        raise ValueError(f"No thinking budget configured for reasoning level {reasoning_level}")
    return budget


def adjust_max_tokens_for_thinking(
    base_max_tokens: int | None,
    model_max_tokens: int,
    reasoning_level: ThinkingLevel,
    custom_budgets: ThinkingBudgets | None = None,
) -> AdjustedThinkingTokens:
    thinking_budget = thinking_budget_for_level(reasoning_level, custom_budgets)

    max_tokens = model_max_tokens if base_max_tokens is None else min(base_max_tokens + thinking_budget, model_max_tokens)
    if max_tokens <= thinking_budget:
        thinking_budget = clamp_thinking_budget_to_answer_room(thinking_budget, max_tokens)

    return AdjustedThinkingTokens(maxTokens=max_tokens, thinkingBudget=thinking_budget)


def thinking_allowance(model: Model, options: SimpleStreamOptions | None) -> int:
    """Thinking a provider adapter requests *beyond* ``maxTokens`` for this request, in tokens.

    Two adapters add one: budget-thinking Claude (``anthropic-messages``, Bedrock), whose wire
    ``max_tokens`` is ``min(base + budget, model max)`` (``adjust_max_tokens_for_thinking``), and
    Gemini 2.5, whose ``thinkingBudget`` is not counted in ``maxOutputTokens`` (a dynamic budget,
    ``-1``, is taken as the model's whole output). Every other adapter -- adaptive Claude, the
    OpenAI APIs, DeepSeek (its ``max_tokens`` covers the reasoning, api-docs.deepseek.com),
    Mistral -- counts thinking inside ``maxTokens``. The one place these rules are written down:
    the token-cap meter (``misaka.core.platform.metering``) budgets a request from it.
    """
    from misaka.utils.values import read_field
    reasoning = read_field(options, "reasoning") if options is not None else None
    if not reasoning or reasoning == "off" or not getattr(model, "reasoning", False):
        return 0
    budgets = read_field(options, "thinkingBudgets")
    if model.api == "anthropic-messages":
        from misaka.ai.providers.anthropic import _force_adaptive_thinking
        return 0 if _force_adaptive_thinking(model) is True else thinking_budget_for_level(reasoning, budgets)
    if model.api == "bedrock-converse-stream":
        from misaka.ai.providers.amazon_bedrock import (
            is_anthropic_claude_model,
            supports_adaptive_thinking,
        )
        if is_anthropic_claude_model(model) and not supports_adaptive_thinking(model.id, model.name):
            return thinking_budget_for_level(reasoning, budgets)
        return 0
    if model.api in ("google-generative-ai", "google-vertex"):
        from misaka.ai.models import clamp_thinking_level
        from misaka.ai.providers.google import get_google_budget
        from misaka.ai.providers.google_shared import (
            resolve_google_thinking_level,
            uses_google_thinking_level,
        )
        clamped = clamp_thinking_level(model, reasoning)
        if clamped == "off" or uses_google_thinking_level(model):
            return 0
        budget = get_google_budget(model, resolve_google_thinking_level(model, clamped), budgets)
        return int(model.maxTokens or 0) if budget == -1 else max(0, int(budget))
    return 0


__all__ = [
    "DEFAULT_THINKING_BUDGETS",
    "thinking_allowance",
    "thinking_budget_for_level",
    ]
