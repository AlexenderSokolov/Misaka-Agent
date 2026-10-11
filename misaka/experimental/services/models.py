"""Translated from pi's ``coding-agent/src/experimental/services/models.ts``.

A ``ModelRef`` is ``{"provider", "modelId"}`` and a ``ModelSummary`` adds ``"name"`` and ``"reasoning"``. The replicated
``ModelsState`` is ``{"catalog": {"revision", "availableModels"}, "configuration": {"model": ModelRef | None,
"thinkingLevel"}, "refresh": {"status": "idle" | "refreshing" | "done"} | {"status": "warning", "errors": {...}}}``.

The service provides ``state`` and ``cycleThinking(context)``, ``getThinkingLevels(context)``, ``refresh(context)``,
``select(model, context)`` and ``selectThinking(level, context)``.
"""

from __future__ import annotations

from misaka.chord import defineService

Models = defineService("pi.models")

__all__ = ["Models"]
