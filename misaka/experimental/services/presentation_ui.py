"""Translated from pi's ``coding-agent/src/experimental/services/presentation-ui.ts``.

Narrow process-local UI capabilities available to presentation facets. A ``PresentationSelectItem`` is ``{"value",
"label", "description"?}``. The service provides ``select(title, items, selectedValue, context)`` (the chosen value, or
``None``) and ``showStatus(message, context)``.
"""

from __future__ import annotations

from misaka.chord import defineService

PresentationUI = defineService("pi.local.presentation-ui", local=True)

__all__ = ["PresentationUI"]
