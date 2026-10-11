"""Translated from pi's ``coding-agent/src/experimental/services/plugins.ts``.

``PresentationPlugins``: server-built plugin generations available to presentations. It provides
``prepareSession({"sessionId", "packagePaths": [str] | None}, context)`` and ``reload(context)``; both return JSON data
the presentation turns into facet loaders.

``SessionPlugins``: plugin facets hosted in the currently attached Session worker. It provides ``reload(context)``.
"""

from __future__ import annotations

from misaka.chord import defineService

PresentationPlugins = defineService("pi.presentation-plugins")
SessionPlugins = defineService("pi.session-plugins")

__all__ = ["PresentationPlugins", "SessionPlugins"]
