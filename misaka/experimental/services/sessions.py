"""Translated from pi's ``coding-agent/src/experimental/services/sessions.ts``.

A ``SessionAddress`` is ``{"serverId", "sessionId"}`` and a ``SessionSummary`` adds ``"createdAt"``. A
``SessionDirectoryState`` is ``{"revision": int, "sessions": [SessionSummary]}``.

``SessionDirectory`` provides the replicated ``state``. ``SessionManagement`` provides ``create(options, context)`` (with
``options`` ``{"id"?}``), ``remove(sessionId, context)``, ``attach(sessionId, context)`` and ``detach(context)``.
"""

from __future__ import annotations

from misaka.chord import defineService

SessionDirectory = defineService("pi.session-directory")
SessionManagement = defineService("pi.session-management")

__all__ = ["SessionDirectory", "SessionManagement"]
