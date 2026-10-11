"""Translated from pi's ``coding-agent/src/experimental/services/transcript.ts``.

The root conversation's durable view: active entries and its live, inbox, agent, and usage documents. The service
provides the replicated ``state``, a ``ConversationView``.
"""

from __future__ import annotations

from misaka.chord import defineService

Transcript = defineService("pi.transcript")

__all__ = ["Transcript"]
