"""Translated from pi's ``coding-agent/src/experimental/services/agent-controller.ts``.

Requests and responses are plain dicts with upstream's keys:

- an ``AgentPromptRequest`` is ``{"message": str, "images": [{"type": "image", "data", "mimeType"}] | None}``;
- an ``AgentOperationResponse`` is ``{"accepted": True, "operationId": str, "error": None}`` or ``{"accepted": False,
  "operationId": None, "error": {"code", "message"}}``, where ``operationId`` identifies the durable submission of a
  prompt, or the task of a compaction;
- an ``AgentQueueResponse`` is the same with ``entryId`` instead of ``operationId``: the durable submission, which
  ``cancelQueued()`` withdraws while it is still queued;
- an ``AgentPromptResult`` is ``{"status": "done", "text": str, "reason": None}`` or ``{"status": "unanswered", "text":
  None, "reason": str}``: the answer's text, or why it got none;
- an ``AgentCompactionRequest`` is ``{"customInstructions": str | None}``.

The service is a presentation-safe command facade over the worker-owned root conversation:

- ``prompt(request, context)``: start a run; rejected with ``busy`` while one is active;
- ``steer(request, context)``: steer the active run, or start one when idle;
- ``followUp(request, context)``: queue input for after the active run, or start one when idle;
- ``cancelQueued(entryId, context)``: ``{"outcome": "cancelled" | "already_consumed" | "not_found"}``;
- ``abort(context)``: withdraw queued input and abort the active run and compaction;
- ``compact(request, context)``: an operation response;
- ``waitForPrompt(operationId, context)``: wait until the prompt with this ``operationId`` is answered or settles
  unanswered.
"""

from __future__ import annotations

from misaka.chord import defineService

AgentController = defineService("pi.agent-controller")

__all__ = ["AgentController"]
