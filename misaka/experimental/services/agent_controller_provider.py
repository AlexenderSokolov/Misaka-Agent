"""Translated from pi's ``coding-agent/src/experimental/services/agent-controller-provider.ts``.

The controller is a ``SimpleNamespace`` of the ``AgentController`` members (see ``agent_controller.py`` for the
request and response dicts).
"""

from __future__ import annotations

import re
from types import SimpleNamespace
from typing import Any

from misaka.chord import Context
from misaka.durable import ConversationBusy
from misaka.utils.js import errorMessage

_SUBMISSION_ID = re.compile(r"[1-9][0-9]*")
_MAX_SAFE_INTEGER = 2**53 - 1


def createAgentController(harness: Any, conversation: Any) -> SimpleNamespace:
    async def queue(whenBusy: str, request: dict[str, Any], context: Context) -> dict[str, Any]:
        try:
            submission = await conversation.submit(
                {"type": "input", "content": toInput(request), "whenBusy": whenBusy}, context
            )
            return {"accepted": True, "entryId": str(submission.id), "error": None}
        except Exception as error:  # noqa: BLE001 - the failure is the response
            return {"accepted": False, "entryId": None, "error": toAgentError(error)}

    async def prompt(request: dict[str, Any], context: Context) -> dict[str, Any]:
        try:
            submission = await conversation.submit(
                {"type": "input", "content": toInput(request), "whenBusy": "reject"}, context
            )
            return {"accepted": True, "operationId": str(submission.id), "error": None}
        except Exception as error:  # noqa: BLE001 - the failure is the response
            return {"accepted": False, "operationId": None, "error": toAgentError(error)}

    def steer(request: dict[str, Any], context: Context) -> Any:
        return queue("steer", request, context)

    def followUp(request: dict[str, Any], context: Context) -> Any:
        return queue("followUp", request, context)

    async def cancelQueued(entryId: str, context: Context) -> dict[str, str]:
        id = parseSubmissionId(entryId)
        if id is None:
            return {"outcome": "not_found"}
        result = await harness.abortSubmission(id, context, conversation.id)
        return {
            "outcome": "cancelled" if result == "aborted" else "not_found" if result == "not_found" else "already_consumed"
        }

    def abort(context: Context) -> Any:
        return conversation.abort(context)

    async def compact(request: dict[str, Any], context: Context) -> dict[str, Any]:
        try:
            id = await conversation.compact(request.get("customInstructions"), context)
            return {"accepted": True, "operationId": str(id), "error": None}
        except Exception as error:  # noqa: BLE001 - the failure is the response
            return {"accepted": False, "operationId": None, "error": toAgentError(error)}

    async def waitForPrompt(operationId: str, context: Context) -> dict[str, Any]:
        id = parseSubmissionId(operationId)
        submission = None if id is None else await harness.submission(id, context)
        if submission is None:
            raise RuntimeError(f"Unknown prompt: {operationId}")
        settled = await submission.wait(context)
        if settled["status"] == "unanswered":
            return {"status": "unanswered", "text": None, "reason": settled["reason"]}
        answer = settled["answer"] if settled["type"] == "input" else None
        entry = None if answer is None else await harness.commit(lambda tx: tx.entry(answer), context)
        model = None if entry is None else entry.get("model")
        message = model[0] if model else None
        text = (
            "".join(content["text"] for content in message["content"] if content["type"] == "text")
            if message is not None and message["role"] == "assistant"
            else ""
        )
        return {"status": "done", "text": text, "reason": None}

    return SimpleNamespace(
        prompt=prompt,
        steer=steer,
        followUp=followUp,
        cancelQueued=cancelQueued,
        abort=abort,
        compact=compact,
        waitForPrompt=waitForPrompt,
    )


def parseSubmissionId(value: str) -> int | None:
    if _SUBMISSION_ID.fullmatch(value) is None:
        return None
    id = int(value)
    return id if id <= _MAX_SAFE_INTEGER else None


def toInput(request: dict[str, Any]) -> Any:
    images = request.get("images")
    if images is None or len(images) == 0:
        return request["message"]
    return [{"type": "text", "text": request["message"]}, *images]


def toAgentError(error: BaseException) -> dict[str, str]:
    if isinstance(error, ConversationBusy):
        return {"code": "busy", "message": errorMessage(error)}
    return {"code": "operation_failed", "message": errorMessage(error)}
