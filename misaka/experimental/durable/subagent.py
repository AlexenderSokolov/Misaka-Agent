"""Translated from pi's ``coding-agent/src/experimental/durable/subagent.ts``."""

from __future__ import annotations

from typing import Any

from misaka.chord import Context
from misaka.durable import AssistantEntry
from misaka.durable.harness.agent import configure
from misaka.durable.harness.define import defineExtension, defineTool


async def answerText(api: Any, answer: int, context: Context) -> str:
    entry = await api.commit(lambda tx: tx.entry(AssistantEntry, answer), context)
    model = None if entry is None else entry.get("model")
    message = model[0] if model else None
    if message is None:
        return ""
    return "".join(content["text"] for content in message["content"] if content["type"] == "text")


async def _execute(args: dict[str, Any], api: Any, context: Context) -> dict[str, Any]:
    async def findOrCreateChild(tx: Any) -> int:
        existing = (await tx.scanConversations({"ownerTaskId": api.taskId}, 1))["items"]
        if existing:
            return existing[0]["id"]  # type: ignore[no-any-return]
        # Starts as a copy of this conversation's agent; without this extension it cannot delegate further.
        created = await tx.createConversation({"ownership": {"kind": "task", "taskId": api.taskId}})
        await configure(tx, created["id"], {"extensions": {"remove": [Subagent]}})
        return created["id"]  # type: ignore[no-any-return]

    # A rerun after a crash finds the child it created and the submission it made.
    child = await api.commit(findOrCreateChild, context)
    await api.details({"conversationId": child}, context)
    handle = await api.conversation(child, context)
    request = {"type": "input", "content": args["task"], "requestId": f"subagent:{api.taskId}"}
    settled = await (await handle.submit(request, context)).wait(context)
    if settled["status"] != "done" or settled["type"] != "input":
        raise RuntimeError(f"Subagent {child} failed: {settled['status']}")
    text = await answerText(api, settled["answer"], context)
    return {"content": [{"type": "text", "text": text}], "details": {"conversationId": child}}


# A foreground subagent: each call creates a child conversation owned by the call's task, runs the task there, and
# returns the child's answer. Aborting the call aborts the child. The child outlives the call, so the user can switch to
# it and keep talking.
Subagent: dict[str, Any] = defineExtension(
    {
        "name": "subagent",
        "tools": [
            defineTool(
                {
                    "name": "subagent",
                    "description": (
                        "Delegate a self-contained task to a subagent with the same tools and get its answer back. Give it "
                        "everything it needs to know; it does not see this conversation."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {"task": {"type": "string", "description": "What the subagent should do"}},
                        "required": ["task"],
                    },
                    "replay": "safe",
                    "execute": _execute,
                }
            )
        ],
    }
)
