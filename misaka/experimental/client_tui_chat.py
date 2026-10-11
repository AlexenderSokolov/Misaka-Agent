"""Translated from pi's ``coding-agent/src/experimental/client-tui-chat.ts``.

``ExperimentalChatView`` renders the root conversation's durable view (a ``ConversationView``: the dict ``{"entries": [...],
"docs": {...}}``) for the service-only experimental presentation.

"""

from __future__ import annotations

import re
from typing import Any

from misaka.core.extensions.types import ToolRenderers
from misaka.core.tools.renderers import createAllToolRenderers
from misaka.ui.tui import Container, Spacer, Text, TruncatedText
from misaka.ui.tui.interactive.components.assistant_message import (
    AssistantMessageComponent,
)
from misaka.ui.tui.interactive.components.status_indicator import WorkingStatusIndicator
from misaka.ui.tui.interactive.components.tool_execution import ToolExecutionComponent
from misaka.ui.tui.interactive.components.user_message import UserMessageComponent
from misaka.ui.tui.interactive.theme.theme import theme


def liveOf(view: dict[str, Any]) -> dict[str, Any]:
    """The ``pi.live`` document of a view: the active run, the streaming answer, and running tools."""
    live = view["docs"].get("pi.live")
    return {} if live is None else live


def userText(content: Any) -> str:
    if isinstance(content, str):
        return content
    return "".join(block["text"] for block in content if block["type"] == "text")


class ExperimentalChatView:
    _renderers: dict[str, ToolRenderers] = createAllToolRenderers()

    def __init__(self, ui: Any, cwd: str) -> None:
        self.transcript = Container()
        self.pendingMessages = Container()
        self.status = Container()
        self._ui = ui
        self._cwd = cwd
        # The newest card per call ID; provider call IDs may repeat across turns.
        self._tools: dict[str, ToolExecutionComponent] = {}
        # Every card shown, also older ones whose call ID a later turn reused.
        self._cards: list[ToolExecutionComponent] = []
        # Call IDs whose cards the streaming answer created; its entry takes them over.
        self._streamingCalls: set[str] = set()
        self._renderedEntryIds: list[int] = []
        self._streaming: AssistantMessageComponent | None = None
        self._indicator: WorkingStatusIndicator | None = None
        self._statusText = ""

    def apply(self, view: dict[str, Any]) -> None:
        live = liveOf(view)
        self._syncTranscript(view["entries"])
        generation = live.get("generation")
        message = None if generation is None else generation.get("message")
        # A partial without its entry was dropped, for example by a retry: render the transcript again.
        if message is None and self._streaming is not None:
            self._rebuild(view["entries"])
        if message is not None:
            self._syncStreaming(message)
        for slot in live.get("tools") or []:
            if slot["status"] == "pending":
                continue
            component = self._tool(slot["name"], slot["callId"])
            component.setArgsComplete()
            if slot["status"] != "running":
                continue
            component.markExecutionStarted()
            if slot.get("output") is not None:
                component.updateResult(
                    {"content": [{"type": "text", "text": slot["output"]}], "details": slot.get("details"), "isError": False},
                    True,
                )
        inbox = view["docs"].get("pi.inbox")
        self._syncQueue({"items": []} if inbox is None else inbox)
        self._syncStatus(live)
        self.transcript.invalidate()
        self.pendingMessages.invalidate()
        self.status.invalidate()

    def refreshTheme(self, view: dict[str, Any]) -> None:
        if self._indicator is not None:
            self._indicator.dispose()
        self._indicator = None
        self._statusText = ""
        self.status.clear()
        self._rebuild(view["entries"])
        self.apply(view)

    def dispose(self) -> None:
        self._discardTools()
        if self._indicator is not None:
            self._indicator.dispose()

    def _discardTools(self) -> None:
        """Finish every card: a running bash card keeps a timer until it gets a final result."""
        for component in self._cards:
            component.updateResult({"content": [], "isError": False}, False)
        self._cards.clear()
        self._tools.clear()
        self._streamingCalls.clear()

    def _syncQueue(self, inbox: dict[str, Any]) -> None:
        self.pendingMessages.clear()
        for item in inbox["items"]:
            text = (
                f"<{item['entry']['kind']}>"
                if item["mode"] == "write"
                else re.sub(r"\s+", " ", userText(item["content"]))
            )
            self.pendingMessages.addChild(TruncatedText(theme.fg("muted", f"[{item['mode']}] {text}"), 1, 0))

    def _syncStatus(self, live: dict[str, Any]) -> None:
        generation: Any = live.get("generation")
        compactions = live.get("compactions")
        compaction: Any = compactions[0] if compactions else None
        runningTool: Any = next((slot for slot in live.get("tools") or [] if slot["status"] == "running"), None)
        text = ""
        if generation is not None and generation.get("retry") is not None:
            text = f"Retrying (attempt {generation['attempt'] + 1}): {generation['retry']['error']}"
        elif generation is not None and generation.get("deferred") is not None:
            text = "Waiting for deferred response..."
        elif compaction is not None:
            text = (
                f"Retrying {compaction['reason']} compaction (attempt {compaction['attempt'] + 1})..."
                if compaction.get("retry")
                else f"Compacting ({compaction['reason']})..."
            )
        elif runningTool is not None:
            text = f"Running {runningTool['name']}... (esc to abort)"
        elif live.get("run") is not None:
            text = "Working... (esc to abort)"
        if text == self._statusText:
            return
        self._statusText = text
        if self._indicator is not None:
            self._indicator.dispose()
        self._indicator = None
        self.status.clear()
        if len(text) > 0:
            self._indicator = WorkingStatusIndicator(self._ui, text)
            self.status.addChild(self._indicator)

    def _syncTranscript(self, entries: list[dict[str, Any]]) -> None:
        # Compaction and resets replace the head of the active transcript.
        if any(index >= len(entries) or entries[index]["id"] != id for index, id in enumerate(self._renderedEntryIds)):
            self._rebuild(entries)
        for entry in entries[len(self._renderedEntryIds) :]:
            self._addEntry(entry)
            self._renderedEntryIds.append(entry["id"])

    def _rebuild(self, entries: list[dict[str, Any]]) -> None:
        self.transcript.clear()
        self._discardTools()
        self._renderedEntryIds = []
        self._streaming = None
        for entry in entries:
            self._addEntry(entry)
            self._renderedEntryIds.append(entry["id"])

    def _addEntry(self, entry: dict[str, Any]) -> None:
        model = entry.get("model")
        message: Any = model[0] if model else None
        role = None if message is None else message.get("role")
        if entry["kind"] == "pi.user" and role == "user":
            self.transcript.addChild(Spacer(1))
            self.transcript.addChild(UserMessageComponent(userText(message["content"])))
        elif entry["kind"] == "pi.assistant" and role == "assistant":
            component = self._streaming if self._streaming is not None else AssistantMessageComponent()
            if self._streaming is None:
                self.transcript.addChild(component)
            self._streaming = None
            component.updateContent(message, False)
            # Only a tool-calling answer runs its calls; an aborted, failed, or truncated one never does.
            ran = message.get("stopReason") == "toolUse"
            for content in message["content"]:
                if content["type"] != "toolCall":
                    continue
                streamed = content["id"] in self._streamingCalls
                if not ran and not streamed:
                    continue
                card = self._tool(content["name"], content["id"], content["arguments"], not streamed)
                card.setArgsComplete()
                if not ran:
                    text = "Not run: the answer was interrupted."
                    card.updateResult({"content": [{"type": "text", "text": text}], "isError": True}, False)
            self._streamingCalls.clear()
        elif entry["kind"] == "pi.tool-result" and role == "toolResult":
            self._tool(message["toolName"], message["toolCallId"]).updateResult(message)
        elif entry["kind"] == "pi.compaction":
            self._addText(theme.fg("muted", "[compaction]"))
            if role == "user":
                self._addText(userText(message["content"]))
        elif entry["kind"] == "pi.reset":
            self._addText(theme.fg("muted", "[new context]"))

    def _syncStreaming(self, message: dict[str, Any]) -> None:
        if self._streaming is None:
            self._streaming = AssistantMessageComponent()
            self.transcript.addChild(self._streaming)
        self._streaming.updateContent(message, True)
        for content in message["content"]:
            if content["type"] != "toolCall":
                continue
            self._tool(content["name"], content["id"], content["arguments"], content["id"] not in self._streamingCalls)
            self._streamingCalls.add(content["id"])

    def _tool(self, toolName: str, toolCallId: str, args: Any = None, fresh: bool = False) -> ToolExecutionComponent:
        """The card of a call; ``fresh`` starts a new one for a call ID an earlier turn used."""
        existing = None if fresh else self._tools.get(toolCallId)
        if existing is not None:
            if args is not None:
                existing.updateArgs(args)
            return existing
        component = ToolExecutionComponent(
            toolName,
            toolCallId,
            args if args is not None else {},
            {},
            ExperimentalChatView._renderers.get(toolName),
            self._ui,
            self._cwd,
        )
        self.transcript.addChild(component)
        self._cards.append(component)
        self._tools[toolCallId] = component
        return component

    def _addText(self, text: str) -> None:
        self.transcript.addChild(Spacer(1))
        self.transcript.addChild(Text(text, 1, 0))
