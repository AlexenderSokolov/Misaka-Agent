"""Translated from pi's ``coding-agent/src/experimental/durable/tui.ts``.

The durable demo's terminal UI: it renders a ``DurableView`` (``runtime.py``) with the interactive components.

MISAKA fork:

- pi's demo runs in the alternate screen with a scrollable transcript and a docked editor (``TuiAltScreen``, ``ScrollView``,
  ``VStack``); MISAKA has only the regular renderer, so the parts are stacked in the order they are docked in pi (transcript,
  tasks, queue, notices, status, editor, footer) and the terminal scrolls.
- The working status is shown above the editor; pi embeds it in the editor's border.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Callable
from typing import Any

from misaka.config.engine import get_agent_dir
from misaka.core.extensions.types import ToolRenderers
from misaka.core.keybindings import KeybindingsManager
from misaka.core.tools.renderers import createAllToolRenderers
from misaka.experimental.durable.runtime import agentOf
from misaka.ui.tui import (
    TUI,
    Box,
    Container,
    DefaultTextStyle,
    EditorOptions,
    Input,
    Markdown,
    ProcessTerminal,
    SelectItem,
    SelectList,
    SelectListTheme,
    Spacer,
    Text,
    TruncatedText,
    TuiMainScreen,
    fuzzyFilter,
    getKeybindings,
    setCapabilityOverrides,
    setKeybindings,
)
from misaka.ui.tui.interactive.components.assistant_message import (
    AssistantMessageComponent,
)
from misaka.ui.tui.interactive.components.custom_editor import CustomEditor
from misaka.ui.tui.interactive.components.dynamic_border import DynamicBorder
from misaka.ui.tui.interactive.components.footer import formatTokens
from misaka.ui.tui.interactive.components.keybinding_hints import keyText
from misaka.ui.tui.interactive.components.status_indicator import WorkingStatusIndicator
from misaka.ui.tui.interactive.components.tool_execution import ToolExecutionComponent
from misaka.ui.tui.interactive.components.user_message import UserMessageComponent
from misaka.ui.tui.interactive.theme.theme import (
    ThemeColor,
    getEditorTheme,
    getMarkdownTheme,
    initTheme,
    theme,
)
from misaka.ui.tui.interactive.theme.theme_controller import InteractiveThemeController
from misaka.utils.js import jsTrim, toFixed

SELECT_THEME = SelectListTheme(
    selectedPrefix=lambda text: theme.fg("accent", text),
    selectedText=lambda text: theme.fg("accent", text),
    description=lambda text: theme.fg("muted", text),
    scrollInfo=lambda text: theme.fg("dim", text),
    noMatch=lambda text: theme.fg("warning", text),
)


class ListSelector(Container):
    """A filterable list in place of the editor."""

    def __init__(
        self, title: str, items: list[SelectItem], onSelect: Callable[[str], None], onCancel: Callable[[], None]
    ) -> None:
        super().__init__()
        self._input = Input()
        self._listContainer = Container()
        self._items = items
        self._onSelect = onSelect
        self._onCancel = onCancel
        self._focused = False
        self._list = self._build(items)
        self.addChild(DynamicBorder())
        self.addChild(Spacer(1))
        self.addChild(Text(theme.fg("accent", theme.bold(title)), 1, 0))
        self.addChild(self._input)
        self.addChild(Spacer(1))
        self.addChild(self._listContainer)
        self.addChild(DynamicBorder())

    @property
    def focused(self) -> bool:
        return self._focused

    @focused.setter
    def focused(self, value: bool) -> None:
        self._focused = value
        self._input.focused = value

    def handleInput(self, data: str) -> None:
        keybindings = getKeybindings()
        if any(
            keybindings.matches(data, action)
            for action in ("tui.select.up", "tui.select.down", "tui.select.confirm", "tui.select.cancel")
        ):
            self._list.handleInput(data)
            return
        self._input.handleInput(data)
        query = self._input.getValue()
        filtered = self._items if len(query) == 0 else fuzzyFilter(self._items, query, lambda item: f"{item.label} {item.value}")
        self._list = self._build(filtered)

    def _build(self, items: list[SelectItem]) -> SelectList:
        selectList = SelectList(items, 10, SELECT_THEME)
        selectList.onSelect = lambda item: self._onSelect(item.value)
        selectList.onCancel = self._onCancel
        self._listContainer.clear()
        self._listContainer.addChild(selectList)
        return selectList


class CompactionComponent(Box):
    """The summary that replaced earlier context: collapsed to one line until expanded."""

    def __init__(self, summary: str, expanded: bool) -> None:
        super().__init__(1, 1, lambda text: theme.bg("customMessageBg", text))
        self._summary = summary
        self.setExpanded(expanded)

    def setExpanded(self, expanded: bool) -> None:
        self.clear()
        self.addChild(Text(theme.fg("customMessageLabel", theme.bold("[compaction]")), 0, 0))
        self.addChild(Spacer(1))
        self.addChild(
            Markdown(
                self._summary,
                0,
                0,
                getMarkdownTheme(),
                DefaultTextStyle(color=lambda text: theme.fg("customMessageText", text)),
            )
            if expanded
            else Text(
                theme.fg("customMessageText", "Earlier context summarized (")
                + theme.fg("dim", keyText("app.tools.expand"))
                + theme.fg("customMessageText", " to expand)"),
                0,
                0,
            )
        )


class DurableTui:
    """``handlers`` is ``{"submit"(text), "followUp"(text), "abort"(), "exit"(), "selectModel"(), "cycleThinking"()}``."""

    _renderers: dict[str, ToolRenderers] = createAllToolRenderers()

    def __init__(self, cwd: str, handlers: dict[str, Callable[..., Any]]) -> None:
        self._cwd = cwd
        self._ui = TuiMainScreen(ProcessTerminal(), False, get_agent_dir())
        self._chat = Container()
        self._tasks = Container()
        self._queue = Container()
        self._notices = Container()
        self._status = Container()
        self._footer = Container()
        self._footerStats = Text("", 1, 0)
        self._footerHints = Text("", 1, 0)
        self._editorContainer = Container()
        # The newest card per call ID; provider call IDs may repeat across turns.
        self._tools: dict[str, ToolExecutionComponent] = {}
        # Every card shown, also older ones whose call ID a later turn reused.
        self._cards: list[ToolExecutionComponent] = []
        # Call IDs whose cards the streaming answer created; its entry takes them over.
        self._streamingCalls: set[str] = set()
        self._summaries: list[CompactionComponent] = []
        # Tool output and summaries shown in full; toggled like pi.
        self._expanded = False
        self._renderedEntryIds: list[int] = []
        self._streaming: AssistantMessageComponent | None = None
        self._indicator: WorkingStatusIndicator | None = None
        self._statusText = ""
        keybindings = KeybindingsManager.create()
        setKeybindings(keybindings)
        self._editor = CustomEditor(self._ui, getEditorTheme(), keybindings, EditorOptions(paddingX=1))
        self._editor.onSubmit = handlers["submit"]
        self._editor.onEscape = handlers["abort"]
        self._editor.onCtrlD = handlers["exit"]
        self._editor.onAction("app.clear", handlers["exit"])
        self._editor.onAction("app.model.select", handlers["selectModel"])
        self._editor.onAction("app.thinking.cycle", handlers["cycleThinking"])

        def toggleExpanded() -> None:
            self._expanded = not self._expanded
            for component in [*self._cards, *self._summaries]:
                component.setExpanded(self._expanded)
            self._ui.requestRender()

        self._editor.onAction("app.tools.expand", toggleExpanded)

        def followUp() -> None:
            text = jsTrim(self._editor.getText())
            if not text:
                return
            self._editor.setText("")
            handlers["followUp"](text)

        self._editor.onAction("app.message.followUp", followUp)

        self._editorContainer.addChild(self._editor)
        self._footer.addChild(self._footerStats)
        self._footer.addChild(self._footerHints)
        # One empty line between the transcript and everything below it.
        self._ui.addChild(self._chat)
        self._ui.addChild(Spacer(1))
        for component in (self._tasks, self._queue, self._notices, self._status, self._editorContainer, self._footer):
            self._ui.addChild(component)
        self._ui.setFocus(self._editor)

    @property
    def ui(self) -> TUI:
        return self._ui

    def start(self) -> None:
        self._ui.start()

    def stop(self) -> None:
        self._discardTools()
        if self._indicator is not None:
            self._indicator.dispose()
        self._ui.stop()

    def _discardTools(self) -> None:
        """Finish every card: a running bash card keeps a timer until it gets a final result."""
        for component in self._cards:
            component.updateResult({"content": [], "isError": False}, False)
        self._cards.clear()
        self._tools.clear()
        self._streamingCalls.clear()

    def mount(self, component: Any) -> None:
        self._editorContainer.clear()
        self._editorContainer.addChild(component)
        self._ui.setFocus(component)
        self._ui.requestRender()

    def restoreEditor(self) -> None:
        self.mount(self._editor)

    def apply(self, view: dict[str, Any]) -> None:
        conversation = view["conversation"]
        live = conversation["docs"].get("pi.live") or {}
        self._syncTranscript(conversation["entries"])
        generation = live.get("generation")
        message = None if generation is None else generation.get("message")
        # A partial without its entry was dropped, for example by a retry: render the transcript again.
        if message is None and self._streaming is not None:
            self._rebuild(conversation["entries"])
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
            details = slot.get("details")
            child = details.get("conversationId") if isinstance(details, dict) else None
            if slot.get("output") is None and child is not None:
                text = f"Subagent {child} is working. /agents switches to it."
                component.updateResult(
                    {"content": [{"type": "text", "text": text}], "details": slot.get("details"), "isError": False}, True
                )
            elif slot.get("output") is not None:
                component.updateResult(
                    {"content": [{"type": "text", "text": slot["output"]}], "details": slot.get("details"), "isError": False},
                    True,
                )
        self._syncTasks(view.get("tasks"))
        inbox = conversation["docs"].get("pi.inbox")
        self._syncQueue({"items": []} if inbox is None else inbox)
        self._syncNotices(view)
        self._editor.borderColor = theme.getThinkingBorderColor(agentOf(conversation).get("thinkingLevel") or "off")
        self._syncStatus(live)
        self._syncFooter(view)
        self._ui.requestRender()

    def _syncTasks(self, graph: dict[str, Any] | None) -> None:
        self._tasks.clear()
        if graph is None:
            return
        nodes = list(graph["tasks"].values())
        lines = [theme.fg("accent", f"Tasks ({len(nodes)} live, /tasks to hide)")]
        # A conversation-owned task sits under the task that owns its conversation, when that task is live.
        owned = {conversationId for node in nodes for conversationId in node["conversations"]}

        def children(node: dict[str, Any]) -> list[dict[str, Any]]:
            return [
                candidate
                for candidate in nodes
                if candidate.get("owner") == node["id"]
                or (candidate.get("owner") is None and candidate["conversationId"] in node["conversations"])
            ]

        def visit(node: dict[str, Any], depth: int) -> None:
            lines.append(f"{'  ' * (depth + 1)}{describeTask(node)}")
            for child in children(node):
                visit(child, depth + 1)

        for node in nodes:
            if node.get("owner") is None and node["conversationId"] not in owned:
                visit(node, 0)
        for line in lines:
            self._tasks.addChild(TruncatedText(theme.fg("muted", line), 1, 0))

    def _syncQueue(self, inbox: dict[str, Any]) -> None:
        self._queue.clear()
        for item in inbox["items"]:
            text = f"<{item['entry']['kind']}>" if item["mode"] == "write" else userText(item["content"])
            self._queue.addChild(TruncatedText(theme.fg("muted", f"[{item['mode']}] {text}"), 1, 0))

    def _syncNotices(self, view: dict[str, Any]) -> None:
        self._notices.clear()
        for item in view["notices"][-4:]:
            color: ThemeColor = "error" if item["level"] == "error" else "warning" if item["level"] == "warning" else "muted"
            self._notices.addChild(TruncatedText(theme.fg(color, item["message"]), 1, 0))

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
        self._status.clear()
        self._indicator = (
            WorkingStatusIndicator(self._ui, text, None, lambda part: self._editor.borderColor(part)) if text else None
        )
        if self._indicator is not None:
            self._status.addChild(self._indicator)

    def _syncFooter(self, view: dict[str, Any]) -> None:
        conversation = view["conversation"]
        agent = agentOf(conversation)
        usage = totalUsage(conversation["docs"].get("pi.usage") or {"models": {}, "tools": {}})
        stats: list[str] = []
        if usage["input"]:
            stats.append(f"↑{formatTokens(usage['input'])}")
        if usage["output"]:
            stats.append(f"↓{formatTokens(usage['output'])}")
        if usage["cacheRead"]:
            stats.append(f"R{formatTokens(usage['cacheRead'])}")
        if usage["cacheWrite"]:
            stats.append(f"W{formatTokens(usage['cacheWrite'])}")
        stats.append(f"${toFixed(usage['cost']['total'], 3)}")
        agentModel = agent.get("model")
        contextWindow = next(
            (
                model["contextWindow"]
                for model in view["models"]
                if agentModel is not None
                and model["provider"] == agentModel["provider"]
                and model["modelId"] == agentModel["modelId"]
            ),
            0,
        )
        if contextWindow > 0:
            tokens = contextTokens(conversation["entries"])
            percent = None if tokens is None else tokens / contextWindow * 100
            text = f"{'?' if percent is None else toFixed(percent, 1)}%/{formatTokens(contextWindow)}"
            stats.append(theme.fg("error", text) if percent is not None and percent > 90 else text)
        self._footerStats.setText(theme.fg("dim", f"{' '.join(stats)}  {view['session']['cwd']}"))
        model = "no model" if agentModel is None else f"{agentModel['provider']}/{agentModel['modelId']}"
        shown = next(
            (candidate for candidate in view["conversations"] if candidate["id"] == conversation["conversation"]["id"]), None
        )
        label = shown["label"] if shown is not None else f"conversation {conversation['conversation']['id']}"
        thinkingLevel = agent.get("thinkingLevel")
        self._footerHints.setText(
            theme.fg("dim" if label == "main" else "accent", label)
            + theme.fg(
                "dim",
                f" · {model} · thinking:{'off' if thinkingLevel is None else thinkingLevel} ({keyText('app.thinking.cycle')})"
                f" · {keyText('app.model.select')} or /model · /agents · /compact · /tasks"
                f" · {keyText('app.message.followUp')} follow-up · {keyText('app.clear')} exit",
            )
        )

    def _syncTranscript(self, entries: list[dict[str, Any]]) -> None:
        # Compaction and resets replace the head of the active transcript.
        if any(index >= len(entries) or entries[index]["id"] != id for index, id in enumerate(self._renderedEntryIds)):
            self._rebuild(entries)
        for entry in entries[len(self._renderedEntryIds) :]:
            self._addEntry(entry)
            self._renderedEntryIds.append(entry["id"])

    def _rebuild(self, entries: list[dict[str, Any]]) -> None:
        self._chat.clear()
        self._discardTools()
        self._summaries.clear()
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
            self._chat.addChild(Spacer(1))
            self._chat.addChild(UserMessageComponent(userText(message["content"])))
        elif entry["kind"] == "pi.assistant" and role == "assistant":
            component = self._streaming if self._streaming is not None else AssistantMessageComponent()
            if self._streaming is None:
                self._chat.addChild(component)
            self._streaming = None
            component.updateContent(message, False)
            # Only a tool-calling answer runs its calls; an aborted, failed, or truncated one never does.
            ran = message.get("stopReason") == "toolUse"
            for content in message["content"]:
                if content["type"] != "toolCall":
                    continue
                streamed = content["id"] in self._streamingCalls
                # Only cards the stream already showed are kept for calls that never run.
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
            summary = CompactionComponent(userText(message["content"]) if role == "user" else "", self._expanded)
            self._summaries.append(summary)
            self._chat.addChild(Spacer(1))
            self._chat.addChild(summary)
        elif entry["kind"] == "pi.reset":
            self._addText("[new context]")

    def _syncStreaming(self, message: dict[str, Any]) -> None:
        if self._streaming is None:
            self._streaming = AssistantMessageComponent()
            self._chat.addChild(self._streaming)
        self._streaming.updateContent(message, True)
        for content in message["content"]:
            if content["type"] != "toolCall":
                continue
            self._tool(content["name"], content["id"], content["arguments"], content["id"] not in self._streamingCalls)
            self._streamingCalls.add(content["id"])

    def _addText(self, text: str) -> None:
        self._chat.addChild(Spacer(1))
        self._chat.addChild(Text(theme.fg("muted", text), 1, 0))

    def _tool(self, name: str, callId: str, args: Any = None, fresh: bool = False) -> ToolExecutionComponent:
        """The card of a call; ``fresh`` starts a new one for a call ID an earlier turn used."""
        existing = None if fresh else self._tools.get(callId)
        if existing is not None:
            if args is not None:
                existing.updateArgs(args)
            return existing
        component = ToolExecutionComponent(
            name, callId, args if args is not None else {}, {}, DurableTui._renderers.get(name), self._ui, self._cwd
        )
        component.setExpanded(self._expanded)
        self._chat.addChild(component)
        self._cards.append(component)
        self._tools[callId] = component
        return component


def describeTask(node: dict[str, Any]) -> str:
    state = node["state"]
    if state["status"] == "waiting":
        status = f"waiting on {', '.join(str(task) for task in state['on'])}"
    elif state["status"] == "completing":
        status = f"completing ({state['outcome']})"
    else:
        status = f"{state['status']} {state['phase']}"
    flags = [flag for flag in ("background" if node.get("background") else "", "aborting" if node.get("abortRequested") else "") if flag]
    owned = f" owns conversation {', '.join(str(id) for id in node['conversations'])}" if node["conversations"] else ""
    return f"{node['kind']} #{node['id']}: {status}{f' [{', '.join(flags)}]' if flags else ''}{owned}"


def userText(content: Any) -> str:
    if isinstance(content, str):
        return content
    return "".join(block["text"] for block in content if block["type"] == "text")


def totalUsage(state: dict[str, Any]) -> dict[str, Any]:
    total: dict[str, Any] = {
        "input": 0,
        "output": 0,
        "cacheRead": 0,
        "cacheWrite": 0,
        "totalTokens": 0,
        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0},
    }
    for usage in [*state["models"].values(), *state["tools"].values()]:
        total["input"] += usage["input"]
        total["output"] += usage["output"]
        total["cacheRead"] += usage["cacheRead"]
        total["cacheWrite"] += usage["cacheWrite"]
        total["cost"]["total"] += usage["cost"]["total"]
    return total


def contextTokens(entries: list[dict[str, Any]]) -> int | None:
    """Context size from the newest successful answer after the newest compaction; unknown before one."""
    # Kept entries follow the summary in the view but are older than it; only later answers measure the new context.
    compacted = max([0, *(entry["id"] for entry in entries if entry["kind"] == "pi.compaction")])
    for entry in reversed(entries):
        model = entry.get("model")
        message = model[0] if model else None
        if entry["id"] < compacted or entry["kind"] != "pi.assistant" or message is None or message.get("role") != "assistant":
            continue
        if message.get("stopReason") in ("aborted", "error"):
            continue
        usage = message["usage"]
        return usage.get("totalTokens") or usage["input"] + usage["output"] + usage["cacheRead"] + usage["cacheWrite"]  # type: ignore[no-any-return]
    return None


async def runDurableTui(source: Any, controller: Any, settings: Any) -> None:
    """Run the demo's TUI until the user exits. ``source`` provides ``current()`` and ``subscribe(listener)``; ``controller``
    is the demo's controller (``runtime.py``); ``settings`` is pi's ``SettingsManager``."""
    setCapabilityOverrides(settings.getTerminalCapabilityOverrides())
    # The default theme until the controller resolves the user's theme against the terminal.
    initTheme()
    loop = asyncio.get_running_loop()
    exited: asyncio.Future[None] = loop.create_future()
    background: set[asyncio.Task[Any]] = set()
    holder: dict[str, DurableTui] = {}

    def exit() -> None:
        loop.call_soon_threadsafe(lambda: exited.done() or exited.set_result(None))

    def run(awaitable: Any) -> None:
        task = asyncio.ensure_future(awaitable)
        background.add(task)
        task.add_done_callback(background.discard)

    def selectModel() -> None:
        snapshot = source.current()
        current = agentOf(snapshot["conversation"]).get("model")

        def isCurrent(model: dict[str, Any]) -> bool:
            return current is not None and model["provider"] == current["provider"] and model["modelId"] == current["modelId"]

        items = [
            SelectItem(value=f"{model['provider']}/{model['modelId']}", label=model["modelId"], description=model["provider"])
            for model in sorted(snapshot["models"], key=lambda model: not isCurrent(model))
        ]

        def onSelect(value: str) -> None:
            holder["view"].restoreEditor()
            separator = value.index("/")
            run(controller.setModel({"provider": value[:separator], "modelId": value[separator + 1 :]}))

        holder["view"].mount(ListSelector("Select model:", items, onSelect, lambda: holder["view"].restoreEditor()))

    def selectConversation() -> None:
        snapshot = source.current()
        shownId = snapshot["conversation"]["conversation"]["id"]
        items = [
            SelectItem(
                value=str(candidate["id"]),
                label=candidate["label"],
                description=f"{'(shown) ' if candidate['id'] == shownId else ''}{candidate.get('title') or ''}",
            )
            for candidate in reversed(snapshot["conversations"])
        ]

        def onSelect(value: str) -> None:
            holder["view"].restoreEditor()
            run(controller.switchConversation(int(value)))

        holder["view"].mount(ListSelector("Switch to:", items, onSelect, lambda: holder["view"].restoreEditor()))

    def submit(text: str) -> None:
        trimmed = jsTrim(text)
        if not trimmed:
            return
        if trimmed == "/model":
            return selectModel()
        if trimmed == "/tasks":
            return run(controller.toggleTasks())
        if trimmed == "/agents":
            return selectConversation()
        if trimmed == "/compact" or trimmed.startswith("/compact "):
            instructions = jsTrim(trimmed[len("/compact") :])
            return run(controller.compact(instructions or None))
        run(controller.submit(trimmed, "steer"))

    view = DurableTui(
        source.current()["session"]["cwd"],
        {
            "submit": submit,
            "followUp": lambda text: run(controller.submit(text, "followUp")),
            "abort": lambda: run(controller.abort()),
            "exit": exit,
            "selectModel": selectModel,
            "cycleThinking": lambda: run(controller.cycleThinking()),
        },
    )
    holder["view"] = view

    # pi's theme handling: the theme setting (also light/dark pairs) resolved against the terminal.
    themes = InteractiveThemeController(
        view.ui,
        getSettingsManager=lambda: settings,
        showError=lambda message: print(message, file=sys.stderr),
        onChanged=lambda: view.ui.requestRender(),
    )
    unsubscribe = source.subscribe(lambda: view.apply(source.current()))
    view.start()
    themes.applyFromSettings()
    view.apply(source.current())
    try:
        await exited
    finally:
        unsubscribe()
        view.stop()
