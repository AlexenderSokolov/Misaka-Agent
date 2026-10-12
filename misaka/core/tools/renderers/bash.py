"""Translated from pi's ``core/tools/renderers/bash.ts`` (v1.1.0).

Presentation for the shell tools.

Renderers live apart from the implementation so a process that only displays tool output does not
load the execution path or its typebox parameter schema. `bash.ts` spreads these into the shell
tool definition, so the tool's public shape is unchanged.

MISAKA fork: the renderer state is the dict ``ToolExecutionComponent`` keeps (``state["startedAt"]`` for
``state.startedAt``).
"""

from __future__ import annotations

import math
import time
from typing import Any, cast

from misaka.core.extensions.types import (
    ToolRenderContext,
    ToolRenderers,
    ToolRenderResultOptions,
)
from misaka.core.tools import render_utils
from misaka.core.tools.render_utils import argOf, getTextOutput, invalidArgText
from misaka.core.tools.truncate import DEFAULT_MAX_BYTES, formatSize
from misaka.modes.interactive.components.keybinding_hints import keyHint
from misaka.modes.interactive.components.visual_truncate import (
    VisualLinePreview,
    VisualLinePreviewOptions,
)
from misaka.modes.interactive.theme.theme import theme
from misaka.ui.tui import Container, Spacer, Text
from misaka.utils.js import (
    clearInterval,
    jsString,
    jsTrim,
    jsTrimEnd,
    setInterval,
    toFixed,
)
from misaka.utils.values import read_field

BASH_PREVIEW_LINES = 5
BASH_UPDATE_THROTTLE_MS = 100


def _now() -> int:
    """``Date.now()``"""
    return int(time.time() * 1000)


def formatDuration(ms: float) -> str:
    seconds = ms / 1000
    if seconds < 60:
        return f"{toFixed(seconds, 1)}s"

    totalSeconds = math.floor(seconds)
    minutes = math.floor(totalSeconds / 60)
    remainder = totalSeconds % 60
    if minutes < 60:
        return f"{minutes}m {remainder}s"

    return f"{math.floor(minutes / 60)}h {minutes % 60}m {remainder}s"


def formatShellCall(args: Any, prompt: str) -> str:
    command = render_utils.str(argOf(args, "command"))
    timeout = argOf(args, "timeout")
    timeoutSuffix = theme.fg("muted", f" (timeout {jsString(timeout)}s)") if timeout else ""
    commandDisplay = invalidArgText(theme) if command is None else command if command else theme.fg("toolOutput", "...")
    return theme.fg("toolTitle", theme.bold(f"{prompt} {commandDisplay}")) + timeoutSuffix


def rebuildBashResultRenderComponent(
    component: Container,
    result: Any,
    options: ToolRenderResultOptions,
    showImages: bool,
    startedAt: int | None,
    endedAt: int | None,
    durationMs: int | None,
) -> None:
    component.clear()

    output = jsTrim(getTextOutput(result, showImages))
    details = read_field(result, "details")
    truncation = read_field(details, "truncation")
    fullOutputPath = read_field(details, "fullOutputPath")
    if not options["isPartial"] and read_field(truncation, "truncated") and fullOutputPath and output.endswith("]"):
        footerStart = output.rfind("\n\n[")
        if footerStart != -1 and fullOutputPath in output[footerStart:]:
            output = jsTrimEnd(output[0:footerStart])

    if output:
        styledOutput = "\n".join(theme.fg("toolOutput", line) for line in output.split("\n"))

        if options["expanded"]:
            component.addChild(Text(f"\n{styledOutput}", 0, 0))
        else:
            component.addChild(Spacer(1))
            component.addChild(
                VisualLinePreview(
                    VisualLinePreviewOptions(
                        text=styledOutput,
                        maxVisualLines=BASH_PREVIEW_LINES,
                        keep="end",
                        formatHint=lambda hidden: theme.fg("muted", f"... ({hidden} earlier lines,")
                        + f" {keyHint('app.tools.expand', 'to expand')}{theme.fg('muted', ')')}",
                    )
                )
            )

    if read_field(truncation, "truncated") or fullOutputPath:
        warnings: list[str] = []
        if fullOutputPath:
            warnings.append(f"Full output: {fullOutputPath}")
        if read_field(truncation, "truncated"):
            if read_field(truncation, "truncatedBy") == "lines":
                warnings.append(
                    f"Truncated: showing {jsString(read_field(truncation, 'outputLines'))} of "
                    f"{jsString(read_field(truncation, 'totalLines'))} lines"
                )
            else:
                warnings.append(
                    f"Truncated: {jsString(read_field(truncation, 'outputLines'))} lines shown "
                    f"({formatSize(read_field(truncation, 'maxBytes', DEFAULT_MAX_BYTES))} limit)"
                )
        component.addChild(Text(f"\n{theme.fg('warning', f'[{'. '.join(warnings)}]')}", 0, 0))

    # A final result's recorded duration wins: it is monotonic and survives reloads. The renderer's own clock is the
    # fallback for live progress and for results stored without one.
    if not options["isPartial"] and durationMs is not None:
        component.addChild(Text(f"\n{theme.fg('muted', f'Took {formatDuration(durationMs)}')}", 0, 0))
    elif startedAt is not None:
        label = "Elapsed" if options["isPartial"] else "Took"
        endTime = endedAt if endedAt is not None else _now()
        component.addChild(Text(f"\n{theme.fg('muted', f'{label} {formatDuration(endTime - startedAt)}')}", 0, 0))


def createShellRenderers(prompt: str) -> ToolRenderers:
    """Shell renderers are shared by bash and powershell, which differ only in the prompt they display."""

    def renderCall(args: Any, _theme: Any, context: ToolRenderContext) -> Text:
        state = context.state
        if context.executionStarted and state.get("startedAt") is None:
            state["startedAt"] = _now()
            state["endedAt"] = None
        text = cast(Text, context.lastComponent) if context.lastComponent is not None else Text("", 0, 0)
        text.setText(formatShellCall(args, prompt))
        return text

    def renderResult(
        result: Any, options: ToolRenderResultOptions, _theme: Any, context: ToolRenderContext
    ) -> Container:
        state = context.state
        if state.get("startedAt") is not None and options["isPartial"] and not state.get("interval"):
            state["interval"] = setInterval(lambda: context.invalidate(), 1000)
        if not options["isPartial"] or context.isError:
            if state.get("endedAt") is None:
                state["endedAt"] = _now()
            if state.get("interval"):
                clearInterval(state["interval"])
                state["interval"] = None
        component = cast(Container, context.lastComponent) if context.lastComponent is not None else Container()
        rebuildBashResultRenderComponent(
            component,
            result,
            options,
            context.showImages,
            state.get("startedAt"),
            state.get("endedAt"),
            context.durationMs,
        )
        component.invalidate()
        return component

    return {"renderCall": renderCall, "renderResult": renderResult}
