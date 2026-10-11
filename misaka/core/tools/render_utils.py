"""Translated from pi's ``core/tools/render-utils.ts`` (v1.1.0)."""

from __future__ import annotations

import builtins
import os
from collections.abc import Mapping
from typing import Any, TypedDict

from misaka.ui.tui.terminal_image import (
    getCapabilities,
    getImageDimensions,
    hyperlink,
    imageFallback,
    pathToFileURL,
)
from misaka.utils.ansi import stripAnsi
from misaka.utils.js import UNDEFINED, jsLength, jsonStringify, jsSlice, jsString
from misaka.utils.paths import resolvePath
from misaka.utils.shell import sanitizeBinaryOutput
from misaka.utils.values import read_field


def shortenPath(path: object) -> builtins.str:
    if not isinstance(path, builtins.str):
        return ""
    home = os.path.expanduser("~")
    if path.startswith(home):
        return f"~{path[len(home) :]}"
    return path


def linkPath(styledText: builtins.str, rawPath: builtins.str, cwd: builtins.str) -> builtins.str:
    if not getCapabilities().hyperlinks:
        return styledText
    absolutePath = resolvePath(rawPath, cwd)
    return hyperlink(styledText, pathToFileURL(absolutePath))


def str(value: object) -> builtins.str | None:
    if isinstance(value, builtins.str):
        return value
    if value is None or value is UNDEFINED:
        return ""
    return None


def argOf(args: object, key: builtins.str) -> Any:
    """MISAKA fork: ``args?.<key>`` on tool arguments, which arrive as dicts (or models from Python callers):
    ``UNDEFINED`` when ``args`` is missing or lacks the key, ``None`` for a JSON null."""
    if args is None or args is UNDEFINED:
        return UNDEFINED
    if isinstance(args, Mapping):
        return args.get(key, UNDEFINED)
    return getattr(args, key, UNDEFINED)


def replaceTabs(text: builtins.str) -> builtins.str:
    return text.replace("\t", "   ")


def normalizeDisplayText(text: builtins.str) -> builtins.str:
    return text.replace("\r", "")


def getTextOutput(result: Any | None, showImages: bool) -> builtins.str:
    """MISAKA fork: ``result`` and its blocks are read with ``read_field`` (models or dicts)."""
    if not result:
        return ""

    content = read_field(result, "content") or []
    textBlocks = [c for c in content if read_field(c, "type") == "text"]
    imageBlocks = [c for c in content if read_field(c, "type") == "image"]

    output = "\n".join(
        sanitizeBinaryOutput(stripAnsi(read_field(c, "text") or "")).replace("\r", "") for c in textBlocks
    )

    caps = getCapabilities()
    if len(imageBlocks) > 0 and (not caps.images or not showImages):

        def indicator(img: Any) -> builtins.str:
            mimeType = read_field(img, "mimeType") or "image/unknown"
            data = read_field(img, "data")
            dims = getImageDimensions(data, read_field(img, "mimeType")) if data and read_field(img, "mimeType") else None
            return imageFallback(mimeType, dims)

        imageIndicators = "\n".join(indicator(img) for img in imageBlocks)
        output = f"{output}\n{imageIndicators}" if output else imageIndicators

    return output


# MISAKA: misaka/core/subagent imports this name, and that package is not edited.
get_text_output = getTextOutput


class ToolRenderResultLike[TDetails](TypedDict):
    content: list[Any]
    details: TDetails


COLLAPSED_ARGS_CHARS = 100


def _stringifyOrString(value: object, indent: int | None = None) -> builtins.str:
    """``JSON.stringify(value, null, indent) ?? String(value)``."""
    try:
        return jsonStringify(value, indent)
    except (TypeError, ValueError):
        return jsString(value)


def formatToolCallWithArgs(title: builtins.str, args: object, theme: Any, expanded: bool) -> builtins.str:
    """Generic tool call header: the title followed by the arguments. Collapsed, they are `key=value`
    pairs on the title line, cut to `COLLAPSED_ARGS_CHARS`. Expanded, each is a `key: value`
    line below the title, with strings shown raw and continuation lines indented.
    """
    header = theme.fg("toolTitle", theme.bold(title))
    if args is None:
        return header
    entries: list[tuple[builtins.str, object]] = (
        [(builtins.str(key), value) for key, value in args.items()] if isinstance(args, Mapping) else [("args", args)]
    )
    if len(entries) == 0:
        return header
    if expanded:
        lines: list[builtins.str] = []
        for key, value in entries:
            text = value if isinstance(value, builtins.str) else _stringifyOrString(value, 2)
            lines.append(f"  {key}: {'\n    '.join(replaceTabs(text).replace('\r', '').split('\n'))}")
        joined = "\n".join(lines)
        return f"{header}\n{theme.fg('muted', joined)}"
    pairs = " ".join(f"{key}={_stringifyOrString(value)}" for key, value in entries)
    preview = f"{jsSlice(pairs, 0, COLLAPSED_ARGS_CHARS - 3)}..." if jsLength(pairs) > COLLAPSED_ARGS_CHARS else pairs
    return f"{header} {theme.fg('muted', preview)}"


def invalidArgText(theme: Any) -> builtins.str:
    return theme.fg("error", "[invalid arg]")


class RenderToolPathOptions(TypedDict, total=False):
    """MISAKA fork: a name for the inline type of ``renderToolPath``'s ``options``."""

    emptyFallback: builtins.str


def renderToolPath(
    rawPath: builtins.str | None,
    theme: Any,
    cwd: builtins.str,
    options: RenderToolPathOptions | None = None,
) -> builtins.str:
    if rawPath is None:
        return invalidArgText(theme)
    value = rawPath or (options.get("emptyFallback") if options is not None else None)
    if not value:
        return theme.fg("toolOutput", "...")
    return linkPath(theme.fg("accent", shortenPath(value)), value, cwd)
