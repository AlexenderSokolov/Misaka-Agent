"""Read tool for text and image files."""

from __future__ import annotations

import asyncio
import dataclasses
import io
import os
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any, Protocol, TextIO

from pydantic import BaseModel, ConfigDict, Field

from misaka.agent.types import AgentTool, AgentToolResult
from misaka.ai.types import Api, ImageContent, Model, TextContent
from misaka.core.extensions.types import ToolDefinition
from misaka.core.tools._common import (
    _ignore_background_task_result,
    abort_race,
    regular_file,
    whole_file_bytes,
)
from misaka.core.tools.path_utils import resolveReadPathAsync
from misaka.core.tools.renderers.read import readRenderers
from misaka.core.tools.tool_definition_wrapper import wrapToolDefinition
from misaka.core.tools.truncate import (
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_LINES,
    TruncationResult,
    formatSize,
    truncateHead,
)
from misaka.utils.image_process import ProcessImageOptions, processImage
from misaka.utils.image_resize import ImageResizeOptions
from misaka.utils.mime import detectSupportedImageMimeTypeFromFile
from misaka.utils.values import read_field, signal_aborted

readToolSystemPromptContribution = {
    "snippet": "Read file contents",
    "guidelines": ["Use read to examine files instead of cat or sed."],
}


class ReadToolInput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    path: str = Field(description="Path to the file to read (relative or absolute)")
    offset: int | None = Field(default=None, description="Line number to start reading from (1-indexed)")
    limit: int | None = Field(default=None, description="Maximum number of lines to read")
    cell_range: str | None = Field(
        default=None,
        description=(
            "Spreadsheet only: an A1-style range such as \"Sheet1!A3:D15\" to read just that "
            "region instead of the whole workbook. Use it to re-read part of a large sheet."
        ),
    )


# Result for programmatic callers such as codemode scripts: the text for text files, and an image
# block for images that codemode's `image()` accepts. `note` is the text that goes with the image,
# such as resize hints. Property descriptions are left out so the type stays on one line in tool
# descriptions.
readOutputSchema: dict[str, Any] = {
    "anyOf": [
        {"type": "string"},
        {
            "type": "object",
            "required": ["type", "data", "mimeType", "note"],
            "properties": {
                "type": {"const": "image", "type": "string"},
                "data": {"type": "string"},
                "mimeType": {"type": "string"},
                "note": {"type": "string"},
            },
        },
    ]
}

# MISAKA fork: `Static<typeof readOutputSchema>` is the text, or a mapping of the image block.
type ReadToolOutput = str | dict[str, str]


@dataclass(slots=True)
class ReadToolDetails:
    truncation: TruncationResult | None = None


class ReadOperations(Protocol):
    readFile: Callable[[str], Awaitable[bytes]]
    access: Callable[[str], Awaitable[None]]
    detectImageMimeType: Callable[[str], Awaitable[str | None | Any]] | None


@dataclass(slots=True)
class ReadToolOptions:
    autoResizeImages: bool = True
    # Used when the session's model declares no `inputLimits.images.resize` of its own.
    resizeOptions: ImageResizeOptions | None = None
    operations: ReadOperations | None = None


@dataclass(slots=True)
class _DefaultReadOperations:
    async def readFile(self, absolute_path: str) -> bytes:
        return await asyncio.to_thread(whole_file_bytes, absolute_path, "read")

    async def access(self, absolute_path: str) -> None:
        await asyncio.to_thread(regular_file, absolute_path)

    async def detectImageMimeType(self, absolute_path: str) -> str | None:
        return await detectSupportedImageMimeTypeFromFile(absolute_path)

    def openText(self, absolute_path: str) -> TextIO:
        # MISAKA fork: pi reads the whole file and splits it. Text is streamed instead (see
        # ``_scan_text``), decoded exactly as ``bytes.decode("utf-8", "replace")`` would be and
        # split only on "\n", so a multi-gigabyte log pages with offset/limit in bounded memory.
        return open(absolute_path, encoding="utf-8", errors="replace", newline="\n")


# MISAKA fork: an Office file is rendered in memory; past this size the rendering is refused.
MAX_OFFICE_READ_BYTES = 100 * 1024 * 1024
_SCAN_CHUNK_CHARS = 1024 * 1024
# The selection's first characters that decide ``truncate_head`` exactly as the whole selection
# would: anything longer is over the byte limit in both, and they cut at the same line.
_WINDOW_CHARS = DEFAULT_MAX_BYTES + 2


@dataclass(slots=True)
class _TextScan:
    total_lines: int          # len(text.split("\n"))
    window: str               # the first _WINDOW_CHARS characters of the selection
    selected_bytes: int       # UTF-8 bytes of the selection, "\n".join(lines[start:end])
    first_line_bytes: int     # UTF-8 bytes of the selection's first line
    last_char: str            # the selection's last character, "" when it is empty


def _nth_newline(text: str, start: int, count: int) -> int:
    """Index of the ``count``-th "\n" in ``text`` at or after ``start``; -1 if there are fewer."""
    index = start - 1
    for _ in range(count):
        index = text.find("\n", index + 1)
        if index < 0:
            return -1
    return index


def _scan_text(stream: Any, start_line: int, limit: int | None, signal: Any = None) -> _TextScan:
    """One pass over ``stream`` that measures ``"\n".join(lines[start_line:start_line + limit])``
    and keeps only its first characters. Lines outside the selection are counted, not split."""
    end_line = None if limit is None else start_line + limit
    line = 0                          # the line the next character belongs to
    window: list[str] = []
    window_chars = selected_bytes = first_line_bytes = 0
    first_line_open = True
    last_char = ""
    while chunk := stream.read(_SCAN_CHUNK_CHARS):
        if signal_aborted(signal):
            raise RuntimeError("Operation aborted")
        position = 0
        while position < len(chunk):
            if line < start_line:
                boundary = _nth_newline(chunk, position, start_line - line)
                if boundary < 0:
                    line += chunk.count("\n", position)
                    break
                line, position = start_line, boundary + 1
            elif end_line is not None and line >= end_line:
                line += chunk.count("\n", position)
                break
            else:
                stop = len(chunk) if end_line is None else _nth_newline(chunk, position, end_line - line)
                part = chunk[position:len(chunk) if stop < 0 else stop]
                if part:
                    selected_bytes += len(part.encode("utf-8"))
                    if first_line_open:
                        cut = part.find("\n")
                        first_line_bytes += len((part if cut < 0 else part[:cut]).encode("utf-8"))
                        first_line_open = cut < 0
                    if window_chars < _WINDOW_CHARS:
                        window.append(part[:_WINDOW_CHARS - window_chars])
                        window_chars += len(window[-1])
                    last_char = part[-1]
                    line += part.count("\n")
                if stop < 0 or stop == len(chunk):
                    break
                line, position = line + 1, stop + 1       # the newline that ends the selection
    return _TextScan(line + 1, "".join(window), selected_bytes, first_line_bytes, last_char)


def _coerce_options(options: ReadToolOptions | Mapping[str, Any] | None) -> ReadToolOptions:
    if options is None:
        return ReadToolOptions()
    if isinstance(options, ReadToolOptions):
        return options
    return ReadToolOptions(
        autoResizeImages=options.get("autoResizeImages", True),
        operations=options.get("operations"),
    )


def toReadOutput(content: list[Any]) -> ReadToolOutput:
    """The image block and its note, or the text for text files and images that could not be processed."""
    text = next((block.text for block in content if read_field(block, "type") == "text"), "")
    image = next((block for block in content if read_field(block, "type") == "image"), None)
    return {"type": "image", "data": image.data, "mimeType": image.mimeType, "note": text} if image else text


def getNonVisionImageNote(model: Model[Api] | None) -> str | None:
    if model is None or "image" in model.input:
        return None
    return "[Current model does not support images. The image will be omitted from this request.]"


def _office_format(absolute_path: str) -> str | None:
    """The Office renderer key for this path, or ``None``.

    A local import: ``documents.office`` pulls in openpyxl, and ``read`` is constructed for
    every session whether or not one ever opens a workbook.
    """
    from misaka.core.documents import office
    return (office.format_of(absolute_path)
            or office.soffice.LEGACY.get(os.path.splitext(absolute_path)[1].lower()))


async def _render_office(absolute_path: str, cell_range: str | None, workspace: str) -> str:
    """The document as markdown, off the event loop.

    Parsing a workbook and walking its cells is seconds of CPU on a large file, and the
    session is a single loop: doing it inline holds every other tool call still. Rendering
    goes through the cache, so paging one document with ``offset`` parses it once.
    ``cell_range`` deliberately bypasses the cache -- it is a different rendering of the
    same file, and keying the store on it would let one range's answer serve another's.
    """
    from misaka.core.documents import office

    size = (await asyncio.to_thread(os.stat, absolute_path)).st_size
    if size > MAX_OFFICE_READ_BYTES:
        raise RuntimeError(f"{absolute_path} is {formatSize(size)}; read renders Office files up to "
                           f"{formatSize(MAX_OFFICE_READ_BYTES)}. Use doc_read or bash for a file this large.")
    if cell_range:
        return await asyncio.to_thread(office.render, absolute_path, cell_range=cell_range)
    return await asyncio.to_thread(
        office.render_cached, absolute_path, lambda: office.render(absolute_path), workspace=workspace)


def _model_resize_options(model: Any) -> ImageResizeOptions | None:
    """The session model's `inputLimits.images.resize`, as the resizer's own options."""
    images = getattr(getattr(model, "inputLimits", None), "images", None)
    resize = getattr(images, "resize", None)
    if resize is None:
        return None
    return ImageResizeOptions(
        maxWidth=resize.maxWidth, maxHeight=resize.maxHeight, maxBytes=resize.maxBytes, jpegQuality=resize.jpegQuality
    )


def createReadToolDefinition(
    cwd: str,
    options: ReadToolOptions | Mapping[str, Any] | None = None,
) -> ToolDefinition[ReadToolInput | dict[str, Any], ReadToolDetails | None]:
    resolved_options = _coerce_options(options)
    auto_resize_images = resolved_options.autoResizeImages
    fallback_resize_options = resolved_options.resizeOptions
    operations = resolved_options.operations or _DefaultReadOperations()

    async def execute(
        _tool_call_id: str,
        params: ReadToolInput | dict[str, Any],
        signal: Any | None = None,
        _on_update: Callable[[AgentToolResult], None] | None = None,
        ctx: Any = None,
    ) -> AgentToolResult:
        parsed = ReadToolInput.model_validate(params)

        if signal_aborted(signal):
            raise RuntimeError("Operation aborted")

        async def worker() -> AgentToolResult:
            absolute_path = await resolveReadPathAsync(parsed.path, read_field(ctx, "cwd") or cwd)
            if signal_aborted(signal):
                return AgentToolResult(content=[], details=None)
            await operations.access(absolute_path)
            if signal_aborted(signal):
                return AgentToolResult(content=[], details=None)

            detect_mime = getattr(operations, "detectImageMimeType", None)
            mime_type = await detect_mime(absolute_path) if callable(detect_mime) else None
            content: list[TextContent | ImageContent]
            details: ReadToolDetails | None = None
            non_vision_image_note = getNonVisionImageNote(getattr(ctx, "model", None))

            if mime_type:
                buffer = await operations.readFile(absolute_path)
                processed = await processImage(
                    buffer,
                    mime_type,
                    ProcessImageOptions(
                        autoResizeImages=auto_resize_images,
                        resizeOptions=_model_resize_options(getattr(ctx, "model", None)) or fallback_resize_options,
                    ),
                )
                if not processed.ok:
                    text_note = f"Read image file [{mime_type}]\n{processed.message}"
                    if non_vision_image_note:
                        text_note += f"\n{non_vision_image_note}"
                    content = [TextContent(text=text_note)]
                else:
                    text_note = f"Read image file [{processed.mimeType}]"
                    if processed.hints:
                        text_note += "\n" + "\n".join(processed.hints)
                    if non_vision_image_note:
                        text_note += f"\n{non_vision_image_note}"
                    content = [
                        TextContent(text=text_note),
                        ImageContent(data=processed.data, mimeType=processed.mimeType),
                    ]
            else:
                # An Office file is rendered rather than decoded: its bytes are a zip, and
                # ``errors="replace"`` on them produces pages of replacement characters that
                # look like content and quote against nothing. The rendering is the corpus'
                # own (``documents/office``), so what is shown here is what ``doc_read``
                # would show -- a quotation copied out of this verifies against the indexed
                # document. Everything below is unchanged: offset/limit page the rendering
                # exactly as they page a text file.
                start_line = max(0, parsed.offset - 1) if parsed.offset else 0
                open_text = getattr(operations, "openText", None)
                if _office_format(absolute_path) is not None:
                    text_stream: Any = io.StringIO(await _render_office(absolute_path, parsed.cell_range, read_field(ctx, "cwd") or cwd))
                elif parsed.cell_range:
                    raise RuntimeError(
                        "cell_range names a sheet and a range, and this is not a spreadsheet. "
                        "Call read again without it."
                    )
                elif callable(open_text):
                    text_stream = await asyncio.to_thread(open_text, absolute_path)
                else:
                    buffer = await operations.readFile(absolute_path)
                    text_stream = io.StringIO(buffer.decode("utf-8", errors="replace"))
                with text_stream:
                    scan = await asyncio.to_thread(_scan_text, text_stream, start_line, parsed.limit, signal)
                total_file_lines = scan.total_lines
                start_line_display = start_line + 1
                if start_line >= total_file_lines:
                    raise RuntimeError(f"Offset {parsed.offset} is beyond end of file ({total_file_lines} lines total)")

                user_limited_lines: int | None = None
                end_line = total_file_lines
                if parsed.limit is not None:
                    end_line = min(start_line + parsed.limit, total_file_lines)
                    user_limited_lines = end_line - start_line

                # What truncate_head reports about the whole selection, from the window it needs
                # plus the scan's measurements (the same numbers it would count on the full text:
                # a selection ending in "\n" has no last line, an empty one has no lines).
                counted_lines = end_line - start_line - (scan.last_char == "\n") if scan.selected_bytes else 0
                truncation = dataclasses.replace(
                    truncateHead(scan.window), totalLines=counted_lines, totalBytes=scan.selected_bytes)
                if truncation.firstLineExceedsLimit:
                    first_line_size = formatSize(scan.first_line_bytes)
                    output_text = (
                        f"[Line {start_line_display} is {first_line_size}, exceeds {formatSize(DEFAULT_MAX_BYTES)} limit. "
                        f"Use bash: sed -n '{start_line_display}p' {parsed.path} | head -c {DEFAULT_MAX_BYTES}]"
                    )
                    details = ReadToolDetails(truncation=truncation)
                elif truncation.truncated:
                    end_line_display = start_line_display + truncation.outputLines - 1
                    next_offset = end_line_display + 1
                    output_text = truncation.content
                    if truncation.truncatedBy == "lines":
                        output_text += (
                            f"\n\n[Showing lines {start_line_display}-{end_line_display} of {total_file_lines}. "
                            f"Use offset={next_offset} to continue.]"
                        )
                    else:
                        output_text += (
                            f"\n\n[Showing lines {start_line_display}-{end_line_display} of {total_file_lines} "
                            f"({formatSize(DEFAULT_MAX_BYTES)} limit). Use offset={next_offset} to continue.]"
                        )
                    details = ReadToolDetails(truncation=truncation)
                elif user_limited_lines is not None and start_line + user_limited_lines < total_file_lines:
                    remaining = total_file_lines - (start_line + user_limited_lines)
                    next_offset = start_line + user_limited_lines + 1
                    output_text = (
                        f"{truncation.content}\n\n[{remaining} more lines in file. Use offset={next_offset} to continue.]"
                    )
                else:
                    output_text = truncation.content

                content = [TextContent(text=output_text)]

            if signal_aborted(signal):
                return AgentToolResult(content=[], details=None)
            return AgentToolResult(content=content, details=details)

        worker_task = asyncio.create_task(worker())
        async with abort_race(signal) as abort_task:
            if abort_task is None:
                return await worker_task

            done, _pending = await asyncio.wait({worker_task, abort_task}, return_when=asyncio.FIRST_COMPLETED)
            if abort_task in done and worker_task not in done:
                _ignore_background_task_result(worker_task)
                raise RuntimeError("Operation aborted")

            result = await worker_task
            if signal_aborted(signal):
                raise RuntimeError("Operation aborted")
            return result

    async def execute_with_output(
        tool_call_id: str,
        params: ReadToolInput | dict[str, Any],
        signal: Any | None = None,
        on_update: Callable[[AgentToolResult], None] | None = None,
        ctx: Any = None,
    ) -> AgentToolResult:
        # Every result carries `structuredContent` for programmatic callers (upstream's `.then()` on execute).
        result = await execute(tool_call_id, params, signal, on_update, ctx)
        return replace(result, structuredContent=toReadOutput(result.content))

    return ToolDefinition(
        name="read",
        label="read",
        description=(
            "Read the contents of a file. Supports text files, Office documents "
            "(xlsx, xlsm, csv, tsv, docx, docm, pptx, pptm), and images "
            "(jpg, png, gif, webp, bmp). "
            f"Images are sent as attachments. For text files, output is truncated to {DEFAULT_MAX_LINES} "
            f"lines or {DEFAULT_MAX_BYTES // 1024}KB (whichever is hit first). Use offset/limit for large files. "
            "When you need the full file, continue with offset until complete. A spreadsheet "
            "is rendered with real cell coordinates, its formulas and its sheet-level "
            "formatting; pass cell_range to re-read one region of a large sheet. A Word "
            "document keeps its headings, lists, tables in body order and its footnotes; a "
            "deck is rendered slide by slide with chart values and connector flow."
        ),
        promptSnippet=readToolSystemPromptContribution["snippet"],
        promptGuidelines=[*readToolSystemPromptContribution["guidelines"]],
        parameters=ReadToolInput,
        outputSchema=readOutputSchema,
        constrainedSampling={"type": "json_schema", "strict": "prefer"},
        execute=execute_with_output,
        **readRenderers,
    )


def createReadTool(cwd: str, options: ReadToolOptions | Mapping[str, Any] | None = None) -> AgentTool:
    return wrapToolDefinition(createReadToolDefinition(cwd, options))



__all__ = [
    "ReadOperations",
    "ReadToolDetails",
    "ReadToolInput",
    "ReadToolOptions",
    "createReadTool",
    "createReadToolDefinition",
    "readToolSystemPromptContribution",
]
