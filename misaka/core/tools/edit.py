"""Edit tool for exact and fuzzy-aware search/replace operations."""

from __future__ import annotations

import asyncio
import errno as errno_module
import json
import os
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from misaka.agent.types import AgentTool, AgentToolResult
from misaka.ai.types import TextContent
from misaka.core.extensions.types import ToolDefinition
from misaka.core.tools._common import (
    _drain_worker,
    abort_race,
    regular_file,
    whole_file_bytes,
    write_file_text,
)
from misaka.core.tools.edit_diff import (
    Edit,
    applyEditsToNormalizedContent,
    detectLineEnding,
    generateDiffString,
    generateUnifiedPatch,
    normalizeToLF,
    restoreLineEndings,
)
from misaka.core.tools.file_mutation_queue import withFileMutationQueue
from misaka.core.tools.path_utils import resolveToCwd
from misaka.core.tools.renderers.edit import editRenderers
from misaka.core.tools.tool_definition_wrapper import wrapToolDefinition
from misaka.utils.text import splitBom
from misaka.utils.values import read_field, signal_aborted


class ReplaceEditInput(BaseModel):
    # Upstream's schema does not set additionalProperties: false, and neither do the
    # other built-in tools.  Rejecting unknown keys would refuse arguments that
    # extensions legitimately annotate on a tool call.
    model_config = ConfigDict(extra="ignore")

    oldText: str = Field(
        description=(
            "Exact text for one targeted replacement. It must be unique in the original file "
            "and must not overlap with any other edits[].oldText in the same call."
        )
    )
    newText: str = Field(description="Replacement text for this targeted edit.")


editToolSystemPromptContribution = {
    "snippet": "Make precise file edits with exact text replacement, including multiple disjoint edits in one call",
    "guidelines": [
        "Use edit for precise changes (edits[].oldText must match exactly)",
        "When changing multiple separate locations in one file, use one edit call with multiple entries in edits[] instead of multiple edit calls",
        "Each edits[].oldText is matched against the original file, not after earlier edits are applied. Do not emit overlapping or nested edits. Merge nearby changes into one edit.",
        "Keep edits[].oldText as small as possible while still being unique in the file. Do not pad with large unchanged regions.",
    ],
}


class EditToolInput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    path: str = Field(description="Path to the file to edit (relative or absolute)")
    edits: list[ReplaceEditInput] = Field(
        description=(
            "One or more targeted replacements. Each edit is matched against the original file, not incrementally. "
            "Do not include overlapping or nested edits. If two changes touch the same block or nearby lines, "
            "merge them into one edit instead."
        )
    )


def _edit_tool_schema() -> dict[str, Any]:
    """Inline the one Pydantic ref that TypeBox does not emit in Pi's edit schema."""
    schema = EditToolInput.model_json_schema()
    definitions = schema.pop("$defs")
    schema["properties"]["edits"]["items"] = definitions["ReplaceEditInput"]
    return schema


@dataclass(slots=True)
class EditToolDetails:
    diff: str
    patch: str
    firstChangedLine: int | None = None


class EditOperations(Protocol):
    readFile: Callable[[str], Awaitable[bytes]]
    writeFile: Callable[[str, str], Awaitable[None]]
    access: Callable[[str], Awaitable[None]]


@dataclass(slots=True)
class EditToolOptions:
    operations: EditOperations | None = None




@dataclass(slots=True)
class _DefaultEditOperations:
    async def readFile(self, absolute_path: str) -> bytes:
        return await asyncio.to_thread(whole_file_bytes, absolute_path, "edit")

    async def writeFile(self, absolute_path: str, content: str) -> None:
        await asyncio.to_thread(write_file_text, absolute_path, content)   # follows symlinks; keeps the file's mode

    async def access(self, absolute_path: str) -> None:
        await asyncio.to_thread(regular_file, absolute_path, writable=True)


def _coerce_options(options: EditToolOptions | Mapping[str, Any] | None) -> EditToolOptions:
    if options is None:
        return EditToolOptions()
    if isinstance(options, EditToolOptions):
        return options
    return EditToolOptions(operations=options.get("operations"))


def prepareEditArguments(input_value: Any) -> Any:
    if not isinstance(input_value, dict):
        return input_value

    def _single_edit(value: Any) -> bool:
        # A single edit object (models often send this): wrap it in a one-element list (upstream #7835/ca21c1686)
        return (isinstance(value, dict) and isinstance(value.get("oldText"), str)
                and isinstance(value.get("newText"), str))

    args = input_value
    edits_value = args.get("edits")
    if isinstance(edits_value, str):
        try:
            parsed = json.loads(edits_value)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            args["edits"] = parsed
        elif _single_edit(parsed):
            args["edits"] = [parsed]
    elif _single_edit(edits_value):
        args["edits"] = [edits_value]

    # Legacy single-edit format: a top-level oldText/newText pair instead of edits[].
    # Fold it onto the end of edits[] and strip it, so replayed transcripts and older
    # callers still execute.  The pair only counts when both halves are strings.
    old_text = args.get("oldText")
    new_text = args.get("newText")
    if not isinstance(old_text, str) or not isinstance(new_text, str):
        return args

    edits = list(args["edits"]) if isinstance(args.get("edits"), list) else []
    edits.append({"oldText": old_text, "newText": new_text})
    rest = {key: value for key, value in args.items() if key not in ("oldText", "newText", "edits")}
    return {**rest, "edits": edits}


def validateEditInput(input_value: EditToolInput) -> tuple[str, list[Edit]]:
    if not input_value.edits:
        raise RuntimeError("Edit tool input is invalid. edits must contain at least one replacement.")
    return input_value.path, [Edit(oldText=edit.oldText, newText=edit.newText) for edit in input_value.edits]


def _format_access_error(error: BaseException) -> str:
    if isinstance(error, OSError) and error.errno is not None:
        code = errno_module.errorcode.get(error.errno)
        if code:
            return f"Error code: {code}"
    if isinstance(error, Exception):
        return f"Error: {error}"
    return str(error)


def _refuse_office_package(absolute_path: str, tool: str) -> None:
    """Send a .docx/.xlsx/.pptx to the office tool instead of corrupting it.

    These are zip packages. Decoding one as text and writing the result back produces a
    file no program will open, and both tools would report success -- the model finds out
    only when someone tries to open the deliverable. A local import: ``documents.office``
    pulls in openpyxl, and these tools are constructed for every session.
    """
    from misaka.core.documents import office

    suffix = os.path.splitext(absolute_path)[1].lower()
    if suffix in office.SUFFIXES and suffix not in _TEXT_OFFICE_SUFFIXES:
        raise RuntimeError(
            f"{tool} cannot write {suffix}: it is a zip package, not text, and writing it "
            f"as text corrupts it. Use the office tool, which edits it through structured "
            f"operations. To read it, use read."
        )


# .csv and .tsv are text and both tools have always handled them; only the packages are
# refused here.
_TEXT_OFFICE_SUFFIXES = frozenset({".csv", ".tsv"})


def createEditToolDefinition(
    cwd: str,
    options: EditToolOptions | Mapping[str, Any] | None = None,
) -> ToolDefinition[EditToolInput | dict[str, Any], EditToolDetails | None]:
    resolved_options = _coerce_options(options)
    operations = resolved_options.operations or _DefaultEditOperations()

    async def execute(
        _tool_call_id: str,
        input_value: EditToolInput | dict[str, Any],
        signal: Any | None = None,
        _on_update: Callable[[AgentToolResult], None] | None = None,
        ctx: Any = None,
    ) -> AgentToolResult:
        parsed = EditToolInput.model_validate(input_value)
        path, edits = validateEditInput(parsed)
        absolute_path = resolveToCwd(path, read_field(ctx, "cwd") or cwd)
        _refuse_office_package(absolute_path, "edit")

        async def mutate() -> AgentToolResult:
            if signal_aborted(signal):
                raise RuntimeError("Operation aborted")

            aborted = False

            async def worker() -> AgentToolResult | None:
                try:
                    try:
                        await operations.access(absolute_path)
                    except Exception as error:  # noqa: BLE001 - any access failure becomes the user-facing edit error
                        raise RuntimeError(f"Could not edit file: {path}. {_format_access_error(error)}.") from None

                    if aborted:
                        return None

                    buffer = await operations.readFile(absolute_path)
                    if aborted:
                        return None

                    try:
                        raw_content = buffer.decode("utf-8")
                    except UnicodeDecodeError as error:
                        raise RuntimeError(
                            f"Could not edit file: {path}. It is not valid UTF-8 text ({error}); "
                            "editing it as text would corrupt it. Use bash for binary files."
                        ) from None
                    split = splitBom(raw_content)
                    bom, content = split["bom"], split["text"]
                    original_ending = detectLineEnding(content)
                    normalized_content = normalizeToLF(content)
                    applied = applyEditsToNormalizedContent(normalized_content, edits, path)

                    if aborted:
                        return None

                    final_content = bom + restoreLineEndings(applied.newContent, original_ending)
                    await operations.writeFile(absolute_path, final_content)
                    if aborted:
                        return None

                    diff_result = generateDiffString(applied.baseContent, applied.newContent)
                    patch = generateUnifiedPatch(path, applied.baseContent, applied.newContent)
                    return AgentToolResult(
                        content=[TextContent(text=f"Successfully replaced {len(edits)} block(s) in {path}.")],
                        details=EditToolDetails(
                            diff=diff_result.diff,
                            patch=patch,
                            firstChangedLine=diff_result.firstChangedLine,
                        ),
                    )
                except Exception:
                    if aborted:
                        return None
                    raise

            worker_task = asyncio.create_task(worker())
            async with abort_race(signal) as abort_task:
                try:
                    if abort_task is None:
                        result = await asyncio.shield(worker_task)
                        if result is None:
                            raise RuntimeError("Operation aborted")
                        return result

                    done, _pending = await asyncio.wait({worker_task, abort_task}, return_when=asyncio.FIRST_COMPLETED)
                    if abort_task in done and worker_task not in done:
                        aborted = True
                        # A to-thread/custom write cannot be stopped by cancelling its Task.
                        # Hold the per-file lock until it really ends so it cannot overtake
                        # and overwrite a later mutation.
                        if await _drain_worker(worker_task):
                            raise asyncio.CancelledError
                        raise RuntimeError("Operation aborted")

                    result = await asyncio.shield(worker_task)
                    if aborted or result is None:
                        raise RuntimeError("Operation aborted")
                    return result
                except BaseException as error:
                    aborted = True
                    cancelled = await _drain_worker(worker_task)
                    if cancelled and not isinstance(error, asyncio.CancelledError):
                        raise asyncio.CancelledError from error
                    raise

        return await withFileMutationQueue(absolute_path, mutate)

    return ToolDefinition(
        name="edit",
        label="edit",
        description=(
            "Edit a single file using exact text replacement. Every edits[].oldText must match a unique, "
            "non-overlapping region of the original file. If two changes affect the same block or nearby lines, "
            "merge them into one edit instead of emitting overlapping edits. Do not include large unchanged regions "
            "just to connect distant changes."
        ),
        promptSnippet=editToolSystemPromptContribution["snippet"],
        promptGuidelines=[*editToolSystemPromptContribution["guidelines"]],
        parameters=_edit_tool_schema(),
        constrainedSampling={"type": "json_schema", "strict": "prefer"},
        renderShell="self",
        prepareArguments=prepareEditArguments,
        execute=execute,
        **editRenderers,
    )


def createEditTool(cwd: str, options: EditToolOptions | Mapping[str, Any] | None = None) -> AgentTool:
    return wrapToolDefinition(createEditToolDefinition(cwd, options))



__all__ = [
    "EditOperations",
    "EditToolDetails",
    "EditToolInput",
    "EditToolOptions",
    "createEditTool",
    "createEditToolDefinition",
    "editToolSystemPromptContribution",
]
