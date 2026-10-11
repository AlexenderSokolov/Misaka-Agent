"""Write tool for creating and overwriting files."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from misaka.agent.types import AgentTool, AgentToolResult
from misaka.ai.types import TextContent
from misaka.core.extensions.types import ToolDefinition
from misaka.core.tools._common import _drain_worker, abort_race, write_file_text
from misaka.core.tools.file_mutation_queue import withFileMutationQueue
from misaka.core.tools.path_utils import resolveToCwd
from misaka.core.tools.renderers.write import writeRenderers
from misaka.core.tools.tool_definition_wrapper import wrapToolDefinition
from misaka.utils.values import read_field, signal_aborted


class WriteToolInput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    path: str = Field(description="Path to the file to write (relative or absolute)")
    content: str = Field(description="Content to write to the file")


writeToolSystemPromptContribution = {
    "snippet": "Create or overwrite files",
    "guidelines": ["Use write only for new files or complete rewrites."],
}


class WriteOperations(Protocol):
    writeFile: Callable[[str, str], Awaitable[None]]
    mkdir: Callable[[str], Awaitable[None]]


@dataclass(slots=True)
class WriteToolOptions:
    operations: WriteOperations | None = None


@dataclass(slots=True)
class _DefaultWriteOperations:
    async def writeFile(self, absolute_path: str, content: str) -> None:
        await asyncio.to_thread(write_file_text, absolute_path, content)   # follows symlinks; keeps the file's mode

    async def mkdir(self, directory: str) -> None:
        await asyncio.to_thread(os.makedirs, directory, exist_ok=True)


def _coerce_options(options: WriteToolOptions | Mapping[str, Any] | None) -> WriteToolOptions:
    if options is None:
        return WriteToolOptions()
    if isinstance(options, WriteToolOptions):
        return options
    return WriteToolOptions(operations=options.get("operations"))


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


def createWriteToolDefinition(
    cwd: str,
    options: WriteToolOptions | Mapping[str, Any] | None = None,
) -> ToolDefinition[WriteToolInput | dict[str, Any], None]:
    resolved_options = _coerce_options(options)
    operations = (
        resolved_options.operations
        if resolved_options.operations is not None
        else _DefaultWriteOperations()
    )

    async def execute(
        _tool_call_id: str,
        params: WriteToolInput | dict[str, Any],
        signal: Any | None = None,
        _on_update: Callable[[AgentToolResult], None] | None = None,
        ctx: Any = None,
    ) -> AgentToolResult:
        parsed = WriteToolInput.model_validate(params)
        absolute_path = resolveToCwd(parsed.path, read_field(ctx, "cwd") or cwd)
        _refuse_office_package(absolute_path, "write")
        directory = os.path.dirname(absolute_path)

        async def mutate() -> AgentToolResult:
            if signal_aborted(signal):
                raise RuntimeError("Operation aborted")

            aborted = False

            def throw_if_aborted() -> None:
                if aborted or signal_aborted(signal):
                    raise RuntimeError("Operation aborted")

            async def worker() -> AgentToolResult | None:
                try:
                    throw_if_aborted()
                    await operations.mkdir(directory)
                    throw_if_aborted()
                    await operations.writeFile(absolute_path, parsed.content)
                    throw_if_aborted()
                    return AgentToolResult(
                        content=[TextContent(text=f"Successfully wrote {len(parsed.content.encode('utf-8'))} bytes to {parsed.path}")],
                        details=None,
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
        name="write",
        label="write",
        description=(
            "Write content to a file. Creates the file if it doesn't exist, overwrites if it does. "
            "Automatically creates parent directories."
        ),
        promptSnippet=writeToolSystemPromptContribution["snippet"],
        promptGuidelines=list(writeToolSystemPromptContribution["guidelines"]),
        parameters=WriteToolInput,
        constrainedSampling={"type": "json_schema", "strict": "prefer"},
        execute=execute,
        **writeRenderers,
    )


def createWriteTool(cwd: str, options: WriteToolOptions | Mapping[str, Any] | None = None) -> AgentTool:
    return wrapToolDefinition(createWriteToolDefinition(cwd, options))



__all__ = [
    "WriteOperations",
    "WriteToolInput",
    "WriteToolOptions",
    "createWriteTool",
    "createWriteToolDefinition",
    "writeToolSystemPromptContribution",
]
