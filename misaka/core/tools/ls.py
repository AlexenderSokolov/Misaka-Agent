"""Directory listing tool."""

from __future__ import annotations

import asyncio
import os
import stat
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, fields
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from misaka.agent.types import AgentTool, AgentToolResult
from misaka.ai.types import TextContent
from misaka.core.extensions.types import ToolDefinition
from misaka.core.tools._common import _ignore_background_task_result, abort_race
from misaka.core.tools.path_utils import resolveToCwd
from misaka.core.tools.renderers.ls import lsRenderers
from misaka.core.tools.tool_definition_wrapper import wrapToolDefinition
from misaka.core.tools.truncate import (
    DEFAULT_MAX_BYTES,
    TruncationOptions,
    TruncationResult,
    formatSize,
    truncateHead,
)
from misaka.utils.values import maybe_await, read_field, signal_aborted

T = TypeVar("T")


DEFAULT_LIMIT = 500


lsToolSystemPromptContribution = {
    "snippet": "List directory contents",
    "guidelines": [],
}


class LsToolInput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    path: str | None = Field(default=None, description="Directory to list (default: current directory)")
    limit: int | None = Field(default=None, description="Maximum number of entries to return (default: 500)")


@dataclass(slots=True)
class LsToolDetails:
    truncation: TruncationResult | None = None
    entryLimitReached: int | None = None


class LsOperations(Protocol):
    exists: Callable[[str], Awaitable[bool] | bool]
    stat: Callable[[str], Awaitable[os.stat_result | Any] | os.stat_result | Any]
    readdir: Callable[[str], Awaitable[list[str]] | list[str]]


@dataclass(slots=True)
class LsToolOptions:
    operations: LsOperations | None = None


@dataclass(slots=True)
class _DefaultLsOperations:
    async def exists(self, absolute_path: str) -> bool:
        return await asyncio.to_thread(os.path.exists, absolute_path)

    async def stat(self, absolute_path: str) -> os.stat_result:
        return await asyncio.to_thread(os.stat, absolute_path)

    async def readdir(self, absolute_path: str) -> list[str]:
        return await asyncio.to_thread(os.listdir, absolute_path)


def _coerce_options(options: LsToolOptions | Mapping[str, Any] | None) -> LsToolOptions:
    if options is None:
        return LsToolOptions()
    if isinstance(options, LsToolOptions):
        return options
    return LsToolOptions(operations=options.get("operations"))


def _is_directory(stat_result: Any) -> bool:
    if hasattr(stat_result, "is_dir"):
        return bool(stat_result.is_dir())
    if hasattr(stat_result, "isDirectory"):
        return bool(stat_result.isDirectory())
    mode = getattr(stat_result, "st_mode", None)
    if isinstance(mode, int):
        return stat.S_ISDIR(mode)
    raise TypeError("Unsupported stat result")


def _details_or_none(details: LsToolDetails) -> LsToolDetails | None:
    if any(getattr(details, field.name) is not None for field in fields(details)):
        return details
    return None


def createLsToolDefinition(
    cwd: str,
    options: LsToolOptions | Mapping[str, Any] | None = None,
) -> ToolDefinition[LsToolInput | dict[str, Any], LsToolDetails | None]:
    operations = _coerce_options(options).operations or _DefaultLsOperations()

    async def execute(
        _tool_call_id: str,
        params: LsToolInput | dict[str, Any],
        signal: Any | None = None,
        _on_update: Callable[[AgentToolResult], None] | None = None,
        ctx: Any = None,
    ) -> AgentToolResult:
        if signal_aborted(signal):
            raise RuntimeError("Operation aborted")

        parsed = LsToolInput.model_validate(params)
        dir_path = resolveToCwd(parsed.path or ".", read_field(ctx, "cwd") or cwd)
        effective_limit = parsed.limit if parsed.limit is not None else DEFAULT_LIMIT

        async def worker() -> AgentToolResult:
            if not await maybe_await(operations.exists(dir_path)):
                raise RuntimeError(f"Path not found: {dir_path}")

            stat_result = await maybe_await(operations.stat(dir_path))
            if not _is_directory(stat_result):
                raise RuntimeError(f"Not a directory: {dir_path}")

            try:
                entries = await maybe_await(operations.readdir(dir_path))
            except Exception as error:  # noqa: BLE001 - any readdir failure becomes the user-facing error
                raise RuntimeError(f"Cannot read directory: {error}") from None

            entries.sort(key=str.lower)
            results: list[str] = []
            entry_limit_reached = False
            for entry in entries:
                if len(results) >= effective_limit:
                    entry_limit_reached = True
                    break
                full_path = os.path.join(dir_path, entry)
                try:
                    entry_stat = await maybe_await(operations.stat(full_path))
                except Exception:  # noqa: BLE001, S112 - an entry that vanished or cannot be read is left out
                    continue
                results.append(f"{entry}/" if _is_directory(entry_stat) else entry)

            if not results:
                return AgentToolResult(content=[TextContent(text="(empty directory)")], details=None)

            raw_output = "\n".join(results)
            truncation = truncateHead(raw_output, TruncationOptions(maxLines=2**31 - 1))
            output = truncation.content
            details = LsToolDetails()
            notices: list[str] = []
            if entry_limit_reached:
                notices.append(f"{effective_limit} entries limit reached. Use limit={effective_limit * 2} for more")
                details.entryLimitReached = effective_limit
            if truncation.truncated:
                notices.append(f"{formatSize(DEFAULT_MAX_BYTES)} limit reached")
                details.truncation = truncation
            if notices:
                output += f"\n\n[{'. '.join(notices)}]"

            return AgentToolResult(
                content=[TextContent(text=output)],
                details=_details_or_none(details),
            )

        worker_task = asyncio.create_task(worker())
        async with abort_race(signal) as abort_task:
            if abort_task is None:
                return await worker_task

            done, _pending = await asyncio.wait({worker_task, abort_task}, return_when=asyncio.FIRST_COMPLETED)
            if abort_task in done and worker_task not in done:
                _ignore_background_task_result(worker_task)
                raise RuntimeError("Operation aborted")
            return await worker_task

    return ToolDefinition(
        name="ls",
        label="ls",
        description=(
            "List directory contents. Returns entries sorted alphabetically, with '/' suffix for directories. "
            "Includes dotfiles. Output is truncated to 500 entries or 50KB (whichever is hit first)."
        ),
        promptSnippet=lsToolSystemPromptContribution["snippet"],
        parameters=LsToolInput,
        execute=execute,
        **lsRenderers,
    )


def createLsTool(cwd: str, options: LsToolOptions | Mapping[str, Any] | None = None) -> AgentTool:
    return wrapToolDefinition(createLsToolDefinition(cwd, options))



__all__ = [
    "LsOperations",
    "LsToolDetails",
    "LsToolInput",
    "LsToolOptions",
    "createLsTool",
    "createLsToolDefinition",
    "lsToolSystemPromptContribution",
]
