"""Glob-based file discovery tool."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, fields
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from misaka.agent.types import AgentTool, AgentToolResult
from misaka.ai.types import TextContent
from misaka.core.extensions.types import ToolDefinition
from misaka.core.tools._common import _ignore_background_task_result, abort_race
from misaka.core.tools.path_utils import resolveToCwd
from misaka.core.tools.renderers.find import findRenderers
from misaka.core.tools.tool_definition_wrapper import wrapToolDefinition
from misaka.core.tools.truncate import (
    DEFAULT_MAX_BYTES,
    TruncationOptions,
    TruncationResult,
    formatSize,
    truncateHead,
)
from misaka.utils.tools_manager import find_tool, missing_tool_message
from misaka.utils.values import maybe_await, read_field, signal_aborted

T = TypeVar("T")


def relativizeFindResultPath(
    resultPath: str,
    searchPath: str,
    pathModule: Any | None = None,
) -> str:
    """Relativize a find result against the search root and normalize it to posix separators (pi #7569/523b5a491).

    Absolute paths go through relpath: prefix slicing loses the first segment at the
    root ("/etc" under "/" becomes "tc") and mistakes shared-prefix siblings
    (/foo-bar vs /foo) for children. Relative paths pass through unchanged: the custom
    glob already yields relative results, and relpath would wrongly resolve them
    against cwd.
    """
    paths = os.path if pathModule is None else pathModule
    had_trailing_separator = resultPath.endswith(paths.sep) or (
        paths.sep == "\\" and resultPath.endswith("/")
    )
    is_absolute = paths.isabs(resultPath) or (
        paths.sep == "\\" and resultPath.startswith(("\\", "/"))
    )
    if is_absolute:
        try:
            relative_path = paths.relpath(resultPath, searchPath)
        except ValueError:
            # Match win32.relative() across drives/UNC roots and root-relative paths.
            def node_normalize(value: str) -> str:
                normalized = paths.normpath(value)
                if paths.sep != "\\":
                    return normalized
                drive, tail = paths.splitdrive(normalized)
                if tail or not drive.startswith("\\\\"):
                    return normalized
                drive_parts = [
                    part for part in drive[2:].split(paths.sep) if part
                ]
                if not drive_parts or drive_parts[0] in {"?", "."}:
                    return normalized
                if len(drive_parts) == 2:
                    return normalized + paths.sep
                return normalized

            relative_path = node_normalize(resultPath)
            source_path = node_normalize(searchPath)
            target_parts = [
                part for part in relative_path.lstrip(paths.sep).split(paths.sep) if part
            ]
            source_parts = [
                part for part in source_path.lstrip(paths.sep).split(paths.sep) if part
            ]
            common = 0
            for source_part, target_part in zip(source_parts, target_parts):
                if source_part.lower() != target_part.lower():
                    break
                common += 1
            if not target_parts:
                relative_path = paths.sep.join([".."] * len(source_parts))
            elif common:
                relative_path = paths.sep.join(
                    [".."] * (len(source_parts) - common) + target_parts[common:]
                )
        if relative_path == paths.curdir:
            # Node's path.relative(path, path) returns "", while Python returns ".".
            relative_path = ""
    else:
        relative_path = resultPath

    posix_path = relative_path.replace(paths.sep, "/")
    if had_trailing_separator and not posix_path.endswith("/"):
        return f"{posix_path}/"
    return posix_path


findToolSystemPromptContribution = {
    "snippet": "Find files by glob pattern (respects .gitignore)",
    "guidelines": [],
}


class FindToolInput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    pattern: str = Field(
        description="Glob pattern to match files, e.g. '*.ts', '**/*.json', or 'src/**/*.spec.ts'"
    )
    path: str | None = Field(default=None, description="Directory to search in (default: current directory)")
    limit: int | None = Field(default=None, description="Maximum number of results (default: 1000)")


DEFAULT_LIMIT = 1000


@dataclass(slots=True)
class FindToolDetails:
    truncation: TruncationResult | None = None
    resultLimitReached: int | None = None


class FindOperations(Protocol):
    exists: Callable[[str], Awaitable[bool] | bool]
    glob: Callable[[str, str, dict[str, Any]], Awaitable[list[str]] | list[str]]


@dataclass(slots=True)
class FindToolOptions:
    operations: FindOperations | None = None


def _coerce_options(options: FindToolOptions | Mapping[str, Any] | None) -> FindToolOptions:
    if options is None:
        return FindToolOptions()
    if isinstance(options, FindToolOptions):
        return options
    return FindToolOptions(operations=options.get("operations"))


def _details_or_none(details: FindToolDetails) -> FindToolDetails | None:
    if any(getattr(details, field.name) is not None for field in fields(details)):
        return details
    return None


def _is_inside_git_repo(search_path: str) -> bool:
    current = search_path
    while True:
        if os.path.exists(os.path.join(current, ".git")):
            return True
        parent = os.path.dirname(current)
        if parent == current:
            return False
        current = parent


async def _run_fd_search(fd_path: str, args: list[str], signal: Any | None) -> tuple[bytes, bytes, int | None]:
    try:
        process = await asyncio.create_subprocess_exec(
            fd_path,
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except Exception as error:  # noqa: BLE001 - any spawn failure is reported as 'failed to run fd'
        raise RuntimeError(f"Failed to run fd: {error}") from None

    communicate_task = asyncio.create_task(process.communicate())
    async with abort_race(signal) as abort_task:
        if abort_task is not None:
            done, _pending = await asyncio.wait({communicate_task, abort_task}, return_when=asyncio.FIRST_COMPLETED)
            if abort_task in done and not communicate_task.done():
                process.kill()
                await communicate_task
                raise RuntimeError("Operation aborted")

        stdout, stderr = await communicate_task

    if signal_aborted(signal):
        raise RuntimeError("Operation aborted")

    return stdout, stderr, process.returncode


def createFindToolDefinition(
    cwd: str,
    options: FindToolOptions | Mapping[str, Any] | None = None,
) -> ToolDefinition[FindToolInput | dict[str, Any], FindToolDetails | None]:
    custom_ops = _coerce_options(options).operations

    async def execute(
        _tool_call_id: str,
        params: FindToolInput | dict[str, Any],
        signal: Any | None = None,
        _on_update: Callable[[AgentToolResult], None] | None = None,
        ctx: Any = None,
    ) -> AgentToolResult:
        if signal_aborted(signal):
            raise RuntimeError("Operation aborted")

        parsed = FindToolInput.model_validate(params)
        search_path = resolveToCwd(parsed.path or ".", read_field(ctx, "cwd") or cwd)
        # MISAKA fork: MISAKA's credentials and the web evidence vault are out of the tools' reach.
        from misaka.core.web.evidence import check_material_read
        check_material_read(search_path)
        effective_limit = parsed.limit if parsed.limit is not None else DEFAULT_LIMIT

        if custom_ops is not None and callable(getattr(custom_ops, "glob", None)):

            async def run_custom_search() -> AgentToolResult:
                if not await maybe_await(custom_ops.exists(search_path)):
                    raise RuntimeError(f"Path not found: {search_path}")
                if signal_aborted(signal):
                    raise RuntimeError("Operation aborted")

                results = await maybe_await(
                    custom_ops.glob(
                        parsed.pattern,
                        search_path,
                        {
                            "ignore": ["**/node_modules/**", "**/.git/**"],
                            "limit": effective_limit,
                        },
                    )
                )
                if signal_aborted(signal):
                    raise RuntimeError("Operation aborted")
                if not results:
                    return AgentToolResult(
                        content=[TextContent(text="No files found matching pattern")],
                        details=None,
                    )

                relativized = [
                    relativizeFindResultPath(path_value, search_path)
                    for path_value in results
                ]
                result_limit_reached = len(relativized) >= effective_limit
                raw_output = "\n".join(relativized)
                truncation = truncateHead(
                    raw_output, TruncationOptions(maxLines=2**31 - 1)
                )
                result_output = truncation.content
                details = FindToolDetails()
                notices: list[str] = []
                if result_limit_reached:
                    notices.append(f"{effective_limit} results limit reached")
                    details.resultLimitReached = effective_limit
                if truncation.truncated:
                    notices.append(f"{formatSize(DEFAULT_MAX_BYTES)} limit reached")
                    details.truncation = truncation
                if notices:
                    result_output += f"\n\n[{'. '.join(notices)}]"
                return AgentToolResult(
                    content=[TextContent(text=result_output)],
                    details=_details_or_none(details),
                )

            worker_task = asyncio.create_task(run_custom_search())
            try:
                async with abort_race(signal) as abort_task:
                    if abort_task is None:
                        return await worker_task

                    done, _pending = await asyncio.wait(
                        {worker_task, abort_task}, return_when=asyncio.FIRST_COMPLETED
                    )
                    if abort_task in done or signal_aborted(signal):
                        if worker_task in done:
                            await asyncio.gather(worker_task, return_exceptions=True)
                        else:
                            _ignore_background_task_result(worker_task)
                        raise RuntimeError("Operation aborted")

                    result = await worker_task
                    if signal_aborted(signal):
                        raise RuntimeError("Operation aborted")
                    return result
            except asyncio.CancelledError:
                worker_task.cancel()
                await asyncio.gather(worker_task, return_exceptions=True)
                raise

        fd_path = find_tool("fd")
        if signal_aborted(signal):
            raise RuntimeError("Operation aborted")
        if not fd_path:
            raise RuntimeError(missing_tool_message("fd"))

        args: list[str] = ["--glob", "--color=never", "--hidden"]
        repo_probe = asyncio.create_task(
            asyncio.to_thread(_is_inside_git_repo, search_path)
        )
        try:
            async with abort_race(signal) as abort_task:
                if abort_task is not None:
                    done, _pending = await asyncio.wait(
                        {repo_probe, abort_task}, return_when=asyncio.FIRST_COMPLETED
                    )
                    if abort_task in done and repo_probe not in done:
                        raise RuntimeError("Operation aborted")
                inside_git_repo = await repo_probe
        finally:
            if not repo_probe.done():
                repo_probe.cancel()
            await asyncio.gather(repo_probe, return_exceptions=True)
        if signal_aborted(signal):
            raise RuntimeError("Operation aborted")
        if not inside_git_repo:
            args.append("--no-require-git")
        args.extend(["--max-results", str(effective_limit)])

        effective_pattern = parsed.pattern
        if "/" in parsed.pattern:
            args.append("--full-path")
            if (
                not parsed.pattern.startswith("/")
                and not parsed.pattern.startswith("**/")
                and parsed.pattern != "**"
            ):
                effective_pattern = f"**/{parsed.pattern}"
            if os.name == "nt":
                effective_pattern = effective_pattern.replace("/", r"[/\\]")
        args.extend(["--", effective_pattern, search_path])

        stdout, stderr, return_code = await _run_fd_search(fd_path, args, signal)
        lines = stdout.decode("utf-8", errors="replace").splitlines()
        stderr_text = stderr.decode("utf-8", errors="replace").strip()
        output = "\n".join(lines)

        if return_code != 0 and not output:
            raise RuntimeError(stderr_text or f"fd exited with code {return_code}")
        if not output:
            return AgentToolResult(content=[TextContent(text="No files found matching pattern")], details=None)

        relativized: list[str] = []
        for raw_line in lines:
            line = raw_line.rstrip("\r").strip()
            if not line:
                continue
            relativized.append(relativizeFindResultPath(line, search_path))

        result_limit_reached = len(relativized) >= effective_limit
        raw_output = "\n".join(relativized)
        truncation = truncateHead(raw_output, TruncationOptions(maxLines=2**31 - 1))
        result_output = truncation.content
        details = FindToolDetails()
        notices: list[str] = []
        if result_limit_reached:
            notices.append(
                f"{effective_limit} results limit reached. Use limit={effective_limit * 2} for more, or refine pattern"
            )
            details.resultLimitReached = effective_limit
        if truncation.truncated:
            notices.append(f"{formatSize(DEFAULT_MAX_BYTES)} limit reached")
            details.truncation = truncation
        if notices:
            result_output += f"\n\n[{'. '.join(notices)}]"

        return AgentToolResult(
            content=[TextContent(text=result_output)],
            details=_details_or_none(details),
        )

    return ToolDefinition(
        name="find",
        label="find",
        description=(
            "Search for files by glob pattern. Returns matching file paths relative to the search directory. "
            "Respects .gitignore. Output is truncated to 1000 results or 50KB (whichever is hit first)."
        ),
        promptSnippet=findToolSystemPromptContribution["snippet"],
        parameters=FindToolInput,
        execute=execute,
        **findRenderers,
    )


def createFindTool(cwd: str, options: FindToolOptions | Mapping[str, Any] | None = None) -> AgentTool:
    return wrapToolDefinition(createFindToolDefinition(cwd, options))



__all__ = [
    "FindOperations",
    "FindToolDetails",
    "FindToolInput",
    "FindToolOptions",
    "createFindTool",
    "createFindToolDefinition",
    "findToolSystemPromptContribution",
    "relativizeFindResultPath",
]
