"""Bash tool for command execution with streaming, truncation, and timeouts."""

from __future__ import annotations

import asyncio
import math
import os
import subprocess
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol, TypedDict

from pydantic import BaseModel, ConfigDict, Field, field_validator

from misaka.agent.types import AgentTool, AgentToolResult
from misaka.ai.types import TextContent
from misaka.core.extensions.types import ToolDefinition
from misaka.core.tools._common import _drain_worker, abort_race
from misaka.core.tools.output_accumulator import (
    OutputAccumulator,
    OutputAccumulatorOptions,
    OutputSnapshot,
)
from misaka.core.tools.renderers.bash import (
    BASH_UPDATE_THROTTLE_MS,
    createShellRenderers,
)
from misaka.core.tools.renderers.bash import (
    formatDuration as _format_duration,  # noqa: F401 - misaka/core/subagent imports this name, and that package is not edited
)
from misaka.core.tools.tool_definition_wrapper import wrapToolDefinition
from misaka.core.tools.truncate import (
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_LINES,
    TruncationResult,
    formatSize,
)
from misaka.utils.child_process import spawn_child_process, waitForChildProcess
from misaka.utils.shell import (
    ShellConfig,
    getShellConfig,
    getShellEnv,
    killProcessTree,
    normalize_command_for_stdin,
    trackDetachedChildPid,
    untrackDetachedChildPid,
)
from misaka.utils.values import read_field, semantic_boolean, signal_aborted

_MAX_TIMEOUT_MS = 2_147_483_647
# Output limit of `structuredContent.output`, which programmatic callers such as codemode scripts
# receive.
STRUCTURED_OUTPUT_MAX_BYTES = 1024 * 1024
_MAX_TIMEOUT_SECONDS = _MAX_TIMEOUT_MS / 1000

# Result for programmatic callers such as codemode scripts. A non-zero exit code is an error
# result for the model, but scripts still resolve to this value. `output` is not limited like the
# model-facing output: callers decide how much of it reaches the model.
BASH_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "output": {"type": "string", "description": "Combined stdout and stderr, possibly truncated"},
        "truncated": {"type": "boolean"},
        "full_output_path": {"type": "string", "description": "Full output, when truncated"},
        "exit_code": {"type": "number"},
        "wall_time_seconds": {"type": "number"},
    },
    "required": ["output", "truncated", "exit_code", "wall_time_seconds"],
}
_SESSION_ENV_KEYS = (
    "PI_SESSION_ID",
    "PI_SESSION_FILE",
    "PI_PROVIDER",
    "PI_MODEL",
    "PI_REASONING_LEVEL",
)
_SESSION_ENV_GUIDELINE = "You can inspect PI_* environment variables for current model and session details."

bashToolSystemPromptContribution = {
    "snippet": "Execute bash commands (ls, grep, find, etc.)",
    "guidelines": [_SESSION_ENV_GUIDELINE],
}


# MISAKA fork: pi's resolveTimeoutMs; asyncio counts seconds, so the checked value stays in seconds.
def _resolve_timeout_seconds(timeout: float | None) -> float | None:
    if timeout is None:
        return None
    if not math.isfinite(timeout) or timeout <= 0:
        raise RuntimeError("Invalid timeout: must be a finite number of seconds")
    if timeout > _MAX_TIMEOUT_SECONDS:
        raise RuntimeError(
            f"Invalid timeout: maximum is {_MAX_TIMEOUT_SECONDS} seconds"
        )
    return timeout


class BashToolInput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    command: str = Field(description="Shell command to execute")
    timeout: float | None = Field(default=None, description="Timeout in seconds (optional, no default timeout)")


# MISAKA fork: the background-capable input misaka's managed bash uses (CCB's run_in_background/description);
# pi's schema is BashToolInput alone.
class BackgroundBashToolInput(BashToolInput):
    # CCB semanticBoolean accepts exactly the two boolean strings, not 0/1/yes.
    timeout: float | None = Field(default=None, description="Foreground wait in seconds; managed commands default to BASH_DEFAULT_TIMEOUT_MS / 1000 (120 seconds if unset). Eligible commands continue in the background at timeout.")
    run_in_background: bool = Field(default=False, strict=True, description="Run in the background; use TaskOutput or TaskStop with the returned task ID")
    description: str | None = Field(default=None, description="Short description of this command")

    _semantic_boolean = field_validator("run_in_background", mode="before")(semantic_boolean)


@dataclass(slots=True)
class BashToolDetails:
    truncation: TruncationResult | None = None
    fullOutputPath: str | None = None


class BashExecOptions(TypedDict, total=False):
    onData: Callable[[bytes], None]
    signal: Any
    timeout: float
    env: dict[str, str]


class BashOperations(Protocol):
    async def exec(self, command: str, cwd: str, options: BashExecOptions) -> dict[str, int | None]: ...


@dataclass(slots=True)
class BashSpawnContext:
    command: str
    cwd: str
    env: dict[str, str]


type BashSpawnHook = Callable[[BashSpawnContext], BashSpawnContext]


@dataclass(slots=True)
class BashToolOptions:
    operations: BashOperations | None = None
    commandPrefix: str | None = None
    shellPath: str | None = None
    exposeSessionEnvironment: bool | None = True
    spawnHook: BashSpawnHook | None = None
    startBackground: Callable[..., AgentToolResult] | None = None
    registerForeground: Callable[..., Any] | None = None


@dataclass(slots=True)
class ShellToolConfig:
    name: str
    label: str
    shellName: str
    prompt: str
    promptSnippet: str
    promptGuidelines: tuple[str, ...] = ()
    # MISAKA fork: a default, where upstream requires the field; output files are named after misaka.
    tempFilePrefix: str = "misaka-shell"


@dataclass(slots=True)
class _LocalShellOperations:
    shellName: str
    resolveShellConfig: Callable[[], ShellConfig]

    async def exec(self, command: str, cwd: str, options: BashExecOptions) -> dict[str, int | None]:
        timeout = _resolve_timeout_seconds(options.get("timeout"))
        signal = options.get("signal")
        if signal_aborted(signal):
            raise RuntimeError("aborted")
        if not os.path.exists(cwd):
            raise RuntimeError(
                f"Working directory does not exist: {cwd}\n"
                f"Cannot execute {self.shellName} commands."
            )

        shell_config = self.resolveShellConfig()
        env = options["env"] if "env" in options else getShellEnv()
        on_data = options["onData"]
        command_from_stdin = shell_config.commandTransport == "stdin"

        process = await spawn_child_process(
            shell_config.shell,
            *(
                shell_config.args
                if command_from_stdin
                else [*shell_config.args, command]
            ),
            cwd=cwd,
            env=dict(env),
            stdin=subprocess.PIPE if command_from_stdin else subprocess.DEVNULL,
            start_new_session=os.name != "nt",
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            on_data=lambda _fd, data: on_data(data),
        )
        if process.pid is not None:
            trackDetachedChildPid(process.pid)

        stdin_error: BaseException | None = None
        if command_from_stdin:
            try:
                process.write_stdin(
                    normalize_command_for_stdin(command).encode("utf-8")
                )
            except OSError:
                pass
            except BaseException as error:  # noqa: BLE001 - preserve the local-operations contract
                stdin_error = error
            finally:
                try:
                    process.close_stdin()
                except OSError:
                    pass
                except BaseException as error:  # noqa: BLE001 - report writer failures after cleanup
                    stdin_error = error
        wait_task = asyncio.create_task(waitForChildProcess(process))
        timeout_task = (
            asyncio.create_task(asyncio.sleep(timeout)) if timeout is not None else None
        )
        timed_out = False

        async with abort_race(signal) as abort_task:
            primary_error: BaseException | None = None
            try:
                pending: set[asyncio.Task[Any]] = {wait_task}
                if abort_task is not None:
                    pending.add(abort_task)
                if timeout_task is not None:
                    pending.add(timeout_task)
                done, _pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)

                timed_out = timeout_task is not None and timeout_task in done
                abort_won = abort_task is not None and abort_task in done
                abort_requested = abort_won or signal_aborted(signal)
                if (timed_out or abort_requested) and process.pid is not None:
                    killProcessTree(process.pid)
                exit_code = await wait_task

                if signal_aborted(signal):
                    raise RuntimeError("aborted")
                if timed_out:
                    raise RuntimeError(f"timeout:{timeout}")
                # A signal-killed shell has no exit code. Use the standard shell convention so
                # callers do not mistake the termination for a successful command (pi #9577).
                # asyncio reports a signal death as the negative signal number.
                if exit_code is not None and exit_code < 0:
                    return {"exitCode": 128 + (-exit_code)}
                return {"exitCode": exit_code if exit_code is not None else 1}
            except BaseException as error:
                primary_error = error
                if isinstance(error, asyncio.CancelledError):
                    try:
                        if process.pid is not None:
                            killProcessTree(process.pid)
                    finally:
                        process.close()
                raise
            finally:
                async def cleanup() -> None:
                    tasks: list[asyncio.Task[Any]] = [wait_task]
                    for task in (abort_task, timeout_task):
                        if task is None:
                            continue
                        if not task.done():
                            task.cancel()
                        tasks.append(task)
                    await asyncio.gather(*tasks, return_exceptions=True)

                cancelled_during_cleanup = await _drain_worker(
                    asyncio.create_task(cleanup())
                )
                if process.pid is not None:
                    untrackDetachedChildPid(process.pid)
                if primary_error is None:
                    if cancelled_during_cleanup:
                        raise asyncio.CancelledError
                    if stdin_error is not None:
                        raise stdin_error


def createLocalShellOperations(
    shell_name: str,
    resolve_shell_config: Callable[[], ShellConfig],
) -> BashOperations:
    return _LocalShellOperations(
        shellName=shell_name,
        resolveShellConfig=resolve_shell_config,
    )


def createLocalBashOperations(options: Mapping[str, Any] | None = None) -> BashOperations:
    shell_path = options.get("shellPath") if options else None
    return createLocalShellOperations(
        "bash",
        lambda: getShellConfig(shell_path),
    )


def _coerce_options(options: BashToolOptions | Mapping[str, Any] | None) -> BashToolOptions:
    if options is None:
        return BashToolOptions()
    if isinstance(options, BashToolOptions):
        return options
    expose_session_environment = options.get("exposeSessionEnvironment")
    return BashToolOptions(
        operations=options.get("operations"),
        commandPrefix=options.get("commandPrefix"),
        shellPath=options.get("shellPath"),
        exposeSessionEnvironment=(
            True if expose_session_environment is None else expose_session_environment
        ),
        spawnHook=options.get("spawnHook"),
        startBackground=options.get("startBackground"),
        registerForeground=options.get("registerForeground"),
    )


def resolveSpawnContext(
    command: str,
    cwd: str,
    spawn_hook: BashSpawnHook | None,
    expose_session_environment: bool,
    ctx: Any,
) -> BashSpawnContext:
    env = {**getShellEnv()}
    for key in _SESSION_ENV_KEYS:
        env.pop(key, None)
    if expose_session_environment and ctx is not None:
        session_manager = read_field(ctx, "sessionManager")
        env["PI_SESSION_ID"] = read_field(session_manager, "getSessionId")()
        session_file = read_field(session_manager, "getSessionFile")()
        if session_file:
            env["PI_SESSION_FILE"] = session_file
        model = read_field(ctx, "model")
        if model is not None:
            env["PI_PROVIDER"] = read_field(model, "provider")
            env["PI_MODEL"] = read_field(model, "id")
        thinking_level = read_field(ctx, "thinkingLevel")
        if thinking_level:
            env["PI_REASONING_LEVEL"] = thinking_level
    # MISAKA fork: a skill's runtime (its venv, a remote host) shapes the environment the command runs in.
    runtime = getattr(read_field(ctx, "sessionManager"), "_skill_runtime", None) if ctx is not None else None
    if runtime is not None:
        env = runtime.execution_env(env)
    base_context = BashSpawnContext(command=command, cwd=cwd, env=env)
    return spawn_hook(base_context) if spawn_hook is not None else base_context


def appendStatus(text: str, status: str) -> str:
    return f"{text}\n\n{status}" if text else status


def createShellToolDefinition(
    cwd: str,
    config: ShellToolConfig,
    options: BashToolOptions | Mapping[str, Any] | None = None,
) -> ToolDefinition[dict[str, Any], BashToolDetails | None]:
    resolved_options = _coerce_options(options)
    expose_session_environment = (
        True
        if resolved_options.exposeSessionEnvironment is None
        else resolved_options.exposeSessionEnvironment
    )
    operations = resolved_options.operations
    if operations is None:
        operations = createLocalBashOperations(
            {"shellPath": resolved_options.shellPath}
        )

    input_model = BackgroundBashToolInput if resolved_options.startBackground is not None else BashToolInput

    async def execute(
        _tool_call_id: str,
        params: dict[str, Any],
        signal: Any | None = None,
        on_update: Callable[[AgentToolResult], None] | None = None,
        ctx: Any = None,
    ) -> AgentToolResult:
        parsed = input_model.model_validate(params)
        resolved_command = (
            f"{resolved_options.commandPrefix}\n{parsed.command}" if resolved_options.commandPrefix else parsed.command
        )
        spawn_context = resolveSpawnContext(
            resolved_command,
            read_field(ctx, "cwd") or cwd,
            resolved_options.spawnHook,
            expose_session_environment,
            ctx,
        )
        # MISAKA fork: a skill whose runtime is remote gets its host ready before the command runs.
        runtime = getattr(read_field(ctx, "sessionManager"), "_skill_runtime", None) if ctx is not None else None
        if runtime is not None and runtime.remote is not None:
            from misaka.utils.async_lifecycle import run_in_thread
            await run_in_thread(runtime.prepare_remote)
        if isinstance(parsed, BackgroundBashToolInput) and parsed.run_in_background:
            if signal_aborted(signal):
                raise RuntimeError("Command aborted")
            timeout = _resolve_timeout_seconds(parsed.timeout)
            # MISAKA fork: a backgrounded command has not finished, so its result carries a task
            # id rather than the declared `structuredContent`; the output arrives later.
            return resolved_options.startBackground(
                command=parsed.command, description=parsed.description, context=ctx,
                spawn_context=spawn_context, operations=operations, timeout=timeout,
            )
        output = OutputAccumulator(
            OutputAccumulatorOptions(tempFilePrefix=config.tempFilePrefix)
        )
        accepting_output = True
        loop = asyncio.get_running_loop()
        update_handle: asyncio.TimerHandle | None = None
        update_dirty = False
        last_update_at = 0.0

        def emit_output_update() -> None:
            nonlocal update_dirty, last_update_at, update_handle
            if on_update is None or not update_dirty:
                return
            update_dirty = False
            last_update_at = loop.time()
            update_handle = None
            snapshot = output.snapshot(persistIfTruncated=True)
            on_update(
                AgentToolResult(
                    content=[TextContent(text=snapshot.content or "")],
                    details=BashToolDetails(
                        truncation=snapshot.truncation if snapshot.truncation.truncated else None,
                        fullOutputPath=snapshot.fullOutputPath,
                    ),
                )
            )

        def clear_update_handle() -> None:
            nonlocal update_handle
            if update_handle is not None:
                update_handle.cancel()
                update_handle = None

        def schedule_output_update() -> None:
            nonlocal update_dirty, update_handle
            if on_update is None:
                return
            update_dirty = True
            delay = BASH_UPDATE_THROTTLE_MS / 1000 - (loop.time() - last_update_at)
            if delay <= 0:
                clear_update_handle()
                emit_output_update()
                return
            if update_handle is None:
                update_handle = loop.call_later(delay, emit_output_update)

        if on_update is not None:
            on_update(AgentToolResult(content=[], details=None))

        def handle_data(data: bytes) -> None:
            if not accepting_output:
                return
            output.append(data)
            schedule_output_update()

        async def finish_output() -> OutputSnapshot:
            nonlocal accepting_output
            accepting_output = False
            output.finish()
            clear_update_handle()
            emit_output_update()
            snapshot = output.snapshot(persistIfTruncated=True)
            await output.closeTempFile()
            return snapshot

        def format_output(snapshot: OutputSnapshot, empty_text: str = "(no output)") -> tuple[str, BashToolDetails | None]:
            truncation = snapshot.truncation
            text = snapshot.content or empty_text
            details: BashToolDetails | None = None
            if truncation.truncated:
                details = BashToolDetails(truncation=truncation, fullOutputPath=snapshot.fullOutputPath)
                start_line = truncation.totalLines - truncation.outputLines + 1
                end_line = truncation.totalLines
                if truncation.lastLinePartial:
                    last_line_size = formatSize(output.getLastLineBytes())
                    text += (
                        f"\n\n[Showing last {formatSize(truncation.outputBytes)} of line {end_line} "
                        f"(line is {last_line_size}). Full output: {snapshot.fullOutputPath}]"
                    )
                elif truncation.truncatedBy == "lines":
                    text += (
                        f"\n\n[Showing lines {start_line}-{end_line} of {truncation.totalLines}. "
                        f"Full output: {snapshot.fullOutputPath}]"
                    )
                else:
                    text += (
                        f"\n\n[Showing lines {start_line}-{end_line} of {truncation.totalLines} "
                        f"({formatSize(DEFAULT_MAX_BYTES)} limit). Full output: {snapshot.fullOutputPath}]"
                    )
            return text, details

        foreground = None
        timeout = parsed.timeout
        started_at = time.perf_counter()
        if resolved_options.registerForeground is not None:
            if signal_aborted(signal):
                raise RuntimeError("Command aborted")
            foreground = resolved_options.registerForeground(
                command=parsed.command, description=read_field(parsed, "description"), context=ctx,
                spawn_context=spawn_context, operations=operations, output=output, on_data=handle_data,
                timeout=timeout,
            )
        try:
            try:
                if foreground is not None:
                    result = await foreground.wait_foreground(signal, timeout, auto_background=True)
                    if result is None:
                        from misaka.core.subagent.shell import background_result
                        clear_update_handle()
                        # MISAKA fork: moved to the background, as above; no `structuredContent` yet.
                        return background_result(foreground, by_user=foreground.backgrounded_by_user)
                else:
                    result = await operations.exec(
                        spawn_context.command,
                        spawn_context.cwd,
                        {
                            "onData": handle_data,
                            "signal": signal,
                            "timeout": parsed.timeout,
                            "env": spawn_context.env,
                        },
                    )
                exit_code = result.get("exitCode")
            except Exception as error:
                snapshot = await finish_output()
                output_text, _details = format_output(snapshot, "")
                message = str(error)
                if message == "aborted":
                    raise RuntimeError(appendStatus(output_text, "Command aborted")) from None
                if message.startswith("timeout:"):
                    timeout_secs = message.split(":", 1)[1]
                    raise RuntimeError(appendStatus(output_text, f"Command timed out after {timeout_secs} seconds")) from None
                raise

            snapshot = await finish_output()
            output_text, details = format_output(snapshot)
            if exit_code is None:
                raise RuntimeError(appendStatus(output_text, "Command terminated without an exit code"))
            # `Math.round`: halves round up.
            wall_time_seconds = math.floor((time.perf_counter() - started_at) * 10 + 0.5) / 10
            full_output = await output.readFullOutput(STRUCTURED_OUTPUT_MAX_BYTES)
            structured_content: dict[str, Any] = {
                "output": full_output.content,
                "truncated": full_output.truncated,
                **(
                    {"full_output_path": snapshot.fullOutputPath}
                    if full_output.truncated and snapshot.fullOutputPath
                    else {}
                ),
                "exit_code": exit_code,
                "wall_time_seconds": wall_time_seconds,
            }
            if exit_code != 0:
                return AgentToolResult(
                    content=[TextContent(text=appendStatus(output_text, f"Command exited with code {exit_code}"))],
                    details=details,
                    structuredContent=structured_content,
                    isError=True,
                )
            return AgentToolResult(
                content=[TextContent(text=output_text)], details=details, structuredContent=structured_content
            )
        finally:
            clear_update_handle()

    return ToolDefinition(
        name=config.name,
        label=config.label,
        description=(
            f"Execute a {config.shellName} command in the current working directory. "
            "Returns stdout and stderr. "
            f"Output is truncated to last {DEFAULT_MAX_LINES} lines or {DEFAULT_MAX_BYTES // 1024}KB "
            "(whichever is hit first). If truncated, full output is saved to a temp file. "
            "Optionally provide a timeout in seconds."
            # MISAKA fork: the background sentence appears only when the session runs managed background commands.
            + (" run_in_background=true returns a task ID immediately; completion arrives automatically. "
               "TaskOutput reads output and TaskStop stops the command. Eligible foreground commands also continue "
               "in the background when the foreground wait times out (default 120 seconds; BASH_DEFAULT_TIMEOUT_MS overrides)." if resolved_options.startBackground is not None else "")
        ),
        promptSnippet=config.promptSnippet,
        promptGuidelines=(
            list(config.promptGuidelines) if expose_session_environment else None
        ),
        parameters=input_model,
        outputSchema=BASH_OUTPUT_SCHEMA,
        constrainedSampling={"type": "json_schema", "strict": "prefer"},
        execute=execute,
        **createShellRenderers(config.prompt),
    )


_BASH_TOOL_CONFIG = ShellToolConfig(
    name="bash",
    label="bash",
    shellName="bash",
    prompt="$",
    promptSnippet=bashToolSystemPromptContribution["snippet"],
    promptGuidelines=tuple(bashToolSystemPromptContribution["guidelines"]),
    # MISAKA fork: output files are named after misaka (upstream: `pi-bash`).
    tempFilePrefix="misaka-bash",
)


def createBashToolDefinition(
    cwd: str,
    options: BashToolOptions | Mapping[str, Any] | None = None,
) -> ToolDefinition[dict[str, Any], BashToolDetails | None]:
    return createShellToolDefinition(cwd, _BASH_TOOL_CONFIG, options)


def createBashTool(cwd: str, options: BashToolOptions | Mapping[str, Any] | None = None) -> AgentTool:
    definition = createBashToolDefinition(cwd, options)
    tool = wrapToolDefinition(definition)
    # Pi assigns these definition-only fields dynamically so registerTool(createBashTool(...))
    # keeps its system-prompt contribution.
    object.__setattr__(tool, "promptSnippet", definition.promptSnippet)
    object.__setattr__(tool, "promptGuidelines", definition.promptGuidelines)
    return tool



__all__ = [
    "BashOperations",
    "BashSpawnContext",
    "BashSpawnHook",
    "BashToolDetails",
    "BashToolInput",
    "BashToolOptions",
    "ShellToolConfig",
    "bashToolSystemPromptContribution",
    "createBashTool",
    "createBashToolDefinition",
    "createLocalBashOperations",
]

# Kept for misaka/core/subagent, which imports these names.
_resolve_spawn_context = resolveSpawnContext
create_local_bash_operations = createLocalBashOperations
