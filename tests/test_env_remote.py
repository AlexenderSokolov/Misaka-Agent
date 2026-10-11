"""pi env's ``test/remote.test.ts``."""

from __future__ import annotations

import asyncio
import os
import signal
import sys
import tempfile
import time
from collections.abc import AsyncIterator, Callable
from typing import Any

import pytest

from misaka.chord.context import BACKGROUND_CONTEXT, withAbortSignal
from misaka.durable.env import getOrThrow
from misaka.env import Connection, RemoteExecutionEnv
from misaka.utils.js import AbortController
from tests.env_support import closeConnection, daemonCommand, removeTempDir

context = BACKGROUND_CONTEXT


@pytest.fixture
async def environments() -> AsyncIterator[Callable[[], tuple[RemoteExecutionEnv, Connection]]]:
    connections: list[Connection] = []
    dirs: list[str] = []

    def environment() -> tuple[RemoteExecutionEnv, Connection]:
        cwd = tempfile.mkdtemp(prefix="pi-env-remote-")
        dirs.append(cwd)
        connection = Connection({"command": daemonCommand()})
        connections.append(connection)
        return RemoteExecutionEnv({"connection": connection, "id": "pi-env:test", "cwd": cwd}), connection

    environment.connections = connections  # type: ignore[attr-defined]
    environment.dirs = dirs  # type: ignore[attr-defined]
    yield environment
    for connection in connections:
        await closeConnection(connection)
    for directory in dirs:
        removeTempDir(directory)


def killProcess(pid: int) -> None:
    """``process.kill(pid, "SIGKILL")``."""
    os.kill(pid, signal.SIGKILL if sys.platform != "win32" else signal.SIGTERM)


async def test_transfers_only_about_a_window_of_output_a_caller_keeps_with_exact_counts(environments):
    env, _ = environments()
    total = 5_000_000
    delivered = 0
    skipped = 0
    tail = ""

    def onOutput(text: str, _context: Any, info: dict[str, Any]) -> None:
        nonlocal delivered, skipped, tail
        delivered += len(text.encode("utf-8"))
        if info.get("skipped"):
            skipped += info["skipped"]["bytes"]
            tail = ""
        tail += text

    result = await env.exec(
        ["sh", "-c", f"yes 0123456789abcdef | head -c {total}"],
        {
            "window": {"maxBytes": 1000, "maxLines": 20, "minIntervalMs": 20, "bytesPerSecond": 1_000_000},
            "onOutput": onOutput,
        },
        context,
    )
    assert getOrThrow(result)["exitCode"] == 0
    assert delivered + skipped == total
    assert skipped > 0
    assert delivered < total / 10
    # After the last skip comes everything to the end, more than the window.
    full = ("0123456789abcdef\n" * -(-total // 17))[:total]
    assert full.endswith(tail)
    assert len(tail) > 1000 or len(tail.split("\n")) - 1 > 20


async def test_aborts_commands_cancelled_right_after_they_are_sent(environments):
    env, _ = environments()
    started = time.monotonic()
    loop = asyncio.get_running_loop()
    for attempt in range(20):
        controller = AbortController()
        running = asyncio.ensure_future(env.exec(["sleep", "5"], None, withAbortSignal(controller.signal, context)))
        if attempt % 2 == 0:
            controller.abort()
        else:
            loop.call_later((attempt % 5) / 1000, controller.abort)
        result = await running
        assert ("ok" if result["ok"] else result["error"].code) == "aborted"
    assert time.monotonic() - started < 10


async def test_times_out_a_command_that_produces_no_output(environments):
    env, _ = environments()
    started = time.monotonic()
    result = await env.exec(["sleep", "5"], {"timeout": 0.3}, context)
    assert ("ok" if result["ok"] else result["error"].code) == "timeout"
    assert time.monotonic() - started < 3


async def test_fails_requests_on_handles_of_a_lost_connection_instead_of_reaching_the_new_daemon(environments):
    env, connection = environments()
    getOrThrow(await env.writeFile("a.txt", "AAA", context))
    getOrThrow(await env.writeFile("b.txt", "BBB", context))
    a = getOrThrow(await env.openBinaryReader("a.txt", None, context))
    pid = (await connection.info())["pid"]
    killProcess(pid)
    await asyncio.sleep(0.3)
    # The new daemon numbers its handles from 1 again.
    b = getOrThrow(await env.openBinaryReader("b.txt", None, context))
    assert (await connection.info())["pid"] != pid
    stale = await a.read(0, 3, context)
    assert (bytes(stale["value"]).decode() if stale["ok"] else stale["error"].code) == "unknown"
    assert bytes(getOrThrow(await b.read(0, 3, context))).decode() == "BBB"
    await a.close(context)
    await b.close(context)


async def test_computes_the_command_at_each_start_and_retries_a_start_that_failed(environments):
    cwd = tempfile.mkdtemp(prefix="pi-env-remote-")
    environments.dirs.append(cwd)
    starts = 0

    async def command() -> list[str]:
        nonlocal starts
        starts += 1
        if starts == 1:
            raise RuntimeError("ssh gpu-box failed: Network is unreachable")
        return daemonCommand()

    connection = Connection({"command": command})
    environments.connections.append(connection)
    env = RemoteExecutionEnv({"connection": connection, "id": "pi-env:test", "cwd": cwd})
    offline = await env.readTextFile("missing.txt", context)
    assert ("ok" if offline["ok"] else [offline["error"].code, offline["error"].message]) == [
        "unknown",
        "ssh gpu-box failed: Network is unreachable",
    ]
    result = await env.exec(["sh", "-c", "exit 0"], None, context)
    assert (result["value"]["exitCode"] if result["ok"] else result["error"].code) == 0
    assert starts == 2


async def test_writes_and_reads_files_larger_than_one_transfer_chunk_in_order(environments):
    env, _ = environments()
    content = bytes((index * 7919) % 251 for index in range(3_500_017))
    getOrThrow(await env.writeFile("big.bin", content, context))
    getOrThrow(await env.appendFile("big.bin", content, context))
    read = getOrThrow(await env.readBinaryFile("big.bin", context))
    assert len(read) == len(content) * 2
    assert read[: len(content)] == content
    assert read[len(content) :] == content
    reader = getOrThrow(await env.openBinaryReader("big.bin", None, context))
    data = getOrThrow(await reader.read(1_000_003, 2_000_000, context))
    await reader.close(context)
    assert data == read[1_000_003:3_000_003]
