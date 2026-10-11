"""pi coding-agent's ``test/experimental-internal-process.test.ts``."""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
import tempfile
from collections.abc import AsyncIterator

import pytest

from misaka.experimental.process import (
    InternalProcess,
    onceSignal,
    spawnInternalProcess,
    terminateInternalProcess,
)
from tests.chat_support import waitFor

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="Unix sockets")


@pytest.fixture
async def children() -> AsyncIterator[tuple[set[InternalProcess], set[str]]]:
    spawned: set[InternalProcess] = set()
    directories: set[str] = set()
    yield spawned, directories
    for child in spawned:
        if child.pid is not None and child.exitCode is None and child.signalCode is None:
            exited: asyncio.Future[None] = asyncio.get_running_loop().create_future()
            child.once("exit", lambda *_, exited=exited: exited.done() or exited.set_result(None))
            child.kill("SIGTERM")
            await exited
    for directory in directories:
        shutil.rmtree(directory, ignore_errors=True)


async def canConnect(path: str) -> bool:
    try:
        _reader, writer = await asyncio.open_unix_connection(path)
    except OSError:
        return False
    writer.close()
    await writer.wait_closed()
    return True


async def test_starts_the_coordinator_through_the_current_runtime(children):
    spawned, directories = children
    # Keep Unix socket paths below macOS's short sun_path limit.
    directory = tempfile.mkdtemp(prefix="pi-ip-", dir="/tmp")
    directories.add(directory)
    publicPath = os.path.join(directory, "p.sock")
    controlPath = os.path.join(directory, "c.sock")
    child = spawnInternalProcess("coordinator", [publicPath, controlPath])
    spawned.add(child)

    await waitFor(lambda: canConnect(controlPath), 10_000)
    assert child.pid != os.getpid()


async def test_waits_for_a_failed_activation_child_to_terminate(children):
    spawned, directories = children
    directory = tempfile.mkdtemp(prefix="pi-ip-stop-", dir="/tmp")
    directories.add(directory)
    child = spawnInternalProcess("coordinator", [os.path.join(directory, "p.sock"), os.path.join(directory, "c.sock")])
    spawned.add(child)

    await waitFor(lambda: isinstance(child.pid, int))
    await terminateInternalProcess(child)
    assert child.signalCode == "SIGKILL"


async def test_reports_a_child_that_cannot_be_executed_through_the_error_event(children, monkeypatch):
    spawned, _directories = children
    monkeypatch.setattr(sys, "executable", "/nonexistent/python")
    errors: list[BaseException] = []
    child = spawnInternalProcess("coordinator", [])
    spawned.add(child)
    child.once("error", errors.append)

    await waitFor(lambda: len(errors) == 1)
    assert isinstance(errors[0], OSError)
    assert child.pid is None


async def test_reports_the_exit_of_a_child_that_fails_to_start(children):
    spawned, _directories = children
    exits: list[tuple[int | None, str | None]] = []
    child = spawnInternalProcess("coordinator", [], {"entry": "misaka.experimental.no_such_module"})
    spawned.add(child)
    child.once("exit", lambda code, signal: exits.append((code, signal)))

    await waitFor(lambda: len(exits) == 1)
    assert exits[0][0] != 0
    assert child.exitCode == exits[0][0]


async def test_a_once_signal_handler_serves_one_delivery_and_restores_the_default_action():
    import signal

    calls: list[int] = []
    onceSignal(signal.SIGUSR1, lambda: calls.append(1))
    os.kill(os.getpid(), signal.SIGUSR1)
    await waitFor(lambda: calls == [1])
    # Node's ``process.once``: the handler is gone, so a second signal takes the default action.
    assert signal.getsignal(signal.SIGUSR1) == signal.SIG_DFL


async def test_a_once_signal_handler_leaves_the_lock_cleanup_in_charge_of_the_next_termination():
    import signal

    from misaka.utils import lockfile

    calls: list[int] = []
    previous = signal.getsignal(signal.SIGTERM)
    onceSignal(signal.SIGTERM, lambda: calls.append(1))
    try:
        os.kill(os.getpid(), signal.SIGTERM)
        await waitFor(lambda: calls == [1])
        assert signal.getsignal(signal.SIGTERM) == lockfile._onSignal
    finally:
        signal.signal(signal.SIGTERM, previous)
