"""Translated from pi's ``coding-agent/src/experimental/commands.ts``.

Development-only command dispatch. Published entrypoints must not import this module.

MISAKA fork: ``runExperimentalCommand`` resolves to the process exit code (``0`` or ``1``) of the command it ran, or to
``None`` when ``args`` is not an experimental command, where upstream resolves to a boolean and sets ``process.exitCode``.
"""

from __future__ import annotations

import asyncio
import os
import signal
import sys
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

from misaka.cli.experimental.cli import cli
from misaka.core.experimental import areExperimentalFeaturesEnabled
from misaka.experimental.client import runClient
from misaka.experimental.client_tui import runClientTui
from misaka.experimental.process import onceSignal
from misaka.experimental.server import startForegroundServer
from misaka.utils.js import errorMessage


def _red(text: str) -> str:
    """chalk's default: colour on a terminal unless ``NO_COLOR`` is set; ``FORCE_COLOR`` decides on its own."""
    force = os.environ.get("FORCE_COLOR")
    colour = force not in ("0", "false") if force is not None else "NO_COLOR" not in os.environ and sys.stdout.isatty()
    return f"\x1b[31m{text}\x1b[39m" if colour else text


async def runServerCommand(command: dict[str, Any]) -> None:
    previousRelayStatus = ""
    relayOutputReady = False
    pendingRelayStatus: dict[str, Any] | None = None

    def reportRelayStatus(status: dict[str, Any]) -> None:
        nonlocal previousRelayStatus
        if status["status"] == "connected":
            description = "connected"
        elif status["status"] == "not_authenticated":
            description = "not connected; local only"
        elif status["status"] == "retrying":
            description = f"reconnecting: {status['error']}"
        else:
            description = "connecting"
        if description == previousRelayStatus or status["status"] == "connecting":
            return
        previousRelayStatus = description
        print(f"Radius: {description}", flush=True)

    def onRelayStatus(status: dict[str, Any]) -> None:
        nonlocal pendingRelayStatus
        if relayOutputReady:
            reportRelayStatus(status)
        else:
            pendingRelayStatus = status

    options: dict[str, Any] = {
        "pluginPackages": command.get("pluginPackages") or [],
        "onRelayStatus": onRelayStatus,
    }
    for key in ("serverId", "sessionDir", "provider", "model"):
        if command.get(key) is not None:
            options[key] = command[key]
    if command.get("auth") is not None:
        options["relayAuth"] = command["auth"]
    runtime = await startForegroundServer(options)
    print(f"Server: {runtime.serverId}", flush=True)
    print(f"Socket: {runtime.socketPath}", flush=True)
    relayOutputReady = True
    if pendingRelayStatus is not None:
        reportRelayStatus(pendingRelayStatus)
    loop = asyncio.get_running_loop()
    waiting: asyncio.Future[None] = loop.create_future()
    handled: list[Callable[[], None]] = []

    def finish() -> None:
        if not waiting.done():
            waiting.set_result(None)

    def onClosed(closed: asyncio.Future[None]) -> None:
        if waiting.done() or closed.cancelled():
            return
        error = closed.exception()
        if error is None:
            finish()
        else:
            waiting.set_exception(error)

    if sys.platform != "win32":
        for signalNumber in (signal.SIGINT, signal.SIGTERM):
            handled.append(onceSignal(signalNumber, finish))
    runtime.closed.add_done_callback(onClosed)
    try:
        await waiting
    finally:
        for off in handled:
            off()
        await runtime.close()


async def runClientCommand(command: dict[str, Any]) -> None:
    # ``process.stdin.isTTY === true``: a closed stdin (``sys.stdin`` is ``None``) is not a terminal.
    if command.get("prompt") is None and sys.stdin is not None and sys.stdin.isatty() and sys.stdout.isatty():
        await runClientTui(command)
        return
    result = await runClient(command)
    if result["kind"] == "attached":
        print(f"{result['serverId']}\t{result['sessionId']}\tattached")
        return
    if result["kind"] == "prompted":
        print(result["text"])
        return
    for session in result["sessions"]:
        print(f"{session['serverId']}\t{session['sessionId']}")


async def runExperimentalCommand(args: list[str]) -> int | None:
    """Run an experimental ``server`` or ``client`` command. Resolves to its exit code, or to ``None`` when ``args`` is not
    one or experimental features are off."""
    if not areExperimentalFeaturesEnabled() or args[:1] not in (["server"], ["client"]):
        return None
    try:
        result = await cli.execute(args, SimpleNamespace(runServer=runServerCommand, runClient=runClientCommand))
        if not result["ok"]:
            for error in result["errors"]:
                print(_red(f"Error: {error}"), file=sys.stderr)
            return 1
    except Exception as error:  # noqa: BLE001 - reported like a parse error
        print(_red(f"Error: {errorMessage(error)}"), file=sys.stderr)
        return 1
    return 0
