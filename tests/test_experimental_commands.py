"""The experimental command dispatch (pi's ``coding-agent/src/experimental/commands.ts``; pi has no test for it). The server and
client behind the commands are replaced by doubles, so nothing here needs a socket.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from misaka.experimental import commands


async def test_runs_nothing_unless_experiments_are_on_and_the_command_is_experimental(monkeypatch):
    monkeypatch.setenv("PI_EXPERIMENTAL", "0")
    assert await commands.runExperimentalCommand(["server"]) is None
    monkeypatch.setenv("PI_EXPERIMENTAL", "1")
    assert await commands.runExperimentalCommand(["chat"]) is None
    assert await commands.runExperimentalCommand([]) is None


async def test_reports_a_parse_error_and_fails(monkeypatch, capsys):
    monkeypatch.setenv("PI_EXPERIMENTAL", "1")
    monkeypatch.setenv("NO_COLOR", "1")

    assert await commands.runExperimentalCommand(["server", "--server-id", "invalid"]) == 1
    captured = capsys.readouterr()
    assert "Error: Invalid --server-id" in captured.err
    assert captured.out == ""


async def test_reports_an_error_a_command_raises(monkeypatch, capsys):
    monkeypatch.setenv("PI_EXPERIMENTAL", "1")
    monkeypatch.setenv("NO_COLOR", "1")

    async def failing(command: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("no server")

    monkeypatch.setattr(commands, "runClient", failing)
    assert await commands.runExperimentalCommand(["client"]) == 1
    assert "Error: no server" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        ({"kind": "list", "sessions": [{"serverId": "s", "sessionId": "a"}, {"serverId": "s", "sessionId": "b"}]}, "s\ta\ns\tb\n"),
        ({"kind": "attached", "serverId": "s", "sessionId": "a"}, "s\ta\tattached\n"),
        ({"kind": "prompted", "serverId": "s", "sessionId": "a", "text": "the answer"}, "the answer\n"),
    ],
)
async def test_prints_what_the_client_resolves_to(monkeypatch, capsys, result, expected):
    monkeypatch.setenv("PI_EXPERIMENTAL", "1")
    seen: list[dict[str, Any]] = []

    async def runClient(command: dict[str, Any]) -> dict[str, Any]:
        seen.append(command)
        return result

    monkeypatch.setattr(commands, "runClient", runClient)

    assert await commands.runExperimentalCommand(["client", "--session-id", "a", "--", "hi"]) == 0
    assert capsys.readouterr().out == expected
    assert seen == [{"command": "client", "sessionId": "a", "prompt": "hi"}]


async def test_a_server_prints_where_it_listens_reports_the_relay_once_and_stops_when_it_closes(monkeypatch, capsys):
    monkeypatch.setenv("PI_EXPERIMENTAL", "1")
    started: dict[str, Any] = {}
    closed: asyncio.Future[None] = asyncio.get_running_loop().create_future()
    closes: list[int] = []

    async def startForegroundServer(options: dict[str, Any]) -> SimpleNamespace:
        started.update(options)
        # The relay reports before the server has printed where it listens: that status waits.
        options["onRelayStatus"]({"status": "connecting"})
        options["onRelayStatus"]({"status": "not_authenticated"})

        async def close() -> None:
            closes.append(1)

        return SimpleNamespace(serverId="the-id", socketPath="/the/socket", closed=closed, close=close)

    monkeypatch.setattr(commands, "startForegroundServer", startForegroundServer)

    running = asyncio.ensure_future(
        commands.runExperimentalCommand(
            ["server", "--server-id", "00000000-0000-4000-8000-000000000001", "-e", "./plugin", "--provider", "p", "--model", "m"]
        )
    )
    await asyncio.sleep(0.05)
    assert not running.done()
    for status in (
        {"status": "retrying", "error": "boom"},
        {"status": "retrying", "error": "boom"},
        {"status": "connected"},
    ):
        started["onRelayStatus"](status)
    closed.set_result(None)
    assert await running == 0

    assert capsys.readouterr().out.splitlines() == [
        "Server: the-id",
        "Socket: /the/socket",
        "Radius: not connected; local only",
        "Radius: reconnecting: boom",
        "Radius: connected",
    ]
    assert closes == [1]
    assert started["serverId"] == "00000000-0000-4000-8000-000000000001"
    assert started["pluginPackages"] == ["./plugin"]
    assert (started["provider"], started["model"]) == ("p", "m")
    assert "relayAuth" not in started


async def test_a_server_that_fails_to_close_reports_the_failure(monkeypatch, capsys):
    monkeypatch.setenv("PI_EXPERIMENTAL", "1")
    monkeypatch.setenv("NO_COLOR", "1")
    closed: asyncio.Future[None] = asyncio.get_running_loop().create_future()

    async def startForegroundServer(options: dict[str, Any]) -> SimpleNamespace:
        async def close() -> None:
            return None

        return SimpleNamespace(serverId="id", socketPath="/s", closed=closed, close=close)

    monkeypatch.setattr(commands, "startForegroundServer", startForegroundServer)
    running = asyncio.ensure_future(commands.runExperimentalCommand(["server"]))
    await asyncio.sleep(0.05)
    closed.set_exception(RuntimeError("the server broke"))

    assert await running == 1
    assert "Error: the server broke" in capsys.readouterr().err
