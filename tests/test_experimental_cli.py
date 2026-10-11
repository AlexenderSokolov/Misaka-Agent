"""pi coding-agent's ``test/experimental-cli-command.test.ts`` and ``test/experimental-cli-resolution.test.ts``."""

from __future__ import annotations

from typing import Any

import pytest

from misaka.cli.experimental.cli import cli

UNSUPPORTED_SERVER_OPTIONS = "The experimental server command does not support existing CLI options yet"
UNSUPPORTED_CLIENT_OPTIONS = "The experimental client command does not support existing CLI options yet"


def test_parses_server_configuration():
    assert cli.parse(
        [
            "server",
            "--server-id",
            "00000000-0000-4000-8000-000000000001",
            "--session-dir",
            "~/pi-sessions",
            "--provider",
            "anthropic",
            "--model",
            "claude-sonnet-4-5",
            "-e",
            "./first-plugin",
            "-e=./second-plugin",
        ]
    ) == {
        "ok": True,
        "command": {
            "command": "server",
            "serverId": "00000000-0000-4000-8000-000000000001",
            "sessionDir": "~/pi-sessions",
            "provider": "anthropic",
            "model": "claude-sonnet-4-5",
            "pluginPackages": ["./first-plugin", "./second-plugin"],
        },
    }


def test_parses_provider_qualified_server_models():
    assert cli.parse(["server", "--model", "anthropic/claude-sonnet-4-5:high"]) == {
        "ok": True,
        "command": {"command": "server", "model": "anthropic/claude-sonnet-4-5:high"},
    }


def test_parses_client_transport_addresses():
    assert cli.parse(["client", "--connect", "unix:///tmp/pi.sock"]) == {
        "ok": True,
        "command": {"command": "client", "connect": {"transport": "unix", "path": "/tmp/pi.sock"}},
    }
    assert cli.parse(
        ["client", "--connect", "radius://00000000-0000-4000-8000-000000000001", "--session-id", "demo-1"]
    ) == {
        "ok": True,
        "command": {
            "command": "client",
            "connect": {"transport": "radius", "serverId": "00000000-0000-4000-8000-000000000001"},
            "sessionId": "demo-1",
        },
    }


@pytest.mark.parametrize(
    ("option", "property"), [("-c", "continue"), ("--continue", "continue"), ("-r", "resume"), ("--resume", "resume")]
)
def test_parses_client_session_selection(option: str, property: str):
    assert cli.parse(["client", option]) == {"ok": True, "command": {"command": "client", property: True}}


def test_parses_a_session_selection_followed_by_a_one_shot_prompt():
    assert cli.parse(["client", "-r", "Explain this project"]) == {
        "ok": True,
        "command": {"command": "client", "resume": True, "prompt": "Explain this project"},
    }


def test_parses_client_model_and_plugin_configuration():
    assert cli.parse(
        ["client", "--model", "anthropic/claude-sonnet-4-5:high", "-e", "./first-plugin", "-e", "./second-plugin"]
    ) == {
        "ok": True,
        "command": {
            "command": "client",
            "model": "anthropic/claude-sonnet-4-5:high",
            "pluginPackages": ["./first-plugin", "./second-plugin"],
        },
    }


@pytest.mark.parametrize(
    ("argv", "auth"),
    [
        (["--auth-token", "secret"], {"type": "token", "token": "secret"}),
        (["--auth-token-file", "/tmp/token"], {"type": "file", "path": "/tmp/token"}),
    ],
)
def test_parses_authentication_source(argv: list[str], auth: dict[str, str]):
    for command in ("server", "client"):
        result = cli.parse([command, *argv])
        assert result["ok"] is True
        assert result["command"]["command"] == command
        assert result["command"]["auth"] == auth


@pytest.mark.parametrize("command", ["server", "client"])
def test_permits_omitted_authentication(command: str):
    assert cli.parse([command]) == {"ok": True, "command": {"command": command}}


@pytest.mark.parametrize(
    ("argv", "error"),
    [
        (["client", "--listen", "unix:///tmp/pi.sock"], UNSUPPORTED_CLIENT_OPTIONS),
        (["server", "--listen", "unix:///tmp/pi.sock"], UNSUPPORTED_SERVER_OPTIONS),
        (["server", "--connect", "unix:///tmp/pi.sock"], UNSUPPORTED_SERVER_OPTIONS),
        (["client", "--connect", "ws://localhost:8080"], 'Unsupported --connect transport "ws:"'),
        (["client", "--connect", "radius://not-a-server"], "Radius transport address requires"),
        (["client", "--connect", "unix://relative.sock"], "Unix transport address must not include an authority"),
        (["client", "--connect", "unix:///tmp/pi.sock?wrong=value"], "Invalid --connect address"),
        (["client", "--provider", "anthropic"], "--provider requires --model"),
        (["client", "-c", "-r"], "--session-id, --continue, and --resume are mutually exclusive"),
        (["client", "--continue=true"], "--continue does not take a value"),
        (["server", "--provider", "anthropic"], "--provider requires --model"),
        (["server", "--server-id", "not-a-uuid"], "Invalid --server-id"),
        (["server", "--server-id"], "--server-id requires a value"),
        (["server", "--session-dir"], "--session-dir requires a value"),
        (["client", "-e"], "-e requires a value"),
        (
            ["server", "--session-dir", "/tmp/first", "--session-dir=/tmp/second"],
            "--session-dir may only be specified once",
        ),
        (["client", "--connect="], "--connect requires a value"),
    ],
)
def test_rejects_invalid_experimental_input(argv: list[str], error: str):
    result = cli.parse(argv)
    assert result["ok"] is False
    assert any(error in message for message in result["errors"])


def test_rejects_unsupported_options_without_parsing_them_through_the_stable_cli():
    assert cli.parse(["client", "--tui-mode", "wrong", "--model", "claude-sonnet"]) == {
        "ok": False,
        "errors": [UNSUPPORTED_CLIENT_OPTIONS],
    }


@pytest.mark.parametrize(
    "value",
    [
        "unix:///tmp/a b.sock",
        "unix:///tmp/../pi.sock",
        "unix:////tmp/pi.sock",
        "unix:///tmp/pi.sock#fragment",
        "unix:///tmp/%zz.sock",
        "unix:/tmp/pi.sock",
        "unix:relative.sock",
        "radius://00000000-0000-4000-8000-000000000001/extra",
        "radius://00000000-0000-4000-8000-000000000001:80",
        "radius://user@00000000-0000-4000-8000-000000000001",
        "radius://00000000-0000-4000-8000-000000000001?x=1",
        "no scheme",
        # The path percent-encode set holds ``^``, so ``URL.href`` differs from the input.
        "unix:///tmp/a^b.sock",
        "unix:///^",
        # Forbidden host code points, and an empty host with a port or credentials, make ``new URL`` throw.
        "radius://a^b",
        "unix://:80/tmp/a.sock",
        "unix://a^b/tmp/a.sock",
        "unix://@/tmp/a.sock",
        "http://a b",
        "http://",
        # A lone surrogate (a non-UTF-8 byte in argv) is read as U+FFFD, which the href percent-encodes.
        "unix:///tmp/" + chr(0xDCFF) + ".sock",
    ],
)
def test_rejects_addresses_that_are_not_in_canonical_form(value: str):
    result = cli.parse(["client", "--connect", value])
    assert result["ok"] is False
    assert any("--connect address" in message or "must not include" in message for message in result["errors"])


def test_an_address_with_leading_spaces_or_inner_tabs_is_read_like_new_url():
    result = cli.parse(["client", "--connect", " http://x"])
    assert result["errors"] == ['Unsupported --connect transport "http:"']
    result = cli.parse(["client", "--connect", "ht\ttp://x"])
    assert result["errors"] == ['Unsupported --connect transport "http:"']


def test_accepts_a_percent_encoded_unix_path_and_a_trailing_slash_on_a_radius_address():
    assert cli.parse(["client", "--connect", "unix:///tmp/pi%20test.sock"])["command"]["connect"] == {
        "transport": "unix",
        "path": "/tmp/pi test.sock",
    }
    assert cli.parse(["client", "--connect", "radius://00000000-0000-4000-8000-000000000001/"])["command"][
        "connect"
    ] == {"transport": "radius", "serverId": "00000000-0000-4000-8000-000000000001"}


# experimental-cli-resolution.test.ts


def test_requires_an_experimental_subcommand():
    assert cli.parse([]) == {"ok": False, "errors": ["Expected experimental command: server or client"]}


async def test_passes_server_options_to_the_command_action():
    calls: list[dict[str, Any]] = []

    class Context:
        def runServer(self, command: dict[str, Any]) -> None:
            calls.append(command)

        def runClient(self, command: dict[str, Any]) -> None:
            raise AssertionError("client must not run")

    result = await cli.execute(
        [
            "server",
            "--server-id",
            "00000000-0000-4000-8000-000000000001",
            "--session-dir",
            "./sessions",
            "--provider",
            "anthropic",
            "--model",
            "claude-sonnet-4-5",
        ],
        Context(),
    )

    command = {
        "command": "server",
        "serverId": "00000000-0000-4000-8000-000000000001",
        "sessionDir": "./sessions",
        "provider": "anthropic",
        "model": "claude-sonnet-4-5",
    }
    assert result == {"ok": True, "command": command}
    assert calls == [command]


@pytest.mark.parametrize("name", ["server", "client"])
async def test_executes_the_parsed_command(name: str):
    calls: dict[str, int] = {"server": 0, "client": 0}

    class Context:
        async def runServer(self, command: dict[str, Any]) -> None:
            calls["server"] += 1

        async def runClient(self, command: dict[str, Any]) -> None:
            calls["client"] += 1

    result = await cli.execute([name], Context())

    assert result == {"ok": True, "command": {"command": name}}
    assert calls == {"server": 1 if name == "server" else 0, "client": 1 if name == "client" else 0}
