"""Translated from pi's ``coding-agent/src/cli/experimental/cli.ts``: the ``experimental`` command group, whose
subcommands are ``server`` and ``client``. The context a command action receives provides ``runServer(command)`` and
``runClient(command)``."""

from __future__ import annotations

from misaka.cli.experimental.command import Command
from misaka.cli.experimental.commands.client import clientCommand
from misaka.cli.experimental.commands.server import serverCommand

experimentalCommand = Command("experimental").build(
    lambda _input: {"ok": False, "errors": ["Expected experimental command: server or client"]}
)

cli = experimentalCommand.command(serverCommand).command(clientCommand)
