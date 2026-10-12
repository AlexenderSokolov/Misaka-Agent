"""Translated from pi's ``coding-agent/src/cli/experimental/commands/server.ts``.

The parsed invocation is a dict ``{"command": "server", "auth"?, "provider"?, "model"?, "pluginPackages"?, "serverId"?,
"sessionDir"?}``; an option that was not given leaves its key out. The command context provides ``runServer(command)``.
"""

from __future__ import annotations

from typing import Any

from misaka.cli.experimental.command import (
    Command,
    ParsedCommandInput,
    stringOption,
    valueOption,
)
from misaka.cli.experimental.command_options import (
    authTokenFileOption,
    authTokenOption,
    parseAuth,
    unsupportedOptions,
)
from misaka.protocol import isServerId

serverIdOption = valueOption(
    "--server-id",
    lambda value: (
        {"ok": True, "value": value}
        if isServerId(value)
        else {"ok": False, "error": f'Invalid --server-id "{value}"; expected a lowercase UUIDv4'}
    ),
)
sessionDirOption = stringOption("--session-dir")
providerOption = stringOption("--provider")
modelOption = stringOption("--model")
pluginPackageOption = stringOption("-e", {"repeatable": True})


def _build(input: ParsedCommandInput) -> dict[str, Any]:
    parsedAuth = parseAuth(input)
    auth, authErrors = parsedAuth.get("auth"), parsedAuth["errors"]
    serverId = input.value(serverIdOption)
    sessionDir = input.value(sessionDirOption)
    provider = input.value(providerOption)
    model = input.value(modelOption)
    pluginPackages = input.values(pluginPackageOption)
    modelErrors = ["--provider requires --model"] if provider is not None and model is None else []
    errors = [*authErrors, *modelErrors, *unsupportedOptions("server", input)]
    if len(errors) > 0:
        return {"ok": False, "errors": errors}
    command: dict[str, Any] = {"command": "server"}
    if auth is not None:
        command["auth"] = auth
    if provider is not None:
        command["provider"] = provider
    if model is not None:
        command["model"] = model
    if len(pluginPackages) > 0:
        command["pluginPackages"] = pluginPackages
    if serverId is not None:
        command["serverId"] = serverId
    if sessionDir is not None:
        command["sessionDir"] = sessionDir
    return {"ok": True, "command": command}


serverCommand = (
    Command("server")
    .option(serverIdOption)
    .option(sessionDirOption)
    .option(providerOption)
    .option(modelOption)
    .option(pluginPackageOption)
    .option(authTokenOption)
    .option(authTokenFileOption)
    .build(_build)
    .action(lambda command, context: context.runServer(command))
)
