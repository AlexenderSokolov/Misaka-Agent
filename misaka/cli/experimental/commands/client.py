"""Translated from pi's ``coding-agent/src/cli/experimental/commands/client.ts``.

The parsed invocation is a dict ``{"command": "client", "auth"?, "connect"?, "sessionId"?, "continue"?, "resume"?,
"provider"?, "model"?, "pluginPackages"?, "prompt"?}``; an option that was not given leaves its key out. The command
context provides ``runClient(command)``.
"""

from __future__ import annotations

from typing import Any

from misaka.cli.experimental.command import (
    Command,
    ParsedCommandInput,
    flagOption,
    stringOption,
)
from misaka.cli.experimental.command_options import (
    authTokenFileOption,
    authTokenOption,
    connectOption,
    parseAuth,
    unsupportedOptions,
)

sessionIdOption = stringOption("--session-id")
continueOption = flagOption("--continue")
continueShortOption = flagOption("-c")
resumeOption = flagOption("--resume")
resumeShortOption = flagOption("-r")
providerOption = stringOption("--provider")
modelOption = stringOption("--model")
pluginPackageOption = stringOption("-e", {"repeatable": True})


def _build(input: ParsedCommandInput) -> dict[str, Any]:
    parsedAuth = parseAuth(input)
    auth, authErrors = parsedAuth.get("auth"), parsedAuth["errors"]
    connect = input.value(connectOption)
    sessionId = input.value(sessionIdOption)
    shouldContinue = input.value(continueOption) is True or input.value(continueShortOption) is True
    shouldResume = input.value(resumeOption) is True or input.value(resumeShortOption) is True
    provider = input.value(providerOption)
    model = input.value(modelOption)
    pluginPackages = input.values(pluginPackageOption)
    promptArgs = input.remainingArgs[1:] if input.remainingArgs[:1] == ["--"] else input.remainingArgs
    prompt = (
        promptArgs[0]
        if len(promptArgs) == 1
        and (input.remainingArgs[0] == "--" or not promptArgs[0].startswith("-"))
        and len(promptArgs[0]) > 0
        else None
    )
    modelErrors = ["--provider requires --model"] if provider is not None and model is None else []
    sessionSelectionErrors = (
        ["--session-id, --continue, and --resume are mutually exclusive"]
        if len([selected for selected in (sessionId is not None, shouldContinue, shouldResume) if selected]) > 1
        else []
    )
    unsupportedErrors = [] if len(input.remainingArgs) == 0 or prompt is not None else unsupportedOptions("client", input)
    errors = [*authErrors, *modelErrors, *sessionSelectionErrors, *unsupportedErrors]
    if len(errors) > 0:
        return {"ok": False, "errors": errors}
    command: dict[str, Any] = {"command": "client"}
    if auth is not None:
        command["auth"] = auth
    if connect is not None:
        command["connect"] = connect
    if sessionId is not None:
        command["sessionId"] = sessionId
    if shouldContinue:
        command["continue"] = True
    if shouldResume:
        command["resume"] = True
    if provider is not None:
        command["provider"] = provider
    if model is not None:
        command["model"] = model
    if len(pluginPackages) > 0:
        command["pluginPackages"] = pluginPackages
    if prompt is not None:
        command["prompt"] = prompt
    return {"ok": True, "command": command}


clientCommand = (
    Command("client")
    .option(connectOption)
    .option(sessionIdOption)
    .option(continueOption)
    .option(continueShortOption)
    .option(resumeOption)
    .option(resumeShortOption)
    .option(providerOption)
    .option(modelOption)
    .option(pluginPackageOption)
    .option(authTokenOption)
    .option(authTokenFileOption)
    .build(_build)
    .action(lambda command, context: context.runClient(command))
)
