"""Translated from pi's ``coding-agent/src/cli/experimental/command.ts``.

A command parses its own options, then hands the rest to a builder. Results are plain dicts with upstream's keys:
``{"ok": True, "command": invocation}`` or ``{"ok": False, "errors": [...]}``; an invocation is a dict with at least
``"command"``. An option parser returns ``{"ok": True, "value": ...}`` or ``{"ok": False, "error": "..."}``. An action may
be a plain function or a coroutine function.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping, Sequence
from typing import Any


class CommandOption:
    """One option: ``name`` is the literal flag (``-e``, ``--model``)."""

    def __init__(
        self,
        name: str,
        parse: Callable[[str], dict[str, Any]],
        flag: bool = False,
        repeatable: bool = False,
    ) -> None:
        self.name = name
        self.flag = flag
        self.repeatable = repeatable
        self.parse = parse


def valueOption(name: str, parse: Callable[[str], dict[str, Any]], options: Mapping[str, bool] | None = None) -> CommandOption:
    return CommandOption(name, parse, repeatable=bool((options or {}).get("repeatable", False)))


def stringOption(name: str, options: Mapping[str, bool] | None = None) -> CommandOption:
    return valueOption(name, lambda value: {"ok": True, "value": value}, options)


def flagOption(name: str) -> CommandOption:
    return CommandOption(name, lambda _value: {"ok": True, "value": True}, flag=True)


class ParsedCommandInput:
    """What an option pass left for a builder: ``remainingArgs`` and the values of the options it recognised."""

    def __init__(self, remainingArgs: list[str], values: dict[str, list[Any]]) -> None:
        self.remainingArgs = remainingArgs
        self._values = values

    def value(self, option: CommandOption) -> Any:
        values = self._values.get(option.name)
        return values[0] if values else None

    def values(self, option: CommandOption) -> list[Any]:
        return list(self._values.get(option.name, []))


class _MutableParsedCommandInput:
    def __init__(self) -> None:
        self.values: dict[str, list[Any]] = {}
        self.remainingArgs: list[str] = []
        self.errors: list[str] = []


class _RegisteredCommand:
    def __init__(self, command: Command) -> None:
        self._command = command

    def parse(self, argv: Sequence[str]) -> dict[str, Any]:
        return self._command.parse(argv)

    async def execute(self, argv: Sequence[str], context: Any) -> dict[str, Any]:
        return await self._command.execute(argv, context)


class Command:
    def __init__(self, name: str) -> None:
        self.name = name
        self._options: dict[str, CommandOption] = {}
        self._subcommands: dict[str, _RegisteredCommand] = {}
        self._builder: Callable[[ParsedCommandInput], dict[str, Any]] | None = None
        self._commandAction: Callable[[Any, Any], Any] | None = None

    def option(self, option: CommandOption) -> Command:
        if option.name in self._options:
            raise RuntimeError(f"Option {option.name} is already registered for {self.name}")
        self._options[option.name] = option
        return self

    def build(self, builder: Callable[[ParsedCommandInput], dict[str, Any]]) -> Command:
        self._builder = builder
        return self

    def action(self, action: Callable[[Any, Any], Any]) -> Command:
        self._commandAction = action
        return self

    def command(self, command: Command) -> Command:
        if command.name in self._subcommands:
            raise RuntimeError(f"Command {command.name} is already registered")
        self._subcommands[command.name] = _RegisteredCommand(command)
        return self

    def parse(self, argv: Sequence[str]) -> dict[str, Any]:
        selected = self._select(argv)
        if selected is not None:
            return selected[0].parse(selected[1])
        return self._parseOwn(argv)

    async def execute(self, argv: Sequence[str], context: Any) -> dict[str, Any]:
        selected = self._select(argv)
        if selected is not None:
            return await selected[0].execute(selected[1], context)

        parsed = self._parseOwn(argv)
        if not parsed["ok"]:
            return parsed
        if self._commandAction is None:
            raise RuntimeError(f"Command {self.name} does not define an action")
        result = self._commandAction(parsed["command"], context)
        if inspect.isawaitable(result):
            await result
        return {"ok": True, "command": parsed["command"]}

    def _select(self, argv: Sequence[str]) -> tuple[_RegisteredCommand, Sequence[str]] | None:
        if len(argv) == 0:
            return None
        command = self._subcommands.get(argv[0])
        return (command, argv[1:]) if command is not None else None

    def _parseOwn(self, argv: Sequence[str]) -> dict[str, Any]:
        if self._builder is None:
            raise RuntimeError(f"Command {self.name} does not define a builder")
        parsed = self._parseOptions(argv)
        built = self._builder(ParsedCommandInput(parsed.remainingArgs, parsed.values))
        errors = [*parsed.errors, *([] if built["ok"] else built["errors"])]
        if len(errors) > 0:
            return {"ok": False, "errors": errors}
        if not built["ok"]:
            raise RuntimeError(f"Command {self.name} failed without an error")
        return {"ok": True, "command": built["command"]}

    def _parseOptions(self, argv: Sequence[str]) -> _MutableParsedCommandInput:
        parsed = _MutableParsedCommandInput()
        index = 0
        while index < len(argv):
            argument = argv[index]
            index += 1
            if argument == "--":
                parsed.remainingArgs.extend(argv[index - 1 :])
                break

            equals = argument.find("=")
            name = argument if equals == -1 else argument[:equals]
            option = self._options.get(name)
            if option is None:
                parsed.remainingArgs.extend(argv[index - 1 :])
                break

            if option.flag:
                if equals != -1:
                    parsed.errors.append(f"{name} does not take a value")
                    continue
                value = ""
            else:
                candidate = None if equals == -1 else argument[equals + 1 :]
                if candidate is None:
                    following = argv[index] if index < len(argv) else None
                    if following is not None and not following.startswith("-"):
                        candidate = following
                        index += 1
                if candidate is None or candidate == "":
                    parsed.errors.append(f"{name} requires a value")
                    continue
                value = candidate

            values = parsed.values.get(name, [])
            if len(values) > 0 and not option.repeatable:
                parsed.errors.append(f"{name} may only be specified once")
                continue
            result = option.parse(value)
            if not result["ok"]:
                parsed.errors.append(result["error"])
                continue
            values.append(result["value"])
            parsed.values[name] = values
        return parsed
