"""Translated from pi's ``coding-agent/src/cli/project-trust.ts`` (v1.1.0)."""

from __future__ import annotations

import sys
from typing import Any

from misaka.core.extensions.types import ProjectTrustContext
from misaka.core.project_trust import AppMode
from misaka.core.settings_manager import SettingsManager
from misaka.utils.chalk import chalk

from .startup_ui import StartupOption, showStartupInput, showStartupSelector


def createProjectTrustContext(
    *, cwd: str, mode: AppMode, settingsManager: SettingsManager, hasUI: bool
) -> ProjectTrustContext:
    async def select(title: str, selectOptions: list[str], _opts: Any = None) -> str | None:
        if not hasUI:
            return None
        if mode != "interactive":
            return None
        return await showStartupSelector(
            settingsManager, title, [StartupOption(label=option, value=option) for option in selectOptions]
        )

    async def confirm(title: str, message: str, _opts: Any = None) -> bool:
        if not hasUI:
            return False
        if mode != "interactive":
            return False
        result = await showStartupSelector(
            settingsManager,
            f"{title}\n{message}",
            [StartupOption(label="Yes", value=True), StartupOption(label="No", value=False)],
        )
        return result if result is not None else False

    async def input(title: str, placeholder: str | None = None, _opts: Any = None) -> str | None:
        if not hasUI:
            return None
        if mode != "interactive":
            return None
        return await showStartupInput(settingsManager, title, placeholder)

    def notify(message: str, type: str = "info") -> None:
        if mode != "interactive":
            color = chalk.red if type == "error" else chalk.yellow if type == "warning" else chalk.cyan
            print(color(message), file=sys.stderr)

    return {
        "cwd": cwd,
        "mode": "tui" if mode == "interactive" else mode,  # type: ignore[typeddict-item]
        "hasUI": hasUI,
        "ui": {"select": select, "confirm": confirm, "input": input, "notify": notify},
    }


__all__ = ["createProjectTrustContext"]
