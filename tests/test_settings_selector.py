"""Translated from pi's ``coding-agent/test/settings-selector.test.ts`` (v1.1.0).

MISAKA forks: "cycles through fullscreen settings" is not ported (D1: no fullscreen rows). The fixed-theme case has no
system theme in the list (D11), so automatic mode comes first. Upstream builds partial configs with a type cast;
``makeConfig``/``makeCallbacks`` fill the rest.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from misaka.core.keybindings import KeybindingsManager
from misaka.modes.interactive.components.settings_selector import (
    SettingsCallbacks,
    SettingsConfig,
    SettingsSelectorComponent,
)
from misaka.modes.interactive.theme.theme import initTheme
from misaka.ui.tui import setKeybindings
from misaka.utils.ansi import stripAnsi
from tests.suite_harness import createHarness


def makeConfig(**overrides: Any) -> SettingsConfig:
    fields: dict[str, Any] = {
        "autoCompact": False,
        "defaultModel": "not set",
        "currentModel": None,
        "availableDefaultModels": [],
        "showImages": False,
        "imageWidthCells": 60,
        "autoResizeImages": True,
        "blockImages": False,
        "enableSkillCommands": True,
        "steeringMode": "one-at-a-time",
        "followUpMode": "one-at-a-time",
        "transport": "sse",
        "httpIdleTimeoutMs": 300_000,
        "cacheWarmingMode": "streaming",
        "thinkingLevel": "medium",
        "availableThinkingLevels": [],
        "modelThinkingLevels": {},
        "currentTheme": "dark",
        "terminalTheme": "dark",
        "availableThemes": [],
        "hideThinkingBlock": False,
        "mermaidRenderingMode": "streaming",
        "showCacheMissNotices": False,
        "collapseChangelog": False,
        "enableInstallTelemetry": False,
        "doubleEscapeAction": "tree",
        "treeFilterMode": "default",
        "showHardwareCursor": False,
        "editorPaddingX": 0,
        "outputPad": 1,
        "autocompleteMaxVisible": 5,
        "quietStartup": False,
        "defaultProjectTrust": "ask",
        "clearOnShrink": False,
        "showTerminalProgress": False,
        "warnings": {},
    }
    fields.update(overrides)
    return SettingsConfig(**fields)


def makeCallbacks(**overrides: Any) -> SettingsCallbacks:
    names = [name for name in SettingsCallbacks.__dataclass_fields__ if name != "onThemePreview"]
    fields: dict[str, Any] = {name: MagicMock() for name in names}
    fields.update(overrides)
    return SettingsCallbacks(**fields)


@pytest.fixture(autouse=True, scope="module")
def darkTheme():
    initTheme("dark")
    setKeybindings(KeybindingsManager())


def test_keeps_the_configured_fixed_theme_marked_while_browsing():
    config = makeConfig(currentTheme="dark", terminalTheme="dark", availableThemes=["dark", "light"])
    callbacks = makeCallbacks(onThemePreview=MagicMock(), onCancel=lambda: None)
    list = SettingsSelectorComponent(config, callbacks).getSettingsList()

    list.selectItem("theme")
    list.handleInput("\r")
    output = stripAnsi("\n".join(list.render(120)))
    assert any(line.startswith("    automatic") and "Use separate themes" in line for line in output.split("\n"))
    assert "→ ✓ dark" in output

    list.handleInput("\x1b[B")
    output = stripAnsi("\n".join(list.render(120)))
    assert "  ✓ dark" in output
    assert "→   light" in output


def test_keeps_a_configured_automatic_theme_marked_while_browsing():
    config = makeConfig(currentTheme="light/dark", terminalTheme="dark", availableThemes=["dark", "light", "other"])
    callbacks = makeCallbacks(onThemePreview=MagicMock(), onCancel=lambda: None)
    list = SettingsSelectorComponent(config, callbacks).getSettingsList()

    list.selectItem("theme")
    list.handleInput("\r")
    list.handleInput("\r")
    output = stripAnsi("\n".join(list.render(120)))
    assert "→ ✓ light" in output

    list.handleInput("\x1b[B")
    output = stripAnsi("\n".join(list.render(120)))
    assert "  ✓ light" in output
    assert "→   other" in output


async def test_keeps_the_configured_per_model_thinking_level_marked_while_browsing():
    async with createHarness(models=[{"id": "thinking-model", "reasoning": True}]) as harness:
        model = harness.getModel("thinking-model")
        modelKey = f"{model.provider}/{model.id}"
        config = makeConfig(
            defaultModel=modelKey,
            availableDefaultModels=[model],
            thinkingLevel="high",
            modelThinkingLevels={modelKey: "medium"},
        )
        callbacks = makeCallbacks(onCancel=lambda: None)
        list = SettingsSelectorComponent(config, callbacks).getSettingsList()

        list.selectItem("model-thinking")
        list.handleInput("\r")
        list.handleInput("\r")

        output = stripAnsi("\n".join(list.render(120)))
        assert "→ ✓ medium" in output
        assert "    (clear override)" in output

        list.handleInput("\x1b[B")
        output = stripAnsi("\n".join(list.render(120)))
        assert "  ✓ medium" in output
        assert "→   high" in output
