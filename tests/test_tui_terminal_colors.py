"""Translated from pi-tui's ``test/terminal-colors.test.ts`` (v1.1.0).

MISAKA fork (D11): the three ``TUI.queryTerminalColors`` cases belong to pi's system theme, which is not ported.
"""

from misaka.ui.tui.terminal_colors import (
    RgbColor,
    parseOscColorResponse,
    parseTerminalColorSchemeReport,
)

# --- parseTerminalColorSchemeReport ---------------------------------------------------------------


def test_parses_color_scheme_reports():
    assert parseTerminalColorSchemeReport("\x1b[?997;1n") == "dark"
    assert parseTerminalColorSchemeReport("\x1b[?997;2n") == "light"
    assert parseTerminalColorSchemeReport("\x1b[?997;2n\x1b[?997;1n\x1b[?997;1n") == "dark"
    assert parseTerminalColorSchemeReport("\x1b[?997;1n\x1b[?997;2n\x1b[?997;2n") == "light"
    assert parseTerminalColorSchemeReport("\x1b[?997;3n") is None
    assert parseTerminalColorSchemeReport("\x1b[?996n") is None
    assert parseTerminalColorSchemeReport("x\x1b[?997;1n") is None


# --- parseOscColorResponse ------------------------------------------------------------------------


def test_parses_osc_10_11_and_4_replies():
    assert parseOscColorResponse("\x1b]10;rgb:ffff/ffff/ffff\x07") == {
        "target": "foreground",
        "rgb": RgbColor(r=255, g=255, b=255),
    }
    assert parseOscColorResponse("\x1b]4;13;#ff0080\x1b\\") == {
        "target": 13,
        "rgb": RgbColor(r=255, g=0, b=128),
    }
    assert parseOscColorResponse("\x1b]4;1;bogus\x07") == {"target": 1, "rgb": None}
    assert parseOscColorResponse("\x1b]12;#ffffff\x07") is None


# --- MISAKA ---------------------------------------------------------------------------------------


def test_a_reply_with_a_trailing_newline_is_not_a_reply():
    # `$` is the end of the input, as in JavaScript.
    assert parseTerminalColorSchemeReport("\x1b[?997;2n\n") is None
    assert parseOscColorResponse("\x1b]11;#ffffff\x07\n") is None
