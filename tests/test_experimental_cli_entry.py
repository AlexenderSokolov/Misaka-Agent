"""pi coding-agent's ``test/experimental-cli-entry.test.ts``.

MISAKA fork: the stable entry is ``python -m misaka`` and the development entry is ``python -m misaka.experimental.cli``.
pi's stable CLI answers ``--version`` wherever it stands, so pi's test passes ``server --server-id invalid --version`` to
both; MISAKA's argument parser takes the command first and rejects ``server`` as a command of its own, so only the stable
entry runs without ``--version``. The assertions keep pi's meaning: the stable entry never dispatches an experimental
command, the development entry does (and wins over ``--version``), and the development entry with experiments off is the
stable command line.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from misaka.config import VERSION

REPOSITORY = str(Path(__file__).resolve().parents[1])


def runEntry(module: str, args: list[str], experimental: bool, directory: Path) -> subprocess.CompletedProcess[str]:
    home = directory / "home"
    return subprocess.run(
        [sys.executable, "-m", module, *args],
        cwd=directory,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
        env={
            **os.environ,
            "MISAKA_HOME": str(home),
            "MISAKA_OFFLINE": "1",
            "PI_EXPERIMENTAL": "1" if experimental else "0",
            "PYTHONPATH": REPOSITORY,
            "PYTHONUTF8": "1",
        },
        check=False,
    )


def test_serves_the_version_from_the_stable_entrypoint(tmp_path):
    result = runEntry("misaka", ["--version"], True, tmp_path)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == VERSION


def test_does_not_dispatch_experimental_commands_from_the_stable_entrypoint(tmp_path):
    # pi #9132: enabling experiments must not pull remote-server dependencies into the published CLI.
    result = runEntry("misaka", ["server", "--server-id", "invalid"], True, tmp_path)
    assert result.returncode == 2
    assert "invalid choice: 'server'" in result.stderr
    assert "Invalid --server-id" not in result.stderr


def test_keeps_experimental_dispatch_in_the_development_entrypoint(tmp_path):
    # As in pi's test, ``--version`` follows: experimental dispatch must win over the stable CLI's version handling.
    result = runEntry("misaka.experimental.cli", ["server", "--server-id", "invalid", "--version"], True, tmp_path)
    assert result.returncode == 1, result.stderr
    assert "Invalid --server-id" in result.stderr
    assert VERSION not in result.stdout


def test_falls_back_to_the_stable_cli_when_experiments_are_disabled(tmp_path):
    version = runEntry("misaka.experimental.cli", ["--version"], False, tmp_path)
    assert version.returncode == 0, version.stderr
    assert version.stdout.strip() == VERSION

    server = runEntry("misaka.experimental.cli", ["server", "--server-id", "invalid"], False, tmp_path)
    assert server.returncode == 2
    assert "invalid choice: 'server'" in server.stderr
    assert "Invalid --server-id" not in server.stderr
