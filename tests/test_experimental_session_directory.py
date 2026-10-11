"""pi coding-agent's ``test/experimental-session-directory.test.ts``.

MISAKA fork: the default directories are rows of the home (``experimental_sessions`` and ``experimental_server``) where pi
puts them under ``PI_CODING_AGENT_DIR`` and ``~/.pi``; the tests assert against the home.
"""

from __future__ import annotations

import os

from misaka.config import home
from misaka.experimental.server import resolveServerDirectory, resolveSessionDirectory


def test_uses_the_experimental_directory_in_the_home_by_default():
    assert resolveSessionDirectory() == os.path.abspath(str(home.path("experimental_sessions")))


def test_resolves_an_explicit_relative_directory_from_the_current_working_directory():
    assert resolveSessionDirectory("relative/sessions") == os.path.abspath("relative/sessions")


def test_expands_a_tilde_in_an_explicit_directory():
    assert resolveSessionDirectory("~/custom-sessions") == os.path.abspath(
        os.path.join(os.path.expanduser("~"), "custom-sessions")
    )


def test_uses_the_server_directory_in_the_home_by_default(monkeypatch):
    monkeypatch.delenv("PI_SERVER_DIR", raising=False)

    assert resolveServerDirectory() == os.path.abspath(str(home.path("experimental_server")))


def test_prefers_an_explicit_server_directory_over_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("PI_SERVER_DIR", str(tmp_path / "from-environment"))

    assert resolveServerDirectory() == os.path.abspath(str(tmp_path / "from-environment"))
    assert resolveServerDirectory(str(tmp_path / "explicit")) == os.path.abspath(str(tmp_path / "explicit"))
