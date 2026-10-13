"""The board does not know the workflows built on it. Its code (``core/platform``, ``core/network``)
and the session catalog import nothing of research and name none of its tables: what the board needs
from research it asks its card owners (``platform.owners``), which research registers at process entry."""

from __future__ import annotations

import ast
import re
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1] / "misaka"
BOARD = [*sorted((PACKAGE / "core" / "platform").rglob("*.py")), *sorted((PACKAGE / "core" / "network").rglob("*.py")),
         PACKAGE / "core" / "session_catalog.py"]
TABLES = re.compile(r"\bresearch_(?:runs|execution|cards|artifacts|findings|claims)\b")


def _imports(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from ((alias.name, node.lineno) for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            yield node.module, node.lineno


def test_the_board_imports_nothing_of_research():
    offenders = [f"{path.relative_to(PACKAGE.parent).as_posix()}:{line}: {module}"
                 for path in BOARD for module, line in _imports(ast.parse(path.read_text(encoding="utf-8")))
                 if module == "misaka.core.research" or module.startswith("misaka.core.research.")]
    assert offenders == []


def test_the_board_names_no_research_table():
    offenders = [f"{path.relative_to(PACKAGE.parent).as_posix()}:{number}"
                 for path in BOARD
                 for number, text in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
                 if TABLES.search(text)]
    assert offenders == []


def test_a_board_without_owners_treats_every_card_as_its_own(monkeypatch):
    from misaka.core.platform import owners

    monkeypatch.setattr(owners, "_OWNERS", [])
    assert owners.card(None, "t_any") == (None, None)
    assert owners.budget_scope(None, "t_any") is None
    assert owners.sender_scope(None, "t_any", None) == (None, None, None)
    assert owners.owns_window() is False
