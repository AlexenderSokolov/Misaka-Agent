"""The experimental client/server stack is development-only: the published command line neither imports it nor the packages it
runs on (pi #9132: enabling experiments must not pull remote-server dependencies into the published CLI).
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1] / "misaka"
REPOSITORY = str(PACKAGE.parent)
EXPERIMENTAL = ("misaka.experimental", "misaka.cli.experimental")
# The packages only the experimental stack runs on; pi-env (misaka.env) is reachable only through it (D5).
STACK = (
    "misaka.experimental",
    "misaka.client",
    "misaka.server",
    "misaka.durable",
    "misaka.protocol",
    "misaka.chord",
    "misaka.env",
)


def _package(path: Path) -> list[str]:
    parts = ["misaka", *path.relative_to(PACKAGE).with_suffix("").parts]
    return parts[:-1]  # the package a module (or an `__init__`) belongs to


def importedModules(tree: ast.AST, path: Path | None = None) -> list[tuple[str, int]]:
    """Every module a file imports: absolute and relative imports, and ``importlib.import_module("misaka...")``."""
    modules: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules += [(alias.name, node.lineno) for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                base = node.module or ""
            elif path is not None:
                package = _package(path)
                package = package[: len(package) - (node.level - 1)] if node.level > 1 else package
                base = ".".join([*package, *([node.module] if node.module else [])])
            else:
                continue
            if base:
                modules += [(base, node.lineno)] + [(f"{base}.{alias.name}", node.lineno) for alias in node.names]
        elif (
            isinstance(node, ast.Call)
            and getattr(node.func, "attr", getattr(node.func, "id", None)) == "import_module"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            modules.append((node.args[0].value, node.lineno))
    return modules


def under(module: str, packages: tuple[str, ...]) -> bool:
    return any(module == package or module.startswith(f"{package}.") for package in packages)


def test_nothing_outside_the_experimental_stack_imports_it():
    offenders = []
    for path in sorted(PACKAGE.rglob("*.py")):
        relative = path.relative_to(PACKAGE).parts
        if relative[:2] == ("cli", "experimental") or under(".".join(["misaka", *relative[:1]]), STACK):
            continue
        for module, line in importedModules(ast.parse(path.read_text(encoding="utf-8")), path):
            if under(module, STACK):
                offenders.append(f"{path.relative_to(PACKAGE.parent)}:{line}: {module}")
    assert offenders == []


def test_the_experimental_command_definitions_do_not_import_the_experimental_stack():
    """The ``server`` and ``client`` command definitions are plain parsing; the stack imports them, never the reverse.
    Only the protocol's ``isServerId`` is used, as upstream's command-options.ts imports it from pi-protocol."""
    offenders = []
    for path in sorted((PACKAGE / "cli" / "experimental").rglob("*.py")):
        for module, line in importedModules(ast.parse(path.read_text(encoding="utf-8")), path):
            if under(module, STACK) and module not in ("misaka.protocol", "misaka.protocol.isServerId"):
                offenders.append(f"{path.relative_to(PACKAGE.parent)}:{line}: {module}")
    assert offenders == []


def test_the_published_command_line_does_not_load_the_experimental_stack(tmp_path):
    program = (
        "import sys\n"
        "from misaka.__main__ import main\n"
        "try:\n"
        "    main(['--version'])\n"
        "except SystemExit:\n"
        "    pass\n"
        f"loaded = sorted(name for name in sys.modules if any(name == p or name.startswith(p + '.') for p in {STACK!r}))\n"
        "print('LOADED', loaded)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
        env={
            **os.environ,
            "MISAKA_HOME": str(tmp_path / "home"),
            "MISAKA_OFFLINE": "1",
            "PI_EXPERIMENTAL": "1",
            "PYTHONPATH": REPOSITORY,
            "PYTHONUTF8": "1",
        },
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[-1] == "LOADED []"


def test_the_chat_panel_and_research_do_not_load_the_experimental_stack(tmp_path):
    program = (
        "import sys\n"
        "import misaka.modes.interactive.interactive_mode, misaka.ui.panel, misaka.core.research.workflow\n"
        f"loaded = sorted(name for name in sys.modules if any(name == p or name.startswith(p + '.') for p in {STACK!r}))\n"
        "print('LOADED', loaded)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
        env={**os.environ, "MISAKA_HOME": str(tmp_path / "home"), "PYTHONPATH": REPOSITORY, "PYTHONUTF8": "1"},
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[-1] == "LOADED []"
