"""Bundled extensions, discovered by folder.

    misaka/extensions/<module>             every role
    misaka/extensions/last_order/<module>  Last Order only
    misaka/extensions/sisters/<module>     every other role (the Sisters)

This is what Pi's ``src/extensions/`` holds -- what ships in the package and reaches a
session through the extension API alone -- with one thing Pi does not have: roles, so
the folder a module sits in decides who gets it. A module takes part by defining
``activate(spec) -> register | None``: ``register(harn)`` installs the extension into the
harness, ``None`` skips it for this session. A module may override its descriptor name
with ``EXTENSION_NAME``, hide it with ``HIDDEN``, restrict itself to session kinds
with ``SESSION_KINDS = {...}`` (the default is every kind except ``bare``), or to roles with
``ROLES = {"last_order" | "sisters", ...}`` (the default is both). A module may also
offer a top-level command, ``misaka <COMMAND> ...``, by defining ``COMMAND`` and
``command(argv)``; the first line of that function's docstring is its help.

Nothing in ``misaka.core`` imports this package. The process entry hands ``discover``
to ``misaka.core.wiring.bundled`` (``misaka.cli.bootstrap``), the way Pi's ``main.ts``
composes ``builtInExtensions`` ahead of everything else, and core calls it with the
session's spec. The engine-level extension protocol stays in :mod:`misaka.core.extensions`.
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import Any

from misaka.config import profiles
from misaka.core.wiring import DEFAULT_KINDS, inline

_ROLE_DIRS = ("last_order", "sisters")
# pi's `builtInExtensions`, in its order; misaka's own bundled extensions follow, by name.
_PI_BUILTINS = ("llama", "codemode", "tool_search", "mcp")


def _order(name: str) -> tuple[int, str]:
    return (_PI_BUILTINS.index(name), "") if name in _PI_BUILTINS else (len(_PI_BUILTINS), name)


def _modules(package: str):
    pkg = importlib.import_module(package)
    for info in sorted(pkgutil.iter_modules(pkg.__path__), key=lambda i: _order(i.name)):
        if info.name.startswith("_") or (package == __name__ and info.name in _ROLE_DIRS):
            continue
        yield importlib.import_module(f"{package}.{info.name}")


def discover(spec) -> list[dict[str, Any]]:
    """Return the inline extensions for one session, by folder and session kind."""
    last_order = profiles.is_last_order(spec.profile_dir)
    kind = "bare" if last_order and spec.kind == "beast" else spec.kind  # a budget-less Last Order runs tool-less
    out: list[dict[str, Any]] = []
    for package in (__name__, f"{__name__}.{'last_order' if last_order else 'sisters'}"):
        for mod in _modules(package):
            activate = getattr(mod, "activate", None)
            if activate is None or kind not in getattr(mod, "SESSION_KINDS", DEFAULT_KINDS):
                continue
            if ("last_order" if last_order else "sisters") not in getattr(mod, "ROLES", _ROLE_DIRS):
                continue
            register = activate(spec)
            if register is not None:
                out.append(
                    inline(
                        getattr(mod, "EXTENSION_NAME", mod.__name__.rsplit(".", 1)[-1]),
                        register,
                        hidden=bool(getattr(mod, "HIDDEN", False)),
                        replaceable=bool(getattr(mod, "REPLACEABLE", False)),
                        builtin=bool(getattr(mod, "BUILTIN", False)),
                    )
                )
    return out


def commands() -> dict[str, Any]:
    """The top-level commands bundled extensions offer: ``{COMMAND: command}``. Every bundled
    module is imported to look, so the process entry asks only for a name that is not its own."""
    found: dict[str, Any] = {}
    for package in (__name__, *(f"{__name__}.{role}" for role in _ROLE_DIRS)):
        for mod in _modules(package):
            name, command = getattr(mod, "COMMAND", None), getattr(mod, "command", None)
            if name and callable(command):
                found[name] = command
    return found


__all__ = ["commands", "discover"]
