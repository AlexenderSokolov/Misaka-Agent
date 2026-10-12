"""The ``tool_search`` tool as an extension. Sessions load it as a built-in extension
(``builtin:tool-search``); SDK users add ``createToolSearchExtension()`` to their extension factories.

Translated from pi's ``coding-agent/src/extensions/tool-search/index.ts``.

``tool_search`` is registered inactive. Activate it with ``--tools``, the ``defaultTools`` setting, or
``setActiveTools()``.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from misaka.core.extensions.types import ExtensionFactory
from misaka.core.wiring import KINDS

from .tool import ToolSearchToolOptions, createToolSearchToolDefinition

EXTENSION_NAME = "tool-search"
# `builtin:tool-search`, as upstream bundles it. Replaceable: an extension that registers `tool_search`
# takes over instead of running alongside it.
BUILTIN = True
REPLACEABLE = True
SESSION_KINDS = KINDS
# MISAKA fork: Sisters only. Every MCP tool of Last Order's is declared to her directly (the MCP glue makes her
# servers `direct`), so she has nothing to search for.
ROLES = {"sisters"}


def createToolSearchExtension() -> ExtensionFactory:
    def factory(pi: Any) -> None:
        pi.registerTool(replace(createToolSearchToolDefinition(ToolSearchToolOptions(tools=pi)), defaultActive=False))

    return factory


register = createToolSearchExtension()


def activate(_spec: Any):
    return register


__all__ = ["BUILTIN", "EXTENSION_NAME", "REPLACEABLE", "ROLES", "SESSION_KINDS", "activate", "createToolSearchExtension", "register"]
