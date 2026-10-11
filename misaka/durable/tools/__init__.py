"""Coding tools: ``read``, ``write``, ``edit``, ``bash`` and ``powershell``.

Translated from pi's ``packages/durable/src/tools/index.ts``.
"""

from __future__ import annotations

from ..harness.define import defineExtension
from .bash import createBashTool, createPowerShellTool
from .edit import createEditTool
from .read import createReadTool
from .write import createWriteTool

CodingTools = defineExtension(
    {
        "name": "coding-tools",
        "tools": [createReadTool(), createWriteTool(), createEditTool(), createBashTool()],
    }
)
"""``read``, ``write``, ``edit``, and ``bash``; nothing installs it automatically. ``createPowerShellTool()`` adds
``powershell``."""

__all__ = [
    "CodingTools",
    "createBashTool",
    "createEditTool",
    "createPowerShellTool",
    "createReadTool",
    "createWriteTool",
]
