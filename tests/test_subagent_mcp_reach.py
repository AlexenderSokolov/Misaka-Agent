"""Sub-agents reach MCP as they did before the MCP extension: an async child may use the extension's doors to
undeclared MCP tools, and a required server counts once it is configured, connected or not."""

from __future__ import annotations

import json
import os
from pathlib import Path

from misaka.core.subagent import agents
from misaka.core.subagent.tool_policy import tool_allowed_for_agent


def test_an_async_child_may_use_codemode_and_tool_search_as_it_may_mcp_tools():
    assert tool_allowed_for_agent("mcp__docs__search", is_async=True)
    assert tool_allowed_for_agent("codemode", is_async=True)
    assert tool_allowed_for_agent("tool_search", is_async=True)
    assert not tool_allowed_for_agent("AskUserQuestion", is_async=True)


def _write(path: Path, servers: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"mcpServers": servers}), encoding="utf-8")


def test_required_servers_count_by_configuration(monkeypatch, tmp_path):
    root = Path(os.environ["MISAKA_HOME"])
    _write(root / "mcp.json", {"github": {"command": "gh-mcp"}, "off": {"command": "x", "enabled": False}})
    sister = root / "profiles" / "sisters" / "10032"
    _write(sister / "mcp.json", {"zotero": {"command": "zotero-mcp"}})

    assert agents.configured_mcp_servers(None) == ["github"]
    assert agents.configured_mcp_servers(root / "profiles" / "last_order") == ["github"]
    assert agents.configured_mcp_servers(sister) == ["zotero"]
