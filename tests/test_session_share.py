"""Translated from pi's ``coding-agent/test/session-share.test.ts`` and ``export-jsonl-share.test.ts`` (v1.1.0).

``vi.mock("node:child_process")`` becomes ``monkeypatch`` of the module's ``subprocess.run`` (``spawnSync``) and
``asyncio.create_subprocess_exec`` (``spawn``). The JSONL export case uses the suite harness session in place of
``createAgentSession``.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from types import SimpleNamespace
from typing import Any

import pytest

from misaka.agent.types import AgentTool, AgentToolResult
from misaka.ai.types import TextContent
from misaka.core.session_manager import SessionManager
from misaka.ui.tui.interactive import session_share
from misaka.ui.tui.interactive.session_share import (
    SessionShareContext,
    exportSessionForShare,
    shareSession,
)
from misaka.ui.tui.interactive.theme.theme import initTheme
from tests.suite_harness import createHarness


@pytest.fixture(autouse=True, scope="module")
def darkTheme():
    initTheme("dark")


async def test_keeps_concurrent_session_exports_isolated(monkeypatch: pytest.MonkeyPatch):
    uploads: list[str] = []

    class _Process:
        def __init__(self, stdout: str) -> None:
            self.returncode: int | None = None
            self._stdout = stdout

        async def communicate(self) -> tuple[bytes, bytes]:
            await asyncio.sleep(0)
            self.returncode = 0
            return self._stdout.encode(), b""

        def kill(self) -> None:
            pass

    async def spawn(*args: Any, **_kwargs: Any) -> _Process:
        with open(args[-1], encoding="utf-8") as file:
            uploads.append(file.read())
        return _Process(f"https://gist.github.com/test/{len(uploads)}\n")

    monkeypatch.setattr(session_share.subprocess, "run", lambda *_args, **_kwargs: SimpleNamespace(returncode=0))
    monkeypatch.setattr(session_share.asyncio, "create_subprocess_exec", spawn)

    loop = asyncio.get_running_loop()
    aWritten = loop.create_future()
    bWritten = loop.create_future()
    releaseB = loop.create_future()
    errors: list[str] = []

    def context(name: str) -> SessionShareContext:
        async def exportToHtml(filePath: str, _options: Any = None) -> None:
            with open(filePath, "w", encoding="utf-8") as file:
                file.write(name)
            if name == "A":
                aWritten.set_result(None)
                await bWritten
            else:
                bWritten.set_result(None)
                await releaseB

        return SessionShareContext(
            session=SimpleNamespace(
                sessionManager=SimpleNamespace(getSessionId=lambda: name, getCwd=lambda: "/tmp", getBranch=list),
                state=SimpleNamespace(systemPrompt=name, tools=[]),
                modelRuntime=SimpleNamespace(getProvider=lambda _id: None),
                exportToHtml=exportToHtml,
            ),
            ui=SimpleNamespace(setFocus=lambda _component: None, requestRender=lambda *_args: None),
            editorContainer=SimpleNamespace(clear=lambda: None, addChild=lambda _component: None),
            editor=SimpleNamespace(),
            showStatus=lambda _message: None,
            showError=errors.append,
        )

    shareA = asyncio.ensure_future(shareSession(context("A")))
    await aWritten
    shareB = asyncio.ensure_future(shareSession(context("B")))
    await bWritten
    await shareA
    releaseB.set_result(None)
    await shareB

    assert uploads == ["A", "B"]
    assert errors == []


def _shareTool() -> AgentTool:
    async def execute(*_args: Any, **_kwargs: Any) -> AgentToolResult:
        return AgentToolResult(content=[TextContent(text="done")], details={})

    return AgentTool(
        name="share_tool",
        label="Share Tool",
        description="Render a value for sharing",
        parameters={
            "type": "object",
            "properties": {"value": {"type": "string", "description": "Value to render"}},
            "required": ["value"],
        },
        execute=execute,
    )


async def test_adds_presentation_data_without_changing_conversation_ids_or_links():
    async with createHarness(tools=[_shareTool()], initialActiveToolNames=["share_tool"]) as harness:
        session = harness.session
        sessionManager = harness.sessionManager
        tempDir = harness.tempDir
        now = int(time.time() * 1000)
        userId = sessionManager.appendMessage({"role": "user", "content": "hello", "timestamp": now})
        assistantId = sessionManager.appendMessage(
            {
                "role": "assistant",
                "content": [{"type": "toolCall", "id": "call-1", "name": "share_tool", "arguments": {"value": "example"}}],
                "api": "anthropic-messages",
                "provider": "anthropic",
                "model": "test",
                "usage": {
                    "input": 0,
                    "output": 0,
                    "cacheRead": 0,
                    "cacheWrite": 0,
                    "totalTokens": 0,
                    "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0},
                },
                "stopReason": "toolUse",
                "timestamp": now,
            }
        )
        resultId = sessionManager.appendMessage(
            {
                "role": "toolResult",
                "toolCallId": "call-1",
                "toolName": "share_tool",
                "content": [{"type": "text", "text": "done"}],
                "details": {},
                "isError": False,
                "timestamp": now,
            }
        )
        originalEntryIds = [entry["id"] for entry in sessionManager.getBranch()]

        def readRecords(path: str) -> list[dict[str, Any]]:
            with open(path, encoding="utf-8") as file:
                return [json.loads(line) for line in file.read().strip().split("\n")]

        normalPath = os.path.join(tempDir, "normal.jsonl")
        session.exportToJsonl(normalPath)
        normalRecords = readRecords(normalPath)
        assert not any(record.get("type") == "custom" and record.get("customType") == "pi.share" for record in normalRecords)

        sharePath = os.path.join(tempDir, "share.jsonl")
        exportSessionForShare(sharePath, session)
        records = readRecords(sharePath)
        conversationRecords = records[1:-1]
        assert [record["id"] for record in conversationRecords] == originalEntryIds
        assert [record["parentId"] for record in conversationRecords] == [None, *originalEntryIds[:-1]]
        assert [record["id"] for record in conversationRecords[-3:]] == [userId, assistantId, resultId]

        shareEntry = records[-1]
        assert shareEntry["type"] == "custom"
        assert shareEntry["customType"] == "pi.share"
        assert shareEntry["parentId"] == resultId
        assert isinstance(shareEntry["timestamp"], str)
        assert shareEntry["data"]["systemPrompt"] == session.state.systemPrompt
        assert [tool["name"] for tool in shareEntry["data"]["tools"]] == ["share_tool"]
        assert shareEntry["data"]["tools"][0]["description"] == "Render a value for sharing"
        assert "renderedTools" not in shareEntry["data"]
        assert "theme" not in shareEntry["data"]
        assert "version" not in shareEntry["data"]

        imported = SessionManager.open(sharePath)
        assert imported.getLeafId() == shareEntry["id"]
        assert [message.role if hasattr(message, "role") else message["role"] for message in imported.buildSessionContext().messages] == [
            "user",
            "assistant",
            "toolResult",
        ]
