"""Shared utilities for coding-agent compaction and branch summarization."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from misaka.ai.types import MessageValue
from misaka.utils.values import read_field

_TOOL_RESULT_MAX_CHARS = 2000

SUMMARIZATION_SYSTEM_PROMPT = (
    "You are a context summarization assistant. Your task is to read a conversation between a user "
    "and an AI assistant, then produce a structured summary following the exact format specified.\n\n"
    "Do NOT continue the conversation. Do NOT respond to any questions in the conversation. "
    "ONLY output the structured summary."
)


@dataclass(slots=True)
class FileOperations:
    read: set[str] = field(default_factory=set)
    written: set[str] = field(default_factory=set)
    edited: set[str] = field(default_factory=set)


def createFileOps() -> FileOperations:
    return FileOperations()


def extractFileOpsFromMessage(message: Any, file_ops: FileOperations) -> None:
    """Extract file operations from tool calls in an assistant message, or from the nested calls
    recorded on a tool result."""
    if read_field(message, "role") == "toolResult":
        # Calls made from codemode scripts are recorded on the script's result.
        nested_calls = read_field(message, "nestedCalls")
        for call in read_field(nested_calls, "calls", []) if nested_calls is not None else []:
            addFileOp(read_field(call, "name"), read_field(call, "arguments"), file_ops)
        return
    if read_field(message, "role") != "assistant":
        return
    content = read_field(message, "content")
    if not isinstance(content, list):
        return

    for block in content:
        if read_field(block, "type") != "toolCall":
            continue
        addFileOp(read_field(block, "name"), read_field(block, "arguments"), file_ops)


def addFileOp(tool_name: Any, arguments: Any, file_ops: FileOperations) -> None:
    path = arguments.get("path") if isinstance(arguments, dict) else None
    if not isinstance(path, str) or not path:
        return
    if tool_name == "read":
        file_ops.read.add(path)
    elif tool_name == "write":
        file_ops.written.add(path)
    elif tool_name == "edit":
        file_ops.edited.add(path)


def computeFileLists(file_ops: FileOperations) -> dict[str, list[str]]:
    modified = {*file_ops.edited, *file_ops.written}
    read_files = sorted(path for path in file_ops.read if path not in modified)
    modified_files = sorted(modified)
    return {"readFiles": read_files, "modifiedFiles": modified_files}


def formatFileOperations(read_files: list[str], modified_files: list[str]) -> str:
    sections: list[str] = []
    if read_files:
        read_block = "\n".join(read_files)
        sections.append(f"<read-files>\n{read_block}\n</read-files>")
    if modified_files:
        modified_block = "\n".join(modified_files)
        sections.append(f"<modified-files>\n{modified_block}\n</modified-files>")
    if not sections:
        return ""
    return f"\n\n{'\n\n'.join(sections)}"


def serializeConversation(messages: list[MessageValue]) -> str:
    parts: list[str] = []

    for message in messages:
        role = read_field(message, "role")
        if role == "user":
            content = _serialize_user_content(read_field(message, "content"))
            if content:
                parts.append(f"[User]: {content}")
            continue

        if role == "assistant":
            text_parts: list[str] = []
            thinking_parts: list[str] = []
            tool_calls: list[str] = []
            for block in read_field(message, "content") or []:
                block_type = read_field(block, "type")
                if block_type == "text":
                    text = read_field(block, "text")
                    if isinstance(text, str):
                        text_parts.append(text)
                elif block_type == "thinking":
                    thinking = read_field(block, "thinking")
                    if isinstance(thinking, str):
                        thinking_parts.append(thinking)
                elif block_type == "toolCall":
                    arguments = read_field(block, "arguments")
                    items = arguments.items() if isinstance(arguments, dict) else []
                    args_str = ", ".join(f"{key}={_safe_json_stringify(value)}" for key, value in items)
                    tool_calls.append(f"{read_field(block, 'name')}({args_str})")
            if thinking_parts:
                parts.append(f"[Assistant thinking]: {'\n'.join(thinking_parts)}")
            if text_parts:
                parts.append(f"[Assistant]: {'\n'.join(text_parts)}")
            if tool_calls:
                parts.append(f"[Assistant tool calls]: {'; '.join(tool_calls)}")
            continue

        if role == "toolResult":
            content = _serialize_tool_result_content(read_field(message, "content"))
            if content:
                parts.append(f"[Tool result]: {truncateForSummary(content, _TOOL_RESULT_MAX_CHARS)}")

    return "\n\n".join(parts)


def _serialize_user_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if read_field(block, "type") == "text":
            text = read_field(block, "text")
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts)


def _serialize_tool_result_content(content: Any) -> str:
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if read_field(block, "type") == "text":
            text = read_field(block, "text")
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts)


def truncateForSummary(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    truncated_chars = len(text) - max_chars
    return f"{text[:max_chars]}\n\n[... {truncated_chars} more characters truncated]"


def _safe_json_stringify(value: Any) -> str:
    try:
        # Compact and unescaped, like `JSON.stringify`: the result is measured by length
        # for the chars/4 token estimate, and default ASCII escaping turns every CJK
        # character into six -- inflating estimates on non-ASCII content by 2-3x, which
        # made compaction fire far too early for those sessions.
        serialized = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        return "[unserializable]"
    return serialized if serialized is not None else "undefined"



__all__ = [
    "SUMMARIZATION_SYSTEM_PROMPT",
    "FileOperations",
    "computeFileLists",
    "createFileOps",
    "extractFileOpsFromMessage",
    "formatFileOperations",
    "serializeConversation",
]


def _assistant_text(message: Any) -> str:
    parts: list[str] = []
    for block in read_field(message, "content") or []:
        if read_field(block, "type") == "text":
            text = read_field(block, "text")
            if isinstance(text, str):
                parts.append(text)
    return "\n".join(parts)
