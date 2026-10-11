"""Translated from pi's ``coding-agent/src/experimental/durable/prompt.ts``.

MISAKA fork: ``buildSystemPromptSections`` (``misaka/core/system_prompt.py``) builds the sections MISAKA's prompt has:
``preamble``, ``addendum``, ``tools``, ``rules``, ``project_context`` and ``cwd``. It has no ``docs`` section, and its skills
are not part of the prompt, so neither is rendered here.
"""

from __future__ import annotations

from typing import Any

from misaka.config.engine import get_agent_dir
from misaka.core.resource_loader import loadProjectContextFiles
from misaka.core.system_prompt import buildSystemPromptSections
from misaka.core.tools.bash import bashToolSystemPromptContribution
from misaka.core.tools.edit import editToolSystemPromptContribution
from misaka.core.tools.read import readToolSystemPromptContribution
from misaka.core.tools.write import writeToolSystemPromptContribution
from misaka.durable import defineExtension, section

CONTRIBUTIONS = {
    "read": readToolSystemPromptContribution,
    "bash": bashToolSystemPromptContribution,
    "edit": editToolSystemPromptContribution,
    "write": writeToolSystemPromptContribution,
}

# pi's section order; ``buildSystemPromptSections()`` omits the ones without content.
KEYS = ("preamble", "addendum", "tools", "rules", "project_context", "cwd")


def createPiPrompt(settings: Any, fallbackCwd: str) -> Any:
    """pi's system prompt as one extension: the sections of ``buildSystemPromptSections()`` for the request's tools and
    the conversation's directory. Context files load once per directory, like pi at startup."""
    resources: dict[str, dict[str, Any]] = {}

    def load(cwd: str) -> dict[str, Any]:
        found = resources.get(cwd)
        if found is None:
            found = {"contextFiles": loadProjectContextFiles({"cwd": cwd, "agentDir": get_agent_dir()})}
            resources[cwd] = found
        return found

    # The sections of one request render from one build: every section of a request gets the same input.
    built: dict[str, Any] = {"input": None, "sections": None}

    def buildSections(input: dict[str, Any]) -> dict[str, str]:
        env = input.get("env")
        cwd = None if env is None else getattr(env, "cwd", None)
        if cwd is None:
            cwd = input["agent"].get("cwd")
        if cwd is None:
            cwd = fallbackCwd
        selectedTools = [tool["name"] for tool in input["agent"]["tools"]]
        snippets: dict[str, str] = {}
        guidelines: dict[str, list[str]] = {}
        for name in selectedTools:
            contribution = CONTRIBUTIONS.get(name)
            if contribution is None:
                continue
            snippets[name] = contribution["snippet"]
            guidelines[name] = list(contribution["guidelines"])
        return buildSystemPromptSections(
            {
                "cwd": cwd,
                "selectedTools": selectedTools,
                "toolSnippets": snippets,
                "toolGuidelines": guidelines,
                **load(cwd),
            }
        )

    def build(input: dict[str, Any]) -> dict[str, str]:
        if built["input"] is not input:
            built["sections"] = buildSections(input)
            built["input"] = input
        return built["sections"]  # type: ignore[no-any-return]

    def renderer(key: str) -> Any:
        return lambda input, *_: build(input).get(key)

    return defineExtension(
        {
            "name": "pi-prompt",
            # The built sections carry their own tags.
            "sections": [section(key, renderer(key), {"tag": False}) for key in KEYS],
        }
    )
