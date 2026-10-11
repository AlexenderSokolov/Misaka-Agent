"""Translated from pi's ``coding-agent/src/experimental/plugin.ts``: the services a plugin facet can use.

``AgentController`` prompts, steers and compacts the Session's root conversation, ``PresentationUI`` selects and shows
status in the presentation, and ``SlashCommands`` contributes slash commands.
"""

from __future__ import annotations

from misaka.experimental.services.agent_controller import AgentController
from misaka.experimental.services.presentation_ui import PresentationUI
from misaka.experimental.services.slash_commands import SlashCommands

__all__ = ["AgentController", "PresentationUI", "SlashCommands"]
