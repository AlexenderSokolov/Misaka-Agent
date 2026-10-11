"""Translated from pi's ``coding-agent/src/experimental/services/transcript-provider.ts``."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from misaka.chord import Context, defineFacet
from misaka.experimental.services.transcript import Transcript


async def createTranscriptServiceFacet(conversation: Any, context: Context) -> Any:
    """Serve the conversation's durable view state. The facet owns and disposes the attached state."""
    state = await conversation.viewState(context)

    def setup(env: Any) -> None:
        env.own(lambda: state.dispose())
        env.provide(Transcript, {"state": state})

    return defineFacet(SimpleNamespace(id="@pi/transcript", setup=setup))
