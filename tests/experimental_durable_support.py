"""pi coding-agent's ``test/experimental-durable-support.ts``."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

from misaka.ai.models import createModels
from misaka.ai.providers.faux import fauxProvider
from misaka.chord.context import BACKGROUND_CONTEXT
from misaka.durable import Harness, MemoryStorage, createRegistry
from misaka.utils.js import abortReason, onAbort


async def openFauxConversation(responses: list[Any] | None = None) -> SimpleNamespace:
    """An in-memory durable Harness whose root conversation answers with the faux provider. Returns
    ``SimpleNamespace(faux, harness, conversation, close)``."""
    faux = fauxProvider()
    faux.setResponses(responses if responses is not None else [])
    models = createModels()
    models.setProvider(faux.provider)
    harness = await Harness.open(MemoryStorage(), {"models": models, "registry": createRegistry()}, BACKGROUND_CONTEXT)
    model = faux.getModel()
    conversation = await harness.root(
        BACKGROUND_CONTEXT, {"agent": {"model": {"provider": model.provider, "modelId": model.id}}}
    )
    return SimpleNamespace(
        faux=faux, harness=harness, conversation=conversation, close=lambda: harness.close(BACKGROUND_CONTEXT)
    )


def pendingResponse() -> SimpleNamespace:
    """A faux response that never arrives; it ends only when its generation is aborted. Returns
    ``SimpleNamespace(step, reached)``; ``reached`` is a future that resolves when the model was asked."""
    reached: asyncio.Future[None] = asyncio.get_running_loop().create_future()

    async def step(_context: Any, options: Any, *_: Any) -> Any:
        if not reached.done():
            reached.set_result(None)
        signal = options.signal
        aborted: asyncio.Future[Any] = asyncio.get_running_loop().create_future()

        def fail() -> None:
            if not aborted.done():
                aborted.set_exception(abortReason(signal))

        onAbort(signal, fail)
        return await aborted

    return SimpleNamespace(step=step, reached=reached)
