"""pi coding-agent's ``test/experimental-plugin-reload.test.ts``."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

from misaka.chord import defineFacet
from misaka.chord.context import BACKGROUND_CONTEXT
from misaka.experimental.services.plugins import SessionPlugins
from misaka.experimental.services.worker import createSessionWorkerServices
from tests.experimental_durable_support import openFauxConversation


async def test_loads_and_cuts_over_a_fresh_session_facet_generation():
    activations: list[int] = []
    disposals: list[int] = []
    generation = 0

    class ReloadableLoader:
        async def load(self):
            nonlocal generation
            generation += 1
            current = generation

            def setup(env):
                env.onActivate(lambda: activations.append(current))

            async def dispose():
                disposals.append(current)

            return SimpleNamespace(
                facets=[defineFacet(SimpleNamespace(id="reloadable-session-plugin", setup=setup))], dispose=dispose
            )

    durable = await openFauxConversation()
    services = await createSessionWorkerServices(
        {
            "harness": durable.harness,
            "conversation": durable.conversation,
            "modelRuntime": None,
            "facetLoader": ReloadableLoader(),
            "publish": AsyncMock(),
        }
    )
    try:
        assert activations == [1]
        await services.invoke(
            {"serviceId": SessionPlugins.id, "member": "reload", "args": []},
            {"serverConnectionId": "server-1", "attachmentId": "attachment-1"},
            BACKGROUND_CONTEXT,
        )
        assert activations == [1, 2]
        assert disposals == [1]
    finally:
        await services.dispose()
        await durable.close()
    assert disposals == [1, 2]
