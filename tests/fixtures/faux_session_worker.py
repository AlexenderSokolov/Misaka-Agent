"""pi coding-agent's ``test/fixtures/faux-session-worker.ts``: a Session worker whose model is the faux provider, and whose facets
provide a ``KeyedProbe``. Run as ``python -m tests.fixtures.faux_session_worker`` in place of the real worker module.
"""

from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace
from typing import Any

from misaka.ai.models import createModels
from misaka.ai.providers.faux import fauxAssistantMessage, fauxProvider
from misaka.chord import createStaticFacetLoader, defineFacet
from misaka.chord.context import BACKGROUND_CONTEXT
from misaka.durable import Harness, createRegistry
from misaka.durable.storage.sqlite.node import openNodeSqliteStorage
from misaka.experimental.process import consumeInternalProcessRole
from misaka.experimental.session_worker import runSessionWorkerWithHarness
from misaka.utils.js import DOMException
from tests.fixtures.keyed_service import KeyedProbe


def abortError(signal: Any) -> BaseException:
    reason = signal.reason
    return reason if isinstance(reason, Exception) else DOMException("The operation was aborted", "AbortError")


def setup(env: Any) -> None:
    probes = env.provideMany(KeyedProbe)

    def spawnProbe(value: str) -> None:
        state = env.replicatedState({"value": value})
        holder: dict[str, Any] = {}

        async def replace(next: str, *_: Any) -> None:
            holder["close"]()
            spawnProbe(next)

        async def wait(context: Any, *_: Any) -> None:
            signal = context.abortSignal
            if signal is None:
                raise RuntimeError("Probe wait requires cancellation")
            if signal.aborted:
                raise abortError(signal)
            aborted: asyncio.Future[None] = asyncio.get_running_loop().create_future()

            def fail() -> None:
                if not aborted.done():
                    aborted.set_exception(abortError(signal))

            signal.addEventListener("abort", fail, True)
            await aborted

        holder["close"] = probes.spawn("probe", {"state": state, "replace": replace, "wait": wait})

    env.onActivate(lambda: spawnProbe("first"))


async def createHarness(databasePath: str, options: dict[str, Any]) -> SimpleNamespace:
    if options.get("provider") != "anthropic" or options.get("model") != "claude-sonnet-4-5":
        raise RuntimeError(f"Unexpected faux worker model: {options.get('provider')}/{options.get('model')}")
    faux = fauxProvider()
    answer = fauxAssistantMessage("deterministic remote answer", {"timestamp": 20})
    faux.setResponses([answer, answer])
    models = createModels()
    models.setProvider(faux.provider)
    harness = await Harness.open(
        await openNodeSqliteStorage(databasePath), {"models": models, "registry": createRegistry()}, BACKGROUND_CONTEXT
    )
    model = faux.getModel()
    conversation = await harness.root(
        BACKGROUND_CONTEXT,
        {"agent": {"cwd": options["metadata"]["cwd"], "model": {"provider": model.provider, "modelId": model.id}}},
    )
    keyedProbeFacet = defineFacet(SimpleNamespace(id="@test/keyed-probe", setup=setup))
    return SimpleNamespace(
        harness=harness, conversation=conversation, facetLoader=createStaticFacetLoader([keyedProbeFacet])
    )


def main() -> None:
    role = consumeInternalProcessRole()
    if role != "session-worker":
        raise RuntimeError("Faux Session worker requires a session-worker invocation")
    try:
        code = asyncio.run(runSessionWorkerWithHarness(sys.argv[1:], createHarness))
    except Exception:  # noqa: BLE001 - the worker has reported its failure
        sys.exit(1)
    sys.exit(code)


if __name__ == "__main__":
    main()
