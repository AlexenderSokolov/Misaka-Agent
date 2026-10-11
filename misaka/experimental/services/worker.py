"""Translated from pi's ``coding-agent/src/experimental/services/worker.ts``.

A ``SessionWorkerRuntime`` is a ``SimpleNamespace`` of ``harness``, ``conversation`` (the root conversation the services
expose), ``modelRuntime``, ``settingsManager``, ``facetLoader`` and ``cleanup`` (``None`` or an async function that releases
resources the Harness does not own, such as execution environments, after it closed); the last four are optional. A
``WorkerServiceScope`` is ``{"serverConnectionId", "attachmentId"}``.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

from misaka.chord import (
    Context,
    createFacetHost,
    createRemoteServiceEndpoint,
    createStaticFacetLoader,
    defineFacet,
)
from misaka.chord.context import BACKGROUND_CONTEXT
from misaka.experimental.services.agent_controller import AgentController
from misaka.experimental.services.agent_controller_provider import createAgentController
from misaka.experimental.services.models_provider import createModelsServiceFacet
from misaka.experimental.services.plugins import SessionPlugins
from misaka.experimental.services.transcript_provider import (
    createTranscriptServiceFacet,
)


async def _settle(operation: asyncio.Future[Any]) -> None:
    try:
        await operation
    except Exception:  # noqa: BLE001, S110 - `operation.catch(() => {})`: the tail only marks that it settled
        pass


async def createSessionWorkerServices(options: dict[str, Any]) -> SimpleNamespace:
    """``options`` is ``{"harness", "conversation", "modelRuntime", "settingsManager"?, "facetLoader"?, "publish"}`` where
    ``publish(scope, subscriptionId, update)`` delivers a service update to the client of ``scope``. Returns a
    ``SimpleNamespace`` of ``invoke(call, scope, context)``, ``removeSubscriptions(matches)`` and ``dispose()``."""
    agentControllerRuntimeFacet = defineFacet(
        SimpleNamespace(
            id="@pi/agent-controller-runtime",
            setup=lambda env: env.provide(
                AgentController, createAgentController(options["harness"], options["conversation"])
            ),
        )
    )

    async def pluginsNotReady() -> None:
        raise RuntimeError("Session plugins are not ready")

    reloadPlugins = [pluginsNotReady]

    pluginRuntimeFacet = defineFacet(
        SimpleNamespace(
            id="@pi/session-plugins-runtime",
            setup=lambda env: env.provide(SessionPlugins, {"reload": lambda *_: reloadPlugins[0]()}),
        )
    )
    builtins = await createStaticFacetLoader(
        [
            agentControllerRuntimeFacet,
            pluginRuntimeFacet,
            await createModelsServiceFacet({**options, "context": BACKGROUND_CONTEXT}),
            await createTranscriptServiceFacet(options["conversation"], BACKGROUND_CONTEXT),
        ]
    ).load()
    pluginLoader = options.get("facetLoader") or createStaticFacetLoader([])
    loadedPlugins = await pluginLoader.load()
    try:
        facetHost = await createFacetHost({"facets": [*builtins.facets, *loadedPlugins.facets]})
    except Exception as error:
        cleanup = await asyncio.gather(loadedPlugins.dispose(), builtins.dispose(), return_exceptions=True)
        cleanupErrors = [result for result in cleanup if isinstance(result, Exception)]
        if len(cleanupErrors) > 0:
            raise ExceptionGroup("Session facets failed to start and clean up", [error, *cleanupErrors]) from None
        raise
    reloadTail: asyncio.Future[None] = asyncio.get_running_loop().create_future()
    reloadTail.set_result(None)

    def reloadPluginGeneration() -> asyncio.Future[None]:
        nonlocal reloadTail, loadedPlugins
        previous = reloadTail

        async def run() -> None:
            nonlocal loadedPlugins
            await previous
            candidate = await pluginLoader.load()
            try:
                await facetHost.reload(candidate.facets)
            except Exception as error:
                try:
                    await candidate.dispose()
                except Exception as cleanupError:  # noqa: BLE001
                    raise ExceptionGroup("Session plugin reload and cleanup failed", [error, cleanupError]) from None
                raise
            retired = loadedPlugins
            loadedPlugins = candidate
            await retired.dispose()

        operation = asyncio.ensure_future(run())
        reloadTail = asyncio.ensure_future(_settle(operation))
        return operation

    reloadPlugins[0] = reloadPluginGeneration  # type: ignore[assignment]
    provider = facetHost.services

    endpoints: dict[str, SimpleNamespace] = {}

    def removeSubscriptions(matches: Any) -> None:
        for key, entry in list(endpoints.items()):
            if not matches(entry.scope):
                continue
            entry.endpoint.dispose()
            del endpoints[key]

    def invoke(call: dict[str, Any], scope: dict[str, str], context: Context) -> Any:
        key = serviceScopeKey(scope)
        entry = endpoints.get(key)
        if entry is None:
            entry = SimpleNamespace(scope=scope, endpoint=createRemoteServiceEndpoint(provider))
            endpoints[key] = entry
        return entry.endpoint.invoke(
            call, lambda subscriptionId, update, *_: options["publish"](scope, subscriptionId, update), context
        )

    async def dispose() -> None:
        removeSubscriptions(lambda _scope: True)
        await reloadTail
        errors: list[Exception] = []
        try:
            await facetHost.dispose()
        except Exception as error:  # noqa: BLE001 - collected
            errors.append(error)
        results = await asyncio.gather(loadedPlugins.dispose(), builtins.dispose(), return_exceptions=True)
        errors.extend(result for result in results if isinstance(result, Exception))
        if len(errors) == 1:
            raise errors[0]
        if len(errors) > 1:
            raise ExceptionGroup("Failed to dispose Session facets", errors)

    return SimpleNamespace(invoke=invoke, removeSubscriptions=removeSubscriptions, dispose=dispose)


def serviceScopeKey(scope: dict[str, str]) -> str:
    return f"{scope['serverConnectionId']}\0{scope['attachmentId']}"
