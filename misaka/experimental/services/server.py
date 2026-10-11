"""Translated from pi's ``coding-agent/src/experimental/services/server.ts``.

``createExperimentalServerServices(options)`` takes ``{"list", "create", "remove", "prepareSessionPlugins",
"reloadPresentationPlugins"}``, async functions over Session summaries and plugin selections, and returns a
``SimpleNamespace`` of ``host`` (the routed server service host), ``refresh(context=BACKGROUND_CONTEXT)`` and ``dispose()``.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

from misaka.chord import (
    Context,
    RemoteServiceProvider,
    createRemoteServiceEndpoint,
    replicatedState,
)
from misaka.chord.context import BACKGROUND_CONTEXT
from misaka.experimental.services.plugins import PresentationPlugins
from misaka.experimental.services.sessions import SessionDirectory, SessionManagement
from misaka.utils.js import rejectedPromise


async def createExperimentalServerServices(options: dict[str, Any]) -> SimpleNamespace:
    revision = 1
    directory = replicatedState({"revision": revision, "sessions": await options["list"](BACKGROUND_CONTEXT)})
    # A JavaScript ``Set`` of objects: keyed by identity, as the attachments are not hashable.
    attachments: dict[int, Any] = {}
    mutationTail: asyncio.Future[None] = asyncio.get_running_loop().create_future()
    mutationTail.set_result(None)

    async def refreshNow(context: Context) -> None:
        nonlocal revision
        sessions = await options["list"](context)
        revision += 1

        def update(draft: Any) -> None:
            draft["revision"] = revision
            draft["sessions"] = sessions

        directory.change(context, update)

    def serialize(operation: Callable[[], Any]) -> asyncio.Future[Any]:
        nonlocal mutationTail
        previous = mutationTail

        async def run() -> Any:
            try:
                await previous
            except Exception:  # noqa: BLE001, S110 - `mutationTail.catch(() => {})`
                pass
            return await operation()

        result = asyncio.ensure_future(run())

        async def settle() -> None:
            try:
                await result
            except Exception:  # noqa: BLE001, S110 - the tail only marks that the operation settled
                pass

        mutationTail = asyncio.ensure_future(settle())
        return result

    def attachClient(presentation: Any, *_: Any) -> Any:
        preparedPluginPackagePaths: list[str] | None = None
        provider = RemoteServiceProvider(
            [
                {"service": SessionDirectory, "mode": "singleton"},
                {"service": SessionManagement, "mode": "singleton"},
                {"service": PresentationPlugins, "mode": "singleton"},
            ]
        )
        provider.provide(SessionDirectory, {"state": directory})

        def prepareSession(request: dict[str, Any], context: Context, *_: Any) -> Any:
            async def prepare() -> Any:
                nonlocal preparedPluginPackagePaths
                selected = await options["prepareSessionPlugins"](request["sessionId"], request.get("packagePaths"), context)
                preparedPluginPackagePaths = selected["packagePaths"]
                return selected["presentationPlugins"]

            return serialize(prepare)

        def reload(context: Context, *_: Any) -> Any:
            async def run() -> Any:
                if preparedPluginPackagePaths is None:
                    raise RuntimeError("No Session plugin selection is prepared")
                return await options["reloadPresentationPlugins"](preparedPluginPackagePaths, context)

            return serialize(run)

        provider.provide(PresentationPlugins, {"prepareSession": prepareSession, "reload": reload})

        def create(createOptions: dict[str, Any], context: Context, *_: Any) -> Any:
            async def run() -> Any:
                created = await options["create"](createOptions, context)
                await refreshNow(context)
                return created

            return serialize(run)

        def remove(sessionId: str, context: Context, *_: Any) -> Any:
            async def run() -> None:
                await presentation.prepareSessionRemoval(sessionId, context)
                await options["remove"](sessionId, context)
                await refreshNow(context)

            return serialize(run)

        def attach(sessionId: str, context: Context, *_: Any) -> Any:
            async def run() -> None:
                await presentation.attachSession(sessionId, context)

            return serialize(run)

        def detach(context: Context, *_: Any) -> Any:
            async def run() -> None:
                nonlocal preparedPluginPackagePaths
                await presentation.detachSession(context)
                preparedPluginPackagePaths = None

            return serialize(run)

        provider.provide(SessionManagement, {"create": create, "remove": remove, "attach": attach, "detach": detach})
        attachment = createProviderAttachment(provider, lambda: attachments.pop(id(attachment), None))
        attachments[id(attachment)] = attachment
        return attachment

    def refresh(context: Context = BACKGROUND_CONTEXT) -> Any:
        return serialize(lambda: refreshNow(context))

    async def dispose() -> None:
        releases = await asyncio.gather(
            *(_release(attachment) for attachment in list(attachments.values())), return_exceptions=True
        )
        attachments.clear()
        await mutationTail
        errors = [result for result in releases if isinstance(result, Exception)]
        if len(errors) == 1:
            raise errors[0]
        if len(errors) > 1:
            raise ExceptionGroup("Failed to release server service attachments", errors)

    return SimpleNamespace(host=SimpleNamespace(attachClient=attachClient), refresh=refresh, dispose=dispose)


async def _release(attachment: Any) -> None:
    result = attachment.release(BACKGROUND_CONTEXT)
    if inspect.isawaitable(result):
        await result


def createProviderAttachment(provider: RemoteServiceProvider, onRelease: Callable[[], None]) -> SimpleNamespace:
    endpoint = createRemoteServiceEndpoint(provider)
    released = False

    def invokeService(call: dict[str, Any], publish: Any, context: Context) -> Any:
        if released:
            return rejectedPromise(RuntimeError("Server service attachment is released"))
        return endpoint.invoke(call, publish, context)

    def release(*_: Any) -> None:
        nonlocal released
        if released:
            return
        released = True
        endpoint.dispose()
        provider.dispose()
        onRelease()

    return SimpleNamespace(invokeService=invokeService, release=release)
