"""pi coding-agent's ``test/experimental-service-binding.ts``: helpers that bind services to a client without a facet host.

Both helpers take ``(client, options)`` with ``options`` ``{"services": [...], "onError"?}`` and return a ``SimpleNamespace`` that is
a service source (``acceptsUnavailableServices``, ``catalogue``, ``open`` and the ``connection`` or ``attachment`` state) and
the bound services (``use``, ``observe``, ``ready``, ``dispose``).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from misaka.chord import Context
from misaka.chord.context import BACKGROUND_CONTEXT
from misaka.experimental.services.connection import (
    createServerServiceSource,
    createSessionServiceSource,
)
from misaka.utils.js import allSettled, startTask


def _ignore(error: Exception) -> None:
    pass


def _allowAccess() -> None:
    pass


def _open(source: Any, options: dict[str, Any]) -> tuple[Any, Any]:
    onError = options.get("onError") or _ignore
    services = source.open({"services": options["services"], "assertAccess": _allowAccess, "onError": onError})
    activation = startTask(services.ready(BACKGROUND_CONTEXT))

    def reported(future: Any) -> None:
        if not future.cancelled() and future.exception() is not None:
            onError(future.exception())

    activation.add_done_callback(reported)
    return services, activation


def createServerServiceBinding(client: Any, options: dict[str, Any]) -> SimpleNamespace:
    """Test helper for binding server services without a facet host."""
    source = createServerServiceSource(client, options)
    services, activation = _open(source, options)

    async def ready(context: Context) -> None:
        await activation
        await services.ready(context)

    async def dispose(context: Context) -> None:
        results = await allSettled([services.dispose(context), source.dispose(context)])
        throwFailures(results, "Failed to dispose server service binding")

    return SimpleNamespace(
        acceptsUnavailableServices=source.acceptsUnavailableServices,
        connection=source.connection,
        catalogue=lambda context: source.catalogue(context),
        open=lambda openOptions: source.open(openOptions),
        use=lambda service: services.use(service),
        observe=lambda service, handler: services.observe(service, handler),
        ready=ready,
        dispose=dispose,
    )


class _SessionServiceBinding:
    def __init__(self, source: Any, services: Any, activation: Any) -> None:
        self._source = source
        self._services = services
        self._activation = activation
        self.attachment = source.attachment

    @property
    def acceptsUnavailableServices(self) -> bool:
        return bool(self._source.acceptsUnavailableServices)

    def catalogue(self, context: Context) -> Any:
        return self._source.catalogue(context)

    def open(self, openOptions: dict[str, Any]) -> Any:
        return self._source.open(openOptions)

    def use(self, service: Any) -> Any:
        return self._services.use(service)

    def observe(self, service: Any, handler: Any) -> Any:
        return self._services.observe(service, handler)

    async def ready(self, context: Context) -> None:
        await self._activation
        attachment = self._source.attachment.value
        if attachment["status"] == "detached":
            await self._source.whenDetached(context)
        else:
            await self._source.whenAttached(attachment["sessionId"], context)

    def whenAttached(self, sessionId: str, context: Context) -> Any:
        return self._source.whenAttached(sessionId, context)

    def whenDetached(self, context: Context) -> Any:
        return self._source.whenDetached(context)

    async def dispose(self, context: Context) -> None:
        results = await allSettled([self._services.dispose(context), self._source.dispose(context)])
        throwFailures(results, "Failed to dispose Session service binding")


def createSessionServiceBinding(client: Any, options: dict[str, Any]) -> _SessionServiceBinding:
    """Test helper for binding selected-Session services without a facet host."""
    source = createSessionServiceSource(client, options)
    services, activation = _open(source, options)
    return _SessionServiceBinding(source, services, activation)


def throwFailures(results: list[Any], message: str) -> None:
    errors = [result.reason for result in results if result.status == "rejected"]
    if len(errors) == 1:
        raise errors[0]
    if len(errors) > 1:
        raise ExceptionGroup(message, errors)
