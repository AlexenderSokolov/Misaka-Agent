"""Translated from pi's ``coding-agent/src/experimental/client-runtime.ts``.

``openClientRuntime(command, options)`` opens live server and Session service namespaces for one experimental presentation
and resolves to ``SimpleNamespace(servers, dispose)``. A route is ``{"transport": "unix", "serverId", "path"}`` or
``{"transport": "radius", "serverId"}``; a server is ``SimpleNamespace(route, client, server, session)`` (the two service
sources of ``services/connection.py``). ``activateBuiltinClientServices(server)`` resolves to the same namespace with
``directory``, ``management``, ``plugins``, ``models``, ``agent`` and ``transcript`` added.

``command`` is the parsed ``client`` command (``cli/experimental/commands/client.py``); ``options`` is ``{"directory"?}``,
the directory searched when ``--connect`` is omitted (``PI_SERVER_DIR`` or the home's ``run/server`` by default).
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any

from misaka.chord.context import BACKGROUND_CONTEXT
from misaka.client import ServerError, connectClient
from misaka.client.unix import createUnixTransportFactory, discoverUnixServers
from misaka.experimental.radius_auth import RadiusRelayAuthResolver
from misaka.experimental.radius_relay import (
    RadiusClientReconnect,
    createRadiusClientTransportFactory,
)
from misaka.experimental.server import (
    ENV_SERVER_ID,
    activateServer,
    resolveServerDirectory,
    resolveSessionDirectory,
)
from misaka.experimental.services.agent_controller import AgentController
from misaka.experimental.services.connection import (
    createServerServiceSource,
    createSessionServiceSource,
)
from misaka.experimental.services.models import Models
from misaka.experimental.services.plugins import PresentationPlugins
from misaka.experimental.services.sessions import SessionDirectory, SessionManagement
from misaka.experimental.services.transcript import Transcript
from misaka.protocol import isServerId
from misaka.utils.js import aggregateError, allSettled, promiseAll


def _ignoreAccess() -> None:
    pass


def _ignoreError(error: Exception) -> None:
    pass


async def openClientRuntime(command: dict[str, Any], options: dict[str, Any] | None = None) -> SimpleNamespace:
    """Open live server/session service namespaces for one experimental presentation."""
    options = options or {}
    connect = command.get("connect")
    if command.get("auth") is not None and (connect is None or connect["transport"] != "radius"):
        raise RuntimeError("Authentication is only supported for experimental Radius connections")
    if command.get("provider") is not None and command.get("model") is None:
        raise RuntimeError("Server model provider requires a model")
    if connect is not None and command.get("model") is not None:
        raise RuntimeError("Model selection is only valid when automatically activating a new server")
    if connect is not None and connect["transport"] == "radius" and command.get("pluginPackages") is not None:
        raise RuntimeError("Plugin package paths can only be configured on a local Unix server")
    directory = resolveServerDirectory(options.get("directory"))
    routes: list[dict[str, Any]]
    activatedClient: Any = None
    if connect is not None:
        routes = [
            {"transport": "radius", "serverId": connect["serverId"]}
            if connect["transport"] == "radius"
            else {"transport": "unix", **routeFromExplicitPath(connect["path"])}
        ]
    else:
        routes = [{"transport": "unix", **route} for route in await discoverUnixServers({"directory": directory})]
        if len(routes) > 0 and command.get("model") is not None:
            raise RuntimeError("Model selection is only valid when automatically activating a new server")
        if len(routes) == 0:
            activated = await activateServer(
                {
                    "directory": directory,
                    "requestedServerId": os.environ.get(ENV_SERVER_ID),
                    "sessionDir": resolveSessionDirectory(),
                    "provider": command.get("provider"),
                    "model": command.get("model"),
                }
            )
            routes = [{"transport": "unix", **activated["route"]}]
            activatedClient = activated["client"]
    if command.get("pluginPackages") is not None and len(routes) != 1:
        raise RuntimeError("Plugin selection requires exactly one local server")

    clients: list[Any] = []
    reconnectors: list[RadiusClientReconnect] = []
    serviceSources: list[Any] = []
    servers: list[SimpleNamespace] = []
    disposed = False

    async def dispose() -> None:
        nonlocal disposed
        if disposed:
            return
        disposed = True
        reconnectResults = await allSettled([reconnector.dispose() for reconnector in reconnectors])
        sourceResults = await allSettled([source.dispose(BACKGROUND_CONTEXT) for source in serviceSources])
        clientResults = await allSettled([client.dispose() for client in clients])
        errors = [
            result.reason
            for result in [*reconnectResults, *sourceResults, *clientResults]
            if result.status == "rejected" and result.reason is not None
        ]
        if len(errors) == 1:
            raise errors[0]
        if len(errors) > 1:
            raise aggregateError(errors, "Failed to dispose experimental client runtime")

    try:
        for route in routes:
            client = activatedClient
            if client is None:
                try:
                    client = await connectClient(
                        {
                            "serverId": route["serverId"],
                            "transportFactory": (
                                createUnixTransportFactory({"path": route["path"]})
                                if route["transport"] == "unix"
                                else createRadiusClientTransportFactory(
                                    {
                                        "serverId": route["serverId"],
                                        "auth": RadiusRelayAuthResolver(command.get("auth")),
                                    }
                                )
                            ),
                        }
                    )
                except Exception as error:
                    if (
                        connect is not None
                        or route["transport"] != "unix"
                        or not isinstance(error, ServerError)
                        or error.code != "version"
                    ):
                        raise
                    client = (
                        await activateServer(
                            {
                                "directory": directory,
                                "requestedServerId": route["serverId"],
                                "sessionDir": resolveSessionDirectory(),
                            }
                        )
                    )["client"]
            activatedClient = None
            clients.append(client)
            server = createServerServiceSource(client)
            serviceSources.append(server)
            if route["transport"] == "radius":
                reconnectServices = server.open(
                    {"services": [SessionManagement], "assertAccess": _ignoreAccess, "onError": _ignoreError}
                )
                reconnectManagement = reconnectServices.use(SessionManagement)
                reconnectors.append(RadiusClientReconnect(client, _reattacher(reconnectServices, reconnectManagement)))
            session = createSessionServiceSource(client)
            serviceSources.append(session)
            servers.append(SimpleNamespace(route=route, client=client, server=server, session=session))
        return SimpleNamespace(servers=servers, dispose=dispose)
    except BaseException as error:
        try:
            await dispose()
        except Exception as cleanupError:  # noqa: BLE001 - reported together with the startup error
            raise aggregateError([error, cleanupError], "Experimental client startup and cleanup failed") from None
        raise


def _reattacher(services: Any, management: Any) -> Any:
    async def reattach(sessionId: str) -> None:
        await services.ready(BACKGROUND_CONTEXT)
        await management.attach(sessionId, BACKGROUND_CONTEXT)

    return reattach


async def activateBuiltinClientServices(server: SimpleNamespace) -> SimpleNamespace:
    """Acquire and connect the built-in service facades used by the non-interactive client."""
    serverServices = server.server.open(
        {
            "services": [SessionDirectory, SessionManagement, PresentationPlugins],
            "assertAccess": _ignoreAccess,
            "onError": _ignoreError,
        }
    )
    sessionServices = server.session.open(
        {"services": [Models, AgentController, Transcript], "assertAccess": _ignoreAccess, "onError": _ignoreError}
    )
    directory = serverServices.use(SessionDirectory)
    remoteManagement = serverServices.use(SessionManagement)

    async def remove(sessionId: str, context: Any) -> None:
        attachment = server.client.attachment
        removesCurrentAttachment = attachment is not None and attachment["sessionId"] == sessionId
        await remoteManagement.remove(sessionId, context)
        if removesCurrentAttachment:
            await server.session.whenDetached(context)

    async def attach(sessionId: str, context: Any) -> None:
        await remoteManagement.attach(sessionId, context)
        await server.session.whenAttached(sessionId, context)

    async def detach(context: Any) -> None:
        await remoteManagement.detach(context)
        await server.session.whenDetached(context)

    management = SimpleNamespace(
        create=lambda options, context: remoteManagement.create(options, context),
        remove=remove,
        attach=attach,
        detach=detach,
    )
    plugins = serverServices.use(PresentationPlugins)
    models = sessionServices.use(Models)
    agent = sessionServices.use(AgentController)
    transcript = sessionServices.use(Transcript)
    await promiseAll([serverServices.ready(BACKGROUND_CONTEXT), sessionServices.ready(BACKGROUND_CONTEXT)])
    return SimpleNamespace(
        **vars(server),
        directory=directory,
        management=management,
        plugins=plugins,
        models=models,
        agent=agent,
        transcript=transcript,
    )


def routeFromExplicitPath(path: str) -> dict[str, str]:
    # ``path.basename`` ignores trailing separators.
    name = os.path.basename(path.rstrip("/"))
    serverId = name[: -len(".sock")] if name.endswith(".sock") else ""
    if not isServerId(serverId):
        raise RuntimeError("--connect path must end with <uuidv4-server-id>.sock")
    return {"serverId": serverId, "path": path}
