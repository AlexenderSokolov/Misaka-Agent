"""Translated from pi's ``coding-agent/src/experimental/client.ts``.

``runClient(command, options)`` discovers servers, then lists Sessions, attaches to one, or creates one for a prompt. The
result is ``{"kind": "list", "sessions": [{"serverId", "sessionId"}]}``, ``{"kind": "attached", "serverId", "sessionId"}`` or
``{"kind": "prompted", "serverId", "sessionId", "text"}``. ``options`` is ``{"directory"?}``, the directory searched when
``--connect`` is omitted (``PI_SERVER_DIR`` or the home's ``run/server`` by default).
"""

from __future__ import annotations

from typing import Any

from misaka.chord.context import BACKGROUND_CONTEXT
from misaka.experimental.client_runtime import (
    activateBuiltinClientServices,
    openClientRuntime,
)
from misaka.utils.collation import localeSortKey
from misaka.utils.js import pathResolve, promiseAll


async def runClient(command: dict[str, Any], options: dict[str, Any] | None = None) -> dict[str, Any]:
    """Discover servers, then list Sessions, attach to one, or create one for a prompt."""
    options = options or {}
    runtime = await openClientRuntime(command, {"directory": options.get("directory")})
    try:
        discovered = await promiseAll([activateBuiltinClientServices(server) for server in runtime.servers])
        sessionId = command.get("sessionId")
        if sessionId is None and command.get("prompt") is None:
            sessions = [
                {"serverId": server.route["serverId"], "sessionId": session["sessionId"]}
                for server in discovered
                for session in server.directory.state.value["sessions"]
            ]
            sessions.sort(key=lambda session: (localeSortKey(session["serverId"]), localeSortKey(session["sessionId"])))
            return {"kind": "list", "sessions": sessions}

        if sessionId is None:
            if len(discovered) != 1:
                raise RuntimeError("Client prompt requires exactly one discovered server to create a Session")
            match = discovered[0]
            sessionId = (await match.management.create({}, BACKGROUND_CONTEXT))["sessionId"]
        else:
            matches = [
                candidate
                for candidate in discovered
                if any(session["sessionId"] == sessionId for session in candidate.directory.state.value["sessions"])
            ]
            if len(matches) > 1:
                raise RuntimeError(f"Session {sessionId} is available from more than one server")
            if matches:
                match = matches[0]
            else:
                connect = command.get("connect")
                if (
                    (connect is not None and connect["transport"] == "radius")
                    or command.get("prompt") is None
                    or len(discovered) != 1
                ):
                    raise RuntimeError(f"No discovered server contains session {sessionId}")
                match = discovered[0]
                await match.management.create({"id": sessionId}, BACKGROUND_CONTEXT)
        pluginPackages = command.get("pluginPackages")
        await match.plugins.prepareSession(
            {
                "sessionId": sessionId,
                "packagePaths": None if pluginPackages is None else [pathResolve(path) for path in pluginPackages],
            },
            BACKGROUND_CONTEXT,
        )
        await match.management.attach(sessionId, BACKGROUND_CONTEXT)
        if command.get("prompt") is None:
            return {"kind": "attached", "serverId": match.route["serverId"], "sessionId": sessionId}

        response = await match.agent.prompt({"message": command["prompt"], "images": None}, BACKGROUND_CONTEXT)
        if not response["accepted"]:
            raise RuntimeError(response["error"]["message"])
        result = await match.agent.waitForPrompt(response["operationId"], BACKGROUND_CONTEXT)
        if result["status"] == "unanswered":
            raise RuntimeError(f"Prompt was not answered: {result['reason']}")
        return {"kind": "prompted", "serverId": match.route["serverId"], "sessionId": sessionId, "text": result["text"]}
    finally:
        await runtime.dispose()
