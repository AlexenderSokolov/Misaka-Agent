"""Translated from pi's ``coding-agent/src/experimental/radius-auth.ts``.

An auth input is ``{"type": "token", "token"}`` or ``{"type": "file", "path"}`` (``command_options.py``). ``resolve`` returns
``{"gateway", "token"}`` or ``None``.

MISAKA fork: offline mode is ``MISAKA_OFFLINE``, as for ``ModelRuntime``, where upstream reads ``PI_OFFLINE``.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

from misaka.ai.providers.radius_config import normalizeRadiusGatewayUrl
from misaka.cli.auth_command import getAuthCredential
from misaka.core.model_runtime import ModelRuntime, ModelRuntimeAuthOverrides
from misaka.core.radius import ENV_RADIUS_GATEWAY, getRadiusGatewayUrl
from misaka.utils.abort import raceWithAbortSignal
from misaka.utils.js import jsTrim
from misaka.utils.paths import resolvePath


def _readText(path: str) -> str:
    # ``readFile(path, "utf8")`` replaces invalid bytes instead of failing.
    with open(path, encoding="utf-8", errors="replace", newline="") as file:
        return file.read()


class RadiusRelayAuthResolver:
    """Resolve explicit or stored Radius credentials anew for every relay connection attempt."""

    def __init__(self, input: dict[str, str] | None = None, gateway: str | None = None) -> None:
        self._input = input
        self._gateway = normalizeRadiusGatewayUrl(gateway if gateway is not None else getRadiusGatewayUrl())
        self._modelRuntime: asyncio.Future[ModelRuntime] | None = None

    @property
    def gateway(self) -> str:
        return self._gateway

    async def resolve(self, options: dict[str, Any]) -> dict[str, str] | None:
        """``options`` is ``{"required": bool, "signal"?}``."""
        signal = options.get("signal")
        if signal is not None:
            signal.throwIfAborted()
        if os.environ.get("MISAKA_OFFLINE") is not None:
            if options["required"]:
                raise RuntimeError("Radius relay connections are unavailable in offline mode")
            return None

        explicit = await self._explicitToken(signal)
        if explicit is not None:
            return {"gateway": self._gateway, "token": explicit}

        if self._modelRuntime is None:
            self._modelRuntime = asyncio.ensure_future(ModelRuntime.create({"refreshOnCreate": False, "allowModelNetwork": False}))
        runtime = await self._modelRuntime
        if signal is not None:
            signal.throwIfAborted()
        token = getAuthCredential(
            await runtime.getAuth("radius", ModelRuntimeAuthOverrides(minOAuthValidityMs=5 * 60_000, signal=signal))
        )
        if token is not None and len(token) > 0:
            return {"gateway": self._gateway, "token": token}
        if options["required"]:
            raise RuntimeError("Radius authentication is required; start Pi and run /login radius, then retry")
        return None

    async def _explicitToken(self, signal: Any) -> str | None:
        if self._input is None:
            return None
        value = (
            self._input["token"]
            if self._input["type"] == "token"
            else await raceWithAbortSignal(asyncio.to_thread(_readText, resolvePath(self._input["path"])), signal)
        )
        token = jsTrim(value)
        if len(token) == 0:
            raise RuntimeError("Radius authentication token must not be empty")
        return token


__all__ = ["ENV_RADIUS_GATEWAY", "RadiusRelayAuthResolver"]
