"""A persisted remote catalog overlay on a static built-in provider.

Translated from pi's ``coding-agent/src/core/remote-catalog-provider.ts``.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from datetime import UTC
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import quote

from pydantic import ValidationError

from misaka.ai.models import ModelsPublication, RefreshModelsContext
from misaka.ai.models_store import ModelsStoreEntry
from misaka.ai.types import ANY_MODEL_ADAPTER, AnyModel
from misaka.ai.utils.model_operations import getModelType, isModelType
from misaka.ai.utils.pi_user_agent import getPiUserAgent
from misaka.utils.management_http import FetchRetryOptions, fetchWithRetry
from misaka.utils.values import signal_aborted
from misaka.utils.whatwg_url import URL

DEFAULT_CATALOG_BASE_URL = "https://pi.dev"
REMOTE_CATALOG_ATTEMPT_TIMEOUT_MS = 4_000
REMOTE_CATALOG_REFRESH_INTERVAL_MS = 4 * 60 * 60 * 1000
# Model types this client can consume. Sent as `?types=` so the catalog server
# returns the full-type shard instead of the chat-only one served to clients
# that predate model types. A server that ignores the parameter still returns
# the chat-only shard, which this client handles unchanged.
REMOTE_CATALOG_MODEL_TYPES: tuple[str, ...] = ("chat", "image", "classifier")


def _now() -> int:
    return int(time.time() * 1000)


def _isSupportedModelType(model: Mapping[str, Any]) -> bool:
    return "type" not in model or (
        isinstance(model["type"], str) and model["type"] in REMOTE_CATALOG_MODEL_TYPES
    )


def _mergeModels(baseline: Sequence[AnyModel], dynamic: Sequence[AnyModel]) -> list[AnyModel]:
    merged: dict[tuple[str, str], AnyModel] = {}
    for model in [*baseline, *dynamic]:
        merged[(getModelType(model), model.id)] = model
    return list(merged.values())


def _toModel(entry: dict[str, Any]) -> AnyModel | None:
    # MISAKA fork: catalog entries become typed models here. A malformed entry is left out
    # instead of failing the whole catalog.
    try:
        return ANY_MODEL_ADAPTER.validate_python(entry)
    except ValidationError:
        return None


def _parseCatalog(providerId: str, value: Any) -> list[AnyModel]:
    entries = (
        value
        if isinstance(value, list)
        else value["models"]
        if isinstance(value, dict) and isinstance(value.get("models"), list)
        else list(value.values())
        if isinstance(value, dict)
        else None
    )
    if entries is None:
        raise RuntimeError(f'Invalid model catalog for provider "{providerId}"')
    models = (
        _toModel({**entry, "provider": providerId})
        for entry in entries
        if isinstance(entry, dict) and "id" in entry and _isSupportedModelType(entry)
    )
    return [model for model in models if model is not None]


def _remoteModels(entry: ModelsStoreEntry | None, localGeneratedAt: int | None) -> list[AnyModel]:
    if entry is None:
        return []
    if localGeneratedAt is not None and (entry.lastModified is None or entry.lastModified <= localGeneratedAt):
        return []
    return list(entry.models)


def _parseHttpDate(value: str | None) -> float:
    """``Date.parse`` of an HTTP date: epoch milliseconds, NaN when unreadable."""
    try:
        parsed = parsedate_to_datetime(value or "")
    except (TypeError, ValueError):
        return float("nan")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.timestamp() * 1000


def withRemoteCatalog(
    provider: Any,
    catalogBaseUrl: str = DEFAULT_CATALOG_BASE_URL,
    localGeneratedAt: int | None = None,
) -> Any:
    """Add a persisted pi.dev catalog overlay to a static built-in provider."""
    dynamicModels: list[AnyModel] = []

    class _RemoteCatalogProvider:
        def __getattr__(self, name: str) -> Any:
            return getattr(provider, name)

        def getModels(self) -> list[AnyModel]:
            return _mergeModels(
                provider.getModels(),
                [model for model in dynamicModels if isModelType(model, "chat")],
            )

        def getAllModels(self) -> list[AnyModel]:
            getAllModels = getattr(provider, "getAllModels", None)
            return _mergeModels(getAllModels() if getAllModels is not None else provider.getModels(), dynamicModels)

        async def refreshModels(self, context: RefreshModelsContext) -> None:
            stored = context.stored
            restored = [model for model in _remoteModels(stored, localGeneratedAt) if model.provider == provider.id]

            def applyRestored() -> None:
                nonlocal dynamicModels
                dynamicModels = restored

            if not await context.publish(ModelsPublication(update=applyRestored)):
                return
            if not context.allowNetwork or signal_aborted(context.signal):
                return
            if (
                not context.force
                and stored is not None
                and stored.checkedAt is not None
                and stored.lastModified is not None
                and _now() - stored.checkedAt < REMOTE_CATALOG_REFRESH_INTERVAL_MS
            ):
                return

            # Only revalidate when a cached body backs the validator, so a 304 can never
            # leave the overlay empty.
            validator = stored.etag if stored is not None and len(stored.models) > 0 else None
            # `encodeURIComponent` leaves !~*'() alone too.
            url = URL(f"/api/models/providers/{quote(provider.id, safe="!~*'()")}", catalogBaseUrl)
            url.searchParams.set("types", ",".join(REMOTE_CATALOG_MODEL_TYPES))
            response = await fetchWithRetry(
                url,
                {
                    "headers": {
                        "accept": "application/json",
                        # MISAKA fork: misaka's anonymous user agent (ai/utils/pi_user_agent.py), not pi's `pi/<version>` one.
                        "User-Agent": getPiUserAgent(),
                        **({"if-none-match": validator} if validator else {}),
                    },
                    "signal": context.signal,
                },
                FetchRetryOptions(attemptTimeoutMs=REMOTE_CATALOG_ATTEMPT_TIMEOUT_MS),
            )
            if signal_aborted(context.signal):
                return
            checkedAt = _now()
            # Unchanged: dynamicModels already holds the stored overlay, so only the
            # freshness window moves.
            if response.status == 304 and stored is not None:
                await context.publish(
                    ModelsPublication(persist=stored.model_copy(update={"checkedAt": checkedAt}), persist_is_set=True)
                )
                return
            if response.status == 404 or response.status == 501:
                await context.publish(
                    ModelsPublication(
                        persist=(stored or ModelsStoreEntry(models=[])).model_copy(
                            update={"checkedAt": checkedAt, "lastModified": 0, "etag": None}
                        ),
                        persist_is_set=True,
                    )
                )
                return
            if not response.ok:
                # Transient failure: the cached body and its validator stay valid, so keep the
                # etag and let the next refresh revalidate instead of downloading the catalog.
                await context.publish(
                    ModelsPublication(
                        persist=(stored or ModelsStoreEntry(models=[])).model_copy(update={"checkedAt": checkedAt}),
                        persist_is_set=True,
                    )
                )
                raise RuntimeError(f"Model catalog request failed for {provider.id}: {response.status}")
            refreshed = _parseCatalog(provider.id, await response.json())
            lastModified = _parseHttpDate(response.headers.get("last-modified"))
            if signal_aborted(context.signal):
                return
            entry = ModelsStoreEntry(
                models=refreshed,
                checkedAt=checkedAt,
                lastModified=0 if math.isnan(lastModified) else int(lastModified),
                etag=response.headers.get("etag"),
            )
            published = _remoteModels(entry, localGeneratedAt)

            def applyPublished() -> None:
                nonlocal dynamicModels
                dynamicModels = published

            await context.publish(ModelsPublication(persist=entry, persist_is_set=True, update=applyPublished))

    return _RemoteCatalogProvider()


__all__ = [
    "DEFAULT_CATALOG_BASE_URL",
    "REMOTE_CATALOG_MODEL_TYPES",
    "REMOTE_CATALOG_REFRESH_INTERVAL_MS",
    "withRemoteCatalog",
]
