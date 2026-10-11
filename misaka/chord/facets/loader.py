"""Translated from pi's ``packages/chord/src/facets/loader.ts``."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

from misaka.chord.types import LoadedFacets


async def disposeLoadedFacets(loaded: Sequence[LoadedFacets]) -> list[Exception]:
    results = await asyncio.gather(*(entry.dispose() for entry in loaded), return_exceptions=True)
    return [result for result in results if isinstance(result, Exception)]


__all__ = ["disposeLoadedFacets"]
