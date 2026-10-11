"""Small helpers shared by the Harness modules.

Translated from pi's ``packages/durable/src/harness/util.ts``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from misaka.chord.context import Context, abortError
from misaka.utils.js import rejectedPromise

from ..types import Cursor, Page


class Waiters:
    """Pending waits by key. Each settles once: through ``resolve``, ``rejectAll``, or cancellation of its context."""

    def __init__(self) -> None:
        self._sets: dict[Any, dict[asyncio.Future[Any], None]] = {}

    def add(self, key: Any, context: Context) -> asyncio.Future[Any]:
        signal = context.abortSignal
        if signal is not None and signal.aborted:
            return rejectedPromise(abortError(signal))
        waiter: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        waiters = self._sets.get(key)
        if waiters is None:
            waiters = {}
            self._sets[key] = waiters
        waiters[waiter] = None
        if signal is not None:

            def onAbort() -> None:
                waiters.pop(waiter, None)
                if len(waiters) == 0 and self._sets.get(key) is waiters:
                    del self._sets[key]
                if not waiter.done():
                    waiter.set_exception(abortError(signal))

            signal.addEventListener("abort", onAbort, once=True)

            def detach(_: asyncio.Future[Any]) -> None:
                signal.removeEventListener("abort", onAbort)

            waiter.add_done_callback(detach)
        return waiter

    def keys(self) -> list[Any]:
        return list(self._sets.keys())

    def resolve(self, key: Any, value: Any) -> None:
        waiters = self._sets.pop(key, None)
        for waiter in list(waiters or ()):
            if not waiter.done():
                waiter.set_result(value)

    def rejectAll(self, error: BaseException) -> None:
        sets = list(self._sets.values())
        self._sets.clear()
        for waiters in sets:
            for waiter in list(waiters):
                if not waiter.done():
                    waiter.set_exception(error)


async def scanAll(scan: Callable[[Cursor | None], Awaitable[Page]]) -> list[Any]:
    """Every item of a paginated scan, in page order."""
    items: list[Any] = []
    cursor: Cursor | None = None
    while True:
        page = await scan(cursor)
        items.extend(page["items"])
        cursor = page.get("next")
        if cursor is None:
            break
    return items


def closedError() -> Exception:
    return RuntimeError("Harness is closed")


__all__ = ["Waiters", "closedError", "scanAll"]
