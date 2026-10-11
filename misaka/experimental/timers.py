"""The timers the experimental lifecycle classes use.

MISAKA fork: pi's classes call ``setTimeout`` and ``clearTimeout`` and its tests replace them with fake timers. Here a class
takes a ``timers`` object with ``setTimeout(callback, delayMs)`` and ``clearTimeout(handle)``; this one runs them on the
running event loop, and the tests pass a fake with a controllable clock.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any


class LoopTimers:
    def setTimeout(self, callback: Callable[[], None], delayMs: float) -> asyncio.TimerHandle:
        return asyncio.get_running_loop().call_later(delayMs / 1000, callback)

    def clearTimeout(self, handle: Any) -> None:
        if handle is not None:
            handle.cancel()


DEFAULT_TIMERS = LoopTimers()
