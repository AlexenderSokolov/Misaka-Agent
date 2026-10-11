"""Translated from pi's ``packages/chord/src/services/state-internals.ts``."""

from __future__ import annotations

import weakref
from collections.abc import Callable, Sequence
from typing import Any, Protocol

from misaka.chord.delta import Op
from misaka.chord.types import Context


class ReplicatedStateInternals(Protocol):
    def snapshot(self) -> dict[str, Any]:
        """Atomically capture the immutable value and its matching publication sequence: ``{"value",
        "sequence"}``."""
        ...

    def subscribe(self, listener: Callable[[Sequence[Op], int, Context], None]) -> Callable[[], None]: ...


# MISAKA fork: `WeakMap<object, ...>`. The state objects are ordinary class instances, which can be weakly
# referenced.
sources: weakref.WeakKeyDictionary[Any, ReplicatedStateInternals] = weakref.WeakKeyDictionary()


def registerReplicatedStateInternals(value: Any, internals: ReplicatedStateInternals) -> None:
    sources[value] = internals


def getReplicatedStateInternals(value: Any) -> ReplicatedStateInternals | None:
    try:
        return sources.get(value)
    except TypeError:
        return None


__all__ = ["ReplicatedStateInternals", "getReplicatedStateInternals", "registerReplicatedStateInternals"]
