"""Translated from pi's ``packages/chord/src/services/loopback.ts``."""

from __future__ import annotations

from typing import Any

from misaka.chord.types import Context, RemoteServiceTransport, ServiceCall, ServiceMode


class _LoopbackSubscription:
    def __init__(self, subscription: Any) -> None:
        self._subscription = subscription
        self.snapshot = subscription.snapshot

    def activate(self) -> None:
        self._subscription.activate()

    def close(self, context: Context | None = None) -> Any:
        return self._subscription.close()


class _LoopbackTransport:
    def __init__(self, provider: Any) -> None:
        self._provider = provider

    async def invoke(self, call: ServiceCall, context: Context) -> Any:
        return await self._provider.invoke(call, context)

    async def subscribe(self, serviceId: str, mode: ServiceMode, listener: Any, context: Context) -> Any:
        subscription = self._provider.subscribe(serviceId, mode, listener)
        return _LoopbackSubscription(subscription)


def createLoopbackServiceTransport(provider: Any) -> RemoteServiceTransport:
    """Connects a provider to a binding without changing remote service semantics."""
    return _LoopbackTransport(provider)


__all__ = ["createLoopbackServiceTransport"]
