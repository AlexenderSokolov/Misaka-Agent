"""pi server's ``test/listener.test.ts``."""

from __future__ import annotations

from typing import Any

import pytest

from misaka.server.testing import createTestServer


class RecordingListener:
    def __init__(self, startError: Exception | None = None) -> None:
        self.accept: Any = None
        self.startCount = 0
        self.closeCount = 0
        self.startError = startError

    async def start(self, accept: Any) -> None:
        self.startCount += 1
        self.accept = accept
        if self.startError is not None:
            raise self.startError

    async def close(self) -> None:
        self.closeCount += 1


class TestServerListenerComposition:
    async def test_starts_and_closes_every_configured_listener(self):
        first = RecordingListener()
        second = RecordingListener()
        server = createTestServer({"listeners": [first, second]}).server

        await server.start()
        assert callable(first.accept)
        assert callable(second.accept)

        await server.close()
        assert first.closeCount == 1
        assert second.closeCount == 1

    async def test_closes_previously_started_listeners_when_startup_fails(self):
        first = RecordingListener()
        failure = Exception("listener failed")
        second = RecordingListener(failure)
        server = createTestServer({"listeners": [first, second]}).server

        with pytest.raises(Exception, match="listener failed") as info:
            await server.start()
        assert info.value is failure
        assert first.closeCount == 1
        assert second.closeCount == 0
