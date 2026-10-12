"""Test doubles for a server host.

Translated from pi's ``packages/server/src/testing/host.ts``. ``Deferred`` and the gates create futures, so they
belong inside a running event loop.
"""

from __future__ import annotations

import asyncio
from typing import Any

from misaka.chord.types import Context

from ..errors import SessionNotFoundError
from ..types import SessionMetadata


class Deferred:
    """A promise with its resolver."""

    def __init__(self) -> None:
        self.promise: asyncio.Future[Any] = asyncio.get_running_loop().create_future()

    def resolve(self, value: Any = None) -> None:
        if not self.promise.done():
            self.promise.set_result(value)


class OpenGate:
    def __init__(self) -> None:
        self.entered = Deferred()
        self.release = Deferred()


class _HarnessLease:
    def __init__(self, harness: TestHarness) -> None:
        self._harness = harness
        self._released = False

    def invokeService(self, call: Any, publish: Any = None, context: Any = None) -> Any:
        return self._harness.invokeService(call)

    def release(self, context: Context | None = None) -> None:
        if self._released:
            return
        harness = self._harness
        harness.attachmentReleaseCount += 1
        if harness.failAttachmentRelease is not None:
            raise harness.failAttachmentRelease
        self._released = True
        harness.attachedClients -= 1


class TestHarness:
    __test__ = False

    def __init__(self, metadata: SessionMetadata) -> None:
        self.metadata = metadata
        self.closed = Deferred()
        self._termination = Deferred()
        self.terminated = self._termination.promise
        self.attachedClients = 0
        self.attachmentReleaseCount = 0
        self.closeCount = 0
        self.serviceCalls: list[Any] = []
        self.failAttachmentRelease: Exception | None = None
        self.failClose: Exception | None = None
        self.nextServiceError: Exception | None = None
        self.nextServiceResult: Any = {"ok": True}
        self._nextCloseGate: OpenGate | None = None
        self._nextServiceGate: OpenGate | None = None

    def attachClient(self, _context: Context | None = None) -> _HarnessLease:
        self.attachedClients += 1
        return _HarnessLease(self)

    async def invokeService(self, call: Any) -> Any:
        self.serviceCalls.append(call)
        if self.nextServiceError is not None:
            error = self.nextServiceError
            self.nextServiceError = None
            raise error
        gate = self._nextServiceGate
        if gate is not None:
            self._nextServiceGate = None
            gate.entered.resolve(None)
            await gate.release.promise
        result = self.nextServiceResult
        self.nextServiceResult = {"ok": True}
        return result

    async def close(self, _context: Context | None = None) -> None:
        self.closeCount += 1
        gate = self._nextCloseGate
        if gate is not None:
            self._nextCloseGate = None
            gate.entered.resolve(None)
            await gate.release.promise
        if self.failClose is not None:
            error = self.failClose
            self.failClose = None
            raise error
        self.closed.resolve(None)
        self._termination.resolve(None)

    async def terminate(self, error: Exception) -> None:
        self._termination.resolve(error)

    def gateNextClose(self) -> OpenGate:
        gate = OpenGate()
        self._nextCloseGate = gate
        return gate

    def gateNextServiceCall(self) -> OpenGate:
        gate = OpenGate()
        self._nextServiceGate = gate
        return gate


class _TestServerServiceAttachment:
    def __init__(self, presentation: Any) -> None:
        self._presentation = presentation

    async def invokeService(self, call: Any, publish: Any = None, context: Any = None) -> Any:
        presentation = self._presentation
        if (
            "instance" not in call
            and call["serviceId"] == "pi.session-management"
            and call["member"] == "attach"
            and len(call["args"]) == 1
            and isinstance(call["args"][0], str)
        ):
            await presentation.attachSession(call["args"][0], context)
            return None
        if (
            "instance" not in call
            and call["serviceId"] == "pi.session-management"
            and call["member"] == "detach"
            and len(call["args"]) == 0
        ):
            await presentation.detachSession(context)
            return None
        raise RuntimeError(f"Unsupported test server service {call['serviceId']}.{call['member']}")

    def release(self, context: Context | None = None) -> None:
        pass


class _TestServerServices:
    def attachClient(self, presentation: Any, context: Context | None = None) -> _TestServerServiceAttachment:
        return _TestServerServiceAttachment(presentation)


def createTestServerServices() -> _TestServerServices:
    return _TestServerServices()


class TestServerHost:
    __test__ = False

    def __init__(self) -> None:
        self.serverServices = createTestServerServices()
        self.sessions: dict[str, SessionMetadata] = {}
        self.harnesses: dict[str, list[TestHarness]] = {}
        self.openSessionCount = 0
        self.nextOpenSessionError: Exception | None = None
        self.nextHarnessCloseError: Exception | None = None
        self._nextOpenSessionGate: OpenGate | None = None

    async def resolveSession(self, sessionId: str, _context: Context | None = None) -> SessionMetadata:
        metadata = self.sessions.get(sessionId)
        if metadata is None:
            raise SessionNotFoundError(f"Unknown session: {sessionId}")
        return metadata

    async def openSession(self, metadata: SessionMetadata, _context: Context | None = None) -> TestHarness:
        self.openSessionCount += 1
        gate = self._nextOpenSessionGate
        if gate is not None:
            self._nextOpenSessionGate = None
            gate.entered.resolve(None)
            await gate.release.promise
        if metadata["id"] not in self.sessions:
            raise SessionNotFoundError(f"Unknown session: {metadata['id']}")
        if self.nextOpenSessionError is not None:
            error = self.nextOpenSessionError
            self.nextOpenSessionError = None
            raise error
        harness = TestHarness(metadata)
        if self.nextHarnessCloseError is not None:
            harness.failClose = self.nextHarnessCloseError
            self.nextHarnessCloseError = None
        harnesses = self.harnesses.get(metadata["id"], [])
        harnesses.append(harness)
        self.harnesses[metadata["id"]] = harnesses
        return harness

    async def seed(self, id: str = "session-1") -> SessionMetadata:
        metadata: SessionMetadata = {"id": id}
        self.sessions[id] = metadata
        return metadata

    def gateNextOpenSession(self) -> OpenGate:
        gate = OpenGate()
        self._nextOpenSessionGate = gate
        return gate

    def latestHarness(self, id: str) -> TestHarness:
        harnesses = self.harnesses.get(id)
        if not harnesses:
            raise RuntimeError(f"No harness for {id}")
        return harnesses[-1]


__all__ = [
    "Deferred",
    "OpenGate",
    "TestHarness",
    "TestServerHost",
    "createTestServerServices",
]
