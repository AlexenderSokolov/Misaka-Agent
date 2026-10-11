"""pi server's ``test/conformance.test.ts``."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest

from misaka.chord.context import BACKGROUND_CONTEXT
from misaka.server import Server, ServerHost, SessionAmbiguousError
from misaka.server.testing import (
    Deferred,
    ProtocolTestClient,
    TestServerHost,
    createTestServerServices,
)
from misaka.utils.js import UNDEFINED
from tests.chord_helpers import matches, waitFor
from tests.server_helpers import connect

serverId = "00000000-0000-4000-8000-000000000001"


@pytest.fixture
async def servers() -> AsyncIterator[set[Server]]:
    created: set[Server] = set()
    yield created
    await asyncio.gather(*(server.close() for server in created))
    created.clear()


def createServer(servers: set[Server], host: ServerHost, id: str = serverId) -> Server:
    server = Server(host, {"listeners": [], "serverId": id})
    servers.add(server)
    return server


def sessionCall(member: str, args: list[Any] | None = None) -> dict[str, Any]:
    return {"serviceId": "test.session", "member": member, "args": [] if args is None else args}


def latestAttachmentId(client: ProtocolTestClient, sessionId: str) -> str:
    for message in reversed(client.messages):
        if (
            message["type"] == "attachment"
            and message["attachment"] is not None
            and message["attachment"]["sessionId"] == sessionId
        ):
            return message["attachment"]["attachmentId"]
    raise RuntimeError(f"Missing attachment for {sessionId}")


class TestSessionProtocol:
    async def test_handshake_identifies_the_logical_server_without_listing_sessions(self, servers):
        host = TestServerHost()
        await host.seed()
        client = connect(createServer(servers, host))

        assert matches(await client.hello(), {"type": "hello", "serverId": serverId})
        assert len(host.harnesses) == 0

    async def test_rejects_a_semantically_invalid_service_call_after_envelope_decoding(self, servers):
        host = TestServerHost()
        client = connect(createServer(servers, host))
        await client.hello()
        response = client.next(lambda message: message["type"] == "response" and message["id"] == "invalid-call")
        await client.sendMessage(
            {"type": "request", "id": "invalid-call", "target": {"serverId": serverId}, "call": {"arbitrary": True}}
        )
        assert matches(await response, {"ok": False, "error": {"code": "invalid_request"}})

    async def test_attach_passes_concrete_repository_metadata_to_the_harness_host(self, servers):
        metadata = {"id": "session-1", "path": "/sessions/session-1.jsonl", "modifiedAt": 2}
        received: list[Any] = []

        class Lease:
            async def invokeService(self, call: Any, publish: Any = None, context: Any = None) -> Any:
                return UNDEFINED

            def release(self, context: Any = None) -> None:
                pass

        class Handle:
            def attachClient(self, context: Any = None) -> Lease:
                return Lease()

            async def close(self, context: Any = None) -> None:
                pass

        class Host:
            serverServices = createTestServerServices()

            async def resolveSession(self, sessionId: str, context: Any = None) -> Any:
                return metadata

            async def openSession(self, candidate: Any, context: Any = None) -> Handle:
                received.append(candidate)
                return Handle()

        client = connect(createServer(servers, Host()))
        await client.hello()

        assert matches(await client.attach(serverId, "session-1"), {"ok": True})
        assert received[0] is metadata

    async def test_routes_opaque_server_services_and_publishes_attachment_changes_out_of_band(self, servers):
        backing = TestServerHost()
        await backing.seed("session-1")
        releaseCount = 0

        class Attachment:
            def __init__(self, presentation: Any) -> None:
                self.presentation = presentation

            async def invokeService(self, call: Any, publish: Any, context: Any) -> Any:
                if call["serviceId"] != "pi.session-management":
                    raise RuntimeError("Unexpected service")
                if call["member"] == "attach" and isinstance(call["args"][0], str):
                    await self.presentation.attachSession(call["args"][0], context)
                    return None
                if call["member"] == "detach":
                    await self.presentation.detachSession(context)
                    return None
                raise RuntimeError("Unexpected service member")

            def release(self, context: Any = None) -> None:
                nonlocal releaseCount
                releaseCount += 1

        class ServerServices:
            def attachClient(self, presentation: Any, context: Any = None) -> Attachment:
                return Attachment(presentation)

        class Host:
            serverServices = ServerServices()

            def resolveSession(self, sessionId: str, context: Any) -> Any:
                return backing.resolveSession(sessionId, context)

            def openSession(self, metadata: Any, context: Any) -> Any:
                return backing.openSession(metadata, context)

        client = connect(createServer(servers, Host()))
        await client.hello()
        attached = client.next(lambda message: message["type"] == "attachment" and message["attachment"] is not None)
        assert matches(await client.attach(serverId, "session-1"), {"ok": True})
        message = await attached
        assert message["type"] == "attachment"
        assert message["attachment"]["sessionId"] == "session-1"
        assert isinstance(message["attachment"]["attachmentId"], str)
        assert backing.latestHarness("session-1").attachedClients == 1

        detached = client.next(lambda message: message["type"] == "attachment" and message["attachment"] is None)
        assert matches(
            await client.requestService(
                {"serverId": serverId}, {"serviceId": "pi.session-management", "member": "detach", "args": []}
            ),
            {"ok": True},
        )
        assert matches(await detached, {"type": "attachment", "attachment": None})
        assert backing.latestHarness("session-1").attachedClients == 0
        await client.close()
        await waitFor(lambda: releaseCount == 1)

    async def test_permits_multiple_client_attachments_per_session(self, servers):
        host = TestServerHost()
        await host.seed("session-1")
        server = createServer(servers, host)
        first = connect(server)
        second = connect(server)
        await asyncio.gather(first.hello(), second.hello())

        assert matches(await first.attach(serverId, "session-1"), {"ok": True})
        assert matches(await first.attach(serverId, "session-1"), {"ok": True})
        assert host.latestHarness("session-1").attachedClients == 1
        assert matches(await second.attach(serverId, "session-1"), {"ok": True})
        assert len(host.harnesses["session-1"]) == 1
        assert host.latestHarness("session-1").attachedClients == 2

        await first.close()
        await waitFor(lambda: host.latestHarness("session-1").attachedClients == 1)
        assert matches(await second.attach(serverId, "session-1"), {"ok": True})

    async def test_clears_connection_ownership_when_attachment_release_fails(self, servers):
        host = TestServerHost()
        await host.seed("session-1")
        errors: list[Exception] = []
        server = Server(host, {"listeners": [], "serverId": serverId, "onError": errors.append})
        servers.add(server)
        first = connect(server)
        second = connect(server)
        await asyncio.gather(first.hello(), second.hello())
        await first.attach(serverId, "session-1")
        harness = host.latestHarness("session-1")
        releaseError = Exception("release failed")
        harness.failAttachmentRelease = releaseError

        await first.close()
        await waitFor(lambda: harness.attachmentReleaseCount == 1)
        await waitFor(lambda: any(error is releaseError for error in errors))
        harness.failAttachmentRelease = None
        assert matches(await second.attach(serverId, "session-1"), {"ok": True})

    async def test_requires_the_requesting_client_to_hold_the_targeted_session_attachment(self, servers):
        host = TestServerHost()
        await asyncio.gather(host.seed("session-1"), host.seed("session-2"))
        server = createServer(servers, host)
        attached = connect(server)
        unattached = connect(server)
        await asyncio.gather(attached.hello(), unattached.hello())

        assert matches(
            await unattached.requestSessionService(serverId, "session-1", sessionCall("run")),
            {"ok": False, "error": {"code": "session_not_attached"}},
        )
        await attached.attach(serverId, "session-1")
        assert matches(
            await attached.requestSessionService(serverId, "session-2", sessionCall("run")),
            {"ok": False, "error": {"code": "session_not_attached"}},
        )
        assert matches(
            await attached.requestSessionService(serverId, "session-1", sessionCall("run", ["Hello"])),
            {"ok": True, "result": {"ok": True}},
        )
        assert host.latestHarness("session-1").serviceCalls == [sessionCall("run", ["Hello"])]

    async def test_rejects_a_stale_attachment_route_after_switching_sessions(self, servers):
        host = TestServerHost()
        await asyncio.gather(host.seed("session-1"), host.seed("session-2"))
        client = connect(createServer(servers, host))
        await client.hello()
        await client.attach(serverId, "session-1")
        firstAttachmentId = latestAttachmentId(client, "session-1")
        await client.attach(serverId, "session-2")

        assert matches(
            await client.requestService(
                {"serverId": serverId, "sessionId": "session-1", "attachmentId": firstAttachmentId},
                sessionCall("run", ["stale"]),
            ),
            {"ok": False, "error": {"code": "session_not_attached"}},
        )
        assert host.latestHarness("session-1").serviceCalls == []

    async def test_preserves_opaque_service_results_and_bounds_adapter_defects(self, servers):
        host = TestServerHost()
        await host.seed("session-1")
        client = connect(createServer(servers, host))
        await client.hello()
        await client.attach(serverId, "session-1")
        harness = host.latestHarness("session-1")
        harness.nextServiceResult = {"accepted": False, "reason": "closed"}
        assert matches(
            await client.requestSessionService(serverId, "session-1", sessionCall("run")),
            {"ok": True, "result": {"accepted": False, "reason": "closed"}},
        )

        harness.nextServiceError = Exception("private adapter detail")
        assert matches(
            await client.requestSessionService(serverId, "session-1", sessionCall("run")),
            {"ok": False, "error": {"code": "internal_error", "message": "Internal server error"}},
        )

    async def test_admits_concurrent_service_calls_to_the_attached_session(self, servers):
        host = TestServerHost()
        await host.seed("session-1")
        client = connect(createServer(servers, host))
        await client.hello()
        await client.attach(serverId, "session-1")
        harness = host.latestHarness("session-1")
        gate = harness.gateNextServiceCall()
        first = client.requestSessionService(serverId, "session-1", sessionCall("run", ["first"]))
        await gate.entered.promise
        second = client.requestSessionService(serverId, "session-1", sessionCall("run", ["second"]))

        assert matches(await second, {"ok": True})
        assert harness.serviceCalls == [sessionCall("run", ["first"]), sessionCall("run", ["second"])]
        gate.release.resolve(None)
        assert matches(await first, {"ok": True})

    async def test_keeps_attachment_demand_until_an_accepted_service_call_settles_after_disconnect(self, servers):
        host = TestServerHost()
        await host.seed("session-1")
        client = connect(createServer(servers, host))
        await client.hello()
        await client.attach(serverId, "session-1")
        harness = host.latestHarness("session-1")
        gate = harness.gateNextServiceCall()
        calling = client.requestSessionService(serverId, "session-1", sessionCall("run"))
        await gate.entered.promise

        await client.close()
        assert harness.attachedClients == 1
        gate.release.resolve(None)
        with pytest.raises(Exception, match="(?i)closed"):
            await calling
        await waitFor(lambda: harness.attachedClients == 0)

    async def test_rejects_requests_addressed_to_another_server_before_repository_access(self, servers):
        host = TestServerHost()
        await host.seed("session-1")
        client = connect(createServer(servers, host))
        await client.hello()

        assert matches(
            await client.attach("00000000-0000-4000-8000-000000000002", "session-1"),
            {"ok": False, "error": {"code": "wrong_server"}},
        )
        assert len(host.harnesses) == 0

    async def test_reports_an_unknown_session_without_creating_a_harness(self, servers):
        host = TestServerHost()
        client = connect(createServer(servers, host))
        await client.hello()

        assert matches(
            await client.attach(serverId, "missing"), {"ok": False, "error": {"code": "session_not_found"}}
        )
        assert len(host.harnesses) == 0

    async def test_rejects_an_ambiguous_session_id_without_creating_a_harness(self, servers):
        class Host:
            serverServices = createTestServerServices()

            async def resolveSession(self, sessionId: str, context: Any = None) -> Any:
                raise SessionAmbiguousError

            async def openSession(self, metadata: Any, context: Any = None) -> Any:
                raise RuntimeError("must not create a Harness for an ambiguous session")

        client = connect(createServer(servers, Host()))
        await client.hello()

        assert matches(
            await client.attach(serverId, "duplicate"), {"ok": False, "error": {"code": "session_ambiguous"}}
        )

    async def test_invalidates_a_terminated_harness_handle_and_allows_a_later_attach(self, servers):
        host = TestServerHost()
        await host.seed("session-1")
        client = connect(createServer(servers, host))
        await client.hello()
        await client.attach(serverId, "session-1")
        firstHarness = host.latestHarness("session-1")

        await firstHarness.terminate(Exception("worker crashed"))
        await firstHarness.terminated
        await waitFor(lambda: firstHarness.attachedClients == 0)
        assert firstHarness.attachmentReleaseCount == 1

        assert matches(await client.attach(serverId, "session-1"), {"ok": True})
        assert len(host.harnesses["session-1"]) == 2

    async def test_connection_loss_releases_its_attachment_while_server_shutdown_closes_the_harness(self, servers):
        host = TestServerHost()
        await host.seed("session-1")
        server = createServer(servers, host)
        client = connect(server)
        await client.hello()
        await client.attach(serverId, "session-1")
        harness = host.latestHarness("session-1")

        await client.close()
        await waitFor(lambda: harness.attachedClients == 0)
        assert harness.closeCount == 0
        await server.close()
        assert harness.closeCount == 1


class TestRoutedSessionAcquisitionFailures:
    async def test_releases_a_lease_acquired_concurrently_with_harness_termination(self, servers):
        metadata = {"id": "session-1"}
        acquiring = Deferred()
        continueAcquiring = Deferred()
        terminated = Deferred()
        releaseCount = 0

        class Lease:
            async def invokeService(self, call: Any, publish: Any = None, context: Any = None) -> Any:
                return UNDEFINED

            def release(self, context: Any = None) -> None:
                nonlocal releaseCount
                releaseCount += 1

        class Handle:
            def __init__(self) -> None:
                self.terminated = terminated.promise

            async def attachClient(self, context: Any = None) -> Lease:
                acquiring.resolve(None)
                await continueAcquiring.promise
                return Lease()

            async def close(self, context: Any = None) -> None:
                pass

        class Host:
            serverServices = createTestServerServices()

            async def resolveSession(self, sessionId: str, context: Any = None) -> Any:
                return metadata

            async def openSession(self, candidate: Any, context: Any = None) -> Handle:
                return Handle()

        client = connect(createServer(servers, Host()))
        await client.hello()
        attach = client.attach(serverId, "session-1")
        await acquiring.promise

        terminated.resolve(Exception("worker crashed"))
        continueAcquiring.resolve(None)
        assert matches(await attach, {"ok": False, "error": {"code": "server_draining"}})
        assert releaseCount == 1

    async def test_shares_a_harness_creation_failure_releases_the_session_and_allows_a_later_retry(self, servers):
        host = TestServerHost()
        await host.seed("session-1")
        host.nextOpenSessionError = Exception("Harness creation failed")
        server = createServer(servers, host)
        first = connect(server)
        second = connect(server)
        await asyncio.gather(first.hello(), second.hello())
        gate = host.gateNextOpenSession()

        firstAttach = first.attach(serverId, "session-1")
        await gate.entered.promise
        secondAttach = second.attach(serverId, "session-1")
        gate.release.resolve(None)

        results = await asyncio.gather(firstAttach, secondAttach)
        assert matches(
            list(results),
            [{"ok": False, "error": {"code": "internal_error"}}, {"ok": False, "error": {"code": "internal_error"}}],
        )
        assert host.openSessionCount == 1

        assert matches(await first.attach(serverId, "session-1"), {"ok": True})
        assert host.openSessionCount == 2
        assert len(host.harnesses["session-1"]) == 1

    async def test_closes_a_harness_acquired_while_server_shutdown_is_in_progress(self, servers):
        host = TestServerHost()
        await host.seed("session-1")
        server = createServer(servers, host)
        client = connect(server)
        await client.hello()
        gate = host.gateNextOpenSession()
        attach = client.attach(serverId, "session-1")
        await gate.entered.promise
        closing = server.close()
        gate.release.resolve(None)

        await closing
        with pytest.raises(Exception, match="(?i)closed"):
            await attach
        assert host.latestHarness("session-1").closeCount == 1

    async def test_fails_shutdown_when_an_in_flight_acquisition_cannot_release_its_harness(self, servers):
        host = TestServerHost()
        await host.seed("session-1")
        cleanupError = Exception("close failed")
        host.nextHarnessCloseError = cleanupError
        server = createServer(servers, host)
        client = connect(server)
        await client.hello()
        gate = host.gateNextOpenSession()
        attach = client.attach(serverId, "session-1")
        await gate.entered.promise

        closing = server.close()
        gate.release.resolve(None)

        with pytest.raises(ExceptionGroup, match="Failed to close routed Sessions"):
            await closing
        with pytest.raises(ExceptionGroup, match="Failed to close routed Sessions"):
            await server.closed
        with pytest.raises(Exception, match="(?i)closed"):
            await attach
        assert host.latestHarness("session-1").closeCount == 1

        servers.discard(server)
        await host.latestHarness("session-1").close(BACKGROUND_CONTEXT)
