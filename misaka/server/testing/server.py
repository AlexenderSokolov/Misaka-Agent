"""Translated from pi's ``packages/server/src/testing/server.ts``."""

from __future__ import annotations

from typing import Any, NotRequired, TypedDict

from misaka.utils.js import UNDEFINED, nullishCoalesce

from ..server import Server
from ..types import ServerHost, ServerOptions
from .host import TestServerHost


class TestServerOptions(TypedDict):
    """``listeners`` is required; ``host`` defaults to a ``TestServerHost`` and ``serverId`` to a fixed test UUID."""

    listeners: Any
    host: NotRequired[ServerHost]
    serverId: NotRequired[str]
    maxFrameLength: NotRequired[int]
    handshakeTimeoutMs: NotRequired[int]
    onError: NotRequired[Any]


class TestServer:
    __test__ = False

    def __init__(self, server: Server, host: ServerHost) -> None:
        self.server = server
        self.host = host


def createTestServer(options: TestServerOptions) -> TestServer:
    """Create an unstarted Server with deterministic defaults for transport conformance tests."""
    host = nullishCoalesce(options.get("host"), None)
    if host is None:
        host = TestServerHost()
    serverOptions: ServerOptions = {
        "listeners": options.get("listeners", UNDEFINED),
        "serverId": nullishCoalesce(options.get("serverId"), "00000000-0000-4000-8000-000000000001"),
    }
    if "maxFrameLength" in options:
        serverOptions["maxFrameLength"] = options["maxFrameLength"]
    if "handshakeTimeoutMs" in options:
        serverOptions["handshakeTimeoutMs"] = options["handshakeTimeoutMs"]
    if "onError" in options:
        serverOptions["onError"] = options["onError"]
    return TestServer(Server(host, serverOptions), host)


__all__ = ["TestServer", "TestServerOptions", "createTestServer"]
