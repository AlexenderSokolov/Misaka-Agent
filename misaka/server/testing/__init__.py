"""Test doubles for ``misaka.server``: a protocol client, an in-memory host and a server factory.

Translated from pi's ``packages/server/src/testing/index.ts``.
"""

from __future__ import annotations

from .client import ProtocolTestClient, WireChannel, connectUnixTestClient
from .host import Deferred, TestHarness, TestServerHost, createTestServerServices
from .server import TestServer, TestServerOptions, createTestServer

__all__ = [
    "Deferred",
    "ProtocolTestClient",
    "TestHarness",
    "TestServer",
    "TestServerHost",
    "TestServerOptions",
    "WireChannel",
    "connectUnixTestClient",
    "createTestServer",
    "createTestServerServices",
]
