"""Unix-domain socket transport for the experimental server.

Translated from pi's ``packages/server/src/transports/unix/index.ts``. Unix sockets are POSIX-only: on Windows
``start()`` fails with an error that says so.
"""

from __future__ import annotations

from .address import getUnixSocketPath
from .listener import createUnixListener
from .preset import createUnixServer
from .types import UnixListenerOptions, UnixServerOptions

__all__ = [
    "UnixListenerOptions",
    "UnixServerOptions",
    "createUnixListener",
    "createUnixServer",
    "getUnixSocketPath",
]
