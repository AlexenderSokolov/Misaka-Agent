"""Translated from pi's ``packages/server/src/transports/unix/address.ts``."""

from __future__ import annotations

import os
import re

_SERVER_ID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}")


def getUnixSocketPath(serverId: str, serverDirectory: str) -> str:
    """Derive the local Unix socket path for one logical server identity."""
    if _SERVER_ID.fullmatch(serverId) is None:
        raise TypeError("Unix serverId must be a canonical lowercase UUIDv4")
    return os.path.normpath(os.path.join(serverDirectory, f"{serverId}.sock"))


__all__ = ["getUnixSocketPath"]
