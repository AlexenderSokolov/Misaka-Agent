"""pi coding-agent's ``test/experimental-server-profile.test.ts``."""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path

import pytest

from misaka.experimental.server import acquireServerProfile

FIRST_SERVER_ID = "00000000-0000-4000-8000-000000000001"
SECOND_SERVER_ID = "00000000-0000-4000-8000-000000000002"


@pytest.fixture
def makeDirectory(tmp_path):
    def make() -> str:
        return tempfile.mkdtemp(prefix="psp-", dir=str(tmp_path))

    return make


async def test_serializes_launchers_and_preserves_the_server_identity(makeDirectory):
    directory = makeDirectory()
    first = await acquireServerProfile(directory)
    state = {"secondAcquired": False}

    async def acquireSecond():
        profile = await acquireServerProfile(directory)
        state["secondAcquired"] = True
        return profile

    pendingSecond = asyncio.ensure_future(acquireSecond())
    await asyncio.sleep(0.1)
    assert state["secondAcquired"] is False
    await first.release()

    second = await pendingSecond
    assert state["secondAcquired"] is True
    assert second.serverId == first.serverId
    assert Path(directory, "default-server-id").read_bytes().decode("utf-8").strip() == first.serverId
    await second.release()


async def test_does_not_serialize_different_server_ids_in_one_directory(makeDirectory):
    directory = makeDirectory()
    first, second = await asyncio.gather(
        acquireServerProfile(directory, FIRST_SERVER_ID), acquireServerProfile(directory, SECOND_SERVER_ID)
    )
    assert first.serverId == FIRST_SERVER_ID
    assert second.serverId == SECOND_SERVER_ID
    assert sorted(os.listdir(directory)) == [f"launcher-{FIRST_SERVER_ID}.lock", f"launcher-{SECOND_SERVER_ID}.lock"]
    await asyncio.gather(first.release(), second.release())


async def test_does_not_share_the_default_identity_across_server_directories(makeDirectory):
    first, second = await asyncio.gather(
        acquireServerProfile(makeDirectory()), acquireServerProfile(makeDirectory())
    )
    assert first.serverId != second.serverId
    await asyncio.gather(first.release(), second.release())


async def test_rejects_a_corrupt_default_identity(makeDirectory):
    directory = makeDirectory()
    Path(directory, "default-server-id").write_bytes(b"invalid\n")

    with pytest.raises(RuntimeError, match="Invalid default experimental server identity"):
        await acquireServerProfile(directory)


async def test_rejects_an_invalid_explicit_server_id(makeDirectory):
    with pytest.raises(RuntimeError, match="Invalid experimental server ID"):
        await acquireServerProfile(makeDirectory(), "invalid")
