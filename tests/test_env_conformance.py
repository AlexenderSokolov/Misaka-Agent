"""pi env's ``test/conformance.test.ts``: Durable's env conformance suite against ``RemoteExecutionEnv`` over a pipe to
the daemon, one connection for the whole suite."""

from __future__ import annotations

import os
import sys
import tempfile
from collections.abc import AsyncIterator
from typing import Any

import pytest
import pytest_asyncio

from misaka.durable.testing import createAssertions, createEnvConformance
from misaka.env import Connection, RemoteExecutionEnv
from tests.env_support import closeConnection, daemonCommand, removeTempDir

pytestmark = pytest.mark.asyncio(loop_scope="module")

windows = sys.platform == "win32"
gitBash = os.path.join(os.environ.get("ProgramFiles", "C:\\Program Files"), "Git", "bin", "bash.exe")

state: dict[str, Connection] = {}


@pytest_asyncio.fixture(loop_scope="module", scope="module")
async def connection() -> AsyncIterator[Connection]:
    connection = Connection({"command": daemonCommand()})
    yield connection
    await closeConnection(connection)


async def withEnv(use: Any) -> None:
    cwd = tempfile.mkdtemp(prefix="pi-env-conformance-")
    try:
        await use(
            RemoteExecutionEnv(
                {"connection": state["connection"], "id": "pi-env:test", "cwd": cwd, "watch": {"pollIntervalMs": 100}}
            )
        )
    finally:
        # On Windows a killed command's processes can hold the directory for a moment.
        removeTempDir(cwd)


# Git Bash's `ln -s` copies instead of linking unless native symlinks are enabled.
cases = createEnvConformance(
    {
        **({"shell": [gitBash, "-c"], "symlinks": False} if windows else {}),
        "assertions": createAssertions(),
        "withEnv": withEnv,
    }
)


@pytest.mark.parametrize("case", cases, ids=[case["name"] for case in cases])
async def test_remote_execution_env_over_a_pipe(case: Any, connection: Connection) -> None:
    state["connection"] = connection
    await case["run"]()
