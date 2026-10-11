"""Places where the experimental server reads files and values the way Node does: ``readFile(path, "utf8")`` replaces invalid
bytes, ``trim()`` removes a byte order mark, ``JSON.parse`` rejects ``NaN`` and ``Infinity``, ``rm -rf`` unlinks a link, and
``path.resolve`` collapses two leading slashes."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from misaka.experimental.plugins.bundled import createPresentationFacetLoaders
from misaka.experimental.plugins.package import (
    normalizePluginPackagePaths,
    readPluginPackageProfile,
)
from misaka.experimental.radius_auth import RadiusRelayAuthResolver
from misaka.experimental.server import acquireServerProfile
from misaka.experimental.session_catalog import deleteSession, listSessions, readSession
from misaka.experimental.session_worker_manager import SessionWorkerManager

SERVER_ID = "00000000-0000-4000-8000-000000000001"
BOM = "﻿"


def writeMeta(root: Path, name: str, content: bytes) -> None:
    directory = root / name
    directory.mkdir(parents=True)
    (directory / "meta.json").write_bytes(content)


async def test_a_session_whose_metadata_json_parse_rejects_is_skipped(tmp_path: Path) -> None:
    writeMeta(tmp_path, "nan", b'{"createdAt": NaN, "cwd": "/x"}')
    writeMeta(tmp_path, "inf", b'{"createdAt": Infinity, "cwd": "/x"}')
    writeMeta(tmp_path, "deep", b"[" * 200_000 + b"]" * 200_000)
    writeMeta(tmp_path, "bom", BOM.encode() + b'{"createdAt": 1, "cwd": "/x"}')
    writeMeta(tmp_path, "good", b'{"createdAt": 1, "cwd": "/x"}')
    for name in ("nan", "inf", "deep", "bom"):
        assert await readSession(str(tmp_path), name) is None
    assert [session["id"] for session in await listSessions(str(tmp_path))] == ["good"]


async def test_a_session_whose_metadata_holds_invalid_utf8_is_listed_with_a_replacement_character(tmp_path: Path) -> None:
    writeMeta(tmp_path, "badutf8", b'{"createdAt":1,"cwd":"/x\xff"}')
    writeMeta(tmp_path, "good", b'{"createdAt": 1, "cwd": "/x"}')
    metadata = await readSession(str(tmp_path), "badutf8")
    assert metadata is not None
    assert metadata["cwd"] == "/x�"
    assert sorted(session["id"] for session in await listSessions(str(tmp_path))) == ["badutf8", "good"]


async def test_deleting_a_session_unlinks_a_link_and_leaves_its_target(tmp_path: Path) -> None:
    target = tmp_path / "elsewhere"
    target.mkdir()
    (target / "meta.json").write_text('{"createdAt": 1, "cwd": "/x"}', encoding="utf-8")
    link = tmp_path / "sessions" / "linked"
    link.parent.mkdir()
    try:
        os.symlink(target, link, target_is_directory=True)
    except OSError:
        pytest.skip("symbolic links are not available")
    metadata = await readSession(str(link.parent), "linked")
    assert metadata is not None
    await deleteSession(metadata)
    assert not os.path.lexists(link)
    assert (target / "meta.json").exists()


async def test_deleting_a_session_removes_a_file_and_ignores_a_missing_path(tmp_path: Path) -> None:
    file = tmp_path / "session"
    file.write_text("x", encoding="utf-8")
    await deleteSession({"path": str(file)})
    assert not file.exists()
    await deleteSession({"path": str(file)})


async def test_the_default_server_identity_may_start_with_a_byte_order_mark(tmp_path: Path) -> None:
    (tmp_path / "default-server-id").write_bytes((BOM + SERVER_ID + "\n").encode("utf-8"))
    profile = await acquireServerProfile(str(tmp_path))
    try:
        assert profile.serverId == SERVER_ID
    finally:
        await profile.release()


async def test_a_default_server_identity_with_invalid_bytes_is_the_documented_error(tmp_path: Path) -> None:
    (tmp_path / "default-server-id").write_bytes(b"\xff\xfe\x00")
    with pytest.raises(RuntimeError, match="Invalid default experimental server identity"):
        await acquireServerProfile(str(tmp_path))


async def test_a_token_file_with_a_byte_order_mark_gives_the_bare_token(tmp_path: Path) -> None:
    path = tmp_path / "token.txt"
    path.write_bytes((BOM + "tok\r\n").encode("utf-8"))
    resolver = RadiusRelayAuthResolver({"type": "file", "path": str(path)})
    assert await resolver._explicitToken(None) == "tok"


@pytest.mark.parametrize("version", [True, 2, "1", None])
async def test_a_plugin_profile_needs_the_number_one_as_its_version(tmp_path: Path, version: object) -> None:
    path = tmp_path / "profile.json"
    path.write_text(json.dumps({"version": version, "packagePaths": [str(tmp_path)]}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="Invalid experimental plugin package profile"):
        await readPluginPackageProfile(str(path), False)


async def test_a_plugin_profile_rejects_json_constants(tmp_path: Path) -> None:
    path = tmp_path / "profile.json"
    path.write_text('{"version": 1, "packagePaths": [], "x": NaN}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="Could not read experimental plugin package profile"):
        await readPluginPackageProfile(str(path), True)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX paths")
def test_plugin_package_paths_are_resolved_like_node() -> None:
    assert normalizePluginPackagePaths(["//home/u/plugin"]) == ("/home/u/plugin",)
    with pytest.raises(RuntimeError, match="unique"):
        normalizePluginPackagePaths(["//home/u/plugin", "/home/u/plugin"])


def test_a_null_presentation_bundle_list_is_invalid_where_an_absent_one_is_empty() -> None:
    assert createPresentationFacetLoaders({}) == []
    with pytest.raises(RuntimeError, match="Invalid presentation plugin bundle list"):
        createPresentationFacetLoaders({"presentationFacetBundles": None})


def test_an_exit_code_of_zero_is_reported_as_zero() -> None:
    manager = SessionWorkerManager.__new__(SessionWorkerManager)
    failures: list[BaseException] = []
    pending = SimpleNamespace(sessionKey="key", peerId="peer")
    manager._pending = {"key": pending}  # type: ignore[attr-defined]
    manager._failPending = lambda key, error: failures.append(error)  # type: ignore[method-assign,attr-defined]
    manager._childExited(pending, 0, None)  # type: ignore[attr-defined]
    assert str(failures[0]) == "Session worker exited before readiness (0)"
    failures.clear()
    manager._childExited(pending, None, None)  # type: ignore[attr-defined]
    assert str(failures[0]) == "Session worker exited before readiness (unknown)"
    failures.clear()
    manager._childExited(pending, 1, "SIGTERM")  # type: ignore[attr-defined]
    assert str(failures[0]) == "Session worker exited before readiness (SIGTERM)"
