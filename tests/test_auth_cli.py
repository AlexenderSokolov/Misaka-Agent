"""``misaka auth`` over the model runtime, after pi's ``auth-check.test.ts`` and
``credential-print.test.ts`` (the command surface is misaka's own, ``misaka/cli/auth.py``)."""

from __future__ import annotations

import json
import time

import pytest

from misaka.ai.auth.types import OAuthCredential
from misaka.cli import auth
from misaka.config import home


def _store(data) -> None:
    path = home.path("auth")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data if isinstance(data, str) else json.dumps(data), encoding="utf-8")


def _command(*args: str):
    return auth.parse_auth_command(list(args))


def _hour_ms() -> int:
    return int(time.time() * 1000) + 60 * 60 * 1000


async def test_reports_a_configured_provider_as_ready():
    _store({"openai": {"type": "api_key", "key": "test-key"}})

    result = await auth._check_auth(_command("check", "--provider", "openai"))

    assert (result.status, result.provider, result.auth_type) == ("ready", "openai", "api_key")


async def test_resolves_the_provider_from_model():
    _store({"openai": {"type": "api_key", "key": "test-key"}})

    result = await auth._check_auth(_command("check", "--model", "openai/gpt-4.1"))
    assert (result.status, result.provider) == ("ready", "openai")
    result = await auth._check_auth(_command("check", "--provider", "openai", "--model", "gpt-4.1"))
    assert (result.status, result.provider) == ("ready", "openai")


def _patch_refresh(monkeypatch, provider_id: str, refreshed: list, access: str | None = None):
    create = auth.create_auth_check_model_runtime

    async def patched(credentials):
        runtime = await create(credentials)
        oauth = runtime.getProvider(provider_id).auth.oauth

        async def refresh(credential, _signal):
            refreshed.append(credential.access)
            return OAuthCredential(type="oauth", access=access or "fresh-token", refresh="refresh-token",
                                   expires=_hour_ms())

        oauth.refresh = refresh
        return runtime

    monkeypatch.setattr(auth, "create_auth_check_model_runtime", patched)


async def test_reads_credentials_without_refreshing_oauth_when_requested(monkeypatch):
    _store({"openai-codex": {"type": "oauth", "access": "old-token", "refresh": "refresh-token", "expires": 0}})
    refreshed: list = []
    _patch_refresh(monkeypatch, "openai-codex", refreshed)

    result = await auth._check_auth(_command("check", "--provider", "openai-codex", "--no-refresh", "--credentials"))

    assert (result.status, result.credential) == ("ready", "old-token")
    assert refreshed == []


async def test_refreshes_oauth_by_default(monkeypatch):
    _store({"openai-codex": {"type": "oauth", "access": "old-token", "refresh": "refresh-token", "expires": 0}})
    refreshed: list = []
    _patch_refresh(monkeypatch, "openai-codex", refreshed)

    result = await auth._check_auth(_command("check", "--provider", "openai-codex"))

    assert result.status == "ready"
    assert refreshed == ["old-token"]


async def test_reports_an_unknown_provider_as_not_ready():
    result = await auth._check_auth(_command("check", "--provider", "not-installed"))

    assert (result.status, result.provider, result.reason) == ("not_ready", "not-installed", "provider_not_found")


async def test_does_not_treat_an_unresolved_stored_environment_reference_as_configured(monkeypatch):
    monkeypatch.delenv("MISSING_AUTH_CHECK_KEY", raising=False)
    _store({"openai": {"type": "api_key", "key": "$MISSING_AUTH_CHECK_KEY"}})

    result = await auth._check_auth(_command("check", "--provider", "openai", "--no-refresh"))

    assert (result.status, result.reason) == ("not_ready", "credentials_not_configured")


async def test_reports_malformed_auth_state_as_invalid():
    _store("{invalid-json")

    result = await auth._check_auth(_command("check", "--provider", "openai", "--no-refresh"))

    assert (result.status, result.provider, result.reason) == ("invalid", "openai", "invalid_state")


async def test_does_not_create_an_auth_file_or_its_parent_directory():
    result = await auth._check_auth(_command("check", "--provider", "openai", "--no-refresh"))

    assert (result.status, result.reason) == ("not_ready", "credentials_not_configured")
    assert not home.path("auth").exists()
    assert not home.path("auth").parent.exists()


async def test_creates_an_auth_check_runtime_without_catalog_storage():
    from misaka.core.auth_storage import AuthStorage

    runtime = await auth.create_auth_check_model_runtime(AuthStorage.inMemory())

    assert runtime.getProvider("openai") is not None
    assert not home.path("models_store").exists()


async def test_prints_a_resolved_api_key():
    _store({"openai": {"type": "api_key", "key": "test-api-key"}})

    assert await auth._print_credential(_command("print-api-key", "--provider", "openai")) == "test-api-key"


async def test_prints_bearer_tokens_resolved_from_an_authorization_header():
    _store({"kimi-coding": {
        "type": "oauth", "access": "header-test-token", "refresh": "test-refresh-token", "expires": _hour_ms(),
    }})

    credential = await auth._print_credential(_command("print-bearer-token", "--provider", "kimi-coding"))

    assert credential == "header-test-token"


async def test_refreshes_an_expired_oauth_token_before_printing_it(monkeypatch):
    _store({"openai-codex": {"type": "oauth", "access": "old-test-token", "refresh": "test-refresh-token",
                             "expires": 0}})
    refreshed: list = []
    _patch_refresh(monkeypatch, "openai-codex", refreshed, access="fresh-test-token")

    credential = await auth._print_credential(_command("print-bearer-token", "--provider", "openai-codex"))

    assert credential == "fresh-test-token"
    assert refreshed == ["old-test-token"]
    assert json.loads(home.path("auth").read_text(encoding="utf-8"))["openai-codex"]["access"] == "fresh-test-token"


async def test_an_oauth_provider_does_not_print_as_an_api_key():
    _store({"openai-codex": {"type": "oauth", "access": "test-token-not-to-be-printed",
                             "refresh": "test-refresh-token", "expires": _hour_ms()}})

    with pytest.raises(auth.AuthCommandError, match="configured with OAuth, not an API key"):
        await auth._print_credential(_command("print-api-key", "--provider", "openai-codex"))


def test_the_legacy_overview_reads_the_runtime_snapshot(capsys):
    _store({"openai": {"type": "api_key", "key": "test-key"}})

    code = auth._run_legacy_check("openai", show=True)

    line = capsys.readouterr().out.strip()
    assert code == 0 and line.startswith("✓ openai  stored") and line.endswith("test-key")


def test_the_legacy_overview_shows_a_token_sent_as_an_authorization_header(capsys):
    _store({"kimi-coding": {
        "type": "oauth", "access": "header-test-token", "refresh": "test-refresh-token", "expires": _hour_ms(),
    }})

    auth._run_legacy_check("kimi-coding", show=True)

    assert capsys.readouterr().out.strip().endswith("header-test-token")
