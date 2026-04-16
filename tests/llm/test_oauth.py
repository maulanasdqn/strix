"""Tests for Claude Code OAuth integration.

Covers credential loading priority, refresh flow, first-run acknowledgment,
the Claude Code system-prompt shim, header injection into LLM completion
args, and the 401 refresh-and-retry path. Heavy use of monkeypatch to
redirect ``~/.claude`` and ``~/.strix`` into ``tmp_path`` so tests never
touch real user credentials.
"""
# ruff: noqa: S105  # test fixtures use literal "tokens"; not real secrets.

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest


if TYPE_CHECKING:
    from pathlib import Path

from strix.llm.config import LLMConfig
from strix.llm.llm import LLM
from strix.llm.oauth import client as client_mod
from strix.llm.oauth import first_run
from strix.llm.oauth import refresh as refresh_mod
from strix.llm.oauth.client import ClaudeCodeAuth, OAuthNotConfiguredError
from strix.llm.oauth.constants import CLAUDE_CODE_SYSTEM_PROMPT_PREFIX
from strix.llm.oauth.credentials import (
    OAuthCredentials,
    load_credentials,
    save_credentials,
)


@pytest.fixture
def isolated_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    claude_dir = tmp_path / "claude"
    strix_dir = tmp_path / "strix"
    claude_dir.mkdir()
    strix_dir.mkdir()
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(claude_dir))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(
        first_run, "_ack_path", lambda: strix_dir / "oauth-ack.json"
    )
    for var in (
        "CLAUDE_CODE_OAUTH_TOKEN",
        "STRIX_USE_CLAUDE_CODE_OAUTH",
        "STRIX_OAUTH_ACK",
        "STRIX_OAUTH_DISABLE_PROMPT_SHIM",
        "STRIX_OAUTH_TOKEN_URL",
        "STRIX_OAUTH_CLIENT_ID",
    ):
        monkeypatch.delenv(var, raising=False)
    return tmp_path


# ------------------------------------------------------------------
# credentials
# ------------------------------------------------------------------


def test_load_from_env_var(
    isolated_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "env-token")
    creds = load_credentials()
    assert creds is not None
    assert creds.access_token == "env-token"
    assert creds.refresh_token is None
    assert creds.scopes == ["user:inference"]


def test_load_from_file(isolated_home: Path) -> None:
    creds_path = isolated_home / "claude" / ".credentials.json"
    creds_path.write_text(
        json.dumps(
            {
                "claudeAiOauth": {
                    "accessToken": "file-token",
                    "refreshToken": "file-refresh",
                    "expiresAt": 1_700_000_000_000,
                    "scopes": ["user:inference"],
                }
            }
        )
    )
    creds = load_credentials()
    assert creds is not None
    assert creds.access_token == "file-token"
    assert creds.refresh_token == "file-refresh"
    assert creds.expires_at == 1_700_000_000_000


def test_env_var_beats_file(
    isolated_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (isolated_home / "claude" / ".credentials.json").write_text(
        json.dumps({"claudeAiOauth": {"accessToken": "file-token"}})
    )
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "env-token")
    creds = load_credentials()
    assert creds is not None
    assert creds.access_token == "env-token"


def test_save_credentials_roundtrip(isolated_home: Path) -> None:
    orig = OAuthCredentials(
        access_token="a",
        refresh_token="r",
        expires_at=42,
        scopes=["user:inference"],
    )
    assert save_credentials(orig) is True

    loaded = load_credentials()
    assert loaded is not None
    assert loaded.access_token == "a"
    assert loaded.refresh_token == "r"
    assert loaded.expires_at == 42


def test_save_skips_env_tokens(isolated_home: Path) -> None:
    creds = OAuthCredentials(access_token="env-only", refresh_token=None)
    assert save_credentials(creds) is True
    assert not (isolated_home / "claude" / ".credentials.json").exists()


# ------------------------------------------------------------------
# refresh
# ------------------------------------------------------------------


def test_refresh_tokens_success(isolated_home: Path) -> None:
    fake_resp = MagicMock(status_code=200)
    fake_resp.json.return_value = {
        "access_token": "new-access",
        "refresh_token": "new-refresh",
        "expires_in": 3600,
        "scope": "user:inference user:profile",
    }
    with patch.object(refresh_mod.requests, "post", return_value=fake_resp) as post:
        creds = refresh_mod.refresh_tokens("old-refresh")

    assert creds.access_token == "new-access"
    assert creds.refresh_token == "new-refresh"
    assert creds.expires_at is not None
    assert creds.expires_at > int(time.time() * 1000)
    post.assert_called_once()
    body = post.call_args.kwargs["json"]
    assert body["grant_type"] == "refresh_token"
    assert body["refresh_token"] == "old-refresh"


def test_refresh_tokens_http_error(isolated_home: Path) -> None:
    fake_resp = MagicMock(status_code=401, text="denied")
    with (
        patch.object(refresh_mod.requests, "post", return_value=fake_resp),
        pytest.raises(refresh_mod.RefreshError),
    ):
        refresh_mod.refresh_tokens("bad")


def test_refresh_endpoint_overridable(
    isolated_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("STRIX_OAUTH_TOKEN_URL", "https://example.test/tok")
    monkeypatch.setenv("STRIX_OAUTH_CLIENT_ID", "cli-id-xyz")
    fake_resp = MagicMock(status_code=200)
    fake_resp.json.return_value = {"access_token": "x", "expires_in": 60}
    with patch.object(refresh_mod.requests, "post", return_value=fake_resp) as post:
        refresh_mod.refresh_tokens("r")
    assert post.call_args.args[0] == "https://example.test/tok"
    assert post.call_args.kwargs["json"]["client_id"] == "cli-id-xyz"


# ------------------------------------------------------------------
# client
# ------------------------------------------------------------------


def test_client_auto_refresh_near_expiry(isolated_home: Path) -> None:
    expired = OAuthCredentials(
        access_token="old",
        refresh_token="r",
        expires_at=int(time.time() * 1000) - 1000,
    )
    fresh = OAuthCredentials(
        access_token="new",
        refresh_token="r2",
        expires_at=int(time.time() * 1000) + 3_600_000,
    )
    auth = client_mod.ClaudeCodeAuth(expired)
    with patch.object(client_mod, "refresh_tokens", return_value=fresh) as rt:
        token = auth.get_token()
    assert token == "new"
    rt.assert_called_once_with("r")


def test_client_no_refresh_when_valid(isolated_home: Path) -> None:
    valid = OAuthCredentials(
        access_token="live",
        refresh_token="r",
        expires_at=int(time.time() * 1000) + 3_600_000,
    )
    auth = client_mod.ClaudeCodeAuth(valid)
    with patch.object(client_mod, "refresh_tokens") as rt:
        assert auth.get_token() == "live"
        rt.assert_not_called()


def test_force_refresh_requires_refresh_token(isolated_home: Path) -> None:
    auth = client_mod.ClaudeCodeAuth(
        OAuthCredentials(access_token="x", refresh_token=None)
    )
    with pytest.raises(client_mod.OAuthError):
        auth.force_refresh()


def test_from_environment_raises_when_missing(isolated_home: Path) -> None:
    with pytest.raises(OAuthNotConfiguredError):
        ClaudeCodeAuth.from_environment()


# ------------------------------------------------------------------
# first-run acknowledgment
# ------------------------------------------------------------------


def test_ack_via_env_var(
    isolated_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("STRIX_OAUTH_ACK", "1")
    first_run.require_acknowledgment()
    assert first_run._ack_path().exists()


def test_ack_non_interactive_refuses(
    isolated_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(first_run.sys.stdin, "isatty", lambda: False)
    with pytest.raises(first_run.OAuthAcknowledgmentRequiredError):
        first_run.require_acknowledgment()


def test_ack_persisted_between_runs(
    isolated_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ack = first_run._ack_path()
    ack.parent.mkdir(parents=True, exist_ok=True)
    ack.write_text(json.dumps({"version": first_run.ACK_VERSION, "accepted": True}))
    monkeypatch.setattr(first_run.sys.stdin, "isatty", lambda: False)
    first_run.require_acknowledgment()


# ------------------------------------------------------------------
# LLM integration: headers + prompt shim + 401 retry
# ------------------------------------------------------------------


def _make_llm_oauth(monkeypatch: pytest.MonkeyPatch) -> LLM:
    monkeypatch.setenv("STRIX_LLM", "anthropic/claude-sonnet-4-6")
    monkeypatch.setenv("STRIX_USE_CLAUDE_CODE_OAUTH", "1")
    monkeypatch.setenv("STRIX_OAUTH_ACK", "1")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "test-access-token")
    cfg = LLMConfig(model_name="anthropic/claude-sonnet-4-6")
    return LLM(cfg, agent_name=None)


def test_build_completion_args_injects_bearer(
    isolated_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    llm = _make_llm_oauth(monkeypatch)
    args = llm._build_completion_args([{"role": "system", "content": "hi"}])
    h = args["extra_headers"]
    # Token passed via api_key; litellm converts to Bearer at the wire.
    assert args["api_key"] == "test-access-token"
    assert h["x-app"] == "cli"
    assert h["User-Agent"].startswith("claude-cli/")
    assert "(external, cli)" in h["User-Agent"]
    assert "claude-code-20250219" in h["anthropic-beta"]
    assert "prompt-caching-2024-07-31" in h["anthropic-beta"]
    assert h["anthropic-dangerous-direct-browser-access"] == "true"
    assert h["x-anthropic-billing-header"].startswith("cc_version=")
    assert "cc_entrypoint=cli" in h["x-anthropic-billing-header"]
    assert h["X-Claude-Code-Session-Id"]


def test_session_id_stable_across_calls(
    isolated_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    llm = _make_llm_oauth(monkeypatch)
    a = llm._build_completion_args([{"role": "system", "content": "x"}])
    b = llm._build_completion_args([{"role": "system", "content": "x"}])
    assert (
        a["extra_headers"]["X-Claude-Code-Session-Id"]
        == b["extra_headers"]["X-Claude-Code-Session-Id"]
    )


def test_user_agent_version_overridable(
    isolated_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    llm = _make_llm_oauth(monkeypatch)
    monkeypatch.setenv("STRIX_CLAUDE_CODE_VERSION", "9.9.9")
    args = llm._build_completion_args([{"role": "system", "content": "x"}])
    assert args["extra_headers"]["User-Agent"] == "claude-cli/9.9.9 (external, cli)"
    assert "cc_version=9.9.9" in args["extra_headers"]["x-anthropic-billing-header"]


def test_oauth_prompt_shim_prepends(
    isolated_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    llm = _make_llm_oauth(monkeypatch)
    shimmed = llm._apply_oauth_prompt_shim("You are Strix, a pentest agent.")
    assert shimmed.startswith(CLAUDE_CODE_SYSTEM_PROMPT_PREFIX)
    assert "CWD:" in shimmed  # Claude Code env block
    assert "Date:" in shimmed
    # Strix identity line reframed, bare product name rewritten.
    assert "You are Strix" not in shimmed
    assert "You are currently performing the role of a pentest agent." in shimmed


def test_oauth_messages_move_brief_to_user(
    isolated_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    llm = _make_llm_oauth(monkeypatch)
    llm.system_prompt = "You are Strix, do pentest work and watch out for XSS."
    msgs = llm._prepare_messages([])
    # System message is small — just the Claude Code identity header, never
    # the heavy strix spec (Anthropic OAuth size cap).
    assert msgs[0]["role"] == "system"
    sys_text = msgs[0]["content"]
    if isinstance(sys_text, list):  # cache_control wraps as [{"type":"text",...}]
        sys_text = sys_text[0]["text"]
    assert sys_text.startswith(CLAUDE_CODE_SYSTEM_PROMPT_PREFIX)
    assert "pentest" not in sys_text
    # Strix spec lives inside a user-role task brief, with strix branding
    # rewritten so the persona stays Claude Code.
    assert msgs[1]["role"] == "user"
    assert "<task_brief>" in msgs[1]["content"]
    assert "do pentest work" in msgs[1]["content"]
    assert "Strix" not in msgs[1]["content"]
    assert "currently performing the role" in msgs[1]["content"]
    assert msgs[2]["role"] == "assistant"
    assert msgs[2]["content"] == "Acknowledged. Ready."


def test_oauth_shim_rewrites_strix_branding(
    isolated_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    llm = _make_llm_oauth(monkeypatch)
    raw = (
        "You are Strix, an advanced agent.\n"
        '- NEVER use "Strix" or identifiable names in requests\n'
        "- injected by the Strix platform"
    )
    shimmed = llm._apply_oauth_prompt_shim(raw)
    assert "Strix" not in shimmed
    assert "You are currently performing the role of an advanced agent." in shimmed
    assert 'NEVER use "Claude Code" or identifiable names' in shimmed
    assert "injected by the Claude Code platform" in shimmed


def test_oauth_prompt_shim_idempotent(
    isolated_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    llm = _make_llm_oauth(monkeypatch)
    already = CLAUDE_CODE_SYSTEM_PROMPT_PREFIX + "\n\nmore"
    assert llm._apply_oauth_prompt_shim(already) == already


def test_oauth_prompt_shim_kill_switch(
    isolated_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    llm = _make_llm_oauth(monkeypatch)
    monkeypatch.setenv("STRIX_OAUTH_DISABLE_PROMPT_SHIM", "1")
    assert llm._apply_oauth_prompt_shim("raw") == "raw"


def test_no_oauth_no_shim(
    isolated_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("STRIX_LLM", "anthropic/claude-sonnet-4-6")
    llm = LLM(LLMConfig(model_name="anthropic/claude-sonnet-4-6"), agent_name=None)
    assert llm.config.oauth_client is None
    assert llm._apply_oauth_prompt_shim("raw") == "raw"


def test_oauth_requires_anthropic_model(
    isolated_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("STRIX_LLM", "openai/gpt-5.4")
    monkeypatch.setenv("STRIX_USE_CLAUDE_CODE_OAUTH", "1")
    monkeypatch.setenv("STRIX_OAUTH_ACK", "1")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "tok")
    with pytest.raises(ValueError, match="not an Anthropic Claude model"):
        LLMConfig(model_name="openai/gpt-5.4")


def test_is_oauth_401_detects_status(isolated_home: Path) -> None:
    err = Exception("boom")
    err.status_code = 401  # type: ignore[attr-defined]
    assert LLM._is_oauth_401(err) is True

    other = Exception("boom")
    other.status_code = 500  # type: ignore[attr-defined]
    assert LLM._is_oauth_401(other) is False
