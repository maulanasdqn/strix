"""Tests for Claude Code OAuth auto-detection on startup.

Validates the precedence matrix: explicit env vars (opt-out, API key,
non-Anthropic model) win over detected credentials, and expired or missing
credentials skip auto-enable instead of failing loudly.
"""
# ruff: noqa: S105  # test fixtures use literal "tokens"; not real secrets.

from __future__ import annotations

import json
import os
import time
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from pathlib import Path

import pytest

from strix.llm.oauth import first_run
from strix.llm.oauth.autodetect import try_autodetect_and_enable


@pytest.fixture
def isolated_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    claude_dir = tmp_path / "claude"
    strix_dir = tmp_path / "strix"
    claude_dir.mkdir()
    strix_dir.mkdir()
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(claude_dir))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(first_run, "_ack_path", lambda: strix_dir / "oauth-ack.json")
    for var in (
        "CLAUDE_CODE_OAUTH_TOKEN",
        "STRIX_USE_CLAUDE_CODE_OAUTH",
        "STRIX_OAUTH_ACK",
        "LLM_API_KEY",
        "STRIX_LLM",
    ):
        monkeypatch.delenv(var, raising=False)
    return tmp_path


def _write_creds(isolated_home: Path, *, expires_at: int | None = None) -> None:
    payload = {"accessToken": "detected-token", "refreshToken": "detected-refresh"}
    if expires_at is not None:
        payload["expiresAt"] = expires_at
    (isolated_home / "claude" / ".credentials.json").write_text(
        json.dumps({"claudeAiOauth": payload})
    )


def test_enables_when_creds_exist_and_no_other_config(isolated_home: Path) -> None:
    _write_creds(isolated_home, expires_at=int(time.time() * 1000) + 3600_000)

    assert try_autodetect_and_enable() is True
    assert os.environ["STRIX_USE_CLAUDE_CODE_OAUTH"] == "1"
    assert os.environ["STRIX_OAUTH_ACK"] == "1"


def test_skipped_when_llm_api_key_set(
    isolated_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_creds(isolated_home)
    monkeypatch.setenv("LLM_API_KEY", "sk-explicit")

    assert try_autodetect_and_enable() is False
    assert "STRIX_USE_CLAUDE_CODE_OAUTH" not in os.environ
    assert "STRIX_OAUTH_ACK" not in os.environ


def test_skipped_when_strix_llm_is_non_anthropic(
    isolated_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_creds(isolated_home)
    monkeypatch.setenv("STRIX_LLM", "openai/gpt-5.4")

    assert try_autodetect_and_enable() is False
    assert "STRIX_USE_CLAUDE_CODE_OAUTH" not in os.environ


def test_runs_when_strix_llm_is_anthropic(
    isolated_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_creds(isolated_home)
    monkeypatch.setenv("STRIX_LLM", "anthropic/claude-sonnet-4-6")

    assert try_autodetect_and_enable() is True


def test_runs_when_strix_llm_is_bare_claude_name(
    isolated_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_creds(isolated_home)
    monkeypatch.setenv("STRIX_LLM", "claude-opus-4-6")

    assert try_autodetect_and_enable() is True


def test_skipped_when_user_opted_out(
    isolated_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_creds(isolated_home)
    monkeypatch.setenv("STRIX_USE_CLAUDE_CODE_OAUTH", "0")

    assert try_autodetect_and_enable() is False
    # Opt-out env var left intact by autodetect.
    assert os.environ["STRIX_USE_CLAUDE_CODE_OAUTH"] == "0"


def test_respects_existing_opt_in(
    isolated_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_creds(isolated_home)
    monkeypatch.setenv("STRIX_USE_CLAUDE_CODE_OAUTH", "1")

    # Already enabled explicitly — autodetect is a no-op.
    assert try_autodetect_and_enable() is False
    assert "STRIX_OAUTH_ACK" not in os.environ


def test_skipped_when_token_expired(isolated_home: Path) -> None:
    _write_creds(isolated_home, expires_at=1)  # epoch ms in 1970

    assert try_autodetect_and_enable() is False
    assert "STRIX_USE_CLAUDE_CODE_OAUTH" not in os.environ


def test_enables_when_expires_at_missing(isolated_home: Path) -> None:
    # expires_at=None means "no expiry info" — treat as valid, let refresh/401 handle it.
    _write_creds(isolated_home, expires_at=None)

    assert try_autodetect_and_enable() is True


def test_skipped_when_no_creds(isolated_home: Path) -> None:
    assert try_autodetect_and_enable() is False
    assert "STRIX_USE_CLAUDE_CODE_OAUTH" not in os.environ
