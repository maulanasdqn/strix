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

from unittest.mock import MagicMock, patch

from strix.llm.oauth import constants as constants_mod
from strix.llm.oauth import first_run
from strix.llm.oauth.autodetect import try_autodetect_and_enable
from strix.llm.oauth.credentials import normalize_claude_code_model


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


@pytest.mark.parametrize(
    ("input_name", "expected"),
    [
        ("sonnet", "claude-sonnet-4-6"),
        ("opus", "claude-opus-4-6"),
        ("haiku", "claude-haiku-4-5"),
        ("default", "claude-sonnet-4-6"),
        ("Sonnet", "claude-sonnet-4-6"),
        ("sonnet-4.6", "claude-sonnet-4-6"),
        ("opus-4.6", "claude-opus-4-6"),
        ("haiku-4.5", "claude-haiku-4-5"),
        ("sonnet-4-6", "claude-sonnet-4-6"),
        ("claude-sonnet-4.6", "claude-sonnet-4-6"),
        ("claude-sonnet-4-6", "claude-sonnet-4-6"),
        # Unknown names pass through untouched — don't silently rewrite
        # non-Claude models or shapes we haven't seen.
        ("gpt-5.4", "gpt-5.4"),
        ("anthropic/claude-sonnet-4-6", "anthropic/claude-sonnet-4-6"),
    ],
)
def test_normalize_claude_code_model(input_name: str, expected: str) -> None:
    assert normalize_claude_code_model(input_name) == expected


@pytest.fixture
def clear_version_cache() -> None:
    constants_mod._detect_installed_claude_code_version.cache_clear()
    yield
    constants_mod._detect_installed_claude_code_version.cache_clear()


def test_claude_code_version_uses_detected_when_no_override(
    clear_version_cache: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("STRIX_CLAUDE_CODE_VERSION", raising=False)
    fake_result = MagicMock(returncode=0, stdout="2.1.200 (Claude Code)\n")
    with (
        patch.object(constants_mod.shutil, "which", return_value="/usr/bin/claude"),
        patch.object(constants_mod.subprocess, "run", return_value=fake_result),
    ):
        assert constants_mod.claude_code_version() == "2.1.200"


def test_claude_code_version_override_beats_detection(
    clear_version_cache: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("STRIX_CLAUDE_CODE_VERSION", "9.9.9")
    # Detection should not even run when override is present — but if it did,
    # the override must still win.
    fake_result = MagicMock(returncode=0, stdout="2.1.200 (Claude Code)\n")
    with (
        patch.object(constants_mod.shutil, "which", return_value="/usr/bin/claude"),
        patch.object(constants_mod.subprocess, "run", return_value=fake_result),
    ):
        assert constants_mod.claude_code_version() == "9.9.9"


def test_claude_code_version_falls_back_when_cli_missing(
    clear_version_cache: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("STRIX_CLAUDE_CODE_VERSION", raising=False)
    with patch.object(constants_mod.shutil, "which", return_value=None):
        assert constants_mod.claude_code_version() == constants_mod.DEFAULT_CLAUDE_CODE_VERSION


def test_claude_code_version_falls_back_on_parse_failure(
    clear_version_cache: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("STRIX_CLAUDE_CODE_VERSION", raising=False)
    fake_result = MagicMock(returncode=0, stdout="unexpected garbage output\n")
    with (
        patch.object(constants_mod.shutil, "which", return_value="/usr/bin/claude"),
        patch.object(constants_mod.subprocess, "run", return_value=fake_result),
    ):
        assert constants_mod.claude_code_version() == constants_mod.DEFAULT_CLAUDE_CODE_VERSION


# ------------------------------------------------------------------
# Message normalization: OpenAI image_url -> Anthropic image block
# ------------------------------------------------------------------


def test_normalize_converts_data_url_image_block() -> None:
    from strix.llm.oauth.direct import _normalize_message

    msg = {
        "role": "user",
        "content": [
            {"type": "text", "text": "here is a screenshot:"},
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,ABCDEF=="},
            },
        ],
    }
    result = _normalize_message(msg)
    assert result["role"] == "user"
    blocks = result["content"]
    assert blocks[0] == {"type": "text", "text": "here is a screenshot:"}
    assert blocks[1] == {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": "ABCDEF==",
        },
    }


def test_normalize_converts_http_url_image_block() -> None:
    from strix.llm.oauth.direct import _normalize_message

    msg = {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": "https://example/x.png"}}
        ],
    }
    result = _normalize_message(msg)
    assert result["content"][0] == {
        "type": "image",
        "source": {"type": "url", "url": "https://example/x.png"},
    }


def test_normalize_drops_malformed_image_block() -> None:
    from strix.llm.oauth.direct import _normalize_message

    msg = {
        "role": "user",
        "content": [
            {"type": "text", "text": "keep me"},
            {"type": "image_url", "image_url": {}},  # no url
            {"type": "image_url"},  # no image_url key at all
        ],
    }
    result = _normalize_message(msg)
    assert result["content"] == [{"type": "text", "text": "keep me"}]


def test_normalize_passes_plain_string_through() -> None:
    from strix.llm.oauth.direct import _normalize_message

    assert _normalize_message({"role": "user", "content": "just text"}) == {
        "role": "user",
        "content": "just text",
    }
