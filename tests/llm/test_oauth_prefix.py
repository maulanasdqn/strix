"""Auto-prefix of bare Claude model names when OAuth is enabled."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from strix.llm.config import LLMConfig
from strix.llm.oauth import first_run


if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def oauth_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    claude_dir = tmp_path / "claude"
    claude_dir.mkdir()
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(claude_dir))
    monkeypatch.setattr(
        first_run, "_ack_path", lambda: tmp_path / "ack.json"
    )
    monkeypatch.setenv("STRIX_USE_CLAUDE_CODE_OAUTH", "1")
    monkeypatch.setenv("STRIX_OAUTH_ACK", "1")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "t")
    return tmp_path


def test_bare_claude_name_gets_anthropic_prefix(
    oauth_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("STRIX_LLM", "claude-sonnet-4-6")
    cfg = LLMConfig(model_name="claude-sonnet-4-6")
    assert cfg.model_name == "anthropic/claude-sonnet-4-6"
    assert cfg.oauth_client is not None


def test_explicit_prefix_preserved(
    oauth_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("STRIX_LLM", "anthropic/claude-opus-4-6")
    cfg = LLMConfig(model_name="anthropic/claude-opus-4-6")
    assert cfg.model_name == "anthropic/claude-opus-4-6"


def test_non_claude_bare_name_not_rewritten(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # No OAuth: bare name must stay as-is.
    monkeypatch.setenv("STRIX_LLM", "gpt-5.4")
    cfg = LLMConfig(model_name="openai/gpt-5.4")
    assert cfg.model_name == "openai/gpt-5.4"


def test_model_read_from_claude_settings(
    oauth_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("STRIX_LLM", raising=False)
    settings = oauth_env / "claude" / "settings.json"
    settings.write_text(json.dumps({"model": "claude-opus-4-6"}))
    cfg = LLMConfig()
    assert cfg.model_name == "anthropic/claude-opus-4-6"


def test_settings_alias_resolves(
    oauth_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("STRIX_LLM", raising=False)
    settings = oauth_env / "claude" / "settings.json"
    settings.write_text(json.dumps({"model": "opus"}))
    cfg = LLMConfig()
    assert cfg.model_name == "anthropic/claude-opus-4-6"


def test_explicit_strix_llm_wins_over_settings(
    oauth_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = oauth_env / "claude" / "settings.json"
    settings.write_text(json.dumps({"model": "claude-opus-4-6"}))
    monkeypatch.setenv("STRIX_LLM", "claude-haiku-4-5")
    cfg = LLMConfig()
    assert cfg.model_name == "anthropic/claude-haiku-4-5"
