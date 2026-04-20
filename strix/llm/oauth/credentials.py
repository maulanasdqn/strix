"""Load/save Claude Code OAuth credentials.

Supports three sources (priority order):

1. ``CLAUDE_CODE_OAUTH_TOKEN`` env var — static access token, no refresh.
2. ``~/.claude/.credentials.json`` (or ``CLAUDE_CONFIG_DIR``) — full token
   bundle written by ``claude /login`` on Linux/Windows and as fallback on
   macOS when the keychain is unavailable.
3. macOS keychain entry ``Claude Code-credentials`` under the current user —
   queried via ``security find-generic-password -w``. Best-effort; failures
   are swallowed so non-macOS or headless hosts fall through to (2).
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import subprocess  # nosec B404 - only invokes /usr/bin/security
import sys
from dataclasses import dataclass, field
from pathlib import Path

from strix.llm.oauth.constants import CREDENTIALS_FILENAME, claude_config_dir


@dataclass
class OAuthCredentials:
    access_token: str
    refresh_token: str | None = None
    # Epoch milliseconds, matching the Claude Code on-disk format.
    expires_at: int | None = None
    scopes: list[str] = field(default_factory=list)
    subscription_type: str | None = None

    @classmethod
    def from_dict(cls, data: dict) -> OAuthCredentials:
        return cls(
            access_token=data["accessToken"],
            refresh_token=data.get("refreshToken"),
            expires_at=data.get("expiresAt"),
            scopes=list(data.get("scopes") or []),
            subscription_type=data.get("subscriptionType"),
        )

    def to_dict(self) -> dict:
        return {
            "accessToken": self.access_token,
            "refreshToken": self.refresh_token,
            "expiresAt": self.expires_at,
            "scopes": self.scopes,
            "subscriptionType": self.subscription_type,
        }


def _credentials_path() -> Path:
    return Path(claude_config_dir()) / CREDENTIALS_FILENAME


def _load_from_env() -> OAuthCredentials | None:
    token = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN")
    if not token:
        return None
    return OAuthCredentials(access_token=token, scopes=["user:inference"])


def _load_from_file() -> OAuthCredentials | None:
    path = _credentials_path()
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    oauth = data.get("claudeAiOauth") or data
    if not isinstance(oauth, dict) or not oauth.get("accessToken"):
        return None
    return OAuthCredentials.from_dict(oauth)


def _load_from_macos_keychain() -> OAuthCredentials | None:
    if sys.platform != "darwin":
        return None
    try:
        result = subprocess.run(  # nosec B603 - fixed argv, no shell
            [
                "/usr/bin/security",
                "find-generic-password",
                "-s",
                "Claude Code-credentials",
                "-w",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None
    try:
        data = json.loads(result.stdout.strip())
    except ValueError:
        return None
    oauth = data.get("claudeAiOauth") or data
    if not isinstance(oauth, dict) or not oauth.get("accessToken"):
        return None
    return OAuthCredentials.from_dict(oauth)


def load_credentials() -> OAuthCredentials | None:
    """Return the best available OAuth credentials, or None if none found."""
    for loader in (_load_from_env, _load_from_macos_keychain, _load_from_file):
        creds = loader()
        if creds is not None:
            return creds
    return None


_MODEL_ALIASES = {
    "default": "claude-sonnet-4-6",
    "sonnet": "claude-sonnet-4-6",
    "opus": "claude-opus-4-6",
    "haiku": "claude-haiku-4-5",
}

# Matches versioned Claude Code shorthand with either dash or dot separators:
# "sonnet-4.6", "sonnet-4-6", "claude-sonnet-4.6", "claude-sonnet-4-6".
_VERSIONED_CLAUDE_RE = re.compile(
    r"^(?:claude-)?(sonnet|opus|haiku)-(\d+)[.\-](\d+)$"
)


def normalize_claude_code_model(name: str) -> str:
    """Translate Claude Code shorthand to the canonical Anthropic model id.

    Handles the bare-family aliases (``sonnet``/``opus``/``haiku``/``default``)
    and versioned forms like ``sonnet-4.6``. Unknown names pass through
    untouched so non-Claude models (or newer shapes we haven't met) don't
    silently break.
    """
    lower = name.strip().lower()
    if lower in _MODEL_ALIASES:
        return _MODEL_ALIASES[lower]
    match = _VERSIONED_CLAUDE_RE.match(lower)
    if match:
        family, major, minor = match.groups()
        return f"claude-{family}-{major}-{minor}"
    return name


def load_claude_code_model() -> str | None:
    """Read the user's selected model from Claude Code's ``settings.json``.

    Returns the resolved bare model name (Claude Code's own spelling) or
    ``None`` if no setting exists. Aliases like ``sonnet`` / ``opus`` are
    expanded to the underlying versioned id so the wire request matches what
    Claude Code itself would send.
    """
    settings_path = Path(claude_config_dir()) / "settings.json"
    if not settings_path.is_file():
        return None
    try:
        data = json.loads(settings_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    model = data.get("model")
    if not isinstance(model, str) or not model:
        return None
    return normalize_claude_code_model(model)


def save_credentials(creds: OAuthCredentials) -> bool:
    """Persist credentials to ``~/.claude/.credentials.json`` with 0600 perms.

    Only writes the file backend; keychain writeback is intentionally skipped
    so strix never mutates the user's macOS keychain. Returns False on any
    IO error — callers treat the in-memory token as authoritative either way.
    """
    if creds.refresh_token is None:
        # Nothing useful to persist (env-var tokens are ephemeral).
        return True
    path = _credentials_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"claudeAiOauth": creds.to_dict()}
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        with contextlib.suppress(OSError):
            tmp.chmod(0o600)  # best effort on Windows
        tmp.replace(path)
    except OSError:
        return False
    return True
