"""Auto-detect existing Claude Code credentials on strix startup.

When a machine is already signed into Claude Code (``claude /login``), strix
should authenticate silently using those credentials instead of forcing the
user to set ``STRIX_USE_CLAUDE_CODE_OAUTH=1`` and step through the ToS prompt.

``try_autodetect_and_enable`` is the single entry point. It only flips the
opt-in env vars; the rest of the OAuth machinery (client, refresh, headers,
model resolution) reads those env vars on first use and needs no other hook.
"""

from __future__ import annotations

import os
import time

from strix.llm.oauth.credentials import load_credentials


def _explicit_opt_in_or_out() -> bool:
    return "STRIX_USE_CLAUDE_CODE_OAUTH" in os.environ


def _has_explicit_api_key() -> bool:
    return bool(os.environ.get("LLM_API_KEY"))


def _strix_llm_is_non_anthropic() -> bool:
    model = os.environ.get("STRIX_LLM", "").strip().lower()
    if not model:
        return False
    return not (model.startswith("anthropic/") or "claude" in model)


def _token_expired(expires_at: int | None) -> bool:
    if not expires_at:
        return False
    return expires_at <= int(time.time() * 1000)


def try_autodetect_and_enable() -> bool:
    """Enable Claude Code OAuth if credentials are available and no other
    provider is explicitly configured. Returns True when auto-enabled."""
    if _explicit_opt_in_or_out():
        return False
    if _has_explicit_api_key():
        return False
    if _strix_llm_is_non_anthropic():
        return False

    creds = load_credentials()
    if creds is None:
        return False
    if _token_expired(creds.expires_at):
        return False

    os.environ["STRIX_USE_CLAUDE_CODE_OAUTH"] = "1"
    os.environ["STRIX_OAUTH_ACK"] = "1"
    return True
