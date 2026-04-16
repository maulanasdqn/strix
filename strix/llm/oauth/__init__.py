"""Claude Code OAuth support for strix.

Opt-in via STRIX_USE_CLAUDE_CODE_OAUTH=1. See constants.py for the ToS risk
notice; clients must surface it before first use.
"""

from strix.llm.oauth.client import ClaudeCodeAuth, OAuthError, OAuthNotConfiguredError
from strix.llm.oauth.constants import (
    CLAUDE_CODE_SYSTEM_PROMPT_PREFIX,
    OAUTH_BETA_HEADER,
    is_oauth_enabled,
)
from strix.llm.oauth.credentials import load_claude_code_model


__all__ = [
    "CLAUDE_CODE_SYSTEM_PROMPT_PREFIX",
    "OAUTH_BETA_HEADER",
    "ClaudeCodeAuth",
    "OAuthError",
    "OAuthNotConfiguredError",
    "is_oauth_enabled",
    "load_claude_code_model",
]
