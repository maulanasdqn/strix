"""Constants for Claude Code OAuth integration.

Prod defaults can be overridden via STRIX_OAUTH_TOKEN_URL and
STRIX_OAUTH_CLIENT_ID because Anthropic may rotate either without notice.
"""

from __future__ import annotations

import functools
import os
import shutil
import subprocess  # nosec B404 - only invokes the user's installed `claude` CLI
from datetime import UTC, datetime
from pathlib import Path


DEFAULT_TOKEN_URL = "https://platform.claude.com/v1/oauth/token"  # noqa: S105 - URL, not a secret
DEFAULT_CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"

OAUTH_BETA_HEADER = "oauth-2025-04-20"

# Full anthropic-beta blob Claude Code sends on every OAuth request. The
# ``claude-code-20250219`` beta in particular is the server-side "this is
# Claude Code" signal; without it the API often returns a generic
# rate_limit_error even when quota is nowhere near the cap.
CLAUDE_CODE_ANTHROPIC_BETA = (
    "oauth-2025-04-20,"
    "interleaved-thinking-2025-05-14,"
    "claude-code-20250219,"
    "prompt-caching-2024-07-31"
)

CLAUDE_AI_INFERENCE_SCOPE = "user:inference"
CLAUDE_AI_OAUTH_SCOPES = (
    "user:profile",
    CLAUDE_AI_INFERENCE_SCOPE,
    "user:sessions:claude_code",
    "user:mcp_servers",
    "user:file_upload",
)

# Required system prompt prefix for OAuth inference tokens. API returns 403
# when the first system message does not start with this string verbatim.
# Kill switch: set STRIX_OAUTH_DISABLE_PROMPT_SHIM=1 to skip the prepend
# (only useful once Anthropic relaxes the constraint).
CLAUDE_CODE_SYSTEM_PROMPT_PREFIX = (
    "You are Claude Code, Anthropic's official CLI for Claude."
)

SHIM_SEPARATOR = "\n\n---\n\n"


DEFAULT_CLAUDE_CODE_VERSION = "2.1.114"


@functools.lru_cache(maxsize=1)
def _detect_installed_claude_code_version() -> str | None:
    """Best-effort read of the local ``claude --version`` output.

    Anthropic rejects OAuth requests whose User-Agent version drifts too far
    from the current Claude Code release (generic ``rate_limit_error`` with
    no Retry-After). Matching the installed CLI makes strix's requests
    indistinguishable from a real Claude Code session on the same machine.
    Returns None if the CLI is missing, times out, or prints something we
    can't parse.
    """
    exe = shutil.which("claude")
    if not exe:
        return None
    try:
        result = subprocess.run(  # nosec B603 - fixed argv, no shell
            [exe, "--version"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    first_token = result.stdout.strip().split()[0:1]
    if not first_token:
        return None
    candidate = first_token[0]
    # Require something that looks like a semver head (digit-dot-digit…).
    if not candidate or not candidate[0].isdigit() or "." not in candidate:
        return None
    return candidate


def claude_code_version() -> str:
    """Claude Code CLI version reported in headers.

    Precedence: ``STRIX_CLAUDE_CODE_VERSION`` env override → live detection
    from ``claude --version`` on PATH → hardcoded default. The hardcoded
    value is a fallback floor; bump it when Anthropic starts rejecting it.
    """
    override = os.environ.get("STRIX_CLAUDE_CODE_VERSION")
    if override:
        return override
    detected = _detect_installed_claude_code_version()
    if detected:
        return detected
    return DEFAULT_CLAUDE_CODE_VERSION


def claude_code_user_agent() -> str:
    # Matches the claude-rust port's UA: ``claude-cli/<version> (external, cli)``.
    # "external" marks non-Anthropic-internal callers; "cli" is the entrypoint.
    return f"claude-cli/{claude_code_version()} (external, cli)"


def claude_code_billing_header() -> str:
    """Attribution payload sent on every Claude Code request.

    Format mirrors ``getAttributionHeader`` in claude-code-rust:
    ``cc_version=<v>; cc_entrypoint=<entry>;``. ``cc_entrypoint`` defaults
    to ``cli`` (interactive shell session); override via
    ``STRIX_CLAUDE_CODE_ENTRYPOINT`` for SDK/CI mimicry.
    """
    entrypoint = os.environ.get("STRIX_CLAUDE_CODE_ENTRYPOINT") or "cli"
    return f"cc_version={claude_code_version()}; cc_entrypoint={entrypoint};"


def claude_code_billing_line() -> str:
    """Billing attribution as Claude Code actually wires it.

    Real Claude Code sends this string as the first text block of the
    request-body ``system`` array (not as an HTTP header), matching
    claude-rust's ``BILLING_HEADER_LINE``. Anthropic reads it server-side
    for session attribution; sending it only as an HTTP header leaves the
    request indistinguishable from non-Claude-Code OAuth traffic and
    triggers a generic ``rate_limit_error``.
    """
    return f"x-anthropic-billing-header: {claude_code_billing_header()}"


def claude_code_prompt_header() -> str:
    """Build the leading system-prompt block exactly like Claude Code.

    Mirrors ``getSystemPrompt`` in claude-code-rust's ``constants/prompts.ts``
    (CLAUDE_CODE_SIMPLE branch): identity line + working directory + date.
    Matching this byte-for-byte makes strix's OAuth requests less
    distinguishable from real Claude Code traffic.
    """
    cwd = Path.cwd()
    date = datetime.now(tz=UTC).date().isoformat()
    return f"{CLAUDE_CODE_SYSTEM_PROMPT_PREFIX}\n\nCWD: {cwd}\nDate: {date}"

# Refresh ~60s before expiry so concurrent requests don't race the clock.
REFRESH_LEEWAY_SECONDS = 60


def oauth_min_interval_seconds() -> float:
    """Minimum gap between two LLM requests when OAuth is active.

    Anthropic enforces a strict per-token rate limit on Claude Code OAuth
    (empirically ~1 req every 30s). Bursting past that returns a generic
    ``rate_limit_error`` with no Retry-After. Default 30s keeps us under
    the cap; override with ``STRIX_OAUTH_MIN_INTERVAL`` if you have a
    higher-tier subscription or are willing to accept retries.
    """
    raw = os.environ.get("STRIX_OAUTH_MIN_INTERVAL")
    if not raw:
        return 30.0
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 30.0

# Path inside the user's claude config dir.
CREDENTIALS_FILENAME = ".credentials.json"


def oauth_token_url() -> str:
    return os.environ.get("STRIX_OAUTH_TOKEN_URL") or DEFAULT_TOKEN_URL


def oauth_client_id() -> str:
    return os.environ.get("STRIX_OAUTH_CLIENT_ID") or DEFAULT_CLIENT_ID


def is_oauth_enabled() -> bool:
    return os.environ.get("STRIX_USE_CLAUDE_CODE_OAUTH", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def prompt_shim_disabled() -> bool:
    return os.environ.get("STRIX_OAUTH_DISABLE_PROMPT_SHIM", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def claude_config_dir() -> str:
    override = os.environ.get("CLAUDE_CONFIG_DIR")
    if override:
        return override
    return str(Path("~/.claude").expanduser())
