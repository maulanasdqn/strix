"""First-run acknowledgment for Claude Code OAuth opt-in.

We refuse to load OAuth credentials until the user has explicitly accepted
the ToS risk. Accepted state is persisted to ``~/.strix/oauth-ack.json`` so
the prompt only fires once per machine.

Non-interactive callers (CI, headless) can pre-accept by exporting
``STRIX_OAUTH_ACK=1`` — logged so the acceptance trail is reconstructable.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path


logger = logging.getLogger(__name__)


ACK_VERSION = 1
ACK_FILENAME = "oauth-ack.json"

WARNING_TEXT = (
    "\n"
    "================================================================\n"
    "  Claude Code OAuth — ToS RISK WARNING\n"
    "================================================================\n"
    "  Anthropic's Terms of Service restrict Claude Code OAuth tokens\n"
    "  to the official Claude Code client. Using them with strix is a\n"
    "  third-party use that may violate the ToS and result in your\n"
    "  Anthropic account being suspended or banned.\n"
    "\n"
    "  You are solely responsible for this risk. strix prepends the\n"
    "  required Claude Code system prompt to every request, which may\n"
    "  degrade pentest-agent behaviour.\n"
    "\n"
    "  To bypass this prompt in CI: export STRIX_OAUTH_ACK=1\n"
    "================================================================\n"
)


def _ack_path() -> Path:
    return Path.home() / ".strix" / ACK_FILENAME


def _already_acknowledged() -> bool:
    path = _ack_path()
    if not path.is_file():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return bool(data.get("accepted")) and data.get("version") == ACK_VERSION


def _persist_acknowledgment(source: str) -> None:
    path = _ack_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "version": ACK_VERSION,
                    "accepted": True,
                    "accepted_at": int(time.time()),
                    "source": source,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    except OSError as exc:
        logger.warning("could not persist oauth acknowledgment: %s", exc)


class OAuthAcknowledgmentRequiredError(RuntimeError):
    """Raised when OAuth is enabled but the user has not accepted the ToS risk."""


def require_acknowledgment() -> None:
    """Block until the user has acknowledged the ToS risk, or raise."""
    if _already_acknowledged():
        return

    if os.environ.get("STRIX_OAUTH_ACK", "").lower() in {"1", "true", "yes"}:
        logger.warning("oauth ToS acknowledged via STRIX_OAUTH_ACK env var")
        _persist_acknowledgment("env")
        return

    if not sys.stdin.isatty():
        raise OAuthAcknowledgmentRequiredError(
            "STRIX_USE_CLAUDE_CODE_OAUTH is enabled in a non-interactive "
            "session without prior acknowledgment. Set STRIX_OAUTH_ACK=1 "
            "after reading the ToS risk notice in docs, or run strix "
            "interactively once to accept."
        )

    sys.stderr.write(WARNING_TEXT)
    sys.stderr.flush()
    try:
        reply = input("Type 'I ACCEPT' to continue: ").strip()
    except (EOFError, KeyboardInterrupt) as exc:
        raise OAuthAcknowledgmentRequiredError("acknowledgment cancelled") from exc

    if reply != "I ACCEPT":
        raise OAuthAcknowledgmentRequiredError(
            "Claude Code OAuth requires explicit acceptance to continue."
        )
    _persist_acknowledgment("interactive")
