"""High-level OAuth client used by LLMConfig.

``ClaudeCodeAuth`` is the only thing the rest of strix touches. It hides the
credential source (env / file / keychain), the expiry check, and the lockfile
that serialises refreshes across processes sharing the same credential file.
"""

from __future__ import annotations

import contextlib
import os
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Iterator

from strix.llm.oauth.constants import (
    REFRESH_LEEWAY_SECONDS,
    claude_config_dir,
)
from strix.llm.oauth.credentials import (
    OAuthCredentials,
    load_credentials,
    save_credentials,
)
from strix.llm.oauth.refresh import RefreshError, refresh_tokens


class OAuthError(Exception):
    """Base class for OAuth-related failures surfaced to callers."""


class OAuthNotConfiguredError(OAuthError):
    """No usable OAuth credentials found on disk / env / keychain."""


def _is_expired(expires_at: int | None) -> bool:
    if not expires_at:
        return False  # static env-var tokens — assume valid, rely on 401
    return expires_at <= int((time.time() + REFRESH_LEEWAY_SECONDS) * 1000)


class ClaudeCodeAuth:
    """Thread-safe holder of Claude Code OAuth credentials.

    ``get_token()`` returns an access token, refreshing first if the stored
    expiry is within ``REFRESH_LEEWAY_SECONDS`` of now. ``force_refresh()`` is
    called from the LLM 401 path to handle server-side rejection even when
    the local clock thinks the token is still valid.
    """

    def __init__(self, creds: OAuthCredentials):
        self._creds = creds
        self._lock = threading.Lock()

    @classmethod
    def from_environment(cls) -> ClaudeCodeAuth:
        creds = load_credentials()
        if creds is None:
            raise OAuthNotConfiguredError(
                "Claude Code OAuth enabled but no credentials found. "
                "Set CLAUDE_CODE_OAUTH_TOKEN, run `claude /login`, or point "
                "CLAUDE_CONFIG_DIR at a directory containing .credentials.json."
            )
        return cls(creds)

    @property
    def credentials(self) -> OAuthCredentials:
        return self._creds

    def get_token(self) -> str:
        with self._lock:
            if _is_expired(self._creds.expires_at) and self._creds.refresh_token:
                self._refresh_locked()
            return self._creds.access_token

    def force_refresh(self) -> str:
        """Refresh even if the local clock thinks the token is valid."""
        with self._lock:
            if not self._creds.refresh_token:
                raise OAuthError(
                    "cannot refresh: credential source has no refresh token "
                    "(CLAUDE_CODE_OAUTH_TOKEN env tokens are not refreshable)"
                )
            self._refresh_locked()
            return self._creds.access_token

    def _refresh_locked(self) -> None:
        refresh_token = self._creds.refresh_token
        assert refresh_token  # guarded by callers
        lockfile = Path(claude_config_dir()) / ".credentials.lock"
        with _file_lock(lockfile):
            # Another process may have refreshed while we waited; prefer that
            # token over making a second network call.
            latest = load_credentials()
            if (
                latest is not None
                and latest.access_token != self._creds.access_token
                and not _is_expired(latest.expires_at)
            ):
                self._creds = latest
                return
            try:
                new_creds = refresh_tokens(refresh_token)
            except RefreshError as exc:
                raise OAuthError(str(exc)) from exc
            self._creds = new_creds
            save_credentials(new_creds)


@contextlib.contextmanager
def _file_lock(path: Path) -> Iterator[None]:
    """Cross-process refresh lock. flock on POSIX, mkdir fallback elsewhere."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import fcntl  # noqa: PLC0415 - POSIX-only, platform-gated import
    except ImportError:
        yield from _mkdir_lock(path)
        return

    fd = os.open(str(path), os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def _mkdir_lock(path: Path) -> Iterator[None]:
    lock_dir = path.with_suffix(path.suffix + ".d")
    deadline = time.time() + 30
    while True:
        try:
            lock_dir.mkdir()
            break
        except FileExistsError as exc:
            if time.time() >= deadline:
                raise OAuthError("timed out waiting for oauth refresh lock") from exc
            time.sleep(0.1)
    try:
        yield
    finally:
        with contextlib.suppress(OSError):
            lock_dir.rmdir()
