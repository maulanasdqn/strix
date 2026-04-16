"""Refresh Claude Code OAuth tokens via Anthropic's token endpoint."""

from __future__ import annotations

import time

import requests

from strix.llm.oauth.constants import (
    CLAUDE_AI_OAUTH_SCOPES,
    oauth_client_id,
    oauth_token_url,
)
from strix.llm.oauth.credentials import OAuthCredentials


class RefreshError(Exception):
    """Raised when the refresh endpoint rejects the request."""


def refresh_tokens(refresh_token: str) -> OAuthCredentials:
    """Exchange a refresh token for a new access+refresh pair."""
    body = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": oauth_client_id(),
        "scope": " ".join(CLAUDE_AI_OAUTH_SCOPES),
    }
    try:
        resp = requests.post(
            oauth_token_url(),
            json=body,
            headers={"Content-Type": "application/json"},
            timeout=15,
        )
    except requests.RequestException as exc:
        raise RefreshError(f"network error during token refresh: {exc}") from exc

    if resp.status_code != 200:
        raise RefreshError(
            f"token refresh failed: HTTP {resp.status_code} {resp.text[:200]}"
        )

    try:
        data = resp.json()
    except ValueError as exc:
        raise RefreshError("token refresh returned non-JSON body") from exc

    access_token = data.get("access_token")
    if not access_token:
        raise RefreshError("token refresh response missing access_token")

    expires_in = int(data.get("expires_in") or 0)
    expires_at = int(time.time() * 1000) + expires_in * 1000 if expires_in else None
    scopes = (data.get("scope") or "").split() or list(CLAUDE_AI_OAUTH_SCOPES)

    return OAuthCredentials(
        access_token=access_token,
        # Endpoint may or may not rotate the refresh token; fall back to prev.
        refresh_token=data.get("refresh_token") or refresh_token,
        expires_at=expires_at,
        scopes=scopes,
    )
