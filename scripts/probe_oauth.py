"""One-shot OAuth probe: sends a minimal Anthropic Messages request with the
exact headers + body shape claude-rust uses, bypassing litellm entirely.

Goal is to answer: is Anthropic rejecting our wire shape, or is litellm doing
something unexpected on top? If this succeeds and strix still fails, the gap
is in the litellm/strix layer, not in our understanding of Claude Code's
protocol. If this also returns ``rate_limit_error {"message":"Error"}``, the
wire shape itself is wrong.

Usage: ``uv run python scripts/probe_oauth.py``
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx


def load_access_token() -> str:
    creds_path = Path.home() / ".claude" / ".credentials.json"
    if not creds_path.is_file():
        sys.exit(f"no credentials file at {creds_path}")
    data = json.loads(creds_path.read_text(encoding="utf-8"))
    oauth = data.get("claudeAiOauth") or data
    token = oauth.get("accessToken")
    if not token:
        sys.exit("no accessToken in credentials")
    return token


def get_claude_version() -> str:
    import shutil
    import subprocess

    exe = shutil.which("claude")
    if not exe:
        return "2.1.114"
    try:
        out = subprocess.run(
            [exe, "--version"], capture_output=True, text=True, timeout=5, check=False
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip().split()[0]
    except Exception:
        pass
    return "2.1.114"


def main() -> None:
    token = load_access_token()
    version = get_claude_version()

    url = "https://api.anthropic.com/v1/messages?beta=true"
    headers = {
        "Authorization": f"Bearer {token}",
        "anthropic-version": "2023-06-01",
        "anthropic-beta": (
            "oauth-2025-04-20,"
            "interleaved-thinking-2025-05-14,"
            "claude-code-20250219,"
            "prompt-caching-2024-07-31"
        ),
        "anthropic-dangerous-direct-browser-access": "true",
        "User-Agent": f"claude-cli/{version} (external, cli)",
        "x-app": "cli",
        "content-type": "application/json",
    }
    billing_line = (
        f"x-anthropic-billing-header: cc_version={version}; cc_entrypoint=cli;"
    )
    body = {
        "model": "claude-sonnet-4-5",
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "Reply with just 'OK'."}],
        "stream": True,
        "system": [
            {
                "type": "text",
                "text": billing_line,
                "cache_control": {"type": "ephemeral"},
            }
        ],
    }

    print("=== request ===")
    print("URL:", url)
    print("Headers:")
    for k, v in headers.items():
        if k == "Authorization":
            v = v[:20] + "...(redacted)"
        print(f"  {k}: {v}")
    print("Body:", json.dumps(body, indent=2))
    print()
    print("=== response ===")

    try:
        with httpx.Client(timeout=30.0) as client:
            with client.stream("POST", url, headers=headers, json=body) as resp:
                print("Status:", resp.status_code)
                print("Response headers:")
                for k, v in resp.headers.items():
                    print(f"  {k}: {v}")
                print()
                print("Body (first 4KB):")
                buf = bytearray()
                for chunk in resp.iter_bytes():
                    buf.extend(chunk)
                    if len(buf) > 4096:
                        break
                print(buf.decode("utf-8", errors="replace"))
    except httpx.HTTPError as exc:
        print(f"HTTP error: {exc}")


if __name__ == "__main__":
    main()
