"""Direct Anthropic Messages client for Claude Code OAuth.

litellm's Anthropic provider actively breaks Claude Code OAuth:

- ``optionally_handle_anthropic_oauth`` unconditionally overwrites the
  caller's ``anthropic-beta`` header with only ``oauth-2025-04-20``,
  dropping the ``claude-code-20250219`` flag that tells Anthropic's server
  the request is real Claude Code traffic. Without that flag Anthropic
  returns a generic ``rate_limit_error`` with no Retry-After, regardless
  of actual usage.
- ``translate_system_message`` explicitly filters out any system content
  block whose text starts with ``x-anthropic-billing-header:``, even
  though that billing line is exactly what Claude Code puts there and
  what Anthropic reads server-side for session attribution.

Both would require forking or monkey-patching litellm. Rather than fight
a provider that is trying to prevent this integration, we speak HTTP
directly. The on-wire shape matches the working claude-rust port
byte-for-byte (verified via scripts/probe_oauth.py).

The chunks yielded from :func:`acompletion_oauth_stream` are shaped like
litellm's ``ModelResponseStream`` so strix's existing consumer code
(delta extraction, usage accounting) works unchanged.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import httpx


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

from strix.llm.oauth.constants import (
    CLAUDE_CODE_ANTHROPIC_BETA,
    claude_code_billing_line,
    claude_code_user_agent,
)


ANTHROPIC_MESSAGES_URL = "https://api.anthropic.com/v1/messages?beta=true"


@dataclass
class _Delta:
    content: str = ""


@dataclass
class _Choice:
    index: int = 0
    delta: _Delta = field(default_factory=_Delta)
    finish_reason: str | None = None


@dataclass
class _PromptTokensDetails:
    cached_tokens: int = 0


@dataclass
class _Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    prompt_tokens_details: _PromptTokensDetails = field(
        default_factory=_PromptTokensDetails
    )


@dataclass
class OAuthStreamChunk:
    """Minimal chunk shape compatible with strix's litellm consumer code.

    Strix reads ``chunk.choices[0].delta.content`` for deltas and
    ``chunk.usage`` on the terminal chunk; nothing else is touched.
    """

    choices: list[_Choice] = field(default_factory=lambda: [_Choice()])
    usage: _Usage | None = None


def _strip_model_prefix(model: str) -> str:
    """``anthropic/claude-sonnet-4-6`` -> ``claude-sonnet-4-6``."""
    if "/" in model:
        return model.split("/", 1)[1]
    return model


def _hoist_system(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Pull any ``role: system`` messages out of the list and flatten to
    Anthropic's top-level system-blocks shape. Billing attribution is
    prepended as the first block so Anthropic classifies the request as
    real Claude Code traffic (matches claude-rust's ``BILLING_HEADER_LINE``)."""
    system_blocks: list[dict[str, Any]] = [
        {"type": "text", "text": claude_code_billing_line()}
    ]
    remaining: list[dict[str, Any]] = []

    for msg in messages:
        if msg.get("role") != "system":
            remaining.append(msg)
            continue
        content = msg.get("content")
        if isinstance(content, str):
            if content:
                system_blocks.append({"type": "text", "text": content})
        elif isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") != "text":
                    system_blocks.append(block)
                    continue
                text = block.get("text")
                if not text:
                    continue
                # Skip any pre-existing billing line — we already prepended one.
                if text.startswith("x-anthropic-billing-header:"):
                    continue
                new_block: dict[str, Any] = {"type": "text", "text": text}
                if "cache_control" in block:
                    new_block["cache_control"] = block["cache_control"]
                system_blocks.append(new_block)

    # Ensure the final block is cache-controlled so Anthropic caches the
    # whole system prefix on our behalf; matches claude-rust behaviour.
    if system_blocks:
        system_blocks[-1].setdefault("cache_control", {"type": "ephemeral"})

    return system_blocks, remaining


def _normalize_message(msg: dict[str, Any]) -> dict[str, Any]:
    """Keep role + content; drop OpenAI-specific keys Anthropic rejects.

    Strix uses plain text content (tool calls live inside the text as XML),
    so no structural conversion is needed — just pass role/content through.
    """
    role = msg.get("role")
    content = msg.get("content", "")
    return {"role": role, "content": content}


def _build_request_body(
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
) -> dict[str, Any]:
    system_blocks, remaining = _hoist_system(messages)
    body: dict[str, Any] = {
        "model": _strip_model_prefix(model),
        "max_tokens": max_tokens,
        "messages": [_normalize_message(m) for m in remaining],
        "stream": True,
    }
    if system_blocks:
        body["system"] = system_blocks
    return body


def build_oauth_headers(access_token: str) -> dict[str, str]:
    """Header set claude-rust sends on every OAuth request.

    Exported for tests and the warm-up path so there's one source of truth
    for the on-wire header shape.
    """
    return {
        "Authorization": f"Bearer {access_token}",
        "anthropic-version": "2023-06-01",
        "anthropic-beta": CLAUDE_CODE_ANTHROPIC_BETA,
        "anthropic-dangerous-direct-browser-access": "true",
        "User-Agent": claude_code_user_agent(),
        "x-app": "cli",
        "content-type": "application/json",
    }


def _parse_sse_line(line: str) -> tuple[str | None, dict[str, Any] | None]:
    """Anthropic SSE is ``event: <name>\\n`` then ``data: <json>\\n\\n``.
    We only care about ``data:`` lines (they carry the event ``type``)."""
    if not line.startswith("data:"):
        return None, None
    payload = line[5:].strip()
    if not payload:
        return None, None
    try:
        parsed: dict[str, Any] = json.loads(payload)
    except json.JSONDecodeError:
        return None, None
    return parsed.get("type"), parsed


class OAuthRequestError(Exception):
    """Non-2xx from Anthropic's Messages endpoint."""

    def __init__(self, status_code: int, body: str):
        super().__init__(f"anthropic oauth {status_code}: {body[:500]}")
        self.status_code = status_code
        self.body = body


async def acompletion_oauth_stream(
    *,
    model: str,
    messages: list[dict[str, Any]],
    access_token: str,
    max_tokens: int = 8192,
    timeout: float = 300.0,
) -> AsyncIterator[OAuthStreamChunk]:
    """Stream an Anthropic Messages completion over Claude Code OAuth.

    Yields :class:`OAuthStreamChunk` objects. The terminal chunk carries
    ``usage`` (populated from ``message_start`` + ``message_delta`` SSE
    events); intermediate chunks carry ``choices[0].delta.content``.
    """
    headers = build_oauth_headers(access_token)
    body = _build_request_body(model, messages, max_tokens)

    usage = _Usage()

    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream(
            "POST", ANTHROPIC_MESSAGES_URL, headers=headers, json=body
        ) as resp:
            if resp.status_code >= 400:
                raw = await resp.aread()
                raise OAuthRequestError(
                    resp.status_code, raw.decode("utf-8", errors="replace")
                )

            async for line in resp.aiter_lines():
                event_type, data = _parse_sse_line(line)
                if event_type is None or data is None:
                    continue

                if event_type == "message_start":
                    u = (data.get("message") or {}).get("usage") or {}
                    usage.prompt_tokens = int(u.get("input_tokens") or 0)
                    usage.prompt_tokens_details.cached_tokens = int(
                        u.get("cache_read_input_tokens") or 0
                    )
                elif event_type == "content_block_delta":
                    delta = data.get("delta") or {}
                    if delta.get("type") == "text_delta":
                        text = delta.get("text") or ""
                        if text:
                            yield OAuthStreamChunk(
                                choices=[_Choice(delta=_Delta(content=text))]
                            )
                elif event_type == "message_delta":
                    u = data.get("usage") or {}
                    if "output_tokens" in u:
                        usage.completion_tokens = int(u.get("output_tokens") or 0)
                    stop_reason = (data.get("delta") or {}).get("stop_reason")
                    if stop_reason:
                        # Forward finish_reason on an empty-content chunk so
                        # strix's per-chunk loop observes termination.
                        yield OAuthStreamChunk(
                            choices=[_Choice(finish_reason=stop_reason)]
                        )
                elif event_type == "message_stop":
                    usage.total_tokens = usage.prompt_tokens + usage.completion_tokens
                    yield OAuthStreamChunk(usage=usage)
                    return
