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

import asyncio
import concurrent.futures
import json
import re
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


_DATA_URL_RE = re.compile(
    r"^data:(?P<media>[a-zA-Z0-9.+-]+/[a-zA-Z0-9.+-]+);base64,(?P<data>.*)$",
    re.DOTALL,
)


def _convert_image_block(block: dict[str, Any]) -> dict[str, Any] | None:
    """Convert an OpenAI ``image_url`` content block to Anthropic's ``image``
    block. Returns None if the shape is unexpected so the caller can drop it
    rather than send something the API will reject."""
    image_url = block.get("image_url") or {}
    url = image_url.get("url") if isinstance(image_url, dict) else None
    if not isinstance(url, str) or not url:
        return None
    match = _DATA_URL_RE.match(url)
    if match:
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": match.group("media"),
                "data": match.group("data"),
            },
        }
    # Non-data URL — send via Anthropic's URL source mode.
    return {"type": "image", "source": {"type": "url", "url": url}}


def _normalize_content(content: Any) -> Any:
    """Convert OpenAI-style content blocks (notably ``image_url``) to the
    Anthropic block shapes the Messages API accepts. Plain strings pass
    through unchanged."""
    if not isinstance(content, list):
        return content
    converted: list[dict[str, Any]] = []
    for block in content:
        if not isinstance(block, dict):
            converted.append(block)
            continue
        block_type = block.get("type")
        if block_type == "image_url":
            image_block = _convert_image_block(block)
            if image_block is not None:
                converted.append(image_block)
            continue
        converted.append(block)
    return converted


_OPENAI_ROLE_REMAP = {
    # OpenAI's tool-response role has no Anthropic equivalent; the closest
    # shape is a user-role message carrying the tool output. Strix encodes
    # tool results as XML inside plain text, so simple re-labelling works.
    "tool": "user",
    "function": "user",
}


def _normalize_message(msg: dict[str, Any]) -> dict[str, Any] | None:
    """Keep only role + content; rewrite OpenAI-only shapes so the Anthropic
    Messages API accepts them. Returns ``None`` if the message ends up
    empty after normalization — callers should drop it rather than forward
    an empty block that Anthropic would 400 on."""
    role = msg.get("role")
    if role in _OPENAI_ROLE_REMAP:
        role = _OPENAI_ROLE_REMAP[role]
    if role not in ("user", "assistant"):
        # ``system`` is hoisted elsewhere; unknown roles get dropped.
        return None
    content = _normalize_content(msg.get("content", ""))
    if not _content_has_payload(content):
        return None
    return {"role": role, "content": content}


def _content_has_payload(content: Any) -> bool:
    """Does this content carry anything Anthropic will accept? Empty strings,
    empty lists, and lists containing only empty text blocks all fail the API
    validator, so we treat them as no-payload."""
    if isinstance(content, str):
        return bool(content.strip())
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                return True
            if block.get("type") == "text":
                if (block.get("text") or "").strip():
                    return True
                continue
            return True
        return False
    return content is not None


def _merge_adjacent_same_role(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Anthropic rejects consecutive same-role messages. This happens in
    practice when sub-agents inherit a user-led context or when tool-result
    user messages land right after the initial user turn. We merge by
    concatenating their contents — strings join with a blank line, lists
    concatenate, and mixed string/list coerce to list so no data is lost."""
    merged: list[dict[str, Any]] = []
    for msg in messages:
        if merged and merged[-1]["role"] == msg["role"]:
            prev = merged[-1]
            prev["content"] = _concat_content(prev["content"], msg["content"])
        else:
            merged.append({**msg})
    return merged


def _concat_content(a: Any, b: Any) -> Any:
    if isinstance(a, str) and isinstance(b, str):
        return f"{a}\n\n{b}"
    if isinstance(a, list) and isinstance(b, list):
        return a + b
    if isinstance(a, str):
        return [{"type": "text", "text": a}, *b]
    # isinstance(b, str)
    return [*a, {"type": "text", "text": b}]


def _build_request_body(
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
) -> dict[str, Any]:
    system_blocks, remaining = _hoist_system(messages)

    normalized: list[dict[str, Any]] = []
    for m in remaining:
        out = _normalize_message(m)
        if out is not None:
            normalized.append(out)

    # Anthropic requires the first message to be user-role. If the
    # conversation starts with assistant (e.g. after a trimming pass
    # removed the opening user turn) prepend a placeholder so the API
    # doesn't 400. Better to surface the history than fail silently.
    if normalized and normalized[0]["role"] != "user":
        normalized.insert(
            0,
            {"role": "user", "content": "<meta>Continue.</meta>"},
        )

    normalized = _merge_adjacent_same_role(normalized)

    body: dict[str, Any] = {
        "model": _strip_model_prefix(model),
        "max_tokens": max_tokens,
        "messages": normalized,
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


async def acompletion_oauth_collect(
    *,
    model: str,
    messages: list[dict[str, Any]],
    access_token: str,
    max_tokens: int = 8192,
    timeout: float = 300.0,
) -> tuple[str, _Usage | None]:
    """Non-streaming convenience wrapper: run the stream to completion and
    return the accumulated text and usage. For callers that don't need
    incremental deltas (e.g. the dedupe judge)."""
    content = ""
    usage: _Usage | None = None
    async for chunk in acompletion_oauth_stream(
        model=model,
        messages=messages,
        access_token=access_token,
        max_tokens=max_tokens,
        timeout=timeout,
    ):
        if chunk.choices and chunk.choices[0].delta.content:
            content += chunk.choices[0].delta.content
        if chunk.usage is not None:
            usage = chunk.usage
    return content, usage


def completion_oauth_collect(
    *,
    model: str,
    messages: list[dict[str, Any]],
    access_token: str,
    max_tokens: int = 8192,
    timeout: float = 300.0,
) -> tuple[str, _Usage | None]:
    """Sync wrapper for callers not already inside an event loop. If a loop
    is running in the current thread we hop to a worker thread to avoid the
    ``asyncio.run() cannot be called from a running event loop`` error."""
    coro = acompletion_oauth_collect(
        model=model,
        messages=messages,
        access_token=access_token,
        max_tokens=max_tokens,
        timeout=timeout,
    )
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()
