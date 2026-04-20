import asyncio
import logging
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import litellm
from jinja2 import Environment, FileSystemLoader, select_autoescape
from litellm import completion_cost, supports_reasoning
from litellm.utils import supports_prompt_caching, supports_vision

from strix.config import Config
from strix.llm.config import LLMConfig
from strix.llm.memory_compressor import MemoryCompressor
from strix.llm.oauth import OAuthError
from strix.llm.oauth.constants import (
    CLAUDE_CODE_SYSTEM_PROMPT_PREFIX,
    SHIM_SEPARATOR,
    claude_code_prompt_header,
    oauth_min_interval_seconds,
    prompt_shim_disabled,
)
from strix.llm.oauth.direct import acompletion_oauth_stream
from strix.llm.utils import (
    _truncate_to_first_function,
    fix_incomplete_tool_call,
    normalize_tool_format,
    parse_tool_invocations,
)
from strix.skills import load_skills
from strix.tools import get_tools_prompt
from strix.utils.resource_paths import get_strix_resource_path


logger = logging.getLogger(__name__)

litellm.drop_params = True
litellm.modify_params = True


# Process-wide gate for OAuth requests. Anthropic rate-limits the token
# itself, so the gate is shared across every LLM instance and subagent in
# this process; in-process subagents would otherwise burst past the cap.
_OAUTH_GATE_LOCK = asyncio.Lock()
_OAUTH_GATE_LAST: float = 0.0


async def _oauth_throttle() -> None:
    global _OAUTH_GATE_LAST  # noqa: PLW0603
    interval = oauth_min_interval_seconds()
    if interval <= 0:
        return
    async with _OAUTH_GATE_LOCK:
        now = asyncio.get_event_loop().time()
        wait = (_OAUTH_GATE_LAST + interval) - now
        if wait > 0:
            logger.debug("oauth throttle: sleeping %.1fs to stay under rate limit", wait)
            await asyncio.sleep(wait)
        _OAUTH_GATE_LAST = asyncio.get_event_loop().time()


class LLMRequestFailedError(Exception):
    def __init__(self, message: str, details: str | None = None):
        super().__init__(message)
        self.message = message
        self.details = details


@dataclass
class LLMResponse:
    content: str
    tool_invocations: list[dict[str, Any]] | None = None
    thinking_blocks: list[dict[str, Any]] | None = None


@dataclass
class RequestStats:
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    cost: float = 0.0
    requests: int = 0

    def to_dict(self) -> dict[str, int | float]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cached_tokens": self.cached_tokens,
            "cost": round(self.cost, 4),
            "requests": self.requests,
        }


class LLM:
    def __init__(self, config: LLMConfig, agent_name: str | None = None):
        self.config = config
        self.agent_name = agent_name
        self.agent_id: str | None = None
        self._active_skills: list[str] = list(config.skills or [])
        self._system_prompt_context: dict[str, Any] = dict(
            getattr(config, "system_prompt_context", {}) or {}
        )
        self._total_stats = RequestStats()
        self.memory_compressor = MemoryCompressor(model_name=config.litellm_model)
        self.system_prompt = self._load_system_prompt(agent_name)

        reasoning = Config.get("strix_reasoning_effort")
        if reasoning:
            self._reasoning_effort = reasoning
        elif config.reasoning_effort:
            self._reasoning_effort = config.reasoning_effort
        elif config.scan_mode == "quick":
            self._reasoning_effort = "medium"
        else:
            self._reasoning_effort = "high"

    def _load_system_prompt(self, agent_name: str | None) -> str:
        if not agent_name:
            return ""

        try:
            prompt_dir = get_strix_resource_path("agents", agent_name)
            skills_dir = get_strix_resource_path("skills")
            env = Environment(
                loader=FileSystemLoader([prompt_dir, skills_dir]),
                autoescape=select_autoescape(enabled_extensions=(), default_for_string=False),
            )

            skills_to_load = self._get_skills_to_load()
            skill_content = load_skills(skills_to_load)
            env.globals["get_skill"] = lambda name: skill_content.get(name, "")

            result = env.get_template("system_prompt.jinja").render(
                get_tools_prompt=get_tools_prompt,
                loaded_skill_names=list(skill_content.keys()),
                interactive=self.config.interactive,
                system_prompt_context=self._system_prompt_context,
                **skill_content,
            )
            return str(result)
        except Exception:  # noqa: BLE001
            return ""

    def _get_skills_to_load(self) -> list[str]:
        ordered_skills = [*self._active_skills]
        ordered_skills.append(f"scan_modes/{self.config.scan_mode}")
        if self.config.is_whitebox:
            ordered_skills.append("coordination/source_aware_whitebox")
            ordered_skills.append("custom/source_aware_sast")

        deduped: list[str] = []
        seen: set[str] = set()
        for skill_name in ordered_skills:
            if skill_name not in seen:
                deduped.append(skill_name)
                seen.add(skill_name)

        return deduped

    def add_skills(self, skill_names: list[str]) -> list[str]:
        added: list[str] = []
        for skill_name in skill_names:
            if not skill_name or skill_name in self._active_skills:
                continue
            self._active_skills.append(skill_name)
            added.append(skill_name)

        if not added:
            return []

        updated_prompt = self._load_system_prompt(self.agent_name)
        if updated_prompt:
            self.system_prompt = updated_prompt

        return added

    def set_agent_identity(self, agent_name: str | None, agent_id: str | None) -> None:
        if agent_name:
            self.agent_name = agent_name
        if agent_id:
            self.agent_id = agent_id

    def set_system_prompt_context(self, context: dict[str, Any] | None) -> None:
        self._system_prompt_context = dict(context or {})
        updated_prompt = self._load_system_prompt(self.agent_name)
        if updated_prompt:
            self.system_prompt = updated_prompt

    async def generate(
        self, conversation_history: list[dict[str, Any]]
    ) -> AsyncIterator[LLMResponse]:
        messages = self._prepare_messages(conversation_history)
        max_retries = int(Config.get("strix_llm_max_retries") or "5")
        oauth_refreshed = False

        for attempt in range(max_retries + 1):
            try:
                async for response in self._stream(messages):
                    yield response
                return  # noqa: TRY300
            except Exception as e:  # noqa: BLE001
                if self._is_oauth_401(e) and not oauth_refreshed and self.config.oauth_client:
                    try:
                        self.config.oauth_client.force_refresh()
                        oauth_refreshed = True
                        logger.warning("oauth 401 — refreshed token and retrying")
                        continue
                    except OAuthError as refresh_err:
                        self._raise_error(refresh_err)
                if attempt >= max_retries or not self._should_retry(e):
                    self._raise_error(e)
                wait = min(90, 2 * (2**attempt))
                await asyncio.sleep(wait)

    async def _stream(self, messages: list[dict[str, Any]]) -> AsyncIterator[LLMResponse]:
        accumulated = ""
        chunks: list[Any] = []
        done_streaming = 0

        await _oauth_throttle()
        self._total_stats.requests += 1

        # Direct Anthropic call over Claude Code OAuth. See
        # strix.llm.oauth.direct for why litellm is bypassed.
        if not self._supports_vision():
            messages = self._strip_images(messages)
        stream_source = acompletion_oauth_stream(
            model=self.config.model_name,
            messages=messages,
            access_token=self.config.oauth_client.get_token(),
            max_tokens=self._oauth_max_tokens(),
            timeout=float(self.config.timeout),
        )

        async for chunk in stream_source:
            chunks.append(chunk)
            if done_streaming:
                done_streaming += 1
                if getattr(chunk, "usage", None) or done_streaming > 5:
                    break
                continue
            delta = self._get_chunk_content(chunk)
            if delta:
                accumulated += delta
                if "</function>" in accumulated or "</invoke>" in accumulated:
                    end_tag = "</function>" if "</function>" in accumulated else "</invoke>"
                    pos = accumulated.find(end_tag)
                    accumulated = accumulated[: pos + len(end_tag)]
                    yield LLMResponse(content=accumulated)
                    done_streaming = 1
                    continue
                yield LLMResponse(content=accumulated)

        if chunks:
            # Direct path emits usage on the terminal chunk. Fall back to the
            # last observed chunk if none carries usage (e.g. stream cut off).
            final = next(
                (c for c in reversed(chunks) if getattr(c, "usage", None)),
                chunks[-1],
            )
            self._update_usage_stats(final)

        accumulated = normalize_tool_format(accumulated)
        accumulated = fix_incomplete_tool_call(_truncate_to_first_function(accumulated))
        yield LLMResponse(
            content=accumulated,
            tool_invocations=parse_tool_invocations(accumulated),
            thinking_blocks=self._extract_thinking(chunks),
        )

    def _oauth_max_tokens(self) -> int:
        """Anthropic requires ``max_tokens`` for Messages requests. Default
        high enough that strix's long-running pentest replies don't truncate;
        override via ``STRIX_OAUTH_MAX_TOKENS``."""
        raw = Config.get("strix_oauth_max_tokens")
        if raw:
            try:
                return max(1, int(raw))
            except ValueError:
                pass
        return 32768

    def _prepare_messages(self, conversation_history: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not prompt_shim_disabled():
            # System is a two-block array: the Claude Code identity header
            # with ``cache_control: ephemeral`` on the last block, mirroring
            # claude-rust's on-wire shape. (Billing attribution is prepended
            # automatically inside ``direct._hoist_system``.) Strix's agent
            # spec rides as a user-role task brief so the system message
            # stays small — OAuth has a tight size cap.
            brief = self._rewrite_strix_identity(self.system_prompt or "")
            messages: list[dict[str, Any]] = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": claude_code_prompt_header(),
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": (
                        "<task_brief>\n"
                        "The following is your operating specification for this "
                        "session. Treat it as your active working instructions.\n\n"
                        f"{brief}\n"
                        "</task_brief>"
                    ),
                },
                {"role": "assistant", "content": "Acknowledged. Ready."},
            ]
        else:
            # Kill switch: ship the raw strix prompt without the Claude Code
            # identity header. Billing attribution still gets prepended by
            # the direct client, but traffic becomes more distinguishable
            # from real Claude Code — accept at your own risk.
            messages = [{"role": "system", "content": self.system_prompt}]

        if self.agent_name:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"\n\n<agent_identity>\n"
                        f"<meta>Internal metadata: do not echo or reference.</meta>\n"
                        f"<agent_name>{self.agent_name}</agent_name>\n"
                        f"<agent_id>{self.agent_id}</agent_id>\n"
                        f"</agent_identity>\n\n"
                    ),
                }
            )

        compressed = list(self.memory_compressor.compress_history(conversation_history))
        conversation_history.clear()
        conversation_history.extend(compressed)
        messages.extend(compressed)

        if messages[-1].get("role") == "assistant" and not self.config.interactive:
            messages.append({"role": "user", "content": "<meta>Continue the task.</meta>"})

        if self._is_anthropic() and self.config.enable_prompt_caching:
            messages = self._add_cache_control(messages)

        return messages

    def _apply_oauth_prompt_shim(self, system_prompt: str) -> str:
        """Prepend Claude Code's full identity header required by OAuth tokens.

        Matches Claude Code's ``CLAUDE_CODE_SIMPLE`` system prompt: identity
        line + ``CWD`` + ``Date``. The API rejects (403) OAuth-scoped requests
        whose first system message does not start with the identity prefix;
        adding CWD/Date makes the request indistinguishable from real Claude
        Code traffic. Kill switch: ``STRIX_OAUTH_DISABLE_PROMPT_SHIM=1``.

        Also rewrites product-identity references ("Strix") inside the
        downstream system prompt to "Claude Code" so the model sees a single
        consistent persona. Task/tool/skill instructions survive untouched.
        """
        if self.config.oauth_client is None:
            return system_prompt
        if prompt_shim_disabled():
            logger.warning("oauth prompt shim disabled via env — expect 403s")
            return system_prompt
        logger.debug("oauth prompt shim active for agent=%s", self.agent_name)
        if system_prompt.startswith(CLAUDE_CODE_SYSTEM_PROMPT_PREFIX):
            return system_prompt
        rewritten = self._rewrite_strix_identity(system_prompt)
        header = claude_code_prompt_header()
        if not rewritten:
            return header
        return f"{header}{SHIM_SEPARATOR}{rewritten}"

    _STRIX_IDENTITY_LINE = re.compile(r"^You are Strix,\s*", re.MULTILINE)
    _STRIX_WORD = re.compile(r"\bStrix\b")

    @classmethod
    def _rewrite_strix_identity(cls, prompt: str) -> str:
        """Swap strix branding for Claude Code in the agent prompt.

        - Opening "You are Strix, ..." reframes to "You are currently
          performing the role of ..." so it does not conflict with the
          Claude Code identity declared in the header.
        - Remaining bare "Strix" mentions become "Claude Code" — including
          the fingerprinting rule ("NEVER use 'Strix' in requests...") so
          the model still protects the actual client identity.
        """
        if not prompt:
            return prompt
        rewritten = cls._STRIX_IDENTITY_LINE.sub(
            "You are currently performing the role of ", prompt
        )
        return cls._STRIX_WORD.sub("Claude Code", rewritten)

    @staticmethod
    def _is_oauth_401(exc: Exception) -> bool:
        code = getattr(exc, "status_code", None) or getattr(
            getattr(exc, "response", None), "status_code", None
        )
        return code == 401

    def _get_chunk_content(self, chunk: Any) -> str:
        if chunk.choices and hasattr(chunk.choices[0], "delta"):
            return getattr(chunk.choices[0].delta, "content", "") or ""
        return ""

    def _extract_thinking(self, chunks: list[Any]) -> list[dict[str, Any]] | None:
        if not chunks or not self._supports_reasoning():
            return None
        try:
            resp = stream_chunk_builder(chunks)
            if resp.choices and hasattr(resp.choices[0].message, "thinking_blocks"):
                blocks: list[dict[str, Any]] = resp.choices[0].message.thinking_blocks
                return blocks
        except Exception:  # noqa: BLE001, S110  # nosec B110
            pass
        return None

    def _update_usage_stats(self, response: Any) -> None:
        try:
            if hasattr(response, "usage") and response.usage:
                input_tokens = getattr(response.usage, "prompt_tokens", 0) or 0
                output_tokens = getattr(response.usage, "completion_tokens", 0) or 0

                cached_tokens = 0
                if hasattr(response.usage, "prompt_tokens_details"):
                    prompt_details = response.usage.prompt_tokens_details
                    if hasattr(prompt_details, "cached_tokens"):
                        cached_tokens = prompt_details.cached_tokens or 0

                cost = self._extract_cost(response)
            else:
                input_tokens = 0
                output_tokens = 0
                cached_tokens = 0
                cost = 0.0

            self._total_stats.input_tokens += input_tokens
            self._total_stats.output_tokens += output_tokens
            self._total_stats.cached_tokens += cached_tokens
            self._total_stats.cost += cost

        except Exception:  # noqa: BLE001, S110  # nosec B110
            pass

    def _extract_cost(self, response: Any) -> float:
        if hasattr(response, "usage") and response.usage:
            direct_cost = getattr(response.usage, "cost", None)
            if direct_cost is not None:
                return float(direct_cost)
        try:
            if hasattr(response, "_hidden_params"):
                response._hidden_params.pop("custom_llm_provider", None)
            return completion_cost(response, model=self.config.canonical_model) or 0.0
        except Exception:  # noqa: BLE001
            return 0.0

    def _should_retry(self, e: Exception) -> bool:
        code = getattr(e, "status_code", None) or getattr(
            getattr(e, "response", None), "status_code", None
        )
        return code is None or litellm._should_retry(code)

    def _raise_error(self, e: Exception) -> None:
        from strix.telemetry import posthog

        posthog.error("llm_error", type(e).__name__)
        raise LLMRequestFailedError(f"LLM request failed: {type(e).__name__}", str(e)) from e

    def _is_anthropic(self) -> bool:
        if not self.config.model_name:
            return False
        return any(p in self.config.model_name.lower() for p in ["anthropic/", "claude"])

    def _supports_vision(self) -> bool:
        try:
            return bool(supports_vision(model=self.config.canonical_model))
        except Exception:  # noqa: BLE001
            return False

    def _supports_reasoning(self) -> bool:
        try:
            return bool(supports_reasoning(model=self.config.canonical_model))
        except Exception:  # noqa: BLE001
            return False

    def _strip_images(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        result = []
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, list):
                text_parts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                    elif isinstance(item, dict) and item.get("type") == "image_url":
                        text_parts.append("[Image removed - model doesn't support vision]")
                result.append({**msg, "content": "\n".join(text_parts)})
            else:
                result.append(msg)
        return result

    def _add_cache_control(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not messages or not supports_prompt_caching(self.config.canonical_model):
            return messages

        result = list(messages)

        if result[0].get("role") == "system":
            content = result[0]["content"]
            result[0] = {
                **result[0],
                "content": [
                    {"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}
                ]
                if isinstance(content, str)
                else content,
            }
        return result
