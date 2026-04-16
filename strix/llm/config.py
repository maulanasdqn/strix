from typing import Any

from strix.config import Config
from strix.config.config import resolve_llm_config
from strix.llm.oauth import ClaudeCodeAuth, is_oauth_enabled, load_claude_code_model
from strix.llm.utils import resolve_strix_model


class LLMConfig:
    def __init__(
        self,
        model_name: str | None = None,
        enable_prompt_caching: bool = True,
        skills: list[str] | None = None,
        timeout: int | None = None,
        scan_mode: str = "deep",
        is_whitebox: bool = False,
        interactive: bool = False,
        reasoning_effort: str | None = None,
        system_prompt_context: dict[str, Any] | None = None,
    ):
        resolved_model, self.api_key, self.api_base = resolve_llm_config()
        self.model_name = model_name or resolved_model

        # When Claude Code OAuth is enabled, mirror Claude Code's own model
        # selection: read ``~/.claude/settings.json`` and use whatever the user
        # picked there as the default. Reduces drift between strix and the
        # caller's Claude Code session, so requests look like they came from
        # the same client (model id, headers, system prompt prefix all match).
        # ``STRIX_LLM`` still wins if set explicitly.
        if not self.model_name and is_oauth_enabled():
            self.model_name = load_claude_code_model()

        if not self.model_name:
            raise ValueError("STRIX_LLM environment variable must be set and not empty")

        # Bare Claude model names (Claude Code's spelling) get the litellm
        # provider prefix so litellm knows where to route. The wire request
        # still carries the bare id — matches Claude Code byte-for-byte.
        if (
            is_oauth_enabled()
            and "/" not in self.model_name
            and self.model_name.lower().startswith("claude")
        ):
            self.model_name = f"anthropic/{self.model_name}"

        api_model, canonical = resolve_strix_model(self.model_name)
        self.litellm_model: str = api_model or self.model_name
        self.canonical_model: str = canonical or self.model_name

        self.enable_prompt_caching = enable_prompt_caching
        self.skills = skills or []

        self.timeout = timeout or int(Config.get("llm_timeout") or "300")

        self.scan_mode = scan_mode if scan_mode in ["quick", "standard", "deep"] else "deep"
        self.is_whitebox = is_whitebox
        self.interactive = interactive
        self.reasoning_effort = reasoning_effort
        self.system_prompt_context = system_prompt_context or {}

        self.oauth_client: ClaudeCodeAuth | None = self._maybe_load_oauth()

    def _maybe_load_oauth(self) -> ClaudeCodeAuth | None:
        """Load Claude Code OAuth credentials when opted in and model-compatible.

        Silently returns None when the opt-in flag is off. When on but the
        model is not an Anthropic Claude model, raises so misconfigurations
        fail loud instead of silently falling back to an API key the user
        thought they had disabled.
        """
        if not is_oauth_enabled():
            return None

        if not self._is_anthropic_claude_model():
            raise ValueError(
                "STRIX_USE_CLAUDE_CODE_OAUTH is enabled but the configured "
                f"model ({self.model_name}) is not an Anthropic Claude model. "
                "OAuth only works with anthropic/claude-* models."
            )

        # Lazy import: first-run flow prompts on stdin; keep it off the import path.
        from strix.llm.oauth.first_run import require_acknowledgment  # noqa: PLC0415

        require_acknowledgment()
        return ClaudeCodeAuth.from_environment()

    def _is_anthropic_claude_model(self) -> bool:
        name = (self.model_name or "").lower()
        return name.startswith("anthropic/") or "claude" in name
