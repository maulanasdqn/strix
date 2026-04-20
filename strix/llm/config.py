from typing import Any

from strix.config import Config
from strix.llm.oauth import ClaudeCodeAuth, load_claude_code_model
from strix.llm.oauth.credentials import normalize_claude_code_model


class LLMConfig:
    """Claude Code OAuth-only configuration.

    Strix authenticates exclusively through Claude Code OAuth credentials —
    if ``~/.claude/.credentials.json`` / ``CLAUDE_CODE_OAUTH_TOKEN`` is
    missing we refuse to start. Model selection follows a simple chain:
    explicit ``model_name`` arg, then ``STRIX_LLM`` env var, then
    ``~/.claude/settings.json``, then a hardcoded default.
    """

    DEFAULT_MODEL = "claude-sonnet-4-6"

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
        raw_model = (
            model_name
            or Config.get("strix_llm")
            or load_claude_code_model()
            or self.DEFAULT_MODEL
        )

        # Normalize Claude Code shorthand (``sonnet-4.6``, ``sonnet``) to the
        # canonical ``claude-*-*-*`` id and then prefix the provider so
        # capability lookups in litellm still resolve.
        self.model_name = raw_model
        if "/" not in self.model_name:
            self.model_name = normalize_claude_code_model(self.model_name)
            if not self.model_name.lower().startswith("anthropic/"):
                self.model_name = f"anthropic/{self.model_name}"

        # Both names are the same under OAuth; kept as distinct attributes
        # so downstream code (memory_compressor, capability lookups) can
        # access them without branching.
        self.litellm_model: str = self.model_name
        self.canonical_model: str = self.model_name

        self.enable_prompt_caching = enable_prompt_caching
        self.skills = skills or []

        self.timeout = timeout or int(Config.get("llm_timeout") or "300")

        self.scan_mode = scan_mode if scan_mode in ["quick", "standard", "deep"] else "deep"
        self.is_whitebox = is_whitebox
        self.interactive = interactive
        self.reasoning_effort = reasoning_effort
        self.system_prompt_context = system_prompt_context or {}

        # Fail loud if Claude Code OAuth credentials aren't available —
        # strix has no other auth method.
        self.oauth_client: ClaudeCodeAuth = ClaudeCodeAuth.from_environment()
