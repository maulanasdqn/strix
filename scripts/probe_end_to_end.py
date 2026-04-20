"""End-to-end probe: exercises strix's LLM class with Claude Code OAuth
so we can confirm the direct-httpx path works outside of running `strix`
itself (which also spins up Docker, TUI, etc.)."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Pretend we're interactive so the OAuth ack gate uses the env bypass.
os.environ.setdefault("STRIX_USE_CLAUDE_CODE_OAUTH", "1")
os.environ.setdefault("STRIX_OAUTH_ACK", "1")
os.environ.setdefault("STRIX_LLM", "sonnet-4.6")

from strix.llm.config import LLMConfig
from strix.llm.llm import LLM


async def main() -> None:
    cfg = LLMConfig(system_prompt_context={})
    llm = LLM(cfg, agent_name=None)
    llm.system_prompt = "You are a test agent."
    got_text = ""
    async for response in llm.generate([{"role": "user", "content": "Reply with just 'OK'."}]):
        if response.content:
            got_text = response.content
    print("=== final content ===")
    print(repr(got_text))
    print("=== usage ===")
    print(llm._total_stats.to_dict())


if __name__ == "__main__":
    asyncio.run(main())
