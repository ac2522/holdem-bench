"""MoonshotAgent — Kimi via OpenAI-compatible endpoint with explicit cache_control.

Unlike plain OpenAI, Moonshot honours an explicit ``cache_control`` field on
message objects (similar to Anthropic).  Override ``_call_provider`` to emit
that hint on the two system messages; everything else inherits from
:class:`OpenAIAgent`.
"""

from __future__ import annotations

from typing import Any

from holdembench.agents.base import DecisionContext
from holdembench.agents.base_adapter import ProviderCall, Usage
from holdembench.agents.openai import OpenAIAgent, OpenAIClientProtocol

DEFAULT_MOONSHOT_BASE_URL = "https://api.moonshot.ai/v1"


class MoonshotAgent(OpenAIAgent):
    async def _call_provider(
        self,
        ctx: DecisionContext,
        *,
        retry_reason: str | None,
    ) -> ProviderCall:
        bundle = self._render(ctx)
        user = bundle.user_session_log + "\n\n" + bundle.user_volatile
        if retry_reason:
            user += f"\n\nRETRY: {retry_reason}"
        kwargs: dict[str, Any] = {
            "model": self._sdk_model_name(),
            "messages": [
                {
                    "role": "system",
                    "content": bundle.system_tournament,
                    "cache_control": {"type": "ephemeral"},
                },
                {
                    "role": "system",
                    "content": bundle.system_session,
                    "cache_control": {"type": "ephemeral"},
                },
                {"role": "user", "content": user},
            ],
            "max_tokens": 1024,
        }
        client: OpenAIClientProtocol = self._client  # type: ignore[assignment]
        resp = await client.chat.completions.create(**kwargs)
        text = resp.choices[0].message.content or ""
        u = resp.usage
        details = getattr(u, "prompt_tokens_details", None)
        cache_read = int(getattr(details, "cached_tokens", 0)) if details else 0
        prompt_tokens = int(getattr(u, "prompt_tokens", 0))
        usage = Usage(
            input_tokens=max(0, prompt_tokens - cache_read),
            output_tokens=int(getattr(u, "completion_tokens", 0)),
            cache_read_tokens=cache_read,
        )
        return ProviderCall(text=text, usage=usage, latency_ms=0)
