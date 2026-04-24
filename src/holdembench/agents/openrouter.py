"""OpenRouterAgent — Llama / DeepSeek / Qwen / Yi / GLM / etc. via OpenRouter.

Thin subclass of :class:`OpenAIAgent`; OpenRouter mirrors the OpenAI schema.
Model IDs follow the convention ``openrouter:<vendor>/<name>``.
"""

from __future__ import annotations

from holdembench.agents.openai import OpenAIAgent

DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterAgent(OpenAIAgent):
    """OpenRouter adapter — vendor/name slug preserved in the SDK model name."""

    def _sdk_model_name(self) -> str:
        # e.g. "openrouter:deepseek/deepseek-chat-v3" -> "deepseek/deepseek-chat-v3"
        return self.model_id.split(":", 1)[1]
