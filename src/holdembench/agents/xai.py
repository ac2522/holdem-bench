"""XAIAgent — Grok via xAI's OpenAI-compatible endpoint.

Thin subclass of :class:`OpenAIAgent`; xAI mirrors the OpenAI schema so all
behaviour (JSON Schema response_format, cache-read usage reporting) carries
over unchanged.  Wire via ``openai.AsyncOpenAI(api_key=..., base_url=...)``
with ``base_url=DEFAULT_XAI_BASE_URL`` at CLI construction time.
"""

from __future__ import annotations

from holdembench.agents.openai import OpenAIAgent

DEFAULT_XAI_BASE_URL = "https://api.x.ai/v1"


class XAIAgent(OpenAIAgent):
    """Grok adapter — inherits every bit of transport + accounting."""
