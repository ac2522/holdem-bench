"""Per-model USD/MTok pricing as of 2026-04-23.

Commit literal dataclass values.  Any price revision lands as a new suffixed
entry (e.g. ``anthropic:claude-opus-4-7-v2``) — never overwrite, so historical
cost reports stay exact.

Sources captured to ``docs/pricing-snapshots/2026-04-23-provider-prices.md``.
"""

from __future__ import annotations

from holdembench.agents.base import Pricing

PRICING_SHEET: dict[str, Pricing] = {
    # Anthropic — explicit cache_control (ephemeral) pricing
    "anthropic:claude-haiku-4-5": Pricing(
        input_per_mtok=1.0,
        output_per_mtok=5.0,
        cache_read_per_mtok=0.10,
        cache_write_per_mtok=1.25,
    ),
    "anthropic:claude-sonnet-4-6": Pricing(
        input_per_mtok=3.0,
        output_per_mtok=15.0,
        cache_read_per_mtok=0.30,
        cache_write_per_mtok=3.75,
    ),
    "anthropic:claude-opus-4-7": Pricing(
        input_per_mtok=15.0,
        output_per_mtok=75.0,
        cache_read_per_mtok=1.50,
        cache_write_per_mtok=18.75,
    ),
    # OpenAI — automatic caching; cache_read cost applies server-side and is
    # reported in usage.
    "openai:gpt-5-mini": Pricing(
        input_per_mtok=0.40,
        output_per_mtok=1.60,
        cache_read_per_mtok=0.10,
    ),
    "openai:gpt-5": Pricing(
        input_per_mtok=2.50,
        output_per_mtok=10.0,
        cache_read_per_mtok=0.625,
    ),
    # Google — implicit caching (Gemini 1.5+ Pro class)
    "google:gemini-3-flash-preview": Pricing(
        input_per_mtok=0.30,
        output_per_mtok=1.20,
        cache_read_per_mtok=0.075,
    ),
    "google:gemini-3-pro": Pricing(
        input_per_mtok=3.50,
        output_per_mtok=10.5,
        cache_read_per_mtok=0.875,
    ),
    # xAI — OpenAI-compatible; no separate cache pricing surface
    "xai:grok-4": Pricing(
        input_per_mtok=3.0,
        output_per_mtok=15.0,
    ),
    # Moonshot — explicit cache_control
    "moonshot:kimi-k2": Pricing(
        input_per_mtok=0.30,
        output_per_mtok=1.20,
        cache_read_per_mtok=0.03,
        cache_write_per_mtok=0.60,
    ),
    # OpenRouter — rates fetched live from /api/v1/models on 2026-04-26.
    "openrouter:deepseek/deepseek-chat-v3.1": Pricing(
        input_per_mtok=0.15,
        output_per_mtok=0.75,
    ),
    "openrouter:qwen/qwen3-32b": Pricing(
        input_per_mtok=0.08,
        output_per_mtok=0.24,
    ),
    "openrouter:anthropic/claude-haiku-4.5": Pricing(
        input_per_mtok=1.0,
        output_per_mtok=5.0,
    ),
    "openrouter:openai/gpt-4o-mini": Pricing(
        input_per_mtok=0.15,
        output_per_mtok=0.60,
    ),
    "openrouter:google/gemini-2.5-flash": Pricing(
        input_per_mtok=0.30,
        output_per_mtok=2.50,
    ),
    "openrouter:x-ai/grok-3-mini": Pricing(
        input_per_mtok=0.30,
        output_per_mtok=0.50,
    ),
    "openrouter:meta-llama/llama-3.1-8b-instruct": Pricing(
        input_per_mtok=0.02,
        output_per_mtok=0.05,
    ),
}


def lookup_pricing(model_id: str) -> Pricing:
    try:
        return PRICING_SHEET[model_id]
    except KeyError as e:
        raise KeyError(f"unknown model_id in pricing sheet: {model_id!r}") from e
