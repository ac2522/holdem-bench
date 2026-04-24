"""Tests for the model-id -> Pricing registry."""

from __future__ import annotations

import pytest

from holdembench.agents.base import Pricing
from holdembench.agents.pricing_sheet import PRICING_SHEET, lookup_pricing


def test_known_models_registered() -> None:
    expected = {
        "anthropic:claude-haiku-4-5",
        "anthropic:claude-opus-4-7",
        "openai:gpt-5-mini",
        "openai:gpt-5",
        "google:gemini-3-flash-preview",
        "google:gemini-3-pro",
        "xai:grok-4",
        "moonshot:kimi-k2",
        "openrouter:deepseek/deepseek-chat-v3",
    }
    assert expected <= set(PRICING_SHEET)


def test_lookup_raises_on_unknown() -> None:
    with pytest.raises(KeyError, match="unknown"):
        lookup_pricing("openai:not-a-real-model")


def test_lookup_returns_same_instance() -> None:
    a = lookup_pricing("anthropic:claude-haiku-4-5")
    b = lookup_pricing("anthropic:claude-haiku-4-5")
    assert a is b


def test_pricing_is_nonzero_and_input_cheaper_than_output() -> None:
    for model_id, price in PRICING_SHEET.items():
        assert isinstance(price, Pricing)
        assert price.input_per_mtok > 0, model_id
        assert price.output_per_mtok > price.input_per_mtok, model_id


def test_cache_read_is_cheaper_than_input() -> None:
    for model_id, price in PRICING_SHEET.items():
        if price.cache_read_per_mtok == 0:
            continue  # provider without explicit cache read pricing
        assert price.cache_read_per_mtok < price.input_per_mtok, model_id
