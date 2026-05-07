"""Pricing.cost_usd correctly bills thinking_tokens."""

from __future__ import annotations

import pytest

from holdembench.agents.base import Pricing


def test_cost_usd_bills_thinking_at_output_rate_by_default() -> None:
    p = Pricing(input_per_mtok=1.0, output_per_mtok=5.0)
    cost = p.cost_usd(input_tokens=0, output_tokens=0, thinking_tokens=1_000_000)
    assert cost == pytest.approx(5.0)


def test_cost_usd_uses_explicit_thinking_rate_when_set() -> None:
    p = Pricing(input_per_mtok=1.0, output_per_mtok=5.0, thinking_per_mtok=2.0)
    cost = p.cost_usd(input_tokens=0, output_tokens=0, thinking_tokens=1_000_000)
    assert cost == pytest.approx(2.0)


def test_cost_usd_thinking_zero_means_no_charge() -> None:
    p = Pricing(input_per_mtok=1.0, output_per_mtok=5.0)
    cost = p.cost_usd(input_tokens=0, output_tokens=0, thinking_tokens=0)
    assert cost == pytest.approx(0.0)


def test_cost_usd_combines_all_token_types() -> None:
    p = Pricing(
        input_per_mtok=1.0,
        output_per_mtok=5.0,
        cache_read_per_mtok=0.1,
        cache_write_per_mtok=1.25,
        thinking_per_mtok=5.0,
    )
    cost = p.cost_usd(
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        cache_read_tokens=1_000_000,
        cache_write_tokens=1_000_000,
        thinking_tokens=1_000_000,
    )
    # 1.0 + 5.0 + 0.1 + 1.25 + 5.0 = 12.35
    assert cost == pytest.approx(12.35)


def test_thinking_tokens_default_zero() -> None:
    """Calling cost_usd without thinking_tokens still works (back-compat)."""
    p = Pricing(input_per_mtok=1.0, output_per_mtok=5.0)
    cost_explicit = p.cost_usd(input_tokens=100, output_tokens=10, thinking_tokens=0)
    cost_implicit = p.cost_usd(input_tokens=100, output_tokens=10)
    assert cost_explicit == pytest.approx(cost_implicit)
