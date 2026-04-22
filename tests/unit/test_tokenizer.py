"""Tests for the canonical cl100k_base tokenizer wrapper."""
from __future__ import annotations

import pytest

from holdembench.chat.tokenizer import count_tokens, truncate_to_budget


@pytest.mark.parametrize(
    ("text", "expected_min", "expected_max"),
    [
        ("", 0, 0),
        ("hello", 1, 2),
        ("The quick brown fox jumps over the lazy dog.", 8, 12),
    ],
)
def test_count_tokens_bounds(text: str, expected_min: int, expected_max: int) -> None:
    n = count_tokens(text)
    assert expected_min <= n <= expected_max


def test_truncate_preserves_short_text() -> None:
    s = "short"
    out = truncate_to_budget(s, max_tokens=10)
    assert out == s


def test_truncate_limits_tokens() -> None:
    max_budget = 10
    long = "word " * 200
    out = truncate_to_budget(long, max_tokens=max_budget)
    assert count_tokens(out) <= max_budget


def test_truncate_zero_budget_returns_empty() -> None:
    assert truncate_to_budget("anything", max_tokens=0) == ""


def test_count_tokens_is_stable() -> None:
    s = "deterministic text"
    assert count_tokens(s) == count_tokens(s)
