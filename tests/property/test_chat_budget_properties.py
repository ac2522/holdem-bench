"""Hypothesis invariants on ChatProtocol."""
from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from holdembench.chat.protocol import ChatProtocol, ChatRuleViolation
from holdembench.chat.tokenizer import count_tokens

_SEATS = tuple(f"Seat{i}" for i in range(1, 7))
_PER_ACTION_CAP = 80


@given(st.integers(min_value=1, max_value=10))
@settings(max_examples=50, deadline=None)
def test_budget_never_goes_negative(hand_count: int) -> None:
    cp = ChatProtocol(seats=_SEATS, budget_per_orbit=400, per_action_cap=_PER_ACTION_CAP)
    cp.start_hand(in_hand=set(_SEATS))
    seat = _SEATS[0]
    for _ in range(hand_count):
        try:
            cp.spend(seat, "short text here to test budget", kind="action")
        except ChatRuleViolation:
            break
    assert cp.budget_remaining(seat) >= 0


@given(st.text(min_size=0, max_size=500))
@settings(max_examples=100, deadline=None)
def test_per_action_cap_enforced(msg: str) -> None:
    cp = ChatProtocol(seats=_SEATS, budget_per_orbit=100_000, per_action_cap=_PER_ACTION_CAP)
    cp.start_hand(in_hand=set(_SEATS))
    if count_tokens(msg) > _PER_ACTION_CAP:
        try:
            cp.spend(_SEATS[0], msg, kind="action")
            pytest.fail("should have raised ChatRuleViolation")
        except ChatRuleViolation:
            pass
