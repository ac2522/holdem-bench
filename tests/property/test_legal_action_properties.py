"""Hypothesis: TDAValidator never accepts a malformed action."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from holdembench.engine.config import TableConfig
from holdembench.engine.table import Table
from holdembench.engine.validator import RawDecision, TDAValidator, ValidationError

_MIN_RAISE_AMOUNT = 40  # 2 * BB (big blind is 20)


def _fresh_table() -> Table:
    cfg = TableConfig(
        seat_count=6,
        small_blind=10,
        big_blind=20,
        ante=0,
        starting_stacks=tuple(1000 for _ in range(6)),
    )
    return Table(cfg)


@given(st.text(min_size=1, max_size=20))
@settings(max_examples=100, deadline=None)
def test_invalid_action_names_always_rejected(bad: str) -> None:
    t = _fresh_table()
    v = TDAValidator(t)
    seat = t.next_actor()
    assert seat is not None
    if bad in {"fold", "check", "call", "raise"}:
        return  # skip — valid name
    try:
        v.check(seat, RawDecision(kind="action", action=bad))  # type: ignore[arg-type]
        pytest.fail(f"validator accepted unknown action {bad!r}")
    except ValidationError:
        pass


@given(st.integers(max_value=40))
@settings(max_examples=50, deadline=None)
def test_raise_below_min_always_rejected(amount: int) -> None:
    t = _fresh_table()
    v = TDAValidator(t)
    seat = t.next_actor()
    assert seat is not None
    # min raise is 2 * current_bet (BB posted), so anything < threshold invalid
    if amount >= _MIN_RAISE_AMOUNT:
        return
    try:
        v.check(seat, RawDecision(kind="action", action="raise", amount=amount))
        pytest.fail(f"validator accepted raise={amount}")
    except ValidationError:
        pass
