"""Tests for the pokerkit Table wrapper."""

from __future__ import annotations

import pytest

from holdembench.engine.config import TableConfig
from holdembench.engine.table import Table

# Expected pot = small_blind (10) + big_blind (20) with auto-posted blinds.
_EXPECTED_INITIAL_POT = 30
_TEST_SEAT_COUNT = 6
_RAISE_AMOUNT = 60


def _cfg(seat_count: int = 9) -> TableConfig:
    return TableConfig(
        seat_count=seat_count,
        small_blind=10,
        big_blind=20,
        ante=0,
        starting_stacks=tuple(1000 for _ in range(seat_count)),
    )


def test_table_initial_state_has_blinds_posted() -> None:
    t = Table(_cfg())
    # small blind = 10, big blind = 20 → pot committed before first action
    assert t.pot_committed() == _EXPECTED_INITIAL_POT


def test_table_seat_count_matches_config() -> None:
    cfg = _cfg(seat_count=_TEST_SEAT_COUNT)
    t = Table(cfg)
    assert t.seat_count == _TEST_SEAT_COUNT


def test_next_actor_advances_after_action() -> None:
    t = Table(_cfg())
    first = t.next_actor()
    assert first is not None
    t.apply_fold(first)
    second = t.next_actor()
    assert second != first


def test_apply_raise_updates_current_bet() -> None:
    t = Table(_cfg())
    seat = t.next_actor()
    assert seat is not None
    t.apply_raise(seat, to=_RAISE_AMOUNT)
    assert t.current_bet() == _RAISE_AMOUNT


def test_apply_invalid_action_raises() -> None:
    t = Table(_cfg())
    seat = t.next_actor()
    assert seat is not None
    with pytest.raises(ValueError, match="invalid raise"):
        t.apply_raise(seat, to=5)  # below min-raise


def test_hand_over_flag() -> None:
    t = Table(_cfg(seat_count=2))
    # One player folds → hand is over
    seat = t.next_actor()
    assert seat is not None
    t.apply_fold(seat)
    assert t.hand_is_over()
