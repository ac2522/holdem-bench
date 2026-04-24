"""P1-A: narrow the ValueError catch in _apply_raw_to_table."""

from __future__ import annotations

import pytest

from holdembench.engine.config import TableConfig
from holdembench.engine.table import Table
from holdembench.engine.validator import RawDecision
from holdembench.harness.runner import _apply_raw_to_table


def _cfg() -> TableConfig:
    return TableConfig(
        seat_count=2,
        small_blind=10,
        big_blind=20,
        ante=0,
        starting_stacks=(1000, 1000),
    )


def test_fold_when_no_reason_downgrades_to_check() -> None:
    """Pokerkit raises 'no reason to fold' when BB can check back; we downgrade."""
    t = Table(_cfg())
    idx = t.next_actor()
    assert idx is not None
    t.apply_check_or_call(idx)  # SB completes
    idx2 = t.next_actor()
    assert idx2 is not None
    # BB has no reason to fold — runner should silently downgrade to check.
    _apply_raw_to_table(t, idx2, RawDecision(kind="action", action="fold"))


def test_unrelated_value_error_from_raise_is_not_swallowed() -> None:
    """Raise with an impossibly large amount must surface its ValueError."""
    t = Table(_cfg())
    idx = t.next_actor()
    assert idx is not None
    with pytest.raises(ValueError, match="greater than|stack|invalid"):
        _apply_raw_to_table(t, idx, RawDecision(kind="action", action="raise", amount=999_999_999))


def test_unexpected_fold_error_message_is_reraised(monkeypatch: pytest.MonkeyPatch) -> None:
    """If pokerkit evolves and fold-rejection has a different message, re-raise."""
    t = Table(_cfg())
    idx = t.next_actor()
    assert idx is not None

    def _raise_unexpected(_idx: int) -> None:
        raise ValueError("some unexpected pokerkit invariant failed")

    monkeypatch.setattr(t, "apply_fold", _raise_unexpected)
    with pytest.raises(ValueError, match="unexpected pokerkit"):
        _apply_raw_to_table(t, idx, RawDecision(kind="action", action="fold"))
