"""Tests for the TDA action-protocol validator."""
from __future__ import annotations

import pytest

from holdembench.engine.config import TableConfig
from holdembench.engine.table import Table
from holdembench.engine.validator import RawDecision, TDAValidator, ValidationError


def _cfg() -> TableConfig:
    return TableConfig(
        seat_count=6,
        small_blind=10,
        big_blind=20,
        ante=0,
        starting_stacks=tuple(1000 for _ in range(6)),
    )


def test_validator_accepts_fold() -> None:
    t = Table(_cfg())
    v = TDAValidator(t)
    seat = t.next_actor()
    assert seat is not None
    decision = RawDecision(kind="action", action="fold")
    v.check(seat, decision)  # does not raise


def test_validator_rejects_unknown_action() -> None:
    t = Table(_cfg())
    v = TDAValidator(t)
    seat = t.next_actor()
    assert seat is not None
    with pytest.raises(ValidationError):
        v.check(seat, RawDecision(kind="action", action="nuke"))  # type: ignore[arg-type]


def test_validator_rejects_below_min_raise() -> None:
    t = Table(_cfg())
    v = TDAValidator(t)
    seat = t.next_actor()
    assert seat is not None
    # BB = 20, so first raise must be >= 40 total
    with pytest.raises(ValidationError, match="min raise"):
        v.check(seat, RawDecision(kind="action", action="raise", amount=30))


def test_validator_accepts_legal_raise() -> None:
    t = Table(_cfg())
    v = TDAValidator(t)
    seat = t.next_actor()
    assert seat is not None
    v.check(seat, RawDecision(kind="action", action="raise", amount=60))


def test_validator_rejects_probe_without_message() -> None:
    t = Table(_cfg())
    v = TDAValidator(t)
    seat = t.next_actor()
    assert seat is not None
    with pytest.raises(ValidationError, match="message"):
        v.check(seat, RawDecision(kind="probe", message=""))


def test_validator_rejects_action_from_wrong_seat() -> None:
    t = Table(_cfg())
    v = TDAValidator(t)
    seat = t.next_actor()
    assert seat is not None
    wrong = (seat + 1) % 6
    with pytest.raises(ValidationError, match="not the current actor"):
        v.check(wrong, RawDecision(kind="action", action="fold"))
