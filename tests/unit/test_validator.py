"""Tests for the TDA action-protocol validator."""

from __future__ import annotations

import pytest
from pydantic import ValidationError as PydanticValidationError

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
    # Unknown actions are now rejected at RawDecision construction time
    # (pydantic Literal enforcement), not at TDAValidator.check().
    with pytest.raises(PydanticValidationError):
        RawDecision(kind="action", action="nuke")  # type: ignore[arg-type]


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
    # Probe/probe_reply without a non-empty message is rejected at
    # RawDecision construction (pydantic model_validator), so the invariant
    # now holds wherever a RawDecision exists — not only at validator time.
    with pytest.raises(PydanticValidationError, match="message"):
        RawDecision(kind="probe", message="")


def test_validator_rejects_action_from_wrong_seat() -> None:
    t = Table(_cfg())
    v = TDAValidator(t)
    seat = t.next_actor()
    assert seat is not None
    wrong = (seat + 1) % 6
    with pytest.raises(ValidationError, match="not the current actor"):
        v.check(wrong, RawDecision(kind="action", action="fold"))


def test_validator_uses_pokerkit_min_raise() -> None:
    """min_raise_to() uses pokerkit's reported value (current_bet + last_raise_size)."""
    cfg = TableConfig(
        seat_count=2,
        small_blind=10,
        big_blind=20,
        ante=0,
        starting_stacks=(1000, 1000),
    )
    t = Table(cfg)
    v = TDAValidator(t)
    seat = t.next_actor()
    assert seat is not None

    # First actor preflop (UTG in 2-seat = SB). BB=20, so min open is 40.
    # Amount 30 is below min (40) — must reject.
    with pytest.raises(ValidationError, match="min raise"):
        v.check(seat, RawDecision(kind="action", action="raise", amount=30))

    # Amount 40 equals min — must accept.
    v.check(seat, RawDecision(kind="action", action="raise", amount=40))


def test_validator_min_raise_after_previous_raise() -> None:
    """After BB=20 and an open to 60, the next raise must be >= 100 (60 + 40)."""
    cfg = TableConfig(
        seat_count=3,
        small_blind=10,
        big_blind=20,
        ante=0,
        starting_stacks=(1000, 1000, 1000),
    )
    t = Table(cfg)
    v = TDAValidator(t)

    # First actor (UTG=seat2 in 3-handed) raises to 60 (raise size = 40 over BB=20)
    seat = t.next_actor()
    assert seat is not None
    v.check(seat, RawDecision(kind="action", action="raise", amount=60))
    t.apply_raise(seat, to=60)

    # Next actor: min re-raise must be 60 + 40 = 100
    seat2 = t.next_actor()
    assert seat2 is not None

    # Amount 80 is below the required 100 — must reject
    with pytest.raises(ValidationError, match="min raise"):
        v.check(seat2, RawDecision(kind="action", action="raise", amount=80))

    # Amount 100 meets the minimum — must accept
    v.check(seat2, RawDecision(kind="action", action="raise", amount=100))
