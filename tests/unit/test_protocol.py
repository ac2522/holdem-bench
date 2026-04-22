"""Tests for the chat protocol — budget accounting and probe mechanics."""

from __future__ import annotations

import pytest

from holdembench.chat.protocol import ChatProtocol, ChatRuleViolation

BUDGET_PER_ORBIT = 400
PER_ACTION_CAP = 80


def _cp(seats: int = 6) -> ChatProtocol:
    return ChatProtocol(
        seats=tuple(f"Seat{i}" for i in range(1, seats + 1)),
        budget_per_orbit=BUDGET_PER_ORBIT,
        per_action_cap=PER_ACTION_CAP,
    )


def test_new_orbit_gives_full_budget_to_every_seat() -> None:
    cp = _cp()
    cp.start_orbit()
    for s in cp.seats:
        assert cp.budget_remaining(s) == BUDGET_PER_ORBIT


def test_attached_message_decrements_budget() -> None:
    cp = _cp()
    cp.start_orbit()
    cp.spend(seat="Seat1", message="hello", kind="action")
    assert cp.budget_remaining("Seat1") < BUDGET_PER_ORBIT


def test_message_over_cap_is_rejected() -> None:
    cp = _cp()
    cp.start_orbit()
    too_long = " ".join(["word"] * 200)
    with pytest.raises(ChatRuleViolation, match="per-action cap"):
        cp.spend(seat="Seat1", message=too_long, kind="action")


def test_probe_requires_min_tokens() -> None:
    cp = _cp()
    cp.start_orbit()
    cp.start_hand(in_hand={"Seat1", "Seat2", "Seat3"})
    with pytest.raises(ChatRuleViolation, match="min"):
        cp.spend(seat="Seat1", message="hi", kind="probe")


def test_max_two_probes_per_hand() -> None:
    cp = _cp()
    cp.start_orbit()
    cp.start_hand(in_hand={"Seat1", "Seat2", "Seat3"})
    cp.spend("Seat1", "probing once please" * 3, kind="probe")
    cp.spend("Seat1", "action between probes", kind="action")
    cp.spend("Seat1", "probing twice please" * 3, kind="probe")
    with pytest.raises(ChatRuleViolation, match="max 2 probes"):
        cp.spend("Seat1", "probing thrice please" * 3, kind="probe")


def test_folded_seat_is_silenced_for_hand() -> None:
    cp = _cp()
    cp.start_orbit()
    cp.start_hand(in_hand={"Seat1", "Seat2"})
    cp.mark_folded("Seat1")
    with pytest.raises(ChatRuleViolation, match="folded"):
        cp.spend("Seat1", "I'm folded but still talking", kind="action")


def test_exhausted_budget_blocks_further_chat() -> None:
    cp = _cp()
    cp.start_orbit()
    cp.start_hand(in_hand={"Seat1"})
    for _ in range(5):
        cp.spend("Seat1", "m " * 79, kind="action")  # eventually drains budget
    with pytest.raises(ChatRuleViolation, match="budget"):
        cp.spend("Seat1", "one more word", kind="action")


def test_cannot_probe_twice_in_a_row() -> None:
    cp = _cp()
    cp.start_orbit()
    cp.start_hand(in_hand={"Seat1", "Seat2", "Seat3"})
    cp.spend("Seat1", "first probe message long enough" * 3, kind="probe")
    with pytest.raises(ChatRuleViolation, match="twice in a row"):
        cp.spend("Seat1", "second probe right after" * 3, kind="probe")


def test_probe_blocked_when_only_two_remain() -> None:
    cp = _cp()
    cp.start_orbit()
    cp.start_hand(in_hand={"Seat1", "Seat2"})
    with pytest.raises(ChatRuleViolation, match="2 players remain"):
        cp.spend("Seat1", "probing heads-up please", kind="probe")
