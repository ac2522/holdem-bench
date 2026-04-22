"""Tests for event schema pydantic models."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from holdembench.events.schema import (
    ActionResponse,
    HandEnd,
    HandStart,
    SessionStart,
    TournamentStart,
    parse_event,
)


def test_tournament_start_minimal() -> None:
    ev = TournamentStart(
        tournament_id="t1",
        schema_version="1.0",
        holdembench_version="0.1.0",
        pokerkit_version="0.7.3",
        git_sha="deadbeef",
        seat_assignments={"Seat1": "stub:random", "Seat2": "stub:tight"},
        master_seed=42,
        anonymization_salt="cafebabe",
        canary_uuid="11111111-1111-1111-1111-111111111111",
    )
    assert ev.type == "tournament_start"
    assert ev.tournament_id == "t1"
    assert ev.seat_assignments["Seat1"] == "stub:random"


def test_action_response_action_kind() -> None:
    amount = 60
    ev = ActionResponse(
        hand_id="s1h001",
        seat="Seat3",
        kind="action",
        action="raise",
        amount=amount,
        message="small raise",
        tokens=3,
        latency_ms=500,
        cost_usd=0.0,
        model_id="stub:random",
        prompt_hash="sha256:abcd",
    )
    assert ev.action == "raise"
    assert ev.amount == amount


def test_action_response_probe_requires_message() -> None:
    """A probe kind must carry a message."""
    with pytest.raises(ValidationError):
        ActionResponse(
            hand_id="s1h001",
            seat="Seat3",
            kind="probe",
            action=None,
            message=None,
            tokens=0,
            latency_ms=0,
            cost_usd=0.0,
            model_id="stub:random",
            prompt_hash="sha256:abcd",
        )


def test_parse_event_roundtrip() -> None:
    """Every event type survives JSON round-trip via parse_event."""
    start = SessionStart(
        session_id=1,
        hand_cap=150,
        small_blind=10,
        big_blind=20,
        ante=0,
        deal_pack_seed=1001,
    )
    blob = start.model_dump_json()
    parsed = parse_event(json.loads(blob))
    assert isinstance(parsed, SessionStart)
    assert parsed.session_id == 1


def test_hand_start_and_end_balance() -> None:
    hs = HandStart(
        hand_id="s1h001",
        button_seat=1,
        stacks={"Seat1": 20000, "Seat2": 20000},
        cards_hash="sha256:" + "0" * 64,
        chat_budgets_remaining={"Seat1": 400, "Seat2": 400},
    )
    he = HandEnd(
        hand_id="s1h001",
        stack_deltas={"Seat1": -100, "Seat2": 100},
        elapsed_s=1.2,
        total_cost_usd=0.0,
    )
    assert hs.hand_id == he.hand_id
    assert sum(he.stack_deltas.values()) == 0


def test_event_discriminator_union() -> None:
    raw = {
        "type": "tournament_start",
        "tournament_id": "t1",
        "schema_version": "1.0",
        "holdembench_version": "0.1.0",
        "pokerkit_version": "0.7.3",
        "git_sha": "deadbeef",
        "seat_assignments": {"Seat1": "stub:random"},
        "master_seed": 42,
        "anonymization_salt": "cafebabe",
        "canary_uuid": "11111111-1111-1111-1111-111111111111",
    }
    parsed = parse_event(raw)
    assert isinstance(parsed, TournamentStart)
