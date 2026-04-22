"""Tests for chip EV + mbb/100 computation from event log."""
from __future__ import annotations

from pathlib import Path

from holdembench.events.log import EventLog
from holdembench.events.schema import (
    HandEnd,
    SessionStart,
    TournamentEnd,
    TournamentStart,
)
from holdembench.scoring.chip_ev import compute_chip_ev, mbb_per_100


def _minimal_log(tmp: Path) -> Path:
    p = tmp / "events.jsonl"
    with EventLog(p) as log:
        log.emit(
            TournamentStart(
                tournament_id="t", schema_version="1.0",
                holdembench_version="0.1", pokerkit_version="0.7",
                git_sha="x", seat_assignments={"Seat1": "a", "Seat2": "b"},
                master_seed=0, anonymization_salt="s", canary_uuid="u",
            )
        )
        log.emit(
            SessionStart(session_id=1, hand_cap=2, small_blind=10, big_blind=20,
                         ante=0, deal_pack_seed=1)
        )
        log.emit(HandEnd(hand_id="s1h001", stack_deltas={"Seat1": 100, "Seat2": -100},
                         elapsed_s=0.1, total_cost_usd=0.0))
        log.emit(HandEnd(hand_id="s1h002", stack_deltas={"Seat1": -50, "Seat2": 50},
                         elapsed_s=0.1, total_cost_usd=0.0))
        log.emit(
            TournamentEnd(
                tournament_id="t", final_chip_totals={"Seat1": 1050, "Seat2": 950},
                winner_seat="Seat1", winner_model="a",
                total_cost_usd=0.0, wall_clock_s=1.0,
            )
        )
    return p


def test_compute_chip_ev_sums_deltas(tmp_path: Path) -> None:
    p = _minimal_log(tmp_path)
    ev = compute_chip_ev(p)
    assert ev["Seat1"] == 50  # noqa: PLR2004
    assert ev["Seat2"] == -50  # noqa: PLR2004


def test_mbb_per_100_conversion() -> None:
    # 50 chip profit at BB=20 → 2.5 BB → 2500 mbb over 2 hands → 125000 mbb/100
    assert mbb_per_100(chip_delta=50, hands=2, big_blind=20) == 125000.0  # noqa: PLR2004


def test_zero_hands_returns_zero() -> None:
    assert mbb_per_100(chip_delta=0, hands=0, big_blind=20) == 0.0
