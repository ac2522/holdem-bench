"""Match-based tournament lifecycle.

A "match" is one full N-hand game with rising blinds.  The tournament
runs ``match_count`` matches; between each match every seat resets to
``starting_stack``, the chat protocol resets, the action log resets,
blinds reset to level 1.  Busted seats sit out the remainder of the
current match but their seat slot is preserved so they can return
fresh in the next match.

Final score for the tournament = sum of per-match final stacks per seat.
Highest sum wins.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from holdembench.baselines.tight_passive import TightPassiveAgent
from holdembench.events.log import EventLog
from holdembench.harness.runner import BlindLevel, TournamentConfig, run_tournament

pytestmark = pytest.mark.asyncio

_STARTING_STACK = 1000
_MATCH_COUNT = 3
_HANDS_PER_MATCH = 8
_SEAT_COUNT = 4
_LEVEL1_SB = 10
_LEVEL1_BB = 20
_LEVEL2_SB = 20
_LEVEL2_BB = 40
_LEVEL2_START_HAND = 5


def _config(tmp_path: Path) -> TournamentConfig:
    return TournamentConfig(
        tournament_id="match-life",
        seats={f"Seat{i}": "stub:tight_passive" for i in range(1, _SEAT_COUNT + 1)},
        small_blind=10,
        big_blind=20,
        ante=0,
        starting_stack=_STARTING_STACK,
        hand_cap=_HANDS_PER_MATCH,
        session_count=_MATCH_COUNT,
        master_seed=11,
        results_dir=tmp_path,
        # Rising-stakes schedule: blinds double every 4 hands.
        blind_levels=(
            BlindLevel(start_hand=1, small_blind=_LEVEL1_SB, big_blind=_LEVEL1_BB),
            BlindLevel(
                start_hand=_LEVEL2_START_HAND,
                small_blind=_LEVEL2_SB,
                big_blind=_LEVEL2_BB,
            ),
        ),
    )


async def _run(tmp_path: Path) -> Path:
    cfg = _config(tmp_path)
    result = await run_tournament(cfg, {"stub:tight_passive": TightPassiveAgent()})
    return result.log_path


def _hand_starts_by_match(log_path: Path) -> dict[int, list[dict[str, int]]]:
    """Group hand_start events by match (session_id)."""
    out: dict[int, list[dict[str, int]]] = {}
    current_session = 0
    for event in EventLog.replay(log_path):
        if event.type == "session_start":
            current_session = event.session_id  # type: ignore[attr-defined]
            out[current_session] = []
        elif event.type == "hand_start":
            out[current_session].append(event.stacks)  # type: ignore[attr-defined]
    return out


async def test_stacks_reset_between_matches(tmp_path: Path) -> None:
    """First hand of every match starts with all seats at starting_stack."""
    log_path = await _run(tmp_path)
    by_match = _hand_starts_by_match(log_path)
    assert len(by_match) == _MATCH_COUNT
    for session_id, hand_stacks in by_match.items():
        assert hand_stacks, f"session {session_id} had no hands"
        first = hand_stacks[0]
        for seat, chips in first.items():
            assert chips == _STARTING_STACK, (
                f"session {session_id} hand 1 had {seat}={chips}, "
                f"expected reset to {_STARTING_STACK}"
            )


async def test_blind_levels_rise_within_match(tmp_path: Path) -> None:
    """Hand 1 uses level-1 blinds (10/20); hand 5 uses level-2 blinds (20/40)."""
    log_path = await _run(tmp_path)
    levels: list[tuple[int, int, int]] = []  # (hand_index_within_match, sb, bb)
    current_session = 0
    hand_in_match = 0
    for event in EventLog.replay(log_path):
        if event.type == "session_start":
            current_session = event.session_id  # type: ignore[attr-defined]
            hand_in_match = 0
        elif event.type == "hand_start" and current_session == 1:
            hand_in_match += 1
            levels.append(
                (hand_in_match, event.small_blind, event.big_blind)  # type: ignore[attr-defined]
            )
    assert any(
        h == 1 and sb == _LEVEL1_SB and bb == _LEVEL1_BB for h, sb, bb in levels
    ), f"hand 1 of match 1 should use level-1 blinds; got: {levels}"
    assert any(
        h == _LEVEL2_START_HAND and sb == _LEVEL2_SB and bb == _LEVEL2_BB
        for h, sb, bb in levels
    ), f"hand {_LEVEL2_START_HAND} of match 1 should use level-2 blinds; got: {levels}"


async def test_final_score_is_sum_across_matches(tmp_path: Path) -> None:
    """tournament_end.final_score per seat equals sum of session-end stacks."""
    log_path = await _run(tmp_path)
    per_session_finals: dict[int, dict[str, int]] = {}
    final_score: dict[str, int] | None = None
    for event in EventLog.replay(log_path):
        if event.type == "session_end":
            per_session_finals[event.session_id] = event.final_stacks  # type: ignore[attr-defined]
        elif event.type == "tournament_end":
            final_score = getattr(event, "final_score", None)
    assert final_score is not None, "tournament_end missing `final_score` field"
    expected = {seat: 0 for seat in per_session_finals[1]}
    for stacks in per_session_finals.values():
        for seat, chips in stacks.items():
            expected[seat] += chips
    assert final_score == expected, (
        f"final_score {final_score} != sum-of-session-finals {expected}"
    )


async def test_chips_conserved_per_match_even_with_busts(tmp_path: Path) -> None:
    """For each match, sum of session_end final_stacks == seat_count * starting_stack."""
    log_path = await _run(tmp_path)
    bank = _SEAT_COUNT * _STARTING_STACK
    saw_session_end = False
    for event in EventLog.replay(log_path):
        if event.type == "session_end":
            saw_session_end = True
            total = sum(event.final_stacks.values())  # type: ignore[attr-defined]
            assert total == bank, (
                f"session {event.session_id} final stacks "  # type: ignore[attr-defined]
                f"sum to {total}, expected {bank}"
            )
    assert saw_session_end, "no session_end events emitted"
