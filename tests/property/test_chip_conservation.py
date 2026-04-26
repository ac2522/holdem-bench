"""Chip conservation invariant.

For any tournament played with stub agents (no LLM calls), every hand's
``stack_deltas`` must sum to zero, and the tournament's final chip totals
must sum to the starting bank.  Anything else means chips are leaking
somewhere — a fatal bug for chip-EV-based scoring.

This file uses *stub* agents only — it makes no network calls, runs in
~seconds, and is the safety net for the runner's pokerkit interaction.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from holdembench.baselines.random_agent import RandomAgent
from holdembench.baselines.tight_passive import TightPassiveAgent
from holdembench.events.log import EventLog
from holdembench.harness.runner import TournamentConfig, run_tournament

pytestmark = pytest.mark.asyncio

_STARTING_STACK = 1000
_BIG_BLIND = 20
_SMALL_BLIND = 10


def _starting_bank(seat_count: int) -> int:
    return seat_count * _STARTING_STACK


async def _run_with_stub(
    *,
    seat_count: int,
    hand_cap: int,
    master_seed: int,
    tmp_path: Path,
    model_id: str,
) -> Path:
    """Run a stub-only tournament and return the events.jsonl path."""
    cfg = TournamentConfig(
        tournament_id=f"chipcon-{seat_count}-{master_seed}",
        seats={f"Seat{i}": model_id for i in range(1, seat_count + 1)},
        small_blind=_SMALL_BLIND,
        big_blind=_BIG_BLIND,
        ante=0,
        starting_stack=_STARTING_STACK,
        hand_cap=hand_cap,
        session_count=1,
        master_seed=master_seed,
        results_dir=tmp_path,
    )
    agents_map = {
        "stub:tight_passive": TightPassiveAgent(),
        "stub:random": RandomAgent(),
    }
    agents = {model_id: agents_map[model_id]}
    result = await run_tournament(cfg, agents)
    return result.log_path


def _assert_per_hand_conservation(log_path: Path) -> None:
    """Every hand_end event must have stack_deltas summing to zero."""
    for event in EventLog.replay(log_path):
        if event.type != "hand_end":
            continue
        delta_sum = sum(event.stack_deltas.values())  # type: ignore[attr-defined]
        assert delta_sum == 0, (
            f"hand {event.hand_id} stack_deltas summed to {delta_sum}, "  # type: ignore[attr-defined]
            f"not 0: {event.stack_deltas}"  # type: ignore[attr-defined]
        )


def _assert_tournament_conservation(log_path: Path, seat_count: int) -> None:
    """The tournament_end final_chip_totals must sum to the starting bank."""
    expected = _starting_bank(seat_count)
    for event in EventLog.replay(log_path):
        if event.type != "tournament_end":
            continue
        actual = sum(event.final_chip_totals.values())  # type: ignore[attr-defined]
        assert actual == expected, (
            f"tournament_end final_chip_totals summed to {actual}, "
            f"expected {expected}: {event.final_chip_totals}"  # type: ignore[attr-defined]
        )


@pytest.mark.parametrize("seat_count", [2, 3, 6, 9])
async def test_chips_conserved_tight_passive(
    seat_count: int, tmp_path: Path
) -> None:
    """Tight-passive baseline: rarely all-in, mostly fold/check/call."""
    log_path = await _run_with_stub(
        seat_count=seat_count,
        hand_cap=10,
        master_seed=11,
        tmp_path=tmp_path,
        model_id="stub:tight_passive",
    )
    _assert_per_hand_conservation(log_path)
    _assert_tournament_conservation(log_path, seat_count)


@pytest.mark.parametrize("seat_count", [2, 3, 6, 9])
@pytest.mark.parametrize("seed", [1, 7, 42, 2026])
async def test_chips_conserved_random_agent(
    seat_count: int, seed: int, tmp_path: Path
) -> None:
    """Random agent: aggressive raise patterns, multi-way pots, all-ins."""
    log_path = await _run_with_stub(
        seat_count=seat_count,
        hand_cap=10,
        master_seed=seed,
        tmp_path=tmp_path,
        model_id="stub:random",
    )
    _assert_per_hand_conservation(log_path)
    _assert_tournament_conservation(log_path, seat_count)


async def test_chips_conserved_long_run(tmp_path: Path) -> None:
    """30-hand 6-seat run — closest to the smoke that exhibited the bug."""
    log_path = await _run_with_stub(
        seat_count=6,
        hand_cap=30,
        master_seed=2026,
        tmp_path=tmp_path,
        model_id="stub:tight_passive",
    )
    _assert_per_hand_conservation(log_path)
    _assert_tournament_conservation(log_path, 6)
