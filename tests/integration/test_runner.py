"""Integration: stub-only tournament runs end-to-end."""

from __future__ import annotations

from pathlib import Path

import pytest

from holdembench.baselines import RandomAgent, TightPassiveAgent
from holdembench.events.log import EventLog
from holdembench.events.schema import HandEnd, TournamentEnd, TournamentStart
from holdembench.harness.runner import TournamentConfig, run_tournament


@pytest.mark.slow
async def test_stub_tournament_produces_complete_log(tmp_path: Path) -> None:
    cfg = TournamentConfig(
        tournament_id="t-smoke",
        seats={f"Seat{i}": "stub:random" if i % 2 else "stub:tight_passive" for i in range(1, 7)},
        small_blind=10,
        big_blind=20,
        ante=0,
        starting_stack=1000,
        hand_cap=20,
        session_count=1,
        master_seed=42,
        results_dir=tmp_path / "results",
    )
    agents = {
        "stub:random": RandomAgent(seed=1),
        "stub:tight_passive": TightPassiveAgent(),
    }
    out = await run_tournament(cfg, agents)
    events = list(EventLog.replay(out.log_path))
    assert isinstance(events[0], TournamentStart)
    assert isinstance(events[-1], TournamentEnd)
    # At least one HandEnd per hand in the cap (modulo hand-is-over-early edge cases)
    hand_ends = [e for e in events if isinstance(e, HandEnd)]
    assert len(hand_ends) <= cfg.hand_cap
    assert len(hand_ends) >= 1


@pytest.mark.slow
async def test_stub_tournament_is_deterministic(tmp_path: Path) -> None:
    """Same master_seed + same agent seeds -> byte-identical log."""
    cfg_kwargs = dict(
        tournament_id="t-det",
        seats={f"Seat{i}": "stub:random" for i in range(1, 5)},
        small_blind=10,
        big_blind=20,
        ante=0,
        starting_stack=1000,
        hand_cap=5,
        session_count=1,
        master_seed=7,
    )
    a = TournamentConfig(**cfg_kwargs, results_dir=tmp_path / "a")  # type: ignore[arg-type]
    b = TournamentConfig(**cfg_kwargs, results_dir=tmp_path / "b")  # type: ignore[arg-type]
    agents_a = {"stub:random": RandomAgent(seed=99)}
    agents_b = {"stub:random": RandomAgent(seed=99)}
    out_a = await run_tournament(a, agents_a)
    out_b = await run_tournament(b, agents_b)
    assert out_a.log_path.read_bytes() == out_b.log_path.read_bytes()
