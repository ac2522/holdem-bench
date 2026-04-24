"""P1-F: canonical action log token budget invariant.

For any N-hand session, the naive canonical action log (one line per action)
must fit under a cache-breakpoint token budget.  This guards the per-session
cache block from overflowing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from holdembench.baselines.tight_passive import TightPassiveAgent
from holdembench.chat.tokenizer import count_tokens
from holdembench.events.log import EventLog
from holdembench.harness.runner import TournamentConfig, run_tournament

pytestmark = pytest.mark.asyncio

CANONICAL_LOG_TOKEN_BUDGET = 8000


async def test_canonical_action_log_fits_in_budget(tmp_path: Path) -> None:
    # 6 seats × 30 hands ≈ a typical per-session body when the session is
    # padded out end-to-end.  Scales linearly, so any cap < hand_cap*seat_count*~6
    # well below the 8K budget.
    cfg = TournamentConfig(
        tournament_id="ttok",
        seats={f"Seat{i}": "stub:tight_passive" for i in range(1, 7)},
        small_blind=10,
        big_blind=20,
        ante=0,
        starting_stack=1000,
        hand_cap=30,
        session_count=1,
        master_seed=3,
        results_dir=tmp_path,
    )
    # TightPassive only folds/calls/checks — deterministic, avoids the
    # RandomAgent raise-legality corner case that's tracked as a separate
    # Phase 1 follow-up.
    agents = {"stub:tight_passive": TightPassiveAgent()}
    result = await run_tournament(cfg, agents)
    events = list(EventLog.replay(result.log_path))
    log_lines: list[str] = []
    for e in events:
        if e.type != "action_response":
            continue
        if e.kind != "action":
            continue
        amount_suffix = f" {e.amount}" if e.amount else ""
        log_lines.append(f"{e.hand_id} {e.seat}:{e.action}{amount_suffix}")
    log_text = "\n".join(log_lines)
    tokens = count_tokens(log_text)
    assert tokens < CANONICAL_LOG_TOKEN_BUDGET, (
        f"canonical log {tokens} tokens exceeds budget {CANONICAL_LOG_TOKEN_BUDGET}"
    )
