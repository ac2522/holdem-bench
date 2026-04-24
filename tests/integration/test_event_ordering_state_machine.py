"""P1-H: event grammar state machine.

The event log must follow:

    tournament_start
      (session_start
        (hand_start deal* (action_request action_response)* community_deal*
         showdown? hand_end)+
       session_end)+
    tournament_end

with auxiliary events (validator_rejection, auto_fold, budget_circuit_break,
probe_response_request) allowed between hand_start and hand_end.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from holdembench.baselines.random_agent import RandomAgent
from holdembench.events.log import EventLog
from holdembench.harness.runner import TournamentConfig, run_tournament

pytestmark = pytest.mark.asyncio


_IN_HAND_EVENTS = {
    "deal",
    "community_deal",
    "action_request",
    "action_response",
    "validator_rejection",
    "auto_fold",
    "budget_circuit_break",
    "probe_response_request",
}


def _validate_order(events: list) -> None:  # noqa: PLR0912
    # State machine is inherently branchy (one branch per state × legal transition);
    # splitting into helpers would obscure the grammar it encodes.
    state = "init"
    for e in events:
        t = e.type
        if state == "init":
            assert t == "tournament_start", f"expected tournament_start, got {t}"
            state = "tournament"
        elif state == "tournament":
            if t == "session_start":
                state = "session"
            elif t == "tournament_end":
                state = "done"
            else:
                raise AssertionError(f"unexpected {t} in tournament state")
        elif state == "session":
            if t == "hand_start":
                state = "hand"
            elif t == "session_end":
                state = "tournament"
            else:
                raise AssertionError(f"unexpected {t} in session state")
        elif state == "hand":
            if t in _IN_HAND_EVENTS:
                pass
            elif t == "showdown":
                state = "showdown"
            elif t == "hand_end":
                state = "session"
            else:
                raise AssertionError(f"unexpected {t} in hand state")
        elif state == "showdown":
            if t == "hand_end":
                state = "session"
            else:
                raise AssertionError(f"unexpected {t} after showdown")
        elif state == "done":
            raise AssertionError(f"events after tournament_end: {t}")
    assert state == "done", f"tournament not terminated, ended in {state}"


async def test_event_order_follows_grammar(tmp_path: Path) -> None:
    cfg = TournamentConfig(
        tournament_id="torder",
        seats={f"Seat{i}": "stub:random" for i in range(1, 5)},
        small_blind=10,
        big_blind=20,
        ante=0,
        starting_stack=1000,
        hand_cap=30,
        session_count=1,
        master_seed=5,
        results_dir=tmp_path,
    )
    agents = {"stub:random": RandomAgent(seed=1)}
    result = await run_tournament(cfg, agents)
    _validate_order(list(EventLog.replay(result.log_path)))


async def test_multi_session_order_still_grammatical(tmp_path: Path) -> None:
    cfg = TournamentConfig(
        tournament_id="tmulti",
        seats={f"Seat{i}": "stub:random" for i in range(1, 4)},
        small_blind=10,
        big_blind=20,
        ante=0,
        starting_stack=1000,
        hand_cap=5,
        session_count=3,
        master_seed=11,
        results_dir=tmp_path,
    )
    agents = {"stub:random": RandomAgent(seed=1)}
    result = await run_tournament(cfg, agents)
    _validate_order(list(EventLog.replay(result.log_path)))
