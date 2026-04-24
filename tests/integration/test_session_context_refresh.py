"""Runner must refresh each adapter's SessionContext at every session boundary.

Regression for a bug found in the Phase 1.1 mid-plan review: the CLI was
wiring one SessionContext(session_id=1) and leaving it pinned for every
subsequent session, so multi-session prompts misreported the session id.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from holdembench.agents.base import Agent, DecisionContext, Pricing
from holdembench.agents.prompt import SessionContext, TournamentContext
from holdembench.engine.validator import RawDecision
from holdembench.harness.runner import TournamentConfig, run_tournament

pytestmark = pytest.mark.asyncio


class _TrackingAgent(Agent):
    model_id = "test:session-tracker"
    pricing = Pricing(input_per_mtok=0.0, output_per_mtok=0.0)

    def __init__(self) -> None:
        self.session_ids_seen: list[int] = []
        self._session: SessionContext | None = None

    def set_context(
        self, *, tournament: TournamentContext, session: SessionContext
    ) -> None:
        _ = tournament
        self._session = session
        self.session_ids_seen.append(session.session_id)

    async def decide(self, ctx: DecisionContext) -> RawDecision:  # noqa: ARG002
        return RawDecision(kind="action", action="fold")


async def test_session_context_refreshed_each_session(tmp_path: Path) -> None:
    agent = _TrackingAgent()
    cfg = TournamentConfig(
        tournament_id="tsess",
        seats={f"Seat{i}": agent.model_id for i in range(1, 4)},
        small_blind=10,
        big_blind=20,
        ante=0,
        starting_stack=1000,
        hand_cap=2,
        session_count=3,
        master_seed=1,
        results_dir=tmp_path,
    )
    agents = {agent.model_id: agent}
    await run_tournament(cfg, agents)
    assert agent.session_ids_seen == [1, 2, 3]
