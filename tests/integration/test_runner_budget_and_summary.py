"""Budget circuit breaker + per-model cost summary on TournamentResult."""

from __future__ import annotations

from pathlib import Path

import pytest

from holdembench.agents.base import DecisionContext, Pricing
from holdembench.engine.validator import RawDecision
from holdembench.events.log import EventLog
from holdembench.harness.runner import TournamentConfig, run_tournament

pytestmark = pytest.mark.asyncio


class _ExpensiveAgent:
    """Fake agent declaring enormous cost per call — trips the breaker fast."""

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self.pricing = Pricing(input_per_mtok=1000.0, output_per_mtok=5000.0)

    class _U:
        input_tokens = 100_000
        output_tokens = 10_000
        cache_read_tokens = 0
        cache_write_tokens = 0

    @property
    def last_usage(self) -> _U:
        return self._U()

    @property
    def last_cost_usd(self) -> float:
        return self.pricing.cost_usd(input_tokens=100_000, output_tokens=10_000)

    last_thinking: str | None = None
    last_prompt_hash: str = "sha256:z"
    last_latency_ms: int = 0

    async def decide(self, ctx: DecisionContext) -> RawDecision:  # noqa: ARG002
        return RawDecision(kind="action", action="call")


class _CheapStub:
    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self.pricing = Pricing(input_per_mtok=0.0, output_per_mtok=0.0)

    last_usage = None
    last_cost_usd: float = 0.0
    last_thinking: str | None = None
    last_prompt_hash: str = ""
    last_latency_ms: int = 0

    async def decide(self, ctx: DecisionContext) -> RawDecision:  # noqa: ARG002
        return RawDecision(kind="action", action="fold")


async def test_budget_circuit_breaker_fires(tmp_path: Path) -> None:
    cfg = TournamentConfig(
        tournament_id="tbb",
        seats={"Seat1": "m:exp", "Seat2": "m:cheap", "Seat3": "m:cheap"},
        small_blind=10,
        big_blind=20,
        ante=0,
        starting_stack=1000,
        hand_cap=3,
        session_count=1,
        master_seed=1,
        results_dir=tmp_path,
        budget_ceilings_usd={"m:exp": 0.01, "m:cheap": 100.0},
    )
    agents = {
        "m:exp": _ExpensiveAgent("m:exp"),
        "m:cheap": _CheapStub("m:cheap"),
    }
    result = await run_tournament(cfg, agents)
    events = list(EventLog.replay(result.log_path))
    breaks = [e for e in events if e.type == "budget_circuit_break"]
    folds = [e for e in events if e.type == "auto_fold" and e.reason == "budget_circuit_break"]
    # Seat1 (the expensive agent) should trip the breaker on its first action,
    # because cost-per-call (≈0.15 USD) far exceeds 2× the 0.01 ceiling.
    assert len(breaks) == 1
    assert breaks[0].seat == "Seat1"
    assert folds


async def test_per_model_summary_in_tournament_result(tmp_path: Path) -> None:
    class _Agent:
        def __init__(self, mid: str, cost: float) -> None:
            self.model_id = mid
            self.pricing = Pricing(input_per_mtok=1.0, output_per_mtok=5.0)
            self._cost = cost

        class _U:
            input_tokens = 100
            output_tokens = 20
            cache_read_tokens = 50
            cache_write_tokens = 0

        @property
        def last_usage(self) -> _U:
            return self._U()

        @property
        def last_cost_usd(self) -> float:
            return self._cost

        last_thinking: str | None = None
        last_prompt_hash: str = "sha256:abc"
        last_latency_ms: int = 12

        async def decide(self, ctx: DecisionContext) -> RawDecision:  # noqa: ARG002
            return RawDecision(kind="action", action="fold")

    # Use 4 seats so both models definitely see voluntary action across the hand
    # rotation — some 2-seat configurations auto-resolve via blinds-only play.
    cfg = TournamentConfig(
        tournament_id="tsum",
        seats={"Seat1": "m:a", "Seat2": "m:a", "Seat3": "m:b", "Seat4": "m:b"},
        small_blind=10,
        big_blind=20,
        ante=0,
        starting_stack=1000,
        hand_cap=4,
        session_count=1,
        master_seed=2,
        results_dir=tmp_path,
    )
    agents = {"m:a": _Agent("m:a", 0.01), "m:b": _Agent("m:b", 0.02)}
    result = await run_tournament(cfg, agents)
    assert result.per_model_cost is not None
    assert "m:a" in result.per_model_cost
    assert "m:b" in result.per_model_cost
    a = result.per_model_cost["m:a"]
    assert a["input_tokens"] > 0
    assert a["usd_total"] > 0
    assert result.total_cost_usd > 0
    # m:b has higher per-call cost so its total should exceed m:a's
    assert (
        result.per_model_cost["m:b"]["usd_total"]
        > result.per_model_cost["m:a"]["usd_total"]
    )
