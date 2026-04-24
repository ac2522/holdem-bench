"""Runner must copy adapter usage + cost into ActionResponse events."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from holdembench.agents.base import DecisionContext, Pricing
from holdembench.baselines.random_agent import RandomAgent
from holdembench.engine.validator import RawDecision
from holdembench.events.log import EventLog
from holdembench.harness.runner import TournamentConfig, run_tournament

pytestmark = pytest.mark.asyncio

_FAKE_AGENT_LATENCY_MS = 12


@dataclass
class _FakeUsage:
    input_tokens: int = 300
    output_tokens: int = 10
    cache_read_tokens: int = 200
    cache_write_tokens: int = 0
    thinking_tokens: int = 0


class _FakeBudgetAgent:
    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self.pricing = Pricing(input_per_mtok=1.0, output_per_mtok=5.0, cache_read_per_mtok=0.1)
        self._usage = _FakeUsage()

    @property
    def last_usage(self) -> _FakeUsage:
        return self._usage

    @property
    def last_cost_usd(self) -> float:
        u = self._usage
        return self.pricing.cost_usd(
            input_tokens=u.input_tokens,
            output_tokens=u.output_tokens,
            cache_read_tokens=u.cache_read_tokens,
            cache_write_tokens=u.cache_write_tokens,
        )

    @property
    def last_thinking(self) -> str | None:
        return "thought for a moment"

    @property
    def last_prompt_hash(self) -> str:
        return "sha256:abc123"

    @property
    def last_latency_ms(self) -> int:
        return _FAKE_AGENT_LATENCY_MS

    async def decide(self, ctx: DecisionContext) -> RawDecision:  # noqa: ARG002
        return RawDecision(kind="action", action="fold")


async def test_action_response_copies_usage_and_cost(tmp_path: Path) -> None:
    cfg = TournamentConfig(
        tournament_id="t",
        seats={"Seat1": "m:a", "Seat2": "m:b"},
        small_blind=10,
        big_blind=20,
        ante=0,
        starting_stack=1000,
        hand_cap=2,
        session_count=1,
        master_seed=1,
        results_dir=tmp_path,
    )
    agents = {"m:a": _FakeBudgetAgent("m:a"), "m:b": _FakeBudgetAgent("m:b")}
    result = await run_tournament(cfg, agents)
    events = list(EventLog.replay(result.log_path))
    action_responses = [e for e in events if e.type == "action_response"]
    assert action_responses
    one = action_responses[0]
    assert one.tokens > 0
    assert one.cost_usd > 0
    assert one.thinking == "thought for a moment"
    assert one.prompt_hash == "sha256:abc123"
    assert one.latency_ms == _FAKE_AGENT_LATENCY_MS


async def test_stub_baseline_still_emits_zero_cost(tmp_path: Path) -> None:
    """Stub agents without cost accounting still work — default 0 values."""
    cfg = TournamentConfig(
        tournament_id="t2",
        seats={"Seat1": "stub:random", "Seat2": "stub:random"},
        small_blind=10,
        big_blind=20,
        ante=0,
        starting_stack=1000,
        hand_cap=2,
        session_count=1,
        master_seed=1,
        results_dir=tmp_path,
    )
    agents = {"stub:random": RandomAgent(seed=1)}
    result = await run_tournament(cfg, agents)
    events = list(EventLog.replay(result.log_path))
    action_responses = [e for e in events if e.type == "action_response"]
    assert action_responses
    # Stub agents don't expose last_usage — cost / tokens / prompt_hash default to zero.
    for ar in action_responses:
        assert ar.cost_usd == 0.0
        assert ar.tokens == 0
        assert ar.prompt_hash == ""
