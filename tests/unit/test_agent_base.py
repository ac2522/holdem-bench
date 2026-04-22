"""Tests for Agent protocol + DecisionContext."""
from __future__ import annotations

import pytest

from holdembench.agents.base import Agent, DecisionContext, Pricing, RawDecision


class _FixedFoldAgent:
    model_id = "test:fold"
    pricing = Pricing(input_per_mtok=0.0, output_per_mtok=0.0, cache_read_per_mtok=0.0)

    async def decide(self, ctx: DecisionContext) -> RawDecision:
        return RawDecision(kind="action", action="fold")


def test_fixed_fold_agent_satisfies_protocol() -> None:
    agent: Agent = _FixedFoldAgent()
    assert agent.model_id == "test:fold"


def test_decision_context_fields() -> None:
    ctx = DecisionContext(
        seat="Seat1",
        hand_id="s1h001",
        street="preflop",
        legal=["fold", "call", "raise"],
        stacks={"Seat1": 1000, "Seat2": 1000},
        board=(),
        hole=("Ah", "Ks"),
        budget_remaining=400,
        is_probe_reply=False,
        deadline_s=60.0,
    )
    assert ctx.seat == "Seat1"
    assert ctx.hole == ("Ah", "Ks")
    assert ctx.is_probe_reply is False


@pytest.mark.asyncio
async def test_agent_decide_returns_rawdecision() -> None:
    """Any Agent instance returns a RawDecision from decide()."""
    a = _FixedFoldAgent()
    ctx = DecisionContext(
        seat="Seat1",
        hand_id="s1h001",
        street="preflop",
        legal=["fold"],
        stacks={"Seat1": 1000},
        board=(),
        hole=("Ah", "Ks"),
        budget_remaining=400,
        is_probe_reply=False,
        deadline_s=60.0,
    )
    d = await a.decide(ctx)
    assert d.kind == "action"
    assert d.action == "fold"


def test_pricing_cost_for_tokens() -> None:
    p = Pricing(input_per_mtok=10.0, output_per_mtok=30.0, cache_read_per_mtok=1.0)
    # 1k input + 0.5k output, no cache → 0.001*10 + 0.0005*30 = 0.025
    cost = p.cost_usd(input_tokens=1000, output_tokens=500, cache_read_tokens=0)
    assert cost == pytest.approx(0.025)
