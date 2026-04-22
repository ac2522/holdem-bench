"""Tests for GTOApproxAgent."""

from __future__ import annotations

from holdembench.agents.base import DecisionContext
from holdembench.baselines.gto_approx import GTOApproxAgent


def _ctx(hole: tuple[str, ...], stack_bb: int, legal: list[str]) -> DecisionContext:
    return DecisionContext(
        seat="Seat1",
        hand_id="s1h001",
        street="preflop",
        legal=legal,  # type: ignore[arg-type]
        stacks={"Seat1": stack_bb * 20},  # BB=20 assumed
        board=(),
        hole=hole,
        budget_remaining=400,
        is_probe_reply=False,
        deadline_s=60.0,
    )


async def test_shoves_aces_at_any_stack() -> None:
    agent = GTOApproxAgent()
    ctx = _ctx(("As", "Ah"), stack_bb=10, legal=["fold", "raise"])
    d = await agent.decide(ctx)
    assert d.action == "raise"
    assert d.amount == ctx.stacks["Seat1"]  # full shove


async def test_folds_72off_at_medium_stack() -> None:
    agent = GTOApproxAgent()
    ctx = _ctx(("7c", "2d"), stack_bb=20, legal=["fold", "call", "raise"])
    d = await agent.decide(ctx)
    assert d.action == "fold"


async def test_never_chats() -> None:
    agent = GTOApproxAgent()
    ctx = _ctx(("As", "Ah"), stack_bb=10, legal=["fold", "raise"])
    d = await agent.decide(ctx)
    assert d.message is None
