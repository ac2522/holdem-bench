"""Tests for the RandomAgent baseline."""

from __future__ import annotations

from holdembench.agents.base import DecisionContext
from holdembench.baselines.random_agent import RandomAgent
from holdembench.types import ActionName


def _ctx(
    legal: tuple[ActionName, ...], *, min_raise_to: int | None = None
) -> DecisionContext:
    return DecisionContext(
        seat="Seat1",
        hand_id="s1h001",
        street="preflop",
        legal=legal,
        stacks={"Seat1": 1000, "Seat2": 1000},
        board=(),
        hole=("Ah", "Ks"),
        budget_remaining=400,
        is_probe_reply=False,
        deadline_s=60.0,
        min_raise_to=min_raise_to,
    )


async def test_random_agent_only_returns_legal_actions() -> None:
    agent = RandomAgent(seed=42)
    ctx = _ctx(("fold", "call"))
    for _ in range(50):
        d = await agent.decide(ctx)
        assert d.kind == "action"
        assert d.action in {"fold", "call"}


async def test_random_agent_is_deterministic_with_seed() -> None:
    a = RandomAgent(seed=1)
    b = RandomAgent(seed=1)
    ctx = _ctx(("fold", "call", "raise"))
    out_a = [(await a.decide(ctx)).action for _ in range(20)]
    out_b = [(await b.decide(ctx)).action for _ in range(20)]
    assert out_a == out_b


async def test_random_agent_never_chats() -> None:
    agent = RandomAgent(seed=1)
    ctx = _ctx(("fold", "call"))
    for _ in range(50):
        d = await agent.decide(ctx)
        assert d.message is None


async def test_random_agent_raises_to_min_raise_to_from_ctx() -> None:
    """RandomAgent reads ctx.min_raise_to (always set by the runner when
    raise is in legal) instead of hardcoding 2× a constructor BB.  This
    keeps the stub legal under rising blinds."""
    expected_min_raise_to = 100
    agent = RandomAgent(seed=1)
    ctx = _ctx(("raise",), min_raise_to=expected_min_raise_to)
    d = await agent.decide(ctx)
    assert d.action == "raise"
    assert d.amount == expected_min_raise_to
