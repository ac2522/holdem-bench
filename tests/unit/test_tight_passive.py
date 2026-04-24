"""Tests for TightPassiveAgent."""

from __future__ import annotations

import pytest

from holdembench.agents.base import DecisionContext
from holdembench.baselines.tight_passive import TightPassiveAgent
from holdembench.types import ActionName


def _ctx(hole: tuple[str, ...], legal: tuple[ActionName, ...]) -> DecisionContext:
    return DecisionContext(
        seat="Seat1",
        hand_id="s1h001",
        street="preflop",
        legal=legal,
        stacks={"Seat1": 1000},
        board=(),
        hole=hole,
        budget_remaining=400,
        is_probe_reply=False,
        deadline_s=60.0,
    )


@pytest.mark.asyncio
async def test_tight_passive_folds_weak_hands_when_facing_bet() -> None:
    agent = TightPassiveAgent()
    ctx = _ctx(("2c", "7d"), ("fold", "call", "raise"))
    d = await agent.decide(ctx)
    assert d.action == "fold"


@pytest.mark.asyncio
async def test_tight_passive_calls_medium_hands() -> None:
    agent = TightPassiveAgent()
    ctx = _ctx(("Js", "Ts"), ("fold", "call", "raise"))
    d = await agent.decide(ctx)
    assert d.action == "call"


@pytest.mark.asyncio
async def test_tight_passive_raises_premium_hands() -> None:
    agent = TightPassiveAgent()
    ctx = _ctx(("As", "Ah"), ("fold", "call", "raise"))
    d = await agent.decide(ctx)
    assert d.action == "raise"


@pytest.mark.asyncio
async def test_tight_passive_checks_when_free() -> None:
    agent = TightPassiveAgent()
    ctx = _ctx(("2c", "7d"), ("check", "raise"))
    d = await agent.decide(ctx)
    assert d.action == "check"


@pytest.mark.asyncio
async def test_tight_passive_never_chats() -> None:
    agent = TightPassiveAgent()
    ctx = _ctx(("As", "Ks"), ("fold", "call", "raise"))
    d = await agent.decide(ctx)
    assert d.message is None
