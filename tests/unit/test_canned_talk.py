"""Tests for CannedTalkAgent."""
from __future__ import annotations

from holdembench.agents.base import DecisionContext
from holdembench.baselines.canned_talk import CannedTalkAgent

_NUM_MESSAGES_TO_TEST = 20


def _ctx(hole: tuple[str, ...], legal: list[str]) -> DecisionContext:
    return DecisionContext(
        seat="Seat1",
        hand_id="s1h001",
        street="preflop",
        legal=legal,  # type: ignore[arg-type]
        stacks={"Seat1": 200},
        board=(),
        hole=hole,
        budget_remaining=400,
        is_probe_reply=False,
        deadline_s=60.0,
    )


async def test_canned_talk_emits_a_message() -> None:
    agent = CannedTalkAgent(seed=1)
    ctx = _ctx(("As", "Ah"), ["fold", "raise"])
    d = await agent.decide(ctx)
    assert d.message is not None
    assert len(d.message) > 0


async def test_canned_talk_messages_drawn_from_pool() -> None:
    agent = CannedTalkAgent(seed=1)
    ctx = _ctx(("As", "Ah"), ["fold", "raise"])
    messages = set()
    for _ in range(_NUM_MESSAGES_TO_TEST):
        d = await agent.decide(ctx)
        if d.message:
            messages.add(d.message)
    # Deterministic seed — should produce a small pool over _NUM_MESSAGES_TO_TEST hands
    assert 1 < len(messages) <= _NUM_MESSAGES_TO_TEST


async def test_canned_talk_actions_match_gto() -> None:
    agent = CannedTalkAgent(seed=1)
    ctx = _ctx(("2c", "7d"), ["fold", "raise"])
    d = await agent.decide(ctx)
    assert d.action == "fold"
