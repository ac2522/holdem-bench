"""HumanAgent — asyncio-Queue handoff + timeout behaviour."""

from __future__ import annotations

import asyncio

import pytest

from holdembench.agents.base import DecisionContext
from holdembench.agents.human import HumanAgent, HumanDecisionQueue


def _ctx() -> DecisionContext:
    return DecisionContext(
        seat="Seat9",
        hand_id="s1h001",
        street="preflop",
        legal=("fold", "call", "raise"),
        stacks={"Seat9": 1000},
        board=(),
        hole=("As", "Kd"),
        budget_remaining=400,
        is_probe_reply=False,
        deadline_s=120.0,
    )


@pytest.mark.asyncio
async def test_queue_handoff() -> None:
    q = HumanDecisionQueue()
    agent = HumanAgent(model_id="human:alex", queue=q)
    await q.submit('{"kind": "action", "action": "call"}')
    raw = await agent.decide(_ctx())
    assert raw.kind == "action"
    assert raw.action == "call"


@pytest.mark.asyncio
async def test_timeout_auto_folds() -> None:
    q = HumanDecisionQueue()
    agent = HumanAgent(model_id="human:alex", queue=q, timeout_s=0.05)
    raw = await agent.decide(_ctx())
    assert raw.kind == "action"
    assert raw.action == "fold"


@pytest.mark.asyncio
async def test_parse_error_auto_folds() -> None:
    q = HumanDecisionQueue()
    agent = HumanAgent(model_id="human:alex", queue=q, timeout_s=1.0)
    await q.submit("not json")
    raw = await agent.decide(_ctx())
    assert raw.kind == "action"
    assert raw.action == "fold"


@pytest.mark.asyncio
async def test_concurrent_submit_then_decide() -> None:
    q = HumanDecisionQueue()
    agent = HumanAgent(model_id="human:alex", queue=q, timeout_s=5.0)

    async def submit_later() -> None:
        await asyncio.sleep(0.02)
        await q.submit('{"kind": "action", "action": "check"}')

    async with asyncio.TaskGroup() as tg:
        _ = tg.create_task(submit_later())
        decide_task = tg.create_task(agent.decide(_ctx()))

    raw = decide_task.result()
    assert raw.action == "check"
