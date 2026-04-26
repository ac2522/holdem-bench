"""XAI / OpenRouter are thin OpenAI subclasses — smoke-check they decide()."""

from __future__ import annotations

from typing import Any

import pytest

from holdembench.agents.base import DecisionContext
from holdembench.agents.openrouter import OpenRouterAgent
from holdembench.agents.prompt import SessionContext, TournamentContext
from holdembench.agents.xai import XAIAgent


class _FakeOpenAI:
    def __init__(self, text: str) -> None:
        self._text = text
        self.last_model: str | None = None
        self.chat = self._Chat(self)

    class _Chat:
        def __init__(self, outer: _FakeOpenAI) -> None:
            self.completions = _FakeOpenAI._Completions(outer)

    class _Completions:
        def __init__(self, outer: _FakeOpenAI) -> None:
            self._outer = outer

        async def create(self, **kwargs: Any) -> object:
            self._outer.last_model = kwargs["model"]
            return _resp(self._outer._text)


def _resp(text: str) -> object:
    class Details:
        cached_tokens = 0

    class Usage:
        prompt_tokens = 100
        completion_tokens = 10
        prompt_tokens_details = Details()

    class Msg:
        def __init__(self, content: str) -> None:
            self.content = content

    class Choice:
        def __init__(self, text: str) -> None:
            self.message = Msg(text)

    class Resp:
        def __init__(self, text: str) -> None:
            self.choices = [Choice(text)]
            self.usage = Usage()

    return Resp(text)


def _ctx() -> DecisionContext:
    return DecisionContext(
        seat="Seat1",
        hand_id="s1h001",
        street="preflop",
        legal=("fold", "call"),
        stacks={"Seat1": 1000},
        board=(),
        hole=("As", "Kd"),
        budget_remaining=400,
        is_probe_reply=False,
        deadline_s=60.0,
    )


def _tctx() -> TournamentContext:
    return TournamentContext(tournament_id="t", seat="Seat1", seat_count=2)


def _sctx() -> SessionContext:
    return SessionContext(
        session_id=1,
        small_blind=10,
        big_blind=20,
        ante=0,
        starting_stack_bb=100,
        orbit_budget_tokens=400,
    )


@pytest.mark.asyncio
async def test_xai_agent_inherits_openai_transport() -> None:
    client = _FakeOpenAI('{"kind": "action", "action": "fold"}')
    agent = XAIAgent(model_id="xai:grok-4", client=client)
    agent.set_context(tournament=_tctx(), session=_sctx())
    raw = await agent.decide(_ctx())
    assert raw.action == "fold"
    assert client.last_model == "grok-4"


@pytest.mark.asyncio
async def test_openrouter_agent_preserves_vendor_slash_name() -> None:
    client = _FakeOpenAI('{"kind": "action", "action": "call"}')
    agent = OpenRouterAgent(
        model_id="openrouter:deepseek/deepseek-chat-v3.1",
        client=client,
    )
    agent.set_context(tournament=_tctx(), session=_sctx())
    raw = await agent.decide(_ctx())
    assert raw.action == "call"
    assert client.last_model == "deepseek/deepseek-chat-v3.1"
