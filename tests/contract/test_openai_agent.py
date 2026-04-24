"""OpenAI adapter contract test — fake OpenAI-shaped client."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from holdembench.agents.base import DecisionContext
from holdembench.agents.openai import OpenAIAgent
from holdembench.agents.prompt import SessionContext, TournamentContext

_EXPECTED_PROMPT_TOKENS = 420
_EXPECTED_COMPLETION_TOKENS = 15
_EXPECTED_CACHED_TOKENS = 380


@dataclass
class _Spy:
    call_count: int = 0
    last_kwargs: dict[str, Any] | None = None


class _FakeOpenAI:
    def __init__(self, responses: list[str], spy: _Spy) -> None:
        self._responses = responses
        self._spy = spy
        self.chat = self._Chat(responses, spy)

    class _Chat:
        def __init__(self, responses: list[str], spy: _Spy) -> None:
            self.completions = _FakeOpenAI._Completions(responses, spy)

    class _Completions:
        def __init__(self, responses: list[str], spy: _Spy) -> None:
            self._responses = responses
            self._spy = spy

        async def create(self, **kwargs: Any) -> object:
            self._spy.call_count += 1
            self._spy.last_kwargs = kwargs
            text = self._responses[min(self._spy.call_count - 1, len(self._responses) - 1)]
            return _openai_response(text)


def _openai_response(text: str) -> object:
    class Details:
        cached_tokens = _EXPECTED_CACHED_TOKENS

    class Usage:
        prompt_tokens = _EXPECTED_PROMPT_TOKENS
        completion_tokens = _EXPECTED_COMPLETION_TOKENS
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


def _tctx() -> TournamentContext:
    return TournamentContext(tournament_id="t", seat="Seat1", seat_count=3)


def _sctx() -> SessionContext:
    return SessionContext(
        session_id=1,
        small_blind=10,
        big_blind=20,
        ante=0,
        starting_stack_bb=100,
        orbit_budget_tokens=400,
    )


def _ctx() -> DecisionContext:
    return DecisionContext(
        seat="Seat1",
        hand_id="s1h001",
        street="preflop",
        legal=("fold", "call", "raise"),
        stacks={"Seat1": 1000},
        board=(),
        hole=("As", "Kd"),
        budget_remaining=400,
        is_probe_reply=False,
        deadline_s=60.0,
    )


@pytest.mark.asyncio
async def test_openai_decide_returns_raw_decision() -> None:
    spy = _Spy()
    client = _FakeOpenAI(['{"kind": "action", "action": "call"}'], spy)
    agent = OpenAIAgent(model_id="openai:gpt-5-mini", client=client)
    agent.set_context(tournament=_tctx(), session=_sctx())
    raw = await agent.decide(_ctx())
    assert raw.action == "call"


@pytest.mark.asyncio
async def test_openai_uses_json_schema_response_format() -> None:
    spy = _Spy()
    client = _FakeOpenAI(['{"kind": "action", "action": "fold"}'], spy)
    agent = OpenAIAgent(model_id="openai:gpt-5-mini", client=client)
    agent.set_context(tournament=_tctx(), session=_sctx())
    await agent.decide(_ctx())
    assert spy.last_kwargs is not None
    rf = spy.last_kwargs.get("response_format")
    assert rf is not None
    assert rf["type"] == "json_schema"


@pytest.mark.asyncio
async def test_openai_reasoning_effort_forwarded() -> None:
    spy = _Spy()
    client = _FakeOpenAI(['{"kind": "action", "action": "fold"}'], spy)
    agent = OpenAIAgent(
        model_id="openai:gpt-5",
        client=client,
        reasoning_effort="medium",
    )
    agent.set_context(tournament=_tctx(), session=_sctx())
    await agent.decide(_ctx())
    assert spy.last_kwargs is not None
    assert spy.last_kwargs.get("reasoning_effort") == "medium"


@pytest.mark.asyncio
async def test_openai_usage_separates_cache_from_input() -> None:
    spy = _Spy()
    client = _FakeOpenAI(['{"kind": "action", "action": "check"}'], spy)
    agent = OpenAIAgent(model_id="openai:gpt-5-mini", client=client)
    agent.set_context(tournament=_tctx(), session=_sctx())
    await agent.decide(_ctx())
    u = agent.last_usage
    assert u is not None
    assert u.output_tokens == _EXPECTED_COMPLETION_TOKENS
    assert u.cache_read_tokens == _EXPECTED_CACHED_TOKENS
    assert u.input_tokens == _EXPECTED_PROMPT_TOKENS - _EXPECTED_CACHED_TOKENS
