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
_EXPECTED_REASONING_TOKENS = 200
_TEST_MIN_RAISE_TO = 120


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


def _openai_response(text: str, *, reasoning_tokens: int = 0) -> object:
    class Details:
        cached_tokens = _EXPECTED_CACHED_TOKENS

    class CompDetails:
        # OpenAI's `completion_tokens_details.reasoning_tokens` is a SUBSET
        # of `completion_tokens`.  Reasoning is billed but reported here.
        reasoning_tokens = 0

    CompDetails.reasoning_tokens = reasoning_tokens

    class Usage:
        prompt_tokens = _EXPECTED_PROMPT_TOKENS
        # When a model emits reasoning, completion_tokens INCLUDES the
        # reasoning slice; visible output is the difference.
        completion_tokens = _EXPECTED_COMPLETION_TOKENS + reasoning_tokens
        prompt_tokens_details = Details()
        completion_tokens_details = CompDetails()

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
    # We use the OpenRouter unified form (extra_body.reasoning.effort)
    # rather than the top-level OpenAI flat form.  Sending both has been
    # observed to 400 on some OR-routed reasoning models.
    assert "reasoning_effort" not in spy.last_kwargs
    extra = spy.last_kwargs.get("extra_body") or {}
    assert extra.get("reasoning") == {"effort": "medium"}


@pytest.mark.asyncio
async def test_openai_captures_reasoning_tokens_separately() -> None:
    """When the response carries reasoning_tokens, Usage.thinking_tokens is set
    and output_tokens is the visible-only slice (completion - reasoning).
    """
    spy = _Spy()

    # Custom fake that injects reasoning_tokens into the response.
    class _ReasoningFakeOpenAI(_FakeOpenAI):
        class _Completions(_FakeOpenAI._Completions):  # type: ignore[misc]
            async def create(self, **kwargs: Any) -> object:
                self._spy.call_count += 1
                self._spy.last_kwargs = kwargs
                return _openai_response(
                    self._responses[0], reasoning_tokens=_EXPECTED_REASONING_TOKENS
                )

        def __init__(self, responses: list[str], spy: _Spy) -> None:
            self._responses = responses
            self._spy = spy
            chat_obj = self._Chat.__new__(self._Chat)
            chat_obj.completions = _ReasoningFakeOpenAI._Completions(responses, spy)
            self.chat = chat_obj

    client = _ReasoningFakeOpenAI(['{"kind": "action", "action": "fold"}'], spy)
    agent = OpenAIAgent(model_id="openai:gpt-5-mini", client=client)
    agent.set_context(tournament=_tctx(), session=_sctx())
    await agent.decide(_ctx())
    u = agent.last_usage
    assert u is not None
    assert u.thinking_tokens == _EXPECTED_REASONING_TOKENS
    # Visible output is completion_tokens minus the reasoning subset.
    assert u.output_tokens == _EXPECTED_COMPLETION_TOKENS


@pytest.mark.asyncio
async def test_openai_cost_includes_thinking_tokens() -> None:
    """Cost should reflect reasoning tokens billed at output rate by default."""
    spy = _Spy()

    class _ReasoningFakeOpenAI(_FakeOpenAI):
        class _Completions(_FakeOpenAI._Completions):  # type: ignore[misc]
            async def create(self, **kwargs: Any) -> object:
                self._spy.call_count += 1
                self._spy.last_kwargs = kwargs
                return _openai_response(
                    self._responses[0], reasoning_tokens=_EXPECTED_REASONING_TOKENS
                )

        def __init__(self, responses: list[str], spy: _Spy) -> None:
            self._responses = responses
            self._spy = spy
            chat_obj = self._Chat.__new__(self._Chat)
            chat_obj.completions = _ReasoningFakeOpenAI._Completions(responses, spy)
            self.chat = chat_obj

    client = _ReasoningFakeOpenAI(['{"kind": "action", "action": "call"}'], spy)
    agent = OpenAIAgent(model_id="openai:gpt-5-mini", client=client)
    agent.set_context(tournament=_tctx(), session=_sctx())
    await agent.decide(_ctx())
    # gpt-5-mini pricing: input 0.40/MTok, output 1.60/MTok, cache 0.10/MTok
    # Cost should include 200 reasoning tokens billed at 1.60/MTok.
    expected_thinking_cost = _EXPECTED_REASONING_TOKENS * 1.60 / 1_000_000
    # Bring some slack: just assert thinking contributes positively.
    assert agent.last_cost_usd > expected_thinking_cost * 0.9


@pytest.mark.asyncio
async def test_openai_amount_has_no_minimum_constraint() -> None:
    """Schema must NOT carry `minimum` on amount — Anthropic-via-OpenRouter
    rejects `minimum` on integer types.  Sub-min raises are caught post-hoc
    by TDAValidator instead.
    """
    spy = _Spy()
    client = _FakeOpenAI(['{"kind": "action", "action": "raise", "amount": 100}'], spy)
    agent = OpenAIAgent(model_id="openai:gpt-5-mini", client=client)
    agent.set_context(tournament=_tctx(), session=_sctx())
    ctx = DecisionContext(
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
        min_raise_to=_TEST_MIN_RAISE_TO,
    )
    await agent.decide(ctx)
    assert spy.last_kwargs is not None
    schema = spy.last_kwargs["response_format"]["json_schema"]["schema"]
    int_branch = next(
        b for b in schema["properties"]["amount"]["anyOf"] if b.get("type") == "integer"
    )
    assert "minimum" not in int_branch
    assert "maximum" not in int_branch


@pytest.mark.asyncio
async def test_openai_response_format_enum_narrowed_to_legal() -> None:
    """Schema's `action` enum must equal ctx.legal so providers reject illegal names."""
    spy = _Spy()
    client = _FakeOpenAI(['{"kind": "action", "action": "fold"}'], spy)
    agent = OpenAIAgent(model_id="openai:gpt-5-mini", client=client)
    agent.set_context(tournament=_tctx(), session=_sctx())
    ctx = DecisionContext(
        seat="Seat1",
        hand_id="s1h001",
        street="preflop",
        legal=("fold", "check"),  # raise NOT legal here
        stacks={"Seat1": 1000},
        board=(),
        hole=("As", "Kd"),
        budget_remaining=400,
        is_probe_reply=False,
        deadline_s=60.0,
    )
    await agent.decide(ctx)
    assert spy.last_kwargs is not None
    schema = spy.last_kwargs["response_format"]["json_schema"]["schema"]
    string_branch = next(
        b for b in schema["properties"]["action"]["anyOf"] if b.get("type") == "string"
    )
    assert string_branch["enum"] == ["fold", "check"]


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
