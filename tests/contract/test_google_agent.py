"""Google Gemini adapter contract test — fake google-genai-shaped client."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from holdembench.agents.base import DecisionContext
from holdembench.agents.google import GoogleAgent
from holdembench.agents.prompt import SessionContext, TournamentContext

_EXPECTED_PROMPT_TOKENS = 420
_EXPECTED_OUTPUT_TOKENS = 12
_EXPECTED_CACHED_TOKENS = 380


@dataclass
class _Spy:
    call_count: int = 0
    last_kwargs: dict[str, Any] | None = None


class _FakeGoogle:
    def __init__(self, responses: list[str], spy: _Spy) -> None:
        self._responses = responses
        self._spy = spy

    async def generate_content(self, **kwargs: Any) -> object:
        self._spy.call_count += 1
        self._spy.last_kwargs = kwargs
        text = self._responses[min(self._spy.call_count - 1, len(self._responses) - 1)]
        return _google_response(text)


def _google_response(text: str) -> object:
    class Usage:
        prompt_token_count = _EXPECTED_PROMPT_TOKENS
        candidates_token_count = _EXPECTED_OUTPUT_TOKENS
        cached_content_token_count = _EXPECTED_CACHED_TOKENS

    class Part:
        def __init__(self, text: str) -> None:
            self.text = text

    class Content:
        def __init__(self, text: str) -> None:
            self.parts = [Part(text)]

    class Candidate:
        def __init__(self, text: str) -> None:
            self.content = Content(text)

    class Resp:
        def __init__(self, text: str) -> None:
            self.candidates = [Candidate(text)]
            self.usage_metadata = Usage()

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
async def test_google_decide_returns_raw_decision() -> None:
    spy = _Spy()
    client = _FakeGoogle(['{"kind": "action", "action": "call"}'], spy)
    agent = GoogleAgent(model_id="google:gemini-3-flash-preview", client=client)
    agent.set_context(tournament=_tctx(), session=_sctx())
    raw = await agent.decide(_ctx())
    assert raw.action == "call"


@pytest.mark.asyncio
async def test_google_response_schema_requested() -> None:
    spy = _Spy()
    client = _FakeGoogle(['{"kind": "action", "action": "fold"}'], spy)
    agent = GoogleAgent(model_id="google:gemini-3-flash-preview", client=client)
    agent.set_context(tournament=_tctx(), session=_sctx())
    await agent.decide(_ctx())
    assert spy.last_kwargs is not None
    cfg = spy.last_kwargs.get("config") or {}
    assert cfg.get("response_mime_type") == "application/json"
    assert "response_schema" in cfg


@pytest.mark.asyncio
async def test_google_response_schema_enum_narrowed_to_legal() -> None:
    """Schema's `action` enum must equal ctx.legal (Gemini-shaped schema)."""
    spy = _Spy()
    client = _FakeGoogle(['{"kind": "action", "action": "fold"}'], spy)
    agent = GoogleAgent(model_id="google:gemini-3-flash-preview", client=client)
    agent.set_context(tournament=_tctx(), session=_sctx())
    ctx = DecisionContext(
        seat="Seat1",
        hand_id="s1h001",
        street="preflop",
        legal=("fold", "call"),  # raise NOT legal here
        stacks={"Seat1": 1000},
        board=(),
        hole=("As", "Kd"),
        budget_remaining=400,
        is_probe_reply=False,
        deadline_s=60.0,
    )
    await agent.decide(ctx)
    assert spy.last_kwargs is not None
    cfg = spy.last_kwargs["config"]
    schema = cfg["response_schema"]
    assert schema["properties"]["action"]["enum"] == ["fold", "call"]


@pytest.mark.asyncio
async def test_google_usage_reports_cached_separately() -> None:
    spy = _Spy()
    client = _FakeGoogle(['{"kind": "action", "action": "check"}'], spy)
    agent = GoogleAgent(model_id="google:gemini-3-flash-preview", client=client)
    agent.set_context(tournament=_tctx(), session=_sctx())
    await agent.decide(_ctx())
    u = agent.last_usage
    assert u is not None
    assert u.output_tokens == _EXPECTED_OUTPUT_TOKENS
    assert u.cache_read_tokens == _EXPECTED_CACHED_TOKENS
    assert u.input_tokens == _EXPECTED_PROMPT_TOKENS - _EXPECTED_CACHED_TOKENS
