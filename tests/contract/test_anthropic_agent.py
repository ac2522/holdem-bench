"""Anthropic adapter contract test.

Uses a fake anthropic-SDK-shaped client so no network is required.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from holdembench.agents.anthropic import AnthropicAgent
from holdembench.agents.base import DecisionContext
from holdembench.agents.prompt import SessionContext, TournamentContext

_EXPECTED_INPUT_TOKENS = 420
_EXPECTED_OUTPUT_TOKENS = 12
_EXPECTED_CACHE_READ_TOKENS = 380
_EXPECTED_CACHE_CREATION_TOKENS = 0
_EXPECTED_SYSTEM_BLOCK_COUNT = 2


@dataclass
class _Spy:
    call_count: int = 0
    cache_control_attempted: bool = False
    last_kwargs: dict[str, Any] | None = None


class _FakeAnthropic:
    def __init__(self, responses: list[str], spy: _Spy) -> None:
        self._responses = responses
        self._spy = spy
        self.messages = self._Messages(responses, spy)

    class _Messages:
        def __init__(self, responses: list[str], spy: _Spy) -> None:
            self._responses = responses
            self._spy = spy

        async def create(self, **kwargs: Any) -> object:
            self._spy.call_count += 1
            self._spy.last_kwargs = kwargs
            for block in kwargs.get("system", []):
                if isinstance(block, dict) and block.get("cache_control"):
                    self._spy.cache_control_attempted = True
            text = self._responses[min(self._spy.call_count - 1, len(self._responses) - 1)]
            return _anthropic_response(text)


def _anthropic_response(text: str) -> object:
    class Usage:
        input_tokens = _EXPECTED_INPUT_TOKENS
        output_tokens = _EXPECTED_OUTPUT_TOKENS
        cache_creation_input_tokens = _EXPECTED_CACHE_CREATION_TOKENS
        cache_read_input_tokens = _EXPECTED_CACHE_READ_TOKENS

    class Block:
        type = "text"

        def __init__(self, text: str) -> None:
            self.text = text

    class Resp:
        content = [Block(text)]
        usage = Usage()

    return Resp()


def _tournament() -> TournamentContext:
    return TournamentContext(tournament_id="t", seat="Seat1", seat_count=3)


def _session() -> SessionContext:
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
async def test_anthropic_decide_returns_raw_decision() -> None:
    spy = _Spy()
    client = _FakeAnthropic(['{"kind": "action", "action": "call"}'], spy)
    agent = AnthropicAgent(model_id="anthropic:claude-haiku-4-5", client=client)
    agent.set_context(tournament=_tournament(), session=_session())
    raw = await agent.decide(_ctx())
    assert raw.kind == "action"
    assert raw.action == "call"


@pytest.mark.asyncio
async def test_anthropic_cache_control_on_system_blocks() -> None:
    spy = _Spy()
    client = _FakeAnthropic(['{"kind": "action", "action": "fold"}'], spy)
    agent = AnthropicAgent(model_id="anthropic:claude-haiku-4-5", client=client)
    agent.set_context(tournament=_tournament(), session=_session())
    await agent.decide(_ctx())
    assert spy.cache_control_attempted
    assert spy.last_kwargs is not None
    system_blocks = spy.last_kwargs["system"]
    assert len(system_blocks) == _EXPECTED_SYSTEM_BLOCK_COUNT
    for block in system_blocks:
        assert block["cache_control"] == {"type": "ephemeral"}


@pytest.mark.asyncio
async def test_anthropic_usage_populated() -> None:
    spy = _Spy()
    client = _FakeAnthropic(['{"kind": "action", "action": "check"}'], spy)
    agent = AnthropicAgent(model_id="anthropic:claude-haiku-4-5", client=client)
    agent.set_context(tournament=_tournament(), session=_session())
    await agent.decide(_ctx())
    u = agent.last_usage
    assert u is not None
    assert u.input_tokens == _EXPECTED_INPUT_TOKENS
    assert u.output_tokens == _EXPECTED_OUTPUT_TOKENS
    assert u.cache_read_tokens == _EXPECTED_CACHE_READ_TOKENS


@pytest.mark.asyncio
async def test_anthropic_thinking_enabled_flag() -> None:
    spy = _Spy()
    client = _FakeAnthropic(['{"kind": "action", "action": "fold"}'], spy)
    agent = AnthropicAgent(
        model_id="anthropic:claude-haiku-4-5",
        client=client,
        enable_thinking=True,
    )
    agent.set_context(tournament=_tournament(), session=_session())
    await agent.decide(_ctx())
    assert spy.last_kwargs is not None
    assert spy.last_kwargs.get("thinking") is not None
    assert spy.last_kwargs["thinking"]["type"] == "enabled"


@pytest.mark.asyncio
async def test_anthropic_sdk_model_name_strips_provider_prefix() -> None:
    spy = _Spy()
    client = _FakeAnthropic(['{"kind": "action", "action": "fold"}'], spy)
    agent = AnthropicAgent(model_id="anthropic:claude-opus-4-7", client=client)
    agent.set_context(tournament=_tournament(), session=_session())
    await agent.decide(_ctx())
    assert spy.last_kwargs is not None
    assert spy.last_kwargs["model"] == "claude-opus-4-7"
