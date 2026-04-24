"""Moonshot adapter contract — verifies explicit cache_control on system msgs."""

from __future__ import annotations

from typing import Any

import pytest

from holdembench.agents.base import DecisionContext
from holdembench.agents.moonshot import MoonshotAgent
from holdembench.agents.prompt import SessionContext, TournamentContext

_EXPECTED_SYSTEM_MSGS_WITH_CACHE = 2


class _FakeClient:
    def __init__(self, text: str) -> None:
        self._text = text
        self.last_kwargs: dict[str, Any] | None = None
        self.chat = self._Chat(self)

    class _Chat:
        def __init__(self, outer: _FakeClient) -> None:
            self.completions = _FakeClient._Completions(outer)

    class _Completions:
        def __init__(self, outer: _FakeClient) -> None:
            self._outer = outer

        async def create(self, **kwargs: Any) -> object:
            self._outer.last_kwargs = kwargs

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

            return Resp(self._outer._text)


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


@pytest.mark.asyncio
async def test_moonshot_emits_explicit_cache_control() -> None:
    client = _FakeClient('{"kind": "action", "action": "fold"}')
    agent = MoonshotAgent(model_id="moonshot:kimi-k2", client=client)
    agent.set_context(
        tournament=TournamentContext(tournament_id="t", seat="Seat1", seat_count=2),
        session=SessionContext(
            session_id=1,
            small_blind=10,
            big_blind=20,
            ante=0,
            starting_stack_bb=100,
            orbit_budget_tokens=400,
        ),
    )
    raw = await agent.decide(_ctx())
    assert raw.action == "fold"
    assert client.last_kwargs is not None
    msgs = client.last_kwargs["messages"]
    sys_msgs_with_cc = [
        m for m in msgs if m["role"] == "system" and "cache_control" in m
    ]
    assert len(sys_msgs_with_cc) == _EXPECTED_SYSTEM_MSGS_WITH_CACHE
    for m in sys_msgs_with_cc:
        assert m["cache_control"] == {"type": "ephemeral"}
