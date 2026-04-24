"""BaseAdapter retry + cost-accounting contract.

Uses a concrete FakeAdapter subclass to exercise the shared machinery without
depending on any real SDK.  Real adapters follow the same contract and are
tested in per-provider files.
"""

from __future__ import annotations

import pytest

from holdembench.agents.base import DecisionContext, Pricing
from holdembench.agents.base_adapter import BaseAdapter, ProviderCall, Usage
from holdembench.agents.prompt import SessionContext, TournamentContext

_EXPECTED_RETRY_ATTEMPTS = 2
_FAKE_INPUT_TOKENS = 100
_FAKE_OUTPUT_TOKENS = 10
_FAKE_CACHE_READ_TOKENS = 80


class _FakeAdapter(BaseAdapter):
    """Test-only adapter that replays a scripted sequence of provider responses."""

    def __init__(
        self,
        *,
        model_id: str,
        responses: list[str],
        pricing: Pricing | None = None,
    ) -> None:
        # Bypass pricing_sheet lookup — inject a local pricing override.
        self.model_id = model_id
        self.pricing = pricing or Pricing(input_per_mtok=1.0, output_per_mtok=5.0)
        self._client: object = object()
        self._tournament: TournamentContext | None = None
        self._session: SessionContext | None = None
        self._last_usage: Usage | None = None
        self._last_cost_usd: float = 0.0
        self._last_thinking: str | None = None
        self._last_prompt_hash: str = ""
        self._last_latency_ms: int = 0
        self._responses = iter(responses)
        self.call_count = 0

    async def _call_provider(
        self,
        ctx: DecisionContext,
        *,
        retry_reason: str | None,
    ) -> ProviderCall:
        self.call_count += 1
        # Render to populate prompt_hash even for the fake adapter.
        self._render(ctx)
        try:
            text = next(self._responses)
        except StopIteration:
            text = ""
        return ProviderCall(
            text=text,
            usage=Usage(
                input_tokens=_FAKE_INPUT_TOKENS,
                output_tokens=_FAKE_OUTPUT_TOKENS,
                cache_read_tokens=_FAKE_CACHE_READ_TOKENS,
            ),
            latency_ms=0,
        )


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
async def test_valid_response_returns_raw_decision() -> None:
    adapter = _FakeAdapter(
        model_id="test:model",
        responses=['{"kind": "action", "action": "call"}'],
    )
    adapter.set_context(tournament=_tournament(), session=_session())
    raw = await adapter.decide(_ctx())
    assert raw.kind == "action"
    assert raw.action == "call"
    assert adapter.call_count == 1


@pytest.mark.asyncio
async def test_parse_failure_then_success_retries_once() -> None:
    adapter = _FakeAdapter(
        model_id="test:model",
        responses=["bad", '{"kind": "action", "action": "fold"}'],
    )
    adapter.set_context(tournament=_tournament(), session=_session())
    raw = await adapter.decide(_ctx())
    assert adapter.call_count == _EXPECTED_RETRY_ATTEMPTS
    assert raw.action == "fold"


@pytest.mark.asyncio
async def test_two_failures_auto_folds() -> None:
    adapter = _FakeAdapter(
        model_id="test:model",
        responses=["bad", "also bad"],
    )
    adapter.set_context(tournament=_tournament(), session=_session())
    raw = await adapter.decide(_ctx())
    assert adapter.call_count == _EXPECTED_RETRY_ATTEMPTS
    assert raw.kind == "action"
    assert raw.action == "fold"


@pytest.mark.asyncio
async def test_cost_and_usage_accumulated() -> None:
    adapter = _FakeAdapter(
        model_id="test:model",
        responses=['{"kind": "action", "action": "check"}'],
    )
    adapter.set_context(tournament=_tournament(), session=_session())
    await adapter.decide(_ctx())
    u = adapter.last_usage
    assert u is not None
    assert u.input_tokens == _FAKE_INPUT_TOKENS
    assert u.output_tokens == _FAKE_OUTPUT_TOKENS
    assert u.cache_read_tokens == _FAKE_CACHE_READ_TOKENS
    expected = _FAKE_INPUT_TOKENS * 1.0 / 1_000_000 + _FAKE_OUTPUT_TOKENS * 5.0 / 1_000_000
    assert adapter.last_cost_usd == pytest.approx(expected)


@pytest.mark.asyncio
async def test_prompt_hash_stashed_on_each_decide() -> None:
    adapter = _FakeAdapter(
        model_id="test:model",
        responses=['{"kind": "action", "action": "fold"}'],
    )
    adapter.set_context(tournament=_tournament(), session=_session())
    await adapter.decide(_ctx())
    assert adapter.last_prompt_hash.startswith("sha256:")


@pytest.mark.asyncio
async def test_cost_accumulates_across_retries() -> None:
    """When first attempt fails JSON parse, cost for BOTH calls must be counted."""
    adapter = _FakeAdapter(
        model_id="test:model",
        responses=["bad", '{"kind": "action", "action": "fold"}'],
    )
    adapter.set_context(tournament=_tournament(), session=_session())
    await adapter.decide(_ctx())
    u = adapter.last_usage
    assert u is not None
    # 2 calls × 100 input / 10 output / 80 cache-read
    assert u.input_tokens == _FAKE_INPUT_TOKENS * _EXPECTED_RETRY_ATTEMPTS
    assert u.output_tokens == _FAKE_OUTPUT_TOKENS * _EXPECTED_RETRY_ATTEMPTS
    assert u.cache_read_tokens == _FAKE_CACHE_READ_TOKENS * _EXPECTED_RETRY_ATTEMPTS
    # And parse-retries should count the one recovered retry.
    assert adapter.last_parse_retries == 1


@pytest.mark.asyncio
async def test_decide_resets_counters_between_calls() -> None:
    """A second decide() call must not carry forward the first call's cost."""
    adapter = _FakeAdapter(
        model_id="test:model",
        responses=[
            '{"kind": "action", "action": "fold"}',
            '{"kind": "action", "action": "call"}',
        ],
    )
    adapter.set_context(tournament=_tournament(), session=_session())
    await adapter.decide(_ctx())
    first_cost = adapter.last_cost_usd
    await adapter.decide(_ctx())
    # Each decide() consumes exactly one response here, so costs should be equal.
    assert adapter.last_cost_usd == pytest.approx(first_cost)
    assert adapter.last_parse_retries == 0


@pytest.mark.asyncio
async def test_thinking_captured_from_output() -> None:
    adapter = _FakeAdapter(
        model_id="test:model",
        responses=['{"kind": "action", "action": "call", "thinking": "pondered"}'],
    )
    adapter.set_context(tournament=_tournament(), session=_session())
    await adapter.decide(_ctx())
    assert adapter.last_thinking == "pondered"


