"""Shared adapter logic: pricing, retry, cost accounting, usage normalisation.

Every concrete provider (Anthropic / OpenAI / Google / xAI / Moonshot /
OpenRouter) subclasses :class:`BaseAdapter` and implements ``_call_provider``.

``BaseAdapter.decide`` encapsulates:
  * prompt rendering (via :func:`holdembench.agents.prompt.render_prompt`)
  * retry-once-then-autofold on JSON / schema errors
  * cost/usage/latency accounting
  * prompt_hash bookkeeping for the event log

The runner reads ``last_usage``, ``last_cost_usd``, ``last_thinking``,
``last_prompt_hash``, ``last_latency_ms`` off the adapter after each
``decide()`` call and copies them onto the emitted ``ActionResponse``.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from holdembench.agents.base import Agent, DecisionContext, Pricing
from holdembench.agents.output_schema import (
    AgentOutput,
    AgentOutputParseError,
    parse_agent_output,
)
from holdembench.agents.pricing_sheet import lookup_pricing
from holdembench.agents.prompt import (
    PromptBundle,
    SessionContext,
    TournamentContext,
    render_prompt,
)
from holdembench.engine.validator import RawDecision

MAX_PARSE_RETRIES = 1  # total attempts = 1 + MAX_PARSE_RETRIES


@dataclass(frozen=True)
class Usage:
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    thinking_tokens: int = 0


@dataclass(frozen=True)
class ProviderCall:
    """One round-trip to a provider SDK."""

    text: str
    usage: Usage
    latency_ms: int


@runtime_checkable
class ClientProtocol(Protocol):
    """Marker type — each provider declares its own narrow subtype."""


class BaseAdapter(ABC, Agent):
    """All concrete adapters inherit from this."""

    model_id: str
    pricing: Pricing

    def __init__(self, *, model_id: str, client: object) -> None:
        self.model_id = model_id
        self.pricing = lookup_pricing(model_id)
        self._client: object = client
        self._tournament: TournamentContext | None = None
        self._session: SessionContext | None = None
        self._last_usage: Usage | None = None
        self._last_cost_usd: float = 0.0
        self._last_thinking: str | None = None
        self._last_prompt_hash: str = ""
        self._last_latency_ms: int = 0
        self._last_parse_retries: int = 0

    # ----- Context management -----

    def set_context(
        self,
        *,
        tournament: TournamentContext,
        session: SessionContext,
    ) -> None:
        """Runner invokes this at session_start / tournament_start."""
        self._tournament = tournament
        self._session = session

    def _render(self, ctx: DecisionContext) -> PromptBundle:
        if self._tournament is None or self._session is None:
            raise RuntimeError(
                f"{type(self).__name__}.set_context(tournament=..., session=...) "
                "must be called before decide()"
            )
        bundle = render_prompt(
            tournament=self._tournament,
            session=self._session,
            decision=ctx,
        )
        self._last_prompt_hash = bundle.prompt_hash
        return bundle

    # ----- Core ``decide`` loop -----

    async def decide(self, ctx: DecisionContext) -> RawDecision:
        last_error: str | None = None
        # Reset per-decision counters.  Cost + usage accumulate across retries
        # within a single decide() call so telemetry reflects the true spend.
        self._reset_per_decision()
        for attempt in range(1 + MAX_PARSE_RETRIES):
            start = time.perf_counter()
            call = await self._call_provider(ctx, retry_reason=last_error)
            self._last_latency_ms += int((time.perf_counter() - start) * 1000)
            self._accumulate_cost(call.usage)
            if attempt > 0:
                self._last_parse_retries += 1
            try:
                parsed: AgentOutput = parse_agent_output(call.text)
            except AgentOutputParseError as e:
                last_error = str(e)
                continue
            self._last_thinking = parsed.thinking
            return parsed.to_raw_decision()
        # All attempts failed → synthetic auto-fold.  The runner still emits
        # its own ValidatorRejection / AutoFold events when relevant.
        self._last_thinking = None
        return RawDecision(kind="action", action="fold")

    def _reset_per_decision(self) -> None:
        self._last_usage = None
        self._last_cost_usd = 0.0
        self._last_thinking = None
        self._last_latency_ms = 0
        self._last_parse_retries = 0

    # ----- Accounting helpers -----

    def _accumulate_cost(self, usage: Usage) -> None:
        """Add this call's tokens + USD to the running per-decision totals.

        Across retries the runner reads a single last_usage snapshot, so we
        sum fields rather than replace — the telemetry reflects all provider
        calls made to produce one RawDecision.
        """
        if self._last_usage is None:
            self._last_usage = usage
        else:
            self._last_usage = Usage(
                input_tokens=self._last_usage.input_tokens + usage.input_tokens,
                output_tokens=self._last_usage.output_tokens + usage.output_tokens,
                cache_read_tokens=self._last_usage.cache_read_tokens + usage.cache_read_tokens,
                cache_write_tokens=self._last_usage.cache_write_tokens + usage.cache_write_tokens,
                thinking_tokens=self._last_usage.thinking_tokens + usage.thinking_tokens,
            )
        self._last_cost_usd += self.pricing.cost_usd(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_read_tokens=usage.cache_read_tokens,
            cache_write_tokens=usage.cache_write_tokens,
        )

    @property
    def last_usage(self) -> Usage | None:
        return self._last_usage

    @property
    def last_cost_usd(self) -> float:
        return self._last_cost_usd

    @property
    def last_thinking(self) -> str | None:
        return self._last_thinking

    @property
    def last_prompt_hash(self) -> str:
        return self._last_prompt_hash

    @property
    def last_latency_ms(self) -> int:
        return self._last_latency_ms

    @property
    def last_parse_retries(self) -> int:
        """Count of JSON/schema-parse retries for the most recent decide() call."""
        return self._last_parse_retries

    @abstractmethod
    async def _call_provider(
        self,
        ctx: DecisionContext,
        *,
        retry_reason: str | None,
    ) -> ProviderCall:
        """Implement: one round-trip to the provider SDK, returning the raw
        text + normalised usage statistics."""
