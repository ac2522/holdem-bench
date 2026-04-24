"""HumanAgent — asyncio-Queue based.  Django PlayView POSTs decisions in.

Spec §8.8: 120s default timeout.  CLI fallback path (stdin prompt) lives in
``cli.py`` — this module is agnostic to where the JSON decision originates.
"""

from __future__ import annotations

import asyncio

from holdembench.agents.base import Agent, DecisionContext, Pricing
from holdembench.agents.output_schema import AgentOutputParseError, parse_agent_output
from holdembench.engine.validator import RawDecision

HUMAN_TIMEOUT_DEFAULT_S = 120.0

_FREE_PRICING = Pricing(input_per_mtok=0.0, output_per_mtok=0.0)


class HumanDecisionQueue:
    """Single-slot async queue; callers (Django view / CLI stdin) push JSON text."""

    def __init__(self) -> None:
        self._q: asyncio.Queue[str] = asyncio.Queue(maxsize=1)

    async def submit(self, json_text: str) -> None:
        await self._q.put(json_text)

    async def wait(self, timeout_s: float) -> str:
        return await asyncio.wait_for(self._q.get(), timeout=timeout_s)


class HumanAgent(Agent):
    model_id: str
    pricing: Pricing

    def __init__(
        self,
        *,
        model_id: str,
        queue: HumanDecisionQueue,
        timeout_s: float = HUMAN_TIMEOUT_DEFAULT_S,
    ) -> None:
        self.model_id = model_id
        self.pricing = _FREE_PRICING
        self._queue = queue
        self._timeout_s = timeout_s

    async def decide(self, ctx: DecisionContext) -> RawDecision:  # noqa: ARG002
        try:
            text = await self._queue.wait(timeout_s=self._timeout_s)
        except TimeoutError:
            return RawDecision(kind="action", action="fold")
        try:
            return parse_agent_output(text).to_raw_decision()
        except AgentOutputParseError:
            return RawDecision(kind="action", action="fold")
