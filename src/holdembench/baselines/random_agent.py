"""Uniform-random baseline — never chats. Never calls the network."""

from __future__ import annotations

import numpy as np

from holdembench.agents.base import DecisionContext, Pricing
from holdembench.engine.validator import RawDecision


class RandomAgent:
    model_id = "stub:random"
    pricing = Pricing(input_per_mtok=0.0, output_per_mtok=0.0)

    def __init__(self, seed: int = 0) -> None:
        self._rng = np.random.default_rng(seed)

    async def decide(self, ctx: DecisionContext) -> RawDecision:
        choice = self._rng.choice(len(ctx.legal))
        action = ctx.legal[choice]
        amount: int | None = None
        if action == "raise":
            # Use the runner-supplied min_raise_to (always set when raise
            # is in legal) so we don't break under rising blinds.  Cap at
            # the seat's stack so we never try to raise more than we have.
            seat_stack = ctx.stacks.get(ctx.seat, 0)
            target = ctx.min_raise_to or 0
            amount = min(max(target, 1), seat_stack) if seat_stack > 0 else target
        return RawDecision(kind="action", action=action, amount=amount, message=None)
