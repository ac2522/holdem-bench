"""Uniform-random baseline — never chats. Never calls the network."""

from __future__ import annotations

import numpy as np

from holdembench.agents.base import DecisionContext, Pricing
from holdembench.engine.validator import RawDecision


class RandomAgent:
    model_id = "stub:random"
    pricing = Pricing(input_per_mtok=0.0, output_per_mtok=0.0)

    def __init__(self, seed: int = 0, big_blind: int = 20) -> None:
        self._rng = np.random.default_rng(seed)
        self._big_blind = big_blind

    async def decide(self, ctx: DecisionContext) -> RawDecision:
        choice = self._rng.choice(len(ctx.legal))
        action = ctx.legal[choice]
        amount: int | None = None
        if action == "raise":
            amount = self._big_blind * 2  # always min-raise
        return RawDecision(kind="action", action=action, amount=amount, message=None)
