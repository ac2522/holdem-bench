"""GTO action policy + rotating canned messages.

Exists to answer 'does any chat at all, even scripted, outperform silence?'
"""

from __future__ import annotations

import numpy as np

from holdembench.agents.base import DecisionContext, Pricing
from holdembench.baselines.gto_approx import GTOApproxAgent
from holdembench.engine.validator import RawDecision

_CANNED_MESSAGES: tuple[str, ...] = (
    "standard spot for me here",
    "I've seen this before",
    "feels like a trap",
    "nothing to write home about",
    "close decision honestly",
    "gonna be tough to call",
    "pot odds say one thing",
    "I'm not loving this board",
    "board doesn't hit me hard",
    "ranges are wide here",
)


class CannedTalkAgent:
    model_id = "stub:canned_talk"
    pricing = Pricing(input_per_mtok=0.0, output_per_mtok=0.0)

    def __init__(self, seed: int = 0, big_blind: int = 20) -> None:
        self._gto = GTOApproxAgent(big_blind=big_blind)
        self._rng = np.random.default_rng(seed)

    async def decide(self, ctx: DecisionContext) -> RawDecision:
        base = await self._gto.decide(ctx)
        message = _CANNED_MESSAGES[self._rng.integers(0, len(_CANNED_MESSAGES))]
        return RawDecision(kind=base.kind, action=base.action, amount=base.amount, message=message)
