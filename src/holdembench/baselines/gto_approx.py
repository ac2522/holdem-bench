"""Coarse GTO-approximation agent — shoves on push/fold chart preflop,
check-or-call postflop. Phase 0 baseline only; refinement in Phase 3.
"""

from __future__ import annotations

from holdembench.agents.base import DecisionContext, Pricing
from holdembench.baselines._pushfold_data import (
    SHOVE_RANGE_10BB,
    SHOVE_RANGE_15BB,
    SHOVE_RANGE_20BB,
)
from holdembench.engine.validator import RawDecision


def _hand_key(hole: tuple[str, ...]) -> str:
    """Return e.g. 'AKs' / 'AKo' / 'AA'."""
    if len(hole) != 2:  # noqa: PLR2004
        return ""
    r1, s1 = hole[0][0], hole[0][1]
    r2, s2 = hole[1][0], hole[1][1]
    order = "AKQJT98765432"
    if order.index(r1) > order.index(r2):
        r1, s1, r2, s2 = r2, s2, r1, s1
    if r1 == r2:
        return r1 + r2
    return r1 + r2  # suitedness ignored in this coarse chart


def _shove_range_for(stack_bb: int) -> frozenset[str]:
    if stack_bb <= 10:  # noqa: PLR2004
        return SHOVE_RANGE_10BB
    if stack_bb <= 15:  # noqa: PLR2004
        return SHOVE_RANGE_15BB
    return SHOVE_RANGE_20BB


class GTOApproxAgent:
    model_id = "stub:gto_approx"
    pricing = Pricing(input_per_mtok=0.0, output_per_mtok=0.0)

    async def decide(self, ctx: DecisionContext) -> RawDecision:
        if ctx.street == "preflop":
            # Use the per-hand BB from context — varies under rising
            # blinds.  Falls back to 20 only for back-compat with very
            # old test harnesses that didn't populate big_blind.
            bb = ctx.big_blind or 20
            stack_bb = max(1, ctx.stacks[ctx.seat] // bb)
            in_range = _hand_key(ctx.hole) in _shove_range_for(stack_bb)
            if in_range and "raise" in ctx.legal:
                return RawDecision(kind="action", action="raise", amount=ctx.stacks[ctx.seat])
            if "check" in ctx.legal:
                return RawDecision(kind="action", action="check")
            return RawDecision(kind="action", action="fold")

        # Postflop stub: check/call only.
        if "check" in ctx.legal:
            return RawDecision(kind="action", action="check")
        if "call" in ctx.legal:
            return RawDecision(kind="action", action="call")
        return RawDecision(kind="action", action="fold")
