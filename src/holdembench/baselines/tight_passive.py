"""Tight-passive preflop heuristic. Postflop: fold-to-aggression."""

from __future__ import annotations

from holdembench.agents.base import DecisionContext, Pricing
from holdembench.engine.validator import RawDecision

_PREMIUM = {
    frozenset(("A", "A")),
    frozenset(("K", "K")),
    frozenset(("Q", "Q")),
    frozenset(("J", "J")),
    frozenset(("A", "K")),
}
_MEDIUM = {
    frozenset(("T", "T")),
    frozenset(("9", "9")),
    frozenset(("8", "8")),
    frozenset(("A", "Q")),
    frozenset(("A", "J")),
    frozenset(("K", "Q")),
    frozenset(("K", "J")),
    frozenset(("Q", "J")),
    frozenset(("J", "T")),
}


def _rank_pair(hole: tuple[str, ...]) -> frozenset[str]:
    """Extract the rank pair from hole cards (ignoring suits)."""
    return frozenset(c[0] for c in hole)


class TightPassiveAgent:
    """A simple baseline that plays premium/medium hands, folds weak ones."""

    model_id = "stub:tight_passive"
    pricing = Pricing(input_per_mtok=0.0, output_per_mtok=0.0)

    async def decide(self, ctx: DecisionContext) -> RawDecision:
        """Decide action based on hand strength and available actions."""
        pair = _rank_pair(ctx.hole)

        # Check for free play (check available)
        if "check" in ctx.legal:
            return RawDecision(kind="action", action="check")

        # Premium hands: raise if possible
        if pair in _PREMIUM and "raise" in ctx.legal:
            return RawDecision(
                kind="action",
                action="raise",
                amount=_compute_raise(ctx),
                message=None,
            )

        # Medium hands: call if possible
        if pair in _MEDIUM and "call" in ctx.legal:
            return RawDecision(kind="action", action="call", message=None)

        # Weak hands: fold if possible (else take first legal action)
        if "fold" in ctx.legal:
            return RawDecision(kind="action", action="fold", message=None)

        # Fallback to first legal action (shouldn't reach in normal play)
        return RawDecision(kind="action", action=ctx.legal[0], message=None)


def _compute_raise(ctx: DecisionContext) -> int:
    """Compute raise amount. Default: 3× BB."""
    return 60  # 3× BB default — refined once we wire real blind data in harness
