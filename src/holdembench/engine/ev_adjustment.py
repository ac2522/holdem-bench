"""All-in Monte Carlo equity using pokerkit StandardHighHand for correctness."""
from __future__ import annotations

import numpy as np
from pokerkit import Card, Hand, StandardHighHand

from holdembench.engine.deck import STANDARD_DECK


def monte_carlo_equity(
    hole_cards: list[list[str]],
    board: list[str],
    samples: int,
    seed: int,
) -> list[float]:
    """Estimate river-equity for each player via Monte Carlo simulation.

    Args:
        hole_cards: list of 2-card lists, one per active player.
        board: 0-5 community cards already dealt.
        samples: number of Monte Carlo samples (5k = ±0.7% error at 1 std dev).
        seed: RNG seed for reproducibility.

    Returns:
        List of win probabilities, one per player, summing to 1.0.
        Ties are split equally among winners.
    """
    rng = np.random.default_rng(seed)
    n_players = len(hole_cards)
    wins = np.zeros(n_players, dtype=np.float64)

    used: set[str] = {c for hole in hole_cards for c in hole} | set(board)
    remaining: list[str] = [c for c in STANDARD_DECK if c not in used]
    needed = 5 - len(board)

    if needed == 0:
        # River already complete — evaluate exactly once, no sampling needed.
        hands = _evaluate_hands(hole_cards, board)
        best = max(hands)
        winners = [i for i, h in enumerate(hands) if h == best]
        share = 1.0 / len(winners)
        for w in winners:
            wins[w] = share
        return wins.tolist()

    for _ in range(samples):
        draw_idx = rng.choice(len(remaining), size=needed, replace=False)
        runout: list[str] = [remaining[int(i)] for i in draw_idx]
        full_board = board + runout
        hands = _evaluate_hands(hole_cards, full_board)
        best = max(hands)
        winners = [i for i, h in enumerate(hands) if h == best]
        share = 1.0 / len(winners)
        for w in winners:
            wins[w] += share

    return (wins / samples).tolist()


def _evaluate_hands(
    hole_cards: list[list[str]],
    board: list[str],
) -> list[Hand]:
    """Return a Hand per player (comparable via <, >, ==)."""
    board_str = "".join(board)
    result: list[Hand] = []
    for hole in hole_cards:
        hole_str = "".join(hole)
        hand: Hand = StandardHighHand.from_game(
            list(Card.parse(hole_str)),
            list(Card.parse(board_str)),
        )
        result.append(hand)
    return result
