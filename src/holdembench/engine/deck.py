"""Deterministic deck shuffling for reproducible deal packs."""
from __future__ import annotations

import numpy as np

_RANKS = "23456789TJQKA"
_SUITS = "cdhs"

STANDARD_DECK: tuple[str, ...] = tuple(f"{r}{s}" for r in _RANKS for s in _SUITS)


def shuffled_deck(seed: int) -> list[str]:
    """Return a deck permutation deterministic in `seed`."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(STANDARD_DECK))
    return [STANDARD_DECK[i] for i in idx]
