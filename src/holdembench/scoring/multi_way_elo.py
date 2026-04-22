"""Plackett-Luce rating — MLE over n-way rankings.

Reference: Plackett (1975). Standard iterative algorithm; Hunter (2004) MM variant.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def fit_plackett_luce(
    rankings: Sequence[Sequence[str]],
    max_iter: int = 500,
    tol: float = 1e-8,
) -> dict[str, float]:
    """Fit Plackett-Luce ratings from a list of rankings.

    Each ranking is a list of player names in order of finish (best first).
    Returns rating per player (higher = stronger), normalized so max == 1.0.
    """
    players = sorted({p for r in rankings for p in r})
    n = len(players)
    idx = {p: i for i, p in enumerate(players)}
    gamma = np.ones(n)

    for _ in range(max_iter):
        new = np.zeros(n)
        denom = np.zeros(n)
        for r in rankings:
            m = len(r)
            for k in range(m):
                subset_idx = [idx[p] for p in r[k:]]
                subset_sum = gamma[subset_idx].sum()
                new[idx[r[k]]] += 1.0
                for p in subset_idx:
                    denom[p] += 1.0 / subset_sum
        next_gamma = np.where(denom > 0, new / denom, gamma)
        next_gamma /= next_gamma.max()
        if np.max(np.abs(next_gamma - gamma)) < tol:
            gamma = next_gamma
            break
        gamma = next_gamma

    return {p: float(gamma[idx[p]]) for p in players}
