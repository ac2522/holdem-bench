"""Tests for bootstrap BCa CI helper."""

from __future__ import annotations

import numpy as np

from holdembench.scoring.bootstrap_ci import bootstrap_mean_ci

TRUE_MEAN = 5.0


def test_ci_contains_true_mean_often() -> None:
    rng = np.random.default_rng(0)
    data = rng.normal(loc=TRUE_MEAN, scale=1.0, size=500)
    lo, hi = bootstrap_mean_ci(data.tolist(), confidence=0.95, n_resamples=2000, seed=1)
    assert lo < TRUE_MEAN < hi


def test_ci_shrinks_with_more_data() -> None:
    rng = np.random.default_rng(1)
    small = rng.normal(0, 1, 30).tolist()
    large = rng.normal(0, 1, 3000).tolist()
    lo_s, hi_s = bootstrap_mean_ci(small, 0.95, 1000, seed=1)
    lo_l, hi_l = bootstrap_mean_ci(large, 0.95, 1000, seed=1)
    assert (hi_l - lo_l) < (hi_s - lo_s)


def test_empty_returns_nan_pair() -> None:
    lo, hi = bootstrap_mean_ci([], 0.95, 1000, seed=1)
    assert np.isnan(lo)
    assert np.isnan(hi)
