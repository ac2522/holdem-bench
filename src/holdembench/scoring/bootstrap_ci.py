"""Bootstrap BCa confidence intervals via scipy.stats.bootstrap."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.stats import bootstrap  # type: ignore[import-untyped]


def bootstrap_mean_ci(
    data: list[float],
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> tuple[float, float]:
    """Return (lo, hi) BCa CI on the mean. (NaN, NaN) for empty input."""
    if not data:
        return (math.nan, math.nan)
    arr = np.asarray(data, dtype=np.float64)
    res: Any = bootstrap(
        (arr,),
        statistic=np.mean,
        confidence_level=confidence,
        n_resamples=n_resamples,
        method="BCa",
        rng=np.random.default_rng(seed),
    )
    lo: float = float(res.confidence_interval.low)  # type: ignore[union-attr]
    hi: float = float(res.confidence_interval.high)  # type: ignore[union-attr]
    return lo, hi
