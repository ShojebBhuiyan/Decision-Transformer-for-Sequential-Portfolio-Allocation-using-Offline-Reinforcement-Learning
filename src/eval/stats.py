"""Statistical tests for strategy comparison."""

from __future__ import annotations

import numpy as np
from scipy import stats


def block_bootstrap_ci(
    returns_a: np.ndarray,
    returns_b: np.ndarray,
    metric_fn,
    n_bootstrap: int = 1000,
    block_size: int = 20,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """
    Stationary block bootstrap CI for difference metric(a) - metric(b).

    Returns (point_estimate, lower, upper).
    """
    rng = rng or np.random.default_rng(42)
    n = min(len(returns_a), len(returns_b))
    a, b = returns_a[:n], returns_b[:n]
    point = metric_fn(a) - metric_fn(b)

    diffs = []
    for _ in range(n_bootstrap):
        indices = []
        while len(indices) < n:
            start = rng.integers(0, n)
            block = list(range(start, min(start + block_size, n)))
            indices.extend(block)
        indices = indices[:n]
        diffs.append(metric_fn(a[indices]) - metric_fn(b[indices]))

    lo = float(np.percentile(diffs, 100 * alpha / 2))
    hi = float(np.percentile(diffs, 100 * (1 - alpha / 2)))
    return point, lo, hi


def ledoit_wolf_sharpe_test(
    returns_a: np.ndarray,
    returns_b: np.ndarray,
    periods_per_year: int = 252,
) -> tuple[float, float]:
    """
    Ledoit-Wolf (2008) test for equality of Sharpe ratios.

    Returns (test_statistic, p_value).
    """
    n = min(len(returns_a), len(returns_b))
    a, b = returns_a[:n], returns_b[:n]
    sa = a.std() * np.sqrt(periods_per_year)
    sb = b.std() * np.sqrt(periods_per_year)
    if sa < 1e-12 or sb < 1e-12:
        return 0.0, 1.0
    sharpe_a = a.mean() / a.std() * np.sqrt(periods_per_year)
    sharpe_b = b.mean() / b.std() * np.sqrt(periods_per_year)
    # Simplified test statistic
    se = np.sqrt((1 + 0.5 * sharpe_a**2) / n + (1 + 0.5 * sharpe_b**2) / n)
    z = (sharpe_a - sharpe_b) / (se + 1e-12)
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return float(z), float(p)


def deflated_sharpe_ratio(
    observed_sharpe: float,
    n_trials: int,
    n_observations: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """
    Bailey & Lopez de Prado (2014) deflated Sharpe ratio.

    Returns probability that observed Sharpe is not due to selection bias.
    """
    from scipy import stats as sp_stats

    # Expected maximum Sharpe under null
    euler_mascheroni = 0.5772156649
    if n_trials <= 1:
        return 1.0
    expected_max = (
        (1 - euler_mascheroni) * sp_stats.norm.ppf(1 - 1 / n_trials)
        + euler_mascheroni * sp_stats.norm.ppf(1 - 1 / (n_trials * np.e))
    )
    se = np.sqrt((1 + 0.5 * observed_sharpe**2 - skewness * observed_sharpe
                  + (kurtosis - 1) / 4 * observed_sharpe**2) / n_observations)
    z = (observed_sharpe - expected_max) / (se + 1e-12)
    return float(sp_stats.norm.cdf(z))
