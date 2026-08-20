"""Statistical tests for strategy comparison."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from src.eval.metrics import sharpe_ratio


def circular_block_indices(
    n: int,
    n_bootstrap: int,
    block_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """(n_bootstrap, n) circular overlapping-block index matrix."""
    block_size = max(1, min(int(block_size), n))
    n_blocks = int(np.ceil(n / block_size))
    starts = rng.integers(0, n, size=(n_bootstrap, n_blocks))
    offsets = np.arange(block_size)
    idx = (starts[..., None] + offsets) % n
    return idx.reshape(n_bootstrap, -1)[:, :n]


def _sharpe_2d(
    returns: np.ndarray,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252,
) -> np.ndarray:
    excess = returns - risk_free_rate / periods_per_year
    vol = excess.std(axis=1)
    out = np.zeros(returns.shape[0], dtype=np.float64)
    ok = vol >= 1e-12
    out[ok] = excess[ok].mean(axis=1) / vol[ok] * np.sqrt(periods_per_year)
    return out


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
    Circular block-bootstrap CI for difference metric(a) - metric(b).

    Returns (point_estimate, lower, upper).
    """
    rng = rng or np.random.default_rng(42)
    n = min(len(returns_a), len(returns_b))
    a, b = np.asarray(returns_a[:n]), np.asarray(returns_b[:n])
    point = float(metric_fn(a) - metric_fn(b))

    idx = circular_block_indices(n, n_bootstrap, block_size, rng)
    diffs = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        diffs[i] = metric_fn(a[idx[i]]) - metric_fn(b[idx[i]])

    lo = float(np.percentile(diffs, 100 * alpha / 2))
    hi = float(np.percentile(diffs, 100 * (1 - alpha / 2)))
    return point, lo, hi


def block_bootstrap_sharpe_diff(
    returns_a: np.ndarray,
    returns_b: np.ndarray,
    n_bootstrap: int = 1000,
    block_size: int = 20,
    alpha: float = 0.05,
    risk_free_rate: float = 0.02,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Vectorized Sharpe-difference CI (circular block bootstrap)."""
    rng = rng or np.random.default_rng(42)
    n = min(len(returns_a), len(returns_b))
    a, b = np.asarray(returns_a[:n], dtype=np.float64), np.asarray(returns_b[:n], dtype=np.float64)
    point = float(
        sharpe_ratio(a, risk_free_rate) - sharpe_ratio(b, risk_free_rate)
    )
    idx = circular_block_indices(n, n_bootstrap, block_size, rng)
    diffs = _sharpe_2d(a[idx], risk_free_rate) - _sharpe_2d(b[idx], risk_free_rate)
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


def significance_table(
    series: dict[str, np.ndarray],
    benchmark: str = "buy_and_hold",
    risk_free_rate: float = 0.02,
    n_bootstrap: int = 1000,
    block_size: int = 20,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Compare every strategy's Sharpe against ``benchmark`` (default buy-and-hold)."""
    rng = rng or np.random.default_rng(42)
    if benchmark not in series:
        raise KeyError(f"Benchmark {benchmark!r} not in series (have {list(series)})")

    n_trials = len(series)
    bah = np.asarray(series[benchmark], dtype=np.float64)
    rows = []
    for name, rets in series.items():
        r = np.asarray(rets, dtype=np.float64)
        n = min(len(r), len(bah))
        r, b = r[:n], bah[:n]
        sr = sharpe_ratio(r, risk_free_rate)
        if name == benchmark:
            diff, lo, hi = 0.0, 0.0, 0.0
            z, p = 0.0, 1.0
        else:
            diff, lo, hi = block_bootstrap_sharpe_diff(
                r, b, n_bootstrap=n_bootstrap, block_size=block_size,
                risk_free_rate=risk_free_rate, rng=rng,
            )
            z, p = ledoit_wolf_sharpe_test(r, b)
        dsr = deflated_sharpe_ratio(
            observed_sharpe=sr,
            n_trials=n_trials,
            n_observations=n,
            skewness=float(stats.skew(r)) if n > 2 else 0.0,
            kurtosis=float(stats.kurtosis(r, fisher=False)) if n > 3 else 3.0,
        )
        rows.append({
            "strategy": name,
            "sharpe": sr,
            "sharpe_diff_vs_bah": diff,
            "ci_lower": lo,
            "ci_upper": hi,
            "ledoit_wolf_z": z,
            "ledoit_wolf_p": p,
            "deflated_sharpe": dsr,
            "n_observations": n,
            "n_trials": n_trials,
        })
    return pd.DataFrame(rows)
