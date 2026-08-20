"""Portfolio performance metrics."""

from __future__ import annotations

import numpy as np


def cumulative_return(log_returns: np.ndarray) -> float:
    return float(np.exp(log_returns.sum()) - 1.0)


def cagr(log_returns: np.ndarray, periods_per_year: int = 252) -> float:
    n = len(log_returns)
    if n == 0:
        return 0.0
    total = np.exp(log_returns.sum())
    years = n / periods_per_year
    return float(total ** (1 / years) - 1) if years > 0 else 0.0


def annualized_volatility(log_returns: np.ndarray, periods_per_year: int = 252) -> float:
    return float(log_returns.std() * np.sqrt(periods_per_year))


def sharpe_ratio(
    log_returns: np.ndarray,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252,
) -> float:
    excess = log_returns - risk_free_rate / periods_per_year
    vol = excess.std()
    if vol < 1e-12:
        return 0.0
    return float(excess.mean() / vol * np.sqrt(periods_per_year))


def sortino_ratio(
    log_returns: np.ndarray,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252,
) -> float:
    excess = log_returns - risk_free_rate / periods_per_year
    downside = excess[excess < 0]
    if len(downside) == 0:
        return 0.0
    downside_std = downside.std()
    if downside_std < 1e-12:
        return 0.0
    return float(excess.mean() / downside_std * np.sqrt(periods_per_year))


def max_drawdown(log_returns: np.ndarray) -> float:
    wealth = np.exp(np.cumsum(log_returns))
    peak = np.maximum.accumulate(wealth)
    dd = (wealth - peak) / peak
    return float(dd.min())


def calmar_ratio(log_returns: np.ndarray, periods_per_year: int = 252) -> float:
    mdd = abs(max_drawdown(log_returns))
    if mdd < 1e-12:
        return 0.0
    return cagr(log_returns, periods_per_year) / mdd


def var_cvar(log_returns: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    if len(log_returns) == 0:
        return 0.0, 0.0
    var = float(np.percentile(log_returns, alpha * 100))
    cvar = float(log_returns[log_returns <= var].mean()) if (log_returns <= var).any() else var
    return var, cvar


def alpha_beta(
    log_returns: np.ndarray,
    benchmark_returns: np.ndarray,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252,
) -> tuple[float, float]:
    n = min(len(log_returns), len(benchmark_returns))
    if n < 2:
        return 0.0, 0.0
    r = log_returns[:n]
    b = benchmark_returns[:n]
    rf = risk_free_rate / periods_per_year
    excess_r = r - rf
    excess_b = b - rf
    cov = np.cov(excess_r, excess_b)
    if cov[1, 1] < 1e-12:
        return 0.0, 0.0
    beta = cov[0, 1] / cov[1, 1]
    alpha = float(excess_r.mean() - beta * excess_b.mean()) * periods_per_year
    return alpha, float(beta)


def compute_all_metrics(
    log_returns: np.ndarray,
    turnovers: np.ndarray | None = None,
    benchmark_returns: np.ndarray | None = None,
    risk_free_rate: float = 0.02,
) -> dict:
    var, cvar = var_cvar(log_returns)
    metrics = {
        "cumulative_return": cumulative_return(log_returns),
        "cagr": cagr(log_returns),
        "volatility": annualized_volatility(log_returns),
        "sharpe": sharpe_ratio(log_returns, risk_free_rate),
        "sortino": sortino_ratio(log_returns, risk_free_rate),
        "max_drawdown": max_drawdown(log_returns),
        "calmar": calmar_ratio(log_returns),
        "var_95": var,
        "cvar_95": cvar,
    }
    if turnovers is not None:
        metrics["mean_turnover"] = float(turnovers.mean())
        metrics["total_turnover"] = float(turnovers.sum())
    if benchmark_returns is not None:
        a, b = alpha_beta(log_returns, benchmark_returns, risk_free_rate)
        metrics["alpha"] = a
        metrics["beta"] = b
    return metrics
