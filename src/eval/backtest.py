"""Backtesting engine for portfolio strategies."""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd

from src.env import PortfolioEnv, equal_weights
from src.policies import Policy


def backtest_policy(
    price_returns: np.ndarray,
    policy: Policy | Callable,
    states: Optional[np.ndarray] = None,
    transaction_cost: float = 0.001,
    reward_epsilon: float = 1e-8,
) -> dict:
    """Run a single policy over the full price return series."""
    env = PortfolioEnv(
        price_returns=price_returns,
        states=states,
        transaction_cost=transaction_cost,
        reward_epsilon=reward_epsilon,
    )

    if isinstance(policy, Policy):
        policy_fn = policy
    else:
        policy_fn = policy

    result = env.run_policy(policy_fn)
    weights_history = np.array([h["weights"] for h in result["history"]])
    turnovers = np.array([h["turnover"] for h in result["history"]])

    log_returns = result["rewards"]
    cumulative = np.cumsum(log_returns)
    wealth = np.exp(cumulative)

    return {
        "log_returns": log_returns,
        "cumulative_log_return": result["cumulative_log_return"],
        "wealth_index": wealth,
        "weights": weights_history,
        "turnovers": turnovers,
        "mean_turnover": float(turnovers.mean()) if len(turnovers) > 0 else 0.0,
        "history": result["history"],
    }


def backtest_all_baselines(
    price_returns: np.ndarray,
    policies: dict[str, Policy],
    states: Optional[np.ndarray] = None,
    transaction_cost: float = 0.001,
) -> pd.DataFrame:
    """Backtest all classical baselines and return summary metrics."""
    from src.eval.metrics import compute_all_metrics

    rows = []
    for name, policy in policies.items():
        result = backtest_policy(price_returns, policy, states, transaction_cost)
        metrics = compute_all_metrics(result["log_returns"], turnovers=result["turnovers"])
        metrics["strategy"] = name
        rows.append(metrics)
    return pd.DataFrame(rows).set_index("strategy")
