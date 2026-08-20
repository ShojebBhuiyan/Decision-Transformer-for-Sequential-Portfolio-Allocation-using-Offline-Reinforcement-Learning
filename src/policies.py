"""Classical portfolio policies and behavior-policy mixture."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Optional

import numpy as np
from sklearn.covariance import LedoitWolf

from src.env import equal_weights, project_to_simplex


class Policy(ABC):
    """Base policy interface."""

    name: str = "base"

    @abstractmethod
    def __call__(self, t: int, state: np.ndarray, env) -> np.ndarray:
        ...


class BuyAndHold(Policy):
    name = "buy_and_hold"

    def __init__(self, n_assets: int):
        self.weights = equal_weights(n_assets)

    def __call__(self, t: int, state: np.ndarray, env) -> np.ndarray:
        return self.weights.copy()


class EqualWeight(Policy):
    name = "equal_weight"

    def __init__(self, n_assets: int):
        self.weights = equal_weights(n_assets)

    def __call__(self, t: int, state: np.ndarray, env) -> np.ndarray:
        return self.weights.copy()


class Momentum(Policy):
    name = "momentum"

    def __init__(self, n_assets: int, lookback: int = 60, top_k: Optional[int] = None):
        self.n_assets = n_assets
        self.lookback = lookback
        self.top_k = top_k or max(1, n_assets // 3)

    def __call__(self, t: int, state: np.ndarray, env) -> np.ndarray:
        if t < self.lookback:
            return equal_weights(self.n_assets)
        rets = env.price_returns[t - self.lookback : t]
        cum = np.prod(rets, axis=0) - 1
        top_idx = np.argsort(cum)[-self.top_k :]
        w = np.zeros(self.n_assets)
        w[top_idx] = 1.0 / self.top_k
        return w


def _optimize_weights(
    returns_window: np.ndarray,
    objective: str = "max_sharpe",
    max_iter: int = 200,
) -> np.ndarray:
    """Long-only portfolio optimization on simplex."""
    n = returns_window.shape[1]
    mu = returns_window.mean(axis=0)
    lw = LedoitWolf().fit(returns_window)
    cov = lw.covariance_

    w = equal_weights(n)
    lr = 0.05
    for _ in range(max_iter):
        if objective == "min_var":
            grad = 2 * cov @ w
        else:  # max_sharpe proxy: maximize mu - 0.5 * w^T cov w
            grad = -(mu - cov @ w)
        w = project_to_simplex(w - lr * grad)
    return w


class MeanVariance(Policy):
    name = "mean_variance"

    def __init__(self, n_assets: int, lookback: int = 252):
        self.n_assets = n_assets
        self.lookback = lookback

    def __call__(self, t: int, state: np.ndarray, env) -> np.ndarray:
        if t < self.lookback:
            return equal_weights(self.n_assets)
        window = env.price_returns[t - self.lookback : t]
        return _optimize_weights(window, objective="max_sharpe")


class MinVariance(Policy):
    name = "min_variance"

    def __init__(self, n_assets: int, lookback: int = 252):
        self.n_assets = n_assets
        self.lookback = lookback

    def __call__(self, t: int, state: np.ndarray, env) -> np.ndarray:
        if t < self.lookback:
            return equal_weights(self.n_assets)
        window = env.price_returns[t - self.lookback : t]
        return _optimize_weights(window, objective="min_var")


class RiskParity(Policy):
    name = "risk_parity"

    def __init__(self, n_assets: int, lookback: int = 60):
        self.n_assets = n_assets
        self.lookback = lookback

    def __call__(self, t: int, state: np.ndarray, env) -> np.ndarray:
        if t < self.lookback:
            return equal_weights(self.n_assets)
        window = env.price_returns[t - self.lookback : t]
        vols = window.std(axis=0) + 1e-8
        inv_vol = 1.0 / vols
        w = inv_vol / inv_vol.sum()
        return w


class NoisyPolicy(Policy):
    """Wrap a base policy with Dirichlet noise."""

    name = "noisy"

    def __init__(self, base: Policy, alpha: float = 10.0, rng: Optional[np.random.Generator] = None):
        self.base = base
        self.alpha = alpha
        self.rng = rng or np.random.default_rng()

    def __call__(self, t: int, state: np.ndarray, env) -> np.ndarray:
        base_w = self.base(t, state, env)
        noise = self.rng.dirichlet(self.alpha * base_w + 0.1)
        return noise / noise.sum()


class RandomDirichlet(Policy):
    name = "random_dirichlet"

    def __init__(self, n_assets: int, alpha: float = 1.0, rng: Optional[np.random.Generator] = None):
        self.n_assets = n_assets
        self.alpha = alpha
        self.rng = rng or np.random.default_rng()

    def __call__(self, t: int, state: np.ndarray, env) -> np.ndarray:
        return self.rng.dirichlet(np.ones(self.n_assets) * self.alpha)


def get_behavior_policies(n_assets: int, seed: int = 42) -> list[Policy]:
    """Return the full behavior-policy mixture."""
    rng = np.random.default_rng(seed)
    policies: list[Policy] = [
        BuyAndHold(n_assets),
        EqualWeight(n_assets),
        MeanVariance(n_assets),
        MinVariance(n_assets),
        RiskParity(n_assets),
        Momentum(n_assets),
    ]
    # Noisy variants at three noise levels
    for base in [MeanVariance(n_assets), Momentum(n_assets), EqualWeight(n_assets)]:
        for alpha in (5.0, 15.0, 50.0):
            policies.append(NoisyPolicy(base, alpha=alpha, rng=rng))
    # Random Dirichlet rollouts
    for alpha in (0.5, 1.0, 2.0):
        policies.append(RandomDirichlet(n_assets, alpha=alpha, rng=rng))
    return policies


def get_classical_baselines(n_assets: int) -> dict[str, Policy]:
    return {
        "buy_and_hold": BuyAndHold(n_assets),
        "equal_weight": EqualWeight(n_assets),
        "mean_variance": MeanVariance(n_assets),
        "min_variance": MinVariance(n_assets),
        "risk_parity": RiskParity(n_assets),
        "momentum": Momentum(n_assets),
    }
