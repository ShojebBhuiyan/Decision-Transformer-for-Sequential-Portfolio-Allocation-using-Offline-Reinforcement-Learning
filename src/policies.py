"""Classical portfolio policies and behavior-policy mixture."""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.covariance import LedoitWolf

from src.config import Config
from src.env import equal_weights, project_to_simplex, project_to_simplex_batch


class Policy(ABC):
    """Base policy interface."""

    name: str = "base"

    @abstractmethod
    def __call__(self, t: int, state: np.ndarray, env) -> np.ndarray:
        ...

    def schedule_key(self) -> str:
        """Unique key for precomputed weight schedule caching."""
        return self.name

    def precompute(self, price_returns: np.ndarray, n_dates: int) -> np.ndarray | None:
        """Return (n_dates, n_assets) weight schedule, or None if stochastic."""
        return None


def _optimize_weights(
    returns_window: np.ndarray,
    objective: str = "max_sharpe",
    max_iter: int = 200,
) -> np.ndarray:
    """Long-only portfolio optimization on simplex (single window)."""
    return _optimize_weights_batch(
        returns_window.mean(axis=0, keepdims=True),
        np.array([LedoitWolf().fit(returns_window).covariance_]),
        objective=objective,
        max_iter=max_iter,
    )[0]


def _optimize_weights_batch(
    mus: np.ndarray,
    covs: np.ndarray,
    objective: str = "max_sharpe",
    max_iter: int = 200,
    lr: float = 0.05,
) -> np.ndarray:
    """
    Batched long-only portfolio optimization on the simplex.

    Args:
        mus: (B, N) mean returns per date
        covs: (B, N, N) covariance matrices per date
    """
    B, n = mus.shape
    w = np.tile(equal_weights(n), (B, 1))
    for _ in range(max_iter):
        if objective == "min_var":
            grad = 2 * np.einsum("bij,bj->bi", covs, w)
        else:
            grad = -(mus - np.einsum("bij,bj->bi", covs, w))
        w = project_to_simplex_batch(w - lr * grad)
    return w


class BuyAndHold(Policy):
    name = "buy_and_hold"

    def __init__(self, n_assets: int):
        self.n_assets = n_assets
        self.weights = equal_weights(n_assets)

    def __call__(self, t: int, state: np.ndarray, env) -> np.ndarray:
        return self.weights.copy()

    def precompute(self, price_returns: np.ndarray, n_dates: int) -> np.ndarray:
        return np.tile(self.weights, (n_dates, 1))


class EqualWeight(Policy):
    name = "equal_weight"

    def __init__(self, n_assets: int):
        self.n_assets = n_assets
        self.weights = equal_weights(n_assets)

    def __call__(self, t: int, state: np.ndarray, env) -> np.ndarray:
        return self.weights.copy()

    def precompute(self, price_returns: np.ndarray, n_dates: int) -> np.ndarray:
        return np.tile(self.weights, (n_dates, 1))


class Momentum(Policy):
    name = "momentum"

    def __init__(self, n_assets: int, lookback: int = 60, top_k: Optional[int] = None):
        self.n_assets = n_assets
        self.lookback = lookback
        self.top_k = top_k or max(1, n_assets // 3)

    def schedule_key(self) -> str:
        return f"momentum_{self.lookback}_{self.top_k}"

    def __call__(self, t: int, state: np.ndarray, env) -> np.ndarray:
        if t < self.lookback:
            return equal_weights(self.n_assets)
        rets = env.price_returns[t - self.lookback : t]
        cum = np.prod(rets, axis=0) - 1
        top_idx = np.argsort(cum)[-self.top_k :]
        w = np.zeros(self.n_assets)
        w[top_idx] = 1.0 / self.top_k
        return w

    def precompute(self, price_returns: np.ndarray, n_dates: int) -> np.ndarray:
        W = np.tile(equal_weights(self.n_assets), (n_dates, 1))
        for t in range(self.lookback, n_dates):
            rets = price_returns[t - self.lookback : t]
            cum = np.prod(rets, axis=0) - 1
            top_idx = np.argsort(cum)[-self.top_k :]
            w = np.zeros(self.n_assets)
            w[top_idx] = 1.0 / self.top_k
            W[t] = w
        return W


class MeanVariance(Policy):
    name = "mean_variance"

    def __init__(self, n_assets: int, lookback: int = 252):
        self.n_assets = n_assets
        self.lookback = lookback

    def schedule_key(self) -> str:
        return f"mean_variance_{self.lookback}"

    def __call__(self, t: int, state: np.ndarray, env) -> np.ndarray:
        if t < self.lookback:
            return equal_weights(self.n_assets)
        window = env.price_returns[t - self.lookback : t]
        return _optimize_weights(window, objective="max_sharpe")

    def precompute(self, price_returns: np.ndarray, n_dates: int) -> np.ndarray:
        return _precompute_optimizer_schedule(
            price_returns, n_dates, self.lookback, self.n_assets, "max_sharpe"
        )


class MinVariance(Policy):
    name = "min_variance"

    def __init__(self, n_assets: int, lookback: int = 252):
        self.n_assets = n_assets
        self.lookback = lookback

    def schedule_key(self) -> str:
        return f"min_variance_{self.lookback}"

    def __call__(self, t: int, state: np.ndarray, env) -> np.ndarray:
        if t < self.lookback:
            return equal_weights(self.n_assets)
        window = env.price_returns[t - self.lookback : t]
        return _optimize_weights(window, objective="min_var")

    def precompute(self, price_returns: np.ndarray, n_dates: int) -> np.ndarray:
        return _precompute_optimizer_schedule(
            price_returns, n_dates, self.lookback, self.n_assets, "min_var"
        )


class RiskParity(Policy):
    name = "risk_parity"

    def __init__(self, n_assets: int, lookback: int = 60):
        self.n_assets = n_assets
        self.lookback = lookback

    def schedule_key(self) -> str:
        return f"risk_parity_{self.lookback}"

    def __call__(self, t: int, state: np.ndarray, env) -> np.ndarray:
        if t < self.lookback:
            return equal_weights(self.n_assets)
        window = env.price_returns[t - self.lookback : t]
        vols = window.std(axis=0) + 1e-8
        inv_vol = 1.0 / vols
        return inv_vol / inv_vol.sum()

    def precompute(self, price_returns: np.ndarray, n_dates: int) -> np.ndarray:
        W = np.tile(equal_weights(self.n_assets), (n_dates, 1))
        for t in range(self.lookback, n_dates):
            window = price_returns[t - self.lookback : t]
            vols = window.std(axis=0) + 1e-8
            inv_vol = 1.0 / vols
            W[t] = inv_vol / inv_vol.sum()
        return W


def _precompute_optimizer_schedule(
    price_returns: np.ndarray,
    n_dates: int,
    lookback: int,
    n_assets: int,
    objective: str,
    batch_size: int = 64,
) -> np.ndarray:
    """Precompute MVO/MinVar weights in batches of dates."""
    W = np.tile(equal_weights(n_assets), (n_dates, 1))
    dates = list(range(lookback, n_dates))
    for i in range(0, len(dates), batch_size):
        batch_dates = dates[i : i + batch_size]
        mus = []
        covs = []
        for t in batch_dates:
            window = price_returns[t - lookback : t]
            mus.append(window.mean(axis=0))
            covs.append(LedoitWolf().fit(window).covariance_)
        mus_arr = np.array(mus)
        covs_arr = np.array(covs)
        W[batch_dates] = _optimize_weights_batch(mus_arr, covs_arr, objective=objective)
    return W


class NoisyPolicy(Policy):
    """Wrap a base policy with Dirichlet noise (per-episode vectorized sampling)."""

    name = "noisy"

    def __init__(self, base: Policy, alpha: float = 10.0, rng: Optional[np.random.Generator] = None):
        self.base = base
        self.alpha = alpha
        self.rng = rng or np.random.default_rng()

    def schedule_key(self) -> str:
        return f"noisy_{self.base.schedule_key()}_{self.alpha}"

    def __call__(self, t: int, state: np.ndarray, env) -> np.ndarray:
        base_w = self.base(t, state, env)
        noise = self.rng.dirichlet(self.alpha * base_w + 0.1)
        return noise / noise.sum()

    def sample_episode_weights(
        self, base_schedule: np.ndarray, start: int, length: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Vectorized Dirichlet noise over an episode via gamma trick."""
        base_w = base_schedule[start : start + length]
        alpha_mat = self.alpha * base_w + 0.1
        g = rng.standard_gamma(alpha_mat)
        return g / g.sum(axis=1, keepdims=True)


class RandomDirichlet(Policy):
    name = "random_dirichlet"

    def __init__(self, n_assets: int, alpha: float = 1.0, rng: Optional[np.random.Generator] = None):
        self.n_assets = n_assets
        self.alpha = alpha
        self.rng = rng or np.random.default_rng()

    def schedule_key(self) -> str:
        return f"random_dirichlet_{self.alpha}"

    def __call__(self, t: int, state: np.ndarray, env) -> np.ndarray:
        return self.rng.dirichlet(np.ones(self.n_assets) * self.alpha)

    def sample_episode_weights(self, length: int, rng: np.random.Generator) -> np.ndarray:
        alpha_mat = np.ones((length, self.n_assets)) * self.alpha
        g = rng.standard_gamma(alpha_mat)
        return g / g.sum(axis=1, keepdims=True)


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
    for base in [MeanVariance(n_assets), Momentum(n_assets), EqualWeight(n_assets)]:
        for alpha in (5.0, 15.0, 50.0):
            policies.append(NoisyPolicy(base, alpha=alpha, rng=rng))
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


def _policy_weights_fingerprint(cfg: Config, n_assets: int, n_dates: int) -> str:
    payload = json.dumps({
        "universe": cfg.get("data", "universe", default="A"),
        "start_date": cfg.get("data", "start_date"),
        "end_date": cfg.get("data", "end_date"),
        "n_assets": n_assets,
        "n_dates": n_dates,
    }, sort_keys=True)
    return hashlib.md5(payload.encode()).hexdigest()


def precompute_all_schedules(
    policies: list[Policy],
    price_returns: np.ndarray,
    cfg: Optional[Config] = None,
) -> dict[str, np.ndarray]:
    """
    Precompute weight schedules for all policies.

    Deterministic policies share schedules by schedule_key.
    Stochastic policies are keyed individually.
    """
    n_dates = price_returns.shape[0]
    n_assets = price_returns.shape[1]
    cache_path = None
    fingerprint = None

    if cfg is not None:
        fingerprint = _policy_weights_fingerprint(cfg, n_assets, n_dates)
        cache_path = cfg.processed_dir() / "policy_weights.npz"
        meta_path = cfg.processed_dir() / "policy_weights_meta.json"
        if cache_path.exists() and meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if meta.get("fingerprint") == fingerprint:
                data = np.load(cache_path)
                return {k: data[k] for k in data.files}

    base_cache: dict[str, np.ndarray] = {}
    schedules: dict[str, np.ndarray] = {}

    for policy in policies:
        key = policy.schedule_key()
        if isinstance(policy, (NoisyPolicy, RandomDirichlet)):
            continue
        if key not in base_cache:
            sched = policy.precompute(price_returns, n_dates)
            if sched is not None:
                base_cache[key] = sched.astype(np.float64)
        if key in base_cache:
            schedules[key] = base_cache[key]

    if cfg is not None and cache_path is not None and fingerprint is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, **schedules)
        with open(cfg.processed_dir() / "policy_weights_meta.json", "w") as f:
            json.dump({"fingerprint": fingerprint, "keys": list(schedules.keys())}, f, indent=2)

    return schedules
