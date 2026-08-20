"""Portfolio allocation environment and simplex utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


def project_to_simplex(v: np.ndarray) -> np.ndarray:
    """Project vector onto probability simplex (Duchi et al. 2008)."""
    if v.ndim == 1:
        return project_to_simplex_batch(v.reshape(1, -1))[0]
    return project_to_simplex_batch(v)


def project_to_simplex_batch(V: np.ndarray) -> np.ndarray:
    """Project each row of V onto the probability simplex (Duchi et al. 2008)."""
    if V.ndim == 1:
        V = V.reshape(1, -1)
    n = V.shape[1]
    u = np.sort(V, axis=1)[:, ::-1]
    cssv = np.cumsum(u, axis=1)
    ranks = np.arange(1, n + 1, dtype=V.dtype)
    cond = u * ranks > (cssv - 1)
    # Last True index per row
    rho = cond.sum(axis=1) - 1
    rho = np.clip(rho, 0, n - 1)
    theta = (cssv[np.arange(V.shape[0]), rho] - 1) / (rho + 1)
    w = np.maximum(V - theta[:, None], 0)
    return w / (w.sum(axis=1, keepdims=True) + 1e-12)


def equal_weights(n: int) -> np.ndarray:
    return np.ones(n, dtype=np.float64) / n


def compute_turnover(w_new: np.ndarray, w_old: np.ndarray) -> float:
    return float(np.abs(w_new - w_old).sum())


def compute_log_reward(
    action: np.ndarray,
    price_returns: np.ndarray,
    prev_weights: np.ndarray,
    transaction_cost: float = 0.001,
    epsilon: float = 1e-8,
) -> float:
    """
    Net logarithmic return with proportional transaction costs.

    R_t = ln(a_t^T y_t - mu * sum|w_{i,t} - w'_{i,t-1}|)
    """
    turnover = compute_turnover(action, prev_weights)
    gross = float(np.dot(action, price_returns))
    net = gross - transaction_cost * turnover
    net = max(net, epsilon)
    return float(np.log(net))


@dataclass
class PortfolioEnv:
    """Multi-asset portfolio environment for backtesting and RL."""

    price_returns: np.ndarray  # (T, N)
    states: Optional[np.ndarray] = None  # (T, state_dim)
    transaction_cost: float = 0.001
    reward_epsilon: float = 1e-8
    n_assets: int = 0

    def __post_init__(self) -> None:
        self.T = self.price_returns.shape[0]
        self.n_assets = self.price_returns.shape[1]
        self.reset()

    def reset(self) -> np.ndarray:
        self.t = 0
        self.weights = equal_weights(self.n_assets)
        self.cumulative_log_return = 0.0
        self.history: list[dict] = []
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        if self.states is not None:
            if self.t >= len(self.states):
                return np.zeros(self.states.shape[1], dtype=np.float32)
            return self.states[self.t]
        # Minimal state: previous weights + latest returns
        latest_ret = self.price_returns[self.t] if self.t < self.T else np.zeros(self.n_assets)
        return np.concatenate([self.weights, latest_ret])

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        """Execute one rebalancing step."""
        action = np.asarray(action, dtype=np.float64)
        action = action / (action.sum() + 1e-12)  # normalize to simplex

        y_t = self.price_returns[self.t]
        reward = compute_log_reward(
            action, y_t, self.weights,
            transaction_cost=self.transaction_cost,
            epsilon=self.reward_epsilon,
        )

        turnover = compute_turnover(action, self.weights)
        info = {
            "turnover": turnover,
            "weights": action.copy(),
            "prev_weights": self.weights.copy(),
            "price_returns": y_t.copy(),
            "gross_return": float(np.dot(action, y_t)),
        }

        self.weights = action
        self.cumulative_log_return += reward
        self.history.append({"t": self.t, "reward": reward, **info})

        self.t += 1
        done = self.t >= self.T
        if done:
            next_state = np.zeros(self.states.shape[1] if self.states is not None else self.n_assets * 2, dtype=np.float32)
        else:
            next_state = self._get_state()
        return next_state, reward, done, info

    def run_policy(self, policy_fn) -> dict:
        """Run a policy function weights = policy_fn(t, state, env) over full episode."""
        self.reset()
        rewards = []
        while self.t < self.T:
            state = self._get_state()
            action = policy_fn(self.t, state, self)
            _, reward, done, _ = self.step(action)
            rewards.append(reward)
            if done:
                break
        return {
            "rewards": np.array(rewards),
            "cumulative_log_return": self.cumulative_log_return,
            "history": self.history,
        }
