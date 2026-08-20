"""Tests for portfolio environment."""

import numpy as np
import pytest

from src.env import (
    PortfolioEnv,
    compute_log_reward,
    compute_turnover,
    equal_weights,
    project_to_simplex,
    project_to_simplex_batch,
)


def test_equal_weights_sum_to_one():
    w = equal_weights(5)
    assert abs(w.sum() - 1.0) < 1e-10


def test_project_to_simplex():
    v = np.array([0.5, -0.2, 0.8, 0.1])
    w = project_to_simplex(v)
    assert abs(w.sum() - 1.0) < 1e-10
    assert (w >= 0).all()


def test_project_to_simplex_batch_rows():
    V = np.array([[0.5, -0.2, 0.8], [0.1, 0.1, 0.1]])
    W = project_to_simplex_batch(V)
    assert W.shape == V.shape
    assert np.allclose(W.sum(axis=1), 1.0)


def test_turnover():
    w_old = np.array([0.5, 0.5])
    w_new = np.array([0.6, 0.4])
    assert abs(compute_turnover(w_new, w_old) - 0.2) < 1e-10


def test_log_reward_hand_computed():
    action = np.array([0.6, 0.4])
    returns = np.array([1.02, 0.98])
    prev = np.array([0.5, 0.5])
    mu = 0.001
    turnover = 0.2
    gross = 0.6 * 1.02 + 0.4 * 0.98
    net = gross - mu * turnover
    expected = np.log(net)
    actual = compute_log_reward(action, returns, prev, transaction_cost=mu)
    assert abs(actual - expected) < 1e-10


def test_zero_cost_buy_and_hold():
    """Buy-and-hold with zero cost should match raw portfolio return."""
    returns = np.array([[1.01, 0.99], [1.02, 1.01], [0.98, 1.03]])
    n = 2
    env = PortfolioEnv(returns, transaction_cost=0.0, reward_epsilon=1e-8)
    w = equal_weights(n)

    def buy_hold(t, state, e):
        return w

    result = env.run_policy(buy_hold)
    expected_log = np.log(returns @ w).sum()
    assert abs(result["cumulative_log_return"] - expected_log) < 1e-6


def test_env_episode_length():
    returns = np.random.uniform(0.98, 1.02, (10, 3))
    env = PortfolioEnv(returns)
    env.reset()
    for _ in range(10):
        action = equal_weights(3)
        _, _, done, _ = env.step(action)
        if done:
            break
    assert env.t == 10
