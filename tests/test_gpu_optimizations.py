"""Tests for GPU pipeline optimizations."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.env import PortfolioEnv, equal_weights, project_to_simplex, project_to_simplex_batch
from src.policies import (
    BuyAndHold,
    EqualWeight,
    MeanVariance,
    MinVariance,
    Momentum,
    RiskParity,
    get_behavior_policies,
    precompute_all_schedules,
)
from src.trajectories import (
    FORMAT_VERSION,
    compute_rtg,
    rollout_episode,
    rollout_episode_vectorized,
)


torch = pytest.importorskip("torch")
from src.models.decision_transformer import DecisionTransformer


def test_project_to_simplex_batch_matches_1d():
    rng = np.random.default_rng(0)
    V = rng.standard_normal((20, 8))
    batch = project_to_simplex_batch(V)
    for i in range(V.shape[0]):
        w = project_to_simplex(V[i])
        assert np.allclose(batch[i], w, atol=1e-10)
    assert np.allclose(batch.sum(axis=1), 1.0, atol=1e-10)


def test_rtg_recursion():
    rewards = np.array([0.1, -0.05, 0.02, 0.0], dtype=np.float32)
    rtg = compute_rtg(rewards)
    for t in range(len(rewards) - 1):
        assert abs(rtg[t] - (rewards[t] + rtg[t + 1])) < 1e-6
    assert abs(rtg[-1] - rewards[-1]) < 1e-6


def test_vectorized_rollout_matches_env():
    """Vectorized rollout equals step-by-step PortfolioEnv rollout."""
    rng = np.random.default_rng(42)
    T, N = 300, 5
    returns = rng.uniform(0.98, 1.02, (T, N))
    states = rng.standard_normal((T, 20)).astype(np.float32)
    env = PortfolioEnv(
        price_returns=returns,
        states=states,
        transaction_cost=0.001,
        reward_epsilon=1e-8,
    )
    start = 60
    episode_length = 50

    deterministic = [
        BuyAndHold(N),
        EqualWeight(N),
        Momentum(N),
        MeanVariance(N, lookback=30),
        MinVariance(N, lookback=30),
        RiskParity(N, lookback=30),
    ]
    schedules = precompute_all_schedules(deterministic, returns)

    for policy in deterministic:
        key = policy.schedule_key()
        sched = schedules[key]
        actions = sched[start : start + episode_length]
        vec = rollout_episode_vectorized(
            actions, returns, start, env.transaction_cost, env.reward_epsilon
        )
        ref = rollout_episode(env, policy, states, start, episode_length, weight_schedule=sched)
        assert np.allclose(vec["actions"], ref["actions"], atol=1e-6)
        assert np.allclose(vec["rewards"], ref["rewards"], atol=1e-6)
        assert np.allclose(vec["rtg"], ref["rtg"], atol=1e-5)


def test_precomputed_schedules_match_policy_calls():
    rng = np.random.default_rng(7)
    T, N = 400, 6
    returns = rng.uniform(0.97, 1.03, (T, N))
    states = rng.standard_normal((T, 30)).astype(np.float32)
    env = PortfolioEnv(price_returns=returns, states=states)

    policies = [
        MeanVariance(N, lookback=60),
        Momentum(N, lookback=40),
        RiskParity(N, lookback=40),
    ]
    schedules = precompute_all_schedules(policies, returns)
    sample_dates = [80, 120, 200, 350]

    for policy in policies:
        key = policy.schedule_key()
        sched = schedules[key]
        for t in sample_dates:
            direct = policy(t, states[t], env)
            precomp = sched[t]
            assert np.allclose(direct, precomp, atol=1e-6), f"{key} mismatch at t={t}"


def test_attention_no_nan_on_fully_padded_row():
    """Regression: right-aligned padding must not produce NaN attention."""
    B, K, state_dim, action_dim = 2, 5, 20, 4
    model = DecisionTransformer(state_dim, action_dim, context_length=K)
    states = torch.randn(B, K, state_dim)
    actions = torch.softmax(torch.randn(B, K, action_dim), dim=-1)
    rtg = torch.randn(B, K, 1)
    mask = torch.zeros(B, K)
    mask[:, -1] = 1.0  # only last position valid

    pred = model(states, actions, rtg, mask)
    assert not torch.isnan(pred).any()
    assert torch.allclose(pred.sum(dim=-1), torch.ones(B, K), atol=1e-5)


def test_format_v2_roundtrip():
    """Episode states reconstructed from global states match slicing."""
    n_dates, state_dim, n_assets = 100, 10, 4
    n_eps, max_len = 5, 20
    global_states = np.random.randn(n_dates, state_dim).astype(np.float32)
    episode_starts = np.array([10, 30, 50, 70, 80], dtype=np.int32)
    actions = np.random.dirichlet(np.ones(n_assets), size=(n_eps, max_len)).astype(np.float32)
    rewards = np.random.randn(n_eps, max_len).astype(np.float32)
    rtg = np.random.randn(n_eps, max_len).astype(np.float32)
    lengths = np.full(n_eps, max_len, dtype=np.int32)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "traj.npz"
        np.savez_compressed(
            path,
            states=global_states,
            episode_starts=episode_starts,
            actions=actions,
            rewards=rewards,
            rtg=rtg,
            lengths=lengths,
            policy_ids=np.arange(n_eps, dtype=np.int32),
            policy_names=np.array(["test"] * n_eps),
            rtg_mean=np.float32(0.0),
            rtg_std=np.float32(1.0),
            format_version=np.int32(FORMAT_VERSION),
        )

        from src.trajectories import TrajectoryDataset

        ds = TrajectoryDataset(path, context_length=5)
        for ep_idx in range(n_eps):
            start = int(episode_starts[ep_idx])
            for t in range(int(lengths[ep_idx])):
                flat_idx = ds.indices.index((ep_idx, t))
                sample = ds[flat_idx]
                g_t = start + t
                assert np.allclose(sample["states"][-1].numpy(), global_states[g_t], atol=1e-6)


def test_noisy_policies_share_base_schedule():
    policies = get_behavior_policies(5, seed=0)
    returns = np.random.uniform(0.99, 1.01, (200, 5))
    schedules = precompute_all_schedules(policies, returns)
    mvo_keys = [k for k in schedules if k.startswith("mean_variance")]
    assert len(mvo_keys) == 1
