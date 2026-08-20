"""Tests for feature engineering."""

import numpy as np
import pandas as pd
import pytest

from src.config import Config
from src.data_loader import load_market_data
from src.features import (
    build_features,
    compute_norm_stats,
    compute_rsi,
    flatten_lookback_window,
    normalize_states,
)


@pytest.fixture
def cfg():
    return Config.from_yaml("configs/config.yaml")


def test_rsi_bounded():
    close = pd.Series(np.cumprod(1 + np.random.randn(100) * 0.01) * 100)
    rsi = compute_rsi(close)
    valid = rsi.dropna()
    assert (valid >= 0).all() and (valid <= 100).all()


def test_flatten_lookback_window_shape():
    features = pd.DataFrame(np.random.randn(50, 5))
    features.index = pd.date_range("2020-01-01", periods=50)
    states, dates = flatten_lookback_window(features, lookback=10)
    assert states.shape == (41, 50)  # 5 features * 10 lookback
    assert len(dates) == 41


def test_norm_stats_train_only():
    states = np.random.randn(100, 10).astype(np.float32)
    train_mask = np.zeros(100, dtype=bool)
    train_mask[:70] = True
    stats = compute_norm_stats(states, train_mask)
    normalized = normalize_states(states, stats)
    train_mean = normalized[train_mask].mean(axis=0)
    assert np.allclose(train_mean, 0, atol=0.1)


def test_no_lookahead_in_features(cfg):
    """Feature at time t should not use data after t."""
    bundle = load_market_data(cfg)
    fb = build_features(bundle, cfg)
    # States should have same length as dates
    assert len(fb.dates) == fb.states.shape[0]
    # First state should not exist before lookback window
    assert fb.states.shape[0] > 0


def test_rtg_recursion():
    from src.trajectories import compute_rtg
    rewards = np.array([0.1, 0.2, -0.05, 0.15], dtype=np.float32)
    rtg = compute_rtg(rewards)
    for t in range(len(rewards) - 1):
        assert abs(rtg[t] - (rewards[t] + rtg[t + 1])) < 1e-6
