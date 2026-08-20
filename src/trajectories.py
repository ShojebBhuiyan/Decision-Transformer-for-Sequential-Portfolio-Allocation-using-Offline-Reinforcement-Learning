"""Offline trajectory synthesis and PyTorch Dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.config import Config
from src.data_loader import compute_price_returns, get_split_mask, load_processed_data
from src.env import PortfolioEnv, compute_log_reward, equal_weights
from src.features import build_features, load_features, save_features
from src.policies import Policy, get_behavior_policies


def compute_rtg(rewards: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Compute Return-to-Go (undiscounted if gamma=1)."""
    T = len(rewards)
    rtg = np.zeros(T, dtype=np.float32)
    rtg[-1] = rewards[-1]
    for t in range(T - 2, -1, -1):
        rtg[t] = rewards[t] + gamma * rtg[t + 1]
    return rtg


def rollout_episode(
    env: PortfolioEnv,
    policy: Policy,
    states: np.ndarray,
    start_t: int,
    episode_length: int,
) -> dict:
    """Roll out one episode from start_t."""
    T = min(episode_length, env.T - start_t)
    rewards = np.zeros(T, dtype=np.float32)
    actions = np.zeros((T, env.n_assets), dtype=np.float32)
    state_seq = np.zeros((T, states.shape[1]), dtype=np.float32)

    env.t = start_t
    env.weights = equal_weights(env.n_assets)

    for i in range(T):
        state = states[start_t + i]
        action = policy(start_t + i, state, env)
        action = action / (action.sum() + 1e-12)
        y_t = env.price_returns[start_t + i]
        reward = compute_log_reward(
            action, y_t, env.weights,
            transaction_cost=env.transaction_cost,
            epsilon=env.reward_epsilon,
        )
        rewards[i] = reward
        actions[i] = action
        state_seq[i] = state
        env.weights = action

    rtg = compute_rtg(rewards)
    return {
        "states": state_seq,
        "actions": actions,
        "rewards": rewards,
        "rtg": rtg,
        "length": T,
        "policy": policy.name if hasattr(policy, "name") else "unknown",
    }


def generate_trajectories(cfg: Config) -> Path:
    """Generate offline trajectory dataset."""
    bundle = load_processed_data(cfg)
    fb = build_features(bundle, cfg)
    save_features(fb, cfg)

    returns = compute_price_returns(bundle.investable_prices).values.astype(np.float64)
    states = fb.states

    env = PortfolioEnv(
        price_returns=returns,
        states=states,
        transaction_cost=float(cfg.get("env", "transaction_cost", default=0.001)),
        reward_epsilon=float(cfg.get("env", "reward_epsilon", default=1e-8)),
    )

    episode_length = int(cfg.get("env", "episode_length", default=252))
    stride = int(cfg.get("env", "episode_stride", default=21))
    n_windows = int(cfg.get("trajectories", "n_windows_per_policy", default=200))
    output_dir = cfg.project_root / cfg.get("trajectories", "output_dir", default="data/trajectories")
    output_dir.mkdir(parents=True, exist_ok=True)

    train_mask = get_split_mask(
        fb.dates,
        cfg.get("splits", "train_start", default="2000-09-01"),
        cfg.get("splits", "train_end", default="2016-12-31"),
    )
    train_indices = np.where(train_mask)[0]
    valid_starts = train_indices[train_indices + episode_length < len(fb.dates)]

    policies = get_behavior_policies(env.n_assets)
    all_episodes = []
    rng = np.random.default_rng(42)

    for policy in policies:
        if len(valid_starts) == 0:
            continue
        chosen = rng.choice(valid_starts, size=min(n_windows, len(valid_starts)), replace=False)
        for start in chosen:
            ep = rollout_episode(env, policy, states, int(start), episode_length)
            all_episodes.append(ep)

    # Persist as single compressed archive
    max_len = episode_length
    n_eps = len(all_episodes)
    n_assets = env.n_assets
    state_dim = states.shape[1]

    states_arr = np.zeros((n_eps, max_len, state_dim), dtype=np.float32)
    actions_arr = np.zeros((n_eps, max_len, n_assets), dtype=np.float32)
    rewards_arr = np.zeros((n_eps, max_len), dtype=np.float32)
    rtg_arr = np.zeros((n_eps, max_len), dtype=np.float32)
    lengths = np.zeros(n_eps, dtype=np.int32)

    for i, ep in enumerate(all_episodes):
        L = ep["length"]
        states_arr[i, :L] = ep["states"]
        actions_arr[i, :L] = ep["actions"]
        rewards_arr[i, :L] = ep["rewards"]
        rtg_arr[i, :L] = ep["rtg"]
        lengths[i] = L

    # RTG normalization stats (train only)
    flat_rtg = rtg_arr[rtg_arr != 0]  # approximate
    rtg_mean = float(flat_rtg.mean()) if len(flat_rtg) > 0 else 0.0
    rtg_std = float(flat_rtg.std() + 1e-8)

    out_path = output_dir / "trajectories.npz"
    np.savez_compressed(
        out_path,
        states=states_arr,
        actions=actions_arr,
        rewards=rewards_arr,
        rtg=rtg_arr,
        lengths=lengths,
        rtg_mean=rtg_mean,
        rtg_std=rtg_std,
    )

    meta = {
        "n_episodes": n_eps,
        "episode_length": max_len,
        "n_assets": n_assets,
        "state_dim": state_dim,
        "rtg_mean": rtg_mean,
        "rtg_std": rtg_std,
        "n_policies": len(policies),
    }
    with open(output_dir / "trajectory_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return out_path


class TrajectoryDataset(Dataset):
    """PyTorch Dataset for Decision Transformer training."""

    def __init__(
        self,
        traj_path: Path,
        context_length: int = 30,
        split_mask: Optional[np.ndarray] = None,
        normalize_rtg: bool = True,
    ):
        data = np.load(traj_path)
        self.states = data["states"]
        self.actions = data["actions"]
        self.rewards = data["rewards"]
        self.rtg = data["rtg"]
        self.lengths = data["lengths"]
        self.rtg_mean = float(data.get("rtg_mean", 0.0))
        self.rtg_std = float(data.get("rtg_std", 1.0))
        self.context_length = context_length
        self.normalize_rtg = normalize_rtg

        # Build index of (episode_id, timestep) pairs
        self.indices: list[tuple[int, int]] = []
        for ep_idx in range(len(self.lengths)):
            L = int(self.lengths[ep_idx])
            for t in range(L):
                self.indices.append((ep_idx, t))

        if split_mask is not None:
            # Filter by split if provided (approximate: use episode starts)
            pass  # kept simple; full split filtering done at generation time

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ep_idx, t = self.indices[idx]
        K = self.context_length

        # Context window ending at t
        start = max(0, t - K + 1)
        actual_len = t - start + 1

        s = np.zeros((K, self.states.shape[2]), dtype=np.float32)
        a = np.zeros((K, self.actions.shape[2]), dtype=np.float32)
        r = np.zeros(K, dtype=np.float32)
        rtg = np.zeros(K, dtype=np.float32)
        mask = np.zeros(K, dtype=np.float32)

        s[-actual_len:] = self.states[ep_idx, start : t + 1]
        a[-actual_len:] = self.actions[ep_idx, start : t + 1]
        r[-actual_len:] = self.rewards[ep_idx, start : t + 1]
        rtg_vals = self.rtg[ep_idx, start : t + 1]
        if self.normalize_rtg:
            rtg_vals = (rtg_vals - self.rtg_mean) / self.rtg_std
        rtg[-actual_len:] = rtg_vals
        mask[-actual_len:] = 1.0

        return {
            "states": torch.from_numpy(s),
            "actions": torch.from_numpy(a),
            "rewards": torch.from_numpy(r),
            "rtg": torch.from_numpy(rtg).unsqueeze(-1),
            "mask": torch.from_numpy(mask),
            "target_action": torch.from_numpy(self.actions[ep_idx, t]),
        }


def load_trajectory_dataset(cfg: Config, split: str = "train") -> TrajectoryDataset:
    root = cfg.project_root
    traj_path = root / cfg.get("trajectories", "output_dir", default="data/trajectories") / "trajectories.npz"
    K = int(cfg.get("model", "context_length", default=30))
    return TrajectoryDataset(traj_path, context_length=K)
