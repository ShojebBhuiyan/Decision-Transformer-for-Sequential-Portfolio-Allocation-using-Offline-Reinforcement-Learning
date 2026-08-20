"""Offline trajectory synthesis, format v2 storage, and GPU batch sampler."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from src.config import Config
from src.data_loader import compute_price_returns, get_split_mask, load_processed_data
from src.env import PortfolioEnv, compute_log_reward, equal_weights
from src.features import build_or_load_features, save_features
from src.policies import (
    NoisyPolicy,
    Policy,
    RandomDirichlet,
    get_behavior_policies,
    precompute_all_schedules,
)

FORMAT_VERSION = 2


def compute_rtg(rewards: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Compute Return-to-Go (undiscounted if gamma=1)."""
    if len(rewards) == 0:
        return rewards.astype(np.float32)
    if gamma == 1.0:
        return np.cumsum(rewards[::-1])[::-1].astype(np.float32)
    T = len(rewards)
    rtg = np.zeros(T, dtype=np.float32)
    rtg[-1] = rewards[-1]
    for t in range(T - 2, -1, -1):
        rtg[t] = rewards[t] + gamma * rtg[t + 1]
    return rtg


def _episode_rewards_from_weights(
    actions: np.ndarray,
    price_returns: np.ndarray,
    start_t: int,
    transaction_cost: float,
    reward_epsilon: float,
) -> np.ndarray:
    """Vectorized reward computation matching PortfolioEnv.step."""
    L = actions.shape[0]
    Y = price_returns[start_t : start_t + L]
    n_assets = actions.shape[1]
    prev = np.vstack([equal_weights(n_assets), actions[:-1]])
    turnover = np.abs(actions - prev).sum(axis=1)
    gross = (actions * Y).sum(axis=1)
    net = np.maximum(gross - transaction_cost * turnover, reward_epsilon)
    return np.log(net).astype(np.float32)


def rollout_episode_vectorized(
    actions: np.ndarray,
    price_returns: np.ndarray,
    start_t: int,
    transaction_cost: float,
    reward_epsilon: float,
) -> dict:
    """Roll out episode from precomputed action weights (no per-day policy calls)."""
    L = actions.shape[0]
    rewards = _episode_rewards_from_weights(
        actions, price_returns, start_t, transaction_cost, reward_epsilon
    )
    rtg = compute_rtg(rewards)
    return {
        "actions": actions.astype(np.float32),
        "rewards": rewards,
        "rtg": rtg,
        "length": L,
    }


def rollout_episode(
    env: PortfolioEnv,
    policy: Policy,
    states: np.ndarray,
    start_t: int,
    episode_length: int,
    weight_schedule: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> dict:
    """Roll out one episode (step-by-step fallback for tests)."""
    T = min(episode_length, env.T - start_t)
    rewards = np.zeros(T, dtype=np.float32)
    actions = np.zeros((T, env.n_assets), dtype=np.float32)

    env.t = start_t
    env.weights = equal_weights(env.n_assets)

    for i in range(T):
        t = start_t + i
        if weight_schedule is not None:
            action = weight_schedule[t].copy()
        else:
            action = policy(t, states[t], env)
        action = action / (action.sum() + 1e-12)
        y_t = env.price_returns[t]
        reward = compute_log_reward(
            action, y_t, env.weights,
            transaction_cost=env.transaction_cost,
            epsilon=env.reward_epsilon,
        )
        rewards[i] = reward
        actions[i] = action
        env.weights = action

    rtg = compute_rtg(rewards)
    return {
        "actions": actions,
        "rewards": rewards,
        "rtg": rtg,
        "length": T,
        "policy": policy.name if hasattr(policy, "name") else "unknown",
    }


def _get_episode_actions(
    policy: Policy,
    schedules: dict[str, np.ndarray],
    price_returns: np.ndarray,
    start: int,
    episode_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Resolve episode action matrix for a policy."""
    L = min(episode_length, price_returns.shape[0] - start)

    if isinstance(policy, NoisyPolicy):
        base_key = policy.base.schedule_key()
        base_sched = schedules[base_key]
        return policy.sample_episode_weights(base_sched, start, L, rng).astype(np.float32)

    if isinstance(policy, RandomDirichlet):
        return policy.sample_episode_weights(L, rng).astype(np.float32)

    key = policy.schedule_key()
    sched = schedules[key]
    return sched[start : start + L].astype(np.float32)


def generate_trajectories(cfg: Config) -> Path:
    """Generate offline trajectory dataset (format v2)."""
    bundle = load_processed_data(cfg)
    fb = build_or_load_features(bundle, cfg)
    save_features(fb, cfg)

    returns = compute_price_returns(bundle.investable_prices).values.astype(np.float64)
    states = fb.states

    transaction_cost = float(cfg.get("env", "transaction_cost", default=0.001))
    reward_epsilon = float(cfg.get("env", "reward_epsilon", default=1e-8))
    episode_length = int(cfg.get("env", "episode_length", default=252))
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

    policies = get_behavior_policies(env_n_assets := returns.shape[1])
    schedules = precompute_all_schedules(policies, returns, cfg)
    rng = np.random.default_rng(42)

    print(f"Generating trajectories: {len(policies)} policies, {n_windows} windows each...")
    episode_starts_list: list[int] = []
    actions_list: list[np.ndarray] = []
    rewards_list: list[np.ndarray] = []
    rtg_list: list[np.ndarray] = []
    policy_ids: list[int] = []
    policy_names: list[str] = []

    for pi, policy in enumerate(policies):
        if len(valid_starts) == 0:
            continue
        chosen = rng.choice(valid_starts, size=min(n_windows, len(valid_starts)), replace=False)
        ep_rng = np.random.default_rng(42 + pi)
        for start in chosen:
            start = int(start)
            action_mat = _get_episode_actions(
                policy, schedules, returns, start, episode_length, ep_rng
            )
            ep = rollout_episode_vectorized(
                action_mat, returns, start, transaction_cost, reward_epsilon
            )
            episode_starts_list.append(start)
            actions_list.append(ep["actions"])
            rewards_list.append(ep["rewards"])
            rtg_list.append(ep["rtg"])
            policy_ids.append(pi)
            policy_names.append(policy.schedule_key())
        if (pi + 1) % 3 == 0:
            print(f"  Policy {pi+1}/{len(policies)}: {len(episode_starts_list)} episodes so far")

    n_eps = len(episode_starts_list)
    max_len = episode_length
    n_assets = env_n_assets

    actions_arr = np.zeros((n_eps, max_len, n_assets), dtype=np.float32)
    rewards_arr = np.zeros((n_eps, max_len), dtype=np.float32)
    rtg_arr = np.zeros((n_eps, max_len), dtype=np.float32)
    lengths = np.zeros(n_eps, dtype=np.int32)
    episode_starts = np.array(episode_starts_list, dtype=np.int32)
    policy_id_arr = np.array(policy_ids, dtype=np.int32)

    for i in range(n_eps):
        L = len(actions_list[i])
        actions_arr[i, :L] = actions_list[i]
        rewards_arr[i, :L] = rewards_list[i]
        rtg_arr[i, :L] = rtg_list[i]
        lengths[i] = L

    # RTG stats: mask by valid episode lengths
    valid_rtg = []
    for i in range(n_eps):
        valid_rtg.append(rtg_arr[i, : lengths[i]])
    flat_rtg = np.concatenate(valid_rtg) if valid_rtg else np.array([0.0])
    rtg_mean = float(flat_rtg.mean())
    rtg_std = float(flat_rtg.std() + 1e-8)

    out_path = output_dir / "trajectories.npz"
    np.savez_compressed(
        out_path,
        states=states.astype(np.float32),
        episode_starts=episode_starts,
        actions=actions_arr,
        rewards=rewards_arr,
        rtg=rtg_arr,
        lengths=lengths,
        policy_ids=policy_id_arr,
        policy_names=np.array(policy_names),
        rtg_mean=np.float32(rtg_mean),
        rtg_std=np.float32(rtg_std),
        format_version=np.int32(FORMAT_VERSION),
    )

    meta = {
        "format_version": FORMAT_VERSION,
        "n_episodes": n_eps,
        "episode_length": max_len,
        "n_assets": n_assets,
        "state_dim": states.shape[1],
        "n_dates": states.shape[0],
        "rtg_mean": rtg_mean,
        "rtg_std": rtg_std,
        "n_policies": len(policies),
    }
    with open(output_dir / "trajectory_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved {n_eps} episodes (format v{FORMAT_VERSION}) to {out_path}")
    return out_path


def _load_trajectory_npz(traj_path: Path) -> dict:
    data = np.load(traj_path, allow_pickle=True)
    result = {k: data[k] for k in data.files}
    fmt = int(result.get("format_version", 1))
    if fmt < FORMAT_VERSION:
        raise ValueError(
            f"Trajectory format v{fmt} is outdated. Re-run with --stage trajectories."
        )
    return result


class TrajectoryDataset(Dataset):
    """PyTorch Dataset for Decision Transformer training (format v2)."""

    def __init__(
        self,
        traj_path: Path,
        context_length: int = 30,
        split_mask: Optional[np.ndarray] = None,
        normalize_rtg: bool = True,
    ):
        data = _load_trajectory_npz(traj_path)
        self.global_states = data["states"]
        self.episode_starts = data["episode_starts"]
        self.actions = data["actions"]
        self.rewards = data["rewards"]
        self.rtg = data["rtg"]
        self.lengths = data["lengths"]
        self.rtg_mean = float(data.get("rtg_mean", 0.0))
        self.rtg_std = float(data.get("rtg_std", 1.0))
        self.context_length = context_length
        self.normalize_rtg = normalize_rtg
        self.state_dim = self.global_states.shape[1]
        self.n_assets = self.actions.shape[2]

        self.indices: list[tuple[int, int]] = []
        for ep_idx in range(len(self.lengths)):
            L = int(self.lengths[ep_idx])
            for t in range(L):
                self.indices.append((ep_idx, t))

    def __len__(self) -> int:
        return len(self.indices)

    def _get_state_at(self, ep_idx: int, global_t: int) -> np.ndarray:
        return self.global_states[global_t]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ep_idx, t = self.indices[idx]
        K = self.context_length
        ep_start = int(self.episode_starts[ep_idx])
        global_t = ep_start + t

        start_local = max(0, t - K + 1)
        actual_len = t - start_local + 1

        s = np.zeros((K, self.state_dim), dtype=np.float32)
        a = np.zeros((K, self.n_assets), dtype=np.float32)
        r = np.zeros(K, dtype=np.float32)
        rtg = np.zeros(K, dtype=np.float32)
        mask = np.zeros(K, dtype=np.float32)

        for j, local_t in enumerate(range(start_local, t + 1)):
            g_t = ep_start + local_t
            pos = K - actual_len + j
            s[pos] = self.global_states[g_t]
            a[pos] = self.actions[ep_idx, local_t]
            r[pos] = self.rewards[ep_idx, local_t]
            rtg_val = self.rtg[ep_idx, local_t]
            if self.normalize_rtg:
                rtg_val = (rtg_val - self.rtg_mean) / self.rtg_std
            rtg[pos] = rtg_val
            mask[pos] = 1.0

        return {
            "states": torch.from_numpy(s),
            "actions": torch.from_numpy(a),
            "rewards": torch.from_numpy(r),
            "rtg": torch.from_numpy(rtg).unsqueeze(-1),
            "mask": torch.from_numpy(mask),
            "target_action": torch.from_numpy(self.actions[ep_idx, t]),
        }


class GPUTrajectoryBuffer:
    """
    GPU-resident trajectory buffer with on-device batch gathering.

    Uploads all arrays once; builds batches via index gather with no
    per-step host-to-device transfers.
    """

    def __init__(
        self,
        cfg: Config,
        device: str,
        context_length: int | None = None,
        last_state_only: bool = False,
        val_fraction: float = 0.1,
        seed: int = 42,
    ):
        self.device = device
        self.last_state_only = last_state_only
        K = context_length or int(cfg.get("model", "context_length", default=30))
        self.context_length = K

        root = cfg.project_root
        traj_path = root / cfg.get("trajectories", "output_dir", default="data/trajectories") / "trajectories.npz"
        data = _load_trajectory_npz(traj_path)

        self.global_states = torch.from_numpy(data["states"].astype(np.float32)).to(device)
        self.episode_starts = torch.from_numpy(data["episode_starts"].astype(np.int64)).to(device)
        self.actions = torch.from_numpy(data["actions"].astype(np.float32)).to(device)
        self.rewards = torch.from_numpy(data["rewards"].astype(np.float32)).to(device)
        self.rtg = torch.from_numpy(data["rtg"].astype(np.float32)).to(device)
        self.lengths = torch.from_numpy(data["lengths"].astype(np.int64)).to(device)
        self.rtg_mean = float(data.get("rtg_mean", 0.0))
        self.rtg_std = float(data.get("rtg_std", 1.0))
        self.state_dim = self.global_states.shape[1]
        self.n_assets = self.actions.shape[2]
        self.n_episodes = len(self.lengths)

        # Flat index: (episode_idx, local_t) for every valid transition
        ep_indices = []
        local_ts = []
        for ep in range(self.n_episodes):
            L = int(self.lengths[ep].item())
            for t in range(L):
                ep_indices.append(ep)
                local_ts.append(t)
        self.ep_indices = torch.tensor(ep_indices, dtype=torch.long, device=device)
        self.local_ts = torch.tensor(local_ts, dtype=torch.long, device=device)
        self.n_samples = len(self.ep_indices)

        # Train/val split
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(self.n_samples, generator=g)
        n_val = max(1, int(val_fraction * self.n_samples))
        self.val_indices = perm[:n_val]
        self.train_indices = perm[n_val:]

    def sample_batch(self, indices: torch.Tensor) -> dict[str, torch.Tensor]:
        """Gather a batch by flat sample indices (already on device)."""
        ep = self.ep_indices[indices]
        t = self.local_ts[indices]

        if self.last_state_only:
            global_t = self.episode_starts[ep] + t
            states = self.global_states[global_t]
            target_action = self.actions[ep, t]
            rewards = self.rewards[ep, t]
            return {
                "states": states.unsqueeze(1),
                "actions": target_action.unsqueeze(1),
                "rewards": rewards.unsqueeze(1),
                "rtg": torch.zeros(len(indices), 1, 1, device=self.device),
                "mask": torch.ones(len(indices), 1, device=self.device),
                "target_action": target_action,
            }

        B = len(indices)
        K = self.context_length
        offsets = torch.arange(K, device=self.device) - (K - 1)  # [-(K-1), ..., 0]
        local_idx = t.unsqueeze(1) + offsets.unsqueeze(0)  # (B, K)
        ep_start_local = torch.zeros(B, dtype=torch.long, device=self.device)
        mask = (local_idx >= 0).float()
        local_idx_clamped = local_idx.clamp(min=0)

        # Gather actions, rewards, rtg per episode
        ep_exp = ep.unsqueeze(1).expand(-1, K)
        batch_actions = self.actions[ep_exp, local_idx_clamped]
        batch_rewards = self.rewards[ep_exp, local_idx_clamped]
        batch_rtg = self.rtg[ep_exp, local_idx_clamped]
        batch_rtg = (batch_rtg - self.rtg_mean) / self.rtg_std

        # Gather global states
        global_idx = self.episode_starts[ep].unsqueeze(1) + local_idx_clamped
        global_idx = global_idx.clamp(max=self.global_states.shape[0] - 1)
        batch_states = self.global_states[global_idx]

        target_action = self.actions[ep, t]

        return {
            "states": batch_states,
            "actions": batch_actions,
            "rewards": batch_rewards,
            "rtg": batch_rtg.unsqueeze(-1),
            "mask": mask,
            "target_action": target_action,
        }

    def train_batch_indices(self, batch_size: int, steps_per_epoch: int | None = None) -> list[torch.Tensor]:
        """Generate shuffled train batch index tensors for one epoch."""
        n = len(self.train_indices)
        steps = steps_per_epoch if steps_per_epoch and steps_per_epoch > 0 else max(1, n // batch_size)
        batches = []
        for _ in range(steps):
            perm = self.train_indices[torch.randperm(n, device=self.device)]
            idx = perm[:batch_size]
            if len(idx) == batch_size:
                batches.append(idx)
        return batches if batches else [self.train_indices[:batch_size]]

    def val_batches(self, batch_size: int) -> list[torch.Tensor]:
        batches = []
        for i in range(0, len(self.val_indices) - batch_size + 1, batch_size):
            batches.append(self.val_indices[i : i + batch_size])
        if not batches and len(self.val_indices) > 0:
            batches.append(self.val_indices)
        return batches


def load_trajectory_dataset(cfg: Config, split: str = "train") -> TrajectoryDataset:
    root = cfg.project_root
    traj_path = root / cfg.get("trajectories", "output_dir", default="data/trajectories") / "trajectories.npz"
    K = int(cfg.get("model", "context_length", default=30))
    return TrajectoryDataset(traj_path, context_length=K)
