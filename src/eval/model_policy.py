"""Checkpoint-backed policies for Decision Transformer and RL agents.

Each policy is a ``(t, state, env) -> weights`` callable so it drops into
``PortfolioEnv.run_policy`` / ``backtest_policy`` unchanged.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.config import Config
from src.env import project_to_simplex
from src.models.bc import TransformerBC
from src.models.decision_transformer import DecisionTransformer
from src.models.offline_rl import CQL, IQL, TD3BC
from src.models.online_rl import A2C, PPO, SAC
from src.policies import Policy

TRANSFORMER_ALGOS = {"dt", "bc"}
AGENT_ALGOS = {"td3bc", "iql", "cql", "ppo", "sac", "a2c"}
AGENT_CLASSES: dict[str, type] = {
    "td3bc": TD3BC,
    "iql": IQL,
    "cql": CQL,
    "ppo": PPO,
    "sac": SAC,
    "a2c": A2C,
}


def parse_manifest_key(key: str) -> tuple[str, int]:
    """Split ``dt_seed42`` / ``td3bc_seed42`` into ``(algo, seed)``."""
    if "_seed" not in key:
        raise ValueError(f"Manifest key {key!r} does not match '<algo>_seed<N>'")
    algo, seed_str = key.rsplit("_seed", 1)
    return algo, int(seed_str)


def assert_checkpoint_dims(
    ckpt_config: dict[str, Any],
    state_dim: int,
    action_dim: int,
    path: Path,
) -> None:
    """Refuse to evaluate a checkpoint trained on a different feature layout."""
    ckpt_sd = int(ckpt_config.get("state_dim", -1))
    ckpt_ad = int(ckpt_config.get("action_dim", -1))
    if ckpt_sd != state_dim or ckpt_ad != action_dim:
        raise ValueError(
            f"Checkpoint {path} was trained with state_dim={ckpt_sd}, "
            f"action_dim={ckpt_ad}, but the current feature bundle has "
            f"state_dim={state_dim}, action_dim={action_dim}. "
            "Re-run --stage train for this universe, or point evaluation "
            "at matching artifacts."
        )


def trajectory_dir(cfg: Config) -> Path:
    return cfg.project_root / cfg.get(
        "trajectories", "output_dir", default="data/trajectories"
    )


def load_rtg_stats(cfg: Config) -> tuple[float, float]:
    """Read RTG mean/std used to normalize DT inputs during training."""
    meta_path = trajectory_dir(cfg) / "trajectory_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Missing {meta_path}. Run --stage trajectories before evaluating DT."
        )
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return float(meta["rtg_mean"]), float(meta["rtg_std"])


def episode_start_rtg(cfg: Config) -> np.ndarray:
    """Per-episode remaining return at t=0 (full-episode log return)."""
    traj_path = trajectory_dir(cfg) / "trajectories.npz"
    if not traj_path.exists():
        raise FileNotFoundError(
            f"Missing {traj_path}. Run --stage trajectories before evaluating DT."
        )
    data = np.load(traj_path)
    return np.asarray(data["rtg"][:, 0], dtype=np.float64)


def rtg_quantile_targets(cfg: Config, quantiles: list[float] | None = None) -> dict[float, float]:
    """Map requested quantiles to target RTG values from training episodes."""
    if quantiles is None:
        quantiles = list(cfg.get("evaluation", "rtg_quantiles", default=[0.1, 0.25, 0.5, 0.75, 0.9]))
    starts = episode_start_rtg(cfg)
    return {float(q): float(np.quantile(starts, q)) for q in quantiles}


def default_target_rtg(cfg: Config) -> float:
    q = float(cfg.get("evaluation", "default_rtg_quantile", default=0.9))
    return rtg_quantile_targets(cfg, [q])[q]


def load_manifest(cfg: Config) -> dict[str, str]:
    path = cfg.project_root / "results" / "training_manifest.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def iter_manifest_checkpoints(cfg: Config) -> list[tuple[str, Path]]:
    """Yield ``(manifest_key, checkpoint_path)`` skipping failed / missing entries."""
    entries: list[tuple[str, Path]] = []
    for key, value in load_manifest(cfg).items():
        if not isinstance(value, str) or value.startswith("failed:"):
            warnings.warn(f"Skipping failed manifest entry {key}: {value}")
            continue
        path = Path(value)
        if not path.exists():
            warnings.warn(f"Skipping missing checkpoint for {key}: {path}")
            continue
        entries.append((key, path))
    return entries


class DTPolicy(Policy):
    """RTG-conditioned (or BC) transformer policy with a rolling K-step context.

    Training stores unshifted actions at each timestep; the causal mask keeps the
    current-step action token after the state token, so it cannot leak. Inference
    therefore zero-fills the last action slot. RTG is the remaining budget,
    decremented by the realized log reward after each env step and optionally
    reset every ``reset_horizon`` steps so a multi-year backtest stays
    in-distribution relative to 252-day training episodes.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: str,
        target_rtg: float,
        rtg_mean: float,
        rtg_std: float,
        reset_horizon: int | None = None,
        name: str = "dt",
    ):
        self.model = model.eval()
        self.device = device
        self.target_rtg = float(target_rtg)
        self.rtg_mean = float(rtg_mean)
        self.rtg_std = float(rtg_std) if rtg_std else 1.0
        self.reset_horizon = reset_horizon
        self.name = name
        self.context_length = int(getattr(model, "context_length", 30))
        self.state_dim = int(model.state_dim)
        self.action_dim = int(model.action_dim)
        self.rtg_budget = self.target_rtg
        self._steps = 0
        self._states: list[np.ndarray] = []
        self._actions: list[np.ndarray] = []
        self._rtgs: list[float] = []

    def reset(self) -> None:
        self.rtg_budget = self.target_rtg
        self._steps = 0
        self._states.clear()
        self._actions.clear()
        self._rtgs.clear()

    def _maybe_start_episode(self, env) -> None:
        if not getattr(env, "history", None):
            self.reset()

    def _update_budget(self, env) -> None:
        if self._steps > 0 and env.history:
            self.rtg_budget -= float(env.history[-1]["reward"])
            self._actions.append(np.asarray(env.history[-1]["weights"], dtype=np.float32))
        if self.reset_horizon and self._steps % self.reset_horizon == 0:
            self.rtg_budget = self.target_rtg

    def _build_context(self, state: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self._states.append(np.asarray(state, dtype=np.float32))
        self._rtgs.append(float(self.rtg_budget))

        K = self.context_length
        n = len(self._states)
        valid = min(K, n)

        states = np.zeros((K, self.state_dim), dtype=np.float32)
        actions = np.zeros((K, self.action_dim), dtype=np.float32)
        rtg = np.zeros((K, 1), dtype=np.float32)
        mask = np.zeros(K, dtype=np.float32)

        states[-valid:] = np.stack(self._states[-valid:], axis=0)
        rtg_raw = np.asarray(self._rtgs[-valid:], dtype=np.float32)
        rtg[-valid:, 0] = (rtg_raw - self.rtg_mean) / self.rtg_std
        mask[-valid:] = 1.0
        # History actions occupy every slot except the current (last) one.
        n_prev = min(len(self._actions), valid - 1)
        if n_prev > 0:
            actions[-valid : -valid + n_prev] = np.stack(self._actions[-n_prev:], axis=0)

        def _t(arr: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(arr).unsqueeze(0).to(self.device)

        return _t(states), _t(actions), _t(rtg), _t(mask)

    def __call__(self, t: int, state: np.ndarray, env) -> np.ndarray:
        self._maybe_start_episode(env)
        self._update_budget(env)
        states, actions, rtg, mask = self._build_context(state)
        with torch.no_grad():
            pred = self.model.get_action(states, actions, rtg, mask)
        action = pred.squeeze(0).detach().cpu().numpy().astype(np.float64)
        self._steps += 1
        return action


class AgentPolicy(Policy):
    """Single-step RL agent (TD3+BC / IQL / CQL / PPO / SAC / A2C)."""

    def __init__(self, agent: Any, device: str, name: str = "agent"):
        self.agent = agent
        self.device = device
        self.name = name
        if hasattr(agent, "actor") and isinstance(agent.actor, torch.nn.Module):
            agent.actor.eval()

    def __call__(self, t: int, state: np.ndarray, env) -> np.ndarray:
        x = torch.from_numpy(np.asarray(state, dtype=np.float32)).view(1, 1, -1)
        x = x.to(self.device)
        with torch.no_grad():
            pred = self.agent.get_action(x)
        action = pred.squeeze(0).detach().cpu().numpy().astype(np.float64)
        return project_to_simplex(action)


def _load_payload(ckpt_path: Path, device: str) -> dict[str, Any]:
    try:
        return torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(ckpt_path, map_location=device)


def load_transformer(
    ckpt_path: Path,
    device: str,
    state_dim: int,
    action_dim: int,
) -> torch.nn.Module:
    payload = _load_payload(ckpt_path, device)
    cfg_dict = payload.get("config", {})
    assert_checkpoint_dims(cfg_dict, state_dim, action_dim, ckpt_path)
    K = int(cfg_dict.get("K", 30))
    d_model = int(cfg_dict.get("d_model", 192))
    n_layers = int(cfg_dict.get("n_layers", 4))
    n_heads = int(cfg_dict.get("n_heads", 6))
    use_rtg = bool(cfg_dict.get("use_rtg", True))
    cls = DecisionTransformer if use_rtg else TransformerBC
    model = cls(state_dim, action_dim, K, d_model, n_layers, n_heads).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model


def load_agent(
    ckpt_path: Path,
    device: str,
    state_dim: int,
    action_dim: int,
    algo: str,
) -> Any:
    payload = _load_payload(ckpt_path, device)
    cfg_dict = payload.get("config", {})
    assert_checkpoint_dims(cfg_dict, state_dim, action_dim, ckpt_path)
    recorded = payload.get("algo", algo)
    if recorded not in AGENT_CLASSES:
        raise ValueError(f"Unknown agent algo {recorded!r} in {ckpt_path}")
    agent = AGENT_CLASSES[recorded](state_dim, action_dim, device=device)
    modules = payload.get("modules")
    if not modules:
        raise ValueError(
            f"Checkpoint {ckpt_path} has no 'modules' weights. "
            "Re-train with the current trainer (it now saves nn.Module state)."
        )
    for name, state_dict in modules.items():
        module = getattr(agent, name, None)
        if not isinstance(module, torch.nn.Module):
            continue
        module.load_state_dict(state_dict)
        module.eval()
    return agent


def policy_from_checkpoint(
    cfg: Config,
    key: str,
    ckpt_path: Path,
    state_dim: int,
    action_dim: int,
    target_rtg: float,
    device: str,
    reset_horizon: int | None = None,
) -> Policy:
    algo, _seed = parse_manifest_key(key)
    if algo in TRANSFORMER_ALGOS:
        model = load_transformer(ckpt_path, device, state_dim, action_dim)
        rtg_mean, rtg_std = load_rtg_stats(cfg)
        return DTPolicy(
            model,
            device=device,
            target_rtg=target_rtg,
            rtg_mean=rtg_mean,
            rtg_std=rtg_std,
            reset_horizon=reset_horizon,
            name=key,
        )
    if algo in AGENT_ALGOS:
        agent = load_agent(ckpt_path, device, state_dim, action_dim, algo)
        return AgentPolicy(agent, device=device, name=key)
    raise ValueError(f"Unsupported algorithm in manifest key {key!r}")


def load_learned_policies(
    cfg: Config,
    state_dim: int,
    action_dim: int,
    target_rtg: float | None = None,
    device: str | None = None,
    reset_horizon: int | None = None,
) -> dict[str, Policy]:
    """Load every successful checkpoint in the training manifest."""
    from src.trainer import get_device

    device = device or get_device(cfg)
    if reset_horizon is None:
        reset_horizon = int(cfg.get("env", "episode_length", default=252))
    if target_rtg is None:
        try:
            target_rtg = default_target_rtg(cfg)
        except FileNotFoundError:
            target_rtg = 0.0

    policies: dict[str, Policy] = {}
    for key, path in iter_manifest_checkpoints(cfg):
        policies[key] = policy_from_checkpoint(
            cfg, key, path, state_dim, action_dim, target_rtg, device, reset_horizon
        )
    return policies
