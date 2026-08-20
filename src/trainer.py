"""Training loop for Decision Transformer and baselines."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.config import Config
from src.models.bc import TransformerBC
from src.models.decision_transformer import DecisionTransformer
from src.models.offline_rl import CQL, IQL, TD3BC
from src.models.online_rl import A2C, PPO, SAC
from src.trajectories import GPUTrajectoryBuffer


def get_device(cfg: Config) -> str:
    requested = cfg.get("project", "device", default="cuda")
    if requested == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _setup_training(cfg: Config) -> None:
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True


def _release_cuda(*objs: object) -> None:
    del objs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _agent_state_dict(agent: object) -> dict[str, Any]:
    """Collect weights of every nn.Module held by an RL agent."""
    return {
        name: module.state_dict()
        for name, module in vars(agent).items()
        if isinstance(module, torch.nn.Module)
    }


def _steps_per_epoch(cfg: Config) -> int | None:
    value = cfg.get("training", "steps_per_epoch", default=None)
    return int(value) if value is not None else None


def _max_val_batches(cfg: Config) -> int | None:
    value = cfg.get("training", "max_val_batches", default=None)
    return int(value) if value is not None else None


def train_dt(
    cfg: Config,
    seed: int = 42,
    use_rtg: bool = True,
    context_length: int | None = None,
    run_tag: str = "",
) -> Path:
    """Train Decision Transformer (or BC if use_rtg=False)."""
    _setup_training(cfg)
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = get_device(cfg)

    K = context_length or int(cfg.get("model", "context_length", default=30))
    d_model = int(cfg.get("model", "d_model", default=192))
    n_layers = int(cfg.get("model", "n_layers", default=4))
    n_heads = int(cfg.get("model", "n_heads", default=6))
    dropout = float(cfg.get("model", "dropout", default=0.1))
    lr = float(cfg.get("model", "learning_rate", default=1e-4))
    batch_size = int(cfg.get("model", "batch_size", default=128))
    max_epochs = int(cfg.get("model", "max_epochs", default=50))
    weight_decay = float(cfg.get("model", "weight_decay", default=1e-4))
    steps_per_epoch = _steps_per_epoch(cfg)
    max_val_batches = _max_val_batches(cfg)

    buffer = GPUTrajectoryBuffer(cfg, device, context_length=K, seed=seed)
    state_dim = buffer.state_dim
    action_dim = buffer.n_assets

    max_samples = int(cfg.get("model", "max_train_samples", default=0))
    if max_samples > 0 and len(buffer.train_indices) > max_samples:
        perm = torch.randperm(len(buffer.train_indices), device=buffer.storage_device)
        buffer.train_indices = buffer.train_indices[perm[:max_samples]]

    if use_rtg:
        model = DecisionTransformer(
            state_dim, action_dim, K, d_model, n_layers, n_heads, dropout
        ).to(device)
        model_name = f"dt{run_tag}"
    else:
        model = TransformerBC(
            state_dim, action_dim, K, d_model, n_layers, n_heads, dropout
        ).to(device)
        model_name = f"bc_transformer{run_tag}"

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    ckpt_dir = cfg.checkpoint_dir()
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir = cfg.run_dir()
    log_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = ckpt_dir / f"{model_name}_seed{seed}.pt"
    log_path = log_dir / f"{model_name}_seed{seed}_history.csv"
    best_val_loss = float("inf")
    history: list[dict[str, float]] = []

    for epoch in range(max_epochs):
        model.train()
        train_loss = 0.0
        n_batches = 0
        for batch_idx in buffer.train_batch_indices(batch_size, steps_per_epoch):
            batch = buffer.sample_batch(batch_idx)
            pred = model(batch["states"], batch["actions"], batch["rtg"], batch["mask"])
            loss = torch.nn.functional.mse_loss(pred[:, -1, :], batch["target_action"])

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1

        scheduler.step()

        model.eval()
        val_loss = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for batch_idx in buffer.val_batch_indices(batch_size, max_val_batches):
                batch = buffer.sample_batch(batch_idx)
                pred = model(batch["states"], batch["actions"], batch["rtg"], batch["mask"])
                loss = torch.nn.functional.mse_loss(pred[:, -1, :], batch["target_action"])
                val_loss += loss.item()
                n_val_batches += 1

        train_loss /= max(n_batches, 1)
        val_loss /= max(n_val_batches, 1)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        print(f"  [{model_name}] epoch {epoch+1}/{max_epochs} train={train_loss:.5f} val={val_loss:.5f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": {"state_dim": state_dim, "action_dim": action_dim, "K": K,
                           "d_model": d_model, "n_layers": n_layers, "n_heads": n_heads,
                           "use_rtg": use_rtg},
                "val_loss": val_loss,
                "epoch": epoch,
            }, ckpt_path)

        # Rewrite the log each epoch so an interrupted run keeps its history
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss"])
            writer.writeheader()
            writer.writerows(history)

    _release_cuda(model, optimizer, scheduler, buffer)
    return ckpt_path


def _train_agent(
    cfg: Config,
    algo: str,
    algo_map: dict[str, Any],
    seed: int,
) -> Path:
    """Shared training loop for offline and online RL agents."""
    _setup_training(cfg)
    torch.manual_seed(seed)
    device = get_device(cfg)
    batch_size = int(cfg.get("model", "batch_size", default=128))
    max_epochs = int(cfg.get("model", "max_epochs", default=50))
    steps_per_epoch = _steps_per_epoch(cfg)

    buffer = GPUTrajectoryBuffer(cfg, device, last_state_only=True, seed=seed)
    agent = algo_map[algo](buffer.state_dim, buffer.n_assets, device=device)

    ckpt_dir = cfg.checkpoint_dir()
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{algo}_seed{seed}.pt"

    last_metrics: dict[str, float] = {}
    for epoch in range(max_epochs):
        for batch_idx in buffer.train_batch_indices(batch_size, steps_per_epoch):
            batch = buffer.sample_batch(batch_idx)
            metrics = agent.train_step(batch)
            if isinstance(metrics, dict):
                last_metrics = metrics
        print(f"  [{algo}] epoch {epoch+1}/{max_epochs} " +
              " ".join(f"{k}={v:.5f}" for k, v in last_metrics.items()))

    torch.save({
        "algo": algo,
        "seed": seed,
        "epoch": max_epochs,
        "modules": _agent_state_dict(agent),
        "config": {"state_dim": buffer.state_dim, "action_dim": buffer.n_assets},
        "metrics": last_metrics,
    }, ckpt_path)
    _release_cuda(agent, buffer)
    return ckpt_path


def train_offline_rl(cfg: Config, algo: str, seed: int = 42) -> Path:
    """Train an offline RL algorithm."""
    return _train_agent(cfg, algo, {"td3bc": TD3BC, "iql": IQL, "cql": CQL}, seed)


def train_online_rl(cfg: Config, algo: str, seed: int = 42) -> Path:
    """Train an online RL algorithm (reference only)."""
    return _train_agent(cfg, algo, {"ppo": PPO, "sac": SAC, "a2c": A2C}, seed)


def train_all_models(cfg: Config) -> dict[str, Any]:
    """Train all models across seeds, persisting progress after each model."""
    seeds = cfg.get("training", "seeds", default=[42])
    out_path = cfg.training_manifest_path()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}

    def _record(key: str, fn) -> None:
        try:
            results[key] = str(fn())
        except Exception as e:  # keep the sweep alive and record the reason
            print(f"  WARNING: {key} failed: {e}")
            results[key] = f"failed: {type(e).__name__}: {e}"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    for seed in seeds:
        print(f"Training seed {seed}...")
        _record(f"dt_seed{seed}", lambda: train_dt(cfg, seed=seed, use_rtg=True))
        _record(f"bc_seed{seed}", lambda: train_dt(cfg, seed=seed, use_rtg=False))
        for algo in ("td3bc", "iql", "cql"):
            print(f"  Training offline RL: {algo}")
            _record(f"{algo}_seed{seed}", lambda a=algo: train_offline_rl(cfg, a, seed=seed))
        for algo in ("ppo", "sac", "a2c"):
            print(f"  Training online RL: {algo}")
            _record(f"{algo}_seed{seed}", lambda a=algo: train_online_rl(cfg, a, seed=seed))

    return results
