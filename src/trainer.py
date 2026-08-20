"""Training loop for Decision Transformer and baselines."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from src.config import Config
from src.dataset import load_trajectory_dataset
from src.models.bc import MLPBC, TransformerBC
from src.models.decision_transformer import DecisionTransformer
from src.models.offline_rl import CQL, IQL, TD3BC
from src.models.online_rl import A2C, PPO, SAC


def get_device(cfg: Config) -> str:
    requested = cfg.get("project", "device", default="cuda")
    if requested == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def train_dt(
    cfg: Config,
    seed: int = 42,
    use_rtg: bool = True,
    context_length: int | None = None,
) -> Path:
    """Train Decision Transformer (or BC if use_rtg=False)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = get_device(cfg)

    K = context_length or int(cfg.get("model", "context_length", default=30))
    d_model = int(cfg.get("model", "d_model", default=192))
    n_layers = int(cfg.get("model", "n_layers", default=4))
    n_heads = int(cfg.get("model", "n_heads", default=6))
    dropout = float(cfg.get("model", "dropout", default=0.1))
    lr = float(cfg.get("model", "learning_rate", default=1e-4))
    batch_size = int(cfg.get("model", "batch_size", default=64))
    max_epochs = int(cfg.get("model", "max_epochs", default=50))
    weight_decay = float(cfg.get("model", "weight_decay", default=1e-4))

    dataset = load_trajectory_dataset(cfg)
    state_dim = dataset.states.shape[2]
    action_dim = dataset.actions.shape[2]

    n_val = max(1, int(0.1 * len(dataset)))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    if use_rtg:
        model = DecisionTransformer(
            state_dim, action_dim, K, d_model, n_layers, n_heads, dropout
        ).to(device)
        model_name = "dt"
    else:
        model = TransformerBC(
            state_dim, action_dim, K, d_model, n_layers, n_heads, dropout
        ).to(device)
        model_name = "bc_transformer"

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    ckpt_dir = cfg.project_root / cfg.get("training", "checkpoint_dir", default="results/checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir = cfg.project_root / cfg.get("training", "log_dir", default="results/runs")
    log_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    history = []

    for epoch in range(max_epochs):
        model.train()
        train_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            states = batch["states"].to(device)
            actions = batch["actions"].to(device)
            rtg = batch["rtg"].to(device)
            mask = batch["mask"].to(device)
            target = batch["target_action"].to(device)

            pred = model(states, actions, rtg, mask)
            loss = torch.nn.functional.mse_loss(pred[:, -1, :], target)

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
            for batch in val_loader:
                states = batch["states"].to(device)
                actions = batch["actions"].to(device)
                rtg = batch["rtg"].to(device)
                mask = batch["mask"].to(device)
                target = batch["target_action"].to(device)
                pred = model(states, actions, rtg, mask)
                loss = torch.nn.functional.mse_loss(pred[:, -1, :], target)
                val_loss += loss.item()
                n_val_batches += 1

        train_loss /= max(n_batches, 1)
        val_loss /= max(n_val_batches, 1)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = ckpt_dir / f"{model_name}_seed{seed}.pt"
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": {"state_dim": state_dim, "action_dim": action_dim, "K": K,
                           "d_model": d_model, "n_layers": n_layers, "n_heads": n_heads,
                           "use_rtg": use_rtg},
                "val_loss": val_loss,
                "epoch": epoch,
            }, ckpt_path)

    # Save training log
    log_path = log_dir / f"{model_name}_seed{seed}_history.csv"
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss"])
        writer.writeheader()
        writer.writerows(history)

    return ckpt_dir / f"{model_name}_seed{seed}.pt"


def train_offline_rl(cfg: Config, algo: str, seed: int = 42) -> Path:
    """Train an offline RL algorithm."""
    torch.manual_seed(seed)
    device = get_device(cfg)
    dataset = load_trajectory_dataset(cfg)
    state_dim = dataset.states.shape[2]
    action_dim = dataset.actions.shape[2]
    batch_size = int(cfg.get("model", "batch_size", default=64))
    max_epochs = int(cfg.get("model", "max_epochs", default=50))

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    algo_map = {
        "td3bc": TD3BC,
        "iql": IQL,
        "cql": CQL,
    }
    agent = algo_map[algo](state_dim, action_dim, device=device)

    ckpt_dir = cfg.project_root / cfg.get("training", "checkpoint_dir", default="results/checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(max_epochs):
        for batch in loader:
            agent.train_step(batch)

    ckpt_path = ckpt_dir / f"{algo}_seed{seed}.pt"
    torch.save({"agent_state": algo, "seed": seed, "epoch": max_epochs}, ckpt_path)
    return ckpt_path


def train_online_rl(cfg: Config, algo: str, seed: int = 42) -> Path:
    """Train an online RL algorithm (reference only)."""
    torch.manual_seed(seed)
    device = get_device(cfg)
    dataset = load_trajectory_dataset(cfg)
    state_dim = dataset.states.shape[2]
    action_dim = dataset.actions.shape[2]
    batch_size = int(cfg.get("model", "batch_size", default=64))
    max_epochs = int(cfg.get("model", "max_epochs", default=50))

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    algo_map = {"ppo": PPO, "sac": SAC, "a2c": A2C}
    agent = algo_map[algo](state_dim, action_dim, device=device)

    ckpt_dir = cfg.project_root / cfg.get("training", "checkpoint_dir", default="results/checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(max_epochs):
        for batch in loader:
            agent.train_step(batch)

    ckpt_path = ckpt_dir / f"{algo}_seed{seed}.pt"
    torch.save({"agent_state": algo, "seed": seed}, ckpt_path)
    return ckpt_path


def train_all_models(cfg: Config) -> dict[str, Any]:
    """Train all models across seeds."""
    seeds = cfg.get("training", "seeds", default=[42])
    results = {}

    for seed in seeds:
        results[f"dt_seed{seed}"] = str(train_dt(cfg, seed=seed, use_rtg=True))
        results[f"bc_seed{seed}"] = str(train_dt(cfg, seed=seed, use_rtg=False))
        for algo in ("td3bc", "iql", "cql"):
            results[f"{algo}_seed{seed}"] = str(train_offline_rl(cfg, algo, seed=seed))
        for algo in ("ppo", "sac", "a2c"):
            results[f"{algo}_seed{seed}"] = str(train_online_rl(cfg, algo, seed=seed))

    out_path = cfg.project_root / "results" / "training_manifest.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    return results
