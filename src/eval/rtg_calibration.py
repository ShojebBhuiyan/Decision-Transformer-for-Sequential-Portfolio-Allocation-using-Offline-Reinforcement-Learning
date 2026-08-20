"""RTG calibration: target return-to-go vs realized backtest return."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import Config
from src.data_loader import get_split_mask
from src.eval.backtest import backtest_policy
from src.eval.metrics import cagr as cagr_fn
from src.eval.metrics import sharpe_ratio
from src.eval.model_policy import (
    DTPolicy,
    iter_manifest_checkpoints,
    load_rtg_stats,
    load_transformer,
    parse_manifest_key,
    rtg_quantile_targets,
)
from src.trainer import get_device


def run_rtg_calibration(
    cfg: Config,
    price_returns: np.ndarray,
    states: np.ndarray,
    dates: pd.DatetimeIndex,
    start: str | None = None,
    end: str | None = None,
    transaction_cost: float | None = None,
) -> pd.DataFrame:
    """
    Sweep target RTG quantiles on the holdout window.

    The RTG budget is reset every ``env.episode_length`` steps so a multi-year
    test window stays in-distribution relative to 252-day training episodes.
    """
    start = start or cfg.get("splits", "test_start")
    end = end or cfg.get("splits", "test_end")
    transaction_cost = (
        float(cfg.get("env", "transaction_cost", default=0.001))
        if transaction_cost is None
        else transaction_cost
    )
    mask = get_split_mask(dates, start, end)
    if mask.sum() == 0:
        return pd.DataFrame()

    split_returns = price_returns[mask]
    split_states = states[mask]
    reset_horizon = int(cfg.get("env", "episode_length", default=252))
    quantiles = list(cfg.get("evaluation", "rtg_quantiles", default=[0.1, 0.25, 0.5, 0.75, 0.9]))

    try:
        targets = rtg_quantile_targets(cfg, quantiles)
        rtg_mean, rtg_std = load_rtg_stats(cfg)
    except FileNotFoundError as e:
        warnings.warn(f"RTG calibration skipped: {e}")
        return pd.DataFrame()

    device = get_device(cfg)
    state_dim = int(states.shape[1])
    action_dim = int(price_returns.shape[1])
    rows: list[dict] = []

    dt_ckpts = [
        (key, path)
        for key, path in iter_manifest_checkpoints(cfg)
        if parse_manifest_key(key)[0] == "dt"
    ]
    if not dt_ckpts:
        warnings.warn("RTG calibration skipped: no Decision Transformer checkpoints in the manifest.")
        return pd.DataFrame()

    for key, path in dt_ckpts:
        _algo, seed = parse_manifest_key(key)
        model = load_transformer(path, device, state_dim, action_dim)
        for q, target in targets.items():
            policy = DTPolicy(
                model,
                device=device,
                target_rtg=target,
                rtg_mean=rtg_mean,
                rtg_std=rtg_std,
                reset_horizon=reset_horizon,
                name=f"{key}_q{q}",
            )
            result = backtest_policy(split_returns, policy, split_states, transaction_cost)
            log_ret = result["log_returns"]
            rows.append({
                "strategy": "dt",
                "seed": seed,
                "quantile": q,
                "target_rtg": target,
                "realized_log_return": float(result["cumulative_log_return"]),
                "realized_cagr": cagr_fn(log_ret),
                "sharpe": sharpe_ratio(
                    log_ret,
                    risk_free_rate=float(cfg.get("evaluation", "risk_free_rate", default=0.02)),
                ),
            })

    return pd.DataFrame(rows)


def plot_rtg_calibration(df: pd.DataFrame, out_path: Path) -> None:
    """Target RTG vs realized cumulative log return."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    for seed, sub in df.groupby("seed"):
        sub = sub.sort_values("target_rtg")
        ax.plot(
            sub["target_rtg"],
            sub["realized_log_return"],
            marker="o",
            alpha=0.5,
            label=f"seed {seed}",
        )
    mean = df.groupby("target_rtg", as_index=False)["realized_log_return"].mean().sort_values("target_rtg")
    ax.plot(
        mean["target_rtg"],
        mean["realized_log_return"],
        color="black",
        linewidth=2,
        marker="s",
        label="mean",
    )
    lo, hi = df["target_rtg"].min(), df["target_rtg"].max()
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="gray", linewidth=1, label="identity")
    ax.set_xlabel("Target RTG (per 252-day reset)")
    ax.set_ylabel("Realized cumulative log return")
    ax.set_title("Decision Transformer RTG calibration")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def save_rtg_calibration(cfg: Config, df: pd.DataFrame) -> Path | None:
    if df.empty:
        return None
    tables = cfg.tables_dir()
    tables.mkdir(parents=True, exist_ok=True)
    csv_path = tables / "rtg_calibration.csv"
    df.to_csv(csv_path, index=False)
    plot_rtg_calibration(df, cfg.figures_eval_dir() / "rtg_calibration.png")
    return csv_path
