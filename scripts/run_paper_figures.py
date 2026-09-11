"""Figures for the term paper: convergence, equity curves, and ablation plots.

    python scripts/run_paper_figures.py

Reads the training histories in ``results/runs/`` and the tables in
``results/tables/``; writes into ``results/figures/paper/``.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import Config

SEEDS = [42, 123, 456, 789, 1011]


def out_dir(cfg: Config) -> Path:
    d = cfg.figures_eval_dir().parent / "paper"
    d.mkdir(parents=True, exist_ok=True)
    return d


def convergence(cfg: Config) -> None:
    runs = cfg.run_dir()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, (prefix, label) in zip(axes, [("dt", "Decision Transformer"),
                                          ("bc_transformer", "Transformer BC")]):
        tr, va = [], []
        for seed in SEEDS:
            p = runs / f"{prefix}_seed{seed}_history.csv"
            if not p.exists():
                continue
            h = pd.read_csv(p)
            tr.append(h["train_loss"].values)
            va.append(h["val_loss"].values)
        if not tr:
            continue
        tr = np.vstack(tr); va = np.vstack(va)
        ep = np.arange(tr.shape[1]) + 1
        ax.plot(ep, tr.mean(0), color="steelblue", label="train (mean of 5 seeds)")
        ax.fill_between(ep, tr.min(0), tr.max(0), color="steelblue", alpha=0.2)
        ax.plot(ep, va.mean(0), color="indianred", label="validation (mean of 5 seeds)")
        ax.fill_between(ep, va.min(0), va.max(0), color="indianred", alpha=0.2)
        ax.set_title(label)
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Action MSE")
    axes[0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir(cfg) / "convergence.png", dpi=150)
    plt.close(fig)
    print("wrote convergence.png")


def cost_sensitivity(cfg: Config) -> None:
    path = cfg.tables_dir() / "ablation_transaction_cost.csv"
    if not path.exists():
        print("skip cost_sensitivity (no table)")
        return
    df = pd.read_csv(path)
    if "algo" not in df.columns:
        df["algo"] = df["strategy"]
    # average the learned models over seeds; classical rows have one row each
    df = df.groupby(["algo", "transaction_cost"], as_index=False).mean(numeric_only=True)
    keep = ["buy_and_hold", "mean_variance", "momentum", "dt", "bc"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for metric, ax, ylab in [("sharpe", axes[0], "Sharpe ratio"),
                             ("cagr", axes[1], "CAGR")]:
        for name in keep:
            sub = df[df["algo"] == name].sort_values("transaction_cost")
            if sub.empty:
                continue
            ax.plot(sub["transaction_cost"] * 1e4, sub[metric], marker="o", label=name)
        ax.set_xlabel(r"Transaction cost $\mu$ (basis points of turnover)")
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.3)
    axes[0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir(cfg) / "cost_sensitivity.png", dpi=150)
    plt.close(fig)
    print("wrote cost_sensitivity.png")


def context_length(cfg: Config) -> None:
    path = cfg.tables_dir() / "ablation_context_length.csv"
    if not path.exists():
        print("skip context_length (no table)")
        return
    df = pd.read_csv(path).sort_values("K")

    seed_sharpes = None
    learned = cfg.tables_dir() / "learned_test_metrics.csv"
    if learned.exists():
        ld = pd.read_csv(learned)
        seed_sharpes = ld[ld["strategy"] == "dt"]["sharpe"].values

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    ax.plot(df["K"], df["val_loss"] * 1e3, marker="s", color="indianred", label="best validation")
    if "final_train_loss" in df.columns:
        ax.plot(df["K"], df["final_train_loss"] * 1e3, marker="o", color="steelblue",
                linestyle="--", label="final training")
    ax.set_xlabel("Context length $K$ (timesteps)")
    ax.set_ylabel(r"Action MSE ($\times 10^{-3}$)")
    ax.set_ylim(3.0, 3.6)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("Imitation loss is flat in $K$")

    ax = axes[1]
    if seed_sharpes is not None and len(seed_sharpes):
        ax.scatter([30] * len(seed_sharpes), seed_sharpes, s=28, color="lightsteelblue",
                   zorder=2, label="$K{=}30$, all five seeds")
        ax.axhline(float(seed_sharpes.mean()), color="lightsteelblue", linestyle=":",
                   linewidth=1.2, zorder=1)
    ax.plot(df["K"], df["sharpe"], marker="o", color="steelblue", zorder=3,
            label="seed 42 at each $K$")
    ax.axhline(1.118, color="darkgreen", linestyle="--", linewidth=1.2, zorder=1,
               label="equal weight ($1/N$)")
    ax.set_xlabel("Context length $K$ (timesteps)")
    ax.set_ylabel("Test Sharpe ratio")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.set_title("No $K$ reaches the baseline")

    fig.tight_layout()
    fig.savefig(out_dir(cfg) / "context_length.png", dpi=150)
    plt.close(fig)
    print("wrote context_length.png")


def equity_curves(cfg: Config) -> None:
    from src.data_loader import compute_price_returns, get_split_mask, load_processed_data
    from src.eval.harness import _backtest_named_policies
    from src.features import build_or_load_features
    from src.policies import get_classical_baselines
    from src.trainer import get_device
    from scripts.run_paper_ablations import make_dt_policy

    bundle = load_processed_data(cfg)
    fb = build_or_load_features(bundle, cfg)
    returns = compute_price_returns(bundle.investable_prices).values
    returns = returns[len(returns) - len(fb.dates):]
    mask = get_split_mask(fb.dates, cfg.get("splits", "test_start"), cfg.get("splits", "test_end"))
    dates = fb.dates[mask]
    mu = float(cfg.get("env", "transaction_cost", default=0.001))
    rf = float(cfg.get("evaluation", "risk_free_rate", default=0.02))

    device = get_device(cfg)
    policies = get_classical_baselines(returns.shape[1])
    policies = {k: v for k, v in policies.items()
                if k in ("buy_and_hold", "momentum", "mean_variance", "risk_parity")}
    # seed 456 is the median-Sharpe DT run; seed 42 is the high-turnover outlier
    for name, fname in (("dt", "dt_seed456.pt"), ("bc", "bc_transformer_seed456.pt")):
        ckpt = cfg.checkpoint_dir() / fname
        if ckpt.exists():
            policies[name] = make_dt_policy(
                cfg, ckpt, fb.states.shape[1], returns.shape[1], device, name
            )
    _, series = _backtest_named_policies(policies, returns[mask], fb.states[mask], mu, rf)

    fig, ax = plt.subplots(figsize=(10, 5))
    # DT and BC land almost exactly on top of each other, so BC is drawn dashed on top
    style = {
        "buy_and_hold": dict(color="tab:blue", linewidth=1.6),
        "mean_variance": dict(color="tab:orange", linewidth=1.2),
        "momentum": dict(color="tab:green", linewidth=1.2),
        "risk_parity": dict(color="tab:brown", linewidth=1.2),
        "dt": dict(color="tab:red", linewidth=2.0),
        "bc": dict(color="black", linewidth=1.0, linestyle="--"),
    }
    order = ["buy_and_hold", "mean_variance", "momentum", "risk_parity", "dt", "bc"]
    for name in order:
        if name not in series:
            continue
        wealth = np.exp(np.cumsum(series[name]))
        ax.plot(dates[: len(wealth)], wealth, label=name, **style.get(name, {}))
    ax.set_yscale("log")
    ax.set_ylabel("Cumulative wealth (log scale, initial = 1)")
    ax.set_xlabel("Date")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir(cfg) / "equity_curves.png", dpi=150)
    plt.close(fig)
    print("wrote equity_curves.png")


if __name__ == "__main__":
    cfg = Config.from_yaml(ROOT / "configs/config.yaml")
    convergence(cfg)
    cost_sensitivity(cfg)
    context_length(cfg)
    equity_curves(cfg)
