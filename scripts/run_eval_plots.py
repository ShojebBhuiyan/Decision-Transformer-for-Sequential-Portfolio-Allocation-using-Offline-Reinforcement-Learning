"""Generate evaluation plots for notebook 02."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

OUT = ROOT / "results" / "figures" / "eval"
OUT.mkdir(parents=True, exist_ok=True)
TABLES = ROOT / "results" / "tables"


def main():
    test_path = TABLES / "headline_test_metrics.csv"
    if not test_path.exists():
        print("No test metrics found. Run evaluation first.")
        return

    df = pd.read_csv(test_path, index_col=0)

    # Sharpe comparison bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    df["sharpe"].sort_values().plot(kind="barh", ax=ax, color="steelblue")
    ax.set_xlabel("Sharpe Ratio")
    ax.set_title("Test Period (2020-2025) — Classical Baseline Sharpe Ratios")
    ax.axvline(0, color="gray", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(OUT / "sharpe_comparison.png", dpi=120)
    plt.close()

    # Risk-return scatter
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df["volatility"], df["cagr"], s=100, alpha=0.7)
    for name, row in df.iterrows():
        ax.annotate(name, (row["volatility"], row["cagr"]), fontsize=8)
    ax.set_xlabel("Annualized Volatility")
    ax.set_ylabel("CAGR")
    ax.set_title("Risk-Return Profile (Test Period)")
    fig.tight_layout()
    fig.savefig(OUT / "risk_return_scatter.png", dpi=120)
    plt.close()

    # Transaction cost ablation
    tc_path = TABLES / "ablation_transaction_cost.csv"
    if tc_path.exists():
        tc = pd.read_csv(tc_path)
        fig, ax = plt.subplots(figsize=(10, 5))
        for strat in tc["strategy"].unique():
            sub = tc[tc["strategy"] == strat]
            ax.plot(sub["transaction_cost"], sub["sharpe"], marker="o", label=strat)
        ax.set_xlabel("Transaction Cost (mu)")
        ax.set_ylabel("Sharpe Ratio")
        ax.set_title("Transaction Cost Sensitivity")
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(OUT / "ablation_transaction_cost.png", dpi=120)
        plt.close()

    # Walk-forward if available
    wf_path = TABLES / "walkforward_metrics.csv"
    if wf_path.exists():
        wf = pd.read_csv(wf_path)
        pivot = wf.pivot_table(index="strategy", columns="period", values="sharpe")
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn", center=0, ax=ax)
        ax.set_title("Walk-Forward Sharpe Ratios by Period")
        fig.tight_layout()
        fig.savefig(OUT / "walkforward_sharpe.png", dpi=120)
        plt.close()

    # RTG calibration
    cal_path = TABLES / "rtg_calibration.csv"
    if cal_path.exists():
        from src.eval.rtg_calibration import plot_rtg_calibration

        cal = pd.read_csv(cal_path)
        plot_rtg_calibration(cal, OUT / "rtg_calibration.png")

    print(f"Evaluation figures saved to {OUT}")


if __name__ == "__main__":
    main()
