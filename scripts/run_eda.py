"""Generate EDA figures for notebook 01."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import Config
from src.data_loader import load_market_data, load_raw_market_csv, align_to_nyse_calendar

OUT = ROOT / "results" / "figures" / "eda"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    cfg = Config.from_yaml(ROOT / "configs/config.yaml")
    bundle = load_market_data(cfg)
    raw = load_raw_market_csv(ROOT / "dataset")
    events = bundle.events

    # 1. Coverage heatmap
    aligned = align_to_nyse_calendar(raw)
    coverage = aligned.notna().astype(float)
    sample = coverage.iloc[::20]  # subsample for readability
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(sample.T, cmap="YlGn", cbar_kws={"label": "Available"}, ax=ax, xticklabels=20)
    ax.set_title("Data Coverage Heatmap (NYSE calendar, subsampled)")
    ax.set_xlabel("Date (subsampled)")
    fig.tight_layout()
    fig.savefig(OUT / "coverage_heatmap.png", dpi=120)
    plt.close()

    # 2. Return distributions
    prices = bundle.investable_prices
    log_rets = np.log(prices / prices.shift(1)).dropna()
    fig, axes = plt.subplots(3, 5, figsize=(18, 10))
    for i, col in enumerate(prices.columns[:15]):
        ax = axes[i // 5, i % 5]
        ax.hist(log_rets[col].dropna(), bins=50, alpha=0.7, edgecolor="black")
        ax.set_title(col[:20], fontsize=8)
    fig.suptitle("Daily Log Return Distributions (Universe A)")
    fig.tight_layout()
    fig.savefig(OUT / "return_distributions.png", dpi=120)
    plt.close()

    # 3. Rolling volatility (SPY proxy)
    spy_col = "S&P 500 ETF"
    if spy_col in log_rets.columns:
        vol20 = log_rets[spy_col].rolling(20).std() * np.sqrt(252)
        vol60 = log_rets[spy_col].rolling(60).std() * np.sqrt(252)
        fig, ax = plt.subplots(figsize=(12, 4))
        vol20.plot(ax=ax, label="20d vol", alpha=0.8)
        vol60.plot(ax=ax, label="60d vol", alpha=0.8)
        for _, ev in events.iterrows():
            ax.axvline(ev["DATE"], color="red", alpha=0.3, linestyle="--")
        ax.set_title(f"Rolling Volatility — {spy_col}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUT / "rolling_volatility.png", dpi=120)
        plt.close()

    # 4. Drawdown timeline
    if spy_col in prices.columns:
        wealth = (1 + log_rets[spy_col]).cumprod()
        peak = wealth.cummax()
        dd = (wealth - peak) / peak
        fig, ax = plt.subplots(figsize=(12, 4))
        dd.plot(ax=ax, color="darkred")
        for _, ev in events.iterrows():
            ax.axvline(ev["DATE"], color="gray", alpha=0.5, linestyle=":")
            ax.text(ev["DATE"], dd.min() * 0.9, ev["FINANCIAL EVENT"][:15], rotation=90, fontsize=6)
        ax.set_title(f"Drawdown — {spy_col}")
        ax.set_ylabel("Drawdown")
        fig.tight_layout()
        fig.savefig(OUT / "drawdown_timeline.png", dpi=120)
        plt.close()

    # 5. Correlation clustering
    corr = log_rets.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=ax, xticklabels=True, yticklabels=True)
    ax.set_title("Cross-Asset Correlation Matrix")
    fig.tight_layout()
    fig.savefig(OUT / "correlation_matrix.png", dpi=120)
    plt.close()

    # 6. Autocorrelation (SPY)
    if spy_col in log_rets.columns:
        from pandas.plotting import autocorrelation_plot
        fig, ax = plt.subplots(figsize=(8, 4))
        pd.plotting.autocorrelation_plot(log_rets[spy_col].dropna(), ax=ax)
        ax.set_title(f"Autocorrelation — {spy_col}")
        fig.tight_layout()
        fig.savefig(OUT / "autocorrelation.png", dpi=120)
        plt.close()

    print(f"EDA figures saved to {OUT}")


if __name__ == "__main__":
    main()
