"""Evaluation harness for headline holdout and walk-forward folds."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.config import Config
from src.data_loader import compute_price_returns, get_split_mask, load_processed_data
from src.eval.backtest import backtest_all_baselines, backtest_policy
from src.eval.metrics import compute_all_metrics
from src.features import build_features, load_features
from src.policies import get_classical_baselines


WALK_FORWARD_FOLDS = [
    ("2014-01-01", "2015-12-31"),
    ("2016-01-01", "2017-12-31"),
    ("2018-01-01", "2019-12-31"),
    ("2020-01-01", "2021-12-31"),
    ("2022-01-01", "2023-12-31"),
    ("2024-01-01", "2025-12-30"),
]


def evaluate_split(
    price_returns: np.ndarray,
    states: np.ndarray,
    dates: pd.DatetimeIndex,
    start: str,
    end: str,
    transaction_cost: float,
    risk_free_rate: float,
) -> pd.DataFrame:
    """Evaluate classical baselines on a date split."""
    mask = get_split_mask(dates, start, end)
    if mask.sum() == 0:
        return pd.DataFrame()

    split_returns = price_returns[mask]
    split_states = states[mask]
    n_assets = price_returns.shape[1]
    baselines = get_classical_baselines(n_assets)

    return backtest_all_baselines(
        split_returns, baselines, split_states, transaction_cost
    )


def run_evaluation(cfg: Config) -> pd.DataFrame:
    """Run headline holdout evaluation."""
    bundle = load_processed_data(cfg)
    try:
        fb = load_features(cfg)
    except FileNotFoundError:
        fb = build_features(bundle, cfg)

    returns = compute_price_returns(bundle.investable_prices).values
    # Align returns with feature dates (features start later due to lookback)
    offset = len(returns) - len(fb.dates)
    returns_aligned = returns[offset:]

    tc = float(cfg.get("env", "transaction_cost", default=0.001))
    rf = float(cfg.get("evaluation", "risk_free_rate", default=0.02))

    results = {}

    # Headline splits
    for split_name, start, end in [
        ("train", cfg.get("splits", "train_start"), cfg.get("splits", "train_end")),
        ("val", cfg.get("splits", "val_start"), cfg.get("splits", "val_end")),
        ("test", cfg.get("splits", "test_start"), cfg.get("splits", "test_end")),
    ]:
        df = evaluate_split(returns_aligned, fb.states, fb.dates, start, end, tc, rf)
        results[split_name] = df

    # Walk-forward folds
    wf_results = []
    for i, (start, end) in enumerate(WALK_FORWARD_FOLDS):
        df = evaluate_split(returns_aligned, fb.states, fb.dates, start, end, tc, rf)
        if not df.empty:
            df["fold"] = i
            df["period"] = f"{start}_{end}"
            wf_results.append(df)

    # Save results
    out_dir = cfg.project_root / "results" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    test_df = results.get("test", pd.DataFrame())
    if not test_df.empty:
        test_df.to_csv(out_dir / "headline_test_metrics.csv")

    if wf_results:
        wf_df = pd.concat(wf_results)
        wf_df.to_csv(out_dir / "walkforward_metrics.csv")

    summary = {
        "test_sharpe": test_df["sharpe"].to_dict() if not test_df.empty else {},
        "n_folds": len(wf_results),
    }
    with open(out_dir / "eval_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    return test_df
