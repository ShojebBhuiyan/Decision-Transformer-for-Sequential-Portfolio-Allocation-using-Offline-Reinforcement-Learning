"""Evaluation harness for headline holdout and walk-forward folds."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import Config
from src.data_loader import compute_price_returns, get_split_mask, load_processed_data
from src.eval.backtest import backtest_policy
from src.eval.metrics import compute_all_metrics
from src.eval.model_policy import (
    load_learned_policies,
    parse_manifest_key,
)
from src.eval.rtg_calibration import run_rtg_calibration, save_rtg_calibration
from src.eval.stats import significance_table
from src.features import build_features, load_features
from src.policies import Policy, get_classical_baselines


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
    df, _series = evaluate_split_series(
        price_returns, states, dates, start, end, transaction_cost, risk_free_rate
    )
    return df


def evaluate_split_series(
    price_returns: np.ndarray,
    states: np.ndarray,
    dates: pd.DatetimeIndex,
    start: str,
    end: str,
    transaction_cost: float,
    risk_free_rate: float,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Classical baselines plus their log-return series."""
    mask = get_split_mask(dates, start, end)
    if mask.sum() == 0:
        return pd.DataFrame(), {}

    split_returns = price_returns[mask]
    split_states = states[mask]
    baselines = get_classical_baselines(price_returns.shape[1])
    df, series = _backtest_named_policies(
        baselines, split_returns, split_states, transaction_cost, risk_free_rate
    )
    if df.empty:
        return df, series
    return df.set_index("strategy"), series


def _backtest_named_policies(
    policies: dict[str, Policy],
    split_returns: np.ndarray,
    split_states: np.ndarray,
    transaction_cost: float,
    risk_free_rate: float,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Backtest a dict of policies; return metrics rows and log-return series."""
    rows = []
    series: dict[str, np.ndarray] = {}
    for name, policy in policies.items():
        result = backtest_policy(split_returns, policy, split_states, transaction_cost)
        metrics = compute_all_metrics(
            result["log_returns"],
            turnovers=result["turnovers"],
            risk_free_rate=risk_free_rate,
        )
        metrics["strategy"] = name
        rows.append(metrics)
        series[name] = result["log_returns"]
    df = pd.DataFrame(rows) if rows else pd.DataFrame()
    return df, series


def evaluate_learned(
    cfg: Config,
    price_returns: np.ndarray,
    states: np.ndarray,
    dates: pd.DatetimeIndex,
    start: str,
    end: str,
    transaction_cost: float,
    risk_free_rate: float,
    policies: dict[str, Policy] | None = None,
) -> pd.DataFrame:
    """Backtest trained checkpoints on a date split (one row per seed)."""
    df, _series = evaluate_learned_series(
        cfg, price_returns, states, dates, start, end,
        transaction_cost, risk_free_rate, policies,
    )
    return df


def evaluate_learned_series(
    cfg: Config,
    price_returns: np.ndarray,
    states: np.ndarray,
    dates: pd.DatetimeIndex,
    start: str,
    end: str,
    transaction_cost: float,
    risk_free_rate: float,
    policies: dict[str, Policy] | None = None,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Learned-model backtest plus per-checkpoint log-return series."""
    mask = get_split_mask(dates, start, end)
    if mask.sum() == 0:
        return pd.DataFrame(), {}

    if policies is None:
        policies = load_learned_policies(
            cfg, state_dim=states.shape[1], action_dim=price_returns.shape[1]
        )
    if not policies:
        warnings.warn(
            "No trained checkpoints found in training_manifest.json; "
            "learned-model evaluation skipped."
        )
        return pd.DataFrame(), {}

    split_returns = price_returns[mask]
    split_states = states[mask]
    df, series = _backtest_named_policies(
        policies, split_returns, split_states, transaction_cost, risk_free_rate
    )
    if df.empty:
        return df, series

    parsed = df["strategy"].map(parse_manifest_key)
    df["seed"] = parsed.map(lambda x: x[1])
    df["strategy"] = parsed.map(lambda x: x[0])
    return df, series


def aggregate_learned_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Mean ± std across seeds; index is the algorithm name."""
    if df.empty:
        return df
    metric_cols = [
        c for c in df.columns if c not in ("strategy", "seed") and np.issubdtype(df[c].dtype, np.number)
    ]
    mean = df.groupby("strategy")[metric_cols].mean()
    std = df.groupby("strategy")[metric_cols].std(ddof=0)
    out = mean.copy()
    for col in metric_cols:
        out[f"{col}_std"] = std[col]
    return out


def fold_is_in_sample(start: str, train_end: str) -> bool:
    """True when the fold begins on or before the training cutoff."""
    return pd.Timestamp(start) <= pd.Timestamp(train_end)


def aggregate_walkforward_learned(df: pd.DataFrame) -> pd.DataFrame:
    """Mean ± std across seeds per strategy and walk-forward fold."""
    if df.empty:
        return df
    keys = ["strategy", "fold", "period", "in_sample"]
    metric_cols = [
        c for c in df.columns
        if c not in keys + ["seed"] and np.issubdtype(df[c].dtype, np.number)
    ]
    mean = df.groupby(keys, as_index=False)[metric_cols].mean()
    std = df.groupby(keys, as_index=False)[metric_cols].std(ddof=0)
    std = std.rename(columns={c: f"{c}_std" for c in metric_cols})
    return mean.merge(std, on=keys)


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

    # Headline splits (train/val); test is computed with series for stats
    for split_name, start, end in [
        ("train", cfg.get("splits", "train_start"), cfg.get("splits", "train_end")),
        ("val", cfg.get("splits", "val_start"), cfg.get("splits", "val_end")),
    ]:
        df = evaluate_split(returns_aligned, fb.states, fb.dates, start, end, tc, rf)
        results[split_name] = df

    test_start = cfg.get("splits", "test_start")
    test_end = cfg.get("splits", "test_end")
    test_df, test_series = evaluate_split_series(
        returns_aligned, fb.states, fb.dates, test_start, test_end, tc, rf
    )
    results["test"] = test_df

    # Walk-forward folds
    wf_results = []
    for i, (start, end) in enumerate(WALK_FORWARD_FOLDS):
        df = evaluate_split(returns_aligned, fb.states, fb.dates, start, end, tc, rf)
        if not df.empty:
            df["fold"] = i
            df["period"] = f"{start}_{end}"
            wf_results.append(df)

    # Learned models on the headline test split (load checkpoints once)
    learned_policies = load_learned_policies(
        cfg, state_dim=fb.states.shape[1], action_dim=returns_aligned.shape[1]
    )
    learned_df, learned_series = evaluate_learned_series(
        cfg,
        returns_aligned,
        fb.states,
        fb.dates,
        test_start,
        test_end,
        tc,
        rf,
        policies=learned_policies,
    )

    train_end = cfg.get("splits", "train_end")
    wf_learned = []
    if learned_policies:
        for i, (start, end) in enumerate(WALK_FORWARD_FOLDS):
            df = evaluate_learned(
                cfg, returns_aligned, fb.states, fb.dates, start, end, tc, rf,
                policies=learned_policies,
            )
            if not df.empty:
                df["fold"] = i
                df["period"] = f"{start}_{end}"
                df["in_sample"] = fold_is_in_sample(start, train_end)
                wf_learned.append(df)

    # Save results
    out_dir = cfg.tables_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    test_df = results.get("test", pd.DataFrame())
    headline = test_df
    if not learned_df.empty:
        learned_df.to_csv(out_dir / "learned_test_metrics.csv", index=False)
        learned_agg = aggregate_learned_metrics(learned_df)
        mean_cols = [c for c in learned_agg.columns if not c.endswith("_std")]
        headline = pd.concat([test_df, learned_agg[mean_cols]]) if not test_df.empty else learned_agg[mean_cols]
        headline.index.name = "strategy"

    if not headline.empty:
        headline.to_csv(out_dir / "headline_test_metrics.csv")

    cal_df = run_rtg_calibration(
        cfg, returns_aligned, fb.states, fb.dates,
        test_start, test_end, tc,
    )
    save_rtg_calibration(cfg, cal_df)

    all_series = {**test_series, **learned_series}
    if "buy_and_hold" in all_series:
        sig = significance_table(all_series, benchmark="buy_and_hold", risk_free_rate=rf)
        sig.to_csv(out_dir / "significance_tests.csv", index=False)
    else:
        sig = pd.DataFrame()

    if wf_results:
        wf_df = pd.concat(wf_results)
        wf_df.to_csv(out_dir / "walkforward_metrics.csv")

    if wf_learned:
        wf_learned_df = aggregate_walkforward_learned(pd.concat(wf_learned, ignore_index=True))
        wf_learned_df.to_csv(out_dir / "walkforward_learned_metrics.csv", index=False)
    else:
        wf_learned_df = pd.DataFrame()

    summary = {
        "test_sharpe": test_df["sharpe"].to_dict() if not test_df.empty else {},
        "learned_sharpe_mean": (
            learned_df.groupby("strategy")["sharpe"].mean().to_dict() if not learned_df.empty else {}
        ),
        "learned_sharpe_std": (
            learned_df.groupby("strategy")["sharpe"].std(ddof=0).to_dict() if not learned_df.empty else {}
        ),
        "n_folds": len(wf_results),
        "n_learned_rows": int(len(learned_df)),
        "n_walkforward_learned_rows": int(len(wf_learned_df)),
        "n_rtg_calibration_rows": int(len(cal_df)),
        "n_significance_rows": int(len(sig)),
    }
    with open(out_dir / "eval_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    return headline
