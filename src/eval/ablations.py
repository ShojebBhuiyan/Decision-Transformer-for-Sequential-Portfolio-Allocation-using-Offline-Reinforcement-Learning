"""Ablation experiments."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pandas as pd

from src.config import Config
from src.data_loader import compute_price_returns, load_processed_data
from src.eval.harness import evaluate_split
from src.features import build_features, load_features


def run_ablations(cfg: Config) -> dict:
    """Run ablation suite and save results."""
    out_dir = cfg.project_root / "results" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    ablation_results = {}

    # 1. Transaction cost sensitivity (fast, no retraining)
    tc_results = []
    for mu in cfg.get("ablations", "transaction_costs", default=[0.0, 0.0005, 0.001, 0.0025]):
        cfg_mu = deepcopy(cfg)
        cfg_mu.raw.setdefault("env", {})["transaction_cost"] = mu
        bundle = load_processed_data(cfg_mu)
        try:
            fb = load_features(cfg_mu)
        except FileNotFoundError:
            fb = build_features(bundle, cfg_mu)
        returns = compute_price_returns(bundle.investable_prices).values
        offset = len(returns) - len(fb.dates)
        returns_aligned = returns[offset:]

        df = evaluate_split(
            returns_aligned, fb.states, fb.dates,
            cfg_mu.get("splits", "test_start"),
            cfg_mu.get("splits", "test_end"),
            mu, float(cfg_mu.get("evaluation", "risk_free_rate", default=0.02)),
        )
        if not df.empty:
            df["transaction_cost"] = mu
            tc_results.append(df)

    if tc_results:
        tc_df = pd.concat(tc_results)
        tc_df.to_csv(out_dir / "ablation_transaction_cost.csv")
        ablation_results["transaction_cost"] = "saved"

    # 2. Context length / RTG / Universe B — documented for full GPU runs
    ablation_results["context_length"] = {
        "note": "Train with model.context_length in {10,20,30,50} via train_dt(..., context_length=K)",
        "values": cfg.get("ablations", "context_lengths", default=[10, 20, 30, 50]),
    }
    ablation_results["rtg_conditioning"] = {
        "note": "Compare train_dt(use_rtg=True) vs train_dt(use_rtg=False)",
        "dt_checkpoint": "results/checkpoints/dt_seed42.pt",
        "bc_checkpoint": "results/checkpoints/bc_transformer_seed42.pt",
    }
    ablation_results["universe_b"] = {
        "note": "Set data.universe=B, data.start_date=2014-09-17 and re-run pipeline",
    }
    ablation_results["feature_ablation"] = {
        "minimal": "data.feature_set=minimal (default in config)",
        "full": "data.feature_set=full",
    }

    manifest_path = out_dir / "ablation_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(ablation_results, f, indent=2, default=str)

    return ablation_results
