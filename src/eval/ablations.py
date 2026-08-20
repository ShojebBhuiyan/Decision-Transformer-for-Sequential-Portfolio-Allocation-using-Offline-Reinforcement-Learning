"""Ablation experiments."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pandas as pd

from src.config import Config
from src.eval.harness import evaluate_split, run_evaluation
from src.trainer import train_dt


def run_ablations(cfg: Config) -> dict:
    """Run ablation suite and save results."""
    out_dir = cfg.project_root / "results" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    ablation_results = {}

    # 1. Context length K ablation
    k_results = []
    for K in cfg.get("ablations", "context_lengths", default=[10, 20, 30, 50]):
        cfg_k = deepcopy(cfg)
        cfg_k.raw.setdefault("model", {})["context_length"] = K
        cfg_k.raw["model"]["max_epochs"] = 10  # shorter for ablations
        ckpt = train_dt(cfg_k, seed=42, use_rtg=True, context_length=K)
        k_results.append({"K": K, "checkpoint": str(ckpt)})
    ablation_results["context_length"] = k_results

    # 2. RTG conditioning on/off (DT vs BC)
    dt_ckpt = train_dt(cfg, seed=42, use_rtg=True)
    bc_ckpt = train_dt(cfg, seed=42, use_rtg=False)
    ablation_results["rtg_conditioning"] = {
        "dt": str(dt_ckpt),
        "bc": str(bc_ckpt),
    }

    # 3. Transaction cost sensitivity
    tc_results = []
    for mu in cfg.get("ablations", "transaction_costs", default=[0.0, 0.0005, 0.001, 0.0025]):
        cfg_mu = deepcopy(cfg)
        cfg_mu.raw.setdefault("env", {})["transaction_cost"] = mu
        from src.data_loader import load_processed_data
        from src.features import load_features, build_features
        from src.data_loader import compute_price_returns

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
        ablation_results["transaction_cost"] = tc_df.to_dict()

    # 4. Universe B robustness (if not already on B)
    if cfg.get("data", "universe", default="A") == "A":
        cfg_b = deepcopy(cfg)
        cfg_b.raw.setdefault("data", {})["universe"] = "B"
        cfg_b.raw["data"]["start_date"] = "2014-09-17"
        ablation_results["universe_b"] = {"note": "Universe B requires re-running prepare stage"}

    # Save ablation manifest
    manifest_path = out_dir / "ablation_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(ablation_results, f, indent=2, default=str)

    return ablation_results
