"""Universe B robustness pipeline (Bitcoin + Aluminum, from 2014-09-17)."""

from __future__ import annotations

from src.config import Config
from src.data_loader import (
    compute_price_returns,
    load_market_data,
    load_processed_data,
    save_processed_data,
    universe_b_config,
)
from src.eval.harness import evaluate_split
from src.features import load_features
from src.trajectories import generate_trajectories


def prepare_universe_b(cfg: Config) -> Config:
    """Load Universe B market data into scoped processed/ directories."""
    cfg_b = universe_b_config(cfg)
    bundle = load_market_data(cfg_b)
    save_processed_data(bundle, cfg_b)
    return cfg_b


def run_universe_b_classical(cfg: Config, generate_traj: bool = True) -> dict:
    """Prepare Universe B, optionally synthesize trajectories, evaluate classical baselines."""
    cfg_b = prepare_universe_b(cfg)
    if generate_traj:
        generate_trajectories(cfg_b)

    bundle = load_processed_data(cfg_b)
    fb = load_features(cfg_b)
    returns = compute_price_returns(bundle.investable_prices).values
    offset = len(returns) - len(fb.dates)
    returns_aligned = returns[offset:]
    tc = float(cfg_b.get("env", "transaction_cost", default=0.001))
    rf = float(cfg_b.get("evaluation", "risk_free_rate", default=0.02))
    df = evaluate_split(
        returns_aligned, fb.states, fb.dates,
        cfg_b.get("splits", "test_start"),
        cfg_b.get("splits", "test_end"),
        tc, rf,
    )
    out = cfg_b.tables_dir()
    out.mkdir(parents=True, exist_ok=True)
    if not df.empty:
        df.to_csv(out / "headline_test_metrics.csv")
    return {
        "universe": "B",
        "n_assets": len(bundle.universe),
        "n_days": int(bundle.metadata.get("n_trading_days", len(bundle.investable_prices))),
        "start_date": bundle.metadata.get("start_date"),
        "end_date": bundle.metadata.get("end_date"),
        "processed_dir": str(cfg_b.processed_dir()),
        "trajectories_dir": str(cfg_b.trajectories_dir()),
        "tables_dir": str(out),
        "test_sharpe": df["sharpe"].to_dict() if not df.empty else {},
    }
