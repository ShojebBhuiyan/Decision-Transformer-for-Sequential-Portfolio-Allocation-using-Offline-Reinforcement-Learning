"""Paper ablations: Decision Transformer context length and transaction-cost sensitivity.

Writes ``results/tables/ablation_context_length.csv`` and
``results/tables/ablation_transaction_cost.csv``. Both are idempotent: the
context-length sweep reuses any checkpoint that already exists (K=30 reuses the
main ``dt_seed42`` checkpoint, which has an identical configuration).
"""

from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.config import Config
from src.data_loader import compute_price_returns, get_split_mask, load_processed_data
from src.eval.harness import _backtest_named_policies
from src.eval.model_policy import (
    DTPolicy,
    default_target_rtg,
    load_rtg_stats,
    load_transformer,
)
from src.features import build_or_load_features
from src.policies import get_classical_baselines
from src.trainer import get_device, train_dt

K_GRID = [10, 20, 30, 50]


def load_test_arrays(cfg: Config):
    bundle = load_processed_data(cfg)
    fb = build_or_load_features(bundle, cfg)
    returns = compute_price_returns(bundle.investable_prices).values
    offset = len(returns) - len(fb.dates)
    returns = returns[offset:]
    mask = get_split_mask(
        fb.dates,
        cfg.get("splits", "test_start"),
        cfg.get("splits", "test_end"),
    )
    return returns[mask], fb.states[mask], fb.states.shape[1], returns.shape[1]


def make_dt_policy(cfg: Config, ckpt: Path, state_dim: int, action_dim: int,
                   device: str, name: str) -> DTPolicy:
    model = load_transformer(ckpt, device, state_dim, action_dim)
    rtg_mean, rtg_std = load_rtg_stats(cfg)
    return DTPolicy(
        model,
        device=device,
        target_rtg=default_target_rtg(cfg),
        rtg_mean=rtg_mean,
        rtg_std=rtg_std,
        reset_horizon=int(cfg.get("env", "episode_length", default=252)),
        name=name,
    )


def context_length_ablation(cfg: Config) -> pd.DataFrame:
    import torch

    device = get_device(cfg)
    ckpt_dir = cfg.checkpoint_dir()
    log_dir = cfg.run_dir()
    mu = float(cfg.get("env", "transaction_cost", default=0.001))
    rf = float(cfg.get("evaluation", "risk_free_rate", default=0.02))
    test_returns, test_states, state_dim, action_dim = load_test_arrays(cfg)

    rows = []
    for K in K_GRID:
        if K == 30:
            ckpt = ckpt_dir / "dt_seed42.pt"
            hist = log_dir / "dt_seed42_history.csv"
        else:
            ckpt = ckpt_dir / f"dt_K{K}_seed42.pt"
            hist = log_dir / f"dt_K{K}_seed42_history.csv"
        if not ckpt.exists():
            print(f"[K={K}] training ...")
            cfg_k = deepcopy(cfg)
            cfg_k.raw.setdefault("model", {})["context_length"] = K
            ckpt = Path(train_dt(cfg_k, seed=42, use_rtg=True,
                                 context_length=K, run_tag=f"_K{K}"))
        print(f"[K={K}] backtesting {ckpt.name}")
        policy = make_dt_policy(cfg, ckpt, state_dim, action_dim, device, f"dt_K{K}")
        df, _ = _backtest_named_policies(
            {f"dt_K{K}": policy}, test_returns, test_states, mu, rf
        )
        row = df.iloc[0].to_dict()
        payload = torch.load(ckpt, map_location="cpu", weights_only=False)
        row["K"] = K
        row["val_loss"] = float(payload.get("val_loss", np.nan))
        row["best_epoch"] = int(payload.get("epoch", -1))
        if hist.exists():
            h = pd.read_csv(hist)
            row["final_train_loss"] = float(h["train_loss"].iloc[-1])
        rows.append(row)

    out = pd.DataFrame(rows).set_index("K")
    path = cfg.tables_dir() / "ablation_context_length.csv"
    out.to_csv(path)
    print(f"wrote {path}")
    return out


def transaction_cost_ablation(cfg: Config) -> pd.DataFrame:
    """Re-backtest classical baselines and every DT/BC seed at several cost levels."""
    device = get_device(cfg)
    ckpt_dir = cfg.checkpoint_dir()
    rf = float(cfg.get("evaluation", "risk_free_rate", default=0.02))
    grid = cfg.get("ablations", "transaction_costs", default=[0.0, 0.0005, 0.001, 0.0025])
    seeds = cfg.get("training", "seeds", default=[42])
    test_returns, test_states, state_dim, action_dim = load_test_arrays(cfg)

    frames = []
    for mu in grid:
        policies = get_classical_baselines(action_dim)
        for algo, stem in (("dt", "dt"), ("bc", "bc_transformer")):
            for seed in seeds:
                ckpt = ckpt_dir / f"{stem}_seed{seed}.pt"
                if ckpt.exists():
                    policies[f"{algo}_seed{seed}"] = make_dt_policy(
                        cfg, ckpt, state_dim, action_dim, device, f"{algo}_seed{seed}"
                    )
        print(f"[mu={mu}] backtesting {len(policies)} strategies")
        df, _ = _backtest_named_policies(policies, test_returns, test_states, float(mu), rf)
        df["transaction_cost"] = mu
        frames.append(df)

    out = pd.concat(frames)
    out["algo"] = out["strategy"].str.replace(r"_seed\d+", "", regex=True)
    out = out.set_index(["transaction_cost", "strategy"])
    path = cfg.tables_dir() / "ablation_transaction_cost.csv"
    out.to_csv(path)
    print(f"wrote {path}")
    return out


CONCENTRATION_MODELS = [
    ("dt", "dt"),
    ("bc", "bc_transformer"),
    ("td3bc", "td3bc"),
    ("iql", "iql"),
    ("cql", "cql"),
    ("ppo", "ppo"),
    ("sac", "sac"),
    ("a2c", "a2c"),
]


def concentration_table(cfg: Config, only: str | None = None) -> pd.DataFrame:
    """Mean test-window allocation per model and seed: Herfindahl index and top holdings."""
    import gc
    import json

    import torch

    from src.eval.backtest import backtest_policy
    from src.eval.model_policy import default_target_rtg, policy_from_checkpoint
    from src.policies import EqualWeight

    device = get_device(cfg)
    mu = float(cfg.get("env", "transaction_cost", default=0.001))
    seeds = cfg.get("training", "seeds", default=[42])
    test_returns, test_states, state_dim, action_dim = load_test_arrays(cfg)
    names = json.loads((cfg.processed_dir() / "universe.json").read_text(encoding="utf-8"))
    try:
        target = default_target_rtg(cfg)
    except FileNotFoundError:
        target = 0.0

    def summarise(model: str, seed, policy) -> dict:
        result = backtest_policy(test_returns, policy, test_states, mu)
        w = np.asarray(result["weights"], dtype=float).mean(axis=0)
        order = np.argsort(-w)[:3]
        return {
            "model": model,
            "seed": seed,
            "hhi": float((w ** 2).sum()),
            "max_mean_weight": float(w.max()),
            "mean_turnover": float(np.mean(result["turnovers"])),
            "top1": f"{names[order[0]]} {w[order[0]]:.3f}",
            "top2": f"{names[order[1]]} {w[order[1]]:.3f}",
            "top3": f"{names[order[2]]} {w[order[2]]:.3f}",
        }

    models = CONCENTRATION_MODELS if only is None else [
        (m, st) for m, st in CONCENTRATION_MODELS if m == only
    ]
    rows = [] if only else [summarise("equal_weight", "-", EqualWeight(action_dim))]
    for model, stem in models:
        for seed in seeds:
            ckpt = cfg.checkpoint_dir() / f"{stem}_seed{seed}.pt"
            if not ckpt.exists():
                continue
            policy = policy_from_checkpoint(
                cfg, f"{model}_seed{seed}", ckpt, state_dim, action_dim, target, device, 252
            )
            rows.append(summarise(model, seed, policy))
            print(f"  {model} seed {seed}: HHI={rows[-1]['hhi']:.4f}", flush=True)
            del policy
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    out = pd.DataFrame(rows).set_index(["model", "seed"])
    path = cfg.tables_dir() / "portfolio_concentration.csv"
    if only and path.exists():
        prev = pd.read_csv(path).set_index(["model", "seed"])
        prev = prev.drop(index=only, errors="ignore")
        out = pd.concat([prev, out])
    out.to_csv(path)
    print(f"wrote {path}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["k", "cost", "conc", "all"], default="all")
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--only", default=None,
                    help="conc only: restrict to this model key (e.g. dt)")
    args = ap.parse_args()
    cfg = Config.from_yaml(args.config)
    cfg.tables_dir().mkdir(parents=True, exist_ok=True)
    if args.part in ("cost", "all"):
        transaction_cost_ablation(cfg)
    if args.part in ("conc", "all"):
        concentration_table(cfg, only=args.only)
    if args.part in ("k", "all"):
        context_length_ablation(cfg)


if __name__ == "__main__":
    main()
