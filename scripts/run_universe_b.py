#!/usr/bin/env python3
"""Run the pipeline against Universe B without touching Universe A artifacts.

    python scripts/run_universe_b.py --stage prepare
    python scripts/run_universe_b.py --stage trajectories
    python scripts/run_universe_b.py --stage train
    python scripts/run_universe_b.py --stage evaluate
    python scripts/run_universe_b.py --stage all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import Config  # noqa: E402
from src.data_loader import universe_b_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Universe B pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Base YAML config (universe is overridden to B)",
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=["prepare", "trajectories", "train", "evaluate", "all"],
        default="all",
    )
    args = parser.parse_args()
    cfg = universe_b_config(Config.from_yaml(ROOT / args.config))
    print(f"Universe {cfg.get('data', 'universe')}  scope={cfg.artifact_scope!r}")
    print(f"processed:     {cfg.processed_dir()}")
    print(f"trajectories:  {cfg.trajectories_dir()}")
    print(f"checkpoints:   {cfg.checkpoint_dir()}")
    print(f"Stage: {args.stage}")

    if args.stage in ("prepare", "all"):
        from src.data_loader import load_market_data, save_processed_data

        bundle = load_market_data(cfg)
        save_processed_data(bundle, cfg)
        print(f"Data prepared: {bundle.metadata}")

    if args.stage in ("trajectories", "all"):
        from src.trajectories import generate_trajectories

        generate_trajectories(cfg)
        print("Trajectories generated.")

    if args.stage in ("train", "all"):
        from src.trainer import train_all_models

        train_all_models(cfg)
        print("Training complete.")

    if args.stage in ("evaluate", "all"):
        from src.eval.harness import run_evaluation

        run_evaluation(cfg)
        print("Evaluation complete.")


if __name__ == "__main__":
    main()
