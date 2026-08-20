#!/usr/bin/env python3
"""Entry point for the portfolio Decision Transformer pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import Config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Portfolio DT pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to YAML config",
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=[
            "prepare",
            "trajectories",
            "train",
            "evaluate",
            "ablations",
            "all",
        ],
        default="all",
        help="Pipeline stage to run",
    )
    args = parser.parse_args()
    cfg = Config.from_yaml(ROOT / args.config)
    print(f"Loaded config: {args.config}")
    print(f"Stage: {args.stage}")

    if args.stage in ("prepare", "all"):
        from src.data_loader import load_market_data, save_processed_data

        bundle = load_market_data(cfg)
        save_processed_data(bundle, cfg)
        print("Data prepared.")

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

    if args.stage in ("ablations", "all"):
        from src.eval.ablations import run_ablations

        run_ablations(cfg)
        print("Ablations complete.")


if __name__ == "__main__":
    main()
