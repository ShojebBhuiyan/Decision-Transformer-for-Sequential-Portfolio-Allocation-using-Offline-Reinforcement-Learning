"""Smoke-check the training pipeline: train briefly, then reload every checkpoint."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Config
from src.models.decision_transformer import DecisionTransformer
from src.models.offline_rl import CQL, IQL, TD3BC
from src.models.online_rl import A2C, PPO, SAC
from src.trainer import train_all_models

AGENT_CLASSES = {
    "td3bc": TD3BC, "iql": IQL, "cql": CQL,
    "ppo": PPO, "sac": SAC, "a2c": A2C,
}


def main() -> int:
    cfg = Config.from_yaml("configs/config.yaml")
    cfg.raw["training"]["seeds"] = [42]
    cfg.raw["model"]["max_epochs"] = 2
    cfg.raw["training"]["steps_per_epoch"] = 40
    cfg.raw["training"]["max_val_batches"] = 10

    started = time.time()
    results = train_all_models(cfg)
    print(f"--- trained in {time.time() - started:.0f}s ---")

    failures = [k for k, v in results.items() if str(v).startswith("failed")]
    if failures:
        print(f"TRAINING FAILURES: {failures}")
        return 1

    ckpt_dir = cfg.checkpoint_dir()
    problems: list[str] = []

    for name, path in results.items():
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        algo = name.split("_seed")[0]

        if algo in ("dt", "bc"):
            conf = ckpt["config"]
            model_cls = DecisionTransformer
            model = model_cls(
                conf["state_dim"], conf["action_dim"], conf["K"],
                conf["d_model"], conf["n_layers"], conf["n_heads"],
            )
            if conf["use_rtg"]:
                model.load_state_dict(ckpt["model_state_dict"])
            n = sum(p.numel() for p in ckpt["model_state_dict"].values())
            print(f"{name:16s} weights={n:>9,} val_loss={ckpt['val_loss']:.5f} epoch={ckpt['epoch']}")
        else:
            modules = ckpt.get("modules")
            if not modules:
                problems.append(f"{name}: no weights saved")
                continue
            conf = ckpt["config"]
            agent = AGENT_CLASSES[algo](conf["state_dim"], conf["action_dim"], device="cpu")
            for attr, state in modules.items():
                getattr(agent, attr).load_state_dict(state)
            n = sum(p.numel() for sd in modules.values() for p in sd.values())
            print(f"{name:16s} weights={n:>9,} modules={list(modules)} reloaded=ok")

    # Every history log should exist and cover all epochs
    for stem in ("dt", "bc_transformer"):
        log = cfg.run_dir() / f"{stem}_seed42_history.csv"
        rows = log.read_text().strip().splitlines()
        if len(rows) != 3:
            problems.append(f"{log.name}: expected 3 lines, got {len(rows)}")
        else:
            print(f"{log.name}: {len(rows) - 1} epochs logged")

    if problems:
        print("PROBLEMS:")
        for p in problems:
            print(" -", p)
        return 1

    print(f"\nAll {len(results)} checkpoints train, save and reload cleanly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
