"""Configuration loader."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Config:
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return cls(raw=raw)

    def get(self, *keys: str, default: Any = None) -> Any:
        node: Any = self.raw
        for key in keys:
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node

    @property
    def project_root(self) -> Path:
        return Path(__file__).resolve().parents[1]

    @property
    def artifact_scope(self) -> str:
        """Subdirectory for non-default universes; empty for Universe A (legacy paths)."""
        universe = str(self.get("data", "universe", default="A")).upper()
        if universe == "B":
            return "universe_B"
        return ""

    def scoped_path(self, relative: str | Path) -> Path:
        path = self.project_root / relative
        if self.artifact_scope:
            path = path / self.artifact_scope
        return path

    def processed_dir(self) -> Path:
        return self.scoped_path("data/processed")

    def trajectories_dir(self) -> Path:
        return self.scoped_path(
            self.get("trajectories", "output_dir", default="data/trajectories")
        )

    def checkpoint_dir(self) -> Path:
        return self.scoped_path(
            self.get("training", "checkpoint_dir", default="results/checkpoints")
        )

    def run_dir(self) -> Path:
        return self.scoped_path(self.get("training", "log_dir", default="results/runs"))

    def tables_dir(self) -> Path:
        return self.scoped_path("results/tables")

    def figures_eval_dir(self) -> Path:
        return self.scoped_path("results/figures/eval")

    def training_manifest_path(self) -> Path:
        if self.artifact_scope:
            return self.project_root / "results" / self.artifact_scope / "training_manifest.json"
        return self.project_root / "results" / "training_manifest.json"
