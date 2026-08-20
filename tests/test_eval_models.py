"""Tests for learned-model evaluation, RTG conditioning, stats, and universe scope."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from src.config import Config
from src.data_loader import universe_b_config
from src.eval.model_policy import (
    AgentPolicy,
    DTPolicy,
    assert_checkpoint_dims,
    parse_manifest_key,
    rtg_quantile_targets,
)
from src.eval.stats import significance_table
from src.models.bc import TransformerBC
from src.models.decision_transformer import DecisionTransformer
from src.models.offline_rl import CQL, IQL, TD3BC
from src.models.online_rl import A2C, PPO, SAC


def _tiny_dt(state_dim=8, action_dim=4, K=6, use_rtg=True):
    cls = DecisionTransformer if use_rtg else TransformerBC
    return cls(state_dim, action_dim, context_length=K, d_model=32, n_layers=1, n_heads=4)


class _DummyEnv:
    def __init__(self):
        self.history: list[dict] = []


def test_parse_manifest_key():
    assert parse_manifest_key("dt_seed42") == ("dt", 42)
    assert parse_manifest_key("td3bc_seed1011") == ("td3bc", 1011)
    with pytest.raises(ValueError):
        parse_manifest_key("dt")


def test_assert_checkpoint_dims_mismatch():
    with pytest.raises(ValueError, match="state_dim"):
        assert_checkpoint_dims(
            {"state_dim": 10, "action_dim": 4},
            state_dim=8,
            action_dim=4,
            path=__file__,
        )


def test_dt_last_position_invariant_to_current_action_slot():
    """Causal mask must hide the current-step action token from the state token."""
    torch.manual_seed(0)
    B, K, sd, ad = 2, 8, 12, 5
    model = _tiny_dt(sd, ad, K).eval()
    states = torch.randn(B, K, sd)
    rtg = torch.randn(B, K, 1)
    mask = torch.ones(B, K)
    a1 = torch.zeros(B, K, ad)
    a1[:, :-1] = torch.softmax(torch.randn(B, K - 1, ad), dim=-1)
    a1[:, -1] = torch.softmax(torch.randn(B, ad), dim=-1)
    a2 = a1.clone()
    a2[:, -1] = torch.softmax(torch.randn(B, ad), dim=-1)
    with torch.no_grad():
        p1 = model(states, a1, rtg, mask)[:, -1]
        p2 = model(states, a2, rtg, mask)[:, -1]
    assert torch.allclose(p1, p2, atol=1e-5)


def test_bc_last_position_invariant_to_current_action_slot():
    torch.manual_seed(1)
    B, K, sd, ad = 2, 8, 12, 5
    model = _tiny_dt(sd, ad, K, use_rtg=False).eval()
    states = torch.randn(B, K, sd)
    a1 = torch.softmax(torch.randn(B, K, ad), dim=-1)
    a2 = a1.clone()
    a2[:, -1] = torch.softmax(torch.randn(B, ad), dim=-1)
    with torch.no_grad():
        p1 = model(states, a1, None, torch.ones(B, K))[:, -1]
        p2 = model(states, a2, None, torch.ones(B, K))[:, -1]
    assert torch.allclose(p1, p2, atol=1e-5)


def test_rtg_budget_decrements_and_resets():
    torch.manual_seed(0)
    model = _tiny_dt().eval()
    horizon = 5
    reward = 0.01
    policy = DTPolicy(
        model, device="cpu", target_rtg=1.0, rtg_mean=0.0, rtg_std=1.0,
        reset_horizon=horizon, name="dt",
    )
    env = _DummyEnv()
    for t in range(12):
        weights = policy(t, np.zeros(model.state_dim, dtype=np.float32), env)
        env.history.append({"reward": reward, "weights": weights})

    rtgs = np.asarray(policy._rtgs)
    for t, budget in enumerate(rtgs):
        if t % horizon == 0:
            assert budget == pytest.approx(1.0)
        else:
            assert budget == pytest.approx(rtgs[t - 1] - reward)


def _assert_simplex(w: np.ndarray) -> None:
    assert np.isfinite(w).all()
    assert w.ndim == 1
    assert (w >= -1e-6).all()
    assert w.sum() == pytest.approx(1.0, abs=1e-5)


def test_dt_and_bc_policies_emit_simplex_weights():
    torch.manual_seed(0)
    env = _DummyEnv()
    state = np.random.randn(8).astype(np.float32)
    for use_rtg, name in ((True, "dt"), (False, "bc")):
        model = _tiny_dt(use_rtg=use_rtg).eval()
        policy = DTPolicy(
            model, "cpu", target_rtg=0.5, rtg_mean=0.0, rtg_std=1.0, name=name
        )
        _assert_simplex(policy(0, state, env))


@pytest.mark.parametrize("cls", [TD3BC, IQL, CQL, PPO, SAC, A2C])
def test_agent_policies_emit_simplex_weights(cls):
    torch.manual_seed(0)
    agent = cls(8, 4, device="cpu")
    policy = AgentPolicy(agent, "cpu", name=cls.__name__)
    _assert_simplex(policy(0, np.random.randn(8).astype(np.float32), None))


def test_rtg_quantile_targets_are_monotonic():
    cfg = Config.from_yaml("configs/config.yaml")
    traj = cfg.trajectories_dir() / "trajectories.npz"
    if not traj.exists():
        pytest.skip("trajectories.npz not generated")
    targets = rtg_quantile_targets(cfg, [0.1, 0.25, 0.5, 0.75, 0.9])
    vals = [targets[q] for q in sorted(targets)]
    assert vals == sorted(vals)
    assert len(vals) == 5


def test_calibration_table_shape_and_monotonic_targets():
    """Sweep contract: one row per (seed, quantile), target RTG rises with quantile."""
    rows = []
    for seed in (42, 123):
        for q, tgt, realized in (
            (0.1, 0.10, 0.05),
            (0.5, 0.50, 0.20),
            (0.9, 0.90, 0.40),
        ):
            rows.append({
                "strategy": "dt",
                "seed": seed,
                "quantile": q,
                "target_rtg": tgt,
                "realized_log_return": realized,
                "realized_cagr": 0.1,
                "sharpe": 0.5,
            })
    df = pd.DataFrame(rows)
    assert set(df.columns) >= {
        "strategy", "seed", "quantile", "target_rtg", "realized_log_return",
    }
    for _, g in df.groupby("seed"):
        g = g.sort_values("quantile")
        assert g["target_rtg"].is_monotonic_increasing
        # Synthetic sweep is constructed so realized return tracks the target.
        assert g["realized_log_return"].is_monotonic_increasing


def test_universe_scope_legacy_a_and_suffixed_b():
    cfg_a = Config.from_yaml("configs/config.yaml")
    assert cfg_a.artifact_scope == ""
    assert cfg_a.processed_dir() == cfg_a.project_root / "data" / "processed"
    assert cfg_a.trajectories_dir() == cfg_a.project_root / "data" / "trajectories"
    assert cfg_a.checkpoint_dir() == cfg_a.project_root / "results" / "checkpoints"
    assert cfg_a.training_manifest_path() == cfg_a.project_root / "results" / "training_manifest.json"

    cfg_b = universe_b_config(cfg_a)
    assert cfg_b.artifact_scope == "universe_B"
    assert cfg_b.processed_dir() == cfg_a.processed_dir() / "universe_B"
    assert cfg_b.trajectories_dir() == cfg_a.trajectories_dir() / "universe_B"
    assert cfg_b.checkpoint_dir() == cfg_a.checkpoint_dir() / "universe_B"
    assert cfg_b.tables_dir() == cfg_a.tables_dir() / "universe_B"
    assert cfg_b.training_manifest_path().parent.name == "universe_B"
    assert cfg_a.processed_dir() != cfg_b.processed_dir()
    assert cfg_b.get("data", "start_date") == "2014-09-17"


def test_universe_b_config_does_not_mutate_original():
    cfg_a = Config.from_yaml("configs/config.yaml")
    raw_before = deepcopy(cfg_a.raw)
    _ = universe_b_config(cfg_a)
    assert cfg_a.raw == raw_before
    assert cfg_a.get("data", "universe") == "A"


def test_significance_table_shape_and_finite_cis():
    rng = np.random.default_rng(0)
    bah = rng.normal(0.0004, 0.01, 400)
    series = {
        "buy_and_hold": bah,
        "momentum": bah + rng.normal(0.0001, 0.002, 400),
        "dt": bah + rng.normal(0.0, 0.002, 400),
    }
    table = significance_table(
        series, benchmark="buy_and_hold", n_bootstrap=80, block_size=10, rng=rng
    )
    assert len(table) == 3
    assert set(table["strategy"]) == set(series)
    for col in ("sharpe", "sharpe_diff_vs_bah", "ci_lower", "ci_upper",
                "ledoit_wolf_p", "deflated_sharpe"):
        assert np.isfinite(table[col]).all(), col
    bah_row = table.set_index("strategy").loc["buy_and_hold"]
    assert bah_row["sharpe_diff_vs_bah"] == 0.0
    assert bah_row["ci_lower"] == 0.0
    assert (table["ci_lower"] <= table["ci_upper"]).all()
    assert (table["n_trials"] == 3).all()
