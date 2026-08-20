"""Regression tests for training-loop and RL-agent correctness fixes."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.models.offline_rl import CQL, IQL, TD3BC
from src.models.online_rl import A2C, PPO, SAC


def _fake_batch(B: int = 16, state_dim: int = 24, action_dim: int = 5) -> dict:
    """Batch shaped like GPUTrajectoryBuffer(last_state_only=True) output."""
    actions = torch.softmax(torch.randn(B, action_dim), dim=-1)
    return {
        "states": torch.randn(B, 1, state_dim),
        "next_states": torch.randn(B, 1, state_dim),
        "dones": torch.zeros(B, 1),
        "actions": actions.unsqueeze(1),
        "rewards": torch.randn(B, 1) * 1e-3,
        "rtg": torch.zeros(B, 1, 1),
        "mask": torch.ones(B, 1),
        "target_action": actions,
    }


AGENTS = [("td3bc", TD3BC), ("iql", IQL), ("cql", CQL),
          ("ppo", PPO), ("sac", SAC), ("a2c", A2C)]


@pytest.mark.parametrize("name,cls", AGENTS)
def test_agent_train_step_runs_and_is_finite(name, cls):
    """Every agent must train on the buffer's batch format without NaN."""
    torch.manual_seed(0)
    agent = cls(24, 5, device="cpu")
    for _ in range(5):
        metrics = agent.train_step(_fake_batch())
    for key, value in metrics.items():
        assert np.isfinite(value), f"{name}.{key} not finite: {value}"


def _first_step_loss(cls, batch: dict) -> float:
    """Loss of a single step taken by a freshly initialized agent."""
    torch.manual_seed(0)
    agent = cls(24, 5, device="cpu")
    torch.manual_seed(1)
    return agent.train_step(dict(batch))["critic_loss"]


@pytest.mark.parametrize("name,cls", [("td3bc", TD3BC), ("iql", IQL), ("cql", CQL)])
def test_offline_agents_use_next_states(name, cls):
    """Bellman target must depend on next_states, not just the current state."""
    torch.manual_seed(0)
    batch = _fake_batch()
    perturbed = dict(batch)
    perturbed["next_states"] = batch["next_states"] + 25.0

    assert _first_step_loss(cls, batch) != _first_step_loss(cls, perturbed), (
        f"{name} critic loss ignores next_states"
    )


@pytest.mark.parametrize("name,cls", [("td3bc", TD3BC), ("iql", IQL), ("cql", CQL)])
def test_target_critic_tracks_online_critic(name, cls):
    """Target networks must be soft-updated, not frozen at initialization."""
    agent = cls(24, 5, device="cpu")
    before = [p.clone() for p in agent.critic_target.parameters()]
    for _ in range(3):
        agent.train_step(_fake_batch())
    after = list(agent.critic_target.parameters())
    assert any(not torch.equal(b, a) for b, a in zip(before, after)), (
        f"{name} critic_target never updated"
    )


def test_terminal_transition_drops_bootstrap():
    """dones=1 must remove the next-state bootstrap, so next_states stop mattering."""
    torch.manual_seed(0)
    batch = _fake_batch()
    batch["dones"] = torch.ones_like(batch["dones"])
    far = dict(batch)
    far["next_states"] = batch["next_states"] + 100.0

    assert _first_step_loss(TD3BC, batch) == pytest.approx(
        _first_step_loss(TD3BC, far), rel=1e-6
    )


def test_cql_penalty_independent_of_batch_size():
    """
    CQL's log-sum-exp must run over sampled actions, not the batch.

    A batch-dimension reduction makes the penalty grow like log(batch_size).
    """
    torch.manual_seed(0)
    agent = CQL(24, 5, device="cpu")
    small = agent.train_step(_fake_batch(B=8))["critic_loss"]
    torch.manual_seed(0)
    agent = CQL(24, 5, device="cpu")
    large = agent.train_step(_fake_batch(B=256))["critic_loss"]
    assert abs(small - large) < 1.0, (
        f"penalty scales with batch size: {small:.3f} vs {large:.3f}"
    )
