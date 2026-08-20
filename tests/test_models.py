"""Tests for Decision Transformer model."""

import pytest

torch = pytest.importorskip("torch")

from src.models.decision_transformer import DecisionTransformer


def test_dt_forward_shape():
    B, K, state_dim, action_dim = 4, 10, 50, 14
    model = DecisionTransformer(state_dim, action_dim, context_length=K)
    states = torch.randn(B, K, state_dim)
    actions = torch.softmax(torch.randn(B, K, action_dim), dim=-1)
    rtg = torch.randn(B, K, 1)
    mask = torch.ones(B, K)

    pred = model(states, actions, rtg, mask)
    assert pred.shape == (B, K, action_dim)
    assert torch.allclose(pred.sum(dim=-1), torch.ones(B, K), atol=1e-5)


def test_dt_get_action():
    B, K, state_dim, action_dim = 2, 5, 30, 14
    model = DecisionTransformer(state_dim, action_dim, context_length=K)
    states = torch.randn(B, K, state_dim)
    actions = torch.softmax(torch.randn(B, K, action_dim), dim=-1)
    rtg = torch.randn(B, K, 1)

    action = model.get_action(states, actions, rtg)
    assert action.shape == (B, action_dim)
    assert torch.allclose(action.sum(dim=-1), torch.ones(B), atol=1e-5)
