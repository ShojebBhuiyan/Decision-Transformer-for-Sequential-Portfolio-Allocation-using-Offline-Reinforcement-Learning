"""Behavior Cloning baselines."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.decision_transformer import DecisionTransformer


class TransformerBC(DecisionTransformer):
    """Behavior Cloning with transformer backbone (no RTG conditioning)."""

    def __init__(self, *args, **kwargs):
        kwargs["use_rtg"] = False
        super().__init__(*args, **kwargs)

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rtg: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, K, _ = states.shape
        # Only use state and action tokens (no RTG)
        s_emb = self.embed_state(states)
        a_emb = self.embed_action(actions)
        tokens = torch.stack([s_emb, a_emb], dim=2).reshape(B, 2 * K, self.d_model)
        pos = self.pos_embed[:, : 2 * K, :]
        x = self.drop(tokens + pos)

        if mask is not None:
            expanded_mask = mask.unsqueeze(-1).repeat(1, 1, 2).reshape(B, 2 * K)
        else:
            expanded_mask = None

        for block in self.blocks:
            x = block(x, expanded_mask)

        x = self.ln_f(x)
        state_tokens = x[:, 0::2, :]
        logits = self.action_head(state_tokens)
        return F.softmax(logits, dim=-1)


class MLPBC(nn.Module):
    """Simple MLP behavior cloning baseline."""

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, states: torch.Tensor, **kwargs) -> torch.Tensor:
        # states: (B, K, state_dim) or (B, state_dim)
        if states.dim() == 3:
            states = states[:, -1, :]
        logits = self.net(states)
        return F.softmax(logits, dim=-1)
