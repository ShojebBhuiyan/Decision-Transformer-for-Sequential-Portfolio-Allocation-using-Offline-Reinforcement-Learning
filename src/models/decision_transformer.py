"""Decision Transformer architecture."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        dropout_p = self.dropout.p if self.training else 0.0

        if mask is None:
            # Fused causal kernel; no mask tensor to materialize
            out = F.scaled_dot_product_attention(
                q, k, v, dropout_p=dropout_p, is_causal=True
            )
        else:
            causal = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
            allowed = causal.view(1, 1, T, T) & mask.bool().view(B, 1, 1, T)
            # Guarantee diagonal is attendable so no row is fully masked (-> NaN)
            eye = torch.eye(T, device=x.device, dtype=torch.bool).view(1, 1, T, T)
            allowed = allowed | eye
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=allowed, dropout_p=dropout_p
            )

        out = out.transpose(1, 2).reshape(B, T, C)
        return self.proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.mlp(self.ln2(x))
        return x


class DecisionTransformer(nn.Module):
    """
    Causal GPT-style Decision Transformer.

    Input tokens per timestep: (RTG, state, action) interleaved.
    Predicts action at each state token position.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        context_length: int = 30,
        d_model: int = 192,
        n_layers: int = 4,
        n_heads: int = 6,
        dropout: float = 0.1,
        use_rtg: bool = True,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.context_length = context_length
        self.d_model = d_model
        self.use_rtg = use_rtg

        self.embed_rtg = nn.Linear(1, d_model)
        self.embed_state = nn.Linear(state_dim, d_model)
        self.embed_action = nn.Linear(action_dim, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, 3 * context_length, d_model))
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dropout) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.action_head = nn.Linear(d_model, action_dim)

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rtg: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            states: (B, K, state_dim)
            actions: (B, K, action_dim) — previous actions (shifted)
            rtg: (B, K, 1)
            mask: (B, K) padding mask
        Returns:
            predicted_actions: (B, K, action_dim) softmax weights
        """
        B, K, _ = states.shape

        rtg_emb = self.embed_rtg(rtg)
        s_emb = self.embed_state(states)
        a_emb = self.embed_action(actions)

        # Interleave: (RTG, S, A) per timestep
        tokens = torch.stack([rtg_emb, s_emb, a_emb], dim=2)  # (B, K, 3, d_model)
        tokens = tokens.reshape(B, 3 * K, self.d_model)

        pos = self.pos_embed[:, : 3 * K, :]
        x = self.drop(tokens + pos)

        # Expand mask for 3 tokens per step. A fully-valid mask is equivalent to
        # no mask, so drop it to reach the fused causal attention kernel.
        if mask is not None and not bool(mask.all()):
            expanded_mask = mask.unsqueeze(-1).repeat(1, 1, 3).reshape(B, 3 * K)
        else:
            expanded_mask = None

        for block in self.blocks:
            x = block(x, expanded_mask)

        x = self.ln_f(x)
        # Extract state token positions (index 1, 4, 7, ...)
        state_tokens = x[:, 1::3, :]  # (B, K, d_model)
        logits = self.action_head(state_tokens)
        return F.softmax(logits, dim=-1)

    def get_action(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rtg: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return action for the last timestep only."""
        pred = self.forward(states, actions, rtg, mask)
        return pred[:, -1, :]
