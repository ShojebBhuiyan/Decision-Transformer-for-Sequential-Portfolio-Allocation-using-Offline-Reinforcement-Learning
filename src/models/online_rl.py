"""
Online RL baselines: PPO, SAC, A2C.

CAVEAT: These algorithms interact with the environment during training,
unlike the offline methods (DT, BC, CQL, IQL, TD3+BC). They serve as
reference upper bounds but are not directly comparable in a fair offline setting.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Dirichlet

from src.models.offline_rl import SimplexActor


class DirichletActor(nn.Module):
    """Policy parameterizing Dirichlet distribution over simplex."""

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, state: torch.Tensor) -> Dirichlet:
        if state.dim() == 3:
            state = state[:, -1, :]
        alpha = F.softplus(self.net(state)) + 1.0
        alpha = torch.nan_to_num(alpha, nan=1.0).clamp(min=1e-3)
        return Dirichlet(alpha)

    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        dist = self.forward(state)
        if deterministic:
            return dist.mean
        return dist.sample()


class ValueNetwork(nn.Module):
    def __init__(self, state_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if state.dim() == 3:
            state = state[:, -1, :]
        return self.net(state)


class PPO:
    """Proximal Policy Optimization."""

    def __init__(self, state_dim: int, action_dim: int, lr: float = 3e-4, clip_eps: float = 0.2, device: str = "cpu"):
        self.device = device
        self.clip_eps = clip_eps
        self.actor = DirichletActor(state_dim, action_dim).to(device)
        self.critic = ValueNetwork(state_dim).to(device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

    def train_step(self, batch: dict) -> dict:
        states = batch["states"].to(self.device)
        actions = batch["target_action"].to(self.device)
        rewards = batch["rewards"][:, -1].to(self.device)
        s = torch.nan_to_num(states[:, -1, :], nan=0.0)

        # Simplified PPO step (single-step for offline-compatible interface)
        dist = self.actor(s)
        log_prob = dist.log_prob(actions / (actions.sum(dim=-1, keepdim=True) + 1e-8))
        value = self.critic(s).squeeze(-1)
        advantage = rewards - value.detach()

        ratio = torch.exp(log_prob.sum(-1) - log_prob.sum(-1).detach())
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantage
        actor_loss = -torch.min(surr1, surr2).mean()

        critic_loss = F.mse_loss(value, rewards)
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        return {"actor_loss": actor_loss.item(), "critic_loss": critic_loss.item()}

    def get_action(self, states: torch.Tensor) -> torch.Tensor:
        return self.actor.get_action(states, deterministic=True)


class SAC:
    """Soft Actor-Critic (discrete/simplex variant)."""

    def __init__(self, state_dim: int, action_dim: int, lr: float = 3e-4, alpha: float = 0.2, device: str = "cpu"):
        self.device = device
        self.alpha = alpha
        self.actor = DirichletActor(state_dim, action_dim).to(device)
        self.critic = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 1),
        ).to(device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

    def train_step(self, batch: dict) -> dict:
        states = batch["states"].to(self.device)
        actions = batch["target_action"].to(self.device)
        rewards = batch["rewards"][:, -1].to(self.device)
        s = torch.nan_to_num(states[:, -1, :], nan=0.0)

        q = self.critic(torch.cat([s, actions], -1)).squeeze(-1)
        critic_loss = F.mse_loss(q, rewards)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        new_actions = self.actor.get_action(s)
        q_pi = self.critic(torch.cat([s, new_actions], -1)).squeeze(-1)
        actor_loss = -q_pi.mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        return {"critic_loss": critic_loss.item(), "actor_loss": actor_loss.item()}

    def get_action(self, states: torch.Tensor) -> torch.Tensor:
        return self.actor.get_action(states, deterministic=True)


class A2C:
    """Advantage Actor-Critic."""

    def __init__(self, state_dim: int, action_dim: int, lr: float = 3e-4, device: str = "cpu"):
        self.device = device
        self.actor = SimplexActor(state_dim, action_dim).to(device)
        self.critic = ValueNetwork(state_dim).to(device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

    def train_step(self, batch: dict) -> dict:
        states = batch["states"].to(self.device)
        actions = batch["target_action"].to(self.device)
        rewards = batch["rewards"][:, -1].to(self.device)
        s = torch.nan_to_num(states[:, -1, :], nan=0.0)

        value = self.critic(s).squeeze(-1)
        advantage = rewards - value.detach()
        pi_action = self.actor(states)
        log_pi = torch.log(pi_action + 1e-8)
        actor_loss = -(advantage * (actions * log_pi).sum(-1)).mean()
        critic_loss = F.mse_loss(value, rewards)

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        return {"actor_loss": actor_loss.item(), "critic_loss": critic_loss.item()}

    def get_action(self, states: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.actor(states)
