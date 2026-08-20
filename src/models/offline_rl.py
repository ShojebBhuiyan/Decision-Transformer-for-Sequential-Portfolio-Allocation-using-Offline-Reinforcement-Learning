"""Offline RL algorithms: TD3+BC, IQL, CQL."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimplexActor(nn.Module):
    """Actor with softmax output on simplex."""

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if state.dim() == 3:
            state = state[:, -1, :]
        return F.softmax(self.net(state), dim=-1)


class TwinCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        input_dim = state_dim + action_dim
        self.q1 = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if state.dim() == 3:
            state = state[:, -1, :]
        x = torch.cat([state, action], dim=-1)
        return self.q1(x), self.q2(x)

    def q1_forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if state.dim() == 3:
            state = state[:, -1, :]
        return self.q1(torch.cat([state, action], dim=-1))


class TD3BC:
    """TD3 + Behavior Cloning regularization."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        alpha: float = 2.5,
        lr: float = 3e-4,
        gamma: float = 0.99,
        device: str = "cpu",
    ):
        self.device = device
        self.alpha = alpha
        self.gamma = gamma
        self.actor = SimplexActor(state_dim, action_dim).to(device)
        self.critic = TwinCritic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

    def train_step(self, batch: dict) -> dict:
        states = batch["states"].to(self.device)
        actions = batch["target_action"].to(self.device)
        rewards = batch["rewards"][:, -1].to(self.device)
        next_states = states  # simplified: same episode context

        # Critic update
        with torch.no_grad():
            next_action = self.actor(next_states)
            q1_t, q2_t = self.critic_target(next_states, next_action)
            target_q = rewards.unsqueeze(-1) + self.gamma * torch.min(q1_t, q2_t)

        q1, q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # Actor update: maximize Q + BC regularization
        pi_action = self.actor(states)
        q_pi = self.critic.q1_forward(states, pi_action)
        bc_loss = F.mse_loss(pi_action, actions)
        actor_loss = -self.alpha * q_pi.mean() + bc_loss
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        return {"critic_loss": critic_loss.item(), "actor_loss": actor_loss.item()}

    def get_action(self, states: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.actor(states)


class IQL:
    """Implicit Q-Learning."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        expectile: float = 0.7,
        temperature: float = 3.0,
        lr: float = 3e-4,
        gamma: float = 0.99,
        device: str = "cpu",
    ):
        self.device = device
        self.expectile = expectile
        self.temperature = temperature
        self.gamma = gamma
        self.actor = SimplexActor(state_dim, action_dim).to(device)
        self.critic = TwinCritic(state_dim, action_dim).to(device)
        self.value = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1),
        ).to(device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.value_opt = torch.optim.Adam(self.value.parameters(), lr=lr)

    def _expectile_loss(self, diff: torch.Tensor) -> torch.Tensor:
        weight = torch.where(diff > 0, self.expectile, 1 - self.expectile)
        return (weight * diff.pow(2)).mean()

    def train_step(self, batch: dict) -> dict:
        states = batch["states"].to(self.device)
        actions = batch["target_action"].to(self.device)
        rewards = batch["rewards"][:, -1].to(self.device)
        s = states[:, -1, :]

        # Value update
        with torch.no_grad():
            q1, q2 = self.critic(states, actions)
            q = torch.min(q1, q2)
        v = self.value(s)
        value_loss = self._expectile_loss(q - v)
        self.value_opt.zero_grad()
        value_loss.backward()
        self.value_opt.step()

        # Critic update
        with torch.no_grad():
            next_v = self.value(s)
            target_q = rewards.unsqueeze(-1) + self.gamma * next_v
        q1, q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # Actor update: advantage-weighted BC
        with torch.no_grad():
            adv = q - self.value(s)
            weights = torch.exp(adv / self.temperature).clamp(max=100.0)
        pi_action = self.actor(states)
        actor_loss = (weights * F.mse_loss(pi_action, actions, reduction="none").mean(-1, keepdim=True)).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        return {"value_loss": value_loss.item(), "critic_loss": critic_loss.item(), "actor_loss": actor_loss.item()}

    def get_action(self, states: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.actor(states)


class CQL:
    """Conservative Q-Learning."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        cql_alpha: float = 1.0,
        lr: float = 3e-4,
        gamma: float = 0.99,
        device: str = "cpu",
    ):
        self.device = device
        self.cql_alpha = cql_alpha
        self.gamma = gamma
        self.actor = SimplexActor(state_dim, action_dim).to(device)
        self.critic = TwinCritic(state_dim, action_dim).to(device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

    def train_step(self, batch: dict) -> dict:
        states = batch["states"].to(self.device)
        actions = batch["target_action"].to(self.device)
        rewards = batch["rewards"][:, -1].to(self.device)

        # Critic with CQL penalty
        q1, q2 = self.critic(states, actions)
        with torch.no_grad():
            next_action = self.actor(states)
            q1_next, q2_next = self.critic(states, next_action)
            target_q = rewards.unsqueeze(-1) + self.gamma * torch.min(q1_next, q2_next)

        td_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        # CQL conservative term: log-sum-exp over random actions
        random_actions = F.softmax(torch.randn(actions.shape[0], actions.shape[1], device=self.device), dim=-1)
        q1_rand, q2_rand = self.critic(states, random_actions)
        cql_penalty = (torch.logsumexp(q1_rand, dim=0).mean() - q1.mean()) + \
                      (torch.logsumexp(q2_rand, dim=0).mean() - q2.mean())
        critic_loss = td_loss + self.cql_alpha * cql_penalty

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # Actor: maximize Q
        pi_action = self.actor(states)
        q_pi = self.critic.q1_forward(states, pi_action)
        actor_loss = -q_pi.mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        return {"critic_loss": critic_loss.item(), "actor_loss": actor_loss.item()}

    def get_action(self, states: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.actor(states)
