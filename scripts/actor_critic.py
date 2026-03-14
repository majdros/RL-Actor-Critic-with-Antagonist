# Source: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/sac_continuous_action.py
import torch
import torch.nn as nn
from torch.distributions import Normal


# Actor gibt Normal-Verteilung aus
class Actor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        self.mu_head = nn.Linear(hidden_dim, act_dim)
        # globale log_std-Parameter
        self.log_std = nn.Parameter(torch.zeros(act_dim))


    def forward(self, obs: torch.Tensor):
        x = self.net(obs)
        mu = self.mu_head(x)
        std = torch.exp(self.log_std).expand_as(mu)
        return mu, std


    def get_dist(self, obs: torch.Tensor):
        mu, std = self.forward(obs)
        return Normal(mu, std)


    def choose_action(self, state):
        mu, std = self.forward(state)
        dist = torch.distributions.Normal(mu, std)
        # raw action aus Gaussian
        raw_action = dist.rsample()
        # squash mit tanh
        squashed_action = torch.tanh(raw_action)
        # log_prob berechnen
        log_prob = dist.log_prob(raw_action)
        # Korrektur term (tanh Transformation)
        log_prob -= torch.log(1 - squashed_action.pow(2) + 1e-6)
        log_prob = log_prob.sum()
        entropy = dist.entropy().sum()

        return squashed_action, log_prob, entropy



# Critic approximiert V(s)
class Critic(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int = 128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor):
        return self.net(obs).squeeze(-1)