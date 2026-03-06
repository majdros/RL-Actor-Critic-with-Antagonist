import torch
import torch.nn as nn
from torch.distributions import Categorical


class A2C(nn.Module):
    def __init__(self, env, hidden_size=128, gamma=.99, random_seed=None):
        """
        Assumes fixed continuous observation space
        and fixed discrete action space (for now)
        """
        super().__init__()

        self.env = env
        self.gamma = gamma
        self.hidden_size = hidden_size

        if random_seed is not None:
            # Gymnasium-style seeding
            try:
                self.env.reset(seed=random_seed)
            except TypeError:
                # Older Gym fallback
                if hasattr(self.env, "seed"):
                    self.env.seed(random_seed)
            torch.manual_seed(random_seed)

        # Infer sizes
        self.in_size = len(env.observation_space.sample().flatten())
        self.out_size = self.env.action_space.n

        self.actor = nn.Sequential(
            nn.Linear(self.in_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.out_size),
        ).double()

        self.critic = nn.Sequential(
            nn.Linear(self.in_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        ).double()

    def _reset(self):
        out = self.env.reset()
        if isinstance(out, tuple):
            obs, _info = out
            return obs
        return out

    def _step(self, action):
        out = self.env.step(action)
        if len(out) == 5:
            obs, reward, terminated, truncated, info = out
            done = terminated or truncated
            return obs, reward, done, info
        obs, reward, done, info = out
        return obs, reward, done, info

    def train_env_episode(self, render=False):
        rewards = []
        critic_vals = []
        action_lp_vals = []

        observation = self._reset()
        done = False

        while not done:
            if render:
                self.env.render()

            observation_t = torch.from_numpy(observation).double()

            action_logits = self.actor(observation_t)
            dist = Categorical(logits=action_logits)
            action = dist.sample()

            # Correct log-prob of sampled action
            action_log_prob = dist.log_prob(action)

            pred = torch.squeeze(self.critic(observation_t).view(-1))

            action_lp_vals.append(action_log_prob)
            critic_vals.append(pred)

            observation, reward, done, info = self._step(action.item())
            rewards.append(torch.tensor(reward).double())

        total_reward = sum(rewards)

        # Expected returns
        for t_i in range(len(rewards)):
            G = 0
            for t in range(t_i, len(rewards)):
                G += rewards[t] * (self.gamma ** (t - t_i))
            rewards[t_i] = G

        def f(inp):
            return torch.stack(tuple(inp), 0)

        rewards = f(rewards)
        rewards = (rewards - torch.mean(rewards)) / (torch.std(rewards) + 1e-12)

        return rewards, f(critic_vals), f(action_lp_vals), total_reward

    def test_env_episode(self, render=True):
        observation = self._reset()
        rewards = []
        done = False

        while not done:
            if render:
                self.env.render()

            observation_t = torch.from_numpy(observation).double()

            action_logits = self.actor(observation_t)
            action = Categorical(logits=action_logits).sample()

            observation, reward, done, info = self._step(action.item())
            rewards.append(reward)

        return sum(rewards)

    @staticmethod
    def compute_loss(action_p_vals, G, V, critic_loss=nn.SmoothL1Loss()):
        assert len(action_p_vals) == len(G) == len(V)
        advantage = G - V.detach()
        return -(torch.sum(action_p_vals * advantage)), critic_loss(G, V)