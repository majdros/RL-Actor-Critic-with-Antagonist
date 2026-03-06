"""
Cart-pole reinforcement learning environment:
Agent learns to balance a pole on a cart

a2c: Agent uses Advantage Actor Critic algorithm
"""
import os
import sys

# Ensure project root is on sys.path so "import src..." works
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    import gymnasium as gym
except ImportError:
    import gym

from src.a2c import A2C
import torch.optim as optim
import torch
from typing import cast


LR = .01  # Learning rate
SEED = None  # Random seed for reproducibility
MAX_EPISODES = 10000  # Max number of episodes

ENV_ID = "CartPole-v1"

# Rendering
RENDER_TRAINING = True
RENDER_TRAIN_EVERY_N_EPISODES = 1
RENDER_TEST = True
TEST_EPISODES = 100

# Train env (no rendering)
train_env = gym.make(ENV_ID, render_mode="human") if RENDER_TRAINING else gym.make(ENV_ID)
agent = A2C(train_env, random_seed=SEED)

# Init optimizers
actor_optim = optim.Adam(agent.actor.parameters(), lr=LR)
critic_optim = optim.Adam(agent.critic.parameters(), lr=LR)

r = []
avg_r = 0.0

for i in range(MAX_EPISODES):
    critic_optim.zero_grad()
    actor_optim.zero_grad()

    render_this_episode = RENDER_TRAINING and (i % RENDER_TRAIN_EVERY_N_EPISODES == 0)
    rewards, critic_vals, action_lp_vals, total_reward = agent.train_env_episode(render=render_this_episode)
    r.append(float(total_reward))

    l_actor, l_critic = agent.compute_loss(action_p_vals=action_lp_vals, G=rewards, V=critic_vals)
    cast(torch.Tensor, l_actor).backward()
    cast(torch.Tensor, l_critic).backward()

    actor_optim.step()
    critic_optim.step()

    if len(r) >= 100:
        episode_count = i - (i % 100)
        prev_episodes = r[len(r) - 100:]
        avg_r = sum(prev_episodes) / len(prev_episodes)
        if len(r) % 100 == 0:
            print(f"Average reward during episodes {episode_count}-{episode_count+100} is {avg_r}")
        if avg_r > 195:
            print(f"Solved CartPole-v1 with average reward {avg_r}")
            break

# Test env (rendering)
if RENDER_TEST:
    render_env = train_env if RENDER_TRAINING else gym.make(ENV_ID, render_mode="human")
    agent.env = render_env
    for _ in range(TEST_EPISODES):
        agent.test_env_episode(render=True)
    if not RENDER_TRAINING:
        render_env.close()

train_env.close()