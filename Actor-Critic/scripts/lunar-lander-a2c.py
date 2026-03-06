"""
Lunar-lander reinforcement learning environment:
Agent learns to land spacecraft

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
import math
from typing import Any

LR = .001
SEED = None
MAX_EPISODES = 10000

ENV_ID = "LunarLander-v3"

# Rendering
RENDER_TRAINING = False
RENDER_TRAIN_EVERY_N_EPISODES = 1
RENDER_TEST = True
TEST_EPISODES = 10

train_env = gym.make(ENV_ID, render_mode="human") if RENDER_TRAINING else gym.make(ENV_ID)
agent = A2C(train_env, random_seed=SEED, gamma=.999)

actor_optim = optim.Adam(agent.actor.parameters(), lr=LR)
critic_optim = optim.Adam(agent.critic.parameters(), lr=LR)

r = []
avg_r = None
max_r = -math.inf

for i in range(MAX_EPISODES):
    critic_optim.zero_grad()
    actor_optim.zero_grad()

    render_this_episode = RENDER_TRAINING and (i % RENDER_TRAIN_EVERY_N_EPISODES == 0)
    rewards, critic_vals, action_lp_vals, total_reward = agent.train_env_episode(render=render_this_episode)
    total_reward_item = getattr(total_reward, "item", None)
    total_reward_raw: Any = total_reward_item() if callable(total_reward_item) else total_reward
    total_reward_value = float(total_reward_raw)
    r.append(total_reward_value)

    if total_reward_value >= 200:
        print("solved")
        break

    if len(r) >= 100:
        episode_count = i - (i % 100)
        prev_episodes = r[len(r) - 100:]
        avg_r = sum(prev_episodes) / len(prev_episodes)
        if len(r) % 100 == 0:
            print(f"Average reward during episodes {episode_count}-{episode_count + 100} is {avg_r}")

    l_actor, l_critic = agent.compute_loss(action_p_vals=action_lp_vals, G=rewards, V=critic_vals)
    l_actor.backward()
    l_critic.backward()

    actor_optim.step()
    critic_optim.step()

if RENDER_TEST:
    render_env = train_env if RENDER_TRAINING else gym.make(ENV_ID, render_mode="human")
    agent.env = render_env
    for _ in range(TEST_EPISODES):
        agent.test_env_episode(render=True)
    if not RENDER_TRAINING:
        render_env.close()

train_env.close()