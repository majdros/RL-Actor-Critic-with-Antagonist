"""
Lunar-lander reinforcement learning environment:
Agent learns to land spacecraft

Baseline: Agent selects moves at random

"""

try:
    import gymnasium as gym
except ImportError:
    import gym

from typing import Any, cast

ENV_ID = "LunarLander-v3"
env = gym.make(ENV_ID, render_mode="human")


t_steps = []
for i_episode in range(1000):

    reset_out = env.reset()  # Get initial observation
    observation = reset_out[0] if isinstance(reset_out, tuple) else reset_out

    for t in range(100):

        env.render()

        action = env.action_space.sample()  # Get a random action

        step_out = cast(tuple[Any, ...], env.step(action))  # Get next step of the game
        if len(step_out) == 5:
            observation, reward, terminated, truncated, info = step_out
            done = terminated or truncated
        else:
            observation, reward, done, info = step_out
        print(reward)
        if done:
            t_steps.append(t + 1)
            break
    break
for t in t_steps:
    print(t)
env.close()
