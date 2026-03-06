"""
Cart-pole reinforcement learning environment:
Agent learns to balance a pole on a cart

Baseline: Agent selects moves at random
"""

try:
    import gymnasium as gym
except ImportError:
    import gym

ENV_ID = "CartPole-v1"
EPISODES = 100
MAX_STEPS = 500

env = gym.make(ENV_ID, render_mode="human")

t_steps = []

for i_episode in range(EPISODES):
    reset_out = env.reset()
    observation = reset_out[0] if isinstance(reset_out, tuple) else reset_out

    for t in range(MAX_STEPS):
        action = env.action_space.sample()

        step_out = env.step(action)
        if len(step_out) == 5:
            observation, reward, terminated, truncated, info = step_out
            done = terminated or truncated
        else:
            observation, reward, done, info = step_out

        if done:
            t_steps.append(t + 1)
            break

print(f"Episodes: {len(t_steps)}")
print(f"Mean steps: {sum(t_steps)/len(t_steps):.1f}" if t_steps else "No completed episodes?")
env.close()