import os
import numpy as np
import torch

from finger_env import FingerEllipseEnv, EnvConfig
from actor_critic import Actor, Critic

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

MODE = "render"   # "render" oder "eval"

def load_model(checkpoint_path):

    checkpoint = torch.load(checkpoint_path, map_location=device)

    cfg_dict = checkpoint["config"]
    cfg = EnvConfig(**cfg_dict)

    env = FingerEllipseEnv(cfg=cfg)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    actor = Actor(obs_dim, act_dim).to(device)
    critic = Critic(obs_dim).to(device)

    actor.load_state_dict(checkpoint["actor_state_dict"])
    critic.load_state_dict(checkpoint["critic_state_dict"])

    actor.eval()

    return actor, cfg


@torch.no_grad()
def select_action(actor, obs, env):

    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)

    mu, _ = actor(obs_tensor)
    action = torch.tanh(mu)

    action_np = (env.cfg.max_delta * action).cpu().numpy()

    return action_np


@torch.no_grad()
def run_episode(actor, env):

    obs, _ = env.reset()

    done = False
    episode_return = 0

    last_info = {}

    while not done:

        action = select_action(actor, obs, env)

        obs, reward, terminated, truncated, info = env.step(action)

        episode_return += reward
        done = terminated or truncated

        last_info = info

    metrics = {
        "return": episode_return,
        "area": last_info.get("area_det_final", 0),
        "closure": last_info.get("closure_dist2", 0),
        "axis_ratio": last_info.get("axis_ratio_b_over_a_final", 0),
    }

    return metrics


def evaluate(actor, cfg, adv_noise_scale, num_episodes=20):

    eval_cfg = EnvConfig(**cfg.__dict__)
    eval_cfg.adv_noise_scale = adv_noise_scale
    # eval_cfg.horizon = 300

    env = FingerEllipseEnv(cfg=eval_cfg)

    results = []

    for i in range(num_episodes):

        metrics = run_episode(actor, env)

        results.append(metrics)

    # Mittelwerte berechnen
    summary = {}

    for key in results[0].keys():

        values = [m[key] for m in results]

        summary[key] = np.mean(values)

    return summary


@torch.no_grad()
def render_trained_episode(checkpoint_path, adv_noise_scale=1.0, seed=0):
    """
    Lädt ein trainiertes Modell und spielt genau eine Episode
    sichtbar mit render_mode='human' ab.
    """
    print(f"Model: {checkpoint_path}")

    actor, cfg = load_model(checkpoint_path)
    eval_cfg = EnvConfig(**cfg.__dict__)        # config werden aus checkpoints geladen!
    eval_cfg.adv_noise_scale = adv_noise_scale
    eval_cfg.horizon = 128

    env = FingerEllipseEnv(cfg=eval_cfg, render_mode="human")

    obs, _ = env.reset(seed=seed)
    done = False
    episode_return = 0.0
    last_info = {}

    while not done:
        action = select_action(actor, obs, env)
        obs, reward, terminated, truncated, info = env.step(action)

        episode_return += reward
        done = terminated or truncated
        last_info = info

    print("\nEpisode fertig.")
    print(f"Return:        {episode_return:.4f}")
    print(f"Final area:    {last_info.get('area_det_final', 0.0):.6f}")
    print(f"Closure dist2: {last_info.get('closure_dist2', 0.0):.6f}")
    print(f"Axis ratio:    {last_info.get('axis_ratio_b_over_a_final', 0.0):.6f}")
    print(f"Penalty:       {last_info.get('terminal_penalty', 0.0):.6f}")



def main():
    if MODE == "render":
        render_trained_episode(
            # checkpoi
            checkpoint_path="checkpoints/100-episoden/best_by_area.pt",
            # checkpoint_path="checkpoints/100-episoden/best_by_return.pt",
            # checkpoint_path="checkpoints/100-episoden/last_model.pt",
            adv_noise_scale=0.0,
            seed=0,
        )


    else:
        checkpoints = [
            "checkpoints/best_by_return.pt",
            "checkpoints/best_by_area.pt"
        ]

        noise_levels = [0.0, 0.05, 0.1, 0.2]

        for ckpt in checkpoints:

            print("=" * 20)
            print("MODEL:", ckpt)
            print("=" * 20)

            actor, cfg = load_model(ckpt)

            for noise in noise_levels:

                summary = evaluate(actor, cfg, noise)

                print(
                    f"noise={noise:.2f} | "
                    f"return={summary['return']:.3f} | "
                    f"area={summary['area']:.3f} | "
                    f"closure={summary['closure']:.5f} | "
                    f"axis_ratio={summary['axis_ratio']:.3f}"
                )


if __name__ == "__main__":
    main()