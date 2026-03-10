import os
from dataclasses import asdict

import torch
import torch.nn.functional as F
import torch.optim as optim

from finger_env import FingerEllipseEnv, EnvConfig
from actor_critic import Actor, Critic
from rollout import collect_rollout


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

EPISODEN = 100
SAVE_DIR = os.path.join("checkpoints", f"{EPISODEN}-episoden")

# Monte-Carlo Returns berechnen
def compute_returns(rewards: torch.Tensor, gamma: float) -> torch.Tensor:
    """
    Berechnet diskontierte Returns:
        G_t = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + ...

    rewards: Tensor der Form (T,)
    return:  Tensor der Form (T,)
    """
    returns = torch.zeros_like(rewards, device=rewards.device)
    G_t = 0.0

    # Rückwärts durch die Episode gehen
    for t in reversed(range(len(rewards))):
        G_t = rewards[t] + gamma * G_t      # G_t=rt+γG_t+1
        returns[t] = G_t

    return returns


# Ein einzelner Trainingsschritt
def train_one_episode(env, actor, critic, optimizer, horizon, gamma, entropy_coef, value_coef):
    """
    Führt einen Rollout aus und macht anschließend ein Update
    für Actor und Critic.

    Rückgabe:
        metrics: dict mit Losses und Diagnosewerten
    """

    # Rollout sammeln
    rollout = collect_rollout(
        actor=actor,
        critic=critic,
        horizon=horizon,
        env=env,
        device=device,
    )

    rewards = rollout["rewards"]         
    log_probs = rollout["log_probs"]     
    values = rollout["values"]           
    entropies = rollout["entropies"]     
    last_info = rollout["last_info"]

    # Diskontierte Returns berechnen
    returns = compute_returns(rewards, gamma=gamma)

    # Advantage: Wie viel besser/schlechter ist das Ergebnis verglichen mit V(s)
    advantages = (returns - values).detach()

    # Advantage-Normalisierung
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Actor-Loss:
    # Wenn Advantage positiv ist, soll log_prob größer werden.
    # Wenn Advantage negativ ist, soll log_prob kleiner werden.
    actor_loss = -(log_probs * advantages.detach()).mean()

    # Critic-Loss: Critic soll Returns möglichst gut approximieren
    critic_loss = F.mse_loss(values, returns.detach())

    # Entropy-Bonus: Fördert Exploration, verhindert frühes Kollabieren der Policy
    entropy_bonus = entropies.mean()

    total_loss = actor_loss + value_coef * critic_loss - entropy_coef * entropy_bonus

    # Backpropagatoion
    optimizer.zero_grad()
    total_loss.backward()

    # Gradienten clipping für stabileres Training
    torch.nn.utils.clip_grad_norm_(
        list(actor.parameters()) + list(critic.parameters()),
        max_norm=0.5
    )

    optimizer.step()

    # Logging / Diagnose
    metrics = {
        "episode_len": len(rewards),
        "episode_return": rewards.sum().item(),
        "actor_loss": actor_loss.item(),
        "critic_loss": critic_loss.item(),
        "entropy": entropy_bonus.item(),
        "mean_return": returns.mean().item(),
        "mean_value": values.mean().item(),
        "area_det_final": float(last_info.get("area_det_final", 0.0)),
        "closure_dist2": float(last_info.get("closure_dist2", 0.0)),
        "axis_ratio_b_over_a_final": float(last_info.get("axis_ratio_b_over_a_final", 0.0)),
        "terminal_penalty": float(last_info.get("terminal_penalty", 0.0)),
    }
    return metrics


def save_checkpoint(path, actor, critic, cfg, episode, metrics):
    torch.save(
        {
            "actor_state_dict": actor.state_dict(),
            "critic_state_dict": critic.state_dict(),
            "config": asdict(cfg),
            "episode": episode,
            "metrics": metrics,
        },
        path,
    )


# Hauptfunktion
def main():
    # Hyperparameter
    cfg = EnvConfig(
        horizon=128,
        adv_noise_scale=0.0,
        w_action=0.1,
    )

    num_episodes = EPISODEN
    gamma = 0.99
    lr = 3e-4
    hidden_dim = 128
    entropy_coef = 1e-3
    value_coef = 0.5

    os.makedirs(SAVE_DIR, exist_ok=True)

    # Environment
    env = FingerEllipseEnv(cfg=cfg, render_mode=None)

    obs_dim = env.observation_space.shape[0]    # 9
    act_dim = env.action_space.shape[0]         # 3

    # Modelle
    actor = Actor(obs_dim=obs_dim, act_dim=act_dim, hidden_dim=hidden_dim).to(device)
    critic = Critic(obs_dim=obs_dim, hidden_dim=hidden_dim).to(device)

    optimizer = optim.Adam(
        list(actor.parameters()) + list(critic.parameters()),
        lr=lr
    )

    print("Training startet ...")
    print(f"obs_dim={obs_dim}, act_dim={act_dim}, device={device}")
    print(f"Config: {asdict(cfg)}")

    best_return = float("-inf")
    best_area = float("-inf")

    # Trainingsloop
    for episode in range(1, num_episodes + 1):

        metrics = train_one_episode(
            env=env,
            actor=actor,
            critic=critic,
            optimizer=optimizer,
            horizon=cfg.horizon,
            gamma=gamma,
            entropy_coef=entropy_coef,
            value_coef=value_coef,
        )

        ep_return = metrics["episode_return"]
        ep_area = metrics["area_det_final"]

        # Bestes Modell speichern
        if ep_return > best_return:
            best_return = ep_return
            save_checkpoint(
                path=os.path.join(SAVE_DIR, "best_by_return.pt"),
                actor=actor,
                critic=critic,
                cfg=cfg,
                episode=episode,
                metrics=metrics,
            )

        # Bestes Modell nach finaler Ellipsenfläche
        if ep_area > best_area:
            best_area = ep_area
            save_checkpoint(
                path=os.path.join(SAVE_DIR, "best_by_area.pt"),
                actor=actor,
                critic=critic,
                cfg=cfg,
                episode=episode,
                metrics=metrics,
            )

        # Regelmäßiges Logging
        if episode % 50 == 0 or episode == 1:
            print(
                f"[Episode {episode:4d}] "
                f"Return={metrics['episode_return']:.4f} | "
                f"Area={metrics['area_det_final']:.4f} | "
                f"Close={metrics['closure_dist2']:.4f} | "
                f"Penalty={metrics['terminal_penalty']:.4f} | "
                f"ActorLoss={metrics['actor_loss']:.4f} | "
                f"CriticLoss={metrics['critic_loss']:.4f}"
            )

    # Letztes Modell speichern
    save_checkpoint(
        path=os.path.join(SAVE_DIR, "last_model.pt"),
        actor=actor,
        critic=critic,
        cfg=cfg,
        episode=num_episodes,
        metrics=metrics,
    )

    print("Training abgeschlossen.")
    print(f"Best episode return: {best_return:.4f}")
    print(f"Best area:   {best_area:.4f}")


if __name__ == "__main__":
    main()