import os
from dataclasses import asdict

import torch
import torch.nn.functional as F
import torch.optim as optim

from finger_env import FingerEllipseEnv, EnvConfig
from actor_critic import Actor, Critic
from rollout import collect_rollout
from evaluate import evaluate
from visualize_results import plot_training_curves_from_logs


device = EnvConfig().device
EPISODEN = 3000          # Anzahl von Training-Episoden

MODE = "NEU"            # Neu: Training ein frische Policy, RESUME: besthende Policy weiter trainineren 

## Training von vorne ##
if MODE == "NEU":
    RESUME_PATH = None 
    SAVE_DIR = os.path.join("checkpoints", f"{EPISODEN}-episoden")

## Resume Training ##
elif MODE == "RESUME":
    RESUME_PATH = os.path.join("checkpoints", "20446-episoden", "best_by_eval_return.pt")     # Bestehendes Policy-path eingeben
    SAVE_DIR = os.path.join("checkpoints","continue-training",f"{EPISODEN}-episoden&{EnvConfig().adv_noise_scale}-noise-scale")

if MODE not in {"NEU", "RESUME"}:
    raise ValueError("Training-Mode 'MODE' entweder 'NEU' oder 'RESUME' eingeben!")

# Training-Hyperparameter
GAMMA = 0.99
ACTOR_LR = 0.0003
CRITIC_LR = 0.0003
ENTROPY_COEF = 0.0025
VALUE_COEF = 0.5


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
def train_one_episode(env, actor, critic, actor_optimizer, critic_optimizer, horizon, gamma, entropy_coef, value_coef):
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

    obs_batch = rollout["obs"]
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

    # Actor update:
    actor_optimizer.zero_grad()
    # Wenn Advantage positiv ist, soll log_prob größer werden.
    # Wenn Advantage negativ ist, soll log_prob kleiner werden.
    actor_loss = -(log_probs * advantages.detach()).mean()
    # Entropy-Bonus: Fördert Exploration, verhindert frühes Kollabieren der Policy
    entropy_bonus = entropies.mean()
    actor_total_loss = actor_loss - entropy_coef * entropy_bonus
    # Backpropagatoion
    actor_total_loss.backward()
    # Gradienten clipping für stabileres Training
    torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=0.5)
    actor_optimizer.step()


    # Critic update: Critic soll Returns möglichst gut approximieren
    critic_optimizer.zero_grad()
    values_pred = critic(obs_batch)
    critic_loss = F.mse_loss(values_pred, returns.detach())
    critic_total_loss = value_coef * critic_loss
    # Backpropagatoion
    critic_total_loss.backward()
    # Gradienten clipping für stabileres Training
    torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=0.5)
    critic_optimizer.step()


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


def save_training_log(path, return_log, area_log, actor_loss_log, critic_loss_log, entropy_log):
    torch.save(
        {
            "return": list(return_log),
            "area": list(area_log),
            "actor_loss": list(actor_loss_log),
            "critic_loss": list(critic_loss_log),
            "entropy": list(entropy_log),

        },
        path,
    )


def main():
    # Hyperparameter
    cfg = EnvConfig()

    num_episodes = EPISODEN
    gamma = GAMMA
    hidden_dim = 128
    entropy_coef = ENTROPY_COEF
    value_coef = VALUE_COEF

    eval_every = 200          # alle 200 Trainings-Episoden evaluieren
    eval_episodes = 20        # Evaluation über 20 Episoden
    eval_noise = EnvConfig().adv_noise_scale      

    # Environment
    env = FingerEllipseEnv(cfg=cfg, render_mode=None)

    obs_dim = env.observation_space.shape[0]    # 9
    act_dim = env.action_space.shape[0]         # 3

    # Modelle
    actor = Actor(obs_dim=obs_dim, act_dim=act_dim, hidden_dim=hidden_dim).to(device)
    critic = Critic(obs_dim=obs_dim, hidden_dim=hidden_dim).to(device)

    actor_optimizer = optim.Adam(actor.parameters(), lr=ACTOR_LR)
    critic_optimizer = optim.Adam(critic.parameters(), lr=CRITIC_LR)

    # Resume: nur Gewichte laden 
    start_episode = 0
    if RESUME_PATH is not None:
        checkpoint = torch.load(RESUME_PATH, map_location=device)
        actor.load_state_dict(checkpoint["actor_state_dict"])
        critic.load_state_dict(checkpoint["critic_state_dict"])
        start_episode = int(checkpoint.get("episode", 0))
        print(f"Resume von: {RESUME_PATH}")
        print(f"Weiter ab Episode {start_episode + 1}")

    os.makedirs(SAVE_DIR, exist_ok=True)

    print("Training startet ...")
    print(f"obs_dim={obs_dim}, act_dim={act_dim}, device={device}")
    print(f"Config: {asdict(cfg)}")

    best_eval_return = float("-inf")
    best_eval_area = float("-inf")


    return_log = []
    area_log = []
    actor_loss_log = []
    critic_loss_log = []
    entropy_log = []



    # Trainingsloop
    for episode in range(start_episode + 1, start_episode + num_episodes + 1):
        metrics = train_one_episode(
            env=env,
            actor=actor,
            critic=critic,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            horizon=cfg.horizon,
            gamma=gamma,
            entropy_coef=entropy_coef,
            value_coef=value_coef,
        )
        # Metriken speichern
        return_log.append(metrics["episode_return"])
        area_log.append(metrics["area_det_final"])
        actor_loss_log.append(metrics["actor_loss"])
        critic_loss_log.append(metrics["critic_loss"])
        entropy_log.append(metrics["entropy"])

        # Periodische Evaluierung
        if episode % eval_every == 0 or episode == start_episode + num_episodes:
            actor.eval()

            eval_summary = evaluate(
                actor=actor,
                cfg=cfg,
                adv_noise_scale=eval_noise,
                num_episodes=eval_episodes,
            )

            mean_eval_return = float(eval_summary["return"])
            mean_eval_area = float(eval_summary["area"])

            print(
                f"[Eval @ Episode {episode:4d}] "
                f"mean_return={mean_eval_return:.4f} | "
                f"mean_area={mean_eval_area:.4f}"
            )

            if mean_eval_return > best_eval_return:
                best_eval_return = mean_eval_return
                save_checkpoint(
                    path=os.path.join(SAVE_DIR, "best_by_eval_return.pt"),
                    actor=actor,
                    critic=critic,
                    cfg=cfg,
                    episode=episode,
                    metrics={
                        "mean_eval_return": mean_eval_return,
                        "mean_eval_area": mean_eval_area,
                    },
                )

            if mean_eval_area > best_eval_area:
                best_eval_area = mean_eval_area
                save_checkpoint(
                    path=os.path.join(SAVE_DIR, "best_by_eval_area.pt"),
                    actor=actor,
                    critic=critic,
                    cfg=cfg,
                    episode=episode,
                    metrics={
                        "mean_eval_return": mean_eval_return,
                        "mean_eval_area": mean_eval_area,
                    },
                )

            actor.train()

        # Regelmäßiges Logging
        if episode % eval_every == 0 or episode == 1:
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
    last_episode = start_episode + num_episodes
    save_checkpoint(
        path=os.path.join(SAVE_DIR, "last_model.pt"),
        actor=actor,
        critic=critic,
        cfg=cfg,
        episode=last_episode,
        metrics=metrics,
    )
    save_training_log(
        path=os.path.join(SAVE_DIR, "training_log.pt"),
        return_log=return_log,
        area_log=area_log,
        actor_loss_log=actor_loss_log,
        critic_loss_log=critic_loss_log,
        entropy_log=entropy_log
    )

    print("Training abgeschlossen.")
    print(f"Best mean eval return: {best_eval_return:.4f}")
    print(f"Best mean eval area:   {best_eval_area:.4f}")

    plot_training_curves_from_logs(
        save_dir=SAVE_DIR,
        cfg=cfg,
        total_episodes=EPISODEN,
        return_log=return_log,
        area_log=area_log,
        actor_loss_log=actor_loss_log,
        critic_loss_log=critic_loss_log,
        entropy_log=entropy_log,
        ma_window=100,
    )

if __name__ == "__main__":
    main()