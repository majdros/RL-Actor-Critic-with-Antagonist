import os
import numpy as np
import matplotlib.pyplot as plt
import torch

from finger_env import FingerEllipseEnv, EnvConfig

SEED = 0
EVALUATE_EPISODES_NUM = 50
CHECKPOINT="checkpoints/20446-episoden/best_by_eval_return.pt"


def _checkpoint_label(checkpoint_path):
    parent = os.path.basename(os.path.dirname(checkpoint_path))
    filename = os.path.splitext(os.path.basename(checkpoint_path))[0]
    return f"{parent}/{filename}"

def load_training_log(checkpoint_path):
    candidate_paths = [
        os.path.join(os.path.dirname(checkpoint_path), "training_log.pt"),
        checkpoint_path,
        os.path.join(os.path.dirname(checkpoint_path), "last_model.pt"),
    ]

    for path in candidate_paths:
        if not os.path.exists(path):
            continue

        payload = torch.load(path, map_location="cpu")

        if isinstance(payload, dict) and "return" in payload and "area" in payload:
            return payload

        if isinstance(payload, dict) and "training_log" in payload:
            log = payload["training_log"]
            if "return" in log and "area" in log:
                return log

    raise FileNotFoundError(
        f"Kein Trainingslog für {checkpoint_path} gefunden. "
        "Speichere beim Training zusätzlich eine training_log.pt-Datei."
    )


def plot_learning_curves_from_checkpoint(checkpoint_path):
    try:
        log = load_training_log(checkpoint_path)
        plot_learning_curves(log)
        return
    except FileNotFoundError:
        pass

    checkpoint_dir = os.path.dirname(checkpoint_path)
    image_specs = [
        ("training_curve_return.png", "Learning curve: Return"),
        ("training_curve_area.png", "Learning curve: Final area"),
        ("training_curve_losses.png", "Learning curve: Actor & Critic Loss"),
        ("training_curve_entropy.png", "Learning curve: Entropy")
    ]

    for image_name, title in image_specs:
        image_path = os.path.join(checkpoint_dir, image_name)
        if not os.path.exists(image_path):
            raise FileNotFoundError(
                f"Weder Trainingslog noch Plotbild gefunden: {image_path}"
            )

        image = plt.imread(image_path)
        plt.figure(figsize=(14, 7))
        plt.imshow(image)
        plt.axis("off")
        plt.title(title)
        plt.tight_layout()
        plt.show()


def run_trained_trajectory(checkpoint_path, adv_noise_scale=0.0, seed=SEED):
    from evaluate import load_model, select_action

    actor, cfg = load_model(checkpoint_path)

    eval_cfg = EnvConfig(**cfg.__dict__)
    eval_cfg.adv_noise_scale = adv_noise_scale

    env = FingerEllipseEnv(cfg=eval_cfg, render_mode=None)

    obs, _ = env.reset(seed=seed)
    done = False

    traj = [env.traj[0].copy()]
    last_info = {}

    while not done:
        action = select_action(actor, obs, env)
        obs, reward, terminated, truncated, info = env.step(action)
        traj.append(info["fingertip_xy"].copy())
        done = terminated or truncated
        last_info = info

    return np.array(traj), last_info


def fit_pca_ellipse(points, k_axis=EnvConfig.k_axis, num_pts=EnvConfig.horizon + 1):
    mu = points.mean(axis=0)
    X = points - mu
    Sigma = np.cov(X.T) + 1e-12 * np.eye(2)

    evals, evecs = np.linalg.eigh(Sigma)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]

    a = k_axis * np.sqrt(max(evals[0], 0.0))
    b = k_axis * np.sqrt(max(evals[1], 0.0))

    t = np.linspace(0, 2 * np.pi, num_pts)
    ellipse_local = np.stack([a * np.cos(t), b * np.sin(t)], axis=0)
    ellipse_global = (evecs @ ellipse_local).T + mu
    return ellipse_global


def plot_final_trajectory_with_ellipse(checkpoint_path):
    traj, info = run_trained_trajectory(checkpoint_path, adv_noise_scale=0.0, seed=SEED)
    ellipse = fit_pca_ellipse(traj)

    plt.figure(figsize=(14, 7))
    plt.plot(traj[:, 0], traj[:, 1], label="Fingertip trajectory")
    plt.plot(ellipse[:, 0], ellipse[:, 1], "--", label="Estimated ellipse")
    plt.scatter(traj[0, 0], traj[0, 1], marker="o", label="Start")
    plt.scatter(traj[-1, 0], traj[-1, 1], marker="x", label="End")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(
        f"Final trajectory | area={info.get('area_det_final', 0):.3f}, "
        f"axis_ratio={info.get('axis_ratio_b_over_a_final', 0):.3f}"
    )
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_learning_curves(log):
    episodes = np.arange(1, len(log["return"]) + 1)

    plt.figure(figsize=(14, 7))
    plt.plot(episodes, log["return"])
    plt.xlabel("Episode")
    plt.ylabel("Episode return")
    plt.title("Learning curve: Return")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(14, 7))
    plt.plot(episodes, log["area"])
    plt.xlabel("Episode")
    plt.ylabel("Final ellipse area")
    plt.title("Learning curve: Final area")
    plt.grid(True)
    plt.show()

    if len(log.get("actor_loss", [])) > 0 and len(log.get("critic_loss", [])) > 0:
        plt.figure(figsize=(14, 7))
        plt.plot(episodes, log["actor_loss"], label="Actor loss")
        plt.plot(episodes, log["critic_loss"], label="Critic loss")
        plt.xlabel("Episode")
        plt.ylabel("Loss")
        plt.title("Learning curve: Actor & Critic Loss")
        plt.grid(True)
        plt.legend()
        plt.show()

    if len(log.get("entropy", [])) > 0:
        plt.figure(figsize=(14, 7))
        plt.plot(episodes, log["entropy"], label="Entropy")
        plt.xlabel("Episode")
        plt.ylabel("Entropy")
        plt.title("Learning curve: Entropy")
        plt.grid(True)
        plt.legend()
        plt.show()


def plot_training_curves_from_logs(
    save_dir,
    cfg,
    total_episodes,
    return_log,
    area_log,
    actor_loss_log,
    critic_loss_log,
    entropy_log,
    ma_window=100,
):
    episodes = np.arange(1, len(return_log) + 1)

    def moving_average(values, window=ma_window):
        if len(values) < window:
            return np.array(values)
        return np.convolve(values, np.ones(window) / window, mode="valid")

    plt.figure(figsize=(12, 6))
    plt.plot(episodes, return_log, alpha=0.4, label="Episode return")
    ret_smooth = moving_average(return_log)
    ret_label = (
        f"Moving average ({ma_window}) "
        f"[min={ret_smooth.min():.4f}, max={ret_smooth.max():.4f}, mean={ret_smooth.mean():.4f}]"
    )
    if len(return_log) >= ma_window:
        plt.plot(np.arange(ma_window, len(return_log) + 1), ret_smooth, label=ret_label)
    else:
        plt.plot(episodes, ret_smooth, label=ret_label)

    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title(
        f"Training Curve: Episode Return, adv_noise_scale:{cfg.adv_noise_scale}, "
        f"Episoden:{total_episodes}"
    )
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_curve_return.png"), dpi=200, bbox_inches="tight")
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(episodes, area_log, alpha=0.4, label="Final ellipse area")
    area_smooth = moving_average(area_log)
    area_label = (
        f"Moving average ({ma_window}) "
        f"[min={area_smooth.min():.4f}, max={area_smooth.max():.4f}, mean={area_smooth.mean():.4f}]"
    )
    if len(area_log) >= ma_window:
        plt.plot(np.arange(ma_window, len(area_log) + 1), area_smooth, label=area_label)
    else:
        plt.plot(episodes, area_smooth, label=area_label)

    plt.xlabel("Episode")
    plt.ylabel("Area")
    plt.title(
        f"Training Curve: Final Ellipse Area, adv_noise_scale={cfg.adv_noise_scale}, "
        f"Episoden:{total_episodes}"
    )
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_curve_area.png"), dpi=200, bbox_inches="tight")
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(episodes, actor_loss_log, alpha=0.4, label="Actor loss")
    plt.plot(episodes, critic_loss_log, alpha=0.4, label="Critic loss")

    actor_loss_smooth = moving_average(actor_loss_log)
    critic_loss_smooth = moving_average(critic_loss_log)

    if len(actor_loss_log) >= ma_window:
        episodes_smooth = np.arange(ma_window, len(actor_loss_log) + 1)
    else:
        episodes_smooth = episodes

    plt.plot(episodes_smooth, actor_loss_smooth, label=f"Actor loss MA({ma_window})")
    plt.plot(episodes_smooth, critic_loss_smooth, label=f"Critic loss MA({ma_window})")

    plt.xlabel("Episode")
    plt.ylabel("Value")
    plt.title(
        f"Training Curve: Actor & Critic Loss, adv_noise_scale={cfg.adv_noise_scale}, "
        f"Episoden:{total_episodes}"
    )
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_curve_losses.png"), dpi=200, bbox_inches="tight")
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(episodes, entropy_log, alpha=0.35, label="Entropy")

    entropy_smooth = moving_average(entropy_log)
    if len(entropy_log) >= ma_window:
        episodes_smooth = np.arange(ma_window, len(entropy_log) + 1)
    else:
        episodes_smooth = episodes

    plt.plot(episodes_smooth, entropy_smooth, label=f"Entropy MA({ma_window})")

    plt.xlabel("Episode")
    plt.ylabel("Entropy")
    plt.title(
        f"Training Curve: Policy Entropy, adv_noise_scale={cfg.adv_noise_scale}, "
        f"Episoden:{total_episodes}"
    )
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_curve_entropy.png"), dpi=200, bbox_inches="tight")
    plt.show()


def plot_robustness(checkpoint_path):
    from evaluate import load_model, evaluate

    actor, cfg = load_model(checkpoint_path)

    noise_levels = [0.0, 0.1, 0.25, 0.5, 0.75]
    mean_areas = []
    mean_returns = []

    for noise in noise_levels:
        summary = evaluate(actor, cfg, adv_noise_scale=noise, num_episodes=EVALUATE_EPISODES_NUM)
        mean_areas.append(summary["area"])
        mean_returns.append(summary["return"])
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharex=True)
    colors = plt.get_cmap("viridis")(np.linspace(0, 1, len(noise_levels)))

    axes[0].plot(noise_levels, mean_areas, linewidth=2, label="Area curve")
    for noise, area, color in zip(noise_levels, mean_areas, colors):
        axes[0].scatter(
            noise,
            area,
            color=color,
            s=70,
            zorder=4,
            label=f"noise={noise:.2f}",
        )
    axes[0].set_xlabel("Antagonist noise scale")
    axes[0].set_ylabel("Mean final area")
    axes[0].set_title("Robustness: Noise vs Area")
    axes[0].grid(True)
    axes[0].legend(loc="best", title="Noise levels")

    axes[1].plot(noise_levels, mean_returns, linewidth=2, label="Return curve")
    for noise, ret, color in zip(noise_levels, mean_returns, colors):
        axes[1].scatter(
            noise,
            ret,
            color=color,
            s=70,
            zorder=4,
            label=f"noise={noise:.2f}",
        )
    axes[1].set_xlabel("Antagonist noise scale")
    axes[1].set_ylabel("Mean return")
    axes[1].set_title("Robustness: Noise vs Return")
    axes[1].grid(True)
    axes[1].legend(loc="best", title="Noise levels")

    fig.suptitle("Robustness Evaluation across adv_noise_scale", y=0.98)
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    plt.show()


def plot_eval_mode_results(
    checkpoints,
    noise_levels,
    metric="return",
    num_episodes=EVALUATE_EPISODES_NUM,
):
    from evaluate import load_model, evaluate

    plt.figure(figsize=(14, 7))

    for checkpoint_path in checkpoints:
        actor, cfg = load_model(checkpoint_path)
        metric_values = []

        for noise in noise_levels:
            summary = evaluate(actor, cfg, adv_noise_scale=noise, num_episodes=num_episodes)
            metric_values.append(float(summary[metric]))

        plt.plot(
            noise_levels,
            metric_values,
            marker="o",
            linewidth=2,
            label=_checkpoint_label(checkpoint_path),
        )

    plt.xlabel("Antagonist noise scale")
    plt.ylabel(f"Mean {metric}")
    plt.title(f"Evaluation across adv_noise_scale ({metric})")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_eval_records(records_by_checkpoint, noise_levels, metric="return"):
    plt.figure(figsize=(14, 7))

    for checkpoint_path, summaries in records_by_checkpoint.items():
        metric_values = [float(summary[metric]) for summary in summaries]
        plt.plot(
            noise_levels,
            metric_values,
            marker="o",
            linewidth=2,
            label=_checkpoint_label(checkpoint_path),
        )

    plt.xlabel("Antagonist noise scale")
    plt.ylabel(f"Mean {metric}")
    plt.title(f"Evaluation across adv_noise_scale ({metric})")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # checkpoint = "checkpoints/20500-episoden-parallel/best_by_eval_return.pt"
    checkpoint=CHECKPOINT
    # checkpoint="checkpoints/continue-training/3000-episoden&0.25-noise-scale/best_by_eval_return.pt"
    
    # checkpoint="checkpoints/100000-episoden-parallel/best_by_eval_return.pt"
    
    # 1) Plot der finalen Trajektorie + Ellipse
    plot_final_trajectory_with_ellipse(checkpoint)

    # 2) + 3) Trainingskurven aus dem Checkpoint-Ordner laden
    plot_learning_curves_from_checkpoint(checkpoint)

    # 4) Robustheit gegen Antagonist
    plot_robustness(checkpoint)