from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import matplotlib.pyplot as plt
# -----------------------------
# Config
# -----------------------------

@dataclass
class EnvConfig:
    # Rollout horizon T
    horizon: int = 96

    # action is delta joint angles per step (rad)
    max_delta: float = 0.08

    # link lengths in meters
    l1: float = 0.05
    l2: float = 0.025
    l3: float = 0.025

    # joint limits ±90°
    theta_min: float = -np.pi / 2
    theta_max: float = +np.pi / 2

    # ellipse estimation scale (PCA)
    k_axis: float = 2.0

    # penalties
    # w_periodic: float = 20.0           # periodicity penalty on tangent mismatch between start/end
    w_close: float = 50.0              # closure penalty weight on ||p_T - p_1||^2
    w_degen: float = 50.0              # degeneracy penalty weight on (max(0, tau - b/a))^2
    min_axis_ratio: float = 0.2        # tau
    w_action: float = 0.05             # smoothness via sum ||action||^2

    # antagonist (disturbance) strength: 0 => off
    adv_noise_scale: float = 0.0       # disturbance magnitude relative to max_delta

    # observation includes (sin,cos) for each theta + fingertip xy + phase
    include_xy: bool = True
    include_phase: bool = True


class FingerEllipseEnv(gym.Env):
    """
    Anthropomorphic index finger (planar 3R) continuous shape optimization.
    Objective: generate largest possible ellipse from fingertip trajectory within reachable workspace.

    - Action: delta joint angles (3,)
    - Observation: [sin/cos(theta1..3), x,y, phase]
    - Episode length: fixed horizon T
    - Reward: sparse terminal (ellipse area - penalties)

    Antagonist: bounded disturbance added to action inside env.step().
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, cfg: EnvConfig = EnvConfig(), render_mode: Optional[str] = None, seed: Optional[int] = None):
        super().__init__()
        assert cfg.horizon >= 16, "horizon should be >= 16 for stable ellipse estimation."

        self.cfg = cfg
        self.render_mode = render_mode

        # RNG
        self.np_random, _ = gym.utils.seeding.np_random(seed)

        # Action space
        self.action_space = spaces.Box(
            low=-self.cfg.max_delta,
            high=+self.cfg.max_delta,
            shape=(3,),
            dtype=np.float32
        )

        # Observation space
        # base: sin/cos for 3 joints -> 6 dims
        obs_dim = 6
        reach = self.cfg.l1 + self.cfg.l2 + self.cfg.l3

        if self.cfg.include_xy:
            obs_dim += 2
        if self.cfg.include_phase:
            obs_dim += 1

        # bounds (safe)
        high = []
        high += [1.0] * 6
        if self.cfg.include_xy:
            high += [reach, reach]
        if self.cfg.include_phase:
            high += [1.0]
        high = np.array(high, dtype=np.float32)

        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        # Internal state
        self.t: int = 0
        self.theta: np.ndarray = np.zeros(3, dtype=np.float64)         # (3,)
        self.traj: np.ndarray = np.zeros((self.cfg.horizon, 2), dtype=np.float64)  # (T,2)
        self.action_energy_sum: float = 0.0

    # -----------------------------
    # Kinematics
    # -----------------------------
    def fingertip_xy(self, theta: np.ndarray) -> np.ndarray:
        th1, th2, th3 = float(theta[0]), float(theta[1]), float(theta[2])
        a12 = th1 + th2
        a123 = a12 + th3
        x = self.cfg.l1 * np.cos(th1) + self.cfg.l2 * np.cos(a12) + self.cfg.l3 * np.cos(a123)
        y = self.cfg.l1 * np.sin(th1) + self.cfg.l2 * np.sin(a12) + self.cfg.l3 * np.sin(a123)
        return np.array([x, y], dtype=np.float64)

    def clip_theta(self, theta: np.ndarray) -> np.ndarray:
        return np.clip(theta, self.cfg.theta_min, self.cfg.theta_max).astype(np.float64)

    # -----------------------------
    # Observation
    # -----------------------------
    def _obs(self) -> np.ndarray:
        parts = [
            np.sin(self.theta[0]), np.cos(self.theta[0]),
            np.sin(self.theta[1]), np.cos(self.theta[1]),
            np.sin(self.theta[2]), np.cos(self.theta[2]),
        ]
        if self.cfg.include_xy:
            x, y = self.fingertip_xy(self.theta)
            parts += [x, y]
        if self.cfg.include_phase:
            phase = self.t / max(1, self.cfg.horizon - 1)
            parts += [phase]

        return np.array(parts, dtype=np.float32)

    # -----------------------------
    # Ellipse estimation (PCA)
    # -----------------------------
    def ellipse_from_pca(self, points: np.ndarray) -> Tuple[float, float, float]:
        """
        Returns (area, a, b) using PCA/covariance ellipse approximation.
        a = k*sqrt(lambda1), b = k*sqrt(lambda2)
        """
        mu = points.mean(axis=0)
        X = points - mu
        Sigma = np.cov(X.T) + 1e-12 * np.eye(2)  # numerical stability
        evals, _ = np.linalg.eigh(Sigma)          # ascending
        evals = np.sort(evals)[::-1]              # descending
        lam1, lam2 = float(evals[0]), float(evals[1])
        a = self.cfg.k_axis * np.sqrt(max(lam1, 0.0))
        b = self.cfg.k_axis * np.sqrt(max(lam2, 0.0))
        area = float(np.pi * a * b)
        return area, a, b

    def terminal_reward(self) -> Tuple[float, Dict[str, Any]]:
        pts = self.traj.copy()
        area, a, b = self.ellipse_from_pca(pts)

        # Closure: enforce p_T ~ p_1 (squared distance)
        close_dist2 = float(np.sum((pts[-1] - pts[0]) ** 2))
        p_close = self.cfg.w_close * close_dist2

        # Periodicity: also match tangent direction at the seam
        # (pts[1]-pts[0]) should be close to (pts[-1]-pts[-2])
        # if pts.shape[0] >= 2:
        #     v0 = pts[1] - pts[0]
        #     vT = pts[-1] - pts[-2]
        #     tangent_mismatch2 = float(np.sum((vT - v0) ** 2))
        # else:
        #     tangent_mismatch2 = 0.0
        # p_periodic = self.cfg.w_periodic * tangent_mismatch2

        # Degeneracy: enforce b/a >= tau
        axis_ratio = (b / a) if a > 1e-12 else 0.0
        tau = self.cfg.min_axis_ratio
        hinge = max(0.0, tau - axis_ratio)
        p_degen = self.cfg.w_degen * (hinge ** 2)

        # Smoothness: penalize action energy
        p_action = self.cfg.w_action * float(self.action_energy_sum)

        reward = area - p_close - p_degen - p_action # - p_periodic

        info = {
            "ellipse_area": area,
            "ellipse_a": a,
            "ellipse_b": b,
            "axis_ratio_b_over_a": axis_ratio,
            "closure_dist2": close_dist2,
            "penalty_close": p_close,
            # "tangent_mismatch2": tangent_mismatch2,
            # "penalty_periodic": p_periodic,
            "penalty_degen": p_degen,
            "penalty_action": p_action,
            "reward_terminal": float(reward),
        }
        return float(reward), info

    # -----------------------------
    # Gym API
    # -----------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)

        self.t = 0
        self.action_energy_sum = 0.0

        # Random initial joint angles within ±90°
        self.theta = self.np_random.uniform(self.cfg.theta_min, self.cfg.theta_max, size=(3,)).astype(np.float64)

        # Initialize trajectory with initial fingertip point
        p0 = self.fingertip_xy(self.theta)
        self.traj[:] = p0  # fill buffer

        obs = self._obs()
        info = {"fingertip_xy": p0.copy()}
        return obs, info

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float64).reshape(3,)
        action = np.clip(action, -self.cfg.max_delta, self.cfg.max_delta)

        # Antagonist disturbance (bounded)
        if self.cfg.adv_noise_scale > 0.0:
            eps = self.np_random.uniform(-1.0, 1.0, size=(3,))
            action = action + (self.cfg.adv_noise_scale * self.cfg.max_delta) * eps
            action = np.clip(action, -self.cfg.max_delta, self.cfg.max_delta)

        # Transition: theta_{t+1} = clip(theta_t + action)
        self.theta = self.clip_theta(self.theta + action)

        # Save point
        p = self.fingertip_xy(self.theta)
        self.traj[self.t] = p

        # Smoothness accumulator
        self.action_energy_sum += float(np.sum(action * action))

        # Advance time
        self.t += 1
        terminated = (self.t >= self.cfg.horizon)
        truncated = False

        # Sparse reward: only at terminal
        if terminated:
            reward, info = self.terminal_reward()
        else:
            reward, info = 0.0, {}

        obs = self._obs()
        info.update({"t": self.t, "fingertip_xy": p.copy()})

        if self.render_mode == "human":
            self.render()

        return obs, float(reward), terminated, truncated, info

    def render(self):

        n = max(1, min(self.t, self.cfg.horizon))
        pts = self.traj[:n]
        th1, th2, th3 = self.theta
        p0 = np.array([0.0, 0.0])
        p1 = np.array([self.cfg.l1 * np.cos(th1), self.cfg.l1 * np.sin(th1)])
        a12 = th1 + th2
        p2 = p1 + np.array([self.cfg.l2 * np.cos(a12), self.cfg.l2 * np.sin(a12)])
        a123 = a12 + th3
        p3 = p2 + np.array([self.cfg.l3 * np.cos(a123), self.cfg.l3 * np.sin(a123)])

        plt.clf()
        plt.plot(pts[:, 0], pts[:, 1])
        plt.plot([p0[0], p1[0], p2[0], p3[0]], [p0[1], p1[1], p2[1], p3[1]], marker="o")

        r = self.cfg.l1 + self.cfg.l2 + self.cfg.l3
        plt.xlim(-r, r)
        plt.ylim(-r, r)
        plt.gca().set_aspect("equal", adjustable="box")
        plt.title(f"FingerEllipseEnv t={self.t}/{self.cfg.horizon} | adv={self.cfg.adv_noise_scale}")
        # plt.grid(True)

        plt.pause(0.001)




# -----------------------------
# Minimal sanity check
# -----------------------------
if __name__ == "__main__":
    cfg = EnvConfig(horizon=960, adv_noise_scale=0.0)  # T = 96, with disturbances
    env = FingerEllipseEnv(cfg=cfg, render_mode="human")

    obs, info = env.reset(seed=0)
    done = False
    ep_return = 0.0

    while not done:
        action = env.action_space.sample()
        obs, r, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        print("terminated:", terminated, " truncated:", truncated, " done:", done)
        ep_return += r

    print("Episode return:", ep_return)
    print("Terminal metrics:", {k: info[k] for k in info if k.startswith("ellipse_") or k.startswith("penalty_") or k in ["axis_ratio_b_over_a", "closure_dist2", "reward_terminal"]})