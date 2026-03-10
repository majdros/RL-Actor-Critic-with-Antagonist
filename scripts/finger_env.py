from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from icecream import ic

# Config
@dataclass
class EnvConfig:
    # Rollout horizon (episode length)
    horizon: int = 128

    # action is delta joint angles per step (rad)
    max_delta: float = 0.05  # rad ~ 2.8°

    # link lengths (cm)
    l1: float = 5.0
    l2: float = 2.5
    l3: float = 2.5

    # joint limits
    theta_min: float = -np.pi / 2
    theta_max: float = +np.pi / 2

    # ellipse estimation scale (PCA)
    k_axis: float = 3.0

    # penalties
    w_close: float = 1.0          # Scließ-Strafe [0.5, 0.75, 1.0]
    w_degen: float = 1.0          # Degenerierung-Strafe
    min_axis_ratio: float = 0.25  # tau
    w_action: float = 0.5        # Energie-Strafe:[0.0, 0.5, 1.0] sum ||action||^2

    # antagonist Stärke
    adv_noise_scale: float = 0.0  # relative zu max_delta

    # observation toggles
    include_xy: bool = True
    include_phase: bool = True


class FingerEllipseEnv(gym.Env):
    """
    Planar 3R finger (anthropomorphic index finger surrogate).

    Goal: create a large ellipse (PCA-based proxy) from the fingertip trajectory.
    - Action: delta joint angles (3,)
    - Observation: [sin/cos(theta1..3), x,y (optional), phase (optional)]
    - Episode length: fixed horizon
    - Reward: dense area increment - action penalty + terminal penalties
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, cfg: EnvConfig = EnvConfig(),render_mode: Optional[str] = None):
        super().__init__()
        assert cfg.horizon >= 16, "horizon musss >= 16 für stabile Ellibse Schätzung."

        self.cfg = cfg
        self.render_mode = render_mode

        # RNG (seed can be None)
        # self.np_random, _ = gym.utils.seeding.np_random(seed)

        # Spaces: alle möglichen Aktionen, die der Agent in der Umgebung ausführen darf.
        self.action_space = spaces.Box(             # kontinuierliches action
            low=-self.cfg.max_delta,
            high=+self.cfg.max_delta,
            shape=(3,),
            dtype=np.float32,
        )

        obs_dim = 6  # sin/cos für 3 joints

        self.reichweite = self.cfg.l1 + self.cfg.l2 + self.cfg.l3
        if self.cfg.include_xy:
            obs_dim += 2
        if self.cfg.include_phase:
            obs_dim += 1

        # observation_space: Alle features skaliert zu [-1, 1]
        high = [1.0] * 6
        if self.cfg.include_xy:
            high += [1.0, 1.0]      # normalisiert x, y
        if self.cfg.include_phase:
            high += [1.0]           # phase in [0,1], allowed bound [-1,1]
        high = np.array(high, dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        # State
        self.t: int = 0
        self.theta: np.ndarray = np.zeros(3, dtype=np.float64)

        # Trajectory: store p0..pT => horizon+1 points
        self.traj: np.ndarray = np.zeros((self.cfg.horizon + 1, 2), dtype=np.float64)

        # Reward caches / diagnostics
        self.prev_area: float = 0.0
        self.action_energy_sum: float = 0.0


    # Kinematics
    def fingertip_xy(self, theta: np.ndarray) -> np.ndarray:
        th1, th2, th3 = float(theta[0]), float(theta[1]), float(theta[2])
        a12 = th1 + th2
        a123 = a12 + th3
        # FK
        x = self.cfg.l1 * np.cos(th1) + self.cfg.l2 * np.cos(a12) + self.cfg.l3 * np.cos(a123)
        y = self.cfg.l1 * np.sin(th1) + self.cfg.l2 * np.sin(a12) + self.cfg.l3 * np.sin(a123)
        return np.array([x, y], dtype=np.float64)


    def clip_theta(self, theta: np.ndarray) -> np.ndarray:
        return np.clip(theta, self.cfg.theta_min, self.cfg.theta_max).astype(np.float64)


    def _obs(self) -> np.ndarray:
        parts = [
            np.sin(self.theta[0]), np.cos(self.theta[0]),
            np.sin(self.theta[1]), np.cos(self.theta[1]),
            np.sin(self.theta[2]), np.cos(self.theta[2]),
        ]

        if self.cfg.include_xy:
            x, y = self.fingertip_xy(self.theta)
            parts += [x /self.reichweite, y /self.reichweite]     # Koordinaten normalisieren [-1, 1]

        if self.cfg.include_phase:
            parts += [self.t / float(self.cfg.horizon)] # Zeit normieren [0, 1]

        return np.array(parts, dtype=np.float32)


    # Ellipse estimation (PCA proxy)
    def ellipse_area_from_covdet(self, points: np.ndarray) -> Tuple[float, float, float, float]:
        """
        Schätzt eine Ellipse aus Punktwolke über die Kovarianz (PCA-Proxy).

        - Kovarianz Σ beschreibt die Streuung der Punkte in x/y.
        - Eigenwerte λ1 >= λ2 geben die Varianz entlang Haupt-/Nebenrichtung.
        - Mit Skalierung k entstehen Halbachsen:
              a = k * sqrt(λ1),  b = k * sqrt(λ2)
        - Daraus folgt die Ellipsenfläche:
              A = πab = π * k^2 * sqrt(det(Σ))

        Returns:
            area_det: pi * k^2 * sqrt(det(Sigma))
            a, b:     PCA semi-axes (k*sqrt(eigenvalues))
            detSigma: det(Sigma)
        """
        # Für <2 Punkte ist Kovarianz nicht sinnvoll definiert -> Rückgabe mit Nullen.
        n = points.shape[0]
        if n < 2:
            return 0.0, 0.0, 0.0, 0.0
        ic(points)
        # Zentrieren der Punkte (Mittelwert auf den Ursprung verschieben).
        mu = points.mean(axis=0)
        X = points - mu

        # 2x2-Kovarianzmatrix; kleiner Diagonalterm verhindert numerische Probleme.
        Sigma = np.cov(X.T) + 1e-12 * np.eye(2)  # numerical stabilizer

        # Determinante der Kovarianz: in 2D proportional zur Flächenstreuung.
        detSigma = float(np.linalg.det(Sigma))
        detSigma = max(detSigma, 0.0)

        # Eigenwerte für symmetrische Matrizen.
        evals, _ = np.linalg.eigh(Sigma)    # aufsteigend sortiert
        # Größter Eigenwert -> Hauptachse, kleinster -> Nebenachse.
        lam1, lam2 = float(evals[-1]), float(evals[0])       # lam1: größerer Eigenwert, lam2: kleinerer Eigenwert

        # Halbachsen der skalierten PCA-Ellipse.
        a = self.cfg.k_axis * np.sqrt(max(lam1, 0.0))
        b = self.cfg.k_axis * np.sqrt(max(lam2, 0.0))

        # Äquivalente Flächenformel über det(Σ): π * k^2 * sqrt(det(Σ)).
        area_det = float(np.pi * (self.cfg.k_axis ** 2) * np.sqrt(detSigma))
        return area_det, a, b, detSigma


    def terminal_penalty(self) -> Tuple[float, Dict[str, Any]]:
        pts = self.traj[: self.t + 1]  # p0..pT

        area, a, b, detSigma = self.ellipse_area_from_covdet(pts)

        # Closure: zwingt pT ~ p0
        close_dist2 = float(np.sum((pts[-1] - pts[0]) ** 2))
        p_close = self.cfg.w_close * close_dist2

        # Degeneracy: zwingt b/a >= tau
        axis_ratio = (b / a) if a > 1e-12 else 0.0      # große Halbachse muss ein wert haben
        hinge = max(0.0, self.cfg.min_axis_ratio - axis_ratio)
        p_degen = self.cfg.w_degen * (hinge ** 2)

        penalty = p_close + p_degen

        info = {
            "area_det_final": float(area),
            "detSigma_final": float(detSigma),
            "ellipse_a_final": float(a),
            "ellipse_b_final": float(b),
            "axis_ratio_b_over_a_final": float(axis_ratio),
            "closure_dist2": float(close_dist2),
            "penalty_close": float(p_close),
            "penalty_degen": float(p_degen),
            "action_energy_sum": float(self.action_energy_sum),
            "terminal_penalty": float(penalty),
        }
        return float(penalty), info


    # Gym API
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)        #  Gymnasiums Seeding

        self.t = 0
        self.action_energy_sum = 0.0
        self.prev_area = 0.0
        # Joints Initialisierung
        self.theta = self.np_random.uniform(self.cfg.theta_min, self.cfg.theta_max, size=(3,)).astype(np.float64)

        p0 = self.fingertip_xy(self.theta)
        self.traj[:] = p0  # alle Punkte der Trajektorie inklusive Start Punkt auf P0 setzen

        obs = self._obs()
        info = {"fingertip_xy": p0.copy()}
        return obs, info


    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float64).reshape(3,)
        action = np.clip(action, -self.cfg.max_delta, self.cfg.max_delta)

        # Antagonist disturbance (bounded)
        if self.cfg.adv_noise_scale > 0.0:
            eps = self.np_random.uniform(-1.0, 1.0, size=(3,))
            noise = (self.cfg.adv_noise_scale * self.cfg.max_delta) * eps
            action = action + noise
            action = np.clip(action, -self.cfg.max_delta, self.cfg.max_delta)

        # Transition
        self.theta = self.clip_theta(self.theta + action)

        # Save point p_{t+1}
        p = self.fingertip_xy(self.theta)
        self.traj[self.t + 1] = p

        # Action energy
        a2 = float(np.sum(action * action))
        self.action_energy_sum += a2

        # Dense Fläche inkremental
        pts_now = self.traj[: self.t + 2]  # p0..p_{t+1}
        area_now, a, b, detSigma = self.ellipse_area_from_covdet(pts_now)

        d_area = area_now - self.prev_area
        self.prev_area = area_now

        reward = float(d_area - self.cfg.w_action * a2)

        # nächster Schritt
        self.t += 1
        truncated = (self.t >= self.cfg.horizon)    # Truncated is for time-limits when time is not part of the observation space. 
        terminated = False                          # Bei Training: terminated => V(st+1) = 0 / truncated => V(st+1)!= 0
        info: Dict[str, Any] = {
            "t": self.t,
            "fingertip_xy": p.copy(),
            "area_det_now": float(area_now),
            "d_area": float(d_area),
            "detSigma_now": float(detSigma),
            "a_now": float(a),
            "b_now": float(b),
            "action_l2": float(a2),
            "reward_dense": float(reward),
        }

        # Abschlussbewertung am Episodenende
        if truncated or terminated: 
            penalty, term_info = self.terminal_penalty()
            reward -= float(penalty)
            info.update(term_info)      # fügt key-value paare aus 'term_info' in 'info' hinzu, alte keys werden überschreiben!
            info["reward_terminal"] = float(-penalty)  # neues key-value paar
        # ic(info)
        obs = self._obs()

        if self.render_mode == "human":
            self.render()

        return obs, float(reward), terminated, truncated, info


    def render(self):
        import matplotlib.pyplot as plt

        n = max(1, min(self.t + 1, self.cfg.horizon + 1))
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
        plt.plot(p3[0], p3[1], "g", marker="o")

        r = self.cfg.l1 + self.cfg.l2 + self.cfg.l3
        plt.xlim(-r, r)
        plt.ylim(-r, r)
        plt.gca().set_aspect("equal", adjustable="box")
        plt.title(f"FingerEllipseEnv t={self.t}/{self.cfg.horizon} | adv={self.cfg.adv_noise_scale}")
        plt.grid()
        plt.pause(0.001)


# TEST
def run_episode(seed):
    cfg = EnvConfig(horizon=128, adv_noise_scale=0.0)   #128 Environment-Horizon(Env. Ebene = maximale Episodenlänge) = 128. Rollout-Horizon (Parameter-Update) ist Training-Ebene
    env = FingerEllipseEnv(cfg=cfg, render_mode='human')   # 'human'

    # WICHTIG: beide RNGs seeden
    env.action_space.seed(seed)
    obs, info = env.reset(seed=seed)

    done = False
    ep_return = 0.0

    traj = []

    while not done:
        action = env.action_space.sample()          # Zufällige Aktion erzeugen 
        obs, r, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        ep_return += r
        traj.append(info["fingertip_xy"])

    return ep_return, np.array(traj), info

def main():

    for i in range(3):
        ep_return, traj, info = run_episode(seed=None)
        print("\n")
        print(f"{i+1}. episode_return: {ep_return}, Ellipse_area: {info['area_det_final']}")
        print('=' * 30)
        print("return:", ep_return)
        print("area:", info["area_det_final"])
        print("closure_dist2:", info["closure_dist2"])
        print("penalty_close:", info["penalty_close"])
        print("penalty_degen:", info["penalty_degen"])
        print("axis_ratio:", info["axis_ratio_b_over_a_final"])
        print("terminal_penalty:", info["terminal_penalty"])

if __name__ == "__main__":
    main()
