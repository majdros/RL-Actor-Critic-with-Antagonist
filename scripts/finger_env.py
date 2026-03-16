from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch


# Configs
@dataclass
class EnvConfig:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Rollout horizon (episode Länge)
    horizon: int = 256

    # Action ist delta joint winkel pro Step (rad)
    max_delta: float = 0.05  # rad ~ 2.8°

    # Links Längen (cm)
    l1, l2, l3 = float(5.0), float(2.5), float(2.5)

    # joint limits
    theta_min: float = -np.pi / 2
    theta_max: float = +np.pi / 2

    # ellipse estimation scale (PCA)
    k_axis: float = 1.0

    # penalties
    w_area: float = 1.0           # Dense Ellipse-Fläche Reward (1.0: deaktiviert!)
    w_close: float = 0.05         # Terminal Scließ-Strafe [0.2, 0.4, 0.5]
    w_close_dense: float = 0.05   # Dense Schließ-Strafe
    w_degen: float = 0.1          # Terminal Degenerierung-Strafe
    w_degen_dense: float = 0.01   # Dense Degenerierung-Strafe
    w_action: float = 0.02        # Dense Energie-Strafe:[0.2, 0.4, 0.5] sum ||action||^2 {0.05 -> 0.02 einfriert}
    min_axis_ratio: float = 0.35  # tau

    # antagonist Stärke in rad [0.0, 0.1, 0.25, 0.5, 0.75]
    adv_noise_scale: float = 0.25  # relative zu max_delta => 0.25 * 0.05 = 0.0125 rad ~ 0.72°


class FingerEllipseEnv(gym.Env):
    """
        Planarer 3R-Finger zur Erzeugung einer großen, nicht-degenerierten Trajektorien-Ellipse.

        - Action: delta-Gelenkwinkel (3,), begrenzt auf [-max_delta, +max_delta]
        - Observation: [sin/cos(theta1..3), x_norm, y_norm, phase]
        - Episode: feste Länge (horizon), terminated=False, Abschluss über truncated
        - Dynamik: optionaler antagonistischer Störterm (adv_noise_scale) auf die Action
        - Reward (dense):
            + Flächenzuwachs (PCA/Kovarianz), phasenabhängig gewichtet
            + Closure-Verbesserung ab spätem Episodenabschnitt
            - Aktionsenergie (L2)
            - Degenerationsstrafe bei kleinem Achsenverhältnis b/a
        - Zusätzlich am Episodenende: terminale Penalty aus Closure + Degeneration
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, cfg: EnvConfig = EnvConfig(),render_mode: Optional[str] = None):
        super().__init__()
        assert cfg.horizon >= 16, "horizon musss >= 16 für stabile Ellibse Schätzung."

        self.cfg = cfg
        self.render_mode = render_mode

        # Spaces: alle möglichen Aktionen, die der Agent in der Umgebung ausführen darf.
        self.action_space = spaces.Box(             # kontinuierliches action
            low=-self.cfg.max_delta,
            high=+self.cfg.max_delta,
            shape=(3,),
            dtype=np.float32,
        )

        obs_dim = 9  # sin/cos für 3 joints + x y von Endeffektor + phase

        self.reichweite = self.cfg.l1 + self.cfg.l2 + self.cfg.l3

        # observation_space: Alle features skaliert zu [-1, 1]
        high = [1.0] * 9
        high = np.array(high, dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        # State
        self.t: int = 0
        self.theta: np.ndarray = np.zeros(3, dtype=np.float64)


        # Trajectory: speichert p0..pT => horizon+1 points
        self.traj: np.ndarray = np.zeros((self.cfg.horizon + 1, 2), dtype=np.float64)

        # Reward speicher / diagnose
        self.prev_area: float = 0.0
        self.prev_close_dist: float = 0.0
        self.action_energy_sum: float = 0.0


    # Forward Kinematik
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

        x, y = self.fingertip_xy(self.theta)
        parts += [x / self.reichweite, y / self.reichweite]  # Koordinaten normalisieren [-1, 1]

        self.phase = [self.t /float(self.cfg.horizon)]       # Zeit normieren [0, 1]
        parts += self.phase            

        return np.array(parts, dtype=np.float32)


    # Ellipse Schätzung (PCA proxy)
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
        eval_a, eval_b = float(evals[-1]), float(evals[0])       # eval_a: größerer Eigenwert, eval_b: kleinerer Eigenwert

        # Halbachsen der skalierten PCA-Ellipse.
        a = self.cfg.k_axis * np.sqrt(max(eval_a, 0.0))
        b = self.cfg.k_axis * np.sqrt(max(eval_b, 0.0))

        # Äquivalente Flächenformel über det(Σ): π * k^2 * sqrt(det(Σ)); k = 1
        area_det = float(np.pi * (self.cfg.k_axis ** 2) * np.sqrt(detSigma))
        return area_det, a, b, detSigma


    def terminal_penalty(self) -> Tuple[float, Dict[str, Any]]:
        pts = self.traj[: self.t + 1]  # p0..pT

        area, a, b, detSigma = self.ellipse_area_from_covdet(pts)

        # Closure: zwingt pT ~ p0
        close_dist = float(np.sum((pts[-1] - pts[0]) ** 2))
        p_close = self.cfg.w_close * close_dist

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
            "closure_dist2": float(close_dist),
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
        self.prev_close_dist = 0.0

        # Joints Initialisierung
        self.theta = self.np_random.uniform(self.cfg.theta_min, self.cfg.theta_max, size=(3,)).astype(np.float64)

        self.p0 = self.fingertip_xy(self.theta)
        self.traj[:] = self.p0  # alle Punkte der Trajektorie inklusive Start Punkt auf P0 setzen

        obs = self._obs()
        info = {"fingertip_xy": self.p0.copy()}
        return obs, info


    def reward_function(self, action: np.ndarray):
        # Dense-REWARD Design
        close_dist = float(np.sum((self.p - self.p0)**2))
        phase = (self.t) / self.cfg.horizon

        ## 1. close reward ab dem dritten Drittel der Episode: fördert Schließung der Ellipse
        alpha_close_t = max(0, (phase - 0.7) / 0.3)       #    Bei 70% der Episode, anfangen mit der Ellipse-Schließung
        r_close_dense = self.cfg.w_close_dense * alpha_close_t * (self.prev_close_dist - close_dist)
        self.prev_close_dist = close_dist

        ## 2.  Action energy zur vermeindung von hektischen Bewegungen
        a2 = float(np.sum(action * action))
        self.action_energy_sum += a2
        r_action_dense = self.cfg.w_action * a2

        ## 3. Haupt-Reward: Ellipsen-Fläche
        ## Alle Points in 'pts_now' speichern und die Fläche der Ellipse berechnen
        pts_now = self.traj[: self.t + 1]  # p0..p_{t+1}
        area_now, a, b, detSigma = self.ellipse_area_from_covdet(pts_now)
        # Bis 80% Phase Fläche komplett als reward nehemen
        # Phase >= 80 % Flächen-reward fällt linear und bleibt mindestens bei 0.2
        alpha_area_t = max(0.2, 1.0 - max(0.0, (phase - 0.8) / 0.2))
        r_area_dense = alpha_area_t * (area_now - self.prev_area)
        self.prev_area = area_now

        ## 4. Degenaration Strafe
        axis_ratio = (b / a) if a > 1e-12 else 0.0           # große Halbachse muss ein wert haben
        hinge = max(0.0, self.cfg.min_axis_ratio - axis_ratio)
        alpha_degne_t = max(0.0, (phase - 0.25) / 0.75)      # Ab 25% der Episode berücksichtigen diese Strafe
        r_degen_dense = self.cfg.w_degen_dense * alpha_degne_t * (hinge ** 2)

        ## dense_reward zusammen summieren
        reward = float(r_area_dense - r_action_dense + r_close_dense - r_degen_dense)

        info: Dict[str, Any] = {
            "alpha_close_t": float(alpha_close_t),
            "r_close_dense": float(r_close_dense),
            "action_l2": float(a2),
            "r_action_dense": float(r_action_dense),
            "area_det_now": float(area_now),
            "r_area_dense": float(r_area_dense),
            "detSigma_now": float(detSigma),
            "a_now": float(a),
            "b_now": float(b),
            "axis_ratio": float(axis_ratio),
            "hinge": float(hinge),
            "r_degen_dense": float(r_degen_dense),
            "reward_dense": float(reward),
        }

        truncated = (self.t >= self.cfg.horizon)
        terminated = False

        # Terinal Penalty
        if truncated or terminated:
            penalty, penalty_info = self.terminal_penalty()
            info.update(penalty_info)
            # Terminal Penalty von Dense-Reward abziehen
            reward -= float(penalty)

        return float(reward), info, truncated, terminated


    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float64).reshape(3,)
        action = np.clip(action, -self.cfg.max_delta, self.cfg.max_delta)

        # Antagonist disturbance (bounded)
        if self.cfg.adv_noise_scale > 0.0:
            eps = self.np_random.uniform(-1.0, 1.0, size=(3,))  # Random-Wert: skaliert noise [-1.0, 1.0]
            noise = (self.cfg.adv_noise_scale * self.cfg.max_delta) * eps
            action = action + noise
            action = np.clip(action, -self.cfg.max_delta, self.cfg.max_delta)

        # Transition
        self.theta = self.clip_theta(self.theta + action)

        # Aktuelle Position von der Fingerspitze bzw. Endeffektor berechnen
        self.p = self.fingertip_xy(self.theta)

        # point p_{t+1} in Trajektorie speichern
        self.traj[self.t + 1] = self.p

        self.t += 1

        reward, info, truncated, terminated  = self.reward_function(action) 
        info["t"] = self.t
        info["fingertip_xy"] = self.p.copy()

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


# Environment-TEST
def run_episode(seed):
    cfg = EnvConfig()   #256 Environment-Horizon(Env. Ebene = maximale Episodenlänge) = 256. Rollout-Horizon (Parameter-Update) ist Training-Ebene
    env = FingerEllipseEnv(cfg=cfg, render_mode='human')

    # seed setzen
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

    for i in range(1):
        ep_return, traj, info = run_episode(seed=None)
        print("\n")
        # print(f"{i+1}. episode_return: {ep_return}, Ellipse_area: {info['area_det_final']}")
        print('=' * 30)
        print("return:", ep_return)
        print(info)
        print("area:", info["area_det_final"])
        print("closure_dist2:", info["closure_dist2"])
        print("penalty_close:", info["penalty_close"])
        print("penalty_degen:", info["penalty_degen"])
        print("axis_ratio:", info["axis_ratio_b_over_a_final"])
        print("r_close_dense:", info["r_close_dense"])
        print("terminal_penalty:", info["terminal_penalty"])

if __name__ == "__main__":
    main()
