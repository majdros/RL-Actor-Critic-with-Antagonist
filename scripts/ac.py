# -------------------------------
# 1️⃣ Import & Setup
# -------------------------------
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

# Finger-Gelenke & Längen
l1, l2, l3 = 5.0, 2.5, 2.5  # cm
joint_limits = [-90, 90]     # Grad

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------
# 2️⃣ Finger-Vorwärtskinematik
# -------------------------------
def forward_kinematics(theta):
    """
    theta = [theta1, theta2, theta3] in Grad
    Rückgabe: Fingertip-Position [x, y]
    """
    th = np.radians(theta)
    x = l1*np.cos(th[0]) + l2*np.cos(th[0]+th[1]) + l3*np.cos(th[0]+th[1]+th[2])
    y = l1*np.sin(th[0]) + l2*np.sin(th[0]+th[1]) + l3*np.sin(th[0]+th[1]+th[2])
    return np.array([x, y])

# -------------------------------
# 3️⃣ Actor & Critic Netzwerke
# -------------------------------
class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 3),   # Δθ1, Δθ2, Δθ3
            nn.Tanh()           # skaliert Aktionen [-1,1]
        )
    
    def forward(self, state):
        return self.net(state)

class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)   # Value V(s)
        )
    
    def forward(self, state):
        return self.net(state)

# Actor & Critic Instanzen
actor = Actor().to(device)
critic = Critic().to(device)
actor_opt = optim.Adam(actor.parameters(), lr=1e-3)
critic_opt = optim.Adam(critic.parameters(), lr=1e-3)

# -------------------------------
# 4️⃣ Antagonist (Störungen)
# -------------------------------
def apply_antagonist(delta_theta, magnitude=5.0):
    """
    delta_theta: Δθ1-3
    magnitude: max Störung in Grad
    """
    noise = np.random.uniform(-magnitude, magnitude, size=3)
    return delta_theta + noise

# -------------------------------
# 5️⃣ Ellipse-Belohnung
# -------------------------------
def compute_ellipse_reward(trajectory):
    """
    trajectory: Liste von Fingertip [x,y]
    Rückgabe: Fläche der Ellipse
    Hinweis: PCA als einfache Approximation
    """
    pts = np.array(trajectory)
    cov = np.cov(pts.T)
    eigvals = np.linalg.eigvalsh(cov)  # reell & stabiler als eigvals
    eigvals = np.maximum(eigvals, 0.0)
    a, b = np.sqrt(eigvals)  # Halbachsen (Skala ~ 1 std)
    return float(np.pi * a * b)

# -------------------------------
# 6️⃣ Trainingsloop (Skizze)
# -------------------------------
for episode in range(10):
    theta = np.array([0.0, 0.0, 0.0])  # Startwinkel
    trajectory = []

    log_probs = []
    values = []

    # feste Policy-Std (einfacher Sketch)
    policy_std = 0.2

    for t in range(5000):  # Schritte pro Episode
        state = torch.as_tensor(theta, dtype=torch.float32, device=device)

        # Critic baseline V(s)
        value = critic(state).squeeze(-1)
        values.append(value)

        # Stochastische Policy um Actor-Output
        mu = actor(state)
        dist = Normal(mu, policy_std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum()
        log_probs.append(log_prob)

        # in Grad skalieren + Störung (in numpy)
        delta_theta = action.detach().cpu().numpy() * 5.0
        delta_theta = apply_antagonist(delta_theta)

        theta = np.clip(theta + delta_theta, joint_limits[0], joint_limits[1])
        fingertip = forward_kinematics(theta)
        trajectory.append(fingertip)

    # Belohnung
    reward = compute_ellipse_reward(trajectory)

    reward_t = torch.tensor(reward, dtype=torch.float32, device=device)
    log_probs_t = torch.stack(log_probs)
    values_t = torch.stack(values)

    # Critic: V(s) ~ terminal reward (einfacher Episoden-Sketch)
    critic_loss = ((values_t - reward_t) ** 2).mean()

    # Actor: REINFORCE mit Baseline
    advantages = (reward_t - values_t.detach())
    actor_loss = -(log_probs_t * advantages).mean()

    critic_opt.zero_grad()
    critic_loss.backward()
    critic_opt.step()

    actor_opt.zero_grad()
    actor_loss.backward()
    actor_opt.step()

    if episode % 50 == 0:
        print(f"Episode {episode}, Reward: {reward:.2f}")

# -------------------------------
# 7️⃣ Visualisierung
# -------------------------------
trajectory = np.array(trajectory)
plt.plot(trajectory[:,0], trajectory[:,1], '-o')
plt.axis('equal')
plt.title("Fingertip Trajektorie")
plt.show()