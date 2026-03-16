import numpy as np
import matplotlib.pyplot as plt

# Beispielwert aus dem Env
w_action = 0.02

# 3 DoF, jede Action-Komponente liegt typischerweise in [-max_delta, max_delta]
max_delta = 0.05

# ||a||² liegt dann zwischen 0 und 3 * max_delta²
a2_max = 3 * (max_delta ** 2)
a2 = np.linspace(0.0, a2_max, 500)

# negativer Beitrag zum Gesamtreward
reward_contrib_action = -w_action * a2

plt.figure(figsize=(8, 4.5))
plt.plot(a2, reward_contrib_action, linewidth=2, label=r'$-w_{action}\|a_t\|^2$')
plt.xlabel(r'Action magnitude $\|a_t\|^2$')
plt.ylabel('Reward contribution')
plt.title('Dense action penalty')
plt.xlim(0.0, a2_max)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("dense_action_penalty.png", dpi=250, bbox_inches="tight")
plt.show()