import numpy as np
import matplotlib.pyplot as plt

phase = np.linspace(0.0, 1.0, 500)

# Area weighting (80% -> linear down to 0.2)
alpha_area = np.maximum(0.2, 1.0 - np.maximum(0.0, (phase - 0.8) / 0.2))

# Degeneration weighting (start at 25%)
alpha_degen = np.maximum(0.0, (phase - 0.25) / 0.75)

# Closure weighting (start at 70%)
alpha_close = np.maximum(0.0, (phase - 0.7) / 0.3)

plt.figure(figsize=(9,5))

plt.plot(phase, alpha_area, linewidth=2, label="Area reward weight")
plt.plot(phase, alpha_degen, linewidth=2, label="Degeneration penalty weight")
plt.plot(phase, alpha_close, linewidth=2, label="Closure reward weight")

# wichtige Phasen markieren
plt.axvline(0.25, linestyle="--", alpha=0.6)
plt.axvline(0.70, linestyle="--", alpha=0.6)
plt.axvline(0.80, linestyle="--", alpha=0.6)

# xticks inklusive der wichtigen Phasen
plt.xticks([0, 0.25, 0.5, 0.7, 0.8, 1.0])

plt.text(0.26, 1.02, "Degeneration start", fontsize=9)
plt.text(0.71, 1.02, "Closing phase", fontsize=9)
plt.text(0.81, 1.02, "Area reduction", fontsize=9)

plt.xlabel("Phase t / T")
plt.ylabel("Reward weighting")
plt.title("Phase-dependent reward design")

plt.xlim(0,1)
plt.ylim(0,1.1)

plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig("reward_design_phase_weightss.png", dpi=300)
plt.show()