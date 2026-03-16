import numpy as np
import matplotlib.pyplot as plt

phase = np.linspace(0.0, 1.0, 500)

alpha_degen = np.maximum(0, (phase - 0.25) / 0.75)

plt.figure(figsize=(8,4.5))
plt.plot(phase, alpha_degen, linewidth=2,
        label=r'$\alpha_{degen}(t)$')

plt.axvline(0.25, linestyle='--', alpha=0.7,
        label='Start degeneration penalty (25%)')

plt.xlabel('Phase t/T')
plt.ylabel('Degeneration weight')
plt.title('Phase-dependent degeneration penalty weighting')

plt.xlim(0,1)
plt.ylim(0,1.1)

plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("dense_degenration_penalty.png", dpi=250, bbox_inches="tight")
plt.show()