import numpy as np
import matplotlib.pyplot as plt

phase = np.linspace(0.0, 1.0, 500)

alpha_area_t = np.maximum(0.2, 1.0 - np.maximum(0.0, (phase - 0.8) / 0.2))

plt.figure(figsize=(8, 4.5))
plt.plot(phase, alpha_area_t, linewidth=2, label=r'$\alpha_{\mathrm{area}}(t)$')
plt.axvline(0.7, linestyle='--', alpha=0.7, label='Start closing phase (70%)')
plt.axvline(0.8, linestyle='--', alpha=0.7, label='Start area reduction (80%)')
plt.xlabel('Phase $t/T$')
plt.ylabel(r'Area weight $\alpha_{\mathrm{area}}$')
plt.title('Phase-dependent weighting of area growth reward')
plt.ylim(0.0, 1.1)
plt.xlim(0.0, 1.0)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("flächen_wachstum.png", dpi=250, bbox_inches="tight")
plt.show()