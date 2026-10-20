"""
Fig. 5 — Os modos de comunicação paraxiais são funções esferoidais prolatas.

Fendas de largura Δx = 12λ separadas por L = 48λ (c = k(Δx/2)²/L ≈ 4.7,
N = 2c/π = 3). Para cada modo:
  - retiramos a fase confocal (Miller, Eqs. 54-55) da fonte e do receptor;
  - o que sobra é real e coincide com S_0n(c, 2x/Δx) (scipy.special.pro_ang1).
A fonte e o receptor têm a MESMA forma de amplitude e curvaturas de fase
opostas: a fonte "converge" para o receptor, o receptor "recebe" a onda curva.
"""

import numpy as np
from _style import C1, C2, INK, INK2, plt, save

from comm_modes import communication_modes, fresnel_kernel_1d, slit_quadrature
from comm_modes.paraxial import confocal_phase, prolate_angular, slepian_c

lam, width, L = 1.0, 12.0, 48.0
x, w = slit_quadrature(width, 120)
m = communication_modes(fresnel_kernel_1d(x, w, x, w, lam, L))
c = slepian_c(width, width, lam, L)
u_fine = np.linspace(-1, 1, 400)

fig, axes = plt.subplots(1, 4, figsize=(12, 3.2), sharey=True)
for n, ax in enumerate(axes):
    # amostras da função contínua (÷ √peso) com a fase confocal removida
    psi = m.psi[:, n] / np.sqrt(w) * np.conj(confocal_phase(x, lam, L, "source"))
    phi = m.phi[:, n] / np.sqrt(w) * np.conj(confocal_phase(x, lam, L, "receiver"))
    S_fine = prolate_angular(n, c, u_fine) / np.sqrt(width / 2)
    S_pts = prolate_angular(n, c, x / (width / 2)) / np.sqrt(width / 2)
    # fase global: alinhar com a prolata
    psi *= np.exp(-1j * np.angle(np.sum(w * np.conj(S_pts) * psi)))
    phi *= np.exp(-1j * np.angle(np.sum(w * np.conj(S_pts) * phi)))
    ov = abs(np.sum(w * np.conj(S_pts) * psi))
    print(f"modo {n + 1}: |s|² = {m.s[n] ** 2:.5f}, sobreposição com S_0{n} = {ov:.8f}, máx |Im| = {np.max(np.abs(psi.imag)):.1e}")

    ax.plot(u_fine * width / 2, S_fine, color=INK, lw=1.2, zorder=1, label="prolata $S_{0n}(c,\\,2x/Δx)$")
    ax.plot(x[::4], psi.real[::4], "o", color=C1, ms=4, label="modo na fonte")
    ax.plot(x[2::4], phi.real[2::4], "x", color=C2, ms=4.5, mew=1.5, label="modo no receptor")
    ax.set(title=f"modo {n + 1}:  |s|² = {m.s[n] ** 2:.3f}", xlabel="x / λ")
axes[0].set_ylabel("amplitude (fase confocal removida)")
axes[0].legend(loc="lower left", fontsize=8)
fig.suptitle(f"Fendas de {width:.0f}λ a {L:.0f}λ: c = {c:.2f}, N = 2c/π = {2 * c / np.pi:.1f}", y=1.02, color=INK)
fig.tight_layout()
save(fig, "fig05_prolate_modes.png")
