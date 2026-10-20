"""
Fig. 2 — Reprodução da Fig. 6 do Miller: 9 fontes e 9 receptores em linhas paralelas.

Mesmo comprimento de 4λ do exemplo 3×3, agora com espaçamento λ/2, L = 5λ.
Mesmo com 9 fontes, só há ~3 canais fortes (Tabela 1): mais pontos num
volume fixo não criam mais canais — é o "limite de difração" via regra da soma.

Para cada um dos 3 modos mais fortes:
  - amplitude e fase das fontes (fase zero no centro, como no artigo);
  - a onda resultante Re[φ(r)] calculada pela Eq. (53), multiplicada por √|z|
    para compensar o decaimento 1/√z de uma linha de fontes (só para a figura).
"""

import numpy as np
from _style import DIVERGING, INK, INK2, plt, save

from comm_modes import (
    communication_modes,
    field_from_point_sources,
    fix_phase_at_center,
    green_scalar_3d,
    line_points,
    sum_rule,
    wavenumber,
)

k = wavenumber(1.0)
Lz = 5.0
rS = line_points(9, 4.0, z=0.0)
rR = line_points(9, 4.0, z=Lz)
G = green_scalar_3d(rR, rS, k)
m = communication_modes(G)

gSR2 = (4 * np.pi * Lz) ** 2
print("g_SR² · |s_j|² =", np.array2string(m.coupling_strengths * gSR2, precision=4))
print("g_SR² · S =", round(sum_rule(G) * gSR2, 2), " (Miller: 72.65)")
print("% de S =", np.round(100 * m.fraction_of_sum_rule(), 2))

# grade no plano (y, z) entre as linhas; evita a vizinhança singular das fontes
z = np.linspace(0.25, Lz, 240)
y = np.linspace(-2.6, 2.6, 250)
Z, Y = np.meshgrid(z, y)
r_eval = np.column_stack([np.zeros(Z.size), Y.ravel(), Z.ravel()])

fig, axes = plt.subplots(3, 3, figsize=(10, 8.6), gridspec_kw={"width_ratios": [1, 1, 1.45]})
for row in range(3):
    h = fix_phase_at_center(m.psi[:, row])
    amp = np.abs(h)
    # fase desenrolada ao longo de y e centrada no meio: mostra a curvatura de
    # fase que a SVD "encontrou". No modo antissimétrico aparece um salto de π
    # no centro: é a troca de sinal da amplitude, não uma descontinuidade física.
    ph = np.unwrap(np.angle(h))
    ph -= ph[len(ph) // 2]
    a0, a1, a2 = axes[row]

    a0.barh(rS[:, 1], amp, height=0.3, color=INK2)
    a0.set(xlim=(0, 0.75), ylim=(-2.6, 2.6), ylabel=f"modo {row + 1}\n\ny / λ")
    a1.scatter(ph, rS[:, 1], s=40, color=INK, zorder=3)
    a1.set(xlim=(-1.6 * np.pi, 1.6 * np.pi), ylim=(-2.6, 2.6), xticks=[-np.pi, 0, np.pi], xticklabels=["-π", "0", "π"])

    phi = field_from_point_sources(r_eval, rS, h, k).reshape(Z.shape) * np.sqrt(Z)
    vmax = np.max(np.abs(phi.real))
    a2.pcolormesh(Z, Y, phi.real, cmap=DIVERGING, vmin=-vmax, vmax=vmax, shading="auto", rasterized=True)
    a2.scatter(np.zeros(9) + 0.12, rS[:, 1], s=10, color=INK)
    a2.scatter(np.zeros(9) + Lz - 0.04, rR[:, 1], s=10, color=INK, marker="s")
    a2.set(xlim=(0, Lz + 0.05), ylim=(-2.6, 2.6))
    a2.set_aspect("equal")
    a2.grid(False)
    a2.text(Lz - 0.1, 2.35, f"{100 * m.fraction_of_sum_rule()[row]:.1f}% de S", ha="right", color=INK, fontsize=9,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1.5))
    if row == 0:
        a0.set_title("amplitude da fonte")
        a1.set_title("fase da fonte (desenrolada)")
        a2.set_title(r"onda Re[$\phi$]·$\sqrt{z}$")
axes[2, 0].set_xlabel(r"$|h_q|$")
axes[2, 1].set_xlabel("fase (rad)")
axes[2, 2].set_xlabel("z / λ")
fig.suptitle("Nove fontes e nove receptores (reprodução da Fig. 6 do Miller)", y=0.995)
fig.tight_layout()
save(fig, "fig02_nine_sources_beams.png")
