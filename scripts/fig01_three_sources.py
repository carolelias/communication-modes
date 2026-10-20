"""
Fig. 1 — Exemplo introdutório do Miller (Seção 4): 3 fontes e 3 receptores.

Fontes espaçadas de 2λ em y, receptores iguais a L_z = 5λ em z.
Imprime a matriz g_SR·G_SR, as forças de acoplamento e a regra da soma,
e compara com as Eqs. (43)-(48) do artigo.
"""

import numpy as np
from _style import C1, INK, INK2, plt, save

from comm_modes import communication_modes, green_scalar_3d, line_points, sum_rule, wavenumber

k = wavenumber(1.0)  # λ = 1: distâncias em comprimentos de onda
Lz = 5.0
rS = line_points(3, 4.0, z=0.0)
rR = line_points(3, 4.0, z=Lz)
G = green_scalar_3d(rR, rS, k)
m = communication_modes(G)

gSR = -4 * np.pi * Lz  # fator de escala do Miller, Eq. (41)
np.set_printoptions(precision=2, suppress=True)
print("g_SR · G_SR =\n", gSR * G)
print("g_SR² · |s_j|² =", m.coupling_strengths * gSR**2, "  (Miller: 3.41, 2.89, 1.37)")
print("g_SR² · S      =", round(sum_rule(G) * gSR**2, 2), "  (Miller: 7.67)")
for j in range(3):
    print(f"|ψ_S{j + 1}⟩ =", np.round(m.psi[:, j] * np.exp(-1j * np.angle(m.psi[0, j])), 2))

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(9, 3.4), gridspec_kw={"width_ratios": [1, 1.3]})

# (a) geometria
ax0.scatter(rS[:, 2], rS[:, 1], s=60, color=C1, zorder=3, label="fontes")
ax0.scatter(rR[:, 2], rR[:, 1], s=60, color=INK, marker="s", zorder=3, label="receptores")
ax0.set(xlabel="z / λ", ylabel="y / λ", xlim=(-1, 6), ylim=(-3, 3), title="(a) Geometria (Miller, Fig. 3)")
ax0.set_aspect("equal")
ax0.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.22))

# (b) forças de acoplamento vs. artigo
j = np.arange(1, 4)
ax1.bar(j, m.coupling_strengths * gSR**2, width=0.5, color=C1, label="este código")
ax1.scatter(j, [3.41, 2.89, 1.37], color=INK, marker="_", s=400, linewidths=2, zorder=3, label="Miller, Eq. (46)")
for jj, v in zip(j, m.coupling_strengths * gSR**2):
    ax1.text(jj, v + 0.08, f"{v:.2f}", ha="center", color=INK2)
ax1.set(xticks=j, xlabel="modo j", ylabel=r"$g_{SR}^2\,|s_j|^2$", ylim=(0, 4), title=f"(b) Forças de acoplamento — soma S·g² = {sum_rule(G) * gSR**2:.2f}")
ax1.legend(loc="upper right")
save(fig, "fig01_three_sources.png")
