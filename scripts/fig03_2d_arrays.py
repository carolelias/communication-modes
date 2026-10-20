"""
Fig. 3 — Reprodução da Fig. 8 do Miller: arranjos 2D de 17 × 17 fontes e receptores.

Quadrados de 16λ × 16λ (espaçamento λ). Observação: o texto do artigo diz
50λ de separação, mas os valores da Fig. 8(c) (12.09% de S no modo 1 ...
5.61% no modo 9, com os pares degenerados 2-3 e 7-8) só são reproduzidos
com 100λ — então usamos 100λ (ver tests/test_miller_examples.py).

(a) |s_j|² em % da regra da soma para os 24 primeiros modos;
(b) Re[φ] no plano do receptor para os 12 primeiros modos.
"""

import numpy as np
from _style import C1, DIVERGING, INK, INK2, plt, save

from comm_modes import communication_modes, field_from_point_sources, green_scalar_3d, plane_grid_points, wavenumber

k = wavenumber(1.0)
L = 100.0
rS = plane_grid_points(17, 16.0, z=0.0)
rR = plane_grid_points(17, 16.0, z=L)
m = communication_modes(green_scalar_3d(rR, rS, k))
pct = 100 * m.fraction_of_sum_rule()
print("% de S (24 primeiros):", np.round(pct[:24], 2))
print("heurística A²/(λL)² =", 16.0**4 / L**2)

# campo num plano z = L um pouco maior que o receptor
c = np.linspace(-11, 11, 111)
X, Y = np.meshgrid(c, c)
r_eval = np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, L)])

fig = plt.figure(figsize=(11, 6.2))
gs = fig.add_gridspec(3, 6, width_ratios=[2.4, 0.15, 1, 1, 1, 1])
ax = fig.add_subplot(gs[:, 0])
ax.bar(np.arange(1, 25), pct[:24], color=C1, width=0.75)
ax.set(xlabel="modo j", ylabel=r"$|s_j|^2$ (% de S)", title="(a) Forças de acoplamento", xlim=(0.3, 24.7))
ax.text(9.6, pct[8] + 0.4, f"modo 9: {pct[8]:.2f}%", color=INK2, fontsize=9)
ax.text(1.4, pct[0] + 0.1, f"modo 1: {pct[0]:.2f}%", color=INK2, fontsize=9)

for j in range(12):
    a = fig.add_subplot(gs[j // 4, 2 + j % 4])
    f = field_from_point_sources(r_eval, rS, m.psi[:, j], k).reshape(X.shape)
    f = f * np.exp(-1j * np.angle(f.ravel()[np.argmax(np.abs(f))]))  # fase global → Re máximo
    v = np.max(np.abs(f))
    a.pcolormesh(X, Y, f.real, cmap=DIVERGING, vmin=-v, vmax=v, shading="auto", rasterized=True)
    a.plot([-8, 8, 8, -8, -8], [-8, -8, 8, 8, -8], "--", color=INK, lw=0.8)
    a.set_aspect("equal")
    a.set(xticks=[], yticks=[])
    a.grid(False)
    a.set_title(f"{j + 1}: {pct[j]:.2f}%", fontsize=9)
fig.text(0.62, 0.995, "(b) Re[φ] no plano do receptor (tracejado: área dos receptores)", ha="center", fontsize=11)
fig.tight_layout()
save(fig, "fig03_2d_arrays.png")
