"""
Fig. 4 — Corte abrupto dos valores singulares no regime paraxial (Miller, Fig. 9).

Geometria do Miller: 97 pares de fontes em duas linhas (espaçamento λ/2 em y,
λ/4 em z) cobrindo 48λ, e 97 receptores numa linha a L = 192λ.
Heurística: N ≈ Δy_S Δy_R/(λL) = 48·48/192 = 12 modos bem acoplados.

Comparamos três cálculos independentes:
  1. fontes pontuais com a função de Green 3D EXATA (Miller);
  2. fendas contínuas com o núcleo de FRESNEL (paraxial);
  3. autovalores de SLEPIAN λ_n(c), c = k(Δy/2)²/L (sem L nem fase: só c).
2 e 3 coincidem (é uma identidade matemática). 1 tem outra normalização
(pontos em vez de aberturas contínuas), então comparamos |s_j|/|s_1|.
"""

import numpy as np
from _style import C1, C2, INK, INK2, plt, save

from comm_modes import communication_modes, double_line_points, fresnel_kernel_1d, green_scalar_3d, line_points, slit_quadrature, wavenumber
from comm_modes.paraxial import heuristic_number_1d, slepian_c, slepian_eigenvalues

lam, width, L = 1.0, 48.0, 192.0
k = wavenumber(lam)

# 1. Green exata, pontos (duas linhas de fontes, uma de receptores)
rS = double_line_points(97, width, z=0.0, dz=0.25)
rR = line_points(97, width, z=L)
s_pts = communication_modes(green_scalar_3d(rR, rS, k)).s

# 2. Fresnel, fendas contínuas
x, w = slit_quadrature(width, 200)
s_fr = communication_modes(fresnel_kernel_1d(x, w, x, w, lam, L)).s

# 3. Slepian
c = slepian_c(width, width, lam, L)
eta_sl = slepian_eigenvalues(c, n_quad=200)
N = heuristic_number_1d(width, width, lam, L)
print(f"c = {c:.3f},  N = 2c/π = {N:.1f}")
print("η_j (Fresnel)  :", np.round(s_fr[:16] ** 2, 4))
print("η_j (Slepian)  :", np.round(eta_sl[:16], 4))
print("|s_j|/|s_1| pontos:", np.round(s_pts[:16] / s_pts[0], 4))

n_show = 30
j = np.arange(1, n_show + 1)
fig, (a0, a1) = plt.subplots(1, 2, figsize=(10.5, 3.8))

a0.bar(j, s_fr[:n_show] ** 2, color=C1, width=0.75, label="Fresnel (fendas contínuas)")
a0.plot(j, eta_sl[:n_show], "o", color=INK, ms=4, label=r"Slepian $\lambda_n(c)$")
a0.axvline(N + 0.5, color=INK2, ls="--", lw=1)
a0.text(N + 0.9, 0.55, f"N = Δy²/(λL) = {N:.0f}", color=INK2)
a0.set(xlabel="modo j", ylabel=r"transmissividade $|s_j|^2$", title="(a) Transmissividade dos modos (paraxial)", ylim=(0, 1.08))
a0.legend(loc="upper right")

a1.semilogy(j, s_pts[:n_show] / s_pts[0], "-", color=C2, label="pontos, Green 3D exata (Miller)")
a1.semilogy(j, s_fr[:n_show] / s_fr[0], "-", color=C1, label="fendas, Fresnel")
a1.axvline(N + 0.5, color=INK2, ls="--", lw=1)
a1.set(xlabel="modo j", ylabel=r"$|s_j|\,/\,|s_1|$", title="(b) Queda exponencial após N", ylim=(1e-8, 2))
a1.legend(loc="lower left")
fig.tight_layout()
save(fig, "fig04_paraxial_cutoff.png")
