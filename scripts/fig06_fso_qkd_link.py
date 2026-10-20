"""
Fig. 6 — Aplicação: enlace de QKD em espaço livre limitado por difração.

λ = 1550 nm, telescópios de 10 cm no transmissor e no receptor, L de 1 a 100 km.
Só difração (sem turbulência, absorção ou apontamento).

(a) Perda geométrica: modo ótimo da SVD (η_1), melhor gaussiano (largura e
    foco otimizados) e a lei de campo distante S = A_T A_R/(λL)².
(b) Número de canais espaciais com η_j > 0.5 e com η_j > 0.1, comparado a S.
(c) Limite PLOB (taxa de chave máxima, bits por uso): só o melhor modo vs. todos
    os modos em paralelo (multiplexação espacial).
"""

import numpy as np
from _style import C1, C2, C3, INK, INK2, plt, save

from comm_modes.fso import FSOLink, best_gaussian_efficiency, link_modes, plob_bound, plob_multimode

L_km = np.geomspace(1, 100, 25)
rows = []
for Lk in L_km:
    link = FSOLink(L=Lk * 1e3)
    _, eta, S, *_ = link_modes(link)
    eta_g, w_T, focus = best_gaussian_efficiency(link)
    rows.append(
        dict(L=Lk, S=link.fresnel_product, eta1=eta[0], eta_g=eta_g, n05=int(np.sum(eta > 0.5)), n01=int(np.sum(eta > 0.1)),
             plob1=plob_bound(eta[0]), plobN=plob_multimode(eta))
    )
    r = rows[-1]
    print(f"L = {Lk:6.2f} km  S = {r['S']:8.3f}  η1 = {r['eta1']:.5f}  η_G = {eta_g:.5f}  (w_T/D = {w_T / link.D_T:.3f})  "
          f"N(η>0.5) = {r['n05']:2d}  PLOB 1 modo = {r['plob1']:.3g}  todos = {r['plobN']:.3g}")

get = lambda key: np.array([r[key] for r in rows])  # noqa: E731

fig, (a0, a1, a2) = plt.subplots(1, 3, figsize=(13, 3.9))

a0.loglog(L_km, get("eta1"), color=C1, label=r"modo ótimo (SVD), $\eta_1$")
a0.loglog(L_km, get("eta_g"), color=C2, label="melhor feixe gaussiano")
a0.loglog(L_km, np.minimum(get("S"), 1), "--", color=INK2, lw=1.2, label=r"$A_T A_R/(\lambda L)^2$  (campo distante)")
a0.set(xlabel="distância L (km)", ylabel="fração da potência recebida", title="(a) Perda geométrica", ylim=(1e-3, 1.5))
a0.legend(loc="lower left", fontsize=8.5)

a1.loglog(L_km, get("S"), "--", color=INK2, lw=1.2, label=r"$S = A_T A_R/(\lambda L)^2$")
nz = lambda n: np.where(n > 0, n, np.nan)  # noqa: E731  (0 canais: fora do eixo log)
a1.step(L_km, nz(get("n05")), where="mid", color=C1, label="nº de modos com η > 0.5")
a1.step(L_km, nz(get("n01")), where="mid", color=C3, label="nº de modos com η > 0.1")
a1.set(xlabel="distância L (km)", ylabel="canais espaciais", title="(b) Graus de liberdade espaciais", ylim=(0.5, 60))
a1.legend(loc="upper right", fontsize=8.5)

a2.loglog(L_km, get("plobN"), color=C1, label="todos os modos (multiplexado)")
a2.loglog(L_km, get("plob1"), color=C2, label="só o modo 1")
a2.set(xlabel="distância L (km)", ylabel="bits por uso do canal", title=r"(c) Limite PLOB  $\sum_j -\log_2(1-\eta_j)$")
a2.legend(loc="lower left", fontsize=8.5)
for a in (a0, a1, a2):
    a.grid(True, which="both", color="#efeeea", lw=0.5)
fig.suptitle("Enlace FSO a 1550 nm, aberturas de 10 cm (só difração)", y=1.02, color=INK)
fig.tight_layout()
save(fig, "fig06_fso_qkd_link.png")
