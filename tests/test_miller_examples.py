"""
Reprodução numérica dos exemplos do tutorial do Miller (2019).
Distâncias em comprimentos de onda (λ = 1), como no artigo.
"""

import numpy as np

from comm_modes import (
    communication_modes,
    green_scalar_3d,
    line_points,
    plane_grid_points,
    sum_rule,
    wavenumber,
)

K = wavenumber(1.0)


def _three_by_three():
    # Fig. 3: 3 fontes espaçadas de 2λ, a 5λ de 3 receptores iguais
    rS = line_points(3, 4.0, z=0.0)
    rR = line_points(3, 4.0, z=5.0)
    return green_scalar_3d(rR, rS, K)


def test_three_sources_matrix_elements():
    # Eq. (43): g_SR · G_SR com g_SR = -4π L_z
    gSR = -4 * np.pi * 5.0
    G = gSR * _three_by_three()
    expected = np.array(
        [
            [1, -0.7 + 0.6j, -0.64 + 0.45j],
            [-0.7 + 0.6j, 1, -0.7 + 0.6j],
            [-0.64 + 0.45j, -0.7 + 0.6j, 1],
        ]
    )
    # o artigo arredonda para 1-2 casas
    assert np.allclose(G, expected, atol=0.015)


def test_three_sources_coupling_strengths_and_sum_rule():
    gSR2 = (4 * np.pi * 5.0) ** 2
    G = _three_by_three()
    m = communication_modes(G)
    # Eq. (46) e (44)
    assert np.allclose(m.coupling_strengths * gSR2, [3.41, 2.89, 1.37], atol=0.005)
    assert np.isclose(sum_rule(G) * gSR2, 7.67, atol=0.005)


def test_three_sources_mode_vectors_up_to_phase():
    m = communication_modes(_three_by_three())
    # Eq. (48): |ψ_S2⟩ = [-0.71, 0, 0.71] (modo antissimétrico)
    psi2 = m.psi[:, 1]
    psi2 = psi2 * np.exp(-1j * np.angle(psi2[2]))  # fixa a fase global
    assert np.allclose(psi2, [-0.71, 0, 0.71], atol=0.01)
    # Eq. (48)-(49): neste problema simétrico, φ_Rj = conj(ψ_Sj) a menos de fase
    for j in range(3):
        overlap = abs(np.vdot(m.phi[:, j].conj(), m.psi[:, j]))
        assert np.isclose(overlap, 1.0, atol=1e-10)


def test_nine_sources_table_1():
    # Seção 5.1: 9 fontes no mesmo comprimento de 4λ (espaçamento λ/2), L = 5λ
    gSR2 = (4 * np.pi * 5.0) ** 2
    G = green_scalar_3d(line_points(9, 4.0, z=5.0), line_points(9, 4.0, z=0.0), K)
    m = communication_modes(G)
    table_1 = np.array([20.73, 20.39, 19.09, 10.41, 1.90, 0.11, 0.0028, 0.000027, 0.000000065])
    # comparação relativa: o artigo dá 2 algarismos significativos para os menores
    assert np.allclose(m.coupling_strengths * gSR2, table_1, rtol=0.03)
    assert np.isclose(sum_rule(G) * gSR2, 72.65, atol=0.01)
    # só ~4 canais fortes apesar de haver 9 fontes: 97.2% da soma nos 4 primeiros
    assert np.isclose(np.sum(m.fraction_of_sum_rule()[:4]), 0.9723, atol=0.001)


def test_two_dimensional_arrays_fig_8():
    # Seção 5.2: 17 × 17 pontos com espaçamento λ (quadrado 16λ × 16λ).
    # O texto diz 50λ de separação, mas os números da Fig. 8(c)
    # (12.09% no modo 1 ... 5.61% no modo 9) só são reproduzidos com 100λ.
    rS = plane_grid_points(17, 16.0, z=0.0)
    rR = plane_grid_points(17, 16.0, z=100.0)
    m = communication_modes(green_scalar_3d(rR, rS, K))
    pct = 100 * m.fraction_of_sum_rule()
    assert np.isclose(pct[0], 12.09, atol=0.01)
    assert np.isclose(pct[8], 5.61, atol=0.01)
    # degenerescências por simetria do quadrado: (2,3) e (7,8) — Miller, Seção 5.2
    assert np.isclose(pct[1], pct[2], rtol=1e-8)
    assert np.isclose(pct[6], pct[7], rtol=1e-8)
    # heurística de Miller: não mais que 100/12.09 ≈ 8 modos tão bem acoplados
    assert 100 / pct[0] < 8.3
