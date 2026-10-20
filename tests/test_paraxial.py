"""
Limite paraxial: núcleo de Fresnel entre fendas ↔ problema de Slepian.
"""

import numpy as np

from comm_modes import communication_modes, fresnel_kernel_1d, slit_quadrature, sum_rule
from comm_modes.paraxial import (
    confocal_phase,
    heuristic_number_1d,
    prolate_angular,
    slepian_c,
    slepian_eigenvalues,
)


def _slits(width, L, n=120, wavelength=1.0):
    x, w = slit_quadrature(width, n)
    M = fresnel_kernel_1d(x, w, x, w, wavelength, L)
    return x, w, M


def test_sum_rule_is_fresnel_number():
    # S = ∫∫|h|² dx_S dx_R = Δx_S Δx_R/(λL), exato para o núcleo de Fresnel
    for width, L in [(48.0, 192.0), (10.0, 7.0), (3.0, 50.0)]:
        _, _, M = _slits(width, L)
        assert np.isclose(sum_rule(M), heuristic_number_1d(width, width, 1.0, L), rtol=1e-12)


def test_transmissivities_are_at_most_one():
    _, _, M = _slits(48.0, 192.0)
    m = communication_modes(M)
    assert m.s[0] ** 2 <= 1 + 1e-12


def test_fresnel_eigenvalues_match_slepian():
    # o caso paraxial do Miller (Fig. 9): 48λ de extensão, 192λ de separação
    width, L = 48.0, 192.0
    _, _, M = _slits(width, L, n=160)
    eta = communication_modes(M).s ** 2
    ref = slepian_eigenvalues(slepian_c(width, width, 1.0, L))
    assert np.allclose(eta[:25], ref[:25], atol=1e-10)


def test_number_of_well_coupled_modes():
    # transição abrupta em N = 2c/π = Δx²/(λL) = 12
    width, L = 48.0, 192.0
    _, _, M = _slits(width, L, n=160)
    eta = communication_modes(M).s ** 2
    assert heuristic_number_1d(width, width, 1.0, L) == 12.0
    assert np.sum(eta > 0.5) == 12
    assert eta[10] > 0.9 and eta[13] < 0.1


def test_modes_are_prolate_spheroidal_functions():
    # c ≈ 4.7: autovalores bem separados, sem mistura de modos quase degenerados
    width, L = 12.0, 48.0
    x, w, M = _slits(width, L)
    m = communication_modes(M)
    c = slepian_c(width, width, 1.0, L)
    u = x / (width / 2)
    for n in range(5):
        S = prolate_angular(n, c, u) / np.sqrt(width / 2)  # normalizada em x
        # amostras da função contínua: dividir pelo √peso da quadratura
        psi = m.psi[:, n] / np.sqrt(w) * np.conj(confocal_phase(x, 1.0, L, "source"))
        phi = m.phi[:, n] / np.sqrt(w) * np.conj(confocal_phase(x, 1.0, L, "receiver"))
        # a fase confocal exata difere da paraxial por O(x⁴/L³): sobreposição ≈ 1
        assert abs(np.sum(w * np.conj(psi) * S)) > 0.999
        assert abs(np.sum(w * np.conj(phi) * S)) > 0.999
