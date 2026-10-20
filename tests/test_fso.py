"""
Enlace FSO entre aberturas circulares: limites físicos e convergência numérica.
"""

import numpy as np

from comm_modes.fso import (
    FSOLink,
    best_gaussian_efficiency,
    link_modes,
    plob_bound,
    plob_multimode,
    quadrature_size,
)


def test_quadrature_converged():
    link = FSOLink(L=2e3)
    _, eta1, *_ = link_modes(link)
    _, eta2, *_ = link_modes(link, quadrature_size(link, extra=1.5))
    assert np.allclose(eta1[:10], eta2[:10], atol=1e-8)


def test_transmissivities_physical_and_sum_rule():
    for L in [1e3, 5e3, 30e3]:
        link = FSOLink(L=L)
        _, eta, S, *_ = link_modes(link)
        assert np.all(eta <= 1 + 1e-12) and np.all(eta >= 0)
        # regra da soma do operador contínuo: A_T A_R / (λL)²
        assert np.isclose(S, link.fresnel_product, rtol=1e-10)
        # os modos discretos esgotam a regra da soma
        assert np.isclose(np.sum(eta), S, rtol=1e-6)


def test_far_field_limit():
    # S ≪ 1: um único modo leva quase toda a regra da soma, η_1 → A_T A_R/(λL)²
    link = FSOLink(L=100e3)
    _, eta, *_ = link_modes(link)
    assert np.isclose(eta[0] / link.fresnel_product, 1.0, atol=0.003)


def test_optimal_mode_beats_any_gaussian():
    for L in [2e3, 5e3, 20e3]:
        link = FSOLink(L=L)
        _, eta, *_ = link_modes(link)
        eta_g, _, _ = best_gaussian_efficiency(link)
        assert eta_g <= eta[0] * (1 + 1e-9)


def test_far_field_gaussian_is_siegman_optimum():
    # Campo distante: o modo ótimo é a iluminação uniforme da abertura; o melhor
    # gaussiano truncado chega a 81.45% dela com a/w ≈ 1.12 (Siegman, "Lasers").
    link = FSOLink(L=100e3)
    _, eta, *_ = link_modes(link)
    eta_g, w_T, _ = best_gaussian_efficiency(link)
    assert np.isclose(eta_g / eta[0], 0.8145, atol=0.002)
    assert np.isclose((link.D_T / 2) / w_T, 1.12, atol=0.01)


def test_plob_multimode_at_least_single_mode():
    _, eta, *_ = link_modes(FSOLink(L=3e3))
    assert plob_multimode(eta) >= plob_bound(eta[0])
    # far field: multimodo ≈ monomodo
    _, eta, *_ = link_modes(FSOLink(L=100e3))
    assert np.isclose(plob_multimode(eta), plob_bound(eta[0]), rtol=0.01)
