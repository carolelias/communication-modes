"""
Propriedades matemáticas da SVD que a teoria garante (Miller, Seção 3.7).
Testadas numa geometria genérica, sem simetria, para não depender de casos especiais.
"""

import numpy as np

from comm_modes import communication_modes, green_scalar_3d, sum_rule, wavenumber

RNG = np.random.default_rng(1234)


def _random_geometry(n_s=7, n_r=11):
    # fontes e receptores em posições aleatórias, em volumes separados em z
    rS = RNG.uniform(-2, 2, size=(n_s, 3))
    rR = RNG.uniform(-3, 3, size=(n_r, 3)) + np.array([0.0, 0.0, 10.0])
    return green_scalar_3d(rR, rS, wavenumber(1.0))


def test_orthonormality_of_both_sets():
    m = communication_modes(_random_geometry())
    n = len(m.s)
    assert np.allclose(m.psi.conj().T @ m.psi, np.eye(n), atol=1e-12)
    assert np.allclose(m.phi.conj().T @ m.phi, np.eye(n), atol=1e-12)


def test_modes_come_in_pairs():
    G = _random_geometry()
    m = communication_modes(G)
    # Eq. (31): G|ψ_Sj⟩ = s_j|φ_Rj⟩ ;  Eq. (32): G†|φ_Rj⟩ = s_j*|ψ_Sj⟩
    assert np.allclose(G @ m.psi, m.phi * m.s, atol=1e-14)
    assert np.allclose(G.conj().T @ m.phi, m.psi * m.s.conj(), atol=1e-14)


def test_two_eigenproblems():
    G = _random_geometry()
    m = communication_modes(G)
    # Eqs. (29) e (30)
    assert np.allclose(G.conj().T @ G @ m.psi, m.psi * m.s**2, atol=1e-14)
    assert np.allclose(G @ G.conj().T @ m.phi, m.phi * m.s**2, atol=1e-14)


def test_svd_reconstruction():
    G = _random_geometry()
    m = communication_modes(G)
    # Eq. (33): G = Σ s_j |φ_Rj⟩⟨ψ_Sj|
    assert np.allclose(m.phi @ np.diag(m.s) @ m.psi.conj().T, G, atol=1e-14)


def test_sum_rule_equals_frobenius_norm():
    G = _random_geometry()
    m = communication_modes(G)
    # Eq. (36): Σ|s_j|² = Σ_ij |g_ij|²
    assert np.isclose(m.sum_rule, sum_rule(G), rtol=1e-12)


def test_maximization_property():
    G = _random_geometry()
    m = communication_modes(G)
    # nenhuma fonte normalizada entrega mais potência que |ψ_S1⟩ ...
    v = RNG.normal(size=(G.shape[1], 2000)) + 1j * RNG.normal(size=(G.shape[1], 2000))
    v /= np.linalg.norm(v, axis=0)
    p = np.sum(np.abs(G @ v) ** 2, axis=0)
    assert np.all(p <= m.s[0] ** 2 * (1 + 1e-12))
    # ... e, entre as ortogonais a |ψ_S1⟩, nenhuma supera |ψ_S2⟩
    v_perp = v - np.outer(m.psi[:, 0], m.psi[:, 0].conj() @ v)
    v_perp /= np.linalg.norm(v_perp, axis=0)
    p_perp = np.sum(np.abs(G @ v_perp) ** 2, axis=0)
    assert np.all(p_perp <= m.s[1] ** 2 * (1 + 1e-12))


def test_singular_values_sorted_and_nonnegative():
    m = communication_modes(_random_geometry())
    assert np.all(m.s >= 0)
    assert np.all(np.diff(m.s) <= 0)
