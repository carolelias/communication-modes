"""
Operadores de acoplamento G_SR entre o espaço da fonte e o espaço do receptor.

Dois modelos são implementados:

1. Função de Green escalar exata em 3D (Miller 2019, Eqs. 4 e 7), para
   conjuntos de fontes e receptores pontuais:

       g_ij = -(1/4π) · exp(i k |r_Ri - r_Sj|) / |r_Ri - r_Sj|

   É a resposta da equação de Helmholtz (∇² + k²)φ = ψ a uma fonte δ(r - r_Sj).

2. Núcleo de Fresnel (aproximação paraxial) entre aberturas CONTÍNUAS em
   planos paralelos separados por L. Aqui a fonte é um campo de abertura
   E_S(x_S) e o receptor é o campo E_R(x_R) no outro plano:

       2D (aberturas em planos):  h(r_R, r_S) = exp(ikL)/(iλL) · exp(ik|r_R - r_S|²/(2L))
       1D (fendas):               h(x_R, x_S) = exp(ikL)/√(iλL) · exp(ik(x_R - x_S)²/(2L))

   Com essa normalização, |E|² é densidade de potência e o operador é uma
   contração (potência recebida ≤ potência transmitida). Por isso os
   valores singulares ao quadrado |s_j|² são TRANSMISSIVIDADES físicas
   (0 ≤ |s_j|² ≤ 1): a fração da potência do modo j que chega ao receptor.

Discretização de um operador integral (método de Nyström/quadratura):
    φ(x_Ri) = ∫ h(x_Ri, x_S) ψ(x_S) dx_S ≈ Σ_j h(x_Ri, x_Sj) w_Sj ψ(x_Sj)

Para que a SVD da MATRIZ aproxime a SVD do OPERADOR contínuo, usamos a
matriz simetrizada M_ij = √w_Ri · h(x_Ri, x_Sj) · √w_Sj e os vetores
u_j = √w_Sj · ψ(x_Sj). Assim o produto interno Cartesiano Σ u_j* u_j
aproxima o produto interno contínuo ∫ |ψ|² dx (Miller, Seção 6.10 e Ap. B).
"""

import numpy as np


def wavenumber(wavelength):
    """Número de onda k = 2π/λ."""
    return 2.0 * np.pi / wavelength


def _pairwise_distance(r_R, r_S):
    """Matriz de distâncias |r_Ri - r_Sj| (linhas = receptores, colunas = fontes)."""
    r_R = np.atleast_2d(np.asarray(r_R, dtype=float))
    r_S = np.atleast_2d(np.asarray(r_S, dtype=float))
    diff = r_R[:, None, :] - r_S[None, :, :]
    return np.sqrt(np.sum(diff**2, axis=-1))


def green_scalar_3d(r_R, r_S, k):
    """
    Matriz G_SR com elementos g_ij da Eq. (7) do Miller.

    Parâmetros
    ----------
    r_R : (N_R, 3) posições dos receptores pontuais
    r_S : (N_S, 3) posições das fontes pontuais
    k   : número de onda

    Retorna
    -------
    G : (N_R, N_S) complexa. Linha i = receptor i, coluna j = fonte j,
        de modo que |φ_R⟩ = G |ψ_S⟩ (Eq. 10).
    """
    d = _pairwise_distance(r_R, r_S)
    if np.any(d == 0):
        raise ValueError("Fonte e receptor coincidem: a função de Green diverge.")
    return -np.exp(1j * k * d) / (4.0 * np.pi * d)


def fresnel_kernel_1d(x_R, w_R, x_S, w_S, wavelength, L):
    """
    Matriz do operador de Fresnel 1D entre duas fendas paralelas,
    já com os pesos de quadratura (M_ij = √w_Ri h(x_Ri, x_Sj) √w_Sj).

    Fisicamente: uma dimensão transversal (x) e propagação em z; vale para
    um campo invariante na outra dimensão transversal (fenda longa).
    """
    k = wavenumber(wavelength)
    dx = np.asarray(x_R)[:, None] - np.asarray(x_S)[None, :]
    # fase de Fresnel: expansão paraxial de k·√(L² + Δx²) ≈ kL + kΔx²/(2L)
    h = np.exp(1j * k * L) / np.sqrt(1j * wavelength * L) * np.exp(1j * k * dx**2 / (2 * L))
    return np.sqrt(w_R)[:, None] * h * np.sqrt(w_S)[None, :]


def fresnel_kernel_2d(r_R, w_R, r_S, w_S, wavelength, L):
    """
    Matriz do operador de Fresnel 2D entre duas aberturas planas paralelas,
    com pesos de quadratura (M_ij = √w_Ri h(r_Ri, r_Sj) √w_Sj).

    r_R, r_S : (N, 2) coordenadas transversais (x, y) nos planos.
    w_R, w_S : (N,) elementos de área de cada ponto.
    """
    k = wavenumber(wavelength)
    r_R = np.asarray(r_R)
    r_S = np.asarray(r_S)
    d2 = (r_R[:, 0, None] - r_S[None, :, 0]) ** 2 + (r_R[:, 1, None] - r_S[None, :, 1]) ** 2
    h = np.exp(1j * k * L) / (1j * wavelength * L) * np.exp(1j * k * d2 / (2 * L))
    return np.sqrt(w_R)[:, None] * h * np.sqrt(w_S)[None, :]


def field_from_point_sources(r_eval, r_S, amplitudes, k):
    """
    Onda gerada em pontos arbitrários por um conjunto de fontes pontuais
    com amplitudes complexas h_q (Miller, Eq. 53):

        φ(r) = -(1/4π) Σ_q exp(ik|r - r_Sq|)/|r - r_Sq| · h_q

    É só a função de Green aplicada a pontos de avaliação quaisquer.
    """
    return green_scalar_3d(r_eval, r_S, k) @ np.asarray(amplitudes)
