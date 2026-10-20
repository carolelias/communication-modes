"""
Limite paraxial: contagem de modos, funções esferoidais prolatas e curvatura confocal.

Resultado central (fendas 1D de larguras Δx_S e Δx_R, separadas por L, Fresnel):

  Com o núcleo h(x_R, x_S) ∝ exp(ik(x_R - x_S)²/(2L)), expandindo o quadrado:
      (x_R - x_S)² = x_R² - 2 x_R x_S + x_S²
  os termos x_S² e x_R² são só fases quadráticas (lentes!) em cada abertura.
  Removendo-as, sobra exp(-ik x_R x_S / L): uma TRANSFORMADA DE FOURIER
  truncada nas duas aberturas. Daí:

      G†G = Φ K Φ†,  Φ = diag(exp(-ik x_S²/(2L))),
      K(x, x') = sin(Ω (x - x')) / (π (x - x')),  Ω = k Δx_R / (2L)

  K é o núcleo do problema de concentração de Slepian (1961). Mudando para
  u = x/(Δx_S/2) ∈ [-1, 1], o parâmetro é

      c = k (Δx_S/2)(Δx_R/2) / L

  e os autovalores λ_n(c) (= |s_n|²) valem ≈ 1 até n ≈ 2c/π e despencam
  depois disso. As autofunções são as funções esferoidais prolatas
  angulares S_0n(c, u), multiplicadas pela fase quadrática Φ.

  Número de graus de liberdade (heurística de Gabor/Miller):
      1D:  N ≈ 2c/π = Δx_S Δx_R / (λ L)
      2D:  N ≈ A_S A_R / (λ L)²

  E a regra da soma do operador de Fresnel normalizado é EXATAMENTE
      S = ∫∫ |h|² = Δx_S Δx_R / (λL)  (1D)   ou   A_S A_R / (λL)²  (2D),
  ou seja, no limite paraxial a regra da soma é o próprio número de Fresnel
  "generalizado" — com S modos de transmissividade ~1, a soma fecha.
"""

import numpy as np
from scipy.special import pro_ang1

from .green import wavenumber


def slepian_c(width_S, width_R, wavelength, L):
    """Parâmetro de largura de banda c = k (Δx_S/2)(Δx_R/2) / L."""
    return wavenumber(wavelength) * (width_S / 2) * (width_R / 2) / L


def heuristic_number_1d(width_S, width_R, wavelength, L):
    """N ≈ Δx_S Δx_R / (λL) = 2c/π (número de modos bem acoplados em 1D)."""
    return width_S * width_R / (wavelength * L)


def heuristic_number_2d(area_S, area_R, wavelength, L):
    """N ≈ A_S A_R / (λL)² (número de modos bem acoplados em 2D)."""
    return area_S * area_R / (wavelength * L) ** 2


def slepian_eigenvalues(c, n_quad=200):
    """
    Autovalores λ_n(c) do núcleo sin(c(u-u'))/(π(u-u')) em [-1, 1],
    calculados por Nyström com Gauss-Legendre (referência independente do
    núcleo de Fresnel: aqui não há fase quadrática nem L, só c).

    Matriz simetrizada A_ij = √w_i K(u_i, u_j) √w_j → autovalores reais.
    """
    u, w = np.polynomial.legendre.leggauss(n_quad)
    du = u[:, None] - u[None, :]
    with np.errstate(invalid="ignore", divide="ignore"):
        K = np.sin(c * du) / (np.pi * du)
    K[np.isclose(du, 0.0)] = c / np.pi  # limite sin(c x)/(π x) → c/π
    A = np.sqrt(w)[:, None] * K * np.sqrt(w)[None, :]
    lam = np.linalg.eigvalsh(A)[::-1]
    return np.clip(lam, 0.0, None)


def prolate_angular(n, c, u):
    """
    Função esferoidal prolata angular S_0n(c, u) (scipy.special.pro_ang1),
    normalizada a ∫_{-1}^{1} S² du = 1 com o sinal ajustado para S(0) > 0
    (n par) ou S'(0) > 0 (n ímpar), só para comparação de formas.
    """
    u = np.asarray(u, dtype=float)
    S = pro_ang1(0, n, c, u)[0]
    ug, wg = np.polynomial.legendre.leggauss(200)
    Sg = pro_ang1(0, n, c, ug)[0]
    S = S / np.sqrt(np.sum(wg * Sg**2))
    s0, ds0 = pro_ang1(0, n, c, 0.0)
    ref = s0 if n % 2 == 0 else ds0
    return S * np.sign(ref)


def confocal_phase(x, wavelength, L, side):
    """
    Fase confocal (Miller, Eqs. 54 e 55), versão exata (não paraxial):

        fonte:     c_S(x) = exp(-ik(√(x² + L²) - L))
        receptor:  c_R(x) = exp(+ik(√(x² + L²) - L))

    Multiplicar um modo por conj(c) remove a curvatura "confocal"
    (cada superfície curvada em torno do centro da outra) e deixa só a
    forma de amplitude, que no caso paraxial é a esferoidal prolata.
    No limite paraxial, √(x² + L²) - L ≈ x²/(2L): é a fase Φ acima.
    """
    k = wavenumber(wavelength)
    x = np.asarray(x, dtype=float)
    sign = -1.0 if side == "source" else +1.0
    return np.exp(sign * 1j * k * (np.sqrt(x**2 + L**2) - L))
