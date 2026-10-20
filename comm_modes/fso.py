"""
Aplicação: enlace óptico em espaço livre (FSO) para QKD entre duas aberturas circulares.

Modelo: aberturas circulares coaxiais de diâmetros D_T (transmissor) e D_R
(receptor), separadas por L, propagação escalar de Fresnel no vácuo.
Ignoramos turbulência atmosférica, absorção, erro de apontamento e
polarização (que só duplicaria os canais). É o limite de DIFRAÇÃO puro.

Cada modo de comunicação j é um canal de perda pura independente com
transmissividade

    η_j = |s_j|²   (0 ≤ η_j ≤ 1, porque o núcleo de Fresnel normalizado é contração)

Consequências para QKD:

1. Perda geométrica mínima: η_1 é a MAIOR fração de potência que qualquer
   campo de transmissão (restrito à abertura D_T) consegue colocar dentro de
   D_R (propriedade de maximização da SVD). Nenhum feixe — gaussiano incluso —
   faz melhor. No campo distante (S ≪ 1), η_1 → S = A_T A_R/(λL)², a
   conhecida lei de perda geométrica ∝ 1/L².

2. Limite PLOB (Pirandola-Laurenza-Ottaviani-Banchi 2017): a capacidade de
   distribuição de chave secreta de um canal de perda pura com
   transmissividade η é -log2(1 - η) bits por uso do modo. Como os modos de
   comunicação são canais ORTOGONAIS independentes, o enlace multimodo
   (multiplexação espacial) admite no máximo

       K_total = Σ_j -log2(1 - η_j)

   bits por uso (por modo temporal e por polarização). É o mesmo limite que
   aparece no projeto bb84-decoy-fiber, agora aplicado modo a modo.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from .geometry import circular_aperture_quadrature
from .green import fresnel_kernel_2d, wavenumber
from .modes import communication_modes, sum_rule


@dataclass(frozen=True)
class FSOLink:
    """Parâmetros do enlace (SI). Padrão: 1550 nm, telescópios de 10 cm."""

    wavelength: float = 1550e-9
    D_T: float = 0.10
    D_R: float = 0.10
    L: float = 10e3

    @property
    def area_T(self):
        return np.pi * self.D_T**2 / 4

    @property
    def area_R(self):
        return np.pi * self.D_R**2 / 4

    @property
    def fresnel_product(self):
        """S = A_T A_R / (λL)²: regra da soma do operador contínuo (≈ nº de modos)."""
        return self.area_T * self.area_R / (self.wavelength * self.L) ** 2


def quadrature_size(link, extra=1.0):
    """
    Número de pontos (n_r, n_theta) da quadratura polar.

    Separando o núcleo de Fresnel, o termo que acopla as aberturas é
    exp(-ik r_R·r_S/L). Em coordenadas polares ele se expande em harmônicos
    e^{imθ} com |m| até ~ c = k a_T a_R / L (expansão de Jacobi-Anger), e na
    direção radial oscila com até ~c/π meios-períodos. A fase quadrática
    k r²/(2L) adiciona no máximo c/2 rad de variação radial.
    Então n_theta ≳ 2c e n_r ≳ c bastam; somamos margens fixas para
    resolver também os modos mais fracos que entram na regra da soma.
    'extra' multiplica tudo (use > 1 para testar convergência).
    """
    k = wavenumber(link.wavelength)
    c = k * (link.D_T / 2) * (link.D_R / 2) / link.L
    n_r = int(np.ceil(extra * (c + 16)))
    n_theta = int(np.ceil(extra * (2 * c + 32)))
    return n_r, n_theta


def link_operator(link, quadrature=None):
    """
    Monta a matriz (com pesos) do operador de Fresnel entre as duas aberturas.
    quadrature = (n_r, n_theta); se None, usa quadrature_size(link).
    """
    n_r, n_theta = quadrature or quadrature_size(link)
    rS, wS = circular_aperture_quadrature(link.D_T, n_r, n_theta)
    rR, wR = circular_aperture_quadrature(link.D_R, n_r, n_theta)
    M = fresnel_kernel_2d(rR, wR, rS, wS, link.wavelength, link.L)
    return M, (rS, wS), (rR, wR)


def link_modes(link, quadrature=None):
    """
    Modos de comunicação do enlace e suas transmissividades η_j = |s_j|².
    Retorna (CommunicationModes, η, regra da soma numérica).
    """
    M, src, rcv = link_operator(link, quadrature)
    modes = communication_modes(M)
    return modes, modes.s**2, sum_rule(M), src, rcv


def gaussian_efficiency(link, w_T, focus, quadrature=None, operator=None):
    """
    Eficiência de um feixe gaussiano lançado pela abertura do transmissor,
    avaliada com o MESMO operador de Fresnel usado para os modos.

    Campo no plano do transmissor:
        E(r) = exp(-r²/w_T²) · exp(-ik r²/(2f)),   1/f = focus / L

    focus = 0 → feixe colimado (cintura no transmissor);
    focus = 1 → focalizado no receptor; valores intermediários colocam a
    cintura entre os dois. (Convenção e^{ikz}: fase -ik r²/(2f) é convergente.)

    η_G = (potência que chega na abertura D_R) / (potência TOTAL do gaussiano),
    então a parte cortada pela abertura do transmissor também conta como perda.
    Potência total de |E|² = exp(-2r²/w_T²) no plano: π w_T²/2.

    Como o gaussiano truncado é só UM campo possível na abertura, η_G ≤ η_1
    sempre (propriedade de maximização); a diferença mede o ganho de usar o
    modo ótimo da SVD em vez de um feixe gaussiano otimizado.
    """
    if operator is None:
        M, (rS, wS), _ = link_operator(link, quadrature)
    else:
        M, (rS, wS) = operator
    k = wavenumber(link.wavelength)
    r2 = np.sum(rS**2, axis=1)
    E = np.exp(-r2 / w_T**2) * np.exp(-1j * k * r2 * focus / (2 * link.L))
    u = np.sqrt(wS) * E  # vetor discretizado com pesos de quadratura
    p_out = np.sum(np.abs(M @ u) ** 2)
    p_total = np.pi * w_T**2 / 2
    return float(p_out / p_total)


def best_gaussian_efficiency(link, quadrature=None):
    """
    Otimiza (w_T, focus) para maximizar η_G: busca em grade grossa seguida de
    refinamento com L-BFGS-B. Retorna (η_G, w_T, focus).
    """
    M, src, _ = link_operator(link, quadrature)
    op = (M, src)
    D = link.D_T

    def cost(p):
        return -gaussian_efficiency(link, p[0] * D, p[1], operator=op)

    # grade grossa: w_T/D em [0.05, 2], focus em [0, 1.5]
    grid = [(a, b) for a in np.linspace(0.05, 2.0, 25) for b in np.linspace(0.0, 1.5, 16)]
    p0 = min(grid, key=cost)
    res = minimize(cost, p0, method="L-BFGS-B", bounds=[(0.02, 3.0), (-0.5, 2.0)])
    return -res.fun, res.x[0] * D, res.x[1]


def plob_bound(eta):
    """Limite PLOB por modo: -log2(1 - η) bits por uso do canal."""
    eta = np.asarray(eta, dtype=float)
    return -np.log2(1.0 - eta)


def plob_multimode(etas):
    """Limite PLOB do enlace multimodo: Σ_j -log2(1 - η_j)."""
    return float(np.sum(plob_bound(etas)))
