"""
Modos de comunicação = decomposição em valores singulares (SVD) de G_SR.

Resumo da teoria (Miller 2019, Seções 3.7 e 3.8):

  G_SR leva vetores do espaço da fonte H_S ao espaço do receptor H_R, então
  não tem autovetores com significado físico (não mapeia um espaço nele mesmo).
  Os operadores G†G (H_S → H_S) e GG† (H_R → H_R) são hermitianos e positivos:
    - autovalores reais e ≥ 0, que escrevemos como |s_j|²;
    - autovetores ortonormais e completos;
    - propriedade de maximização: |ψ_S1⟩ é a fonte normalizada que produz o
      maior ⟨φ|φ⟩ no receptor; |ψ_S2⟩ é a melhor entre as ortogonais a |ψ_S1⟩, etc.

      G†G |ψ_Sj⟩ = |s_j|² |ψ_Sj⟩      (Eq. 29)
      GG† |φ_Rj⟩ = |s_j|² |φ_Rj⟩      (Eq. 30)
      G |ψ_Sj⟩   = s_j |φ_Rj⟩         (Eq. 31)  ← os modos vêm em PARES
      G = Σ_j s_j |φ_Rj⟩⟨ψ_Sj|        (Eq. 33)  ← a SVD propriamente dita

  Regra da soma (Eq. 36): S = Σ_j |s_j|² = Σ_ij |g_ij|² = traço(G†G)
  (norma de Frobenius ao quadrado). Dá para calcular S sem resolver o
  autoproblema, e S limita quantos canais podem ter uma dada força de acoplamento.

Numericamente não resolvemos os dois autoproblemas: a rotina SVD do LAPACK
(numpy.linalg.svd) devolve G = V · diag(s) · U† diretamente (Eq. 35), mais
estável do que diagonalizar G†G (que eleva ao quadrado o número de condição).
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CommunicationModes:
    """
    Resultado da SVD de G_SR, ordenado por |s_j| decrescente.

    s   : (N_m,) valores singulares (reais, ≥ 0; a SVD escolhe a fase de
          forma que s_j seja real e jogue a fase para os vetores)
    psi : (N_S, N_m) coluna j = |ψ_Sj⟩, modo na fonte (ortonormal)
    phi : (N_R, N_m) coluna j = |φ_Rj⟩, modo no receptor (ortonormal)
    """

    s: np.ndarray
    psi: np.ndarray
    phi: np.ndarray

    @property
    def coupling_strengths(self):
        """Forças de acoplamento de potência |s_j|² (autovalores de G†G)."""
        return self.s**2

    @property
    def sum_rule(self):
        """S = Σ |s_j|² (igual a Σ|g_ij|², ver sum_rule())."""
        return float(np.sum(self.s**2))

    def fraction_of_sum_rule(self):
        """|s_j|²/S: fração da regra da soma em cada modo (Tabela 1 do Miller)."""
        return self.coupling_strengths / self.sum_rule


def communication_modes(G):
    """
    Calcula os modos de comunicação (pares de fonte e receptor) de G_SR.

    np.linalg.svd devolve G = V diag(s) Uh, com:
      colunas de V  = |φ_Rj⟩  (vetores singulares à esquerda)
      linhas de Uh  = ⟨ψ_Sj|  (logo, U = Uh† tem colunas |ψ_Sj⟩)
    full_matrices=False: guardamos só N_m = min(N_S, N_R) pares.
    """
    V, s, Uh = np.linalg.svd(np.asarray(G), full_matrices=False)
    return CommunicationModes(s=s, psi=Uh.conj().T, phi=V)


def sum_rule(G):
    """Regra da soma S = Σ_ij |g_ij|² (Eq. 36), sem resolver a SVD."""
    return float(np.sum(np.abs(G) ** 2))


def count_modes_above(strengths, threshold):
    """
    Número de canais com |s_j|² ≥ threshold (Eq. 38). Com threshold relativo,
    passe strengths/strengths[0]; com transmissividades físicas, passe |s_j|².
    """
    return int(np.sum(np.asarray(strengths) >= threshold))


def fix_phase_at_center(vec):
    """
    Remove a fase global arbitrária de um autovetor: multiplica por uma
    constante de módulo 1 para que o elemento central seja real e positivo
    (Miller escolhe fase zero no meio do modo, Fig. 6). Se o elemento central
    for ~0 (modos ímpares), usa o elemento de maior módulo.

    Autovetores só são definidos a menos de uma fase global: se G†Gψ = |s|²ψ,
    então e^{iθ}ψ também é autovetor. Fixar a fase só serve para comparar e plotar.
    """
    vec = np.asarray(vec)
    mid = vec[len(vec) // 2]
    ref = mid if abs(mid) > 1e-6 * np.max(np.abs(vec)) else vec[np.argmax(np.abs(vec))]
    return vec * np.exp(-1j * np.angle(ref))
