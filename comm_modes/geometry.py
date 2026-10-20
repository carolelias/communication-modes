"""
Geometrias de fontes e receptores.

- Pontos (fontes/receptores pontuais) para a função de Green 3D exata.
- Grades com pesos de quadratura para aberturas contínuas (núcleo de Fresnel).

Convenção: propagação ao longo de z; y é a direção "vertical" das figuras
do Miller; todas as distâncias na unidade que o usuário escolher
(nos exemplos do Miller, em comprimentos de onda, λ = 1).
"""

import numpy as np


def line_points(n, length, z=0.0, x=0.0, center_y=0.0):
    """
    n pontos igualmente espaçados ao longo de y, cobrindo um comprimento total
    'length' (do primeiro ao último ponto), no plano z dado.

    Ex.: Miller, Fig. 3: line_points(3, 4.0) → y = -2, 0, 2 (espaçamento 2λ).
    Retorna array (n, 3) com colunas (x, y, z).
    """
    y = center_y + (np.linspace(-length / 2, length / 2, n) if n > 1 else np.array([0.0]))
    return np.column_stack([np.full(n, x), y, np.full(n, z)])


def double_line_points(n, length, z=0.0, dz=0.25):
    """
    Duas linhas paralelas de fontes separadas por dz em z (Miller usa λ/4).
    A SVD "descobre" sozinha que as duas colunas devem formar um dipolo
    espaço-temporal que cancela a radiação para trás (Miller, Fig. 7).
    """
    return np.vstack([line_points(n, length, z=z - dz), line_points(n, length, z=z)])


def plane_grid_points(n, size, z=0.0):
    """
    Grade quadrada n × n no plano z, lado total 'size' (Miller, Fig. 8:
    17 × 17 pontos com espaçamento λ → size = 16λ). Retorna (n², 3).
    """
    c = np.linspace(-size / 2, size / 2, n)
    X, Y = np.meshgrid(c, c, indexing="xy")
    return np.column_stack([X.ravel(), Y.ravel(), np.full(n * n, z)])


def slit_quadrature(width, n):
    """
    Pontos e pesos de Gauss-Legendre numa fenda [-width/2, width/2].

    Gauss-Legendre integra polinômios de grau ≤ 2n-1 exatamente, então
    converge muito rápido para funções suaves (como os modos bem acoplados).
    """
    u, wu = np.polynomial.legendre.leggauss(n)
    return u * width / 2, wu * width / 2


def circular_aperture_quadrature(diameter, n_r, n_theta):
    """
    Quadratura polar para um disco de diâmetro D:

        ∫∫ f dA = ∫_0^a ∫_0^{2π} f(r, θ) r dθ dr,   a = D/2

    - em r: Gauss-Legendre em [0, a] (o fator r do jacobiano entra no peso);
    - em θ: regra do trapézio com n_theta pontos, que para funções periódicas
      converge exponencialmente (é exata para harmônicos e^{imθ} com |m| < n_theta).

    Por que não uma grade cartesiana recortada? A borda "em escada" dá erro
    ~1/n na área e nas integrais e chega a quebrar a propriedade de contração
    (dava |s_1|² > 1). A quadratura polar não tem esse problema: a área
    Σ w = πa² sai exata.

    Retorna (pontos (n_r·n_theta, 2), pesos).
    """
    a = diameter / 2
    u, wu = np.polynomial.legendre.leggauss(n_r)
    r = (u + 1) * a / 2  # mapeia [-1, 1] → [0, a]
    wr = wu * a / 2 * r  # peso radial com o jacobiano r
    th = 2 * np.pi * np.arange(n_theta) / n_theta
    R, TH = np.meshgrid(r, th, indexing="ij")
    W = wr[:, None] * np.full(n_theta, 2 * np.pi / n_theta)[None, :]
    pts = np.column_stack([(R * np.cos(TH)).ravel(), (R * np.sin(TH)).ravel()])
    return pts, W.ravel()
