"""
comm_modes — modos de comunicação (SVD do operador de Green) entre fontes e receptores.

Baseado em D. A. B. Miller, "Waves, modes, communications, and optics: a tutorial",
Advances in Optics and Photonics 11(3), 679 (2019), doi:10.1364/AOP.11.000679.
"""

from .geometry import (
    circular_aperture_quadrature,
    double_line_points,
    line_points,
    plane_grid_points,
    slit_quadrature,
)
from .green import (
    field_from_point_sources,
    fresnel_kernel_1d,
    fresnel_kernel_2d,
    green_scalar_3d,
    wavenumber,
)
from .modes import (
    CommunicationModes,
    communication_modes,
    count_modes_above,
    fix_phase_at_center,
    sum_rule,
)

__all__ = [
    "CommunicationModes",
    "circular_aperture_quadrature",
    "communication_modes",
    "count_modes_above",
    "double_line_points",
    "field_from_point_sources",
    "fix_phase_at_center",
    "fresnel_kernel_1d",
    "fresnel_kernel_2d",
    "green_scalar_3d",
    "line_points",
    "plane_grid_points",
    "slit_quadrature",
    "sum_rule",
    "wavenumber",
]
