"""Shared plotting utilities for egt figures.

Modules:
    density_alpha — per-point alpha scaled by local x-density (counters
                    saturation at clustered TimeTree divergence ages).
    style         — Science Advances–compliant rcParams, the Okabe-Ito
                    palette, and holozoan callout colors/labels.
"""
from egt.plot.density_alpha import density_alpha, rgba_array
from egt.plot.style import (
    ANIMAL_COLOR,
    HOLOZOAN_COLORS,
    HOLOZOAN_LABELS,
    HOLOZOAN_SPECIES,
    OKABE_ITO,
    apply_rc,
    style_axes,
)

__all__ = [
    "density_alpha",
    "rgba_array",
    "apply_rc",
    "style_axes",
    "ANIMAL_COLOR",
    "HOLOZOAN_COLORS",
    "HOLOZOAN_LABELS",
    "HOLOZOAN_SPECIES",
    "OKABE_ITO",
]
