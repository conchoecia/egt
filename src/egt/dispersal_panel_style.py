"""Shared helpers for the dispersal-figure panels (Science Advances Fig 2).

Combines the density-scaled per-point alpha algorithm with the SA-compliant
matplotlib rcParams + palettes + holozoan callout constants. Consumed by
``plot_decay_pairwise_steps.plot_pairwise_decay_sp1_vs_all`` and any other
dispersal-figure plotter that wants the same visual contract.

Public API
----------
``density_alpha(x, ...)``
    Per-point alpha inverse to local x-density. Counters saturation at
    clustered TimeTree divergence ages.
``rgba_array(color, alphas)``
    Build an N×4 RGBA array from one base color + per-point alphas.
``apply_rc()``
    Mutate :data:`matplotlib.rcParams` to the Science Advances spec
    (Helvetica, ``pdf.fonttype=42``, 0.4 pt axes, no minor ticks).
``style_axes(ax)``
    Drop top/right spines and set remaining spine widths.
``OKABE_ITO`` / ``ANIMAL_COLOR``
    Colorblind-safe palette + the default animal-target color.
``HOLOZOAN_COLORS`` / ``HOLOZOAN_LABELS`` / ``HOLOZOAN_SPECIES``
    Manuscript-specific outgroup callout palette and labels.
``ALG_ORDER`` / ``alg_colors()``
    BCnS ALG ordering + 29 viridis colors aligned to that order.
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


__all__ = [
    "density_alpha",
    "rgba_array",
    "apply_rc",
    "style_axes",
    "alg_colors",
    "OKABE_ITO",
    "ANIMAL_COLOR",
    "HOLOZOAN_COLORS",
    "HOLOZOAN_LABELS",
    "HOLOZOAN_SPECIES",
    "ALG_ORDER",
]


# --------------------------------------------------------------------- #
# Palette + manuscript-specific constants
# --------------------------------------------------------------------- #
OKABE_ITO = {
    "black":      "#000000",
    "orange":     "#E69F00",
    "skyblue":    "#56B4E9",
    "green":      "#009E73",
    "yellow":     "#F0E442",
    "blue":       "#0072B2",
    "vermillion": "#D55E00",
    "purple":     "#CC79A7",
}

ANIMAL_COLOR = OKABE_ITO["blue"]

# Holozoan callout palette (manuscript MainFig 2 spec).
HOLOZOAN_COLORS = {
    "Salpingoecarosetta-946362-GCA033442325.1":      "#777777",
    "Capsasporaowczarzaki-192875-GCA033442345.1":    OKABE_ITO["green"],
    "Creolimaxfragrantissima-196028-GCA033442365.1": OKABE_ITO["purple"],
}

HOLOZOAN_LABELS = {
    "Salpingoecarosetta-946362-GCA033442325.1":      "Salpingoeca",
    "Capsasporaowczarzaki-192875-GCA033442345.1":    "Capsaspora",
    "Creolimaxfragrantissima-196028-GCA033442365.1": "Creolimax",
}

HOLOZOAN_SPECIES = list(HOLOZOAN_COLORS.keys())

# BCnS ALG names in canonical x-axis ordering (smallest → largest).
ALG_ORDER = [
    "Qb", "Qc", "C2", "Qd", "R",  "Qa", "A2", "B3", "O2", "Eb",
    "A1b","J1", "O1", "J2", "P",  "B2", "I",  "B1", "M",  "L",
    "N",  "Ea", "K",  "H",  "G",  "C1", "F",  "D",  "A1a",
]


# --------------------------------------------------------------------- #
# Density-scaled per-point alpha
# --------------------------------------------------------------------- #
def density_alpha(
    x: np.ndarray,
    *,
    bin_width: float = 40.0,
    base_alpha: float = 0.40,
    alpha_min: float = 0.04,
    alpha_max: float = 0.60,
    reference: str = "median",
) -> np.ndarray:
    """Per-point alpha inverse to local x-density.

    For each x_i, count points within ±bin_width/2; alpha scales as
    ``1/sqrt(count)`` then is clamped to ``[alpha_min, alpha_max]``.
    Doubling local density only shrinks alpha by ``sqrt(2)``.

    Parameters
    ----------
    x : array
        x-coordinates of the points (raw, pre-jitter).
    bin_width : float
        Window (same units as x) used to estimate local density. For
        divergence-time data with ±20 Mya jitter, 40 Mya is a natural
        choice — points cluster together iff their pre-jitter Mya is
        identical.
    base_alpha : float
        Alpha applied to a point in a bin of reference density.
    alpha_min, alpha_max : float
        Clamp range. ``alpha_min`` keeps sparse points visible;
        ``alpha_max`` keeps the densest bins from saturating.
    reference : {"median", "min"}
        Density used as the "typical" baseline. ``"median"`` gives a
        balanced rescaling; ``"min"`` preserves the look of the sparsest
        bin and only attenuates denser ones.

    Returns
    -------
    np.ndarray
        Per-point alpha values, shape ``== x.shape``.
    """
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return np.empty(0, dtype=float)

    sorted_x = np.sort(x)
    half = bin_width / 2.0
    lo = np.searchsorted(sorted_x, x - half, side="left")
    hi = np.searchsorted(sorted_x, x + half, side="right")
    counts = (hi - lo).astype(float)
    counts = np.clip(counts, 1.0, None)

    if reference == "median":
        ref = float(np.median(counts))
    elif reference == "min":
        ref = float(counts.min())
    else:
        raise ValueError(f"unknown reference: {reference!r}")

    scale = np.sqrt(ref / counts)
    return np.clip(base_alpha * scale, alpha_min, alpha_max)


def rgba_array(color, alphas) -> np.ndarray:
    """Build an N×4 RGBA array from one base color + per-point alphas.

    Use with ``ax.scatter(c=rgba, alpha=None, ...)`` — matplotlib reads
    the alpha channel of ``c`` only when the scalar ``alpha`` argument
    is None.
    """
    r, g, b = mcolors.to_rgb(color)
    alphas = np.asarray(alphas, dtype=float)
    out = np.empty((alphas.size, 4), dtype=float)
    out[:, 0] = r
    out[:, 1] = g
    out[:, 2] = b
    out[:, 3] = alphas
    return out


# --------------------------------------------------------------------- #
# Science Advances styling
# --------------------------------------------------------------------- #
def apply_rc() -> None:
    """Mutate :data:`matplotlib.rcParams` to the Science Advances spec.

    Idempotent. Call once at the start of a figure pipeline before any
    ``plt.subplots`` call so the new rcParams take effect on the figure
    being built. SA contract:

    * Helvetica with Arial / DejaVu Sans fallback.
    * ``pdf.fonttype = 42`` (TrueType) + ``svg.fonttype = "none"`` so
      all text remains editable in Illustrator.
    * Min line weight 0.4 pt (SA hard minimum 0.28 pt).
    * Ticks 6 pt, body 7 pt, panel letters 9 pt bold (panel letters
      drawn at the call site, not via rcParams).
    * No minor ticks, no gridlines, no top/right spines.
    """
    mpl.rcParams.update({
        "pdf.fonttype":      42,
        "ps.fonttype":       42,
        "svg.fonttype":      "none",
        "font.family":       "sans-serif",
        "font.sans-serif":   ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size":         7,
        "axes.titlesize":    7,
        "axes.labelsize":    7,
        "xtick.labelsize":   6,
        "ytick.labelsize":   6,
        "legend.fontsize":   6,
        "axes.linewidth":    0.4,
        "xtick.major.width": 0.4,
        "ytick.major.width": 0.4,
        "xtick.minor.width": 0.0,
        "ytick.minor.width": 0.0,
        "xtick.major.size":  2.2,
        "ytick.major.size":  2.2,
        "xtick.minor.size":  0.0,
        "ytick.minor.size":  0.0,
        "xtick.direction":   "out",
        "ytick.direction":   "out",
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.labelpad":     2,
        "legend.frameon":    False,
        "lines.linewidth":   0.6,
        "patch.linewidth":   0.4,
    })


def style_axes(ax) -> None:
    """Drop top/right spines + set remaining spine widths. Idempotent."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)


def alg_colors():
    """29 viridis colors lining up positionally with :data:`ALG_ORDER`."""
    cmap = plt.get_cmap("viridis")
    return [cmap(i) for i in np.linspace(0.05, 0.95, len(ALG_ORDER))]
