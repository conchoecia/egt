"""Science Advances–compliant matplotlib styling + shared palettes.

Constants
---------
OKABE_ITO         — colorblind-safe 8-color palette.
ANIMAL_COLOR      — default scatter color for animal-target points.
HOLOZOAN_SPECIES  — list of full species IDs treated as outgroup callouts.
HOLOZOAN_COLORS   — species-ID → hex color (callout markers).
HOLOZOAN_LABELS   — species-ID → short genus label for legends.
ALG_ORDER         — BCnS ALG names in their canonical x-axis ordering.

Functions
---------
apply_rc()        — mutates :data:`matplotlib.rcParams` to SA-compliant
                    typography + line weights. Call once per figure
                    pipeline before any ``plt.subplots`` call.
style_axes(ax)    — drops top/right spines, sets remaining spine widths.
alg_colors()      — 29 colors (viridis) lining up with ``ALG_ORDER``.

The Science Advances figure spec implemented here:

* Helvetica with Arial / DejaVu Sans fallback.
* ``pdf.fonttype = 42`` (TrueType), ``svg.fonttype = "none"`` so all
  text remains editable in Illustrator.
* Min line weight 0.4 pt (SA's hard minimum is 0.28 pt).
* Ticks 6 pt, body 7 pt, panel letters 9 pt bold (drawn at the call
  site, not via rcParams).
* No minor ticks, no gridlines, no top/right spines.
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


# 8-color Okabe-Ito colorblind-safe palette.
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

# BCnS ALG names in their canonical x-axis ordering (smallest → largest).
ALG_ORDER = [
    "Qb", "Qc", "C2", "Qd", "R",  "Qa", "A2", "B3", "O2", "Eb",
    "A1b","J1", "O1", "J2", "P",  "B2", "I",  "B1", "M",  "L",
    "N",  "Ea", "K",  "H",  "G",  "C1", "F",  "D",  "A1a",
]


def apply_rc() -> None:
    """Mutate :data:`matplotlib.rcParams` to the Science Advances spec.

    Idempotent; safe to call multiple times. Call once at the start of
    a figure pipeline before any ``plt.subplots`` call so the new
    rcParams take effect on the figure being built.
    """
    mpl.rcParams.update({
        "pdf.fonttype":     42,
        "ps.fonttype":      42,
        "svg.fonttype":     "none",
        "font.family":      "sans-serif",
        "font.sans-serif":  ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size":        7,
        "axes.titlesize":   7,
        "axes.labelsize":   7,
        "xtick.labelsize":  6,
        "ytick.labelsize":  6,
        "legend.fontsize":  6,
        "axes.linewidth":   0.4,
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
    """Drop top/right spines + set remaining spine widths.

    Applied per-axes after rcParams have created the figure. Idempotent.
    """
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)


def alg_colors():
    """29 viridis colors lining up positionally with :data:`ALG_ORDER`."""
    cmap = plt.get_cmap("viridis")
    return [cmap(i) for i in np.linspace(0.05, 0.95, len(ALG_ORDER))]
