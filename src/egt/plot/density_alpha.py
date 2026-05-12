"""Density-scaled per-point alpha for x-clustered scatter plots.

When divergence-time data clusters at discrete TimeTree splits (e.g. 691 Mya
with thousands of species, adjacent 631 Mya with hundreds), a single global
alpha makes dense clusters saturate to solid blocks while sparse regions
nearly vanish. This module computes a per-point alpha that scales inversely
with local x-density so all density bands carry comparable visual ink.

Algorithm
---------
For each point at x_i, count the number of points within ±bin_width/2.
Per-point alpha scales as 1/sqrt(count) so doubling the local density only
shrinks each point's alpha by sqrt(2). Clamp to [alpha_min, alpha_max] so
the sparsest points stay visible and the densest don't disappear entirely.

Usage
-----
    >>> from egt.plot import density_alpha, rgba_array
    >>> alphas = density_alpha(x, bin_width=40, base_alpha=0.40)
    >>> rgba   = rgba_array("#0072B2", alphas)
    >>> ax.scatter(x, y, c=rgba, s=2.0, linewidths=0, rasterized=True)
"""
from __future__ import annotations

import numpy as np
import matplotlib.colors as mcolors

__all__ = ["density_alpha", "rgba_array"]


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
        Alpha applied to a point in a bin of reference density. Tune to
        taste.
    alpha_min, alpha_max : float
        Clamp range. alpha_min keeps sparse points visible; alpha_max
        keeps the densest bins from saturating.
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
    alpha = np.clip(base_alpha * scale, alpha_min, alpha_max)
    return alpha


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
