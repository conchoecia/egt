"""Statistical primitives for GO enrichment.

Kept independent of pandas so test fixtures stay trivial.

- `log_binom`: log C(n, k) via lgamma.
- `hypergeom_sf`: exact one-tailed upper hypergeometric P(X >= k), log-space.
- `binom_sf`: one-tailed upper binomial P(X >= k), via scipy.
- `bh_qvalues`: Benjamini & Hochberg (1995) step-up FDR on a p-value vector.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from scipy.stats import binom as _scipy_binom


def log_binom(n: int, k: int) -> float:
    """log C(n, k); returns -inf if k is out of range."""
    if k < 0 or k > n:
        return float("-inf")
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def hypergeom_sf(k: int, N_total: int, K: int, n: int) -> float:
    """P(X >= k) where X ~ Hypergeometric(N_total, K, n).

    Exact log-space sum over the upper tail. Returns 1.0 for k <= 0 and
    0.0 for k above the support.
    """
    if k <= 0:
        return 1.0
    if k > min(K, n):
        return 0.0
    denom = log_binom(N_total, n)
    logs = [log_binom(K, i) + log_binom(N_total - K, n - i) - denom
            for i in range(k, min(K, n) + 1)]
    m = max(logs)
    return math.exp(m) * sum(math.exp(x - m) for x in logs)


def binom_sf(k: int, n: int, p: float) -> float:
    """P(X >= k) where X ~ Binomial(n, p).

    Edge cases without calling scipy:
      p == 0 → 1.0 if k <= 0 else 0.0
      p >= 1 → 1.0 if k <= n else 0.0
    """
    if p <= 0.0:
        return 1.0 if k <= 0 else 0.0
    if p >= 1.0:
        return 1.0 if k <= n else 0.0
    return float(_scipy_binom.sf(k - 1, n, p))


def bh_qvalues(pvals: Sequence[float]) -> np.ndarray:
    """Benjamini-Hochberg step-up FDR.

    Returns a float ndarray of q-values aligned with the input order.
    Empty input returns an empty float array (never passes through integer
    dtype).
    """
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    q_sorted = np.minimum.accumulate(ranked[::-1])[::-1]
    q_sorted = np.minimum(q_sorted, 1.0)
    q = np.empty_like(q_sorted)
    q[order] = q_sorted
    return q
