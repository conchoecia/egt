"""Unit tests for egt.go.stats."""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import binom as sb_binom, hypergeom as sb_hypergeom

from egt.go.stats import bh_qvalues, binom_sf, hypergeom_sf, log_binom


# ---------- log_binom ----------
def test_log_binom_matches_math_comb_log():
    for n, k in [(10, 0), (10, 10), (10, 5), (20, 7), (100, 40)]:
        assert math.isclose(log_binom(n, k), math.log(math.comb(n, k)), rel_tol=1e-9)


def test_log_binom_out_of_range():
    assert log_binom(5, -1) == float("-inf")
    assert log_binom(5, 6) == float("-inf")


# ---------- hypergeom_sf ----------
@pytest.mark.parametrize("N,K,n,k", [
    (100, 30, 20, 0),
    (100, 30, 20, 1),
    (100, 30, 20, 10),
    (100, 30, 20, 20),   # = min(K, n), so sf = P(X == 20) = pmf(20)
    (50, 25, 25, 13),
    (1000, 100, 50, 12),
    (20, 10, 5, 3),
])
def test_hypergeom_sf_matches_scipy(N, K, n, k):
    expected = float(sb_hypergeom.sf(k - 1, N, K, n))
    got = hypergeom_sf(k, N, K, n)
    assert math.isclose(got, expected, rel_tol=1e-8, abs_tol=1e-12)


def test_hypergeom_sf_edges():
    # k <= 0 → 1.0 (the entire support covers k = 0, 1, …)
    assert hypergeom_sf(0, 100, 30, 20) == 1.0
    assert hypergeom_sf(-5, 100, 30, 20) == 1.0
    # k above the support → 0.0
    assert hypergeom_sf(11, 100, 10, 5) == 0.0  # k > K=10
    assert hypergeom_sf(6, 100, 10, 5) == 0.0   # k > n=5


def test_hypergeom_sf_random_vs_scipy():
    rng = np.random.default_rng(0)
    for _ in range(50):
        N = int(rng.integers(20, 500))
        K = int(rng.integers(1, N))
        n = int(rng.integers(1, N))
        kmax = min(K, n)
        k = int(rng.integers(0, kmax + 1))
        expected = float(sb_hypergeom.sf(k - 1, N, K, n))
        assert math.isclose(
            hypergeom_sf(k, N, K, n), expected, rel_tol=1e-7, abs_tol=1e-12,
        ), f"mismatch at N={N} K={K} n={n} k={k}"


# ---------- binom_sf ----------
@pytest.mark.parametrize("k,n,p", [
    (0, 10, 0.1),
    (1, 10, 0.1),
    (5, 10, 0.5),
    (10, 10, 0.5),
    (11, 10, 0.5),
    (3, 100, 0.01),
])
def test_binom_sf_matches_scipy(k, n, p):
    expected = float(sb_binom.sf(k - 1, n, p))
    assert math.isclose(binom_sf(k, n, p), expected, rel_tol=1e-9, abs_tol=1e-12)


def test_binom_sf_edges():
    # p == 0: any k > 0 has probability 0
    assert binom_sf(0, 10, 0.0) == 1.0
    assert binom_sf(1, 10, 0.0) == 0.0
    assert binom_sf(-1, 10, 0.0) == 1.0
    # p == 1: all trials succeed, so k <= n is certain, k > n impossible
    assert binom_sf(10, 10, 1.0) == 1.0
    assert binom_sf(11, 10, 1.0) == 0.0
    # p slightly >1 treated as 1 in the edge branch
    assert binom_sf(5, 10, 1.5) == 1.0


# ---------- bh_qvalues ----------
def test_bh_empty():
    q = bh_qvalues([])
    assert isinstance(q, np.ndarray)
    assert q.size == 0


def test_bh_monotone_and_bounded():
    rng = np.random.default_rng(1)
    for _ in range(20):
        p = rng.uniform(0, 1, size=rng.integers(5, 50))
        q = bh_qvalues(p)
        assert q.shape == p.shape
        # bounds
        assert (q >= 0).all() and (q <= 1).all()
        # sorted-by-p q is non-decreasing
        order = np.argsort(p)
        q_sorted = q[order]
        assert (np.diff(q_sorted) >= -1e-12).all()


def test_bh_known_case():
    # 10 p-values — 5 uniform large, 5 small. Check the smallest hits ~ n*p/rank
    p = np.array([0.001, 0.01, 0.03, 0.1, 0.2, 0.4, 0.5, 0.6, 0.7, 0.9])
    q = bh_qvalues(p)
    # first p * n / 1 = 0.01 → but monotonicity can only lower this using values
    # after it; here the forward walk stays at 0.01 until the next rank bumps it.
    n = 10
    expected_raw = p * n / np.arange(1, n + 1)
    # right-to-left cummin
    expected = np.minimum.accumulate(expected_raw[::-1])[::-1]
    np.testing.assert_allclose(q, expected)


def test_bh_input_order_preserved():
    p = np.array([0.5, 0.01, 0.2])
    q = bh_qvalues(p)
    # q[1] corresponds to p[1] = 0.01, the smallest p.
    assert q[1] == min(q)
