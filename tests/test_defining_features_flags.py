"""TODO_tests.md section F — defining_features_plot2 z-scores + flags.

The clade-defining flags used downstream (close_in_clade,
stable_in_clade, unique_to_clade, distant_in_clade, unstable_in_clade)
are derived inside ``egt.legacy.defining_features_plot2.main`` via a
chain of:

    df = add_ratio_columns(df)
    df = df[df["occupancy_in"] >= 0.5]
    df = compute_z_scores(df)
    df = assign_flags(df, sd_number=2)

These helpers were refactored out of the inline main() body so they
can be unit-tested without running the full pipeline. Tests below
replicate TODO §F's formulas on hand-built fixtures and assert the
flag derivation agrees.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from egt.legacy.defining_features_plot2 import (
    add_ratio_columns,
    assign_flags,
    compute_z_scores,
)


# --- Fixture helpers ------------------------------------------------------

def _per_clade_df(
    pair_ids,
    notna_in, notna_out,
    mean_in, sd_in,
    mean_out, sd_out,
    occupancy_in, occupancy_out,
):
    """Return a per-clade DataFrame in the shape emitted by
    ``defining_features.process_coo_file``."""
    return pd.DataFrame({
        "pair": pair_ids,
        "notna_in": notna_in, "notna_out": notna_out,
        "mean_in": mean_in, "sd_in": sd_in,
        "mean_out": mean_out, "sd_out": sd_out,
        "occupancy_in": occupancy_in, "occupancy_out": occupancy_out,
    })


# --- add_ratio_columns ----------------------------------------------------

def test_add_ratio_columns_applies_pseudocount_and_logs():
    """Pseudocount of 1 gets added before the log10-ratio. That's the
    exact behavior of the inline code (``df["mean_in"] = df["mean_in"] + 1``).
    """
    df = _per_clade_df(
        pair_ids=[0, 1],
        notna_in=[10, 10], notna_out=[10, 10],
        mean_in=[9.0, 99.0], sd_in=[4.0, 99.0],
        mean_out=[99.0, 9.0], sd_out=[99.0, 4.0],
        occupancy_in=[1.0, 1.0], occupancy_out=[1.0, 1.0],
    )
    out = add_ratio_columns(df, pseudocount=1)
    # sd_in: 4+1=5, sd_out: 99+1=100; ratio = 0.05; log10 = -1.3010...
    assert np.isclose(out.loc[0, "sd_in_out_ratio"], 5.0 / 100.0)
    assert np.isclose(out.loc[0, "sd_in_out_ratio_log"], np.log10(0.05))
    # mean_in: 9+1=10, mean_out: 99+1=100; ratio 0.1; log -1
    assert np.isclose(out.loc[0, "mean_in_out_ratio"], 10.0 / 100.0)
    assert np.isclose(out.loc[0, "mean_in_out_ratio_log"], -1.0)
    # Row 1 is the mirror.
    assert np.isclose(out.loc[1, "sd_in_out_ratio_log"], np.log10(20.0))
    assert np.isclose(out.loc[1, "mean_in_out_ratio_log"], np.log10(10.0))


def test_add_ratio_columns_does_not_mutate_input():
    """Caller's df must be untouched — regression bugs about df mutation
    have bitten this module before."""
    df = _per_clade_df(
        pair_ids=[0],
        notna_in=[10], notna_out=[10],
        mean_in=[9.0], sd_in=[4.0],
        mean_out=[99.0], sd_out=[99.0],
        occupancy_in=[1.0], occupancy_out=[1.0],
    )
    before = df.copy(deep=True)
    _ = add_ratio_columns(df)
    pd.testing.assert_frame_equal(df, before)


# --- compute_z_scores -----------------------------------------------------

def test_compute_z_scores_matches_pandas_default():
    """z-score = (x - mean) / std using pandas ddof=1. Hand-check on a
    3-element vector so we can compute mean/std exactly."""
    # Build a df with 3 pairs and known log-ratios.
    df = pd.DataFrame({
        "pair": [0, 1, 2],
        "sd_in_out_ratio_log":   [-2.0,  0.0,  2.0],
        "mean_in_out_ratio_log": [-1.0,  0.0,  1.0],
        "occupancy_in": [1.0, 1.0, 1.0],
    })
    out = compute_z_scores(df)
    # Mean = 0, std (ddof=1) of [-2, 0, 2] == 2
    expected_sd_sigma = np.array([-1.0, 0.0, 1.0])
    np.testing.assert_allclose(
        out["sd_in_out_ratio_log_sigma"].to_numpy(), expected_sd_sigma
    )
    # Mean = 0, std (ddof=1) of [-1, 0, 1] == 1
    expected_mean_sigma = np.array([-1.0, 0.0, 1.0])
    np.testing.assert_allclose(
        out["mean_in_out_ratio_log_sigma"].to_numpy(), expected_mean_sigma
    )


def test_compute_z_scores_preserves_column_prefix():
    """compute_z_scores must not clobber the input columns."""
    df = pd.DataFrame({
        "pair": [0, 1, 2],
        "sd_in_out_ratio_log":   [-2.0,  0.0,  2.0],
        "mean_in_out_ratio_log": [-1.0,  0.0,  1.0],
        "occupancy_in": [1.0, 1.0, 1.0],
    })
    out = compute_z_scores(df)
    # Input columns still there.
    for c in ("pair", "sd_in_out_ratio_log", "mean_in_out_ratio_log"):
        assert c in out.columns
    # New sigma columns added.
    assert "sd_in_out_ratio_log_sigma" in out.columns
    assert "mean_in_out_ratio_log_sigma" in out.columns


# --- assign_flags ---------------------------------------------------------

def _flag_fixture(sd_sigma, mean_sigma, notna_out, occupancy_in):
    """Build a minimal df that assign_flags can operate on."""
    n = len(sd_sigma)
    return pd.DataFrame({
        "pair": list(range(n)),
        "sd_in_out_ratio_log_sigma":   np.asarray(sd_sigma, dtype=float),
        "mean_in_out_ratio_log_sigma": np.asarray(mean_sigma, dtype=float),
        "notna_out":    np.asarray(notna_out, dtype=np.int64),
        "occupancy_in": np.asarray(occupancy_in, dtype=float),
    })


def test_stable_flag_fires_on_negative_sd_sigma_with_high_occupancy():
    """stable_in_clade == 1  iff  sd_sigma < -sd_number  AND  occ_in >= 0.5."""
    df = _flag_fixture(
        sd_sigma=  [-3.0, -2.5, -1.99, -3.0, -3.0],
        mean_sigma=[ 0.0,  0.0,  0.0,   0.0,  0.0],
        notna_out= [   5,    5,    5,     5,    5],
        occupancy_in=[1.0, 1.0, 1.0,   0.49, 0.5],
    )
    out = assign_flags(df, sd_number=2)
    # Rows 0, 1 qualify. Row 2 fails the threshold. Row 3 fails occupancy.
    # Row 4 has occupancy exactly 0.5 -- spec says >= 0.5 passes.
    assert list(out["stable_in_clade"]) == [1, 1, 0, 0, 1]


def test_unstable_flag_fires_on_positive_sd_sigma():
    """unstable_in_clade == 1  iff  sd_sigma > sd_number  AND  occ_in >= 0.5."""
    df = _flag_fixture(
        sd_sigma=  [3.0,  2.5,  1.99, 3.0, 3.0],
        mean_sigma=[0.0,  0.0,  0.0,  0.0, 0.0],
        notna_out= [  5,    5,    5,    5,   5],
        occupancy_in=[1.0, 1.0, 1.0,  0.49, 0.5],
    )
    out = assign_flags(df, sd_number=2)
    assert list(out["unstable_in_clade"]) == [1, 1, 0, 0, 1]


def test_close_flag_fires_on_negative_mean_sigma():
    """close_in_clade == 1  iff  mean_sigma < -sd_number  AND  occ_in >= 0.5."""
    df = _flag_fixture(
        sd_sigma=  [ 0.0,  0.0,  0.0,   0.0,  0.0],
        mean_sigma=[-3.0, -2.5, -1.99, -3.0, -3.0],
        notna_out= [   5,    5,    5,     5,    5],
        occupancy_in=[1.0, 1.0, 1.0,   0.49, 0.5],
    )
    out = assign_flags(df, sd_number=2)
    assert list(out["close_in_clade"]) == [1, 1, 0, 0, 1]


def test_distant_flag_fires_on_positive_mean_sigma():
    """distant_in_clade == 1 iff mean_sigma > sd_number AND occ_in >= 0.5."""
    df = _flag_fixture(
        sd_sigma=  [0.0,  0.0,  0.0,  0.0, 0.0],
        mean_sigma=[3.0,  2.5,  1.99, 3.0, 3.0],
        notna_out= [  5,    5,    5,    5,   5],
        occupancy_in=[1.0, 1.0, 1.0,  0.49, 0.5],
    )
    out = assign_flags(df, sd_number=2)
    assert list(out["distant_in_clade"]) == [1, 1, 0, 0, 1]


def test_unique_to_clade_fires_iff_notna_out_is_zero():
    """unique_to_clade == 1 iff notna_out == 0 (no occupancy gate)."""
    df = _flag_fixture(
        sd_sigma=  [0.0, 0.0, 0.0, 0.0],
        mean_sigma=[0.0, 0.0, 0.0, 0.0],
        notna_out= [  0,   1,   5,   0],
        # Note: occupancy_in has no influence on unique_to_clade.
        occupancy_in=[1.0, 1.0, 1.0, 0.1],
    )
    out = assign_flags(df, sd_number=2)
    assert list(out["unique_to_clade"]) == [1, 0, 0, 1]


def test_flags_are_integer_dtype():
    """All five flag columns must be integer 0/1 (downstream
    SupplementaryTable code casts to int, so non-int dtypes break the
    aggregation)."""
    df = _flag_fixture(
        sd_sigma=[ -3.0,  3.0, 0.0, 0.0, 0.0],
        mean_sigma=[0.0,  0.0, -3.0, 3.0, 0.0],
        notna_out=[   5,    5,    5,   5,   0],
        occupancy_in=[1.0, 1.0, 1.0, 1.0, 1.0],
    )
    out = assign_flags(df, sd_number=2)
    for col in (
        "stable_in_clade", "unstable_in_clade",
        "close_in_clade",  "distant_in_clade",
        "unique_to_clade",
    ):
        assert pd.api.types.is_integer_dtype(out[col]), (
            f"flag column {col!r} has dtype {out[col].dtype}; "
            f"expected integer."
        )
        # All values are 0 or 1.
        assert set(out[col].tolist()).issubset({0, 1})


# --- Integrated pipeline on a synthetic per-clade df -----------------------

def test_full_pipeline_derives_flags_as_expected():
    """Full chain: add_ratio_columns -> filter occupancy_in>=0.5 ->
    compute_z_scores -> assign_flags. Hand-build a df where exactly
    one pair sits well beyond 2 sigma in both directions of the
    in/out ratio distributions, and verify the stable/close flags fire.

    Uses a 50-pair fixture so the outlier row pulls < -2 sigma even
    with ddof=1 and the outlier included in the std (typical for
    real clades with many pairs).
    """
    rng = np.random.default_rng(42)
    n = 50
    # 49 pairs near the centroid (small jitter), one extreme stable+close
    # outlier (huge out-of-clade sd + mean relative to in-clade).
    mean_in = np.concatenate([[1.0], 100.0 + rng.standard_normal(n - 1)])
    sd_in   = np.concatenate([[1.0], 100.0 + rng.standard_normal(n - 1)])
    mean_out = np.concatenate([[1.0e6], 100.0 + rng.standard_normal(n - 1)])
    sd_out   = np.concatenate([[1.0e6], 100.0 + rng.standard_normal(n - 1)])
    df = pd.DataFrame({
        "pair": list(range(n)),
        "notna_in":  [10] * n,
        "notna_out": [10] * n,
        "mean_in": mean_in, "sd_in": sd_in,
        "mean_out": mean_out, "sd_out": sd_out,
        "occupancy_in":  [1.0] * n,
        "occupancy_out": [1.0] * n,
    })
    df = add_ratio_columns(df, pseudocount=1)
    df = df[df["occupancy_in"] >= 0.5].reset_index(drop=True)
    df = compute_z_scores(df)
    df = assign_flags(df, sd_number=2)

    # Pair 0 dominates both sigma distributions (far left tail), so
    # its sigma_sd and sigma_mean both land below -sd_number=2.
    row0 = df[df["pair"] == 0].iloc[0]
    assert row0["stable_in_clade"] == 1, (
        f"Expected stable flag set on outlier; "
        f"sd_sigma={row0['sd_in_out_ratio_log_sigma']}"
    )
    assert row0["close_in_clade"] == 1, (
        f"Expected close flag set on outlier; "
        f"mean_sigma={row0['mean_in_out_ratio_log_sigma']}"
    )
    # The outlier is not in the positive tail.
    assert row0["unstable_in_clade"] == 0
    assert row0["distant_in_clade"] == 0
    # All other rows are centered near 0 sigma and flag nothing stable/close.
    rest = df[df["pair"] != 0]
    assert (rest["stable_in_clade"] == 0).all()
    assert (rest["close_in_clade"] == 0).all()


# --- Module-import smoke --------------------------------------------------

def test_module_imports_after_stale_source_fix():
    """Regression for the ``import source.rbh_tools as rbh_tools`` bug:
    the stale import would raise ModuleNotFoundError at import time.
    A direct ``from egt.legacy.defining_features_plot2 import X`` should
    now succeed cleanly.
    """
    from egt.legacy import defining_features_plot2 as m
    # Sanity: the pure-function helpers are exposed at module level.
    assert callable(m.add_ratio_columns)
    assert callable(m.compute_z_scores)
    assert callable(m.assign_flags)
    # Sanity: the module correctly binds rbh_tools from egt, not "source".
    assert m.rbh_tools.__name__ == "egt.rbh_tools"
