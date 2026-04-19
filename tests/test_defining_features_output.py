"""Per-clade defining-features output invariants (TODO_tests.md section D).

Covers the math & schema inside `defining_features.process_coo_file` and
its pure-function building blocks `compute_col_aggregates` and
`_mean_std_sample`. The aggregation is vectorized sparse numpy and would
silently miscount (off-by-one `notna`, ddof mismatch, wrong occupancy
denominator, etc.) without these checks.

Fixture shape: a hand-built CSR with known values per species so every
aggregate (notna_in, mean_in, sd_in, occupancy_in) can be computed
exactly and compared against pandas' default (which is what the project
relied on before the sparse rewrite).
"""
from __future__ import annotations

import gzip
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix, save_npz

from egt.defining_features import (
    _mean_std_sample,
    compute_col_aggregates,
    process_coo_file,
)


# --- Helpers --------------------------------------------------------------

def _build_csr_fixture():
    """Build a 6-species x 4-pair CSR with hand-picked distances.

    Species 0..2 are in-clade for taxid 100.
    Species 3..5 are out-clade for taxid 100.

    Pair 0: observed in all 6 species.
    Pair 1: observed only in species 0..2 (in-clade only) -> unique to clade.
    Pair 2: observed only in species 3..5 (out-clade only).
    Pair 3: observed in exactly 1 in-clade species (sd_in should be NaN).
    """
    n_species = 6
    n_pairs = 4
    # rows x cols filled by hand; 0 means "not observed".
    dense = np.array([
        # p0   p1   p2   p3
        [10.0, 100.0,  0.0, 7.0],   # sp0 in
        [20.0, 200.0,  0.0, 0.0],   # sp1 in
        [30.0, 300.0,  0.0, 0.0],   # sp2 in
        [40.0,  0.0, 400.0, 0.0],   # sp3 out
        [50.0,  0.0, 500.0, 0.0],   # sp4 out
        [60.0,  0.0, 600.0, 0.0],   # sp5 out
    ], dtype=np.float64)
    csr = csr_matrix(dense)
    # load_coo_sparse would eliminate stored zeros; emulate that here.
    csr.eliminate_zeros()
    return csr, dense, n_species, n_pairs


def _write_process_coo_inputs(tmp_path: Path, csr, taxid_list_per_row):
    """Persist csr + sampledf + combo_to_index on disk in the format
    process_coo_file expects. Returns (sampledf_path, combo_path, coo_path).
    """
    coo_path = tmp_path / "fx.coo.npz"
    save_npz(str(coo_path), csr.tocoo())

    n_species, n_pairs = csr.shape
    # Minimal sampledf — process_coo_file reads index_col=0 then
    # eval()s "taxid_list".
    sampledf = pd.DataFrame({
        "sample": [f"species_{i:02d}" for i in range(n_species)],
        "taxid_list": [str(list(tl)) for tl in taxid_list_per_row],
    })
    sampledf.index.name = "idx"
    sampledf_path = tmp_path / "sampledf.tsv"
    sampledf.to_csv(sampledf_path, sep="\t")

    combo_path = tmp_path / "combo_to_index.txt"
    with open(combo_path, "w") as fh:
        for i in range(n_pairs):
            fh.write(f"('fam_{i:04d}_a', 'fam_{i:04d}_b')\t{i}\n")
    return sampledf_path, combo_path, coo_path


def _run_process_and_load(tmp_path, csr, taxid_list_per_row, taxid):
    """Run process_coo_file in an isolated cwd, read the per-clade tsv
    back, and return the DataFrame."""
    sampledf_path, combo_path, coo_path = _write_process_coo_inputs(
        tmp_path, csr, taxid_list_per_row)
    cwd_before = os.getcwd()
    os.chdir(tmp_path)
    try:
        process_coo_file(
            str(sampledf_path), str(combo_path), str(coo_path),
            dfoutfilepath="unused.df", taxid_list=[taxid])
    finally:
        os.chdir(cwd_before)
    # Output file pattern: {nodename}_{taxid}_unique_pair_df.tsv.gz
    matches = list(Path(tmp_path).glob(f"*_{taxid}_unique_pair_df.tsv.gz"))
    assert matches, f"no output tsv.gz for taxid {taxid} in {tmp_path}"
    return pd.read_csv(matches[0], sep="\t")


# --- Direct function tests (pure math) ------------------------------------

def test_mean_std_sample_matches_pandas_ddof1():
    """`_mean_std_sample` must match pandas .mean() / .std() (ddof=1)."""
    # Hand-construct three groups with different cardinalities.
    groups = [
        np.array([10.0, 20.0, 30.0, 40.0]),
        np.array([7.0, 7.0]),
        np.array([42.0]),         # single value -> sd should be NaN
    ]
    notna = np.array([g.size for g in groups], dtype=np.int64)
    sum_v = np.array([g.sum() for g in groups], dtype=np.float64)
    sumsq_v = np.array([(g * g).sum() for g in groups], dtype=np.float64)

    mean, std = _mean_std_sample(notna, sum_v, sumsq_v)

    for i, g in enumerate(groups):
        assert mean[i] == pytest.approx(g.mean())
        if g.size >= 2:
            assert std[i] == pytest.approx(pd.Series(g).std(ddof=1))
        else:
            assert np.isnan(std[i]), \
                f"sd must be NaN for notna<2, got {std[i]} at i={i}"


def test_mean_std_sample_nan_when_notna_is_zero():
    notna = np.array([0], dtype=np.int64)
    mean, std = _mean_std_sample(
        notna, np.array([0.0]), np.array([0.0]))
    assert np.isnan(mean[0])
    assert np.isnan(std[0])


def test_compute_col_aggregates_matches_dense_sum():
    """Per-column aggregates: notna, sum, sum-of-squares."""
    dense = np.array([
        [1.0, 0.0, 5.0],
        [2.0, 0.0, 6.0],
        [0.0, 0.0, 7.0],
    ])
    csr = csr_matrix(dense)
    csr.eliminate_zeros()
    notna, sum_v, sumsq_v = compute_col_aggregates(csr)
    np.testing.assert_array_equal(notna, np.array([2, 0, 3]))
    np.testing.assert_allclose(sum_v, np.array([3.0, 0.0, 18.0]))
    np.testing.assert_allclose(sumsq_v, np.array([5.0, 0.0, 110.0]))


def test_compute_col_aggregates_dtype_is_float64():
    """sum-of-squares must be float64 to avoid overflow on large bp
    distances (TODO_tests.md H.massive floats).
    """
    # 1e8 bp squared is 1e16; summed over many species -> 1e19.
    big = 1e8
    dense = np.full((10, 1), big, dtype=np.float64)
    csr = csr_matrix(dense)
    csr.eliminate_zeros()
    _, sum_v, sumsq_v = compute_col_aggregates(csr)
    assert sum_v.dtype == np.float64
    assert sumsq_v.dtype == np.float64
    # Exact check: no float-precision loss at this scale in f64.
    assert sumsq_v[0] == pytest.approx(10 * big * big)


# --- process_coo_file output tests (section D) ---------------------------

def test_per_clade_tsv_schema_and_order(tmp_path):
    """Schema: every emitted per-clade tsv has columns in stable order."""
    csr, _, _, _ = _build_csr_fixture()
    taxid_lists = [[100], [100], [100], [200], [200], [200]]
    df = _run_process_and_load(tmp_path, csr, taxid_lists, taxid=100)
    expected_cols = ["pair", "notna_in", "notna_out", "mean_in", "sd_in",
                     "mean_out", "sd_out", "occupancy_in", "occupancy_out"]
    assert list(df.columns) == expected_cols


def test_per_clade_integer_counts_are_ints(tmp_path):
    csr, _, _, _ = _build_csr_fixture()
    taxid_lists = [[100], [100], [100], [200], [200], [200]]
    df = _run_process_and_load(tmp_path, csr, taxid_lists, taxid=100)
    # pandas read_csv may infer int dtype; allow int64 or plain int.
    assert pd.api.types.is_integer_dtype(df["notna_in"])
    assert pd.api.types.is_integer_dtype(df["notna_out"])
    assert (df["notna_in"] >= 0).all()
    assert (df["notna_out"] >= 0).all()


def test_per_clade_counts_bounded_by_clade_size(tmp_path):
    """notna_in <= n_in_clade; occupancy_in == notna_in / n_in_clade."""
    csr, _, _, _ = _build_csr_fixture()
    taxid_lists = [[100], [100], [100], [200], [200], [200]]
    n_in = 3
    n_out = 3
    df = _run_process_and_load(tmp_path, csr, taxid_lists, taxid=100)
    assert (df["notna_in"] <= n_in).all()
    assert (df["notna_out"] <= n_out).all()
    np.testing.assert_allclose(
        df["occupancy_in"].to_numpy(),
        df["notna_in"].to_numpy() / n_in,
    )
    np.testing.assert_allclose(
        df["occupancy_out"].to_numpy(),
        df["notna_out"].to_numpy() / n_out,
    )


def test_per_clade_means_nonneg_and_match_hand(tmp_path):
    """Means ≥ 0 where notna>0, and they match the hand-built values."""
    csr, _, _, _ = _build_csr_fixture()
    taxid_lists = [[100], [100], [100], [200], [200], [200]]
    df = _run_process_and_load(tmp_path, csr, taxid_lists, taxid=100)
    # Pair 0: in-clade values 10, 20, 30 -> mean 20, sd 10
    p0 = df[df["pair"] == 0].iloc[0]
    assert p0["mean_in"] == pytest.approx(20.0)
    assert p0["sd_in"] == pytest.approx(pd.Series([10.0, 20.0, 30.0]).std())
    # Pair 0 out: 40, 50, 60 -> mean 50
    assert p0["mean_out"] == pytest.approx(50.0)
    # All means ≥ 0 where notna > 0.
    has_in = df["notna_in"] > 0
    has_out = df["notna_out"] > 0
    assert (df.loc[has_in, "mean_in"] >= 0).all()
    assert (df.loc[has_out, "mean_out"] >= 0).all()


def test_per_clade_sd_nan_iff_notna_lt_2(tmp_path):
    """ddof=1 means sd_in is NaN iff notna_in < 2, same for _out."""
    csr, _, _, _ = _build_csr_fixture()
    taxid_lists = [[100], [100], [100], [200], [200], [200]]
    df = _run_process_and_load(tmp_path, csr, taxid_lists, taxid=100)
    # Pair 3 has exactly one in-clade observation (sp0 only) -> sd_in NaN.
    # It has zero out-clade observations -> sd_out NaN, mean_out NaN.
    p3 = df[df["pair"] == 3]
    # Pair 3 must be present (in-clade observed it once, so notna_in>0 kept).
    assert len(p3) == 1
    p3 = p3.iloc[0]
    assert p3["notna_in"] == 1
    assert np.isnan(p3["sd_in"]), f"expected NaN sd_in, got {p3['sd_in']}"
    assert p3["notna_out"] == 0
    assert np.isnan(p3["sd_out"])
    assert np.isnan(p3["mean_out"])

    # Cross-check: any row with notna_in >= 2 has finite sd_in
    mask_fin = df["notna_in"] >= 2
    assert df.loc[mask_fin, "sd_in"].notna().all()
    mask_lt2 = df["notna_in"] < 2
    assert df.loc[mask_lt2, "sd_in"].isna().all()


def test_per_clade_sd_matches_pandas_ddof1(tmp_path):
    """Hand-compute with pandas ddof=1 to catch ddof=0 regressions."""
    csr, _, _, _ = _build_csr_fixture()
    taxid_lists = [[100], [100], [100], [200], [200], [200]]
    df = _run_process_and_load(tmp_path, csr, taxid_lists, taxid=100)
    # pair 0 in-clade values
    hand_sd = pd.Series([10.0, 20.0, 30.0]).std(ddof=1)
    assert df[df["pair"] == 0].iloc[0]["sd_in"] == pytest.approx(hand_sd)
    # pair 0 out-clade
    hand_sd_out = pd.Series([40.0, 50.0, 60.0]).std(ddof=1)
    assert df[df["pair"] == 0].iloc[0]["sd_out"] == pytest.approx(hand_sd_out)


def test_per_clade_pair_column_unique(tmp_path):
    csr, _, _, _ = _build_csr_fixture()
    taxid_lists = [[100], [100], [100], [200], [200], [200]]
    df = _run_process_and_load(tmp_path, csr, taxid_lists, taxid=100)
    assert df["pair"].is_unique


def test_per_clade_only_keeps_pairs_with_in_clade_observations(tmp_path):
    """Pair 2 is observed only out-of-clade (notna_in == 0) and must be
    dropped from the in-clade's tsv. This is the documented
    keep = notna_in > 0 filter."""
    csr, _, _, _ = _build_csr_fixture()
    taxid_lists = [[100], [100], [100], [200], [200], [200]]
    df = _run_process_and_load(tmp_path, csr, taxid_lists, taxid=100)
    assert 2 not in df["pair"].tolist()
    # but pairs 0, 1, 3 (in-clade observations exist) must all appear
    for p in (0, 1, 3):
        assert p in df["pair"].tolist(), f"pair {p} missing"


def test_per_clade_unique_to_clade_has_notna_out_zero(tmp_path):
    """Pair 1 is a clade-unique pair (observed in-clade only).
    Assert notna_out == 0 in the output."""
    csr, _, _, _ = _build_csr_fixture()
    taxid_lists = [[100], [100], [100], [200], [200], [200]]
    df = _run_process_and_load(tmp_path, csr, taxid_lists, taxid=100)
    p1 = df[df["pair"] == 1].iloc[0]
    assert p1["notna_in"] == 3
    assert p1["notna_out"] == 0
    # occupancy_out should be 0/3 = 0
    assert p1["occupancy_out"] == pytest.approx(0.0)


def test_per_clade_respects_fresh_nan_semantics(tmp_path):
    """When notna_in == 0 the row is filtered out (not written as NaN).
    When notna_out == 0 the row IS written (because in-clade observed it)
    and mean_out / sd_out should be NaN.
    """
    csr, _, _, _ = _build_csr_fixture()
    taxid_lists = [[100], [100], [100], [200], [200], [200]]
    df = _run_process_and_load(tmp_path, csr, taxid_lists, taxid=100)
    zero_out = df[df["notna_out"] == 0]
    assert len(zero_out) >= 1
    for _, row in zero_out.iterrows():
        assert np.isnan(row["mean_out"]), \
            f"mean_out must be NaN when notna_out==0, got {row['mean_out']}"
        assert np.isnan(row["sd_out"])
