"""TODO_tests.md section E — sparse-native vs legacy equivalence.

Both paths co-exist in ``egt.defining_features``:

- Legacy: ``load_coo`` (dense with missing_value_as=NaN) +
  ``compute_statistics`` applied column-wise via ``df.apply``.
- Sparse: ``load_coo_sparse`` + ``compute_col_aggregates`` +
  ``_mean_std_sample`` inside ``process_coo_file``.

The sparse rewrite must produce byte-equivalent per-clade TSVs (floats
within 1e-7, ints exact) on any fixture where both paths are well-
defined. This test hand-builds a small COO with a mix of

- stored zeros (upstream placeholders -> both paths treat as missing),
- observed nonzero distances,
- entirely-missing (row, col) combinations (no stored entry -> missing),

runs both paths end-to-end, and diffs the emitted per-clade dataframes.

Skipped: the memory/speedup bullets of TODO §E -- those are production-
scale observational checks, not unit tests.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import coo_matrix, save_npz

from egt.defining_features import (
    compute_statistics,
    load_coo,
    process_coo_file,
)


# --- Fixture construction -------------------------------------------------

N_SPECIES = 10
N_PAIRS = 50

# Three taxids:
#  - taxid 101 is in-clade for species 0..4 (5 species)
#  - taxid 202 is in-clade for species 5..9 (5 species)
#  - taxid 303 covers ALL species (whole-dataset clade; gets skipped)
TAXID_IN = 101
TAXID_LISTS = [
    [101, 303] if s < 5 else [202, 303]
    for s in range(N_SPECIES)
]


def _build_fixture(tmp_path: Path, rng_seed: int = 0):
    """Hand-build a 10x50 COO covering the cases listed in TODO §E.

    Construction:
      - Every (species, pair) has a deterministic value
        distance = 100 * (species+1) + pair so the (row, col) is
        uniquely decodable.
      - ~10% of cells are left unobserved (no stored entry). These
        represent "pair not on same scaffold in this species".
      - ~5% of cells are stored zeros (upstream placeholders — the
        sparse path drops them, and the legacy path converts them to
        NaN before compute_statistics because load_coo treats the
        explicit 0.0 as "missing").

    Returns (coo, sampledf, combo_to_ix_dict).
    """
    rng = np.random.default_rng(rng_seed)
    rows = []
    cols = []
    vals = []
    for s in range(N_SPECIES):
        for p in range(N_PAIRS):
            roll = rng.random()
            if roll < 0.10:
                # unobserved: don't store anything
                continue
            if roll < 0.15:
                # stored zero (placeholder)
                rows.append(s)
                cols.append(p)
                vals.append(0.0)
                continue
            rows.append(s)
            cols.append(p)
            vals.append(float(100 * (s + 1) + p))
    coo = coo_matrix(
        (np.array(vals, dtype=np.float64),
         (np.array(rows, dtype=np.int64),
          np.array(cols, dtype=np.int64))),
        shape=(N_SPECIES, N_PAIRS),
    )

    sampledf = pd.DataFrame({
        "sample": [f"sp_{i:03d}" for i in range(N_SPECIES)],
        "taxid_list": [str(tl) for tl in TAXID_LISTS],
    })
    sampledf.index.name = "idx"

    # Build combo_to_ix: "fam_<i>_a" -> "fam_<i>_b", ix = i for i in 0..49
    combo_to_ix = {
        (f"fam_{i:04d}_a", f"fam_{i:04d}_b"): i
        for i in range(N_PAIRS)
    }
    return coo, sampledf, combo_to_ix


def _write_inputs(tmp_path: Path, coo, sampledf, combo_to_ix):
    """Persist the fixture to disk as process_coo_file expects."""
    coo_path = tmp_path / "fx.coo.npz"
    save_npz(str(coo_path), coo)
    sampledf_path = tmp_path / "sampledf.tsv"
    sampledf.to_csv(sampledf_path, sep="\t")
    combo_path = tmp_path / "combo_to_index.txt"
    with open(combo_path, "w") as fh:
        for (a, b), ix in combo_to_ix.items():
            fh.write(f"('{a}', '{b}')\t{ix}\n")
    return sampledf_path, combo_path, coo_path


def _run_legacy_path(cdf, coofile, combo_to_ix, taxid):
    """Reimplement the legacy per-clade aggregation exactly as the
    original code did: load_coo with missing_value_as=NaN, DataFrame-ify
    the dense matrix, select in/out rows by taxid, apply
    ``compute_statistics`` columnwise, and keep only columns with at
    least one in-clade observation (matching process_coo_file's
    ``keep = notna_in > 0``).
    """
    # Ensure cdf.taxid_list is a list, as the legacy code did via eval.
    cdf = cdf.copy()
    if len(cdf) > 0 and isinstance(cdf["taxid_list"].iloc[0], str):
        cdf["taxid_list"] = cdf["taxid_list"].apply(eval)
    # load_coo returns a dense ndarray with NaN for missing cells.
    matrix = load_coo(cdf, coofile, combo_to_ix, missing_value_as=np.nan)
    # Row membership masks.
    in_mask = np.fromiter(
        (taxid in tl for tl in cdf["taxid_list"]),
        count=len(cdf), dtype=bool,
    )
    out_mask = ~in_mask
    n_in = int(in_mask.sum())
    n_out = int(out_mask.sum())
    if n_in < 2 or n_out == 0:
        return None
    # Reconstruct a DataFrame for df.apply-style compute_statistics.
    df = pd.DataFrame(matrix)
    inindex = df.index[in_mask]
    outindex = df.index[out_mask]
    stats_rows = []
    for colname, col in df.items():
        rec = compute_statistics(col, inindex, outindex)
        rec["pair"] = int(colname)
        stats_rows.append(rec)
    out = pd.DataFrame(stats_rows)
    # Apply the same "keep if in-clade observed" filter the sparse path uses.
    out = out[out["notna_in"] > 0].reset_index(drop=True)
    # Reorder columns to match the sparse-path output.
    col_order = [
        "pair", "notna_in", "notna_out",
        "mean_in", "sd_in", "mean_out", "sd_out",
        "occupancy_in", "occupancy_out",
    ]
    out = out[col_order]
    # ensure numeric dtypes match the sparse path's output after CSV round trip
    out["pair"] = out["pair"].astype(np.int64)
    out["notna_in"] = out["notna_in"].astype(np.int64)
    out["notna_out"] = out["notna_out"].astype(np.int64)
    return out


def _run_sparse_path(tmp_path, sampledf_path, combo_path, coo_path, taxid):
    """Run process_coo_file (sparse) and read back the per-clade TSV."""
    cwd_before = os.getcwd()
    os.chdir(tmp_path)
    try:
        process_coo_file(
            str(sampledf_path), str(combo_path), str(coo_path),
            dfoutfilepath="unused.df", taxid_list=[taxid],
        )
    finally:
        os.chdir(cwd_before)
    matches = list(Path(tmp_path).glob(f"*_{taxid}_unique_pair_df.tsv.gz"))
    assert matches, f"sparse path did not emit a file for taxid {taxid}"
    return pd.read_csv(matches[0], sep="\t")


# --- Tests ----------------------------------------------------------------

def test_sparse_and_legacy_agree_on_schema(tmp_path):
    """Both paths produce the same columns in the same order."""
    coo, sampledf, combo_to_ix = _build_fixture(tmp_path)
    sampledf_path, combo_path, coo_path = _write_inputs(
        tmp_path, coo, sampledf, combo_to_ix)
    sparse_df = _run_sparse_path(
        tmp_path, sampledf_path, combo_path, coo_path, taxid=TAXID_IN)

    cdf = pd.read_csv(sampledf_path, sep="\t", index_col=0)
    legacy_df = _run_legacy_path(cdf, str(coo_path), combo_to_ix, taxid=TAXID_IN)
    assert legacy_df is not None
    assert list(sparse_df.columns) == list(legacy_df.columns)


def test_sparse_and_legacy_agree_row_count(tmp_path):
    """Both paths retain the same set of pairs (those with notna_in>0)."""
    coo, sampledf, combo_to_ix = _build_fixture(tmp_path)
    sampledf_path, combo_path, coo_path = _write_inputs(
        tmp_path, coo, sampledf, combo_to_ix)
    sparse_df = _run_sparse_path(
        tmp_path, sampledf_path, combo_path, coo_path, taxid=TAXID_IN)

    cdf = pd.read_csv(sampledf_path, sep="\t", index_col=0)
    legacy_df = _run_legacy_path(cdf, str(coo_path), combo_to_ix, taxid=TAXID_IN)
    assert len(sparse_df) == len(legacy_df)
    assert set(sparse_df["pair"]) == set(legacy_df["pair"])


def test_sparse_and_legacy_int_columns_exact(tmp_path):
    """Integer counts (notna_in, notna_out) match exactly."""
    coo, sampledf, combo_to_ix = _build_fixture(tmp_path)
    sampledf_path, combo_path, coo_path = _write_inputs(
        tmp_path, coo, sampledf, combo_to_ix)
    sparse_df = _run_sparse_path(
        tmp_path, sampledf_path, combo_path, coo_path, taxid=TAXID_IN)

    cdf = pd.read_csv(sampledf_path, sep="\t", index_col=0)
    legacy_df = _run_legacy_path(cdf, str(coo_path), combo_to_ix, taxid=TAXID_IN)

    # Join on pair and compare int columns elementwise.
    merged = sparse_df.merge(
        legacy_df, on="pair", suffixes=("_sp", "_lg"))
    assert (merged["notna_in_sp"] == merged["notna_in_lg"]).all()
    assert (merged["notna_out_sp"] == merged["notna_out_lg"]).all()


def test_sparse_and_legacy_float_columns_within_1e_minus_7(tmp_path):
    """Float columns (means, sds, occupancies) match within 1e-7."""
    coo, sampledf, combo_to_ix = _build_fixture(tmp_path)
    sampledf_path, combo_path, coo_path = _write_inputs(
        tmp_path, coo, sampledf, combo_to_ix)
    sparse_df = _run_sparse_path(
        tmp_path, sampledf_path, combo_path, coo_path, taxid=TAXID_IN)

    cdf = pd.read_csv(sampledf_path, sep="\t", index_col=0)
    legacy_df = _run_legacy_path(cdf, str(coo_path), combo_to_ix, taxid=TAXID_IN)

    merged = sparse_df.merge(
        legacy_df, on="pair", suffixes=("_sp", "_lg")).sort_values("pair")

    # For float columns, allow NaN-equal: np.testing.assert_allclose with
    # equal_nan=True semantics would work, but we roll our own to be
    # explicit about the policy.
    for col in ("mean_in", "sd_in", "mean_out", "sd_out",
                "occupancy_in", "occupancy_out"):
        a = merged[f"{col}_sp"].to_numpy(dtype=np.float64)
        b = merged[f"{col}_lg"].to_numpy(dtype=np.float64)
        # Both NaN is fine; otherwise must be within 1e-7.
        both_nan = np.isnan(a) & np.isnan(b)
        close = np.isclose(a, b, rtol=1e-7, atol=1e-7, equal_nan=False)
        ok = both_nan | close
        if not ok.all():
            diffs = merged.loc[~ok, ["pair", f"{col}_sp", f"{col}_lg"]]
            raise AssertionError(
                f"Column {col!r} disagreement between sparse/legacy:\n"
                f"{diffs.head(10).to_string(index=False)}"
            )


def test_sparse_legacy_agree_when_clade_has_no_stored_zeros(tmp_path):
    """Secondary fixture without any stored-zero placeholders -- the
    two paths must agree even more trivially (no zero-drop divergence).
    """
    # Build a dense fixture: every species observes every pair, no
    # stored zeros, all values positive.
    rows, cols, vals = [], [], []
    for s in range(N_SPECIES):
        for p in range(N_PAIRS):
            rows.append(s)
            cols.append(p)
            vals.append(float(100 * (s + 1) + p))
    coo = coo_matrix(
        (np.array(vals, dtype=np.float64),
         (np.array(rows, dtype=np.int64),
          np.array(cols, dtype=np.int64))),
        shape=(N_SPECIES, N_PAIRS),
    )
    sampledf = pd.DataFrame({
        "sample": [f"sp_{i:03d}" for i in range(N_SPECIES)],
        "taxid_list": [str(tl) for tl in TAXID_LISTS],
    })
    sampledf.index.name = "idx"
    combo_to_ix = {
        (f"fam_{i:04d}_a", f"fam_{i:04d}_b"): i for i in range(N_PAIRS)
    }
    sampledf_path, combo_path, coo_path = _write_inputs(
        tmp_path, coo, sampledf, combo_to_ix)
    sparse_df = _run_sparse_path(
        tmp_path, sampledf_path, combo_path, coo_path, taxid=TAXID_IN)

    cdf = pd.read_csv(sampledf_path, sep="\t", index_col=0)
    legacy_df = _run_legacy_path(cdf, str(coo_path), combo_to_ix, taxid=TAXID_IN)

    # All 50 pairs present in both.
    assert len(sparse_df) == N_PAIRS
    assert len(legacy_df) == N_PAIRS

    merged = sparse_df.merge(
        legacy_df, on="pair", suffixes=("_sp", "_lg"))
    assert (merged["notna_in_sp"] == merged["notna_in_lg"]).all()
    assert (merged["notna_out_sp"] == merged["notna_out_lg"]).all()
    for col in ("mean_in", "sd_in", "mean_out", "sd_out",
                "occupancy_in", "occupancy_out"):
        a = merged[f"{col}_sp"].to_numpy(dtype=np.float64)
        b = merged[f"{col}_lg"].to_numpy(dtype=np.float64)
        np.testing.assert_allclose(a, b, rtol=1e-7, atol=1e-7)
