"""Edge-case tests for `defining_features.process_coo_file`
(TODO_tests.md section H).

The per-clade loop has a handful of branches that must behave
cleanly under:

  - Singleton clade (n_in == 1): skip, no division-by-zero.
  - Whole-dataset clade (n_out == 0): skip, no corrupted output.
  - Species with empty gb.gz -> empty CSR row: still aligned, COO row
    ordering preserved.
  - Pair appears in exactly one species: `notna_in + notna_out == 1`
    and sd is NaN; downstream must tolerate.
  - Genuine observed zero (two orthologs at identical pos): current
    convention drops these via `csr.eliminate_zeros()` at load time —
    the test documents this decision so any flip to "keep observed 0s"
    is a conscious edit, not a silent change.
  - Massive distances (float64 internal aggregation, no float32 drift).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix, save_npz

from egt.defining_features import (
    _mean_std_sample,
    compute_col_aggregates,
    load_coo_sparse,
    process_coo_file,
)


# --- Helpers (parallel to section D's) ------------------------------------

def _write_fixture(tmp_path: Path, dense: np.ndarray, taxid_lists, n_pairs=None):
    """Persist a dense-matrix fixture + sampledf + combo_to_index."""
    csr = csr_matrix(dense)
    csr.eliminate_zeros()
    coo_path = tmp_path / "fx.coo.npz"
    save_npz(str(coo_path), csr.tocoo())

    n_species = dense.shape[0]
    n_pairs = n_pairs or dense.shape[1]
    sampledf = pd.DataFrame({
        "sample": [f"species_{i:02d}" for i in range(n_species)],
        "taxid_list": [str(list(tl)) for tl in taxid_lists],
    })
    sampledf.index.name = "idx"
    sampledf_path = tmp_path / "sampledf.tsv"
    sampledf.to_csv(sampledf_path, sep="\t")

    combo_path = tmp_path / "combo_to_index.txt"
    with open(combo_path, "w") as fh:
        for i in range(n_pairs):
            fh.write(f"('fam_{i:04d}_a', 'fam_{i:04d}_b')\t{i}\n")
    return sampledf_path, combo_path, coo_path


def _run_process(tmp_path, sampledf_path, combo_path, coo_path, taxid_list):
    cwd_before = os.getcwd()
    os.chdir(tmp_path)
    try:
        process_coo_file(
            str(sampledf_path), str(combo_path), str(coo_path),
            dfoutfilepath="unused.df", taxid_list=list(taxid_list))
    finally:
        os.chdir(cwd_before)


def _outputs_in(tmp_path, taxid):
    return list(Path(tmp_path).glob(f"*_{taxid}_unique_pair_df.tsv.gz"))


# --- Tests ----------------------------------------------------------------

def test_singleton_clade_is_skipped_cleanly(tmp_path):
    """n_in == 1 => skip this clade, no output file, no crash."""
    dense = np.array([
        [10.0, 20.0],
        [11.0, 21.0],
        [12.0, 22.0],
    ], dtype=np.float64)
    # taxid 100 is only on species 0 -> singleton.
    taxid_lists = [[100], [200], [200]]
    s, c, coo = _write_fixture(tmp_path, dense, taxid_lists)
    _run_process(tmp_path, s, c, coo, taxid_list=[100])
    assert _outputs_in(tmp_path, 100) == [], \
        "singleton clade must be skipped (no tsv.gz emitted)"


def test_whole_dataset_clade_is_skipped_cleanly(tmp_path):
    """n_out == 0 => skip. (This is Metazoa under the paper's species set.)"""
    dense = np.array([
        [10.0, 20.0],
        [11.0, 21.0],
        [12.0, 22.0],
    ], dtype=np.float64)
    taxid_lists = [[100], [100], [100]]
    s, c, coo = _write_fixture(tmp_path, dense, taxid_lists)
    _run_process(tmp_path, s, c, coo, taxid_list=[100])
    assert _outputs_in(tmp_path, 100) == []


def test_species_with_empty_gbgz_does_not_scramble_rows(tmp_path):
    """A species with no observations (all-zeros row in the CSR) must
    still occupy its row. Other species' notna counts must be correct.
    """
    # Species 1 is "empty" (no observations). Species 0, 2 are in-clade.
    # Species 3, 4 are out-clade.
    dense = np.array([
        [10.0, 20.0],  # sp0 in
        [ 0.0,  0.0],  # sp1 in (empty)
        [12.0, 22.0],  # sp2 in
        [40.0, 50.0],  # sp3 out
        [41.0, 51.0],  # sp4 out
    ], dtype=np.float64)
    taxid_lists = [[100], [100], [100], [200], [200]]
    s, c, coo = _write_fixture(tmp_path, dense, taxid_lists)
    _run_process(tmp_path, s, c, coo, taxid_list=[100])
    out = _outputs_in(tmp_path, 100)
    assert len(out) == 1
    df = pd.read_csv(out[0], sep="\t")
    # 2 observations per pair in-clade (sp0 and sp2 — sp1 is empty).
    # occupancy_in is notna_in / n_in_clade (3), so 2/3.
    p0 = df[df["pair"] == 0].iloc[0]
    assert p0["notna_in"] == 2
    assert p0["occupancy_in"] == pytest.approx(2.0 / 3.0)
    # out-clade has 2 species, both observed.
    assert p0["notna_out"] == 2
    assert p0["occupancy_out"] == pytest.approx(1.0)


def test_pair_observed_in_only_one_species(tmp_path):
    """Pair observed in exactly one species: sd is NaN; downstream
    flag derivation must not over-claim stability."""
    dense = np.array([
        [ 0.0, 10.0],   # sp0 in-clade observes pair 1 only
        [ 0.0,  0.0],   # sp1 in-clade empty
        [99.0,  0.0],   # sp2 out-clade observes pair 0 only
        [88.0,  0.0],   # sp3 out-clade observes pair 0 only
    ], dtype=np.float64)
    taxid_lists = [[100], [100], [200], [200]]
    s, c, coo = _write_fixture(tmp_path, dense, taxid_lists)
    _run_process(tmp_path, s, c, coo, taxid_list=[100])
    out = _outputs_in(tmp_path, 100)
    assert len(out) == 1
    df = pd.read_csv(out[0], sep="\t")
    # pair 1: in-clade notna=1, sd_in NaN; out-clade notna=0, sd_out NaN.
    p1 = df[df["pair"] == 1].iloc[0]
    assert p1["notna_in"] == 1
    assert p1["notna_out"] == 0
    assert np.isnan(p1["sd_in"])
    assert np.isnan(p1["sd_out"])
    assert (p1["notna_in"] + p1["notna_out"]) == 1


def test_genuine_observed_zero_dropped_current_convention(tmp_path):
    """Observed zero-distance cells are dropped at load time (current
    convention). This test documents the decision; a pipeline change
    that starts keeping them must update this test deliberately.
    """
    # A hand-built CSR where species 0 has distance 0 stored explicitly.
    # scipy's coo_matrix will store the zero; save_npz + load preserves
    # storage. load_coo_sparse then calls eliminate_zeros() — so the
    # cell should not appear in aggregates.
    from scipy.sparse import coo_matrix
    row = np.array([0, 1])
    col = np.array([0, 0])
    data = np.array([0.0, 5.0])   # species 0 observes distance 0
    coo = coo_matrix((data, (row, col)), shape=(2, 1))
    assert coo.nnz == 2, "scipy should store the explicit zero"

    coo_path = tmp_path / "zero_test.coo.npz"
    save_npz(str(coo_path), coo)

    # Build a matching sampledf / combo.
    sampledf = pd.DataFrame({
        "sample": ["sp0", "sp1"],
        "taxid_list": [str([100]), str([100])],
    })
    sampledf.index.name = "idx"
    sdf_path = tmp_path / "sampledf.tsv"
    sampledf.to_csv(sdf_path, sep="\t")
    combo_path = tmp_path / "combo.txt"
    with open(combo_path, "w") as fh:
        fh.write("('fam_a', 'fam_b')\t0\n")

    # Use load_coo_sparse directly to confirm the eliminate_zeros behavior.
    cdf = pd.read_csv(sdf_path, sep="\t", index_col=0)
    from egt.phylotreeumap import algcomboix_file_to_dict
    ALGcomboix = algcomboix_file_to_dict(str(combo_path))
    csr = load_coo_sparse(cdf, str(coo_path), ALGcomboix)
    # After eliminate_zeros, nnz should be 1 (the 5.0 only).
    assert csr.nnz == 1, (
        f"expected stored-zeros dropped (nnz=1); got {csr.nnz}. "
        f"If this is intentional, update the test + load_coo_sparse docstring.")
    # Value at (1, 0) should be 5.0
    assert csr[1, 0] == pytest.approx(5.0)
    # Value at (0, 0) should read as 0 (unstored after eliminate_zeros).
    assert csr[0, 0] == 0.0


def test_massive_distances_float64_internal(tmp_path):
    """Distances up to 1e8 bp squared -> 1e16, summed over N species ->
    1e19. float32 mantissa (~7 digits) would lose precision here; this
    test asserts internal float64 aggregation.
    """
    # 10 species, distance 1e8 each, one pair.
    dense = np.full((10, 1), 1e8, dtype=np.float64)
    csr = csr_matrix(dense)
    csr.eliminate_zeros()
    notna, sum_v, sumsq_v = compute_col_aggregates(csr)
    # Exact expected values in float64.
    expected_sum = 10 * 1e8
    expected_sumsq = 10 * (1e8) ** 2  # 1e17
    assert sum_v[0] == expected_sum
    assert sumsq_v[0] == expected_sumsq, (
        f"sum-of-squares mismatch at large scale: got {sumsq_v[0]}, "
        f"expected {expected_sumsq}. Likely float32 accumulation.")
    # Sanity: mean/std from the aggregates is exact too.
    mean, std = _mean_std_sample(notna, sum_v, sumsq_v)
    assert mean[0] == pytest.approx(1e8)
    # All values equal -> variance 0, std 0 (not NaN because notna>=2).
    assert std[0] == pytest.approx(0.0)


def test_pair_in_no_species_kept_structurally_but_filtered_out(tmp_path):
    """A pair (column) that has no observations at all still needs its
    column index to exist in the COO (stable pair addressing), but
    process_coo_file's keep = notna_in > 0 filter excludes it from the
    per-clade tsv. This guards against "why is pair X missing downstream".
    """
    # 3 columns; column 2 is entirely empty.
    dense = np.array([
        [10.0, 20.0, 0.0],
        [11.0, 21.0, 0.0],
        [30.0, 40.0, 0.0],
        [31.0, 41.0, 0.0],
    ], dtype=np.float64)
    taxid_lists = [[100], [100], [200], [200]]
    s, c, coo = _write_fixture(tmp_path, dense, taxid_lists, n_pairs=3)
    _run_process(tmp_path, s, c, coo, taxid_list=[100])
    df = pd.read_csv(_outputs_in(tmp_path, 100)[0], sep="\t")
    # Pairs 0 and 1 present in-clade; pair 2 not.
    assert 0 in df["pair"].tolist()
    assert 1 in df["pair"].tolist()
    assert 2 not in df["pair"].tolist()


def test_out_clade_aggregates_via_subtraction_match_direct(tmp_path):
    """process_coo_file computes out-clade aggregates by subtraction
    (total - in). Assert this matches the direct slice under float64.
    """
    dense = np.array([
        [10.0, 20.0, 0.0],
        [11.0, 21.0, 30.0],
        [12.0, 22.0, 31.0],
        [40.0, 50.0, 0.0],
        [41.0, 51.0, 0.0],
    ], dtype=np.float64)
    csr = csr_matrix(dense)
    csr.eliminate_zeros()
    total_notna, total_sum, total_sumsq = compute_col_aggregates(csr)
    in_mask = np.array([True, True, True, False, False])
    notna_in, sum_in, sumsq_in = compute_col_aggregates(csr[in_mask, :])
    notna_out_sub = total_notna - notna_in
    sum_out_sub = total_sum - sum_in
    sumsq_out_sub = total_sumsq - sumsq_in
    notna_out_dir, sum_out_dir, sumsq_out_dir = compute_col_aggregates(
        csr[~in_mask, :])
    np.testing.assert_array_equal(notna_out_sub, notna_out_dir)
    np.testing.assert_allclose(sum_out_sub, sum_out_dir)
    np.testing.assert_allclose(sumsq_out_sub, sumsq_out_dir)
