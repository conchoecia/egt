"""Cross-format round-trip tests for sparse COO / CSR / NPZ persistence
(TODO_tests.md section J).

scipy has historically had regressions in both NPZ round-trips and
`.eliminate_zeros()` semantics on different sparse layouts. The whole
pipeline leans on these being exact (save_npz in the builder,
load_npz + eliminate_zeros in defining_features, CSR slicing in the
clade loop). A silent regression here would silently corrupt the paper.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import (
    coo_matrix, csr_matrix, load_npz, save_npz,
)

from egt.defining_features import load_coo_sparse
from egt.phylotreeumap import (
    algcomboix_file_to_dict,
    construct_coo_matrix_from_sampledf,
)


# --- Helpers --------------------------------------------------------------

def _hand_built_coo():
    """A small 5x6 COO with a few stored zeros, a few non-zeros, and one
    explicit zero distance at (2, 3)."""
    row = np.array([0, 0, 1, 2, 3, 4, 2])
    col = np.array([0, 2, 4, 3, 1, 5, 5])
    data = np.array([1.0, 3.0, 5.0, 0.0, 9.0, 11.0, 13.0])
    return coo_matrix((data, (row, col)), shape=(5, 6))


# --- COO <-> CSR round-trip ----------------------------------------------

def test_coo_to_csr_to_coo_preserves_cells():
    """`coo.tocsr().tocoo()` preserves every stored cell (value + dtype)."""
    orig = _hand_built_coo()
    rt = orig.tocsr().tocoo()
    # Cell set equality.
    orig_cells = set(zip(orig.row.tolist(), orig.col.tolist(),
                         orig.data.tolist()))
    rt_cells = set(zip(rt.row.tolist(), rt.col.tolist(), rt.data.tolist()))
    assert orig_cells == rt_cells
    # Dtype preserved.
    assert rt.dtype == orig.dtype
    # Shape preserved.
    assert rt.shape == orig.shape


def test_eliminate_zeros_drops_stored_zero_cells_only():
    """`csr.eliminate_zeros()` must drop exactly the stored-zero cells,
    leaving every non-zero cell intact. defining_features.load_coo_sparse
    relies on this."""
    orig = _hand_built_coo()
    csr = orig.tocsr().copy()
    n_before = csr.nnz
    csr.eliminate_zeros()
    n_after = csr.nnz
    # Exactly one zero cell (value 0.0 at (2, 3)).
    assert n_before - n_after == 1
    # The removed cell is the one at (2, 3).
    assert csr[2, 3] == 0.0  # unstored reads as 0
    # Non-zero cells unchanged.
    assert csr[0, 0] == 1.0
    assert csr[0, 2] == 3.0
    assert csr[1, 4] == 5.0
    assert csr[3, 1] == 9.0
    assert csr[4, 5] == 11.0
    assert csr[2, 5] == 13.0


# --- NPZ persistence ------------------------------------------------------

def test_save_npz_load_npz_round_trip_preserves_structure(tmp_path):
    orig = _hand_built_coo()
    p = tmp_path / "rt.coo.npz"
    save_npz(str(p), orig)
    rt = load_npz(str(p))
    assert rt.shape == orig.shape
    assert rt.dtype == orig.dtype
    # Same stored cells.
    o_sorted = sorted(zip(orig.row.tolist(), orig.col.tolist(),
                          orig.data.tolist()))
    r_coo = rt.tocoo()
    r_sorted = sorted(zip(r_coo.row.tolist(), r_coo.col.tolist(),
                          r_coo.data.tolist()))
    assert o_sorted == r_sorted
    assert rt.nnz == orig.nnz


def test_save_npz_preserves_explicit_zero_cells(tmp_path):
    """scipy should persist stored zero cells across NPZ — a property
    the pipeline relies on because construct_coo_matrix_from_sampledf
    passes raw distances (including possibly-zero) to coo_matrix, and
    load_coo_sparse later strips the zeros. If save_npz silently drops
    zero cells the "stored-zero = placeholder" semantics break.
    """
    row = np.array([0, 1, 2])
    col = np.array([0, 1, 2])
    data = np.array([0.0, 5.0, 0.0])
    coo = coo_matrix((data, (row, col)), shape=(3, 3))
    assert coo.nnz == 3

    p = tmp_path / "zeros.coo.npz"
    save_npz(str(p), coo)
    rt = load_npz(str(p))
    # scipy preserves the stored-zero entries.
    assert rt.nnz == 3, (
        f"NPZ round-trip dropped stored-zero cells: nnz={rt.nnz} (want 3). "
        f"If this is a scipy regression, pin scipy version.")


def test_dtype_preserved_across_npz_for_int_and_float(tmp_path):
    """distance dtype round-trip. gb.gz distances are integer (bp);
    COO should preserve int32/int64 without silent cast to float.
    """
    for dt in (np.int32, np.int64, np.float32, np.float64):
        coo = coo_matrix(np.array([[1, 0], [0, 2]], dtype=dt))
        p = tmp_path / f"rt_{np.dtype(dt).name}.npz"
        save_npz(str(p), coo)
        rt = load_npz(str(p))
        assert rt.dtype == dt, (
            f"dtype drift across NPZ: got {rt.dtype}, expected {dt}")


# --- CSR slicing sanity ---------------------------------------------------

def test_csr_row_slice_matches_dense_slice():
    """csr[mask, :] must match dense[mask, :] value-for-value. The
    clade loop uses csr[in_mask, :] heavily and any regression would
    mis-attribute observations."""
    rng = np.random.default_rng(0)
    dense = rng.integers(0, 3, size=(20, 15)).astype(np.float64)
    # Sparsify roughly half the entries.
    dense[dense < 2] = 0
    csr = csr_matrix(dense)
    csr.eliminate_zeros()
    mask = rng.integers(0, 2, size=20, dtype=bool)
    sliced_csr = csr[mask, :]
    sliced_dense = dense[mask, :]
    # Convert sliced_csr to dense and compare.
    np.testing.assert_array_equal(sliced_csr.toarray(), sliced_dense)


# --- End-to-end: builder -> NPZ -> loader is exact -----------------------

def test_build_save_load_exact_values(tmp_path):
    """Full path: construct_coo_matrix_from_sampledf -> save_npz ->
    load_coo_sparse should recover exact values for each stored cell.

    Uses the synthetic fixture style from test_coo_integrity.py with
    a distinct-value encoding per (species, pair) so row scrambles
    would be visible as value mismatches.
    """
    import gzip
    import pandas as pd

    fams = [f"fam_{i:04d}" for i in range(6)]
    combo = {}
    k = 0
    for i in range(len(fams)):
        for j in range(i + 1, len(fams)):
            combo[(fams[i], fams[j])] = k
            k += 1

    gb_dir = tmp_path / "gbgz"
    gb_dir.mkdir()
    rows_list = []
    expected = {}
    for s in range(4):
        sample = f"species_{s:03d}"
        gbgz = gb_dir / f"{sample}.gb.gz"
        # observe pairs 0..4 with a unique encoding
        with gzip.open(gbgz, "wt") as fh:
            fh.write("rbh1\trbh2\tdistance\n")
            for p in range(5):
                r1, r2 = sorted([fam for fam, ix in combo.items() if ix == p][0])
                d = 1000 * (s + 1) + p
                fh.write(f"{r1}\t{r2}\t{d}\n")
                expected[(s, p)] = d
        rows_list.append({"sample": sample,
                          "dis_filepath_abs": str(gbgz)})
    sampledf = pd.DataFrame(rows_list)
    coo = construct_coo_matrix_from_sampledf(
        sampledf, combo, check_paths_exist=True)

    # Save, then reload through load_coo_sparse (used in production).
    coo_path = tmp_path / "out.coo.npz"
    save_npz(str(coo_path), coo)
    # build a combo file + sampledf.tsv for load_coo_sparse.
    combo_path = tmp_path / "combo.txt"
    with open(combo_path, "w") as fh:
        for (a, b), ix in combo.items():
            fh.write(f"('{a}', '{b}')\t{ix}\n")
    sampledf_for_load = sampledf.copy()
    sampledf_for_load["taxid_list"] = [str([1]) for _ in range(len(sampledf))]
    sampledf_for_load.to_csv(tmp_path / "sdf.tsv", sep="\t",
                              index_label="idx")

    cdf = pd.read_csv(tmp_path / "sdf.tsv", sep="\t", index_col=0)
    ALGcomboix = algcomboix_file_to_dict(str(combo_path))
    csr = load_coo_sparse(cdf, str(coo_path), ALGcomboix)
    # Every expected cell must be present with the exact expected value.
    for (r, c), want in expected.items():
        got = csr[r, c]
        assert got == want, (
            f"value mismatch at (r={r}, c={c}): expected {want}, got {got}")
