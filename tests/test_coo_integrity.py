"""Integrity tests for the COO build path.

Covers the two documented symptoms of the 2026-04-19 row-scramble bug:

1. `construct_coo_matrix_from_sampledf` with a sampledf whose index is
   not 0..N-1 used to silently assign rows by label value, producing a
   COO with the right values at the wrong species' rows. The patch at
   `phylotreeumap.py:2612` (reset_index + positional assignment) must
   make this class of scramble either not happen or trip an assertion.

2. `grid_verify_coo` samples a regular grid of (row, col) cells and
   checks each against the per-species gb.gz. It catches any scramble
   that leaves traces anywhere in the matrix — critical for code paths
   that downsample rows or columns.

Both tests run against synthetic fixtures built in a tmp dir so they
don't depend on the 202509 production data.
"""
from __future__ import annotations

import gzip
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from scipy.sparse import save_npz

from egt._testing import grid_verify_coo


# --- Fixture construction helpers -----------------------------------------

FAMS = [f"fam_{i:04d}" for i in range(12)]      # 12 fake BCnS families
# Every (fam_i, fam_j) with i < j, indexed 0..65
COMBO_TO_IX = {
    (FAMS[i], FAMS[j]): k
    for k, (i, j) in enumerate(
        [(i, j) for i in range(len(FAMS)) for j in range(i + 1, len(FAMS))]
    )
}
IX_TO_PAIR = {v: k for k, v in COMBO_TO_IX.items()}


def _write_gbgz(gb_dir: Path, sample: str, rows: list[tuple[str, str, int]]):
    """Write a per-species gb.gz with the three required columns."""
    gbgz = gb_dir / f"{sample}.gb.gz"
    with gzip.open(gbgz, "wt") as fh:
        fh.write("rbh1\trbh2\tdistance\n")
        for r1, r2, d in rows:
            a, b = sorted((r1, r2))
            fh.write(f"{a}\t{b}\t{d}\n")
    return gbgz


def _build_fixture(tmp_path: Path, n_species: int = 6, rng_seed: int = 0):
    """Return (sampledf, gb_dir, expected_values) for a tiny fixture.

    Each species has a deterministic distance assigned to each pair,
    computed as `1000 * (species_idx + 1) + pair_idx`. This means the
    (row, col) cell uniquely encodes the species, which makes
    row-scramble detection trivial: if COO[r, c] / 1000 == r + 1 then
    the row matches; otherwise scramble is detected.
    """
    rng = np.random.default_rng(rng_seed)
    gb_dir = tmp_path / "gbgz"
    gb_dir.mkdir()
    sampledf_rows = []
    expected = {}
    for s in range(n_species):
        sample = f"species_{s:03d}"
        # pick a random subset of pairs so not every pair is observed
        # in every species — catches bugs that only fire on density < 1.
        pair_indices = rng.choice(len(COMBO_TO_IX), size=len(COMBO_TO_IX) // 2,
                                    replace=False).tolist()
        rows = []
        for p in pair_indices:
            r1, r2 = IX_TO_PAIR[p]
            d = 1000 * (s + 1) + p       # value uniquely encodes (s, p)
            rows.append((r1, r2, d))
            expected[(s, p)] = d
        _write_gbgz(gb_dir, sample, rows)
        sampledf_rows.append({"sample": sample,
                               "dis_filepath_abs": str(gb_dir / f"{sample}.gb.gz")})
    sampledf = pd.DataFrame(sampledf_rows)
    return sampledf, gb_dir, expected


# --- Tests ----------------------------------------------------------------

def test_happy_path_grid_verify(tmp_path):
    """Clean build on a tiny fixture; grid check passes."""
    from egt.phylotreeumap import construct_coo_matrix_from_sampledf
    sampledf, gb_dir, expected = _build_fixture(tmp_path)
    coo = construct_coo_matrix_from_sampledf(
        sampledf, COMBO_TO_IX, check_paths_exist=True)
    # spot-check a few cells directly
    csr = coo.tocsr()
    for (row_idx, col_idx), want in list(expected.items())[:5]:
        assert csr[row_idx, col_idx] == want, (
            f"value mismatch at ({row_idx},{col_idx}): "
            f"expected {want}, got {csr[row_idx, col_idx]}")
    # grid-verify against the gb.gz files — should pass cleanly
    summary = grid_verify_coo(coo, sampledf, COMBO_TO_IX, gb_dir,
                               n_rows=3, n_cols=3, strict=True)
    assert summary["failed"] == 0


def test_scrambled_sampledf_index_resets_and_still_correct(tmp_path):
    """Pathological case: sampledf.index has non-positional labels.

    Before the fix, this caused the COO to be populated with each
    species' data at the row matching their LABEL value, not their
    positional index. After the fix, the builder resets the index
    internally, and the resulting COO is still correct.
    """
    from egt.phylotreeumap import construct_coo_matrix_from_sampledf
    sampledf, gb_dir, expected = _build_fixture(tmp_path, n_species=6)
    # Install pathological labels: 0..5 shuffled to [3, 7, 1, 9, 2, 5].
    pathological_labels = [3, 7, 1, 9, 2, 5]
    sampledf = sampledf.copy()
    sampledf.index = pathological_labels
    coo = construct_coo_matrix_from_sampledf(
        sampledf, COMBO_TO_IX, check_paths_exist=True)
    # If the fix is in place, after the internal reset_index the COO
    # has each species' data at its positional row. We verify by
    # checking that the stored COO value at row 0 column P decodes to
    # species 0, row 1 column P decodes to species 1, etc.
    csr = coo.tocsr()
    for (row_idx, col_idx), want in list(expected.items())[:6]:
        assert csr[row_idx, col_idx] == want, (
            f"scramble survived: at ({row_idx},{col_idx}) expected "
            f"{want} (species {row_idx}), got {csr[row_idx, col_idx]}")


def test_grid_verify_catches_a_planted_scramble(tmp_path):
    """Synthesize a broken COO by hand and confirm grid_verify_coo
    raises.
    """
    from egt.phylotreeumap import construct_coo_matrix_from_sampledf
    sampledf, gb_dir, expected = _build_fixture(tmp_path, n_species=6)
    coo = construct_coo_matrix_from_sampledf(
        sampledf, COMBO_TO_IX, check_paths_exist=True)
    # Scramble: swap rows 0 and 3 in the COO.
    csr = coo.tocsr().tolil()
    row0 = csr.getrow(0).toarray().ravel()
    row3 = csr.getrow(3).toarray().ravel()
    csr[0, :] = row3
    csr[3, :] = row0
    coo_scrambled = csr.tocsr().tocoo()
    with pytest.raises(AssertionError,
                        match=r"grid_verify_coo:.*cells mismatched"):
        grid_verify_coo(coo_scrambled, sampledf, COMBO_TO_IX, gb_dir,
                         n_rows=5, n_cols=5, strict=True)


def test_grid_verify_downsampled_rows_still_correct(tmp_path):
    """Downsample rows (drop half the species) and confirm the grid
    check still passes. Simulates the clade-subset code paths in the
    paper's analysis.
    """
    from egt.phylotreeumap import construct_coo_matrix_from_sampledf
    sampledf, gb_dir, expected = _build_fixture(tmp_path, n_species=8)
    coo = construct_coo_matrix_from_sampledf(
        sampledf, COMBO_TO_IX, check_paths_exist=True)
    # Keep only even-indexed species. After this, the downsampled
    # matrix's row i should correspond to sampledf_ds.iloc[i] — a
    # common trap when subsampling.
    keep = np.arange(0, len(sampledf), 2)
    csr = coo.tocsr()
    coo_ds = csr[keep, :].tocoo()
    sampledf_ds = sampledf.iloc[keep].reset_index(drop=True)
    summary = grid_verify_coo(coo_ds, sampledf_ds, COMBO_TO_IX, gb_dir,
                                n_rows=3, n_cols=3, strict=True)
    assert summary["failed"] == 0


def test_grid_indices_span_corners():
    """Unit test on the grid-index helper: endpoints are always there."""
    from egt._testing.grid_verify_coo import _grid_indices
    ix = _grid_indices(100, 5)
    assert ix[0] == 0
    assert ix[-1] == 99
    assert len(ix) == 5
    ix = _grid_indices(100, 200)        # k > n clamps
    assert len(ix) == 100
    ix = _grid_indices(100, 1)
    assert ix.tolist() == [0]
    ix = _grid_indices(0, 5)
    assert len(ix) == 1                  # clamped to min 1 index
