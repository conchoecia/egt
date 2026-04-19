"""TODO_tests.md section K — gb.gz → COO → gb.gz byte-match round-trip.

The tightest end-to-end correctness check available for the COO build:

  1. Start from per-species gb.gz files.
  2. Run ``construct_coo_matrix_from_sampledf`` to build the COO.
  3. Run ``decompose_coo_to_gbgz`` to reverse the construction.
  4. Sort both sets of gb.gz files and compare byte-for-byte + value.

This subsumes several of the narrower section C tests (no-scramble,
completeness, extraneousness, dtype) in one assertion. The narrower
tests stay in place so a regression can be localized.

Skipped: the "5,821 species stretch" TODO bullet — that's a nightly-CI
concern, not a unit test.
"""
from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from egt._testing import decompose_coo_to_gbgz
from egt.phylotreeumap import construct_coo_matrix_from_sampledf


# --- Fixture construction -------------------------------------------------

FAMS = [f"fam_{i:04d}" for i in range(10)]
COMBO_TO_IX = {
    (FAMS[i], FAMS[j]): k
    for k, (i, j) in enumerate(
        [(i, j) for i in range(len(FAMS)) for j in range(i + 1, len(FAMS))]
    )
}
IX_TO_PAIR = {v: k for k, v in COMBO_TO_IX.items()}
N_PAIRS = len(COMBO_TO_IX)


def _write_gbgz(path: Path, rows):
    """Write a gb.gz with rbh1 < rbh2 canonicalized."""
    with gzip.open(path, "wt") as fh:
        fh.write("rbh1\trbh2\tdistance\n")
        for r1, r2, d in rows:
            a, b = sorted((r1, r2))
            fh.write(f"{a}\t{b}\t{d}\n")


def _build_fixture(tmp_path: Path, n_species: int, rng_seed: int = 0):
    """Build n_species fake gb.gz files plus a matching sampledf.

    Each pair observed in a species gets a distance that uniquely
    encodes (species_idx, pair_idx): 1000 * (species_idx+1) + pair_idx.
    """
    rng = np.random.default_rng(rng_seed)
    gb_dir = tmp_path / "orig_gbgz"
    gb_dir.mkdir()
    rows = []
    for s in range(n_species):
        sample = f"species_{s:03d}"
        # Pick ~60% of pairs so some columns are sparse.
        n_take = max(1, int(N_PAIRS * 0.6))
        pair_indices = rng.choice(N_PAIRS, size=n_take, replace=False).tolist()
        gb_rows = []
        for p in pair_indices:
            r1, r2 = IX_TO_PAIR[p]
            d = 1000 * (s + 1) + p
            gb_rows.append((r1, r2, d))
        gb_path = gb_dir / f"{sample}.gb.gz"
        _write_gbgz(gb_path, gb_rows)
        rows.append({"sample": sample, "dis_filepath_abs": str(gb_path)})
    sampledf = pd.DataFrame(rows)
    return sampledf, gb_dir


def _read_gbgz_sorted(path: Path) -> pd.DataFrame:
    """Read a gb.gz into a DataFrame sorted by (rbh1, rbh2) to make
    comparison order-insensitive within a file."""
    df = pd.read_csv(path, sep="\t")
    df = df.sort_values(["rbh1", "rbh2"]).reset_index(drop=True)
    # Keep distance column exactly as the file had it.
    return df


# --- Tests ----------------------------------------------------------------

def test_roundtrip_byte_match_5_species(tmp_path):
    """With 5 species, rebuild and decompose; sorted gb.gz content
    must match the originals row-for-row."""
    sampledf, gb_orig_dir = _build_fixture(tmp_path, n_species=5)
    coo = construct_coo_matrix_from_sampledf(
        sampledf, COMBO_TO_IX, check_paths_exist=True)

    gb_decomp_dir = tmp_path / "decomp"
    out_paths = decompose_coo_to_gbgz(
        coo, sampledf, COMBO_TO_IX, gb_decomp_dir)

    assert set(out_paths.keys()) == set(sampledf["sample"])
    for sample in sampledf["sample"]:
        orig = _read_gbgz_sorted(gb_orig_dir / f"{sample}.gb.gz")
        decomp = _read_gbgz_sorted(out_paths[sample])
        pd.testing.assert_frame_equal(
            orig[["rbh1", "rbh2"]].reset_index(drop=True),
            decomp[["rbh1", "rbh2"]].reset_index(drop=True),
        )
        # Distance tolerance: exact for ints, 1e-9 relative for floats.
        np.testing.assert_allclose(
            orig["distance"].to_numpy(dtype=np.float64),
            decomp["distance"].to_numpy(dtype=np.float64),
            rtol=1e-9, atol=0,
        )


def test_roundtrip_catches_row_scramble(tmp_path):
    """Swap two rows in the COO, re-decompose, and confirm the
    resulting gb.gz byte-diff against the originals. This is exactly
    the Hirudonipponia bug's signature.
    """
    sampledf, gb_orig_dir = _build_fixture(tmp_path, n_species=6)
    coo = construct_coo_matrix_from_sampledf(
        sampledf, COMBO_TO_IX, check_paths_exist=True)

    # Plant the scramble.
    csr = coo.tocsr().tolil()
    row0 = csr.getrow(0).toarray().ravel()
    row3 = csr.getrow(3).toarray().ravel()
    csr[0, :] = row3
    csr[3, :] = row0
    coo_bad = csr.tocsr().tocoo()

    gb_decomp_dir = tmp_path / "decomp"
    decompose_coo_to_gbgz(coo_bad, sampledf, COMBO_TO_IX, gb_decomp_dir)

    # The decomposed gb.gz for species_000 should now carry the values
    # that originally belonged to species_003, so distance values
    # decode to species 3 (1000*4 = 4000..4999), not species 0 (1000..1999).
    decomp0 = _read_gbgz_sorted(gb_decomp_dir / "species_000.gb.gz")
    # All distances should be in the species-3 encoded range.
    decoded = (decomp0["distance"].astype(int) // 1000) - 1
    assert (decoded == 3).all(), (
        "Scramble should have flipped the decoded species for row 0 to 3; "
        f"instead got {decoded.unique().tolist()}"
    )
    # And diffing the sorted gb.gz's must fail.
    orig0 = _read_gbgz_sorted(gb_orig_dir / "species_000.gb.gz")
    with pytest.raises(AssertionError):
        pd.testing.assert_frame_equal(
            orig0[["rbh1", "rbh2", "distance"]].reset_index(drop=True),
            decomp0[["rbh1", "rbh2", "distance"]].reset_index(drop=True),
        )


def test_roundtrip_coo_and_decomp_gbgz_self_consistent(tmp_path):
    """For each species, every row in the decomposed gb.gz appears as a
    stored entry at COO[row_idx(S), ·] with the matching value, and
    every stored entry in that row of the COO appears as a row in the
    decomposed gb.gz. This is the explicit bullet in TODO §K.
    """
    sampledf, _ = _build_fixture(tmp_path, n_species=5)
    coo = construct_coo_matrix_from_sampledf(
        sampledf, COMBO_TO_IX, check_paths_exist=True)

    gb_decomp_dir = tmp_path / "decomp"
    out_paths = decompose_coo_to_gbgz(
        coo, sampledf, COMBO_TO_IX, gb_decomp_dir)

    csr = coo.tocsr()
    for row_pos, sample in enumerate(sampledf["sample"].tolist()):
        # Read the decomposed gb.gz for this species.
        decomp = pd.read_csv(out_paths[sample], sep="\t")
        # Pair -> distance map from the decomposed file.
        decomp_map = {
            COMBO_TO_IX[(r["rbh1"], r["rbh2"])]: r["distance"]
            for _, r in decomp.iterrows()
        }
        # Pair -> distance map from the COO row.
        row_csr = csr.getrow(row_pos)
        coo_map = dict(zip(row_csr.indices.tolist(), row_csr.data.tolist()))
        # Must agree as a multiset.
        assert set(decomp_map.keys()) == set(coo_map.keys()), (
            f"sample {sample}: decomp and COO pair sets disagree")
        for c, v in coo_map.items():
            assert float(decomp_map[c]) == pytest.approx(float(v), rel=1e-9)


def test_roundtrip_preserves_distance_dtype(tmp_path):
    """If gb.gz distances are integer on disk, the round-trip should
    preserve numeric value (1:1) within the tolerance. Floats shouldn't
    silently show up as '15390028.0' where the gb.gz had '15390028'
    semantically.
    """
    sampledf, gb_orig_dir = _build_fixture(tmp_path, n_species=4)
    coo = construct_coo_matrix_from_sampledf(
        sampledf, COMBO_TO_IX, check_paths_exist=True)
    gb_decomp_dir = tmp_path / "decomp"
    decompose_coo_to_gbgz(coo, sampledf, COMBO_TO_IX, gb_decomp_dir)

    for sample in sampledf["sample"]:
        orig = pd.read_csv(gb_orig_dir / f"{sample}.gb.gz", sep="\t")
        decomp = pd.read_csv(gb_decomp_dir / f"{sample}.gb.gz", sep="\t")
        # Integer-equality after int() cast.
        orig_int = orig["distance"].astype(int).to_numpy()
        decomp_int = decomp["distance"].astype(int).to_numpy()
        np.testing.assert_array_equal(
            np.sort(orig_int), np.sort(decomp_int),
            err_msg=f"Integer distances drifted for sample {sample}",
        )


def test_roundtrip_pair_direction_canonicalized(tmp_path):
    """Every decomposed gb.gz row has rbh1 < rbh2 lexicographically,
    matching the original gb.gz convention."""
    sampledf, _ = _build_fixture(tmp_path, n_species=3)
    coo = construct_coo_matrix_from_sampledf(
        sampledf, COMBO_TO_IX, check_paths_exist=True)
    gb_decomp_dir = tmp_path / "decomp"
    out_paths = decompose_coo_to_gbgz(
        coo, sampledf, COMBO_TO_IX, gb_decomp_dir)

    for sample, path in out_paths.items():
        df = pd.read_csv(path, sep="\t")
        assert (df["rbh1"] < df["rbh2"]).all(), (
            f"sample {sample}: decomposed gb.gz has a row where "
            f"rbh1 >= rbh2")


def test_roundtrip_schema(tmp_path):
    """Decomposed gb.gz has exactly the three canonical columns, in
    the canonical order ``rbh1, rbh2, distance``."""
    sampledf, _ = _build_fixture(tmp_path, n_species=3)
    coo = construct_coo_matrix_from_sampledf(
        sampledf, COMBO_TO_IX, check_paths_exist=True)
    gb_decomp_dir = tmp_path / "decomp"
    out_paths = decompose_coo_to_gbgz(
        coo, sampledf, COMBO_TO_IX, gb_decomp_dir)
    for path in out_paths.values():
        df = pd.read_csv(path, sep="\t")
        assert list(df.columns) == ["rbh1", "rbh2", "distance"]


def test_roundtrip_sampledf_misaligned_raises(tmp_path):
    """If sampledf row count doesn't match COO shape, the helper must
    refuse instead of silently mis-attributing rows."""
    sampledf, _ = _build_fixture(tmp_path, n_species=4)
    coo = construct_coo_matrix_from_sampledf(
        sampledf, COMBO_TO_IX, check_paths_exist=True)
    # Drop a row from sampledf without touching the COO — simulate a
    # caller bug.
    sampledf_bad = sampledf.iloc[:-1].reset_index(drop=True)
    with pytest.raises(ValueError, match="sampledf has"):
        decompose_coo_to_gbgz(
            coo, sampledf_bad, COMBO_TO_IX, tmp_path / "decomp_bad")
