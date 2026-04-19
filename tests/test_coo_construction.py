"""TODO_tests.md section C — COO construction.

Targets ``egt.phylotreeumap.construct_coo_matrix_from_sampledf``.

Builds a small synthetic corpus of gb.gz files in a tmp dir, runs the
COO builder, then checks:

- Shape == (n_species, n_pairs)
- No-row-scramble: every (species_i, pair_j) with a known value
  ends up at COO[i, j].
- Completeness: every gb.gz row appears exactly once.
- Extraneousness: no cells outside the gb.gz footprint.
- Dtype is numeric.
- Stored-zero semantics: genuine 0-distance rows get stored, so the
  "stored zeros are placeholders" convention is enforced by whatever
  filtering happens downstream (not by rbh_to_gb).

Complements ``test_coo_integrity.py`` (which focuses on scramble-bug
detection via grid_verify_coo) by asserting structural invariants
directly on the sparse matrix.
"""
from __future__ import annotations

import gzip
import numpy as np
import pandas as pd
import pytest
from pathlib import Path


FAMS = [f"fam_{i:03d}" for i in range(8)]  # 8 families -> C(8,2) = 28 pairs
COMBO_TO_IX = {
    (FAMS[i], FAMS[j]): k
    for k, (i, j) in enumerate(
        [(i, j) for i in range(len(FAMS)) for j in range(i + 1, len(FAMS))]
    )
}
IX_TO_PAIR = {v: k for k, v in COMBO_TO_IX.items()}
N_PAIRS = len(COMBO_TO_IX)


def _write_gbgz(path: Path, rows: list[tuple[str, str, int]]):
    """Write a gb.gz. rbh1/rbh2 are sorted so rbh1 < rbh2."""
    with gzip.open(path, "wt") as fh:
        fh.write("rbh1\trbh2\tdistance\n")
        for r1, r2, d in rows:
            a, b = sorted((r1, r2))
            fh.write(f"{a}\t{b}\t{d}\n")


def _build_fixture(tmp_path: Path, n_species: int = 5, rng_seed: int = 0):
    """Return (sampledf, gb_dir, expected) for a small tmp-dir fixture.

    Each species gets distances encoded as 1000*(species_idx+1) + pair_idx,
    so COO[i, j] is decodable back to (i, j). A species observes a
    random ~60% subset of all pairs — density < 1 exercises the
    "pair not present in species" path.
    """
    rng = np.random.default_rng(rng_seed)
    gb_dir = tmp_path / "gbgz"
    gb_dir.mkdir()
    rows = []
    expected = {}
    for s in range(n_species):
        sample = f"sp_{s:03d}"
        n_take = int(N_PAIRS * 0.6)
        pair_indices = rng.choice(N_PAIRS, size=n_take, replace=False).tolist()
        gb_rows = []
        for p in pair_indices:
            r1, r2 = IX_TO_PAIR[p]
            d = 1000 * (s + 1) + p   # uniquely encodes (species_idx, pair_idx)
            gb_rows.append((r1, r2, d))
            expected[(s, p)] = d
        gb_path = gb_dir / f"{sample}.gb.gz"
        _write_gbgz(gb_path, gb_rows)
        rows.append({"sample": sample, "dis_filepath_abs": str(gb_path)})
    sampledf = pd.DataFrame(rows)
    return sampledf, gb_dir, expected


# ---------- shape ---------------------------------------------------------

def test_coo_shape(tmp_path):
    from egt.phylotreeumap import construct_coo_matrix_from_sampledf
    sampledf, _, _ = _build_fixture(tmp_path, n_species=6)
    coo = construct_coo_matrix_from_sampledf(
        sampledf, COMBO_TO_IX, check_paths_exist=True)
    assert coo.shape == (len(sampledf), N_PAIRS)


# ---------- no-row-scramble ----------------------------------------------

def test_coo_values_at_expected_cells(tmp_path):
    """For every (species_idx, pair_idx) written into a gb.gz, COO[i, j]
    must equal the exact distance value."""
    from egt.phylotreeumap import construct_coo_matrix_from_sampledf
    sampledf, _, expected = _build_fixture(tmp_path, n_species=10)
    coo = construct_coo_matrix_from_sampledf(
        sampledf, COMBO_TO_IX, check_paths_exist=True)
    csr = coo.tocsr()
    for (row_idx, col_idx), want in expected.items():
        got = csr[row_idx, col_idx]
        assert got == want, (
            f"COO[{row_idx}, {col_idx}] == {got!r}, expected {want}. "
            f"Decoded from value: species={int(got)//1000 - 1}, "
            f"pair={int(got) % 1000}. This is the row-scramble signature.")


def test_coo_row_has_only_that_species_values(tmp_path):
    """For every COO row, the nonzero values must all decode back to the
    same species index. A row-scramble bug would mix species here."""
    from egt.phylotreeumap import construct_coo_matrix_from_sampledf
    sampledf, _, expected = _build_fixture(tmp_path, n_species=8)
    coo = construct_coo_matrix_from_sampledf(
        sampledf, COMBO_TO_IX, check_paths_exist=True)
    csr = coo.tocsr()
    for row_idx in range(csr.shape[0]):
        row = csr.getrow(row_idx).toarray().ravel()
        nonzero = row[row != 0]
        if len(nonzero) == 0:
            continue
        species_idx_decoded = nonzero.astype(int) // 1000 - 1
        # All non-zero values in row i should decode to species i.
        assert np.all(species_idx_decoded == row_idx), (
            f"row {row_idx} contains values decoding to species "
            f"{np.unique(species_idx_decoded).tolist()}; expected only "
            f"{row_idx}. Row-scramble detected.")


# ---------- completeness / extraneousness ---------------------------------

def test_coo_no_duplicate_cells(tmp_path):
    """Every (row, col) appears at most once in COO triplets."""
    from egt.phylotreeumap import construct_coo_matrix_from_sampledf
    sampledf, _, _ = _build_fixture(tmp_path, n_species=6)
    coo = construct_coo_matrix_from_sampledf(
        sampledf, COMBO_TO_IX, check_paths_exist=True)
    # Stack (row, col) pairs and look for duplicates.
    pairs = np.stack([coo.row, coo.col], axis=1)
    # Treat as structured view for uniqueness detection.
    view = pairs.view([("r", pairs.dtype), ("c", pairs.dtype)]).reshape(-1)
    unique = np.unique(view)
    assert unique.size == pairs.shape[0], (
        f"COO has {pairs.shape[0] - unique.size} duplicate (row, col) cells")


def test_coo_nnz_matches_expected_count(tmp_path):
    """nnz equals the union of gb.gz rows across species (no drops)."""
    from egt.phylotreeumap import construct_coo_matrix_from_sampledf
    sampledf, _, expected = _build_fixture(tmp_path, n_species=7)
    coo = construct_coo_matrix_from_sampledf(
        sampledf, COMBO_TO_IX, check_paths_exist=True)
    # All distances in the fixture are nonzero by construction
    # (1000*(s+1) + p with s>=0, so min is 1000). nnz should match
    # exactly the number of gb.gz rows.
    assert coo.nnz == len(expected)


def test_coo_has_no_extraneous_cells(tmp_path):
    """Every stored COO cell is present in the expected dict."""
    from egt.phylotreeumap import construct_coo_matrix_from_sampledf
    sampledf, _, expected = _build_fixture(tmp_path, n_species=6)
    coo = construct_coo_matrix_from_sampledf(
        sampledf, COMBO_TO_IX, check_paths_exist=True)
    for r, c, v in zip(coo.row, coo.col, coo.data):
        assert (int(r), int(c)) in expected, (
            f"COO has unexpected cell ({r}, {c}) = {v}")
        assert expected[(int(r), int(c))] == v, (
            f"COO cell ({r}, {c}) = {v}, expected "
            f"{expected[(int(r), int(c))]}")


# ---------- dtype ---------------------------------------------------------

def test_coo_value_dtype_is_numeric(tmp_path):
    from egt.phylotreeumap import construct_coo_matrix_from_sampledf
    sampledf, _, _ = _build_fixture(tmp_path, n_species=4)
    coo = construct_coo_matrix_from_sampledf(
        sampledf, COMBO_TO_IX, check_paths_exist=True)
    assert np.issubdtype(coo.dtype, np.number), \
        f"COO dtype {coo.dtype} is not numeric — would silently break sums"


# ---------- stored-zero semantics ----------------------------------------

def test_genuine_zero_distance_reaches_coo(tmp_path):
    """TODO_tests.md/C + Related: two orthologs at identical _pos in
    the RBH produce distance 0 in gb.gz. That 0 then gets stored into
    the COO — the builder does not drop it. Downstream code that
    treats "stored zeros" as placeholders must therefore rely on its
    own filter, not on the builder."""
    import scipy.sparse as sp
    from egt.phylotreeumap import construct_coo_matrix_from_sampledf
    gb_dir = tmp_path / "gb"
    gb_dir.mkdir()
    # Species 0: one pair at distance 0, one at distance 100.
    gb0 = gb_dir / "sp_000.gb.gz"
    _write_gbgz(gb0, [
        ("fam_000", "fam_001", 0),
        ("fam_000", "fam_002", 100),
    ])
    # Species 1: just a single nonzero pair so it has some content.
    gb1 = gb_dir / "sp_001.gb.gz"
    _write_gbgz(gb1, [
        ("fam_001", "fam_002", 250),
    ])
    sampledf = pd.DataFrame([
        {"sample": "sp_000", "dis_filepath_abs": str(gb0)},
        {"sample": "sp_001", "dis_filepath_abs": str(gb1)},
    ])
    coo = construct_coo_matrix_from_sampledf(
        sampledf, COMBO_TO_IX, check_paths_exist=True)
    # scipy COO keeps explicit zeros in .data until eliminate_zeros().
    # Look for (fam_000, fam_001) at species 0.
    col0_1 = COMBO_TO_IX[("fam_000", "fam_001")]
    triples = list(zip(coo.row.tolist(), coo.col.tolist(), coo.data.tolist()))
    assert (0, col0_1, 0) in triples, (
        "Genuine zero-distance observation was dropped during COO "
        "construction — stored-zero convention broken")


# ---------- pair missing from any species --------------------------------

def test_column_for_unobserved_pair_is_empty(tmp_path):
    """TODO_tests.md/H: a pair that appears in zero species still has a
    valid column (for stable column indexing); that column is just
    empty."""
    from egt.phylotreeumap import construct_coo_matrix_from_sampledf
    # Build a fixture that only ever touches pairs at col indices 0..4.
    # Cols 5..N_PAIRS-1 should be absent from the COO triplets.
    gb_dir = tmp_path / "gb"
    gb_dir.mkdir()
    touched_pairs = list(IX_TO_PAIR.keys())[:5]
    gb0 = gb_dir / "sp_000.gb.gz"
    _write_gbgz(gb0, [(r1, r2, 100 + i) for i, (r1, r2)
                       in enumerate([IX_TO_PAIR[k] for k in range(5)])])
    sampledf = pd.DataFrame([
        {"sample": "sp_000", "dis_filepath_abs": str(gb0)},
    ])
    coo = construct_coo_matrix_from_sampledf(
        sampledf, COMBO_TO_IX, check_paths_exist=True)
    # Shape must still cover all pairs.
    assert coo.shape == (1, N_PAIRS)
    # All stored cols are within [0, 4].
    assert coo.col.max() <= 4
    # Columns 5..N_PAIRS-1 are empty.
    csc = coo.tocsc()
    for unused_col in range(5, N_PAIRS):
        assert csc.getcol(unused_col).nnz == 0


# ---------- empty gb.gz does not corrupt ordering -------------------------

def test_empty_gbgz_keeps_row_ordering_intact(tmp_path):
    """TODO_tests.md/H: one species with an empty gb.gz must not shift
    downstream species' rows."""
    from egt.phylotreeumap import construct_coo_matrix_from_sampledf
    gb_dir = tmp_path / "gb"
    gb_dir.mkdir()
    # sp_000: empty; sp_001: has data; sp_002: has data.
    _write_gbgz(gb_dir / "sp_000.gb.gz", [])
    _write_gbgz(gb_dir / "sp_001.gb.gz",
                [("fam_000", "fam_001", 101)])
    _write_gbgz(gb_dir / "sp_002.gb.gz",
                [("fam_000", "fam_002", 202)])
    sampledf = pd.DataFrame([
        {"sample": "sp_000", "dis_filepath_abs": str(gb_dir / "sp_000.gb.gz")},
        {"sample": "sp_001", "dis_filepath_abs": str(gb_dir / "sp_001.gb.gz")},
        {"sample": "sp_002", "dis_filepath_abs": str(gb_dir / "sp_002.gb.gz")},
    ])
    coo = construct_coo_matrix_from_sampledf(
        sampledf, COMBO_TO_IX, check_paths_exist=True)
    assert coo.shape == (3, N_PAIRS)
    csr = coo.tocsr()
    # Row 0 is empty.
    assert csr.getrow(0).nnz == 0
    # Row 1 holds sp_001's value.
    col = COMBO_TO_IX[("fam_000", "fam_001")]
    assert csr[1, col] == 101
    # Row 2 holds sp_002's value.
    col = COMBO_TO_IX[("fam_000", "fam_002")]
    assert csr[2, col] == 202


# ---------- unknown pair key fails loudly ---------------------------------

def test_unknown_pair_raises_keyerror(tmp_path):
    """TODO_tests.md: if gb.gz has a pair not in combo_to_ix, the
    builder must fail loudly rather than silently drop the row."""
    from egt.phylotreeumap import construct_coo_matrix_from_sampledf
    gb_dir = tmp_path / "gb"
    gb_dir.mkdir()
    gb0 = gb_dir / "sp_000.gb.gz"
    _write_gbgz(gb0, [
        ("fam_000", "fam_001", 100),
        ("mystery_x", "mystery_y", 200),   # not in COMBO_TO_IX
    ])
    sampledf = pd.DataFrame([
        {"sample": "sp_000", "dis_filepath_abs": str(gb0)},
    ])
    with pytest.raises(KeyError, match="missing from alg_combo_to_ix"):
        construct_coo_matrix_from_sampledf(
            sampledf, COMBO_TO_IX, check_paths_exist=True)


# ---------- path-resolution overrides -------------------------------------

def test_gbgz_paths_dict_override_used(tmp_path):
    """If `gbgz_paths` is supplied, the builder reads from those paths
    (not `dis_filepath_abs`). Simulates Snakemake shadow copy."""
    from egt.phylotreeumap import construct_coo_matrix_from_sampledf
    # Write the REAL data to one dir and a dummy (wrong) path in the df.
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    wrong_path = tmp_path / "nonexistent.gb.gz"
    _write_gbgz(real_dir / "sp_000.gb.gz",
                [("fam_000", "fam_001", 777)])
    sampledf = pd.DataFrame([
        {"sample": "sp_000", "dis_filepath_abs": str(wrong_path)},
    ])
    gbgz_paths = {"sp_000": str(real_dir / "sp_000.gb.gz")}
    coo = construct_coo_matrix_from_sampledf(
        sampledf, COMBO_TO_IX, gbgz_paths=gbgz_paths,
        check_paths_exist=True)
    col = COMBO_TO_IX[("fam_000", "fam_001")]
    assert coo.tocsr()[0, col] == 777


def test_gbgz_paths_dict_missing_sample_raises(tmp_path):
    """If a sample listed in sampledf is missing from gbgz_paths, it's
    the same as a missing file — builder must surface the discrepancy."""
    from egt.phylotreeumap import construct_coo_matrix_from_sampledf
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    _write_gbgz(real_dir / "sp_000.gb.gz",
                [("fam_000", "fam_001", 100)])
    _write_gbgz(real_dir / "sp_001.gb.gz",
                [("fam_000", "fam_002", 100)])
    sampledf = pd.DataFrame([
        {"sample": "sp_000",
         "dis_filepath_abs": str(real_dir / "sp_000.gb.gz")},
        {"sample": "sp_001",
         "dis_filepath_abs": str(real_dir / "sp_001.gb.gz")},
    ])
    gbgz_paths = {"sp_000": str(real_dir / "sp_000.gb.gz")}   # sp_001 missing
    with pytest.raises(KeyError, match="missing in gbgz_paths"):
        construct_coo_matrix_from_sampledf(
            sampledf, COMBO_TO_IX, gbgz_paths=gbgz_paths,
            check_paths_exist=True)


def test_duplicate_sample_names_raise(tmp_path):
    """Duplicate sample names would cause dict-based file mapping to
    collide silently; builder must reject up front."""
    from egt.phylotreeumap import construct_coo_matrix_from_sampledf
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    _write_gbgz(real_dir / "sp_000.gb.gz",
                [("fam_000", "fam_001", 100)])
    # Two rows, same sample name.
    sampledf = pd.DataFrame([
        {"sample": "sp_000",
         "dis_filepath_abs": str(real_dir / "sp_000.gb.gz")},
        {"sample": "sp_000",
         "dis_filepath_abs": str(real_dir / "sp_000.gb.gz")},
    ])
    with pytest.raises(ValueError, match="Duplicate 'sample' values"):
        construct_coo_matrix_from_sampledf(
            sampledf, COMBO_TO_IX, check_paths_exist=True)


# ---------- npz round-trip (J partial) -----------------------------------

def test_coo_npz_persistence_roundtrip(tmp_path):
    """TODO_tests.md/J: save_npz → load_npz preserves shape, dtype, and
    every stored cell. Quick guard against scipy regressions."""
    from scipy.sparse import save_npz, load_npz
    from egt.phylotreeumap import construct_coo_matrix_from_sampledf
    sampledf, _, expected = _build_fixture(tmp_path, n_species=5)
    coo = construct_coo_matrix_from_sampledf(
        sampledf, COMBO_TO_IX, check_paths_exist=True)
    npz = tmp_path / "small_coo.npz"
    save_npz(str(npz), coo)
    reloaded = load_npz(str(npz))
    assert reloaded.shape == coo.shape
    assert reloaded.dtype == coo.dtype
    assert reloaded.nnz == coo.nnz
    # Cellwise equality via dense diff on this small fixture.
    assert (reloaded.toarray() == coo.toarray()).all()


# ---------- dense <-> sparse round-trip (J partial) -----------------------

def test_coo_csr_roundtrip_preserves_cells(tmp_path):
    """TODO_tests.md/J: coo.tocsr().tocoo() preserves every cell."""
    from egt.phylotreeumap import construct_coo_matrix_from_sampledf
    sampledf, _, _ = _build_fixture(tmp_path, n_species=5)
    coo = construct_coo_matrix_from_sampledf(
        sampledf, COMBO_TO_IX, check_paths_exist=True)
    rt = coo.tocsr().tocoo()
    # Dense equality is the cheapest complete check at this size.
    assert (rt.toarray() == coo.toarray()).all()
    assert rt.shape == coo.shape
