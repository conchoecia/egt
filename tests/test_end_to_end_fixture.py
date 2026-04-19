"""TODO_tests.md section G — end-to-end golden fixture.

Runs the full RBH -> gb.gz -> COO -> defining-features chain on a
tiny synthetic fixture (5 fake species, a modest number of pairs,
known distances, known clade membership) and diffs the emitted
per-clade TSV against a hand-computed golden DataFrame.

A hand-computable fixture is used rather than a blob on disk so the
expected values are legible in-source. If the shape of the fixture
ever changes, the golden gets regenerated in the test itself.

Skipped: the "Golden SupplementaryTable_16 snapshot" bullet — that's
too coupled to the production dataset to be a useful unit test.
"""
from __future__ import annotations

import gzip
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import save_npz

from egt.defining_features import process_coo_file
from egt.phylotreeumap import (
    construct_coo_matrix_from_sampledf,
    rbh_to_gb,
)


ALG = "BCnSSimakov2022"
# 5 species, each gets a real-ish sample name. The sample key must
# contain a '-' with an integer taxid in the middle if it ever gets
# routed through rbh_to_distance_gbgz, but here we call rbh_to_gb
# directly so the exact name doesn't matter. Using recognizable names
# so diagnostics are legible.
SAMPLES = [
    "spA-1001-GCAtest.1",
    "spB-1002-GCAtest.1",
    "spC-1003-GCAtest.1",
    "spD-1004-GCAtest.1",
    "spE-1005-GCAtest.1",
]

# 6 gene families -> 15 pairs.
FAMS = [f"fam_{i:04d}" for i in range(6)]
COMBO_TO_IX = {
    (FAMS[i], FAMS[j]): k
    for k, (i, j) in enumerate(
        [(i, j) for i in range(len(FAMS)) for j in range(i + 1, len(FAMS))]
    )
}
N_PAIRS = len(COMBO_TO_IX)


def _synth_rbhdf(sample: str, positions: dict[str, tuple[str, int]]):
    """Return a minimal RBH DataFrame for `sample`.

    `positions` maps rbh -> (scaffold, pos). Other required columns
    are filled with deterministic strings so rbh_to_gb sees a
    well-formed frame.
    """
    fams = list(positions.keys())
    n = len(fams)
    scafs = [positions[f][0] for f in fams]
    poss  = [positions[f][1] for f in fams]
    return pd.DataFrame({
        "rbh": fams,
        "gene_group": [f"gg_{i}" for i in range(n)],
        f"{ALG}_gene": [f"a_{i}" for i in range(n)],
        f"{ALG}_scaf": [f"alg{i % 3}" for i in range(n)],
        f"{ALG}_pos":  list(range(n)),
        f"{sample}_gene": [f"g_{i}" for i in range(n)],
        f"{sample}_scaf": scafs,
        f"{sample}_pos":  poss,
    })


# Golden layout: 5 species, 6 families, all on one scaffold ("scfA")
# except species E which puts family 5 on a different scaffold so the
# (fam_0005, ·) pair is unobserved in E.
#
# In-clade taxid 100 is carried by species A, B, C (3 in-clade, 2 out).
SAMPLE_POSITIONS = {
    "spA-1001-GCAtest.1": {
        "fam_0000": ("scfA", 100),
        "fam_0001": ("scfA", 200),
        "fam_0002": ("scfA", 400),
        "fam_0003": ("scfA", 800),
        "fam_0004": ("scfA", 1600),
        "fam_0005": ("scfA", 3200),
    },
    "spB-1002-GCAtest.1": {
        "fam_0000": ("scfA", 150),
        "fam_0001": ("scfA", 250),
        "fam_0002": ("scfA", 500),
        "fam_0003": ("scfA", 1000),
        "fam_0004": ("scfA", 2000),
        "fam_0005": ("scfA", 4000),
    },
    "spC-1003-GCAtest.1": {
        "fam_0000": ("scfA", 120),
        "fam_0001": ("scfA", 220),
        "fam_0002": ("scfA", 450),
        "fam_0003": ("scfA", 900),
        "fam_0004": ("scfA", 1800),
        "fam_0005": ("scfA", 3600),
    },
    "spD-1004-GCAtest.1": {
        "fam_0000": ("scfA", 1000),
        "fam_0001": ("scfA", 5000),
        "fam_0002": ("scfA", 10000),
        "fam_0003": ("scfA", 20000),
        "fam_0004": ("scfA", 40000),
        "fam_0005": ("scfA", 80000),
    },
    "spE-1005-GCAtest.1": {
        "fam_0000": ("scfA", 900),
        "fam_0001": ("scfA", 4500),
        "fam_0002": ("scfA", 9000),
        "fam_0003": ("scfA", 18000),
        "fam_0004": ("scfA", 36000),
        # different scaffold -> no pair involving fam_0005 from E appears
        "fam_0005": ("scfZ", 100),
    },
}

# Taxid 100 is in-clade for A/B/C, out-clade for D/E; taxid 200 is
# carried by every species (whole-dataset clade -- skipped).
TAXID_LISTS = [
    [100, 200],   # A: in 100, in 200
    [100, 200],   # B: in 100
    [100, 200],   # C: in 100
    [300, 200],   # D: out of 100
    [300, 200],   # E: out of 100
]


def _build_fixture(tmp_path: Path):
    """Materialize the fixture on disk: 5 .gb.gz files + sampledf.tsv +
    combo_to_index.txt + .coo.npz. Returns the paths and the sampledf.
    """
    gb_dir = tmp_path / "gbgz"
    gb_dir.mkdir()
    for sample, pos_map in SAMPLE_POSITIONS.items():
        rbhdf = _synth_rbhdf(sample, pos_map)
        rbh_to_gb(sample, rbhdf, str(gb_dir / f"{sample}.gb.gz"))

    sampledf = pd.DataFrame([
        {"sample": s, "dis_filepath_abs": str(gb_dir / f"{s}.gb.gz")}
        for s in SAMPLES
    ])
    # Attach taxid_list for the defining-features step later.
    sampledf["taxid_list"] = [str(tl) for tl in TAXID_LISTS]

    coo = construct_coo_matrix_from_sampledf(
        sampledf, COMBO_TO_IX, check_paths_exist=True)
    coo_path = tmp_path / "allsamples.coo.npz"
    save_npz(str(coo_path), coo)

    sampledf_path = tmp_path / "sampledf.tsv"
    # process_coo_file reads index_col=0, so give it an index.
    sampledf_for_disk = sampledf.copy()
    sampledf_for_disk.index.name = "idx"
    sampledf_for_disk.to_csv(sampledf_path, sep="\t")

    combo_path = tmp_path / "combo_to_index.txt"
    with open(combo_path, "w") as fh:
        for (a, b), ix in COMBO_TO_IX.items():
            fh.write(f"('{a}', '{b}')\t{ix}\n")

    return {
        "gb_dir": gb_dir,
        "sampledf": sampledf,
        "sampledf_path": sampledf_path,
        "combo_path": combo_path,
        "coo_path": coo_path,
        "coo": coo,
    }


# --- Golden: hand-compute expected per-pair stats -------------------------

def _golden_in_clade_stats():
    """Build the expected per-pair dataframe for taxid 100 (in-clade
    = {A, B, C}, out-clade = {D, E}). Values are computed directly
    from SAMPLE_POSITIONS so this function stays honest if the
    positions dict is edited.
    """
    in_samples = SAMPLES[:3]
    out_samples = SAMPLES[3:]
    n_in = len(in_samples)
    n_out = len(out_samples)

    # For each pair, compute the observed distances in each group.
    records = []
    for (fam1, fam2), pair_idx in COMBO_TO_IX.items():
        in_dists, out_dists = [], []
        for grp, sample_list in (
            (in_dists, in_samples), (out_dists, out_samples)
        ):
            for s in sample_list:
                pos = SAMPLE_POSITIONS[s]
                if pos[fam1][0] == pos[fam2][0]:   # same scaffold
                    grp.append(abs(pos[fam1][1] - pos[fam2][1]))
        notna_in = len(in_dists)
        notna_out = len(out_dists)
        if notna_in == 0:
            continue   # process_coo_file drops these via keep = notna_in > 0
        in_arr = np.asarray(in_dists, dtype=np.float64)
        out_arr = np.asarray(out_dists, dtype=np.float64)
        records.append({
            "pair": pair_idx,
            "notna_in": notna_in,
            "notna_out": notna_out,
            "mean_in": float(in_arr.mean()),
            "sd_in": (pd.Series(in_arr).std(ddof=1)
                      if notna_in >= 2 else np.nan),
            "mean_out": (float(out_arr.mean())
                         if notna_out > 0 else np.nan),
            "sd_out": (pd.Series(out_arr).std(ddof=1)
                       if notna_out >= 2 else np.nan),
            "occupancy_in":  notna_in / n_in,
            "occupancy_out": notna_out / n_out,
        })
    return pd.DataFrame(records).sort_values("pair").reset_index(drop=True)


# --- Tests ----------------------------------------------------------------

def test_gbgz_files_generated(tmp_path):
    """Each species gets a gb.gz with the correct schema and row count."""
    fx = _build_fixture(tmp_path)
    # spA-C / spD have 6 fams all on scfA -> C(6, 2) = 15 rows.
    # spE has 5 fams on scfA (fam_0005 is on scfZ alone) -> C(5, 2) = 10 rows.
    expected_row_counts = {
        "spA-1001-GCAtest.1": 15,
        "spB-1002-GCAtest.1": 15,
        "spC-1003-GCAtest.1": 15,
        "spD-1004-GCAtest.1": 15,
        "spE-1005-GCAtest.1": 10,
    }
    for sample, want in expected_row_counts.items():
        gb = pd.read_csv(fx["gb_dir"] / f"{sample}.gb.gz", sep="\t")
        assert len(gb) == want, (
            f"{sample}: got {len(gb)} rows, expected {want}")
        assert list(gb.columns) == ["rbh1", "rbh2", "distance"]


def test_coo_shape_and_nnz(tmp_path):
    """COO shape matches (n_species, n_pairs); nnz is the sum of gb.gz
    row counts across species."""
    fx = _build_fixture(tmp_path)
    coo = fx["coo"]
    assert coo.shape == (len(SAMPLES), N_PAIRS)
    # 15 + 15 + 15 + 15 + 10 = 70
    assert coo.nnz == 70


def test_coo_per_pair_distances_match_hand_compute(tmp_path):
    """For every stored COO cell, the value equals abs(pos_x - pos_y)
    computed directly from SAMPLE_POSITIONS for the same species+pair."""
    fx = _build_fixture(tmp_path)
    csr = fx["coo"].tocsr()
    for species_idx, sample in enumerate(SAMPLES):
        pos_map = SAMPLE_POSITIONS[sample]
        for (fam1, fam2), pair_idx in COMBO_TO_IX.items():
            s1, p1 = pos_map[fam1]
            s2, p2 = pos_map[fam2]
            if s1 != s2:
                # Different scaffolds: cell must be unstored (reads as 0).
                assert csr[species_idx, pair_idx] == 0
            else:
                want = abs(p1 - p2)
                got = csr[species_idx, pair_idx]
                assert int(got) == want, (
                    f"{sample} [{fam1},{fam2}]: expected {want}, got {got}")


def test_defining_features_output_matches_golden(tmp_path):
    """Run the full defining-features step and compare against a
    hand-computed golden DataFrame."""
    fx = _build_fixture(tmp_path)
    cwd_before = os.getcwd()
    os.chdir(tmp_path)
    try:
        process_coo_file(
            str(fx["sampledf_path"]),
            str(fx["combo_path"]),
            str(fx["coo_path"]),
            dfoutfilepath="unused.df",
            taxid_list=[100],
        )
    finally:
        os.chdir(cwd_before)
    # Find the emitted per-clade tsv.
    matches = list(Path(tmp_path).glob("*_100_unique_pair_df.tsv.gz"))
    assert matches, "no per-clade tsv emitted for taxid 100"
    got = pd.read_csv(matches[0], sep="\t").sort_values("pair").reset_index(
        drop=True)
    want = _golden_in_clade_stats()
    # Schema matches.
    assert list(got.columns) == list(want.columns)
    # Same set of pairs.
    assert set(got["pair"]) == set(want["pair"])
    # Joined comparison of each column.
    merged = got.merge(want, on="pair", suffixes=("_g", "_w"))
    assert (merged["notna_in_g"] == merged["notna_in_w"]).all()
    assert (merged["notna_out_g"] == merged["notna_out_w"]).all()
    for col in ("mean_in", "sd_in", "mean_out", "sd_out",
                "occupancy_in", "occupancy_out"):
        g = merged[f"{col}_g"].to_numpy(dtype=np.float64)
        w = merged[f"{col}_w"].to_numpy(dtype=np.float64)
        both_nan = np.isnan(g) & np.isnan(w)
        close = np.isclose(g, w, rtol=1e-9, atol=1e-9, equal_nan=False)
        ok = both_nan | close
        assert ok.all(), (
            f"col {col!r} mismatch:\n"
            f"got:  {g}\nwant: {w}"
        )


def test_unobserved_pair_dropped_from_per_clade_tsv(tmp_path):
    """Pair (fam_0004, fam_0005) involves fam_0005 which is on a
    different scaffold in species E, but is still observed in A/B/C/D.
    The per-clade tsv for taxid 100 should include this pair with
    notna_in=3, notna_out=1 (only D of the out-clade observed it; E did
    not)."""
    fx = _build_fixture(tmp_path)
    cwd_before = os.getcwd()
    os.chdir(tmp_path)
    try:
        process_coo_file(
            str(fx["sampledf_path"]), str(fx["combo_path"]),
            str(fx["coo_path"]), dfoutfilepath="unused.df",
            taxid_list=[100])
    finally:
        os.chdir(cwd_before)
    got = pd.read_csv(
        list(Path(tmp_path).glob("*_100_unique_pair_df.tsv.gz"))[0],
        sep="\t",
    )
    pair_idx = COMBO_TO_IX[("fam_0004", "fam_0005")]
    row = got[got["pair"] == pair_idx]
    assert len(row) == 1
    assert int(row.iloc[0]["notna_in"]) == 3
    assert int(row.iloc[0]["notna_out"]) == 1
