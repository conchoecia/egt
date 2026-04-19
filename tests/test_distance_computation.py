"""TODO_tests.md section B — Distance computation (RBH → gb.gz).

Targets ``egt.phylotreeumap.rbh_to_gb`` and the thin wrapper
``rbh_to_distance_gbgz``. Verifies:

- Same-scaffold condition is strict (no cross-scaffold pairs emitted).
- Reflexivity: identical positions produce distance 0.
- Symmetry: swapping rbh1/rbh2 in the input doesn't change the emitted
  distance; rbh1 < rbh2 in the output.
- Integer distances (no float drift).
- Empty-output edge cases (single gene per scaffold).

All fixtures are built in tmp dirs.
"""
from __future__ import annotations

import gzip
import pandas as pd
import pytest
from pathlib import Path


ALG = "BCnSSimakov2022"
SAMPLE = "spc1-12345-GCAtest.1"


def _rbhdf(rbhs, sample_scafs, sample_poss,
           alg_scafs=None, alg_poss=None, sample=SAMPLE):
    """Return a minimal RBH DataFrame."""
    n = len(rbhs)
    if alg_scafs is None:
        alg_scafs = ["algA"] * n
    if alg_poss is None:
        alg_poss = list(range(10, 10 + n))
    return pd.DataFrame({
        "rbh": list(rbhs),
        "gene_group": [f"gg_{i}" for i in range(n)],
        f"{ALG}_gene": [f"a_{i}" for i in range(n)],
        f"{ALG}_scaf": alg_scafs,
        f"{ALG}_pos":  alg_poss,
        f"{sample}_gene": [f"s_{i}" for i in range(n)],
        f"{sample}_scaf": sample_scafs,
        f"{sample}_pos":  sample_poss,
    })


def _read_gbgz(path: Path) -> pd.DataFrame:
    with gzip.open(path, "rt") as fh:
        return pd.read_csv(fh, sep="\t")


# ---------- rbh_to_gb: same-scaffold condition ----------------------------

def test_no_rows_for_different_scaffold_pairs(tmp_path):
    """Two families on different scaffolds -> gb.gz must be empty."""
    from egt.phylotreeumap import rbh_to_gb
    df = _rbhdf(
        rbhs=["fam_A", "fam_B"],
        sample_scafs=["scfX", "scfY"],
        sample_poss=[100, 200],
    )
    out = tmp_path / f"{SAMPLE}.gb.gz"
    rbh_to_gb(SAMPLE, df, str(out))
    gb = _read_gbgz(out)
    assert len(gb) == 0


def test_cross_scaffold_pairs_filtered_when_mixed(tmp_path):
    """Mixed: 3 genes, 2 on scfX and 1 on scfY. Only the one
    scfX-scfX pair should appear in gb.gz."""
    from egt.phylotreeumap import rbh_to_gb
    df = _rbhdf(
        rbhs=["fam_A", "fam_B", "fam_C"],
        sample_scafs=["scfX", "scfX", "scfY"],
        sample_poss=[100, 300, 500],
    )
    out = tmp_path / f"{SAMPLE}.gb.gz"
    rbh_to_gb(SAMPLE, df, str(out))
    gb = _read_gbgz(out)
    assert len(gb) == 1
    row = gb.iloc[0]
    assert {row["rbh1"], row["rbh2"]} == {"fam_A", "fam_B"}
    assert int(row["distance"]) == 200


def test_single_gene_per_scaffold_yields_empty(tmp_path):
    """If every scaffold has a single gene, no pairs are emitted."""
    from egt.phylotreeumap import rbh_to_gb
    df = _rbhdf(
        rbhs=["fam_A", "fam_B", "fam_C"],
        sample_scafs=["scf1", "scf2", "scf3"],
        sample_poss=[100, 200, 300],
    )
    out = tmp_path / f"{SAMPLE}.gb.gz"
    rbh_to_gb(SAMPLE, df, str(out))
    gb = _read_gbgz(out)
    assert len(gb) == 0


# ---------- rbh_to_gb: distances ------------------------------------------

def test_pairwise_distances_match_abs_diff(tmp_path):
    """All-on-one-scaffold: distances are |pos_i - pos_j| exactly."""
    from egt.phylotreeumap import rbh_to_gb
    poss = [100, 500, 1500]
    df = _rbhdf(
        rbhs=["fam_A", "fam_B", "fam_C"],
        sample_scafs=["scf1", "scf1", "scf1"],
        sample_poss=poss,
    )
    out = tmp_path / f"{SAMPLE}.gb.gz"
    rbh_to_gb(SAMPLE, df, str(out))
    gb = _read_gbgz(out)
    # 3-choose-2 = 3 pairs expected
    assert len(gb) == 3
    # Build pair -> distance from the output
    observed = {(row["rbh1"], row["rbh2"]): int(row["distance"])
                for _, row in gb.iterrows()}
    expected = {
        ("fam_A", "fam_B"): abs(poss[0] - poss[1]),
        ("fam_A", "fam_C"): abs(poss[0] - poss[2]),
        ("fam_B", "fam_C"): abs(poss[1] - poss[2]),
    }
    assert observed == expected


def test_reflexivity_identical_positions_gives_zero(tmp_path):
    """TODO_tests.md/B: two orthologs at the same _pos -> distance 0.
    This underpins the 'stored zero is a placeholder' convention — a
    genuine zero-distance observation DOES get written to gb.gz."""
    from egt.phylotreeumap import rbh_to_gb
    df = _rbhdf(
        rbhs=["fam_A", "fam_B"],
        sample_scafs=["scf1", "scf1"],
        sample_poss=[777, 777],
    )
    out = tmp_path / f"{SAMPLE}.gb.gz"
    rbh_to_gb(SAMPLE, df, str(out))
    gb = _read_gbgz(out)
    assert len(gb) == 1
    assert int(gb["distance"].iloc[0]) == 0


def test_symmetry_rbh_input_order_does_not_change_distance(tmp_path):
    """TODO_tests.md/B: swapping two rbh rows in the INPUT must not
    change the emitted distance. Also confirms rbh1 < rbh2 in output."""
    from egt.phylotreeumap import rbh_to_gb
    df_ab = _rbhdf(
        rbhs=["fam_A", "fam_B"],
        sample_scafs=["scf1", "scf1"],
        sample_poss=[100, 5000],
    )
    df_ba = _rbhdf(
        rbhs=["fam_B", "fam_A"],
        sample_scafs=["scf1", "scf1"],
        sample_poss=[5000, 100],
    )
    out_ab = tmp_path / "ab.gb.gz"
    out_ba = tmp_path / "ba.gb.gz"
    rbh_to_gb(SAMPLE, df_ab, str(out_ab))
    rbh_to_gb(SAMPLE, df_ba, str(out_ba))
    gb_ab = _read_gbgz(out_ab)
    gb_ba = _read_gbgz(out_ba)
    # Both should have one row with distance 4900, rbh1 < rbh2.
    for gb in (gb_ab, gb_ba):
        assert len(gb) == 1
        row = gb.iloc[0]
        assert row["rbh1"] < row["rbh2"]
        assert row["rbh1"] == "fam_A"
        assert row["rbh2"] == "fam_B"
        assert int(row["distance"]) == 4900


def test_output_pair_order_is_lexicographic(tmp_path):
    """rbh_to_gb sorts rbh1 < rbh2 on output regardless of input order."""
    from egt.phylotreeumap import rbh_to_gb
    # 4 families on one scaffold, names in a non-lex order.
    df = _rbhdf(
        rbhs=["fam_D", "fam_A", "fam_C", "fam_B"],
        sample_scafs=["scf1"] * 4,
        sample_poss=[10, 20, 30, 40],
    )
    out = tmp_path / f"{SAMPLE}.gb.gz"
    rbh_to_gb(SAMPLE, df, str(out))
    gb = _read_gbgz(out)
    assert (gb["rbh1"] < gb["rbh2"]).all()


def test_distance_dtype_is_integer(tmp_path):
    """TODO_tests.md/B: the gb.gz `distance` column must be integer
    (bp). Float drift on write-then-read would betray hidden coercion
    somewhere in the pipeline."""
    from egt.phylotreeumap import rbh_to_gb
    df = _rbhdf(
        rbhs=["fam_A", "fam_B", "fam_C"],
        sample_scafs=["scf1", "scf1", "scf1"],
        sample_poss=[100, 1000, 10000],
    )
    out = tmp_path / f"{SAMPLE}.gb.gz"
    rbh_to_gb(SAMPLE, df, str(out))
    gb = _read_gbgz(out)
    # Parsed back, distance should be int64 (no decimal point in file).
    assert pd.api.types.is_integer_dtype(gb["distance"]), \
        f"distance dtype on reload was {gb['distance'].dtype}, expected int"
    # Raw bytes: no decimal points in the distance column.
    with gzip.open(out, "rt") as fh:
        text = fh.read()
    # Skip the header; check the distance column for any '.'.
    lines = text.strip().splitlines()
    for line in lines[1:]:
        dist_str = line.split("\t")[-1]
        assert "." not in dist_str, \
            f"distance column contains '.' in gb.gz: {dist_str!r}"


def test_large_distance_fits_without_overflow(tmp_path):
    """Distances up to ~10^8 bp must round-trip losslessly (human chr
    sizes). Catches silent float32 truncation."""
    from egt.phylotreeumap import rbh_to_gb
    big = 150_000_000
    df = _rbhdf(
        rbhs=["fam_A", "fam_B"],
        sample_scafs=["scf1", "scf1"],
        sample_poss=[0, big],
    )
    out = tmp_path / f"{SAMPLE}.gb.gz"
    rbh_to_gb(SAMPLE, df, str(out))
    gb = _read_gbgz(out)
    assert int(gb["distance"].iloc[0]) == big


# ---------- rbh_to_gb: multi-scaffold case --------------------------------

def test_two_scaffolds_emits_only_intra_pairs(tmp_path):
    """Two scaffolds, two genes each: emit 1 pair per scaffold = 2 total."""
    from egt.phylotreeumap import rbh_to_gb
    df = _rbhdf(
        rbhs=["fam_A", "fam_B", "fam_C", "fam_D"],
        sample_scafs=["scf1", "scf1", "scf2", "scf2"],
        sample_poss=[100, 500, 1000, 1300],
    )
    out = tmp_path / f"{SAMPLE}.gb.gz"
    rbh_to_gb(SAMPLE, df, str(out))
    gb = _read_gbgz(out)
    assert len(gb) == 2
    observed = {(row["rbh1"], row["rbh2"]): int(row["distance"])
                for _, row in gb.iterrows()}
    assert observed == {
        ("fam_A", "fam_B"): 400,
        ("fam_C", "fam_D"): 300,
    }


# ---------- rbh_to_distance_gbgz (thin wrapper) ---------------------------

def test_rbh_to_distance_gbgz_end_to_end(tmp_path):
    """Full RBH file → gb.gz via the production entry point.
    Fixture path encodes a valid taxid to satisfy the filename parser."""
    from egt.phylotreeumap import rbh_to_distance_gbgz
    # Filename must pass the "spc-<integer>-" taxid check.
    rbh_path = tmp_path / f"{SAMPLE}.rbh"
    df = _rbhdf(
        rbhs=["fam_A", "fam_B", "fam_C"],
        sample_scafs=["scf1", "scf1", "scf2"],
        sample_poss=[100, 400, 50],
    )
    df.to_csv(rbh_path, sep="\t", index=False)
    out = tmp_path / f"{SAMPLE}.gb.gz"
    rbh_to_distance_gbgz(str(rbh_path), str(out), ALG)
    assert out.exists()
    gb = _read_gbgz(out)
    assert len(gb) == 1
    row = gb.iloc[0]
    assert {row["rbh1"], row["rbh2"]} == {"fam_A", "fam_B"}
    assert int(row["distance"]) == 300


def test_rbh_to_distance_gbgz_bad_filename_raises(tmp_path):
    """rbh_to_distance_gbgz parses a taxid out of the RBH filename.
    A filename with a non-integer in the taxid slot must raise."""
    from egt.phylotreeumap import rbh_to_distance_gbgz
    bad = tmp_path / "spc-NOTINT-GCA.rbh"
    # contents can be valid; filename-parse fails first.
    df = _rbhdf(
        rbhs=["fam_A", "fam_B"],
        sample_scafs=["scf1", "scf1"],
        sample_poss=[0, 100],
    )
    df.to_csv(bad, sep="\t", index=False)
    out = tmp_path / "out.gb.gz"
    with pytest.raises(ValueError, match="non-numeric character in the taxid"):
        rbh_to_distance_gbgz(str(bad), str(out), ALG)


def test_rbh_to_distance_gbgz_bad_suffix_raises(tmp_path):
    """Output file must end with .gb.gz by contract."""
    from egt.phylotreeumap import rbh_to_distance_gbgz
    rbh_path = tmp_path / f"{SAMPLE}.rbh"
    df = _rbhdf(
        rbhs=["fam_A", "fam_B"],
        sample_scafs=["scf1", "scf1"],
        sample_poss=[0, 100],
    )
    df.to_csv(rbh_path, sep="\t", index=False)
    with pytest.raises(IOError, match=r"does not end in \.gb\.gz"):
        rbh_to_distance_gbgz(str(rbh_path), str(tmp_path / "out.txt"), ALG)
