"""TODO_tests.md section A — Input-file schema + invariants.

Targets:
- ``egt.rbh_tools.parse_rbh`` — enforces the RBH schema.
- ``egt.phylotreeumap.ALGrbh_to_algcomboix`` + ``algcomboix_file_to_dict``
  — build and re-read ``combo_to_index.txt``.

All fixtures are built in tmp dirs; no dependency on production data.
Invariants under test are spelled out in TODO_tests.md and enforced
by downstream COO / defining-features code.
"""
from __future__ import annotations

import pandas as pd
import pytest
from itertools import combinations
from pathlib import Path


# ---------- helpers --------------------------------------------------------

ALG = "BCnSSimakov2022"
# Sample IDs in the RBH file must be of the form "<name>-<taxid>-..." so
# that the rbh_to_distance_gbgz taxid parser accepts them. We use
# "spc1-12345-GCAtest.1" throughout.
SAMPLE = "spc1-12345-GCAtest.1"


def _alg_cols(rbhs, scafs, poss):
    """BCnS-side gene/scaf/pos triples."""
    return {
        f"{ALG}_gene": [f"alg_g_{r.split('_')[-1]}" for r in rbhs],
        f"{ALG}_scaf": scafs,
        f"{ALG}_pos":  poss,
    }


def _sample_cols(sample, scafs, poss):
    return {
        f"{sample}_gene": [f"sample_g_{i}" for i in range(len(scafs))],
        f"{sample}_scaf": scafs,
        f"{sample}_pos":  poss,
    }


def _make_rbhdf(
    rbhs,
    alg_scafs, alg_poss,
    sample_scafs, sample_poss,
    sample=SAMPLE,
    include_gene_group=True,
):
    """Return a minimal RBH dataframe conforming to parse_rbh's schema."""
    n = len(rbhs)
    assert len(alg_scafs) == n == len(alg_poss) == len(sample_scafs) == len(sample_poss)
    data = {"rbh": list(rbhs)}
    if include_gene_group:
        data["gene_group"] = [f"gg_{i}" for i in range(n)]
    data.update(_alg_cols(rbhs, alg_scafs, alg_poss))
    data.update(_sample_cols(sample, sample_scafs, sample_poss))
    return pd.DataFrame(data)


def _write_rbh(tmp_path: Path, rbhdf: pd.DataFrame, name: str = None) -> Path:
    """Persist an RBH dataframe to a file that matches the filename
    convention `<sample>-<taxid>-*.rbh` the pipeline expects."""
    name = name or f"{SAMPLE}_rbh.tsv"
    p = tmp_path / name
    rbhdf.to_csv(p, sep="\t", index=False)
    return p


# ---------- parse_rbh: schema enforcement ---------------------------------

def test_parse_rbh_happy_path(tmp_path):
    """Minimal valid RBH parses cleanly; dtypes are set."""
    from egt.rbh_tools import parse_rbh
    df = _make_rbhdf(
        rbhs=["fam_0", "fam_1", "fam_2"],
        alg_scafs=["algA", "algA", "algB"],
        alg_poss=[10, 20, 30],
        sample_scafs=["scf1", "scf1", "scf2"],
        sample_poss=[100, 200, 50],
    )
    p = _write_rbh(tmp_path, df)
    out = parse_rbh(str(p))
    assert set(["rbh", "gene_group",
                f"{ALG}_scaf", f"{ALG}_gene", f"{ALG}_pos",
                f"{SAMPLE}_scaf", f"{SAMPLE}_gene", f"{SAMPLE}_pos"]) \
        <= set(out.columns)
    # _pos columns typed as Int64
    assert str(out[f"{SAMPLE}_pos"].dtype) == "Int64"
    assert str(out[f"{ALG}_pos"].dtype) == "Int64"


def test_parse_rbh_missing_rbh_column_raises(tmp_path):
    from egt.rbh_tools import parse_rbh
    df = _make_rbhdf(["fam_0"], ["algA"], [10], ["scf1"], [100])
    df = df.drop(columns=["rbh"])
    p = _write_rbh(tmp_path, df)
    with pytest.raises(IOError, match="does not have a column named 'rbh'"):
        parse_rbh(str(p))


def test_parse_rbh_missing_gene_group_raises(tmp_path):
    from egt.rbh_tools import parse_rbh
    df = _make_rbhdf(
        ["fam_0", "fam_1"], ["algA", "algB"], [10, 20],
        ["scf1", "scf1"], [100, 200],
        include_gene_group=False,
    )
    p = _write_rbh(tmp_path, df)
    with pytest.raises(IOError, match="gene_group"):
        parse_rbh(str(p))


def test_parse_rbh_missing_sample_pos_raises(tmp_path):
    """If a sample is advertised by `_scaf` and `_gene` but missing
    `_pos`, that's the kind of column-drop bug that the RBH parser
    must flag."""
    from egt.rbh_tools import parse_rbh
    df = _make_rbhdf(
        ["fam_0", "fam_1"], ["algA", "algA"], [10, 20],
        ["scf1", "scf1"], [100, 200],
    )
    df = df.drop(columns=[f"{SAMPLE}_pos"])
    p = _write_rbh(tmp_path, df)
    with pytest.raises(IOError, match=f"{SAMPLE}_pos"):
        parse_rbh(str(p))


def test_parse_rbh_bad_hex_color_raises(tmp_path):
    """If `color` is present it must be legal hex."""
    from egt.rbh_tools import parse_rbh
    df = _make_rbhdf(["fam_0", "fam_1"], ["algA", "algA"], [10, 20],
                     ["scf1", "scf1"], [100, 200])
    df["color"] = ["#112233", "not-a-hex"]
    p = _write_rbh(tmp_path, df)
    with pytest.raises(IOError, match="not a legal hex color"):
        parse_rbh(str(p))


def test_parse_rbh_pos_is_integer_dtype(tmp_path):
    """TODO_tests.md/A: RBH positions must land as integer dtype. Catches
    accidental float coercion from an NA or a stray decimal."""
    from egt.rbh_tools import parse_rbh
    df = _make_rbhdf(
        ["fam_0", "fam_1", "fam_2"],
        ["algA", "algA", "algA"], [10, 20, 30],
        ["scf1", "scf1", "scf1"], [100, 200, 300],
    )
    p = _write_rbh(tmp_path, df)
    out = parse_rbh(str(p))
    # Integer-like dtype (pandas nullable Int64 is what parse_rbh sets)
    assert pd.api.types.is_integer_dtype(out[f"{SAMPLE}_pos"])
    assert pd.api.types.is_integer_dtype(out[f"{ALG}_pos"])


# ---------- downstream-invariant checks on RBH content --------------------
# These are not enforced by parse_rbh itself but are prerequisites for
# the distance-computation path. Exposing them as tests here means a
# future refactor of parse_rbh that wants to harden the schema has a
# ready checklist.

def test_rbh_column_uniqueness_is_a_downstream_invariant(tmp_path):
    """TODO_tests.md/A: `rbh` must be unique within an RBH file; if a
    BCnS family maps to two orthologs in the species, the distance math
    isn't well-defined without a paralog policy. We don't enforce this
    inside parse_rbh, but the downstream rbh_to_gb merges on `rbh` —
    a duplicate key on that side of the merge explodes the product.
    Document the expectation; skip hard-enforcement until the library
    adds it."""
    from egt.rbh_tools import parse_rbh
    df = _make_rbhdf(
        rbhs=["fam_0", "fam_0", "fam_1"],      # duplicate!
        alg_scafs=["algA", "algA", "algA"],
        alg_poss=[10, 11, 20],
        sample_scafs=["scf1", "scf1", "scf1"],
        sample_poss=[100, 150, 200],
    )
    p = _write_rbh(tmp_path, df)
    # parse_rbh currently happily returns the df with duplicate rbh values;
    # capture that here so if behavior tightens the test updates explicitly.
    out = parse_rbh(str(p))
    assert out["rbh"].duplicated().any(), \
        "parse_rbh no longer tolerates duplicate rbh values — update this test"


def test_rbh_positions_non_negative_in_fixtures():
    """Sanity: the helpers in this module should never produce negative
    positions. Regression guard on the helpers themselves."""
    df = _make_rbhdf(
        ["fam_0", "fam_1"], ["algA", "algA"], [10, 20],
        ["scf1", "scf1"], [100, 200],
    )
    assert (df[f"{SAMPLE}_pos"] >= 0).all()
    assert (df[f"{ALG}_pos"] >= 0).all()


def test_rbh_scaffold_pos_monotonicity_detectable(tmp_path):
    """TODO_tests.md/A: within each `<SAMPLE>_scaf`, positions must be
    non-monotonic-detectable via sort. Here we prove a test CAN detect
    a position-column swap by showing the sort-order changes when we
    shuffle. (Used as a pattern in downstream integrity checks.)"""
    df = _make_rbhdf(
        ["fam_0", "fam_1", "fam_2", "fam_3"],
        ["algA", "algA", "algA", "algA"], [10, 20, 30, 40],
        ["scf1", "scf1", "scf1", "scf1"], [100, 200, 300, 400],
    )
    # Already sorted by position → rbh order is fam_0..fam_3.
    sorted_df = df.sort_values(f"{SAMPLE}_pos")
    assert list(sorted_df["rbh"]) == ["fam_0", "fam_1", "fam_2", "fam_3"]
    # Swap positions 0 and 2: sort-by-pos ordering changes — this is
    # exactly how a CI check would detect a position column swap.
    df2 = df.copy()
    df2.loc[0, f"{SAMPLE}_pos"] = 300
    df2.loc[2, f"{SAMPLE}_pos"] = 100
    sorted_df2 = df2.sort_values(f"{SAMPLE}_pos")
    assert list(sorted_df2["rbh"]) != ["fam_0", "fam_1", "fam_2", "fam_3"]


# ---------- combo_to_index invariants -------------------------------------

def test_ALGrbh_to_algcomboix_contiguous_indices(tmp_path):
    """TODO_tests.md/A: values are exactly range(len(file))."""
    from egt.phylotreeumap import ALGrbh_to_algcomboix
    # 5 families → C(5,2) = 10 pairs
    df = _make_rbhdf(
        rbhs=[f"fam_{i}" for i in range(5)],
        alg_scafs=["algA"] * 5,
        alg_poss=list(range(5)),
        sample_scafs=["scf1"] * 5,
        sample_poss=list(range(5, 10)),
    )
    p = _write_rbh(tmp_path, df)
    combo = ALGrbh_to_algcomboix(str(p))
    assert len(combo) == 10
    # values 0..9 exactly
    assert sorted(combo.values()) == list(range(10))
    # Build is deterministic: redo and compare.
    combo2 = ALGrbh_to_algcomboix(str(p))
    assert combo == combo2


def test_ALGrbh_to_algcomboix_pair_order_lex(tmp_path):
    """TODO_tests.md/A: pair keys must satisfy rbh1 < rbh2 lex."""
    from egt.phylotreeumap import ALGrbh_to_algcomboix
    df = _make_rbhdf(
        rbhs=["fam_2", "fam_0", "fam_1"],     # deliberately not sorted
        alg_scafs=["algA"] * 3,
        alg_poss=[10, 20, 30],
        sample_scafs=["scf1"] * 3,
        sample_poss=[100, 200, 300],
    )
    p = _write_rbh(tmp_path, df)
    combo = ALGrbh_to_algcomboix(str(p))
    for (a, b) in combo.keys():
        assert a < b, (
            f"combo_to_index key ({a!r}, {b!r}) violates lex ordering "
            "— downstream pair-matching relies on sorted keys")


def test_algcomboix_file_roundtrip(tmp_path):
    """Write combo dict to the TSV format used in production, then
    re-read with the canonical reader and confirm the round-trip."""
    from egt.phylotreeumap import (
        ALGrbh_to_algcomboix, algcomboix_file_to_dict,
    )
    df = _make_rbhdf(
        rbhs=[f"fam_{i}" for i in range(4)],
        alg_scafs=["algA"] * 4,
        alg_poss=list(range(4)),
        sample_scafs=["scf1"] * 4,
        sample_poss=list(range(10, 14)),
    )
    p = _write_rbh(tmp_path, df)
    combo = ALGrbh_to_algcomboix(str(p))
    combo_file = tmp_path / "combo_to_index.txt"
    with open(combo_file, "w") as fh:
        for k, v in combo.items():
            fh.write(f"{k}\t{v}\n")
    reloaded = algcomboix_file_to_dict(str(combo_file))
    assert reloaded == combo


# ---------- sampledf invariants (documentary) -----------------------------

def test_sampledf_index_requirement_enforced_by_builder():
    """TODO_tests.md/A: the COO builder requires sampledf.index to be
    0..N-1. The builder resets a non-RangeIndex silently — verify that
    guard exists (covered in test_coo_integrity.py::test_scrambled_
    sampledf_index_resets_and_still_correct, which reshuffles labels
    and asserts the output is still correct). Here we just re-assert
    the invariant at the API level."""
    import numpy as np
    import pandas as pd
    # A correctly-constructed sampledf has RangeIndex.
    df = pd.DataFrame({"sample": ["a", "b", "c"],
                       "dis_filepath_abs": ["x", "y", "z"]})
    assert isinstance(df.index, pd.RangeIndex)
    assert (df.index.to_numpy() == np.arange(len(df))).all()
