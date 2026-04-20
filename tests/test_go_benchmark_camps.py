"""End-to-end tests for egt.go.benchmarks.camps using synthetic fixtures."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


GO_DB = Path(__file__).resolve().parent / "testdb" / "go"


def _fixture_gene2accession(tmp_path: Path) -> Path:
    p = tmp_path / "g2a.tsv"
    with p.open("w") as fh:
        fh.write("#tax\tGeneID\tstatus\tRNA\trnagi\tprotein\tpgi\tgenomic\tggi\tsp\tep\to\tasm\tmpep\tmpgi\tSymbol\n")
        fh.write("9606\t1000\tREVIEWED\tNM.1\t-\tNP_000001.1\t-\tNC.1\t-\t1\t2\t+\tG\t-\t-\tGENE_A\n")
        fh.write("9606\t1001\tREVIEWED\tNM.2\t-\tNP_000002.1\t-\tNC.1\t-\t1\t2\t+\tG\t-\t-\tGENE_B\n")
        fh.write("9606\t1002\tREVIEWED\tNM.3\t-\tNP_000003.1\t-\tNC.2\t-\t1\t2\t+\tG\t-\t-\tGENE_C\n")
        fh.write("9606\t1003\tREVIEWED\tNM.4\t-\tNP_000004.1\t-\tNC.2\t-\t1\t2\t+\tG\t-\t-\tGENE_D\n")
        # Two-part hyphenated symbol to exercise the hyphen disambiguation.
        fh.write("9606\t1004\tREVIEWED\tNM.5\t-\tNP_000005.1\t-\tNC.3\t-\t1\t2\t+\tG\t-\t-\tGENE_X\n")
        fh.write("9606\t1005\tREVIEWED\tNM.6\t-\tNP_000006.1\t-\tNC.3\t-\t1\t2\t+\tG\t-\t-\tGENE-HYPHEN.2\n")
    return p


def _fixture_family_map(tmp_path: Path) -> Path:
    p = tmp_path / "fm.tsv"
    with p.open("w") as fh:
        fh.write("family_id\talg\thuman_gene\thuman_scaf\tsource\tnote\n")
        fh.write("FAM_A\tA1a\tNP_000001.1\tNC.1\trbh\t\n")
        fh.write("FAM_B\tA1a\tNP_000002.1\tNC.1\trbh\t\n")
        fh.write("FAM_C\tA1b\tNP_000003.1\tNC.2\trbh\t\n")
        fh.write("FAM_D\tA1b\tNP_000004.1\tNC.2\trbh\t\n")
        fh.write("FAM_X\tB1\tNP_000005.1\tNC.3\trbh\t\n")
    return p


def _fixture_unique_pairs(tmp_path: Path) -> Path:
    p = tmp_path / "up.tsv.gz"
    df = pd.DataFrame([
        dict(nodename="X", ortholog1="FAM_A", ortholog2="FAM_B",
             stable_in_clade=True),
        dict(nodename="X", ortholog1="FAM_C", ortholog2="FAM_D",
             stable_in_clade=False),
        dict(nodename="Y", ortholog1="FAM_A", ortholog2="FAM_C",
             stable_in_clade=True),
    ])
    df.to_csv(p, sep="\t", index=False, compression="gzip")
    return p


def _fixture_camps_xlsx(tmp_path: Path) -> Path:
    p = tmp_path / "camps.xlsx"
    df = pd.DataFrame([
        # representable + in our defining union
        dict(i=1, **{"Best Human Hit": "GENE_A-GENE_B"}, **{"All #sp": 17},
             Type="CAMP"),
        # representable but not in our defining union
        dict(i=2, **{"Best Human Hit": "GENE_C-GENE_D"}, **{"All #sp": 14},
             Type="CAMP"),
        # non-representable (GENE_UNKNOWN absent from gene2accession)
        dict(i=3, **{"Best Human Hit": "GENE_A-GENE_UNKNOWN"}, **{"All #sp": 10},
             Type="CAMP"),
        # hyphenated symbol — exercises the per-split disambiguation.
        dict(i=4, **{"Best Human Hit": "GENE_X-GENE-HYPHEN.2"}, **{"All #sp": 6},
             Type="CAMP"),
        # singleton (can't split)
        dict(i=5, **{"Best Human Hit": "LONE"}, **{"All #sp": 4},
             Type="CAMP"),
        # same-symbol tautology
        dict(i=6, **{"Best Human Hit": "GENE_A-GENE_A"}, **{"All #sp": 2},
             Type="CAMP"),
    ])
    df.to_excel(p, index=False)
    return p


def test_camps_main_end_to_end(tmp_path):
    from egt.go.benchmarks import camps
    rc = camps.main([
        "--camps", str(_fixture_camps_xlsx(tmp_path)),
        "--family-map", str(_fixture_family_map(tmp_path)),
        "--gene2accession", str(_fixture_gene2accession(tmp_path)),
        "--unique-pairs", str(_fixture_unique_pairs(tmp_path)),
        "--out-dir", str(tmp_path / "out"),
    ])
    assert rc == 0
    out = tmp_path / "out"
    for name in ("irimia2012_camps_mapped.tsv",
                 "summary.tsv", "report.md"):
        assert (out / name).exists()
    mapped = pd.read_csv(out / "irimia2012_camps_mapped.tsv", sep="\t")
    # CAMP 1 resolves to (FAM_A, FAM_B) which is in our defining union.
    row1 = mapped[mapped["camp_i"] == 1].iloc[0]
    assert row1["status"] == "representable"
    assert row1["in_our_defining_union"]


def test_camps_helpers_individually(tmp_path):
    from egt.go.benchmarks import camps
    # parse_gene2accession_symbol
    p2g, s2g = camps.parse_gene2accession_symbol(_fixture_gene2accession(tmp_path))
    assert p2g["NP_000001.1"] == "1000"
    assert s2g["GENE_A"] == {"1000"}
    # parse_family_map
    g2f, dup = camps.parse_family_map(_fixture_family_map(tmp_path), p2g)
    assert g2f["1000"] == "FAM_A"
    # load_camps with symbol_set
    camps_df = camps.load_camps(_fixture_camps_xlsx(tmp_path),
                                  symbol_set=set(s2g.keys()))
    assert len(camps_df) >= 3
    # symbols_to_families — representable
    pairs, status = camps.symbols_to_families(
        frozenset(("GENE_A", "GENE_B")), s2g, g2f,
    )
    assert status == ("representable",)
    assert frozenset(("FAM_A", "FAM_B")) in pairs
    # symbols_to_families — both_same_symbol
    pairs, status = camps.symbols_to_families(
        frozenset(("GENE_A",)), s2g, g2f,
    )
    assert status == ("both_same_symbol",)
    # symbol not in gene2accession
    pairs, status = camps.symbols_to_families(
        frozenset(("GENE_A", "UNKNOWN")), s2g, g2f,
    )
    assert "not_found" in status[0] or "family_map" in status[0]


def test_camps_load_our_pair_universe_handles_missing(tmp_path):
    from egt.go.benchmarks import camps
    # Non-existent path → None.
    assert camps.load_our_pair_universe(tmp_path / "missing.npz") is None
    # Existent but skip branch.
    existent = tmp_path / "present.npz"
    existent.write_bytes(b"stub")
    assert camps.load_our_pair_universe(existent) is None


def test_camps_load_unique_pairs_round_trip(tmp_path):
    from egt.go.benchmarks import camps
    p = _fixture_unique_pairs(tmp_path)
    per_clade, union, df = camps.load_unique_pairs(p)
    assert set(per_clade) == {"X", "Y"}
    assert frozenset(("FAM_A", "FAM_B")) in per_clade["X"]
    assert frozenset(("FAM_A", "FAM_B")) in union
