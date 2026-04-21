from __future__ import annotations

import gzip
from pathlib import Path

import pandas as pd

from egt import entanglement_go_enrich as ego


def test_parse_gaf_reads_gzipped_annotations(tmp_path: Path):
    gaf = tmp_path / "goa_human.gaf.gz"
    with gzip.open(gaf, "wt") as fh:
        fh.write("!gaf-version: 2.2\n")
        fh.write("UniProtKB\tP1\tGENE1\t\tGO:0001\tPMID:1\tIDA\t\tF\tName1\t\tprotein\ttaxon:9606\t20240101\tUniProt\n")
        fh.write("UniProtKB\tP2\tGENE1\t\tGO:0002\tPMID:2\tIDA\t\tF\tName1\t\tprotein\ttaxon:9606\t20240101\tUniProt\n")

    parsed = ego._parse_gaf(gaf)
    assert parsed["GENE1"] == {"GO:0001", "GO:0002"}


def test_main_writes_go_enrichment_table(tmp_path: Path):
    alg_rbh = tmp_path / "alg.tsv"
    pd.DataFrame(
        {
            "rbh": ["fam1", "fam2", "fam3", "fam4"],
            "gene_group": ["A", "B", "C", "D"],
        }
    ).to_csv(alg_rbh, sep="\t", index=False)

    family_map = tmp_path / "family_map.tsv"
    pd.DataFrame(
        {
            "family_id": ["fam1", "fam2", "fam3", "fam4"],
            "human_gene": ["GENE1", "GENE2", "GENE3", "GENE4"],
        }
    ).to_csv(family_map, sep="\t", index=False)

    entangled = tmp_path / "pairs.tsv"
    pd.DataFrame(
        {
            "clade": ["Metazoa"],
            "alg_a": ["A"],
            "alg_b": ["B"],
            "n_in_clade": ["2"],
        }
    ).to_csv(entangled, sep="\t", index=False)

    gaf = tmp_path / "goa_human.gaf.gz"
    with gzip.open(gaf, "wt") as fh:
        fh.write("!gaf-version: 2.2\n")
        fh.write("UniProtKB\tP1\tGENE1\t\tGO:0001\tPMID:1\tIDA\t\tF\tName1\t\tprotein\ttaxon:9606\t20240101\tUniProt\n")
        fh.write("UniProtKB\tP2\tGENE2\t\tGO:0001\tPMID:2\tIDA\t\tF\tName2\t\tprotein\ttaxon:9606\t20240101\tUniProt\n")
        fh.write("UniProtKB\tP3\tGENE3\t\tGO:0002\tPMID:3\tIDA\t\tF\tName3\t\tprotein\ttaxon:9606\t20240101\tUniProt\n")
        fh.write("UniProtKB\tP4\tGENE4\t\tGO:0003\tPMID:4\tIDA\t\tF\tName4\t\tprotein\ttaxon:9606\t20240101\tUniProt\n")

    out_dir = tmp_path / "out"
    assert ego.main(
        [
            "--alg-rbh",
            str(alg_rbh),
            "--family-gene-map",
            str(family_map),
            "--entangled-pairs",
            str(entangled),
            "--human-go",
            str(gaf),
            "--fdr",
            "1.0",
            "--min-term-hits",
            "1",
            "--out-dir",
            str(out_dir),
        ]
    ) == 0

    out_tsv = out_dir / "go_enrichment_per_clade.tsv"
    assert out_tsv.exists()
    enriched = pd.read_csv(out_tsv, sep="\t")
    assert not enriched.empty
    assert set(enriched["clade"]) == {"Metazoa"}
    assert "q_value" in enriched.columns
