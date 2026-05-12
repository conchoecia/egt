"""Tests for egt.go.io loaders.

Uses tiny fixture files under tests/testdb/go/ so the exact column layout
each loader expects is documented and enforced.
"""
from __future__ import annotations

import gzip
from pathlib import Path

import pandas as pd
import pytest

from egt.go.io import (
    build_family_gene_annotations,
    load_obo_names,
    load_unique_pairs,
    parse_family_map,
    parse_gene2accession,
    parse_gene2go,
)


GO_DB = Path(__file__).resolve().parent / "testdb" / "go"


# ---------- parse_gene2accession ----------
def test_parse_gene2accession_plain():
    prot_to_gene, gene_to_symbol = parse_gene2accession(GO_DB / "gene2accession.tsv")
    # GeneID 1000 has two protein rows (NP_000001 + XP_000002), both recorded.
    assert prot_to_gene["NP_000001.1"] == "1000"
    assert prot_to_gene["XP_000002.1"] == "1000"
    assert prot_to_gene["NP_000003.1"] == "1001"
    assert prot_to_gene["NP_000005.1"] == "1002"
    # GeneID 1003 has no protein accession → not in map.
    assert "NP_000009.1" not in prot_to_gene
    # Symbol table: first seen per GeneID.
    assert gene_to_symbol["1000"] == "GENE1"
    assert gene_to_symbol["1001"] == "GENE2"
    assert gene_to_symbol["1002"] == "GENE3"
    # GeneID 1003 only appears on a row with no protein accession — but
    # it has a Symbol GENE4, which should still be recorded.
    assert gene_to_symbol["1003"] == "GENE4"
    # Row with tax_id set but "-" GeneID does not contribute.
    assert "-" not in gene_to_symbol


def test_parse_gene2accession_gzip(tmp_path: Path):
    src = (GO_DB / "gene2accession.tsv").read_bytes()
    gz_path = tmp_path / "g2a.tsv.gz"
    with gzip.open(gz_path, "wb") as fh:
        fh.write(src)
    p2g, _ = parse_gene2accession(gz_path)
    assert p2g["NP_000001.1"] == "1000"


def test_parse_gene2accession_short_line(tmp_path: Path):
    # A row with fewer than 7 fields should be skipped without error.
    p = tmp_path / "short.tsv"
    p.write_text("# header\n1\t2\t3\n9606\t1\tR\tNM_x\t1\tNP_x.1\t2\n")
    p2g, _ = parse_gene2accession(p)
    assert p2g == {"NP_x.1": "1"}


# ---------- parse_gene2go ----------
def test_parse_gene2go_plain():
    gene_to_terms, term_ns = parse_gene2go(GO_DB / "gene2go.tsv")
    assert gene_to_terms["1000"] == {"GO:0000001", "GO:0000002"}
    assert gene_to_terms["1001"] == {"GO:0000001"}   # NOT-qualifier row skipped
    assert gene_to_terms["1002"] == {"GO:0000002", "GO:0000004"}
    assert gene_to_terms["1003"] == {"GO:0000001"}
    assert term_ns["GO:0000001"] == "BP"
    assert term_ns["GO:0000002"] == "MF"
    assert term_ns["GO:0000004"] == "CC"


def test_parse_gene2go_short_line(tmp_path: Path):
    p = tmp_path / "short.tsv"
    p.write_text("# header\ntoo\tfew\tcols\n")
    g2t, ns = parse_gene2go(p)
    assert g2t == {}
    assert ns == {}


def test_parse_gene2go_gzip(tmp_path: Path):
    src = (GO_DB / "gene2go.tsv").read_bytes()
    gz = tmp_path / "g2g.tsv.gz"
    with gzip.open(gz, "wb") as fh:
        fh.write(src)
    g2t, _ = parse_gene2go(gz)
    assert g2t["1000"] == {"GO:0000001", "GO:0000002"}


# ---------- parse_family_map ----------
def test_parse_family_map():
    p2g, _ = parse_gene2accession(GO_DB / "gene2accession.tsv")
    fam_to_genes, stats = parse_family_map(GO_DB / "family_map.tsv", p2g)
    # RefSeq-hitting families land with their resolved GeneIDs.
    assert fam_to_genes["FAM_A"] == {"1000"}
    assert fam_to_genes["FAM_B"] == {"1001"}
    assert fam_to_genes["FAM_C"] == {"1002"}
    # FAM_D is Swiss-Prot → not mapped.
    assert "FAM_D" not in fam_to_genes
    # FAM_E is RefSeq but unknown → counted as miss.
    assert "FAM_E" not in fam_to_genes
    # FAM_F empty → empty bucket.
    assert "FAM_F" not in fam_to_genes
    # FAM_G "other" → counted.
    assert "FAM_G" not in fam_to_genes
    assert stats["n_rows"] == 7
    assert stats["n_refseq_hit"] == 3
    assert stats["n_refseq_miss"] == 1
    assert stats["n_sp_uniprot"] == 1
    assert stats["n_other"] == 1
    assert stats["n_empty"] == 1
    assert stats["n_families_mapped"] == 3


def test_parse_family_map_no_human_gene_column(tmp_path: Path):
    # If the second-from-left column is used as gene_col (fallback when
    # 'human_gene' is absent), the same RefSeq semantics apply.
    p = tmp_path / "fm.tsv"
    p.write_text("fam\talg\tprotein\nFAM_A\tA1a\tNP_000001.1\n")
    p2g = {"NP_000001.1": "999"}
    fam, stats = parse_family_map(p, p2g)
    assert fam["FAM_A"] == {"999"}
    assert stats["n_refseq_hit"] == 1


# ---------- load_obo_names ----------
def test_load_obo_names():
    names, ns = load_obo_names(GO_DB / "mini.obo")
    assert names["GO:0000001"] == "process-one"
    assert names["GO:0000002"] == "function-one"
    assert names["GO:0000004"] == "component-one"
    assert ns["GO:0000001"] == "BP"
    assert ns["GO:0000002"] == "MF"
    assert ns["GO:0000004"] == "CC"
    # Term with an unrecognised namespace string → "?" sentinel.
    assert ns["GO:0000999"] == "?"


def test_load_obo_names_missing(tmp_path: Path):
    names, ns = load_obo_names(None)
    assert names == {} and ns == {}
    names, ns = load_obo_names(tmp_path / "nope.obo")
    assert names == {} and ns == {}


# ---------- load_unique_pairs ----------
def test_load_unique_pairs_tsv():
    df = load_unique_pairs(GO_DB / "unique_pairs_mini.tsv")
    assert set(df["nodename"].unique()) == {"CladeX", "CladeY"}
    for c in ("ortholog1", "ortholog2", "occupancy_in",
              "sd_in_out_ratio_log_sigma", "mean_in_out_ratio_log_sigma"):
        assert c in df.columns


def test_load_unique_pairs_tsv_gz(tmp_path: Path):
    src = (GO_DB / "unique_pairs_mini.tsv").read_bytes()
    gz = tmp_path / "up.tsv.gz"
    with gzip.open(gz, "wb") as fh:
        fh.write(src)
    df = load_unique_pairs(gz)
    assert len(df) == 15


def test_load_unique_pairs_xlsx(tmp_path: Path):
    df_src = pd.read_csv(GO_DB / "unique_pairs_mini.tsv", sep="\t")
    xlsx = tmp_path / "up.xlsx"
    df_src.to_excel(xlsx, index=False)
    df = load_unique_pairs(xlsx)
    assert len(df) == 15


def test_load_unique_pairs_missing_col(tmp_path: Path):
    bad = tmp_path / "bad.tsv"
    bad.write_text("nodename\tortholog1\n")
    with pytest.raises(ValueError, match="missing required columns"):
        load_unique_pairs(bad)


# ---------- build_family_gene_annotations ----------
def test_build_family_gene_annotations():
    p2g, _ = parse_gene2accession(GO_DB / "gene2accession.tsv")
    gene_to_terms, _ = parse_gene2go(GO_DB / "gene2go.tsv")
    fam_to_genes, _ = parse_family_map(GO_DB / "family_map.tsv", p2g)
    bg, bg_to_terms = build_family_gene_annotations(fam_to_genes, gene_to_terms)
    # background = {1000, 1001, 1002}
    assert bg == {"1000", "1001", "1002"}
    # All three have annotations, so bg_to_terms covers all three.
    assert set(bg_to_terms) == bg
    assert bg_to_terms["1000"] == {"GO:0000001", "GO:0000002"}


def test_build_family_gene_annotations_empty():
    bg, bg_to_terms = build_family_gene_annotations({}, {})
    assert bg == set()
    assert bg_to_terms == {}
