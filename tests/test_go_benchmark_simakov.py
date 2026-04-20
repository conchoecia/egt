"""End-to-end tests for egt.go.benchmarks.simakov using synthetic
GENCODE GTF + synthetic microsynteny xls fixture."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


def _fixture_gene2accession(tmp_path: Path) -> Path:
    p = tmp_path / "g2a.tsv"
    with p.open("w") as fh:
        fh.write("#tax\tGeneID\tstatus\tRNA\trnagi\tprotein\tpgi\tgenomic\tggi\tsp\tep\to\tasm\tmpep\tmpgi\tSymbol\n")
        for i, (gid, sym) in enumerate([
            ("1000", "GENE_A"), ("1001", "GENE_B"), ("1002", "GENE_C"),
            ("1003", "GENE_D"), ("1004", "GENE_E"),
        ]):
            fh.write(f"9606\t{gid}\tREVIEWED\tNM.{i}\t-\tNP_{i:06d}.1\t-\tNC.1\t-\t1\t2\t+\tG\t-\t-\t{sym}\n")
    return p


def _fixture_family_map(tmp_path: Path) -> Path:
    p = tmp_path / "fm.tsv"
    with p.open("w") as fh:
        fh.write("family_id\talg\thuman_gene\n")
        fh.write("FAM_A\tA1a\tNP_000000.1\n")
        fh.write("FAM_B\tA1a\tNP_000001.1\n")
        fh.write("FAM_C\tA1b\tNP_000002.1\n")
        fh.write("FAM_D\tA1b\tNP_000003.1\n")
    return p


def _fixture_unique_pairs(tmp_path: Path) -> Path:
    p = tmp_path / "up.tsv.gz"
    df = pd.DataFrame([
        dict(nodename="X", ortholog1="FAM_A", ortholog2="FAM_B",
             stable_in_clade="True"),
        dict(nodename="X", ortholog1="FAM_C", ortholog2="FAM_D",
             stable_in_clade="False"),
    ])
    df.to_csv(p, sep="\t", index=False, compression="gzip")
    return p


def _fixture_gencode_gtf(tmp_path: Path) -> Path:
    p = tmp_path / "gencode.gtf"
    # 4 gene entries on chr1, spanning the test coords. GTF is 1-based
    # inclusive; our block coord 100-500 covers all 4.
    with p.open("w") as fh:
        fh.write("##description: test GENCODE stub\n")
        for i, (name, start, end) in enumerate([
            ("GENE_A", 100, 150),
            ("GENE_B", 200, 250),
            ("GENE_C", 300, 350),
            ("GENE_D", 400, 450),
        ]):
            # transcript-level line (ignored because it's not type 'gene')
            fh.write(f"chr1\tHAVANA\ttranscript\t{start}\t{end}\t.\t+\t.\t"
                     f'gene_id "G{i}"; transcript_id "T{i}"; '
                     f'gene_name "{name}"; gene_type "protein_coding";\n')
            # gene-level line (what the parser picks up)
            fh.write(f"chr1\tHAVANA\tgene\t{start}\t{end}\t.\t+\t.\t"
                     f'gene_id "G{i}"; gene_name "{name}"; '
                     f'gene_type "protein_coding";\n')
        # Short / malformed line (exercises the <9 fields and non-gene skip).
        fh.write("short\tline\n")
    return p


def _fixture_microsynteny_xls(tmp_path: Path) -> Path:
    p = tmp_path / "ms.xlsx"
    df = pd.DataFrame([
        dict(ClusID="blk1", **{"Species ID": "Hsa"},
             Classification="bilaterianAnc",
             **{"Chrom:begin-end": "chr1:99-500"}),
        # A non-Hsa row — filtered out in the Hsa step.
        dict(ClusID="blk2", **{"Species ID": "Dro"},
             Classification="bilaterianAnc",
             **{"Chrom:begin-end": "chr2:1-100"}),
        # Coord that fails parse_coord.
        dict(ClusID="blk3", **{"Species ID": "Hsa"},
             Classification="bilaterianAnc",
             **{"Chrom:begin-end": "garbage"}),
        # Coord on a chrom we don't index.
        dict(ClusID="blk4", **{"Species ID": "Hsa"},
             Classification="bilaterianAnc",
             **{"Chrom:begin-end": "chrY:1000-2000"}),
        # Wrong classification tag.
        dict(ClusID="blk5", **{"Species ID": "Hsa"},
             Classification="NotBilaterian",
             **{"Chrom:begin-end": "chr1:100-200"}),
    ])
    df.to_excel(p, index=False)
    return p


def test_simakov_helpers():
    from egt.go.benchmarks import simakov as sim
    # parse_coord happy path + None branches.
    assert sim.parse_coord("chr1:100-200") == ("1", 100, 200)
    assert sim.parse_coord("garbage") is None
    assert sim.parse_coord(None) is None
    # genes_overlapping: half-open, sorted shortcut.
    chrom_idx = {"1": [(100, 150, "A"), (200, 250, "B"),
                       (300, 350, "C"), (500, 600, "D")]}
    # Block spanning A and B only.
    out = sim.genes_overlapping(("1", 50, 260), chrom_idx)
    assert set(out) == {"A", "B"}
    # Block before anything — empty.
    assert sim.genes_overlapping(("1", 0, 10), chrom_idx) == []
    # Non-indexed chrom — empty.
    assert sim.genes_overlapping(("X", 0, 1000), chrom_idx) == []


def test_simakov_load_gencode_genes(tmp_path):
    from egt.go.benchmarks import simakov as sim
    df = sim.load_gencode_genes(_fixture_gencode_gtf(tmp_path))
    # 4 gene-type entries (transcripts rejected by type filter)
    assert len(df) == 4
    assert "GENE_A" in df["gene_name"].values


def test_simakov_load_unique_pairs_coerces_bool_strings(tmp_path):
    from egt.go.benchmarks import simakov as sim
    per_clade, union, stable = sim.load_unique_pairs(
        _fixture_unique_pairs(tmp_path)
    )
    assert frozenset(("FAM_A", "FAM_B")) in stable
    assert frozenset(("FAM_C", "FAM_D")) not in stable
    assert frozenset(("FAM_A", "FAM_B")) in union


def test_simakov_main_end_to_end(tmp_path):
    from egt.go.benchmarks import simakov as sim
    rc = sim.main([
        "--microsynteny-xls", str(_fixture_microsynteny_xls(tmp_path)),
        "--classification", "bilaterianAnc",
        "--species", "Hsa",
        "--family-map", str(_fixture_family_map(tmp_path)),
        "--gene2accession", str(_fixture_gene2accession(tmp_path)),
        "--gencode-gtf", str(_fixture_gencode_gtf(tmp_path)),
        "--unique-pairs", str(_fixture_unique_pairs(tmp_path)),
        "--out-dir", str(tmp_path / "out"),
    ])
    assert rc == 0
    out = tmp_path / "out"
    for name in ("per_block.tsv", "per_pair.tsv",
                 "summary.tsv", "report.md"):
        assert (out / name).exists(), f"missing {name}"
    per_block = pd.read_csv(out / "per_block.tsv", sep="\t")
    assert (per_block["ClusID"] == "blk1").any()
    # Block blk1 covers GENE_A through GENE_D → 4 families → 6 pairs.
    b1 = per_block[per_block["ClusID"] == "blk1"].iloc[0]
    assert b1["n_intra_block_pairs"] == 6
    # FAM_A-FAM_B is in our union → one hit.
    assert b1["n_hits_union"] >= 1
