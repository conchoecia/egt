from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from egt.build_family_naming_map import _pass1_from_human_rbh, _pass2_merge_hmm, main


def test_pass1_from_human_rbh_joins_family_and_scaffold(tmp_path: Path):
    alg_rbh = tmp_path / "alg.rbh"
    alg_rbh.write_text("rbh\tgene_group\nfam1\tA1a\nfam2\tB\n")
    human_rbh = tmp_path / "human.rbh"
    human_rbh.write_text(
        "rbh\tHomosapiens_gene\tHomosapiens_scaf\nfam1\tGENE1\tchr1\nfam2\t\t\n"
    )

    df = _pass1_from_human_rbh(alg_rbh, human_rbh)

    assert list(df.columns) == ["family_id", "alg", "human_gene", "human_scaf", "source", "note"]
    assert df.loc[0, "family_id"] == "fam1"
    assert df.loc[0, "human_gene"] == "GENE1"
    assert df.loc[0, "human_scaf"] == "chr1"
    assert df.loc[0, "source"] == "human_rbh"
    assert df.loc[1, "source"] == ""


def test_pass1_requires_rbh_column(tmp_path: Path):
    alg_rbh = tmp_path / "alg.rbh"
    alg_rbh.write_text("gene_group\nA1a\n")
    human_rbh = tmp_path / "human.rbh"
    human_rbh.write_text("rbh\tHomosapiens_gene\nfam1\tGENE1\n")

    with pytest.raises(SystemExit, match="missing 'rbh' column"):
        _pass1_from_human_rbh(alg_rbh, human_rbh)


def test_pass1_requires_species_gene_column(tmp_path: Path):
    alg_rbh = tmp_path / "alg.rbh"
    alg_rbh.write_text("rbh\tgene_group\nfam1\tA1a\n")
    human_rbh = tmp_path / "human.rbh"
    human_rbh.write_text("rbh\tBCnSSimakov2022_gene\nfam1\tALG1\n")

    with pytest.raises(SystemExit, match="No species `_gene` column found"):
        _pass1_from_human_rbh(alg_rbh, human_rbh)


def test_pass2_merge_hmm_fills_only_missing_rows(tmp_path: Path):
    pass1 = pd.DataFrame(
        [
            {"family_id": "fam1", "alg": "A1a", "human_gene": "GENE1", "human_scaf": "chr1", "source": "human_rbh", "note": ""},
            {"family_id": "fam2", "alg": "B", "human_gene": "", "human_scaf": "", "source": "", "note": ""},
        ]
    )
    hmm_map = tmp_path / "hmm.tsv"
    hmm_map.write_text("family_id\thuman_gene\nfam2\tGENE2\n")

    merged = _pass2_merge_hmm(pass1, hmm_map)

    assert merged.loc[0, "human_gene"] == "GENE1"
    assert merged.loc[1, "human_gene"] == "GENE2"
    assert merged.loc[1, "source"] == "hmm_consensus"


def test_pass2_missing_file_is_skipped(tmp_path: Path, capsys):
    pass1 = pd.DataFrame(
        [{"family_id": "fam1", "alg": "A1a", "human_gene": "", "human_scaf": "", "source": "", "note": ""}]
    )

    merged = _pass2_merge_hmm(pass1, tmp_path / "missing.tsv")

    assert merged.loc[0, "human_gene"] == ""
    captured = capsys.readouterr()
    assert "--hmm-map not found" in captured.err


def test_main_writes_output_and_reports_coverage(tmp_path: Path, capsys):
    alg_rbh = tmp_path / "alg.rbh"
    alg_rbh.write_text("rbh\tgene_group\nfam1\tA1a\nfam2\tB\n")
    human_rbh = tmp_path / "human.rbh"
    human_rbh.write_text("rbh\tHomosapiens_gene\tHomosapiens_scaf\nfam1\tGENE1\tchr1\nfam2\t\t\n")
    hmm_map = tmp_path / "hmm.tsv"
    hmm_map.write_text("family_id\thuman_gene\nfam2\tGENE2\n")
    out = tmp_path / "family_map.tsv"

    rc = main(
        [
            "--alg-rbh",
            str(alg_rbh),
            "--human-rbh",
            str(human_rbh),
            "--hmm-map",
            str(hmm_map),
            "--output",
            str(out),
        ]
    )

    assert rc == 0
    written = pd.read_csv(out, sep="\t")
    assert list(written["human_gene"]) == ["GENE1", "GENE2"]
    captured = capsys.readouterr()
    assert "with human gene : 2 (100%)" in captured.err

