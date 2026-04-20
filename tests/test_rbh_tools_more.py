from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from egt import rbh_tools


def _write_rbh(path: Path, text: str) -> Path:
    path.write_text(text)
    return path


def test_combine_rbh_db_fills_missing_colors_and_blocks_duplicates(tmp_path: Path):
    a = _write_rbh(
        tmp_path / "a.tsv",
        "rbh\tgene_group\tcolor\nr1\tA\t#112233\n",
    )
    b = _write_rbh(
        tmp_path / "b.tsv",
        "rbh\tgene_group\nr2\tB\n",
    )
    merged = rbh_tools.combine_rbh_db(a, b)
    assert set(merged["rbh"]) == {"r1", "r2"}
    assert merged.loc[merged["rbh"] == "r2", "color"].iloc[0] == "#000000"

    dup = _write_rbh(
        tmp_path / "dup.tsv",
        "rbh\tgene_group\nr1\tC\n",
    )
    with pytest.raises(IOError, match="shared entries"):
        rbh_tools.combine_rbh_db(a, dup)


def test_combine_rbh_merges_two_files_with_one_shared_sample(tmp_path: Path):
    left = _write_rbh(
        tmp_path / "left.tsv",
        "\t".join(
            [
                "rbh", "gene_group", "A_scaf", "A_gene", "A_pos",
                "Shared_scaf", "Shared_gene", "Shared_pos",
            ]
        )
        + "\n"
        + "\t".join(["l1", "ALG1", "a1", "ag1", "10", "s1", "sg1", "100"])
        + "\n",
    )
    right = _write_rbh(
        tmp_path / "right.tsv",
        "\t".join(
            [
                "rbh", "gene_group", "B_scaf", "B_gene", "B_pos",
                "Shared_scaf", "Shared_gene", "Shared_pos",
            ]
        )
        + "\n"
        + "\t".join(["r1", "ALG2", "b1", "bg1", "20", "s1", "sg2", "200"])
        + "\n",
    )

    merged = rbh_tools.combine_rbh(left, right)

    assert "MergedCol_scaf" in merged.columns
    assert list(merged["Shared_pos"]) == [100, 200]


def test_rbh_to_scafnum_and_parse_alg_rbh_to_colordf(tmp_path: Path):
    df = pd.DataFrame({"sample_scaf": ["s1", "s1", "s2"]})
    assert rbh_tools.rbh_to_scafnum(df, "sample") == 2

    rbh = _write_rbh(
        tmp_path / "alg.tsv",
        "gene_group\tcolor\nA\t#111111\nA\t#111111\nB\t#222222\n",
    )
    colors = rbh_tools.parse_ALG_rbh_to_colordf(rbh)
    assert list(colors.columns) == ["ALGname", "Color", "Size"]
    assert list(colors["ALGname"]) == ["B", "A"]


def test_rbhdf_to_alglocdf_builds_split_dataframe():
    df = pd.DataFrame(
        {
            "gene_group": ["ALG1", "ALG1", "ALG2"],
            "BCnSSimakov2022_scaf": ["alg", "alg", "alg"],
            "sample_scaf": ["chr1", "chr1", "chr2"],
            "whole_FET": [0.001, 0.001, 0.002],
        }
    )

    splits, sample = rbh_tools.rbhdf_to_alglocdf(df, minsig=0.005, ALGname="BCnSSimakov2022")

    assert sample == "sample"
    assert set(splits.columns) == {
        "sample", "gene_group", "scaffold", "pvalue", "num_genes",
        "frac_of_this_ALG_on_this_scaffold",
    }
    assert set(splits["gene_group"]) == {"ALG1", "ALG2"}

