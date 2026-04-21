from __future__ import annotations

import gzip
from pathlib import Path

import pandas as pd
import pytest

from egt import phylotreeumap as ptu


class FakeNCBI:
    def get_taxid_translator(self, taxids):
        return {taxid: f"Taxon {taxid}" for taxid in taxids}


def _write_gbgz(path: Path, rows: list[tuple[str, str, int]]) -> Path:
    df = pd.DataFrame(rows, columns=["rbh1", "rbh2", "distance"])
    with gzip.open(path, "wt") as fh:
        df.to_csv(fh, sep="\t", index=False)
    return path


def test_taxids_to_analyses_builds_names_and_validates(monkeypatch):
    monkeypatch.setattr(ptu, "NCBITaxa", lambda: FakeNCBI())
    analyses = ptu.taxids_to_analyses([[[6340], [42113]], [[10197, 6040], []]])
    assert "Taxon6340_6340_without_42113" in analyses
    assert "Taxon10197_Taxon6040_10197_6040_without_None" in analyses

    with pytest.raises(ValueError, match="There are no taxids to parse"):
        ptu.taxids_to_analyses([])
    with pytest.raises(ValueError, match="must have two lists"):
        ptu.taxids_to_analyses([[6340]])


def test_filter_sample_df_by_clades():
    sampledf = pd.DataFrame(
        {
            "sample": ["a", "b", "c"],
            "taxid_list": ["[1, 2, 3]", "[4, 5]", "[1, 9]"],
        }
    )
    filtered = ptu.filter_sample_df_by_clades(sampledf, [1], [9])
    assert list(filtered["sample"]) == ["a"]


def test_algcomboix_helpers_and_rbh_to_samplename(tmp_path: Path):
    rbh = tmp_path / "alg.rbh"
    rbh.write_text(
        "rbh\tgene_group\tcolor\n"
        "fam1\tA\t#111111\n"
        "fam2\tB\t#222222\n"
        "fam3\tC\t#333333\n"
    )
    combo = ptu.ALGrbh_to_algcomboix(rbh)
    assert len(combo) == 3

    combo_file = tmp_path / "combo.tsv"
    combo_file.write_text("('fam1', 'fam2')\t0\n('fam1', 'fam3')\t1\n")
    parsed = ptu.algcomboix_file_to_dict(combo_file)
    assert parsed[("fam1", "fam2")] == 0

    sample_name = ptu.rbh_to_samplename(
        "BCnSSimakov2022_Zonotrichialeucophrys-44393-GCA028769735.1_xy_reciprocal_best_hits.plotted.rbh",
        "BCnSSimakov2022",
    )
    assert sample_name == "Zonotrichialeucophrys-44393-GCA028769735.1"


def test_construct_coo_matrix_from_sampledf_with_override_paths(tmp_path: Path):
    g1 = _write_gbgz(
        tmp_path / "one.gb.gz",
        [("fam1", "fam2", 10), ("fam1", "fam3", 20)],
    )
    g2 = _write_gbgz(
        tmp_path / "two.gb.gz",
        [("fam1", "fam2", 11), ("fam2", "fam3", 30)],
    )
    sampledf = pd.DataFrame(
        {
            "sample": ["s1", "s2"],
            "dis_filepath_abs": ["/missing/one.gb.gz", "/missing/two.gb.gz"],
        }
    )
    alg_combo_to_ix = {("fam1", "fam2"): 0, ("fam1", "fam3"): 1, ("fam2", "fam3"): 2}

    matrix = ptu.construct_coo_matrix_from_sampledf(
        sampledf,
        alg_combo_to_ix,
        gbgz_paths={"s1": str(g1), "s2": str(g2)},
    )

    assert matrix.shape == (2, 3)
    dense = matrix.toarray()
    assert dense[0, 0] == 10
    assert dense[0, 1] == 20
    assert dense[1, 0] == 11
    assert dense[1, 2] == 30


def test_construct_coo_matrix_from_sampledf_rejects_missing_pairs(tmp_path: Path):
    g1 = _write_gbgz(tmp_path / "one.gb.gz", [("fam1", "fam9", 10)])
    sampledf = pd.DataFrame({"sample": ["s1"], "dis_filepath_abs": [str(g1)]})
    with pytest.raises(KeyError, match="pairs missing from alg_combo_to_ix"):
        ptu.construct_coo_matrix_from_sampledf(sampledf, {("fam1", "fam2"): 0})
