from __future__ import annotations

from types import SimpleNamespace

import matplotlib.pyplot as plt
import pandas as pd

from egt.legacy import plot_alg_fusions_v2 as paf2


def test_legacy_helper_functions(monkeypatch, tmp_path):
    assert paf2.hex_to_rgb("#ff7f00") == (255, 127, 0)
    assert "Qa" in paf2.dict_BCnSALG_to_color()

    df = pd.DataFrame(
        {
            "species": ["sp1", "sp2"],
            "taxid": [1, 2],
            "taxidstring": ["1;10", "1;20"],
            "changestrings": ["x", "y"],
            "A": [0, 0],
            "B": [1, 1],
            ("A", "B"): [0, 1],
        }
    )
    missing, present = paf2.missing_present_ALGs(df, min_for_missing=0.6)
    assert "A" in missing
    assert "B" in present
    assert paf2.separate_ALG_pairs(df.assign(A=[2, 2], B=[2, 2]), min_for_noncolocalized=0.5) == [("A", "B")]
    assert "A" in paf2.unsplit_ALGs(df.assign(A=[1, 2], B=[1, 1]), max_frac_split=0.5)

    monkeypatch.setattr(paf2, "NCBITaxa", lambda: type("FakeNCBI", (), {"get_lineage": lambda self, taxid: [1, int(taxid)]})())
    taxidstrings = paf2.taxids_to_taxidstringdict([10, 20])
    assert taxidstrings[10] == "1;10"

    img = paf2.image_sp_matrix_to_lineage(["1;10", "1;20"])
    assert img.size[1] == 2
    pa = paf2.image_sp_matrix_to_presence_absence(df, color_dict={"A": "#ff0000", "B": "#00ff00"})
    coloc = paf2.image_colocalization_matrix(df, color_dict={"A": "#ff0000", "B": "#00ff00"}, clustering=False)
    assert pa.size[1] >= 2
    assert coloc.size[0] >= 1

    monkeypatch.setattr(paf2, "image_sp_matrix_to_lineage", lambda _series: paf2.image_vertical_barrier(1, len(df), "#111111"))
    monkeypatch.setattr(paf2, "image_sp_matrix_to_presence_absence", lambda _df, color_dict=None: paf2.image_vertical_barrier(2, len(df), "#222222"))
    monkeypatch.setattr(paf2, "image_colocalization_matrix", lambda _df, color_dict=None, clustering=False, missing_data_color="#5d001e": paf2.image_vertical_barrier(1, len(df), "#333333"))
    paf2.standard_plot_out(df.copy(), str(tmp_path / "legacy"))
    assert (tmp_path / "legacy_composite_image.png").exists()


def test_legacy_parse_and_plot(tmp_path):
    rbh = tmp_path / "sample.rbh"
    pd.DataFrame(
        {
            "rbh": ["r1", "r2"],
            "gene_group": ["A", "B"],
            "color": ["#111111", "#222222"],
            "ALG_scaf": ["alg1", "alg2"],
            "ALG_gene": ["ag1", "ag2"],
            "ALG_pos": [1, 2],
            "Species_scaf": ["chr1", "chr1"],
            "Species_gene": ["g1", "g2"],
            "Species_pos": [11, 12],
            "whole_FET": [0.001, 0.001],
        }
    ).to_csv(rbh, sep="\t", index=False)
    alg_df = pd.DataFrame({"ALGname": ["A", "B"], "Color": ["#aa0000", "#00aa00"], "Size": [10, 20]})

    fusion_df = paf2.parse_ALG_fusions([str(rbh)], alg_df, "ALG", minsig=0.01)
    paf2.plot_ALG_fusions(fusion_df, alg_df, "ALG", outprefix=str(tmp_path / "legacy_plot"))
    assert (tmp_path / "legacy_plot_ALG_fusions.pdf").exists()


def test_legacy_main_smoke(tmp_path, monkeypatch):
    rbh_dir = tmp_path / "rbhs"
    rbh_dir.mkdir()
    (rbh_dir / "SpeciesA-10-GCF1.1.rbh").write_text("x\n")
    (rbh_dir / "SpeciesB-20-GCF1.1.rbh").write_text("x\n")
    alg_rbh = tmp_path / "alg.rbh"
    alg_rbh.write_text("placeholder\n")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        paf2,
        "parse_args",
        lambda: SimpleNamespace(directory=str(rbh_dir), ALGname="ALG", prefix="Species", ALG_rbh=str(alg_rbh), minsig=0.01),
    )
    monkeypatch.setattr(
        paf2.rbh_tools,
        "parse_ALG_rbh_to_colordf",
        lambda _path: pd.DataFrame({"ALGname": ["A", "B"], "Color": ["#aa0000", "#00aa00"], "Size": [2, 1]}),
    )
    monkeypatch.setattr(paf2.rbh_tools, "parse_rbh", lambda _path: pd.DataFrame({"dummy": [1]}))
    def fake_rbhdf_to_alglocdf(rbhdf, _minsig, _algname):
        sample = rbhdf.attrs["sample"]
        return (
            pd.DataFrame({"sample": [sample, sample], "gene_group": ["A", "B"], "scaffold": ["chr1", "chr1"]}),
            sample,
        )

    def fake_parse_rbh(path):
        df = pd.DataFrame({"dummy": [1]})
        df.attrs["sample"] = path.stem if hasattr(path, "stem") else str(path).split("/")[-1].replace(".rbh", "")
        return df

    monkeypatch.setattr(paf2.rbh_tools, "parse_rbh", fake_parse_rbh)
    monkeypatch.setattr(paf2.rbh_tools, "rbhdf_to_alglocdf", fake_rbhdf_to_alglocdf)
    monkeypatch.setattr(paf2.rbh_tools, "rbh_to_scafnum", lambda _rbhdf, _sample: 4)
    monkeypatch.setattr(
        paf2,
        "taxids_to_taxidstringdict",
        lambda _taxids: {
            10: "1;131567;2759;33154;33208;10",
            20: "1;131567;2759;33154;33208;20",
        },
    )

    assert paf2.main() is None
    assert (tmp_path / "selected_genomes.txt").exists()
    assert (tmp_path / "per_species_ALG_presence_fusions.tsv").exists()
