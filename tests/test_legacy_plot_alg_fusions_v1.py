from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pandas as pd
import pytest

from egt.legacy import plot_alg_fusions_v1 as paf1


def test_legacy_v1_helpers(tmp_path, monkeypatch):
    assert paf1.hex_to_rgb("#ff7f00") == (255, 127, 0)
    assert "Qa" in paf1.dict_BCnSALG_to_color()

    rbh = tmp_path / "alg.rbh"
    pd.DataFrame(
        {
            "gene_group": ["A", "A", "B"],
            "color": ["#aa0000", "#aa0000", "#00aa00"],
        }
    ).to_csv(rbh, sep="\t", index=False)
    alg_df = paf1.parse_ALG_rbh_to_colordf(rbh)
    assert list(alg_df["ALGname"]) == ["B", "A"]

    df = pd.DataFrame(
        {
            "species": ["sp1", "sp2"],
            "taxid": [1, 2],
            "taxidstring": ["1;10", "1;20"],
            "A": [0, 0],
            "B": [1, 1],
            ("A", "B"): [0, 1],
        }
    )
    missing, present = paf1.missing_present_ALGs(df, min_for_missing=0.6)
    assert "A" in missing
    assert "B" in present
    separate_df = df.copy()
    separate_df["A"] = [1, 1]
    separate_df["B"] = [1, 1]
    separate_df[("A", "B")] = [0, 1]
    assert paf1.separate_ALG_pairs(separate_df, min_for_noncolocalized=0.5) == [("A", "B")]

    monkeypatch.setattr(paf1, "image_sp_matrix_to_lineage", lambda _series: paf1.image_vertical_barrier(1, len(df), "#111111"))
    monkeypatch.setattr(paf1, "image_sp_matrix_to_presence_absence", lambda _df, color_dict=None: paf1.image_vertical_barrier(2, len(df), "#222222"))
    monkeypatch.setattr(paf1, "image_colocalization_matrix", lambda _df, color_dict=None, clustering=False, missing_data_color="#5d001e": paf1.image_vertical_barrier(1, len(df), "#333333"))
    paf1.standard_plot_out(df.copy(), str(tmp_path / "legacy_v1"))
    assert (tmp_path / "legacy_v1_composite_image.png").exists()

    left = paf1.image_vertical_barrier(2, 3, "#ffffff")
    right = paf1.image_vertical_barrier(1, 2, "#000000")
    centered = paf1.image_concatenate_horizontally([left, right], valign="center")
    stacked = paf1.image_concatenate_vertically([left, right])
    assert centered.size == (3, 3)
    assert centered.getpixel((2, 0)) == (0, 0, 0)
    assert stacked.size == (2, 5)


def test_legacy_v1_parse_args_and_image_helpers(tmp_path, monkeypatch):
    directory = tmp_path / "rbhs"
    directory.mkdir()
    alg_rbh = tmp_path / "alg.rbh"
    alg_rbh.write_text("rbh\n")

    monkeypatch.setattr(
        sys,
        "argv",
        ["prog", "-d", str(directory), "-a", "ALG", "-p", "Species", "-r", str(alg_rbh)],
    )
    args = paf1.parse_args()
    assert args.prefix == "Species"

    monkeypatch.setattr(
        sys,
        "argv",
        ["prog", "-d", str(tmp_path / "missing"), "-a", "ALG", "-p", "Species", "-r", str(alg_rbh)],
    )
    with pytest.raises(ValueError, match="directory .* does not exist"):
        paf1.parse_args()

    lineage = paf1.image_sp_matrix_to_lineage(["1;33208;10197;101", "1;2759;301"])
    assert lineage.size[1] == 2

    presabs_df = pd.DataFrame({"A": [1, 0], "B": [0, 1], ("A", "B"): [1, 0]})
    presence = paf1.image_sp_matrix_to_presence_absence(
        presabs_df, color_dict={"A": "#ff0000", "B": "#00ff00"}
    )
    coloc = paf1.image_colocalization_matrix(
        pd.DataFrame({"A": [1, 0], "B": [1, 1], ("A", "B"): [1, 0]}),
        color_dict={"A": "#ff0000", "B": "#00ff00"},
        clustering=False,
        missing_data_color="#990000",
    )
    assert presence.size == (2, 12)
    assert coloc.getpixel((0, 11)) == (153, 0, 0)


def test_legacy_v1_parse_and_plot(tmp_path):
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

    fusion_df = paf1.parse_ALG_fusions([str(rbh)], alg_df, "ALG", minsig=0.01)
    paf1.plot_ALG_fusions(fusion_df, alg_df, "ALG", outprefix=str(tmp_path / "legacy_v1_plot"))
    assert (tmp_path / "legacy_v1_plot_ALG_fusions.pdf").exists()


def test_legacy_v1_parse_validation_errors(tmp_path):
    bad_alg = tmp_path / "bad_alg.rbh"
    pd.DataFrame({"gene_group": ["A"]}).to_csv(bad_alg, sep="\t", index=False)
    with pytest.raises(IOError, match="does not have the correct columns"):
        paf1.parse_ALG_rbh_to_colordf(bad_alg)

    alg_df = pd.DataFrame({"ALGname": ["A"], "Color": ["#aa0000"], "Size": [10]})

    missing_alg_cols = tmp_path / "missing_alg_cols.rbh"
    pd.DataFrame(
        {
            "rbh": ["r1"],
            "gene_group": ["A"],
            "color": ["#111111"],
            "Species_scaf": ["chr1"],
            "Species_gene": ["g1"],
            "Species_pos": [5],
            "whole_FET": [0.001],
        }
    ).to_csv(missing_alg_cols, sep="\t", index=False)
    with pytest.raises(IOError, match="correct columns for the ALG"):
        paf1.parse_ALG_fusions([str(missing_alg_cols)], alg_df, "ALG", minsig=0.01)

    missing_species_cols = tmp_path / "missing_species_cols.rbh"
    pd.DataFrame(
        {
            "rbh": ["r1"],
            "gene_group": ["A"],
            "color": ["#111111"],
            "ALG_scaf": ["alg1"],
            "ALG_gene": ["ag1"],
            "ALG_pos": [1],
            "Species_scaf": ["chr1"],
            "whole_FET": [0.001],
        }
    ).to_csv(missing_species_cols, sep="\t", index=False)
    with pytest.raises(IOError, match="correct columns for the species"):
        paf1.parse_ALG_fusions([str(missing_species_cols)], alg_df, "ALG", minsig=0.01)


def test_legacy_v1_main_smoke(tmp_path, monkeypatch):
    rbh_dir = tmp_path / "rbhs"
    rbh_dir.mkdir()
    (rbh_dir / "SpeciesA-10-GCF1.1.rbh").write_text("x\n")
    (rbh_dir / "SpeciesB-20-GCF1.1.rbh").write_text("x\n")
    alg_rbh = tmp_path / "alg.rbh"
    alg_rbh.write_text("placeholder\n")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        paf1,
        "parse_args",
        lambda: SimpleNamespace(directory=str(rbh_dir), ALGname="ALG", prefix="Species", ALG_rbh=str(alg_rbh), minsig=0.01),
    )
    monkeypatch.setattr(
        paf1,
        "parse_ALG_rbh_to_colordf",
        lambda _path: pd.DataFrame({"ALGname": ["A", "B"], "Color": ["#aa0000", "#00aa00"], "Size": [2, 1]}),
    )

    def fake_parse_rbh(path):
        sample = path.stem if hasattr(path, "stem") else str(path).split("/")[-1].replace(".rbh", "")
        other = sample
        df = pd.DataFrame(
            {
                "whole_FET": [0.001, 0.001],
                "ALG_scaf": ["A", "B"],
                "ALG_gene": ["ag1", "ag2"],
                "ALG_pos": [1, 2],
                f"{other}_scaf": ["chr1", "chr1"],
                f"{other}_gene": ["g1", "g2"],
                f"{other}_pos": [11, 12],
            }
        )
        df.attrs["sample"] = sample
        return df

    def fake_rbhdf_to_alglocdf(rbhdf, _minsig, _algname):
        sample = rbhdf.attrs["sample"]
        return (
            pd.DataFrame({"sample": [sample, sample], "gene_group": ["A", "B"], "scaffold": ["chr1", "chr1"]}),
            sample,
        )

    monkeypatch.setattr(paf1.rbh_tools, "parse_rbh", fake_parse_rbh)
    monkeypatch.setattr(paf1.rbh_tools, "rbhdf_to_alglocdf", fake_rbhdf_to_alglocdf)
    monkeypatch.setattr(paf1.rbh_tools, "rbh_to_scafnum", lambda _rbhdf, _sample: 4)

    class FakeNCBI:
        def get_lineage(self, taxid):
            return [1, 131567, 2759, 33154, 33208, int(taxid)]
        def get_taxid_translator(self, taxids):
            return {int(t): f"name_{t}" for t in taxids}
        def get_topology(self, taxids):
            class Leaf:
                def __init__(self, taxid):
                    self.name = str(taxid)
                def __str__(self):
                    return self.name
            class FakeTree:
                def __init__(self, taxids):
                    self._leaves = [Leaf(t) for t in taxids]
                def get_leaves(self):
                    return self._leaves
                def write(self, format=1, outfile=None):
                    if outfile is not None:
                        with open(outfile, "w", encoding="utf-8") as handle:
                            handle.write("(fake);\n")
            return FakeTree(taxids)

    monkeypatch.setitem(sys.modules, "ete4", types.SimpleNamespace(NCBITaxa=lambda: FakeNCBI(), Tree=object))
    monkeypatch.setattr(paf1, "standard_plot_out", lambda *_args, **_kwargs: None)

    assert paf1.main() is None
    assert (tmp_path / "selected_genomes.txt").exists()
    assert (tmp_path / "per_species_ALG_presence_fusions.tsv").exists()
