from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from egt.custom_taxonomy import CustomTopologyWarning
from egt import plot_alg_fusions as paf


def test_color_helpers_and_alg_qc_functions():
    assert paf.hex_to_rgb("#ff7f00") == (255, 127, 0)
    assert paf.rgb_255_float_to_hex((255, 127, 0)) == "#ff7f00"
    assert "Qa" in paf.dict_BCnSALG_to_color()

    df = pd.DataFrame(
        {
            "species": ["sp1", "sp2", "sp3"],
            "taxid": [1, 2, 3],
            "taxidstring": ["1", "2", "3"],
            "changestrings": ["x", "y", "z"],
            "A": [0, 0, 1],
            "B": [1, 0, 1],
            ("A", "B"): [0, 0, 1],
        }
    )

    missing, present = paf.missing_present_ALGs(df, min_for_missing=2 / 3)
    assert missing == ["A"]
    assert "B" in present

    separated = paf.separate_ALG_pairs(df.assign(A=[2, 2, 2], B=[2, 2, 2]), min_for_noncolocalized=0.5)
    assert separated == [("A", "B")]

    unsplit = paf.unsplit_ALGs(df.assign(A=[1, 1, 2], B=[0, 1, 1]), max_frac_split=0.5)
    assert "A" in unsplit
    assert "B" in unsplit


def test_image_helpers_and_image_composition():
    df = pd.DataFrame(
        {
            "A": [1, 0],
            "B": [0, 1],
            ("A", "B"): [1, 0],
        }
    )
    assert paf._image_helper_get_ALG_columns(df) == ["A", "B"]

    img = paf.image_sp_matrix_to_presence_absence(df, color_dict={"A": "#ff0000", "B": "#00ff00"})
    assert img.size == (2, 12)
    assert img.getpixel((0, 10)) == (255, 255, 255)

    coloc = paf.image_colocalization_matrix(
        pd.DataFrame(
            {
                "A": [1, 0],
                "B": [1, 1],
                ("A", "B"): [1, 0],
            }
        ),
        color_dict={"A": "#ff0000", "B": "#00ff00"},
        clustering=False,
        missing_data_color="#990000",
    )
    assert coloc.size == (1, 12)
    assert coloc.getpixel((0, 10)) == (255, 255, 255)
    assert coloc.getpixel((0, 11)) == (153, 0, 0)

    left = paf.image_vertical_barrier(2, 3, "#ffffff")
    right = paf.image_vertical_barrier(1, 2, "#000000")
    hcat = paf.image_concatenate_horizontally([left, right], valign="bottom")
    centered = paf.image_concatenate_horizontally([left, right], valign="center")
    vcat = paf.image_concatenate_vertically([left, right])
    assert hcat.size == (3, 3)
    assert centered.size == (3, 3)
    assert centered.getpixel((2, 0)) == (0, 0, 0)
    assert vcat.size == (2, 5)


def test_parse_args_validates_inputs(tmp_path: Path):
    directory = tmp_path / "rbhs"
    directory.mkdir()
    alg_rbh = tmp_path / "alg.rbh"
    alg_rbh.write_text("rbh\n")

    args = paf.parse_args(["-d", str(directory), "-a", "ALG", "-r", str(alg_rbh)])
    assert args.ALGname == "ALG"

    with pytest.raises(ValueError, match="tree info file"):
        paf.parse_args(
            ["-d", str(directory), "-a", "ALG", "-r", str(alg_rbh), "-t", str(tmp_path / "missing.tsv")]
        )
    with pytest.raises(ValueError, match="directory .* does not exist"):
        paf.parse_args(["-d", str(tmp_path / "missingdir"), "-a", "ALG", "-r", str(alg_rbh)])


def test_parse_alg_fusions_and_plot(tmp_path: Path):
    rbh = tmp_path / "sample.rbh"
    pd.DataFrame(
        {
            "rbh": ["r1", "r2"],
            "gene_group": ["A", "B"],
            "color": ["#111111", "#222222"],
            "ALG_scaf": ["alg1", "alg2"],
            "ALG_gene": ["ag1", "ag2"],
            "ALG_pos": [1, 2],
            "species_scaf": ["chr1", "chr1"],
            "species_gene": ["g1", "g2"],
            "species_pos": [11, 12],
            "whole_FET": [0.001, 0.001],
        }
    ).to_csv(rbh, sep="\t", index=False)

    alg_df = pd.DataFrame(
        {
            "ALGname": ["A", "B"],
            "Color": ["#aa0000", "#00aa00"],
            "Size": [10, 20],
        }
    )

    fusion_df = paf.parse_ALG_fusions([str(rbh)], alg_df, "ALG", minsig=0.01)
    assert set(fusion_df["ALGname"]) == {"A", "B"}
    assert fusion_df["fused_quantity"].tolist() == [1, 1]

    paf.plot_ALG_fusions(fusion_df, alg_df, "ALG", outprefix=str(tmp_path / "fusion_plot"))
    assert (tmp_path / "fusion_plot_ALG_fusions.pdf").exists()


def test_parse_alg_fusions_validation_errors(tmp_path: Path):
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
        paf.parse_ALG_fusions([str(missing_alg_cols)], alg_df, "ALG", minsig=0.01)

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
        paf.parse_ALG_fusions([str(missing_species_cols)], alg_df, "ALG", minsig=0.01)


def test_phylogeny_and_lineage_image_helpers(monkeypatch):
    class FakeNCBI:
        def get_lineage(self, taxid):
            mapping = {
                101: [1, 33208, 10197, 101],
                201: [1, 33208, 6040, 201],
                301: [1, 2759, 301],
                401: [1, 33208, 6072, 33213, 401],
            }
            return mapping[taxid]

    monkeypatch.setattr(paf, "NCBITaxa", lambda: FakeNCBI())

    with pytest.warns(CustomTopologyWarning, match="Eumetazoa"):
        assert paf.apply_custom_phylogeny([1, 33208, 6072, 10197, 101], 101, None) == [1, 33208, 10197, 101]
    assert paf.apply_custom_phylogeny([1, 33208, 6040, 201], 201, None) == [1, 33208, -67, 6040, 201]
    with pytest.warns(CustomTopologyWarning, match="Eumetazoa"):
        assert paf.apply_custom_phylogeny([1, 33208, 6072, 33213, 401], 401, None) == [
            1,
            33208,
            -67,
            -68,
            33213,
            401,
        ]
    with pytest.warns(CustomTopologyWarning, match="Eumetazoa"):
        assert paf.apply_custom_phylogeny([1, 33208, 6072, 501], 501, None) == [1, 33208, -67, -68, 501]

    with pytest.warns(CustomTopologyWarning, match="Eumetazoa"):
        taxidstrings = paf.taxids_to_taxidstringdict([101, 201, 301, 401], use_custom_phylogeny=True)
    assert taxidstrings[101] == "1;33208;10197;101"
    assert taxidstrings[201] == "1;33208;-67;6040;201"
    assert taxidstrings[301] == "1;2759;301"
    assert taxidstrings[401] == "1;33208;-67;-68;33213;401"

    image = paf.image_sp_matrix_to_lineage(
        [
            "1;33208;10197;101",
            "1;33208;-67;6040;201",
            "1;2759;301",
        ]
    )
    assert image.size[1] == 3


def test_load_calibrated_tree_and_assign_colors(tmp_path: Path):
    node_info = tmp_path / "node_info.tsv"
    pd.DataFrame(
        {
            "taxid": [10, 20, 30],
            "lineage_string": ["1;10", None, "1;30"],
            "nodeage": [0.5, 1.5, np.nan],
        }
    ).to_csv(node_info, sep="\t", index=False)

    loaded = paf.load_calibrated_tree(node_info)
    assert loaded["lineages"] == {10: "1;10", 30: "1;30"}
    assert loaded["ages"] == {10: 0.5, 20: 1.5}

    import networkx as nx

    graph = nx.DiGraph()
    graph.add_edges_from([("root", "a"), ("root", "b")])
    colors = {"a": np.array([1.0, 0.0, 0.0]), "b": np.array([0.0, 0.0, 1.0])}
    paf.assign_colors_to_nodes(graph, "root", colors)
    assert tuple(colors["root"]) == (0.5, 0.0, 0.5)


def test_standard_plot_out_and_missing_vs_colocalized(tmp_path: Path, monkeypatch):
    perspchrom = pd.DataFrame(
        {
            "species": ["sp1", "sp2"],
            "taxid": [1, 2],
            "taxidstring": ["1;10197", "1;6040"],
            "A": [1, 0],
            "B": [0, 1],
            "('A', 'B')": [1, 0],
        }
    )

    monkeypatch.setattr(
        paf,
        "image_sp_matrix_to_lineage",
        lambda _series: paf.image_vertical_barrier(2, len(perspchrom), "#111111"),
    )
    monkeypatch.setattr(
        paf,
        "image_sp_matrix_to_presence_absence",
        lambda _df, color_dict=None: paf.image_vertical_barrier(3, len(perspchrom), "#222222"),
    )
    monkeypatch.setattr(
        paf,
        "image_colocalization_matrix",
        lambda _df, clustering, color_dict=None, missing_data_color="#5d001e": paf.image_vertical_barrier(
            1 if clustering else 2, len(perspchrom), "#333333"
        ),
    )
    monkeypatch.setattr(plt, "show", lambda: None)

    paf.standard_plot_out(perspchrom.copy(), str(tmp_path / "plot"))
    assert (tmp_path / "plot_composite_image.png").exists()
    assert (tmp_path / "plot_composite_image_unclustered.png").exists()

    paf.plot_missing_vs_colocalized(perspchrom.rename(columns={"('A', 'B')": ("A", "B")}), str(tmp_path / "scatter"))


def test_standard_plot_out_validates_taxid_order(tmp_path: Path):
    perspchrom = pd.DataFrame(
        {
            "species": ["sp1"],
            "taxid": [1],
            "taxidstring": ["1;10197"],
            "A": [1],
            ("A", "B"): [0],
        }
    )

    with pytest.raises(ValueError, match="taxid 9 is not in the dataframe"):
        paf.standard_plot_out(perspchrom, str(tmp_path / "plot"), taxid_order=[9], safe=True)


def test_standard_plot_out_filters_missing_taxids_when_not_safe(tmp_path: Path, monkeypatch):
    perspchrom = pd.DataFrame(
        {
            "species": ["sp1", "sp2"],
            "taxid": [1, 2],
            "taxidstring": ["1;10197", "1;6040"],
            "A": [1, 0],
            ("A", "B"): [0, 1],
        }
    )

    monkeypatch.setattr(
        paf,
        "image_sp_matrix_to_lineage",
        lambda _series: paf.image_vertical_barrier(2, len(_series), "#111111"),
    )
    monkeypatch.setattr(
        paf,
        "image_sp_matrix_to_presence_absence",
        lambda _df, color_dict=None: paf.image_vertical_barrier(2, len(_df), "#222222"),
    )
    monkeypatch.setattr(
        paf,
        "image_colocalization_matrix",
        lambda _df, clustering, color_dict=None, missing_data_color="#5d001e": paf.image_vertical_barrier(
            1, len(_df), "#333333"
        ),
    )
    paf.standard_plot_out(perspchrom, str(tmp_path / "ordered_plot"), taxid_order=[9, 2], safe=False)
    assert (tmp_path / "ordered_plot_composite_image.png").exists()


def test_rbh_files_to_locdf_and_perspchrom_filters_missing_calibrated_taxa(tmp_path: Path, monkeypatch):
    rbh1 = tmp_path / "SpeciesA-101-GCA1.rbh"
    rbh2 = tmp_path / "SpeciesB-202-GCA1.rbh"
    alg_rbh = tmp_path / "ALG.rbh"
    for path in [rbh1, rbh2, alg_rbh]:
        path.write_text("placeholder\n")

    alg_df = pd.DataFrame(
        {
            "ALGname": ["A", "B"],
            "Color": ["#aa0000", "#00aa00"],
            "Size": [10, 5],
        }
    )

    def fake_parse_rbh(path):
        sample = Path(path).stem
        return pd.DataFrame(
            {
                "ALG_scaf": ["alg1", "alg2"],
                "ALG_gene": ["ag1", "ag2"],
                "ALG_pos": [1, 2],
                f"{sample}_scaf": ["chr1", "chr1"],
                f"{sample}_gene": ["g1", "g2"],
                f"{sample}_pos": [11, 12],
            }
        )

    def fake_rbhdf_to_alglocdf(_rbhdf, _minsig, _algname):
        sample = [c[:-5] for c in _rbhdf.columns if c.endswith("_scaf") and not c.startswith("ALG_")][0]
        splitdf = pd.DataFrame(
            {
                "sample": [sample, sample],
                "gene_group": ["A", "B"],
                "scaffold": ["chr1", "chr1"],
                "pvalue": [0.001, 0.001],
                "num_genes": [1, 1],
                "frac_of_this_ALG_on_this_scaffold": [1.0, 1.0],
            }
        )
        return splitdf, sample

    monkeypatch.setattr(paf.rbh_tools, "parse_ALG_rbh_to_colordf", lambda _path: alg_df)
    monkeypatch.setattr(paf.rbh_tools, "parse_rbh", fake_parse_rbh)
    monkeypatch.setattr(paf.rbh_tools, "rbhdf_to_alglocdf", fake_rbhdf_to_alglocdf)
    monkeypatch.setattr(paf.rbh_tools, "rbh_to_scafnum", lambda _rbhdf, _sample: 1)

    cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        locdf, perspchrom = paf.rbh_files_to_locdf_and_perspchrom(
            [str(rbh1), str(rbh2)],
            str(alg_rbh),
            0.01,
            "ALG",
            calibrated_tree_data={"lineages": {101: "1;101"}, "ages": {101: 0.5}},
        )
    finally:
        os.chdir(cwd)

    assert list(locdf["sample"].unique()) == ["SpeciesA-101-GCA1"]
    assert list(perspchrom["species"]) == ["SpeciesA-101-GCA1"]
    assert perspchrom.loc[0, "A"] == 1
    assert perspchrom.loc[0, "B"] == 1
    coloc_value = perspchrom[[("A", "B")]].iloc[0, 0]
    assert coloc_value == 1
    assert (tmp_path / "missing_taxa_from_calibrated_tree.txt").exists()


def test_compute_changestring_for_species_and_batch_loading(tmp_path: Path):
    perspchrom = pd.DataFrame(
        {
            "species": ["target", "sister", "outgroup"],
            "taxidstring": ["1;2;3", "1;2;4", "1;5"],
            "A": [1, 1, 0],
            "B": [0, 1, 1],
            ("A", "B"): [0, 0, 0],
        }
    )
    row = perspchrom.iloc[0]

    changestring, was_loaded = paf.compute_changestring_for_species(
        row,
        perspchrom,
        ALG_columns=["A", "B"],
        ALG_combos=[("A", "B")],
        min_for_missing=0.8,
        min_for_noncolocalized=0.5,
        checkpoint_dir=str(tmp_path),
        ALG_node="1",
    )
    assert was_loaded is False
    assert changestring.startswith("1-")
    assert "['B']" in changestring

    loaded_changestring, loaded_flag = paf.compute_changestring_for_species(
        row,
        perspchrom,
        ALG_columns=["A", "B"],
        ALG_combos=[("A", "B")],
        min_for_missing=0.8,
        min_for_noncolocalized=0.5,
        checkpoint_dir=str(tmp_path),
        ALG_node="1",
    )
    assert loaded_flag is True
    assert loaded_changestring == changestring

    batch = paf.process_species_batch(
        (
            [0],
            perspchrom,
            ["A", "B"],
            [("A", "B")],
            0.8,
            0.5,
            str(tmp_path),
            "1",
        )
    )
    assert batch == [(0, changestring, True)]


def test_save_umap_plotly_and_matplotlib(monkeypatch, tmp_path: Path):
    tree_df = pd.DataFrame(
        {
            "metric1": [0.0, 1.0, 2.0],
            "metric2": [2.0, 1.0, 0.0],
            "color": ["#111111", "#222222", "#333333"],
        },
        index=["node", "sp-1", "sp-2"],
    )

    class FakeReducer:
        def fit_transform(self, X):
            return np.column_stack([np.arange(len(X), dtype=float), np.arange(len(X), dtype=float) + 0.5])

    monkeypatch.setattr(paf.umap, "UMAP", lambda: FakeReducer())
    monkeypatch.setattr(np.random, "normal", lambda loc, scale, shape: np.zeros(shape))

    written = []

    class FakeFig:
        def write_html(self, path):
            written.append(path)

    fake_px = types.SimpleNamespace(scatter=lambda *args, **kwargs: FakeFig())
    sys.modules["plotly"] = types.ModuleType("plotly")
    sys.modules["plotly.express"] = fake_px

    paf.save_UMAP_plotly(tree_df, str(tmp_path / "umap"))
    assert str(tmp_path / "umap.withnodes.html") in written
    assert str(tmp_path / "umap.withoutnodes.html") in written

    scatter_calls = []
    monkeypatch.setattr(plt, "scatter", lambda *args, **kwargs: scatter_calls.append((args, kwargs)))
    monkeypatch.setattr(plt, "show", lambda: None)
    paf.save_UMAP(tree_df)
    assert scatter_calls


def test_main_uses_cached_inputs_and_generates_changestrings(tmp_path: Path, monkeypatch):
    rbhs = tmp_path / "rbhs"
    rbhs.mkdir()
    (rbhs / "dummy.rbh").write_text("placeholder\n")

    pd.DataFrame({"sample": ["s1"], "gene_group": ["A"]}).to_csv(tmp_path / "locdf.tsv", sep="\t", index=False)
    perspchrom = pd.DataFrame(
        {
            "species": ["target"],
            "taxid": [999],
            "taxidstring": ["1;131567;2759;33154;33208;999"],
            "A": [1],
            "B": [0],
            "('A', 'B')": [0],
        }
    )
    perspchrom.to_csv(tmp_path / "perspchrom.tsv", sep="\t", index=False)
    pd.DataFrame({"metric": [1], "color": ["#111111"]}, index=["node"]).to_csv(
        tmp_path / "tree1.tsv.gz", sep="\t", compression="gzip"
    )

    args = type(
        "Args",
        (),
        {
            "directory": str(rbhs),
            "ALG_rbh": str(tmp_path / "ALG.rbh"),
            "minsig": 0.01,
            "ALGname": "ALG",
            "tree_info": None,
            "parallel": False,
            "ncores": 1,
        },
    )()
    monkeypatch.setattr(paf, "parse_args", lambda argv=None: args)
    monkeypatch.setattr(paf, "save_UMAP_plotly", lambda *args, **kwargs: None)
    monkeypatch.chdir(tmp_path)

    assert paf.main([]) == 0
    saved = pd.read_csv(tmp_path / "per_species_ALG_presence_fusions.tsv", sep="\t")
    assert "changestrings" in saved.columns
    assert "['B']" in saved.loc[0, "changestrings"]
    assert (tmp_path / "changestring_checkpoints" / "target.txt").exists()
