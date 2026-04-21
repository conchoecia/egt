from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix, save_npz

from egt import phylotreeumap as ptu


def test_mgt_mlt_plot_html_and_pdf_exports(tmp_path: Path, monkeypatch):
    umap_df = tmp_path / "mgt.tsv"
    pd.DataFrame(
        {
            "sample": ["s1", "s2"],
            "taxid": [1, 2],
            "taxname": ["one", "two"],
            "taxname_list_str": ["one;alpha", "two;beta"],
            "level_1": ["alpha", "beta"],
            "UMAP1": [0.0, 1.0],
            "UMAP2": [1.0, 0.0],
            "color": ["#111111", "#222222"],
        }
    ).to_csv(umap_df, sep="\t", index=False)

    html_out = tmp_path / "plot.html"
    ptu.mgt_mlt_plot_HTML(str(umap_df), str(html_out), analysis_type="MGT", plot_sizing_mode="stretch_width")
    assert html_out.exists()

    mlt_html_df = tmp_path / "mlt_html.tsv"
    pd.DataFrame(
        {
            "rbh": ["fam1", "fam2"],
            "gene_group": ["A", "B"],
            "UMAP1": [0.0, 1.0],
            "UMAP2": [1.0, 0.0],
            "color": ["#111111", "#222222"],
        }
    ).to_csv(mlt_html_df, sep="\t", index=False)
    ptu.mgt_mlt_plot_HTML(str(mlt_html_df), str(tmp_path / "mlt_plot.html"), analysis_type="MLT")
    assert (tmp_path / "mlt_plot.html").exists()

    mlt_df = tmp_path / "mlt.df"
    pd.DataFrame(
        {
            "UMAP1": [0.0, 1.0],
            "UMAP2": [1.0, 0.0],
            "color": ["#111111", "#222222"],
            "gene_group": ["A", "B"],
        }
    ).to_csv(mlt_df, sep="\t")
    ptu.plot_umap_pdf(str(mlt_df), str(tmp_path / "umap.pdf"), "Title", color_by_clade=False)
    assert (tmp_path / "umap.pdf").exists()

    ptu.plot_umap_pdf(str(tmp_path / "missing.df"), str(tmp_path / "empty.pdf"), "Missing", color_by_clade=False)
    assert (tmp_path / "empty.pdf").exists()


def test_plot_umap_phylogeny_pdf_and_df_only_pipeline(tmp_path: Path, monkeypatch):
    phylo_df = tmp_path / "phylo.df"
    pd.DataFrame(
        {
            "UMAP1": [0.0, 1.0],
            "UMAP2": [1.0, 0.0],
            "color": ["#111111", "#222222"],
            "taxid_list": ["[1, 10]", "[1, 20]"],
        }
    ).to_csv(phylo_df, sep="\t")

    class FakeNCBI:
        def get_taxid_translator(self, taxids):
            return {int(t): f"Taxon{t}" for t in taxids}

    class FakePhylogeny:
        def __init__(self, *_args, **_kwargs):
            self.plot_edges = [{"x1": 0.0, "x2": 1.0, "y1": 0.5, "y2": 0.5, "color": (0, 0, 0, 0.5)}]

    monkeypatch.setattr(ptu, "NCBITaxa", lambda: FakeNCBI())
    monkeypatch.setattr(ptu, "phylogeny_plotting", FakePhylogeny)
    monkeypatch.setattr(ptu.SplitLossColocTree, "color_dict_top", {"10": "#aa0000", "20": "#00aa00"})

    ptu.plot_umap_phylogeny_pdf(str(phylo_df), str(tmp_path / "phylogeny.pdf"), "sample", "small", 5, 0.1)
    assert (tmp_path / "phylogeny.pdf").exists()

    sampledf = tmp_path / "sampledf.tsv"
    pd.DataFrame({"sample": ["s1", "s2"]}).to_csv(sampledf, sep="\t")
    combo = tmp_path / "combo.tsv"
    combo.write_text("('fam1', 'fam2')\t0\n")
    coo = tmp_path / "matrix.npz"
    save_npz(coo, csr_matrix(np.array([[0.0], [1.0]])))

    fake_umap_mod = types.ModuleType("umap")

    class FakeReducer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def fit(self, matrix):
            return types.SimpleNamespace(embedding_=np.array([[0.0, 1.0], [1.0, 0.0]]))

    fake_umap_mod.UMAP = FakeReducer
    sys.modules["umap"] = fake_umap_mod

    monkeypatch.setattr(ptu, "algcomboix_file_to_dict", lambda _path: {("fam1", "fam2"): 0})
    outdf = tmp_path / "out.df"
    ptu.plot_umap_from_files_just_df(
        str(sampledf),
        str(combo),
        str(coo),
        "sample",
        999,
        1,
        0.1,
        str(outdf),
        print_prefix="[test] ",
        threads=1,
    )
    written = pd.read_csv(outdf, sep="\t", index_col=0)
    assert list(written.columns) == ["sample", "UMAP1", "UMAP2"]


def test_phylogeny_plotting_class_builds_edges():
    df = pd.DataFrame(
        {
            "taxid": [3, 3, 4, 5],
            "taxid_list": ["[1, 2, 3]", "[1, 2, 3]", "[1, 2, 4]", "[1, 5]"],
            "UMAP1": [0.0, 0.2, 1.0, 2.0],
            "UMAP2": [0.0, 0.2, 1.0, 0.0],
        }
    )
    plotter = ptu.phylogeny_plotting(df, "taxid_list", "UMAP1", "UMAP2")
    assert plotter.find_root() == 1
    assert any(edge["color"] == (1.0, 0.0, 0.0, 0.5) for edge in plotter.plot_edges)
    assert len(plotter.plot_edges) >= 3


def test_plot_umap_from_files_and_mlt_html(tmp_path: Path, monkeypatch):
    sampledf = tmp_path / "sampledf.tsv"
    cdf = pd.DataFrame(
        {
            "sample": ["s1", "s2"],
            "taxname": ["Taxon1", "Taxon2"],
            "color": ["#111111", "#222222"],
        }
    )
    cdf.to_csv(sampledf, sep="\t")

    combo = tmp_path / "combo.tsv"
    combo.write_text("('fam1', 'fam2')\t0\n")

    coo = tmp_path / "matrix.npz"
    save_npz(coo, csr_matrix(np.array([[0.0], [1.0]])))

    class FakeReducer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def fit(self, matrix):
            return types.SimpleNamespace(embedding_=np.array([[0.0, 1.0], [1.0, 0.0]]))

    monkeypatch.setattr(ptu.umap, "UMAP", FakeReducer)
    monkeypatch.setattr(ptu, "algcomboix_file_to_dict", lambda _path: {("fam1", "fam2"): 0})
    monkeypatch.setattr(
        ptu,
        "umap_mapper_to_bokeh",
        lambda mapper, cdf, outpath, plot_title=None: Path(outpath).write_text(plot_title or "html"),
    )
    monkeypatch.setattr(
        ptu,
        "umap_mapper_to_bokeh_topoumap",
        lambda mapper, df, outpath, plot_title=None: Path(outpath).write_text(plot_title or "html"),
        raising=False,
    )
    monkeypatch.setattr(
        ptu,
        "umap_mapper_to_df",
        lambda mapper, df: df.assign(UMAP1=[row[0] for row in mapper.embedding_], UMAP2=[row[1] for row in mapper.embedding_]),
    )
    monkeypatch.setattr(
        ptu,
        "umap_mapper_to_connectivity",
        lambda mapper, outfile, title=None: Path(outfile).write_text(title or "connectivity"),
    )
    monkeypatch.setattr(
        ptu.rbh_tools,
        "parse_rbh",
        lambda _path: pd.DataFrame({"rbh": ["fam1", "fam2"], "gene_group": ["A", "B"]}),
    )

    dfout = tmp_path / "odog.df"
    htmlout = tmp_path / "odog.html"
    connout = tmp_path / "odog_connectivity.svg"
    assert (
        ptu.plot_umap_from_files(
            str(sampledf),
            str(combo),
            str(coo),
            "sample",
            "small",
            1,
            0.1,
            str(dfout),
            str(htmlout),
            str(connout),
        )
        is None
    )
    assert dfout.exists()
    assert htmlout.exists()
    assert connout.exists()

    skip_df = tmp_path / "skip.df"
    skip_html = tmp_path / "skip.html"
    ptu.plot_umap_from_files(
        str(sampledf),
        str(combo),
        str(coo),
        "sample",
        "small",
        2,
        0.1,
        str(skip_df),
        str(skip_html),
    )
    assert skip_df.read_text() == ""
    assert skip_html.read_text() == ""

    algrbh = tmp_path / "alg.rbh"
    algrbh.write_text("placeholder\n")
    mlt_df = tmp_path / "mlt.df"
    mlt_html = tmp_path / "mlt.html"
    mlt_jpeg = tmp_path / "mlt_connectivity.svg"
    assert (
        ptu.mlt_umapHTML(
            "sample",
            str(sampledf),
            str(algrbh),
            str(coo),
            "small",
            1,
            0.1,
            str(mlt_df),
            str(mlt_html),
            str(mlt_jpeg),
            plot_jpeg=True,
        )
        == 0
    )
    assert mlt_df.exists()
    assert mlt_html.exists()
    assert mlt_jpeg.exists()


def test_legacy_phylo_tree_merge_and_unused_plot_helpers(tmp_path: Path, monkeypatch):
    class FakeNCBI:
        def get_lineage(self, taxid):
            return [1, int(taxid)]

        def get_taxid_translator(self, taxids):
            return {int(t): f"Taxon{t}" for t in taxids}

    class FakeReducer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def fit(self, matrix):
            return types.SimpleNamespace(embedding_=np.array([[0.0, 1.0], [1.0, 0.0]]))

    class FakePlotlyFigure:
        def __init__(self):
            self.traces = []

        def add_scatter(self, **kwargs):
            self.traces.append(kwargs)

        def write_html(self, outfile):
            Path(outfile).write_text("plotly\n")

    class FakeAxis:
        def __init__(self):
            self.title = None

        def set_title(self, title):
            self.title = title

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ptu, "NCBITaxa", lambda: FakeNCBI())
    monkeypatch.setattr(ptu.umap, "UMAP", FakeReducer)
    monkeypatch.setattr(
        ptu.umap.plot,
        "interactive",
        lambda mapper, color_key=None, labels=None, hover_data=None, point_size=None: {"labels": labels},
    )
    monkeypatch.setitem(
        sys.modules,
        "umap",
        types.SimpleNamespace(
            UMAP=FakeReducer,
            plot=types.SimpleNamespace(
                interactive=lambda mapper, color_key=None, labels=None, hover_data=None, point_size=None: {"labels": labels},
                diagnostic=lambda mapper, diagnostic_type=None: None,
                connectivity=lambda mapper, show_points=True, edge_bundling=None: None,
            ),
        ),
    )
    monkeypatch.setattr(ptu.bokeh.io, "output_file", lambda outfile: Path(outfile).write_text("bokeh\n"))
    monkeypatch.setattr(ptu.bokeh.io, "save", lambda plot: Path("distances_UMAP_sparse_bokeh.html").write_text(str(plot)))
    monkeypatch.setattr(
        ptu,
        "assign_colors_to_nodes",
        lambda G, root, node_colors: [node_colors.setdefault(node, np.array([0.1, 0.2, 0.3])) for node in G.nodes()],
    )
    monkeypatch.setattr(ptu, "rgb_255_float_to_hex", lambda _rgb: "#123456")
    monkeypatch.setattr(ptu.SplitLossColocTree, "color_dict_top", {"10": "#aa0000", "20": "#00aa00"})
    fake_plotly_express = types.SimpleNamespace(scatter=lambda: FakePlotlyFigure())
    monkeypatch.setitem(sys.modules, "plotly", types.SimpleNamespace(express=fake_plotly_express))
    monkeypatch.setitem(sys.modules, "plotly.express", fake_plotly_express)
    monkeypatch.setattr(ptu.plt, "clf", lambda: None)
    monkeypatch.setattr(ptu.plt, "figure", lambda *args, **kwargs: object())
    monkeypatch.setattr(ptu.plt, "scatter", lambda *args, **kwargs: object())
    monkeypatch.setattr(ptu.plt, "legend", lambda *args, **kwargs: object())
    monkeypatch.setattr(ptu.plt, "savefig", lambda outfile, *args, **kwargs: Path(outfile).write_text("pdf\n"))
    monkeypatch.setattr(ptu.sys, "exit", lambda: (_ for _ in ()).throw(SystemExit()))

    legacy_tree = ptu.PhyloTree()
    legacy_tree.add_lineage_string_sample_distances(
        "1;10",
        "sample-10",
        "ALG",
        pd.DataFrame({"rbh1": ["fam1"], "rbh2": ["fam2"], "distance": [5]}),
    )
    legacy_tree.add_lineage_string_sample_distances(
        "1;20",
        "sample-20",
        "ALG",
        pd.DataFrame({"rbh1": ["fam1"], "rbh2": ["fam2"], "distance": [7]}),
    )
    legacy_tree.alg_combo_to_ix = {("fam1", "fam2"): 0, ("fam2", "fam1"): 0}

    with pytest.raises(SystemExit):
        legacy_tree.merge_sampledistances_to_locdf()

    assert (tmp_path / "distances_UMAP_sparse_bokeh.html").exists()
    assert (tmp_path / "distances_UMAP_sparse_plotly.html").exists()
    assert (tmp_path / "distances_UMAP_sparse_matplotlib.pdf").exists()

    fake_axis = FakeAxis()
    monkeypatch.setattr(ptu, "ax", fake_axis, raising=False)
    monkeypatch.setattr(ptu.umap.plot, "diagnostic", lambda mapper, diagnostic_type=None: None)
    saved = []
    monkeypatch.setattr(ptu.plt, "savefig", lambda outfile, *args, **kwargs: saved.append((outfile, kwargs.get("dpi"))))
    ptu.umap_mapper_to_QC_plots(object(), str(tmp_path / "qc.png"), title="QC")
    assert fake_axis.title == "QC"
    assert saved[0][1] == 900

    connectivity_calls = []

    def fake_connectivity(mapper, show_points=True, edge_bundling=None):
        connectivity_calls.append(edge_bundling)
        return fake_axis

    monkeypatch.setattr(ptu.umap.plot, "connectivity", fake_connectivity)
    ptu.umap_mapper_to_connectivity(object(), str(tmp_path / "connectivity.svg"), bundled=False, title="Plain")
    ptu.umap_mapper_to_connectivity(object(), str(tmp_path / "connectivity2.png"), bundled=True, title="Bundled")
    assert connectivity_calls == [None, "hammer"]


def test_plot_umap_pdf_color_by_clade_and_warning_branches(tmp_path: Path, monkeypatch):
    class FakeNCBI:
        def get_taxid_translator(self, taxids):
            return {int(t): f"Taxon{t}" for t in taxids}

    class WarningReducer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def fit(self, matrix):
            raise UserWarning("Graph is not fully connected")

    phylo_df = tmp_path / "phylo.df"
    pd.DataFrame(
        {
            "UMAP1": [0.0, 1.0],
            "UMAP2": [1.0, 0.0],
            "color": ["#111111", "#222222"],
            "taxid_list": ["[1, 10]", "[1, 20]"],
        }
    ).to_csv(phylo_df, sep="\t")

    monkeypatch.setattr(ptu, "NCBITaxa", lambda: FakeNCBI())
    monkeypatch.setattr(ptu.SplitLossColocTree, "color_dict_top", {"10": "#aa0000", "20": "#00aa00"})
    ptu.plot_umap_pdf(str(phylo_df), str(tmp_path / "clade.pdf"), "Clades", color_by_clade=True)
    assert (tmp_path / "clade.pdf").exists()

    sampledf = tmp_path / "sampledf.tsv"
    pd.DataFrame({"sample": ["s1", "s2"]}).to_csv(sampledf, sep="\t")
    combo = tmp_path / "combo.tsv"
    combo.write_text("('fam1', 'fam2')\t0\n")
    coo = tmp_path / "warn_matrix.npz"
    save_npz(coo, csr_matrix(np.array([[0.0], [1.0]])))

    monkeypatch.setattr(ptu.umap, "UMAP", WarningReducer)
    monkeypatch.setattr(ptu, "algcomboix_file_to_dict", lambda _path: {("fam1", "fam2"): 0})

    dfout = tmp_path / "warn.df"
    htmlout = tmp_path / "warn.html"
    ptu.plot_umap_from_files(
        str(sampledf),
        str(combo),
        str(coo),
        "sample",
        "small",
        1,
        0.1,
        str(dfout),
        str(htmlout),
    )
    assert dfout.read_text() == ""
    assert "not fully connected" in htmlout.read_text()

    fake_umap_mod = types.ModuleType("umap")

    class WarningReducerDfOnly:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def fit(self, matrix):
            raise UserWarning("Graph is not fully connected")

    fake_umap_mod.UMAP = WarningReducerDfOnly
    sys.modules["umap"] = fake_umap_mod

    coo_with_zero = tmp_path / "warn_zero_matrix.npz"
    save_npz(coo_with_zero, csr_matrix(np.array([[0.0], [1.0]])))
    outdf = tmp_path / "warn_only.df"
    ptu.plot_umap_from_files_just_df(
        str(sampledf),
        str(combo),
        str(coo_with_zero),
        "sample",
        999,
        1,
        0.1,
        str(outdf),
        print_prefix="[warn] ",
        threads=1,
    )
    assert outdf.read_text() == ""
