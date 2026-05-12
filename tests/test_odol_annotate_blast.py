from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from egt import odol_annotate_blast as oab


class _FakeMapper:
    def __init__(self, n_rows: int):
        self.embedding_ = np.arange(n_rows * 2, dtype=float).reshape(n_rows, 2)


class _FakeUMAP:
    def __init__(self, n_neighbors=None, min_dist=None):
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist

    def fit(self, matrix):
        return _FakeMapper(matrix.shape[0])


def test_tsvgz_calc_umap_writes_embedding(tmp_path: Path, monkeypatch):
    matrix_path = tmp_path / "matrix.tsv"
    pd.DataFrame({"a": [0, 1], "b": [1, 0]}, index=["rbh1", "rbh2"]).to_csv(matrix_path, sep="\t")
    sampledf = tmp_path / "samples.tsv"
    pd.DataFrame({"sample": ["s1"]}).to_csv(sampledf, sep="\t")
    alg_rbh = tmp_path / "alg.rbh"
    alg_rbh.write_text("placeholder\n")
    outdf = tmp_path / "umap.tsv"

    monkeypatch.setattr(oab, "parse_rbh", lambda _path: pd.DataFrame({"rbh": ["rbh1", "rbh2"], "gene_group": ["A", "B"], "color": ["#111111", "#222222"]}))
    monkeypatch.setattr(oab.umap, "UMAP", _FakeUMAP)

    oab.tsvgz_calcUMAP("sample", str(sampledf), str(alg_rbh), str(matrix_path), "large", 5, 0.1, str(outdf))

    written = pd.read_csv(outdf, sep="\t", index_col=0)
    assert {"UMAP1", "UMAP2", "gene_group", "color"} <= set(written.columns)
    assert list(written["gene_group"]) == ["A", "B"]


def test_tsvgz_calc_umap_validates_types(tmp_path: Path):
    matrix_path = tmp_path / "matrix.tsv"
    pd.DataFrame({"a": [0], "b": [1]}, index=["rbh1"]).to_csv(matrix_path, sep="\t")
    sampledf = tmp_path / "samples.tsv"
    pd.DataFrame({"sample": ["s1"]}).to_csv(sampledf, sep="\t")
    alg_rbh = tmp_path / "alg.rbh"
    alg_rbh.write_text("placeholder\n")

    with pytest.raises(ValueError, match="n_neighbors"):
        oab.tsvgz_calcUMAP("sample", str(sampledf), str(alg_rbh), str(matrix_path), "large", "5", 0.1, str(tmp_path / "out.tsv"))
    with pytest.raises(ValueError, match="min_dist"):
        oab.tsvgz_calcUMAP("sample", str(sampledf), str(alg_rbh), str(matrix_path), "large", 5, 1, str(tmp_path / "out.tsv"))


def test_umapdf_one_species_one_query_handles_empty_inputs(tmp_path: Path):
    umapdf = tmp_path / "empty_umap.tsv"
    blastp = tmp_path / "empty.blastp"
    umapdf.write_text("")
    blastp.write_text("")
    outpdf = tmp_path / "annotated.pdf"

    oab.umapdf_one_species_one_query(str(umapdf), str(blastp), "analysis", "ALG", 5, 0.1, "query", str(outpdf), species="species1")

    assert outpdf.exists()


def test_umapdf_one_species_one_query_writes_annotations(tmp_path: Path):
    umapdf = tmp_path / "umap.tsv"
    pd.DataFrame(
        {
            "UMAP1": [0.0, 1.0],
            "UMAP2": [1.0, 0.0],
            "color": ["#111111", "#222222"],
        },
        index=["alg1", "alg2"],
    ).to_csv(umapdf, sep="\t")
    blastp = tmp_path / "hits.blastp"
    pd.DataFrame(
        [
            ["query1", "prot1", 99.0, 10, 0, 0, 1, 10, 1, 10, 0.0, 100, "chr1", "['alg1']", "[5]", "[42]"],
            ["query2", "prot2", 99.0, 10, 0, 0, 1, 10, 1, 10, 0.0, 90, "chr1", "['missing']", "[5]", "[42]"],
        ]
    ).to_csv(blastp, sep="\t", header=False, index=False)
    outpdf = tmp_path / "annotated.pdf"

    oab.umapdf_one_species_one_query(str(umapdf), str(blastp), "analysis", "ALG", 5, 0.1, "query", str(outpdf), species="species1")

    annotated = pd.read_csv(outpdf.with_suffix(".df"), sep="\t", index_col=0)
    assert annotated.loc["alg1", "blastp_best_hits"] != "[]"
    assert outpdf.exists()


def test_umapdf_reimbedding_bokeh_plot_one_species_writes_html(tmp_path: Path, monkeypatch):
    blastdf = tmp_path / "blast.tsv"
    pd.DataFrame(
        {
            "UMAP1": [0.0, 1.0],
            "UMAP2": [1.0, 0.0],
            "blastp_best_hits": [["q1"], ["q2"]],
            "blastp_color": [np.nan, "#aa0000"],
            "color": ["#00aa00", "blue"],
        },
        index=["loc1", "loc2"],
    ).to_csv(blastdf, sep="\t")

    called = {}
    monkeypatch.setattr(oab, "output_file", lambda path: called.setdefault("output", path))
    monkeypatch.setattr(oab, "save", lambda layout: called.setdefault("saved", layout))

    oab.umapdf_reimbedding_bokeh_plot_one_species(str(blastdf), "Plot", str(tmp_path / "plot.html"))

    assert called["output"].endswith("plot.html")
    assert called["saved"] is not None


def test_umapdf_reimbedding_bokeh_plot_one_species_validates_inputs(tmp_path: Path):
    missing = tmp_path / "missing.tsv"
    with pytest.raises(IOError, match="does not exist"):
        oab.umapdf_reimbedding_bokeh_plot_one_species(str(missing), "Plot", str(tmp_path / "plot.html"))

    blastdf = tmp_path / "blast.tsv"
    pd.DataFrame({"UMAP1": [0.0], "UMAP2": [1.0], "blastp_best_hits": [["x"]], "blastp_color": [np.nan], "color": ["#00aa00"]}).to_csv(
        blastdf, sep="\t"
    )
    with pytest.raises(IOError, match="plot_title"):
        oab.umapdf_reimbedding_bokeh_plot_one_species(str(blastdf), "", str(tmp_path / "plot.html"))
    with pytest.raises(IOError, match="must end with .html"):
        oab.umapdf_reimbedding_bokeh_plot_one_species(str(blastdf), "Plot", str(tmp_path / "plot.txt"))
    with pytest.raises(IOError, match="scalar must be an int or a float"):
        oab.umapdf_reimbedding_bokeh_plot_one_species(str(blastdf), "Plot", str(tmp_path / "plot.html"), scalar="bad")
