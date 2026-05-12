from __future__ import annotations

import gzip
from pathlib import Path

import pandas as pd
import pytest

from egt import phylotreeumap as ptu


class _FakeNCBI:
    def get_taxid_translator(self, taxids):
        names = {
            1: "Root",
            2: "Clade-A",
            3: "Clade B",
            4: "Removed.Clade",
            6340: "Annelida",
            42113: "Clitellata",
            6392: "Lumbricidae",
            10197: "Ctenophora",
            6040: "Porifera",
            6073: "Cnidaria",
        }
        return {taxid: names[taxid] for taxid in taxids}


class _FakeMapper:
    embedding_ = [[0.0, 1.0], [1.0, 0.0]]


class _FakeReducer:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit(self, values):
        self.values = values
        return _FakeMapper()


def test_taxid_analysis_helpers(monkeypatch):
    monkeypatch.setattr(ptu, "NCBITaxa", lambda: _FakeNCBI())

    analyses = ptu.taxids_to_analyses([[[2], [4]], [[3], []]])
    assert "CladeA_2_without_4" in analyses
    assert analyses["CladeB_3_without_None"] == [[3], []]

    with pytest.raises(ValueError):
        ptu.taxids_to_analyses([])
    with pytest.raises(ValueError):
        ptu.taxids_to_analyses([[[], []]])

    captured = {}
    def fake_taxids_to_analyses(taxids):
        captured["taxids"] = taxids
        return {"ok": taxids}
    monkeypatch.setattr(ptu, "taxids_to_analyses", fake_taxids_to_analyses)
    result = ptu.taxids_of_interest_to_analyses()
    assert "taxids" in captured
    assert isinstance(result, dict)


def test_odog_iter_pairwise_distance_matrix_and_precomputed_umap(tmp_path: Path, monkeypatch):
    sampledf = tmp_path / "samples.tsv"
    dis1 = tmp_path / "a.tsv.gz"
    dis2 = tmp_path / "b.tsv.gz"
    pd.DataFrame(
        {"rbh1": ["x", "x"], "rbh2": ["y", "z"], "distance": [1.0, 3.0]}
    ).to_csv(dis1, sep="\t", index=False, compression="gzip")
    pd.DataFrame(
        {"rbh1": ["x", "x"], "rbh2": ["y", "z"], "distance": [2.0, 5.0]}
    ).to_csv(dis2, sep="\t", index=False, compression="gzip")
    pd.DataFrame(
        {"sample": ["s1", "s2"], "dis_filepath": [str(dis1), str(dis2)]}
    ).to_csv(sampledf, sep="\t")

    out_tsv = tmp_path / "dist.tsv"
    assert ptu.odog_iter_pairwise_distance_matrix(str(sampledf), str(out_tsv), metric="mad") == 0
    written = pd.read_csv(out_tsv, sep="\t", index_col=0)
    assert written.loc["s1", "s2"] == pytest.approx(1.5)

    out_corr = tmp_path / "dist_corr.tsv"
    assert ptu.odog_iter_pairwise_distance_matrix(str(sampledf), str(out_corr), metric="corr") == 0
    corr_df = pd.read_csv(out_corr, sep="\t", index_col=0)
    assert corr_df.loc["s1", "s2"] == pytest.approx(0.0)

    monkeypatch.setattr(ptu.umap, "UMAP", _FakeReducer)
    calls = {}
    monkeypatch.setattr(ptu, "umap_mapper_to_bokeh", lambda mapper, cdf, outpath, plot_title=None: calls.setdefault("html", (outpath, plot_title)))
    monkeypatch.setattr(
        ptu,
        "umap_mapper_to_df",
        lambda mapper, cdf: cdf.assign(UMAP1=[row[0] for row in mapper.embedding_], UMAP2=[row[1] for row in mapper.embedding_]),
    )

    emb_out = tmp_path / "embedding.tsv"
    html_out = tmp_path / "embedding.html"
    assert ptu.plot_precomputed_umap(str(sampledf), str(out_tsv), 0, 5, 0.1, str(emb_out), str(html_out)) == 0
    assert emb_out.exists()
    assert calls["html"][0] == str(html_out)
