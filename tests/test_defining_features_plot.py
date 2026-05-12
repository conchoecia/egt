from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _import_module():
    fake_source = types.ModuleType("source")
    fake_rbh_tools = types.ModuleType("source.rbh_tools")
    fake_rbh_tools.parse_rbh = lambda _path: pd.DataFrame(
        {"rbh": ["fam1", "fam2"], "gene_group": ["ALG_A", "ALG_B"]}
    )
    fake_source.rbh_tools = fake_rbh_tools
    sys.modules["source"] = fake_source
    sys.modules["source.rbh_tools"] = fake_rbh_tools
    return importlib.import_module("egt.defining_features_plot")


def test_parse_args():
    dfp = _import_module()
    args = dfp.parse_args(
        [
            "--unique_pairs_path",
            "pairs.tsv",
            "--coo_path",
            "matrix.coo",
            "--sample_df_path",
            "sample.tsv",
            "--umapdf_path",
            "umap.tsv",
            "--coo_combination_path",
            "combo.tsv",
            "--rbh_file",
            "alg.rbh",
            "--min_num_samples",
            "2",
        ]
    )
    assert args.min_num_samples == 2


def test_main_writes_pdf_with_monkeypatched_inputs(tmp_path: Path, monkeypatch):
    dfp = _import_module()

    class FakeNCBI:
        def get_lineage(self, taxid):
            return [1, int(taxid)]

        def get_taxid_translator(self, taxids):
            return {int(t): f"Taxon{t}" for t in taxids}

    monkeypatch.setattr(dfp.ete3, "NCBITaxa", lambda: FakeNCBI())
    monkeypatch.setattr(dfp.odp_plot, "format_matplotlib", lambda: None)
    monkeypatch.setattr(dfp, "algcomboix_file_to_dict", lambda _path: {("fam1", "fam2"): 0})
    monkeypatch.setattr(dfp, "load_coo", lambda cdf, _coo, _combo, missing_value_as=None: np.array([[4.0], [6.0]]))
    monkeypatch.setattr(
        dfp.rbh_tools,
        "parse_rbh",
        lambda _path: pd.DataFrame({"rbh": ["fam1", "fam2"], "gene_group": ["ALG_A", "ALG_B"]}),
    )

    def fake_read_csv(path, sep="\t", index_col=None, **kwargs):
        name = Path(path).name
        if name == "umap.tsv":
            return pd.DataFrame({"sample": ["sp1", "sp2"], "UMAP1": [0.0, 1.0], "UMAP2": [1.0, 0.0]}, index=[0, 1])
        if name == "sample.tsv":
            return pd.DataFrame({"sample": ["sp1", "sp2"], "taxid_list": ["[1, 10]", "[1, 10, 20]"]}, index=[0, 1])
        if name == "pairs.tsv":
            return pd.DataFrame(
                {
                    "taxid": [10, 20],
                    "num_samples_in_taxid": [5, 0],
                    "unique_pairs": ["[(0, 1.0)]", "[]"],
                }
            )
        raise AssertionError(name)

    monkeypatch.setattr(dfp.pd, "read_csv", fake_read_csv)

    saved = []

    class FakePdf:
        def __init__(self, path):
            self.path = path

        def __enter__(self):
            saved.append(self.path)
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def savefig(self):
            return None

    monkeypatch.setattr(dfp, "PdfPages", FakePdf)

    assert dfp.main(
        [
            "--unique_pairs_path",
            "pairs.tsv",
            "--coo_path",
            "matrix.coo",
            "--sample_df_path",
            "sample.tsv",
            "--umapdf_path",
            "umap.tsv",
            "--coo_combination_path",
            "combo.tsv",
            "--rbh_file",
            "alg.rbh",
            "--min_num_samples",
            "1",
        ]
    ) == 0

    assert saved == ["definitive_colocalizations.pdf"]


def test_main_rejects_pairs_taxid_missing_from_sample_df(monkeypatch):
    dfp = _import_module()

    class FakeNCBI:
        def get_lineage(self, taxid):
            return [1, int(taxid)]

        def get_taxid_translator(self, taxids):
            return {int(t): f"Taxon{t}" for t in taxids}

    monkeypatch.setattr(dfp.ete3, "NCBITaxa", lambda: FakeNCBI())
    monkeypatch.setattr(dfp.odp_plot, "format_matplotlib", lambda: None)
    monkeypatch.setattr(dfp, "algcomboix_file_to_dict", lambda _path: {("fam1", "fam2"): 0})
    monkeypatch.setattr(dfp, "load_coo", lambda cdf, _coo, _combo, missing_value_as=None: np.array([[4.0]]))
    monkeypatch.setattr(
        dfp.rbh_tools,
        "parse_rbh",
        lambda _path: pd.DataFrame({"rbh": ["fam1", "fam2"], "gene_group": ["ALG_A", "ALG_B"]}),
    )

    def fake_read_csv(path, sep="\t", index_col=None, **kwargs):
        name = Path(path).name
        if name == "umap.tsv":
            return pd.DataFrame({"sample": ["sp1"], "UMAP1": [0.0], "UMAP2": [1.0]}, index=[0])
        if name == "sample.tsv":
            return pd.DataFrame({"sample": ["sp1"], "taxid_list": ["[1, 10]"]}, index=[0])
        if name == "pairs.tsv":
            return pd.DataFrame({"taxid": [99], "num_samples_in_taxid": [5], "unique_pairs": ["[(0, 1.0)]"]})
        raise AssertionError(name)

    monkeypatch.setattr(dfp.pd, "read_csv", fake_read_csv)

    with pytest.raises(ValueError, match="Taxid 99"):
        dfp.main(
            [
                "--unique_pairs_path",
                "pairs.tsv",
                "--coo_path",
                "matrix.coo",
                "--sample_df_path",
                "sample.tsv",
                "--umapdf_path",
                "umap.tsv",
                "--coo_combination_path",
                "combo.tsv",
                "--rbh_file",
                "alg.rbh",
            ]
        )
