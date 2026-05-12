from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest


def _import_module():
    fake_source = types.ModuleType("source")
    fake_rbh_tools = types.ModuleType("source.rbh_tools")
    fake_rbh_tools.parse_rbh = lambda _path: pd.DataFrame()
    fake_source.rbh_tools = fake_rbh_tools
    sys.modules["source"] = fake_source
    sys.modules["source.rbh_tools"] = fake_rbh_tools

    fake_fasta = types.ModuleType("fasta")
    fake_fasta.parse = lambda _path: []
    sys.modules["fasta"] = fake_fasta
    return importlib.import_module("egt.defining_features_plotRBH")


def test_parse_args_and_color_helpers(tmp_path: Path):
    dfp = _import_module()
    rbh = tmp_path / "rbh.tsv"
    pairs = tmp_path / "pairs.tsv"
    genome = tmp_path / "genome.fa"
    for path in [rbh, pairs, genome]:
        path.write_text("x\n")

    args = dfp.parse_args(
        [
            "--rbh_file",
            str(rbh),
            "--unique_pairs_path",
            str(pairs),
            "--clade_for_pairs",
            "123",
            "--genome_file",
            str(genome),
        ]
    )
    assert args.clade_for_pairs == 123

    mixed = dfp.interpolate_color("#000000", "#ffffff", 0.5)
    assert np.allclose(mixed[:3], np.array([0.5, 0.5, 0.5]))

    points = dfp.de_casteljau(
        np.array([0.0, 0.0]),
        np.array([0.0, 1.0]),
        np.array([1.0, 1.0]),
        np.array([1.0, 0.0]),
        0.5,
    )
    assert len(points) == 6
    assert dfp.legal_color("#ABCDEF") is True
    assert dfp.legal_color("#abcDEF") is False
    assert dfp.legal_color("ABCDEF") is False


def test_plot_bezier_arc_and_plot_arcs():
    dfp = _import_module()
    fig, ax = plt.subplots()
    dfp.plot_bezier_arc(ax, 0, 0, 2, 0, 1, "#000000", 0.8)
    dfp.plot_bezier_arc(ax, 0, 0, 2, 0, 1, "#000000", 0.8, color_gradient=("#ff0000", "#00ff00"))
    assert len(ax.patches) >= 4

    rbh = pd.DataFrame(
        {
            "rbh": ["alg1", "alg2"],
            "BCnSSimakov2022_gene": ["alg1", "alg2"],
            "BCnSSimakov2022_plotpos": [5, 15],
            "Species_plotpos": [10, 20],
            "Species_scaf": ["chr1", "chr1"],
            "BCnSSimakov2022_scaf": ["algchr", "algchr"],
            "color": ["#111111", "#222222"],
        }
    )
    unique_pairs = pd.DataFrame(
        {
            "ortholog1": ["alg1"],
            "ortholog2": ["alg2"],
            "stable_in_clade": [1],
            "unstable_in_clade": [0],
            "close_in_clade": [0],
        }
    )
    fig2, ax2 = plt.subplots()
    panel, indices = dfp.plot_arcs(ax2, "Species", "stable_in_clade", rbh, unique_pairs, fontsize=6)
    assert panel is ax2
    assert indices == []


def test_plot_arcs_gradient_and_argument_validation():
    dfp = _import_module()
    fig, ax = plt.subplots()
    rbh = pd.DataFrame(
        {
            "rbh": ["alg1", "alg2"],
            "BCnSSimakov2022_gene": ["alg1", "alg2"],
            "BCnSSimakov2022_plotpos": [5, 15],
            "Species_plotpos": [10, 20],
            "Species_scaf": ["chr1", "chr1"],
            "BCnSSimakov2022_scaf": ["algchr", "algchr"],
            "color": ["#111111", "#222222"],
        }
    )
    unique_pairs = pd.DataFrame(
        {
            "ortholog1": ["alg1"],
            "ortholog2": ["alg2"],
            "stable_in_clade": [1],
            "unstable_in_clade": [0],
            "close_in_clade": [0],
        }
    )
    blastdf = pd.DataFrame({"qseqid": ["q1"], "sseqid": ["prot1"], "scaf": ["chr1"]})
    chromdf = pd.DataFrame({"genome": ["prot1"], "start": [5], "stop": [25]})

    panel, indices = dfp.plot_arcs(
        ax,
        "Species",
        "stable_in_clade",
        rbh,
        unique_pairs,
        fontsize=6,
        blastdf=blastdf,
        blast_rbh_list={"alg1": {"q1"}},
        chromdf=chromdf,
        scaf_order=["chr1"],
        scaf_to_len={"chr1": 100},
        attempt_to_gradient_color=True,
    )
    assert panel is ax
    assert indices == [0]

    with pytest.raises(ValueError, match="color column"):
        dfp.plot_arcs(
            ax,
            "Species",
            "stable_in_clade",
            rbh.drop(columns=["color"]),
            unique_pairs,
            fontsize=6,
            attempt_to_gradient_color=True,
        )


def test_main_writes_outputs_with_stubbed_inputs(tmp_path: Path, monkeypatch):
    dfp = _import_module()

    class FakeNCBI:
        def get_taxid_translator(self, taxids):
            return {int(taxids[0]): "CladeName"}

    args = SimpleNamespace(
        rbh_file=str(tmp_path / "rbh.tsv"),
        unique_pairs_path=str(tmp_path / "pairs.tsv"),
        clade_for_pairs=123,
        genome_file=str(tmp_path / "genome.fa"),
        blast_file=str(tmp_path / "blast.tsv"),
        chrom_file=str(tmp_path / "chrom.tsv"),
        window=1000000,
    )
    monkeypatch.setattr(dfp, "parse_args", lambda argv=None: args)
    monkeypatch.setattr(dfp.odp_plot, "format_matplotlib", lambda: None)
    monkeypatch.setattr(dfp.ete3, "NCBITaxa", lambda: FakeNCBI())
    monkeypatch.setattr(
        dfp.fasta,
        "parse",
        lambda _path: [SimpleNamespace(id="chr1", seq="A" * 100), SimpleNamespace(id="chr2", seq="A" * 200)],
    )
    monkeypatch.setattr(
        dfp.rbh_tools,
        "parse_rbh",
        lambda _path: pd.DataFrame(
            {
                "rbh": ["alg1", "alg2"],
                "Species_gene": ["g1", "g2"],
                "Species_scaf": ["chr1", "chr2"],
                "Species_plotpos": [10, 30],
                "BCnSSimakov2022_gene": ["alg1", "alg2"],
                "BCnSSimakov2022_scaf": ["algchr1", "algchr2"],
                "BCnSSimakov2022_plotpos": [5, 25],
                "color": ["#111111", "#222222"],
            }
        ),
    )

    def fake_read_csv(path, sep="\t", header=None, **kwargs):
        name = Path(path).name
        if name == "pairs.tsv":
            return pd.DataFrame(
                {
                    "taxid": [123],
                    "ortholog1": ["alg1"],
                    "ortholog2": ["alg2"],
                    "stable_in_clade": [1],
                    "unstable_in_clade": [0],
                    "close_in_clade": [0],
                }
            )
        if name == "blast.tsv":
            return pd.DataFrame(
                [
                    [
                        "query1",
                        "prot1",
                        99.0,
                        10,
                        0,
                        0,
                        1,
                        10,
                        1,
                        10,
                        0.0,
                        100,
                        "chr1",
                        "['alg1']",
                        "[10]",
                        "[500]",
                    ]
                ]
            )
        if name == "chrom.tsv":
            return pd.DataFrame([["prot1", "chr1", "+", 10, 40]])
        raise AssertionError(name)

    monkeypatch.setattr(dfp.pd, "read_csv", fake_read_csv)
    monkeypatch.setattr(dfp, "plot_arcs", lambda panel, *args, **kwargs: (panel, [0]))

    cwd = Path.cwd()
    try:
        monkeypatch.chdir(tmp_path)
        assert dfp.main([]) == 0
    finally:
        Path.cwd()

    prefix = tmp_path / "Species_pairsfrom_CladeName_123_blast_query1"
    assert prefix.with_suffix(".pdf").exists()
    assert Path(str(prefix) + ".stable_by_blast.tsv").exists()
    assert Path(str(prefix) + ".unstable_by_blast.tsv").exists()
    assert Path(str(prefix) + ".close_by_blast.tsv").exists()
    assert Path(str(prefix) + ".scaffold_summary.tsv").exists()


def test_parse_args_and_main_empty_unique_pairs(tmp_path: Path, monkeypatch):
    dfp = _import_module()
    missing = tmp_path / "missing.tsv"
    with pytest.raises(FileNotFoundError):
        dfp.parse_args(
            [
                "--rbh_file",
                str(missing),
                "--unique_pairs_path",
                str(missing),
                "--clade_for_pairs",
                "123",
                "--genome_file",
                str(missing),
            ]
        )

    args = SimpleNamespace(
        rbh_file=str(tmp_path / "rbh.tsv"),
        unique_pairs_path=str(tmp_path / "pairs.tsv"),
        clade_for_pairs=123,
        genome_file=str(tmp_path / "genome.fa"),
        blast_file=None,
        chrom_file=None,
        window=1000000,
    )
    for path in [Path(args.rbh_file), Path(args.unique_pairs_path), Path(args.genome_file)]:
        path.write_text("x\n")

    monkeypatch.setattr(dfp, "parse_args", lambda argv=None: args)
    monkeypatch.setattr(dfp.odp_plot, "format_matplotlib", lambda: None)
    monkeypatch.setattr(dfp.fasta, "parse", lambda _path: [SimpleNamespace(id="chr1", seq="A" * 100)])
    monkeypatch.setattr(
        dfp.rbh_tools,
        "parse_rbh",
        lambda _path: pd.DataFrame(
            {
                "rbh": ["alg1"],
                "Species_gene": ["g1"],
                "Species_scaf": ["chr1"],
                "Species_plotpos": [10],
                "BCnSSimakov2022_gene": ["alg1"],
                "BCnSSimakov2022_scaf": ["algchr1"],
                "BCnSSimakov2022_plotpos": [5],
                "color": ["#111111"],
            }
        ),
    )
    monkeypatch.setattr(dfp.ete3, "NCBITaxa", lambda: type("FakeNCBI", (), {"get_taxid_translator": lambda self, taxids: {123: "Clade"}})())
    monkeypatch.setattr(dfp.pd, "read_csv", lambda path, sep="\t", header=None, **kwargs: pd.DataFrame(columns=["taxid", "ortholog1", "ortholog2", "stable_in_clade", "unstable_in_clade", "close_in_clade"]))

    with pytest.raises(SystemExit):
        dfp.main([])
