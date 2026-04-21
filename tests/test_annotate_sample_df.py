from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest

class FakeRecord:
    def __init__(self, ident: str, seq: str):
        self.id = ident
        self.seq = seq


fake_fasta = types.SimpleNamespace(parse=lambda _path: [])
sys.modules.setdefault("fasta", fake_fasta)
asd = importlib.import_module("egt.annotate_sample_df")


def test_gen_rbh_stats_and_stats_filepath_to_dict(monkeypatch, tmp_path: Path):
    alg_rbh = tmp_path / "alg.rbh"
    sample_rbh = tmp_path / "sample.rbh"
    alg_rbh.write_text("x\n")
    sample_rbh.write_text("x\n")

    alg_df = pd.DataFrame(
        {
            "rbh": ["r1", "r2", "r3", "r4"],
            "gene_group": ["A", "A", "B", "B"],
        }
    )
    sample_df = pd.DataFrame(
        {
            "rbh": ["r1", "r2", "r3"],
            "gene_group": ["A", "A", "B"],
            "whole_FET": [0.01, 0.20, 0.01],
            "Species_scaf": ["chr1", "chr2", "chr3"],
            "ALG_scaf": ["a", "a", "b"],
        }
    )

    monkeypatch.setattr(
        asd,
        "parse_rbh",
        lambda path: alg_df if str(path) == str(alg_rbh) else sample_df,
    )
    outfile = tmp_path / "stats.txt"
    asd.gen_rbh_stats(str(sample_rbh), str(alg_rbh), "ALG", str(outfile))
    parsed = asd.stats_filepath_to_dict(outfile)
    assert parsed["frac_ologs"] == 0.75
    assert parsed["frac_ologs_A"] == 0.5
    assert parsed["frac_ologs_B"] == 0.5


def test_data_availability_and_annotation_stats(monkeypatch, tmp_path: Path):
    sampledf = tmp_path / "sampledf.tsv"
    pd.DataFrame(
        {
            "sample": ["Alpha-111-GCA123456.1", "Beta-222-LOCALACC"],
            "taxname": ["Alpha species", "Beta species"],
            "taxid": [111, 222],
        }
    ).to_csv(sampledf, sep="\t")

    html = tmp_path / "availability.html"
    asd.gen_data_availability_statement(str(sampledf), str(html))
    text = html.read_text()
    assert "GCA_123456.1" in text
    assert "LOC_ALACC" in text

    algrbh = tmp_path / "alg.rbh"
    algrbh.write_text("x\n")
    monkeypatch.setattr(asd, "parse_rbh", lambda _path: pd.DataFrame({"rbh": ["protA", "protB"]}))
    monkeypatch.setattr(
        asd.fasta,
        "parse",
        lambda _path: [
            FakeRecord("protA_1", "MKT"),
            FakeRecord("protB_1", "MKTAA"),
            FakeRecord("other", "MK"),
        ],
    )
    annot = tmp_path / "annot.txt"
    asd.gen_annotation_stats("proteins.fa", str(algrbh), str(annot))
    parsed = asd.stats_filepath_to_dict(annot)
    assert parsed["num_proteins"] == 3
    assert parsed["from_rbh"] is True


def test_gen_genome_stats_and_plot_decay(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        asd.fasta,
        "parse",
        lambda _path: [
            FakeRecord("scaf1", "ATGCNNNNNNNNNNAT"),
            FakeRecord("scaf2", "GGCC"),
        ],
    )
    out = tmp_path / "genome_stats.txt"
    asd.gen_genome_stats("genome.fa", str(out))
    parsed = asd.stats_filepath_to_dict(out)
    assert parsed["num_scaffolds"] == 2
    assert parsed["number_of_gaps"] == 1

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    df = pd.DataFrame(
        {
            "sample": ["s1", "s2"],
            "frac_ologs_A": [0.2, 0.4],
            "frac_ologs_B": [0.8, 0.6],
        }
    )
    alg_sizes = {"A": 10, "B": 20}
    asd.plot_decay(axes[0], df, alg_sizes, "absolute", plot_bars=True)
    asd.plot_decay(axes[1], df, alg_sizes, "ranked", x_axis_labels_are_ALGs=True)
    asd.plot_decay(axes[2], df, alg_sizes, "boxplot")
    assert axes[0].get_title() == ""
    plt.close(fig)

    with pytest.raises(IOError, match="must not be empty"):
        asd.plot_decay(plt.subplots()[1], df, {}, "absolute")


def test_bin_and_plot_decay(monkeypatch, tmp_path: Path):
    algrbh = tmp_path / "alg.rbh"
    stats = tmp_path / "rbhstats.tsv"
    outpdf = tmp_path / "out.pdf"
    algrbh.write_text("x\n")
    pd.DataFrame(
        {
            "sample": ["s1", "s2"],
            "frac_ologs_sig": [0.9, 0.4],
            "frac_ologs_A": [0.2, 0.3],
            "frac_ologs_B": [0.8, 0.7],
        }
    ).to_csv(stats, sep="\t", index=False)
    monkeypatch.setattr(
        asd,
        "parse_rbh",
        lambda _path: pd.DataFrame({"gene_group": ["A", "A", "B"]}),
    )
    asd.bin_and_plot_decay(str(algrbh), str(stats), str(outpdf), "ALG", num_bins=2)
    assert outpdf.exists()
