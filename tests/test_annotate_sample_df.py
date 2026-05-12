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


def test_data_availability_validation_and_multiple_local_accessions(tmp_path: Path):
    with pytest.raises(ValueError, match="must end in .html"):
        asd.gen_data_availability_statement("sample.tsv", str(tmp_path / "bad.txt"))

    sampledf = tmp_path / "locals.tsv"
    pd.DataFrame(
        {
            "sample": [
                "Gamma-333-LOCALA",
                "Gamma-333-LOCALB",
                "Delta-444-GCF123456.1",
            ],
            "taxname": ["Gamma species", "Gamma species", "Delta species"],
            "taxid": [333, 333, 444],
        }
    ).to_csv(sampledf, sep="\t")

    html = tmp_path / "locals.html"
    asd.gen_data_availability_statement(str(sampledf), str(html))
    text = html.read_text()
    assert "LOC_ALA" in text
    assert "LOC_ALB" in text
    assert "GCF_123456.1" in text


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


def test_stats_filepath_and_plot_decay_validation(tmp_path: Path):
    stats = tmp_path / "stats.txt"
    stats.write_text("count: 5\ntruth: False\nname: value\n")
    parsed = asd.stats_filepath_to_dict(stats)
    assert parsed == {"count": 5, "truth": False, "name": "value"}

    fig, ax = plt.subplots()
    df = pd.DataFrame({"frac_ologs_A": [0.1], "sample": ["s1"]})
    with pytest.raises(ValueError, match="must be either 'absolute' or 'ranked', or 'boxplot'"):
        asd.plot_decay(ax, df, {"A": 10}, "badmode")
    with pytest.raises(ValueError, match="missing from the df_dict"):
        asd.plot_decay(ax, pd.DataFrame({"sample": ["s1"]}), {"A": 10}, "absolute")
    plt.close(fig)


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


def test_bin_and_plot_decay_validation_and_taxid_boolean(tmp_path: Path):
    missing = tmp_path / "missing.tsv"
    with pytest.raises(ValueError, match="does not exist"):
        asd.bin_and_plot_decay(str(missing), str(missing), str(tmp_path / "out.pdf"), "ALG")

    algrbh = tmp_path / "alg.rbh"
    stats = tmp_path / "stats.tsv"
    algrbh.write_text("x\n")
    stats.write_text("sample\tfrac_ologs_sig\n")

    with pytest.raises(ValueError, match="ALGname must be a string"):
        asd.bin_and_plot_decay(str(algrbh), str(stats), str(tmp_path / "out.pdf"), 5)
    with pytest.raises(ValueError, match="num_bins must be an int"):
        asd.bin_and_plot_decay(str(algrbh), str(stats), str(tmp_path / "out.pdf"), "ALG", num_bins=1.5)

    assert asd.taxid_list_include_exclude_boolean([1, 2, 3], [2], [9]) is True
    assert asd.taxid_list_include_exclude_boolean("[1, 2, 3]", [2], [3]) is False


def test_plot_umap_highlight_subclade_and_main(tmp_path: Path):
    df = pd.DataFrame(
        {
            "UMAP1": [0.0, 1.0],
            "UMAP2": [1.0, 0.0],
            "color": ["#111111", "#222222"],
            "taxid_list": ["[1, 2, 3]", "[1, 4, 5]"],
        }
    )
    infile = tmp_path / "umap.tsv"
    df.to_csv(infile, sep="\t", index=False)
    outpdf = tmp_path / "highlight.pdf"
    asd.plot_UMAP_highlight_subclade(str(infile), "Title", [2], [9], str(outpdf))
    assert outpdf.exists()

    with pytest.raises(ValueError, match="must be a string"):
        asd.plot_UMAP_highlight_subclade(123, "Title", [2], [9], str(outpdf))
    with pytest.raises(ValueError, match="taxids_to_include must be a list"):
        asd.plot_UMAP_highlight_subclade(str(infile), "Title", "2", [9], str(outpdf))
    with pytest.raises(ValueError, match="title must be a string"):
        asd.plot_UMAP_highlight_subclade(str(infile), 5, [2], [9], str(outpdf))
    with pytest.raises(ValueError, match="taxids_to_exclude must be a list"):
        asd.plot_UMAP_highlight_subclade(str(infile), "Title", [2], "9", str(outpdf))
    with pytest.raises(ValueError, match="pdfout must be a string"):
        asd.plot_UMAP_highlight_subclade(str(infile), "Title", [2], [9], 5)
    with pytest.raises(ValueError, match="must be integers"):
        asd.plot_UMAP_highlight_subclade(str(infile), "Title", ["2"], [9], str(outpdf))
    with pytest.raises(ValueError, match="does not exist"):
        asd.plot_UMAP_highlight_subclade(str(tmp_path / "missing.tsv"), "Title", [2], [9], str(outpdf))

    assert asd.main([]) == 0
