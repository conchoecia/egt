from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from egt import plot_chrom_number_vs_changes as pcnc


def _toy_vectors():
    labels = ["a", "b", "c", "d"]
    chromnum = [10, 20, 30, 40]
    colocs = [1, 3, 5, 7]
    losses = [-2, -4, -6, -8]
    return labels, chromnum, colocs, losses


def test_panel_helpers_render_axes():
    labels, chromnum, colocs, losses = _toy_vectors()
    fig, axes = plt.subplots(3, 2, figsize=(10, 10))

    pcnc.panel_chromnum_vs_fusions_hexbin(axes[0, 0], labels, chromnum, colocs, losses)
    pcnc.panel_chromnum_vs_losses_hexbin(axes[0, 1], labels, chromnum, colocs, losses)
    pcnc.panel_coloc_vs_losses_hexbin(axes[1, 0], labels, chromnum, colocs, losses)
    pcnc.panel_fraction_fusions_and_loss_counts(axes[1, 1], labels, chromnum, colocs, losses, num_ALGs=29)
    pcnc.panel_chromsize_vs_changes_bins(axes[2, 0], labels, chromnum, colocs, losses)
    pcnc.panel_chromsize_vs_changes(axes[2, 1], labels, chromnum, colocs, losses)

    assert axes[0, 0].get_xlabel() == "Number of chromosomes"
    assert "Losses" in axes[0, 1].get_title()
    assert "Fusions vs Losses" in axes[1, 0].get_title()
    assert "Fusion fraction" in axes[1, 1].get_title()
    assert axes[2, 0].get_xlim()[1] == 100
    assert axes[2, 1].get_xlim()[1] == 100
    plt.close(fig)


def test_remaining_panel_helpers_render_axes():
    labels, chromnum, colocs, losses = _toy_vectors()
    fig, axes = plt.subplots(3, 2, figsize=(10, 10))

    pcnc.panel_chromosomes_vs_losses(axes[0, 0], labels, chromnum, colocs, losses)
    pcnc.panel_colocalization_vs_losses(axes[0, 1], labels, chromnum, colocs, losses)
    pcnc.panel_rank_coloc_losses(axes[1, 0], colocs, losses, "fusions", "losses")
    pcnc.panel_chromosomes_vs_fractionloss(axes[1, 1], labels, chromnum, colocs, losses, num_ALGs=29)
    pcnc.panel_fraction_fusions_losses_possible(axes[2, 0], labels, chromnum, colocs, losses, num_ALGs=29)

    assert axes[0, 0].get_ylabel() == "Number of losses"
    assert "colocalizations and losses" in axes[0, 1].get_title()
    assert "Rank of number of fusions" in axes[1, 0].get_title()
    assert axes[1, 1].get_ylabel() == "Fraction of possible fusions"
    assert "Fraction of possible fusions" in axes[2, 0].get_ylabel()
    plt.close(fig)


def test_plot_chrom_number_vs_changes_writes_output(tmp_path, monkeypatch):
    chromsizes = tmp_path / "chrom.tsv"
    chromsizes.write_text("sample\tchromosomes\nsp1\t10\nsp2\t20\nsp3\t30\n")

    changes = tmp_path / "changes.tsv"
    pd.DataFrame({"placeholder": [1, 2, 3]}).to_csv(changes, sep="\t", index=False)

    parsed = pd.DataFrame(
        {
            "samplename": ["sp1", "sp1", "sp2", "sp3"],
            "colocalizations": [list("ABCDE"), list("FGHIJ"), ["K"], ["L"]],
            "losses": [[], ["X"], ["Y", "Z"], []],
        }
    )
    monkeypatch.setattr(pcnc, "parse_gain_loss_from_perspchrom_df", lambda _df: parsed)
    monkeypatch.setattr(pcnc.odp_plot, "format_matplotlib", lambda: None)

    outfile = tmp_path / "out.pdf"
    pcnc.plot_chrom_number_vs_changes(str(changes), str(chromsizes), str(outfile))

    assert outfile.exists()


def test_main_dispatches_to_plot_function(monkeypatch):
    seen = {}

    def fake_plot(input1, input2, outfilename):
        seen["args"] = (input1, input2, outfilename)

    monkeypatch.setattr(pcnc, "plot_chrom_number_vs_changes", fake_plot)
    assert pcnc.main(["changes.tsv", "chrom.tsv"]) == 0
    assert seen["args"] == ("changes.tsv", "chrom.tsv", "chrom_number_vs_changes.pdf")
