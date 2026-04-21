from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from egt import algs_split_across_scaffolds as asas


def test_parse_args_with_existing_directory(tmp_path):
    args = asas.parse_args(["-d", str(tmp_path), "-a", "ALG"])
    assert args.rbh_directory == str(tmp_path)
    assert args.minsig == 0.005


def test_plot_helpers_produce_expected_axes_labels():
    splitsdf = pd.DataFrame(
        {
            "sample": ["sp1", "sp1", "sp1", "sp2", "sp2"],
            "gene_group": ["A", "A", "B", "A", "C"],
            "scaffold": ["chr1", "chr2", "chr1", "chr5", "chr5"],
        }
    )
    inferred = {"sp1": 10, "sp2": 6}

    fig, axes = plt.subplots(1, 2)
    asas.plot_chrom_number_vs_number_ALGs_split(axes[0], splitsdf, 2, inferred)
    asas.plot_chrom_number_vs_number_ALGs_perchrom(axes[1], splitsdf, inferred)

    assert axes[0].get_xlabel() == "Number of chromosomes"
    assert "split across 2 or more scaffolds" in axes[0].get_ylabel()
    assert axes[1].get_ylabel() == "Mean number of ALGs on each chromosome"
    plt.close(fig)
