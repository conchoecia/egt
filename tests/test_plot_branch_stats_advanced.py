from __future__ import annotations

from pathlib import Path

import pandas as pd

from egt import plot_branch_stats_advanced as pbsa


def _node_df():
    return pd.DataFrame(
        {
            "taxid": [1, 2, 3],
            "dist_crown": [10.0, 20.0, 30.0],
            "chromsize_median": [5.0, 6.0, 7.0],
            "chromsize_mean": [5.5, 6.5, 7.5],
            "nodeage": [100.0, 200.0, 300.0],
            "fusions_in_this_clade": [4.0, 7.0, 15.0],
            "losses_in_this_clade": [1.0, 5.0, 8.0],
            "extinction_fusion_spearman_r": [0.1, 0.2, 0.3],
            "extinction_losses_spearman_r": [0.2, 0.1, 0.0],
            "origination_fusion_spearman_r": [0.3, 0.2, 0.1],
            "origination_losses_spearman_r": [0.0, 0.1, 0.2],
        }
    )


def test_parse_args_and_plot_functions(tmp_path: Path):
    edge = tmp_path / "edge.tsv"
    node = tmp_path / "node.tsv"
    pd.DataFrame({"x": [1]}).to_csv(edge, sep="\t", index=False)
    _node_df().to_csv(node, sep="\t", index=False)

    args = pbsa.parse_args(["-e", str(edge), "-n", str(node)])
    assert args.edgefile == str(edge)

    out1 = tmp_path / "corr.pdf"
    out2 = tmp_path / "rate.pdf"
    out3 = tmp_path / "allvars.pdf"
    pbsa.plot_median_chrom_vs_correlations(_node_df(), str(out1))
    pbsa.plot_median_chrom_vs_clade_evo_rate(_node_df(), str(out2))
    pbsa.plot_all_variables_node(_node_df(), str(out3))
    assert out1.exists()
    assert out2.exists()
    assert out3.exists()
