from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from egt import plot_branch_stats_vs_time as pbst


def _tiny_edge_df():
    return pd.DataFrame(
        {
            "parent_taxid": [1, 1, 2, 2],
            "child_taxid": [2, 3, 4, 5],
            "branch_length": [4.0, 3.0, 2.0, 2.0],
            "parent_age": [10.0, 10.0, 6.0, 6.0],
            "child_age": [6.0, 7.0, 4.0, 4.0],
            "child_lineage_list": [[1, 2], [1, 3], [1, 2, 4], [1, 2, 5]],
            "num_fusions_per_my_this_branch": [1.0, np.nan, 0.5, np.inf],
            "num_dispersals_per_my_this_branch": [2.0, 1.0, np.nan, 0.25],
        }
    )


def _tiny_node_df():
    return pd.DataFrame({"taxid": [1, 2, 3, 4, 5]})


def test_weighted_quantile_and_event_totals():
    assert pbst.weighted_quantile([], [], 0.5) != pbst.weighted_quantile([], [], 0.5)
    q = pbst.weighted_quantile([1, 2, 3], [1, 1, 2], 0.5)
    assert q == pytest.approx(2.0)

    fusions, dispersals = pbst.calculate_clade_event_totals(2, _tiny_edge_df())
    assert fusions == pytest.approx(5.0)
    assert dispersals == pytest.approx(8.5)


def test_renormalize_weights_for_clade():
    class FakeWeighter:
        def _load_cached_weights(self, time_slice=None):
            assert time_slice == 5
            return {4: 2.0, 5: 4.0, 99: 10.0}

    renorm = pbst.renormalize_weights_for_clade(FakeWeighter(), {4, 5}, 5)
    assert renorm == {4: pytest.approx(2 / 3), 5: pytest.approx(4 / 3)}
    assert pbst.renormalize_weights_for_clade(FakeWeighter(), {77}, 5) == {}

    class ZeroWeighter:
        def _load_cached_weights(self, time_slice=None):
            return {4: 0.0, 5: 0.0}

    assert pbst.renormalize_weights_for_clade(ZeroWeighter(), {4, 5}, 5) == {4: 1.0, 5: 1.0}


def test_phylo_weighting_helpers_and_caching(tmp_path: Path):
    edgedf = _tiny_edge_df()
    weighter = pbst.PhyloWeighting(edgedf, _tiny_node_df(), cache_dir=str(tmp_path), verbose=False)

    assert weighter.root == 1
    assert weighter._path_to_root(4, {4: (2, 2.0), 2: (1, 4.0)}) == [(4, (2, 2.0))]
    assert weighter._get_descendants(2) == {4, 5}
    assert set(weighter._get_branches_at_time(5.0)) == {4, 5}

    distances = weighter.compute_tip_distances()
    assert distances[(4, 4)] == 0
    assert (4, 5) in distances

    weights = weighter.compute_branch_weights(edgedf)
    assert len(weights) == len(edgedf)
    assert np.mean(weights) == pytest.approx(1.0)

    pruned_dag, tips_at_time, edgedf_subset = weighter.create_pruned_tree_at_time(5.0)
    assert tips_at_time == {4, 5}
    assert set(edgedf_subset["child_taxid"]) == {2, 3, 4, 5}

    time_weights = weighter.compute_branch_weights(edgedf_subset[edgedf_subset["child_taxid"].isin(tips_at_time)], time_slice=5.0)
    assert set(time_weights) == {4, 5}

    worker_time, worker_weights = pbst._compute_time_slice_weights_worker(
        (5.0, edgedf.copy(), (_tiny_node_df().copy(), str(tmp_path)))
    )
    assert worker_time == 5.0
    assert set(worker_weights) == {4, 5}


def test_phylo_weighting_zero_mean_falls_back_to_uniform(tmp_path: Path):
    edgedf = pd.DataFrame(
        {
            "parent_taxid": [1, 1],
            "child_taxid": [2, 3],
            "branch_length": [0.0, 0.0],
            "parent_age": [10.0, 10.0],
            "child_age": [10.0, 10.0],
        }
    )
    nodedf = pd.DataFrame({"taxid": [1, 2, 3]})
    weighter = pbst.PhyloWeighting(edgedf, nodedf, cache_dir=str(tmp_path), verbose=False)
    weights = weighter.compute_branch_weights(edgedf)
    assert np.allclose(weights, np.array([1.0, 1.0]))


def test_parse_args_parses_optional_lists(tmp_path: Path):
    edge = tmp_path / "edge.tsv"
    node = tmp_path / "node.tsv"
    stats = tmp_path / "stats.tsv"
    extinct = tmp_path / "extinct.tsv"
    for path in [edge, node, stats, extinct]:
        path.write_text("x\n")

    args = pbst.parse_args(
        [
            "-e", str(edge),
            "-n", str(node),
            "-s", str(stats),
            "-i", str(extinct),
            "-o", "1,2",
            "--analyze_single_clade", "33208",
            "--exclude_subclades", "3,4",
        ]
    )
    assert args.omit_taxids == [1, 2]
    assert args.exclude_subclades == [3, 4]

    with pytest.raises(ValueError, match="requires --analyze_single_clade"):
        pbst.parse_args(
            [
                "-e", str(edge),
                "-n", str(node),
                "-s", str(stats),
                "--exclude_subclades", "3,4",
            ]
        )


def test_stats_helpers_and_add_events_to_edge_df():
    df = pd.DataFrame(
        {
            "x": [1.0, 2.0, np.nan, np.inf, 4.0],
            "y": [2.0, 4.0, 6.0, 8.0, 16.0],
        }
    )
    sub = pbst._subdf_no_missing(df, "x", "y")
    assert len(sub) == 3
    assert pbst._full_sk_stats(df, "x", "y", "spearman")[2] == 3
    assert pbst._full_sk_stats(df, "x", "y", "kendall")[2] == 3
    assert pbst._full_sk_stats(df, "x", "y", "pearson")[2] == 3

    edgedf = pd.DataFrame(
        {
            "parent_taxid": [1, 2],
            "child_taxid": [2, 3],
            "branch_length": [4.0, 2.0],
            "num_losses_this_branch": [9, 9],
        }
    )
    eventdf = pd.DataFrame(
        {
            "target_taxid": [2, 2, 3, 99],
            "change": ["ALG1", "('ALG1', 'ALG2')", "ALG2", "ALG9"],
        }
    )
    out = pbst.add_events_to_edge_df(edgedf, eventdf)
    assert out.loc[out["child_taxid"] == 2, "num_dispersals_this_branch"].iloc[0] == 1
    assert out.loc[out["child_taxid"] == 2, "num_ALG1+ALG2_this_branch"].iloc[0] == 1
    assert out.loc[out["child_taxid"] == 3, "num_ALG2_this_branch"].iloc[0] == 1


def test_plot_helpers_write_outputs(tmp_path: Path):
    edgedf = pd.DataFrame(
        {
            "phylo_weight": [0.8, 1.0, 1.2],
            "branch_length": [2.0, 4.0, 6.0],
            "parent_age": [100.0, 200.0, 300.0],
            "child_age": [90.0, 180.0, 280.0],
        }
    )
    pbst.plot_phylogenetic_weight_diagnostics(edgedf, str(tmp_path), "ALL")
    assert (tmp_path / "ALL_phylogenetic_weights_diagnostic.pdf").exists()

    time_slice_weights = {0: {1: 1.0, 2: 2.0}, 100: {1: 0.5, 2: 1.5}}
    pbst.plot_phylogenetic_weight_temporal_heatmap(time_slice_weights, str(tmp_path), 100, output_prefix="ALL")
    assert (tmp_path / "ALL_temporal_heatmap.pdf").exists()

    temporal = pd.DataFrame(
        {
            "parent_age": [50.0, 120.0, 120.0],
            "phylo_weight": [0.9, 1.1, 1.3],
        }
    )
    pbst.plot_phylogenetic_weights_temporal(temporal, str(tmp_path), "ALL")
    assert (tmp_path / "ALL_phylogenetic_weights_temporal.pdf").exists()
