from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from egt import plot_branch_stats_vs_time as pbst


def test_main_special_mode_single_clade(monkeypatch, tmp_path):
    args = SimpleNamespace(
        outdir=str(tmp_path / "out"),
        edge_information=str(tmp_path / "edge.tsv"),
        node_information=str(tmp_path / "node.tsv"),
        statsdf=str(tmp_path / "stats.tsv"),
        intensity_of_extinction=None,
        analyze_single_clade=2,
        exclude_subclades=[4],
        threads=1,
        suppress_plotting=True,
    )
    monkeypatch.setattr(pbst, "parse_args", lambda argv=None: args)

    edgedf = pd.DataFrame(
        {
            "parent_taxid": [1, 2, 2],
            "child_taxid": [2, 3, 4],
            "child_lineage": ["[1, 2]", "[1, 2, 3]", "[1, 2, 4]"],
            "parent_age": [10.0, 5.0, 5.0],
            "child_age": [5.0, 0.0, 0.0],
            "branch_length": [5.0, 5.0, 5.0],
            "dist_crown": [1.0, 1.0, 1.0],
            "dist_crown_plus_root": [2.0, 2.0, 2.0],
        }
    )
    nodedf = pd.DataFrame(
        {
            "taxid": [1, 2, 3, 4],
            "parent": [None, 1, 2, 2],
            "name": ["root", "Target", "LeafA", "LeafB"],
            "dist_crown": [1.0, 1.0, 1.0, 1.0],
            "dist_crown_plus_root": [2.0, 2.0, 2.0, 2.0],
        }
    )
    eventdf = pd.DataFrame({"target_taxid": [3], "change": ["ALG1"]})

    def fake_read_csv(path, sep="\t", index_col=False, **kwargs):
        if path == args.edge_information:
            return edgedf.copy()
        if path == args.node_information:
            return nodedf.copy()
        if path == args.statsdf:
            return eventdf.copy()
        raise AssertionError(path)

    monkeypatch.setattr(pbst.pd, "read_csv", fake_read_csv)
    monkeypatch.setattr(pbst, "add_events_to_edge_df", lambda edge_df, _eventdf: edge_df.assign(num_changes=1))

    class FakeWeighter:
        def __init__(self, edgedf, nodedf, cache_dir=None, verbose=True):
            self.edgedf = edgedf
            self.nodedf = nodedf
            self.cache_dir = cache_dir
            self.verbose = verbose

    monkeypatch.setattr(pbst, "PhyloWeighting", FakeWeighter)

    weighted = pd.DataFrame({"age": [-5, -1], "fusion_rate_at_this_age_mean": [1.0, 2.0]})
    unweighted = pd.DataFrame({"age": [-5, -1], "fusion_rate_at_this_age_mean": [0.5, 1.5]})
    monkeypatch.setattr(
        pbst,
        "get_edge_stats_single_taxid",
        lambda *args, **kwargs: (weighted.copy(), unweighted.copy(), (3.0, 4.0)),
    )

    called = {"compare": 0, "conservation": 0, "temporal": 0}
    monkeypatch.setattr(
        pbst,
        "plot_weighted_vs_unweighted_comparison",
        lambda *args, **kwargs: called.__setitem__("compare", called["compare"] + 1),
    )
    monkeypatch.setattr(
        pbst,
        "plot_event_count_conservation",
        lambda *args, **kwargs: called.__setitem__("conservation", called["conservation"] + 1),
    )
    monkeypatch.setattr(
        pbst,
        "plot_phylogenetic_weights_temporal",
        lambda *args, **kwargs: called.__setitem__("temporal", called["temporal"] + 1),
    )

    assert pbst.main([]) is None
    assert called == {"compare": 1, "conservation": 1, "temporal": 0}

    custom_dir = tmp_path / "out" / "custom_clade_analyses" / "Target_2_minus_LeafB_4"
    assert (custom_dir / "Target_minus_LeafB_changes_vs_age_weighted.tsv").exists()
    assert (custom_dir / "Target_minus_LeafB_changes_vs_age_unweighted.tsv").exists()


def test_main_normal_mode_generates_summary_and_per_clade_outputs(monkeypatch, tmp_path):
    intensity = tmp_path / "extinction.tsv"
    pd.DataFrame(
        {
            "Time (Ma)": [5, 10],
            "Extinction Intensity (%)": [1.0, 2.0],
            "Origination Intensity (%)": [3.0, 4.0],
        }
    ).to_csv(intensity, sep="\t", index=False)

    args = SimpleNamespace(
        outdir=str(tmp_path / "out"),
        edge_information=str(tmp_path / "edge.tsv"),
        node_information=str(tmp_path / "node.tsv"),
        statsdf=str(tmp_path / "stats.tsv"),
        intensity_of_extinction=str(intensity),
        analyze_single_clade=None,
        exclude_subclades=[],
        threads=1,
        suppress_plotting=True,
    )
    monkeypatch.setattr(pbst, "parse_args", lambda argv=None: args)

    edge_rows = [
        {
            "parent_taxid": 1,
            "child_taxid": 2,
            "child_lineage": "[1, 2]",
            "parent_age": 10.0,
            "child_age": 5.0,
            "branch_length": 5.0,
            "dist_crown": 10.0,
            "dist_crown_plus_root": 20.0,
        }
    ]
    edge_rows.extend(
        {
            "parent_taxid": 2,
            "child_taxid": 100 + i,
            "child_lineage": f"[1, 2, {100 + i}]",
            "parent_age": 5.0,
            "child_age": 0.0,
            "branch_length": 5.0,
            "dist_crown": 2.0,
            "dist_crown_plus_root": 7.0,
        }
        for i in range(51)
    )
    edgedf = pd.DataFrame(edge_rows)
    nodedf = pd.DataFrame(
        {
            "taxid": [1, 2],
            "parent": [None, 1],
            "name": ["root", "LargeClade"],
            "dist_crown": [10.0, 2.0],
            "dist_crown_plus_root": [20.0, 7.0],
        }
    )
    eventdf = pd.DataFrame({"target_taxid": [2], "change": ["ALG1"]})

    real_read_csv = pd.read_csv

    def fake_read_csv(path, sep="\t", index_col=False, **kwargs):
        if path == args.edge_information:
            return edgedf.copy()
        if path == args.node_information:
            return nodedf.copy()
        if path == args.statsdf:
            return eventdf.copy()
        if path == args.intensity_of_extinction:
            return real_read_csv(intensity, sep="\t")
        raise AssertionError(path)

    monkeypatch.setattr(pbst.pd, "read_csv", fake_read_csv)
    monkeypatch.setattr(pbst, "add_events_to_edge_df", lambda edge_df, _eventdf: edge_df.assign(num_changes=1))

    class FakeWeighter:
        def __init__(self, edgedf, nodedf, cache_dir=None, verbose=True):
            self.edgedf = edgedf
            self.nodedf = nodedf
            self.cache_dir = cache_dir
            self.verbose = verbose

    monkeypatch.setattr(pbst, "PhyloWeighting", FakeWeighter)
    monkeypatch.setattr(pbst, "calculate_clade_event_totals", lambda node, _edgedf: (float(node), float(node) + 0.5))

    def fake_get_edge_stats(node, edgedf, **kwargs):
        weighted = pd.DataFrame({"age": [-5, -1], "fusion_rate_at_this_age_mean": [1.0, 2.0]})
        unweighted = pd.DataFrame({"age": [-5, -1], "fusion_rate_at_this_age_mean": [0.5, 1.5]})
        if kwargs.get("clade_tips_for_renorm") is not None:
            weighted["phylo_weight"] = [1.0, 1.0]
            unweighted["phylo_weight"] = [1.0, 1.0]
        return weighted, unweighted, (3.0, 4.0)

    monkeypatch.setattr(pbst, "get_edge_stats_single_taxid", fake_get_edge_stats)

    called = {"compare": 0, "conservation": 0, "clade_dist": 0, "temporal": 0, "intensity": 0}
    monkeypatch.setattr(
        pbst,
        "plot_weighted_vs_unweighted_comparison",
        lambda *args, **kwargs: called.__setitem__("compare", called["compare"] + 1),
    )
    monkeypatch.setattr(
        pbst,
        "plot_event_count_conservation",
        lambda *args, **kwargs: called.__setitem__("conservation", called["conservation"] + 1),
    )
    monkeypatch.setattr(
        pbst,
        "plot_clade_weight_distribution_from_cache",
        lambda *args, **kwargs: called.__setitem__("clade_dist", called["clade_dist"] + 1),
    )
    monkeypatch.setattr(
        pbst,
        "plot_phylogenetic_weights_temporal",
        lambda *args, **kwargs: called.__setitem__("temporal", called["temporal"] + 1),
    )
    monkeypatch.setattr(
        pbst,
        "plot_intensity_of_extinction",
        lambda *args, **kwargs: called.__setitem__("intensity", called["intensity"] + 1) or {
            "extinction_fusion_numpbranch_spearman_r": 0.5
        },
    )

    assert pbst.main([]) == 0
    summary = real_read_csv(tmp_path / "out" / "modified_node_list.tsv", sep="\t")
    assert "fusions_in_this_clade_unweighted" in summary.columns
    assert (tmp_path / "out" / "per_clade_analyses" / "LargeClade_2_changes_vs_age_weighted.tsv").exists()
    assert (tmp_path / "out" / "per_clade_analyses" / "LargeClade_2_changes_vs_age_unweighted.tsv").exists()
    assert called == {"compare": 3, "conservation": 3, "clade_dist": 1, "temporal": 0, "intensity": 1}
