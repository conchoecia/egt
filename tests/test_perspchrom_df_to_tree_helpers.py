from __future__ import annotations

import os

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import pytest

from egt import perspchrom_df_to_tree as pdt


class FakeNCBI:
    def get_taxid_translator(self, taxids):
        return {taxid: f"name_{taxid}" for taxid in taxids}


def test_rgb_float_to_hex():
    assert pdt.rgb_float_to_hex((1.0, 0.5, 0.0)) == "#ff7f00"


def test_parse_gain_loss_string_and_dataframe_wrapper():
    changes = "1-([('A','B')]|['L1']|['S1'])-2-([]|[]|[])-3"
    df = pdt.parse_gain_loss_string(changes, "sampleA")
    assert list(df["source_taxid"]) == [1, 2]
    assert df.loc[0, "colocalizations"] == [("A", "B")]
    assert df.loc[0, "losses"] == ["L1"]
    assert df.loc[0, "splits"] == ["S1"]

    wrapped = pdt.parse_gain_loss_from_perspchrom_df(
        pd.DataFrame({"changestrings": [changes], "species": ["sampleA"]})
    )
    assert len(wrapped) == 2


def test_safe_get_taxid_translator_adds_custom_negative_taxids():
    names = pdt.safe_get_taxid_translator(FakeNCBI(), [1, -67, -68])
    assert names[1] == "name_1"
    assert names[-67] == "Myriazoa"
    assert names[-68] == "Parahoxozoa"


def test_nodes_in_same_cc_and_colocalize_these_nodes():
    graph = nx.Graph()
    graph.add_nodes_from(["A", "B", "C"])
    assert pdt.nodes_in_same_CC(graph, ["A", "B"]) is False
    graph = pdt.colocalize_these_nodes(graph, ["A", "B", "C"])
    assert pdt.nodes_in_same_CC(graph, ["A", "B", "C"]) is True
    assert graph.has_edge("A", "C")


def test_stats_on_changedf(monkeypatch):
    monkeypatch.setattr(pdt, "NCBITaxa", lambda: FakeNCBI())
    sampledf = pd.DataFrame(
        {
            "taxidstring": ["1;2", "1;3"],
        }
    )
    changedf = pd.DataFrame(
        {
            "source_taxid": [1],
            "target_taxid": [2],
            "colocalizations": [[("A", "B")]],
            "losses": [["L1"]],
            "samplename": ["sampleA"],
            "sample_taxid": [999],
        }
    )

    stats = pdt.stats_on_changedf(sampledf, changedf)

    assert set(stats["change"]) == {("A", "B"), "L1"}
    assert set(stats["target_taxid_name"]) == {"name_2"}
    assert all(stats["counts"] == 1)


def test_size_helpers_and_delete_node_resize(monkeypatch):
    graph = nx.Graph()
    graph.add_node("A", size=2)
    graph.add_node("B", size=3)
    graph.add_edge("A", "B")

    assert pdt.node_size_fraction_of_total_size(graph, "A") == 2 / 5
    assert pdt.node_size_fraction_of_largest(graph, "A") == 2 / 3
    assert pdt.node_size_CC(graph, "A") == 5
    assert pdt.node_size_CC_fraction_of_total_size(graph, "A") == 1.0
    assert pdt.node_size_CC_fraction_of_largest(graph, "A") == 1.0

    monkeypatch.setattr(pdt.random, "choice", lambda seq: seq[0])
    resized = pdt.delete_node_resize(graph, "A")
    assert "A" not in resized.nodes
    assert resized.nodes["B"]["size"] == 5


def test_trace_cache_roundtrip_and_staleness(tmp_path):
    cache = {
        1: {
            ("ALG", "frac", "abs"): {
                "lines": {"bin1": {"num_sim": [10, 20], "value": [0.5, 0.7]}},
                "num_sims": 20,
            }
        }
    }
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    pdt.save_trace_cache_to_file(cache, str(cache_dir))
    loaded = pdt.load_trace_cache_from_file(str(cache_dir))
    assert loaded[1][("ALG", "frac", "abs")]["num_sims"] == 20
    assert loaded[1][("ALG", "frac", "abs")]["lines"]["bin1"]["value"] == [0.5, 0.7]

    sim = tmp_path / "sim.tsv"
    sim.write_text("x\n")
    os.utime(sim, (os.path.getmtime(sim) + 5, os.path.getmtime(sim) + 5))
    assert pdt.is_trace_cache_stale([str(sim)], str(cache_dir)) is True


def test_assert_single_size_abs_condition_and_df_to_obs_exp_dict():
    df = pd.DataFrame(
        {
            "ob_ex": ["observed", "expected"],
            "size_frac": ["frac", "frac"],
            "abs_CC": ["abs", "abs"],
            "bin": ["b1", "b1"],
            "counts": [3, 1],
        }
    )
    assert pdt.assert_single_size_abs_condition(df) is True
    ove_df = pdt.df_to_obs_exp_dict(df)
    assert "b1" in ove_df
    ove_dict = pdt.df_to_obs_exp_dict({"observed": {"b1": 3}, "expected": {"b1": 1}})
    assert ove_dict["b1"] > 0


def test_generate_trace_panel_and_phylo_tree_helpers(tmp_path, monkeypatch):
    fig, ax = plt.subplots()
    trace_cache = {"lines": {"bin1": {"num_sim": [10, 20], "value": [0.1, 0.2]}}, "num_sims": 20}
    pdt.generate_trace_panel(ax, [("1", "2")], [], "ALG", "frac", "abs", trace_cache=trace_cache)
    assert ax.get_xlim()[1] == 20
    plt.close(fig)

    monkeypatch.setattr(pdt, "NCBITaxa", lambda: FakeNCBI())
    tree = pdt.PhyloTree()
    assert tree.add_lineage_string("1;2;3") == 0
    df = pd.DataFrame({"taxidstring": ["1;2;3", "1;4"]})
    assert tree.build_tree_from_per_sp_chrom_df(df) == 0
    tree.add_taxname_to_all_nodes()
    assert tree.G.nodes[1]["taxname"] == "name_1"
    edges = tree.get_edges_in_clade(2)
    assert (1, 2) in edges and (2, 3) in edges

    outfile = tmp_path / "nodes.tsv"
    pdt.generate_node_taxid_file_from_per_sp_chrom_df(df, outfile)
    assert "1\tname_1" in outfile.read_text()
