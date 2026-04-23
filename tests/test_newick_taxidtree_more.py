from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from egt import newick_to_common_ancestors as n2ca


class FakeNCBI:
    def get_taxid_translator(self, taxids):
        mapping = {
            1: "root",
            2: "Chordata",
            3: "Arthropoda",
            4: "Homo sapiens",
            5: "Drosophila melanogaster",
        }
        return {taxid: mapping.get(taxid, f"Taxon{taxid}") for taxid in taxids}

    def get_name_translator(self, names):
        mapping = {
            "Homo sapiens": [4],
            "Drosophila melanogaster": [5],
        }
        return {name: mapping[name] for name in names if name in mapping}


def _build_tree():
    tree = n2ca.TaxIDtree()
    tree.NCBI = FakeNCBI()
    tree.add_edge(1, 2)
    tree.add_edge(1, 3)
    tree.add_edge(2, 4)
    tree.add_edge(3, 5)
    tree.root = 1
    tree.nodes[1].name = "root"
    tree.nodes[2].name = "Chordata"
    tree.nodes[3].name = "Arthropoda"
    tree.nodes[4].name = "Homo_sapiens"
    tree.nodes[5].name = "Drosophila_melanogaster"
    tree.nodes[1].nodeage = 10.0
    tree.nodes[2].nodeage = 6.0
    tree.nodes[3].nodeage = 7.0
    tree.nodes[4].nodeage = 0.0
    tree.nodes[5].nodeage = 0.0
    for node in [1, 2, 3, 4, 5]:
        tree.nodes[node].nodeages = Counter({tree.nodes[node].nodeage: 1})
    tree.add_lineage_info()
    tree.calc_dist_crown()
    for edge, vals in {
        (1, 2): (0.2, 0.4),
        (1, 3): (0.0, 0.1),
        (2, 4): (0.3, 0.0),
        (3, 5): (0.5, 0.2),
    }.items():
        tree.edges[edge].num_losses_per_my_this_branch = vals[0]
        tree.edges[edge].num_fusions_per_my_this_branch = vals[1]
    return tree


def test_sort_calc_and_export_helpers(tmp_path: Path):
    tree = _build_tree()

    tree.sort_nodes("lineage")
    assert tree.leaf_order == [4, 5]
    tree.sort_nodes("ascending")
    assert set(tree.leaf_order) == {4, 5}

    assert tree.edges[(1, 2)].branch_length == 4.0
    assert tree.nodes[1].dist_crown == 20.0
    assert tree.nodes[4].lineage_string == "1;2;4"

    out_newick = tmp_path / "tree.nwk"
    tree.write_newick(out_newick)
    assert "[4]" in out_newick.read_text()

    edge_file = tmp_path / "edges.tsv"
    node_file = tmp_path / "nodes.tsv"
    tree.print_edge_information(edge_file)
    tree.print_node_information(node_file)
    assert "parent_taxid" in edge_file.read_text()
    assert "taxid" in node_file.read_text()


def test_plot_and_report_helpers(tmp_path: Path):
    tree = _build_tree()

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    tree.plot_tree(
        axes[0],
        sort="ascending",
        variable=("num_losses_per_my_this_branch", "num_fusions_per_my_this_branch"),
        split_side_mode=True,
        split_draw_zero_branches=True,
        text_older_than=5,
    )
    n2ca.TaxIDtree.create_2d_colorbar_legend(
        axes[1],
        var1_range=(0.01, 1.0),
        var2_range=(0.01, 1.0),
    )
    assert axes[1].get_title().startswith("2D Color Scale")
    plt.close(fig)

    report = tmp_path / "report.txt"
    tree.generate_tree_report(report)
    text = report.read_text()
    assert "TREE STRUCTURE SUMMARY" in text
    assert "ZERO-LENGTH BRANCH ANALYSIS" in text


def test_zero_length_analysis_and_fixing():
    tree = _build_tree()
    tree.nodes[2].nodeage = 10.0
    tree.nodes[2].nodeages = Counter({10.0: 1})
    before = tree.analyze_zero_length_branches(tolerance=0.01)
    assert before["total_zero_length"] >= 1

    tree.fix_zero_length_branches(tolerance=0.01)
    after = tree.analyze_zero_length_branches(tolerance=0.01)
    assert after["total_zero_length"] <= before["total_zero_length"]


def test_error_paths_and_root_detection(monkeypatch):
    tree = _build_tree()
    with pytest.raises(ValueError, match="not a child"):
        tree.get_lineage_length([1, 3, 4])

    tree.nodes[4].parent = 2
    tree.nodes[4].nodeages = Counter({1.0: 1})
    with pytest.raises(ValueError, match="leaf"):
        tree.ensure_all_leaves_have_age_zero()

    empty_tree = n2ca.TaxIDtree()
    empty_tree.NCBI = FakeNCBI()
    with pytest.raises(ValueError, match="There is no root"):
        empty_tree.find_root()

    multi_root = n2ca.TaxIDtree()
    multi_root.NCBI = FakeNCBI()
    multi_root.add_node(1)
    multi_root.add_node(2)
    with pytest.raises(ValueError, match="more than one root"):
        multi_root.find_root()

    with pytest.raises(ValueError, match="different"):
        tree.add_edge(3, 4)

    with pytest.raises(ValueError, match="not recognized"):
        tree.sort_nodes("sideways")


def test_ingest_node_edge_from_files_and_plot_single_variable(monkeypatch, tmp_path: Path):
    tree = n2ca.TaxIDtree()
    tree.NCBI = FakeNCBI()

    node_df = pd.DataFrame(
        {
            "taxid": [1, 2, 3],
            "name": ["root", np.nan, "leaf"],
            "parent": [None, 1, 2],
            "children": ["{2}", "{3}", "set()"],
            "nodeages": ["Counter({10.0: 1})", "Counter({5.0: 1})", "Counter({0.0: 1})"],
            "lineage": ["[1]", "[1, 2]", "[1, 2, 3]"],
            "chromsize_list": ["[]", "[]", "[7]"],
            "num_genomes": [-1, -1, -1],
        }
    )
    edge_df = pd.DataFrame(
        {
            "parent_taxid": [1, 2],
            "child_taxid": [2, 3],
            "num_dispersals_this_branch": [2.0, 0.0],
            "num_dispersals_per_my_this_branch": [0.5, float("inf")],
            "num_fusions_per_my_this_branch": [0.3, np.nan],
            "num_A+B_this_branch": [4.0, 0.0],
        }
    )

    node_path = tmp_path / "nodes.tsv"
    edge_path = tmp_path / "edges.tsv"
    node_df.to_csv(node_path, sep="\t", index=False)
    edge_df.to_csv(edge_path, sep="\t", index=False)

    tree.ingest_node_edge(str(node_path), str(edge_path))
    tree.nodes[1].parent = None
    tree.root = tree.find_root()
    assert tree.edges[(1, 2)].num_losses_this_branch == 2.0
    assert tree.edges[(1, 2)].num_losses_per_my_this_branch == 0.5
    assert tree.edges[(1, 2)].fusions["num_A+B_this_branch"] == 4.0
    for node in tree.nodes.values():
        if node.nodeages:
            node.nodeage = node.nodeages.most_common(1)[0][0]
    tree.add_lineage_info()
    tree.calc_dist_crown()

    assert tree.nodes[3].num_genomes == 1
    tree.edges[(1, 2)].num_losses_per_my_this_branch = 0.5
    tree.edges[(2, 3)].num_losses_per_my_this_branch = 0.1

    monkeypatch.setattr(n2ca.random, "shuffle", lambda items: items.reverse())
    fig, ax = plt.subplots()
    tree.plot_tree(
        ax,
        sort="descending",
        variable="num_losses_per_my_this_branch",
        randomize_order=True,
        text_older_than=4,
    )
    plt.close(fig)


def test_build_from_newick_and_missing_taxonomy(monkeypatch):
    tree = n2ca.TaxIDtree()
    tree.NCBI = FakeNCBI()

    class MissingNCBI(FakeNCBI):
        def get_name_translator(self, names):
            if "Unknown species" in names:
                raise KeyError("Unknown species")
            return super().get_name_translator(names)

    newick = n2ca.PhyloTree(
        "((Homo_sapiens[4],Unknown_species)Myriazoa[-67],Drosophila_melanogaster);",
        parser=1,
    )
    root_id = tree.build_from_newick_tree(newick, MissingNCBI())

    assert root_id < 0
    assert 4 in tree.nodes
    assert 5 in tree.nodes
    assert len([node_id for node_id in tree.nodes if node_id < 0]) >= 2


def test_correct_missing_nodes_and_percolation(monkeypatch):
    tree = n2ca.TaxIDtree()
    tree.NCBI = FakeNCBI()
    for parent, child in [(1, 2), (2, 4), (1, 3)]:
        tree.add_edge(parent, child)
    tree.root = 1
    for node in tree.nodes:
        tree.nodes[node].name = f"N{node}"
    tree.nodes[1].nodeages = Counter({10.0: 1})
    tree.nodes[1].nodeage = 10.0
    tree.nodes[3].nodeages = Counter({0.0: 1})
    tree.nodes[3].nodeage = 0.0
    tree.nodes[4].nodeages = Counter({0.0: 1})
    tree.nodes[4].nodeage = 0.0

    tree.correct_missing_nodes(priority_node_ages={2: (6.0, "internal calibration")})
    assert tree.nodes[2].lock_age is True
    assert tree.nodes[2].nodeage == 6.0
    assert tree.check_nodeages_descending(enforce_nomissing=True) is True
    assert tree.get_lineage_length([1, 2, 4]) == 10.0


def test_build_from_newick_tree_uniquifies_duplicate_internal_taxids(monkeypatch):
    tree = n2ca.TaxIDtree()
    tree.NCBI = FakeNCBI()
    newick = n2ca.PhyloTree("(Homo_sapiens[4])internal[4];", parser=1)

    root_id = tree.build_from_newick_tree(newick, FakeNCBI())

    assert root_id < 0
    assert 4 in tree.nodes
    assert root_id in tree.nodes
    assert tree.nodes[4].parent is not None
    assert tree.nodes[4].parent in tree.nodes
    assert tree.nodes[4].parent != 4


def test_percolation_interpolation_and_calc_edges(monkeypatch):
    tree = n2ca.TaxIDtree()
    tree.NCBI = FakeNCBI()
    for parent, child in [(1, 2), (1, 3), (2, 4)]:
        tree.add_edge(parent, child)
    tree.root = 1

    tree.nodes[1].nodeages = Counter({10.0: 1})
    tree.nodes[1].nodeage = 10.0
    tree.nodes[2].nodeages = Counter({3.0: 1, 8.0: 1})
    tree.nodes[2].nodeage = 3.0
    tree.nodes[3].nodeages = Counter({0.0: 1})
    tree.nodes[3].nodeage = 0.0
    tree.nodes[4].nodeages = Counter({4.0: 1})
    tree.nodes[4].nodeage = 4.0

    tree.percolate_ascending_tip_to_root()
    assert tree.nodes[2].nodeage == 8.0

    tree.nodes[2].nodeage = 11.0
    tree.percolate_descending_root_to_tip()
    assert tree.nodes[2].nodeage == 8.0
    assert tree.check_nodeages_descending() is True

    tree.nodes[2].nodeage = 1.0
    tree.nodes[2].nodeages = Counter({1.0: 1})
    tree.nodes[2].nodeageinterpolated = True
    tree.nodes[4].nodeage = 4.0
    tree.nodes[4].nodeages = Counter({4.0: 1})
    assert tree.fix_broken_interpolated_entries() == 1
    assert tree.nodes[2].nodeage == 5.0

    tree.nodes[2].nodeages = Counter()
    tree.nodes[2].nodeage = None
    tree.interpolate_nodes()
    assert tree.nodes[2].nodeageinterpolated is True
    assert tree.nodes[2].nodeage > tree.nodes[4].nodeage

    tree.calc_edges()
    assert tree.edges[(1, 2)].branch_length > 0
