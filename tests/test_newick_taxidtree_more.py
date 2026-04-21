from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt

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
