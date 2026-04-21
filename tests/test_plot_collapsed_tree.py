from __future__ import annotations

from types import SimpleNamespace

import matplotlib.pyplot as plt

from egt import plot_collapsed_tree as pct


class FakeTree:
    def __init__(self):
        self.nodes = {
            1: SimpleNamespace(taxid=1, lineage=[1], y=0, isleaf=False),
            10: SimpleNamespace(taxid=10, lineage=[1, 10], y=1, isleaf=False),
            20: SimpleNamespace(taxid=20, lineage=[1, 10, 20], y=2, isleaf=True),
        }
        self.edges = {
            (1, 10): SimpleNamespace(
                parent_taxid=1,
                child_taxid=10,
                num_fusions_this_branch=2,
                num_losses_this_branch=3,
                parent_age=100,
                child_age=80,
            ),
            (10, 20): SimpleNamespace(
                parent_taxid=10,
                child_taxid=20,
                num_fusions_this_branch=1,
                num_losses_this_branch=0,
                parent_age=80,
                child_age=0,
            ),
        }

    def plot_tree(self, ax, **_kwargs):
        return ax


def test_parse_args_and_helper_functions(tmp_path):
    node = tmp_path / "node.tsv"
    edge = tmp_path / "edge.tsv"
    node.write_text("x\n")
    edge.write_text("x\n")
    args = pct.parse_args(["-n", str(node), "-e", str(edge)])
    assert args.output == "collapsed_tree.pdf"

    tree = FakeTree()
    assert pct.get_clade_mrca(tree, 10).taxid == 10
    stats = pct.aggregate_clade_statistics(tree, 10)
    assert stats["fusions"] == 3
    assert stats["dispersals"] == 3

    stem = pct.get_stem_branch_statistics(tree, 10)
    assert stem["fusions"] == 2
    assert stem["dispersals"] == 3


def test_plot_collapsed_tree_custom_renders_labels():
    tree = FakeTree()
    fig, ax = plt.subplots()
    pct.plot_collapsed_tree_custom(
        tree,
        {10: {"name": "Chordata", "color": "#112233"}},
        ax,
        show_within_clade_stats=True,
    )
    assert len(ax.texts) >= 1
    plt.close(fig)


def test_main_runs_with_fake_tree(monkeypatch, tmp_path):
    node = tmp_path / "node.tsv"
    edge = tmp_path / "edge.tsv"
    pd = __import__("pandas")
    pd.DataFrame({"taxid": [1, 10], "name": ["root", "tip"], "in_this_clade": [1, 2]}).to_csv(node, sep="\t", index=False)
    pd.DataFrame(
        {
            "parent_taxid": [1],
            "child_taxid": [10],
            "parent_age": [100],
            "child_age": [80],
            "branch_length": [20],
            "num_fusions_this_branch": [2],
            "num_losses_this_branch": [3],
            "extra": [1],
        }
    ).to_csv(edge, sep="\t", index=False)

    class FakeMainTree(FakeTree):
        def ingest_node_edge(self, nodedf, edgedf):
            self.nodedf = nodedf
            self.edgedf = edgedf

    monkeypatch.setattr(pct, "TaxIDtree", FakeMainTree)
    monkeypatch.chdir(tmp_path)
    rc = pct.main(["-n", str(node), "-e", str(edge), "-o", "collapsed.pdf", "--clades", "10"])
    assert rc == 0
    assert (tmp_path / "collapsed.pdf").exists()
