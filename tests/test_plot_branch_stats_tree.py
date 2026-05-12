from __future__ import annotations

from pathlib import Path

import pandas as pd

from egt import plot_branch_stats_tree as pbstree
from egt import plot_alg_fusions as paf


class FakeTaxIDtree:
    def __init__(self):
        self.leaf_order = [10, 20]
        self.nodes = {
            10: type("N", (), {"lineage": [1, 10]})(),
            20: type("N", (), {"lineage": [1, 20]})(),
        }

    def ingest_node_edge(self, nodedf, edgedf):
        self.nodedf = nodedf
        self.edgedf = edgedf

    def plot_tree(self, ax, **_kwargs):
        return ax

    @staticmethod
    def create_2d_colorbar_legend(ax, **_kwargs):
        return ax


def test_parse_args_and_main(monkeypatch, tmp_path: Path):
    node = tmp_path / "node.tsv"
    edge = tmp_path / "edge.tsv"
    pf = tmp_path / "presence.tsv"
    pd.DataFrame(
        {
            "taxid": [1, 10],
            "name": ["root", "tip"],
            "in_this_clade": [1, 2],
        }
    ).to_csv(node, sep="\t", index=False)
    pd.DataFrame(
        {
            "parent_taxid": [1],
            "child_taxid": [10],
            "parent_age": [100],
            "child_age": [0],
            "branch_length": [100],
            "num_fusions_per_my_this_branch": [0.1],
            "num_dispersals_per_my_this_branch": [0.2],
            "extra_col": [9],
        }
    ).to_csv(edge, sep="\t", index=False)
    pd.DataFrame({"species": ["s1"]}).to_csv(pf, sep="\t", index=False)

    args = pbstree.parse_args(["-n", str(node), "-e", str(edge), "-p", str(pf)])
    assert args.node_stats == str(node)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(pbstree, "TaxIDtree", FakeTaxIDtree)
    monkeypatch.setattr(pbstree, "format_matplotlib", lambda: None)
    monkeypatch.setattr(paf, "standard_plot_out", lambda *_args, **_kwargs: None)
    rc = pbstree.main(["-n", str(node), "-e", str(edge), "-p", str(pf)])
    assert rc == 0
    assert (tmp_path / "tree.pdf").exists()
