from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd
from ete4 import Tree

from egt import newick_to_common_ancestors as n2ca


class FakeNCBI:
    def __init__(self):
        self.taxid_to_name = {
            1: "root",
            2: "Chordata",
            3: "Arthropoda",
            4: "Homo sapiens",
            5: "Drosophila melanogaster",
        }
        self.name_to_taxid = {v: k for k, v in self.taxid_to_name.items()}
        self.lineages = {
            4: [1, 2, 4],
            5: [1, 3, 5],
            999: [9999, 1, 2, 4],
        }

    def get_taxid_translator(self, taxids):
        return {taxid: self.taxid_to_name[taxid] for taxid in taxids if taxid in self.taxid_to_name}

    def get_name_translator(self, names):
        return {name: [self.name_to_taxid[name]] for name in names if name in self.name_to_taxid}

    def get_lineage(self, taxid):
        return self.lineages[taxid]


def _make_tree(monkeypatch):
    fake_ncbi = FakeNCBI()
    monkeypatch.setattr(n2ca, "NCBITaxa", lambda: fake_ncbi)
    tree = n2ca.TaxIDtree()
    tree.NCBI = fake_ncbi
    return tree, fake_ncbi


def test_taxnode_and_taxedge_string_reprs():
    node = n2ca.TaxNode(1, "root")
    edge = n2ca.TaxEdge(1, 2)
    assert "TaxNode" in str(node)
    assert "TaxEdge" in str(edge)


def test_taxidtree_add_node_edge_root_and_lca(monkeypatch):
    tree, _fake = _make_tree(monkeypatch)
    tree.add_edge(1, 2)
    tree.add_edge(1, 3)
    tree.add_edge(2, 4)
    tree.add_edge(3, 5)

    assert tree.find_root() == 1
    assert tree.get_lineage(4) == [1, 2, 4]
    assert tree.find_LCA(4, 5) == 1
    assert tree.find_closest_relative(FakeNCBI(), 999) == 4
    assert "|_ 2" in str(tree)


def test_taxidtree_basic_age_helpers(monkeypatch):
    tree, _fake = _make_tree(monkeypatch)
    tree.add_edge(1, 2)
    tree.add_edge(2, 4)
    tree.nodes[1].nodeage = 10
    tree.nodes[2].nodeage = 4
    tree.nodes[4].nodeage = 0
    tree.nodes[1].nodeages = Counter({10: 2, 9: 1})
    tree.nodes[4].nodeages = Counter()

    assert tree.get_lineage_length([1, 2, 4]) == 10
    assert tree.get_dominant_age(1) == 10
    assert tree.find_children_with_ages(1) == [1]

    tree.set_leaf_ages_to_zero()
    assert tree.nodes[4].nodeage == 0
    tree.ensure_all_leaves_have_age_zero()


def test_taxidtree_build_from_newick_tree_and_chromosome_info(monkeypatch, tmp_path: Path):
    tree, fake_ncbi = _make_tree(monkeypatch)
    newick = n2ca.PhyloTree("(Homo_sapiens,Drosophila_melanogaster)Root[-67];", parser=1)

    root_id = tree.build_from_newick_tree(newick, fake_ncbi)

    assert root_id == -1000
    assert set(tree.nodes[-1000].children) == {4, 5}

    chrom = tmp_path / "chrom.tsv"
    chrom.write_text(
        "sample_string\tnum_chromosomes\n"
        "hs-4-x\t10\n"
        "dm-5-y\t6\n"
    )
    tree.add_chromosome_info_file(chrom)
    assert tree.nodes[4].chromsize_list == [10]
    assert tree.nodes[-1000].num_genomes == 2


def test_taxidtree_ingest_node_edge(monkeypatch):
    tree, _fake = _make_tree(monkeypatch)
    node_df = pd.DataFrame(
        {
            "taxid": [1, 2],
            "name": ["root", "leaf"],
            "parent": [None, 1],
            "children": ["{2}", "set()"],
            "nodeages": ["Counter({10: 1})", "Counter({0: 1})"],
            "lineage": ["[1]", "[1, 2]"],
            "chromsize_list": ["[10]", "[10]"],
            "num_genomes": [1, 1],
        }
    )
    edge_df = pd.DataFrame({"parent_taxid": [1], "child_taxid": [2], "branch_length": [10]})

    tree.ingest_node_edge(node_df, edge_df)

    assert tree.nodes[2].parent == 1
    assert (1, 2) in tree.edges
