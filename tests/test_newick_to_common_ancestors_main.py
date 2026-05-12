from __future__ import annotations

from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import yaml

from egt import newick_to_common_ancestors as n2ca


class FakeLeaf:
    def __init__(self, name, dist=0.0, support=None, up=None, is_leaf=True, children=None):
        self.name = name
        self.dist = dist
        self.support = support
        self.up = up
        self.is_leaf = is_leaf
        self.children = children or []

    def write(self, format=1):
        return f"{self.name or 'None'};"


class FakeTree:
    def __init__(self, *_args, **_kwargs):
        self._leaves = [
            FakeLeaf("Homo_sapiens[9606]"),
            FakeLeaf("Drosophila_melanogaster[7227]"),
            FakeLeaf("Spongeus[6040]"),
            FakeLeaf("CnidA[6073]"),
            FakeLeaf("CnidB[6074]"),
        ]

    def leaves(self):
        return self._leaves


class FakeNode:
    def __init__(self, name, parent=None, children=None):
        self.name = name
        self.parent = parent
        self.children = set(children or [])
        self.nodeages = Counter()
        self.nodeage = None

    def __repr__(self):
        return f"FakeNode(name={self.name!r})"


class FakeTaxIDTree:
    def __init__(self):
        self.nodes = {}

    def _ensure_node(self, node_id, name=None, parent=None):
        if node_id not in self.nodes:
            self.nodes[node_id] = FakeNode(name or f"Node{node_id}", parent=parent)
        if name is not None:
            self.nodes[node_id].name = name
        if parent is not None:
            self.nodes[node_id].parent = parent
        return self.nodes[node_id]

    def build_from_newick_tree(self, _tree, _ncbi):
        structure = {
            1: ("Metazoa", None, [100, 6040, 200, -67]),
            100: ("Bilateria", 1, [9606, 7227]),
            200: ("Cnidaria", 1, [6073, 6074]),
            -67: ("Myriazoa", 1, []),
            9606: ("Homo sapiens", 100, []),
            7227: ("Drosophila melanogaster", 100, []),
            6040: ("Porifera species", 1, []),
            6073: ("CnidA", 200, []),
            6074: ("CnidB", 200, []),
        }
        for node_id, (name, parent, children) in structure.items():
            node = self._ensure_node(node_id, name=name, parent=parent)
            node.children = set(children)
        return 0

    def add_lineage_info(self):
        return 0

    def get_lineage(self, taxid):
        mapping = {
            9606: [1, 100, 9606],
            7227: [1, 100, 7227],
            6040: [1, 6040],
            6073: [1, 200, 6073],
            6074: [1, 200, 6074],
            -67: [1, -67],
        }
        return mapping[int(taxid)]

    def add_edge(self, parent, child):
        parent = int(parent)
        child = int(child)
        self._ensure_node(parent)
        self._ensure_node(child, parent=parent)
        self.nodes[parent].children.add(child)
        return 0

    def find_closest_relative(self, _ncbi, taxid):
        return int(taxid)

    def set_leaf_ages_to_zero(self):
        for node_id, node in self.nodes.items():
            if not node.children:
                node.nodeage = 0.0
                node.nodeages.update([0.0])

    def find_LCA(self, taxid1, taxid2):
        pair = {int(taxid1), int(taxid2)}
        if pair <= {9606, 7227}:
            return 100
        if pair <= {6073, 6074}:
            return 200
        return 1

    def find_root(self):
        return 1

    def correct_missing_nodes(self, priority_node_ages=None):
        priority_node_ages = priority_node_ages or {}
        for node_id, node in self.nodes.items():
            if node_id in priority_node_ages:
                node.nodeage = priority_node_ages[node_id][0]
            elif node.nodeages:
                node.nodeage = node.nodeages.most_common(1)[0][0]
            elif not node.children:
                node.nodeage = 0.0
            else:
                node.nodeage = 100.0

    def ensure_all_leaves_have_age_zero(self):
        self.set_leaf_ages_to_zero()

    def analyze_zero_length_branches(self, tolerance=0.01, label=""):
        return {"total_zero_length": 0, "zero_length_with_children": 0}

    def fix_zero_length_branches(self, tolerance=0.01, max_iterations=5):
        return 0

    def calc_dist_crown(self):
        return 0

    def add_chromosome_info_file(self, _path):
        return 0

    def generate_tree_report(self, path, tolerance=0.01):
        Path(path).write_text("tree report\n")

    def print_edge_information(self, path):
        Path(path).write_text("parent\tchild\n1\t2\n")

    def print_node_information(self, path):
        Path(path).write_text("taxid\tname\n1\tMetazoa\n")

    def write_newick(self, path):
        Path(path).write_text("(A,B);\n")


class FakeNCBI:
    def get_name_translator(self, names):
        mapping = {
            "Homo sapiens": [9606],
            "Drosophila melanogaster": [7227],
            "Spongeus": [6040],
            "CnidA": [6073],
            "CnidB": [6074],
        }
        return {name: mapping[name] for name in names}

    def get_taxid_translator(self, taxids):
        names = {
            1: "Metazoa",
            100: "Bilateria",
            200: "Cnidaria",
            -67: "Myriazoa",
            9606: "Homo sapiens",
            7227: "Drosophila melanogaster",
            6040: "Spongeus",
            6073: "CnidA",
            6074: "CnidB",
        }
        return {int(t): names[int(t)] for t in taxids if int(t) in names}

    def get_lineage(self, taxid):
        return FakeTaxIDTree().get_lineage(int(taxid))


def test_main_smoke_with_fake_topology(tmp_path, monkeypatch):
    topology = tmp_path / "topology.nwk"
    timetree = tmp_path / "time.nwk"
    config = tmp_path / "config.yaml"
    for path in [topology, timetree]:
        path.write_text("(A,B);\n")
    config.write_text(
        yaml.safe_dump(
            {
                "species": {
                    "human": {"taxid": 9606},
                    "fly": {"taxid": 7227},
                    "sponge": {"taxid": 6040},
                    "cnidb": {"taxid": 6074},
                }
            }
        )
    )

    args = SimpleNamespace(
        time_newick=str(timetree),
        topology_newick=str(topology),
        config=str(config),
        prefix=str(tmp_path / "out"),
        chromosome_sizes=None,
    )
    monkeypatch.setattr(n2ca, "parse_args", lambda argv=None: args)
    monkeypatch.setattr(n2ca, "NCBITaxa", lambda: FakeNCBI())
    monkeypatch.setattr(n2ca, "PhyloTree", FakeTree)
    monkeypatch.setattr(n2ca, "TaxIDtree", FakeTaxIDTree)
    monkeypatch.setattr(n2ca, "annotate_custom_tree_with_timetree_ages", lambda *args, **kwargs: ({(9606, 7227): 500.0, (6073, 6074): 600.0}, {}, {}))
    monkeypatch.setattr(n2ca, "extract_timetree_root_age_as_metazoa", lambda _tree: 700.0)
    monkeypatch.setattr(n2ca, "is_subspecies_or_below", lambda taxid, ncbi: False)
    monkeypatch.setattr(n2ca, "get_species_level_taxid", lambda taxid, ncbi: taxid)
    monkeypatch.setattr(n2ca, "report_divergence_time_all_vs_all", lambda tree, prefix: Path(f"{prefix}.divergence_times.txt").write_text("9606\t7227\t500.0\n"))

    assert n2ca.main([]) == 0
    assert (tmp_path / "out.node_ages_for_config.tsv").exists()
    assert (tmp_path / "out.tree_report.txt").exists()
    assert (tmp_path / "out.edge_information.tsv").exists()
    assert (tmp_path / "out.node_information.tsv").exists()
    assert (tmp_path / "out.divergence_times.txt").exists()
    assert (tmp_path / "out.calibrated_tree.nwk").exists()


def test_main_handles_none_and_unknown_leaves(tmp_path, monkeypatch):
    topology = tmp_path / "topology.nwk"
    timetree = tmp_path / "time.nwk"
    config = tmp_path / "config.yaml"
    for path in [topology, timetree]:
        path.write_text("(A,B);\n")
    config.write_text(yaml.safe_dump({"species": {"human": {"taxid": 9606}}}))

    args = SimpleNamespace(
        time_newick=str(timetree),
        topology_newick=str(topology),
        config=str(config),
        prefix=str(tmp_path / "weird"),
        chromosome_sizes=None,
    )

    parent = FakeLeaf("parent", dist=1.0, support=0.5, up=None, is_leaf=False, children=[])
    none_leaf = FakeLeaf(None, dist=0.1, support=0.2, up=parent)
    bad_leaf = FakeLeaf("Unknown_species", dist=0.2, support=0.3, up=parent)
    good_leaf = FakeLeaf("Known_species[9606]", dist=0.4, support=0.9, up=parent)
    parent.children = [none_leaf, bad_leaf, good_leaf]

    class WeirdTree:
        def __init__(self, *_args, **_kwargs):
            self._leaves = [none_leaf, bad_leaf, good_leaf]

        def leaves(self):
            return self._leaves

    class SmallTaxTree(FakeTaxIDTree):
        def build_from_newick_tree(self, _tree, _ncbi):
            self._ensure_node(1, name="Metazoa", parent=None)
            self._ensure_node(9606, name="Homo sapiens", parent=1)
            self.nodes[1].children = {9606}
            self.nodes[9606].children = set()
            return 0

    class SmallNCBI(FakeNCBI):
        def get_name_translator(self, names):
            mapping = {"Known species": [9606]}
            return {name: mapping[name] for name in names if name in mapping}

    monkeypatch.setattr(n2ca, "parse_args", lambda argv=None: args)
    monkeypatch.setattr(n2ca, "NCBITaxa", lambda: SmallNCBI())
    monkeypatch.setattr(n2ca, "PhyloTree", WeirdTree)
    monkeypatch.setattr(n2ca, "TaxIDtree", SmallTaxTree)
    monkeypatch.setattr(n2ca, "annotate_custom_tree_with_timetree_ages", lambda *args, **kwargs: ({}, {}, {}))
    monkeypatch.setattr(n2ca, "extract_timetree_root_age_as_metazoa", lambda _tree: 700.0)
    monkeypatch.setattr(n2ca, "is_subspecies_or_below", lambda taxid, ncbi: False)
    monkeypatch.setattr(n2ca, "get_species_level_taxid", lambda taxid, ncbi: taxid)
    monkeypatch.setattr(n2ca, "report_divergence_time_all_vs_all", lambda tree, prefix: {})

    assert n2ca.main([]) == 0
    assert (tmp_path / "weird.node_information.tsv").exists()
