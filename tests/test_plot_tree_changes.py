from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path


class FakeNode:
    def __init__(self, name):
        self.name = name
        self._is_leaf = True
        self.styled = False

    def is_leaf(self):
        return self._is_leaf

    def set_style(self, _style):
        self.styled = True


class FakeTree:
    last_instance = None

    def __init__(self, path, format=1):
        self.path = path
        self.format = format
        self.nodes = [FakeNode("10"), FakeNode("20")]
        self.pruned_to = None
        self.rendered = None
        FakeTree.last_instance = self

    def traverse(self):
        return list(self.nodes)

    def prune(self, collapse_list, preserve_branch_length=True):
        self.pruned_to = set(collapse_list)

    def render(self, filename, tree_style=None):
        self.rendered = (filename, tree_style)


class FakeNCBI:
    def get_taxid_translator(self, taxids):
        return {int(t): f"Taxon{t}" for t in taxids}

    def get_lineage(self, taxid):
        return [1, int(taxid)]

    def get_rank(self, lineage):
        return {lineage[-1]: "class"}


class FakeNodeStyle(dict):
    pass


class FakeTreeStyle:
    def __init__(self):
        self.show_leaf_name = False
        self.show_branch_length = True
        self.layout_fn = None


class FakeAttrFace:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


class FakeFaces:
    @staticmethod
    def add_face_to_node(face, node, column=0):
        node.face = (face, column)


def test_main_renders_cladogram(monkeypatch, tmp_path: Path):
    newick = tmp_path / "tree.nwk"
    newick.write_text("(10,20)1;")

    fake_ete4 = types.SimpleNamespace(
        Tree=FakeTree,
        TreeStyle=FakeTreeStyle,
        AttrFace=FakeAttrFace,
        faces=FakeFaces,
        NodeStyle=FakeNodeStyle,
        NCBITaxa=lambda: FakeNCBI(),
    )
    monkeypatch.setitem(sys.modules, "ete4", fake_ete4)
    ptc = importlib.import_module("egt.plot_tree_changes")

    rc = ptc.main([str(newick)])
    tree = FakeTree.last_instance
    assert rc == 0
    assert tree.pruned_to == {"10", "20"}
    assert tree.rendered[0] == "cladogram.pdf"
