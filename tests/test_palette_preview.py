from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

from egt import palette_preview as pp


class FakeColor:
    def __init__(self, color: str, label: str):
        self.color = color
        self.label = label


class FakePalette:
    def __init__(self):
        self.fallback = FakeColor("#999999", "fallback")
        self.source_path = "fake.yaml"
        self._items = {1: FakeColor("#ff0000", "red clade"), 2: FakeColor("#00ff00", "green clade")}

    def __len__(self):
        return len(self._items)

    def items(self):
        return self._items.items()

    def for_lineage(self, lineage):
        return self._items.get(lineage[0], self.fallback)


class FakeNCBI:
    def get_lineage(self, taxid):
        return [int(taxid), 1]

    def get_name_translator(self, names):
        return {"Name One": [1], "Name Two": [2]}


class FakeNode:
    def __init__(self, name, dist=1.0, children=None):
        self.name = name
        self.dist = dist
        self.children = children or []
        self.up = None
        for child in self.children:
            child.up = self

    @property
    def is_leaf(self):
        return len(self.children) == 0

    def leaves(self):
        if self.is_leaf:
            return [self]
        out = []
        for child in self.children:
            out.extend(child.leaves())
        return out

    def traverse(self, order=None):
        if order == "postorder":
            for child in self.children:
                yield from child.traverse(order)
            yield self
        else:
            yield self
            for child in self.children:
                yield from child.traverse(order)

    def detach(self):
        if self.up is not None:
            self.up.children = [c for c in self.up.children if c is not self]
            self.up = None

    def write(self):
        return "ignored"


class FakeTree(FakeNode):
    def __init__(self, *_args, **_kwargs):
        super().__init__(
            "Root",
            0.0,
            [
                FakeNode("Name_One[1]", 1.0),
                FakeNode("Sample-2-GCA123", 1.5),
            ],
        )


def test_extract_taxid_and_lineage_helpers():
    assert pp._extract_taxid("Metazoa[33208]") == 33208
    assert pp._extract_taxid("Branchiostoma-7739-GCF000003815.2") == 7739
    assert pp._extract_taxid("9606") == 9606
    assert pp._extract_taxid("Homo sapiens") is None

    assert pp._lineage_via_ncbi(3, FakeNCBI()) == [1, 3]


def test_render_and_export_helpers(monkeypatch, tmp_path: Path):
    fake_ete4 = types.SimpleNamespace(Tree=FakeTree, NCBITaxa=lambda: FakeNCBI())
    monkeypatch.setitem(sys.modules, "ete4", fake_ete4)

    tree_path = tmp_path / "tree.nwk"
    tree_path.write_text("(a,b);")
    outpdf = tmp_path / "preview.pdf"
    colored = tmp_path / "colored.nwk"
    nexus = tmp_path / "tree.nex"
    collapsed = tmp_path / "collapsed.nwk"
    collapsed_nex = tmp_path / "collapsed.nex"

    pp.render_palette_preview(
        tree_path=tree_path,
        palette=FakePalette(),
        output_path=outpdf,
        title="preview",
        show_labels=True,
        align_tips=True,
        colored_newick_path=colored,
        nexus_path=nexus,
        collapsed_newick_path=collapsed,
        collapsed_nexus_path=collapsed_nex,
        collapse_dominance=1.0,
    )

    assert outpdf.exists()
    assert "[&!color=" in colored.read_text()
    assert "#NEXUS" in nexus.read_text()
    assert collapsed.exists()
    assert collapsed_nex.exists()


def test_build_color_label_lookup_and_main(monkeypatch, tmp_path: Path):
    lut = pp._build_color_label_lookup(FakePalette())
    assert lut["#ff0000"] == "red clade"
    assert lut["#999999"] == "fallback"

    monkeypatch.setattr(pp.Palette, "from_yaml", staticmethod(lambda _path: FakePalette()))
    called = {}

    def fake_render(**kwargs):
        called.update(kwargs)

    monkeypatch.setattr(pp, "render_palette_preview", fake_render)
    tree = tmp_path / "tree.nwk"
    tree.write_text("(a,b);")
    out = tmp_path / "out.pdf"
    rc = pp.main(["--tree", str(tree), "--out", str(out)])
    assert rc == 0
    assert called["tree_path"] == tree
