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
        self._items = {
            1: FakeColor("#ff0000", "red clade"),
            2: FakeColor("#00ff00", "green clade"),
            3: FakeColor("#0000ff", "blue clade"),
            4: FakeColor("#ffaa00", "orange clade"),
        }
        self.by_taxid = self._items

    def __len__(self):
        return len(self._items)

    def items(self):
        return self._items.items()

    def canonicalize_taxid(self, taxid):
        return int(taxid) if taxid is not None else None

    def has_taxid(self, taxid):
        try:
            return int(taxid) in self._items
        except (TypeError, ValueError):
            return False

    def for_taxid(self, taxid):
        try:
            return self._items.get(int(taxid), self.fallback)
        except (TypeError, ValueError):
            return self.fallback

    def for_lineage(self, lineage):
        return self._items.get(lineage[0], self.fallback)


class FakeNCBI:
    def get_lineage(self, taxid):
        if int(taxid) == 1:
            return [1]
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


class FakeCollapsedTree(FakeNode):
    def __init__(self, *_args, **_kwargs):
        super().__init__(
            "Root",
            0.0,
            [
                FakeNode(
                    "Inner",
                    0.5,
                    [
                        FakeNode("Name_A[1]", 1.0),
                        FakeNode("Name_B[1]", 1.0),
                    ],
                ),
                FakeNode("Name_C[2]", 1.0),
            ],
        )


class FakeTaxTreeNode:
    def __init__(self, parent, children, x):
        self.parent = parent
        self.children = set(children)
        self.x = x


class FakeCircularTaxTreeNode:
    def __init__(self, parent, children, nodeage, x=None):
        self.parent = parent
        self.children = set(children)
        self.nodeage = nodeage
        self.x = x


class FakeCircularTaxTree:
    def __init__(self):
        self.root = 900
        self.leaf_order = [11, 12, 21]
        self.nodes = {
            900: FakeCircularTaxTreeNode(None, [910, 920], 12.0),
            910: FakeCircularTaxTreeNode(900, [11, 12], 6.0),
            920: FakeCircularTaxTreeNode(900, [21], 5.0),
            11: FakeCircularTaxTreeNode(910, [], 0.0),
            12: FakeCircularTaxTreeNode(910, [], 0.0),
            21: FakeCircularTaxTreeNode(920, [], 0.0),
        }

    def find_root(self):
        return self.root

    def sort_nodes(self, sort="ascending"):
        assert sort == "ascending"


def test_extract_taxid_and_lineage_helpers():
    assert pp._extract_taxid("Metazoa[33208]") == 33208
    assert pp._extract_taxid("Branchiostoma-7739-GCF000003815.2") == 7739
    assert pp._extract_taxid("9606") == 9606
    assert pp._extract_taxid("Homo sapiens") is None

    assert pp._lineage_via_ncbi(3, FakeNCBI()) == [1, 3]


def test_configure_vector_font_output_sets_type42():
    fake_matplotlib = types.SimpleNamespace(rcParams={})
    pp._configure_vector_font_output(fake_matplotlib)
    assert fake_matplotlib.rcParams["pdf.fonttype"] == 42
    assert fake_matplotlib.rcParams["ps.fonttype"] == 42


def test_triangle_layout_puts_root_up_and_tips_down():
    tree = FakeTree()
    leaves = tree.leaves()
    node_xy, leaf_label_xy = pp._compute_triangle_layout(tree, leaves)

    root_x, root_y = node_xy[tree]
    left_x, left_y = leaf_label_xy[leaves[0]]
    right_x, right_y = leaf_label_xy[leaves[1]]

    assert root_y < left_y
    assert root_y < right_y
    assert left_y == right_y
    assert left_x == 0.0
    assert right_x == 1.0
    assert root_x == 0.5


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


def test_render_triangle_layout(monkeypatch, tmp_path: Path):
    fake_ete4 = types.SimpleNamespace(Tree=FakeTree, NCBITaxa=lambda: FakeNCBI())
    monkeypatch.setitem(sys.modules, "ete4", fake_ete4)

    tree_path = tmp_path / "tree.nwk"
    tree_path.write_text("(a,b);")
    outpdf = tmp_path / "triangle.pdf"

    pp.render_palette_preview(
        tree_path=tree_path,
        palette=FakePalette(),
        output_path=outpdf,
        title="triangle",
        show_labels=True,
        layout="triangle",
    )

    assert outpdf.exists()


def test_render_triangle_layout_with_max_leaves(monkeypatch, tmp_path: Path):
    fake_ete4 = types.SimpleNamespace(Tree=FakeTree, NCBITaxa=lambda: FakeNCBI())
    monkeypatch.setitem(sys.modules, "ete4", fake_ete4)

    tree_path = tmp_path / "tree.nwk"
    tree_path.write_text("(a,b);")
    outpdf = tmp_path / "triangle_subset.pdf"

    pp.render_palette_preview(
        tree_path=tree_path,
        palette=FakePalette(),
        output_path=outpdf,
        layout="triangle",
        max_leaves=1,
        show_labels=True,
    )

    assert outpdf.exists()


def test_collapse_tree_copy_collapses_monophyletic_subtree(monkeypatch):
    fake_ete4 = types.SimpleNamespace(Tree=FakeCollapsedTree, NCBITaxa=lambda: FakeNCBI())
    monkeypatch.setitem(sys.modules, "ete4", fake_ete4)
    tree = FakeCollapsedTree()
    leaves = tree.leaves()
    leaf_colors = {
        leaves[0]: "#ff0000",
        leaves[1]: "#ff0000",
        leaves[2]: "#00ff00",
    }
    work, render_leaves, render_leaf_colors, collapsed_meta = pp._collapse_tree_copy(
        tree,
        leaves,
        leaf_colors,
        FakePalette(),
        dominance=1.0,
    )

    assert len(render_leaves) == 2
    collapsed = [leaf for leaf in render_leaves if id(leaf) in collapsed_meta]
    assert len(collapsed) == 1
    assert collapsed[0].name.startswith("red clade")
    assert render_leaf_colors[collapsed[0]] == "#ff0000"


def test_render_collapsed_tips_layout(monkeypatch, tmp_path: Path):
    fake_ete4 = types.SimpleNamespace(Tree=FakeCollapsedTree, NCBITaxa=lambda: FakeNCBI())
    monkeypatch.setitem(sys.modules, "ete4", fake_ete4)

    tree_path = tmp_path / "tree.nwk"
    tree_path.write_text("((a,b),c);")
    outpdf = tmp_path / "collapsed_tips.pdf"

    pp.render_palette_preview(
        tree_path=tree_path,
        palette=FakePalette(),
        output_path=outpdf,
        layout="collapsed-tips",
        show_labels=True,
    )

    assert outpdf.exists()


def test_build_palette_breadth_first_order():
    class TinyPalette:
        def __init__(self):
            self.by_taxid = {
                1: FakeColor("#ff0000", "root clade"),
                100: FakeColor("#00ff00", "implicit child"),
                200: FakeColor("#0000ff", "sibling child"),
            }

    class FakeNCBIForOrder:
        def get_lineage(self, taxid):
            mapping = {
                1: [1],
                100: [1, 100],
                200: [1, 200],
            }
            return mapping[int(taxid)]

    placements = {
        1: pp._PaletteCladePlacement(component_roots=[10], leaf_count=3, x_key=0.0),
        100: pp._PaletteCladePlacement(component_roots=[11], leaf_count=2, x_key=0.0),
        200: pp._PaletteCladePlacement(component_roots=[12], leaf_count=1, x_key=2.0),
    }

    order = pp._build_palette_breadth_first_order(TinyPalette(), placements, FakeNCBIForOrder())
    assert order == [1, 100, 200]


def test_resolve_palette_clade_placements_finds_implicit_component():
    class FakeTaxTree:
        def __init__(self):
            self.root = 900
            self.nodes = {
                900: FakeTaxTreeNode(None, [910, 920], 1.0),
                910: FakeTaxTreeNode(900, [11, 12], 0.5),
                920: FakeTaxTreeNode(900, [21], 2.0),
                11: FakeTaxTreeNode(910, [], 0.0),
                12: FakeTaxTreeNode(910, [], 1.0),
                21: FakeTaxTreeNode(920, [], 2.0),
            }

        def find_root(self):
            return self.root

    class TinyPalette:
        def __init__(self):
            self.by_taxid = {
                1: FakeColor("#ff0000", "root clade"),
                100: FakeColor("#00ff00", "implicit child"),
            }

    class FakeNCBIImplicit:
        def get_lineage(self, taxid):
            mapping = {
                11: [1, 100, 11],
                12: [1, 100, 12],
                21: [1, 21],
                100: [1, 100],
                1: [1],
            }
            return mapping[int(taxid)]

    placements = pp._resolve_palette_clade_placements(FakeTaxTree(), TinyPalette(), FakeNCBIImplicit())
    assert placements[100].component_roots == [910]
    assert placements[100].leaf_count == 2
    assert placements[100].x_mean == 0.5
    assert placements[1].component_roots == [900]


def test_resolve_palette_clade_placements_prefers_exact_node():
    class FakeTaxTree:
        def __init__(self):
            self.root = 900
            self.nodes = {
                900: FakeTaxTreeNode(None, [100, 21], 1.0),
                100: FakeTaxTreeNode(900, [11, 12], 0.5),
                21: FakeTaxTreeNode(900, [], 2.0),
                11: FakeTaxTreeNode(100, [], 0.0),
                12: FakeTaxTreeNode(100, [], 1.0),
            }

        def find_root(self):
            return self.root

    class TinyPalette:
        def __init__(self):
            self.by_taxid = {
                100: FakeColor("#00ff00", "custom exact clade"),
            }

    class FakeNCBIImplicit:
        def get_lineage(self, taxid):
            mapping = {
                11: [1, 100, 11],
                12: [1, 100, 12],
                21: [1, 21],
            }
            return mapping[int(taxid)]

    placements = pp._resolve_palette_clade_placements(FakeTaxTree(), TinyPalette(), FakeNCBIImplicit())
    assert placements[100].component_roots == [100]
    assert placements[100].leaf_count == 2


def test_compute_circular_tax_tree_layout_maps_age_to_radius():
    tree = FakeCircularTaxTree()
    ordered_leaf_ids, theta_by_node, radius_by_node, theta_start, available_sweep = pp._compute_circular_tax_tree_layout(
        tree,
        inner_radius=0.68,
    )

    assert ordered_leaf_ids == [11, 12, 21]
    assert radius_by_node[900] < radius_by_node[910] < radius_by_node[11]
    assert theta_by_node[11] < theta_by_node[12] < theta_by_node[21]

    tree_thickness = radius_by_node[11] - radius_by_node[900]
    assert round(tree_thickness, 6) == round(0.68 * 0.5, 6)
    assert theta_start == 0.0
    assert round(available_sweep, 6) == round((2 * 3.141592653589793) * 0.9, 6)


def test_compute_best_umap_rotation_uses_tree_clade_angles():
    umap_points = [
        (1.0, 0.0, 1, 1, "#ff0000"),
        (0.0, 1.0, 2, 2, "#00ff00"),
    ]
    tree_theta_by_taxid = {
        1: 3.141592653589793 / 2.0,
        2: 3.141592653589793,
    }
    rotation, reflect = pp._compute_best_umap_rotation(umap_points, tree_theta_by_taxid)
    assert round(rotation, 6) == round(3.141592653589793 / 2.0, 6)
    assert reflect is False


def test_quantized_age_ticks_prefers_round_numbers():
    assert pp._quantized_age_ticks(734.0, target_count=6) == [0.0, 200.0, 400.0, 600.0]
    assert pp._quantized_age_ticks(410.0, target_count=6) == [0.0, 100.0, 200.0, 300.0, 400.0]


def test_center_umap_inset_fraction_stays_inside_donut_hole():
    inset_fraction = pp._compute_center_umap_inset_fraction(
        inner_radius=0.68,
        total_radius=0.88,
    )
    assert inset_fraction < (0.68 / 0.88)
    assert inset_fraction <= 0.92


def test_scale_figure_to_target_width_hits_requested_mm():
    matplotlib = importlib.import_module("matplotlib")
    matplotlib.use("Agg")
    pp._configure_vector_font_output(matplotlib)
    plt = importlib.import_module("matplotlib.pyplot")

    fig = plt.figure(figsize=(4.0, 4.0))
    ax = fig.add_subplot(111)
    ax.text(0.5, 0.5, "test", fontsize=5)
    ax.set_axis_off()

    pp._scale_figure_to_target_width(fig, target_width_mm=184.0)

    width_in = pp._tight_bbox_width_inches(fig)
    assert abs((width_in * 25.4) - 184.0) < 1.0
    plt.close(fig)


def test_render_circular_nested_tree_coloring(monkeypatch, tmp_path: Path):
    fake_ete4 = types.SimpleNamespace(NCBITaxa=lambda: FakeNCBI())
    monkeypatch.setitem(sys.modules, "ete4", fake_ete4)
    monkeypatch.setattr(pp, "_build_taxidtree_from_source_tree", lambda *_args, **_kwargs: FakeCircularTaxTree())

    outpdf = tmp_path / "circular_nested.pdf"
    leaf_count, clade_count = pp._render_circular_nested_tree_coloring(
        source_tree=FakeTree(),
        palette=FakePalette(),
        output_path=outpdf,
        show_labels=True,
    )

    assert outpdf.exists()
    assert leaf_count == 3
    assert clade_count >= 1


def test_render_circular_nested_tree_coloring_emits_sidecar_layers(monkeypatch, tmp_path: Path):
    fake_ete4 = types.SimpleNamespace(NCBITaxa=lambda: FakeNCBI())
    monkeypatch.setitem(sys.modules, "ete4", fake_ete4)
    monkeypatch.setattr(pp, "_build_taxidtree_from_source_tree", lambda *_args, **_kwargs: FakeCircularTaxTree())
    monkeypatch.setattr(pp, "_draw_center_umap_overlay", lambda *args, **kwargs: None)

    outpdf = tmp_path / "circular_nested_layers.pdf"
    umap = tmp_path / "umap.tsv"
    umap.write_text("UMAP1\tUMAP2\ttaxid_list_str\n0\t0\t1\n")

    pp._render_circular_nested_tree_coloring(
        source_tree=FakeTree(),
        palette=FakePalette(),
        output_path=outpdf,
        show_labels=True,
        umap_df_path=umap,
        emit_layer_pdfs=True,
    )

    assert outpdf.exists()
    assert pp._sidecar_pdf_path(outpdf, "tree").exists()
    assert pp._sidecar_pdf_path(outpdf, "labels").exists()
    assert pp._sidecar_pdf_path(outpdf, "umap").exists()


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
    rc = pp.main([
        "--tree", str(tree),
        "--out", str(out),
        "--layout", "nested-tree-coloring",
        "--show-labels",
        "--font-size-pt", "6",
        "--target-width-mm", "184",
        "--nested-height-scale", "0.25",
    ])
    assert rc == 0
    assert called["tree_path"] == tree
    assert called["layout"] == "nested-tree-coloring"
    assert called["show_labels"] is True
    assert called["font_size_pt"] == 6.0
    assert called["target_width_mm"] == 184.0
    assert called["nested_height_scale"] == 0.25


def test_main_accepts_circular_nested_tree_layout(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(pp.Palette, "from_yaml", staticmethod(lambda _path: FakePalette()))
    called = {}

    def fake_render(**kwargs):
        called.update(kwargs)

    monkeypatch.setattr(pp, "render_palette_preview", fake_render)
    tree = tmp_path / "tree.nwk"
    tree.write_text("(a,b);")
    out = tmp_path / "out.pdf"
    umap = tmp_path / "umap.tsv"
    umap.write_text("UMAP1\tUMAP2\ttaxid_list_str\n0\t0\t1\n")
    rc = pp.main([
        "--tree", str(tree),
        "--out", str(out),
        "--layout", "circular-nested-tree-coloring",
        "--show-labels",
        "--umap-df", str(umap),
        "--font-size-pt", "5",
        "--target-width-mm", "184",
        "--min-linewidth-pt", "0.5",
        "--emit-layer-pdfs",
    ])
    assert rc == 0
    assert called["layout"] == "circular-nested-tree-coloring"
    assert called["show_labels"] is True
    assert called["umap_df_path"] == umap
    assert called["font_size_pt"] == 5.0
    assert called["target_width_mm"] == 184.0
    assert called["min_linewidth_pt"] == 0.5
    assert called["emit_layer_pdfs"] is True
