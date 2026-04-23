"""egt palette-preview — render a phylogenetic tree colored by the paper palette.

This is a sanity-check tool: plot the calibrated (or uncalibrated) Newick
tree with every leaf colored by its palette-resolved clade, and a legend
of all palette entries. If a color looks wrong on the tree, fix the YAML
and re-render. No analysis — pure visualization.

Accepts either:
  - a calibrated Newick emitted by `egt newick-to-common-ancestors`, whose
    internal nodes are labeled `Name[taxid]` (e.g. `Metazoa[33208]`), OR
  - a plain NCBI Newick whose internal + leaf names are species-level
    taxids or strings that can be resolved via the shipped NCBITaxa DB.
"""
from __future__ import annotations

import argparse
from collections import deque
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from egt.palette import Palette, add_palette_argument


# ---------- helpers ----------

_TAXID_IN_BRACKETS = re.compile(r"\[(\-?\d+)\]\s*$")
_TAXID_TRAIL_DASH  = re.compile(r"-(\d+)-")
_BARE_TAXID        = re.compile(r"^-?\d+$")


@dataclass
class _PaletteAssignmentNode:
    """Palette-entry node induced from the source phylogeny."""

    taxid: int | None
    label: str
    color: str
    depth: float
    order: int
    source_node: object | None = None
    children: list["_PaletteAssignmentNode"] = field(default_factory=list)
    slot: int | None = None
    span: tuple[float, float] | None = None
    x: float | None = None
    y: float | None = None


@dataclass
class _PaletteCladePlacement:
    """How a palette clade maps onto the plotted tree."""

    component_roots: list[int]
    leaf_count: int
    x_key: float
    x_mean: float = 0.0


def _extract_taxid(label: str) -> int | None:
    """Best-effort extraction of an NCBI taxid from a node label.

    Handles common conventions:
      - `Metazoa[33208]`            → 33208
      - `Branchiostomafloridae-7739-GCF000003815.2`  → 7739
      - `9606`                      → 9606
      - `Homo sapiens` or `Celegans` → None (caller may resolve via NCBITaxa)
    """
    if label is None:
        return None
    s = str(label).strip()
    if not s:
        return None
    m = _TAXID_IN_BRACKETS.search(s)
    if m:
        return int(m.group(1))
    m = _TAXID_TRAIL_DASH.search(s)
    if m:
        return int(m.group(1))
    if _BARE_TAXID.match(s):
        return int(s)
    return None


def _lineage_via_ncbi(taxid: int, ncbi) -> list[int]:
    """Return [most-specific … most-general] lineage for a taxid."""
    try:
        lineage = ncbi.get_lineage(taxid) or []
    except Exception:
        return [taxid]
    return list(reversed(lineage))


def _configure_vector_font_output(matplotlib) -> None:
    """Use Type 42 fonts so PDF text stays editable in Illustrator."""
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42


def _tight_bbox_width_inches(fig) -> float:
    """Return the figure's tight-bbox export width in inches."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox = fig.get_tightbbox(renderer)
    return float(bbox.width)


def _scale_figure_to_target_width(fig, *, target_width_mm: float | None) -> None:
    """Scale the figure so a tight export lands at the requested physical width."""
    if target_width_mm is None or target_width_mm <= 0:
        return
    target_width_in = float(target_width_mm) / 25.4
    if target_width_in <= 0:
        return
    for _ in range(3):
        current_width_in = _tight_bbox_width_inches(fig)
        if current_width_in <= 0:
            return
        scale = target_width_in / current_width_in
        if abs(scale - 1.0) < 0.002:
            return
        width_in, height_in = fig.get_size_inches()
        fig.set_size_inches(width_in * scale, height_in * scale, forward=True)


def _sidecar_pdf_path(output_path: Path, suffix: str) -> Path:
    """Return a sibling PDF path with an extra descriptive suffix."""
    return output_path.with_name(f"{output_path.stem}.{suffix}{output_path.suffix}")


def _node_depth(node) -> float:
    """Return cumulative branch length from root to node."""
    d = 0.0
    cur = node
    while cur and cur.up is not None:
        d += float(getattr(cur, "dist", 1.0) or 1.0)
        cur = cur.up
    return d


def _triangle_height(node) -> int:
    """Return topological height above the tips for triangle layout."""
    if node.is_leaf:
        return 0
    return 1 + max(_triangle_height(child) for child in node.children)


def _compute_rectilinear_layout(tree, leaves, *, align_tips: bool) -> dict[object, tuple[float, float]]:
    """Current left-to-right phylogram layout."""
    node_xy: dict[object, tuple[float, float]] = {}
    leaf_set = set(leaves)
    leaf_y = {leaf: i for i, leaf in enumerate(leaves)}
    max_leaf_depth = max(_node_depth(leaf) for leaf in leaves) if align_tips else None

    for leaf in leaves:
        x = max_leaf_depth if align_tips else _node_depth(leaf)
        node_xy[leaf] = (x, leaf_y[leaf])

    def _fill_internal(node):
        if node.is_leaf:
            return node_xy[node] if node in leaf_set else None
        child_xys = [xy for xy in (_fill_internal(c) for c in node.children) if xy is not None]
        if not child_xys:
            return None
        y = sum(xy[1] for xy in child_xys) / len(child_xys)
        x = _node_depth(node)
        node_xy[node] = (x, y)
        return (x, y)

    _fill_internal(tree)
    return node_xy


def _compute_triangle_layout(tree, leaves) -> tuple[dict[object, tuple[float, float]], dict[object, tuple[float, float]]]:
    """Topological rooted-triangle layout with tips aligned at the bottom."""
    leaf_set = set(leaves)
    leaf_x = {leaf: float(i) for i, leaf in enumerate(leaves)}
    max_height = _triangle_height(tree)
    node_xy: dict[object, tuple[float, float]] = {}
    leaf_label_xy: dict[object, tuple[float, float]] = {}

    def _fill(node):
        if node.is_leaf:
            if node not in leaf_set:
                return None
            x = leaf_x[node]
            y = float(max_height)
            node_xy[node] = (x, y)
            leaf_label_xy[node] = (x, y)
            return x, x
        spans = [span for span in (_fill(child) for child in node.children) if span is not None]
        if not spans:
            return None
        min_x = min(span[0] for span in spans)
        max_x = max(span[1] for span in spans)
        x = (min_x + max_x) / 2.0
        y = float(max_height - _triangle_height(node))
        node_xy[node] = (x, y)
        return min_x, max_x

    _fill(tree)
    return node_xy, leaf_label_xy


def _draw_rectilinear_edges(ax, tree, node_xy) -> None:
    def _draw(node):
        if node not in node_xy:
            for child in node.children:
                _draw(child)
            return
        if node is tree:
            px, py = node_xy[node]
        else:
            if node.up not in node_xy:
                for child in node.children:
                    _draw(child)
                return
            px, py = node_xy[node.up]
        cx, cy = node_xy[node]
        ax.plot([px, px, cx], [py, cy, cy], color="black", linewidth=0.4, solid_capstyle="butt")
        for child in node.children:
            _draw(child)

    _draw(tree)


def _draw_rectilinear_edges_with_collapsed_tips(ax, tree, node_xy, collapsed_meta) -> None:
    def _draw(node):
        if node not in node_xy:
            for child in node.children:
                _draw(child)
            return
        if node is tree:
            for child in node.children:
                _draw(child)
            return
        if node.up not in node_xy:
            for child in node.children:
                _draw(child)
            return

        px, py = node_xy[node.up]
        cx, cy = node_xy[node]
        if id(node) in collapsed_meta:
            base_x = px + float(collapsed_meta[id(node)]["orig_dist"])
            ax.plot([px, px, base_x], [py, cy, cy], color="black", linewidth=0.4, solid_capstyle="butt")
        else:
            ax.plot([px, px, cx], [py, cy, cy], color="black", linewidth=0.4, solid_capstyle="butt")

        for child in node.children:
            _draw(child)

    _draw(tree)


def _draw_triangle_edges(ax, tree, node_xy) -> None:
    for node in tree.traverse():
        if node not in node_xy or node.up is None or node.up not in node_xy:
            continue
        px, py = node_xy[node.up]
        cx, cy = node_xy[node]
        ax.plot([px, cx], [py, cy], color="black", linewidth=0.5, solid_capstyle="round")


def _compute_weighted_rectilinear_layout(
    tree,
    leaves,
    *,
    align_tips: bool,
    leaf_weights: dict[object, float] | None = None,
) -> tuple[dict[object, tuple[float, float]], dict[object, tuple[float, float]]]:
    """Rectilinear layout that reserves vertical space for collapsed wedges."""
    node_xy: dict[object, tuple[float, float]] = {}
    node_spans: dict[object, tuple[float, float]] = {}
    leaf_set = set(leaves)
    max_leaf_depth = max(_node_depth(leaf) for leaf in leaves) if align_tips else None
    cursor = 0.0

    for leaf in leaves:
        weight = float((leaf_weights or {}).get(leaf, 1.0))
        weight = max(1.0, weight)
        y0 = cursor
        y1 = cursor + weight - 1.0
        y = (y0 + y1) / 2.0
        x = max_leaf_depth if align_tips else _node_depth(leaf)
        node_xy[leaf] = (x, y)
        node_spans[leaf] = (y0, y1)
        cursor = y1 + 1.0

    def _fill_internal(node):
        if node.is_leaf:
            return node_spans[node] if node in leaf_set else None
        child_spans = [span for span in (_fill_internal(c) for c in node.children) if span is not None]
        if not child_spans:
            return None
        y0 = min(span[0] for span in child_spans)
        y1 = max(span[1] for span in child_spans)
        y = (y0 + y1) / 2.0
        x = _node_depth(node)
        node_xy[node] = (x, y)
        node_spans[node] = (y0, y1)
        return (y0, y1)

    _fill_internal(tree)
    return node_xy, node_spans


def _build_palette_assignment_tree(tree, palette: Palette) -> tuple[_PaletteAssignmentNode, list[_PaletteAssignmentNode]]:
    """Lift palette taxids present in the source tree into an induced tree.

    Each palette entry becomes one display unit. Internal palette clades stay
    internal in the induced tree; they do not become fake bifurcating branches.
    """
    leaf_order = {leaf: idx for idx, leaf in enumerate(tree.leaves())}
    leftmost_cache: dict[object, int] = {}

    def _leftmost_leaf_index(node) -> int:
        if node in leftmost_cache:
            return leftmost_cache[node]
        if node.is_leaf:
            idx = leaf_order.get(node, 0)
        else:
            idx = min(_leftmost_leaf_index(child) for child in node.children)
        leftmost_cache[node] = idx
        return idx

    assignments_by_taxid: dict[int, _PaletteAssignmentNode] = {}
    for node in tree.traverse():
        tid = _extract_taxid(getattr(node, "name", ""))
        if tid is None or not palette.has_taxid(tid):
            continue
        cc = palette.for_taxid(tid)
        assignment_taxid = cc.taxid
        depth = _node_depth(node)
        order = _leftmost_leaf_index(node)
        prev = assignments_by_taxid.get(assignment_taxid)
        if prev is None or depth < prev.depth or (depth == prev.depth and order < prev.order):
            assignments_by_taxid[assignment_taxid] = _PaletteAssignmentNode(
                taxid=assignment_taxid,
                label=cc.label,
                color=cc.color,
                depth=depth,
                order=order,
                source_node=node,
            )

    for assignment in assignments_by_taxid.values():
        assignment.children = []

    roots: list[_PaletteAssignmentNode] = []
    for tid, assignment in assignments_by_taxid.items():
        cur = getattr(assignment.source_node, "up", None)
        parent_assignment = None
        while cur is not None:
            parent_tid = _extract_taxid(getattr(cur, "name", ""))
            canonical_parent_tid = palette.canonicalize_taxid(parent_tid)
            if canonical_parent_tid in assignments_by_taxid:
                parent_assignment = assignments_by_taxid[canonical_parent_tid]
                break
            cur = getattr(cur, "up", None)
        if parent_assignment is None:
            roots.append(assignment)
        else:
            parent_assignment.children.append(assignment)

    def _sort_children(node: _PaletteAssignmentNode) -> None:
        node.children.sort(key=lambda child: child.order)
        for child in node.children:
            _sort_children(child)

    for root in roots:
        _sort_children(root)

    roots.sort(key=lambda node: node.order)
    synthetic_root = _PaletteAssignmentNode(
        taxid=None,
        label="root",
        color="#000000",
        depth=0.0,
        order=min((root.order for root in roots), default=0),
        source_node=None,
        children=roots,
    )

    def _assign_slots(node: _PaletteAssignmentNode, next_slot: int) -> int:
        if not node.children:
            if node.taxid is not None:
                node.slot = next_slot
                return next_slot + 1
            return next_slot
        split = len(node.children) // 2
        cursor = next_slot
        for child in node.children[:split]:
            cursor = _assign_slots(child, cursor)
        if node.taxid is not None:
            node.slot = cursor
            cursor += 1
        for child in node.children[split:]:
            cursor = _assign_slots(child, cursor)
        return cursor

    _assign_slots(synthetic_root, 0)

    ordered_assignments: list[_PaletteAssignmentNode] = []

    def _finalize(node: _PaletteAssignmentNode) -> tuple[float, float]:
        spans: list[tuple[float, float]] = []
        for child in node.children:
            spans.append(_finalize(child))
        if node.slot is not None:
            own_span = (float(node.slot), float(node.slot))
            spans.append(own_span)
            ordered_assignments.append(node)
        if not spans:
            spans = [(0.0, 0.0)]
        x0 = min(span[0] for span in spans)
        x1 = max(span[1] for span in spans)
        node.span = (x0, x1)
        node.x = (x0 + x1) / 2.0
        node.y = node.depth
        return node.span

    _finalize(synthetic_root)
    ordered_assignments.sort(key=lambda node: (-1 if node.slot is None else node.slot))
    return synthetic_root, ordered_assignments


def _draw_topdown_rectilinear_edges(ax, root: _PaletteAssignmentNode) -> None:
    for child in root.children:
        if root.x is None or root.y is None or child.x is None or child.y is None:
            continue
        ax.plot(
            [root.x, root.x, child.x],
            [root.y, child.y, child.y],
            color="black",
            linewidth=0.45,
            solid_capstyle="butt",
            zorder=1,
        )
        _draw_topdown_rectilinear_edges(ax, child)


def _render_palette_assignment_triangles(
    *,
    tree,
    palette: Palette,
    output_path: Path,
    title: str | None = None,
    show_labels: bool = False,
) -> int:
    """Render one labeled bottom-axis slot per palette entry present in the tree."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    root, ordered_assignments = _build_palette_assignment_tree(tree, palette)
    if not ordered_assignments:
        raise SystemExit("No palette taxids from the YAML were found in the tree.")

    n_assignments = len(ordered_assignments)
    max_depth = max(node.depth for node in ordered_assignments)
    bottom_y = max_depth + max(0.8, 0.12 * max_depth)
    width = max(18.0, 0.22 * n_assignments)
    height = max(8.0, 7.0 + 0.02 * max_depth)
    fig, ax = plt.subplots(1, 1, figsize=(width, height))

    _draw_topdown_rectilinear_edges(ax, root)

    for node in ordered_assignments:
        if node.slot is None or node.x is None or node.y is None:
            continue
        tri = Polygon(
            [
                (node.slot - 0.42, bottom_y),
                (node.slot + 0.42, bottom_y),
                (node.x, node.y),
            ],
            closed=True,
            facecolor=node.color,
            edgecolor="none",
            alpha=0.92,
            zorder=3,
        )
        ax.add_patch(tri)

    if show_labels:
        tick_positions = [node.slot for node in ordered_assignments if node.slot is not None]
        tick_labels = [node.label for node in ordered_assignments if node.slot is not None]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=7)
        for tick, node in zip(ax.get_xticklabels(), ordered_assignments):
            tick.set_color(node.color)
    else:
        ax.set_xticks([])

    ax.set_xlim(-1.0, n_assignments)
    ax.set_ylim(bottom_y + 0.6, -0.4)
    ax.set_xlabel("palette assignments")
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    if title:
        ax.set_title(title, fontsize=10)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close(fig)
    return n_assignments


def _build_taxidtree_from_source_tree(source_tree, ncbi):
    """Convert a calibrated ete tree into the existing TaxIDtree plotting backbone."""
    from egt.newick_to_common_ancestors import TaxIDtree

    tax_tree = TaxIDtree()
    tax_tree.build_from_newick_tree(source_tree, ncbi)
    root_to_tip = max((_node_depth(leaf) for leaf in source_tree.leaves()), default=0.0)
    for node in source_tree.traverse():
        tid = _extract_taxid(getattr(node, "name", ""))
        if tid is None or tid not in tax_tree.nodes:
            continue
        tax_tree.nodes[tid].nodeage = root_to_tip - _node_depth(node)

    def _fill_missing_nodeages(node_id: int, parent_age: float | None = None) -> float:
        node = tax_tree.nodes[node_id]
        if len(node.children) == 0:
            if node.nodeage is None:
                node.nodeage = 0.0
            return float(node.nodeage)

        current_age = float(node.nodeage) if node.nodeage is not None else None
        child_ages = [
            _fill_missing_nodeages(child_id, current_age)
            for child_id in node.children
        ]

        if node.nodeage is None:
            max_child_age = max(child_ages)
            if parent_age is not None and parent_age > max_child_age:
                # Synthetic unary internal nodes should sit between their parent and child.
                node.nodeage = (parent_age + max_child_age) / 2.0
            else:
                node.nodeage = max_child_age + 1.0
        return float(node.nodeage)

    if tax_tree.root is None:
        tax_tree.root = tax_tree.find_root()
    if tax_tree.root is not None:
        _fill_missing_nodeages(tax_tree.root)

    tax_tree.calc_dist_crown()
    tax_tree.add_lineage_info()
    return tax_tree


def _collect_taxid_subtree(tax_tree, root_taxid: int) -> set[int]:
    """Return all descendants including the root taxid."""
    descendants: set[int] = set()
    stack = [root_taxid]
    while stack:
        current = stack.pop()
        if current in descendants or current not in tax_tree.nodes:
            continue
        descendants.add(current)
        stack.extend(tax_tree.nodes[current].children)
    return descendants


def _tax_tree_postorder_nodes(tax_tree) -> list[int]:
    """Return tax-tree node ids in postorder."""
    if tax_tree.root is None:
        tax_tree.root = tax_tree.find_root()
    if tax_tree.root is None:
        return []
    order: list[int] = []
    stack: list[tuple[int, bool]] = [(tax_tree.root, False)]
    while stack:
        node_id, seen = stack.pop()
        if seen:
            order.append(node_id)
            continue
        stack.append((node_id, True))
        for child_id in tax_tree.nodes[node_id].children:
            stack.append((child_id, False))
    return order


def _compute_total_leaf_counts(tax_tree, postorder: list[int]) -> dict[int, int]:
    """Count descendant leaves for every plotted node."""
    total_leaf_counts: dict[int, int] = {}
    for node_id in postorder:
        node = tax_tree.nodes[node_id]
        if len(node.children) == 0:
            total_leaf_counts[node_id] = 1
        else:
            total_leaf_counts[node_id] = sum(total_leaf_counts[child_id] for child_id in node.children)
    return total_leaf_counts


def _resolve_palette_clade_placements(
    tax_tree,
    palette: Palette,
    ncbi,
    *,
    ordered_leaf_ids: list[int] | None = None,
) -> dict[int, _PaletteCladePlacement]:
    """Resolve palette clades onto exact or implicit tree components via leaf membership."""
    postorder = _tax_tree_postorder_nodes(tax_tree)
    total_leaf_counts = _compute_total_leaf_counts(tax_tree, postorder)
    leaf_ids = [node_id for node_id, node in tax_tree.nodes.items() if len(node.children) == 0]
    if ordered_leaf_ids is None:
        if hasattr(tax_tree, "sort_nodes"):
            tax_tree.sort_nodes("ascending")
            ordered_leaf_ids = list(getattr(tax_tree, "leaf_order", leaf_ids))
        else:
            ordered_leaf_ids = sorted(
                leaf_ids,
                key=lambda leaf_id: (
                    getattr(tax_tree.nodes[leaf_id], "x", None) is None,
                    getattr(tax_tree.nodes[leaf_id], "x", 0.0),
                    leaf_id,
                ),
            )
    leaf_x = {leaf_id: float(i) for i, leaf_id in enumerate(ordered_leaf_ids)}
    leaf_lineages = {leaf_id: set(_lineage_via_ncbi(leaf_id, ncbi)) for leaf_id in leaf_ids}

    placements: dict[int, _PaletteCladePlacement] = {}
    for tid in palette.by_taxid:
        if tid in tax_tree.nodes:
            descendants = _collect_taxid_subtree(tax_tree, tid)
            target_leaf_ids = {
                node_id for node_id in descendants
                if len(tax_tree.nodes[node_id].children) == 0
            }
        else:
            target_leaf_ids = {leaf_id for leaf_id, lineage in leaf_lineages.items() if tid in lineage}
        if not target_leaf_ids:
            continue

        target_counts: dict[int, int] = {}
        for node_id in postorder:
            node = tax_tree.nodes[node_id]
            if len(node.children) == 0:
                target_counts[node_id] = 1 if node_id in target_leaf_ids else 0
            else:
                target_counts[node_id] = sum(target_counts[child_id] for child_id in node.children)

        component_roots: list[int] = []
        for node_id in postorder:
            target_count = target_counts[node_id]
            if target_count == 0 or target_count != total_leaf_counts[node_id]:
                continue
            parent_id = tax_tree.nodes[node_id].parent
            parent_is_full = (
                parent_id is not None
                and parent_id != -1
                and target_counts.get(parent_id, 0) == total_leaf_counts.get(parent_id, 0)
            )
            if not parent_is_full:
                component_roots.append(node_id)

        if not component_roots:
            continue

        x_positions = [
            leaf_x[leaf_id]
            for leaf_id in target_leaf_ids
            if leaf_id in leaf_x
        ]
        x_key = min(float(x) for x in x_positions) if x_positions else 0.0
        x_mean = (
            sum(float(x) for x in x_positions) / len(x_positions)
            if x_positions
            else x_key
        )
        placements[tid] = _PaletteCladePlacement(
            component_roots=component_roots,
            leaf_count=len(target_leaf_ids),
            x_key=x_key,
            x_mean=x_mean,
        )
    return placements


def _build_palette_breadth_first_order(
    palette: Palette,
    placements: dict[int, _PaletteCladePlacement],
    ncbi,
) -> list[int]:
    """Order palette clades from broad to fine using taxonomy, not exact tree labels."""
    present = set(placements)
    if not present:
        return []
    palette_children: dict[int, list[int]] = {tid: [] for tid in present}
    roots: list[int] = []
    for tid in present:
        palette_parent = next(
            (ancestor for ancestor in _lineage_via_ncbi(tid, ncbi)[1:] if ancestor in present),
            None,
        )
        if palette_parent is None:
            roots.append(tid)
        else:
            palette_children[palette_parent].append(tid)

    roots.sort(key=lambda tid: placements[tid].x_key)
    for tid in palette_children:
        palette_children[tid].sort(key=lambda child_tid: placements[child_tid].x_key)

    order: list[int] = []
    queue: deque[int] = deque(roots)
    while queue:
        current = queue.popleft()
        order.append(current)
        queue.extend(palette_children[current])
    return order


def _overlay_palette_subtree(ax, tax_tree, root_taxid: int, color: str, *, alpha: float, linewidth: float) -> None:
    """Overlay one connected subtree on the existing full-tree geometry."""
    descendants = _collect_taxid_subtree(tax_tree, root_taxid)
    for node_id in descendants:
        node = tax_tree.nodes[node_id]
        if len(node.children) > 1:
            child_xs = [
                tax_tree.nodes[child].x
                for child in node.children
                if child in descendants and getattr(tax_tree.nodes[child], "x", None) is not None
            ]
            if len(child_xs) > 1:
                ax.plot(
                    [min(child_xs), max(child_xs)],
                    [node.nodeage, node.nodeage],
                    color=color,
                    linewidth=linewidth,
                    alpha=alpha,
                    solid_capstyle="butt",
                    zorder=4,
                )
        if node_id == root_taxid:
            continue
        parent_id = node.parent
        if parent_id is None or parent_id == -1 or parent_id not in descendants:
            continue
        ax.plot(
            [node.x, node.x],
            [node.nodeage, tax_tree.nodes[parent_id].nodeage],
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            solid_capstyle="butt",
            zorder=4,
        )


def _overlay_palette_components(
    ax,
    tax_tree,
    component_roots: list[int],
    color: str,
    *,
    alpha: float,
    linewidth: float,
) -> None:
    """Overlay all disconnected components for one palette clade."""
    for root_taxid in component_roots:
        _overlay_palette_subtree(
            ax,
            tax_tree,
            root_taxid,
            color,
            alpha=alpha,
            linewidth=linewidth,
        )


def _compute_tax_tree_x_positions(tax_tree, *, reverse: bool = False) -> list[int]:
    """Populate ``node.x`` using the same ascending leaf order as the tree plot."""
    if hasattr(tax_tree, "sort_nodes"):
        tax_tree.sort_nodes("ascending")
        ordered_leaf_ids = list(getattr(tax_tree, "leaf_order", []))
    else:
        ordered_leaf_ids = sorted(
            node_id
            for node_id, node in tax_tree.nodes.items()
            if len(node.children) == 0
        )
    if reverse:
        ordered_leaf_ids = list(reversed(ordered_leaf_ids))
    for i, node_id in enumerate(ordered_leaf_ids):
        tax_tree.nodes[node_id].x = float(i)

    for node_id in _tax_tree_postorder_nodes(tax_tree):
        node = tax_tree.nodes[node_id]
        if len(node.children) == 0:
            continue
        child_xs = [
            tax_tree.nodes[child_id].x
            for child_id in node.children
            if getattr(tax_tree.nodes[child_id], "x", None) is not None
        ]
        if not child_xs:
            continue
        if len(child_xs) == 1:
            node.x = float(child_xs[0])
        else:
            node.x = (min(child_xs) + max(child_xs)) / 2.0
    return ordered_leaf_ids


def _compute_circular_tax_tree_layout(
    tax_tree,
    *,
    inner_radius: float,
    ring_fraction_of_inner: float = 0.5,
    reverse_order: bool = False,
    start_gap_fraction: float = 0.0,
    end_gap_fraction: float = 0.1,
) -> tuple[list[int], dict[int, float], dict[int, float], float, float]:
    """Return per-node polar coordinates for the donut tree renderer.

    The rectilinear tree is mapped directly into polar coordinates:
    leaf order -> angle and node age -> radius.

    The tree ring thickness is controlled relative to the inner white-circle
    radius via ``ring_fraction_of_inner``.
    """
    total_gap_fraction = max(0.0, min(0.95, float(start_gap_fraction) + float(end_gap_fraction)))
    ordered_leaf_ids = _compute_tax_tree_x_positions(tax_tree, reverse=reverse_order)
    leaf_count = max(len(ordered_leaf_ids), 1)
    available_sweep = (2.0 * math.pi) * (1.0 - total_gap_fraction)
    theta_step = available_sweep / float(leaf_count)
    theta_start = (2.0 * math.pi) * max(0.0, float(start_gap_fraction))
    radial_span = inner_radius * float(ring_fraction_of_inner)
    max_age = max(
        float(node.nodeage)
        for node in tax_tree.nodes.values()
        if getattr(node, "nodeage", None) is not None
    )
    if max_age <= 0:
        max_age = 1.0

    theta_by_node: dict[int, float] = {}
    radius_by_node: dict[int, float] = {}
    for node_id, node in tax_tree.nodes.items():
        x = float(getattr(node, "x", 0.0) or 0.0)
        theta_by_node[node_id] = theta_start + theta_step * (x + 0.5)
        node_age = float(getattr(node, "nodeage", 0.0) or 0.0)
        radius_by_node[node_id] = inner_radius + ((max_age - node_age) / max_age) * radial_span
    return ordered_leaf_ids, theta_by_node, radius_by_node, theta_start, available_sweep


def _theta_for_leaf_position(
    x_value: float,
    leaf_count: int,
    *,
    theta_start: float,
    available_sweep: float,
) -> float:
    """Map a leaf-order x position onto the circular angular sweep."""
    return theta_start + ((float(x_value) + 0.5) * available_sweep / float(max(leaf_count, 1)))


def _circular_mean(angles: list[float], weights: list[float] | None = None) -> float:
    """Return the weighted circular mean angle in radians."""
    if not angles:
        return 0.0
    if weights is None:
        weights = [1.0] * len(angles)
    sum_sin = sum(float(w) * math.sin(float(a)) for a, w in zip(angles, weights))
    sum_cos = sum(float(w) * math.cos(float(a)) for a, w in zip(angles, weights))
    return math.atan2(sum_sin, sum_cos) % (2.0 * math.pi)


def _circular_distance(a: float, b: float) -> float:
    """Smallest signed angular distance between two angles in radians."""
    return ((float(a) - float(b) + math.pi) % (2.0 * math.pi)) - math.pi


def _quantized_age_ticks(max_age: float, target_count: int = 6) -> list[float]:
    """Return sensible MYA ticks snapped to 50/100-style values."""
    max_age = float(max(max_age, 1.0))
    raw_step = max_age / max(float(target_count - 1), 1.0)
    candidates = [25.0, 50.0, 100.0, 200.0, 250.0, 500.0]
    step = next((cand for cand in candidates if cand >= raw_step), candidates[-1])
    tick_max = math.floor(max_age / step) * step
    if tick_max < step:
        tick_max = step
    ticks = [0.0]
    current = step
    while current <= tick_max + 1e-9:
        ticks.append(float(current))
        current += step
    return ticks


def _draw_circular_age_axis(
    ax,
    *,
    inner_radius: float,
    outer_radius: float,
    max_age: float,
    label_radius: float,
    tick_fontsize: float,
    title_fontsize: float,
    linewidth: float,
    tick_count: int = 6,
) -> None:
    """Draw a MYA guide axis at 12 o'clock inside the circular-gap region."""
    import numpy as np

    axis_theta = 0.0
    ax.plot(
        [axis_theta, axis_theta],
        [inner_radius, outer_radius],
        color="#333333",
        linewidth=linewidth,
        alpha=0.9,
        solid_capstyle="butt",
        zorder=6,
    )

    tick_half_angle = math.radians(0.9)
    label_theta = -math.radians(1.6)
    ticks = _quantized_age_ticks(max_age, target_count=tick_count)
    for age in ticks:
        radius = inner_radius + ((max_age - float(age)) / max(max_age, 1e-9)) * (outer_radius - inner_radius)
        _plot_polar_arc(
            ax,
            axis_theta - tick_half_angle,
            axis_theta,
            radius,
            color="#333333",
            linewidth=linewidth,
            alpha=0.9,
            zorder=6,
        )
        label = f"{int(round(float(age)))}"
        ax.text(
            label_theta,
            radius,
            label,
            color="#333333",
            fontsize=tick_fontsize,
            ha="right",
            va="center",
            rotation=0,
            clip_on=False,
            zorder=7,
        )

    ax.text(
        label_theta,
        label_radius + 0.012,
        "MYA",
        color="#333333",
        fontsize=title_fontsize,
        ha="right",
        va="bottom",
        rotation=0,
        clip_on=False,
        zorder=7,
    )


def _compute_center_umap_inset_fraction(
    *,
    inner_radius: float,
    total_radius: float,
    safety_fraction: float = 0.96,
    max_fraction: float = 0.92,
) -> float:
    """Return a conservative inset size that stays inside the donut hole."""
    if total_radius <= 0:
        return 0.0
    hole_fraction = float(inner_radius) / float(total_radius)
    return min(float(max_fraction), float(safety_fraction) * hole_fraction)


def _load_circular_umap_overlay(umap_df_path: Path, palette: Palette):
    """Load a recolored UMAP dataframe and resolve palette clades for each point."""
    import pandas as pd

    df = pd.read_csv(umap_df_path, sep="\t")
    required = {"UMAP1", "UMAP2", "taxid_list_str"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"UMAP dataframe missing required columns: {', '.join(sorted(missing))}"
        )

    scaffold_labels = {
        "cellular organisms",
        "eukaryota",
        "opisthokonta",
        "metazoa",
        "eumetazoa",
        "bilateria",
        "protostomia",
        "deuterostomia",
        "spiralia",
        "ecdysozoa",
        "panarthropoda",
    }

    rows: list[tuple[float, float, int | None, int | None, str]] = []
    for row in df.itertuples(index=False):
        lineage_str = getattr(row, "taxid_list_str", "")
        lineage_taxids = [
            int(chunk)
            for chunk in str(lineage_str).split(";")
            if str(chunk).strip()
        ]
        clade = palette.for_lineage_string(lineage_str)
        anchor_taxid = None
        for raw_taxid in lineage_taxids:
            canonical_taxid = palette.canonicalize_taxid(raw_taxid)
            if canonical_taxid not in palette.by_taxid:
                continue
            label = str(palette.by_taxid[canonical_taxid].label).strip().lower()
            if label in scaffold_labels:
                continue
            anchor_taxid = canonical_taxid
            break
        if anchor_taxid is None:
            anchor_taxid = clade.taxid
        rows.append(
            (
                float(getattr(row, "UMAP1")),
                float(getattr(row, "UMAP2")),
                clade.taxid,
                anchor_taxid,
                clade.color,
            )
        )
    return rows


def _compute_best_umap_rotation(
    umap_points: list[tuple[float, float, int | None, int | None, str]],
    tree_theta_by_taxid: dict[int, float],
) -> tuple[float, bool]:
    """Return a rotation angle and reflection flag for best tree/UMAP alignment."""
    centered = [
        (x, y, anchor_taxid, color)
        for x, y, _taxid, anchor_taxid, color in umap_points
        if anchor_taxid in tree_theta_by_taxid
    ]
    if not centered:
        return 0.0, False

    mean_x = sum(x for x, _y, _taxid, _color in centered) / len(centered)
    mean_y = sum(y for _x, y, _taxid, _color in centered) / len(centered)

    best_rotation = 0.0
    best_reflect = False
    best_error = float("inf")
    for reflect in (False, True):
        grouped: dict[int, list[float]] = {}
        for x, y, anchor_taxid, _color in centered:
            x_use = -(x - mean_x) if reflect else (x - mean_x)
            y_use = y - mean_y
            angle = math.atan2(y_use, x_use) % (2.0 * math.pi)
            grouped.setdefault(int(anchor_taxid), []).append(angle)

        diffs: list[float] = []
        weights: list[float] = []
        group_angles: dict[int, float] = {}
        for taxid, angles in grouped.items():
            if taxid not in tree_theta_by_taxid or not angles:
                continue
            mean_angle = _circular_mean(angles)
            group_angles[taxid] = mean_angle
            diffs.append((tree_theta_by_taxid[taxid] - mean_angle) % (2.0 * math.pi))
            weights.append(math.sqrt(float(len(angles))))
        if not diffs:
            continue

        rotation = _circular_mean(diffs, weights=weights)
        error = 0.0
        for taxid, mean_angle in group_angles.items():
            weight = math.sqrt(float(len(grouped[taxid])))
            error += weight * (_circular_distance((mean_angle + rotation) % (2.0 * math.pi), tree_theta_by_taxid[taxid]) ** 2)
        if error < best_error:
            best_error = error
            best_rotation = rotation
            best_reflect = reflect
    return best_rotation, best_reflect


def _draw_center_umap_overlay(
    fig,
    *,
    umap_df_path: Path,
    palette: Palette,
    tree_theta_by_taxid: dict[int, float],
    inner_radius: float,
    total_radius: float,
) -> None:
    """Draw a rotated UMAP scatter inside the donut's white center."""
    from matplotlib.patches import Circle
    import numpy as np

    umap_points = _load_circular_umap_overlay(umap_df_path, palette)
    if not umap_points:
        return

    rotation, reflect = _compute_best_umap_rotation(umap_points, tree_theta_by_taxid)
    xs = np.array([p[0] for p in umap_points], dtype=float)
    ys = np.array([p[1] for p in umap_points], dtype=float)
    colors = [p[4] for p in umap_points]

    xs = xs - xs.mean()
    ys = ys - ys.mean()
    if reflect:
        xs = -xs
    cos_a = math.cos(rotation)
    sin_a = math.sin(rotation)
    x_rot = cos_a * xs - sin_a * ys
    y_rot = sin_a * xs + cos_a * ys
    max_norm = float(np.sqrt((x_rot**2) + (y_rot**2)).max()) if len(x_rot) else 1.0
    scale = 0.96 / max(max_norm, 1e-9)
    x_plot = x_rot * scale
    y_plot = y_rot * scale

    inset_fraction = _compute_center_umap_inset_fraction(
        inner_radius=inner_radius,
        total_radius=total_radius,
    )
    inset_left = 0.5 - inset_fraction / 2.0
    inset_ax = fig.add_axes([inset_left, inset_left, inset_fraction, inset_fraction], zorder=2)
    inset_ax.set_aspect("equal")
    inset_ax.set_facecolor("none")
    clip_circle = Circle((0.5, 0.5), 0.5, transform=inset_ax.transAxes, facecolor="white", edgecolor="none")
    inset_ax.add_patch(clip_circle)
    points = inset_ax.scatter(
        x_plot,
        y_plot,
        s=5 * 0.66,
        c=colors,
        edgecolors="none",
        alpha=0.75,
        zorder=3,
    )
    points.set_clip_path(clip_circle)
    inset_ax.set_xlim(-1.02, 1.02)
    inset_ax.set_ylim(-1.02, 1.02)
    inset_ax.axis("off")


def _plot_polar_arc(
    ax,
    theta_start: float,
    theta_end: float,
    radius: float,
    *,
    color: str,
    linewidth: float,
    alpha: float,
    zorder: int,
) -> None:
    """Draw a constant-radius arc on a polar axis."""
    if theta_end < theta_start:
        theta_start, theta_end = theta_end, theta_start
    if abs(theta_end - theta_start) < 1e-9:
        return

    import numpy as np

    theta = np.linspace(theta_start, theta_end, 64)
    radius_values = np.full_like(theta, fill_value=radius, dtype=float)
    ax.plot(
        theta,
        radius_values,
        color=color,
        linewidth=linewidth,
        alpha=alpha,
        solid_capstyle="butt",
        zorder=zorder,
    )


def _draw_circular_tree_base(
    ax,
    tax_tree,
    theta_by_node: dict[int, float],
    radius_by_node: dict[int, float],
    *,
    color: str,
    alpha: float,
    linewidth: float,
) -> None:
    """Draw the full tree in donut form using arcs and radial segments."""
    for node_id in _tax_tree_postorder_nodes(tax_tree):
        node = tax_tree.nodes[node_id]
        if len(node.children) > 1:
            child_thetas = [
                theta_by_node[child_id]
                for child_id in node.children
                if child_id in theta_by_node
            ]
            if len(child_thetas) > 1:
                _plot_polar_arc(
                    ax,
                    min(child_thetas),
                    max(child_thetas),
                    radius_by_node[node_id],
                    color=color,
                    linewidth=linewidth,
                    alpha=alpha,
                    zorder=1,
                )

        parent_id = getattr(node, "parent", None)
        if parent_id is None or parent_id == -1 or parent_id not in radius_by_node:
            continue
        ax.plot(
            [theta_by_node[node_id], theta_by_node[node_id]],
            [radius_by_node[parent_id], radius_by_node[node_id]],
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            solid_capstyle="butt",
            zorder=1,
        )


def _overlay_palette_components_circular(
    ax,
    tax_tree,
    component_roots: list[int],
    theta_by_node: dict[int, float],
    radius_by_node: dict[int, float],
    *,
    color: str,
    alpha: float,
    linewidth: float,
) -> None:
    """Overlay all disconnected components for one palette clade in donut form."""
    for root_taxid in component_roots:
        descendants = _collect_taxid_subtree(tax_tree, root_taxid)
        for node_id in descendants:
            node = tax_tree.nodes[node_id]
            if len(node.children) > 1:
                child_thetas = [
                    theta_by_node[child_id]
                    for child_id in node.children
                    if child_id in descendants and child_id in theta_by_node
                ]
                if len(child_thetas) > 1:
                    _plot_polar_arc(
                        ax,
                        min(child_thetas),
                        max(child_thetas),
                        radius_by_node[node_id],
                        color=color,
                        linewidth=linewidth,
                        alpha=alpha,
                        zorder=4,
                    )

            if node_id == root_taxid:
                continue
            parent_id = getattr(node, "parent", None)
            if parent_id is None or parent_id == -1 or parent_id not in descendants:
                continue
            ax.plot(
                [theta_by_node[node_id], theta_by_node[node_id]],
                [radius_by_node[parent_id], radius_by_node[node_id]],
                color=color,
                linewidth=linewidth,
                alpha=alpha,
                solid_capstyle="butt",
                zorder=4,
            )


def _render_circular_nested_tree_coloring(
    *,
    source_tree,
    palette: Palette,
    output_path: Path,
    title: str | None = None,
    show_labels: bool = False,
    umap_df_path: Path | None = None,
    font_size_pt: float = 7.0,
    target_width_mm: float | None = None,
    min_linewidth_pt: float = 0.5,
    emit_layer_pdfs: bool = False,
) -> tuple[int, int]:
    """Render the nested tree coloring as a donut-shaped PDF preview."""
    from ete4 import NCBITaxa  # type: ignore
    import matplotlib
    matplotlib.use("Agg")
    _configure_vector_font_output(matplotlib)
    import matplotlib.pyplot as plt

    ncbi = NCBITaxa()
    tax_tree = _build_taxidtree_from_source_tree(source_tree, ncbi)
    leaf_count = len([tid for tid in tax_tree.nodes if len(tax_tree.nodes[tid].children) == 0])
    start_gap_fraction = 0.0005
    end_gap_fraction = 0.025
    inner_radius = 0.68
    ordered_leaf_ids, theta_by_node, radius_by_node, theta_start, available_sweep = _compute_circular_tax_tree_layout(
        tax_tree,
        inner_radius=inner_radius,
        ring_fraction_of_inner=0.2,
        reverse_order=True,
        start_gap_fraction=start_gap_fraction,
        end_gap_fraction=end_gap_fraction,
    )
    placements = _resolve_palette_clade_placements(
        tax_tree,
        palette,
        ncbi,
        ordered_leaf_ids=ordered_leaf_ids,
    )
    palette_order = _build_palette_breadth_first_order(palette, placements, ncbi)
    if not palette_order:
        raise SystemExit("No palette taxids from the YAML were found in the tree.")

    outer_radius = max(radius_by_node.values(), default=inner_radius)
    label_radius = outer_radius + (0.004 if show_labels else 0.03)
    base_linewidth = max(float(min_linewidth_pt), 0.5)
    overlay_linewidth = max(float(min_linewidth_pt), 0.6)
    axis_linewidth = max(float(min_linewidth_pt), 0.55)
    label_fontsize = float(font_size_pt)
    axis_fontsize = float(font_size_pt)
    title_fontsize = max(float(font_size_pt), 5.0)
    max_age = max(
        float(node.nodeage)
        for node in tax_tree.nodes.values()
        if getattr(node, "nodeage", None) is not None
    )

    fig_size = 16.0 if show_labels else 13.0
    tree_theta_by_taxid = {
        tid: _theta_for_leaf_position(
            placements[tid].x_mean,
            len(ordered_leaf_ids),
            theta_start=theta_start,
            available_sweep=available_sweep,
        )
        for tid in palette_order
    }

    def _draw_labels(ax) -> None:
        label_tids = sorted(palette_order, key=lambda tid: placements[tid].x_mean)
        for tid in label_tids:
            theta = tree_theta_by_taxid[tid]
            theta_deg = math.degrees(theta) % 360.0
            rotation = ((90.0 - theta_deg + 180.0) % 360.0) - 180.0
            ha = "left"
            if 180.0 < theta_deg < 360.0:
                rotation = ((rotation + 180.0 + 180.0) % 360.0) - 180.0
                ha = "right"
            ax.text(
                theta,
                label_radius,
                palette.by_taxid[tid].label,
                color=palette.by_taxid[tid].color,
                rotation=rotation,
                rotation_mode="anchor",
                ha=ha,
                va="center",
                fontsize=label_fontsize,
                clip_on=False,
            )

    def _make_figure(
        *,
        draw_tree: bool,
        draw_labels_layer: bool,
        draw_umap_layer: bool,
        draw_title: bool,
        figsize_override=None,
    ):
        current_figsize = figsize_override or (fig_size, fig_size)
        fig = plt.figure(figsize=current_figsize)
        ax = fig.add_subplot(111, projection="polar")
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_facecolor("none")
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        if "polar" in ax.spines:
            ax.spines["polar"].set_visible(False)

        if draw_tree:
            _draw_circular_tree_base(
                ax,
                tax_tree,
                theta_by_node,
                radius_by_node,
                color="#b5b5b5",
                alpha=0.75,
                linewidth=base_linewidth,
            )

            for tid in palette_order:
                _overlay_palette_components_circular(
                    ax,
                    tax_tree,
                    placements[tid].component_roots,
                    theta_by_node,
                    radius_by_node,
                    color=palette.by_taxid[tid].color,
                    alpha=0.75,
                    linewidth=overlay_linewidth,
                )

            _draw_circular_age_axis(
                ax,
                inner_radius=inner_radius,
                outer_radius=outer_radius,
                max_age=max_age,
                label_radius=label_radius,
                tick_fontsize=axis_fontsize,
                title_fontsize=title_fontsize,
                linewidth=axis_linewidth,
            )

        if draw_umap_layer and umap_df_path is not None:
            _draw_center_umap_overlay(
                fig,
                umap_df_path=umap_df_path,
                palette=palette,
                tree_theta_by_taxid=tree_theta_by_taxid,
                inner_radius=inner_radius,
                total_radius=label_radius + 0.06,
            )

        if draw_labels_layer and show_labels:
            _draw_labels(ax)

        ax.set_ylim(0.0, label_radius + 0.06)
        if draw_title and title:
            ax.set_title(title, fontsize=max(label_fontsize + 1.0, 6.0), va="bottom")
        fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
        return fig, ax

    fig, _ax = _make_figure(
        draw_tree=True,
        draw_labels_layer=show_labels,
        draw_umap_layer=(umap_df_path is not None),
        draw_title=True,
    )
    _scale_figure_to_target_width(fig, target_width_mm=target_width_mm)
    fig.canvas.draw()
    tight_bbox = fig.get_tightbbox(fig.canvas.get_renderer())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches=tight_bbox, pad_inches=0.0)

    if emit_layer_pdfs:
        layer_figsize = tuple(float(v) for v in fig.get_size_inches())
        sidecars = [
            (_sidecar_pdf_path(output_path, "tree"), True, False, False),
        ]
        if show_labels:
            sidecars.append((_sidecar_pdf_path(output_path, "labels"), False, True, False))
        if umap_df_path is not None:
            sidecars.append((_sidecar_pdf_path(output_path, "umap"), False, False, True))

        for sidecar_path, draw_tree, draw_labels_layer, draw_umap_layer in sidecars:
            layer_fig, _layer_ax = _make_figure(
                draw_tree=draw_tree,
                draw_labels_layer=draw_labels_layer,
                draw_umap_layer=draw_umap_layer,
                draw_title=False,
                figsize_override=layer_figsize,
            )
            sidecar_path.parent.mkdir(parents=True, exist_ok=True)
            layer_fig.savefig(
                sidecar_path,
                dpi=200,
                bbox_inches=tight_bbox,
                pad_inches=0.0,
                transparent=True,
            )
            print(f"[palette-preview] wrote {sidecar_path}", file=sys.stderr)
            plt.close(layer_fig)

    plt.close(fig)
    return leaf_count, len(placements)


def _render_nested_tree_coloring(
    *,
    source_tree,
    palette: Palette,
    output_path: Path,
    title: str | None = None,
    show_labels: bool = False,
    font_size_pt: float = 6.0,
    target_width_mm: float | None = None,
    nested_height_scale: float = 1.0,
) -> tuple[int, int]:
    """Render the full tree topology, recoloring branches breadth-first by palette."""
    from ete4 import NCBITaxa  # type: ignore
    import matplotlib
    matplotlib.use("Agg")
    _configure_vector_font_output(matplotlib)
    import matplotlib.pyplot as plt
    from matplotlib.transforms import blended_transform_factory

    ncbi = NCBITaxa()
    tax_tree = _build_taxidtree_from_source_tree(source_tree, ncbi)
    leaf_count = len([tid for tid in tax_tree.nodes if len(tax_tree.nodes[tid].children) == 0])
    placements = _resolve_palette_clade_placements(tax_tree, palette, ncbi)
    palette_order = _build_palette_breadth_first_order(palette, placements, ncbi)
    if not palette_order:
        raise SystemExit("No palette taxids from the YAML were found in the tree.")

    width = max(18.0, min(42.0, 0.0045 * leaf_count))
    height_scale = max(float(nested_height_scale), 0.05)
    base_fontsize = float(font_size_pt)
    if show_labels:
        height = 11.0 * height_scale
        fig = plt.figure(figsize=(width, height))
        gs = fig.add_gridspec(2, 1, height_ratios=[9.85, 0.15], hspace=0.0)
        ax = fig.add_subplot(gs[0, 0])
        label_ax = fig.add_subplot(gs[1, 0], sharex=ax)
    else:
        height = 10.0 * height_scale
        fig, ax = plt.subplots(1, 1, figsize=(width, height))
        label_ax = None
    ax = tax_tree.plot_tree(
        ax,
        sort="ascending",
        draw_horizontal_bars=True,
        text_older_than=1e12,
        lw_standard=0.3,
    )
    for line in ax.lines:
        line.set_alpha(0.75)
        if line.get_color() == "grey":
            line.set_color("#b5b5b5")

    for tid in palette_order:
        _overlay_palette_components(
            ax,
            tax_tree,
            placements[tid].component_roots,
            palette.by_taxid[tid].color,
            alpha=0.75,
            linewidth=0.55,
        )

    ax.set_xticks([])
    ax.set_xlabel("")
    ax.set_ylabel("MYA", fontsize=base_fontsize)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    if title:
        ax.set_title(title, fontsize=max(base_fontsize + 1.0, 7.0))

    if show_labels and label_ax is not None:
        label_tids = sorted(palette_order, key=lambda tid: placements[tid].x_mean)
        label_ax.set_xlim(ax.get_xlim())
        label_ax.set_ylim(0.0, 1.0)
        label_ax.set_yticks([])
        label_ax.set_xticks([])
        label_ax.set_xlabel("")
        text_transform = blended_transform_factory(label_ax.transData, label_ax.transAxes)
        for tid in label_tids:
            label_ax.text(
                placements[tid].x_mean,
                0.995,
                palette.by_taxid[tid].label,
                color=palette.by_taxid[tid].color,
                rotation=270,
                rotation_mode="anchor",
                ha="left",
                va="top",
                fontsize=base_fontsize,
                transform=text_transform,
                clip_on=False,
            )
        label_ax.set_axis_off()

    plt.tight_layout()
    _scale_figure_to_target_width(fig, target_width_mm=target_width_mm)
    fig.canvas.draw()
    tight_bbox = fig.get_tightbbox(fig.canvas.get_renderer())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches=tight_bbox, pad_inches=0.0)
    plt.close(fig)
    return leaf_count, len(placements)


# ---------- core rendering ----------

def render_palette_preview(
    *,
    tree_path: Path,
    palette: Palette,
    output_path: Path,
    title: str | None = None,
    max_leaves: int | None = None,
    show_labels: bool = False,
    align_tips: bool = False,
    layout: str = "rectilinear",
    colored_newick_path: Path | None = None,
    nexus_path: Path | None = None,
    collapsed_newick_path: Path | None = None,
    collapsed_nexus_path: Path | None = None,
    collapse_dominance: float = 1.0,
    umap_df_path: Path | None = None,
    font_size_pt: float = 7.0,
    target_width_mm: float | None = None,
    min_linewidth_pt: float = 0.5,
    emit_layer_pdfs: bool = False,
    nested_height_scale: float = 1.0,
) -> None:
    """Render a tree with leaves colored by palette + clade-color legend."""
    # Heavy imports kept inside the function so --help is cheap.
    from ete4 import Tree  # type: ignore
    try:
        from ete4 import NCBITaxa  # type: ignore
        ncbi = NCBITaxa()
    except Exception:
        ncbi = None  # we'll fall back to taxid-in-label extraction only

    import matplotlib
    matplotlib.use("Agg")
    _configure_vector_font_output(matplotlib)
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Polygon

    t = Tree(str(tree_path), parser=1) if hasattr(Tree, "__call__") else Tree(open(tree_path).read())
    # ete4's Tree() accepts a newick string or a path in different versions;
    # the above two-step dance is resilient.

    if layout == "nested-tree-coloring":
        leaf_count, clade_count = _render_nested_tree_coloring(
            source_tree=t,
            palette=palette,
            output_path=output_path,
            title=title,
            show_labels=show_labels,
            font_size_pt=font_size_pt,
            target_width_mm=target_width_mm,
            nested_height_scale=nested_height_scale,
        )
        print(f"[palette-preview] wrote {output_path}", file=sys.stderr)
        print(f"  leaves rendered: {leaf_count}", file=sys.stderr)
        print(f"  palette entries in tree: {clade_count}/{len(palette)}", file=sys.stderr)
        print(f"  layout: {layout}", file=sys.stderr)
        if any(path is not None for path in (colored_newick_path, nexus_path, collapsed_newick_path, collapsed_nexus_path)):
            print("  side exports skipped for nested-tree-coloring layout", file=sys.stderr)
        return
    if layout == "circular-nested-tree-coloring":
        leaf_count, clade_count = _render_circular_nested_tree_coloring(
            source_tree=t,
            palette=palette,
            output_path=output_path,
            title=title,
            show_labels=show_labels,
            umap_df_path=umap_df_path,
            font_size_pt=font_size_pt,
            target_width_mm=target_width_mm,
            min_linewidth_pt=min_linewidth_pt,
            emit_layer_pdfs=emit_layer_pdfs,
        )
        print(f"[palette-preview] wrote {output_path}", file=sys.stderr)
        print(f"  leaves rendered: {leaf_count}", file=sys.stderr)
        print(f"  palette entries in tree: {clade_count}/{len(palette)}", file=sys.stderr)
        print(f"  layout: {layout}", file=sys.stderr)
        if any(path is not None for path in (colored_newick_path, nexus_path, collapsed_newick_path, collapsed_nexus_path)):
            print("  side exports skipped for circular-nested-tree-coloring layout", file=sys.stderr)
        return

    # Walk the tree, compute per-leaf coordinates + color.
    leaves = list(t.leaves())
    if max_leaves is not None and len(leaves) > max_leaves:
        leaves = leaves[:max_leaves]

    if not leaves:
        raise SystemExit(f"Tree at {tree_path} has no leaves.")

    # Palette resolution per leaf: prefer taxid in the label; else use ete4
    # NCBITaxa to resolve a species name.
    def _resolve_color(leaf) -> str:
        tid = _extract_taxid(leaf.name)
        if tid is None and ncbi is not None:
            try:
                name_hits = ncbi.get_name_translator([leaf.name.replace("_", " ")])
                if leaf.name.replace("_", " ") in name_hits:
                    tid = name_hits[leaf.name.replace("_", " ")][0]
            except Exception:
                tid = None
        if tid is None:
            return palette.fallback.color
        lineage = _lineage_via_ncbi(tid, ncbi) if ncbi is not None else [tid]
        return palette.for_lineage(lineage).color

    leaf_colors = {leaf: _resolve_color(leaf) for leaf in leaves}
    render_tree = t
    render_leaves = leaves
    render_leaf_colors = leaf_colors
    collapsed_meta: dict[int, dict[str, object]] = {}

    if layout == "triangle":
        node_xy, leaf_label_xy = _compute_triangle_layout(render_tree, render_leaves)
        node_spans = {leaf: (xy[1], xy[1]) for leaf, xy in leaf_label_xy.items()}
    elif layout == "collapsed-tips":
        render_tree, render_leaves, render_leaf_colors, collapsed_meta = _collapse_tree_copy(
            t,
            leaves,
            leaf_colors,
            palette,
            dominance=collapse_dominance,
        )
        leaf_weights = {
            leaf: float(collapsed_meta.get(id(leaf), {}).get("leaf_weight", 1.0))
            for leaf in render_leaves
        }
        node_xy, node_spans = _compute_weighted_rectilinear_layout(
            render_tree,
            render_leaves,
            align_tips=align_tips,
            leaf_weights=leaf_weights,
        )
        leaf_label_xy = {leaf: node_xy[leaf] for leaf in render_leaves}
    else:
        node_xy = _compute_rectilinear_layout(render_tree, render_leaves, align_tips=align_tips)
        leaf_label_xy = {leaf: node_xy[leaf] for leaf in render_leaves}
        node_spans = {leaf: (xy[1], xy[1]) for leaf, xy in leaf_label_xy.items()}

    # Figure.
    h = max(6, 0.04 * len(leaves))
    fig, (ax_tree, ax_legend) = plt.subplots(
        1, 2, figsize=(14, h), gridspec_kw={"width_ratios": [4, 1]}
    )

    if layout == "triangle":
        _draw_triangle_edges(ax_tree, render_tree, node_xy)
    elif layout == "collapsed-tips":
        _draw_rectilinear_edges_with_collapsed_tips(ax_tree, render_tree, node_xy, collapsed_meta)
    else:
        _draw_rectilinear_edges(ax_tree, render_tree, node_xy)

    # Leaf markers.
    for leaf, (x, y) in node_xy.items():
        if leaf not in leaf_label_xy:
            continue
        if layout == "collapsed-tips" and id(leaf) in collapsed_meta:
            y0, y1 = node_spans[leaf]
            if leaf.up is not None and leaf.up in node_xy:
                px, _py = node_xy[leaf.up]
                base_x = px + float(collapsed_meta[id(leaf)]["orig_dist"])
            else:
                base_x = x
            tri = Polygon(
                [(base_x, y0), (base_x, y1), (x, y)],
                closed=True,
                facecolor=render_leaf_colors[leaf],
                edgecolor="none",
                alpha=0.9,
                zorder=3,
            )
            ax_tree.add_patch(tri)
            if show_labels:
                ax_tree.text(x + 1, y, leaf.name, fontsize=4, va="center")
        else:
            ax_tree.scatter([x], [y], s=12, color=render_leaf_colors[leaf], edgecolors="none", zorder=3)
            if show_labels and layout != "triangle":
                ax_tree.text(x + 1, y, leaf.name, fontsize=4, va="center")

    if layout == "triangle":
        ax_tree.set_yticks([])
        if show_labels:
            tick_positions = [leaf_label_xy[leaf][0] for leaf in leaves]
            tick_labels = [leaf.name for leaf in leaves]
            ax_tree.set_xticks(tick_positions)
            ax_tree.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=5)
            for tick, leaf in zip(ax_tree.get_xticklabels(), leaves):
                tick.set_color(leaf_colors[leaf])
        else:
            ax_tree.set_xticks([])
        ax_tree.set_xlabel("palette assignment order")
        ax_tree.set_ylabel("")
        ax_tree.spines["left"].set_visible(False)
        ax_tree.spines["top"].set_visible(False)
        ax_tree.spines["right"].set_visible(False)
        ax_tree.margins(x=0.01)
        ax_tree.set_ylim(max(y for _x, y in node_xy.values()) + 0.6, -0.4)
    else:
        ax_tree.invert_yaxis()
        ax_tree.set_yticks([])
        ax_tree.set_xlabel("cumulative branch length")
        ax_tree.spines["left"].set_visible(False)
        ax_tree.spines["top"].set_visible(False)
        ax_tree.spines["right"].set_visible(False)
    if title:
        ax_tree.set_title(title, fontsize=10)

    # Legend panel (swatches, ordered by frequency in the tree then by label).
    color_counts: dict[str, int] = {}
    color_to_label: dict[str, str] = {}
    for leaf in leaves:
        c = leaf_colors[leaf]
        color_counts[c] = color_counts.get(c, 0) + 1
    # Map each palette entry's color to a label for display.
    palette_colors = {cc.color: cc.label for _, cc in palette.items()}
    palette_colors[palette.fallback.color] = palette.fallback.label
    for color, count in sorted(color_counts.items(), key=lambda kv: (-kv[1], palette_colors.get(kv[0], ""))):
        color_to_label[color] = f"{palette_colors.get(color, '?')}  (n={count})"

    handles = [Patch(color=c, label=lab) for c, lab in color_to_label.items()]
    ax_legend.axis("off")
    ax_legend.legend(
        handles=handles,
        loc="center left",
        frameon=False,
        fontsize=7,
        title=f"Palette ({len(palette)} clades)\nsource: {palette.source_path or 'egt built-in'}",
        title_fontsize=7,
    )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close(fig)

    # Report summary.
    print(f"[palette-preview] wrote {output_path}", file=sys.stderr)
    print(f"  leaves rendered: {len(leaves)}", file=sys.stderr)
    print(f"  palette entries: {len(palette)}", file=sys.stderr)
    resolved = sum(1 for c in leaf_colors.values() if c != palette.fallback.color)
    print(f"  leaves resolved: {resolved}/{len(leaves)}  "
          f"({resolved * 100 // max(1, len(leaves))}%)", file=sys.stderr)
    if layout == "collapsed-tips":
        print(f"  collapsed tip count: {sum(1 for leaf in render_leaves if id(leaf) in collapsed_meta)}", file=sys.stderr)
    print(f"  layout: {layout}", file=sys.stderr)

    # Optional side-exports — FigTree-compatible.
    if colored_newick_path is not None:
        _write_colored_newick(t, leaves, leaf_colors, colored_newick_path)
        print(f"[palette-preview] wrote {colored_newick_path}", file=sys.stderr)

    if nexus_path is not None:
        _write_figtree_nexus(t, leaves, leaf_colors, nexus_path)
        print(f"[palette-preview] wrote {nexus_path}", file=sys.stderr)

    if collapsed_newick_path is not None or collapsed_nexus_path is not None:
        _write_collapsed_newick(
            t, leaves, leaf_colors, palette,
            collapsed_newick_path or Path("/tmp/_egt_collapsed_discard.nwk"),
            dominance=collapse_dominance,
            nexus_path=collapsed_nexus_path,
        )
        if collapsed_newick_path is not None:
            print(f"[palette-preview] wrote {collapsed_newick_path}  "
                  f"(dominance={collapse_dominance})", file=sys.stderr)
        if collapsed_nexus_path is not None:
            print(f"[palette-preview] wrote {collapsed_nexus_path}", file=sys.stderr)


def _build_color_label_lookup(palette: Palette) -> dict[str, str]:
    """Map each palette hex color to its clade label (first match wins)."""
    lut: dict[str, str] = {}
    for _, cc in palette.items():
        lut.setdefault(cc.color, cc.label)
    lut.setdefault(palette.fallback.color, palette.fallback.label)
    return lut


def _collapse_tree_copy(tree, leaves, leaf_colors, palette: Palette, dominance: float = 1.0):
    """Return a copied tree with dominant-color subtrees collapsed.

    Collapsed nodes remain as leaves in the copied tree and carry metadata
    needed for wedge-tip rendering.
    """
    from ete4 import Tree as ETree
    from collections import Counter

    work = ETree(tree.write(), parser=1)
    orig_leaves = {l.name: leaf_colors[l] for l in leaves}
    copy_leaves = list(work.leaves())
    copy_colors = {id(l): orig_leaves.get(l.name) for l in copy_leaves}
    color_to_label = _build_color_label_lookup(palette)
    collapsed_meta: dict[int, dict[str, object]] = {}

    def _dominant_color(leaf_descendants):
        cts = Counter(copy_colors.get(id(l)) for l in leaf_descendants)
        if not cts:
            return None, 0.0
        color, n = cts.most_common(1)[0]
        total = sum(cts.values())
        return color, (n / total if total else 0.0)

    collapsed_final_count: dict[str, int] = {}
    for node in list(work.traverse("postorder")):
        if node.is_leaf:
            continue
        leaf_descendants = list(node.leaves())
        if not leaf_descendants:
            continue
        color, frac = _dominant_color(leaf_descendants)
        if color is None or frac < dominance:
            continue
        parent = node.up
        if parent is not None:
            p_color, p_frac = _dominant_color(list(parent.leaves()))
            if p_color == color and p_frac >= dominance:
                continue
        label = color_to_label.get(color, "clade")
        collapsed_final_count[label] = collapsed_final_count.get(label, 0) + 1

    collapsed_counts: dict[str, int] = {}
    for node in list(work.traverse("postorder")):
        if node.is_leaf:
            continue
        leaf_descendants = list(node.leaves())
        if not leaf_descendants:
            continue
        color, frac = _dominant_color(leaf_descendants)
        if color is None or frac < dominance:
            continue

        label = color_to_label.get(color, "clade")
        dominant_leaves = [l for l in leaf_descendants if copy_colors.get(id(l)) == color]

        def _depth_from(n):
            d = 0.0
            cur = n
            while cur.up is not None and cur is not node:
                d += float(getattr(cur, "dist", 1.0) or 1.0)
                cur = cur.up
            return d

        orig_dist = float(node.dist or 0.0)
        tip_depth = max(_depth_from(l) for l in (dominant_leaves or leaf_descendants))
        collapsed_counts[label] = collapsed_counts.get(label, 0) + 1
        idx = collapsed_counts[label]
        is_only_one = collapsed_final_count.get(label, 0) == 1
        n_sp = len(dominant_leaves)
        n_total = len(leaf_descendants)
        size_str = f"n={n_sp}" if n_sp == n_total else f"n={n_sp}/{n_total}"
        node.name = (
            f"{label} ({size_str})"
            if is_only_one
            else f"{label} #{idx} ({size_str})"
        )
        for child in list(node.children):
            child.detach()
        node.dist = orig_dist + tip_depth
        copy_colors[id(node)] = color
        collapsed_meta[id(node)] = {
            "color": color,
            "label": label,
            "n_sp": n_sp,
            "n_total": n_total,
            "orig_dist": orig_dist,
            "tip_depth": tip_depth,
            "leaf_weight": max(1.0, float(n_total)),
        }

    render_leaves = list(work.leaves())
    render_leaf_colors = {
        leaf: copy_colors.get(id(leaf)) or palette.fallback.color
        for leaf in render_leaves
    }
    return work, render_leaves, render_leaf_colors, collapsed_meta


def _write_collapsed_newick(
    tree, leaves, leaf_colors, palette: Palette, out_path: Path,
    dominance: float = 1.0,
    nexus_path: Path | None = None,
) -> None:
    """Write a collapsed Newick: any subtree whose dominant color covers
    at least `dominance` fraction of its descendant leaves is merged into
    a single named tip with an inline color tag.

    `dominance=1.0` → strict monophyletic collapse (every leaf shares one color).
    `dominance=0.9` → collapse any subtree that is ≥90% one color, dropping
                      minority leaves. Useful for paraphyletic-ish groups.

    Mutates a COPY of the tree — not the one passed in.
    """
    work, work_leaves, work_leaf_colors, _collapsed_meta = _collapse_tree_copy(
        tree,
        leaves,
        leaf_colors,
        palette,
        dominance=dominance,
    )
    copy_colors = {id(leaf): work_leaf_colors[leaf] for leaf in work_leaves}

    # Serialise with inline color tags. Quote any label that contains
    # Newick-special characters so FigTree/iTOL parse correctly.
    def _quote_if_needed(name: str) -> str:
        if not name:
            return ""
        if any(c in name for c in "() ,;:[]'"):
            return "'" + name.replace("'", "''") + "'"
        return name

    def _to_newick(node) -> str:
        dist = float(getattr(node, "dist", 0.0) or 0.0)
        if node.is_leaf:
            name = _quote_if_needed(node.name or "")
            color = copy_colors.get(id(node))
            colortag = f"[&!color={color}]" if color else ""
            return f"{name}{colortag}:{dist:.6f}"
        name = _quote_if_needed(node.name or "")
        child_strs = ",".join(_to_newick(c) for c in node.children)
        return f"({child_strs}){name}:{dist:.6f}"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_to_newick(work) + ";\n", encoding="utf-8")

    # Optional companion NEXUS (FigTree tends to respect colors better from NEXUS).
    if nexus_path is not None:
        # Collect tip labels in traversal order.
        tips = [(_quote_if_needed(l.name or ""), copy_colors.get(id(l)))
                for l in work.leaves()]
        taxlabels = "\n  ".join(name for name, _ in tips)
        tree_str = _to_newick(work) + ";"
        nex = (
            "#NEXUS\n"
            "begin taxa;\n"
            f"  dimensions ntax={len(tips)};\n"
            "  taxlabels\n"
            f"  {taxlabels}\n"
            "  ;\n"
            "end;\n\n"
            "begin trees;\n"
            f"  tree tree_1 = [&R] {tree_str}\n"
            "end;\n"
        )
        nexus_path.parent.mkdir(parents=True, exist_ok=True)
        nexus_path.write_text(nex, encoding="utf-8")


def _write_colored_newick(tree, leaves, leaf_colors, out_path: Path) -> None:
    """Write a Newick with inline [&!color=#xxxxxx] tags on every leaf.

    FigTree and iTOL both read this format; leaves render with the given hex
    color. Writing from scratch keeps ete4 version quirks out of the way.
    """
    lut = {id(leaf): leaf_colors[leaf] for leaf in leaves}

    def _quote_if_needed(name: str) -> str:
        if not name:
            return ""
        if any(c in name for c in "() ,;:[]'"):
            return "'" + name.replace("'", "''") + "'"
        return name

    def _to_newick(node) -> str:
        dist = float(getattr(node, "dist", 0.0) or 0.0)
        if node.is_leaf:
            name = _quote_if_needed(node.name or "")
            color = lut.get(id(node))
            colortag = f"[&!color={color}]" if color else ""
            return f"{name}{colortag}:{dist:.6f}"
        name = _quote_if_needed(node.name or "")
        child_strs = ",".join(_to_newick(c) for c in node.children)
        return f"({child_strs}){name}:{dist:.6f}"

    nwk = _to_newick(tree) + ";"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(nwk + "\n", encoding="utf-8")


def _write_figtree_nexus(tree, leaves, leaf_colors, out_path: Path) -> None:
    """Write a NEXUS block that FigTree opens and displays with colors."""
    lut = {id(leaf): leaf_colors[leaf] for leaf in leaves}

    def _safe(name: str) -> str:
        # NEXUS taxon labels can't contain spaces or punctuation without quoting.
        bad = set("() ,;:")
        return "'" + name.replace("'", "''") + "'" if any(c in bad for c in name) else name

    def _to_newick(node) -> str:
        name = _safe(node.name or "")
        dist = float(getattr(node, "dist", 0.0) or 0.0)
        if node.is_leaf:
            color = lut.get(id(node), "#bfbfbf")
            return f"{name}[&!color={color}]:{dist:.6f}"
        child_strs = ",".join(_to_newick(c) for c in node.children)
        return f"({child_strs}){name}:{dist:.6f}"

    taxlabels = "\n  ".join(_safe(leaf.name or "") for leaf in leaves)
    tree_str = _to_newick(tree) + ";"

    nex = (
        "#NEXUS\n"
        "begin taxa;\n"
        f"  dimensions ntax={len(leaves)};\n"
        "  taxlabels\n"
        f"  {taxlabels}\n"
        "  ;\n"
        "end;\n\n"
        "begin trees;\n"
        f"  tree tree_1 = [&R] {tree_str}\n"
        "end;\n"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(nex, encoding="utf-8")


# ---------- CLI ----------

def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="egt palette-preview",
        description=(
            "Render a Newick tree colored by the paper palette for visual "
            "sign-off before retrofitting plotting CLIs."
        ),
    )
    parser.add_argument(
        "--tree",
        required=True,
        type=Path,
        help="Newick tree (calibrated or not). Labels with [taxid] brackets "
             "or trailing -taxid- are auto-detected.",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Output PDF path.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional title string for the figure.",
    )
    parser.add_argument(
        "--max-leaves",
        type=int,
        default=None,
        help="Cap on leaves rendered (for quick preview of large trees).",
    )
    parser.add_argument(
        "--show-labels",
        action="store_true",
        help="Print leaf labels next to the tips (default: off — labels are noisy at 4k leaves).",
    )
    parser.add_argument(
        "--align-tips",
        action="store_true",
        help="Render as an ultrametric phylogram (all tips at x=max). Right for calibrated trees.",
    )
    parser.add_argument(
        "--layout",
        choices=["rectilinear", "triangle", "collapsed-tips", "nested-tree-coloring", "circular-nested-tree-coloring"],
        default="rectilinear",
        help=(
            "Tree layout mode. 'rectilinear' is the current left-to-right phylogram. "
            "'triangle' ignores branch lengths and draws the whole tree as rooted nested "
            "triangles. 'collapsed-tips' keeps the tree rectilinear but replaces collapsed "
            "monophyletic palette groups with triangle wedges at the tips. "
            "'nested-tree-coloring' reuses the existing full-tree topology plot and "
            "recolors descendant branches breadth-first from broad palette clades to fine ones. "
            "'circular-nested-tree-coloring' maps that same rectilinear tree into polar coordinates "
            "as a donut with the root on the inner edge and tips on the outer edge."
        ),
    )
    parser.add_argument(
        "--emit-colored-newick",
        type=Path,
        default=None,
        help="Also write a Newick with inline [&!color=#xxxxxx] tags on every leaf. "
             "Opens in FigTree / iTOL with colors preserved.",
    )
    parser.add_argument(
        "--emit-figtree-nexus",
        type=Path,
        default=None,
        help="Also write a FigTree-compatible NEXUS file with colored taxa.",
    )
    parser.add_argument(
        "--emit-collapsed-newick",
        type=Path,
        default=None,
        help="Also write a collapsed Newick: every subtree whose dominant "
             "palette color covers >= --collapse-dominance fraction of "
             "leaves is merged into a single named tip.",
    )
    parser.add_argument(
        "--emit-collapsed-nexus",
        type=Path,
        default=None,
        help="Companion collapsed NEXUS file (FigTree tends to respect "
             "color tags better from NEXUS than from plain Newick).",
    )
    parser.add_argument(
        "--collapse-dominance",
        type=float,
        default=1.0,
        help="Fraction of leaves that must share the dominant color for a "
             "subtree to be collapsed (default 1.0 = strict monophyletic; "
             "0.9 would collapse 90-percent-one-color subtrees, dropping "
             "minority leaves). Useful for paraphyletic-ish groups.",
    )
    parser.add_argument(
        "--umap-df",
        type=Path,
        default=None,
        help="Optional recolored UMAP dataframe to draw in the center of the circular nested-tree layout.",
    )
    parser.add_argument(
        "--font-size-pt",
        type=float,
        default=7.0,
        help="Base font size in points for circular nested-tree labels and MYA axis text.",
    )
    parser.add_argument(
        "--target-width-mm",
        type=float,
        default=None,
        help="Scale the tight exported figure width to this physical size in millimeters.",
    )
    parser.add_argument(
        "--min-linewidth-pt",
        type=float,
        default=0.5,
        help="Minimum branch and MYA-axis stroke width in points for circular nested-tree layout.",
    )
    parser.add_argument(
        "--emit-layer-pdfs",
        action="store_true",
        help="For circular nested-tree layout, also emit aligned tree/labels/UMAP sidecar PDFs.",
    )
    parser.add_argument(
        "--nested-height-scale",
        type=float,
        default=1.0,
        help="Scale factor for nested-tree-coloring figure height; useful for compressed linear previews.",
    )
    add_palette_argument(parser)

    args = parser.parse_args(argv)
    palette = Palette.from_yaml(args.palette)
    render_palette_preview(
        tree_path=args.tree,
        palette=palette,
        output_path=args.out,
        title=args.title,
        max_leaves=args.max_leaves,
        show_labels=args.show_labels,
        align_tips=args.align_tips,
        layout=args.layout,
        colored_newick_path=args.emit_colored_newick,
        nexus_path=args.emit_figtree_nexus,
        collapsed_newick_path=args.emit_collapsed_newick,
        collapsed_nexus_path=args.emit_collapsed_nexus,
        collapse_dominance=args.collapse_dominance,
        umap_df_path=args.umap_df,
        font_size_pt=args.font_size_pt,
        target_width_mm=args.target_width_mm,
        min_linewidth_pt=args.min_linewidth_pt,
        emit_layer_pdfs=args.emit_layer_pdfs,
        nested_height_scale=args.nested_height_scale,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
