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
import re
import sys
from pathlib import Path

from egt.palette import Palette, add_palette_argument


# ---------- helpers ----------

_TAXID_IN_BRACKETS = re.compile(r"\[(\-?\d+)\]\s*$")
_TAXID_TRAIL_DASH  = re.compile(r"-(\d+)-")
_BARE_TAXID        = re.compile(r"^-?\d+$")


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
    colored_newick_path: Path | None = None,
    nexus_path: Path | None = None,
    collapsed_newick_path: Path | None = None,
    collapsed_nexus_path: Path | None = None,
    collapse_dominance: float = 1.0,
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
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    t = Tree(str(tree_path), parser=1) if hasattr(Tree, "__call__") else Tree(open(tree_path).read())
    # ete4's Tree() accepts a newick string or a path in different versions;
    # the above two-step dance is resilient.

    # Walk the tree, compute per-leaf coordinates + color.
    leaves = list(t.leaves())
    if max_leaves is not None and len(leaves) > max_leaves:
        leaves = leaves[:max_leaves]

    if not leaves:
        raise SystemExit(f"Tree at {tree_path} has no leaves.")

    # Assign y-coordinates in leaf-traversal order, x = cumulative depth.
    # ete4's node `.dist` attribute gives edge length; fall back to 1 if absent.
    node_xy: dict[object, tuple[float, float]] = {}
    leaf_y = {leaf: i for i, leaf in enumerate(leaves)}

    def _depth(node) -> float:
        d = 0.0
        cur = node
        while cur and cur.up is not None:
            d += float(getattr(cur, "dist", 1.0) or 1.0)
            cur = cur.up
        return d

    max_leaf_depth = max(_depth(leaf) for leaf in leaves) if align_tips else None
    for leaf in leaves:
        x = max_leaf_depth if align_tips else _depth(leaf)
        node_xy[leaf] = (x, leaf_y[leaf])

    # Internal node x = max leaf depth under it; y = mean of child y's.
    def _fill_internal(node):
        if node.is_leaf:
            return node_xy[node]
        child_xys = [_fill_internal(c) for c in node.children]
        y = sum(xy[1] for xy in child_xys) / len(child_xys)
        x = _depth(node)
        node_xy[node] = (x, y)
        return (x, y)
    _fill_internal(t)

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

    # Figure.
    h = max(6, 0.04 * len(leaves))
    fig, (ax_tree, ax_legend) = plt.subplots(
        1, 2, figsize=(14, h), gridspec_kw={"width_ratios": [4, 1]}
    )

    # Draw edges.
    def _draw_edges(node):
        if node is t:
            px, py = node_xy[node]
        else:
            px, py = node_xy[node.up]
        cx, cy = node_xy[node]
        ax_tree.plot([px, px, cx], [py, cy, cy], color="black", linewidth=0.4, solid_capstyle="butt")
        for c in node.children:
            _draw_edges(c)
    _draw_edges(t)

    # Leaf markers.
    for leaf, (x, y) in node_xy.items():
        if leaf not in leaf_y:
            continue
        ax_tree.scatter([x], [y], s=12, color=leaf_colors[leaf], edgecolors="none", zorder=3)
        if show_labels:
            ax_tree.text(x + 1, y, leaf.name, fontsize=4, va="center")

    ax_tree.invert_yaxis()
    ax_tree.set_yticks([])
    ax_tree.set_xlabel("cumulative branch length")
    ax_tree.spines["top"].set_visible(False)
    ax_tree.spines["right"].set_visible(False)
    ax_tree.spines["left"].set_visible(False)
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
    # Deep-copy via re-parsing. ete4's Tree.write() default returns a Newick string.
    from ete4 import Tree as ETree
    work = ETree(tree.write(), parser=1)
    # rebuild leaf_colors for the copy by walking both in parallel
    orig_leaves = {l.name: leaf_colors[l] for l in leaves}
    copy_leaves = list(work.leaves())
    copy_colors = {id(l): orig_leaves.get(l.name) for l in copy_leaves}

    color_to_label = _build_color_label_lookup(palette)

    from collections import Counter

    def _dominant_color(leaf_descendants):
        """Return (color, fraction) for the majority color under a node."""
        cts = Counter(copy_colors.get(id(l)) for l in leaf_descendants)
        if not cts:
            return None, 0.0
        color, n = cts.most_common(1)[0]
        total = sum(cts.values())
        return color, (n / total if total else 0.0)

    # Pre-scan: count how many independent collapsible subtrees will
    # form under each label (so we know whether to add a `#N` suffix).
    _collapsed_final_count: dict[str, int] = {}
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
                continue  # parent will eat this subtree; don't count here
        label = color_to_label.get(color, "clade")
        _collapsed_final_count[label] = _collapsed_final_count.get(label, 0) + 1

    _collapsed_counts: dict[str, int] = {}

    # Post-order traversal so children are visited before parents.
    # Using list() snapshots the iteration before mutating.
    for node in list(work.traverse("postorder")):
        if node.is_leaf:
            continue
        # After earlier iterations detached a subtree, some internal nodes
        # may no longer have leaves (they've been pruned to zero children).
        leaf_descendants = list(node.leaves())
        if not leaf_descendants:
            continue
        color, frac = _dominant_color(leaf_descendants)
        if color is not None and frac >= dominance:
            label = color_to_label.get(color, "clade")
            # Count tips before collapsing (for the (n=N) label) — use only
            # dominant-color leaves when dominance<1 so the reported count
            # reflects the clade, not the stragglers being dropped.
            dominant_leaves = [l for l in leaf_descendants
                               if copy_colors.get(id(l)) == color]
            # Total branch length from node to a tip — use max over dominant
            # leaves so the collapsed tip still reaches the present day.
            def _depth_from(n):
                d = 0.0; cur = n
                while cur.up is not None and cur is not node:
                    d += float(getattr(cur, "dist", 1.0) or 1.0)
                    cur = cur.up
                return d
            tip_depth = max(_depth_from(l) for l in (dominant_leaves or leaf_descendants))
            label = color_to_label.get(color, "clade")
            # Prune children.
            for child in list(node.children):
                child.detach()
            # Mark node as a leaf: rename + add tip_depth to own dist.
            n_sp = len(dominant_leaves)
            n_total = len(leaf_descendants)
            # FigTree rejects duplicate taxon names. Two independent subtrees
            # that both collapse to the same clade need distinct labels.
            _collapsed_counts[label] = _collapsed_counts.get(label, 0) + 1
            idx = _collapsed_counts[label]
            is_only_one = _collapsed_final_count.get(label, 0) == 1
            size_str = f"n={n_sp}" if n_sp == n_total else f"n={n_sp}/{n_total}"
            node.name = (
                f"{label} ({size_str})"
                if is_only_one
                else f"{label} #{idx} ({size_str})"
            )
            try:
                node.dist = float(node.dist or 0.0) + tip_depth
            except Exception:
                pass
            # Attach a color attribute so the writer emits [&!color=#…]
            copy_colors[id(node)] = color

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
            color = copy_colors.get(id(node)) or orig_leaves.get(node.name)
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
        colored_newick_path=args.emit_colored_newick,
        nexus_path=args.emit_figtree_nexus,
        collapsed_newick_path=args.emit_collapsed_newick,
        collapsed_nexus_path=args.emit_collapsed_nexus,
        collapse_dominance=args.collapse_dominance,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
