"""Publication-style GO enrichment plots from the sweep output.

Produces three artifacts conventional for GO-enrichment papers:

1. significant_terms_annotated.tsv -- significant_terms.tsv augmented
   with a human-readable go_name column from go-basic.obo.

2. dotplots.pdf -- one page per clade, clusterProfiler-style dotplot:
   y = GO term name (top-20 unique terms by best q), x = log2 fold,
   dot size = k (gene overlap), dot color = -log10(q). Faceted into
   BP / MF / CC subpanels.

3. heatmap.pdf -- cross-clade heatmap of the most-recurring enriched
   terms (rows) across clades (columns). Cell = -log10(q). Surfaces
   pan-clade vs clade-specific signals. One page per namespace.

For each (clade, go_id) we pick the minimum q across sweep cells -- this
summarizes "best evidence that this term is enriched in this clade."
The q is still the within-cell BH q from the sweep; no extra correction
is applied.
"""
import argparse
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.cluster.hierarchy import linkage, leaves_list, dendrogram
from scipy.spatial.distance import pdist

try:
    import networkx as nx
except ImportError:
    nx = None

try:
    import upsetplot as _upsetplot
except ImportError:
    _upsetplot = None


# ---------------------------------------------------------------------
# OBO name + namespace parsing
# ---------------------------------------------------------------------
OBO_NS_MAP = {
    "biological_process": "BP",
    "molecular_function": "MF",
    "cellular_component": "CC",
}


# ---------------------------------------------------------------------
# 20-clade NCBI topology for cross-clade heatmap column ordering + tree
# ---------------------------------------------------------------------
# Child-list order determines the L-to-R column ordering. A clade is a
# "leaf of the clade set" iff its value here is an empty list — those
# get a solid tree-tip connector to the heatmap column; non-leaf clades
# in the set get a dotted connector from their internal-node position
# (since the clade is both a tree node AND a heatmap column).
CLADE_CHILDREN = {
    # Paper uses a Ctenophora-sister topology with the custom taxa
    # Myriazoa (-67) and Parahoxozoa (-68). Ctenophora is basal;
    # Porifera sits inside Myriazoa alongside Parahoxozoa
    # (Cnidaria + Bilateria). See pipeline/step1_generate_newick/run.sh
    # (--custom_phylogeny). Virtual (double-underscore) nodes provide
    # the intermediate topology without claiming a heatmap column.
    "__root__": ["Ctenophora", "__Myriazoa__"],
    "Ctenophora": [],
    "__Myriazoa__": ["Porifera", "__Parahoxozoa__"],
    "Porifera": [],
    "__Parahoxozoa__": ["Cnidaria", "Bilateria"],
    "Cnidaria": [],
    # Bilateria (33213) → Deuterostomia + Protostomia (33317).
    "Bilateria": ["Deuterostomia", "Protostomia"],
    # Chordata (7711) unites Vertebrata with the non-vertebrate chordates
    # (tunicates, lancelets); here it becomes a parent of Vertebrata.
    "Deuterostomia": ["Chordata", "Echinodermata"],
    "Chordata": ["Vertebrata"],
    # Mammalia + Teleostei are vertebrate subclades with rich data in
    # the 202509 release; included at the same nesting depth as Diptera
    # under Insecta. Everything else under Vertebrata sits in the
    # residual (unnamed in our set).
    "Vertebrata": ["Mammalia", "Teleostei"],
    "Mammalia": [],
    "Teleostei": [],
    "Echinodermata": [],
    # Protostomia branches: Nematoda (Ecdysozoa branch w/o Arthropoda),
    # Arthropoda (new), Spiralia. Ecdysozoa itself isn't a column.
    "Protostomia": ["Nematoda", "Arthropoda", "Spiralia"],
    "Nematoda": [],
    "Arthropoda": ["Hexapoda"],
    "Hexapoda": ["Insecta"],
    "Insecta": ["Neoptera"],
    "Neoptera": ["Diptera"],
    "Diptera": [],
    # Spiralia branches: Annelida + Platyhelminthes (new) + Mollusca.
    # Clitellata is a subclade of Annelida (terrestrial/freshwater worms).
    "Spiralia": ["Platyhelminthes", "Annelida", "Mollusca"],
    "Platyhelminthes": [],
    "Annelida": ["Clitellata"],
    "Clitellata": [],
    # Mollusca expanded to include Cephalopoda as the parent of Coleoidea.
    "Mollusca": ["Bivalvia", "Gastropoda", "Scaphopoda", "Cephalopoda"],
    "Bivalvia": [],
    "Gastropoda": [],
    "Scaphopoda": [],
    "Cephalopoda": ["Coleoidea"],
    "Coleoidea": ["Decapodiformes"],
    "Decapodiformes": [],
}


def build_clade_layout(root="__root__"):
    """Pre-order DFS.

    Returns dict-of-dicts with per-clade fields:
      col   — x position in the heatmap (0, 1, 2 …)
      depth — tree depth from virtual root (root=0)
      is_leaf — True if the clade has no in-set descendants
      tree_x — x position of the clade's tree node (midpoint of children
                for internal clades, equal to col for leaves)
    """
    info = {}
    col = [0]

    def dfs_pre(node, depth):
        info.setdefault(node, {})["depth"] = depth
        info[node]["is_leaf"] = not CLADE_CHILDREN.get(node, [])
        if not node.startswith("__"):
            info[node]["col"] = col[0]
            col[0] += 1
        for child in CLADE_CHILDREN.get(node, []):
            dfs_pre(child, depth + 1)

    dfs_pre(root, 0)

    def compute_tree_x(node):
        kids = CLADE_CHILDREN.get(node, [])
        if not kids:
            info[node]["tree_x"] = info[node]["col"]
            return info[node]["tree_x"]
        xs = [compute_tree_x(c) for c in kids]
        tx = (min(xs) + max(xs)) / 2.0
        if not node.startswith("__"):
            info[node]["tree_x"] = tx
        else:
            info[node]["tree_x"] = tx
        return tx

    compute_tree_x(root)
    return info


def draw_clade_tree(ax, info, root="__root__", y_heatmap_top=0.0,
                     y_tip=1.0, y_root=None):
    """Draw a rectangular cladogram above a heatmap.

    Horizontal span of each clade's row = 1 unit per column; max depth
    determines how tall the tree is. Terminal clades get solid vertical
    connectors from the tip level to the heatmap top; non-terminal
    clades (those with in-set descendants) get a dotted diagonal from
    their internal-node position down to their own heatmap column.
    """
    max_depth = max(v["depth"] for v in info.values())
    if y_root is None:
        y_root = y_tip + (max_depth + 1) * 0.6
    # Normalize: map depth→y such that depth=0 → y_root and depth=max → y_tip.
    def depth_to_y(d):
        return y_root - (d / max(max_depth, 1)) * (y_root - y_tip)

    # Horizontal + vertical tree branches for every parent→children. If
    # the current heatmap column set doesn't populate any descendant of a
    # given clade, its tree_x stays None and we skip that subtree — this
    # keeps the function working on small clade subsets (e.g. tests).
    for node, kids in CLADE_CHILDREN.items():
        if not kids:
            continue
        if info[node].get("tree_x") is None:
            continue
        parent_y = depth_to_y(info[node]["depth"])
        child_xs = [info[c].get("tree_x") for c in kids]
        live_xs = [x for x in child_xs if x is not None]
        if not live_xs:
            continue
        ax.plot([min(live_xs), max(live_xs)], [parent_y, parent_y],
                color="black", lw=0.9, zorder=4)
        for c, cx in zip(kids, child_xs):
            if cx is None:
                continue
            cy_top = parent_y
            cy_bot = depth_to_y(info[c]["depth"]) if not info[c]["is_leaf"] else y_tip
            ax.plot([cx, cx], [cy_top, cy_bot],
                    color="black", lw=0.9, zorder=4)

    # Connectors from tree to heatmap column positions.
    for name, v in info.items():
        if name.startswith("__"):
            continue
        if "col" not in v:
            continue
        col_x = v["col"]
        if v["is_leaf"]:
            ax.plot([col_x, col_x], [y_tip, y_heatmap_top],
                    color="black", lw=0.7, zorder=3)
        else:
            tree_x = v.get("tree_x")
            if tree_x is None:
                continue
            node_y = depth_to_y(v["depth"])
            ax.plot([tree_x, col_x], [node_y, y_heatmap_top],
                    color="black", lw=0.7, linestyle=(0, (2, 2)), zorder=3)

    ax.set_xlim(-0.5, max(v["col"] for v in info.values()
                           if "col" in v) + 0.5)
    ax.set_ylim(y_heatmap_top, y_root + 0.2)
    ax.axis("off")


def parse_obo(path):
    """Return {go_id: (name, namespace_abbrev)}."""
    term_info = {}
    cur_id = cur_name = cur_ns = None
    in_term = False
    obsolete = False
    with open(path) as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if line == "[Term]":
                if in_term and cur_id and cur_name and cur_ns and not obsolete:
                    term_info[cur_id] = (cur_name, cur_ns)
                cur_id = cur_name = cur_ns = None
                in_term = True
                obsolete = False
                continue
            if line.startswith("[") and line != "[Term]":
                if in_term and cur_id and cur_name and cur_ns and not obsolete:
                    term_info[cur_id] = (cur_name, cur_ns)
                in_term = False
                continue
            if not in_term or not line:
                continue
            if line.startswith("id: "):
                cur_id = line[4:].strip()
            elif line.startswith("name: "):
                cur_name = line[6:].strip()
            elif line.startswith("namespace: "):
                cur_ns = OBO_NS_MAP.get(line[11:].strip())
            elif line.startswith("is_obsolete: true"):
                obsolete = True
    if in_term and cur_id and cur_name and cur_ns and not obsolete:
        term_info[cur_id] = (cur_name, cur_ns)
    return term_info


# ---------------------------------------------------------------------
# Dotplot (clusterProfiler style)
# ---------------------------------------------------------------------
def draw_dotplot(fig_ax, terms_df, title):
    """Draw one dotplot on an existing matplotlib Axes.

    terms_df: columns ['go_name', 'log2fold', 'mlog10q', 'k'] sorted by
    mlog10q desc (highest-significance first at top of y-axis).
    """
    if terms_df.empty:
        fig_ax.set_title(title + "  (no hits)", fontsize=9)
        fig_ax.set_xticks([])
        fig_ax.set_yticks([])
        return None
    ys = np.arange(len(terms_df))[::-1]  # highest-q at top
    sc = fig_ax.scatter(
        terms_df["log2fold"], ys,
        s=40 + 10 * terms_df["k"].clip(upper=50),
        c=terms_df["mlog10q"],
        cmap="viridis",
        edgecolors="#333", linewidths=0.4,
    )
    fig_ax.set_yticks(ys)
    fig_ax.set_yticklabels(
        [f"{n[:56]}" + ("…" if len(n) > 56 else "")
         for n in terms_df["go_name"]],
        fontsize=7)
    fig_ax.axvline(math.log2(3), ls=":", color="gray", lw=0.5)
    fig_ax.axvline(0, color="black", lw=0.3, alpha=0.3)
    fig_ax.set_xlabel("log2 fold-enrichment", fontsize=8)
    fig_ax.set_title(title, fontsize=9)
    fig_ax.grid(axis="x", alpha=0.15)
    return sc


def make_dotplots(sig_df, out_path, top_n=15, min_fold=3.0):
    """Per clade, three subplots (BP/MF/CC), each showing top-N terms.

    Applies a fold-enrichment gate BEFORE per-term deduplication so the
    large-N hypergeometric artifact (broad terms like GO:0005829 flipping
    to q~1e-5 at fold~1.1 when the foreground approaches N/2) doesn't
    overwrite the real narrow-N signal.
    """
    clades = sorted(sig_df["clade"].dropna().unique())
    # Gate: only keep cells whose fold is high enough to be biologically
    # meaningful. Dedupe afterward.
    gated = sig_df[sig_df["fold"] >= min_fold]
    with PdfPages(out_path) as pdf:
        for clade in clades:
            sub = gated[gated["clade"] == clade].copy()
            if sub.empty:
                continue
            best = (sub.sort_values("q")
                       .drop_duplicates(subset=["go_id"]))
            best["log2fold"] = best["fold"].map(
                lambda f: math.log2(f) if f and f > 0 and math.isfinite(f) else float("nan"))
            best["mlog10q"] = best["q"].map(
                lambda q: 300.0 if q == 0 else (-math.log10(q) if q and q > 0 else float("nan")))
            best = best.dropna(subset=["log2fold", "mlog10q"])

            fig, axes = plt.subplots(1, 3, figsize=(17, 6))
            mappables = []
            per_ns_sizes = []
            for col, ns in enumerate(("BP", "MF", "CC")):
                s = best[best["go_namespace"] == ns]
                # Select top-N by strongest q (clusterProfiler-style), then
                # re-sort the selected rows by log2fold descending so the
                # x-axis decreases monotonically top-to-bottom on the plot
                # (matches clusterProfiler's enrichplot::dotplot convention).
                s = s.sort_values("q").head(top_n)
                s = s.sort_values("log2fold", ascending=False)
                sc = draw_dotplot(axes[col], s, f"{ns}  (top {len(s)})")
                if sc is not None:
                    mappables.append(sc)
                    per_ns_sizes.extend(s["k"].tolist())
            fig.suptitle(f"{clade} — top GO enrichments per namespace  "
                          f"(fold ≥ {min_fold:g}×)",
                          fontsize=11, y=1.01)
            # Right-side colorbar for -log10(q).
            if mappables:
                cbar = fig.colorbar(mappables[-1], ax=axes,
                                     orientation="vertical", fraction=0.03,
                                     pad=0.02)
                cbar.set_label("-log10(q)", fontsize=8)
            # Size legend: k = foreground genes annotated to that term
            # (dot size scales as 40 + 10 * min(k, 50)).
            if per_ns_sizes:
                k_min = int(min(per_ns_sizes))
                k_max = min(50, int(max(per_ns_sizes)))
                k_mid = max(k_min + 1, (k_min + k_max) // 2)
                size_vals = sorted({k_min, k_mid, k_max})
                handles = [plt.scatter([], [], s=40 + 10 * min(k, 50),
                                       c="#777", edgecolors="#333",
                                       linewidths=0.4,
                                       label=f"k = {k}"
                                       + ("+" if k >= 50 else ""))
                           for k in size_vals]
                axes[-1].legend(handles=handles, title="k (foreground\ngenes in term)",
                                 loc="center left",
                                 bbox_to_anchor=(1.25, 0.5),
                                 fontsize=7, title_fontsize=7,
                                 frameon=True)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
    print(f"[write] {out_path}  clades={len(clades)}  top_n_per_ns={top_n}  "
          f"fold_gate={min_fold:g}")


# ---------------------------------------------------------------------
# Cross-clade heatmap
# ---------------------------------------------------------------------
def make_heatmap(sig_df, out_path, min_clades=2, max_terms_per_ns=100,
                  min_fold=3.0):
    """Pivot (term × clade) with -log10(q). One page per namespace.

    Columns are ordered by the 20-clade NCBI topology (pre-order DFS of
    CLADE_CHILDREN) and a cladogram is drawn above the heatmap. Terminal
    clades (no in-set descendants) get solid tip-to-column connectors;
    non-terminal clades that are ALSO heatmap columns get dotted
    connectors from their internal-node position down to the column.

    Gates on fold >= min_fold to suppress large-N hypergeometric artifacts.
    """
    gated = sig_df[sig_df["fold"] >= min_fold]
    best = (gated.sort_values("q")
                 .drop_duplicates(subset=["clade", "go_id"]))
    best["mlog10q"] = best["q"].map(
        lambda q: 300.0 if q == 0 else (-math.log10(q) if q and q > 0 else float("nan")))
    best = best.dropna(subset=["mlog10q"])

    # Column ordering + layout from the 20-clade NCBI topology.
    layout = build_clade_layout()
    ordered_clades = [c for c in sorted(layout, key=lambda k: layout[k].get("col", -1))
                      if "col" in layout[c]]
    clades = [c for c in ordered_clades if c in set(best["clade"])]
    # Clades present in the sweep but not in the hardcoded topology end up
    # at the right; keep them so nothing silently disappears.
    extras = sorted(set(best["clade"]) - set(ordered_clades))
    clades = clades + extras

    with PdfPages(out_path) as pdf:
        for ns in ("BP", "MF", "CC"):
            nsub = best[best["go_namespace"] == ns]
            if nsub.empty:
                continue
            # Filter to terms hit in at least min_clades clades.
            term_counts = nsub.groupby("go_id")["clade"].nunique()
            terms_kept = term_counts[term_counts >= min_clades].index.tolist()
            if not terms_kept:
                # Fallback: keep the most widely hit terms anyway.
                terms_kept = term_counts.sort_values(ascending=False).head(
                    max_terms_per_ns).index.tolist()
            # Rank remaining terms by count then by median significance.
            if len(terms_kept) > max_terms_per_ns:
                ranking = (nsub[nsub["go_id"].isin(terms_kept)]
                           .groupby("go_id")
                           .agg(n_clades=("clade", "nunique"),
                                med_mlog10q=("mlog10q", "median"))
                           .sort_values(["n_clades", "med_mlog10q"],
                                         ascending=[False, False]))
                terms_kept = ranking.head(max_terms_per_ns).index.tolist()

            mat = nsub[nsub["go_id"].isin(terms_kept)].pivot_table(
                index="go_id", columns="clade",
                values="mlog10q", aggfunc="max")
            mat = mat.reindex(index=terms_kept, columns=clades)

            # Ward clustering on the rows (GO terms) only; columns
            # (clades) stay in phylogenetic order. Fill NaN with 0 so
            # "no enrichment" clusters terms that share an absent-everywhere
            # pattern. Needs >=2 terms; skip otherwise.
            arr_for_dist = mat.to_numpy(dtype=float)
            arr_for_dist = np.nan_to_num(arr_for_dist, nan=0.0)
            Z_rows = None
            if arr_for_dist.shape[0] >= 2:
                Z_rows = linkage(arr_for_dist, method="ward")
                order = leaves_list(Z_rows).tolist()
                terms_kept = [terms_kept[i] for i in order]
                mat = mat.reindex(index=terms_kept)

            labels = [f"{gid}  {nsub[nsub.go_id == gid].iloc[0]['go_name'][:44]}"
                      for gid in terms_kept]

            h_heatmap = max(4, 0.22 * len(terms_kept) + 1.5)
            # Tree panel takes ~15% of figure height.
            h_tree = max(1.5, h_heatmap * 0.18)
            h = h_heatmap + h_tree + 0.5
            # Extra width for the left dendrogram + right colorbar panel.
            w = max(10, 0.45 * len(clades) + 7.5)
            fig = plt.figure(figsize=(w, h))
            gs = fig.add_gridspec(
                2, 3,
                height_ratios=[h_tree, h_heatmap],
                # dendrogram | heatmap | right-side padding for labels + colorbar
                width_ratios=[1.6, 10, 0.35],
                hspace=0.02, wspace=0.02,
            )
            ax_tree_topleft = fig.add_subplot(gs[0, 0])  # blank, for alignment
            ax_tree_topleft.axis("off")
            ax_tree = fig.add_subplot(gs[0, 1])
            ax_dendro = fig.add_subplot(gs[1, 0])
            ax = fig.add_subplot(gs[1, 1], sharex=ax_tree)
            cax = fig.add_subplot(gs[:, 2])  # colorbar spans both rows

            arr = mat.to_numpy(dtype=float)
            vmax = np.nanpercentile(arr, 95) if np.any(np.isfinite(arr)) else 1.0
            im = ax.imshow(arr, aspect="auto", cmap="magma_r",
                            vmin=0, vmax=max(vmax, math.log10(0.05) * -1))
            ax.set_xticks(range(len(clades)))
            ax.set_xticklabels(clades, rotation=45, ha="right", fontsize=8)
            ax.set_yticks(range(len(terms_kept)))
            ax.set_yticklabels(labels, fontsize=7)
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position("right")
            ax.set_title(f"Cross-clade GO enrichment — {ns}  "
                          f"(cell = -log10 best q; terms in \u2265{min_clades} clades)",
                          fontsize=10)

            # Ward-cluster dendrogram on the left, sharing y-coords with
            # the heatmap. scipy places leaves at y=5,15,25,... in
            # dendrogram data-coords; we remap so y=0..n-1 is the
            # imshow row range.
            if Z_rows is not None:
                dn = dendrogram(Z_rows, orientation="left", ax=ax_dendro,
                                no_labels=True, color_threshold=0,
                                above_threshold_color="0.4")
                ax_dendro.set_ylim(5 + 10 * (len(terms_kept) - 1) + 5, -5)
                ax_dendro.set_xticks([])
                ax_dendro.set_yticks([])
                for s in ("top", "right", "bottom", "left"):
                    ax_dendro.spines[s].set_visible(False)
            else:
                ax_dendro.axis("off")

            # Colorbar — dedicated axis in the right-side gridspec column
            # so it sits to the RIGHT of the y-axis term labels (which
            # are on the right side of the heatmap) instead of overlapping.
            # Shrink its vertical extent so it doesn't span the full height.
            cbar_h = min(0.6, 3.5 / h)
            pos = cax.get_position()
            cax.set_position([pos.x0, pos.y0 + (pos.height - cbar_h * pos.height) / 2,
                              pos.width, cbar_h * pos.height])
            cbar = fig.colorbar(im, cax=cax)
            cbar.set_label("-log10(q)", fontsize=8)
            for i in range(arr.shape[0]):
                for j in range(arr.shape[1]):
                    v = arr[i, j]
                    if np.isfinite(v) and v >= 1.301:
                        ax.text(j, i, "*", ha="center", va="center",
                                fontsize=7, color="white")

            # Remap the layout's "col" coordinate to the actual heatmap
            # column index (since `clades` may have extras appended and
            # column 0 lines up with x=0 in imshow).
            col_of = {c: i for i, c in enumerate(clades)}
            sub_info = {}
            for c, v in layout.items():
                vv = dict(v)
                if c in col_of:
                    vv["col"] = col_of[c]
                    vv["tree_x"] = col_of[c] if vv.get("is_leaf", True) else None
                sub_info[c] = vv
            # Recompute tree_x on the remapped positions.
            def recompute_tree_x(node):
                kids = CLADE_CHILDREN.get(node, [])
                if not kids:
                    if node in col_of:
                        sub_info[node]["tree_x"] = col_of[node]
                        return col_of[node]
                    return None
                xs = [recompute_tree_x(k) for k in kids]
                xs = [x for x in xs if x is not None]
                if not xs:
                    sub_info[node]["tree_x"] = None
                    return None
                tx = (min(xs) + max(xs)) / 2.0
                sub_info[node]["tree_x"] = tx
                return tx
            recompute_tree_x("__root__")

            draw_clade_tree(ax_tree, sub_info)
            ax_tree.set_xlim(ax.get_xlim())
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
    print(f"[write] {out_path}")


# ---------------------------------------------------------------------
# Helpers for gene-set-aware plots (treeplot / emapplot / cnetplot)
# ---------------------------------------------------------------------
def load_term_gene_lists(path):
    """Read sweep.py's term_gene_lists.tsv.gz.

    Returns a pandas DataFrame; columns:
      clade, axis, N_threshold, go_id, k, gene_ids (comma-joined str)
    """
    if not Path(path).exists():
        return pd.DataFrame(columns=["clade", "axis", "N_threshold", "go_id",
                                      "k", "gene_ids"])
    return pd.read_csv(path, sep="\t", compression="infer",
                       dtype={"gene_ids": str}, keep_default_na=False)


def best_cell_gene_sets(sig_df, gene_lists_df, min_fold=3.0):
    """Pick one (axis, N_threshold) per (clade, go_id) — the cell with
    the strongest q at fold >= min_fold — and attach its gene set.

    Returns DataFrame with columns:
      clade, go_id, go_namespace, go_name, k, q, fold, gene_set (frozenset of GeneIDs)
    Only rows with a non-empty gene list survive.
    """
    if sig_df.empty:
        return pd.DataFrame()
    # For each (clade, axis, N, go_id), keep the best q per (clade, go_id);
    # this mirrors the selection make_heatmap uses (min q cell).
    gated = sig_df[sig_df["fold"] >= min_fold].copy()
    if gated.empty:
        return pd.DataFrame()
    best = (gated.sort_values("q")
                  .drop_duplicates(subset=["clade", "axis",
                                            "N_threshold", "go_id"])
                  .sort_values("q")
                  .drop_duplicates(subset=["clade", "go_id"])
                  .copy())
    # `significant_terms.tsv` now carries gene_ids inline (canonical
    # schema). Drop it before joining against the term_gene_lists sidecar
    # so pandas doesn't disambiguate the merge into gene_ids_x / _y.
    best = best.drop(columns=[c for c in ("gene_ids", "gene_symbols")
                              if c in best.columns])
    # Merge with the per-cell gene list file.
    if gene_lists_df is None or gene_lists_df.empty:
        return pd.DataFrame()
    merged = best.merge(
        gene_lists_df[["clade", "axis", "N_threshold", "go_id", "gene_ids"]],
        on=["clade", "axis", "N_threshold", "go_id"], how="left")
    merged = merged[merged["gene_ids"].notna() & (merged["gene_ids"] != "")]
    # term_gene_lists.tsv.gz uses comma-joined IDs; significant_terms.tsv
    # uses semicolon. Support both so we can later drop the sidecar.
    merged["gene_set"] = merged["gene_ids"].map(
        lambda s: frozenset(s.replace(";", ",").split(",")))
    return merged[["clade", "go_id", "go_namespace", "go_name", "k",
                   "q", "fold", "gene_set"]]


def jaccard(a, b):
    """Jaccard similarity between two iterables-cast-to-set."""
    if not a or not b:
        return 0.0
    A, B = set(a), set(b)
    u = A | B
    if not u:
        return 0.0
    return len(A & B) / len(u)


def _truncate(s, n=48):
    s = str(s or "")
    return s if len(s) <= n else s[:n - 1] + "…"


# ---------------------------------------------------------------------
# treeplot — enrichplot::treeplot analog
# ---------------------------------------------------------------------
def make_treeplots(best_df, out_path, top_n=25):
    """One page per clade; 3 subplots (BP/MF/CC).

    For each (clade, namespace) take the top-N terms by q, compute
    Jaccard distance (1 - J) on their foreground gene sets, Ward-cluster,
    and draw a horizontal dendrogram with term labels + a -log10(q)
    annotation strip. Single-term / empty cases are skipped.
    """
    if best_df is None or best_df.empty:
        print(f"[skip] {out_path}  (no gene-set data)")
        return
    clades = sorted(best_df["clade"].dropna().unique())
    with PdfPages(out_path) as pdf:
        n_pages = 0
        for clade in clades:
            cdf = best_df[best_df["clade"] == clade]
            # Build one column per namespace; tolerate missing NS.
            panels = []
            for ns in ("BP", "MF", "CC"):
                sub = (cdf[cdf["go_namespace"] == ns]
                        .sort_values("q")
                        .head(top_n)
                        .copy())
                panels.append((ns, sub))
            if all(p[1].shape[0] < 2 for p in panels):
                continue
            # Stack the 3 namespaces VERTICALLY (not horizontally) so each
            # panel gets the full figure width for the dendrogram + long
            # GO term labels. Prior 1x3 layout clipped labels and bled
            # them into neighbouring panels no matter how wide we made
            # the figure.
            per_panel_h = [max(3.0, 0.24 * p[1].shape[0] + 0.8) for p in panels]
            fig_h = sum(per_panel_h) + 1.0
            fig, axes = plt.subplots(
                3, 1, figsize=(13, fig_h),
                gridspec_kw={"height_ratios": per_panel_h, "hspace": 0.55})
            for ax, (ns, sub) in zip(axes, panels):
                if sub.shape[0] < 2:
                    ax.set_title(f"{ns}  (n={sub.shape[0]}; cluster needs ≥2)",
                                 fontsize=9)
                    ax.axis("off")
                    continue
                gs = sub["gene_set"].tolist()
                n_terms = len(gs)
                # Condensed Jaccard-distance matrix.
                dmat = np.zeros(n_terms * (n_terms - 1) // 2, dtype=float)
                idx = 0
                for i in range(n_terms - 1):
                    for j in range(i + 1, n_terms):
                        dmat[idx] = 1.0 - jaccard(gs[i], gs[j])
                        idx += 1
                try:
                    Z = linkage(dmat, method="average")
                except Exception as exc:
                    print(f"[warn] treeplot {clade}/{ns}: linkage failed "
                          f"({exc}); skipping panel")
                    ax.axis("off")
                    continue
                mlq = sub["q"].map(
                    lambda q: 300.0 if q == 0
                    else (-math.log10(q) if q and q > 0 else 0.0)).to_numpy()
                labels = [f"{gid} {_truncate(nm, 44)}"
                          for gid, nm in zip(sub["go_id"], sub["go_name"])]
                # orientation="left": tree on LEFT (root at leftmost x,
                # branches going right), leaves + labels on RIGHT edge.
                # Matches clusterProfiler::treeplot convention.
                dn = dendrogram(Z, orientation="left", ax=ax,
                                 labels=labels, color_threshold=0.7,
                                 leaf_font_size=7,
                                 above_threshold_color="0.3")
                # scipy attaches leaf labels to the y-axis, which defaults
                # to the LEFT side of the axes — so labels extend leftward
                # over the tree branches. Move y-ticks/labels to the RIGHT
                # so they sit at the leaf x=0 position and extend outward
                # into the wspace between panels.
                ax.yaxis.tick_right()
                ax.yaxis.set_label_position("right")
                leaf_order = dn["leaves"]
                ordered_mlq = mlq[leaf_order]
                # With orientation="left" scipy sets xlim such that leaves
                # are at x=0 and the root is at a negative x. Labels render
                # on the right (positive x). Put a small colored dot at
                # each leaf — colored by −log10(q) — so the significance
                # annotation travels with the label without needing a
                # separate strip that can collide with text.
                y_positions = np.arange(len(leaf_order)) * 10 + 5
                vmax = max(float(np.nanmax(ordered_mlq)), 1.3)
                cmap = plt.get_cmap("magma_r")
                for y, v in zip(y_positions, ordered_mlq):
                    color = cmap(min(v, vmax) / vmax)
                    ax.scatter([0], [y], s=45, color=color,
                               edgecolors="0.25", linewidths=0.4,
                               clip_on=False, zorder=6)
                ax.set_title(f"{ns}  (top {len(sub)} by q, "
                             f"Jaccard-Ward on foreground genes)",
                             fontsize=9)
                ax.set_xlabel("1 − Jaccard (tree)", fontsize=8)
                # Hide the y-axis spine clutter — the leaf labels carry
                # the information, ticks/spine just add noise.
                ax.tick_params(axis="y", length=0)
                for s in ("top", "right"):
                    ax.spines[s].set_visible(False)
            fig.suptitle(f"{clade} — GO term clustering by shared "
                          f"foreground genes",
                          fontsize=11, y=1.01)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            n_pages += 1
    print(f"[write] {out_path}  pages={n_pages}")


# ---------------------------------------------------------------------
# emapplot — enrichment map (term-term network)
# ---------------------------------------------------------------------
def make_emapplots(best_df, out_path, top_n=50, edge_threshold=0.3,
                    label_top=20):
    """One page per clade (3-panel BP/MF/CC).

    Nodes = top-N enriched GO terms (by q).
    Edges = Jaccard similarity between foreground gene sets >= edge_threshold.
    Node size ∝ k, color by -log10(q). Spring layout.
    """
    if nx is None:
        print(f"[skip] {out_path}  (networkx not available)")
        return
    if best_df is None or best_df.empty:
        print(f"[skip] {out_path}  (no gene-set data)")
        return
    clades = sorted(best_df["clade"].dropna().unique())
    with PdfPages(out_path) as pdf:
        n_pages = 0
        for clade in clades:
            cdf = best_df[best_df["clade"] == clade]
            panels = []
            for ns in ("BP", "MF", "CC"):
                sub = (cdf[cdf["go_namespace"] == ns]
                        .sort_values("q")
                        .head(top_n)
                        .copy())
                panels.append((ns, sub))
            if all(p[1].empty for p in panels):
                continue
            fig, axes = plt.subplots(1, 3, figsize=(21, 7))
            drew_any = False
            for ax, (ns, sub) in zip(axes, panels):
                if sub.empty:
                    ax.set_title(f"{ns}  (no hits)", fontsize=9)
                    ax.axis("off")
                    continue
                G = nx.Graph()
                sub = sub.reset_index(drop=True)
                gs = sub["gene_set"].tolist()
                for i, row in sub.iterrows():
                    G.add_node(row["go_id"],
                                k=row["k"],
                                mlq=(300.0 if row["q"] == 0
                                     else (-math.log10(row["q"])
                                           if row["q"] > 0 else 0.0)),
                                name=row["go_name"])
                for i in range(len(sub) - 1):
                    for j in range(i + 1, len(sub)):
                        jc = jaccard(gs[i], gs[j])
                        if jc >= edge_threshold:
                            G.add_edge(sub.loc[i, "go_id"],
                                       sub.loc[j, "go_id"], weight=jc)
                if G.number_of_edges() == 0:
                    ax.set_title(f"{ns}  (n={G.number_of_nodes()}, "
                                  f"no edges ≥ {edge_threshold})",
                                  fontsize=9)
                    ax.axis("off")
                    continue
                # Layout: only connected component(s). Isolates are dropped
                # so the plot focuses on the map structure.
                G_sub = G.subgraph(
                    [n for n in G.nodes if G.degree(n) > 0]).copy()
                pos = nx.spring_layout(G_sub, weight="weight",
                                        seed=42, k=0.6 / math.sqrt(
                                            max(G_sub.number_of_nodes(), 1)))
                ks = np.array([G_sub.nodes[n]["k"] for n in G_sub.nodes])
                mlqs = np.array([G_sub.nodes[n]["mlq"] for n in G_sub.nodes])
                sizes = 40 + 8 * np.clip(ks, 0, 80)
                nx.draw_networkx_edges(G_sub, pos, ax=ax,
                                         edge_color="#999",
                                         width=[0.5 + 2.0 * G_sub[u][v]["weight"]
                                                for u, v in G_sub.edges],
                                         alpha=0.6)
                nodes_collection = nx.draw_networkx_nodes(
                    G_sub, pos, ax=ax,
                    node_size=sizes.tolist(),
                    node_color=mlqs.tolist(),
                    cmap="magma_r",
                    vmin=0,
                    vmax=max(float(mlqs.max()), 1.3),
                    edgecolors="#333", linewidths=0.4)
                # Label only the top-label_top by mlq
                top_ids = sorted(G_sub.nodes,
                                  key=lambda n: -G_sub.nodes[n]["mlq"]
                                  )[:label_top]
                label_map = {n: _truncate(G_sub.nodes[n]["name"], 32)
                              for n in top_ids}
                nx.draw_networkx_labels(G_sub, pos, labels=label_map,
                                          font_size=6, ax=ax)
                ax.set_title(f"{ns}  (nodes={G_sub.number_of_nodes()}, "
                             f"edges={G_sub.number_of_edges()}, "
                             f"Jaccard ≥ {edge_threshold})", fontsize=9)
                ax.axis("off")
                fig.colorbar(nodes_collection, ax=ax, fraction=0.04,
                              pad=0.02, label="-log10(q)")
                drew_any = True
            if drew_any:
                fig.suptitle(f"{clade} — enrichment map "
                             f"(edges: Jaccard on foreground gene sets)",
                             fontsize=11, y=1.02)
                pdf.savefig(fig, bbox_inches="tight")
                n_pages += 1
            plt.close(fig)
    print(f"[write] {out_path}  pages={n_pages}")


# ---------------------------------------------------------------------
# cnetplot — gene-concept network
# ---------------------------------------------------------------------
def make_cnetplots(best_df, out_path, fam_map_path=None,
                    gene_symbols_path=None, top_terms=10, max_genes=50,
                    min_fold=3.0):
    """One page per clade; bipartite term-gene network.

    Term nodes: top-`top_terms` GO terms across BP/MF/CC by q.
    Gene nodes: up to `max_genes` foreground genes, ranked by membership count.
    Gene labels from gene2accession Symbol column if gene_symbols_path given.
    """
    if nx is None:
        print(f"[skip] {out_path}  (networkx not available)")
        return
    if best_df is None or best_df.empty:
        print(f"[skip] {out_path}  (no gene-set data)")
        return

    gene_sym = {}
    if gene_symbols_path and Path(gene_symbols_path).exists():
        s = pd.read_csv(gene_symbols_path, sep="\t",
                         dtype={"gene_id": str, "symbol": str})
        gene_sym = dict(zip(s["gene_id"].astype(str), s["symbol"]))

    ns_color = {"BP": "#4C72B0", "MF": "#DD8452", "CC": "#55A868"}
    clades = sorted(best_df["clade"].dropna().unique())
    with PdfPages(out_path) as pdf:
        n_pages = 0
        for clade in clades:
            cdf = best_df[best_df["clade"] == clade].copy()
            cdf = cdf[cdf["fold"] >= min_fold]
            if cdf.empty:
                continue
            cdf = cdf.sort_values("q").head(top_terms).reset_index(drop=True)
            # Gene frequency across selected terms.
            gene_freq = defaultdict(int)
            for gs in cdf["gene_set"]:
                for g in gs:
                    gene_freq[g] += 1
            # Top max_genes by membership count, break ties by GeneID
            # (deterministic) for reproducibility.
            top_genes = [g for g, _ in sorted(
                gene_freq.items(),
                key=lambda kv: (-kv[1], kv[0]))[:max_genes]]
            if not top_genes:
                continue
            G = nx.Graph()
            for _, row in cdf.iterrows():
                G.add_node(("T", row["go_id"]),
                            kind="term",
                            ns=row["go_namespace"],
                            name=row["go_name"],
                            k=row["k"],
                            q=row["q"])
            for g in top_genes:
                G.add_node(("G", g), kind="gene",
                            symbol=gene_sym.get(g, g),
                            freq=gene_freq[g])
            for _, row in cdf.iterrows():
                for g in row["gene_set"]:
                    if g in set(top_genes):
                        G.add_edge(("T", row["go_id"]), ("G", g))
            if G.number_of_edges() == 0:
                continue
            pos = nx.spring_layout(G, seed=42,
                                    k=0.9 / math.sqrt(
                                        max(G.number_of_nodes(), 1)))
            term_nodes = [n for n in G.nodes if n[0] == "T"]
            gene_nodes = [n for n in G.nodes if n[0] == "G"]
            fig, ax = plt.subplots(figsize=(14, 10))
            nx.draw_networkx_edges(G, pos, ax=ax,
                                     edge_color="#bbb", width=0.4, alpha=0.6)
            nx.draw_networkx_nodes(G, pos,
                                     nodelist=gene_nodes,
                                     node_color="#d0d0d0",
                                     node_size=60,
                                     edgecolors="#555", linewidths=0.3,
                                     ax=ax)
            # Term nodes sized by k, colored by namespace.
            for ns in ("BP", "MF", "CC"):
                sub = [n for n in term_nodes if G.nodes[n]["ns"] == ns]
                if not sub:
                    continue
                sizes = [200 + 12 * min(G.nodes[n]["k"], 80) for n in sub]
                nx.draw_networkx_nodes(
                    G, pos, nodelist=sub,
                    node_color=ns_color[ns],
                    node_size=sizes,
                    edgecolors="black", linewidths=0.6,
                    label=ns, ax=ax)
            gene_labels = {n: G.nodes[n]["symbol"] for n in gene_nodes}
            nx.draw_networkx_labels(G, pos, labels=gene_labels,
                                      font_size=6, ax=ax)
            term_labels = {n: _truncate(G.nodes[n]["name"], 36)
                            for n in term_nodes}
            nx.draw_networkx_labels(G, pos, labels=term_labels,
                                      font_size=7,
                                      font_color="black",
                                      font_weight="bold", ax=ax)
            ax.set_title(
                f"{clade} — cnetplot: top-{len(cdf)} terms × "
                f"top-{len(top_genes)} foreground genes "
                f"(fold ≥ {min_fold:g})",
                fontsize=11)
            ax.axis("off")
            ax.legend(loc="lower right", fontsize=8, frameon=True,
                      title="namespace")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            n_pages += 1
    print(f"[write] {out_path}  pages={n_pages}")


# ---------------------------------------------------------------------
# upsetplot — cross-clade intersection of enriched term sets
# ---------------------------------------------------------------------
def _draw_manual_upset(pdf, ns, clade_sets, min_fold, q_thresh,
                        max_intersections):
    """Manual horizontal-bar upset rendering. Used when upsetplot lib
    fails; keeps the output self-contained and publishable.
    """
    presence = defaultdict(list)
    all_terms = set().union(*clade_sets.values())
    for t in all_terms:
        sig = frozenset(c for c, s in clade_sets.items() if t in s)
        presence[sig].append(t)
    groups = sorted(presence.items(),
                     key=lambda kv: (-len(kv[1]), -len(kv[0])))[:max_intersections]
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1,
        figsize=(max(12, 0.5 * len(groups) + 6),
                 max(7, 0.3 * len(clade_sets) + 5)),
        gridspec_kw={"height_ratios": [2.5, 2]})
    xs = np.arange(len(groups))
    ax_top.bar(xs, [len(g[1]) for g in groups], color="#444")
    ax_top.set_ylabel("Intersection size\n(# terms)", fontsize=9)
    ax_top.set_xticks([])
    for spine in ("top", "right"):
        ax_top.spines[spine].set_visible(False)
    for x, g in zip(xs, groups):
        ax_top.text(x, len(g[1]) + 0.5, str(len(g[1])),
                     ha="center", fontsize=7)
    clades_order = list(clade_sets.keys())
    for yi, c in enumerate(clades_order):
        for xi, (sig, _terms) in enumerate(groups):
            if c in sig:
                ax_bot.scatter(xi, yi, s=30, color="#444")
            else:
                ax_bot.scatter(xi, yi, s=10, color="#ddd")
        # Connect filled dots within a column with a thin vertical line
        # (matches conventional upset rendering).
    for xi, (sig, _terms) in enumerate(groups):
        members_y = [yi for yi, c in enumerate(clades_order) if c in sig]
        if len(members_y) >= 2:
            ax_bot.plot([xi, xi], [min(members_y), max(members_y)],
                         color="#444", lw=1.2, zorder=0)
    ax_bot.set_yticks(range(len(clades_order)))
    ax_bot.set_yticklabels(clades_order, fontsize=8)
    ax_bot.set_xticks(xs)
    ax_bot.set_xticklabels([""] * len(xs))
    ax_bot.invert_yaxis()
    for spine in ("top", "right", "bottom"):
        ax_bot.spines[spine].set_visible(False)
    fig.suptitle(f"Cross-clade GO term intersections — {ns}  "
                 f"(fold ≥ {min_fold:g}, q ≤ {q_thresh}; "
                 f"top {len(groups)} groups)",
                 fontsize=11)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def make_upsetplots(sig_df, out_path, min_fold=3.0, q_thresh=0.05,
                     max_intersections=30):
    """One page per namespace.

    "Enriched set for clade X" = set of go_id where ANY (axis, N) cell
    had fold >= min_fold AND q <= q_thresh.
    Uses upsetplot library when available; falls back to a stacked-bar
    horizontal manual rendering.
    """
    if sig_df is None or sig_df.empty:
        print(f"[skip] {out_path}  (no data)")
        return
    gated = sig_df[(sig_df["fold"] >= min_fold)
                    & (sig_df["q"] <= q_thresh)]
    if gated.empty:
        print(f"[skip] {out_path}  (no cells pass fold≥{min_fold} & q≤{q_thresh})")
        return
    with PdfPages(out_path) as pdf:
        n_pages = 0
        for ns in ("BP", "MF", "CC"):
            nsub = gated[gated["go_namespace"] == ns]
            if nsub.empty:
                continue
            clade_sets = {
                c: set(nsub[nsub["clade"] == c]["go_id"].unique())
                for c in sorted(nsub["clade"].unique())
            }
            clade_sets = {c: s for c, s in clade_sets.items() if s}
            if len(clade_sets) < 2:
                continue
            # Try the upsetplot library first; fall back to the manual
            # rendering if it errors out (upsetplot 0.9.0 is flaky on
            # pandas 3.x / matplotlib 3.10 due to the library using
            # chained-assignment fillna that leaves NaN colors behind).
            used_lib = False
            if _upsetplot is not None:
                try:
                    data = _upsetplot.from_contents(clade_sets)
                    fig = plt.figure(
                        figsize=(max(12, 0.35 * len(clade_sets) + 8), 8))
                    _upsetplot.UpSet(
                        data, subset_size="count",
                        sort_by="cardinality",
                        show_counts=True,
                        min_subset_size=1,
                    ).plot(fig=fig)
                    fig.suptitle(
                        f"Cross-clade GO term intersections — {ns}  "
                        f"(fold ≥ {min_fold:g}, q ≤ {q_thresh})",
                        fontsize=11)
                    pdf.savefig(fig, bbox_inches="tight")
                    plt.close(fig)
                    used_lib = True
                except Exception as exc:
                    print(f"[warn] upsetplot {ns}: lib failed ({exc}); "
                          f"drawing manual fallback")
                    try:
                        plt.close(fig)
                    except Exception:
                        pass
            if not used_lib:
                _draw_manual_upset(pdf, ns, clade_sets, min_fold,
                                    q_thresh, max_intersections)
            n_pages += 1
    print(f"[write] {out_path}  pages={n_pages}")


# ---------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="egt go plot",
        description=__doc__,
    )
    ap.add_argument("--significant-terms", required=True)
    ap.add_argument("--obo", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--top-n", type=int, default=15,
                    help="top-N terms per (clade, namespace) in dotplots")
    ap.add_argument("--heatmap-min-clades", type=int, default=2,
                    help="terms must be enriched in at least this many clades "
                         "to appear on the heatmap (default: 2)")
    ap.add_argument("--heatmap-max-terms", type=int, default=100,
                    help="cap heatmap rows per namespace (default: 40)")
    ap.add_argument("--min-fold", type=float, default=3.0,
                    help="drop cells whose fold-enrichment is below this "
                         "BEFORE per-term dedupe, so large-N artifacts "
                         "don't mask real signal (default: 3.0)")
    ap.add_argument("--term-gene-lists",
                    help="sweep.py term_gene_lists.tsv.gz — required for "
                         "treeplot / emapplot / cnetplot")
    ap.add_argument("--gene-symbols",
                    help="sweep.py gene_symbols.tsv — optional; used to "
                         "label gene nodes on cnetplot")
    ap.add_argument("--treeplot-top-n", type=int, default=25,
                    help="top-N terms per (clade, namespace) in treeplot")
    ap.add_argument("--emap-top-n", type=int, default=50,
                    help="top-N terms per (clade, namespace) in emapplot")
    ap.add_argument("--emap-jaccard", type=float, default=0.3,
                    help="Jaccard threshold for emapplot edges")
    ap.add_argument("--cnet-top-terms", type=int, default=10,
                    help="total terms per clade on cnetplot")
    ap.add_argument("--cnet-max-genes", type=int, default=50,
                    help="max gene nodes per cnetplot page")
    ap.add_argument("--upset-q", type=float, default=0.05,
                    help="q threshold for inclusion in upsetplot sets")
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[load] OBO …")
    term_info = parse_obo(args.obo)
    print(f"  terms_in_obo={len(term_info)}")

    print("[load] significant_terms …")
    from ..io import load_significant_terms
    sig = load_significant_terms(args.significant_terms)
    print(f"  rows={len(sig)}")

    # Annotate with name + authoritative namespace from OBO.
    def name_of(g):
        v = term_info.get(g)
        return v[0] if v else g
    def ns_of(g):
        v = term_info.get(g)
        return v[1] if v else "?"
    sig["go_name"] = sig["go_id"].map(name_of)
    # Overwrite go_namespace from OBO (authoritative) where available.
    sig["go_namespace"] = sig["go_id"].map(ns_of).where(
        sig["go_id"].map(lambda g: g in term_info),
        sig["go_namespace"])

    annotated_path = out_dir / "significant_terms_annotated.tsv"
    sig.to_csv(annotated_path, sep="\t", index=False)
    print(f"[write] {annotated_path}  rows={len(sig)}")

    make_dotplots(sig, out_dir / "dotplots.pdf", top_n=args.top_n,
                   min_fold=args.min_fold)
    # Three heatmap variants: top-25, top-50, top-N (CLI default). All
    # same rules, different caps.
    for cap, name in [(25, "heatmap_top25.pdf"),
                      (50, "heatmap_top50.pdf"),
                      (args.heatmap_max_terms, "heatmap.pdf")]:
        make_heatmap(sig, out_dir / name,
                     min_clades=args.heatmap_min_clades,
                     max_terms_per_ns=cap,
                  min_fold=args.min_fold)

    # Gene-set-aware plots — need sweep.py's term_gene_lists.tsv.gz.
    # Treeplot, emapplot, cnetplot, upsetplot all go under out_dir/.
    if args.term_gene_lists:
        print(f"[load] term_gene_lists …")
        glists = load_term_gene_lists(args.term_gene_lists)
        print(f"  rows={len(glists)}")
        best_df = best_cell_gene_sets(sig, glists, min_fold=args.min_fold)
        print(f"  best-cell-per-(clade,go_id) rows={len(best_df)}")
    else:
        print("[skip] treeplot/emapplot/cnetplot — pass --term-gene-lists "
              "to enable gene-set-aware plots")
        best_df = None

    if best_df is not None:
        make_treeplots(best_df, out_dir / "treeplot.pdf",
                        top_n=args.treeplot_top_n)
        make_emapplots(best_df, out_dir / "emapplot.pdf",
                        top_n=args.emap_top_n,
                        edge_threshold=args.emap_jaccard)
        make_cnetplots(best_df, out_dir / "cnetplot.pdf",
                        gene_symbols_path=args.gene_symbols,
                        top_terms=args.cnet_top_terms,
                        max_genes=args.cnet_max_genes,
                        min_fold=args.min_fold)
    # upsetplot needs only sig; no gene lists required.
    make_upsetplots(sig, out_dir / "upsetplot.pdf",
                     min_fold=args.min_fold,
                     q_thresh=args.upset_q)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
