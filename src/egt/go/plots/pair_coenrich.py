"""Summary figures for pair co-enrichment (pair_co_vs_bag.tsv.gz).

Three-page PDF:
  1. per-clade quadrant scatter: q_pair vs q_bag with shaded regions
     (both-significant, pair-only, bag-only, neither).
  2. observed k_co vs expected k_co scatter, log–log.
  3. horizontal stacked bar of per-clade category counts.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle


CATEGORY_COLORS = {
    "both":      "#2ca02c",
    "pair-only": "#d62728",
    "bag-only":  "#1f77b4",
    "neither":   "#aaaaaa",
}


PAIR_CO_SHORT_ALIASES = {
    "pair_cohits_[k]": "k_co",
    "pair_count_[n]": "n_pairs",
    "fam_hits_[K]": "K_fams",
    "fam_count_[N]": "N_fams",
    "pair_either_hits": "k_either",
}


def load(in_path) -> pd.DataFrame:
    df = pd.read_csv(in_path, sep="\t")
    df = df.rename(columns=PAIR_CO_SHORT_ALIASES)
    df["q_pair"] = df["q_pair_co"]
    df["q_bag_eff"] = df["q_bag"].fillna(1.0)
    df["mlog10_q_pair"] = -np.log10(np.clip(df["q_pair"], 1e-300, 1.0))
    df["mlog10_q_bag"] = -np.log10(np.clip(df["q_bag_eff"], 1e-300, 1.0))
    df["category"] = "neither"
    df.loc[(df["q_pair"] <= 0.05) & (df["q_bag_eff"] <= 0.05), "category"] = "both"
    df.loc[(df["q_pair"] <= 0.05) & (df["q_bag_eff"] > 0.05), "category"] = "pair-only"
    df.loc[(df["q_pair"] > 0.05) & (df["q_bag_eff"] <= 0.05), "category"] = "bag-only"
    return df


def plot_quadrant(df, ax, clade, label_top_n=3):
    d = df[df["clade"] == clade]
    if d.empty:
        ax.axis("off")
        return
    xmax = max(d["mlog10_q_bag"].max(), 2.5)
    ymax = max(d["mlog10_q_pair"].max(), 2.5)
    thr = -math.log10(0.05)
    ax.add_patch(Rectangle((thr, thr), xmax - thr, ymax - thr,
                           color=CATEGORY_COLORS["both"], alpha=0.08, zorder=0))
    ax.add_patch(Rectangle((0, thr), thr, ymax - thr,
                           color=CATEGORY_COLORS["pair-only"], alpha=0.08, zorder=0))
    ax.add_patch(Rectangle((thr, 0), xmax - thr, thr,
                           color=CATEGORY_COLORS["bag-only"], alpha=0.08, zorder=0))
    ax.axhline(thr, color="black", lw=0.4, ls=":")
    ax.axvline(thr, color="black", lw=0.4, ls=":")
    for cat, color in CATEGORY_COLORS.items():
        m = d["category"] == cat
        if m.any():
            ax.scatter(d.loc[m, "mlog10_q_bag"], d.loc[m, "mlog10_q_pair"],
                       s=14, alpha=0.8, color=color, edgecolors="none",
                       label=cat if cat != "neither" else None)
    pair_only = d[d["category"] == "pair-only"].sort_values(
        "fold_pair_co", ascending=False).head(label_top_n)
    for _, r in pair_only.iterrows():
        name = r.get("go_name", "") or r["go_id"]
        if isinstance(name, str) and len(name) > 28:
            name = name[:28] + "…"
        ax.annotate(name, xy=(r["mlog10_q_bag"], r["mlog10_q_pair"]),
                    xytext=(4, 2), textcoords="offset points",
                    fontsize=6, color=CATEGORY_COLORS["pair-only"])
    ax.set_xlim(0, xmax * 1.05)
    ax.set_ylim(0, ymax * 1.05)
    ax.set_title(f"{clade}  (n={len(d)})", fontsize=9)
    ax.tick_params(labelsize=7)


def page_quadrant(df, pdf):
    clades = sorted(df["clade"].unique())
    clade_counts = (df[df["category"] == "pair-only"]
                    .groupby("clade").size().sort_values(ascending=False))
    ordered = (list(clade_counts.index) +
               [c for c in clades if c not in clade_counts.index])
    ncols = 4
    nrows = (len(ordered) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(3.2 * ncols, 3.0 * nrows),
                             squeeze=False)
    for i, c in enumerate(ordered):
        plot_quadrant(df, axes[i // ncols][i % ncols], c)
    for i in range(nrows):
        axes[i][0].set_ylabel("-log10(q) pair-co-enrichment", fontsize=8)
    for j in range(ncols):
        axes[nrows - 1][j].set_xlabel("-log10(q) bag-of-genes", fontsize=8)
    for k in range(len(ordered), nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")
    handles = [
        plt.Line2D([0], [0], marker='o', ls='',
                   color=CATEGORY_COLORS[c], label=lbl)
        for c, lbl in (("both", "both significant"),
                       ("pair-only", "pair-co only (novel)"),
                       ("bag-only", "bag-of-genes only"),
                       ("neither", "neither"))
    ]
    fig.legend(handles=handles, loc="lower right",
               bbox_to_anchor=(0.99, 0.01), fontsize=9,
               title="q ≤ 0.05 in:", title_fontsize=9, frameon=True)
    fig.suptitle("Pair co-enrichment vs bag-of-genes: per-clade q-q scatter",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    pdf.savefig(fig)
    plt.close(fig)


def page_expected_observed(df, pdf):
    fig, ax = plt.subplots(figsize=(7, 6))
    xs = df["expected_k_co"].to_numpy()
    ys = df["k_co"].to_numpy()
    xs_safe = np.maximum(xs, 1e-4)
    for cat in ("neither", "bag-only", "both", "pair-only"):
        m = df["category"] == cat
        if m.any():
            ax.scatter(xs_safe[m.to_numpy()], ys[m.to_numpy()],
                       s=14, alpha=0.7, color=CATEGORY_COLORS[cat],
                       edgecolors="none",
                       label=cat if cat != "neither" else None)
    lo = max(1e-4, min(xs_safe.min(), ys.min() if ys.min() > 0 else 1))
    hi = max(xs_safe.max(), ys.max())
    xs_ref = np.logspace(np.log10(lo), np.log10(hi), 100)
    ax.plot(xs_ref, xs_ref, ls="--", color="black", lw=0.7, label="y = x (null)")
    ax.plot(xs_ref, 3 * xs_ref, ls=":", color="grey", lw=0.5, label="3×, 10×")
    ax.plot(xs_ref, 10 * xs_ref, ls=":", color="grey", lw=0.5)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Expected k_co = n_pairs · p(fam)² (independence null)")
    ax.set_ylabel("Observed k_co (pairs with both partners carrying term X)")
    ax.set_title("Pair co-enrichment: observed vs expected co-occurrence")
    ax.legend(loc="upper left", fontsize=9, title="q ≤ 0.05 in:")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def page_category_bars(df, pdf):
    cats = ["both", "pair-only", "bag-only"]
    counts = (df[df["category"].isin(cats)]
              .groupby(["clade", "category"]).size().unstack(fill_value=0))
    for c in cats:
        if c not in counts.columns:
            counts[c] = 0
    counts = counts[cats]
    counts["total"] = counts.sum(axis=1)
    counts = counts.sort_values("total", ascending=True)
    fig, ax = plt.subplots(figsize=(8, max(3, 0.25 * len(counts))))
    y = np.arange(len(counts))
    left = np.zeros(len(counts))
    for c in cats:
        vals = counts[c].to_numpy()
        ax.barh(y, vals, left=left, color=CATEGORY_COLORS[c],
                label=c, edgecolor="none")
        left = left + vals
    ax.set_yticks(y)
    ax.set_yticklabels(counts.index, fontsize=8)
    ax.set_xlabel("# GO terms at q ≤ 0.05")
    ax.set_title("Per-clade breakdown: where did the signal come from?")
    ax.legend(loc="lower right", fontsize=9, title="q ≤ 0.05 in:")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def run(in_path, out_path) -> None:
    df = load(in_path)
    print(f"loaded {len(df)} (clade, term) rows from {in_path}")
    with PdfPages(out_path) as pdf:
        page_quadrant(df, pdf)
        page_expected_observed(df, pdf)
        page_category_bars(df, pdf)
    print(f"wrote {out_path}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="egt go plot-pair-coenrich",
        description="Summary PDF for pair-coenrichment output.",
    )
    ap.add_argument("--in", dest="in_path", required=True,
                    help="pair_co_vs_bag.tsv.gz")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    run(args.in_path, args.out)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
