"""Volcano plots of GO-enrichment results from `significant_terms.tsv`.

x = log2(fold_enrichment), y = -log10(q). One page per clade, three
subplots (stability / closeness / intersection). Each dot is a
(N_threshold, GO-term) config labeled with its N; colored by GO
namespace (BP/MF/CC).
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


NS_COLOR = {"BP": "#1f77b4", "MF": "#ff7f0e", "CC": "#2ca02c", "?": "#999999"}


def log10_safe(q) -> float:
    try:
        qf = float(q)
    except (TypeError, ValueError):
        return float("nan")
    if qf <= 0:
        return 300.0
    return -math.log10(qf)


def log2_safe(f) -> float:
    try:
        ff = float(f)
    except (TypeError, ValueError):
        return float("nan")
    if not math.isfinite(ff) or ff <= 0:
        return float("nan")
    return math.log2(ff)


def _clade_panel(ax, sub: pd.DataFrame, clade: str, axis: str, col: int) -> None:
    s = sub[sub["axis"] == axis]
    if s.empty:
        ax.set_title(f"{clade} — {axis}  (no sig terms)")
        ax.set_xlabel("log2 fold-enrichment")
        if col == 0:
            ax.set_ylabel("-log10(q)")
        return
    for ns in ("BP", "MF", "CC", "?"):
        t = s[s["go_namespace"] == ns]
        if t.empty:
            continue
        ax.scatter(t["log2fold"], t["mlog10q"],
                   c=NS_COLOR[ns], s=28, alpha=0.7,
                   edgecolors="white", linewidths=0.5,
                   label=ns if ns != "?" else None, zorder=3)
    for _, r in s.iterrows():
        ax.annotate(str(int(r["N_threshold"])),
                    xy=(r["log2fold"], r["mlog10q"]),
                    xytext=(3, 2), textcoords="offset points",
                    fontsize=5.5, color="#333", alpha=0.8, zorder=4)
    ax.axhline(-math.log10(0.05), ls="--", color="red", lw=0.6,
               label="q=0.05" if col == 0 else None)
    ax.axhline(-math.log10(0.25), ls=":", color="orange", lw=0.6,
               label="q=0.25" if col == 0 else None)
    ax.axvline(math.log2(3), ls=":", color="#888", lw=0.6,
               label="fold=3×" if col == 0 else None)
    ax.axvline(0, color="black", lw=0.4, alpha=0.4)
    ax.set_title(f"{clade} — {axis}  (n={len(s)})")
    ax.set_xlabel("log2 fold-enrichment")
    if col == 0:
        ax.set_ylabel("-log10(q)")
        ax.legend(fontsize=7, loc="upper left")


def run(significant_terms_path, out_path, min_fold: float | None = None) -> None:
    df = pd.read_csv(significant_terms_path, sep="\t")
    df = df[df["sweep_namespace"] == "all"].copy()
    df = df.sort_values("q").drop_duplicates(
        subset=["clade", "axis", "go_id"], keep="first"
    )
    df["log2fold"] = df["fold"].map(log2_safe)
    df["mlog10q"] = df["q"].map(log10_safe)
    df = df.dropna(subset=["log2fold", "mlog10q"])
    if min_fold is not None:
        df = df[df["fold"] >= min_fold]

    clades = sorted(df["clade"].dropna().unique())
    axes_order = ("stability", "closeness", "intersection")

    with PdfPages(out_path) as pdf:
        for clade in clades:
            sub = df[df["clade"] == clade]
            if sub.empty:
                continue
            fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
            for col, axis in enumerate(axes_order):
                _clade_panel(axes[col], sub, clade, axis, col)
            fig.suptitle(clade, fontsize=12, y=1.02)
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
    print(f"[write] {out_path}  clades_plotted={len(clades)}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="egt go plot-volcano",
        description="Volcano plot per clade from significant_terms.tsv.",
    )
    ap.add_argument("--significant-terms", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-fold", type=float, default=None,
                    help="optional fold-enrichment floor (e.g. 3.0)")
    args = ap.parse_args(argv)
    run(args.significant_terms, args.out, args.min_fold)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
