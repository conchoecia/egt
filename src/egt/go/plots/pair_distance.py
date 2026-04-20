"""Per-clade pair-distance scatter (log10 mean_in vs log10 mean_out).

For each clade, all pairs are plotted as faint background dots; pairs
landing in the top-N foreground for each ranking axis (stability /
closeness / intersection) are highlighted in red. The intersection
panel uses the literal overlap of the other two foregrounds.

best_N is picked as the lowest-q `sweep_namespace="all"` cell whose
top term has fold >= `min_fold` (default 3.0) to suppress the
large-N hypergeometric rebound artifact.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from ..io import load_sweep_summary, load_unique_pairs


OCCUPANCY_MIN = 0.5


def pick_best_N(
    summary: pd.DataFrame, clade: str, axis: str, min_fold: float = 3.0,
):
    """(N, top_q, top_term, fold) or None."""
    s = summary[(summary["clade"] == clade) & (summary["axis"] == axis)
                & (summary["namespace"] == "all")]
    s = s.dropna(subset=["top_q", "top_term_fold"])
    s = s[s["top_term_fold"] >= min_fold]
    if s.empty:
        return None
    r = s.sort_values("top_q").iloc[0]
    return (int(r["N_threshold"]), float(r["top_q"]),
            str(r["top_term"]), float(r["top_term_fold"]))


def _panel(ax, panel_info, log_mean_in, log_mean_out, sub_index, lims, axis, N_stab, N_clos):
    N_used, best_q, top_term, top_fold, fg_idx = panel_info[axis]
    fg_mask = sub_index.isin(fg_idx)
    ax.scatter(log_mean_in[~fg_mask], log_mean_out[~fg_mask],
               c="#cccccc", s=5, alpha=0.25, edgecolors="none",
               label=f"other pairs (n={int((~fg_mask).sum())})", zorder=2)
    if axis == "intersection":
        fg_label = f"top-{N_stab}-stable ∩ top-{N_clos}-close ({fg_mask.sum()} pairs)"
    else:
        fg_label = f"top-{N_used} ({fg_mask.sum()} pairs)"
    ax.scatter(log_mean_in[fg_mask], log_mean_out[fg_mask],
               c="#d62728", s=14, alpha=0.8, edgecolors="white",
               linewidths=0.3, label=fg_label, zorder=3)
    lim_lo, lim_hi = lims
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], ls=":", color="black",
            lw=0.6, alpha=0.6, label="y = x")
    ax.set_xlim(lim_lo, lim_hi)
    ax.set_ylim(lim_lo, lim_hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("log10 mean_in")
    if axis == "intersection":
        title = (f"intersection\npairs in both top-{N_stab}-stable "
                 f"and top-{N_clos}-close")
    else:
        q_label = f"q={best_q:.2e}" if best_q and best_q > 0 else "q=0"
        title = (f"{axis}\nbest_N={N_used}  top={top_term}  "
                 f"fold={top_fold:.1f}×  {q_label}")
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(alpha=0.15)


def run(supp_table_path, summary_path, out_path) -> None:
    supp = load_unique_pairs(supp_table_path)
    summary = load_sweep_summary(summary_path)
    clades = sorted(supp["nodename"].dropna().unique())
    axes_order = ("stability", "closeness", "intersection")

    with PdfPages(out_path) as pdf:
        for clade in clades:
            sub = supp[supp["nodename"] == clade].copy()
            sub = sub[sub["occupancy_in"].fillna(0) >= OCCUPANCY_MIN]
            sub = sub.dropna(subset=["mean_in", "mean_out",
                                     "sd_in_out_ratio_log_sigma",
                                     "mean_in_out_ratio_log_sigma"])
            sub = sub.reset_index(drop=True)
            if sub.empty:
                continue
            log_mean_in = np.log10(sub["mean_in"].clip(lower=1))
            log_mean_out = np.log10(sub["mean_out"].clip(lower=1))
            best_stab = pick_best_N(summary, clade, "stability")
            best_clos = pick_best_N(summary, clade, "closeness")
            if best_stab is None or best_clos is None:
                continue
            N_stab, N_clos = best_stab[0], best_clos[0]
            stab_idx = set(sub.sort_values("sd_in_out_ratio_log_sigma")
                           .index[:N_stab].tolist())
            clos_idx = set(sub.sort_values("mean_in_out_ratio_log_sigma")
                           .index[:N_clos].tolist())
            inter_set = stab_idx & clos_idx
            panel_info = {
                "stability": (N_stab, best_stab[1], best_stab[2], best_stab[3],
                              pd.Index(sorted(stab_idx))),
                "closeness": (N_clos, best_clos[1], best_clos[2], best_clos[3],
                              pd.Index(sorted(clos_idx))),
                "intersection": (None, None, None, None,
                                 pd.Index(sorted(inter_set))),
            }
            fg_any = stab_idx | clos_idx | inter_set
            if fg_any:
                fg_in = log_mean_in.iloc[sorted(fg_any)]
                fg_out = log_mean_out.iloc[sorted(fg_any)]
                fg_lo = float(min(fg_in.min(), fg_out.min()))
                fg_hi = float(max(fg_in.max(), fg_out.max()))
            else:
                fg_lo = float("inf")
                fg_hi = float("-inf")
            p1 = min(float(np.percentile(log_mean_in, 1)),
                     float(np.percentile(log_mean_out, 1)), fg_lo)
            p99 = max(float(np.percentile(log_mean_in, 99)),
                      float(np.percentile(log_mean_out, 99)), fg_hi)
            span = p99 - p1
            lims = (p1 - 0.05 * span, p99 + 0.05 * span)

            fig, axes = plt.subplots(1, 3, figsize=(15, 5.5),
                                     sharex=True, sharey=True)
            for col, axis in enumerate(axes_order):
                _panel(axes[col], panel_info, log_mean_in, log_mean_out,
                       sub.index, lims, axis, N_stab, N_clos)
            axes[0].set_ylabel("log10 mean_out")
            fig.suptitle(clade, fontsize=13, y=1.02)
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
    print(f"[write] {out_path}  clades_plotted={len(clades)}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="egt go plot-pair-distance",
        description="Per-clade pair-distance scatter with best-cell overlays.",
    )
    ap.add_argument("--supp-table", required=True)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    run(args.supp_table, args.summary, args.out)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
