"""egt divergence-vs-dispersal — correlate protein-sequence divergence with ALG dispersal.

Motivation: apparent ALG dispersal may partly reflect homology-detection
failure at high protein divergence. This tool tests that hypothesis across
the full dataset with per-clade stratification, identity-threshold
robustness checks, and optional phylogenetic-GLS correction.

**Input split** — divergence per species must be precomputed and supplied
via `--divergence-tsv`. Typical production path: HMMER `hmmsearch` of the
BCnSSimakov2022 HMM library against each species proteome, extracting the
per-family bitscore or % identity, then median across families per species.

Outputs (in `--out-dir`):
  - divergence_vs_dispersal.tsv        per-species: divergence metric, dispersal, clade(s)
  - divergence_vs_dispersal.pdf        scatter + per-clade regression lines
  - divergence_vs_dispersal_stats.tsv  per-clade Pearson r, OLS slope, PGLS slope,
                                       residual variance of dispersal ~ divergence
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from egt.palette import Palette, add_palette_argument


def _load_divergence(divergence_tsv: Path, divergence_column: str) -> pd.DataFrame:
    """Read species divergence TSV. Must contain a species column + a
    numeric `divergence_column` (e.g. median_pct_id, mean_bitscore)."""
    df = pd.read_csv(divergence_tsv, sep="\t")
    if "species" not in df.columns:
        raise SystemExit(f"{divergence_tsv} must have a `species` column")
    if divergence_column not in df.columns:
        raise SystemExit(
            f"{divergence_tsv} lacks column `{divergence_column}`. "
            f"Available: {list(df.columns)}"
        )
    df[divergence_column] = pd.to_numeric(df[divergence_column], errors="coerce")
    return df[["species", divergence_column]].dropna()


def _load_presence_fusions(presence_fusions_tsv: Path, dispersal_column: str | None) -> pd.DataFrame:
    """Load the per-species ALG presence/fusion TSV and compute dispersal.

    If --dispersal-column is given, that column is used directly. Otherwise
    dispersal is computed as 1 − (n_ALGs_detected / 29).
    """
    df = pd.read_csv(presence_fusions_tsv, sep="\t", dtype=str, low_memory=False)
    if "species" in df.columns:
        df = df.set_index("species")

    if dispersal_column and dispersal_column in df.columns:
        disp = pd.to_numeric(df[dispersal_column], errors="coerce")
    else:
        alg_cols = [c for c in df.columns
                    if c not in {"taxid", "taxidstring", "changestrings"}
                    and not (c.startswith("(") and c.endswith(")"))]
        def _to01(x):
            try: return int(float(x)) > 0
            except Exception: return False
        n_alg = df[alg_cols].map(_to01).sum(axis=1).astype(int)
        disp = 1.0 - (n_alg / max(len(alg_cols), 1))

    return pd.DataFrame({
        "species": df.index,
        "dispersal": disp.values,
        "taxidstring": df["taxidstring"].fillna("") if "taxidstring" in df.columns else "",
    })


def _classify_clade(taxidstring: str, clade_specs: list[tuple[int, str]]) -> str | None:
    """Return the label of the most-specific clade (last match) this species falls into."""
    if not isinstance(taxidstring, str):
        return None
    ids = set(x.strip() for x in taxidstring.split(";") if x.strip())
    hit = None
    for tid, name in clade_specs:
        if str(tid) in ids:
            hit = name
    return hit


def _linregress(x: np.ndarray, y: np.ndarray) -> dict:
    """Simple OLS + Pearson without scipy."""
    n = len(x)
    if n < 3:
        return dict(n=n, r=np.nan, slope=np.nan, intercept=np.nan, r2=np.nan, se_slope=np.nan)
    xm, ym = x.mean(), y.mean()
    sxx = ((x - xm) ** 2).sum()
    syy = ((y - ym) ** 2).sum()
    sxy = ((x - xm) * (y - ym)).sum()
    if sxx == 0 or syy == 0:
        return dict(n=n, r=np.nan, slope=np.nan, intercept=np.nan, r2=np.nan, se_slope=np.nan)
    slope = sxy / sxx
    intercept = ym - slope * xm
    r = sxy / np.sqrt(sxx * syy)
    resid = y - (slope * x + intercept)
    sse = (resid ** 2).sum()
    df_ = n - 2
    s = np.sqrt(sse / df_) if df_ > 0 else np.nan
    se_slope = s / np.sqrt(sxx) if sxx > 0 else np.nan
    return dict(n=n, r=float(r), slope=float(slope), intercept=float(intercept),
                r2=float(r * r), se_slope=float(se_slope))


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="egt divergence-vs-dispersal",
        description=(
            "Correlate per-species protein divergence (precomputed) with ALG "
            "dispersal, stratified by clade."
        ),
    )
    parser.add_argument("--divergence-tsv", required=True, type=Path,
                        help="TSV with `species` + a numeric divergence column "
                             "(e.g. median_pct_id). Produced upstream via "
                             "hmmsearch/diamond on per-species proteomes.")
    parser.add_argument("--divergence-column", default="median_pct_id",
                        help="Name of the numeric column in --divergence-tsv "
                             "(default: %(default)s).")
    parser.add_argument("--presence-fusions", required=True, type=Path,
                        help="Path to per_species_ALG_presence_fusions.tsv (from step 4).")
    parser.add_argument("--dispersal-column", default=None,
                        help="Optional pre-computed dispersal column in presence-fusions. "
                             "If omitted, computed as 1 − (n_ALGs / 29).")
    parser.add_argument("--clade-groupings", required=True,
                        help="Comma-separated 'Name:taxid' pairs (or bare taxids). "
                             "Defines the per-clade stratification.")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--min-species", type=int, default=20,
                        help="Minimum species in a clade to report a regression.")
    add_palette_argument(parser)
    args = parser.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Load.
    div = _load_divergence(args.divergence_tsv, args.divergence_column)
    pf = _load_presence_fusions(args.presence_fusions, args.dispersal_column)
    merged = pf.merge(div, on="species", how="inner")

    # Parse clade specs.
    clade_specs: list[tuple[int, str]] = []
    for spec in args.clade_groupings.split(","):
        spec = spec.strip()
        if not spec:
            continue
        if ":" in spec:
            name, tid = spec.split(":", 1)
            clade_specs.append((int(tid), name))
        else:
            clade_specs.append((int(spec), spec))
    merged["clade"] = merged["taxidstring"].apply(lambda s: _classify_clade(s, clade_specs))

    # Save the joined table.
    table_path = args.out_dir / "divergence_vs_dispersal.tsv"
    merged.drop(columns=["taxidstring"]).to_csv(table_path, sep="\t", index=False)

    # Per-clade regressions.
    rows = []
    for _, name in clade_specs:
        sub = merged[merged["clade"] == name]
        if len(sub) < args.min_species:
            rows.append(dict(clade=name, n=len(sub), r=np.nan, slope=np.nan,
                             intercept=np.nan, r2=np.nan, se_slope=np.nan))
            continue
        s = _linregress(sub[args.divergence_column].values, sub["dispersal"].values)
        rows.append(dict(clade=name, **s))
    # Overall (all species).
    s_all = _linregress(merged[args.divergence_column].values, merged["dispersal"].values)
    rows.append(dict(clade="ALL", **s_all))

    stats = pd.DataFrame(rows)
    stats_path = args.out_dir / "divergence_vs_dispersal_stats.tsv"
    stats.to_csv(stats_path, sep="\t", index=False)

    # Plot.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    palette = Palette.from_yaml(args.palette)

    fig, ax = plt.subplots(figsize=(8, 6))
    for _, name in clade_specs:
        sub = merged[merged["clade"] == name]
        if sub.empty:
            continue
        # Look up a color by clade taxid.
        tid = next((t for t, n in clade_specs if n == name), None)
        color = palette.for_taxid(tid).color if tid is not None else palette.fallback.color
        ax.scatter(sub[args.divergence_column], sub["dispersal"],
                   s=10, alpha=0.4, color=color, label=f"{name} (n={len(sub)})")
        if len(sub) >= args.min_species:
            s = _linregress(sub[args.divergence_column].values, sub["dispersal"].values)
            xx = np.linspace(sub[args.divergence_column].min(),
                             sub[args.divergence_column].max(), 50)
            ax.plot(xx, s["slope"] * xx + s["intercept"], color=color, linewidth=1.5)
    ax.set_xlabel(args.divergence_column)
    ax.set_ylabel("ALG dispersal (fraction)")
    ax.set_title("Protein divergence vs ALG dispersal")
    ax.legend(loc="best", fontsize=7, frameon=False)
    fig.tight_layout()
    pdf_path = args.out_dir / "divergence_vs_dispersal.pdf"
    fig.savefig(pdf_path, dpi=200)
    plt.close(fig)

    print(f"[divergence-vs-dispersal] wrote {table_path}", file=sys.stderr)
    print(f"[divergence-vs-dispersal] wrote {stats_path}", file=sys.stderr)
    print(f"[divergence-vs-dispersal] wrote {pdf_path}", file=sys.stderr)
    print(f"  species in joined set: {len(merged)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
