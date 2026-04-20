"""Per-clade N-sweep GO enrichment driver.

Given a defining-pair table, a family → human-gene map, and NCBI GO
references, compute GO-term enrichment for every
`(clade, axis, N_threshold, namespace)` sweep cell.

Axes:
  - "stability"    — pairs ranked by sd_in_out_ratio_log_sigma (ascending)
  - "closeness"    — pairs ranked by mean_in_out_ratio_log_sigma (ascending)
  - "intersection" — top-N under both above

For each axis, N is drawn from a per-clade logarithmic grid anchored at
N_MIN = 10 up to the clade row count. The floor at N = 10 matches the
point where hypergeometric power under BH ramps up.

Outputs written under `out_dir`:
  per_clade/{clade}.tsv            — per-cell summary rows
  summary.tsv                      — concatenated per-cell summary
  significant_terms.tsv            — every term with q <= 0.25 across all cells
  term_gene_lists.tsv.gz           — foreground GeneIDs per (clade, axis, N, go_id)
  gene_symbols.tsv                 — GeneID → Symbol side-car
  curves.pdf                       — -log10(top q) vs N, per clade × axis
"""
from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from .enrichment import enrich_for_foreground, reduce_pairs_to_geneset
from .io import (
    build_family_gene_annotations,
    load_unique_pairs,
    parse_family_map,
    parse_gene2accession,
    parse_gene2go,
)


OCCUPANCY_MIN = 0.5
N_MIN = 10
NUM_POINTS = 8


def _axis_orders(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    stab = df.sort_values("sd_in_out_ratio_log_sigma").index.to_numpy()
    close = df.sort_values("mean_in_out_ratio_log_sigma").index.to_numpy()
    return stab, close


def _prepare_clade(clade_rows: pd.DataFrame) -> pd.DataFrame:
    df = clade_rows.copy()
    df = df[df["occupancy_in"].fillna(0) >= OCCUPANCY_MIN]
    df = df.dropna(
        subset=["sd_in_out_ratio_log_sigma", "mean_in_out_ratio_log_sigma"]
    )
    return df.reset_index(drop=True)


def _n_grid(n_rows: int) -> list[int]:
    if n_rows < N_MIN:
        return [n_rows]
    return np.unique(
        np.geomspace(N_MIN, n_rows, num=NUM_POINTS).round().astype(int)
    ).tolist()


def _foreground_indices(axis: str, N: int, stab: np.ndarray, close: np.ndarray) -> set[int]:
    if axis == "stability":
        return set(stab[:N].tolist())
    if axis == "closeness":
        return set(close[:N].tolist())
    return set(stab[:N].tolist()) & set(close[:N].tolist())


def sweep_clade(
    clade_rows: pd.DataFrame,
    fam_to_genes: Mapping[str, set[str]],
    background_to_terms: Mapping[str, set[str]],
    term_namespace: Mapping[str, str],
) -> tuple[list[dict], dict]:
    """Run the per-cell enrichment sweep for a single clade.

    Returns (records, curve_data). `records` is a list of dicts, one per
    (axis, N, namespace) cell. `curve_data` is a dict keyed by
    (axis, namespace) with lists of (N, -log10(top_q)) points for the
    curves.pdf figure. -log10(0) is rendered as a 300.0 sentinel to keep
    the axis finite.
    """
    df = _prepare_clade(clade_rows)
    if df.empty:
        return [], {}
    stab, close = _axis_orders(df)
    n_grid = _n_grid(len(df))

    records: list[dict] = []
    curve_data: dict = defaultdict(list)
    for N in n_grid:
        for axis in ("stability", "closeness", "intersection"):
            idxs = _foreground_indices(axis, N, stab, close)
            if not idxs:
                continue
            sub = df.loc[list(idxs)]
            families = pd.concat(
                [sub["ortholog1"], sub["ortholog2"]]
            ).dropna().unique()
            foreground = reduce_pairs_to_geneset(families, fam_to_genes)
            if not foreground:
                continue
            by_ns = enrich_for_foreground(
                foreground, background_to_terms, term_namespace
            )
            for ns in ("all", "BP", "MF", "CC"):
                res = by_ns.get(ns, [])
                n_q05 = sum(1 for r in res if r["q"] <= 0.05)
                n_q25 = sum(1 for r in res if r["q"] <= 0.25)
                top = res[0] if res else None
                top_q = top["q"] if top else float("nan")
                records.append(dict(
                    axis=axis,
                    N_threshold=N,
                    pairs_used=len(idxs),
                    namespace=ns,
                    foreground_size=len(foreground),
                    n_families=len(families),
                    n_terms_tested=len(res),
                    n_hits_q05=n_q05,
                    n_hits_q25=n_q25,
                    top_term=top["go_id"] if top else "",
                    top_term_fold=top["fold"] if top else float("nan"),
                    top_term_k=top["k"] if top else 0,
                    top_term_K=top["K"] if top else 0,
                    top_q=top_q,
                ))
                if not math.isnan(top_q) and top_q > 0:
                    curve_data[(axis, ns)].append((N, -math.log10(top_q)))
                elif top and top_q == 0:
                    curve_data[(axis, ns)].append((N, 300.0))
    return records, dict(curve_data)


def harvest_significant_terms(
    clade: str,
    clade_rows: pd.DataFrame,
    records: list[dict],
    fam_to_genes: Mapping[str, set[str]],
    background_to_terms: Mapping[str, set[str]],
    term_namespace: Mapping[str, str],
    go_names: Mapping[str, str] | None = None,
    gene_to_symbol: Mapping[str, str] | None = None,
) -> tuple[list[dict], list[dict]]:
    """For every (axis, N) cell that produced q25 hits, emit every term.

    Each emitted row follows the publication-standard GO-enrichment
    schema (see docs of `run()`): go_id, go_name, namespace, k, n, K, N,
    ratio_in_study, ratio_in_pop, fold, p, correction_method, q, and the
    inline gene_ids (semicolon-joined) + gene_symbols (if a gene→symbol
    map is supplied).

    Also emits a compact (clade, axis, N, go_id) → gene_ids side-car
    list used downstream by the Jaccard-based plot builders.
    """
    out: list[dict] = []
    gene_lists: list[dict] = []
    df = _prepare_clade(clade_rows)
    if df.empty:
        return out, gene_lists
    stab, close = _axis_orders(df)
    go_names = go_names or {}
    gene_to_symbol = gene_to_symbol or {}

    for axis in ("stability", "closeness", "intersection"):
        cells = sorted({
            r["N_threshold"] for r in records
            if r["axis"] == axis and r["namespace"] == "all"
            and r["n_hits_q25"] > 0
        })
        for N in cells:
            idxs = _foreground_indices(axis, N, stab, close)
            if not idxs:
                continue
            s2 = df.loc[list(idxs)]
            families = pd.concat(
                [s2["ortholog1"], s2["ortholog2"]]
            ).dropna().unique()
            fg = reduce_pairs_to_geneset(families, fam_to_genes)
            fg_in_bg = {g for g in fg if g in background_to_terms}
            ns_res = enrich_for_foreground(
                fg, background_to_terms, term_namespace
            )
            seen_terms_this_cell: set[str] = set()
            for ns, rows in ns_res.items():
                for rr in rows:
                    if rr["q"] > 0.25:
                        continue
                    # Compute the gene set contributing to this term's
                    # hit count once per (axis, N, go_id) (it's identical
                    # across sweep namespaces at the same cell).
                    hit_genes = sorted(
                        g for g in fg_in_bg
                        if rr["go_id"] in background_to_terms[g]
                    )
                    hit_symbols = [gene_to_symbol.get(g, g) for g in hit_genes]
                    k_val = int(rr["k"]); n_val = int(rr["n"])
                    K_val = int(rr["K"]); N_val = int(rr["N"])
                    out.append(dict(
                        clade=clade, axis=axis, N_threshold=N,
                        sweep_namespace=ns,
                        go_id=rr["go_id"],
                        go_name=go_names.get(rr["go_id"], ""),
                        go_namespace=rr["term_namespace"],
                        k=k_val, n=n_val, K=K_val, N=N_val,
                        ratio_in_study=f"{k_val}/{n_val}",
                        ratio_in_pop=f"{K_val}/{N_val}",
                        fold=rr["fold"], p=rr["p"],
                        correction_method="fdr_bh", q=rr["q"],
                        gene_ids=";".join(hit_genes),
                        gene_symbols=";".join(hit_symbols),
                    ))
                    if rr["go_id"] in seen_terms_this_cell:
                        continue
                    seen_terms_this_cell.add(rr["go_id"])
                    gene_lists.append(dict(
                        clade=clade, axis=axis, N_threshold=N,
                        go_id=rr["go_id"],
                        k=k_val,
                        gene_ids=",".join(hit_genes),
                    ))
    return out, gene_lists


def _write_curves_pdf(all_curves: dict, out_path: Path) -> None:
    """Render the per-clade -log10(top_q) vs N curves into one PDF."""
    if not all_curves:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter, ScalarFormatter

    n_clades = len(all_curves)
    fig, axes = plt.subplots(
        n_clades, 3, figsize=(15, 3 * n_clades), squeeze=False
    )
    colors = {"all": "black", "BP": "C0", "MF": "C1", "CC": "C2"}
    for row, (clade, cdata) in enumerate(sorted(all_curves.items())):
        for col, axis in enumerate(("stability", "closeness", "intersection")):
            ax = axes[row][col]
            for ns in ("all", "BP", "MF", "CC"):
                pts = cdata.get((axis, ns), [])
                if not pts:
                    continue
                xs, ys = zip(*pts)
                ax.plot(xs, ys, marker="o", markersize=3, label=ns,
                        color=colors[ns])
            ax.axhline(-math.log10(0.05), ls="--", color="red", lw=0.5,
                       label="q=0.05" if row == 0 and col == 0 else None)
            ax.axhline(-math.log10(0.25), ls=":", color="orange", lw=0.5,
                       label="q=0.25" if row == 0 and col == 0 else None)
            ax.set_xscale("log")
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.set_title(f"{clade} — {axis}")
            if col == 0:
                ax.set_ylabel("-log10(top q)")
            if row == n_clades - 1:
                ax.set_xlabel("top-N threshold")
            if row == 0 and col == 0:
                ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def run(
    supp_table,
    family_map,
    gene2accession,
    gene2go,
    out_dir,
    obo=None,
    write_curves: bool = True,
    verbose: bool = True,
) -> Path:
    """End-to-end driver. Writes all sweep outputs under `out_dir`.

    Emits a `significant_terms.tsv` with one row per (clade, axis, N,
    sweep_namespace, go_id) enriched term, following the publication-
    standard GO-enrichment schema:

      clade, axis, N_threshold, sweep_namespace,
      go_id, go_name, go_namespace,
      k, n, K, N,                  # foreground/background counts
      ratio_in_study (k/n),        # string "k/n"
      ratio_in_pop (K/N),          # string "K/N"
      fold,                        # (k/n) / (K/N)
      p,                           # raw hypergeometric upper-tail
      correction_method,           # "fdr_bh"
      q,                           # BH-adjusted p-value
      gene_ids,                    # ";"-joined Entrez GeneIDs driving k
      gene_symbols                 # ";"-joined HGNC symbols (same order)

    When `obo` is supplied the `go_name` column is populated from the
    ontology file; otherwise it is empty. The `gene_symbols` column is
    populated from the NCBI `gene2accession` Symbol side-map.

    Returns the Path of `summary.tsv`.
    """
    out = Path(out_dir)
    (out / "per_clade").mkdir(parents=True, exist_ok=True)

    def _log(msg: str) -> None:
        if verbose:
            print(msg)

    _log(f"[load] supp table: {supp_table}")
    df = load_unique_pairs(supp_table)
    _log(f"  shape={df.shape}  clades={df['nodename'].nunique()}")

    _log(f"[load] gene2accession: {gene2accession}")
    prot_to_gene, gene_to_symbol = parse_gene2accession(gene2accession)
    _log(f"  protein-acc → GeneID rows={len(prot_to_gene)}  "
         f"gene-symbols={len(gene_to_symbol)}")

    _log(f"[load] family map: {family_map}")
    fam_to_genes, stats = parse_family_map(family_map, prot_to_gene)
    _log(f"  mapped_families={stats['n_families_mapped']}  stats={stats}")

    _log(f"[load] gene2go: {gene2go}")
    gene_to_terms_all, term_namespace = parse_gene2go(gene2go)
    _log(f"  GeneIDs-with-GO={len(gene_to_terms_all)}  "
         f"terms-in-namespace-map={len(term_namespace)}")

    go_names: dict[str, str] = {}
    if obo:
        _log(f"[load] OBO: {obo}")
        from .io import load_obo_names
        go_names, obo_ns = load_obo_names(obo)
        # Prefer OBO namespace where provided (authoritative).
        term_namespace = {**term_namespace, **obo_ns}
        _log(f"  OBO terms with name: {len(go_names)}")

    background_gene_ids, background_to_terms = build_family_gene_annotations(
        fam_to_genes, gene_to_terms_all
    )
    _log(f"  background_family_GeneIDs={len(background_gene_ids)}  "
         f"background_with_any_term={len(background_to_terms)}")

    all_records: list[dict] = []
    all_curves: dict = {}
    all_significant: list[dict] = []
    all_gene_lists: list[dict] = []
    for clade in sorted(df["nodename"].dropna().unique()):
        sub = df[df["nodename"] == clade]
        records, curves = sweep_clade(
            sub, fam_to_genes, background_to_terms, term_namespace
        )
        any_hit = any(r["n_hits_q25"] > 0 for r in records)
        _log(f"[clade] {clade}  rows={len(sub)}  "
             f"sweep_configs={len(records)}  any_q25_hit={any_hit}")
        if records:
            cdf = pd.DataFrame(records)
            cdf.insert(0, "clade", clade)
            cdf.to_csv(out / "per_clade" / f"{clade}.tsv",
                       sep="\t", index=False)
            all_records.extend(cdf.to_dict("records"))
        if curves:
            all_curves[clade] = curves
        sig_rows, gene_rows = harvest_significant_terms(
            clade, sub, records, fam_to_genes,
            background_to_terms, term_namespace,
            go_names=go_names,
            gene_to_symbol=gene_to_symbol,
        )
        all_significant.extend(sig_rows)
        all_gene_lists.extend(gene_rows)

    summary_path = out / "summary.tsv"
    if all_records:
        sdf = pd.DataFrame(all_records).sort_values(
            ["n_hits_q25", "n_hits_q05"], ascending=False
        )
        sdf.to_csv(summary_path, sep="\t", index=False)
        _log(f"[write] summary.tsv rows={len(sdf)}")

    sig_cols = [
        "clade", "axis", "N_threshold", "sweep_namespace",
        "go_id", "go_name", "go_namespace",
        "k", "n", "K", "N",
        "ratio_in_study", "ratio_in_pop",
        "fold", "p", "correction_method", "q",
        "gene_ids", "gene_symbols",
    ]
    if all_significant:
        sig_df = pd.DataFrame(all_significant).drop_duplicates(
            subset=["clade", "axis", "N_threshold",
                    "sweep_namespace", "go_id"]
        ).sort_values(["clade", "q"])
        # Emit in the canonical column order so downstream consumers
        # don't depend on dict-iteration order.
        sig_df = sig_df[[c for c in sig_cols if c in sig_df.columns]]
        sig_df.to_csv(out / "significant_terms.tsv",
                      sep="\t", index=False)
        _log(f"[write] significant_terms.tsv rows={len(sig_df)}")
    else:
        pd.DataFrame(columns=sig_cols).to_csv(
            out / "significant_terms.tsv", sep="\t", index=False
        )
        _log("[write] significant_terms.tsv (empty)")

    gl_cols = ["clade", "axis", "N_threshold", "go_id", "k", "gene_ids"]
    if all_gene_lists:
        gdf = pd.DataFrame(all_gene_lists).drop_duplicates(
            subset=["clade", "axis", "N_threshold", "go_id"]
        )
        gdf.to_csv(out / "term_gene_lists.tsv.gz",
                   sep="\t", index=False, compression="gzip")
        _log(f"[write] term_gene_lists.tsv.gz rows={len(gdf)}")
    else:
        pd.DataFrame(columns=gl_cols).to_csv(
            out / "term_gene_lists.tsv.gz",
            sep="\t", index=False, compression="gzip"
        )
        _log("[write] term_gene_lists.tsv.gz (empty)")

    sym_rows = [(g, gene_to_symbol[g]) for g in sorted(background_to_terms)
                if g in gene_to_symbol]
    pd.DataFrame(sym_rows, columns=["gene_id", "symbol"]).to_csv(
        out / "gene_symbols.tsv", sep="\t", index=False
    )
    _log(f"[write] gene_symbols.tsv rows={len(sym_rows)}")

    if write_curves and all_curves:
        _write_curves_pdf(all_curves, out / "curves.pdf")
        _log(f"[write] curves.pdf clades={len(all_curves)}")

    return summary_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="egt go sweep",
        description="Per-clade N-sweep GO enrichment.",
    )
    ap.add_argument("--supp-table", required=True)
    ap.add_argument("--family-map", required=True)
    ap.add_argument("--gene2accession", required=True)
    ap.add_argument("--gene2go", required=True)
    ap.add_argument("--obo", default=None,
                    help="go-basic.obo; populates go_name in the output "
                         "(optional but recommended)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--no-curves", action="store_true",
                    help="skip the curves.pdf figure")
    args = ap.parse_args(argv)
    run(
        supp_table=args.supp_table,
        family_map=args.family_map,
        gene2accession=args.gene2accession,
        gene2go=args.gene2go,
        out_dir=args.out_dir,
        obo=args.obo,
        write_curves=not args.no_curves,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
