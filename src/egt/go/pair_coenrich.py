"""Pair-level co-enrichment test (binomial-null).

For a single (clade, axis, N) cell, for each GO term X ask: are the
foreground pairs enriched for pairs whose **both** partners' families
carry X, vs a binomial null in which pair co-carriage is `p(X)^2` under
independent family draws?

Per-family carriage is the union of GO terms across that family's
mapped GeneIDs.

This is complementary to the bag-of-genes hypergeometric: a term that
appears in the gene pool but whose carriers are not specifically
partnered will show up in bag-of-genes but not here.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from .io import (
    load_obo_names,
    load_unique_pairs,
    parse_family_map,
    parse_gene2accession,
    parse_gene2go,
)
from .stats import bh_qvalues, binom_sf


OCCUPANCY_MIN = 0.5
MIN_CO_HITS = 2


def family_term_table(
    fam_to_genes: Mapping[str, set[str]],
    gene_to_terms: Mapping[str, set[str]],
) -> dict[str, set[str]]:
    """{family: union of GO terms across its mapped GeneIDs}.

    Families with no annotated gene are omitted.
    """
    out: dict[str, set[str]] = {}
    for fam, genes in fam_to_genes.items():
        terms: set[str] = set()
        for g in genes:
            terms |= gene_to_terms.get(g, set())
        if terms:
            out[fam] = terms
    return out


def best_cell_per_clade(summary_path) -> dict[str, dict]:
    """Pull the `(axis, N)` cell with the smallest top_q per clade from
    summary.tsv, using namespace='all'."""
    from .io import load_sweep_summary
    sdf = load_sweep_summary(summary_path)
    sdf = sdf[(sdf["namespace"] == "all") & sdf["top_q"].notna()]
    best = (sdf.sort_values("top_q")
               .drop_duplicates(subset=["clade"])
               .set_index("clade"))
    return {c: dict(axis=row["axis"], N=int(row["N_threshold"]))
            for c, row in best.iterrows()}


def _select_pair_indices(
    df: pd.DataFrame, axis: str, N: int
) -> list[int]:
    stab = df.sort_values("sd_in_out_ratio_log_sigma").index.to_numpy()
    close = df.sort_values("mean_in_out_ratio_log_sigma").index.to_numpy()
    if axis == "stability":
        return stab[:N].tolist()
    if axis == "closeness":
        return close[:N].tolist()
    return list(set(stab[:N].tolist()) & set(close[:N].tolist()))


def pair_coenrich_for_clade(
    clade_rows: pd.DataFrame,
    axis: str,
    N: int,
    fam_to_terms: Mapping[str, set[str]],
    term_namespace: Mapping[str, str],
) -> list[dict]:
    """One-cell binomial-null pair co-enrichment test.

    Returns per-term rows sorted ascending by q. Terms with fewer than
    `MIN_CO_HITS` observed co-occurrences are not tested.
    """
    df = clade_rows.copy()
    df = df[df["occupancy_in"].fillna(0) >= OCCUPANCY_MIN]
    df = df.dropna(subset=["sd_in_out_ratio_log_sigma",
                           "mean_in_out_ratio_log_sigma",
                           "ortholog1", "ortholog2"])
    df = df.reset_index(drop=True)
    if df.empty:
        return []

    idxs = _select_pair_indices(df, axis, N)
    if not idxs:
        return []
    sub = df.loc[idxs, ["ortholog1", "ortholog2"]]

    mask = sub["ortholog1"].isin(fam_to_terms) & sub["ortholog2"].isin(fam_to_terms)
    pairs = sub[mask].to_numpy().tolist()
    n_pairs = len(pairs)
    if n_pairs == 0:
        return []

    k_co: dict[str, int] = defaultdict(int)
    k_either: dict[str, int] = defaultdict(int)
    for f1, f2 in pairs:
        t1 = fam_to_terms.get(f1, set())
        t2 = fam_to_terms.get(f2, set())
        for t in t1 & t2:
            k_co[t] += 1
        for t in t1 | t2:
            k_either[t] += 1

    N_fams = len(fam_to_terms)
    K_fams: dict[str, int] = defaultdict(int)
    for _, terms in fam_to_terms.items():
        for t in terms:
            K_fams[t] += 1

    out: list[dict] = []
    terms_to_test = [t for t, k in k_co.items() if k >= MIN_CO_HITS]
    pvals: list[float] = []
    for t in terms_to_test:
        K = K_fams[t]
        p_fam = K / N_fams if N_fams else 0.0
        p_co = p_fam ** 2
        kco = k_co[t]
        expected = n_pairs * p_co
        pval = binom_sf(kco, n_pairs, p_co)
        fold = kco / expected if expected > 0 else float("inf")
        out.append(dict(
            go_id=t,
            go_namespace=term_namespace.get(t, "?"),
            k_co=kco,
            k_either=k_either[t],
            n_pairs=n_pairs,
            K_fams=K,
            N_fams=N_fams,
            p_fam=p_fam,
            p_co_null=p_co,
            expected_k_co=expected,
            fold=fold,
            p=pval,
        ))
        pvals.append(pval)

    if out:
        qs = bh_qvalues(np.array(pvals))
        for r, q in zip(out, qs):
            r["q"] = float(q)
    out.sort(key=lambda d: d["q"])
    return out


def _compare_to_bag(
    summ_co: pd.DataFrame, sig_path: Path, out_dir: Path
) -> Path:
    from .io import load_significant_terms
    sig = load_significant_terms(sig_path)
    sig = sig[sig["sweep_namespace"] == "all"]
    merged = summ_co.merge(
        sig[["clade", "axis", "N_threshold", "go_id", "q", "fold"]].rename(
            columns={"q": "q_bag", "fold": "fold_bag"}),
        on=["clade", "axis", "N_threshold", "go_id"], how="left",
    )
    merged = merged.rename(
        columns={"q": "q_pair_co", "fold": "fold_pair_co"}
    )
    out_path = out_dir / "pair_co_vs_bag.tsv.gz"
    merged.to_csv(out_path, sep="\t", index=False, compression="gzip")
    return out_path


def run(
    supp_table,
    family_map,
    gene2accession,
    gene2go,
    summary,
    out_dir,
    obo=None,
    verbose: bool = True,
) -> Path:
    """End-to-end driver. Writes per-clade TSVs + aggregate gz table.

    Returns the Path of `pair_coenrich.tsv.gz` (may not exist if no rows
    were produced; tests explicitly assert existence of its parent dir).
    """
    out = Path(out_dir)
    (out / "per_clade").mkdir(parents=True, exist_ok=True)

    def _log(msg: str) -> None:
        if verbose:
            print(msg)

    _log(f"[load] gene2accession: {gene2accession}")
    prot_to_gene, _ = parse_gene2accession(gene2accession)
    _log(f"[load] gene2go: {gene2go}")
    gene_to_terms, term_ns = parse_gene2go(gene2go)
    _log(f"[load] family map: {family_map}")
    fam_to_genes, _ = parse_family_map(family_map, prot_to_gene)
    fam_to_terms = family_term_table(fam_to_genes, gene_to_terms)
    _log(f"  families with ≥1 annotated gene: {len(fam_to_terms)}")

    obo_names, obo_ns = ({}, {})
    if obo:
        obo_names, obo_ns = load_obo_names(obo)
    namespace_resolved = dict(term_ns)
    namespace_resolved.update(obo_ns)

    _log(f"[load] supp table: {supp_table}")
    supp = load_unique_pairs(supp_table)
    best = best_cell_per_clade(summary)

    all_rows: list[dict] = []
    for clade in sorted(best):
        bc = best[clade]
        axis, N = bc["axis"], bc["N"]
        clade_rows = supp[supp["nodename"] == clade]
        rows = pair_coenrich_for_clade(
            clade_rows, axis, N, fam_to_terms, namespace_resolved
        )
        if not rows:
            _log(f"[skip] {clade}  (no testable terms at {axis} N={N})")
            continue
        for r in rows:
            r["go_name"] = obo_names.get(r["go_id"], "")
        cdf = pd.DataFrame(rows)
        cdf.insert(0, "clade", clade)
        cdf.insert(1, "axis", axis)
        cdf.insert(2, "N_threshold", N)
        cols_order = ["clade", "axis", "N_threshold",
                      "go_id", "go_namespace", "go_name",
                      "k_co", "k_either", "n_pairs",
                      "K_fams", "N_fams", "p_fam", "p_co_null",
                      "expected_k_co", "fold", "p", "q"]
        cdf = cdf[cols_order]
        cdf.to_csv(out / "per_clade" / f"{clade}.tsv",
                   sep="\t", index=False)
        all_rows.extend(cdf.to_dict("records"))
        _log(f"[done] {clade} rows={len(cdf)}")

    agg_path = out / "pair_coenrich.tsv.gz"
    if all_rows:
        summ = pd.DataFrame(all_rows).sort_values(["clade", "q"])
        summ.to_csv(agg_path, sep="\t", index=False, compression="gzip")
        _log(f"[write] {agg_path} rows={len(summ)}")
        top10 = summ.groupby("clade").head(10)
        top10.to_csv(out / "pair_coenrich_top10.tsv",
                     sep="\t", index=False)
        _log(f"[write] pair_coenrich_top10.tsv rows={len(top10)}")
        sig_path = Path(summary).parent / "significant_terms.tsv"
        if sig_path.exists():
            cmp_path = _compare_to_bag(summ, sig_path, out)
            _log(f"[write] {cmp_path}")
    return agg_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="egt go pair-coenrich",
        description="Pair-level binomial-null GO co-enrichment.",
    )
    ap.add_argument("--supp-table", required=True)
    ap.add_argument("--family-map", required=True)
    ap.add_argument("--gene2accession", required=True)
    ap.add_argument("--gene2go", required=True)
    ap.add_argument("--summary", required=True,
                    help="summary.tsv from egt go sweep")
    ap.add_argument("--obo", default=None)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args(argv)
    run(
        supp_table=args.supp_table,
        family_map=args.family_map,
        gene2accession=args.gene2accession,
        gene2go=args.gene2go,
        summary=args.summary,
        out_dir=args.out_dir,
        obo=args.obo,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
