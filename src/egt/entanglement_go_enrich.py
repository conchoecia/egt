"""egt entanglement-go-enrich — GO enrichment on clade-characteristic ALG genes.

For each clade, identify the BCnS ALG families that participate in the
clade-enriched fusion pairs (from `entanglement-browse`), map them to
their human gene representative (from `build-family-naming-map`), and
test the resulting gene set for GO-term enrichment using a simple
hypergeometric test against the full BCnS → human background.

Useful for asking whether clade-stable locus co-localizations are
enriched for particular biological functions.

Inputs (all user-supplied):
  --alg-rbh                BCnSSimakov2022.rbh (for per-ALG family → ALG letter)
  --family-gene-map        output of build-family-naming-map
  --entangled-pairs        output of entanglement-browse (entangled_pairs_per_clade.tsv)
  --human-go               EBI QuickGO GAF for human (goa_human.gaf.gz)

Output (in --out-dir):
  go_enrichment_per_clade.tsv  per (clade, go_term) hypergeometric test results
                               with BH-adjusted q-values
"""
from __future__ import annotations

import argparse
import gzip
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def _parse_gaf(gaf_path: Path) -> dict[str, set[str]]:
    """Parse a GOA GAF file → {gene_symbol: {go_id, ...}}."""
    opener = gzip.open if str(gaf_path).endswith(".gz") else open
    out: dict[str, set[str]] = defaultdict(set)
    with opener(gaf_path, "rt") as fh:
        for line in fh:
            if line.startswith("!") or not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                continue
            # Col 3 = DB_Object_Symbol, col 5 = GO_ID
            sym = parts[2].strip()
            go = parts[4].strip()
            if sym and go.startswith("GO:"):
                out[sym].add(go)
    return out


def _hypergeom_p(k: int, K: int, n: int, N: int) -> float:
    """P(X >= k) where X ~ Hypergeometric(N, K, n).

    No scipy dependency: sum PMF from k..min(K,n).
    """
    from math import comb
    if N <= 0 or k < 0 or K < 0 or n < 0:
        return 1.0
    top = min(K, n)
    if k > top:
        return 0.0
    total_choose = comb(N, n)
    if total_choose == 0:
        return 1.0
    s = 0
    for i in range(k, top + 1):
        s += comb(K, i) * comb(N - K, n - i)
    return s / total_choose


def _bh_adjust(pvals: np.ndarray) -> np.ndarray:
    n = len(pvals)
    if n == 0:
        return pvals
    order = np.argsort(pvals)
    ranks = np.arange(1, n + 1)
    adj = pvals[order] * n / ranks
    # enforce monotonicity
    for i in range(n - 2, -1, -1):
        adj[i] = min(adj[i], adj[i + 1])
    q = np.empty(n)
    q[order] = np.clip(adj, 0, 1)
    return q


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="egt entanglement-go-enrich",
        description=(
            "GO enrichment for clade-characteristic BCnS ALG gene sets, "
            "using human orthologs and a standard GAF annotation file."
        ),
    )
    parser.add_argument("--alg-rbh", required=True, type=Path)
    parser.add_argument("--family-gene-map", required=True, type=Path,
                        help="Output of build-family-naming-map.")
    parser.add_argument("--entangled-pairs", required=True, type=Path,
                        help="Output TSV from entanglement-browse.")
    parser.add_argument("--human-go", required=True, type=Path,
                        help="GOA GAF file for human (e.g. goa_human.gaf.gz).")
    parser.add_argument("--fdr", type=float, default=0.05,
                        help="BH-adjusted q-value threshold to include in output (default: %(default)s).")
    parser.add_argument("--min-term-hits", type=int, default=3,
                        help="Minimum foreground hits per GO term to evaluate.")
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Load family→human gene map. Background = all families that HAVE a human gene.
    fg_map = pd.read_csv(args.family_gene_map, sep="\t", dtype=str)
    if "family_id" not in fg_map.columns or "human_gene" not in fg_map.columns:
        raise SystemExit("--family-gene-map must have family_id + human_gene columns")
    fg_map = fg_map.dropna(subset=["human_gene"])
    family_to_gene = dict(zip(fg_map["family_id"], fg_map["human_gene"]))

    # Load ALG RBH to know which families belong to which ALG letter.
    alg_rbh = pd.read_csv(args.alg_rbh, sep="\t", dtype=str, low_memory=False)
    if "rbh" not in alg_rbh.columns or "gene_group" not in alg_rbh.columns:
        raise SystemExit("--alg-rbh missing rbh or gene_group columns")
    alg_to_families: dict[str, list[str]] = defaultdict(list)
    for _, r in alg_rbh.iterrows():
        alg_to_families[r["gene_group"]].append(r["rbh"])

    # Background universe: all human genes reached by any family in the ALG DB.
    background_genes = {family_to_gene[f] for fams in alg_to_families.values() for f in fams
                        if f in family_to_gene}

    # GO annotations.
    go_by_gene = _parse_gaf(args.human_go)
    # Restrict to genes in background.
    background_go = {g: go_by_gene.get(g, set()) for g in background_genes}
    # GO-term universe from background.
    go_universe = {go for gos in background_go.values() for go in gos}

    # Load entangled-pair table.
    pairs = pd.read_csv(args.entangled_pairs, sep="\t", dtype=str)
    pairs["n_in_clade"] = pd.to_numeric(pairs["n_in_clade"], errors="coerce")

    # For each clade, compose the foreground gene set = union of human genes
    # across ALL families of both ALGs in the clade's entangled pairs.
    all_rows = []
    for clade, grp in pairs.groupby("clade"):
        fg_genes: set[str] = set()
        for _, row in grp.iterrows():
            for alg_letter in (row.get("alg_a"), row.get("alg_b")):
                if not isinstance(alg_letter, str):
                    continue
                for fam in alg_to_families.get(alg_letter, []):
                    g = family_to_gene.get(fam)
                    if g:
                        fg_genes.add(g)
        if not fg_genes:
            continue
        N = len(background_genes)
        n = len(fg_genes)
        rows = []
        for go in go_universe:
            K = sum(1 for g in background_genes if go in background_go.get(g, set()))
            k = sum(1 for g in fg_genes if go in background_go.get(g, set()))
            if k < args.min_term_hits or K == 0:
                continue
            p = _hypergeom_p(k, K, n, N)
            fold = (k / n) / (K / N) if (n > 0 and N > 0 and K > 0) else np.nan
            rows.append(dict(clade=clade, go_id=go, k=k, K=K, n=n, N=N,
                             fold_enrichment=fold, p_value=p))
        if not rows:
            continue
        clade_df = pd.DataFrame(rows)
        clade_df["q_value"] = _bh_adjust(clade_df["p_value"].to_numpy())
        all_rows.append(clade_df[clade_df["q_value"] < args.fdr])

    out_path = args.out_dir / "go_enrichment_per_clade.tsv"
    if all_rows:
        combined = pd.concat(all_rows, ignore_index=True).sort_values(
            ["clade", "q_value", "fold_enrichment"], ascending=[True, True, False]
        )
    else:
        combined = pd.DataFrame(columns=["clade", "go_id", "k", "K", "n", "N",
                                         "fold_enrichment", "p_value", "q_value"])
    combined.to_csv(out_path, sep="\t", index=False)
    print(f"[entanglement-go-enrich] wrote {out_path}", file=sys.stderr)
    print(f"  background genes: {len(background_genes)}", file=sys.stderr)
    print(f"  enriched terms  : {len(combined)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
