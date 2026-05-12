"""Bag-of-genes hypergeometric GO enrichment.

Used by `egt go sweep` (the N-sweep driver) and by the goatools benchmark
harness. Given a foreground gene set and a background of annotatable
genes, compute the one-sided upper hypergeometric p-value for every GO
term with >= `min_term_hits` foreground hits, BH-corrected per namespace.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping, Sequence

import numpy as np

from .stats import bh_qvalues, hypergeom_sf


def reduce_pairs_to_geneset(
    families: Iterable[str],
    fam_to_genes: Mapping[str, set[str]],
) -> set[str]:
    """Bag-of-genes reduction: union of mapped GeneIDs across a family set."""
    out: set[str] = set()
    for f in families:
        out |= fam_to_genes.get(f, set())
    return out


def enrich_for_foreground(
    foreground: Iterable[str],
    background_to_terms: Mapping[str, set[str]],
    term_namespace: Mapping[str, str],
    namespaces: Sequence[str] = ("all", "BP", "MF", "CC"),
    min_term_hits: int = 2,
) -> dict[str, list[dict]]:
    """One-shot hypergeometric + BH-FDR for a single foreground.

    Returns `{namespace: [row_dict]}`. `row_dict` keys:
      go_id, k, K, n, N, fold, p, q, term_namespace.

    k < `min_term_hits` is the pre-correction filter so BH isn't forced
    to down-weight every singleton-hit term.
    """
    fg = {g for g in foreground if g in background_to_terms}
    n = len(fg)
    N_total = len(background_to_terms)
    if n == 0 or N_total == 0:
        return {ns: [] for ns in namespaces}

    term_K: dict[str, int] = defaultdict(int)
    for g, terms in background_to_terms.items():
        for t in terms:
            term_K[t] += 1

    term_k: dict[str, int] = defaultdict(int)
    for g in fg:
        for t in background_to_terms[g]:
            term_k[t] += 1

    rows: list[tuple] = []
    for t, k in term_k.items():
        if k < min_term_hits:
            continue
        K = term_K[t]
        p = hypergeom_sf(k, N_total, K, n)
        fold = (k / n) / (K / N_total) if K > 0 else float("inf")
        rows.append((t, k, K, n, N_total, fold, p))

    out_by_ns: dict[str, list[dict]] = {ns: [] for ns in namespaces}
    if not rows:
        return out_by_ns

    for ns in namespaces:
        if ns == "all":
            sub = rows
        else:
            sub = [r for r in rows if term_namespace.get(r[0]) == ns]
        if not sub:
            continue
        pvals = np.array([r[6] for r in sub])
        qvals = bh_qvalues(pvals)
        enriched = [
            dict(go_id=t, k=k, K=K, n=ntot_, N=N_, fold=fold, p=p, q=q,
                 term_namespace=term_namespace.get(t, "?"))
            for (t, k, K, ntot_, N_, fold, p), q in zip(sub, qvals)
        ]
        # Secondary sort on go_id makes top_term selection reproducible
        # across Python hash seeds when multiple terms tie on q.
        enriched.sort(key=lambda d: (d["q"], d["go_id"]))
        out_by_ns[ns] = enriched
    return out_by_ns
