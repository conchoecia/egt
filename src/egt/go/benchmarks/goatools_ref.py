"""
Benchmark our native hypergeometric + BH-FDR (sweep.py) against goatools
(Klopfenstein et al. 2018).

For each clade, picks the best-q sweep cell (axis, N) from summary.tsv,
re-builds the foreground gene set exactly as sweep.py does, and runs
goatools' GOEnrichmentStudy with `propagate_counts=False` so the
annotation scope matches our code (NCBI gene2go is treated as direct
annotations, no DAG walk). BH-FDR is requested via `methods=['fdr_bh']`.

Emits:
  benchmark_goatools/per_clade/<clade>.tsv  — side-by-side per-term rows
  benchmark_goatools/summary.tsv            — per-clade agreement metrics
  benchmark_goatools/scatter.pdf            — −log10(q) scatter, all clades
"""
import argparse
import gzip
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# goatools is a heavy optional dependency — imported lazily inside the
# run / helper functions so `import egt.go.benchmarks.goatools_ref`
# succeeds on envs that only install egt's core dependencies.

REFSEQ_PROTEIN_RE = re.compile(r"^(NP|XP|YP)_[0-9]+\.[0-9]+$")
NCBI_CATEGORY_MAP = {
    "Process": "BP", "Function": "MF", "Component": "CC",
    "biological_process": "BP", "molecular_function": "MF",
    "cellular_component": "CC",
}
OCCUPANCY_MIN = 0.5


def parse_gene2accession(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    prot_to_gene = {}
    with opener(path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 7:
                continue
            prot, gene = f[5], f[1]
            if prot and prot != "-" and gene and gene != "-":
                prot_to_gene[prot] = gene
    return prot_to_gene


def parse_gene2go(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    gene_to_terms = defaultdict(set)
    term_ns = {}
    with opener(path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 8:
                continue
            gene, go_id, qualifier, category = f[1], f[2], f[4], f[7]
            if "NOT" in qualifier.upper().split("|"):
                continue
            gene_to_terms[gene].add(go_id)
            ns = NCBI_CATEGORY_MAP.get(category)
            if ns and go_id not in term_ns:
                term_ns[go_id] = ns
    return gene_to_terms, term_ns


def parse_family_map(path, prot_to_gene):
    df = pd.read_csv(path, sep="\t")
    family_col = df.columns[0]
    gene_col = "human_gene" if "human_gene" in df.columns else df.columns[2]
    fam_to_genes = defaultdict(set)
    for _, row in df.iterrows():
        fam = row[family_col]
        val = row[gene_col]
        if not isinstance(val, str) or not val.strip():
            continue
        v = val.strip()
        if REFSEQ_PROTEIN_RE.match(v):
            g = prot_to_gene.get(v)
            if g:
                fam_to_genes[fam].add(g)
    return fam_to_genes


def best_sweep_cell_per_clade(summary_path):
    """For each clade, pick the (axis, N, namespace='all') row with the
    smallest top_q. Returns dict {clade: {axis, N_threshold}}."""
    from ..io import load_sweep_summary
    sdf = load_sweep_summary(summary_path)
    sdf = sdf[(sdf["namespace"] == "all") & sdf["top_q"].notna()]
    best = (sdf.sort_values("top_q")
               .drop_duplicates(subset=["clade"])
               .set_index("clade"))
    return {c: dict(axis=row["axis"], N=int(row["N_threshold"]))
            for c, row in best.iterrows()}


def foreground_for_cell(clade_rows, axis, N, fam_to_genes):
    df = clade_rows.copy()
    df = df[df["occupancy_in"].fillna(0) >= OCCUPANCY_MIN]
    df = df.dropna(subset=["sd_in_out_ratio_log_sigma",
                           "mean_in_out_ratio_log_sigma"])
    df = df.reset_index(drop=True)
    if df.empty:
        return set(), 0
    stability_order = df.sort_values("sd_in_out_ratio_log_sigma").index.to_numpy()
    closeness_order = df.sort_values("mean_in_out_ratio_log_sigma").index.to_numpy()
    if axis == "stability":
        idxs = stability_order[:N]
    elif axis == "closeness":
        idxs = closeness_order[:N]
    else:
        idxs = list(set(stability_order[:N].tolist())
                    & set(closeness_order[:N].tolist()))
    if len(idxs) == 0:
        return set(), 0
    sub = df.loc[list(idxs)]
    fams = pd.concat([sub["ortholog1"], sub["ortholog2"]]).dropna().unique()
    fg = set()
    for f in fams:
        fg |= fam_to_genes.get(f, set())
    return fg, len(idxs)


def bh_qvalues(pvals):
    """Benjamini-Hochberg. Returns ndarray of q-values aligned with input."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    q_sorted = np.minimum.accumulate(ranked[::-1])[::-1]
    q_sorted = np.minimum(q_sorted, 1.0)
    q = np.empty_like(q_sorted)
    q[order] = q_sorted
    return q


def log_binom(n, k):
    if k < 0 or k > n:
        return float("-inf")
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def hypergeom_sf(k, N_total, K, n):
    """P(X >= k) — exact log-space sum; matches sweep.py."""
    if k <= 0:
        return 1.0
    if k > min(K, n):
        return 0.0
    denom = log_binom(N_total, n)
    logs = [log_binom(K, i) + log_binom(N_total - K, n - i) - denom
            for i in range(k, min(K, n) + 1)]
    m = max(logs)
    return math.exp(m) * sum(math.exp(x - m) for x in logs)


def run_our_hypergeom(foreground, pop_assoc, min_term_hits=2):
    """Run the same hypergeom + BH that sweep.py runs, but without the
    q<=0.25 write-time filter. Returns DataFrame with every term that
    passes the k>=min_term_hits floor."""
    fg = {g for g in foreground if g in pop_assoc}
    n = len(fg)
    N_total = len(pop_assoc)
    if n == 0 or N_total == 0:
        return pd.DataFrame()
    term_K = defaultdict(int)
    for g, terms in pop_assoc.items():
        for t in terms:
            term_K[t] += 1
    term_k = defaultdict(int)
    for g in fg:
        for t in pop_assoc[g]:
            term_k[t] += 1
    rows = []
    for t, k in term_k.items():
        if k < min_term_hits:
            continue
        K = term_K[t]
        p = hypergeom_sf(k, N_total, K, n)
        fold = (k / n) / (K / N_total) if K > 0 else float("inf")
        rows.append(dict(go_id=t, k=k, K=K, n=n, N=N_total,
                          fold=fold, p=p))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["q"] = bh_qvalues(df["p"].to_numpy())
    return df


def run_goatools_for_clade(clade, pop_assoc, obodag, foreground,
                             min_term_hits=2):
    """Run goatools GOEnrichmentStudy on one foreground.

    pop_assoc: {gene_id (str): set(GO_ID)}  — restricted to background that
    has at least one annotation (matches our sweep.py's background_to_terms).
    Returns a pandas DataFrame with per-term p_theirs, q_theirs, k, K, n, N,
    plus q_theirs_matched: BH-FDR recomputed on the subset with k >= 2,
    matching our code's `min_term_hits` pre-filter. The q_theirs_matched
    column is what should be compared term-by-term against our q; the raw
    q_theirs is goatools' native output.
    """
    from goatools.go_enrichment import GOEnrichmentStudy
    pop = set(pop_assoc.keys())
    fg = {g for g in foreground if g in pop}
    if not fg:
        return pd.DataFrame()
    study = GOEnrichmentStudy(
        pop=pop,
        assoc=pop_assoc,
        obo_dag=obodag,
        propagate_counts=False,
        alpha=0.05,
        methods=["fdr_bh"],
    )
    results = study.run_study(fg)
    rows = []
    for r in results:
        if r.study_count == 0:
            continue
        rows.append(dict(
            go_id=r.GO,
            name=r.name,
            namespace=r.NS,
            enrichment=r.enrichment,
            k=r.study_count, n=r.study_n,
            K=r.pop_count, N=r.pop_n,
            p_theirs=r.p_uncorrected,
            q_theirs=r.p_fdr_bh,
        ))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # Upper-tail only.
    df = df[df["enrichment"] == "e"].reset_index(drop=True)
    # Re-apply BH on the k>=min_term_hits subset so `m` matches our code.
    kept = df[df["k"] >= min_term_hits].copy()
    if not kept.empty:
        kept["q_theirs_matched"] = bh_qvalues(kept["p_theirs"].to_numpy())
    else:
        kept["q_theirs_matched"] = []
    df = df.merge(kept[["go_id", "q_theirs_matched"]], on="go_id", how="left")
    return df


def compare_per_clade(ours_rows, theirs_df, q_col="q_theirs_matched"):
    """Join on go_id, report term-count and agreement metrics.

    Uses `q_col` from theirs_df: by default 'q_theirs_matched' (BH re-run
    on k>=2 subset to match our pre-filter). Pass 'q_theirs' to compare
    goatools' native m=#-all-terms BH.
    """
    if theirs_df.empty:
        return dict(our_q05=0, their_q05=0, our_q25=0, their_q25=0,
                    overlap_q05=0, overlap_q25=0,
                    jaccard_q05=float("nan"), jaccard_q25=float("nan"),
                    spearman_mlog10q=float("nan"),
                    pearson_logp=float("nan"),
                    n_terms_both=0)
    ours = ours_rows.set_index("go_id")
    theirs = theirs_df.set_index("go_id")
    both = ours.index.intersection(theirs.index)
    if len(both) < 2:
        spearman = float("nan"); pearson_logp = float("nan")
    else:
        a = np.clip(ours.loc[both, "q"].astype(float), 1e-300, 1.0)
        b = np.clip(theirs.loc[both, q_col].astype(float), 1e-300, 1.0)
        spearman = pd.Series(-np.log10(a.to_numpy())).corr(
            pd.Series(-np.log10(b.to_numpy())), method="spearman")
        # Also pearson on per-term log(p) — tests whether uncorrected
        # stats agree (if they don't, BH post-hoc won't save it).
        po = np.clip(ours.loc[both, "p"].astype(float), 1e-300, 1.0)
        pt = np.clip(theirs.loc[both, "p_theirs"].astype(float), 1e-300, 1.0)
        pearson_logp = pd.Series(-np.log10(po.to_numpy())).corr(
            pd.Series(-np.log10(pt.to_numpy())), method="pearson")
    our_q05 = set(ours.index[ours["q"] <= 0.05])
    their_q05 = set(theirs.index[theirs[q_col].fillna(1.0) <= 0.05])
    our_q25 = set(ours.index[ours["q"] <= 0.25])
    their_q25 = set(theirs.index[theirs[q_col].fillna(1.0) <= 0.25])

    def jac(a, b):
        u = a | b
        return len(a & b) / len(u) if u else float("nan")

    return dict(
        our_q05=len(our_q05), their_q05=len(their_q05),
        our_q25=len(our_q25), their_q25=len(their_q25),
        overlap_q05=len(our_q05 & their_q05),
        overlap_q25=len(our_q25 & their_q25),
        jaccard_q05=jac(our_q05, their_q05),
        jaccard_q25=jac(our_q25, their_q25),
        spearman_mlog10q=spearman,
        pearson_logp=pearson_logp,
        n_terms_both=len(both),
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--supp-table", required=True)
    ap.add_argument("--family-map", required=True)
    ap.add_argument("--gene2accession", required=True)
    ap.add_argument("--gene2go", required=True)
    ap.add_argument("--obo", required=True)
    ap.add_argument("--summary", required=True,
                    help="out/summary.tsv from sweep.py")
    ap.add_argument("--significant", required=True,
                    help="out/significant_terms.tsv from sweep.py")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args(argv)

    out = Path(args.out_dir)
    (out / "per_clade").mkdir(parents=True, exist_ok=True)

    from goatools.obo_parser import GODag
    print("[load] OBO ..."); obodag = GODag(args.obo, prt=None)

    print("[load] gene2accession ...")
    prot_to_gene = parse_gene2accession(args.gene2accession)

    print("[load] gene2go ...")
    gene_to_terms_all, term_ns = parse_gene2go(args.gene2go)

    print("[load] family map ...")
    fam_to_genes = parse_family_map(args.family_map, prot_to_gene)

    # Background exactly as in sweep.py: GeneIDs in family map AND annotated.
    background_gene_ids = set().union(*fam_to_genes.values())
    pop_assoc = {g: gene_to_terms_all[g] for g in background_gene_ids
                 if g in gene_to_terms_all}
    print(f"[bg] family-map GeneIDs={len(background_gene_ids)}  "
          f"with-annotation={len(pop_assoc)}")

    print("[load] supp table ...")
    p = str(args.supp_table).lower()
    if p.endswith(".xlsx") or p.endswith(".xls"):
        supp = pd.read_excel(args.supp_table, engine="openpyxl")
    else:
        supp = pd.read_csv(args.supp_table, sep="\t",
                           compression="infer", low_memory=False)

    print("[load] sweep summary ...")
    best_cells = best_sweep_cell_per_clade(args.summary)

    print("[load] significant_terms ...")
    from ..io import load_significant_terms
    sig = load_significant_terms(args.significant)

    summary_rows = []
    scatter_pts = []  # (clade, log10q_ours, log10q_theirs)

    for clade in sorted(best_cells):
        bc = best_cells[clade]
        axis, N = bc["axis"], bc["N"]
        clade_rows = supp[supp["nodename"] == clade]
        fg, n_pairs_used = foreground_for_cell(clade_rows, axis, N,
                                                fam_to_genes)
        if not fg:
            print(f"[skip] {clade}  (no foreground)")
            continue
        theirs = run_goatools_for_clade(clade, pop_assoc, obodag, fg)
        # Rerun OUR native hypergeom on the same foreground so we get the
        # full term set (not just q<=0.25 as significant_terms.tsv stores).
        # This is what makes the side-by-side TSV apples-to-apples: every
        # term that goatools tested, we also tested, and vice versa.
        ours_full = run_our_hypergeom(fg, pop_assoc, min_term_hits=2)
        if ours_full.empty:
            ours = pd.DataFrame(columns=["go_id", "k_ours", "K_ours",
                                         "n_ours", "N_ours", "fold_ours",
                                         "p_ours", "q_ours"])
        else:
            ours = ours_full.rename(columns={
                "p": "p_ours", "q": "q_ours",
                "k": "k_ours", "K": "K_ours",
                "n": "n_ours", "N": "N_ours",
                "fold": "fold_ours"})
        # Keep the significant_terms lookup too, for the per-clade
        # reporting metrics (those still use our official sweep output).
        ours_rows = sig[(sig["clade"] == clade)
                        & (sig["axis"] == axis)
                        & (sig["N_threshold"] == N)
                        & (sig["sweep_namespace"] == "all")]
        if theirs.empty:
            side = ours.copy()
            for c in ("p_theirs", "q_theirs", "q_theirs_matched"):
                side[c] = float("nan")
        else:
            th = theirs[["go_id", "k", "K", "n", "N", "p_theirs",
                         "q_theirs", "q_theirs_matched", "name",
                         "namespace"]].rename(
                columns={"k": "k_theirs", "K": "K_theirs",
                         "n": "n_theirs", "N": "N_theirs"})
            # Inner-join: only terms tested by BOTH pipelines. Our
            # min_term_hits=2 filter matches goatools' k>=2 post-hoc
            # subset, so the intersection is ~all terms either tested.
            side = ours.merge(th, on="go_id", how="inner")
        side.insert(0, "clade", clade)
        side.insert(1, "axis", axis)
        side.insert(2, "N_threshold", N)
        side.to_csv(out / "per_clade" / f"{clade}.tsv",
                    sep="\t", index=False)

        # Scatter prep: terms with finite q in both (matched-BH).
        both = side.dropna(subset=["q_ours", "q_theirs_matched"])
        for _, r in both.iterrows():
            qo = max(float(r["q_ours"]), 1e-300)
            qt = max(float(r["q_theirs_matched"]), 1e-300)
            scatter_pts.append((clade, -math.log10(qo), -math.log10(qt)))

        # Metrics: use the freshly-computed full ours table rather than
        # the q<=0.25 truncated significant_terms.tsv, so Jaccard/Spearman
        # are computed on the whole matched term set.
        ours_for_metrics = ours.rename(columns={
            "p_ours": "p", "q_ours": "q"})
        metrics = compare_per_clade(ours_for_metrics, theirs)
        metrics.update(dict(clade=clade, axis=axis, N_threshold=N,
                            foreground_size=len(fg),
                            pairs_used=n_pairs_used,
                            their_terms_tested=len(theirs)))
        summary_rows.append(metrics)
        print(f"[done] {clade:<18} axis={axis:<12} N={N:<5}  "
              f"ours q<=0.05: {metrics['our_q05']:<4} theirs: {metrics['their_q05']:<4}  "
              f"jac05={metrics['jaccard_q05']:.2f}  "
              f"rho={metrics['spearman_mlog10q']:.3f}")

    # Summary TSV
    cols = ["clade", "axis", "N_threshold", "foreground_size", "pairs_used",
            "n_terms_both", "our_q05", "their_q05", "overlap_q05",
            "jaccard_q05", "our_q25", "their_q25", "overlap_q25",
            "jaccard_q25", "spearman_mlog10q", "pearson_logp"]
    sdf = pd.DataFrame(summary_rows)[cols]
    sdf = sdf.sort_values("clade").reset_index(drop=True)
    sdf.to_csv(out / "summary.tsv", sep="\t", index=False)
    print(f"[write] {out}/summary.tsv  rows={len(sdf)}")

    # Overall numbers
    tot_both = sdf["n_terms_both"].sum()
    med_rho = sdf["spearman_mlog10q"].median()
    med_pearson_p = sdf["pearson_logp"].median()
    med_jac05 = sdf["jaccard_q05"].median()
    med_jac25 = sdf["jaccard_q25"].median()
    print(f"[overall] clades={len(sdf)}  "
          f"terms-matched={tot_both}  "
          f"median pearson(-log10p)={med_pearson_p:.3f}  "
          f"median rho(-log10q_matched)={med_rho:.3f}  "
          f"median Jaccard(q<=0.05)={med_jac05:.2f}  "
          f"median Jaccard(q<=0.25)={med_jac25:.2f}")

    # Scatter — two panels: raw p-agreement + matched-BH q-agreement.
    # Also a per-clade scatter stack.
    if scatter_pts:
        spdf = pd.DataFrame(scatter_pts, columns=["clade", "ours", "theirs"])
        with PdfPages(out / "scatter.pdf") as pdf:
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.scatter(spdf["ours"], spdf["theirs"], s=6, alpha=0.4,
                       color="C0", edgecolors="none")
            lim = max(spdf["ours"].max(), spdf["theirs"].max())
            ax.plot([0, lim], [0, lim], color="k", ls="--", lw=0.7,
                    label="y=x")
            ax.set_xlabel("-log10(q)  ours (sweep.py)")
            ax.set_ylabel("-log10(q)  goatools (fdr_bh, k≥2 subset)")
            ax.set_title(
                f"Per-term q-agreement — matched BH denominator "
                f"({len(spdf)} terms, {spdf['clade'].nunique()} clades)")
            ax.legend()
            fig.tight_layout()
            pdf.savefig(fig); plt.close(fig)

            # Per-clade grid.
            clades_sorted = sorted(spdf["clade"].unique())
            ncols = 4
            nrows = (len(clades_sorted) + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows, ncols,
                                      figsize=(3 * ncols, 3 * nrows),
                                      squeeze=False)
            for i, c in enumerate(clades_sorted):
                ax = axes[i // ncols][i % ncols]
                sc = spdf[spdf["clade"] == c]
                ax.scatter(sc["ours"], sc["theirs"], s=8, alpha=0.5,
                            color="C0", edgecolors="none")
                lim = max(sc["ours"].max(), sc["theirs"].max(), 1.0)
                ax.plot([0, lim], [0, lim], color="k", ls="--", lw=0.6)
                ax.set_title(f"{c} (n={len(sc)})", fontsize=9)
                ax.tick_params(labelsize=7)
                if i // ncols == nrows - 1:
                    ax.set_xlabel("-log10(q) ours", fontsize=8)
                if i % ncols == 0:
                    ax.set_ylabel("-log10(q) goatools (matched)", fontsize=8)
            # Blank unused axes.
            for j in range(len(clades_sorted), nrows * ncols):
                axes[j // ncols][j % ncols].axis("off")
            fig.tight_layout()
            pdf.savefig(fig); plt.close(fig)
        print(f"[write] {out}/scatter.pdf")

    print("[done]")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
