"""
Within-ALG vs between-ALG analysis of our defining-pair catalog.

This is the Simakov-style benchmark at the level our data supports.
Simakov 2013 / Simakov 2022 introduced the concept of Ancestral Linkage
Groups (ALGs): bilaterian-ancestor chromosome-scale units, each carrying
a specific set of gene families. The BCnS marker set we use comes from
Simakov 2022, which assigns every marker family to one of 29 ALGs
(A1a, A1b, A2, B1, B2, B3, C1, C2, D, Ea, Eb, F, G, H, I, J1, J2, K,
L, M, N, O1, O2, P, Qa, Qb, Qc, Qd, R).

The specific question this script answers is: **within our defining-pair
catalog (close / distant / stable / unstable / unique), what fraction
of pairs link two families on the same ALG?** This is the empirical
analog of the Simakov 2013 microsyntenic-block finding: ancestral
chromosome-scale units preserve gene neighborhoods within themselves
and lose them between. Pairs flagged `close_in_clade` or
`stable_in_clade` are our method's strongest candidates for that
preservation signal, so we expect them to be dramatically enriched for
within-ALG membership relative to an independent-pairing null.

Null model. The "if ALG membership were independent" probability of
a random BCnS pair being within-ALG is ∑ᵢ (nᵢ/N)² where nᵢ is the
number of mapped families in ALG i and N is the total. Under uniform
random pair assignment, the expected fraction of within-ALG pairs
equals this sum. Observed fractions above this are evidence of
ancestral chromosomal preservation.

Per-flag breakdown (close / distant / stable / unstable / unique)
tells us which of our flags best tracks the Simakov-style signal.

Writing note on circularity. Our marker set inherits its ALG
definitions from Simakov 2022 (which descends from Simakov 2013), so
the ALG column here is not an *independent* reference. What IS
independent is our scoring: the decision to flag a pair as stable or
close was made purely from per-species scaffold-distance statistics on
our 5,821-species dataset, with no input from ALG assignments. An
observed enrichment of within-ALG pairs in the flag lists is therefore
validation that the distance-based flags recover the ALG structure
they were never told about.
"""
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def load_family_alg(family_map_path):
    df = pd.read_csv(family_map_path, sep="\t")
    m = dict(zip(df["family_id"], df["alg"]))
    alg_counts = df["alg"].value_counts().to_dict()
    return m, alg_counts


def expected_same_alg_fraction(alg_counts):
    total = sum(alg_counts.values())
    return sum((n / total) ** 2 for n in alg_counts.values())


def per_pair_within_alg(up_df, fam_to_alg):
    """Add 'alg_1', 'alg_2', 'same_alg' columns to a unique_pairs DataFrame."""
    df = up_df.copy()
    df["alg_1"] = df["ortholog1"].map(fam_to_alg)
    df["alg_2"] = df["ortholog2"].map(fam_to_alg)
    df["same_alg"] = (df["alg_1"].notna() & df["alg_2"].notna()
                       & (df["alg_1"] == df["alg_2"]))
    return df


def summarize_flag(df, flag):
    sub = df[df[flag] == True]
    n_total = len(sub)
    annotated = sub[sub["alg_1"].notna() & sub["alg_2"].notna()]
    n_ann = len(annotated)
    n_same = int(annotated["same_alg"].sum())
    frac_same = n_same / n_ann if n_ann else float("nan")
    return dict(flag=flag, n_pairs=n_total, n_both_alg_annotated=n_ann,
                n_within_alg=n_same, n_between_alg=n_ann - n_same,
                frac_within_alg=frac_same)


def per_clade_flag_breakdown(df, flag):
    """For a given flag, give per-clade within-ALG fraction."""
    sub = df[df[flag] == True].copy()
    sub = sub[sub["alg_1"].notna() & sub["alg_2"].notna()]
    g = sub.groupby("nodename").agg(
        n_ann=("same_alg", "size"),
        n_same=("same_alg", "sum"),
    )
    g["frac_within_alg"] = g["n_same"] / g["n_ann"]
    return g.reset_index()


def plot_alg_heatmap(df, flag, ax, alg_order):
    """Between-ALG pair counts as a square heatmap for a given flag."""
    sub = df[df[flag] == True].dropna(subset=["alg_1", "alg_2"])
    if sub.empty:
        ax.set_title(f"{flag} (empty)"); ax.axis("off"); return
    # Canonicalise ordered pair.
    a = np.minimum(
        sub["alg_1"].map({k: i for i, k in enumerate(alg_order)}).to_numpy(),
        sub["alg_2"].map({k: i for i, k in enumerate(alg_order)}).to_numpy())
    b = np.maximum(
        sub["alg_1"].map({k: i for i, k in enumerate(alg_order)}).to_numpy(),
        sub["alg_2"].map({k: i for i, k in enumerate(alg_order)}).to_numpy())
    n = len(alg_order)
    mat = np.zeros((n, n), dtype=int)
    for i, j in zip(a, b):
        mat[i, j] += 1
        if i != j:
            mat[j, i] += 1
    # log scale for color
    im = ax.imshow(np.log10(mat + 1), cmap="viridis", origin="lower",
                    aspect="equal")
    ax.set_xticks(range(n))
    ax.set_xticklabels(alg_order, rotation=90, fontsize=6)
    ax.set_yticks(range(n))
    ax.set_yticklabels(alg_order, fontsize=6)
    ax.set_title(f"{flag}  (n={len(sub):,} pairs)", fontsize=9)
    return im


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family-map", required=True)
    ap.add_argument("--unique-pairs", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args(argv)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("[load] family map ...")
    fam_to_alg, alg_counts = load_family_alg(args.family_map)
    print(f"  families with ALG: {len(fam_to_alg)}  unique ALGs: {len(alg_counts)}")
    p_null = expected_same_alg_fraction(alg_counts)
    print(f"  expected within-ALG fraction under independence: {p_null:.4f}"
          f" ({p_null*100:.2f}%)")

    print("[load] unique_pairs ...")
    up = pd.read_csv(args.unique_pairs, sep="\t", compression="infer",
                      low_memory=False)
    print(f"  rows: {len(up):,}  clades: {up['nodename'].nunique()}")
    up = per_pair_within_alg(up, fam_to_alg)
    print(f"  rows with both partners ALG-annotated: "
          f"{(up['alg_1'].notna() & up['alg_2'].notna()).sum():,}")

    # Per-flag headline.
    flags = ["close_in_clade", "distant_in_clade", "stable_in_clade",
             "unstable_in_clade", "unique_to_clade"]
    rows = [summarize_flag(up, f) for f in flags]
    # Also a "any_flag" row = rows where any flag is true (our whole
    # defining set).
    mask_any = up[flags].sum(axis=1) > 0
    sub_any = up[mask_any & up["alg_1"].notna() & up["alg_2"].notna()]
    rows.append(dict(flag="any_flag", n_pairs=int(mask_any.sum()),
                      n_both_alg_annotated=len(sub_any),
                      n_within_alg=int(sub_any["same_alg"].sum()),
                      n_between_alg=int((~sub_any["same_alg"]).sum()),
                      frac_within_alg=float(sub_any["same_alg"].mean())))
    sdf = pd.DataFrame(rows)
    sdf["expected_frac_null"] = p_null
    sdf["enrichment_over_null"] = sdf["frac_within_alg"] / p_null
    sdf.to_csv(out / "flag_summary.tsv", sep="\t", index=False)
    print(f"[write] {out}/flag_summary.tsv")
    print(sdf.to_string(index=False))

    # Per-clade × per-flag breakdown for the two most relevant flags.
    for f in ("close_in_clade", "stable_in_clade"):
        g = per_clade_flag_breakdown(up, f)
        g["expected_frac_null"] = p_null
        g["enrichment_over_null"] = g["frac_within_alg"] / p_null
        g = g.sort_values("frac_within_alg", ascending=False)
        g.to_csv(out / f"per_clade_{f}.tsv", sep="\t", index=False)
        print(f"[write] {out}/per_clade_{f}.tsv")

    # ALG × ALG heatmap PDF (one page per flag).
    alg_order = sorted(alg_counts.keys())
    with PdfPages(out / "alg_heatmaps.pdf") as pdf:
        for f in flags + ["any_flag"]:
            fig, ax = plt.subplots(figsize=(7, 6.5))
            flag_for_mask = f
            if f == "any_flag":
                up["_any"] = mask_any
                plot_alg_heatmap(up, "_any", ax, alg_order)
                ax.set_title(f"any defining flag  (n={int(mask_any.sum()):,} pairs)",
                              fontsize=10)
            else:
                plot_alg_heatmap(up, f, ax, alg_order)
            fig.tight_layout()
            pdf.savefig(fig); plt.close(fig)
    print(f"[write] {out}/alg_heatmaps.pdf")

    # Bar chart: within-ALG fraction per flag vs null.
    fig, ax = plt.subplots(figsize=(8, 4))
    xs = np.arange(len(sdf))
    ys = sdf["frac_within_alg"].to_numpy()
    ax.bar(xs, ys, color="C0", width=0.6,
            label="observed within-ALG fraction")
    ax.axhline(p_null, color="red", ls="--", lw=1,
                label=f"null expectation ({p_null:.3f})")
    for x, y, e in zip(xs, ys, sdf["enrichment_over_null"].to_numpy()):
        ax.text(x, y + 0.01, f"{e:.1f}×", ha="center", fontsize=9)
    ax.set_xticks(xs)
    ax.set_xticklabels(sdf["flag"], rotation=30, ha="right")
    ax.set_ylabel("Fraction of pairs within same ALG")
    ax.set_title("Within-ALG enrichment — defining-pair flags vs Simakov-style null")
    ax.set_ylim(0, min(1.0, max(ys) * 1.15))
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "within_alg_by_flag.pdf")
    plt.close(fig)
    print(f"[write] {out}/within_alg_by_flag.pdf")

    print("[done]")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
