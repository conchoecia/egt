"""egt entanglement-browse — rank clade-characteristic ALG fusion pairs.

For each user-supplied clade, compute the frequency of every ALG pair
co-localization inside vs outside the clade, and emit the top-N pairs
that are enriched within the clade. Useful for identifying
clade-defining fusion events beyond the usual Hox cluster example.

Inputs:
  --presence-fusions  per_species_ALG_presence_fusions.tsv (from step 4)
  --clade-groupings   Name:taxid pairs (or bare taxids)
  --top-n             top pairs to report per clade

Outputs (in --out-dir):
  entangled_pairs_per_clade.tsv   long-form: one row per (clade, pair)
  entangled_pairs_top.md          paragraph-style writeup, top pairs per clade
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _pair_cols(columns: list[str]) -> list[str]:
    return [c for c in columns if c.startswith("(") and c.endswith(")") and "," in c]


def _parse_pair(col: str) -> tuple[str, str]:
    inner = col.strip()[1:-1]  # drop ()
    a, b = [x.strip().strip("'\"") for x in inner.split(",", 1)]
    return a, b


def _lineage_match(taxidstring: str, clade_taxid: int) -> bool:
    if not isinstance(taxidstring, str):
        return False
    return str(clade_taxid) in {x.strip() for x in taxidstring.split(";") if x.strip()}


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="egt entanglement-browse",
        description=(
            "Rank clade-characteristic ALG fusion pairs: for each clade, "
            "compute frequency of each ALG pair inside vs outside the "
            "clade, and report the top-N most clade-enriched pairs."
        ),
    )
    parser.add_argument("--presence-fusions", required=True, type=Path,
                        help="Path to per_species_ALG_presence_fusions.tsv (from step 4).")
    parser.add_argument("--clade-groupings", required=True,
                        help="Comma-separated list of 'Name:taxid' pairs (or bare taxids).")
    parser.add_argument("--top-n", type=int, default=20,
                        help="Top pairs to report per clade (default: %(default)s).")
    parser.add_argument("--min-species-in-clade", type=int, default=5,
                        help="Minimum species in clade to report (default: %(default)s).")
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    pf = pd.read_csv(args.presence_fusions, sep="\t", dtype=str, low_memory=False)
    if "species" in pf.columns:
        pf = pf.set_index("species")

    pair_cols = _pair_cols(list(pf.columns))
    if not pair_cols:
        raise SystemExit("No pair columns found (expected '(ALG1, ALG2)' headers).")

    # Binarize pair columns.
    def _to01(x):
        try: return int(float(x)) > 0
        except Exception: return False

    P = pf[pair_cols].map(_to01).astype(int)
    taxidstrings = pf["taxidstring"] if "taxidstring" in pf.columns else pd.Series("", index=pf.index)

    clade_specs = []
    for spec in args.clade_groupings.split(","):
        spec = spec.strip()
        if not spec:
            continue
        if ":" in spec:
            name, tid = spec.split(":", 1)
            clade_specs.append((int(tid), name))
        else:
            clade_specs.append((int(spec), spec))

    all_rows = []
    md_lines = ["# Clade-enriched ALG fusion pairs\n"]

    for tid, name in clade_specs:
        in_mask = taxidstrings.apply(lambda s: _lineage_match(s, tid))
        n_in = int(in_mask.sum())
        n_out = int((~in_mask).sum())
        if n_in < args.min_species_in_clade:
            continue

        freq_in = P.loc[in_mask].mean()
        freq_out = P.loc[~in_mask].mean() if n_out > 0 else pd.Series(0, index=P.columns)
        # Enrichment: fold-change with a small pseudocount.
        eps = 1.0 / max(1, n_in + n_out)
        fold = (freq_in + eps) / (freq_out + eps)

        # Simple Fisher's exact-like statistic (approximate: use observed deltas).
        # count in/out occurrences:
        k_in = P.loc[in_mask].sum()
        k_out = P.loc[~in_mask].sum() if n_out > 0 else pd.Series(0, index=P.columns)

        summary = pd.DataFrame({
            "pair": pair_cols,
            "alg_a": [_parse_pair(c)[0] for c in pair_cols],
            "alg_b": [_parse_pair(c)[1] for c in pair_cols],
            "freq_in_clade": freq_in.values,
            "freq_outside": freq_out.values,
            "n_in_clade": k_in.values,
            "n_outside": k_out.values,
            "fold_enrichment": fold.values,
        })
        summary["clade"] = name
        summary["clade_taxid"] = tid
        summary["n_species_in_clade"] = n_in
        summary["n_species_outside"] = n_out

        # Rank by fold_enrichment then freq_in_clade, keeping pairs
        # present in at least a quarter of the clade.
        ranked = summary[summary["freq_in_clade"] >= 0.25].sort_values(
            ["fold_enrichment", "freq_in_clade"], ascending=False
        ).head(args.top_n)

        all_rows.append(ranked)

        md_lines.append(f"## {name} (taxid {tid}, n={n_in})\n")
        if ranked.empty:
            md_lines.append("_No pairs pass the 25% in-clade frequency threshold._\n")
        else:
            md_lines.append("| ALG pair | % in clade | % outside | fold | n_in | n_out |")
            md_lines.append("|---|---:|---:|---:|---:|---:|")
            for _, r in ranked.iterrows():
                md_lines.append(
                    f"| {r.alg_a} × {r.alg_b} | {100*r.freq_in_clade:.1f} | "
                    f"{100*r.freq_outside:.1f} | {r.fold_enrichment:.2f}× | "
                    f"{int(r.n_in_clade)}/{n_in} | {int(r.n_out_clade) if 'n_out_clade' in r else int(r.n_outside)}/{n_out} |"
                )
        md_lines.append("")

    out_tsv = args.out_dir / "entangled_pairs_per_clade.tsv"
    out_md = args.out_dir / "entangled_pairs_top.md"
    if all_rows:
        pd.concat(all_rows, ignore_index=True).to_csv(out_tsv, sep="\t", index=False)
    else:
        pd.DataFrame().to_csv(out_tsv, sep="\t", index=False)
    out_md.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"[entanglement-browse] wrote {out_tsv}", file=sys.stderr)
    print(f"[entanglement-browse] wrote {out_md}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
