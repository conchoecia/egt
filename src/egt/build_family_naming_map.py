"""egt build-family-naming-map — map BCnS ALG families to human gene IDs.

Pass 1 (fast, in-package): for each BCnSSimakov2022 family, look up the
human representative via the per-species human RBH file (5,821 such files
sit under RBH_DIR after the pipeline runs). Most families have a direct
human ortholog — ~90% coverage expected.

Pass 2 (cluster-only, slow): for the ~10% without human hits in pass 1,
emit an HMM consensus from BCnSSimakov2022.hmm and DIAMOND-search against
Swiss-Prot human. See the companion shell script
`post_analyses/build_family_naming_map/pass2_hmm_consensus.sh` in the
schultz-et-al-2026 repo for the cluster workflow.

Prereqs for downstream analyses:
  - entanglement-browse (#22)  — needs family → human gene symbols
  - entanglement-go-enrich (#23) — needs human gene symbols for GO lookup

Output:
  bcns_family_to_human_gene.tsv  with columns:
    family_id     e.g. Simakov2022BCnS_genefamily_11671
    alg           the BCnS ALG letter (A1a, D, F, … Qb)
    human_gene    the human gene identifier from the RBH
    human_scaf    scaffold/chromosome of the human ortholog
    source        'human_rbh' (pass 1) or 'hmm_consensus' (pass 2; via helper)
    note          optional provenance
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def _parse_rbh(rbh_path: Path) -> pd.DataFrame:
    """Read an ODP RBH TSV (tab-separated, header row)."""
    return pd.read_csv(rbh_path, sep="\t", dtype=str, low_memory=False)


def _pass1_from_human_rbh(alg_rbh: Path, human_rbh: Path) -> pd.DataFrame:
    """Join BCnSSimakov2022.rbh with the human per-species RBH by the
    `rbh` family-ID column. Return (family_id, alg, human_gene, human_scaf)."""
    alg = _parse_rbh(alg_rbh)
    human = _parse_rbh(human_rbh)

    if "rbh" not in alg.columns:
        raise SystemExit(f"{alg_rbh} missing 'rbh' column")
    if "rbh" not in human.columns:
        raise SystemExit(f"{human_rbh} missing 'rbh' column")

    # Identify the "human gene" column. Convention: "<SpeciesToken>_gene".
    gene_cols = [c for c in human.columns if c.endswith("_gene")
                 and not c.startswith("BCnSSimakov2022")]
    if not gene_cols:
        raise SystemExit(f"No species `_gene` column found in {human_rbh}. "
                         f"Columns: {list(human.columns)}")
    human_gene_col = gene_cols[0]
    human_scaf_col = human_gene_col.replace("_gene", "_scaf")

    # Identify the ALG-letter column in alg_rbh. Convention: `gene_group`.
    alg_letter_col = "gene_group" if "gene_group" in alg.columns else None

    alg_slim = alg[["rbh"] + ([alg_letter_col] if alg_letter_col else [])].copy()
    alg_slim = alg_slim.rename(columns={"rbh": "family_id",
                                        alg_letter_col: "alg"} if alg_letter_col
                                        else {"rbh": "family_id"})
    human_slim = human[["rbh", human_gene_col] +
                       ([human_scaf_col] if human_scaf_col in human.columns else [])].copy()
    human_slim = human_slim.rename(columns={
        "rbh": "family_id",
        human_gene_col: "human_gene",
        human_scaf_col: "human_scaf" if human_scaf_col in human_slim.columns else None,
    })

    merged = alg_slim.merge(human_slim, on="family_id", how="left")
    merged["source"] = merged["human_gene"].apply(
        lambda v: "human_rbh" if isinstance(v, str) and v.strip() else ""
    )
    merged["note"] = ""
    return merged[["family_id", "alg", "human_gene", "human_scaf", "source", "note"]]


def _pass2_merge_hmm(pass1: pd.DataFrame, hmm_map_tsv: Path | None) -> pd.DataFrame:
    """Optionally fill holes using a TSV of family_id → human_gene produced
    by the cluster-side HMM-consensus DIAMOND script. Rows whose human_gene
    was empty after pass 1 get filled in; source column becomes
    'hmm_consensus'."""
    if hmm_map_tsv is None:
        return pass1
    if not hmm_map_tsv.exists():
        print(f"[build-family-naming-map] --hmm-map not found: {hmm_map_tsv}; "
              f"skipping pass 2 merge", file=sys.stderr)
        return pass1
    hmm = pd.read_csv(hmm_map_tsv, sep="\t", dtype=str)
    if "family_id" not in hmm.columns or "human_gene" not in hmm.columns:
        raise SystemExit(
            f"{hmm_map_tsv} must have `family_id` and `human_gene` columns"
        )
    missing_mask = pass1["human_gene"].fillna("") == ""
    # Index hmm-derived rows by family_id for fast update.
    hmm_idx = hmm.set_index("family_id")["human_gene"].to_dict()
    def _fill(row):
        if row["human_gene"] and isinstance(row["human_gene"], str):
            return row
        if row["family_id"] in hmm_idx:
            row["human_gene"] = hmm_idx[row["family_id"]]
            row["source"] = "hmm_consensus"
        return row
    pass1 = pass1.apply(_fill, axis=1)
    return pass1


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="egt build-family-naming-map",
        description=(
            "Build BCnSSimakov2022 family → human gene ID map. Pass 1 uses "
            "the human per-species RBH (fast). Optional pass 2 fills "
            "remaining holes from a precomputed HMM-consensus DIAMOND TSV."
        ),
    )
    parser.add_argument("--alg-rbh", required=True, type=Path,
                        help="BCnSSimakov2022 ALG RBH file (from reference_data/ or odp LG_db).")
    parser.add_argument("--human-rbh", required=True, type=Path,
                        help="Per-species RBH file for Homo sapiens "
                             "(BCnSSimakov2022_Homosapiens-9606-*.rbh).")
    parser.add_argument("--hmm-map", default=None, type=Path,
                        help="Optional TSV (family_id, human_gene) from the HMM-consensus "
                             "DIAMOND pass (see post_analyses/build_family_naming_map).")
    parser.add_argument("--output", required=True, type=Path,
                        help="Output TSV. Conventional location: "
                             "src/egt/data/bcns_family_to_human_gene.tsv (committed in egt).")
    args = parser.parse_args(argv)

    pass1 = _pass1_from_human_rbh(args.alg_rbh, args.human_rbh)
    result = _pass2_merge_hmm(pass1, args.hmm_map)

    # Report coverage.
    total = len(result)
    covered = (result["human_gene"].fillna("") != "").sum()
    by_source = result["source"].value_counts().to_dict()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, sep="\t", index=False)
    print(f"[build-family-naming-map] wrote {args.output}", file=sys.stderr)
    print(f"  families total  : {total}", file=sys.stderr)
    print(f"  with human gene : {covered} ({100 * covered // max(1,total)}%)",
          file=sys.stderr)
    print(f"  source breakdown: {by_source}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
