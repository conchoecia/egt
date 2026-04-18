"""egt pigeonhole-check — null-model test for shared ALG fusions.

A species with a small karyotype must, by pigeonhole, co-localize many
ALG pairs on the same chromosome even under random assortment. Without a
null, any apparent "shared fusion" signal between small-karyotype clades
(e.g. nematodes with 5–6 chromosomes) could be statistical necessity
rather than shared ancestry.

For each species:
  n_algs     = number of BCnS ALGs detected (presence columns sum)
  n_chroms   = haploid chromosome count (species_chrom_counts.tsv)
  obs_pairs  = observed co-localized pairs (pair columns sum)

Null: randomly assort n_algs into n_chroms bins (uniform with replacement);
count pairs falling on the same bin; repeat N times. Empirical p-value =
P(null_pairs >= obs_pairs).

Per-clade roll-up: for each clade in --clade-groupings, aggregate
species-level counts and report mean null ± σ vs mean observed, plus
one-sided Fisher-combined p-value across species in the clade.
"""
from __future__ import annotations

import argparse
import re
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


def _is_pair_col(name: str) -> bool:
    return name.startswith("(") and "," in name and name.endswith(")")


def _load_inputs(presence_fusions_tsv: Path, chrom_counts_tsv: Path):
    pf = pd.read_csv(presence_fusions_tsv, sep="\t", dtype=str, low_memory=False)
    cc = pd.read_csv(chrom_counts_tsv, sep="\t", dtype=str)
    # canonicalize
    if "species" in pf.columns:
        pf = pf.set_index("species")
    if "sample" in cc.columns:
        cc = cc.set_index("sample")
    # chromosome counts: coerce to int where possible
    cc["chromosomes"] = pd.to_numeric(cc["chromosomes"], errors="coerce")

    alg_cols = [c for c in pf.columns if c not in {"taxid", "taxidstring", "changestrings"}
                and not _is_pair_col(c)]
    pair_cols = [c for c in pf.columns if _is_pair_col(c)]

    # Coerce presence and pair columns to 0/1 int.
    def _to01(x):
        try:
            return int(float(x)) > 0
        except Exception:
            return False

    n_algs = pf[alg_cols].map(_to01).sum(axis=1).astype(int)
    obs_pairs = pf[pair_cols].map(_to01).sum(axis=1).astype(int)
    return pf, cc, alg_cols, pair_cols, n_algs, obs_pairs


def _lineage_match(taxidstring: str, clade_taxid: int) -> bool:
    """True if `clade_taxid` appears anywhere in the species' taxidstring."""
    if not isinstance(taxidstring, str):
        return False
    ids = [x.strip() for x in taxidstring.split(";") if x.strip()]
    return str(clade_taxid) in ids


def _resolve_clade_spec(spec: str, ncbi=None) -> tuple[int, str]:
    """Accept either 'Name' (look up taxid) or 'Name:taxid' or plain taxid."""
    spec = spec.strip()
    if ":" in spec:
        name, tid = spec.split(":", 1)
        return int(tid), name
    if spec.isdigit() or (spec.startswith("-") and spec[1:].isdigit()):
        tid = int(spec)
        if ncbi is not None:
            nm = ncbi.get_taxid_translator([tid]).get(tid, spec)
        else:
            nm = spec
        return tid, nm
    # treat as name
    if ncbi is None:
        raise ValueError(
            f"Clade '{spec}' is a name but --clade-groupings passed without "
            "NCBITaxa available; pass 'Name:taxid' or just the numeric taxid."
        )
    t = ncbi.get_name_translator([spec])
    if spec not in t:
        raise ValueError(f"Clade name '{spec}' not resolvable to a taxid")
    return int(t[spec][0]), spec


def _simulate_pairs(n_algs: int, n_chroms: int, n_sims: int,
                    rng: np.random.Generator) -> np.ndarray:
    """Return an array of simulated pair-count per simulation."""
    if n_algs < 2 or n_chroms < 1:
        return np.zeros(n_sims, dtype=int)
    # Assign each of n_algs to a uniformly-random chromosome bin.
    assign = rng.integers(0, n_chroms, size=(n_sims, n_algs))
    # Count pairs sharing a bin: for each sim, bincount+choose(2,k).
    out = np.empty(n_sims, dtype=int)
    for s in range(n_sims):
        _, counts = np.unique(assign[s], return_counts=True)
        out[s] = int(np.sum(counts * (counts - 1) // 2))
    return out


def _empirical_p(obs: int, sims: np.ndarray) -> float:
    """One-sided P(null >= obs)."""
    return float((sims >= obs).sum() + 1) / (len(sims) + 1)


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="egt pigeonhole-check",
        description=(
            "Null-model test for shared ALG fusions under random ALG-to-chromosome "
            "assortment. Useful when comparing fusion patterns between clades with "
            "very different karyotype sizes."
        ),
    )
    parser.add_argument("--presence-fusions", required=True, type=Path,
                        help="Path to per_species_ALG_presence_fusions.tsv from step 4.")
    parser.add_argument("--chrom-counts", required=True, type=Path,
                        help="Path to species_chrom_counts.tsv (sample, chromosomes).")
    parser.add_argument("--clade-groupings", default=None,
                        help="Comma-separated list of clade names (or Name:taxid, or "
                             "taxid). If omitted, reports species-level only.")
    parser.add_argument("--n-simulations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args(argv)

    # Optional: NCBITaxa for name→taxid lookups (only if user passes names).
    ncbi = None
    needs_ncbi = bool(args.clade_groupings) and any(
        not s.strip().lstrip("-").isdigit() and ":" not in s
        for s in (args.clade_groupings or "").split(",")
    )
    if needs_ncbi:
        try:
            from ete4 import NCBITaxa  # type: ignore
            ncbi = NCBITaxa()
        except Exception as exc:
            raise SystemExit(f"Need ete4 NCBITaxa for clade-name lookup: {exc}")

    pf, cc, alg_cols, pair_cols, n_algs, obs_pairs = _load_inputs(
        args.presence_fusions, args.chrom_counts
    )
    joined = pd.DataFrame({
        "n_algs": n_algs,
        "obs_pairs": obs_pairs,
        "taxidstring": pf["taxidstring"] if "taxidstring" in pf.columns else "",
        "taxid": pf["taxid"] if "taxid" in pf.columns else "",
    }).join(cc[["chromosomes"]], how="left")
    joined["chromosomes"] = joined["chromosomes"].fillna(0).astype(int)

    # Species-level sim.
    rng = np.random.default_rng(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    per_species_rows = []
    for sp, row in joined.iterrows():
        n_a = int(row["n_algs"])
        n_c = int(row["chromosomes"])
        obs = int(row["obs_pairs"])
        if n_a < 2 or n_c < 1:
            per_species_rows.append({
                "species": sp, "n_algs": n_a, "n_chroms": n_c, "obs_pairs": obs,
                "null_mean": np.nan, "null_sd": np.nan, "fold_enrichment": np.nan,
                "p_value": np.nan,
            })
            continue
        sims = _simulate_pairs(n_a, n_c, args.n_simulations, rng)
        mean, sd = float(sims.mean()), float(sims.std(ddof=1))
        fold = (obs / mean) if mean > 0 else np.nan
        per_species_rows.append({
            "species": sp, "n_algs": n_a, "n_chroms": n_c, "obs_pairs": obs,
            "null_mean": mean, "null_sd": sd, "fold_enrichment": fold,
            "p_value": _empirical_p(obs, sims),
        })

    per_species = pd.DataFrame(per_species_rows)
    per_species_path = args.out_dir / "per_species.tsv"
    per_species.to_csv(per_species_path, sep="\t", index=False)

    # Per-clade rollup.
    if args.clade_groupings:
        clade_specs = [_resolve_clade_spec(s, ncbi) for s in args.clade_groupings.split(",") if s.strip()]
        rows = []
        for tid, name in clade_specs:
            mask = joined["taxidstring"].apply(lambda s: _lineage_match(s, tid))
            sp_ok = mask & (joined["n_algs"] >= 2) & (joined["chromosomes"] >= 1)
            n_species = int(sp_ok.sum())
            if n_species == 0:
                rows.append({
                    "clade": name, "clade_taxid": tid, "n_species": 0,
                    "obs_mean": np.nan, "null_mean": np.nan, "null_sd": np.nan,
                    "fold_enrichment": np.nan, "fisher_combined_log10p": np.nan,
                })
                continue
            sub = joined.loc[sp_ok]
            obs_mean = float(sub["obs_pairs"].mean())
            null_means = []
            null_sds = []
            p_values = []
            for sp, r in sub.iterrows():
                n_a = int(r["n_algs"]); n_c = int(r["chromosomes"])
                sims = _simulate_pairs(n_a, n_c, args.n_simulations, rng)
                null_means.append(float(sims.mean()))
                null_sds.append(float(sims.std(ddof=1)))
                p_values.append(_empirical_p(int(r["obs_pairs"]), sims))
            null_mean_overall = float(np.mean(null_means))
            null_sd_overall = float(np.sqrt(np.mean(np.square(null_sds))))
            fold = obs_mean / null_mean_overall if null_mean_overall > 0 else np.nan
            # Fisher's method in log space.
            clipped = np.clip(np.array(p_values), 1e-6, 1.0)
            chi2 = -2 * np.log(clipped).sum()
            # Chi-squared df = 2k; use survival fn for tail. We'll just report
            # log10(p) via survival; avoid scipy import by returning the
            # standardized Z of chi2 at df 2k.
            df = 2 * len(p_values)
            # Approximate log10p using Wilson-Hilferty cube-root transform.
            z = (((chi2 / df) ** (1/3)) - (1 - 2/(9*df))) / np.sqrt(2/(9*df))
            # log10-survival-from-normal ≈ -z^2/(2 ln 10)
            log10p = -float(z**2 / (2 * np.log(10))) if z > 0 else 0.0
            rows.append({
                "clade": name, "clade_taxid": tid, "n_species": n_species,
                "obs_mean": obs_mean,
                "null_mean": null_mean_overall,
                "null_sd": null_sd_overall,
                "fold_enrichment": fold,
                "fisher_combined_log10p": log10p,
            })
        per_clade = pd.DataFrame(rows)
        per_clade_path = args.out_dir / "per_clade.tsv"
        per_clade.to_csv(per_clade_path, sep="\t", index=False)
        print(f"[pigeonhole-check] wrote {per_clade_path}", file=sys.stderr)

    print(f"[pigeonhole-check] wrote {per_species_path}", file=sys.stderr)
    print(f"  species processed: {len(per_species_rows)}", file=sys.stderr)
    print(f"  n-simulations    : {args.n_simulations}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
