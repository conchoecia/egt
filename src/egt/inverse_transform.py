#!/usr/bin/env python
"""Inverse-transform a fitted UMAP manifold back into feature space.

The UMAP reductions in :mod:`egt.phylotreeumap` place each *sample* (genome /
species) as a point in a 2-D embedding whose coordinates are driven by a long
vector of pairwise gene-family (RBH locus) distances. Reading that plot, a user
can see clusters of genomes — but the plot does not say *which* gene-family
pairs are responsible for a region of the manifold.

This module closes that loop. UMAP supports an approximate ``inverse_transform``
that maps a 2-D embedding coordinate back into the original high-dimensional
feature space. Given a fitted reducer and the ordered list of feature
(RBH-pair) names, ``inverse_transform`` at an embedding coordinate yields one
reconstructed distance value per gene-family pair. The pairs with the most
extreme values are the features characteristic of that location, and the
difference between two coordinates ranks the pairs that distinguish one region
of the manifold from another.

Two actions:

``egt inverse-transform fit``
    Fit a dense, Euclidean UMAP reducer from a COO distance matrix and persist
    it (``.reducer``) together with the ordered feature list (``.features``) and
    the 2-D embedding (``.embedding.tsv``). The embedding tells you which
    coordinates to query.

``egt inverse-transform query``
    Load a persisted reducer + feature list, inverse-transform one or two
    embedding coordinates, and write a ranked table of characteristic gene-family
    pairs (single point) or the pairs that most distinguish two points.

Notes
-----
- ``inverse_transform`` is unsupported by UMAP for sparse input and for the
  ``metric='precomputed'`` reductions, so ``fit`` densifies the COO matrix and
  fits with the default Euclidean metric. At genome-by-(tens-of-thousands-of-
  pairs) scale the dense matrix can be large; ``fit`` prints the dense shape so
  the memory cost is visible.
- UMAP's inverse transform is approximate by construction: it reconstructs
  plausible feature values for a coordinate, not observed data. Treat the output
  as "what is characteristic here", not as a measurement.
"""
from __future__ import annotations

import argparse
import pickle
import sys

import numpy as np
import pandas as pd
from scipy.sparse import load_npz

from egt.phylotreeumap import algcomboix_file_to_dict


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def _ordered_feature_table(alg_combo_to_ix: dict, n_features: int) -> pd.DataFrame:
    """Return a DataFrame indexed by column position 0..n_features-1.

    Columns: ``feature_index``, ``rbh1``, ``rbh2``. Column positions that have
    no entry in ``alg_combo_to_ix`` (a pair never observed in any sample but
    still allotted a stable column) are labelled ``NA``.
    """
    ix_to_combo = {v: k for k, v in alg_combo_to_ix.items()}
    rows = []
    for ix in range(n_features):
        rbh1, rbh2 = ix_to_combo.get(ix, ("NA", "NA"))
        rows.append({"feature_index": ix, "rbh1": rbh1, "rbh2": rbh2})
    return pd.DataFrame(rows)


def _features_path_for(reducer_path: str) -> str:
    if reducer_path.endswith(".reducer"):
        return reducer_path[: -len(".reducer")] + ".features"
    return reducer_path + ".features"


# --------------------------------------------------------------------------- #
# fit
# --------------------------------------------------------------------------- #
def run_fit(args) -> int:
    import umap

    print(f"[inverse-transform fit] Loading COO matrix: {args.coo}")
    mat = load_npz(args.coo)
    n_samples, n_features = mat.shape
    print(f"  Matrix shape: {n_samples} samples x {n_features} feature columns")

    print(f"  Loading feature index: {args.algcomboix}")
    alg_combo_to_ix = algcomboix_file_to_dict(args.algcomboix)
    max_ix = max(alg_combo_to_ix.values()) if alg_combo_to_ix else -1
    if max_ix >= n_features:
        print(
            f"Error: feature index file references column {max_ix} but the COO "
            f"matrix only has {n_features} columns.",
            file=sys.stderr,
        )
        return 2

    # Densify. UMAP's inverse_transform does not accept sparse input. Build the
    # dense array explicitly so we can distinguish a stored 0 (a real 0-distance
    # observation) from a never-observed cell, and fill the latter as requested.
    coo = mat.tocoo()
    if args.fill_missing == "max":
        fill_value = float(coo.data.max()) if coo.nnz else 0.0
    else:
        fill_value = 0.0
    print(
        f"  Densifying ({n_samples} x {n_features}); "
        f"missing cells -> {args.fill_missing} ({fill_value})"
    )
    dense = np.full(mat.shape, fill_value, dtype=float)
    dense[coo.row, coo.col] = coo.data.astype(float)

    print(
        f"  Fitting UMAP (n_neighbors={args.n_neighbors}, "
        f"min_dist={args.min_dist}, metric=euclidean, n_components=2)"
    )
    reducer = umap.UMAP(
        low_memory=True,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        random_state=args.random_state,
        # random_state already forces single-threaded execution; set n_jobs=1
        # explicitly so UMAP does not warn about overriding it.
        n_jobs=1,
    )
    reducer.fit(dense)

    # Sample names for the embedding table.
    sample_names = [str(i) for i in range(n_samples)]
    if args.sample_df:
        sdf = pd.read_csv(args.sample_df, sep="\t")
        if args.sample_col not in sdf.columns:
            print(
                f"Error: --sample-col {args.sample_col!r} not in {args.sample_df}",
                file=sys.stderr,
            )
            return 2
        if len(sdf) != n_samples:
            print(
                f"Error: sample df has {len(sdf)} rows but the matrix has "
                f"{n_samples} samples.",
                file=sys.stderr,
            )
            return 2
        sample_names = sdf[args.sample_col].astype(str).tolist()

    reducer_path = args.out_prefix + ".reducer"
    features_path = args.out_prefix + ".features"
    embedding_path = args.out_prefix + ".embedding.tsv"

    with open(reducer_path, "wb") as fh:
        pickle.dump(reducer, fh)
    print(f"  Saved reducer: {reducer_path}")

    feat = _ordered_feature_table(alg_combo_to_ix, n_features)
    feat.to_csv(features_path, sep="\t", index=False)
    print(f"  Saved feature list ({n_features} features): {features_path}")

    emb = pd.DataFrame(
        {
            "sample": sample_names,
            "UMAP1": reducer.embedding_[:, 0],
            "UMAP2": reducer.embedding_[:, 1],
        }
    )
    emb.to_csv(embedding_path, sep="\t", index=False)
    print(f"  Saved embedding: {embedding_path}")
    print(
        "  Inspect the embedding (or its plot) to choose coordinates, then run "
        "`egt inverse-transform query`."
    )
    return 0


# --------------------------------------------------------------------------- #
# query
# --------------------------------------------------------------------------- #
def run_query(args) -> int:
    reducer_path = args.reducer
    features_path = args.features or _features_path_for(reducer_path)

    print(f"[inverse-transform query] Loading reducer: {reducer_path}")
    with open(reducer_path, "rb") as fh:
        reducer = pickle.load(fh)

    print(f"  Loading feature list: {features_path}")
    feat = pd.read_csv(features_path, sep="\t").sort_values("feature_index")

    two_point = args.embx2 is not None and args.emby2 is not None
    if (args.embx2 is None) != (args.emby2 is None):
        print("Error: pass both --embx2 and --emby2, or neither.", file=sys.stderr)
        return 2

    query_points = [[args.embx, args.emby]]
    if two_point:
        query_points.append([args.embx2, args.emby2])
        print(
            f"  Comparing point 1 ({args.embx}, {args.emby}) vs "
            f"point 2 ({args.embx2}, {args.emby2})"
        )
    else:
        print(f"  Analysing single point ({args.embx}, {args.emby})")
    query_points = np.asarray(query_points, dtype=float)

    print("  Running UMAP inverse transform ...")
    inv = reducer.inverse_transform(query_points)

    n_features = inv.shape[1]
    if len(feat) != n_features:
        print(
            f"Error: feature list has {len(feat)} rows but the reducer "
            f"reconstructs {n_features} features. Mismatched files?",
            file=sys.stderr,
        )
        return 2

    res = feat[["feature_index", "rbh1", "rbh2"]].reset_index(drop=True)

    if two_point:
        res["Val_P1"] = inv[0]
        res["Val_P2"] = inv[1]
        # Positive Difference => higher near point 1; negative => higher near point 2.
        res["Difference"] = res["Val_P1"] - res["Val_P2"]
        res["Abs_Difference"] = res["Difference"].abs()
        res = res.sort_values("Abs_Difference", ascending=False)

        out_name = args.out or (
            reducer_path
            + f".inv_diff_{args.embx}_{args.emby}_vs_{args.embx2}_{args.emby2}.tsv"
        )
        print(f"\n--- Top {args.top_n} distinguishing pairs (P1 vs P2) ---")
        print(
            res[["rbh1", "rbh2", "Difference", "Val_P1", "Val_P2"]]
            .head(args.top_n)
            .to_string(index=False)
        )
    else:
        res["Value"] = inv[0]
        res = res.sort_values("Value", ascending=False)

        out_name = args.out or (
            reducer_path + f".inv_vals_{args.embx}_{args.emby}.tsv"
        )
        print(f"\n--- Top {args.top_n} highest-value pairs (farthest) ---")
        print(res.head(args.top_n).to_string(index=False))
        print(f"\n--- Top {args.top_n} lowest-value pairs (closest) ---")
        print(res.tail(args.top_n).to_string(index=False))

    res.to_csv(out_name, sep="\t", index=False)
    print(f"\n  Saved: {out_name}")
    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="egt inverse-transform",
        description=(
            "Inverse-transform a UMAP manifold to find the gene-family pairs "
            "characteristic of an embedding location."
        ),
    )
    sub = parser.add_subparsers(dest="action", required=True, metavar="<action>")

    p_fit = sub.add_parser(
        "fit",
        help="Fit a dense Euclidean UMAP reducer and persist it for querying.",
    )
    p_fit.add_argument("--coo", required=True, help="Path to the COO matrix (.npz).")
    p_fit.add_argument(
        "--algcomboix",
        required=True,
        help="Feature-index file mapping each (rbh1, rbh2) pair to its column.",
    )
    p_fit.add_argument(
        "--sample-df",
        default=None,
        help="Optional TSV of samples (one row per matrix row) for embedding labels.",
    )
    p_fit.add_argument(
        "--sample-col",
        default="sample",
        help="Column in --sample-df holding the sample name (default: sample).",
    )
    p_fit.add_argument("--n-neighbors", type=int, default=15, dest="n_neighbors")
    p_fit.add_argument("--min-dist", type=float, default=0.1, dest="min_dist")
    p_fit.add_argument(
        "--fill-missing",
        choices=("zero", "max"),
        default="zero",
        help=(
            "How to fill never-observed cells when densifying (stored 0s are "
            "kept as 0 either way). Default: zero."
        ),
    )
    p_fit.add_argument(
        "--random-state",
        type=int,
        default=42,
        dest="random_state",
        help="UMAP random_state for a reproducible embedding (default: 42).",
    )
    p_fit.add_argument(
        "--out-prefix",
        required=True,
        dest="out_prefix",
        help="Output prefix; writes <prefix>.reducer/.features/.embedding.tsv.",
    )
    p_fit.set_defaults(func=run_fit)

    p_q = sub.add_parser(
        "query",
        help="Inverse-transform one or two embedding coordinates.",
    )
    p_q.add_argument("--reducer", required=True, help="Persisted .reducer file.")
    p_q.add_argument(
        "--features",
        default=None,
        help="Feature list (.features). Default: alongside the reducer.",
    )
    p_q.add_argument("--embx", type=float, required=True, help="Point 1 x-coordinate.")
    p_q.add_argument("--emby", type=float, required=True, help="Point 1 y-coordinate.")
    p_q.add_argument("--embx2", type=float, default=None, help="Optional point 2 x.")
    p_q.add_argument("--emby2", type=float, default=None, help="Optional point 2 y.")
    p_q.add_argument(
        "--top-n",
        type=int,
        default=20,
        dest="top_n",
        help="How many ranked pairs to print (default: 20).",
    )
    p_q.add_argument(
        "--out",
        default=None,
        help="Output TSV path. Default: derived from the reducer + coordinates.",
    )
    p_q.set_defaults(func=run_query)

    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
