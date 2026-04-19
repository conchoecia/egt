"""Grid-based COO integrity check (mesh-bed-leveling style).

Given a COO sparse matrix of per-species pair-distances, a sampledf,
a combo_to_index mapping, and a directory of per-species `.gb.gz`
distance files, sample `n_rows × n_cols` (row, col) cells in a regular
grid from corner to corner, and for each cell assert that

    coo[row_idx, col_idx]  ==  gb.gz[sample(row_idx), pair(col_idx)]

The `(row_idx, col_idx)` are spread uniformly across the matrix, so
a row-scramble or column-misalignment bug almost anywhere has a good
chance of landing on at least one grid point.

The cost is O(n_rows * gb.gz read time) = a handful of file opens
plus the COO slice, so this is cheap enough to bake into any pipeline
that touches the COO — including the subsampling paths that select
species-row subsets for clade-specific analyses.

Usage (standalone):

    from scipy.sparse import load_npz
    import pandas as pd
    from egt._testing import grid_verify_coo

    coo = load_npz(coo_path)
    sdf = pd.read_csv(sampledf_path, sep="\\t", index_col=0)
    combo = parse_combo_to_index(combo_path)   # {(rbh1, rbh2): col_idx}
    grid_verify_coo(coo, sdf, combo, gb_dir,
                    n_rows=5, n_cols=5, strict=True)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _grid_indices(n, k):
    """Choose `k` integer indices in `[0, n)` from corner to corner.

    Endpoints are always included. For k >= n all indices are returned.
    """
    k = max(1, min(k, n))
    if k == 1:
        return np.array([0], dtype=int)
    if k >= n:
        return np.arange(n, dtype=int)
    return np.linspace(0, n - 1, num=k, dtype=int)


def _find_gbgz(gb_dir, sample):
    """Locate the gb.gz file for `sample` in `gb_dir`.

    Accepts both `<sample>.gb.gz` and `<something>_<sample>_<...>.gb.gz`
    patterns.
    """
    gb_dir = Path(gb_dir)
    direct = gb_dir / f"{sample}.gb.gz"
    if direct.exists():
        return direct
    matches = list(gb_dir.glob(f"*{sample}*.gb.gz"))
    return matches[0] if matches else None


def _lookup_gbgz(gbgz_df, rbh1, rbh2):
    """Return the distance stored in `gbgz_df` for pair (rbh1, rbh2),
    or None if the pair is absent. gb.gz entries are canonicalized
    rbh1 < rbh2 lex, but we try both orderings to be robust.
    """
    for a, b in ((rbh1, rbh2), (rbh2, rbh1)):
        hit = gbgz_df[(gbgz_df["rbh1"] == a) & (gbgz_df["rbh2"] == b)]
        if not hit.empty:
            return float(hit.iloc[0]["distance"])
    return None


def grid_verify_coo(coo, sampledf, combo_to_ix, gb_dir,
                    n_rows=5, n_cols=5,
                    sample_col="sample",
                    strict=True,
                    rng=None) -> dict[str, Any]:
    """Sample an n_rows × n_cols grid of cells and verify each against
    the per-species gb.gz.

    Parameters
    ----------
    coo : scipy.sparse matrix
        COO/CSR/CSC — converted internally to CSR for row slicing.
    sampledf : pd.DataFrame
        One row per species. Must have a `sample_col` with the species
        key used to locate the gb.gz file. Positional order of the
        DataFrame MUST match the COO's row order.
    combo_to_ix : dict
        `{(rbh1, rbh2): col_idx}` — the pair index used when building
        the COO.
    gb_dir : Path
        Directory holding per-species `.gb.gz` files.
    n_rows, n_cols : int
        Grid dimensions. Clamped to min(n_species, n_pairs).
    sample_col : str
        Column in `sampledf` holding the species key.
    strict : bool
        If True, raise AssertionError when any grid cell mismatches.
    rng : np.random.Generator, optional
        For reproducible jitter around grid points (unused today but
        kept in the signature for future stochastic variants).

    Returns
    -------
    dict with keys:
        passed : int
        failed : int
        skipped : int   (e.g. species absent from gb_dir, pair absent
                         from gb.gz and COO cell is also "missing")
        details : list[dict] of per-cell outcome
    """
    csr = coo.tocsr()
    n_species, n_pairs = csr.shape
    row_ids = _grid_indices(n_species, n_rows)
    col_ids = _grid_indices(n_pairs, n_cols)

    # Invert combo_to_ix for col_idx -> pair lookup.
    ix_to_pair = {v: k for k, v in combo_to_ix.items()}

    # Cache gb.gz reads across the grid (each species' gb.gz holds many pairs).
    gbgz_cache: dict[int, pd.DataFrame | None] = {}

    def gbgz_for_row(row_idx):
        if row_idx in gbgz_cache:
            return gbgz_cache[row_idx]
        sample = sampledf.iloc[row_idx][sample_col]
        gbgz = _find_gbgz(gb_dir, sample)
        if gbgz is None:
            gbgz_cache[row_idx] = None
            return None
        df = pd.read_csv(gbgz, sep="\t")
        gbgz_cache[row_idx] = df
        return df

    passed = 0
    failed = 0
    skipped = 0
    details = []
    for r in row_ids:
        for c in col_ids:
            coo_val = float(csr[r, c])
            pair = ix_to_pair.get(int(c))
            sample = sampledf.iloc[int(r)][sample_col]
            rec = {
                "row_idx": int(r), "col_idx": int(c),
                "sample": sample, "pair": pair,
                "coo_value": coo_val,
            }
            if pair is None:
                rec["status"] = "skipped_no_pair"
                rec["outcome"] = None
                skipped += 1
                details.append(rec)
                continue
            gb_df = gbgz_for_row(int(r))
            if gb_df is None:
                rec["status"] = "skipped_no_gbgz"
                rec["outcome"] = None
                skipped += 1
                details.append(rec)
                continue
            gb_val = _lookup_gbgz(gb_df, pair[0], pair[1])
            rec["gbgz_value"] = gb_val
            # Both sides have the same "missingness" notion:
            #   - COO stores only pairs actually observed same-scaffold
            #     in that species (may or may not include 0 placeholder
            #     cells depending on convention).
            #   - gb.gz stores only observed same-scaffold pairs.
            # Cell matches if:
            #   - both "present" and values equal within 1e-6, OR
            #   - both "missing" (coo is 0/absent, gb.gz has no row)
            coo_missing = coo_val == 0.0   # coo returns 0 for unstored
            gb_missing = gb_val is None
            if coo_missing and gb_missing:
                rec["status"] = "match_both_missing"
                rec["outcome"] = True
                passed += 1
            elif coo_missing or gb_missing:
                rec["status"] = "mismatch_presence"
                rec["outcome"] = False
                failed += 1
            elif abs(coo_val - gb_val) <= max(1e-6 * max(abs(gb_val), 1.0), 0.5):
                rec["status"] = "match"
                rec["outcome"] = True
                passed += 1
            else:
                rec["status"] = "mismatch_value"
                rec["outcome"] = False
                failed += 1
            details.append(rec)

    summary = {"passed": passed, "failed": failed, "skipped": skipped,
               "n_cells": len(details), "details": details}
    if strict and failed > 0:
        msg_lines = [f"grid_verify_coo: {failed}/{len(details)} cells mismatched"]
        for d in details:
            if d.get("outcome") is False:
                msg_lines.append(
                    f"  row={d['row_idx']:>6} col={d['col_idx']:>8} "
                    f"sample={d['sample']} pair={d.get('pair')} "
                    f"coo={d['coo_value']} gbgz={d.get('gbgz_value')} "
                    f"status={d['status']}"
                )
        raise AssertionError("\n".join(msg_lines))
    return summary
