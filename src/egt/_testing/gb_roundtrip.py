"""gb.gz <-> COO round-trip helper (TODO_tests.md section K).

The invariant: all information in the per-species gb.gz files is
losslessly represented by the COO, so decomposing the COO back into
per-species gb.gz files must reproduce the originals exactly.

This helper lives in ``egt._testing`` rather than the main library
because it's a correctness-checking utility, not a production code
path; importing it from tests is fine.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix


def decompose_coo_to_gbgz(
    coo,
    sampledf,
    combo_to_ix,
    out_dir,
    sample_col: str = "sample",
):
    """Reverse ``phylotreeumap.construct_coo_matrix_from_sampledf``.

    For every row of the COO, emit a per-species gb.gz that contains
    one row per stored (col, value) with the ``rbh1``/``rbh2`` pair
    recovered from ``combo_to_ix``. The emitted gb.gz uses the same
    three-column schema (``rbh1 TAB rbh2 TAB distance``) as the
    original pipeline's gb.gz files.

    Parameters
    ----------
    coo : scipy.sparse matrix
        COO / CSR / CSC accepted — converted internally to CSR for
        row slicing.
    sampledf : pandas.DataFrame
        One row per species, same positional order as the COO's rows.
        Must contain a ``sample_col`` with the species key.
    combo_to_ix : dict[(rbh1, rbh2), int]
        Pair-tuple -> column index map; the same dict used when the
        COO was built.
    out_dir : str | Path
        Directory to write the decomposed ``{sample}.gb.gz`` files into.
        Created if missing.
    sample_col : str
        Column in ``sampledf`` that holds the species key.

    Returns
    -------
    dict[str, Path]
        Mapping of sample key -> output path.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csr = coo.tocsr()
    n_species, n_pairs = csr.shape
    if len(sampledf) != n_species:
        raise ValueError(
            f"sampledf has {len(sampledf)} rows, but COO has "
            f"{n_species}. They must match."
        )

    # Invert combo_to_ix once for col_idx -> (rbh1, rbh2) lookup.
    ix_to_pair: dict[int, tuple[str, str]] = {
        int(v): k for k, v in combo_to_ix.items()
    }
    if len(ix_to_pair) != len(combo_to_ix):
        raise ValueError(
            "combo_to_ix has duplicate integer values — cannot invert."
        )

    # Validate column coverage: every COO column we might store must
    # have a pair in ix_to_pair. We check max used col later; here
    # just ensure 0..n_pairs-1 coverage if needed.

    out_paths: dict[str, Path] = {}
    for row_pos in range(n_species):
        sample = str(sampledf.iloc[row_pos][sample_col])
        # Pull the row's stored (col, value) entries. CSR.getrow returns
        # a 1-by-n CSR; .indices gives the column positions and .data
        # the stored values.
        row_csr = csr.getrow(row_pos)
        cols = row_csr.indices.tolist()
        vals = row_csr.data.tolist()
        rbh1_list: list[str] = []
        rbh2_list: list[str] = []
        dist_list: list = []
        for c, v in zip(cols, vals):
            if int(c) not in ix_to_pair:
                raise KeyError(
                    f"COO stored a cell at col {c} but that index is "
                    f"missing from combo_to_ix. Either the COO was "
                    f"built with a different combo_to_ix or the map "
                    f"has been truncated."
                )
            r1, r2 = ix_to_pair[int(c)]
            # Re-canonicalize rbh1 < rbh2 lexicographically, matching
            # the original gb.gz emitter.
            a, b = (r1, r2) if r1 < r2 else (r2, r1)
            rbh1_list.append(a)
            rbh2_list.append(b)
            dist_list.append(v)
        out_df = pd.DataFrame({
            "rbh1": rbh1_list,
            "rbh2": rbh2_list,
            "distance": dist_list,
        })
        out_path = out_dir / f"{sample}.gb.gz"
        out_df.to_csv(out_path, sep="\t", index=False, compression="gzip")
        out_paths[sample] = out_path
    return out_paths
