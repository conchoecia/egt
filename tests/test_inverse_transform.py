"""Tests for ``egt inverse-transform`` (egt.inverse_transform).

Builds a tiny synthetic COO distance matrix + feature-index file, fits a dense
Euclidean UMAP reducer, then inverse-transforms embedding coordinates and checks
the structure of the ranked output (single point and two-point comparison).

UMAP embeddings are not asserted by exact value — only the shapes, column
schema, ordering invariants, and round-trip wiring are checked.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from scipy.sparse import coo_matrix, save_npz


FAMS = [f"fam_{i:03d}" for i in range(6)]  # 6 families -> C(6,2) = 15 pairs
COMBO_TO_IX = {
    (FAMS[i], FAMS[j]): k
    for k, (i, j) in enumerate(
        [(i, j) for i in range(len(FAMS)) for j in range(i + 1, len(FAMS))]
    )
}
N_PAIRS = len(COMBO_TO_IX)  # 15


def _build_fixture(tmp_path: Path, n_samples: int = 15, seed: int = 0):
    """Write a small COO .npz + algcomboix file. Returns (coo_path, ix_path)."""
    rng = np.random.default_rng(seed)
    rows, cols, vals = [], [], []
    for s in range(n_samples):
        n_take = int(N_PAIRS * 0.7)
        pairs = rng.choice(N_PAIRS, size=n_take, replace=False)
        for p in pairs:
            rows.append(s)
            cols.append(int(p))
            vals.append(1000.0 * (s + 1) + int(p))
    mat = coo_matrix(
        (vals, (rows, cols)), shape=(n_samples, N_PAIRS), dtype=float
    )
    coo_path = tmp_path / "small.npz"
    save_npz(str(coo_path), mat)

    ix_path = tmp_path / "algcomboix.txt"
    with open(ix_path, "w") as fh:
        for (r1, r2), ix in COMBO_TO_IX.items():
            fh.write(f"('{r1}', '{r2}')\t{ix}\n")
    return coo_path, ix_path


def _fit(tmp_path: Path):
    from egt.inverse_transform import main

    coo_path, ix_path = _build_fixture(tmp_path)
    prefix = tmp_path / "run"
    rc = main(
        [
            "fit",
            "--coo", str(coo_path),
            "--algcomboix", str(ix_path),
            "--n-neighbors", "5",
            "--min-dist", "0.1",
            "--out-prefix", str(prefix),
        ]
    )
    assert rc == 0
    return prefix


# --------------------------------------------------------------------------- #
# fit
# --------------------------------------------------------------------------- #
def test_fit_writes_all_artifacts(tmp_path):
    prefix = _fit(tmp_path)
    assert Path(str(prefix) + ".reducer").exists()
    assert Path(str(prefix) + ".features").exists()
    assert Path(str(prefix) + ".embedding.tsv").exists()


def test_fit_feature_list_is_ordered_and_complete(tmp_path):
    prefix = _fit(tmp_path)
    feat = pd.read_csv(str(prefix) + ".features", sep="\t")
    assert list(feat.columns) == ["feature_index", "rbh1", "rbh2"]
    assert len(feat) == N_PAIRS
    # feature_index is exactly 0..N_PAIRS-1 in order.
    assert feat["feature_index"].tolist() == list(range(N_PAIRS))
    # The pair at a known index matches COMBO_TO_IX.
    ix_to_pair = {v: k for k, v in COMBO_TO_IX.items()}
    row7 = feat[feat["feature_index"] == 7].iloc[0]
    assert (row7["rbh1"], row7["rbh2"]) == ix_to_pair[7]


def test_fit_embedding_has_one_row_per_sample(tmp_path):
    prefix = _fit(tmp_path)
    emb = pd.read_csv(str(prefix) + ".embedding.tsv", sep="\t")
    assert list(emb.columns) == ["sample", "UMAP1", "UMAP2"]
    assert len(emb) == 15


# --------------------------------------------------------------------------- #
# query — single point
# --------------------------------------------------------------------------- #
def test_query_single_point(tmp_path):
    from egt.inverse_transform import main

    prefix = _fit(tmp_path)
    emb = pd.read_csv(str(prefix) + ".embedding.tsv", sep="\t")
    x, y = float(emb["UMAP1"].iloc[0]), float(emb["UMAP2"].iloc[0])
    out = tmp_path / "single.tsv"
    rc = main(
        [
            "query",
            "--reducer", str(prefix) + ".reducer",
            "--embx", str(x),
            "--emby", str(y),
            "--out", str(out),
        ]
    )
    assert rc == 0
    df = pd.read_csv(out, sep="\t")
    assert len(df) == N_PAIRS
    assert "Value" in df.columns
    assert {"rbh1", "rbh2"}.issubset(df.columns)
    # Sorted descending by Value.
    assert df["Value"].is_monotonic_decreasing
    assert np.isfinite(df["Value"]).all()


# --------------------------------------------------------------------------- #
# query — two points
# --------------------------------------------------------------------------- #
def test_query_two_point_difference(tmp_path):
    from egt.inverse_transform import main

    prefix = _fit(tmp_path)
    emb = pd.read_csv(str(prefix) + ".embedding.tsv", sep="\t")
    x1, y1 = float(emb["UMAP1"].iloc[0]), float(emb["UMAP2"].iloc[0])
    x2, y2 = float(emb["UMAP1"].iloc[-1]), float(emb["UMAP2"].iloc[-1])
    out = tmp_path / "diff.tsv"
    rc = main(
        [
            "query",
            "--reducer", str(prefix) + ".reducer",
            "--embx", str(x1), "--emby", str(y1),
            "--embx2", str(x2), "--emby2", str(y2),
            "--out", str(out),
        ]
    )
    assert rc == 0
    df = pd.read_csv(out, sep="\t")
    assert len(df) == N_PAIRS
    assert {"Val_P1", "Val_P2", "Difference", "Abs_Difference"}.issubset(df.columns)
    # Difference == Val_P1 - Val_P2, and sorted by Abs_Difference descending.
    # UMAP reconstructs in float32, so allow float32-scale round-trip noise.
    np.testing.assert_allclose(
        df["Difference"].to_numpy(),
        (df["Val_P1"] - df["Val_P2"]).to_numpy(),
        rtol=1e-4,
        atol=1e-2,
    )
    assert df["Abs_Difference"].is_monotonic_decreasing


def test_query_requires_both_second_coords(tmp_path):
    from egt.inverse_transform import main

    prefix = _fit(tmp_path)
    # Only embx2, no emby2 -> error exit 2.
    rc = main(
        [
            "query",
            "--reducer", str(prefix) + ".reducer",
            "--embx", "0", "--emby", "0",
            "--embx2", "1",
            "--out", str(tmp_path / "x.tsv"),
        ]
    )
    assert rc == 2
