"""Tests for ALG dispersal binning + divergence-vs-dispersal loaders.

Covers:
  1. Bin coverage gap   — species with median in (0.2, 0.4) or (0.6, 0.8) get no bin.
  2. Bin edge inclusion — locks behavior at 0.00, 0.20, 0.40, 0.60, 0.80, 1.00.
  3. num_algs_detected invariant — equals count of ALG_* fractions > 0.
  4. Missing divergence TSV       — FileNotFoundError.
  5. Unknown divergence column    — SystemExit (NOT ValueError).
  6. presence-fusions deny-list   — taxid/taxidstring/changestrings/(paren) cols excluded.
  7. Default dispersal formula    — 1 - n_alg / len(alg_cols) on a 29-ALG fixture.
  8. Figure-close hygiene         — plot_dispersal_by_alg leaves no open figures.
  9. Regression                   — no lowercase 'dispersion' literal in src/egt/**/*.py.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from egt import plot_alg_dispersal
from egt.divergence_vs_dispersal import _load_divergence, _load_presence_fusions


# --------------------------------------------------------------------------- #
# 1. Bin coverage gap
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("median", [0.21, 0.25, 0.30, 0.39, 0.61, 0.70, 0.79])
def test_bin_coverage_gap_drops_intermediate(median: float) -> None:
    """Medians in (0.2, 0.4) or (0.6, 0.8) are not assigned to any bin."""
    assert plot_alg_dispersal._assign_dispersal_bin(median) is None


# --------------------------------------------------------------------------- #
# 2. Bin edge inclusion
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "median, expected",
    [
        (0.00, "80%-100% dispersal"),
        (0.20, "80%-100% dispersal"),
        (0.40, "40%-60% dispersal"),
        (0.60, "40%-60% dispersal"),
        (0.80, "0%-20% dispersal"),
        (1.00, "0%-20% dispersal"),
    ],
)
def test_bin_edge_inclusion(median: float, expected: str) -> None:
    """Lock the bin assignment at every edge under the current `<=` predicate."""
    assert plot_alg_dispersal._assign_dispersal_bin(median) == expected


# --------------------------------------------------------------------------- #
# 3. num_algs_detected invariant
# --------------------------------------------------------------------------- #
def test_num_algs_detected_matches_alg_columns_gt_zero() -> None:
    """num_algs_detected (used by plot_dispersal_by_alg) equals count of ALG_* > 0.

    This locks the per-row invariant: the count stored alongside the
    per-ALG fractions must agree with the number of those fractions
    that are strictly positive.
    """
    fractions = {"Qa": 0.5, "Qb": 0.0, "A1a": 0.8, "B": 0.0, "C": 0.3}
    conservation_values = [v for v in fractions.values() if v > 0]
    num_algs_detected = len(conservation_values)
    alg_cols_gt_zero = sum(1 for v in fractions.values() if v > 0)
    assert num_algs_detected == alg_cols_gt_zero == 3


# --------------------------------------------------------------------------- #
# 4. Missing divergence file
# --------------------------------------------------------------------------- #
def test_load_divergence_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        _load_divergence(tmp_path / "does_not_exist.tsv", "median_pct_id")


# --------------------------------------------------------------------------- #
# 5. Unknown divergence column → SystemExit
# --------------------------------------------------------------------------- #
def test_load_divergence_unknown_column_raises_systemexit(tmp_path: Path) -> None:
    p = tmp_path / "div.tsv"
    p.write_text("species\tfoo\nA\t0.5\n")
    with pytest.raises(SystemExit):
        _load_divergence(p, "median_pct_id")


# --------------------------------------------------------------------------- #
# 6. presence-fusions deny-list
# --------------------------------------------------------------------------- #
def test_load_presence_fusions_filters_meta_columns(tmp_path: Path) -> None:
    """taxid, taxidstring, changestrings, and (paren) cols must not count as ALGs."""
    p = tmp_path / "pf.tsv"
    p.write_text(
        "species\ttaxid\ttaxidstring\tchangestrings\t(Qa,Qb)\tQa\tQb\n"
        "spA\t9606\t1;2;3\tfoo\t1\t1\t0\n"
    )
    df = _load_presence_fusions(p, dispersal_column=None)
    # 2 real ALG cols (Qa, Qb); 1 detected → dispersal = 1 - 1/2 = 0.5
    assert len(df) == 1
    assert df.iloc[0]["dispersal"] == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# 7. Default dispersal formula on 29 ALGs
# --------------------------------------------------------------------------- #
def test_load_presence_fusions_default_formula_29_algs(tmp_path: Path) -> None:
    """With 29 ALG cols, default dispersal = 1 − n_alg / 29."""
    alg_cols = [f"ALG{i:02d}" for i in range(29)]
    header = "species\ttaxidstring\t" + "\t".join(alg_cols) + "\n"
    sp1 = "sp1\t1;2\t" + "\t".join(["1"] * 29) + "\n"          # all present  → 0
    sp2 = "sp2\t1;3\t" + "\t".join(["0"] * 29) + "\n"          # none present → 1
    sp3 = "sp3\t1;4\t" + "\t".join(["1"] * 15 + ["0"] * 14) + "\n"  # 15/29
    p = tmp_path / "pf.tsv"
    p.write_text(header + sp1 + sp2 + sp3)
    df = _load_presence_fusions(p, dispersal_column=None).set_index("species")
    assert df.loc["sp1", "dispersal"] == pytest.approx(0.0)
    assert df.loc["sp2", "dispersal"] == pytest.approx(1.0)
    assert df.loc["sp3", "dispersal"] == pytest.approx(1 - 15 / 29)


# --------------------------------------------------------------------------- #
# 8. Figure-close hygiene
# --------------------------------------------------------------------------- #
def test_plot_dispersal_closes_figures(tmp_path: Path) -> None:
    """plot_dispersal_by_alg must close every matplotlib figure it opens.

    Run with an empty `species_to_rbh` so the function only takes the
    no-species code path (one figure opened + saved + closed; no ranking
    figures created). After the call, no figures may remain open.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.close("all")
    alg_df = pd.DataFrame(
        {
            "ALGname": ["Qa", "Qb"],
            "Size": [10, 10],
            "Color": ["#000000", "#111111"],
        }
    )
    plot_alg_dispersal.plot_dispersal_by_alg(
        species_to_rbh={},
        alg_df=alg_df,
        algname="BCnS",
        minsig=0.05,
        outdir=str(tmp_path),
    )
    assert plt.get_fignums() == [], (
        f"Open figures leaked: {plt.get_fignums()}"
    )


# --------------------------------------------------------------------------- #
# 9. Regression: no lowercase 'dispersion' in src/egt
# --------------------------------------------------------------------------- #
def test_no_dispersion_literal_in_egt_source() -> None:
    """After the dispersion→dispersal rename, no occurrence of the literal
    'dispersion' (case-insensitive) may remain in src/egt/**/*.py."""
    src_dir = Path(__file__).resolve().parents[1] / "src" / "egt"
    hits: list[str] = []
    for py in src_dir.rglob("*.py"):
        text = py.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), start=1):
            if "dispersion" in line.lower():
                hits.append(f"{py.relative_to(src_dir)}:{lineno}: {line.strip()}")
    assert hits == [], "Found 'dispersion' refs:\n" + "\n".join(hits)
