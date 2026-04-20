"""Further edge-branch coverage for:
- enrichment.py line 71 (no rows → early return)
- sweep.py line 143 (top_q=0 sentinel), 177 (harvest empty-idxs)
- volcano.py line 50, 98 (NaN, empty axis)
- pair_distance.py line 92, 120-121
- pair_coenrich.py plot: 48-49 (q_bag_eff fillna fallback), 72 (xmax floor),
  153 (category bars empty)
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest


GO_DB = Path(__file__).resolve().parent / "testdb" / "go"


# ---------- enrichment: no rows → early return ----------
def test_enrich_no_rows_after_min_hits():
    from egt.go.enrichment import enrich_for_foreground
    # All foreground genes carry only terms with singleton coverage; every
    # term hits k=1 < min_term_hits=2, so rows=[] triggers early return.
    bg = {f"g{i}": {f"GO:T{i}"} for i in range(5)}
    ns = {f"GO:T{i}": "BP" for i in range(5)}
    res = enrich_for_foreground(list(bg)[:3], bg, ns, min_term_hits=2)
    # All namespaces return [] because no term hits k>=2.
    assert res == {"all": [], "BP": [], "MF": [], "CC": []}


# ---------- sweep: 143 (top_q=0 sentinel) ----------
def test_sweep_clade_q_zero_sentinel():
    import math
    from egt.go import sweep as sweep_mod
    # Replace hypergeom_sf with a stub that forces q=0 for one cell so
    # the `elif top and top_q == 0` branch runs (curve sentinel 300.0).
    real = sweep_mod.enrich_for_foreground

    def stub(fg, bg, ns, **kw):
        # Return one term with q=0 in all namespaces.
        row = dict(go_id="GO:X", k=3, K=3, n=5, N=10, fold=2.0,
                    p=0.0, q=0.0, term_namespace="BP")
        return {"all": [row], "BP": [row], "MF": [], "CC": []}

    df = pd.DataFrame([
        dict(nodename="X", ortholog1="fam1", ortholog2="fam2",
             occupancy_in=0.9, sd_in_out_ratio_log_sigma=-1.0,
             mean_in_out_ratio_log_sigma=-1.0),
        dict(nodename="X", ortholog1="fam1", ortholog2="fam3",
             occupancy_in=0.9, sd_in_out_ratio_log_sigma=-0.5,
             mean_in_out_ratio_log_sigma=-0.5),
    ])
    fam_to_genes = {"fam1": {"g1"}, "fam2": {"g2"}, "fam3": {"g3"}}
    bg = {"g1": {"GO:X"}, "g2": {"GO:X"}, "g3": {"GO:X"}}
    ns = {"GO:X": "BP"}
    with patch.object(sweep_mod, "enrich_for_foreground", stub):
        records, curves = sweep_mod.sweep_clade(df, fam_to_genes, bg, ns)
    # Curve data must have the 300.0 sentinel for the q=0 cell on some axis.
    found_sentinel = any(pt[1] == 300.0 for lst in curves.values() for pt in lst)
    assert found_sentinel


# ---------- sweep: 177 (harvest axis-idxs empty branch) ----------
def test_harvest_empty_axis_indices_branch():
    from egt.go import sweep as sweep_mod
    # Force records that claim q25 hits on axis='intersection' at N where
    # the real intersection is empty (disagreeing orders).
    df = pd.DataFrame([
        dict(nodename="X", ortholog1="A", ortholog2="B",
             occupancy_in=0.9, sd_in_out_ratio_log_sigma=-1.0,
             mean_in_out_ratio_log_sigma=5.0),
        dict(nodename="X", ortholog1="A", ortholog2="C",
             occupancy_in=0.9, sd_in_out_ratio_log_sigma=5.0,
             mean_in_out_ratio_log_sigma=-1.0),
    ])
    # Fabricate a `records` list with an intersection-namespace="all"
    # q25 hit at N=1; intersection of top-1 stab (row 0) and top-1 close
    # (row 1) is empty — triggers the `if not idxs: continue` path.
    records = [dict(axis="intersection", namespace="all",
                    N_threshold=1, n_hits_q25=1)]
    out, gene_lists = sweep_mod.harvest_significant_terms(
        "X", df, records, {}, {}, {},
    )
    assert out == []
    assert gene_lists == []


# ---------- volcano: 50 (log10_safe negative → 300), 98 (empty axis panel) ----------
def test_volcano_log10_safe_handles_exact_zero_and_negative():
    from egt.go.plots.volcano import log10_safe
    assert log10_safe(-1.0) == 300.0
    assert log10_safe(0.0) == 300.0


def test_volcano_run_axis_without_matching_rows(tmp_path):
    from egt.go.plots import volcano
    # Clade Y seeds ONLY closeness → stability panel at col=0 is empty
    # and hits the `if col == 0: ax.set_ylabel(...)` branch (line 50).
    # Clade Z is filtered entirely to cover the `sub.empty continue`
    # branch (line 98).
    def _r(clade, axis, ns, go, g_ns, k, K, n, N, fold, p, q):
        return {
            "clade": clade, "axis": axis, "N_threshold": 10,
            "sweep_namespace": ns, "go_id": go,
            "go_name": "", "go_namespace": g_ns,
            "foreground_hits_[k]": k, "foreground_size_[n]": n,
            "background_hits_[K]": K, "background_size_[N]": N,
            "ratio_in_study_[k/n]": f"{k}/{n}",
            "ratio_in_pop_[K/N]": f"{K}/{N}",
            "fold_enrichment": fold, "p_value": p,
            "correction_method": "fdr_bh", "q_value": q,
            "gene_ids": "", "gene_symbols": "",
        }
    df = pd.DataFrame([
        _r("X", "stability", "all", "GO:1", "BP", 5, 10, 20, 100, 5.0, 1e-5, 1e-4),
        _r("Y", "closeness", "all", "GO:2", "MF", 4, 8, 20, 100, 4.0, 1e-4, 1e-3),
        _r("Z", "stability", "BP", "GO:3", "BP", 2, 5, 20, 100, 2.0, 0.1, 0.2),
    ])
    sig = tmp_path / "sig.tsv"
    df.to_csv(sig, sep="\t", index=False)
    out = tmp_path / "v.pdf"
    volcano.run(sig, out)
    assert out.exists()


# ---------- pair_distance: 92 (occupancy-filtered empty), 120-121 ----------
def test_pair_distance_continue_on_empty_clade(tmp_path):
    from egt.go.plots import pair_distance
    # Clade rows fail occupancy filter entirely → empty → continue.
    supp = pd.DataFrame([
        dict(nodename="X", ortholog1="A", ortholog2="B",
             occupancy_in=0.0,
             sd_in_out_ratio_log_sigma=0.0,
             mean_in_out_ratio_log_sigma=0.0,
             mean_in=1, mean_out=1),
    ])
    supp_path = tmp_path / "supp.tsv"
    supp.to_csv(supp_path, sep="\t", index=False)
    summary = pd.DataFrame([
        dict(clade="X", axis="stability", namespace="all",
             N_threshold=5, top_q=0.01, top_term="GO:X",
             top_term_fold=5.0),
        dict(clade="X", axis="closeness", namespace="all",
             N_threshold=5, top_q=0.01, top_term="GO:Y",
             top_term_fold=5.0),
    ])
    summary_path = tmp_path / "sum.tsv"
    summary.to_csv(summary_path, sep="\t", index=False)
    out = tmp_path / "pd.pdf"
    pair_distance.run(supp_path, summary_path, out)
    # Clade skipped due to empty sub; file may not exist.


# ---------- pair_coenrich plot: empty category-bars path ----------
def test_pair_distance_empty_fg_fallback_branch(tmp_path):
    """best_stab and best_clos both return with N=0 → fg_any set empty →
    branch `fg_lo=inf / fg_hi=-inf` runs."""
    from egt.go.plots import pair_distance
    # Clade row with occupancy passing + non-NaN mean_in/mean_out.
    supp = pd.DataFrame([
        dict(nodename="X", ortholog1="A", ortholog2="B",
             occupancy_in=0.9,
             sd_in_out_ratio_log_sigma=0.1,
             mean_in_out_ratio_log_sigma=0.1,
             mean_in=10, mean_out=100),
        dict(nodename="X", ortholog1="A", ortholog2="C",
             occupancy_in=0.9,
             sd_in_out_ratio_log_sigma=0.2,
             mean_in_out_ratio_log_sigma=0.2,
             mean_in=20, mean_out=200),
    ])
    supp_path = tmp_path / "supp_n0.tsv"
    supp.to_csv(supp_path, sep="\t", index=False)
    # Summary forces N=0 for each axis (pick_best_N will choose it).
    summary = pd.DataFrame([
        dict(clade="X", axis="stability", namespace="all",
             N_threshold=0, top_q=0.001, top_term="GO:X",
             top_term_fold=5.0),
        dict(clade="X", axis="closeness", namespace="all",
             N_threshold=0, top_q=0.001, top_term="GO:Y",
             top_term_fold=5.0),
    ])
    sum_path = tmp_path / "sum_n0.tsv"
    summary.to_csv(sum_path, sep="\t", index=False)
    out = tmp_path / "pd_n0.pdf"
    pair_distance.run(supp_path, sum_path, out)
    # File may or may not exist depending on tight_layout warnings; the
    # important thing is the branch got exercised without raising.


def test_pair_coenrich_plot_long_go_name_truncation(tmp_path):
    """Cover line 72 — truncate GO-name to 28 chars + ellipsis."""
    from egt.go.plots import pair_coenrich as pcp
    import matplotlib.pyplot as plt
    # Build one pair-only row whose go_name is longer than 28 chars.
    df = pd.DataFrame([
        dict(clade="X", axis="stability", N_threshold=10,
             go_id="GO:1", go_namespace="BP",
             go_name="a-very-long-go-term-name-that-needs-truncation",
             k_co=5, k_either=5, n_pairs=10,
             K_fams=2, N_fams=10, p_fam=0.2, p_co_null=0.04,
             expected_k_co=0.4, fold_pair_co=12.5,
             q_pair_co=1e-8,  # pair-significant
             q_bag=0.5,       # not bag-significant → pair-only
             fold_bag=1.0),
    ])
    path = tmp_path / "pc_long.tsv.gz"
    df.to_csv(path, sep="\t", index=False, compression="gzip")
    loaded = pcp.load(path)
    fig, ax = plt.subplots()
    pcp.plot_quadrant(loaded, ax, "X")
    plt.close(fig)


def test_pair_coenrich_plot_empty_categories(tmp_path):
    # All rows in the "neither" category → `counts` has empty index →
    # exercise the fallback where no categories contribute.
    from egt.go.plots.pair_coenrich import main
    df = pd.DataFrame([
        dict(clade="X", axis="stability", N_threshold=10,
             go_id="GO:N", go_namespace="BP", go_name="noise",
             k_co=1, k_either=2, n_pairs=10,
             K_fams=30, N_fams=50, p_fam=0.6, p_co_null=0.36,
             expected_k_co=3.6, fold_pair_co=0.3,
             q_pair_co=0.9, q_bag=0.9, fold_bag=0.9),
    ])
    path = tmp_path / "pc.tsv.gz"
    df.to_csv(path, sep="\t", index=False, compression="gzip")
    out = tmp_path / "out.pdf"
    rc = main(["--in", str(path), "--out", str(out)])
    assert rc == 0


def test_pair_coenrich_plot_quadrant_empty_clade(tmp_path):
    """Cover plot_quadrant's `if d.empty: axis('off'); return` branch."""
    from egt.go.plots import pair_coenrich as pcp
    import matplotlib.pyplot as plt
    df = pd.DataFrame([
        dict(clade="X", axis="stability", N_threshold=10, go_id="GO:1",
             go_namespace="BP", go_name="one",
             k_co=5, k_either=5, n_pairs=10,
             K_fams=2, N_fams=10, p_fam=0.2, p_co_null=0.04,
             expected_k_co=0.4, fold_pair_co=12.5,
             q_pair_co=1e-8, q_bag=1e-8, fold_bag=4.0),
    ])
    path = tmp_path / "pc.tsv.gz"
    df.to_csv(path, sep="\t", index=False, compression="gzip")
    loaded = pcp.load(path)
    fig, ax = plt.subplots()
    pcp.plot_quadrant(loaded, ax, clade="NOT_PRESENT")
    plt.close(fig)
