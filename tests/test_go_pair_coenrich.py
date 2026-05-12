"""Tests for egt.go.pair_coenrich."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from egt.go import pair_coenrich as pc


GO_DB = Path(__file__).resolve().parent / "testdb" / "go"


def test_family_term_table_unions_and_drops_empties():
    fam = {"A": {"g1", "g2"}, "B": {"g3"}, "C": {"g_noterms"}}
    g2t = {"g1": {"T1"}, "g2": {"T2"}, "g3": {"T1", "T3"}}
    out = pc.family_term_table(fam, g2t)
    assert out["A"] == {"T1", "T2"}
    assert out["B"] == {"T1", "T3"}
    # C has genes but none has terms → dropped.
    assert "C" not in out


def test_best_cell_per_clade(tmp_path):
    sdf = pd.DataFrame([
        # clade X — two "all" rows; pick the smaller top_q
        dict(clade="X", axis="stability", N_threshold=10,
             namespace="all", top_q=0.01),
        dict(clade="X", axis="closeness", N_threshold=50,
             namespace="all", top_q=0.001),
        # NaN top_q should be dropped
        dict(clade="X", axis="intersection", N_threshold=5,
             namespace="all", top_q=float("nan")),
        # namespace!=all should be ignored
        dict(clade="X", axis="stability", N_threshold=10,
             namespace="BP", top_q=1e-20),
        # clade Y — single candidate
        dict(clade="Y", axis="intersection", N_threshold=200,
             namespace="all", top_q=0.05),
    ])
    p = tmp_path / "summary.tsv"
    sdf.to_csv(p, sep="\t", index=False)
    best = pc.best_cell_per_clade(p)
    assert best["X"] == dict(axis="closeness", N=50)
    assert best["Y"] == dict(axis="intersection", N=200)


def test_select_pair_indices():
    df = pd.DataFrame({
        "sd_in_out_ratio_log_sigma":  [0.1, 0.2, 0.3, 0.4, 0.5],
        "mean_in_out_ratio_log_sigma":[0.5, 0.4, 0.3, 0.2, 0.1],
    })
    stab = pc._select_pair_indices(df, "stability", 3)
    assert stab == [0, 1, 2]
    close = pc._select_pair_indices(df, "closeness", 3)
    assert set(close) == {2, 3, 4}
    inter = pc._select_pair_indices(df, "intersection", 3)
    assert set(inter) == {2}


def test_pair_coenrich_for_clade_detects_planted_signal():
    # Synthesize a clade where pairs *intentionally* partner two
    # carriers of "GO:PLANTED" more than chance. Build a larger
    # family-term universe so p_fam^2 is small enough to make k_co
    # statistically interesting.
    rng_fams = [f"fam{i}" for i in range(50)]
    fam_to_terms: dict[str, set[str]] = {}
    for i, f in enumerate(rng_fams):
        terms = {"GO:UB1"}  # ubiquitous
        if i < 4:            # rare planted carriers
            terms.add("GO:PLANTED")
        fam_to_terms[f] = terms
    # Build 10 pairs, all of which pair two planted carriers.
    import itertools
    planted = [f for i, f in enumerate(rng_fams) if i < 4]
    # Pair combinations of the 4 planted families: C(4,2)=6, then pad
    # with another 4 pairs that reuse them to reach 10.
    planted_pairs = list(itertools.combinations(planted, 2))
    pairs = planted_pairs + planted_pairs[:4]
    rows = []
    for i, (f1, f2) in enumerate(pairs):
        rows.append(dict(
            nodename="X",
            ortholog1=f1, ortholog2=f2,
            occupancy_in=0.9,
            sd_in_out_ratio_log_sigma=0.1 * i,
            mean_in_out_ratio_log_sigma=0.1 * i,
        ))
    # pad with 5 "ubiquitous" pairs at higher sd so the top-N stability
    # picks only the planted ones.
    for j in range(5):
        rows.append(dict(
            nodename="X",
            ortholog1=rng_fams[10 + j], ortholog2=rng_fams[20 + j],
            occupancy_in=0.9,
            sd_in_out_ratio_log_sigma=10.0 + j,
            mean_in_out_ratio_log_sigma=10.0 + j,
        ))
    df = pd.DataFrame(rows)
    ns = {"GO:PLANTED": "BP", "GO:UB1": "BP"}
    out = pc.pair_coenrich_for_clade(df, axis="stability", N=10,
                                      fam_to_terms=fam_to_terms,
                                      term_namespace=ns)
    planted_row = next(r for r in out if r["go_id"] == "GO:PLANTED")
    # Every one of the 10 pairs co-carried the planted term.
    assert planted_row["pair_cohits_[k]"] == 10
    assert planted_row["pair_count_[n]"] == 10
    # p_fam = 4/50 = 0.08; p_co = 0.0064; expected ≈ 0.064 → k=10
    # is absurdly over-represented, p should be essentially 0.
    assert planted_row["p_value"] < 1e-8
    assert planted_row["q_value"] < 1e-7
    assert planted_row["fold_enrichment"] > 100


def test_pair_coenrich_for_clade_filters_and_noop_paths():
    # All rows fail occupancy → empty.
    df = pd.DataFrame([
        dict(nodename="X", ortholog1="A", ortholog2="B", occupancy_in=0.1,
             sd_in_out_ratio_log_sigma=0.1,
             mean_in_out_ratio_log_sigma=0.1),
    ])
    assert pc.pair_coenrich_for_clade(df, "stability", 5, {}, {}) == []

    # occupancy ok but N=0 path.
    df2 = pd.DataFrame([
        dict(nodename="X", ortholog1="A", ortholog2="B", occupancy_in=0.9,
             sd_in_out_ratio_log_sigma=0.1,
             mean_in_out_ratio_log_sigma=0.1),
    ])
    # N=0 → intersection path yields empty, caught by the idxs-empty branch.
    assert pc.pair_coenrich_for_clade(
        df2, "intersection", N=0, fam_to_terms={}, term_namespace={}
    ) == []

    # Pairs exist but neither family is in fam_to_terms → n_pairs=0.
    assert pc.pair_coenrich_for_clade(
        df2, "stability", N=5, fam_to_terms={}, term_namespace={}
    ) == []


def test_pair_coenrich_min_co_hits_floor():
    # A term observed in 1 pair should not be tested (MIN_CO_HITS=2).
    fam_to_terms = {"A": {"T"}, "B": {"T"}, "C": set()}
    df = pd.DataFrame([
        dict(nodename="X", ortholog1="A", ortholog2="B", occupancy_in=0.9,
             sd_in_out_ratio_log_sigma=0.1,
             mean_in_out_ratio_log_sigma=0.1),
        dict(nodename="X", ortholog1="A", ortholog2="C", occupancy_in=0.9,
             sd_in_out_ratio_log_sigma=0.2,
             mean_in_out_ratio_log_sigma=0.2),
    ])
    out = pc.pair_coenrich_for_clade(df, "stability", 10, fam_to_terms, {})
    # Only 1 co-occurrence (pair A-B), below MIN_CO_HITS=2 → no rows.
    assert out == []


# ---------- run() + CLI ----------
def _prepare_inputs(tmp_path: Path) -> dict:
    from egt.go import sweep as sweep_mod
    # Run sweep first so we get summary.tsv + significant_terms.tsv for
    # pair_coenrich to read.
    sweep_mod.run(
        supp_table=str(GO_DB / "unique_pairs_mini.tsv"),
        family_map=str(GO_DB / "family_map.tsv"),
        gene2accession=str(GO_DB / "gene2accession.tsv"),
        gene2go=str(GO_DB / "gene2go.tsv"),
        out_dir=str(tmp_path),
        write_curves=False,
        verbose=False,
    )
    return dict(
        supp_table=str(GO_DB / "unique_pairs_mini.tsv"),
        family_map=str(GO_DB / "family_map.tsv"),
        gene2accession=str(GO_DB / "gene2accession.tsv"),
        gene2go=str(GO_DB / "gene2go.tsv"),
        summary=str(tmp_path / "summary.tsv"),
    )


def test_run_writes_per_clade_dir(tmp_path):
    inp = _prepare_inputs(tmp_path)
    out_dir = tmp_path / "pair"
    pc.run(**inp, out_dir=str(out_dir), obo=str(GO_DB / "mini.obo"),
           verbose=False)
    # per_clade dir is always created by run(), even if empty.
    assert (out_dir / "per_clade").exists()


def test_main_argv(tmp_path):
    inp = _prepare_inputs(tmp_path)
    out_dir = tmp_path / "pair2"
    rc = pc.main([
        "--supp-table", inp["supp_table"],
        "--family-map", inp["family_map"],
        "--gene2accession", inp["gene2accession"],
        "--gene2go", inp["gene2go"],
        "--summary", inp["summary"],
        "--out-dir", str(out_dir),
        "--obo", str(GO_DB / "mini.obo"),
    ])
    assert rc == 0
    assert (out_dir / "per_clade").exists()


def test_compare_to_bag_merges_both_tables(tmp_path):
    # Build a minimal summ_co and significant_terms pair to drive the
    # comparison path. summ_co uses the canonical column names emitted
    # by pair_coenrich_for_clade; sig uses the canonical significant
    # terms schema (short aliases come from load_significant_terms).
    summ_co = pd.DataFrame([
        dict(clade="X", axis="stability", N_threshold=10,
             go_id="GO:A", q_value=0.01, fold_enrichment=5.0),
        dict(clade="X", axis="stability", N_threshold=10,
             go_id="GO:B", q_value=0.5, fold_enrichment=1.1),
    ])
    sig = pd.DataFrame([{
        "clade": "X", "axis": "stability", "N_threshold": 10,
        "sweep_namespace": "all", "go_id": "GO:A",
        "q_value": 0.001, "fold_enrichment": 6.0,
    }])
    sig_path = tmp_path / "significant_terms.tsv"
    sig.to_csv(sig_path, sep="\t", index=False)
    out = pc._compare_to_bag(summ_co, sig_path, tmp_path)
    assert out.exists()
    merged = pd.read_csv(out, sep="\t", compression="infer")
    assert "q_pair_co" in merged.columns
    assert "q_bag" in merged.columns
    # GO:A merges, GO:B has NaN q_bag because it isn't in sig.
    a_row = merged[merged["go_id"] == "GO:A"].iloc[0]
    assert a_row["q_bag"] == 0.001
    b_row = merged[merged["go_id"] == "GO:B"].iloc[0]
    assert pd.isna(b_row["q_bag"])
