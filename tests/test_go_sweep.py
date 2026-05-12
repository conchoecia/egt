"""Tests for egt.go.sweep — the N-sweep driver and CLI entry point."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from egt.go import sweep as sweep_mod
from egt.go.sweep import (
    _foreground_indices,
    _n_grid,
    _prepare_clade,
    harvest_significant_terms,
    main as sweep_main,
    run as sweep_run,
    sweep_clade,
)


GO_DB = Path(__file__).resolve().parent / "testdb" / "go"


# ---------- small helpers ----------
def test_n_grid_small():
    # n_rows below N_MIN collapses to a single-element grid.
    assert _n_grid(5) == [5]


def test_n_grid_log_spaced():
    g = _n_grid(200)
    assert g[0] == 10
    assert g[-1] == 200
    # monotonic, unique
    assert g == sorted(set(g))


def test_foreground_indices_axes():
    import numpy as np
    stab = np.array([0, 1, 2, 3, 4])
    close = np.array([4, 3, 2, 1, 0])
    assert _foreground_indices("stability", 2, stab, close) == {0, 1}
    assert _foreground_indices("closeness", 2, stab, close) == {4, 3}
    # intersection of first-3 on each side = {2}
    assert _foreground_indices("intersection", 3, stab, close) == {2}


def test_prepare_clade_filters_and_resets_index():
    df = pd.DataFrame({
        "occupancy_in": [0.9, 0.3, 0.8, None],
        "sd_in_out_ratio_log_sigma": [0.1, 0.2, None, 0.4],
        "mean_in_out_ratio_log_sigma": [0.1, 0.2, 0.3, 0.4],
        "ortholog1": ["A"] * 4,
        "ortholog2": ["B"] * 4,
    })
    out = _prepare_clade(df)
    # Rows 1 (low occupancy), 2 (NaN sd), 3 (NaN occupancy) are dropped.
    assert len(out) == 1
    assert list(out.index) == [0]


# ---------- sweep_clade / harvest_significant_terms ----------
@pytest.fixture(scope="module")
def loaded_refs():
    from egt.go.io import (
        build_family_gene_annotations,
        load_unique_pairs,
        parse_family_map,
        parse_gene2accession,
        parse_gene2go,
    )
    df = load_unique_pairs(GO_DB / "unique_pairs_mini.tsv")
    p2g, _ = parse_gene2accession(GO_DB / "gene2accession.tsv")
    fam_to_genes, _ = parse_family_map(GO_DB / "family_map.tsv", p2g)
    g2t, term_ns = parse_gene2go(GO_DB / "gene2go.tsv")
    _, bg_terms = build_family_gene_annotations(fam_to_genes, g2t)
    return df, fam_to_genes, bg_terms, term_ns


def test_sweep_clade_shape(loaded_refs):
    df, fam, bg, ns = loaded_refs
    sub = df[df["nodename"] == "CladeX"]
    records, curves = sweep_clade(sub, fam, bg, ns)
    # Non-empty sub with enough rows above occupancy produces records.
    assert records
    # Each record carries the fixed schema keys.
    required = {"axis", "N_threshold", "pairs_used", "namespace",
                "foreground_size_[n]", "background_size_[N]",
                "n_families", "foreground_raw_geneids",
                "n_terms_tested", "n_hits_q05", "n_hits_q25",
                "top_term", "top_term_fold_enrichment",
                "top_term_hits_[k]", "top_term_bg_hits_[K]",
                "top_q_value"}
    for r in records:
        assert required <= set(r.keys())
    # Axes covered.
    assert {"stability", "closeness", "intersection"} == {
        r["axis"] for r in records
    }
    # Namespaces covered.
    assert {"all", "BP", "MF", "CC"} >= {r["namespace"] for r in records}
    # curve_data keys tagged by (axis, ns).
    if curves:
        for k in curves:
            assert isinstance(k, tuple) and len(k) == 2


def test_sweep_clade_empty_after_filter():
    # All rows fail occupancy → empty returns.
    df = pd.DataFrame({
        "occupancy_in": [0.1, 0.2],
        "sd_in_out_ratio_log_sigma": [1.0, 2.0],
        "mean_in_out_ratio_log_sigma": [1.0, 2.0],
        "ortholog1": ["A", "B"],
        "ortholog2": ["B", "C"],
    })
    records, curves = sweep_clade(df, {}, {}, {})
    assert records == []
    assert curves == {}


def test_harvest_significant_terms_without_records(loaded_refs):
    df, fam, bg, ns = loaded_refs
    sub = df[df["nodename"] == "CladeX"]
    # No records → nothing to harvest.
    out, gl = harvest_significant_terms("CladeX", sub, [], fam, bg, ns)
    assert out == []
    assert gl == []


def test_harvest_significant_terms_empty_clade_returns_empty(loaded_refs):
    df, fam, bg, ns = loaded_refs
    empty = df.iloc[0:0]
    out, gl = harvest_significant_terms("X", empty, [], fam, bg, ns)
    assert out == [] and gl == []


# ---------- run() end-to-end ----------
def test_run_end_to_end_writes_expected_files(tmp_path, loaded_refs):
    summary = sweep_run(
        supp_table=str(GO_DB / "unique_pairs_mini.tsv"),
        family_map=str(GO_DB / "family_map.tsv"),
        gene2accession=str(GO_DB / "gene2accession.tsv"),
        gene2go=str(GO_DB / "gene2go.tsv"),
        out_dir=str(tmp_path),
        obo=str(GO_DB / "mini.obo"),
        write_curves=True,
        verbose=False,
    )
    assert summary == tmp_path / "summary.tsv"
    # Required artifacts.
    assert (tmp_path / "summary.tsv").exists()
    assert (tmp_path / "significant_terms.tsv").exists()
    assert (tmp_path / "term_gene_lists.tsv.gz").exists()
    assert (tmp_path / "gene_symbols.tsv").exists()
    # Per-clade dir non-empty.
    per = list((tmp_path / "per_clade").glob("*.tsv"))
    assert per, "expected at least one per-clade tsv"
    # Summary shape.
    sdf = pd.read_csv(tmp_path / "summary.tsv", sep="\t")
    for c in ("axis", "N_threshold", "namespace", "top_q_value"):
        assert c in sdf.columns
    # significant_terms follows the publication-standard schema,
    # with k/n/K/N/fold_enrichment/q_value as one contiguous block.
    sig = pd.read_csv(tmp_path / "significant_terms.tsv", sep="\t")
    expected = ["clade", "axis", "N_threshold", "sweep_namespace",
                "go_id", "go_name", "go_namespace",
                "foreground_hits_[k]", "foreground_size_[n]",
                "background_hits_[K]", "background_size_[N]",
                "fold_enrichment", "q_value",
                "ratio_in_study_[k/n]", "ratio_in_pop_[K/N]",
                "p_value", "correction_method",
                "gene_ids", "gene_symbols", "bcns_families"]
    assert list(sig.columns) == expected


def test_run_no_curves_flag(tmp_path):
    sweep_run(
        supp_table=str(GO_DB / "unique_pairs_mini.tsv"),
        family_map=str(GO_DB / "family_map.tsv"),
        gene2accession=str(GO_DB / "gene2accession.tsv"),
        gene2go=str(GO_DB / "gene2go.tsv"),
        out_dir=str(tmp_path),
        write_curves=False,
        verbose=False,
    )
    assert not (tmp_path / "curves.pdf").exists()


def test_run_verbose_prints(tmp_path, capsys):
    sweep_run(
        supp_table=str(GO_DB / "unique_pairs_mini.tsv"),
        family_map=str(GO_DB / "family_map.tsv"),
        gene2accession=str(GO_DB / "gene2accession.tsv"),
        gene2go=str(GO_DB / "gene2go.tsv"),
        out_dir=str(tmp_path),
        write_curves=False,
        verbose=True,
    )
    out = capsys.readouterr().out
    assert "[load]" in out
    assert "[clade]" in out


# ---------- CLI ----------
def test_main_argv_round_trip(tmp_path):
    rc = sweep_main([
        "--supp-table", str(GO_DB / "unique_pairs_mini.tsv"),
        "--family-map", str(GO_DB / "family_map.tsv"),
        "--gene2accession", str(GO_DB / "gene2accession.tsv"),
        "--gene2go", str(GO_DB / "gene2go.tsv"),
        "--out-dir", str(tmp_path),
        "--no-curves",
    ])
    assert rc == 0
    assert (tmp_path / "summary.tsv").exists()


# ---------- curves.pdf rendering ----------
def test_write_curves_pdf_renders(tmp_path):
    # Synthesize minimal curve_data to exercise the PDF path including
    # the q=0 sentinel branch and the row/col legend gating.
    curves = {
        "CladeA": {
            ("stability", "all"): [(10, 3.0), (100, 5.0)],
            ("closeness", "BP"): [(10, 1.0)],
            ("intersection", "MF"): [(50, 10.0)],
        },
        "CladeB": {
            ("stability", "CC"): [(10, 2.0), (50, 4.0)],
        },
    }
    out_path = tmp_path / "curves.pdf"
    sweep_mod._write_curves_pdf(curves, out_path)
    assert out_path.exists() and out_path.stat().st_size > 0


def test_write_curves_pdf_noop_on_empty(tmp_path):
    out_path = tmp_path / "curves.pdf"
    sweep_mod._write_curves_pdf({}, out_path)
    assert not out_path.exists()
