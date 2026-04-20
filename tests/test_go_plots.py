"""Smoke tests for egt.go.plots.*

Each plotter is exercised end-to-end on a synthesised sweep output and
verified by checking that the PDF file is non-empty.
"""
from __future__ import annotations

import gzip
from pathlib import Path

import pandas as pd
import pytest


GO_DB = Path(__file__).resolve().parent / "testdb" / "go"


@pytest.fixture(scope="module")
def sweep_outputs(tmp_path_factory) -> Path:
    """Run the sweep once + append a synthetic significant_terms row so
    every downstream plotter has data to render, even when the fixture
    is too small to reach q ≤ 0.25 organically."""
    from egt.go import sweep as sweep_mod
    out = tmp_path_factory.mktemp("sweep_out")
    sweep_mod.run(
        supp_table=str(GO_DB / "unique_pairs_mini.tsv"),
        family_map=str(GO_DB / "family_map.tsv"),
        gene2accession=str(GO_DB / "gene2accession.tsv"),
        gene2go=str(GO_DB / "gene2go.tsv"),
        out_dir=str(out),
        write_curves=False,
        verbose=False,
    )
    # Inject synthetic significant rows so plot code has clades to draw.
    sig_path = out / "significant_terms.tsv"
    sig = pd.read_csv(sig_path, sep="\t")
    synth_rows = []
    gene_lists_rows = []
    # Seed two clades with 3 BP + 2 MF + 1 CC each. Use real clade names
    # from the enrich-plots CLADE_CHILDREN topology so the heatmap's
    # tree-layout recursion resolves a tree_x for every internal node.
    for clade in ("Bilateria", "Mollusca"):
        for axis in ("stability", "closeness", "intersection"):
            for i, (go_id, ns, name) in enumerate([
                ("GO:0000001", "BP", "process-one"),
                ("GO:0000011", "BP", "process-two"),
                ("GO:0000012", "BP", "process-three"),
                ("GO:0000002", "MF", "function-one"),
                ("GO:0000021", "MF", "function-two"),
                ("GO:0000004", "CC", "component-one"),
            ]):
                synth_rows.append({
                    "clade": clade, "axis": axis, "N_threshold": 10,
                    "sweep_namespace": "all",
                    "go_id": go_id, "go_name": name, "go_namespace": ns,
                    "foreground_hits_[k]": 5 + i,
                    "foreground_size_[n]": 20,
                    "background_hits_[K]": 10 + i,
                    "background_size_[N]": 100,
                    "ratio_in_study_[k/n]": f"{5+i}/20",
                    "ratio_in_pop_[K/N]": f"{10+i}/100",
                    "fold_enrichment": 5.0 - 0.3 * i,
                    "p_value": 1e-5 / (i + 1),
                    "correction_method": "fdr_bh",
                    "q_value": 1e-4 / (i + 1),
                    "gene_ids": ";".join(f"{1000+j}" for j in range(3)),
                    "gene_symbols": ";".join(f"GENE{j}" for j in range(3)),
                })
                gene_lists_rows.append(dict(
                    clade=clade, axis=axis, N_threshold=10, go_id=go_id,
                    k=5 + i,
                    gene_ids=",".join(f"{1000+j}" for j in range(3)),
                ))
    sig_full = pd.concat([sig, pd.DataFrame(synth_rows)], ignore_index=True)
    sig_full.to_csv(sig_path, sep="\t", index=False)
    # Also re-write term_gene_lists with the synthetic gene sets.
    gl_path = out / "term_gene_lists.tsv.gz"
    existing = pd.read_csv(gl_path, sep="\t", compression="infer")
    all_gl = pd.concat([existing, pd.DataFrame(gene_lists_rows)],
                       ignore_index=True)
    all_gl.to_csv(gl_path, sep="\t", index=False, compression="gzip")
    # Inject a gene_symbols.tsv with the synthetic GeneIDs.
    sym = pd.DataFrame([dict(gene_id=str(1000 + j), symbol=f"GENE{j}")
                        for j in range(3)])
    sym.to_csv(out / "gene_symbols.tsv", sep="\t", index=False)
    return out


# ---------- volcano ----------
def test_volcano_log_helpers():
    from egt.go.plots.volcano import log10_safe, log2_safe
    assert log10_safe(1.0) == 0.0
    assert log10_safe(0.0) == 300.0
    assert log10_safe("not-a-number") != log10_safe("not-a-number")  # NaN
    assert log2_safe(8.0) == 3.0
    assert log2_safe(-1) != log2_safe(-1)  # NaN
    assert log2_safe("x") != log2_safe("x")


def test_volcano_run_min_fold(sweep_outputs, tmp_path):
    from egt.go.plots.volcano import run as volcano_run
    out_path = tmp_path / "volcano.pdf"
    volcano_run(sweep_outputs / "significant_terms.tsv", out_path,
                min_fold=None)
    assert out_path.exists()
    # With min_fold=3 almost nothing may remain on a tiny fixture; run
    # it anyway to cover the branch.
    out2 = tmp_path / "volcano_3.pdf"
    volcano_run(sweep_outputs / "significant_terms.tsv", out2, min_fold=3.0)
    assert out2.exists()


def test_volcano_main_cli(sweep_outputs, tmp_path):
    from egt.go.plots.volcano import main
    rc = main([
        "--significant-terms", str(sweep_outputs / "significant_terms.tsv"),
        "--out", str(tmp_path / "volcano.pdf"),
    ])
    assert rc == 0


# ---------- pair_distance ----------
def test_pair_distance_pick_best_N():
    from egt.go.plots.pair_distance import pick_best_N
    s = pd.DataFrame([
        dict(clade="X", axis="stability", namespace="all",
             N_threshold=10, top_q=0.01, top_term="GO:X", top_term_fold=5.0),
        dict(clade="X", axis="stability", namespace="all",
             N_threshold=50, top_q=0.02, top_term="GO:Y", top_term_fold=1.2),
        dict(clade="X", axis="stability", namespace="all",
             N_threshold=100, top_q=0.001, top_term="GO:Z", top_term_fold=0.5),
    ])
    # Below fold=3: the fold=0.5 and fold=1.2 rows are dropped. Only the
    # first row (fold=5.0) survives.
    N, q, term, fold = pick_best_N(s, "X", "stability")
    assert N == 10 and term == "GO:X"
    # None when no rows match.
    assert pick_best_N(s, "Y", "stability") is None


def test_pair_distance_run(sweep_outputs, tmp_path):
    from egt.go.plots.pair_distance import main
    # Synthesize a summary.tsv that HAS fold>=3 cells so pick_best_N
    # returns non-None and the panel render path executes.
    synth = pd.DataFrame([
        dict(clade="CladeX", axis=axis, namespace="all",
             N_threshold=5, top_q=0.01, top_term="GO:X",
             top_term_fold=5.0)
        for axis in ("stability", "closeness", "intersection")
    ])
    synth_path = tmp_path / "summary_synth.tsv"
    synth.to_csv(synth_path, sep="\t", index=False)
    rc = main([
        "--supp-table", str(GO_DB / "unique_pairs_mini.tsv"),
        "--summary", str(synth_path),
        "--out", str(tmp_path / "pair_distance.pdf"),
    ])
    assert rc == 0
    assert (tmp_path / "pair_distance.pdf").exists()


# ---------- pair_coenrich plots ----------
def _build_pair_covs_bag(tmp_path: Path) -> Path:
    """Synthesize a pair_co_vs_bag.tsv.gz with all four category shades."""
    rows = [
        # both-significant
        dict(clade="X", axis="stability", N_threshold=10,
             go_id="GO:1", go_namespace="BP", go_name="proc-one",
             k_co=10, k_either=10, n_pairs=10,
             K_fams=5, N_fams=50, p_fam=0.1, p_co_null=0.01,
             expected_k_co=0.1, fold_pair_co=100.0,
             q_pair_co=1e-10, q_bag=1e-5, fold_bag=4.0),
        # pair-only
        dict(clade="X", axis="stability", N_threshold=10,
             go_id="GO:2", go_namespace="MF", go_name="function-two",
             k_co=5, k_either=8, n_pairs=10,
             K_fams=10, N_fams=50, p_fam=0.2, p_co_null=0.04,
             expected_k_co=0.4, fold_pair_co=12.5,
             q_pair_co=1e-4, q_bag=0.5, fold_bag=1.0),
        # bag-only
        dict(clade="X", axis="stability", N_threshold=10,
             go_id="GO:3", go_namespace="CC", go_name="comp-three",
             k_co=1, k_either=5, n_pairs=10,
             K_fams=20, N_fams=50, p_fam=0.4, p_co_null=0.16,
             expected_k_co=1.6, fold_pair_co=0.6,
             q_pair_co=0.8, q_bag=0.01, fold_bag=5.0),
        # neither
        dict(clade="X", axis="stability", N_threshold=10,
             go_id="GO:4", go_namespace="BP", go_name="noise-four",
             k_co=1, k_either=2, n_pairs=10,
             K_fams=30, N_fams=50, p_fam=0.6, p_co_null=0.36,
             expected_k_co=3.6, fold_pair_co=0.3,
             q_pair_co=0.9, q_bag=0.9, fold_bag=0.9),
        # Another clade with missing q_bag to exercise the fillna path.
        dict(clade="Y", axis="closeness", N_threshold=20,
             go_id="GO:5", go_namespace="BP", go_name="y-only",
             k_co=3, k_either=4, n_pairs=5,
             K_fams=3, N_fams=50, p_fam=0.06, p_co_null=0.0036,
             expected_k_co=0.018, fold_pair_co=166.7,
             q_pair_co=1e-6, q_bag=None, fold_bag=None),
    ]
    df = pd.DataFrame(rows)
    path = tmp_path / "pair_co_vs_bag.tsv.gz"
    df.to_csv(path, sep="\t", index=False, compression="gzip")
    return path


def test_pair_coenrich_load_and_categorize(tmp_path):
    from egt.go.plots.pair_coenrich import load
    path = _build_pair_covs_bag(tmp_path)
    df = load(path)
    cats = set(df["category"])
    assert {"both", "pair-only", "bag-only", "neither"} <= cats


def test_pair_coenrich_plot_main(tmp_path):
    from egt.go.plots.pair_coenrich import main
    path = _build_pair_covs_bag(tmp_path)
    out = tmp_path / "pair_coenrich_plots.pdf"
    rc = main(["--in", str(path), "--out", str(out)])
    assert rc == 0
    assert out.exists() and out.stat().st_size > 0


# ---------- enrich (umbrella plotting driver) ----------
def _build_obo(path: Path) -> None:
    path.write_text(
        "format-version: 1.2\n\n"
        "[Term]\nid: GO:0000001\nname: process-one\nnamespace: biological_process\n\n"
        "[Term]\nid: GO:0000002\nname: function-one\nnamespace: molecular_function\n\n"
        "[Term]\nid: GO:0000004\nname: component-one\nnamespace: cellular_component\n\n"
        "[Term]\nid: GO:0000999\nname: obsolete-term\nnamespace: biological_process\nis_obsolete: true\n\n"
        "[Typedef]\nid: part_of\nname: part of\n\n"
    )


def test_enrich_parse_obo(tmp_path):
    from egt.go.plots.enrich import parse_obo
    p = tmp_path / "mini.obo"
    _build_obo(p)
    info = parse_obo(p)
    assert info["GO:0000001"][0] == "process-one"
    assert info["GO:0000001"][1] == "BP"
    assert info["GO:0000002"][1] == "MF"
    assert info["GO:0000004"][1] == "CC"
    # Obsolete should be dropped.
    assert "GO:0000999" not in info


def test_enrich_build_clade_layout():
    from egt.go.plots.enrich import build_clade_layout, CLADE_CHILDREN
    info = build_clade_layout()
    # Every non-virtual clade has a col.
    for name in CLADE_CHILDREN:
        if not name.startswith("__"):
            assert "col" in info[name]
    # Ctenophora sits at col=0 under the paper topology.
    assert info["Ctenophora"]["col"] == 0


def test_enrich_jaccard_and_truncate():
    from egt.go.plots.enrich import jaccard, _truncate
    assert jaccard({"a", "b"}, {"b", "c"}) == 1 / 3
    assert jaccard(set(), {"a"}) == 0.0
    assert jaccard({"a"}, set()) == 0.0
    # disjoint union-less case
    assert jaccard([], []) == 0.0
    s = "x" * 60
    assert _truncate(s, 10).endswith("…")
    assert _truncate(None, 10) == ""


def _build_enrich_inputs(tmp_path: Path, sweep_outputs: Path) -> dict:
    """Return a ready-to-use argv dict for enrich.main()."""
    obo = tmp_path / "mini.obo"
    _build_obo(obo)
    return dict(
        significant_terms=str(sweep_outputs / "significant_terms.tsv"),
        obo=str(obo),
        out_dir=str(tmp_path / "plots_out"),
        term_gene_lists=str(sweep_outputs / "term_gene_lists.tsv.gz"),
        gene_symbols=str(sweep_outputs / "gene_symbols.tsv"),
    )


def test_enrich_main_smoke(sweep_outputs, tmp_path):
    from egt.go.plots.enrich import main as enrich_main
    args = _build_enrich_inputs(tmp_path, sweep_outputs)
    rc = enrich_main([
        "--significant-terms", args["significant_terms"],
        "--obo", args["obo"],
        "--out-dir", args["out_dir"],
        "--term-gene-lists", args["term_gene_lists"],
        "--gene-symbols", args["gene_symbols"],
        "--min-fold", "0.0",  # keep rows for fixture sanity
    ])
    assert rc == 0
    # An annotated table is always written.
    assert (Path(args["out_dir"]) / "significant_terms_annotated.tsv").exists()


# ---------- CLI dispatch ----------
def test_nested_cli_dispatch_lists_go_subgroup():
    # Ensure the nested dispatch recognises `egt go` as a group.
    from egt.cli import SUBCOMMANDS
    entry = SUBCOMMANDS["go"]
    assert isinstance(entry, tuple)
    registry, hlp = entry
    assert isinstance(registry, dict)
    # Every registered subcommand imports cleanly.
    import importlib
    for name, (mod_path, _hlp) in registry.items():
        importlib.import_module(mod_path)
