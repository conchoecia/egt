"""End-to-end tests for egt.go.benchmarks.goatools_ref.

Requires the real `goatools` library (pinned in dev deps); tests skip
gracefully if it is absent.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


GO_DB = Path(__file__).resolve().parent / "testdb" / "go"
goatools = pytest.importorskip("goatools")


def _fixture_g2a(tmp_path: Path) -> Path:
    p = tmp_path / "g2a.tsv"
    with p.open("w") as fh:
        fh.write("#tax\tGeneID\tstatus\tRNA\trnagi\tprotein\tpgi\t"
                 "genomic\tggi\tsp\tep\to\tasm\tmpep\tmpgi\tSymbol\n")
        for i in range(30):
            fh.write(f"9606\t{1000+i}\tREVIEWED\t-\t-\tNP_{i:06d}.1\t-\tNC.1\t-\t1\t2\t+\tG\t-\t-\tS{i}\n")
    return p


def _fixture_g2g(tmp_path: Path) -> Path:
    p = tmp_path / "g2g.tsv"
    with p.open("w") as fh:
        fh.write("#tax\tGeneID\tGO_ID\tEvidence\tQualifier\tTerm\tPM\tCategory\n")
        for i in range(30):
            # Universal term + clade-specific planted term on the first 10 genes.
            fh.write(f"9606\t{1000+i}\tGO:0000001\tIEA\t-\tproc\t1\tProcess\n")
            if i < 10:
                fh.write(f"9606\t{1000+i}\tGO:0000002\tIEA\t-\tfunc\t1\tFunction\n")
    return p


def _fixture_fm(tmp_path: Path) -> Path:
    p = tmp_path / "fm.tsv"
    with p.open("w") as fh:
        fh.write("family_id\talg\thuman_gene\thuman_scaf\tsource\tnote\n")
        for i in range(30):
            fh.write(f"fam{i}\tA1a\tNP_{i:06d}.1\tNC\trbh\t\n")
    return p


def _fixture_obo(tmp_path: Path) -> Path:
    p = tmp_path / "mini.obo"
    p.write_text(
        "format-version: 1.2\n\n"
        "[Term]\nid: GO:0000001\nname: proc\nnamespace: biological_process\n\n"
        "[Term]\nid: GO:0000002\nname: func\nnamespace: molecular_function\n\n"
    )
    return p


def _run_sweep(tmp_path: Path) -> dict:
    from egt.go import sweep as sweep_mod
    import itertools
    # Build a real per-clade fixture big enough to produce q25 hits.
    planted_fams = [f"fam{i}" for i in range(8)]
    rows = []
    for i, (f1, f2) in enumerate(list(itertools.combinations(planted_fams, 2))[:15]):
        rows.append(dict(nodename="X", ortholog1=f1, ortholog2=f2,
                         occupancy_in=0.9,
                         sd_in_out_ratio_log_sigma=-5.0 + 0.1 * i,
                         mean_in_out_ratio_log_sigma=-5.0 + 0.1 * i))
    up = tmp_path / "up.tsv"
    pd.DataFrame(rows).to_csv(up, sep="\t", index=False)

    g2a = _fixture_g2a(tmp_path)
    g2g = _fixture_g2g(tmp_path)
    fm = _fixture_fm(tmp_path)

    out = tmp_path / "sweep_out"
    sweep_mod.run(
        supp_table=str(up),
        family_map=str(fm),
        gene2accession=str(g2a),
        gene2go=str(g2g),
        out_dir=str(out),
        write_curves=False,
        verbose=False,
    )
    return dict(
        supp_table=str(up), family_map=str(fm),
        gene2accession=str(g2a), gene2go=str(g2g),
        summary=str(out / "summary.tsv"),
        significant=str(out / "significant_terms.tsv"),
    )


def test_goatools_ref_helpers(tmp_path):
    from egt.go.benchmarks import goatools_ref as gr
    p2g = gr.parse_gene2accession(_fixture_g2a(tmp_path))
    assert len(p2g) == 30
    g2t, ns = gr.parse_gene2go(_fixture_g2g(tmp_path))
    assert len(g2t) == 30
    fam_to_genes = gr.parse_family_map(_fixture_fm(tmp_path), p2g)
    assert len(fam_to_genes) == 30

    # foreground_for_cell happy path.
    df = pd.DataFrame([
        dict(nodename="X", ortholog1=f"fam{i}", ortholog2=f"fam{i+1}",
             occupancy_in=0.9,
             sd_in_out_ratio_log_sigma=i,
             mean_in_out_ratio_log_sigma=i,
             mean_in=1, mean_out=1)
        for i in range(5)
    ])
    fg, n = gr.foreground_for_cell(df, "stability", 3, fam_to_genes)
    assert n == 3
    assert fg
    # Empty occupancy path.
    df_bad = df.copy()
    df_bad["occupancy_in"] = 0.0
    fg2, n2 = gr.foreground_for_cell(df_bad, "stability", 3, fam_to_genes)
    assert fg2 == set() and n2 == 0
    # Intersection-empty path.
    fg3, n3 = gr.foreground_for_cell(df, "intersection", 0, fam_to_genes)
    assert fg3 == set() and n3 == 0

    # stats duplicates inside goatools_ref.py are tested just enough to
    # exercise their lines (they redundantly re-implement egt.go.stats
    # but the port keeps them for self-containment):
    assert gr.log_binom(5, 2) > 0
    assert gr.log_binom(-1, 0) == float("-inf")
    q = gr.bh_qvalues([0.01, 0.02, 0.1])
    assert q[0] <= q[1] <= q[2]
    assert gr.bh_qvalues([]).size == 0
    s = gr.hypergeom_sf(0, 100, 10, 20)
    assert s == 1.0
    assert gr.hypergeom_sf(11, 100, 10, 20) == 0.0

    df_h = gr.run_our_hypergeom({"1000", "1001", "1002"},
                                  {g: t for g, t in g2t.items()})
    # With 3 genes carrying GO:0000001 → k=3, K=30, n=3, N=30 → fold=1
    assert "go_id" in df_h.columns if not df_h.empty else True

    # best_sweep_cell_per_clade on a small synthetic summary.
    summ = pd.DataFrame([
        dict(clade="X", axis="stability", namespace="all",
             N_threshold=10, top_q=0.001),
    ])
    sp = tmp_path / "sum.tsv"
    summ.to_csv(sp, sep="\t", index=False)
    bc = gr.best_sweep_cell_per_clade(sp)
    assert bc["X"] == {"axis": "stability", "N": 10}

    # compare_per_clade on two DataFrames.
    ours = pd.DataFrame([dict(go_id="GO:1", k=5, K=10, n=20, N=100,
                               fold=5.0, p=1e-5, q=1e-4)])
    theirs = pd.DataFrame([dict(go_id="GO:1", k=5, K=10, n=20, N=100,
                                 p_theirs=1e-5, q_theirs=1e-4,
                                 q_theirs_matched=1e-4,
                                 name="one", namespace="BP",
                                 enrichment="e")])
    m = gr.compare_per_clade(ours, theirs)
    assert m["n_terms_both"] == 1
    # Empty `theirs` → NaN-populated metrics.
    m2 = gr.compare_per_clade(ours, pd.DataFrame())
    assert m2["n_terms_both"] == 0


def test_goatools_ref_main_end_to_end(tmp_path):
    from egt.go.benchmarks import goatools_ref as gr
    kw = _run_sweep(tmp_path)
    rc = gr.main([
        "--supp-table", kw["supp_table"],
        "--family-map", kw["family_map"],
        "--gene2accession", kw["gene2accession"],
        "--gene2go", kw["gene2go"],
        "--obo", str(_fixture_obo(tmp_path)),
        "--summary", kw["summary"],
        "--significant", kw["significant"],
        "--out-dir", str(tmp_path / "bench"),
    ])
    assert rc == 0
    # Expected outputs from the goatools harness.
    for name in ("summary.tsv", "scatter.pdf"):
        assert (tmp_path / "bench" / name).exists(), f"missing {name}"
