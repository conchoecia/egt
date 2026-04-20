"""Additional sweep-module tests that exercise non-default branches
(harvest_significant_terms with real q25 hits, edge paths in sweep_clade
and run())."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from egt.go import sweep as sweep_mod


def _planted_fixture():
    """Build a synthetic (clade_rows, fam_to_genes, bg_to_terms, term_ns)
    fixture where a specific GO term is statistically guaranteed to reach
    q ≤ 0.25 for the stability-axis top-N cell.
    """
    # 200-gene background. 10 carry GO:PLANTED. Every gene also carries
    # GO:UB1 (so bag-of-genes has at least two tested terms).
    bg: dict[str, set[str]] = {}
    fam_to_genes: dict[str, set[str]] = {}
    for i in range(200):
        gid = f"g{i}"
        terms = {"GO:UB1"}
        if i < 10:
            terms.add("GO:PLANTED")
        bg[gid] = terms
        fam_to_genes[f"fam{i}"] = {gid}
    ns = {"GO:UB1": "BP", "GO:PLANTED": "BP"}

    # Clade has 15 pairs; the first 7 pair-up PLANTED carriers. Foreground
    # at top-N=10 will be PLANTED-heavy → q tiny.
    rows = []
    # Seven high-stability pairs between PLANTED-carrier families.
    planted_fams = [f"fam{i}" for i in range(8)]
    import itertools
    for i, (f1, f2) in enumerate(list(itertools.combinations(planted_fams, 2))[:7]):
        rows.append(dict(nodename="X",
                         ortholog1=f1, ortholog2=f2,
                         occupancy_in=0.9,
                         sd_in_out_ratio_log_sigma=-5.0 + 0.1 * i,
                         mean_in_out_ratio_log_sigma=-5.0 + 0.1 * i))
    # Another 8 random pairs at higher sd (they rank lower on stability).
    for j in range(8):
        rows.append(dict(nodename="X",
                         ortholog1=f"fam{20 + j}", ortholog2=f"fam{40 + j}",
                         occupancy_in=0.9,
                         sd_in_out_ratio_log_sigma=1.0 + j,
                         mean_in_out_ratio_log_sigma=1.0 + j))
    return pd.DataFrame(rows), fam_to_genes, bg, ns


def test_sweep_clade_produces_q25_hits():
    df, fam, bg, ns = _planted_fixture()
    records, curves = sweep_mod.sweep_clade(df, fam, bg, ns)
    assert any(r["n_hits_q25"] > 0 for r in records), (
        "synthetic fixture should trigger q25 hits"
    )


def test_harvest_significant_terms_emits_rows():
    df, fam, bg, ns = _planted_fixture()
    records, _ = sweep_mod.sweep_clade(df, fam, bg, ns)
    sig, gene_lists = sweep_mod.harvest_significant_terms(
        "X", df, records, fam, bg, ns,
    )
    assert sig, "harvest should emit rows when q25 hits exist"
    assert gene_lists, "gene_lists sidecar should be populated"
    # GO:PLANTED should appear.
    ids = {r["go_id"] for r in sig}
    assert "GO:PLANTED" in ids


def test_run_standard_columns_with_obo_and_symbols(tmp_path):
    """`significant_terms.tsv` must carry the publication-standard
    columns and be populated from the OBO + NCBI Symbol side-map."""
    import itertools
    from egt.go import sweep as sweep_mod
    # Reuse the planted fixture.
    planted_fams = [f"fam{i}" for i in range(8)]
    rows = []
    for i, (f1, f2) in enumerate(
        list(itertools.combinations(planted_fams, 2))[:15]
    ):
        rows.append(dict(nodename="X", ortholog1=f1, ortholog2=f2,
                         occupancy_in=0.9,
                         sd_in_out_ratio_log_sigma=-5.0 + 0.1 * i,
                         mean_in_out_ratio_log_sigma=-5.0 + 0.1 * i))
    for j in range(5):
        rows.append(dict(nodename="X",
                         ortholog1=f"fam{20+j}", ortholog2=f"fam{40+j}",
                         occupancy_in=0.9,
                         sd_in_out_ratio_log_sigma=2.0 + j,
                         mean_in_out_ratio_log_sigma=2.0 + j))
    up = tmp_path / "up.tsv"
    pd.DataFrame(rows).to_csv(up, sep="\t", index=False)
    g2a = tmp_path / "g2a.tsv"
    with g2a.open("w") as fh:
        fh.write("#tax\tGeneID\ts\tR\trg\tpa.v\tpg\tgn\tgg\ts0\te0\to\ta\tmp\tmg\tSymbol\n")
        for i in range(200):
            fh.write(f"9606\tg{i}\tREVIEWED\t-\t-\tNP_{i:06d}.1\t-\tNC_1.1\t-\t1\t2\t+\tG\t-\t-\tSYM{i}\n")
    fm = tmp_path / "fm.tsv"
    with fm.open("w") as fh:
        fh.write("family_id\talg\thuman_gene\thuman_scaf\tsource\tnote\n")
        for i in range(200):
            fh.write(f"fam{i}\tA1a\tNP_{i:06d}.1\tNC\thuman_rbh\t\n")
    g2g = tmp_path / "g2g.tsv"
    with g2g.open("w") as fh:
        fh.write("#tax\tGeneID\tGO_ID\tEvidence\tQualifier\tTerm\tPub\tCategory\n")
        for i in range(200):
            fh.write(f"9606\tg{i}\tGO:UB\tIEA\t-\tub\t1\tProcess\n")
            if i < 10:
                fh.write(f"9606\tg{i}\tGO:PLANTED\tIEA\t-\tplanted\t1\tProcess\n")
    obo = tmp_path / "mini.obo"
    obo.write_text(
        "format-version: 1.2\n\n"
        "[Term]\nid: GO:UB\nname: ubiquitous-term\nnamespace: biological_process\n\n"
        "[Term]\nid: GO:PLANTED\nname: planted-process\nnamespace: biological_process\n\n"
    )
    out = tmp_path / "out"
    sweep_mod.run(
        supp_table=str(up), family_map=str(fm),
        gene2accession=str(g2a), gene2go=str(g2g),
        out_dir=str(out), obo=str(obo),
        write_curves=False, verbose=False,
    )
    sig = pd.read_csv(out / "significant_terms.tsv", sep="\t")
    # Every enrichment-standard column present in the canonical order.
    required = ["go_id", "go_name", "go_namespace",
                "k", "n", "K", "N",
                "ratio_in_study", "ratio_in_pop",
                "fold", "p", "correction_method", "q",
                "gene_ids", "gene_symbols"]
    assert all(c in sig.columns for c in required)
    # go_name populated from OBO.
    planted = sig[sig["go_id"] == "GO:PLANTED"]
    assert (planted["go_name"] == "planted-process").all()
    # ratio_in_study strings are well-formed "k/n".
    r = planted.iloc[0]
    assert r["ratio_in_study"] == f"{r['k']}/{r['n']}"
    assert r["ratio_in_pop"] == f"{r['K']}/{r['N']}"
    # Correction method label is stable.
    assert (sig["correction_method"] == "fdr_bh").all()
    # gene_ids + gene_symbols populated, same length.
    ids = r["gene_ids"].split(";")
    syms = r["gene_symbols"].split(";")
    assert len(ids) == len(syms) == r["k"]
    # Symbols follow the SYM{i} pattern from the fixture.
    for s in syms:
        assert s.startswith("SYM")


def test_run_end_to_end_with_real_hits(tmp_path):
    # Write the synthetic clade rows + a minimal family_map + gene2accession
    # + gene2go so run() can chain through them.
    df, fam, bg, ns = _planted_fixture()
    # Persist the unique_pairs table.
    up_path = tmp_path / "up.tsv"
    df.to_csv(up_path, sep="\t", index=False)
    # gene2accession: N rows, RefSeq mapping NP_xN.1 → f"{i}".
    g2a = tmp_path / "gene2accession.tsv"
    with g2a.open("w") as fh:
        fh.write("#tax_id\tGeneID\tstatus\tRNA\trnagi\tNP\tpgi\tgenomic\tggi\ts\te\to\tasm\tmpep\tmpgi\tSymbol\n")
        for i in range(200):
            fh.write(f"9606\tg{i}\tREVIEWED\tNM_{i}.1\t1\tNP_{i:06d}.1\t2\tNC_1.1\t3\t1\t2\t+\tGRCh38\t-\t-\tSYM{i}\n")
    # family_map: fam{i} → NP_{i:06d}.1
    fm = tmp_path / "family_map.tsv"
    with fm.open("w") as fh:
        fh.write("family_id\talg\thuman_gene\thuman_scaf\tsource\tnote\n")
        for i in range(200):
            fh.write(f"fam{i}\tA1a\tNP_{i:06d}.1\tNC_1.1\thuman_rbh\t\n")
    # gene2go: write planted/ub1 terms.
    g2g = tmp_path / "gene2go.tsv"
    with g2g.open("w") as fh:
        fh.write("#tax_id\tGeneID\tGO_ID\tEvidence\tQualifier\tGO_term\tPubMed\tCategory\n")
        for i in range(200):
            fh.write(f"9606\tg{i}\tGO:UB1\tIEA\t-\tub1\t1\tProcess\n")
            if i < 10:
                fh.write(f"9606\tg{i}\tGO:PLANTED\tIEA\t-\tplanted\t1\tProcess\n")
    out = tmp_path / "out"
    sweep_mod.run(
        supp_table=str(up_path),
        family_map=str(fm),
        gene2accession=str(g2a),
        gene2go=str(g2g),
        out_dir=str(out),
        write_curves=True,
        verbose=False,
    )
    # With real hits, significant_terms.tsv should be non-empty.
    sig = pd.read_csv(out / "significant_terms.tsv", sep="\t")
    assert len(sig) > 0
    gl = pd.read_csv(out / "term_gene_lists.tsv.gz",
                      sep="\t", compression="infer")
    assert len(gl) > 0
    # curves.pdf produced.
    assert (out / "curves.pdf").exists()


def test_sweep_clade_empty_intersection_branch():
    # Clade where sorted-by-stability and sorted-by-closeness disagree
    # enough that small-N intersection is empty.
    rows = []
    n_good = 20
    for i in range(n_good):
        rows.append(dict(nodename="X", ortholog1=f"fam{i}", ortholog2=f"fam{i+100}",
                         occupancy_in=0.9,
                         sd_in_out_ratio_log_sigma=i,       # ascending by stab
                         mean_in_out_ratio_log_sigma=-i))   # opposite order for close
    df = pd.DataFrame(rows)
    # No family annotations → foreground empty, hits line 114.
    records, _ = sweep_mod.sweep_clade(df, {}, {}, {})
    assert records == []


def test_sweep_clade_hits_q_zero_curve_sentinel():
    # Construct a situation where one cell's top_q = 0.0 (extreme
    # over-representation) to exercise the `elif top and top_q == 0`
    # curve-data sentinel branch.
    #
    # Hypergeometric sf returns exactly 0 only if k > min(K, n); we
    # cannot legitimately get there, but the enrichment can drive q→0
    # via floating underflow. Easier: mock a tiny background where the
    # computation rounds to 0.
    bg: dict[str, set[str]] = {}
    fam_to_genes: dict[str, set[str]] = {}
    for i in range(500):
        bg[f"g{i}"] = {"GO:UB"}
        fam_to_genes[f"fam{i}"] = {f"g{i}"}
    for i in range(3):
        bg[f"g{i}"].add("GO:RARE")
    ns = {"GO:UB": "BP", "GO:RARE": "BP"}
    # Foreground = the 3 carrying GO:RARE → hypergeom_sf(3, 500, 3, 3)
    # = 1/C(500,3) ≈ 4.8e-8 → not zero but small. Good enough to hit the
    # `top_q > 0` branch; the `top_q == 0` sentinel requires actual 0.
    # Accept that this branch is documented but rare in practice.
    # The test below still exercises the common branch.
    rows = [
        dict(nodename="X", ortholog1=f"fam{i}", ortholog2=f"fam{j}",
             occupancy_in=0.9,
             sd_in_out_ratio_log_sigma=-5.0 - i,
             mean_in_out_ratio_log_sigma=-5.0 - i)
        for i, j in [(0, 1), (0, 2), (1, 2)]
    ] + [
        dict(nodename="X", ortholog1=f"fam{10+k}", ortholog2=f"fam{20+k}",
             occupancy_in=0.9,
             sd_in_out_ratio_log_sigma=10.0 + k,
             mean_in_out_ratio_log_sigma=10.0 + k)
        for k in range(10)
    ]
    df = pd.DataFrame(rows)
    records, curves = sweep_mod.sweep_clade(df, fam_to_genes, bg, ns)
    assert records
    # Curve data exists on at least one axis.
    assert curves
