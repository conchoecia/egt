"""Coverage for dotplot features added after the initial port:

- min_k gate (k-floor companion dotplots at k ≥ 3 and k ≥ 4);
- q > 0.05 diagonal-slash overlay (the dotted red "did-not-clear"
  marker that matches the 0.05 colorbar reference line);
- empty-namespace panel on a per-clade dotplot page (draw_dotplot's
  `if terms_df.empty:` branch).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


def _synth_sig(rows):
    """Return a DataFrame in the short-name format that enrich.py
    expects after load_significant_terms aliasing. Every row carries
    clade/axis/N_threshold/sweep_namespace/go_id/go_name/go_namespace
    plus the k/n/K/N/fold/q short aliases."""
    return pd.DataFrame(rows)


def _row(clade, ns, go_id, go_name, k, n, K, N, fold, q, N_thr=10):
    return {
        "clade": clade, "axis": "stability", "N_threshold": N_thr,
        "sweep_namespace": "all",
        "go_id": go_id, "go_name": go_name, "go_namespace": ns,
        "k": k, "n": n, "K": K, "N": N,
        "fold": fold, "p": q, "q": q,
    }


def test_make_dotplots_min_k_gate_drops_low_k_terms(tmp_path):
    from egt.go.plots.enrich import make_dotplots
    # A k=2 hit (filtered out by min_k=3) and a k=3 hit (kept).
    sig = _synth_sig([
        _row("Bilateria", "BP", "GO:1", "k=2-term",
             k=2, n=20, K=30, N=2290, fold=7.6, q=0.01),
        _row("Bilateria", "BP", "GO:2", "k=3-term",
             k=3, n=20, K=30, N=2290, fold=11.4, q=0.001),
    ])
    out_no_gate = tmp_path / "dotplots.pdf"
    out_k3 = tmp_path / "dotplots_kge3.pdf"
    make_dotplots(sig, out_no_gate, min_fold=3.0, min_k=None)
    make_dotplots(sig, out_k3, min_fold=3.0, min_k=3)
    assert out_no_gate.exists() and out_no_gate.stat().st_size > 0
    # min_k=3 drops the k=2 term but keeps the k=3 one → file still drawn.
    assert out_k3.exists() and out_k3.stat().st_size > 0


def test_make_dotplots_min_k_drops_everything(tmp_path):
    from egt.go.plots.enrich import make_dotplots
    # All rows have k=2 so min_k=3 drops every term → no clade page is
    # drawn (PdfPages context closes without writing), but the function
    # must still complete cleanly.
    sig = _synth_sig([
        _row("Bilateria", "BP", "GO:1", "only-k2",
             k=2, n=20, K=30, N=2290, fold=7.6, q=0.01),
    ])
    out = tmp_path / "dotplots_allempty.pdf"
    make_dotplots(sig, out, min_fold=3.0, min_k=3)
    # matplotlib's PdfPages may write a stub file; the invariant we care
    # about is that make_dotplots didn't raise.


def test_make_dotplots_slash_overlay_for_q_above_0p05(tmp_path):
    from egt.go.plots.enrich import make_dotplots
    # Mix of q≤0.05 and q>0.05 so the q>0.05 slash overlay branch runs.
    sig = _synth_sig([
        _row("Bilateria", "BP", "GO:A", "sig-term",
             k=5, n=20, K=30, N=2290, fold=19.0, q=1e-4),
        _row("Bilateria", "BP", "GO:B", "not-sig",
             k=3, n=20, K=30, N=2290, fold=11.4, q=0.20),
        _row("Bilateria", "BP", "GO:C", "not-sig-2",
             k=4, n=20, K=30, N=2290, fold=15.3, q=0.30),
    ])
    out = tmp_path / "dotplots_slash.pdf"
    make_dotplots(sig, out, min_fold=3.0, min_k=None)
    assert out.exists() and out.stat().st_size > 0


def test_make_dotplots_empty_namespace_panel(tmp_path):
    from egt.go.plots.enrich import make_dotplots
    # Clade has BP hits only — the MF and CC panels are drawn empty,
    # exercising draw_dotplot's `terms_df.empty` branch (lines 280-283).
    sig = _synth_sig([
        _row("Bilateria", "BP", "GO:BP1", "bp-term",
             k=3, n=20, K=30, N=2290, fold=11.4, q=0.001),
    ])
    out = tmp_path / "dotplots_emptyns.pdf"
    make_dotplots(sig, out, min_fold=3.0, min_k=None)
    assert out.exists() and out.stat().st_size > 0


def test_make_dotplots_color_scheme_and_colorbar(tmp_path):
    """The ENRICHPLOT_CMAP + LogNorm + invert_yaxis stack should render
    without matplotlib complaints across a ~3-log-decade q range, with
    the 0.05 reference line inside the colorbar range."""
    from egt.go.plots.enrich import make_dotplots
    sig = _synth_sig([
        _row("Bilateria", "BP", f"GO:{i:03d}", f"term-{i}",
             k=3 + i, n=40, K=60, N=2290,
             fold=5.0 + i, q=10 ** (-3 - i * 0.3))
        for i in range(5)
    ])
    out = tmp_path / "dotplots_color.pdf"
    make_dotplots(sig, out, min_fold=3.0, min_k=None)
    assert out.exists() and out.stat().st_size > 0


def test_cli_emits_three_dotplot_variants(tmp_path):
    """The enrich.main driver must emit `dotplots.pdf`,
    `dotplots_kge3.pdf`, and `dotplots_kge4.pdf` as a triple."""
    # Build a tiny significant_terms.tsv + term_gene_lists + gene_symbols
    # + OBO fixture on the fly. Use a real clade name so the heatmap
    # tree-draw doesn't trip.
    from egt.go.plots.enrich import main as enrich_main
    # synth sig with canonical column names (what sweep.py actually
    # writes on disk — load_significant_terms aliases them).
    sig_rows = []
    gl_rows = []
    for i, (gid, name, ns) in enumerate([
        ("GO:0000001", "process-one", "BP"),
        ("GO:0000002", "function-one", "MF"),
        ("GO:0000004", "component-one", "CC"),
    ]):
        for clade in ("Bilateria", "Mollusca"):
            sig_rows.append({
                "clade": clade, "axis": "stability", "N_threshold": 10,
                "sweep_namespace": "all",
                "go_id": gid, "go_name": name, "go_namespace": ns,
                "foreground_hits_[k]": 5, "foreground_size_[n]": 20,
                "background_hits_[K]": 30, "background_size_[N]": 2290,
                "ratio_in_study_[k/n]": "5/20", "ratio_in_pop_[K/N]": "30/2290",
                "fold_enrichment": 19.1, "p_value": 1e-5,
                "correction_method": "fdr_bh", "q_value": 1e-4,
                "gene_ids": "1000;1001;1002;1003;1004",
                "gene_symbols": "G1;G2;G3;G4;G5",
            })
            gl_rows.append({
                "clade": clade, "axis": "stability", "N_threshold": 10,
                "go_id": gid, "k": 5,
                "gene_ids": "1000,1001,1002,1003,1004",
            })
    sig_path = tmp_path / "sig.tsv"
    pd.DataFrame(sig_rows).to_csv(sig_path, sep="\t", index=False)
    gl_path = tmp_path / "tg.tsv.gz"
    pd.DataFrame(gl_rows).to_csv(gl_path, sep="\t", index=False,
                                   compression="gzip")
    sym_path = tmp_path / "sym.tsv"
    pd.DataFrame([{"gene_id": str(1000 + i), "symbol": f"G{i + 1}"}
                  for i in range(5)]).to_csv(
        sym_path, sep="\t", index=False
    )
    obo_path = tmp_path / "mini.obo"
    obo_path.write_text(
        "format-version: 1.2\n\n"
        "[Term]\nid: GO:0000001\nname: process-one\nnamespace: biological_process\n\n"
        "[Term]\nid: GO:0000002\nname: function-one\nnamespace: molecular_function\n\n"
        "[Term]\nid: GO:0000004\nname: component-one\nnamespace: cellular_component\n\n"
    )
    out_dir = tmp_path / "plots"
    rc = enrich_main([
        "--significant-terms", str(sig_path),
        "--obo", str(obo_path),
        "--out-dir", str(out_dir),
        "--term-gene-lists", str(gl_path),
        "--gene-symbols", str(sym_path),
        "--min-fold", "0.0",  # keep synthetic data visible
    ])
    assert rc == 0
    for name in ("dotplots.pdf", "dotplots_kge3.pdf", "dotplots_kge4.pdf"):
        p = out_dir / name
        assert p.exists(), f"missing {name}"
        assert p.stat().st_size > 0
