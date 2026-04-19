"""Per-clade QC plot tests for ``egt defining-features``.

These tests guard the QC plotting code ported from
``dev_scripts/defining_features_plot2.py`` on odp's ``decay-branch``
at commit 898101525a8a84e4e9aad3b51504d578d08a3844. See
``src/egt/defining_features_qc_plots.py`` for the port; the primitive
``make_marginal_plot`` / ``scatter_hist`` / ``qq_plot`` helpers live
in ``egt.legacy.defining_features_plot2`` and are reused verbatim.

Coverage:
  1. ``write_qc_plots`` produces a non-empty multi-page PDF given a
     realistic per-clade stats DataFrame.
  2. ``process_coo_file`` emits the PDF alongside the TSV by default.
  3. The ``--no-qc-plots`` CLI flag (and programmatic
     ``qc_plots=False``) suppresses PDF emission without breaking the
     per-clade TSV.
  4. ``write_qc_plots_from_tsv`` round-trips a TSV back into a PDF
     (used for re-generating QC plots after the fact).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix, save_npz


# --- Fixture helpers --------------------------------------------------------

def _per_clade_df(n_pairs: int = 40, rng_seed: int = 0) -> pd.DataFrame:
    """Build a per-clade stats DataFrame matching the schema written
    by ``defining_features.process_coo_file``.

    We draw ``n_pairs`` pairs with mean_in / sd_in / mean_out / sd_out
    from a log-normal so the derived ratio columns are well-behaved
    (no -inf after log10), and occupancy values uniform on [0, 1].
    """
    rng = np.random.default_rng(rng_seed)
    mean_in = np.exp(rng.normal(loc=10.0, scale=1.0, size=n_pairs))
    sd_in = np.exp(rng.normal(loc=9.0, scale=0.5, size=n_pairs))
    mean_out = np.exp(rng.normal(loc=10.0, scale=1.0, size=n_pairs))
    sd_out = np.exp(rng.normal(loc=9.0, scale=0.5, size=n_pairs))
    notna_in = rng.integers(2, 20, size=n_pairs)
    notna_out = rng.integers(2, 20, size=n_pairs)
    # Occupancy uniform on [0, 1], but with some >=0.5 pairs so the
    # subset plots aren't empty.
    occ_in = rng.uniform(0.0, 1.0, size=n_pairs)
    occ_in[: n_pairs // 2] = rng.uniform(0.5, 1.0, size=n_pairs // 2)
    occ_out = rng.uniform(0.0, 1.0, size=n_pairs)
    return pd.DataFrame({
        "pair": np.arange(n_pairs, dtype=np.int64),
        "notna_in": notna_in.astype(np.int64),
        "notna_out": notna_out.astype(np.int64),
        "mean_in": mean_in, "sd_in": sd_in,
        "mean_out": mean_out, "sd_out": sd_out,
        "occupancy_in": occ_in, "occupancy_out": occ_out,
    })


def _build_csr_fixture():
    """Small 6 sp x 4 pair CSR. Same as test_defining_features_output."""
    dense = np.array([
        [10.0, 100.0,  0.0, 7.0],
        [20.0, 200.0,  0.0, 0.0],
        [30.0, 300.0,  0.0, 0.0],
        [40.0,  0.0, 400.0, 0.0],
        [50.0,  0.0, 500.0, 0.0],
        [60.0,  0.0, 600.0, 0.0],
    ], dtype=np.float64)
    csr = csr_matrix(dense)
    csr.eliminate_zeros()
    return csr


def _write_inputs(tmp_path: Path, csr, taxid_list_per_row):
    coo_path = tmp_path / "fx.coo.npz"
    save_npz(str(coo_path), csr.tocoo())
    n_species, n_pairs = csr.shape
    sampledf = pd.DataFrame({
        "sample": [f"species_{i:02d}" for i in range(n_species)],
        "taxid_list": [str(list(tl)) for tl in taxid_list_per_row],
    })
    sampledf.index.name = "idx"
    sampledf_path = tmp_path / "sampledf.tsv"
    sampledf.to_csv(sampledf_path, sep="\t")
    combo_path = tmp_path / "combo_to_index.txt"
    with open(combo_path, "w") as fh:
        for i in range(n_pairs):
            fh.write(f"('fam_{i:04d}_a', 'fam_{i:04d}_b')\t{i}\n")
    return sampledf_path, combo_path, coo_path


# --- Direct write_qc_plots tests -------------------------------------------

def test_write_qc_plots_produces_nonempty_pdf(tmp_path):
    """write_qc_plots returns a path, and that file is a non-empty PDF."""
    from egt.defining_features_qc_plots import write_qc_plots
    df = _per_clade_df()
    out = tmp_path / "Testclade_42_unique_pairs_qc.pdf"
    returned = write_qc_plots(df, out, nodename="Testclade", taxid=42)
    assert returned == out
    assert out.exists()
    # Not merely non-zero — a real PDF starts with the %PDF-1. magic.
    with open(out, "rb") as fh:
        head = fh.read(5)
    assert head == b"%PDF-", f"File at {out} is not a PDF (header={head!r})"
    assert out.stat().st_size > 1024, (
        f"PDF at {out} is suspiciously small ({out.stat().st_size} bytes) "
        f"— likely the plotting code produced an empty figure."
    )


def test_write_qc_plots_multipage(tmp_path):
    """The QC PDF must have multiple pages (one per axis-pair variant).
    The original decay-branch code emitted five separate single-page
    PDFs; we consolidate. Page count should be >= 5.

    We avoid adding pypdf as a new dep (not currently in the env) by
    grepping the raw PDF bytes for ``/Type /Page`` entries, which is a
    simple-but-reliable proxy for page count.
    """
    from egt.defining_features_qc_plots import write_qc_plots
    df = _per_clade_df()
    out = tmp_path / "Testclade_42_unique_pairs_qc.pdf"
    write_qc_plots(df, out, nodename="Testclade", taxid=42)
    raw = out.read_bytes()
    # Matplotlib's PdfPages emits one ``/Type /Page`` per page plus a
    # single ``/Type /Pages`` root for the tree; so page-count ~= count
    # of ``/Type /Page``  minus one for the root. Be lenient and just
    # check that the raw count is comfortably above the "one page"
    # baseline (>=6 hits -> >=5 pages).
    import re
    n_hits = len(re.findall(rb"/Type\s*/Page[^s]", raw))
    assert n_hits >= 5, (
        f"Expected >= 5 /Type /Page entries in QC PDF, got {n_hits}. "
        f"The original decay-branch code emitted five distinct "
        f"axis-pair figures and this port consolidates them."
    )


def test_write_qc_plots_empty_df_still_writes_something(tmp_path):
    """If the per-clade df has zero rows after NaN-dropping, we still
    want a valid PDF (possibly empty) rather than an exception — the
    caller (``process_coo_file``) already guards on ``len(df) > 0``,
    but defensive behavior here makes programmatic use safer."""
    from egt.defining_features_qc_plots import write_qc_plots
    # An all-NaN df still has a valid schema.
    df = pd.DataFrame({
        "pair": [0], "notna_in": [0], "notna_out": [0],
        "mean_in": [np.nan], "sd_in": [np.nan],
        "mean_out": [np.nan], "sd_out": [np.nan],
        "occupancy_in": [0.0], "occupancy_out": [0.0],
    })
    out = tmp_path / "x_1_unique_pairs_qc.pdf"
    # Should not raise; may produce a PDF with fewer pages.
    write_qc_plots(df, out, nodename="x", taxid=1)
    assert out.exists()


# --- write_qc_plots_from_tsv round-trip ------------------------------------

def test_write_qc_plots_from_tsv_round_trip(tmp_path):
    """A per-clade TSV on disk can be replayed back into a QC PDF."""
    from egt.defining_features_qc_plots import write_qc_plots_from_tsv
    df = _per_clade_df()
    tsv_path = tmp_path / "Clade_123_unique_pair_df.tsv.gz"
    df.to_csv(tsv_path, sep="\t", index=False, compression="gzip")
    out = write_qc_plots_from_tsv(tsv_path)
    # Default output name derives from the TSV stem.
    assert out.name == "Clade_123_unique_pair_df_qc.pdf"
    assert out.exists()
    with open(out, "rb") as fh:
        assert fh.read(5) == b"%PDF-"


def test_write_qc_plots_from_tsv_parses_nodename_and_taxid(tmp_path):
    """Nodename + taxid parsed from the filename so downstream scripts
    can just point at the TSV without re-supplying the clade metadata."""
    from egt.defining_features_qc_plots import write_qc_plots_from_tsv
    df = _per_clade_df()
    tsv_path = tmp_path / "Arthropoda_6656_unique_pair_df.tsv.gz"
    df.to_csv(tsv_path, sep="\t", index=False, compression="gzip")
    out_pdf = tmp_path / "out.pdf"
    result = write_qc_plots_from_tsv(tsv_path, out_pdf=out_pdf)
    assert result == out_pdf
    assert out_pdf.exists()


# --- process_coo_file integration ------------------------------------------

def test_process_coo_file_emits_qc_pdf_by_default(tmp_path):
    """End-to-end: running process_coo_file must drop a QC PDF next to
    the per-clade TSV when qc_plots is left at the default True."""
    from egt.defining_features import process_coo_file
    csr = _build_csr_fixture()
    taxid_lists = [[100], [100], [100], [200], [200], [200]]
    sampledf_path, combo_path, coo_path = _write_inputs(
        tmp_path, csr, taxid_lists)
    cwd_before = os.getcwd()
    os.chdir(tmp_path)
    try:
        process_coo_file(
            str(sampledf_path), str(combo_path), str(coo_path),
            dfoutfilepath="unused.df", taxid_list=[100])
    finally:
        os.chdir(cwd_before)
    # Per-clade TSV still written.
    tsvs = list(tmp_path.glob("*_100_unique_pair_df.tsv.gz"))
    assert len(tsvs) == 1, f"expected 1 TSV, got {tsvs}"
    # QC PDF emitted alongside it.
    pdfs = list(tmp_path.glob("*_100_unique_pairs_qc.pdf"))
    assert len(pdfs) == 1, (
        f"expected 1 QC PDF, got {pdfs}. "
        f"Directory listing: {sorted(p.name for p in tmp_path.iterdir())}"
    )
    assert pdfs[0].stat().st_size > 0


def test_process_coo_file_respects_qc_plots_false(tmp_path):
    """qc_plots=False suppresses the PDF but still writes the TSV."""
    from egt.defining_features import process_coo_file
    csr = _build_csr_fixture()
    taxid_lists = [[100], [100], [100], [200], [200], [200]]
    sampledf_path, combo_path, coo_path = _write_inputs(
        tmp_path, csr, taxid_lists)
    cwd_before = os.getcwd()
    os.chdir(tmp_path)
    try:
        process_coo_file(
            str(sampledf_path), str(combo_path), str(coo_path),
            dfoutfilepath="unused.df", taxid_list=[100],
            qc_plots=False)
    finally:
        os.chdir(cwd_before)
    tsvs = list(tmp_path.glob("*_100_unique_pair_df.tsv.gz"))
    assert len(tsvs) == 1
    pdfs = list(tmp_path.glob("*_100_unique_pairs_qc.pdf"))
    assert len(pdfs) == 0, (
        f"qc_plots=False should suppress PDFs, but found: {pdfs}"
    )


# --- CLI --no-qc-plots flag -------------------------------------------------

def test_cli_no_qc_plots_flag_parsed():
    """parse_args must accept --no-qc-plots and set args.qc_plots=False.
    The default (no flag) must leave qc_plots=True.
    """
    from egt.defining_features import parse_args
    # Provide all required args except --no-qc-plots; point them at /dev/null
    # workarounds that exist. Use a throwaway file set that parse_args's
    # os.path.exists check will accept.
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".npz") as fc, \
         tempfile.NamedTemporaryFile(suffix=".tsv") as fs, \
         tempfile.NamedTemporaryFile(suffix=".txt") as fk:
        argv_base = [
            "--coo_path", fc.name,
            "--sample_df_path", fs.name,
            "--coo_combination_path", fk.name,
        ]
        args_default = parse_args(argv_base)
        assert args_default.qc_plots is True, (
            f"Default should keep qc_plots=True, got {args_default.qc_plots}"
        )
        args_flag = parse_args(argv_base + ["--no-qc-plots"])
        assert args_flag.qc_plots is False, (
            f"--no-qc-plots should set qc_plots=False, "
            f"got {args_flag.qc_plots}"
        )


def test_cli_end_to_end_no_qc_plots(tmp_path):
    """End-to-end: calling egt.defining_features.main with --no-qc-plots
    on a real fixture produces the TSV but no QC PDF."""
    from egt.defining_features import main as defining_features_main
    csr = _build_csr_fixture()
    taxid_lists = [[100], [100], [100], [200], [200], [200]]
    sampledf_path, combo_path, coo_path = _write_inputs(
        tmp_path, csr, taxid_lists)
    cwd_before = os.getcwd()
    os.chdir(tmp_path)
    try:
        rv = defining_features_main([
            "--coo_path", str(coo_path),
            "--sample_df_path", str(sampledf_path),
            "--coo_combination_path", str(combo_path),
            "--taxid_list", "100",
            "--no-qc-plots",
        ])
    finally:
        os.chdir(cwd_before)
    assert rv == 0
    tsvs = list(tmp_path.glob("*_100_unique_pair_df.tsv.gz"))
    assert len(tsvs) == 1
    pdfs = list(tmp_path.glob("*_100_unique_pairs_qc.pdf"))
    assert len(pdfs) == 0


def test_cli_end_to_end_default_emits_qc_pdf(tmp_path):
    """End-to-end: no --no-qc-plots flag -> QC PDF emitted alongside TSV.

    Mirrors the expected production behavior of
    ``run_defining_features_new.sh``.
    """
    from egt.defining_features import main as defining_features_main
    csr = _build_csr_fixture()
    taxid_lists = [[100], [100], [100], [200], [200], [200]]
    sampledf_path, combo_path, coo_path = _write_inputs(
        tmp_path, csr, taxid_lists)
    cwd_before = os.getcwd()
    os.chdir(tmp_path)
    try:
        rv = defining_features_main([
            "--coo_path", str(coo_path),
            "--sample_df_path", str(sampledf_path),
            "--coo_combination_path", str(combo_path),
            "--taxid_list", "100",
        ])
    finally:
        os.chdir(cwd_before)
    assert rv == 0
    tsvs = list(tmp_path.glob("*_100_unique_pair_df.tsv.gz"))
    assert len(tsvs) == 1
    pdfs = list(tmp_path.glob("*_100_unique_pairs_qc.pdf"))
    assert len(pdfs) == 1
    # Header-sniff: real PDF.
    with open(pdfs[0], "rb") as fh:
        assert fh.read(5) == b"%PDF-"
