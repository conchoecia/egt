"""
End-to-end MGT/MLT test on REAL Canidae data (opt-in, skipped without data).

This complements test_canidae_synthetic_end_to_end.py: the synthetic test pins
exact matrix values on toy data; this one proves the same CLI stage chain works
on a real comparative-genomics extract. It is gated behind an environment
variable so CI (which has no data bundle) skips it silently:

    EGT_CANIDAE_REALDATA=/path/to/bundle pytest tests/test_canidae_realdata_end_to_end.py

The bundle directory must contain four files:

    canid_sampledf.tsv        sample dataframe of canid genomes (a subset of a
                              full GTUMAP sampledf, reindexed 0..N-1, with the
                              dis_filepath columns dropped)
    canid_allsamples.coo.npz  the matching rows of the genome-by-locus-pair
                              distance matrix (allsamples.coo.npz)
    combo_to_index.txt        the ALG locus-pair -> column index map the matrix
                              was built with
    BCnSSimakov2022.rbh       the ALG RBH file the combo map was built from

Such a bundle is made by loading a full GTUMAP run, selecting the rows whose
`taxid_list` contains the clade taxid (9608, Canidae), and saving those rows
as a new .npz — i.e. real distances from real chromosome-scale assemblies,
small enough to run in minutes.

Everything is deterministic except the UMAP embedding: the mlt-matrix output
is validated against the inputs, the UMAP dataframes must carry the input
metadata unchanged, and the paramsweep PDFs must be legal PDF files whose
scatter-dot count equals panels x points exactly.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import load_npz

from egt import phylotreeumap as ptu
from egt import phylotreeumap_plotdfs as plotdfs
from egt import rbh_tools

from tests.test_canidae_synthetic_end_to_end import _count_pdf_dots

BUNDLE_ENV = "EGT_CANIDAE_REALDATA"
CANIDAE_TAXID = 9608
MLT_SWEEP = [(15, 0.1), (15, 0.5), (50, 0.1), (50, 0.5)]
MGT_SWEEP = [(5, 0.1), (5, 0.5), (10, 0.1), (10, 0.5)]

pytestmark = pytest.mark.skipif(
    not os.environ.get(BUNDLE_ENV) or not Path(os.environ.get(BUNDLE_ENV, "")).is_dir(),
    reason=f"real-data bundle not available; set {BUNDLE_ENV} to the bundle directory to run",
)


@pytest.fixture(scope="module")
def bundle():
    b = Path(os.environ[BUNDLE_ENV])
    paths = {
        "sampledf": b / "canid_sampledf.tsv",
        "coo": b / "canid_allsamples.coo.npz",
        "combo": b / "combo_to_index.txt",
        "rbh": b / "BCnSSimakov2022.rbh",
    }
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        pytest.skip(f"bundle incomplete, missing: {missing}")
    return paths


@pytest.fixture(scope="module")
def realdata(bundle, tmp_path_factory):
    """Run the full CLI stage chain on the real bundle once."""
    tmp = tmp_path_factory.mktemp("canidae_realdata")
    rbhdf = rbh_tools.parse_rbh(str(bundle["rbh"]))
    sampledf = pd.read_csv(bundle["sampledf"], sep="\t", index_col=0)
    n_loci, n_genomes = len(rbhdf), len(sampledf)

    mltcoo = tmp / "canidae_mlt.coo.npz"
    mltsampledf = tmp / "canidae_mlt.sampledf.tsv"
    assert ptu.main([
        "mlt-matrix",
        "--sampledf", str(bundle["sampledf"]),
        "--algcomboix", str(bundle["combo"]),
        "--coo", str(bundle["coo"]),
        "--alg-rbh", str(bundle["rbh"]),
        "--taxids-to-keep", str(CANIDAE_TAXID),
        "--nan-mode", "small",
        "--method", "mean",
        "--coo-out", str(mltcoo),
        "--sampledf-out", str(mltsampledf),
    ]) == 0

    mlt_dfs, mgt_dfs = [], []
    for n, m in MLT_SWEEP:
        out = tmp / f"CanidaeMLT.neighbors_{n}.mind_{m}.missing_large.method_mean.df"
        assert ptu.main([
            "mgt-mlt-umap",
            "--sampledf", str(mltsampledf),
            "--locus-file", str(bundle["rbh"]),
            "--coo", str(mltcoo),
            "--nan-mode", "large",
            "--n-neighbors", str(n),
            "--min-dist", str(m),
            "--df-out", str(out),
        ]) == 0
        mlt_dfs.append(out)
    for n, m in MGT_SWEEP:
        out = tmp / f"CanidaeMGT.neighbors_{n}.mind_{m}.missing_large.method_raw.df"
        assert ptu.main([
            "mgt-mlt-umap",
            "--sampledf", str(bundle["sampledf"]),
            "--locus-file", str(bundle["combo"]),
            "--coo", str(bundle["coo"]),
            "--nan-mode", "large",
            "--n-neighbors", str(n),
            "--min-dist", str(m),
            "--df-out", str(out),
        ]) == 0
        mgt_dfs.append(out)

    mltpdf = tmp / "CanidaeMLT.paramsweep.pdf"
    assert plotdfs.main(["-f", " ".join(str(p) for p in mlt_dfs), "-p", str(tmp / "CanidaeMLT.paramsweep"), "--pdf"]) == 0
    mgtpdf = tmp / "CanidaeMGT.paramsweep.pdf"
    assert plotdfs.main(["-f", " ".join(str(p) for p in mgt_dfs), "-p", str(tmp / "CanidaeMGT.paramsweep"), "--pdf"]) == 0

    return {
        "rbhdf": rbhdf,
        "sampledf": sampledf,
        "n_loci": n_loci,
        "n_genomes": n_genomes,
        "mltcoo": mltcoo,
        "mltsampledf": mltsampledf,
        "mlt_dfs": mlt_dfs,
        "mgt_dfs": mgt_dfs,
        "mltpdf": mltpdf,
        "mgtpdf": mgtpdf,
    }


def test_bundle_is_internally_consistent(bundle, realdata):
    combo = ptu.algcomboix_file_to_dict(str(bundle["combo"]))
    n = realdata["n_loci"]
    assert len(combo) == n * (n - 1) // 2
    coo = load_npz(bundle["coo"])
    assert coo.shape == (realdata["n_genomes"], len(combo))
    # every genome in the bundle is a canid, so the clade filter keeps them all
    kept = pd.read_csv(realdata["mltsampledf"], sep="\t", index_col=0)
    assert list(kept["sample"]) == list(realdata["sampledf"]["sample"])


def test_mlt_matrix_shape_and_symmetry(realdata):
    mlt = load_npz(realdata["mltcoo"]).toarray()
    n = realdata["n_loci"]
    assert mlt.shape == (n, n)
    assert np.allclose(mlt, mlt.T)
    assert np.allclose(np.diag(mlt), 0)
    assert (mlt > 0).any()


def test_mlt_umap_dfs_carry_rbh_metadata(realdata):
    for dfpath in realdata["mlt_dfs"]:
        df = pd.read_csv(dfpath, sep="\t", index_col=0)
        assert len(df) == realdata["n_loci"]
        meta = df.drop(columns=["UMAP1", "UMAP2"])
        pd.testing.assert_frame_equal(meta, realdata["rbhdf"], check_dtype=False)
        assert np.isfinite(df[["UMAP1", "UMAP2"]].to_numpy()).all()


def test_mgt_umap_dfs_carry_sample_metadata(realdata):
    for dfpath in realdata["mgt_dfs"]:
        df = pd.read_csv(dfpath, sep="\t", index_col=0)
        assert len(df) == realdata["n_genomes"]
        assert list(df["sample"]) == list(realdata["sampledf"]["sample"])
        assert np.isfinite(df[["UMAP1", "UMAP2"]].to_numpy()).all()


def test_paramsweep_pdfs_are_legal_with_expected_dot_counts(realdata):
    for pdf, n_points, sweep in [
        (realdata["mltpdf"], realdata["n_loci"], MLT_SWEEP),
        (realdata["mgtpdf"], realdata["n_genomes"], MGT_SWEEP),
    ]:
        raw = pdf.read_bytes()
        assert raw.startswith(b"%PDF-"), f"{pdf.name} lacks a PDF header"
        assert b"%%EOF" in raw[-1024:], f"{pdf.name} lacks a PDF trailer"
        assert _count_pdf_dots(pdf) == n_points * len(sweep)
