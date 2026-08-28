"""
End-to-end MGT/MLT regression test on a synthetic Canidae-like dataset.

This module was motivated by the 0.2.5 MLT crashes: `mgt-mlt-umap` could not
parse an ALG RBH locus file, and `--nan-mode small|large` crashed before the
UMAP was computed. Those bugs survived because the CLI tests monkeypatched the
pipeline functions, so nothing ever ran the real stage chain. This test runs
the real chain, through the public CLI, on data small enough for CI:

    algcomboix -> combine-distances -> mlt-matrix -> mgt-mlt-umap (MGT + MLT)
    -> phylotreeumap_plotdfs paramsweep PDFs

ALL DATA IN THIS MODULE ARE SYNTHETIC. The species names and taxids are public
NCBI identifiers (e.g. 9608 Canidae, 33554 Carnivora), used only so the
lineage-filtering logic is exercised with realistic-looking lineages. The
genomes, gene families, and distances are generated from a fixed seed.

The simulated karyotypes encode a simple, visually checkable story:
  - 4 synthetic ALGs (A, B, C, D) with 20 loci each = 80 loci.
  - wolf-like canids carry an A+B fusion chromosome; fox-like canids keep all
    four ALGs on separate chromosomes; the two outgroups (cat-like, human-like)
    have their own arrangements and are excluded from the MLT averaging by
    `--taxids-to-keep 9608`.
  - Expected MLT UMAP: loci cluster by ALG; A and B are entangled with each
    other (the fusion mixes them) while C and D form isolated clusters.
  - Expected MGT UMAP: 24 genomes (8 species x 3 simulated assemblies) cluster
    by karyotype arrangement.

Everything is deterministic EXCEPT the UMAP embedding coordinates: the input
matrices, the row/column identities of every output dataframe, and the number
of dots in the PDFs are all asserted exactly; the embedding is only checked
for shape, finiteness, and coarse cluster structure.
"""
from __future__ import annotations

import gzip
import re
import zlib
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import load_npz

from egt import phylotreeumap as ptu
from egt import phylotreeumap_plotdfs as plotdfs

RNG_SEED = 9608  # the Canidae taxid

N_LOCI_PER_ALG = 20
ALG_COLORS = {
    "CanidaeSimALG_A": "#1f77b4",
    "CanidaeSimALG_B": "#d62728",
    "CanidaeSimALG_C": "#2ca02c",
    "CanidaeSimALG_D": "#9467bd",
}
ALG_NAMES = list(ALG_COLORS)

# Karyotypes: each inner tuple is one chromosome carrying those ALGs.
WOLFLIKE = (("CanidaeSimALG_A", "CanidaeSimALG_B"), ("CanidaeSimALG_C",), ("CanidaeSimALG_D",))
FOXLIKE = (("CanidaeSimALG_A",), ("CanidaeSimALG_B",), ("CanidaeSimALG_C",), ("CanidaeSimALG_D",))
CATLIKE = (("CanidaeSimALG_A",), ("CanidaeSimALG_B",), ("CanidaeSimALG_C", "CanidaeSimALG_D"))
HUMANLIKE = (("CanidaeSimALG_A", "CanidaeSimALG_C"), ("CanidaeSimALG_B", "CanidaeSimALG_D"))

# (species, taxid, lineage root->tip, karyotype, plotting color)
# Lineage taxids: 40674 Mammalia, 33554 Carnivora, 9608 Canidae, 9681 Felidae, 9443 Primates.
SPECIES = [
    ("Canis_lupus", 9612, [1, 40674, 33554, 9608, 9612], WOLFLIKE, "#33658a"),
    ("Canis_latrans", 9614, [1, 40674, 33554, 9608, 9614], WOLFLIKE, "#33658a"),
    ("Lycaon_pictus", 9622, [1, 40674, 33554, 9608, 9622], WOLFLIKE, "#33658a"),
    ("Vulpes_vulpes", 9627, [1, 40674, 33554, 9608, 9627], FOXLIKE, "#f26419"),
    ("Vulpes_lagopus", 494514, [1, 40674, 33554, 9608, 494514], FOXLIKE, "#f26419"),
    ("Nyctereutes_procyonoides", 34880, [1, 40674, 33554, 9608, 34880], FOXLIKE, "#f26419"),
    # outgroups; dropped from the MLT averaging by --taxids-to-keep 9608
    ("Felis_catus", 9685, [1, 40674, 33554, 9681, 9685], CATLIKE, "#758e4f"),
    ("Homo_sapiens", 9606, [1, 40674, 9443, 9606], HUMANLIKE, "#888888"),
]
N_REPLICATES = 3  # simulated assemblies per species
N_LOCI = N_LOCI_PER_ALG * len(ALG_NAMES)
N_GENOMES = len(SPECIES) * N_REPLICATES
N_CANID_GENOMES = 6 * N_REPLICATES

MLT_SWEEP = [(5, 0.05), (5, 0.5), (15, 0.05), (15, 0.5)]
MGT_SWEEP = [(3, 0.05), (3, 0.5), (7, 0.05), (7, 0.5)]


def _locus_names():
    return [f"CanidaeSim_genefamily_{i:03d}" for i in range(N_LOCI)]


def _alg_of_locus():
    """locus name -> ALG name, 20 loci per ALG in order A, B, C, D."""
    names = _locus_names()
    return {names[i]: ALG_NAMES[i // N_LOCI_PER_ALG] for i in range(N_LOCI)}


def _write_alg_rbh(path: Path) -> pd.DataFrame:
    locus_to_alg = _alg_of_locus()
    names = _locus_names()
    rbhdf = pd.DataFrame(
        {
            "rbh": names,
            "gene_group": [locus_to_alg[n] for n in names],
            "color": [ALG_COLORS[locus_to_alg[n]] for n in names],
            "REF_scaf": [locus_to_alg[n] for n in names],
            "REF_gene": [f"refgene_{i:03d}" for i in range(N_LOCI)],
            "REF_pos": [(i % N_LOCI_PER_ALG) + 1 for i in range(N_LOCI)],
        }
    )
    rbhdf.to_csv(path, sep="\t", index=False)
    return rbhdf


def _simulate_genomes(rng):
    """
    Returns {sample_name: {(rbh1, rbh2): distance}} with rbh1 < rbh2, plus the
    sampledf rows. Loci on the same chromosome are ~1 Mb apart with seeded
    jitter; loci on different chromosomes never co-occur (missing data).
    """
    locus_to_alg = _alg_of_locus()
    by_alg = {alg: [n for n in _locus_names() if locus_to_alg[n] == alg] for alg in ALG_NAMES}
    genome_dists = {}
    rows = []
    for species, taxid, lineage, karyotype, color in SPECIES:
        for rep in range(N_REPLICATES):
            sample = f"{species}-{taxid}-SIM{rep}"
            dists = {}
            for chromosome in karyotype:
                loci = [n for alg in chromosome for n in by_alg[alg]]
                positions = {
                    locus: (k + 1) * 1_000_000 + int(rng.integers(0, 200_001))
                    for k, locus in enumerate(loci)
                }
                for a, b in combinations(loci, 2):
                    key = tuple(sorted((a, b)))
                    dists[key] = abs(positions[a] - positions[b])
            genome_dists[sample] = dists
            rows.append(
                {
                    "sample": sample,
                    "taxid": taxid,
                    "taxid_list": str(lineage),
                    "color": color,
                }
            )
    return genome_dists, pd.DataFrame(rows)


def _count_pdf_dots(pdf_path: Path) -> int:
    """
    Count scatter markers painted in the PDF. matplotlib's PDF backend draws
    each scatter point as one `Do` invocation of a marker XObject named /P<n>
    (older versions used /M<n>); content streams are FlateDecode-compressed.
    """
    raw = pdf_path.read_bytes()
    text = b""
    for stream in re.findall(rb"stream\r?\n(.*?)endstream", raw, flags=re.DOTALL):
        try:
            text += zlib.decompress(stream)
        except zlib.error:
            text += stream
    return len(re.findall(rb"/[MP]\d+ Do", text))


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    """Build the synthetic dataset and run the full CLI stage chain once."""
    tmp = tmp_path_factory.mktemp("canidae_synthetic")
    rng = np.random.default_rng(RNG_SEED)

    rbhfile = tmp / "CanidaeSim.rbh"
    rbhdf = _write_alg_rbh(rbhfile)

    genome_dists, sampledf = _simulate_genomes(rng)
    for sample, dists in genome_dists.items():
        gbgz = tmp / f"{sample}.gb.gz"
        rows = [(a, b, d) for (a, b), d in sorted(dists.items())]
        with gzip.open(gbgz, "wt") as fh:
            pd.DataFrame(rows, columns=["rbh1", "rbh2", "distance"]).to_csv(fh, sep="\t", index=False)
    sampledf["dis_filepath_abs"] = [str(tmp / f"{s}.gb.gz") for s in sampledf["sample"]]
    sampledffile = tmp / "sampledf.tsv"
    sampledf.to_csv(sampledffile, sep="\t")

    combofile = tmp / "combo_to_index.tsv"
    assert ptu.main(["algcomboix", "--alg-rbh", str(rbhfile), "--output", str(combofile)]) == 0

    mgtcoo = tmp / "allsamples.coo.npz"
    assert ptu.main([
        "combine-distances",
        "--sampledf", str(sampledffile),
        "--algcomboix", str(combofile),
        "--output", str(mgtcoo),
    ]) == 0

    mltcoo = tmp / "canidae_mlt.coo.npz"
    mltsampledf = tmp / "canidae_mlt.sampledf.tsv"
    assert ptu.main([
        "mlt-matrix",
        "--sampledf", str(sampledffile),
        "--algcomboix", str(combofile),
        "--coo", str(mgtcoo),
        "--alg-rbh", str(rbhfile),
        "--taxids-to-keep", "9608",
        "--nan-mode", "small",
        "--method", "mean",
        "--coo-out", str(mltcoo),
        "--sampledf-out", str(mltsampledf),
    ]) == 0

    mlt_dfs, mgt_dfs = [], []
    for n, m in MLT_SWEEP:
        out = tmp / f"CanidaeSimMLT.neighbors_{n}.mind_{m}.missing_large.method_mean.df"
        assert ptu.main([
            "mgt-mlt-umap",
            "--sampledf", str(mltsampledf),
            "--locus-file", str(rbhfile),
            "--coo", str(mltcoo),
            "--nan-mode", "large",
            "--n-neighbors", str(n),
            "--min-dist", str(m),
            "--df-out", str(out),
        ]) == 0
        mlt_dfs.append(out)
    for n, m in MGT_SWEEP:
        out = tmp / f"CanidaeSimMGT.neighbors_{n}.mind_{m}.missing_large.method_raw.df"
        assert ptu.main([
            "mgt-mlt-umap",
            "--sampledf", str(sampledffile),
            "--locus-file", str(combofile),
            "--coo", str(mgtcoo),
            "--nan-mode", "large",
            "--n-neighbors", str(n),
            "--min-dist", str(m),
            "--df-out", str(out),
        ]) == 0
        mgt_dfs.append(out)

    mltpdf = tmp / "CanidaeSimMLT.paramsweep.pdf"
    assert plotdfs.main(["-f", " ".join(str(p) for p in mlt_dfs), "-p", str(tmp / "CanidaeSimMLT.paramsweep"), "--pdf"]) == 0
    mgtpdf = tmp / "CanidaeSimMGT.paramsweep.pdf"
    assert plotdfs.main(["-f", " ".join(str(p) for p in mgt_dfs), "-p", str(tmp / "CanidaeSimMGT.paramsweep"), "--pdf"]) == 0

    return {
        "tmp": tmp,
        "rbhdf": rbhdf,
        "sampledf": sampledf,
        "genome_dists": genome_dists,
        "combofile": combofile,
        "mgtcoo": mgtcoo,
        "mltcoo": mltcoo,
        "mltsampledf": mltsampledf,
        "mlt_dfs": mlt_dfs,
        "mgt_dfs": mgt_dfs,
        "mltpdf": mltpdf,
        "mgtpdf": mgtpdf,
    }


def test_mgt_coo_matrix_is_deterministic(pipeline):
    combo = ptu.algcomboix_file_to_dict(str(pipeline["combofile"]))
    coo = load_npz(pipeline["mgtcoo"]).toarray()
    assert coo.shape == (N_GENOMES, len(combo))
    assert len(combo) == N_LOCI * (N_LOCI - 1) // 2
    expected = np.zeros_like(coo)
    for g, sample in enumerate(pipeline["sampledf"]["sample"]):
        for pair, dist in pipeline["genome_dists"][sample].items():
            expected[g, combo[pair]] = dist
    assert np.array_equal(coo, expected)


def test_mlt_matrix_filters_to_canidae_and_is_deterministic(pipeline):
    kept = pd.read_csv(pipeline["mltsampledf"], sep="\t", index_col=0)
    # --taxids-to-keep 9608 must drop the 6 outgroup genomes
    assert len(kept) == N_CANID_GENOMES
    assert not kept["sample"].str.contains("Felis|Homo").any()

    mlt = load_npz(pipeline["mltcoo"]).toarray()
    assert mlt.shape == (N_LOCI, N_LOCI)
    # method=mean, nan-mode=small: entry (i, j) is the plain mean over the kept
    # genomes of the pair's distance, with 0 where the pair never co-occurs.
    names = _locus_names()
    ix = {n: i for i, n in enumerate(names)}
    expected = np.zeros((N_LOCI, N_LOCI))
    for sample in kept["sample"]:
        for (a, b), dist in pipeline["genome_dists"][sample].items():
            expected[ix[a], ix[b]] += dist
            expected[ix[b], ix[a]] += dist
    expected /= N_CANID_GENOMES
    assert np.allclose(mlt, expected)
    # the simulated fusion: A-B pairs co-occur (in wolf-like genomes only),
    # A-C / A-D / B-C / B-D / C-D pairs never do
    assert expected[0, N_LOCI_PER_ALG] > 0  # first A locus vs first B locus
    assert expected[0, 2 * N_LOCI_PER_ALG] == 0  # first A locus vs first C locus


def test_mlt_matrix_phylogenetic_method_runs(pipeline):
    out = pipeline["tmp"] / "canidae_mlt_phylo.coo.npz"
    outsample = pipeline["tmp"] / "canidae_mlt_phylo.sampledf.tsv"
    assert ptu.main([
        "mlt-matrix",
        "--sampledf", str(pipeline["tmp"] / "sampledf.tsv"),
        "--algcomboix", str(pipeline["combofile"]),
        "--coo", str(pipeline["mgtcoo"]),
        "--alg-rbh", str(pipeline["tmp"] / "CanidaeSim.rbh"),
        "--taxids-to-keep", "9608",
        "--nan-mode", "small",
        "--method", "phylogenetic",
        "--coo-out", str(out),
        "--sampledf-out", str(outsample),
    ]) == 0
    phylo = load_npz(out).toarray()
    assert phylo.shape == (N_LOCI, N_LOCI)
    assert np.allclose(np.diag(phylo), 0)
    # same sparsity structure as the mean method: fused A-B pairs present,
    # never-co-occurring cross-ALG pairs absent
    assert phylo[0, N_LOCI_PER_ALG] > 0
    assert phylo[0, 2 * N_LOCI_PER_ALG] == 0


def test_mlt_umap_dfs_are_deterministic_except_embedding(pipeline):
    for dfpath in pipeline["mlt_dfs"]:
        df = pd.read_csv(dfpath, sep="\t", index_col=0)
        assert len(df) == N_LOCI
        # every column except the embedding must round-trip the rbh file exactly
        meta = df.drop(columns=["UMAP1", "UMAP2"])
        pd.testing.assert_frame_equal(meta, pipeline["rbhdf"], check_dtype=False)
        assert np.isfinite(df[["UMAP1", "UMAP2"]].to_numpy()).all()


def test_mgt_umap_dfs_are_deterministic_except_embedding(pipeline):
    for dfpath in pipeline["mgt_dfs"]:
        df = pd.read_csv(dfpath, sep="\t", index_col=0)
        assert len(df) == N_GENOMES
        assert list(df["sample"]) == list(pipeline["sampledf"]["sample"])
        assert list(df["color"]) == list(pipeline["sampledf"]["color"])
        assert np.isfinite(df[["UMAP1", "UMAP2"]].to_numpy()).all()


def test_mlt_embedding_recovers_alg_clusters(pipeline):
    """
    Coarse, embedding-tolerant structure check: loci of the unfused ALGs C and
    D must sit closer to their own ALG than to other loci. (A and B are
    deliberately entangled by the simulated fusion, so they are not checked.)
    """
    locus_to_alg = _alg_of_locus()
    for dfpath in pipeline["mlt_dfs"]:
        df = pd.read_csv(dfpath, sep="\t", index_col=0)
        xy = df[["UMAP1", "UMAP2"]].to_numpy()
        algs = np.array([locus_to_alg[n] for n in df["rbh"]])
        for alg in ["CanidaeSimALG_C", "CanidaeSimALG_D"]:
            inside = xy[algs == alg]
            outside = xy[algs != alg]
            intra = np.median(
                [np.hypot(*(p - q)) for i, p in enumerate(inside) for q in inside[i + 1:]]
            )
            inter = np.median([np.hypot(*(p - q)) for p in inside for q in outside])
            assert intra < inter, f"{dfpath.name}: {alg} intra {intra} !< inter {inter}"


def test_paramsweep_pdfs_are_legal_with_expected_dot_counts(pipeline):
    for pdf, n_points, sweep in [
        (pipeline["mltpdf"], N_LOCI, MLT_SWEEP),
        (pipeline["mgtpdf"], N_GENOMES, MGT_SWEEP),
    ]:
        raw = pdf.read_bytes()
        assert raw.startswith(b"%PDF-"), f"{pdf.name} lacks a PDF header"
        assert b"%%EOF" in raw[-1024:], f"{pdf.name} lacks a PDF trailer"
        assert _count_pdf_dots(pdf) == n_points * len(sweep)
