from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix, load_npz, save_npz

from egt import phylotreeumap as ptu


class FakeNCBI:
    def get_lineage(self, taxid):
        return [1, int(taxid)]

    def get_taxid_translator(self, taxids):
        mapping = {1: "root"}
        mapping.update({int(t): f"Taxon{t}" for t in taxids})
        return {int(k): v for k, v in mapping.items() if int(k) in [int(x) for x in taxids]}


def _write_gbgz(path: Path, rows: list[tuple[str, str, int]]) -> Path:
    df = pd.DataFrame(rows, columns=["rbh1", "rbh2", "distance"])
    with gzip.open(path, "wt") as fh:
        df.to_csv(fh, sep="\t", index=False)
    return path


def test_misc_helpers_and_taxdict(tmp_path: Path):
    class Mapper:
        embedding_ = [[0.0, 1.0], [2.0, 3.0]]

    merged = ptu.umap_mapper_to_df(Mapper(), pd.DataFrame({"sample": ["a", "b"]}))
    assert list(merged.columns) == ["sample", "UMAP1", "UMAP2"]
    assert ptu.get_text_color("#ffffff") == "#000000"
    assert ptu.get_text_color("#000000") == "#FFFFFF"

    nested = tmp_path / "a" / "b" / "out.tsv"
    ptu.create_directories_if_not_exist(str(nested))
    assert (tmp_path / "a" / "b").is_dir()

    taxdict = ptu.NCBI_taxid_to_taxdict(FakeNCBI(), 123)
    assert taxdict["taxname"] == "Taxon123"
    assert taxdict["taxid_list_str"] == "1;123"


def test_rbh_to_gb_and_distance_wrapper(monkeypatch, tmp_path: Path):
    sample = "Species-123-GCA1"
    rbhdf = pd.DataFrame(
        {
            "rbh": ["fam1", "fam2", "fam3"],
            f"{sample}_scaf": ["chr1", "chr1", "chr2"],
            f"{sample}_pos": [10, 40, 5],
            "ALG_scaf": ["a", "a", "b"],
            "ALG_gene": ["g1", "g2", "g3"],
            "ALG_pos": [1, 2, 3],
            f"{sample}_gene": ["x1", "x2", "x3"],
        }
    )

    outfile = tmp_path / "dist.gb.gz"
    ptu.rbh_to_gb(sample, rbhdf, outfile)
    written = pd.read_csv(outfile, sep="\t", compression="gzip")
    assert list(written["distance"]) == [30]

    rbhfile = tmp_path / f"ALG_{sample}_xy_reciprocal_best_hits.plotted.rbh"
    rbhfile.write_text("placeholder\n")
    monkeypatch.setattr(ptu.rbh_tools, "parse_rbh", lambda _path: rbhdf)
    wrapped_out = tmp_path / "wrapped.gb.gz"
    ptu.rbh_to_distance_gbgz(str(rbhfile), str(wrapped_out), "ALG")
    assert wrapped_out.exists()


def test_sample_matrix_helpers(monkeypatch, tmp_path: Path):
    sample = "Species-123-GCA1"
    rbhfile = tmp_path / f"{sample}.rbh"
    rbhfile.write_text("placeholder\n")
    gbgz = _write_gbgz(tmp_path / f"{sample}.gb.gz", [("fam1", "fam2", 9)])

    rbhdf = pd.DataFrame(
        {
            "rbh": ["fam1", "fam2"],
            "ALG_scaf": ["alg_chr", "alg_chr"],
            "ALG_gene": ["alg1", "alg2"],
            "ALG_pos": [1, 2],
            f"{sample}_scaf": ["chr1", "chr2"],
            f"{sample}_gene": ["gene1", "gene2"],
            f"{sample}_pos": [100, 200],
        }
    )

    monkeypatch.setattr(ptu, "NCBITaxa", lambda: FakeNCBI())
    monkeypatch.setattr(ptu.rbh_tools, "parse_rbh", lambda _path: rbhdf)

    sampledf = ptu.sampleToRbhFileDict_to_sample_matrix(
        {sample: str(rbhfile)},
        "ALG",
        str(tmp_path),
        str(tmp_path / "sampledf.tsv"),
    )
    assert list(sampledf["sample"]) == [sample]
    assert sampledf.loc[0, "number_of_chromosomes"] == 2

    matrix = ptu.construct_lil_matrix_from_sampledf(
        pd.DataFrame({"dis_filepath_abs": [str(gbgz)]}),
        {("fam1", "fam2"): 0},
    )
    assert matrix.shape == (1, 1)
    assert matrix.toarray()[0, 0] == 9


def test_sample_matrix_helper_validation_errors(monkeypatch, tmp_path: Path):
    sample = "Species-123-GCA1"
    rbhfile = tmp_path / f"{sample}.rbh"
    rbhfile.write_text("placeholder\n")

    bad_columns = pd.DataFrame(
        {
            "rbh": ["fam1"],
            "ALG_scaf": ["alg_chr"],
            "ALG_gene": ["alg1"],
            f"{sample}_scaf": ["chr1"],
            f"{sample}_gene": ["gene1"],
            f"{sample}_pos": [100],
        }
    )
    monkeypatch.setattr(ptu, "NCBITaxa", lambda: FakeNCBI())
    monkeypatch.setattr(ptu.rbh_tools, "parse_rbh", lambda _path: bad_columns)

    with pytest.raises(IOError, match="ALG_pos"):
        ptu.sampleToRbhFileDict_to_sample_matrix(
            {sample: str(rbhfile)},
            "ALG",
            str(tmp_path),
            str(tmp_path / "sampledf.tsv"),
        )

    mismatched = pd.DataFrame(
        {
            "rbh": ["fam1"],
            "ALG_scaf": ["alg_chr"],
            "ALG_gene": ["alg1"],
            "ALG_pos": [1],
            "Other-123-GCA1_scaf": ["chr1"],
            "Other-123-GCA1_gene": ["gene1"],
            "Other-123-GCA1_pos": [100],
        }
    )
    monkeypatch.setattr(ptu.rbh_tools, "parse_rbh", lambda _path: mismatched)

    with pytest.raises(ValueError, match="is not the same as the key"):
        ptu.sampleToRbhFileDict_to_sample_matrix(
            {sample: str(rbhfile)},
            "ALG",
            str(tmp_path),
            str(tmp_path / "sampledf.tsv"),
        )


def test_rbh_directory_to_distance_matrix_builds_sampledf(monkeypatch, tmp_path: Path):
    rbh_dir = tmp_path / "rbhs"
    rbh_dir.mkdir()
    rbhfile = rbh_dir / "Species-123-GCA1.rbh"
    rbhfile.write_text("placeholder\n")

    rbhdf = pd.DataFrame(
        {
            "rbh": ["fam1", "fam2"],
            "ALG_scaf": ["alg_chr", "alg_chr"],
            "ALG_gene": ["alg1", "alg2"],
            "ALG_pos": [1, 2],
            "Species-123-GCA1_scaf": ["chr1", "chr2"],
            "Species-123-GCA1_gene": ["gene1", "gene2"],
            "Species-123-GCA1_pos": [100, 200],
        }
    )

    monkeypatch.setattr(ptu, "NCBITaxa", lambda: FakeNCBI())
    monkeypatch.setattr(ptu.rbh_tools, "parse_rbh", lambda _path: rbhdf)
    monkeypatch.setattr(ptu, "rbh_to_gb", lambda sample, df, outfile: Path(outfile).write_text("ok\n"))

    outtsv = tmp_path / "GTUMAP" / "sampledf.tsv"
    outputdir = tmp_path / "GTUMAP" / "distance_matrices"
    sampledf = ptu.rbh_directory_to_distance_matrix(str(rbh_dir), "ALG", outtsv=str(outtsv), outputdir=str(outputdir))

    assert list(sampledf["sample"]) == ["Species-123-GCA1"]
    assert sampledf.loc[0, "number_of_chromosomes"] == 2
    assert Path(sampledf.loc[0, "dis_filepath"]).exists()
    assert outtsv.exists()


def test_command_wrappers_and_main_dispatch(monkeypatch, tmp_path: Path):
    calls = []

    monkeypatch.setattr(ptu, "rbh_directory_to_distance_matrix", lambda **kwargs: calls.append(("build", kwargs)))
    monkeypatch.setattr(ptu, "ALGrbh_to_algcomboix", lambda _path: {("fam1", "fam2"): 0})
    monkeypatch.setattr(
        ptu,
        "construct_coo_matrix_from_sampledf",
        lambda *args, **kwargs: type("Coo", (), {"shape": (1, 2), "nnz": 2})(),
    )
    monkeypatch.setattr(ptu, "save_npz", lambda output, coo: Path(output).write_text("npz\n"))
    monkeypatch.setattr(ptu, "plot_umap_from_files", lambda **kwargs: calls.append(("odog", kwargs)))
    monkeypatch.setattr(ptu, "mgt_mlt_umap", lambda **kwargs: calls.append(("mgt", kwargs)))
    monkeypatch.setattr(ptu, "mlt_umapHTML", lambda **kwargs: calls.append(("mlt", kwargs)))
    monkeypatch.setattr(ptu, "mgt_mlt_plot_HTML", lambda **kwargs: calls.append(("html", kwargs)))

    sampledf = tmp_path / "sampledf.tsv"
    pd.DataFrame({"sample": ["s1"], "dis_filepath_abs": ["/tmp/fake.gb.gz"]}).to_csv(sampledf, sep="\t")
    algcombo = tmp_path / "algcombo.tsv"
    algcombo.write_text("('fam1', 'fam2')\t0\n")
    umap_df = tmp_path / "umap.tsv"
    pd.DataFrame({"UMAP1": [0.0], "UMAP2": [1.0]}).to_csv(umap_df, sep="\t")

    assert ptu.main(["build-distances", "--rbh-dir", str(tmp_path), "--alg-name", "ALG"]) == 0
    assert ptu.main(["algcomboix", "--alg-rbh", str(algcombo), "--output", str(tmp_path / "combo_out.tsv")]) == 0
    assert (
        ptu.main(
            [
                "combine-distances",
                "--sampledf",
                str(sampledf),
                "--algcomboix",
                str(algcombo),
                "--output",
                str(tmp_path / "combined.npz"),
                "--no-check-paths",
            ]
        )
        == 0
    )
    assert (
        ptu.main(
            [
                "odog-umap",
                "--sampledf",
                str(sampledf),
                "--algcomboix",
                str(algcombo),
                "--coo",
                str(tmp_path / "coo.npz"),
                "--sample",
                "sample",
                "--nan-mode",
                "small",
                "--n-neighbors",
                "5",
                "--min-dist",
                "0.1",
                "--df-out",
                str(tmp_path / "df.tsv"),
                "--html-out",
                str(tmp_path / "plot.html"),
            ]
        )
        == 0
    )
    assert (
        ptu.main(
            [
                "mgt-mlt-umap",
                "--sampledf",
                str(sampledf),
                "--locus-file",
                str(algcombo),
                "--coo",
                str(tmp_path / "coo.npz"),
                "--nan-mode",
                "large",
                "--n-neighbors",
                "5",
                "--min-dist",
                "0.1",
                "--df-out",
                str(tmp_path / "mgt.tsv"),
            ]
        )
        == 0
    )
    assert (
        ptu.main(
            [
                "mlt-html",
                "--sample",
                "sample",
                "--sampledf",
                str(sampledf),
                "--alg-rbh",
                str(algcombo),
                "--coo",
                str(tmp_path / "coo.npz"),
                "--nan-mode",
                "small",
                "--n-neighbors",
                "5",
                "--min-dist",
                "0.1",
                "--df-out",
                str(tmp_path / "mlt.tsv"),
                "--html-out",
                str(tmp_path / "mlt.html"),
            ]
        )
        == 0
    )
    assert (
        ptu.main(
            [
                "plot-html",
                "--umap-df",
                str(umap_df),
                "--html-out",
                str(tmp_path / "render.html"),
                "--analysis-type",
                "MLT",
                "--tree-height",
                "333",
            ]
        )
        == 0
    )

    assert [name for name, _ in calls] == ["build", "odog", "mgt", "mlt", "html"]
    assert (tmp_path / "combo_out.tsv").exists()
    assert (tmp_path / "combined.npz").exists()
    assert calls[-1][1]["tree_newick"] is None
    assert calls[-1][1]["tree_palette"] is None
    assert calls[-1][1]["tree_height"] == 333


def test_construct_lil_matrix_and_rbh_to_samplename_error_paths(tmp_path: Path):
    sampledf = pd.DataFrame({"dis_filepath_abs": [str(tmp_path / "missing.gz")]}, index=[2])
    with pytest.raises(ValueError, match="maximum index"):
        ptu.construct_lil_matrix_from_sampledf(sampledf, {("fam1", "fam2"): 0})

    good = _write_gbgz(tmp_path / "one.gb.gz", [("fam1", "fam2", 10)])
    broken = pd.DataFrame({"dis_filepath_abs": [str(good)]}, index=[0])
    with pytest.raises(ValueError, match="missing from alg_combo_to_ix"):
        ptu.construct_lil_matrix_from_sampledf(broken, {("fam2", "fam1"): 0})

    with pytest.raises(ValueError, match="does not start with the ALGname"):
        ptu.rbh_to_samplename("badname.rbh", "ALG")
    with pytest.raises(ValueError, match="does not have three fields"):
        ptu.rbh_to_samplename("ALG_onlytwofields-123_xy_reciprocal_best_hits.plotted.rbh", "ALG")
    with pytest.raises(ValueError, match="non-numeric character"):
        ptu.rbh_to_samplename("ALG_species-bad-GCA1_xy_reciprocal_best_hits.plotted.rbh", "ALG")


def test_topoumap_genmatrix_validation_and_outputs(monkeypatch, tmp_path: Path):
    sampledf = tmp_path / "sampledf.tsv"
    pd.DataFrame(
        {
            "sample": ["s1", "s2"],
            "taxid_list": ["[1, 10]", "[1, 20]"],
        }
    ).to_csv(sampledf, sep="\t")

    combo = tmp_path / "combo.tsv"
    combo.write_text("('fam1', 'fam2')\t0\n")
    coofile = tmp_path / "coo.npz"
    save_npz(coofile, csr_matrix([[2.0], [4.0]]))
    rbhfile = tmp_path / "alg.rbh"
    rbhfile.write_text("placeholder\n")

    monkeypatch.setattr(
        ptu.rbh_tools,
        "parse_rbh",
        lambda _path: pd.DataFrame({"rbh": ["fam1", "fam2"]}),
    )

    with pytest.raises(ValueError, match="does not end with '.npz'"):
        ptu.topoumap_genmatrix(str(sampledf), str(combo), str(coofile), str(rbhfile), "sample", [1], [], "bad.tsv", str(tmp_path / "out.tsv"), "small")
    with pytest.raises(ValueError, match="does not end with '.tsv' or '.df'"):
        ptu.topoumap_genmatrix(str(sampledf), str(combo), str(coofile), str(rbhfile), "sample", [1], [], str(tmp_path / "out.npz"), str(tmp_path / "out.txt"), "small")
    with pytest.raises(ValueError, match="method bad"):
        ptu.topoumap_genmatrix(str(sampledf), str(combo), str(coofile), str(rbhfile), "sample", [1], [], str(tmp_path / "out.npz"), str(tmp_path / "out.tsv"), "small", method="bad")
    with pytest.raises(ValueError, match="missing_values bad"):
        ptu.topoumap_genmatrix(str(sampledf), str(combo), str(coofile), str(rbhfile), "sample", [1], [], str(tmp_path / "out.npz"), str(tmp_path / "out.tsv"), "bad")
    with pytest.raises(ValueError, match="not of type int"):
        ptu.topoumap_genmatrix(str(sampledf), str(combo), str(coofile), str(rbhfile), "sample", [1], [], str(tmp_path / "out.npz"), str(tmp_path / "out.tsv"), "small", missing_value_as=1.5)
    with pytest.raises(ValueError, match="not a list"):
        ptu.topoumap_genmatrix(str(sampledf), str(combo), str(coofile), str(rbhfile), "sample", "nope", [], str(tmp_path / "out.npz"), str(tmp_path / "out.tsv"), "small")
    with pytest.raises(ValueError, match="not an integer"):
        ptu.topoumap_genmatrix(str(sampledf), str(combo), str(coofile), str(rbhfile), "sample", ["x"], [], str(tmp_path / "out.npz"), str(tmp_path / "out.tsv"), "small")
    with pytest.raises(ValueError, match="There are no samples to process"):
        ptu.topoumap_genmatrix(str(sampledf), str(combo), str(coofile), str(rbhfile), "sample", [999], [], str(tmp_path / "out.npz"), str(tmp_path / "out.tsv"), "small")

    outcoo = tmp_path / "topo_mean.npz"
    outsample = tmp_path / "topo_mean.tsv"
    assert (
        ptu.topoumap_genmatrix(
            str(sampledf),
            str(combo),
            str(coofile),
            str(rbhfile),
            "sample",
            [1],
            [],
            str(outcoo),
            str(outsample),
            "small",
            method="mean",
        )
        is None
    )
    saved = load_npz(outcoo).toarray()
    assert saved[0, 1] == pytest.approx(3.0)
    assert outsample.exists()

    phylo_out = tmp_path / "topo_phylo.npz"
    phylo_sample = tmp_path / "topo_phylo.tsv"
    assert (
        ptu.topoumap_genmatrix(
            str(sampledf),
            str(combo),
            str(coofile),
            str(rbhfile),
            "sample",
            [1],
            [],
            str(phylo_out),
            str(phylo_sample),
            "small",
            method="phylogenetic",
        )
        is None
    )
    assert load_npz(phylo_out).toarray()[0, 1] > 0


def test_topoumap_mean_filters_coo_rows_by_position_not_dataframe_label(monkeypatch, tmp_path: Path):
    """A non-RangeIndex must not make excluded genomes leak into the mean."""
    sampledf = tmp_path / "sampledf.tsv"
    pd.DataFrame(
        {
            "sample": ["kept_a", "excluded", "kept_b"],
            "taxid_list": ["[1, 10]", "[1, 20]", "[1, 10]"],
        },
        index=[10, 20, 30],
    ).to_csv(sampledf, sep="\t")

    combo = tmp_path / "combo.tsv"
    combo.write_text("('fam1', 'fam2')\t0\n")
    coofile = tmp_path / "coo.npz"
    # The excluded middle row is deliberately extreme. The correct unweighted
    # mean of the retained positional rows is (2 + 4) / 2 = 3, not 335.33.
    save_npz(coofile, csr_matrix([[2.0], [1000.0], [4.0]]))
    rbhfile = tmp_path / "alg.rbh"
    rbhfile.write_text("placeholder\n")
    monkeypatch.setattr(
        ptu.rbh_tools,
        "parse_rbh",
        lambda _path: pd.DataFrame({"rbh": ["fam1", "fam2"]}),
    )

    outcoo = tmp_path / "filtered_mean.npz"
    outsample = tmp_path / "filtered_mean.tsv"
    ptu.topoumap_genmatrix(
        str(sampledf),
        str(combo),
        str(coofile),
        str(rbhfile),
        "sample",
        [10],
        [],
        str(outcoo),
        str(outsample),
        "small",
        method="mean",
    )

    result = load_npz(outcoo).toarray()
    assert result[0, 1] == pytest.approx(3.0)
    kept = pd.read_csv(outsample, sep="\t", index_col=0)
    assert list(kept.index) == [10, 30]
    assert list(kept["sample"]) == ["kept_a", "kept_b"]


def _stub_umap(monkeypatch):
    """Replace umap.UMAP with a stub so mgt_mlt_umap tests are fast and deterministic."""

    fitted_matrices = []

    class _StubMapper:
        def __init__(self, n_points):
            self.embedding_ = np.zeros((n_points, 2))

    class _StubUMAP:
        def __init__(self, **_kwargs):
            pass

        def fit(self, matrix):
            dense = matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)
            fitted_matrices.append(np.array(dense, copy=True))
            return _StubMapper(matrix.shape[0])

    monkeypatch.setattr(ptu.umap, "UMAP", _StubUMAP)
    return fitted_matrices


def _write_alg_rbh(path: Path, n_loci: int) -> pd.DataFrame:
    rbhdf = pd.DataFrame(
        {
            "rbh": [f"TestALG_genefamily_{i}" for i in range(n_loci)],
            "gene_group": [f"G{i % 2}" for i in range(n_loci)],
            "color": ["#000000"] * n_loci,
            "sp1_scaf": ["scaf1"] * n_loci,
            "sp1_gene": [f"g1.{i}" for i in range(n_loci)],
            "sp1_pos": list(range(1, n_loci + 1)),
            "sp2_scaf": ["scaf1"] * n_loci,
            "sp2_gene": [f"g2.{i}" for i in range(n_loci)],
            "sp2_pos": list(range(1, n_loci + 1)),
        }
    )
    rbhdf.to_csv(path, sep="\t", index=False)
    return rbhdf


def test_locus_file_analysis_type(tmp_path: Path):
    combo = tmp_path / "combo.tsv"
    combo.write_text("('fam1', 'fam2')\t0\n('fam1', 'fam3')\t1\n")
    assert ptu.locus_file_analysis_type(str(combo)) == "MGT"

    rbhfile = tmp_path / "alg.rbh"
    _write_alg_rbh(rbhfile, 4)
    assert ptu.locus_file_analysis_type(str(rbhfile)) == "MLT"

    # ``rbh`` and ``gene_group`` are sufficient for parse_rbh(), so column
    # count alone cannot distinguish this valid minimal RBH from a combo file.
    minimal_rbh = tmp_path / "minimal.rbh"
    minimal_rbh.write_text("rbh\tgene_group\nfam1\tA\nfam2\tB\n")
    assert ptu.locus_file_analysis_type(str(minimal_rbh)) == "MLT"

    malformed_combo = tmp_path / "malformed_combo.tsv"
    malformed_combo.write_text("['fam1', 'fam2']\t0\n")
    with pytest.raises(ValueError, match="neither an ALG RBH file"):
        ptu.locus_file_analysis_type(str(malformed_combo))
    with pytest.raises(ValueError, match="tuple of exactly two RBH names"):
        ptu.algcomboix_file_to_dict(str(malformed_combo))

    empty = tmp_path / "empty.tsv"
    empty.write_text("\n\n")
    with pytest.raises(ValueError, match="is empty"):
        ptu.locus_file_analysis_type(str(empty))

    with pytest.raises(IOError, match="does not exist"):
        ptu.locus_file_analysis_type(str(tmp_path / "missing.tsv"))


def test_mgt_mlt_umap_mgt_mode_accepts_small_and_large_strings(monkeypatch, tmp_path: Path):
    # Regression test: the CLI passes --nan-mode "small"/"large", which used to crash
    #  in int(smalllargeNaN) before the UMAP was ever calculated.
    _stub_umap(monkeypatch)
    n_samples, n_pairs = 5, 6
    sampledf = tmp_path / "sampledf.tsv"
    pd.DataFrame(
        {"sample": [f"sp{i}" for i in range(n_samples)], "taxid_list": ["[1, 10]"] * n_samples}
    ).to_csv(sampledf, sep="\t")
    combo = tmp_path / "combo.tsv"
    combo.write_text("".join(f"('fam{i}', 'fam{i+1}')\t{i}\n" for i in range(n_pairs)))
    coofile = tmp_path / "mgt.coo.npz"
    save_npz(coofile, csr_matrix(np.arange(n_samples * n_pairs, dtype=float).reshape(n_samples, n_pairs)))

    for nan_mode in ["small", "large"]:
        out = tmp_path / f"mgt_{nan_mode}.df"
        assert ptu.mgt_mlt_umap(str(sampledf), str(combo), str(coofile), nan_mode, 2, 0.1, str(out)) == 0
        result = pd.read_csv(out, sep="\t", index_col=0)
        assert len(result) == n_samples
        assert {"sample", "UMAP1", "UMAP2"}.issubset(result.columns)


def test_mgt_mlt_umap_mlt_mode_uses_rbh_locus_file(monkeypatch, tmp_path: Path):
    # Regression test: MLT mode passes the multi-column ALG rbh file as the LocusFile,
    #  which used to crash in algcomboix_file_to_dict with
    #  "ValueError: too many values to unpack (expected 2)".
    fitted_matrices = _stub_umap(monkeypatch)
    n_loci = 8
    sampledf = tmp_path / "sampledf.tsv"
    pd.DataFrame({"sample": ["sp0", "sp1"], "taxid_list": ["[1, 10]"] * 2}).to_csv(sampledf, sep="\t")
    rbhfile = tmp_path / "alg.rbh"
    _write_alg_rbh(rbhfile, n_loci)
    dense = np.arange(n_loci * n_loci, dtype=float).reshape(n_loci, n_loci)
    dense = dense + dense.T
    np.fill_diagonal(dense, 0)
    # A zero off the diagonal represents another structurally missing pair.
    dense[0, 3] = dense[3, 0] = 0
    coofile = tmp_path / "mlt.coo.npz"
    save_npz(coofile, csr_matrix(dense))

    missing_sentinel = 123456789
    for nan_mode in ["small", "large"]:
        out = tmp_path / f"mlt_{nan_mode}.df"
        assert ptu.mgt_mlt_umap(
            str(sampledf),
            str(rbhfile),
            str(coofile),
            nan_mode,
            2,
            0.1,
            str(out),
            missing_value_as=missing_sentinel,
        ) == 0
        result = pd.read_csv(out, sep="\t", index_col=0)
        # the points are loci, so the output rows carry the rbh metadata
        assert len(result) == n_loci
        assert {"rbh", "gene_group", "color", "UMAP1", "UMAP2"}.issubset(result.columns)
        assert list(result["rbh"]) == [f"TestALG_genefamily_{i}" for i in range(n_loci)]

        expected_umap_input = dense.copy()
        if nan_mode == "large":
            expected_umap_input[expected_umap_input == 0] = missing_sentinel
        np.testing.assert_array_equal(fitted_matrices[-1], expected_umap_input)
        # MLT contains N-choose-2 pairs of distinct loci. The diagonal is not
        # an observed self-distance; it follows the selected missing-data mode.
        expected_diagonal = 0 if nan_mode == "small" else missing_sentinel
        np.testing.assert_array_equal(
            np.diag(fitted_matrices[-1]),
            np.full(n_loci, expected_diagonal),
        )

    # a non-square matrix cannot be an MLT locus-by-locus matrix
    badcoo = tmp_path / "bad.coo.npz"
    save_npz(badcoo, csr_matrix(np.ones((n_loci, 3))))
    with pytest.raises(ValueError, match="does not match the number of loci"):
        ptu.mgt_mlt_umap(str(sampledf), str(rbhfile), str(badcoo), "small", 2, 0.1, str(tmp_path / "bad.df"))


def test_mlt_matrix_cli_dispatch(monkeypatch, tmp_path: Path):
    calls = []
    monkeypatch.setattr(ptu, "topoumap_genmatrix", lambda **kwargs: calls.append(kwargs))
    assert (
        ptu.main(
            [
                "mlt-matrix",
                "--sampledf", str(tmp_path / "sampledf.tsv"),
                "--algcomboix", str(tmp_path / "combo.tsv"),
                "--coo", str(tmp_path / "allsamples.coo.npz"),
                "--alg-rbh", str(tmp_path / "alg.rbh"),
                "--taxids-to-keep", "33208",
                "--taxids-to-remove", "7742", "6231",
                "--nan-mode", "small",
                "--coo-out", str(tmp_path / "mlt.coo.npz"),
                "--sampledf-out", str(tmp_path / "mlt.sampledf.tsv"),
            ]
        )
        == 0
    )
    assert len(calls) == 1
    kwargs = calls[0]
    assert kwargs["taxids_to_keep"] == [33208]
    assert kwargs["taxids_to_remove"] == [7742, 6231]
    assert kwargs["missing_values"] == "small"
    assert kwargs["method"] == "phylogenetic"
    assert kwargs["outcoofile"].endswith("mlt.coo.npz")
