from __future__ import annotations

import gzip
from pathlib import Path

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
