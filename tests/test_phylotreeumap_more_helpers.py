from __future__ import annotations

import gzip
from pathlib import Path

import pandas as pd
import pytest

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
