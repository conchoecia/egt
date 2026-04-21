from __future__ import annotations

import sys

import pandas as pd
import pytest

from egt import perspchrom_df_to_tree as pdt


def test_coloc_array_add_matrix_and_save_roundtrip(tmp_path):
    arr = pdt.coloc_array(abs_bin_size=10, frac_bin_size=0.1)
    assert arr._size_to_bin(23) == 20
    assert arr._frac_to_bin(0.34) == pytest.approx(0.3)

    coloc_df = pd.DataFrame(
        {
            "thisedge": [("p", "c")],
            "coloc0_size": [12],
            "coloc1_size": [27],
            "coloc0_CC_size": [20],
            "coloc1_CC_size": [40],
            "coloc0_percent_of_largest": [0.4],
            "coloc1_percent_of_largest": [0.6],
            "coloc0_CC_percent_of_largest": [0.5],
            "coloc1_CC_percent_of_largest": [0.9],
        }
    )
    arr.add_matrix(coloc_df, "observed")
    arr.add_matrix(coloc_df, "expected")
    assert arr.observed_matrix_size_abs[("p", "c")][(10, 20)] == 1
    assert arr.expected_matrix_frac_CC[("p", "c")][(0.5, 0.9)] == 1

    alg_df = pd.DataFrame({"thisedge": [("p", "c")], "thiscolor": [("red", "blue")]})
    arr.add_matrix_ALGs(alg_df, "observed")
    arr.add_matrix_ALGs(alg_df, "expected")
    assert arr.observed_matrix_ALG[("p", "c")][("blue", "red")] == 2

    outfile = tmp_path / "obs_expected.tsv"
    arr.save_obs_expected_file(outfile)
    written = pd.read_csv(outfile, sep="\t")
    assert set(written["ALG_num"]) == {"num", "ALG"}


def test_coloc_array_plotmatrix_listoffiles_to_plotmatrix(tmp_path):
    df1 = pd.DataFrame(
        {
            "ALG_num": ["num", "num", "ALG", "ALG"],
            "branch": ["b1", "b1", "b1", "b1"],
            "bin": ["x", "x", "x", "x"],
            "ob_ex": ["observed", "expected", "observed", "expected"],
            "size_frac": ["size", "size", "size", "size"],
            "abs_CC": ["abs", "abs", "abs", "abs"],
            "counts": [2, 1, 3, 3],
            "obs_count": [1, 1, 1, 1],
        }
    )
    df2 = pd.DataFrame(
        {
            "ALG_num": ["num", "num", "ALG", "ALG"],
            "branch": ["b1", "b1", "b1", "b1"],
            "bin": ["x", "x", "x", "x"],
            "ob_ex": ["observed", "expected", "observed", "expected"],
            "size_frac": ["frac", "frac", "size", "size"],
            "abs_CC": ["CC", "CC", "abs", "abs"],
            "counts": [5, 4, 1, 1],
            "obs_count": [2, 2, 2, 2],
        }
    )
    f1 = tmp_path / "one.tsv"
    f2 = tmp_path / "two.tsv"
    df1.to_csv(f1, sep="\t", index=False)
    df2.to_csv(f2, sep="\t", index=False)

    arr = pdt.coloc_array()
    arr.plotmatrix_listoffiles_to_plotmatrix([str(f1), str(f2)])
    assert arr.num_observed_observations == 3
    assert arr.num_expected_observations == 3
    assert set(arr.plotmatrix_sumdf["size_frac"]) == {"size", "frac"}


def test_coloc_array_validation_and_error_paths(tmp_path):
    with pytest.raises(Exception, match="abs_bin_size must be an integer"):
        pdt.coloc_array(abs_bin_size=1.5)
    with pytest.raises(Exception, match="frac_bin_size must be a float"):
        pdt.coloc_array(frac_bin_size=1)

    arr = pdt.coloc_array(abs_bin_size=10, frac_bin_size=0.1)
    with pytest.raises(Exception, match="ooe must be either observed or expected"):
        arr.add_matrix(pd.DataFrame(), "bad")
    with pytest.raises(Exception, match="ooe must be either 'observed' or 'expected'"):
        arr.add_matrix_ALGs(pd.DataFrame(), "bad")

    nested = tmp_path / "missing" / "obs.tsv"
    with pytest.raises(Exception, match="Directory does not exist"):
        arr.save_obs_expected_file(nested)

    outfile = tmp_path / "obs.tsv"
    outfile.write_text("taken\n")
    with pytest.raises(Exception, match="File already exists"):
        arr.save_obs_expected_file(outfile)

    bad_counts = pdt.coloc_array()
    bad_counts.num_observed_observations = 1
    bad_counts.num_expected_observations = 0
    with pytest.raises(Exception, match="must equal"):
        bad_counts.save_obs_expected_file(tmp_path / "bad_counts.tsv")

    bad_alg_counts = pdt.coloc_array()
    bad_alg_counts.num_observed_observations = 0
    bad_alg_counts.num_expected_observations = 0
    bad_alg_counts.num_observed_observations_ALGs = 1
    bad_alg_counts.num_expected_observations_ALGs = 0
    with pytest.raises(Exception, match="must equal"):
        bad_alg_counts.save_obs_expected_file(tmp_path / "bad_alg_counts.tsv")


def test_coloc_array_plotmatrix_error_paths(tmp_path):
    arr = pdt.coloc_array()
    with pytest.raises(Exception, match="File does not exist"):
        arr.plotmatrix_listoffiles_to_plotmatrix([str(tmp_path / "missing.tsv")])

    empty = tmp_path / "empty.tsv"
    pd.DataFrame(columns=["ALG_num", "branch", "bin", "ob_ex", "size_frac", "abs_CC", "counts", "obs_count"]).to_csv(
        empty, sep="\t", index=False
    )
    with pytest.raises(Exception, match="length of the observed_obs_count is not 1"):
        arr.plotmatrix_listoffiles_to_plotmatrix([str(empty)])

    wrong = tmp_path / "wrong.tsv"
    pd.DataFrame(
        {
            "ALG_num": ["bad", "bad"],
            "branch": ["b1", "b1"],
            "bin": ["x", "x"],
            "ob_ex": ["observed", "expected"],
            "size_frac": ["size", "size"],
            "abs_CC": ["abs", "abs"],
            "counts": [1, 1],
            "obs_count": [1, 1],
        }
    ).to_csv(wrong, sep="\t", index=False)
    with pytest.raises(Exception, match="ALG_num"):
        arr.plotmatrix_listoffiles_to_plotmatrix([str(wrong)])


def test_run_n_simulations_and_generate_stats_df(tmp_path, monkeypatch):
    sampledf = tmp_path / "samples.tsv"
    sampledf.write_text("species\ttaxidstring\tchangestrings\nsp1\t1;2\t1-([]|[]|[])-2\n")
    algdf = tmp_path / "alg.rbh"
    algdf.write_text("placeholder\n")

    monkeypatch.setattr(
        pdt.rbh_tools,
        "parse_ALG_rbh_to_colordf",
        lambda _path: pd.DataFrame({"ALGname": ["A"], "Color": ["#aa0000"], "Size": [5]}),
    )

    observed_coloc = pd.DataFrame(
        {
            "thisedge": [("p", "c")],
            "coloc0_size": [12],
            "coloc1_size": [24],
            "coloc0_CC_size": [30],
            "coloc1_CC_size": [40],
            "coloc0_percent_of_largest": [0.4],
            "coloc1_percent_of_largest": [0.8],
            "coloc0_CC_percent_of_largest": [0.5],
            "coloc1_CC_percent_of_largest": [0.9],
        }
    )
    observed_alg = pd.DataFrame({"thisedge": [("p", "c")], "thiscolor": [("red", "blue")]})
    monkeypatch.setattr(
        pdt,
        "stats_df_to_loss_fusion_dfs",
        lambda *args, **kwargs: (pd.DataFrame(), observed_coloc.copy(), observed_alg.copy()),
    )
    monkeypatch.setattr(pdt.random, "randint", lambda *_args: 7)

    outfile = tmp_path / "sim.tsv"
    assert pdt.run_n_simulations_save_results(str(sampledf), str(algdf), str(outfile), num_sims=2) == 0
    written = pd.read_csv(outfile, sep="\t")
    assert set(written["ALG_num"]) == {"num", "ALG"}
    assert set(written["ob_ex"]) == {"observed", "expected"}

    stats_out = tmp_path / "stats.tsv"
    monkeypatch.setattr(pdt, "parse_gain_loss_from_perspchrom_df", lambda _df: pd.DataFrame({"target_taxid": [2], "change": ["A"]}))
    monkeypatch.setattr(pdt, "stats_on_changedf", lambda _sampledf, _changedf: pd.DataFrame({"target_taxid": [2], "change": ["A"], "counts": [1]}))
    assert pdt.generate_stats_df(str(sampledf), str(stats_out)) == 0
    assert stats_out.exists()

    with pytest.raises(Exception, match="File does not exist"):
        pdt.generate_stats_df(str(tmp_path / "missing.tsv"), str(stats_out))
    with pytest.raises(Exception, match="Directory does not exist"):
        pdt.generate_stats_df(str(sampledf), str(tmp_path / "missing" / "stats.tsv"))


def test_unit_test_coloc_array_identical(tmp_path, monkeypatch):
    sampledf = tmp_path / "samples.tsv"
    sampledf.write_text("species\ttaxidstring\tchangestrings\nsp1\t1;2\t1-([]|[]|[])-2\n")
    algdf = tmp_path / "alg.rbh"
    algdf.write_text("placeholder\n")

    same_disp = pd.DataFrame({"thisedge": [(1, 2)]})
    same_coloc = pd.DataFrame({"thisedge": [(1, 2)], "thiscoloc": [("A", "B")]})
    same_alg = pd.DataFrame({"x": [1]})
    monkeypatch.setattr(
        pdt.rbh_tools,
        "parse_ALG_rbh_to_colordf",
        lambda _path: pd.DataFrame({"ALGname": ["A"], "Color": ["#aa0000"], "Size": [1]}),
    )
    monkeypatch.setattr(
        pdt,
        "stats_df_to_loss_fusion_dfs",
        lambda *args, **kwargs: (same_disp.copy(), same_coloc.copy(), same_alg.copy()),
    )
    monkeypatch.setattr(sys, "argv", ["prog", str(sampledf), str(algdf)])

    assert pdt.unit_test_coloc_array_identical() is None
