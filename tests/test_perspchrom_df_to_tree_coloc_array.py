from __future__ import annotations

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
