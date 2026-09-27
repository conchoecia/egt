"""Order and missingness invariants for chromosome-local retained-marker ranks."""
import numpy as np
import pandas as pd
import pytest

from egt.gene_index import index_coordinates


def make():
    return pd.DataFrame({"rbh": ["c", "a", "b", "d"], "S_scaf": ["1", "1", "1", "2"],
                         "S_pos": [9000, 100, 150, 5], "aux": [0, 1, 2, 3]})


def ranks(df):
    return index_coordinates("S", df).set_index("rbh")["S_pos"].to_dict()


def test_spacing_and_coordinate_transforms_do_not_change_ranks():
    df = make(); original = df.copy(deep=True)
    assert ranks(df) == {"c": 3, "a": 1, "b": 2, "d": 1}
    df["S_pos"] = df["S_pos"] ** 2 + 700
    assert ranks(df) == ranks(original)
    assert index_coordinates("S", original)["aux"].tolist() == [0, 1, 2, 3]
    assert original.equals(make())


def test_reversal_preserves_within_chromosome_distances():
    df = make(); before = ranks(df)
    df["S_pos"] = -df["S_pos"]
    after = ranks(df)
    for a, b in [("a", "b"), ("a", "c"), ("b", "c")]:
        assert abs(before[a] - before[b]) == abs(after[a] - after[b])


def test_ties_and_numeric_strings_are_reproducible():
    df = make(); df["S_pos"] = ["10", "9", "10", "1"]
    assert ranks(df) == {"c": 3, "a": 1, "b": 2, "d": 1}
    assert ranks(df.sample(frac=1, random_state=5)) == ranks(df)


def test_missing_rows_do_not_occupy_rank_positions():
    df = make(); df.loc[2, "S_pos"] = np.nan; df.loc[3, "S_scaf"] = None
    assert ranks(df) == {"c": 2, "a": 1}


def test_repeated_dataframe_indices_are_preserved_without_row_expansion():
    df = make(); df.index = [7, 7, 2, 2]
    out = index_coordinates("S", df)
    assert out.index.tolist() == [7, 7, 2, 2]
    assert out["S_pos"].tolist() == [3, 1, 2, 1]


@pytest.mark.parametrize("bad", [True, 1j, "garbage", np.inf, -np.inf])
def test_invalid_coordinates_rejected(bad):
    df = make().astype({"S_pos": object}); df.loc[0, "S_pos"] = bad
    with pytest.raises(ValueError, match="finite real"):
        index_coordinates("S", df)


def test_duplicate_markers_rejected_even_across_chromosomes():
    df = make(); df.loc[3, "rbh"] = "a"
    with pytest.raises(ValueError, match="unique"):
        index_coordinates("S", df)


@pytest.mark.parametrize("column", ["rbh", "S_scaf"])
def test_blank_identifiers_rejected(column):
    df = make(); df.loc[0, column] = " "
    with pytest.raises(ValueError, match="nonempty strings"):
        index_coordinates("S", df)


@pytest.mark.parametrize("all_missing", [False, True])
def test_empty_rank_column_has_integer_type(all_missing):
    df = make() if all_missing else make().iloc[:0]
    if all_missing:df["S_pos"] = np.nan
    out = index_coordinates("S", df)
    assert out.empty and out["S_pos"].dtype == np.int64
