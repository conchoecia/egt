from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from egt import phylotreeumap_plotdfs as plotdfs


def test_generate_distinct_colors_and_benedictus_helpers():
    colors = plotdfs.generate_distinct_colors(4)
    assert len(colors) == 4
    assert all(color.startswith("#") and len(color) == 7 for color in colors)
    assert len(plotdfs.benedictus_n(5)) == 5


def test_parse_args_validates_conflicts_and_metadata(tmp_path: Path):
    df = tmp_path / "subsample_phylum.neighbors_15.mind_0.1.df"
    pd.DataFrame({"x": [1], "y": [2]}).to_csv(df, sep="\t")
    meta = tmp_path / "meta.tsv"
    meta.write_text("rbh\tflag\nr1\tTrue\n")

    args = plotdfs.parse_args(["-f", str(df), "-p", "out", "--metadata", str(meta), "--pdf"])
    assert args.metadata == [str(meta)]
    assert args.benedictus is False

    with pytest.raises(ValueError, match="Both directory and filelist"):
        plotdfs.parse_args(["-d", str(tmp_path), "-f", str(df), "-p", "out"])


def test_generate_df_dict_and_filename_parsing(tmp_path: Path):
    df_path = tmp_path / "subsample_phylum.neighbors_15.mind_0.1.missing_large.method_mean.euclidean.df"
    pd.DataFrame({"x": [1], "y": [2]}).to_csv(df_path, sep="\t")
    args = plotdfs.parse_args(["-f", str(df_path), "-p", "out"])

    parsed = plotdfs._parse_df_filename(str(df_path))
    assert parsed == (15, 0.1, "mean", "large", "euclidean", "subsample_phylum")

    df_dict = plotdfs.generate_df_dict(args)
    record = df_dict[(15, 0.1)]
    assert list(record["df"].columns) == ["x", "y"]
    assert record["metric"] == "euclidean"


def test_infer_rank_from_subsample_filename_accepts_tokens_and_rejects_unknown():
    assert plotdfs.infer_rank_from_subsample_filename("subsample_species_group.neighbors_15.mind_0.1.df") == "species group"
    with pytest.raises(ValueError, match="Unrecognized rank token"):
        plotdfs.infer_rank_from_subsample_filename("subsample_notarank.neighbors_15.mind_0.1.df")


def test_set_square_limits_and_auto_point_size():
    fig, ax = plt.subplots()
    plotdfs.set_square_limits(ax, [0, 2], [0, 1], q=None, pad=0.0)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    assert pytest.approx(xlim[1] - xlim[0]) == ylim[1] - ylim[0]
    assert plotdfs.auto_point_size(100, ax=ax) > 0
    assert plotdfs.auto_point_size(0) == 0.25
    plt.close(fig)


def test_parse_metadata_dfs_colors_numeric_boolean_and_categorical(tmp_path: Path, capsys):
    meta1 = tmp_path / "meta1.tsv"
    meta1.write_text("rbh\tscore\tflag\tlabel\nr1\t1\tTrue\tA\nr2\t2\tFalse\tB\n")
    meta2 = tmp_path / "meta2.tsv"
    meta2.write_text("rbh\tcustom\tcustom_color\nr1\tX\t#112233\nr2\tY\t#445566\n")

    merged = plotdfs.parse_metadata_dfs([str(meta1), str(meta2)])

    assert set(merged.columns) >= {"score_color", "flag_color", "label_color", "custom_color"}
    assert merged.loc[merged["rbh"] == "r1", "custom_color"].iloc[0] == "#112233"
    assert merged.loc[merged["rbh"] == "r1", "flag_color"].iloc[0] != merged.loc[merged["rbh"] == "r2", "flag_color"].iloc[0]
    assert capsys.readouterr().out == ""


def test_parse_metadata_dfs_rejects_invalid_files(tmp_path: Path):
    meta = tmp_path / "bad.tsv"
    meta.write_text("rbh\trbh_color\nr1\t#112233\n")
    with pytest.raises(ValueError, match="conflicts with the column we will merge against"):
        plotdfs.parse_metadata_dfs([str(meta)])


def test_load_phylo_df_by_rank_from_phylolist(tmp_path: Path):
    path1 = tmp_path / "subsample_phylum.neighbors_15.mind_0.1.df"
    path2 = tmp_path / "subsample_class.neighbors_30.mind_0.2.df"
    pd.DataFrame({"x": [1], "y": [2]}).to_csv(path1, sep="\t")
    pd.DataFrame({"x": [3], "y": [4]}).to_csv(path2, sep="\t")

    df_by_rank, all_params, row_labels = plotdfs.load_phylo_df_by_rank_from_phylolist([str(path1), str(path2)])

    assert set(df_by_rank) == {"phylum", "class"}
    assert all_params == [(15, 0.1), (30, 0.2)]
    assert row_labels["phylum"] == "phylum"

