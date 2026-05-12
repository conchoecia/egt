from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from egt import palette as palette_module
from egt import phylotreeumap_plotdfs as plotdfs


def test_generate_distinct_colors_and_benedictus_helpers():
    colors = plotdfs.generate_distinct_colors(4)
    assert len(colors) == 4
    assert all(color.startswith("#") and len(color) == 7 for color in colors)
    assert len(plotdfs.benedictus_n(5)) == 5


def test_resolve_palette_phylum_color_uses_nearest_covered_ancestor_when_exact_missing(tmp_path: Path):
    palette_yaml = tmp_path / "palette.yaml"
    palette_yaml.write_text(
        """
schema_version: 1
clades:
  panarthropoda:
    taxid: 88770
    label: "Panarthropoda"
    color: "#aa1122"
    phylopic_uuid: null
  deuterostomia:
    taxid: 33511
    label: "Deuterostomia"
    color: "#334455"
    phylopic_uuid: null
fallback:
  label: "other"
  color: "#cccccc"
""".lstrip()
    )

    palette = plotdfs.load_palette(str(palette_yaml))

    arthropod_lineages = [
        [1, 2759, 33154, 33208, 6072, 33213, 88770, 6656, 6961],
        [1, 2759, 33154, 33208, 6072, 33213, 88770, 6656, 7041],
    ]
    color, taxid, label = plotdfs.resolve_palette_phylum_color(
        "Arthropoda", arthropod_lineages, palette
    )
    assert (color, taxid, label) == ("#aa1122", 88770, "Panarthropoda")

    chordate_lineages = [
        [1, 2759, 33154, 33208, 6072, 33213, 33511, 7711, 7898],
        [1, 2759, 33154, 33208, 6072, 33213, 33511, 7711, 8292],
    ]
    color, taxid, label = plotdfs.resolve_palette_phylum_color(
        "Chordata", chordate_lineages, palette
    )
    assert (color, taxid, label) == ("#334455", 33511, "Deuterostomia")


def test_recolor_df_with_palette_canonicalizes_merged_taxids(monkeypatch, tmp_path: Path):
    class FakeCanonicalizer:
        def canonicalize(self, taxid):
            return {50: 100}.get(int(taxid), int(taxid))

    monkeypatch.setattr(
        palette_module,
        "_get_shared_taxid_canonicalizer",
        lambda: FakeCanonicalizer(),
    )

    palette_yaml = tmp_path / "palette.yaml"
    palette_yaml.write_text(
        """
schema_version: 1
clades:
  merged_target:
    taxid: 100
    label: "Merged Target"
    color: "#123abc"
    phylopic_uuid: null
fallback:
  label: "other"
  color: "#cccccc"
""".lstrip()
    )

    palette = plotdfs.load_palette(str(palette_yaml))
    df = pd.DataFrame({"taxid_list_str": ["1;50", "1;999"], "UMAP1": [0.0, 1.0], "UMAP2": [0.0, 1.0]})

    recolored = plotdfs.recolor_df_with_palette(df, palette)

    assert list(recolored["color"]) == ["#123abc", "#cccccc"]


def test_parse_args_validates_conflicts_and_metadata(tmp_path: Path):
    df = tmp_path / "subsample_phylum.neighbors_15.mind_0.1.df"
    pd.DataFrame({"x": [1], "y": [2]}).to_csv(df, sep="\t")
    meta = tmp_path / "meta.tsv"
    meta.write_text("rbh\tflag\nr1\tTrue\n")

    args = plotdfs.parse_args(["-f", str(df), "-p", "out", "--metadata", str(meta), "--pdf"])
    assert args.metadata == [str(meta)]
    assert args.benedictus is False
    assert args.palette is None
    assert args.color_source == "df"
    assert args.reference_df is None
    assert args.recolored_df_out is None

    with pytest.raises(ValueError, match="Both directory and filelist"):
        plotdfs.parse_args(["-d", str(tmp_path), "-f", str(df), "-p", "out"])


def test_parse_args_normalizes_phylolist_and_validates_ranges(tmp_path: Path):
    df1 = tmp_path / "subsample_phylum.neighbors_15.mind_0.1.df"
    df2 = tmp_path / "subsample_class.neighbors_20.mind_0.2.df"
    pd.DataFrame({"UMAP1": [0.0], "UMAP2": [1.0]}).to_csv(df1, sep="\t")
    pd.DataFrame({"UMAP1": [1.0], "UMAP2": [0.0]}).to_csv(df2, sep="\t")

    args = plotdfs.parse_args(
        ["-p", "out", "--phylolist", f"{df1} {df2}"]
    )
    assert args.phylolist == [str(df1), str(df2)]

    with pytest.raises(ValueError, match="genome-min-bp must be < genome-max-bp"):
        plotdfs.parse_args(
            ["-f", str(df1), "-p", "out", "--genome-min-bp", "5", "--genome-max-bp", "5"]
        )

    with pytest.raises(ValueError, match="File does not exist"):
        plotdfs.parse_args(["-p", "out", "--phylolist", str(tmp_path / "missing.df")])


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
    assert plotdfs.auto_point_size(10_000_000, min_size=0.5) == 0.5
    assert plotdfs.auto_point_size(1, max_size=2.0) == 2.0
    assert plotdfs.auto_point_alpha(0) == 0.84
    assert plotdfs.auto_point_alpha(100) > plotdfs.auto_point_alpha(10000)
    assert plotdfs.auto_point_fill(0) == 0.48
    assert plotdfs.auto_point_fill(100) > plotdfs.auto_point_fill(10000)
    assert 0.40 <= plotdfs.auto_point_alpha(10000) <= 0.84
    assert 0.20 <= plotdfs.auto_point_fill(10000) <= 0.48
    compact = plotdfs.estimate_panel_occupancy(
        [i / 19 for i in range(20)],
        [i / 19 for i in range(20)],
        q=None,
        pad=0.0,
        bins=20,
    )
    spread = plotdfs.estimate_panel_occupancy(
        [i / 4 for i in range(5) for _ in range(5)],
        [j / 4 for _ in range(5) for j in range(5)],
        q=None,
        pad=0.0,
        bins=20,
    )
    assert spread > compact
    assert plotdfs.occupancy_scaled_value(
        1.0, spread, n_points=1000, occ_ref=0.05, min_factor=0.5, max_factor=2.0
    ) > plotdfs.occupancy_scaled_value(
        1.0, compact, n_points=1000, occ_ref=0.05, min_factor=0.5, max_factor=2.0
    )
    assert plotdfs.neighbor_scaled_value(1.0, 10, 125) > plotdfs.neighbor_scaled_value(1.0, 10, 500)
    assert plotdfs.neighbor_scaled_value(1.0, 999, 1500) > plotdfs.neighbor_scaled_value(1.0, 20, 1500)
    assert plotdfs.neighbor_scaled_value(1.0, 20, 0) == 1.0
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


def test_recolor_df_from_reference_df_uses_sample_or_rbh(tmp_path: Path):
    df = pd.DataFrame(
        {
            "sample": ["s1", "s2", "s3"],
            "UMAP1": [0.0, 1.0, 2.0],
            "UMAP2": [1.0, 0.0, -1.0],
            "color": ["#111111", "#222222", "#333333"],
        }
    )
    ref = pd.DataFrame(
        {
            "sample": ["s1", "s2"],
            "color": ["#abcdef", "#fedcba"],
        }
    )
    recolored = plotdfs.recolor_df_from_reference_df(df, ref)
    assert list(recolored["color"]) == ["#abcdef", "#fedcba", "#333333"]

    df_rbh = pd.DataFrame({"rbh": ["r1"], "UMAP1": [0.0], "UMAP2": [0.0]})
    ref_rbh = pd.DataFrame({"rbh": ["r1"], "color": ["#123456"]})
    recolored_rbh = plotdfs.recolor_df_from_reference_df(df_rbh, ref_rbh)
    assert list(recolored_rbh["color"]) == ["#123456"]


def test_parse_args_requires_reference_df_for_reference_color_source(tmp_path: Path):
    df = tmp_path / "subsample_phylum.neighbors_15.mind_0.1.df"
    pd.DataFrame({"UMAP1": [0.0], "UMAP2": [1.0]}).to_csv(df, sep="\t")
    ref = tmp_path / "ref.tsv"
    pd.DataFrame({"sample": ["s1"], "color": ["#111111"]}).to_csv(ref, sep="\t")

    args = plotdfs.parse_args(
        ["-f", str(df), "-p", "out", "--color-source", "reference-df", "--reference-df", str(ref)]
    )
    assert args.reference_df == str(ref)

    with pytest.raises(ValueError, match="--reference-df is required"):
        plotdfs.parse_args(["-f", str(df), "-p", "out", "--color-source", "reference-df"])


def test_parse_metadata_dfs_rejects_invalid_files(tmp_path: Path):
    meta = tmp_path / "bad.tsv"
    meta.write_text("rbh\trbh_color\nr1\t#112233\n")
    with pytest.raises(ValueError, match="conflicts with the column we will merge against"):
        plotdfs.parse_metadata_dfs([str(meta)])


def test_parse_metadata_dfs_additional_validation_and_warnings(tmp_path: Path, capsys):
    with pytest.raises(ValueError, match="must be a list"):
        plotdfs.parse_metadata_dfs("not-a-list")

    missing = tmp_path / "missing.tsv"
    with pytest.raises(ValueError, match="Metadata file does not exist"):
        plotdfs.parse_metadata_dfs([str(missing)])

    no_rbh = tmp_path / "no_rbh.tsv"
    no_rbh.write_text("value\n1\n")
    with pytest.raises(ValueError, match="does not contain a 'rbh' column"):
        plotdfs.parse_metadata_dfs([str(no_rbh)])

    only_rbh = tmp_path / "only_rbh.tsv"
    only_rbh.write_text("rbh\nr1\n")
    with pytest.raises(ValueError, match="does not contain any columns other than 'rbh'"):
        plotdfs.parse_metadata_dfs([str(only_rbh)])

    no_pair = tmp_path / "no_pair.tsv"
    no_pair.write_text("rbh\tstatus_color\nr1\t#112233\n")
    with pytest.raises(ValueError, match="does not contain a corresponding column"):
        plotdfs.parse_metadata_dfs([str(no_pair)])

    bad_color = tmp_path / "bad_color.tsv"
    bad_color.write_text("rbh\tstatus\tstatus_color\nr1\tA\tred\n")
    with pytest.raises(ValueError, match="does not contain valid hex colors"):
        plotdfs.parse_metadata_dfs([str(bad_color)])

    one_value = tmp_path / "one_value.tsv"
    one_value.write_text("rbh\tlabel\nr1\tsame\nr2\tsame\n")
    merged = plotdfs.parse_metadata_dfs([str(one_value)])
    assert set(merged["label_color"]) == {"#000000"}
    assert "contains only one unique value" in capsys.readouterr().out

    dup1 = tmp_path / "dup1.tsv"
    dup1.write_text("rbh\tshared\tshared_color\nr1\t1\t#112233\n")
    dup2 = tmp_path / "dup2.tsv"
    dup2.write_text("rbh\tshared\tshared_color\nr2\t2\t#445566\n")
    with pytest.raises(ValueError, match="Conflicting column names across metadata files"):
        plotdfs.parse_metadata_dfs([str(dup1), str(dup2)])


def test_load_phylo_df_by_rank_from_phylolist(tmp_path: Path):
    path1 = tmp_path / "subsample_phylum.neighbors_15.mind_0.1.df"
    path2 = tmp_path / "subsample_class.neighbors_30.mind_0.2.df"
    pd.DataFrame({"x": [1], "y": [2]}).to_csv(path1, sep="\t")
    pd.DataFrame({"x": [3], "y": [4]}).to_csv(path2, sep="\t")

    df_by_rank, all_params, row_labels = plotdfs.load_phylo_df_by_rank_from_phylolist([str(path1), str(path2)])

    assert set(df_by_rank) == {"phylum", "class"}
    assert all_params == [(15, 0.1), (30, 0.2)]
    assert row_labels["phylum"] == "phylum"
