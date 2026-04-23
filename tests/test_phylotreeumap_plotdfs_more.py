from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from egt import phylotreeumap_plotdfs as plotdfs


def _write_df(path: Path, colors: bool = True) -> Path:
    df = pd.DataFrame(
        {
            "rbh": ["r1", "r2", "r3"],
            "UMAP1": [0.0, 1.0, 2.0],
            "UMAP2": [1.0, 0.5, -0.5],
        }
    )
    if colors:
        df["color"] = ["#111111", "#222222", "#333333"]
    df.to_csv(path, sep="\t")
    return path


def test_plot_paramsweep_writes_pdf(tmp_path: Path):
    p1 = _write_df(tmp_path / "sample.neighbors_10.mind_0.1.df")
    p2 = _write_df(tmp_path / "sample.neighbors_20.mind_0.2.df")
    args = plotdfs.parse_args(["-f", f"{p1} {p2}", "-p", str(tmp_path / "out"), "--pdf"])
    df_dict = plotdfs.generate_df_dict(args)
    outpdf = tmp_path / "grid.pdf"
    plotdfs.plot_paramsweep(df_dict, outpdf)
    assert outpdf.exists()


def test_plot_phylo_resampling_grid_and_main_phylolist(tmp_path: Path):
    p1 = _write_df(tmp_path / "subsample_phylum.neighbors_10.mind_0.1.df")
    p2 = _write_df(tmp_path / "subsample_class.neighbors_20.mind_0.2.df")

    df_by_rank, all_params, row_labels = plotdfs.load_phylo_df_by_rank_from_phylolist([str(p1), str(p2)])
    outpdf = tmp_path / "phylo.pdf"
    plotdfs.plot_phylo_resampling_grid(df_by_rank, all_params, row_labels, outpdf)
    assert outpdf.exists()

    prefix = tmp_path / "pref"
    rc = plotdfs.main(["-p", str(prefix), "--phylolist", str(p1), str(p2)])
    assert rc is None
    assert (tmp_path / "pref.phyloresample.pdf").exists()


def test_plot_phylo_resampling_grid_additional_branches(tmp_path: Path):
    def _df(n: int, *, color: bool) -> pd.DataFrame:
        df = pd.DataFrame(
            {
                "UMAP1": [i % 17 for i in range(n)],
                "UMAP2": [i // 17 for i in range(n)],
            }
        )
        if color:
            df["color"] = ["#123456"] * n
        return df

    df_by_rank = {
        "phylum": {(20, 0.1): {"df": _df(125, color=True)}},
        "subphylum": {(20, 0.1): {"df": _df(170, color=True)}},
        "family": {(20, 0.1): {"df": _df(1500, color=False)}},
        "genus": {(20, 0.1): {"df": pd.DataFrame(columns=["UMAP1", "UMAP2"])}} ,
    }
    row_labels = {rank: rank for rank in df_by_rank}
    outpdf = tmp_path / "phylo_branches.pdf"
    plotdfs.plot_phylo_resampling_grid(
        df_by_rank,
        [(20, 0.1)],
        row_labels,
        outpdf,
        point_min_alpha=0.75,
        point_max_alpha=0.95,
    )
    assert outpdf.exists()

    with pytest.raises(ValueError, match="No ranks to plot"):
        plotdfs.plot_phylo_resampling_grid({}, [], {}, tmp_path / "empty.pdf")


def test_plot_phyla_writes_clean_and_annotation_outputs(tmp_path: Path):
    df = pd.DataFrame(
        {
            "rbh": ["r1", "r2", "r3", "r4"],
            "UMAP1": [0.0, 1.0, 2.0, 3.0],
            "UMAP2": [3.0, 2.0, 1.0, 0.0],
            "taxid_list_str": [
                "1;2759;33154;33208;6040",
                "1;2759;33154;33208;6073",
                "1;2759;33154;33208;6040",
                "1;2759;33154;33208;6073",
            ],
            "color": ["#111111", "#222222", "#333333", "#444444"],
        }
    )
    infile = tmp_path / "phyla.tsv"
    df.to_csv(infile, sep="\t")

    args = SimpleNamespace(
        filelist=str(infile),
        phyla_order="Cnidaria Porifera MissingPhylum",
        phyla_rotation="maximize_square",
        num_cols=2,
        phyla_clean_output=True,
    )
    outpdf = tmp_path / "phyla.pdf"
    plotdfs.plot_phyla(args, str(outpdf))

    assert outpdf.exists()
    assert (tmp_path / "phyla_clean.pdf").exists()


def test_plot_phyla_recolors_from_palette_and_writes_recolored_df(tmp_path: Path):
    df = pd.DataFrame(
        {
            "rbh": ["r1", "r2"],
            "UMAP1": [0.0, 1.0],
            "UMAP2": [1.0, 0.0],
            "taxid_list_str": [
                "1;2759;33154;33208;6040",
                "1;2759;33154;33208;6073",
            ],
            "color": ["#111111", "#222222"],
        }
    )
    infile = tmp_path / "phyla_palette.tsv"
    df.to_csv(infile, sep="\t")

    palette_yaml = tmp_path / "palette.yaml"
    palette_yaml.write_text(
        """
schema_version: 1
clades:
  porifera:
    taxid: 6040
    label: "Porifera"
    color: "#abcdef"
    phylopic_uuid: null
  cnidaria:
    taxid: 6073
    label: "Cnidaria"
    color: "#fedcba"
    phylopic_uuid: null
fallback:
  label: "other"
  color: "#123456"
""".lstrip()
    )

    recolored_out = tmp_path / "recolored.tsv"
    args = SimpleNamespace(
        filelist=str(infile),
        phyla_order=None,
        phyla_rotation="maximize_square",
        num_cols=2,
        phyla_clean_output=False,
        color_source="palette",
        palette=str(palette_yaml),
        recolored_df_out=str(recolored_out),
    )
    outpdf = tmp_path / "phyla_palette.pdf"
    plotdfs.plot_phyla(args, str(outpdf))

    assert outpdf.exists()
    assert recolored_out.exists()
    recolored = pd.read_csv(recolored_out, sep="\t", index_col=0)
    assert list(recolored["color"]) == ["#abcdef", "#fedcba"]


def test_plot_phyla_preserves_df_colors_when_requested(tmp_path: Path):
    df = pd.DataFrame(
        {
            "rbh": ["r1", "r2"],
            "UMAP1": [0.0, 1.0],
            "UMAP2": [1.0, 0.0],
            "taxid_list_str": [
                "1;2759;33154;33208;6040",
                "1;2759;33154;33208;6073",
            ],
            "color": ["#111111", "#222222"],
        }
    )
    infile = tmp_path / "phyla_dfcolor.tsv"
    df.to_csv(infile, sep="\t")

    recolored_out = tmp_path / "dfcolor_out.tsv"
    args = SimpleNamespace(
        filelist=str(infile),
        phyla_order=None,
        phyla_rotation="maximize_square",
        num_cols=2,
        phyla_clean_output=False,
        color_source="df",
        palette=None,
        recolored_df_out=str(recolored_out),
    )
    outpdf = tmp_path / "phyla_dfcolor.pdf"
    plotdfs.plot_phyla(args, str(outpdf))

    assert outpdf.exists()
    preserved = pd.read_csv(recolored_out, sep="\t", index_col=0)
    assert list(preserved["color"]) == ["#111111", "#222222"]


def test_plot_phyla_uses_reference_df_colors_when_requested(tmp_path: Path):
    df = pd.DataFrame(
        {
            "sample": ["s1", "s2"],
            "UMAP1": [0.0, 1.0],
            "UMAP2": [1.0, 0.0],
            "taxid_list_str": [
                "1;2759;33154;33208;6040",
                "1;2759;33154;33208;6073",
            ],
            "color": ["#111111", "#222222"],
        }
    )
    infile = tmp_path / "phyla_reference.tsv"
    df.to_csv(infile, sep="\t")

    reference = pd.DataFrame(
        {
            "sample": ["s1", "s2"],
            "color": ["#abcdef", "#fedcba"],
        }
    )
    reference_file = tmp_path / "reference.tsv"
    reference.to_csv(reference_file, sep="\t")

    recolored_out = tmp_path / "reference_out.tsv"
    args = SimpleNamespace(
        filelist=str(infile),
        phyla_order=None,
        phyla_rotation="maximize_square",
        num_cols=2,
        phyla_clean_output=False,
        color_source="reference-df",
        reference_df=str(reference_file),
        palette=None,
        recolored_df_out=str(recolored_out),
    )
    outpdf = tmp_path / "phyla_reference.pdf"
    plotdfs.plot_phyla(args, str(outpdf))

    assert outpdf.exists()
    recolored = pd.read_csv(recolored_out, sep="\t", index_col=0)
    assert list(recolored["color"]) == ["#abcdef", "#fedcba"]


def test_plot_phyla_without_color_column_and_with_taxid_list(tmp_path: Path, capsys):
    df = pd.DataFrame(
        {
            "rbh": ["r1", "r2", "r3", "r4"],
            "UMAP1": [0.0, 1.0, 2.0, 0.7],
            "UMAP2": [2.0, 0.2, 1.6, -0.8],
            "taxid_list": [
                "[1, 2759, 33154, 33208, 6040]",
                [1, 2759, 33154, 33208, 6073],
                "[1, 2759, 10232]",
                "[1, 2759, 33154, 33208, 6040]",
            ],
        }
    )
    infile = tmp_path / "phyla_nocolor.tsv"
    df.to_csv(infile, sep="\t")

    args = SimpleNamespace(
        filelist=str(infile),
        phyla_order="Cnidaria Rotifera",
        phyla_rotation="minimize_vertical",
        num_cols=1,
        phyla_clean_output=False,
    )
    outpdf = tmp_path / "phyla_nocolor.pdf"
    plotdfs.plot_phyla(args, str(outpdf))

    output = capsys.readouterr().out
    assert outpdf.exists()
    assert "Using custom phyla order" in output
    assert "Plotting 2 phyla in custom order" in output
    assert "Rotifera" in output
    assert "Using rectangular panels" in output


def test_plot_phyla_validation_errors(tmp_path: Path):
    bad = tmp_path / "bad.tsv"
    pd.DataFrame({"rbh": ["r1"], "UMAP1": [0.0]}).to_csv(bad, sep="\t")

    args = SimpleNamespace(
        filelist=str(bad),
        phyla_order=None,
        phyla_rotation="maximize_square",
        num_cols=2,
        phyla_clean_output=False,
    )
    with pytest.raises(ValueError, match="must contain 'UMAP1' and 'UMAP2'"):
        plotdfs.plot_phyla(args, str(tmp_path / "bad.pdf"))

    no_tax = tmp_path / "no_tax.tsv"
    pd.DataFrame({"rbh": ["r1"], "UMAP1": [0.0], "UMAP2": [1.0]}).to_csv(no_tax, sep="\t")
    args.filelist = str(no_tax)
    with pytest.raises(ValueError, match="must contain either 'taxid_list' or 'taxid_list_str'"):
        plotdfs.plot_phyla(args, str(tmp_path / "no_tax.pdf"))

    df = pd.DataFrame(
        {
            "rbh": ["r1"],
            "UMAP1": [0.0],
            "UMAP2": [1.0],
            "taxid_list_str": ["999999"],
        }
    )
    infile = tmp_path / "missing_phyla.tsv"
    df.to_csv(infile, sep="\t")
    args.filelist = str(infile)
    args.phyla_order = "Cnidaria"
    with pytest.raises(ValueError, match="None of the requested phyla were found in the data"):
        plotdfs.plot_phyla(args, str(tmp_path / "missing_phyla.pdf"))


def test_plot_features_with_metadata_and_genome_colorbars(tmp_path: Path):
    df = pd.DataFrame(
        {
            "rbh": ["r1", "r2", "r3"],
            "UMAP1": [0.0, 1.0, 2.0],
            "UMAP2": [2.0, 1.0, 0.0],
            "color": ["#111111", "#222222", "#333333"],
            "num_scaffolds": [10, 20, 30],
            "GC_content": [0.4, 0.5, 0.6],
            "genome_size": [1_000_000, 2_000_000, 3_000_000],
            "genome_size_log10": [6.0, 6.3, 6.5],
            "median_scaffold_length": [1000, 2000, 3000],
            "mean_scaffold_length": [1200, 2200, 3200],
            "scaffold_N50": [5000, 6000, 7000],
            "longest_scaffold": [10000, 11000, 12000],
            "smallest_scaffold": [100, 200, 300],
            "fraction_Ns": [0.01, 0.02, 0.03],
            "number_of_gaps": [1, 2, 3],
            "num_proteins": [100, 110, 120],
            "mean_protein_length": [300, 320, 340],
            "median_protein_length": [250, 260, 270],
            "longest_protein": [1000, 1100, 1200],
            "smallest_protein": [50, 55, 60],
            "from_rbh": [1, 1, 1],
            "frac_ologs": [0.1, 0.2, 0.3],
            "frac_ologs_sig": [0.05, 0.1, 0.15],
            "frac_ologs_single": [0.02, 0.04, 0.06],
            "frac_ologs_extra": [0.3, 0.4, 0.5],
        }
    )
    infile = tmp_path / "features.tsv"
    df.to_csv(infile, sep="\t")
    metadata = pd.DataFrame(
        {
            "rbh": ["r1", "r2", "r3"],
            "habitat": ["marine", "freshwater", "marine"],
            "habitat_color": ["#aa0000", "#00aa00", "#aa0000"],
        }
    )
    args = SimpleNamespace(filelist=str(infile), num_cols=3)
    outpdf = tmp_path / "features.pdf"
    plotdfs.plot_features(
        args,
        str(outpdf),
        metadata_df=metadata,
        legend_scale=0.8,
        genome_min_bp=1_500_000,
        genome_max_bp=2_500_000,
        use_benedictus=True,
    )
    assert outpdf.exists()


def test_generate_umap_grid_bokeh_and_main_dispatch(tmp_path: Path, monkeypatch):
    p1 = _write_df(tmp_path / "sample.neighbors_10.mind_0.1.df")
    p2 = _write_df(tmp_path / "sample.neighbors_20.mind_0.2.df")
    args = plotdfs.parse_args(["-f", f"{p1} {p2}", "-p", str(tmp_path / "out"), "--html"])
    df_dict = plotdfs.generate_df_dict(args)

    html_out = tmp_path / "grid.html"
    plotdfs.generate_umap_grid_bokeh(df_dict, str(html_out))
    assert html_out.exists()

    calls = []
    monkeypatch.setattr(plotdfs, "plot_paramsweep", lambda df_dict, outpdf: calls.append(("pdf", outpdf)))
    monkeypatch.setattr(plotdfs, "generate_umap_grid_bokeh", lambda df_dict, outhtml: calls.append(("html", outhtml)))

    assert plotdfs.main(["-f", f"{p1} {p2}", "-p", str(tmp_path / "mainout"), "--pdf", "--html"]) == 0
    assert ("pdf", str(tmp_path / "mainout.pdf")) in calls
    assert ("html", str(tmp_path / "mainout.html")) in calls


def test_main_plot_features_with_metadata_parse_and_phylolist_errors(tmp_path: Path):
    infile = _write_df(tmp_path / "sample.neighbors_10.mind_0.1.df")
    meta = tmp_path / "meta.tsv"
    meta.write_text("rbh\tgroup\nr1\tA\nr2\tB\nr3\tC\n")

    assert (
        plotdfs.main(
            [
                "-f",
                str(infile),
                "-p",
                str(tmp_path / "featfull"),
                "--plot_features",
                "--metadata",
                str(meta),
                "--pdf",
            ]
        )
        == 0
    )
    assert (tmp_path / "featfull.features.pdf").exists()

    with pytest.raises(ValueError, match="--phylolist cannot be used with --directory/--filelist"):
        plotdfs.main(["-f", str(infile), "-p", "out", "--phylolist", str(infile)])


def test_main_plot_features_and_plot_phyla_with_metadata(tmp_path: Path, monkeypatch):
    infile = _write_df(tmp_path / "sample.neighbors_10.mind_0.1.df")
    meta = tmp_path / "meta.tsv"
    meta.write_text("rbh\tflag\nr1\tTrue\nr2\tFalse\nr3\tTrue\n")

    calls = {}
    monkeypatch.setattr(plotdfs, "plot_features", lambda args, outpdf, **kwargs: calls.setdefault("features", (outpdf, kwargs)))
    monkeypatch.setattr(plotdfs, "plot_phyla", lambda args, outpdf, metadata_df=None: calls.setdefault("phyla", (outpdf, metadata_df is not None)))

    assert (
        plotdfs.main(
            [
                "-f",
                str(infile),
                "-p",
                str(tmp_path / "feat"),
                "--plot_features",
                "--metadata",
                str(meta),
                "--legend-scale",
                "0.7",
                "--genome-min-bp",
                "1000",
                "--genome-max-bp",
                "2000",
                "--threecolor",
            ]
        )
        == 0
    )
    assert calls["features"][0] == str(tmp_path / "feat.features.pdf")
    assert calls["features"][1]["use_benedictus"] is True

    assert plotdfs.main(
        [
            "-f",
            str(infile),
            "-p",
            str(tmp_path / "phy"),
            "--plot-phyla",
            "--metadata",
            str(meta),
        ]
    ) is None
    assert calls["phyla"] == (str(tmp_path / "phy.phyla.pdf"), True)
