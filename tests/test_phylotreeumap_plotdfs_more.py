from __future__ import annotations

from pathlib import Path

import pandas as pd

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
