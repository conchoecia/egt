from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from egt import plot_decay_many_species as pdms


def _write_decay(path: Path, rows: list[tuple[str, int, int, int]]) -> None:
    pd.DataFrame(rows, columns=["ALG", "conserved", "scattered", "total"]).to_csv(path, sep="\t", index=False)


def test_parse_args_and_parse_all_sizes(tmp_path: Path):
    _write_decay(tmp_path / "a.tsv", [("A", 5, 1, 6), ("B", 4, 2, 6)])
    args = pdms.parse_args(["-d", str(tmp_path)])
    assert args.directory == str(tmp_path)

    sizes = pdms.parse_all_ALGs_to_size_df([str(tmp_path / "a.tsv")])
    assert sizes == {"A": 6, "B": 6}

    with pytest.raises(Exception, match="does not exist"):
        pdms.parse_args(["-d", str(tmp_path / "missing")])


def test_filelist_to_plot_data_structure_and_plot_helpers(tmp_path: Path, monkeypatch):
    file1 = tmp_path / "sp1.tsv"
    file2 = tmp_path / "sp2.tsv"
    _write_decay(file1, [("A", 6, 0, 6), ("B", 4, 2, 6)])
    _write_decay(file2, [("A", 3, 3, 6), ("C", 2, 2, 4)])

    plot_these = pdms.filelist_to_plot_data_structure([str(file1), str(file2)], bins=3)
    assert len(plot_these) == 3
    assert any(plot_these.values())

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    first_bin = next(v for v in plot_these.values() if v)
    alg_sizes = pdms.parse_all_ALGs_to_size_df([str(file1), str(file2)])

    pdms.plot_decay_log(axes[0], first_bin)
    pdms.plot_decay(
        axes[1],
        first_bin,
        y_column="fraction_conserved",
        x_column="rank",
        alg_to_size_dict=alg_sizes,
        x_axis_labels_are_ALGs=True,
        plot_bars=True,
        ymin=0,
        ymax=1.1,
    )
    assert axes[0].get_yscale() == "log"
    assert axes[1].get_ylim()[1] == pytest.approx(1.1)
    plt.close(fig)

    fig, ax = plt.subplots()
    with pytest.raises(Exception, match="alg_to_size_dict"):
        pdms.plot_decay(ax, first_bin, x_axis_labels_are_ALGs=True)
    plt.close(fig)


def test_figures_and_main(monkeypatch, tmp_path: Path):
    file1 = tmp_path / "BCnS_LGs_sp1_ALG_decay.tsv"
    file2 = tmp_path / "BCnS_LGs_sp2_ALG_decay.tsv"
    _write_decay(file1, [("A", 6, 0, 6), ("B", 4, 2, 6)])
    _write_decay(file2, [("A", 5, 1, 6), ("B", 2, 4, 6)])

    plot_these = pdms.filelist_to_plot_data_structure([str(file1), str(file2)], bins=3)
    index_to_bin = sorted(plot_these.keys(), reverse=True)
    alg_sizes = pdms.parse_all_ALGs_to_size_df([str(file1), str(file2)])

    saved = []
    monkeypatch.setattr(pdms.odp_plot, "format_matplotlib", lambda: None)
    monkeypatch.setattr(plt, "savefig", lambda path, format=None: saved.append((Path(path).name, format)))

    pdms.fig1(plot_these, index_to_bin)
    pdms.fig2(plot_these, index_to_bin, alg_sizes)
    assert ("output_plot.jpg", "jpeg") in saved
    assert ("decay_plots_final.jpg", "jpeg") in saved
    assert ("decay_plots_final.pdf", "pdf") in saved

    saved.clear()
    assert pdms.main(["-d", str(tmp_path)]) == 0
    assert ("output_plot.jpg", "jpeg") in saved
