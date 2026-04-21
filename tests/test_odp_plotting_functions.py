from __future__ import annotations

from pathlib import Path

from egt._vendor import odp_plotting_functions as odp


def test_format_matplotlib_sets_expected_rcparams():
    odp.format_matplotlib()
    assert odp.matplotlib.rcParams["pdf.fonttype"] == 42
    assert odp.matplotlib.rcParams["ps.fonttype"] == 42
    assert odp.matplotlib.rcParams["image.composite_image"] is False


def test_plot_decay_writes_plot_and_tsv(tmp_path: Path):
    datastruct = {
        "ALG1": ({"chr1": 3}, {"chr2": 1}),
        "ALG2": ({"chr3": 2}, {"chr4": 2}),
    }
    outplot = tmp_path / "decay.pdf"
    outtsv = tmp_path / "decay.tsv"

    odp.plot_decay(datastruct, outplot, outtsv)

    assert outplot.exists()
    lines = outtsv.read_text().splitlines()
    assert lines[0] == "ALG\tconserved\tscattered\ttotal"
    assert any("ALG1\t3\t1\t4" in line for line in lines[1:])
