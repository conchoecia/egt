from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from egt.plot_fourier_support_vs_time import main, parse_args


def test_parse_args_defaults():
    args = parse_args(["-d", "indir"])
    assert args.directory == "indir"
    assert args.output == "support_vs_time_window.pdf"


def test_main_errors_when_no_input_files(tmp_path: Path):
    with pytest.raises(SystemExit):
        main(["-d", str(tmp_path), "-o", str(tmp_path / "out.pdf")])


def test_main_plots_fusion_and_dispersal_support(tmp_path: Path, capsys):
    fusion = pd.DataFrame(
        {
            "clade": ["fusion_100_padded", "fusion_100_padded"],
            "obs_exp": ["observed", "expected"],
            "period": [12, 25],
            "support_mean": [0.5, 0.1],
        }
    )
    dispersal = pd.DataFrame(
        {
            "clade": ["dispersal_200_unpadded"],
            "obs_exp": ["observed"],
            "period": [44],
            "support_mean": [0.7],
        }
    )
    fusion.to_csv(tmp_path / "fusion_rsims.tsv", sep="\t", index=False)
    dispersal.to_csv(tmp_path / "dispersal_rsims.tsv", sep="\t", index=False)
    out = tmp_path / "support.pdf"

    rc = main(["-d", str(tmp_path), "-o", str(out)])

    assert rc == 0
    assert out.exists()
    captured = capsys.readouterr()
    assert "Found 2 total files" in captured.out
    assert "Saved:" in captured.out

