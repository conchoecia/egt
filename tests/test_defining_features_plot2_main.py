from __future__ import annotations

from types import SimpleNamespace

import matplotlib.pyplot as plt
import pandas as pd

from egt.legacy import defining_features_plot2 as dfp2


def test_main_smoke_writes_summary_outputs(tmp_path, monkeypatch):
    stats_dir = tmp_path / "clade_stats"
    stats_dir.mkdir()
    stats_path = stats_dir / "NodeA_123_unique_pair_df.tsv.gz"
    pd.DataFrame(
        {
            "pair": [0, 1],
            "notna_in": [4, 4],
            "notna_out": [0, 2],
            "occupancy_in": [1.0, 1.0],
            "occupancy_out": [0.0, 0.5],
            "mean_in": [5.0, 20.0],
            "mean_out": [0.0, 2.0],
            "sd_in": [1.0, 10.0],
            "sd_out": [0.0, 1.0],
        }
    ).to_csv(stats_path, sep="\t", index=False, compression="gzip")

    rbh_file = tmp_path / "alg.rbh"
    pair_file = tmp_path / "pairs.tsv"
    rbh_file.write_text("placeholder\n")
    pair_file.write_text("placeholder\n")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        dfp2,
        "parse_args",
        lambda: SimpleNamespace(
            clade_stats_dir=str(stats_dir),
            rbh_file=str(rbh_file),
            pair_combination_path=str(pair_file),
            sigma=1,
        ),
    )
    monkeypatch.setattr(
        dfp2.rbh_tools,
        "parse_rbh",
        lambda _path: pd.DataFrame({"rbh": ["orth1", "orth2", "orth3", "orth4"], "gene_group": ["ALG1", "ALG2", "ALG3", "ALG4"]}),
    )
    monkeypatch.setattr(dfp2, "algcomboix_file_to_dict", lambda _path: {("orth1", "orth2"): 0, ("orth3", "orth4"): 1})
    monkeypatch.setattr(dfp2, "make_marginal_plot", lambda *args, **kwargs: plt.figure(figsize=(2, 2)))

    assert dfp2.main() is None
    assert (tmp_path / "unique_pairs.tsv").exists()
    assert (tmp_path / "summary_stats.tsv").exists()
    assert (tmp_path / "NodeA_123_sdratiolog_occupancy.pdf").exists()
