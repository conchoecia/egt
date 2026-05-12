from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from egt import algs_split_across_scaffolds as ass


def test_parse_args_and_plot_helpers(tmp_path):
    rbh_dir = tmp_path / "rbhs"
    rbh_dir.mkdir()

    args = ass.parse_args(["-d", str(rbh_dir), "-m", "0.01", "-a", "ALG"])
    assert args.rbh_directory == str(rbh_dir)
    assert args.minsig == 0.01
    assert args.alg == "ALG"

    splitsdf = pd.DataFrame(
        {
            "sample": ["sp1", "sp1", "sp1", "sp2", "sp2"],
            "gene_group": ["A", "A", "B", "A", "B"],
            "scaffold": ["chr1", "chr2", "chr1", "chr1", "chr2"],
        }
    )
    inferred = {"sp1": 3, "sp2": 2}
    fig, axes = plt.subplots(1, 2)
    ass.plot_chrom_number_vs_number_ALGs_split(axes[0], splitsdf, 2, inferred)
    ass.plot_chrom_number_vs_number_ALGs_perchrom(axes[1], splitsdf, inferred)
    assert axes[0].get_xlabel() == "Number of chromosomes"
    assert axes[1].get_ylabel() == "Mean number of ALGs on each chromosome"
    plt.close(fig)


def test_main_builds_combined_plot(tmp_path, monkeypatch):
    rbh_dir = tmp_path / "rbhs"
    rbh_dir.mkdir()
    (rbh_dir / "a.rbh").write_text("x\n")
    (rbh_dir / "b.rbh").write_text("x\n")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ass, "parse_args", lambda argv=None: type("Args", (), {"rbh_directory": str(rbh_dir), "minsig": 0.01, "alg": "ALG"})())
    monkeypatch.setattr(ass.odp_plot, "format_matplotlib", lambda: None)

    def fake_parse_rbh(path):
        return pd.DataFrame({"path": [path]})

    def fake_rbhdf_to_alglocdf(_rbhdf, _minsig, _alg):
        return (
            pd.DataFrame(
                {
                    "sample": ["sample1", "sample1"],
                    "gene_group": ["A", "A"],
                    "scaffold": ["chr1", "chr2"],
                }
            ),
            "sample1",
        )

    monkeypatch.setattr(ass.rbh_tools, "parse_rbh", fake_parse_rbh)
    monkeypatch.setattr(ass.rbh_tools, "rbhdf_to_alglocdf", fake_rbhdf_to_alglocdf)
    monkeypatch.setattr(ass.rbh_tools, "rbh_to_scafnum", lambda _rbhdf, _sample: 7)

    assert ass.main([]) == 0
    assert (tmp_path / "ALGs_split_across_two_or_more_scaffolds.pdf").exists()
