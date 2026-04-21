from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pandas as pd

source_pkg = types.ModuleType("source")
rbh_tools_mod = types.ModuleType("source.rbh_tools")
rbh_tools_mod.parse_rbh = lambda _path: pd.DataFrame()
source_pkg.rbh_tools = rbh_tools_mod
sys.modules.setdefault("source", source_pkg)
sys.modules.setdefault("source.rbh_tools", rbh_tools_mod)

pbstp = importlib.import_module("egt.plot_branch_stats_tree_pair")


def test_parse_args_and_main(monkeypatch, tmp_path: Path):
    df_file = tmp_path / "genomes.tsv"
    node = tmp_path / "node.tsv"
    edge = tmp_path / "edge.tsv"
    pairs = tmp_path / "pairs.tsv"
    rbh_dir = tmp_path / "rbhs"
    rbh_dir.mkdir()

    pd.DataFrame(
        {
            "sample": ["spin-10-GCA1", "spout-20-GCA1"],
            "taxid_list": ["[1, 10]", "[1, 20]"],
        }
    ).to_csv(df_file, sep="\t", index=False)
    pd.DataFrame({"taxid": [1]}).to_csv(node, sep="\t", index=False)
    pd.DataFrame({"parent_taxid": [1], "child_taxid": [10]}).to_csv(edge, sep="\t", index=False)
    pd.DataFrame(
        {
            "sd_in_out_ratio_log_sigma": [-3.0],
            "num_species_in": [1],
            "nodename": ["TestClade"],
            "rbh1": ["ALG1"],
            "rbh2": ["ALG2"],
            "ortholog1": ["orth1"],
            "ortholog2": ["orth2"],
            "taxid": [10],
        }
    ).to_csv(pairs, sep="\t", index=False)

    rbh = rbh_dir / "sample.rbh"
    rbh.write_text("x\n")
    fake_rbh = pd.DataFrame(
        {
            "rbh": ["orth1", "orth2"],
            "spin-10-GCA1_scaf": ["chr1", "chr1"],
            "spin-10-GCA1_pos": [10, 30],
            "BCnSSimakov2022_scaf": ["a", "a"],
        }
    )
    monkeypatch.setattr(pbstp.rbh_tools, "parse_rbh", lambda _path: fake_rbh)
    monkeypatch.setattr(pbstp, "format_matplotlib", lambda: None)
    monkeypatch.chdir(tmp_path)

    args = pbstp.parse_args(
        [
            "-d", str(df_file),
            "-n", str(node),
            "-e", str(edge),
            "-p", str(pairs),
            "-r", str(rbh_dir),
            "-m", "1",
        ]
    )
    assert args.df_file == str(df_file)

    rc = pbstp.main(
        [
            "-d", str(df_file),
            "-n", str(node),
            "-e", str(edge),
            "-p", str(pairs),
            "-r", str(rbh_dir),
            "-m", "1",
        ]
    )
    assert rc == 0
    assert any(path.suffix == ".pdf" for path in tmp_path.iterdir())
