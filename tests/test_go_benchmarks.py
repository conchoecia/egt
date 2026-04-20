"""Tests for egt.go.benchmarks.*

alg_structure has a tiny self-contained fixture path (no external refs).
The others (goatools_ref, camps, simakov) are exercised as CLI-help smoke
tests only — each requires heavy external reference data that we do not
stage into the fixture tree.
"""
from __future__ import annotations

import importlib
from pathlib import Path

import pytest


GO_DB = Path(__file__).resolve().parent / "testdb" / "go"


# ---------- alg_structure — full end-to-end on fixture ----------
def test_alg_structure_helpers():
    from egt.go.benchmarks import alg_structure as alg
    fam, counts = alg.load_family_alg(GO_DB / "family_map_alg.tsv")
    assert fam == {"FAM_A": "A1a", "FAM_B": "A1a", "FAM_C": "A1b",
                   "FAM_D": "B1", "FAM_E": "B1",
                   "FAM_F": "B2", "FAM_G": "C1"}
    # A1a:2, A1b:1, B1:2, B2:1, C1:1 — total 7
    assert counts["A1a"] == 2
    # Null = sum(n_i/N)^2 = (4 + 1 + 4 + 1 + 1) / 49 = 11/49
    p = alg.expected_same_alg_fraction(counts)
    assert abs(p - 11 / 49) < 1e-12


def test_alg_structure_per_pair_within_alg():
    import pandas as pd
    from egt.go.benchmarks import alg_structure as alg
    fam, _ = alg.load_family_alg(GO_DB / "family_map_alg.tsv")
    up = pd.read_csv(GO_DB / "unique_pairs_alg.tsv", sep="\t")
    up2 = alg.per_pair_within_alg(up, fam)
    # FAM_A & FAM_B both A1a → same_alg True.
    row_ab = up2[(up2["ortholog1"] == "FAM_A") & (up2["ortholog2"] == "FAM_B")].iloc[0]
    assert row_ab["same_alg"] is True or row_ab["same_alg"]
    # FAM_A & FAM_C are A1a vs A1b → different.
    row_ac = up2[(up2["ortholog1"] == "FAM_A") & (up2["ortholog2"] == "FAM_C")].iloc[0]
    assert not row_ac["same_alg"]


def test_alg_structure_summarize_flag():
    import pandas as pd
    from egt.go.benchmarks import alg_structure as alg
    fam, _ = alg.load_family_alg(GO_DB / "family_map_alg.tsv")
    up = pd.read_csv(GO_DB / "unique_pairs_alg.tsv", sep="\t")
    up2 = alg.per_pair_within_alg(up, fam)
    s = alg.summarize_flag(up2, "stable_in_clade")
    assert s["flag"] == "stable_in_clade"
    assert s["n_pairs"] >= 1
    assert 0.0 <= s["frac_within_alg"] <= 1.0


def test_alg_structure_per_clade_breakdown():
    import pandas as pd
    from egt.go.benchmarks import alg_structure as alg
    fam, _ = alg.load_family_alg(GO_DB / "family_map_alg.tsv")
    up = pd.read_csv(GO_DB / "unique_pairs_alg.tsv", sep="\t")
    up2 = alg.per_pair_within_alg(up, fam)
    g = alg.per_clade_flag_breakdown(up2, "stable_in_clade")
    assert "nodename" in g.columns
    assert "frac_within_alg" in g.columns


def test_alg_structure_plot_heatmap_noop_and_render(tmp_path):
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from egt.go.benchmarks import alg_structure as alg
    fam, counts = alg.load_family_alg(GO_DB / "family_map_alg.tsv")
    up = pd.read_csv(GO_DB / "unique_pairs_alg.tsv", sep="\t")
    up2 = alg.per_pair_within_alg(up, fam)
    # Flag with matches → returns an imshow mappable.
    fig, ax = plt.subplots()
    im = alg.plot_alg_heatmap(up2, "stable_in_clade", ax,
                                sorted(counts.keys()))
    assert im is not None
    plt.close(fig)
    # Empty flag → axis turned off; returns None.
    fig, ax = plt.subplots()
    empty = up2.copy()
    empty["nonexistent"] = False
    rv = alg.plot_alg_heatmap(empty, "nonexistent", ax,
                                sorted(counts.keys()))
    assert rv is None
    plt.close(fig)


def test_alg_structure_main_end_to_end(tmp_path):
    from egt.go.benchmarks import alg_structure as alg
    rc = alg.main([
        "--family-map", str(GO_DB / "family_map_alg.tsv"),
        "--unique-pairs", str(GO_DB / "unique_pairs_alg.tsv"),
        "--out-dir", str(tmp_path),
    ])
    assert rc == 0
    for name in ("flag_summary.tsv",
                 "per_clade_close_in_clade.tsv",
                 "per_clade_stable_in_clade.tsv",
                 "alg_heatmaps.pdf",
                 "within_alg_by_flag.pdf"):
        assert (tmp_path / name).exists(), f"missing {name}"


# ---------- goatools_ref / camps / simakov — help-only smoke ----------
@pytest.mark.parametrize("modpath", [
    "egt.go.benchmarks.goatools_ref",
    "egt.go.benchmarks.camps",
    "egt.go.benchmarks.simakov",
])
def test_benchmark_module_importable_and_help(modpath, capsys):
    mod = importlib.import_module(modpath)
    assert hasattr(mod, "main")
    # --help exits 0 via SystemExit; argparse raises it.
    with pytest.raises(SystemExit) as excinfo:
        mod.main(["--help"])
    # argparse exits with code 0 on --help.
    assert excinfo.value.code == 0


# ---------- benchmark dispatcher ----------
def test_benchmark_dispatch_help(capsys):
    from egt.go.benchmark_dispatch import main, REFS
    # Bare --help (no --ref) prints the dispatch help and exits 0.
    rc = main(["--help"])
    assert rc == 0
    # Every ref resolves to a real module.
    for name, modpath in REFS.items():
        importlib.import_module(modpath)


def test_benchmark_dispatch_dispatches_alg(tmp_path):
    from egt.go.benchmark_dispatch import main
    rc = main([
        "--ref", "alg",
        "--family-map", str(GO_DB / "family_map_alg.tsv"),
        "--unique-pairs", str(GO_DB / "unique_pairs_alg.tsv"),
        "--out-dir", str(tmp_path),
    ])
    assert rc == 0
    assert (tmp_path / "flag_summary.tsv").exists()


def test_benchmark_dispatch_unknown_ref_errors():
    from egt.go.benchmark_dispatch import main
    with pytest.raises(SystemExit):
        main(["--ref", "does-not-exist"])
