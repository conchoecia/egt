from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from egt import divergence_vs_dispersal as dvd


class FakeColor:
    def __init__(self, color: str):
        self.color = color


class FakePalette:
    fallback = FakeColor("#777777")

    def for_taxid(self, taxid):
        return FakeColor("#112233" if taxid == 2 else "#445566")


def test_load_divergence_validates_columns(tmp_path: Path):
    good = tmp_path / "div.tsv"
    good.write_text("species\tmedian_pct_id\nsp1\t0.8\nsp2\tbad\n")
    df = dvd._load_divergence(good, "median_pct_id")
    assert list(df["species"]) == ["sp1"]

    missing_species = tmp_path / "missing_species.tsv"
    missing_species.write_text("name\tmedian_pct_id\nsp1\t0.8\n")
    with pytest.raises(SystemExit, match="species"):
        dvd._load_divergence(missing_species, "median_pct_id")

    missing_col = tmp_path / "missing_col.tsv"
    missing_col.write_text("species\tother\nsp1\t1.0\n")
    with pytest.raises(SystemExit, match="lacks column"):
        dvd._load_divergence(missing_col, "median_pct_id")


def test_load_presence_fusions_supports_explicit_or_computed_dispersal(tmp_path: Path):
    explicit = tmp_path / "explicit.tsv"
    explicit.write_text(
        "species\tdisp\ttaxidstring\n"
        "sp1\t0.25\t1;2\n"
        "sp2\t0.50\t1;3\n"
    )
    df = dvd._load_presence_fusions(explicit, "disp")
    assert list(df["dispersal"]) == [0.25, 0.5]

    computed = tmp_path / "computed.tsv"
    computed.write_text(
        "species\ttaxidstring\talg1\talg2\tchangestrings\t(ignore)\n"
        "sp1\t1;2\t1\t0\tx\ty\n"
        "sp2\t1;3\t0\t0\tx\ty\n"
    )
    computed_df = dvd._load_presence_fusions(computed, None)
    assert computed_df.loc[0, "dispersal"] == 0.5
    assert computed_df.loc[1, "dispersal"] == 1.0


def test_classify_clade_and_linregress_edge_cases():
    assert dvd._classify_clade("1;2;3", [(1, "root"), (2, "metazoa")]) == "metazoa"
    assert dvd._classify_clade("1;4", [(2, "metazoa")]) is None
    assert dvd._classify_clade(None, [(2, "metazoa")]) is None

    small = dvd._linregress(pd.Series([1, 2]).to_numpy(), pd.Series([3, 4]).to_numpy())
    assert pd.isna(small["slope"])

    zero_var = dvd._linregress(pd.Series([1, 1, 1]).to_numpy(), pd.Series([2, 3, 4]).to_numpy())
    assert pd.isna(zero_var["r"])

    stats = dvd._linregress(pd.Series([1.0, 2.0, 3.0]).to_numpy(), pd.Series([2.0, 4.0, 6.0]).to_numpy())
    assert stats["slope"] == pytest.approx(2.0)
    assert stats["r2"] == pytest.approx(1.0)


def test_main_writes_outputs(monkeypatch, tmp_path: Path):
    div = tmp_path / "div.tsv"
    div.write_text("species\tmedian_pct_id\nsp1\t0.8\nsp2\t0.6\nsp3\t0.4\n")
    pf = tmp_path / "presence.tsv"
    pf.write_text(
        "species\ttaxidstring\talg1\talg2\n"
        "sp1\t1;2\t1\t1\n"
        "sp2\t1;2\t1\t0\n"
        "sp3\t1;3\t0\t0\n"
    )
    outdir = tmp_path / "out"

    monkeypatch.setattr(dvd.Palette, "from_yaml", staticmethod(lambda _path: FakePalette()))

    rc = dvd.main(
        [
            "--divergence-tsv", str(div),
            "--presence-fusions", str(pf),
            "--clade-groupings", "Metazoa:2,Other:3",
            "--out-dir", str(outdir),
            "--min-species", "1",
        ]
    )

    assert rc == 0
    table = pd.read_csv(outdir / "divergence_vs_dispersal.tsv", sep="\t")
    stats = pd.read_csv(outdir / "divergence_vs_dispersal_stats.tsv", sep="\t")
    assert set(table["clade"]) == {"Metazoa", "Other"}
    assert "ALL" in set(stats["clade"])
    assert (outdir / "divergence_vs_dispersal.pdf").exists()
