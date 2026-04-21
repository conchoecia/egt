from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from egt import plot_alg_dispersion as pad


class FakeNCBI:
    def get_taxid_translator(self, taxids):
        return {taxid: f"Taxon{taxid}" for taxid in taxids}


def test_parse_args_and_metadata_helpers(monkeypatch, tmp_path: Path):
    directory = tmp_path / "rbhs"
    directory.mkdir()
    alg_rbh = tmp_path / "alg.rbh"
    alg_rbh.write_text("x\n")
    args = pad.parse_args(["-d", str(directory), "-a", str(alg_rbh)])
    assert args.outdir.endswith("alg_dispersion_plots")

    monkeypatch.setattr(pad, "NCBITaxa", lambda: FakeNCBI())
    assert pad.taxidstring_to_lineage("1;-67;2", FakeNCBI()) == "Taxon1;Myriazoa;Taxon2"

    meta = tmp_path / "meta.tsv"
    pd.DataFrame({"species": ["sp1"], "taxidstring": ["1;2"]}).to_csv(meta, sep="\t", index=False)
    parsed = pad.parse_metadata(str(meta))
    assert parsed["sp1"] == "Taxon1;Taxon2"


def test_species_file_discovery_and_conservation_calc(monkeypatch, tmp_path: Path):
    f1 = tmp_path / "BCnSSimakov2022_SpeciesA_xy_reciprocal_best_hits.plotted.rbh"
    f2 = tmp_path / "SpeciesB_vs_BCnS.rbh"
    f1.write_text("x\n")
    f2.write_text("x\n")
    found = pad.find_species_rbh_files(str(tmp_path), "BCnS")
    assert "SpeciesA" in found

    df = pd.DataFrame(
        {
            "whole_FET": [0.01, 0.2, 0.01],
            "gene_group": ["A", "A", "B"],
            "Species_gene": ["g1", "g2", "g3"],
        }
    )
    monkeypatch.setattr(pad.pd, "read_csv", lambda *_args, **_kwargs: df.copy())
    cons = pad.calculate_alg_conservation_per_species("ignored", "ALG", ["A", "B", "C"], minsig=0.05)
    assert cons == {"A": 1, "B": 1, "C": 0}


def test_plot_dispersion_by_alg_and_main(monkeypatch, tmp_path: Path):
    species_to_rbh = {"sp1": "a.rbh", "sp2": "b.rbh"}
    alg_df = pd.DataFrame(
        {
            "ALGname": ["A", "B"],
            "Size": [10, 20],
            "Color": ["#111111", "#222222"],
        }
    )

    def fake_calc(rbh_file, algname, sorted_algs, minsig=0.05):
        return {"A": 10 if "a.rbh" in rbh_file else 2, "B": 18 if "a.rbh" in rbh_file else 4}

    monkeypatch.setattr(pad, "calculate_alg_conservation_per_species", fake_calc)
    monkeypatch.setattr(pad, "NCBITaxa", lambda: FakeNCBI())
    outdir = tmp_path / "out"
    pad.plot_dispersion_by_alg(species_to_rbh, alg_df, "ALG", 0.05, str(outdir), species_to_lineage={"sp1": "L1", "sp2": "L2"})
    assert any(path.suffix == ".pdf" for path in outdir.iterdir())

    alg_rbh = tmp_path / "alg.rbh"
    alg_rbh.write_text("x\n")
    rbhs = tmp_path / "rbhs"
    rbhs.mkdir()
    (rbhs / "BCnSSimakov2022_sp1_xy_reciprocal_best_hits.plotted.rbh").write_text("x\n")
    monkeypatch.setattr(
        pad.rbh_tools,
        "parse_ALG_rbh_to_colordf",
        lambda _path: pd.DataFrame({"ALGname": ["A", "B"], "Size": [10, 20], "Color": ["#111111", "#222222"]}),
    )
    monkeypatch.setattr(pad, "plot_dispersion_by_alg", lambda *args, **kwargs: outdir.mkdir(exist_ok=True))
    rc = pad.main(["-d", str(rbhs), "-a", str(alg_rbh), "-o", str(outdir)])
    assert rc == 0
