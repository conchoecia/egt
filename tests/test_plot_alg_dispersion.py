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

    csv_meta = tmp_path / "meta.csv"
    pd.DataFrame({"species": ["sp2"], "taxidstring": [pd.NA]}).to_csv(csv_meta, index=False)
    parsed_csv = pad.parse_metadata(str(csv_meta))
    assert parsed_csv["sp2"] == "Unknown"


def test_parse_args_and_metadata_validation_errors(tmp_path: Path):
    alg_rbh = tmp_path / "alg.rbh"
    alg_rbh.write_text("x\n")

    with pytest.raises(SystemExit):
        pad.parse_args(["-d", str(tmp_path / "missing"), "-a", str(alg_rbh)])
    with pytest.raises(SystemExit):
        pad.parse_args(["-d", str(tmp_path), "-a", str(tmp_path / "missing.rbh")])

    bad_meta = tmp_path / "bad_meta.tsv"
    pd.DataFrame({"wrong": ["sp1"], "taxidstring": ["1;2"]}).to_csv(bad_meta, sep="\t", index=False)
    with pytest.raises(SystemExit):
        pad.parse_metadata(str(bad_meta))

    bad_meta2 = tmp_path / "bad_meta2.tsv"
    pd.DataFrame({"species": ["sp1"], "wrong": ["1;2"]}).to_csv(bad_meta2, sep="\t", index=False)
    with pytest.raises(SystemExit):
        pad.parse_metadata(str(bad_meta2))


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

    monkeypatch.setattr(pad.pd, "read_csv", lambda *_args, **_kwargs: pd.DataFrame())
    assert pad.calculate_alg_conservation_per_species("ignored", "ALG", ["A"], minsig=0.05) == {}

    no_sig = pd.DataFrame({"whole_FET": [0.9], "gene_group": ["A"], "Species_gene": ["g1"]})
    monkeypatch.setattr(pad.pd, "read_csv", lambda *_args, **_kwargs: no_sig.copy())
    assert pad.calculate_alg_conservation_per_species("ignored", "ALG", ["A"], minsig=0.05) == {}

    no_species_cols = pd.DataFrame({"whole_FET": [0.001], "gene_group": ["A"]})
    monkeypatch.setattr(pad.pd, "read_csv", lambda *_args, **_kwargs: no_species_cols.copy())
    assert pad.calculate_alg_conservation_per_species("ignored", "ALG", ["A"], minsig=0.05) == {}


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
