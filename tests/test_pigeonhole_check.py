from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from egt import pigeonhole_check as pc


def test_helper_functions_and_simulation():
    assert pc._is_pair_col("(A, B)") is True
    assert pc._is_pair_col("A") is False
    assert pc._lineage_match("1;2;3", 2) is True
    assert pc._lineage_match(None, 2) is False

    sims = pc._simulate_pairs(4, 2, 8, np.random.default_rng(0))
    assert len(sims) == 8
    assert sims.dtype.kind in {"i", "u"}
    assert pc._empirical_p(2, np.array([0, 1, 2, 3])) == (2 + 1) / (4 + 1)


def test_load_inputs_and_resolve_clade_spec(tmp_path: Path):
    pf = tmp_path / "presence.tsv"
    cc = tmp_path / "chrom.tsv"
    pd.DataFrame(
        {
            "species": ["sp1", "sp2"],
            "taxidstring": ["1;33208", "1;2759"],
            "taxid": ["10", "20"],
            "changestrings": ["x", "y"],
            "A": [1, 0],
            "B": [0, 1],
            "(A, B)": [1, 0],
        }
    ).to_csv(pf, sep="\t", index=False)
    pd.DataFrame({"sample": ["sp1", "sp2"], "chromosomes": ["5", "7"]}).to_csv(cc, sep="\t", index=False)

    _pf, _cc, alg_cols, pair_cols, n_algs, obs_pairs = pc._load_inputs(pf, cc)
    assert alg_cols == ["A", "B"]
    assert pair_cols == ["(A, B)"]
    assert list(n_algs) == [1, 1]
    assert list(obs_pairs) == [1, 0]

    class FakeNCBI:
        def get_taxid_translator(self, taxids):
            return {taxids[0]: "Metazoa"}

        def get_name_translator(self, names):
            return {"Metazoa": [33208]}

    assert pc._resolve_clade_spec("Metazoa:33208", None) == (33208, "Metazoa")
    assert pc._resolve_clade_spec("33208", FakeNCBI()) == (33208, "Metazoa")
    assert pc._resolve_clade_spec("Metazoa", FakeNCBI()) == (33208, "Metazoa")


def test_main_writes_species_and_clade_outputs(tmp_path: Path):
    pf = tmp_path / "presence.tsv"
    cc = tmp_path / "chrom.tsv"
    pd.DataFrame(
        {
            "species": ["sp1", "sp2", "sp3"],
            "taxidstring": ["1;33208", "1;33208", "1;2759"],
            "taxid": ["10", "11", "20"],
            "A": [1, 1, 1],
            "B": [1, 0, 1],
            "C": [0, 1, 1],
            "(A, B)": [1, 0, 0],
            "(A, C)": [0, 1, 1],
        }
    ).to_csv(pf, sep="\t", index=False)
    pd.DataFrame(
        {"sample": ["sp1", "sp2", "sp3"], "chromosomes": [2, 3, 2]}
    ).to_csv(cc, sep="\t", index=False)

    outdir = tmp_path / "out"
    assert pc.main(
        [
            "--presence-fusions",
            str(pf),
            "--chrom-counts",
            str(cc),
            "--clade-groupings",
            "Metazoa:33208,Euk:2759",
            "--n-simulations",
            "20",
            "--seed",
            "1",
            "--out-dir",
            str(outdir),
        ]
    ) == 0

    per_species = pd.read_csv(outdir / "per_species.tsv", sep="\t")
    per_clade = pd.read_csv(outdir / "per_clade.tsv", sep="\t")
    assert set(per_species["species"]) == {"sp1", "sp2", "sp3"}
    assert set(per_clade["clade"]) == {"Metazoa", "Euk"}
    assert "fold_enrichment" in per_clade.columns
