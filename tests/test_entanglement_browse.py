from __future__ import annotations

from pathlib import Path

import pandas as pd

from egt import entanglement_browse as eb


def test_helper_functions():
    cols = ["species", "(A, B)", "('C','D')", "not_a_pair"]
    assert eb._pair_cols(cols) == ["(A, B)", "('C','D')"]
    assert eb._parse_pair("('A', 'B')") == ("A", "B")
    assert eb._lineage_match("1;2759;33208", 2759) is True
    assert eb._lineage_match(None, 2759) is False


def test_main_writes_ranked_outputs(tmp_path: Path):
    pf = tmp_path / "presence.tsv"
    pd.DataFrame(
        {
            "species": ["sp1", "sp2", "sp3"],
            "taxidstring": ["1;33208", "1;33208", "1;2759"],
            "('A', 'B')": [1, 1, 0],
            "('A', 'C')": [1, 0, 1],
        }
    ).to_csv(pf, sep="\t", index=False)

    out_dir = tmp_path / "out"
    assert eb.main(
        [
            "--presence-fusions",
            str(pf),
            "--clade-groupings",
            "Metazoa:33208,Eukaryota:2759",
            "--top-n",
            "2",
            "--min-species-in-clade",
            "1",
            "--out-dir",
            str(out_dir),
        ]
    ) == 0

    out_tsv = out_dir / "entangled_pairs_per_clade.tsv"
    out_md = out_dir / "entangled_pairs_top.md"
    assert out_tsv.exists()
    assert out_md.exists()

    ranked = pd.read_csv(out_tsv, sep="\t")
    assert set(ranked["clade"]) == {"Metazoa", "Eukaryota"}
    assert "fold_enrichment" in ranked.columns
    assert "Metazoa" in out_md.read_text()
