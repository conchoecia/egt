from __future__ import annotations

from pathlib import Path

import pandas as pd

from egt.count_unique_changes_per_branch import main, parse_changestring


def test_parse_changestring_extracts_branch_events():
    changestring = "1-([('A','B')]|['L1']|['S1'])-2-([('B','C')]|[]|[])-3"

    branches = parse_changestring(changestring)

    assert branches == [
        (1, 2, [("A", "B")], ["L1"], ["S1"]),
        (2, 3, [("B", "C")], [], []),
    ]


def test_parse_changestring_handles_negative_taxids():
    changestring = "--67-([('A','B')]|[]|[])-2"
    branches = parse_changestring(changestring)
    assert branches[0][0] == -67
    assert branches[0][1] == 2


def test_main_aggregates_unique_changes(tmp_path: Path, capsys):
    input_tsv = tmp_path / "changes.tsv"
    pd.DataFrame(
        {
            "changestrings": [
                "1-([('A','B')]|['L1']|['S1'])-2",
                "1-([('B','A')]|['L1']|['S2'])-2",
                None,
            ]
        }
    ).to_csv(input_tsv, sep="\t", index=False)

    rc = main([str(input_tsv)])

    assert rc == 0
    out = capsys.readouterr()
    assert "source_taxid\ttarget_taxid" in out.out
    # Fusion tuple is sorted, so A/B from both rows collapses to one unique fusion.
    assert "1\t2\t1\t1\t2\tA+B\tL1\tS1; S2" in out.out
    assert "Parsing 3 changestrings" in out.err
