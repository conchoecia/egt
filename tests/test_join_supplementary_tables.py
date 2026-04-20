from __future__ import annotations

from pathlib import Path

import pandas as pd

from egt.join_supplementary_tables import main, parse_args


def test_parse_args_defaults():
    args = parse_args(["--subsample", "a.tsv", "--merged", "b.tsv", "--output", "c.tsv"])
    assert args.join_type == "left"


def test_main_joins_on_normalized_assembly_accession(tmp_path: Path, capsys):
    subsample = tmp_path / "subsample.tsv"
    subsample.write_text(
        "sample\tvalue\nSpecies-9606-GCF964019385.1-extra\t1\nSpecies-9607-GCA000000001.1-extra\t2\n"
    )
    merged = tmp_path / "merged.tsv"
    merged.write_text(
        "Assembly Accession\tmeta\nGCF_964019385.1\talpha\nGCA_000000001.1\tbeta\n"
    )
    out = tmp_path / "joined.tsv"

    rc = main(
        [
            "--subsample",
            str(subsample),
            "--merged",
            str(merged),
            "--output",
            str(out),
        ]
    )

    assert rc == 0
    joined = pd.read_csv(out, sep="\t")
    assert list(joined["meta"]) == ["alpha", "beta"]
    captured = capsys.readouterr()
    assert "Performing left join" in captured.out
    assert "Extracted accession: GCF_964019385.1" in captured.out


def test_main_reports_unmatched_left_rows(tmp_path: Path, capsys):
    subsample = tmp_path / "subsample.tsv"
    subsample.write_text("sample\tvalue\nSpecies-9606-GCF999999999.1-extra\t1\n")
    merged = tmp_path / "merged.tsv"
    merged.write_text("Assembly Accession\tmeta\nGCF_000000001.1\talpha\n")
    out = tmp_path / "joined.tsv"

    main(
        [
            "--subsample",
            str(subsample),
            "--merged",
            str(merged),
            "--output",
            str(out),
        ]
    )

    captured = capsys.readouterr()
    assert "no match in merged file" in captured.out

