from __future__ import annotations

import csv
from pathlib import Path

from egt.get_assembly_sizes import extract_assembly_info, main, process_directory, write_to_tsv


def test_extract_assembly_info_reads_nested_fields(tmp_path: Path):
    payload = tmp_path / "one.jsonl"
    payload.write_text('{"accession": "GCF_1", "assemblyStats": {"totalSequenceLength": 12345}}')

    accession, total = extract_assembly_info(payload)

    assert accession == "GCF_1"
    assert total == 12345


def test_process_directory_recurses_and_collects_jsonl_files(tmp_path: Path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "one.jsonl").write_text('{"accession": "A", "assemblyStats": {"totalSequenceLength": 1}}')
    (tmp_path / "a" / "skip.txt").write_text("ignore")
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "two.jsonl").write_text('{"accession": "B", "assemblyStats": {"totalSequenceLength": 2}}')

    results = sorted(process_directory(tmp_path))

    assert results == [("A", 1), ("B", 2)]


def test_write_to_tsv_writes_header_and_rows(tmp_path: Path):
    out = tmp_path / "sizes.tsv"
    write_to_tsv(out, [("A", 1), ("B", 2)])

    with out.open() as fh:
        rows = list(csv.reader(fh, delimiter="\t"))

    assert rows == [
        ["Assembly Accession", "Total Sequence Length"],
        ["A", "1"],
        ["B", "2"],
    ]


def test_main_generates_output_file(tmp_path: Path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "one.jsonl").write_text('{"accession": "A", "assemblyStats": {"totalSequenceLength": 55}}')
    out = tmp_path / "out.tsv"

    rc = main(["--genomes-dir", str(tmp_path / "data"), "--output", str(out)])

    assert rc == 0
    assert "A\t55" in out.read_text()

