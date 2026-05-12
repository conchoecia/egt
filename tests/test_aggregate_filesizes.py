from __future__ import annotations

from pathlib import Path

import pytest

from egt.aggregate_filesizes import bytes_to_mb, main, size_mb_or_na


def test_bytes_to_mb_uses_requested_base():
    assert bytes_to_mb(1024 * 1024, 1024) == 1.0
    assert bytes_to_mb(1_000_000, 1000) == 1.0


def test_size_mb_or_na_formats_existing_file(tmp_path: Path):
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"x" * 2048)
    assert size_mb_or_na(str(payload), 1024) == "0.002"


def test_size_mb_or_na_handles_missing_and_none(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    missing = tmp_path / "missing.bin"
    assert size_mb_or_na(None) == "NA"
    assert size_mb_or_na(str(missing)) == "NA"
    captured = capsys.readouterr()
    assert "WARNING: file not found" in captured.err


def test_main_writes_expected_summary_table(tmp_path: Path):
    proteins = tmp_path / "proteins.fa"
    proteins.write_bytes(b"p" * 1024)
    genome = tmp_path / "genome.fa"
    genome.write_bytes(b"g" * 2048)

    config = tmp_path / "config.yaml"
    config.write_text(
        f"""
species:
  sampleA:
    assembly_accession: GCF_000001.1
    proteins: {proteins}
    chrom:
    genome: {genome}
""".lstrip()
    )

    out = tmp_path / "nested" / "sizes.tsv"
    rc = main(["--config", str(config), "--out", str(out), "--base", "1024"])

    assert rc == 0
    lines = out.read_text().splitlines()
    assert lines[0] == "sample\tassembly_accession\tproteins_MB\tchrom_MB\tgenome_MB"
    assert lines[1] == "sampleA\tGCF_000001.1\t0.001\tNA\t0.002"

