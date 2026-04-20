from __future__ import annotations

from pathlib import Path

import pandas as pd

from egt.aggregate_filechecker_benchmarks import _input_paths, aggregate, main


def test_input_paths_cover_supported_rules():
    config = {
        "species": {
            "SAMPLE": {
                "genome": "/tmp/genome.fa",
                "proteins": "/tmp/proteins.fa",
                "chrom": "/tmp/chrom.tsv",
            }
        }
    }

    assert _input_paths("check_genome_legality", "SAMPLE", config) == {
        "genome_bytes": "/tmp/genome.fa"
    }
    assert _input_paths("check_protein_legality", "SAMPLE", config) == {
        "proteins_bytes": "/tmp/proteins.fa"
    }
    assert _input_paths("check_chrom_legality", "SAMPLE", config) == {
        "genome_bytes": "/tmp/genome.fa",
        "proteins_bytes": "/tmp/proteins.fa",
        "chrom_bytes": "/tmp/chrom.tsv",
    }
    assert _input_paths("unknown_rule", "SAMPLE", config) == {}


def test_aggregate_reads_benchmarks_and_file_sizes(tmp_path: Path):
    genome = tmp_path / "genome.fa"
    genome.write_bytes(b"g" * 11)
    proteins = tmp_path / "proteins.fa"
    proteins.write_bytes(b"p" * 7)
    chrom = tmp_path / "chrom.tsv"
    chrom.write_bytes(b"c" * 5)

    config = tmp_path / "config.yaml"
    config.write_text(
        f"""
species:
  SAMPLE:
    genome: {genome}
    proteins: {proteins}
    chrom: {chrom}
""".lstrip()
    )

    bench_dir = tmp_path / "benchmarks" / "check_chrom_legality"
    bench_dir.mkdir(parents=True)
    pd.DataFrame([{"s": 1.5, "max_rss": 1234}]).to_csv(
        bench_dir / "SAMPLE.check_chrom_legality.benchmark.txt",
        sep="\t",
        index=False,
    )
    pd.DataFrame(columns=["s", "max_rss"]).to_csv(
        bench_dir / "EMPTY.check_chrom_legality.benchmark.txt",
        sep="\t",
        index=False,
    )
    (bench_dir / "not_a_benchmark.txt").write_text("ignore\n")

    df = aggregate(str(config), str(tmp_path / "benchmarks"))

    assert list(df["sample"]) == ["SAMPLE"]
    row = df.iloc[0]
    assert row["rule"] == "check_chrom_legality"
    assert row["genome_bytes"] == 11
    assert row["proteins_bytes"] == 7
    assert row["chrom_bytes"] == 5


def test_aggregate_handles_missing_species_and_missing_input_files(tmp_path: Path):
    config = tmp_path / "config.yaml"
    config.write_text("species:\n  OTHER:\n    genome: /nope.fa\n")

    bench_dir = tmp_path / "benchmarks" / "check_genome_legality"
    bench_dir.mkdir(parents=True)
    pd.DataFrame([{"s": 2.0}]).to_csv(
        bench_dir / "MISSING.check_genome_legality.benchmark.txt",
        sep="\t",
        index=False,
    )

    df = aggregate(str(config), str(tmp_path / "benchmarks"))
    row = df.iloc[0]
    assert row["sample"] == "MISSING"
    assert "genome_bytes" not in row or pd.isna(row.get("genome_bytes"))


def test_main_writes_output_tsv(tmp_path: Path):
    genome = tmp_path / "genome.fa"
    genome.write_bytes(b"abc")
    config = tmp_path / "config.yaml"
    config.write_text(
        f"""
species:
  SAMPLE:
    genome: {genome}
    proteins: {genome}
    chrom: {genome}
""".lstrip()
    )

    bench_dir = tmp_path / "benchmarks"
    (bench_dir / "check_genome_legality").mkdir(parents=True)
    pd.DataFrame([{"s": 1.0}]).to_csv(
        bench_dir / "check_genome_legality" / "SAMPLE.check_genome_legality.benchmark.txt",
        sep="\t",
        index=False,
    )

    out = tmp_path / "aggregated.tsv"
    rc = main(["--config", str(config), "--benchmarks", str(bench_dir), "--out", str(out)])

    assert rc == 0
    written = pd.read_csv(out, sep="\t")
    assert list(written["sample"]) == ["SAMPLE"]

