from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from egt import plot_decay_pairwise_steps as pdps


def _write_rbh(path: Path, rows: list[dict]) -> Path:
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)
    return path


def test_load_divergence_times_from_file_and_parse_config(tmp_path: Path):
    divergence_file = tmp_path / "divergence.tsv"
    divergence_file.write_text(
        "\n".join(
            [
                "1\t2\t100.5",
                "2\t3\t250",
                "1\t999\t4",
                "bad\tline",
                "",
            ]
        )
    )
    config = {
        "species": {
            "A": {"taxid": 1},
            "B": {"taxid": 2},
            "C": {"taxid": 3},
        }
    }

    mapped = pdps.load_divergence_times_from_file(str(divergence_file), config=config)
    assert mapped["A"]["B"] == pytest.approx(100.5)
    assert mapped["B"]["C"] == pytest.approx(250.0)
    assert "999" not in mapped

    unmapped = pdps.load_divergence_times_from_file(str(divergence_file), config=None)
    assert unmapped["1"]["2"] == pytest.approx(100.5)
    assert unmapped["2"]["3"] == pytest.approx(250.0)

    rbh_dir = tmp_path / "rbhs"
    rbh_dir.mkdir()
    for name in ["A_B_pair.rbh", "A_C_pair.rbh"]:
        (rbh_dir / name).write_text("A_gene\tA_scaf\tA_pos\tB_gene\tB_scaf\tB_pos\n")

    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(
        "target_species:\n"
        "  - A\n"
        "divergence_times:\n"
        "  A:\n"
        "    B: 100.5\n"
        "    C: 200.0\n"
        "  B:\n"
        "    A: 100.5\n"
        "  C:\n"
        "    A: 200.0\n"
    )

    parsed = pdps.parse_config(str(yaml_path), str(rbh_dir), "A")
    assert parsed["analyses"]["A"] == {"B": 100.5, "C": 200.0}
    assert parsed["analysis_files"]["A"]["B"].endswith("A_B_pair.rbh")
    assert parsed["analysis_files"]["A"]["C"].endswith("A_C_pair.rbh")

    yaml_missing = tmp_path / "config_missing.yaml"
    yaml_missing.write_text(
        "divergence_times:\n"
        "  A:\n"
        "    B: 100.5\n"
        "  B:\n"
        "    A: 100.5\n"
    )
    with pytest.raises(IOError, match="target_species"):
        pdps.parse_config(str(yaml_missing), str(rbh_dir), "missing")


def test_decay_pair_helpers_and_caching(tmp_path: Path, monkeypatch):
    rawdf = pd.DataFrame(
        {
            "A_scaf": ["chr1", "chr1", "chr2", "tiny"],
            "B_scaf": ["b1", "b2", "b1", "b1"],
            "A_gene": ["a1", "a2", "a3", "a4"],
            "B_gene": ["g1", "g2", "g3", "g4"],
            "A_pos": [10, 20, 30, 40],
            "B_pos": [100, 200, 300, 400],
            "whole_FET": [0.01, 0.20, 0.01, 0.01],
        }
    )
    sp_to_chr_to_size = {
        "A": {"chr1": 900000, "chr2": 800000, "tiny": 10},
        "B": {"b1": 900000, "b2": 800000},
    }

    decay = pdps.decay_of_one_species_pair(rawdf, "A", "B", sp_to_chr_to_size, min_scaf_len=100)
    chr1 = decay[decay["sp1_scaf"] == "chr1"].iloc[0]
    chr2 = decay[decay["sp1_scaf"] == "chr2"].iloc[0]
    tiny = decay[decay["sp1_scaf"] == "tiny"].iloc[0]
    assert chr1["sp2_scaf"] == ["b1"]
    assert chr1["sp1_scaf_genecount"] == 2
    assert chr1["conserved"] == 1
    assert chr1["scattered"] == 1
    assert chr2["conserved"] == 1
    assert tiny["sp1_scaf_genecount"] == 1
    assert tiny["conserved"] == 0

    jittered = pdps.jitter([0, 1, 2], 0.5)
    assert len(jittered) == 3
    assert min(jittered) >= -0.5

    one = _write_rbh(
        tmp_path / "A_B.rbh",
        [
            {"A_gene": "a1", "A_scaf": "chr1", "A_pos": 10, "B_gene": "b1", "B_scaf": "b1", "B_pos": 100},
            {"A_gene": "a2", "A_scaf": "chr1", "A_pos": 20, "B_gene": "b2", "B_scaf": "b1", "B_pos": 120},
        ],
    )
    two = _write_rbh(
        tmp_path / "A_C.rbh",
        [
            {"A_gene": "a1", "A_scaf": "chr1", "A_pos": 15, "C_gene": "c1", "C_scaf": "c1", "C_pos": 1000},
            {"A_gene": "a3", "A_scaf": "chr2", "A_pos": 30, "C_gene": "c2", "C_scaf": "c2", "C_pos": 2000},
        ],
    )

    chrom_sizes, gene_counts = pdps.rbh_files_to_sp_to_chr_to_size([str(one), str(two)])
    assert chrom_sizes["A"]["chr1"] == 20
    assert chrom_sizes["A"]["chr2"] == 30
    assert gene_counts["A"]["chr1"] == 2
    assert gene_counts["A"]["chr2"] == 1

    cache_dir = tmp_path / "cache"
    cached_once = pdps.get_chromosome_sizes_cached([str(one), str(two)], cache_dir=str(cache_dir))
    assert (cache_dir).exists()

    def _boom(_paths):
        raise AssertionError("cache should have been used")

    monkeypatch.setattr(pdps, "rbh_files_to_sp_to_chr_to_size", _boom)
    cached_twice = pdps.get_chromosome_sizes_cached([str(one), str(two)], cache_dir=str(cache_dir))
    assert cached_once == cached_twice


def test_calculate_pairwise_decay_and_plot_dispersion(tmp_path: Path, monkeypatch):
    rbh = _write_rbh(
        tmp_path / "A_B.rbh",
        [
            {
                "A_scaf": "Qa_chr1",
                "B_scaf": "b1",
                "A_gene": "a1",
                "B_gene": "b1",
                "A_pos": 10,
                "B_pos": 100,
                "whole_FET": 0.001,
            },
            {
                "A_scaf": "Qa_chr1",
                "B_scaf": "b1",
                "A_gene": "a2",
                "B_gene": "b2",
                "A_pos": 20,
                "B_pos": 120,
                "whole_FET": 0.001,
            },
            {
                "A_scaf": "Qb_chr2",
                "B_scaf": "b2",
                "A_gene": "a3",
                "B_gene": "b3",
                "A_pos": 30,
                "B_pos": 130,
                "whole_FET": 0.001,
            },
        ],
    )
    config = {
        "target_species": ["A"],
        "analyses": {"A": {"B": 123.0}},
        "analysis_files": {"A": {"B": str(rbh)}},
    }
    sp_to_chr_to_size = {"A": {"Qa_chr1": 900000, "Qb_chr2": 900000}, "B": {"b1": 900000, "b2": 900000}}
    keep_scafs = {"A": {"Qa_chr1", "Qb_chr2"}}

    result = pdps.calculate_pairwise_decay_sp1_vs_many(
        "A", config, sp_to_chr_to_size, keep_scafs, outdir=str(tmp_path / "out"), min_scaf_len=100
    )
    output = Path(result["A"]["B"])
    assert output.exists()
    saved = pd.read_csv(output, sep="\t")
    assert saved["fraction_conserved"].max() == pytest.approx(1.0)
    assert saved["divergence_time"].iloc[0] == pytest.approx(123.0)

    alg_db = tmp_path / "alg.rbh"
    alg_db.write_text("rbh\n")
    monkeypatch.setattr(
        pdps.rbh_tools,
        "parse_ALG_rbh_to_colordf",
        lambda _path: pd.DataFrame(
            {"ALGname": ["Qa", "Qb"], "Size": [10, 20], "Color": ["#aa0000", "#00aa00"]}
        ),
    )
    pdps.plot_dispersion_by_ALG("A", result, str(alg_db), algname="ALG", outdir=str(tmp_path / "plots"))
    assert (tmp_path / "plots" / "A_ALG_dispersion_by_conservation.pdf").exists()


def test_main_orchestrates_with_stubbed_helpers(monkeypatch, tmp_path: Path):
    args = SimpleNamespace(
        config=str(tmp_path / "config.yaml"),
        divergence_file=str(tmp_path / "divergence.tsv"),
        directory=str(tmp_path),
        target_species="A",
        cache_dir=str(tmp_path / "cache"),
        min_scaf_size=500000,
        fet_threshold=0.05,
        bin_size=50,
        ALG_rbh=str(tmp_path / "alg.rbh"),
        ALGname="ALG",
        ALG_rbh_dir=str(tmp_path / "algdir"),
    )

    monkeypatch.setattr(pdps, "parse_args", lambda argv=None: args)
    monkeypatch.setattr(pdps, "read_yaml_file", lambda _path: {"species": {"A": {"taxid": 1}}})
    monkeypatch.setattr(pdps, "load_divergence_times_from_file", lambda _path, config=None: {"A": {"B": 1.0}, "B": {"A": 1.0}})
    monkeypatch.setattr(
        pdps,
        "parse_config",
        lambda *_args, **_kwargs: {
            "target_species": ["A"],
            "analysis_files": {"A": {"B": str(tmp_path / "A_B.rbh")}},
            "analyses": {"A": {"B": 1.0}},
        },
    )
    monkeypatch.setattr(
        pdps,
        "get_chromosome_sizes_cached",
        lambda *_args, **_kwargs: ({"A": {"chr1": 1}, "B": {"chr2": 1}}, {"A": {"chr1": 100}, "B": {"chr2": 100}}),
    )
    monkeypatch.setattr(pdps, "calculate_pairwise_decay_sp1_vs_many", lambda *args, **kwargs: {"A": {"B": "dummy.tsv"}})
    called = {"pairwise": 0, "dispersion": 0}
    monkeypatch.setattr(
        pdps,
        "plot_pairwise_decay_sp1_vs_all",
        lambda *args, **kwargs: called.__setitem__("pairwise", called["pairwise"] + 1),
    )
    monkeypatch.setattr(
        pdps,
        "plot_dispersion_by_ALG",
        lambda *args, **kwargs: called.__setitem__("dispersion", called["dispersion"] + 1),
    )

    assert pdps.main([]) == 0
    assert called == {"pairwise": 1, "dispersion": 1}


def test_chromosome_alg_mapping_and_plotting_helpers(tmp_path: Path, monkeypatch):
    alg_db = tmp_path / "ALG.rbh"
    alg_db.write_text("rbh\n")
    species = "Species-101-GCA1"
    species_rbh = tmp_path / f"{species}_vs_ALG.rbh"
    pd.DataFrame(
        {
            "gene_group": ["Qa", "Qa", "Qb"],
            f"{species}_scaf": ["chr1", "chr1", "chr2"],
        }
    ).to_csv(species_rbh, sep="\t", index=False)

    monkeypatch.setattr(
        pdps.rbh_tools,
        "parse_ALG_rbh_to_colordf",
        lambda _path: pd.DataFrame({"ALGname": ["Qa", "Qb"], "Color": ["#aa0000", "#00aa00"]}),
    )
    chrom_to_alg, colors = pdps.get_chromosome_to_dominant_alg(species, str(tmp_path), str(alg_db), "ALG")
    assert chrom_to_alg["chr1"] == ("Qa", "#aa0000")
    assert colors["Qb"] == "#00aa00"

    missing_map, _ = pdps.get_chromosome_to_dominant_alg("Other", str(tmp_path), str(alg_db), "ALG")
    assert missing_map == {}

    decay_tsv = tmp_path / "decay.tsv"
    pd.DataFrame(
        {
            "sp1_scaf": ["chr1", "chr2"],
            "sp2_scaf": ["['x']", "['y']"],
            "sp1_scaf_genecount": [10, 20],
            "conserved": [8, 5],
            "scattered": [2, 15],
            "divergence_time": [100.0, 100.0],
            "fraction_conserved": [0.8, 0.25],
        }
    ).to_csv(decay_tsv, sep="\t", index=False)
    filestruct = {"Species-101-GCA1": {"Other-202-GCA1": str(decay_tsv)}}

    monkeypatch.setattr(pdps.odp_plot, "format_matplotlib", lambda: None)
    pdps.plot_pairwise_decay_sp1_vs_all(
        "Species-101-GCA1",
        filestruct,
        outdir=str(tmp_path / "overview"),
        bin_size=50,
        alg_rbh_file=str(alg_db),
        alg_rbh_dir=str(tmp_path),
        algname="ALG",
    )
    assert (tmp_path / "overview" / "Species-101-GCA1_decay_plot_vs_divergence_time.pdf").exists()

    pdps.plot_decay_twospecies(
        "Species-101-GCA1",
        "Other-202-GCA1",
        str(decay_tsv),
        {"chr1", "chr2"},
        str(tmp_path / "twospecies"),
    )
    assert (tmp_path / "twospecies" / "Species-101-GCA1_and_Other-202-GCA1_chromosome_conservation.pdf").exists()
