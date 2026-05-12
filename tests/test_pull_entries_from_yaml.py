from __future__ import annotations

import yaml

import pytest

from egt.pull_entries_from_yaml import (
    filter_yaml_entries,
    get_genus_species_list,
    get_yaml_entries,
    main,
    parse_args,
)


def test_parse_args_reads_keeps_and_yaml():
    args = parse_args(["-k", "keep1.txt", "keep2.txt", "-y", "config.yaml"])
    assert args.keeps == ["keep1.txt", "keep2.txt"]
    assert args.yaml == "config.yaml"


def test_get_genus_species_list_ignores_comments_and_blanks(tmp_path):
    keep = tmp_path / "keep.txt"
    keep.write_text("# comment\n\nSampleA\nHomo sapiens\n")
    assert get_genus_species_list([keep]) == ["SampleA", "Homo sapiens"]


def test_get_yaml_entries_loads_yaml(tmp_path):
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text("species:\n  SampleA:\n    genus: Homo\n    species: sapiens\n")
    loaded = get_yaml_entries(yaml_path)
    assert loaded["species"]["SampleA"]["genus"] == "Homo"


def test_filter_yaml_entries_matches_sample_and_binomial_name():
    yaml_dict = {
        "species": {
            "SampleA": {"genus": "Homo", "species": "sapiens"},
            "SampleB": {"genus": "Pan", "species": "troglodytes"},
            "SampleC": {"genus": "Mus", "species": "musculus"},
        }
    }

    filtered = filter_yaml_entries(yaml_dict, ["SampleA", "Pan troglodytes"])

    assert set(filtered) == {"SampleA", "SampleB"}


def test_filter_yaml_entries_requires_species_key():
    with pytest.raises(IOError, match="species field wasn't found"):
        filter_yaml_entries({}, ["SampleA"])


def test_main_prints_filtered_yaml(tmp_path, capsys):
    keep = tmp_path / "keep.txt"
    keep.write_text("Pan troglodytes\n")
    config = tmp_path / "config.yaml"
    config.write_text(
        "species:\n"
        "  SampleA:\n"
        "    genus: Homo\n"
        "    species: sapiens\n"
        "  SampleB:\n"
        "    genus: Pan\n"
        "    species: troglodytes\n"
    )

    rc = main(["-k", str(keep), "-y", str(config)])

    assert rc == 0
    out = yaml.safe_load(capsys.readouterr().out)
    assert set(out["species"]) == {"SampleB"}

