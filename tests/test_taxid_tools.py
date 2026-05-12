from __future__ import annotations

import pytest

from egt.taxid_tools import NCBI_taxid_to_taxdict


class FakeNCBI:
    def __init__(self, lineages=None, names=None):
        self.lineages = lineages or {}
        self.names = names or {}

    def get_lineage(self, taxid):
        return list(self.lineages.get(taxid, []))

    def get_taxid_translator(self, lineage):
        return {taxid: self.names[taxid] for taxid in lineage if taxid in self.names}


def test_taxdict_builds_expected_fields_for_integer_taxid():
    ncbi = FakeNCBI(
        lineages={9606: [1, 2759, 33208, 9606]},
        names={1: "root", 2759: "Eukaryota", 33208: "Metazoa", 9606: "Homo sapiens"},
    )

    entry = NCBI_taxid_to_taxdict(ncbi, 9606)

    assert entry["taxid"] == 9606
    assert entry["taxname"] == "Homo sapiens"
    assert entry["taxid_list"] == [1, 2759, 33208, 9606]
    assert entry["taxid_list_str"] == "1;2759;33208;9606"
    assert entry["taxname_list_str"] == "root;Eukaryota;Metazoa;Homo sapiens"
    assert "root (1)" in entry["level_1"]
    assert "Homo sapiens (9606)" in entry["printstring"]


def test_taxdict_accepts_numeric_string_taxid():
    ncbi = FakeNCBI(
        lineages={9606: [1, 9606]},
        names={1: "root", 9606: "Homo sapiens"},
    )

    entry = NCBI_taxid_to_taxdict(ncbi, "9606")

    assert entry["taxid"] == 9606
    assert entry["taxname"] == "Homo sapiens"


def test_taxdict_rejects_non_numeric_string():
    ncbi = FakeNCBI()

    with pytest.raises(ValueError, match="non-numeric character"):
        NCBI_taxid_to_taxdict(ncbi, "96O6")


def test_taxdict_rejects_unsupported_type():
    ncbi = FakeNCBI()

    with pytest.raises(ValueError, match="not a string or an integer"):
        NCBI_taxid_to_taxdict(ncbi, 96.06)


def test_taxdict_translates_legacy_taxid_and_preserves_original_terminal_id():
    translated = 3126489
    ncbi = FakeNCBI(
        lineages={translated: [1, 2759, translated]},
        names={1: "root", 2759: "Eukaryota", translated: "Ochlodes sylvanus"},
    )

    entry = NCBI_taxid_to_taxdict(ncbi, 876063)

    assert entry["taxid"] == translated
    assert entry["taxid_list"][-1] == 876063
    assert entry["taxname"] == "Ochlodes sylvanus"


def test_taxdict_rejects_empty_lineage():
    ncbi = FakeNCBI(lineages={9606: []}, names={})

    with pytest.raises(ValueError, match="lineage is empty"):
        NCBI_taxid_to_taxdict(ncbi, 9606)


def test_taxdict_rejects_empty_name_map():
    ncbi = FakeNCBI(lineages={9606: [1, 9606]}, names={})

    with pytest.raises(ValueError, match="names are empty"):
        NCBI_taxid_to_taxdict(ncbi, 9606)
