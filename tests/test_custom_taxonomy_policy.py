from __future__ import annotations

import pandas as pd
import pytest

from egt import phylotreeumap as ptu
from egt import plot_alg_fusions as paf
from egt.custom_taxonomy import (
    CustomTopologyWarning,
    EUMETAZOA_TAXID,
    apply_custom_animal_topology_to_taxid_lineage,
)


def test_custom_taxonomy_warns_when_eumetazoa_is_replaced():
    with pytest.warns(CustomTopologyWarning, match="Eumetazoa"):
        lineage = apply_custom_animal_topology_to_taxid_lineage([1, 33208, 6072, 33213, 9606])

    assert EUMETAZOA_TAXID not in lineage
    assert lineage == [1, 33208, -67, -68, 33213, 9606]


def test_custom_taxonomy_can_rewrite_without_warning(recwarn):
    lineage = apply_custom_animal_topology_to_taxid_lineage([1, 33208, 6072, 10197, 27923], warn=False)

    assert not [warning for warning in recwarn if issubclass(warning.category, CustomTopologyWarning)]
    assert lineage == [1, 33208, 10197, 27923]


def test_phylotreeumap_taxdict_warns_and_uses_manuscript_topology():
    class FakeNCBI:
        def get_lineage(self, taxid):
            return [1, 33208, 6072, 33213, taxid]

        def get_taxid_translator(self, lineage):
            return {
                1: "root",
                33208: "Metazoa",
                6072: "Eumetazoa",
                33213: "Bilateria",
                9606: "Homo sapiens",
            }

    with pytest.warns(CustomTopologyWarning, match="Eumetazoa"):
        entry = ptu.NCBI_taxid_to_taxdict(FakeNCBI(), 9606)

    assert "Eumetazoa" not in entry["taxname_list_str"]
    assert "6072" not in entry["taxid_list_str"]
    assert "Myriazoa;Parahoxozoa;Bilateria" in entry["taxname_list_str"]


def test_phylotreeumap_table_normalizer_warns_once_for_stale_eumetazoa(recwarn):
    stale = pd.DataFrame(
        {
            "taxid_list_str": [
                "1;33208;6072;10197;27923",
                "1;33208;6072;33213;9606",
            ],
            "taxname_list_str": [
                "root;Metazoa;Eumetazoa;Ctenophora;Mnemiopsis leidyi",
                "root;Metazoa;Eumetazoa;Bilateria;Homo sapiens",
            ],
            "taxid_list": ["[1, 33208, 6072, 10197, 27923]", "[1, 33208, 6072, 33213, 9606]"],
            "taxname_list": [
                "['root', 'Metazoa', 'Eumetazoa', 'Ctenophora', 'Mnemiopsis leidyi']",
                "['root', 'Metazoa', 'Eumetazoa', 'Bilateria', 'Homo sapiens']",
            ],
            "printstring": ["", ""],
        }
    )

    normalized = ptu._normalize_custom_taxonomy_columns(stale)
    topology_warnings = [warning for warning in recwarn if issubclass(warning.category, CustomTopologyWarning)]

    assert len(topology_warnings) == 1
    assert "Eumetazoa" not in ";".join(normalized["taxname_list_str"])
    assert "6072" not in ";".join(normalized["taxid_list_str"])


def test_plot_alg_fusions_no_custom_phylogeny_preserves_eumetazoa_without_warning(monkeypatch, recwarn):
    class FakeNCBI:
        def get_lineage(self, taxid):
            return [1, 33208, 6072, 33213, taxid]

    monkeypatch.setattr(paf, "NCBITaxa", lambda: FakeNCBI())

    taxidstrings = paf.taxids_to_taxidstringdict([9606], use_custom_phylogeny=False)

    assert taxidstrings[9606] == "1;33208;6072;33213;9606"
    assert not [warning for warning in recwarn if issubclass(warning.category, CustomTopologyWarning)]
