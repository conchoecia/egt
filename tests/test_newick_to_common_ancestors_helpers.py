from __future__ import annotations

from pathlib import Path

import yaml
from ete4 import Tree

from egt import newick_to_common_ancestors as n2ca


class FakeTaxNode:
    def __init__(self, nodeage, children):
        self.nodeage = nodeage
        self.children = children


class FakeTaxIDTree:
    def __init__(self):
        self.nodes = {
            1: FakeTaxNode(100.0, [2, 3]),
            2: FakeTaxNode(None, []),
            3: FakeTaxNode(40.0, []),
        }

    def find_LCA(self, taxid1, taxid2):
        return 1


class FakeNCBI:
    def __init__(self, mapping):
        self.mapping = mapping

    def get_name_translator(self, names):
        result = {}
        for name in names:
            if name in self.mapping:
                result[name] = [self.mapping[name]]
            else:
                raise KeyError(name)
        return result


def test_create_directories_recursive_notouch_handles_file_path(tmp_path: Path):
    target = tmp_path / "a" / "b" / "out.tsv"
    rc = n2ca.create_directories_recursive_notouch(str(target))
    assert rc == 0
    assert (tmp_path / "a").is_dir()
    assert (tmp_path / "a" / "b").is_dir()


def test_get_lineage_and_get_all_lineages():
    tree = Tree("((A:1,B:1):2,C:3);")
    lineages = n2ca.get_all_lineages(tree)
    assert [node.name for node in lineages["A"]] == [None, "A"]
    assert [node.name for node in n2ca.get_lineage(tree, lineages["B"][-1])] == [None, "B"]


def test_find_common_ancestor_age_uses_unique_branch_lengths():
    tree = Tree("((A:1,B:1):2,C:4);")
    lineages = n2ca.get_all_lineages(tree)
    ancestor, age = n2ca.find_common_ancestor_age(lineages["A"], lineages["B"])
    assert ancestor is not None
    assert age == 1.0


def test_annotate_custom_tree_with_timetree_ages_supports_bracket_taxids():
    custom = Tree("(A_species,B_species);")
    timetree = Tree("(A_species:1,B_species:1);")

    divergences, tt_map, custom_map = n2ca.annotate_custom_tree_with_timetree_ages(
        custom, timetree, FakeNCBI({"A species": 1, "B species": 2})
    )

    assert divergences[(1, 2)] == 1.0
    assert tt_map["A_species"] == 1
    assert custom_map["B_species"] == 2


def test_extract_timetree_root_age_as_metazoa():
    timetree = Tree("((A:1,B:1):2,C:4);")
    assert n2ca.extract_timetree_root_age_as_metazoa(timetree) == 3.0


def test_get_divergence_time_all_vs_all_taxidtree_and_report(tmp_path: Path):
    tree = FakeTaxIDTree()

    yielded = list(n2ca.get_divergence_time_all_vs_all_taxidtree(tree))
    assert yielded == [(2, 3, 100.0)]

    output = n2ca.report_divergence_time_all_vs_all(tree, str(tmp_path / "ages"))
    assert output == {(2, 3): 100.0}
    assert "2\t3\t100.0" in (tmp_path / "ages.divergence_times.txt").read_text()


def test_convert_ncbi_entry_to_dict():
    converted = n2ca.convert_ncbi_entry_to_dict(
        {
            "TaxId": "9606",
            "ScientificName": "Homo sapiens",
            "Lineage": "root;Eukaryota",
            "LineageEx": [{"TaxId": "1", "ScientificName": "root", "Rank": "no rank"}],
        }
    )
    assert converted["TaxID"] == 9606
    assert converted["LineageEx"][0]["TaxID"] == 1


def test_taxinfo_download_or_load_downloads_and_reuses_file(tmp_path: Path, monkeypatch):
    target = tmp_path / "tax" / "info.yaml"
    monkeypatch.setattr(n2ca, "get_taxonomy_info", lambda name: {"ScientificName": name})
    monkeypatch.setattr(n2ca.time, "sleep", lambda *_args, **_kwargs: None)

    assert n2ca.taxinfo_download_or_load("Homo sapiens", str(target)) == 0
    assert yaml.safe_load(target.read_text()) == {"ScientificName": "Homo sapiens"}
    assert n2ca.taxinfo_download_or_load("Homo sapiens", str(target)) == 0


def test_taxinfo_download_or_load_handles_empty_cached_file(tmp_path: Path):
    target = tmp_path / "tax.yaml"
    target.write_text("")
    assert n2ca.taxinfo_download_or_load("ignored", str(target)) == 1
    assert not target.exists()


def test_yaml_file_legal_checks_nonempty_file(tmp_path: Path):
    good = tmp_path / "good.yaml"
    bad = tmp_path / "bad.yaml"
    good.write_text("a: 1\n")
    bad.write_text("")
    assert n2ca.yaml_file_legal(good) is True
    assert n2ca.yaml_file_legal(bad) is False
