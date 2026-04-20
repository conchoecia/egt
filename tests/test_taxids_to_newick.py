from __future__ import annotations

from pathlib import Path

import pytest

from egt import taxids_to_newick as ttn


class FakeNCBI:
    def __init__(self, lineages=None, ranks=None, names=None):
        self.lineages = lineages or {}
        self.ranks = ranks or {}
        self.names = names or {}

    def get_lineage(self, taxid):
        return self.lineages[taxid]

    def get_rank(self, lineage):
        return {tid: self.ranks.get(tid, "") for tid in lineage}

    def get_taxid_translator(self, taxids):
        return {taxid: self.names[taxid] for taxid in taxids if taxid in self.names}

    def get_topology(self, taxids):
        from ete4 import PhyloTree
        if len(taxids) == 1:
            node = PhyloTree()
            node.name = str(taxids[0])
            return node
        leaves = ",".join(str(tid) for tid in taxids)
        return PhyloTree(f"({leaves});", parser=1)


def test_parse_args_requires_one_input_source():
    args = ttn.parse_args(["-t", "taxids.txt", "-o", "tree.nwk"])
    assert args.taxid_file == "taxids.txt"
    assert args.output_file == "tree.nwk"
    assert args.custom_phylogeny is False


def test_is_subspecies_or_below_and_get_species_level_taxid():
    ncbi = FakeNCBI(
        lineages={10: [1, 2, 3, 10]},
        ranks={1: "root", 2: "genus", 3: "species", 10: "subspecies"},
    )
    assert ttn.is_subspecies_or_below(10, ncbi) is True
    assert ttn.get_species_level_taxid(10, ncbi) == 3
    assert ttn.get_species_level_taxid(999, FakeNCBI()) == 999


def test_read_taxids_from_config_converts_subspecies(tmp_path: Path, monkeypatch, capsys):
    config = tmp_path / "config.yaml"
    config.write_text(
        "species:\n"
        "  entryA:\n"
        "    taxid: 10\n"
        "  entryB:\n"
        "    taxid: 20\n"
        "  entryBad:\n"
        "    taxid: bad\n"
    )

    fake_ncbi = FakeNCBI(
        lineages={10: [1, 2, 3, 10], 20: [1, 20], 3: [1, 3]},
        ranks={1: "root", 2: "genus", 3: "species", 10: "subspecies", 20: "species"},
        names={10: "Name ten", 20: "Name twenty", 3: "Species three"},
    )
    monkeypatch.setattr(ttn, "NCBITaxa", lambda: fake_ncbi)

    cwd = Path.cwd()
    try:
        import os
        os.chdir(tmp_path)
        taxids = ttn.read_taxids_from_config(str(config))
    finally:
        os.chdir(cwd)

    assert taxids == {3, 20}
    assert (tmp_path / "subspecies_to_species_conversions.tsv").exists()
    captured = capsys.readouterr()
    assert "Extracted 2 unique species-level taxids" in captured.out


def test_export_timetree_list_skips_non_binomials(tmp_path: Path, capsys):
    ncbi = FakeNCBI(names={1: "Homo sapiens", 2: "Unclassified"})
    out = tmp_path / "species.txt"

    ttn.export_timetree_list([1, 2], ncbi, str(out))

    assert out.read_text() == "Homo sapiens\n"
    assert "Skipped 1 non-species-level taxa" in capsys.readouterr().out


def test_build_custom_topology_tree_groups_major_clades():
    ncbi = FakeNCBI(
        lineages={
            101: [1, 33208, 10197, 101],
            201: [1, 33208, 6040, 201],
            301: [1, 33208, 6073, 301],
            401: [1, 33208, 10226, 401],
            501: [1, 33208, 33213, 501],
        },
        names={
            101: "cteno one",
            201: "porifera one",
            301: "cnidaria one",
            401: "placozoa one",
            501: "bilateria one",
        },
    )

    tree = ttn.build_custom_topology_tree([101, 201, 301, 401, 501], ncbi)

    assert tree.name == "Metazoa[33208]"
