from __future__ import annotations

from pathlib import Path
import os

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


class FakeNode:
    def __init__(self, name="", children=None):
        self.name = name
        self.children = []
        if children:
            for child in children:
                self.add_child(child)

    @property
    def is_leaf(self):
        return len(self.children) == 0

    def add_child(self, child):
        self.children.append(child)
        return child

    def remove_child(self, child):
        self.children.remove(child)

    def traverse(self):
        yield self
        for child in self.children:
            yield from child.traverse()


def test_collapse_single_child_internal_nodes():
    tree = FakeNode(
        "root",
        [
            FakeNode("10"),
            FakeNode("inner", [FakeNode("20")]),
        ],
    )

    collapsed = ttn.collapse_single_child_internal_nodes(tree)

    assert collapsed == 1
    assert [child.name for child in tree.children] == ["10", "20"]


def test_parse_args_requires_one_input_source():
    args = ttn.parse_args(["-t", "taxids.txt", "-o", "tree.nwk"])
    assert args.taxid_file == "taxids.txt"
    assert args.output_file == "tree.nwk"
    assert args.custom_phylogeny is False
    assert args.preserve_single_child_internal_nodes is False


def test_parse_args_preserve_single_child_internal_nodes_flag():
    args = ttn.parse_args(["-t", "taxids.txt", "--preserve-single-child-internal-nodes"])
    assert args.taxid_file == "taxids.txt"
    assert args.preserve_single_child_internal_nodes is True


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


def test_build_subtree_with_labels_names_internal_nodes():
    class SubtreeNCBI(FakeNCBI):
        def get_topology(self, taxids):
            return FakeNode(
                "",
                [
                    FakeNode(str(taxids[0])),
                    FakeNode("", [FakeNode(str(taxids[1])), FakeNode(str(taxids[2]))]),
                ],
            )

    ncbi = SubtreeNCBI(
        lineages={
            101: [1, 10, 101],
            102: [1, 10, 20, 102],
            103: [1, 10, 20, 103],
            10: [1, 10],
            20: [1, 10, 20],
        },
        names={10: "Major Clade", 20: "Inner Node"},
    )

    tree = ttn.build_subtree_with_labels([101, 102, 103], ncbi, 10, "Major")

    assert tree.name == "Major[10]"
    internal_names = [node.name for node in tree.traverse() if not node.is_leaf]
    assert "Inner_Node[20]" in internal_names


def test_build_custom_topology_tree_adds_other_taxa_root():
    ncbi = FakeNCBI(
        lineages={
            101: [1, 33208, 10197, 101],
            999: [1, 2759, 999],
        },
        names={101: "cteno", 999: "other"},
    )

    tree = ttn.build_custom_topology_tree([101, 999], ncbi)

    assert tree.name == "root"
    assert len(tree.children) == 2


def test_main_builds_newick_from_taxid_file(tmp_path: Path, monkeypatch):
    class MainNCBI(FakeNCBI):
        def get_topology(self, taxids):
            return FakeNode(
                "root node",
                [
                    FakeNode(str(taxids[0])),
                    FakeNode("", [FakeNode("20")]),
                ],
            )

    ncbi = MainNCBI(
        lineages={10: [1, 10], 20: [1, 20], 5: [1, 5], 6: [1, 6]},
        names={5: "Tax five", 6: "Tax six", 10: "Alpha beta", 20: "Gamma delta"},
    )
    monkeypatch.setattr(ttn, "NCBITaxa", lambda: ncbi)
    monkeypatch.setattr(ttn, "is_subspecies_or_below", lambda taxid, _ncbi: taxid == 10)
    monkeypatch.setattr(ttn, "get_species_level_taxid", lambda taxid, _ncbi: 5 if taxid == 10 else taxid)

    export_calls = []

    def fake_export(taxids, _ncbi, outfile):
        export_calls.append((tuple(taxids), Path(outfile).name))
        Path(outfile).write_text("ok\n")

    monkeypatch.setattr(ttn, "export_timetree_list", fake_export)

    taxid_file = tmp_path / "taxids.txt"
    taxid_file.write_text("10\n20\nnonnumeric\n")
    output = tmp_path / "tree.nwk"

    cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        assert ttn.main(["-t", str(taxid_file), "-o", str(output)]) == 0
    finally:
        os.chdir(cwd)

    newick = output.read_text()
    assert "'Tax_five[5]'" in newick
    assert "'Gamma_delta[20]'" in newick
    assert export_calls[0][0] == (5, 20)
    assert (tmp_path / "subspecies_to_species_conversions.tsv").exists()
    assert (tmp_path / "tree.log").exists()


def test_main_preserves_single_child_internal_nodes(tmp_path: Path, monkeypatch):
    class MainNCBI(FakeNCBI):
        def get_topology(self, taxids):
            return FakeNode(
                "root node",
                [
                    FakeNode(str(taxids[0])),
                    FakeNode("", [FakeNode("20")]),
                ],
            )

    ncbi = MainNCBI(
        lineages={10: [1, 10], 20: [1, 20]},
        names={10: "Alpha beta", 20: "Gamma delta"},
    )
    monkeypatch.setattr(ttn, "NCBITaxa", lambda: ncbi)
    monkeypatch.setattr(ttn, "is_subspecies_or_below", lambda taxid, _ncbi: False)
    monkeypatch.setattr(ttn, "export_timetree_list", lambda *_args, **_kwargs: None)

    taxid_file = tmp_path / "taxids.txt"
    taxid_file.write_text("10\n20\n")
    output = tmp_path / "tree.nwk"

    cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        assert ttn.main(
            [
                "-t",
                str(taxid_file),
                "-o",
                str(output),
                "--preserve-single-child-internal-nodes",
            ]
        ) == 0
    finally:
        os.chdir(cwd)

    newick = output.read_text()
    assert "('Gamma_delta[20]')" in newick


def test_main_uses_custom_phylogeny_and_custom_timetree_output(tmp_path: Path, monkeypatch):
    tree = FakeNode("root", [FakeNode("101"), FakeNode("201")])
    ncbi = FakeNCBI(
        lineages={101: [1, 101], 201: [1, 201]},
        names={101: "One taxon", 201: "Two taxon"},
    )
    monkeypatch.setattr(ttn, "NCBITaxa", lambda: ncbi)
    monkeypatch.setattr(ttn, "read_taxids_from_config", lambda _cfg: {101, 201})
    monkeypatch.setattr(ttn, "build_custom_topology_tree", lambda taxids, _ncbi: tree)

    exported = []
    monkeypatch.setattr(
        ttn,
        "export_timetree_list",
        lambda taxids, _ncbi, outfile: exported.append((tuple(taxids), Path(outfile).name)),
    )

    config = tmp_path / "cfg.yaml"
    config.write_text("species: {}\n")
    out = tmp_path / "custom.newick"
    custom_species = tmp_path / "species_custom.txt"

    assert ttn.main(
        [
            "-c",
            str(config),
            "-o",
            str(out),
            "--custom_phylogeny",
            "--timetree_list",
            str(custom_species),
        ]
    ) == 0

    assert out.exists()
    assert exported == [((101, 201), "species_list.txt"), ((101, 201), "species_custom.txt")]
