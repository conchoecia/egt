from __future__ import annotations

import argparse
from importlib import resources

from egt import palette as palette_module
from egt.palette import CladeColor, Palette, add_palette_argument, load_palette


def test_load_palette_default():
    palette = load_palette()
    assert isinstance(palette, Palette)
    assert len(palette) > 0
    assert palette.source_path is None


def test_load_palette_from_yaml_override(tmp_path):
    palette_yaml = tmp_path / "palette.yaml"
    palette_yaml.write_text(
        """
schema_version: 1
clades:
  teleostei:
    taxid: 32443
    label: "Teleostei"
    color: "#ABCDEF"
    phylopic_uuid: "uuid-1"
fallback:
  label: "fallback"
  color: "#123456"
""".lstrip()
    )

    palette = load_palette(palette_yaml)

    assert palette.source_path == palette_yaml
    assert palette.for_taxid(32443) == CladeColor(
        taxid=32443,
        label="Teleostei",
        color="#abcdef",
        phylopic_uuid="uuid-1",
    )
    assert palette.fallback.color == "#123456"


def test_palette_lineage_resolution_prefers_most_specific():
    palette = load_palette()
    # Most-specific-first lineage: a Drosophilinae species inside Diptera inside Panarthropoda.
    resolved = palette.for_lineage([43845, 7147, 88770, 33213])
    assert resolved.taxid == 43845
    assert resolved.label == "Drosophilinae"


def test_palette_lineage_string_reverses_root_to_leaf_order():
    palette = load_palette()
    resolved = palette.for_lineage_string("33213;88770;7147;43845")
    assert resolved.taxid == 43845


def test_simple_palette_keeps_requested_manuscript_clades(monkeypatch):
    monkeypatch.setattr(palette_module, "_get_shared_taxid_canonicalizer", lambda: None)
    palette_path = resources.files("egt.data").joinpath("paper_palette_simple.yaml")
    palette = load_palette(palette_path)

    trematode = palette.for_lineage_string("1;33208;-67;-68;33213;33317;2697495;6178")
    lepidoptera = palette.for_lineage_string("1;33208;-67;-68;33213;33317;88770;6656;6960;50557;7088")
    ovis = palette.for_lineage_string("1;33208;-67;-68;33213;33511;7711;32523;40674;9935")
    sheep = palette.for_lineage_string("1;33208;-67;-68;33213;33511;7711;32523;40674;9443;9940")
    pig = palette.for_lineage_string("1;33208;-67;-68;33213;33511;7711;32523;40674;9443;9823")
    muroidea = palette.for_lineage_string("1;33208;-67;-68;33213;33511;7711;32523;40674;337687;10066")

    assert trematode.taxid == 6178
    assert trematode.label == "Trematoda"
    assert lepidoptera.taxid == 7088
    assert lepidoptera.label == "Lepidoptera (other)"
    assert ovis.taxid == 9935
    assert ovis.label == "Ovis"
    assert ovis.color == "#6f9fbd"
    assert sheep.taxid == 9940
    assert sheep.label == "Sheep"
    assert pig.taxid == 9823
    assert pig.label == "Pig"
    assert muroidea.taxid == 337687
    assert muroidea.label == "Muroidea"
    assert muroidea.color == "#9b5c8f"


def test_palette_lineage_falls_back_on_unknown_values():
    palette = load_palette()
    resolved = palette.for_lineage(["bad", None, 999999999])
    assert resolved == palette.fallback


def test_palette_canonicalizes_merged_taxids(monkeypatch, tmp_path):
    class FakeCanonicalizer:
        def canonicalize(self, taxid):
            return {50: 100}.get(int(taxid), int(taxid))

    monkeypatch.setattr(
        palette_module,
        "_get_shared_taxid_canonicalizer",
        lambda: FakeCanonicalizer(),
    )

    palette_yaml = tmp_path / "palette.yaml"
    palette_yaml.write_text(
        """
schema_version: 1
clades:
  merged_target:
    taxid: 100
    label: "Merged Target"
    color: "#ABCDEF"
    phylopic_uuid: null
fallback:
  label: "fallback"
  color: "#123456"
""".lstrip()
    )

    palette = load_palette(palette_yaml)

    assert palette.has_taxid(50) is True
    assert palette.for_taxid(50).taxid == 100
    assert palette.for_lineage([50, 1]).taxid == 100
    assert palette.for_lineage_string("1;50").taxid == 100


def test_add_palette_argument_registers_default_dest():
    parser = argparse.ArgumentParser()
    add_palette_argument(parser)
    args = parser.parse_args([])
    assert hasattr(args, "palette")
    assert args.palette is None
