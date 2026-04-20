from __future__ import annotations

import argparse

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


def test_palette_lineage_falls_back_on_unknown_values():
    palette = load_palette()
    resolved = palette.for_lineage(["bad", None, 999999999])
    assert resolved == palette.fallback


def test_add_palette_argument_registers_default_dest():
    parser = argparse.ArgumentParser()
    add_palette_argument(parser)
    args = parser.parse_args([])
    assert hasattr(args, "palette")
    assert args.palette is None

