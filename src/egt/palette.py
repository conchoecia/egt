"""Paper-wide clade palette + resolver.

Every plotting subcommand that colors species or clades should go through
this module instead of carrying its own hardcoded hex table. The canonical
palette ships as `src/egt/data/paper_palette.yaml`; plotting CLIs accept
`--palette` to override.

Resolution rule: given a species' ancestry chain (root → species), the
color for that species is the palette entry for the *most specific*
ancestor that has an entry. E.g. a Teleostei species resolves to
Teleostei, not Vertebrata. This keeps sub-clade colors from being washed
out by parent-clade assignments.
"""
from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Iterable

try:
    import yaml  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "PyYAML is required for egt.palette — install via `pip install pyyaml`"
    ) from exc


@dataclass(frozen=True)
class CladeColor:
    taxid: int
    label: str
    color: str
    phylopic_uuid: str | None


@dataclass
class Palette:
    """Loaded palette indexed by taxid, with a fallback color."""

    by_taxid: dict[int, CladeColor]
    fallback: CladeColor
    source_path: Path | None

    # ---------- construction ----------

    @classmethod
    def from_yaml(cls, path: Path | str | None = None) -> "Palette":
        """Load a palette YAML.

        `path=None` loads the package default `src/egt/data/paper_palette.yaml`.
        """
        if path is None:
            with resources.files("egt.data").joinpath("paper_palette.yaml").open(
                "r", encoding="utf-8"
            ) as fh:
                data = yaml.safe_load(fh)
            source: Path | None = None
        else:
            path = Path(path)
            with path.open("r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh)
            source = path

        by_taxid: dict[int, CladeColor] = {}
        for key, block in (data.get("clades") or {}).items():
            taxid = int(block["taxid"])
            by_taxid[taxid] = CladeColor(
                taxid=taxid,
                label=str(block.get("label", key)),
                color=str(block["color"]).lower(),
                phylopic_uuid=(
                    str(block["phylopic_uuid"])
                    if block.get("phylopic_uuid")
                    else None
                ),
            )

        fb = data.get("fallback", {}) or {}
        fallback = CladeColor(
            taxid=-1,
            label=str(fb.get("label", "other")),
            color=str(fb.get("color", "#bfbfbf")).lower(),
            phylopic_uuid=None,
        )
        return cls(by_taxid=by_taxid, fallback=fallback, source_path=source)

    # ---------- lookups ----------

    def for_taxid(self, taxid: int) -> CladeColor:
        """Exact match on a single taxid. Use `for_lineage` for species lookups."""
        return self.by_taxid.get(int(taxid), self.fallback)

    def for_lineage(self, lineage: Iterable[int]) -> CladeColor:
        """Resolve a species' color by walking its lineage most-specific first.

        `lineage` is an iterable of taxids from most-specific (the species
        itself) to least-specific (root). Returns the palette entry for
        the first ancestor that has one, falling back if nothing matches.
        """
        for tid in lineage:
            try:
                tid_int = int(tid)
            except (TypeError, ValueError):
                continue
            if tid_int in self.by_taxid:
                return self.by_taxid[tid_int]
        return self.fallback

    def for_lineage_string(self, taxid_list_str: str, sep: str = ";") -> CladeColor:
        """Convenience for the 'taxid_list_str' column in sample dataframes.

        The string is assumed to be in ROOT-TO-LEAF order (as produced by
        `egt newick-to-common-ancestors`); we reverse it so the most
        specific clade wins.
        """
        if not taxid_list_str:
            return self.fallback
        chunks = [c.strip() for c in str(taxid_list_str).split(sep) if c.strip()]
        return self.for_lineage(reversed(chunks))

    # ---------- introspection ----------

    def __len__(self) -> int:
        return len(self.by_taxid)

    def items(self) -> Iterable[tuple[int, CladeColor]]:
        return self.by_taxid.items()


# Convenience module-level loaders ----------------------------------------

def load_palette(path: Path | str | None = None) -> Palette:
    """Top-level helper used by plotting CLIs."""
    return Palette.from_yaml(path)


def add_palette_argument(parser, dest: str = "palette") -> None:
    """Shared argparse helper so every plot CLI exposes `--palette`."""
    parser.add_argument(
        "--palette",
        dest=dest,
        default=None,
        help=(
            "Path to a paper-palette YAML. Defaults to the file bundled "
            "with egt (src/egt/data/paper_palette.yaml)."
        ),
    )
