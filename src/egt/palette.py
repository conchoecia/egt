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

from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Iterable
import warnings

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
    taxid_aliases: dict[int, int] = field(default_factory=dict)
    _canonicalizer: object | None = None

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

        canonicalizer = _get_shared_taxid_canonicalizer()
        by_taxid: dict[int, CladeColor] = {}
        taxid_aliases: dict[int, int] = {}
        for key, block in (data.get("clades") or {}).items():
            original_taxid = int(block["taxid"])
            canonical_taxid = (
                canonicalizer.canonicalize(original_taxid)
                if canonicalizer is not None
                else original_taxid
            )
            clade = CladeColor(
                taxid=canonical_taxid,
                label=str(block.get("label", key)),
                color=str(block["color"]).lower(),
                phylopic_uuid=(
                    str(block["phylopic_uuid"])
                    if block.get("phylopic_uuid")
                    else None
                ),
            )
            previous = by_taxid.get(canonical_taxid)
            if previous is not None and previous != clade:
                raise ValueError(
                    "Palette contains multiple entries that canonicalize to "
                    f"taxid {canonical_taxid}: {previous.label!r} and {clade.label!r}"
                )
            by_taxid[canonical_taxid] = clade
            taxid_aliases[original_taxid] = canonical_taxid
            taxid_aliases.setdefault(canonical_taxid, canonical_taxid)

        fb = data.get("fallback", {}) or {}
        fallback = CladeColor(
            taxid=-1,
            label=str(fb.get("label", "other")),
            color=str(fb.get("color", "#bfbfbf")).lower(),
            phylopic_uuid=None,
        )
        return cls(
            by_taxid=by_taxid,
            fallback=fallback,
            source_path=source,
            taxid_aliases=taxid_aliases,
            _canonicalizer=canonicalizer,
        )

    # ---------- lookups ----------

    def canonicalize_taxid(self, taxid: int | str | None) -> int | None:
        """Return the current NCBI taxid when a merged/obsolete id is given."""
        try:
            tid_int = int(taxid)
        except (TypeError, ValueError):
            return None

        if tid_int in self.taxid_aliases:
            return self.taxid_aliases[tid_int]

        canonicalizer = self._canonicalizer
        canonical_tid = (
            canonicalizer.canonicalize(tid_int)
            if canonicalizer is not None
            else tid_int
        )
        self.taxid_aliases[tid_int] = canonical_tid
        self.taxid_aliases.setdefault(canonical_tid, canonical_tid)
        return canonical_tid

    def has_taxid(self, taxid: int | str | None) -> bool:
        """Return True when ``taxid`` resolves to a palette-covered clade."""
        canonical_tid = self.canonicalize_taxid(taxid)
        return canonical_tid in self.by_taxid if canonical_tid is not None else False

    def for_taxid(self, taxid: int) -> CladeColor:
        """Exact match on a single taxid. Use `for_lineage` for species lookups."""
        canonical_tid = self.canonicalize_taxid(taxid)
        if canonical_tid is None:
            return self.fallback
        return self.by_taxid.get(canonical_tid, self.fallback)

    def for_lineage(self, lineage: Iterable[int]) -> CladeColor:
        """Resolve a species' color by walking its lineage most-specific first.

        `lineage` is an iterable of taxids from most-specific (the species
        itself) to least-specific (root). Returns the palette entry for
        the first ancestor that has one, falling back if nothing matches.
        """
        for tid in lineage:
            canonical_tid = self.canonicalize_taxid(tid)
            if canonical_tid is not None and canonical_tid in self.by_taxid:
                return self.by_taxid[canonical_tid]
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


class _NCBITaxidCanonicalizer:
    """Translate merged/obsolete NCBI taxids to their current ids."""

    def __init__(self, ncbi) -> None:
        self.ncbi = ncbi
        self._cache: dict[int, int] = {}

    def canonicalize(self, taxid: int) -> int:
        tid_int = int(taxid)
        if tid_int < 1:
            return tid_int
        cached = self._cache.get(tid_int)
        if cached is not None:
            return cached

        canonical_tid = tid_int
        translator = getattr(self.ncbi, "_translate_merged", None)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                if callable(translator):
                    _, merged = translator([tid_int])
                    canonical_tid = int(merged.get(tid_int, tid_int))
                else:
                    lineage = self.ncbi.get_lineage(tid_int) or []
                    if lineage:
                        canonical_tid = int(lineage[-1])
        except Exception:
            canonical_tid = tid_int

        self._cache[tid_int] = canonical_tid
        self._cache.setdefault(canonical_tid, canonical_tid)
        return canonical_tid


_SHARED_TAXID_CANONICALIZER: _NCBITaxidCanonicalizer | bool | None = None


def _get_shared_taxid_canonicalizer() -> _NCBITaxidCanonicalizer | None:
    """Best-effort lazy loader for NCBI merged-taxid translation."""
    global _SHARED_TAXID_CANONICALIZER

    if _SHARED_TAXID_CANONICALIZER is False:
        return None
    if _SHARED_TAXID_CANONICALIZER is not None:
        return _SHARED_TAXID_CANONICALIZER

    try:
        from ete4 import NCBITaxa  # type: ignore
    except Exception:
        _SHARED_TAXID_CANONICALIZER = False
        return None

    try:
        _SHARED_TAXID_CANONICALIZER = _NCBITaxidCanonicalizer(NCBITaxa())
    except Exception:
        _SHARED_TAXID_CANONICALIZER = False
        return None
    return _SHARED_TAXID_CANONICALIZER


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
