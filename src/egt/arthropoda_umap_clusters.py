"""Backward-compatible wrapper for Arthropoda-focused UMAP clustering."""
from __future__ import annotations

from .umap_taxonomy_clusters import (
    ARTHROPODA_TAXID,
    BASE_GENERIC_LABELS,
    choose_cluster_label,
    cluster_subset,
    parse_args as _parse_args,
    relabel_by_centroid,
    run,
    subset_by_taxid,
    taxon_aware_cluster_subset,
)


def parse_args(argv: list[str] | None = None):
    return _parse_args(
        argv,
        prog="egt arthropoda-umap-clusters",
        description=(
            "Cluster Arthropoda points in an existing UMAP dataframe without "
            "rerunning the manifold."
        ),
        default_subset_taxid=ARTHROPODA_TAXID,
        default_subset_label="Arthropoda",
        default_subset_slug="arthropoda",
        default_mixed_label="Mixed arthropods",
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    outputs = run(args)
    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
