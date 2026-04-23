"""Cluster points inside an existing UMAP embedding using taxonomy-aware labels.

This helper does not rerun UMAP. It takes a fixed 2D embedding already on disk,
optionally subsets rows by a lineage taxid, fits KMeans on ``UMAP1``/``UMAP2``,
and writes cluster plots plus summary tables. Cluster labels are chosen from the
full lineage of each point rather than from a fixed taxonomic depth.
"""
from __future__ import annotations

import argparse
import ast
import math
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors


ARTHROPODA_TAXID = 6656
GREY = "#D5D7DF"
CLUSTER_COLORS = [
    "#d1495b",
    "#edae49",
    "#66a182",
    "#00798c",
    "#4c5c68",
    "#9c6644",
    "#6d597a",
    "#b56576",
    "#457b9d",
    "#588157",
    "#f4a261",
    "#2a9d8f",
]
BASE_GENERIC_LABELS = {
    "root",
    "cellular organisms",
    "Eukaryota",
    "Opisthokonta",
    "Metazoa",
    "Eumetazoa",
    "Bilateria",
    "Protostomia",
    "Deuterostomia",
}


def parse_args(
    argv: list[str] | None = None,
    *,
    prog: str = "egt umap-taxonomy-clusters",
    description: str | None = None,
    default_subset_taxid: int | None = None,
    default_subset_label: str | None = None,
    default_subset_slug: str | None = None,
    default_mixed_label: str = "Mixed cluster",
) -> argparse.Namespace:
    if default_subset_taxid is None:
        subset_help = (
            "Optional lineage taxid to subset before clustering. "
            "If omitted, cluster the full UMAP."
        )
    else:
        subset_help = (
            "Lineage taxid to subset before clustering "
            f"(default: {default_subset_taxid})."
        )
    parser = argparse.ArgumentParser(
        prog=prog,
        description=description or (
            "Cluster points in an existing UMAP dataframe without rerunning the manifold."
        ),
    )
    parser.add_argument(
        "--df",
        required=True,
        help="Input dataframe TSV with UMAP1/UMAP2 and lineage columns.",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Directory for plots and TSV outputs.",
    )
    parser.add_argument(
        "--ks",
        nargs="+",
        type=int,
        default=[6, 8],
        help="KMeans cluster counts to emit (default: 6 8).",
    )
    parser.add_argument(
        "--subset-taxid",
        "--taxid",
        type=int,
        dest="subset_taxid",
        default=default_subset_taxid,
        help=subset_help,
    )
    parser.add_argument(
        "--subset-label",
        type=str,
        default=default_subset_label,
        help=(
            "Optional human-readable label for the clustered subset, used in plot titles. "
            "If omitted, inferred from the taxid when possible."
        ),
    )
    parser.add_argument(
        "--subset-slug",
        type=str,
        default=default_subset_slug,
        help=(
            "Optional output filename token for the clustered subset. "
            "If omitted, derived from the subset label or taxid."
        ),
    )
    parser.set_defaults(mixed_label=default_mixed_label)
    parser.add_argument(
        "--random-state",
        type=int,
        default=0,
        help="Random seed for KMeans (default: 0).",
    )
    parser.add_argument(
        "--background-alpha",
        type=float,
        default=0.18,
        help="Alpha for non-target background points in the context panel.",
    )
    parser.add_argument(
        "--taxon-aware",
        action="store_true",
        help=(
            "Run taxon-aware overclustering: fit many microclusters in UMAP space, "
            "then merge adjacent microclusters when they share a sufficiently "
            "specific lineage signal."
        ),
    )
    parser.add_argument(
        "--taxon-aware-micro-k",
        type=int,
        default=32,
        help="Microcluster count for --taxon-aware mode (default: 32).",
    )
    parser.add_argument(
        "--taxon-aware-neighbors",
        type=int,
        default=3,
        help="How many nearest microclusters to consider for merging (default: 3).",
    )
    parser.add_argument(
        "--taxon-aware-min-fraction",
        type=float,
        default=0.55,
        help=(
            "Minimum within-microcluster fraction required to call a taxon as a "
            "consensus label (default: 0.55)."
        ),
    )
    parser.add_argument(
        "--taxon-aware-min-common-depth",
        type=int,
        default=20,
        help=(
            "Minimum shared lineage depth for merging two nearby microclusters when "
            "their consensus labels are not identical (default: 20)."
        ),
    )
    parser.add_argument(
        "--taxon-aware-max-dist-factor",
        type=float,
        default=1.6,
        help=(
            "Distance cutoff multiplier for taxon-aware microcluster merging, based "
            "on the median nearest-neighbor centroid distance (default: 1.6)."
        ),
    )
    return parser.parse_args(argv)


def _parse_lineage_taxids(row: pd.Series) -> list[int]:
    if "taxid_list_str" in row.index and pd.notna(row["taxid_list_str"]):
        return [int(x) for x in str(row["taxid_list_str"]).split(";") if x.strip()]
    if "taxid_list" in row.index and pd.notna(row["taxid_list"]):
        value = row["taxid_list"]
        if isinstance(value, list):
            return [int(x) for x in value]
        if isinstance(value, str):
            return [int(x) for x in ast.literal_eval(value)]
    return []


def subset_by_taxid(df: pd.DataFrame, taxid: int) -> pd.DataFrame:
    if "taxid_list_str" in df.columns:
        pattern = rf"(?:^|;){int(taxid)}(?:;|$)"
        mask = df["taxid_list_str"].astype(str).str.contains(pattern, regex=True)
        return df[mask].copy()
    mask = df.apply(lambda row: int(taxid) in _parse_lineage_taxids(row), axis=1)
    return df[mask].copy()


def _lineage_names(row: pd.Series) -> list[str]:
    if "taxname_list_str" in row.index and pd.notna(row["taxname_list_str"]):
        return [x.strip() for x in str(row["taxname_list_str"]).split(";") if x.strip()]
    if "taxname_list" in row.index and pd.notna(row["taxname_list"]):
        value = row["taxname_list"]
        if isinstance(value, list):
            return [str(x).strip() for x in value]
        if isinstance(value, str):
            return [str(x).strip() for x in ast.literal_eval(value)]
    return []


def _rank_mode(df: pd.DataFrame, depth: int) -> tuple[str | None, int]:
    counts: Counter[str] = Counter()
    for _, row in df.iterrows():
        names = _lineage_names(row)
        if len(names) > depth:
            counts[names[depth]] += 1
    if not counts:
        return None, 0
    return counts.most_common(1)[0]


def _normalize_slug(text: str | None) -> str | None:
    if not text:
        return None
    slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in text.strip())
    slug = "_".join(part for part in slug.split("_") if part)
    return slug or None


def _infer_subset_name_from_taxid(df: pd.DataFrame, taxid: int) -> str | None:
    for _, row in df.iterrows():
        taxids = _parse_lineage_taxids(row)
        if taxid not in taxids:
            continue
        names = _lineage_names(row)
        if len(names) != len(taxids):
            continue
        idx = taxids.index(taxid)
        return names[idx]
    return None


def _generic_labels_for_subset(
    df: pd.DataFrame,
    subset_taxid: int | None = None,
    extra_generic_labels: set[str] | None = None,
) -> set[str]:
    labels = set(BASE_GENERIC_LABELS)
    if extra_generic_labels:
        labels.update(extra_generic_labels)
    if subset_taxid is None:
        return labels

    for _, row in df.iterrows():
        taxids = _parse_lineage_taxids(row)
        if subset_taxid not in taxids:
            continue
        names = _lineage_names(row)
        if len(names) != len(taxids):
            continue
        idx = taxids.index(subset_taxid)
        labels.update(name for name in names[: idx + 1] if name)
        break
    return labels


def _lineage_name_depth_counts(df: pd.DataFrame) -> Counter[tuple[int, str]]:
    counts: Counter[tuple[int, str]] = Counter()
    for _, row in df.iterrows():
        for depth, name in enumerate(_lineage_names(row)):
            if not name:
                continue
            counts[(depth, name)] += 1
    return counts


def _terminal_lineage_name_depth_counts(df: pd.DataFrame) -> Counter[tuple[int, str]]:
    counts: Counter[tuple[int, str]] = Counter()
    for _, row in df.iterrows():
        names = _lineage_names(row)
        if not names:
            continue
        depth = len(names) - 1
        counts[(depth, names[-1])] += 1
    return counts


def _max_lineage_depth(df: pd.DataFrame) -> int:
    max_depth = 0
    for _, row in df.iterrows():
        names = _lineage_names(row)
        if names:
            max_depth = max(max_depth, len(names) - 1)
    return max_depth


def _strip_rich(label: str | None) -> str | None:
    if not label:
        return label
    return label[:-5] if label.endswith("-rich") else label


def _informative_signature(
    df: pd.DataFrame,
    *,
    generic_labels: set[str] | None = None,
    limit: int = 6,
) -> str:
    labels_to_skip = generic_labels or BASE_GENERIC_LABELS
    counts = _lineage_name_depth_counts(df)
    ranked: list[tuple[int, int, str]] = []
    for (depth, label), count in counts.items():
        if (
            not label
            or label in labels_to_skip
            or _looks_like_species_name(label)
            or _looks_like_placeholder_label(label)
        ):
            continue
        ranked.append((count, depth, label))
    ranked.sort(key=lambda item: (-item[0], -item[1], item[2]))
    seen: set[str] = set()
    parts: list[str] = []
    for count, _depth, label in ranked:
        if label in seen:
            continue
        seen.add(label)
        parts.append(f"{label}:{count}")
        if len(parts) >= limit:
            break
    return "; ".join(parts)


def _deepest_common_informative_name(
    path_a: list[str | None],
    path_b: list[str | None],
    generic_labels: set[str] | None = None,
) -> tuple[str | None, int]:
    labels_to_skip = generic_labels or BASE_GENERIC_LABELS
    common_label: str | None = None
    common_depth = -1
    for depth, (name_a, name_b) in enumerate(zip(path_a, path_b)):
        if name_a != name_b:
            break
        if name_a and name_a not in labels_to_skip:
            common_label = name_a
            common_depth = depth
    return common_label, common_depth


def _looks_like_species_name(label: str | None) -> bool:
    if not label:
        return False
    parts = str(label).strip().split()
    if len(parts) != 2:
        return False
    genus, epithet = parts
    if not genus or not epithet:
        return False
    return genus[0].isupper() and genus[1:].islower() and epithet.islower()


def _looks_like_placeholder_label(label: str | None) -> bool:
    if not label:
        return False
    text = str(label).strip()
    low = text.lower()
    if low in {"species", "unknown", "unclassified", "uncultured"}:
        return True
    if low.startswith(("sample", "cluster", "placeholder")):
        return True
    if low.startswith("sp") and len(text) <= 6 and text[2:].replace("_", "").replace("-", "").isalnum():
        return True
    return False


def _cluster_taxonomy_info(
    df: pd.DataFrame,
    min_fraction: float = 0.55,
    generic_labels: set[str] | None = None,
) -> dict[str, object]:
    labels_to_skip = generic_labels or BASE_GENERIC_LABELS
    n = len(df)
    max_depth = _max_lineage_depth(df)
    mode_path: list[str | None] = []
    best_label: str | None = None
    best_depth = -1
    best_count = 0
    best_fraction = 0.0

    for depth in range(max_depth + 1):
        label, count = _rank_mode(df, depth)
        mode_path.append(label)

    for depth in range(max_depth, -1, -1):
        label, count = _rank_mode(df, depth)
        if (
            not label
            or label in labels_to_skip
            or _looks_like_species_name(label)
            or _looks_like_placeholder_label(label)
        ):
            continue
        fraction = count / n if n else 0.0
        if fraction >= min_fraction:
            best_label = label
            best_depth = depth
            best_count = count
            best_fraction = fraction
            break

    return {
        "mode_path": mode_path,
        "consensus_label": best_label,
        "consensus_depth": best_depth,
        "consensus_count": best_count,
        "consensus_fraction": best_fraction,
    }


def choose_cluster_label(
    df: pd.DataFrame,
    global_counts: Counter[tuple[int, str]] | None = None,
    global_total: int | None = None,
    min_fraction: float = 0.55,
    strong_fraction: float = 0.75,
    enriched_min_fraction: float = 0.18,
    enriched_min_fold: float = 2.5,
    enriched_min_count: int = 3,
    generic_labels: set[str] | None = None,
    mixed_label: str = "Mixed cluster",
) -> str:
    labels_to_skip = generic_labels or BASE_GENERIC_LABELS
    n = len(df)
    dynamic_min_count = max(enriched_min_count, int(math.ceil(n * 0.06)))
    local_counts = _lineage_name_depth_counts(df)

    best_candidate: tuple[int, float, float, int, str] | None = None
    for (depth, label), count in local_counts.items():
        if (
            not label
            or label in labels_to_skip
            or _looks_like_species_name(label)
            or _looks_like_placeholder_label(label)
            or count < dynamic_min_count
        ):
            continue
        fraction = count / n if n else 0.0
        if fraction < enriched_min_fraction:
            continue

        if global_counts is not None and global_total:
            global_fraction = global_counts.get((depth, label), 0) / global_total
            if global_fraction <= 0:
                fold = float("inf")
            else:
                fold = fraction / global_fraction
        else:
            fold = 1.0

        if fold < enriched_min_fold:
            continue

        candidate = (depth, fold, fraction, count, label)
        if best_candidate is None or candidate > best_candidate:
            best_candidate = candidate

    if best_candidate is not None:
        _, _fold, fraction, _count, label = best_candidate
        if fraction >= strong_fraction:
            return str(label)
        return f"{label}-rich"

    taxinfo = _cluster_taxonomy_info(
        df,
        min_fraction=min_fraction,
        generic_labels=labels_to_skip,
    )
    label = taxinfo["consensus_label"]
    fraction = float(taxinfo["consensus_fraction"])
    if label:
        if fraction >= strong_fraction:
            return str(label)
        if fraction >= min_fraction:
            return f"{label}-rich"

    return mixed_label


def relabel_by_centroid(labels: np.ndarray, centers: np.ndarray) -> np.ndarray:
    order = sorted(
        range(len(centers)),
        key=lambda idx: (float(centers[idx][0]), float(centers[idx][1])),
    )
    mapping = {old: new for new, old in enumerate(order)}
    return np.array([mapping[int(label)] for label in labels], dtype=int)


def cluster_subset(
    df: pd.DataFrame,
    k: int,
    random_state: int = 0,
    min_fraction: float = 0.55,
    generic_labels: set[str] | None = None,
    mixed_label: str = "Mixed cluster",
) -> pd.DataFrame:
    global_counts = _lineage_name_depth_counts(df)
    global_total = len(df)
    labels_to_skip = generic_labels or BASE_GENERIC_LABELS
    model = KMeans(n_clusters=k, random_state=random_state, n_init=20)
    raw_labels = model.fit_predict(df[["UMAP1", "UMAP2"]].to_numpy())
    labels = relabel_by_centroid(raw_labels, model.cluster_centers_)
    clustered = df.copy()
    clustered["cluster"] = labels
    summary_rows = []
    for cluster_id in sorted(clustered["cluster"].unique()):
        sub = clustered[clustered["cluster"] == cluster_id]
        label = choose_cluster_label(
            sub,
            global_counts=global_counts,
            global_total=global_total,
            min_fraction=min_fraction,
            generic_labels=labels_to_skip,
            mixed_label=mixed_label,
        )
        taxinfo = _cluster_taxonomy_info(
            sub,
            min_fraction=min_fraction,
            generic_labels=labels_to_skip,
        )
        summary_rows.append(
            {
                "cluster": cluster_id,
                "cluster_label": label,
                "n": len(sub),
                "center_umap1": float(sub["UMAP1"].mean()),
                "center_umap2": float(sub["UMAP2"].mean()),
                "umap1_min": float(sub["UMAP1"].min()),
                "umap1_max": float(sub["UMAP1"].max()),
                "umap2_min": float(sub["UMAP2"].min()),
                "umap2_max": float(sub["UMAP2"].max()),
                "top_depth19": _rank_mode(sub, 19)[0],
                "top_depth19_n": _rank_mode(sub, 19)[1],
                "top_depth20": _rank_mode(sub, 20)[0],
                "top_depth20_n": _rank_mode(sub, 20)[1],
                "top_depth21": _rank_mode(sub, 21)[0],
                "top_depth21_n": _rank_mode(sub, 21)[1],
                "consensus_label": taxinfo["consensus_label"],
                "consensus_depth": taxinfo["consensus_depth"],
                "consensus_fraction": taxinfo["consensus_fraction"],
                "major_signature": _informative_signature(
                    sub,
                    generic_labels=labels_to_skip,
                ),
            }
        )
        clustered.loc[sub.index, "cluster_label"] = label
    summary = pd.DataFrame(summary_rows).sort_values("cluster").reset_index(drop=True)
    return clustered, summary


def taxon_aware_cluster_subset(
    df: pd.DataFrame,
    micro_k: int = 32,
    random_state: int = 0,
    min_fraction: float = 0.55,
    min_common_depth: int = 20,
    n_neighbors: int = 3,
    max_dist_factor: float = 1.6,
    generic_labels: set[str] | None = None,
    mixed_label: str = "Mixed cluster",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    global_counts = _lineage_name_depth_counts(df)
    global_total = len(df)
    labels_to_skip = generic_labels or BASE_GENERIC_LABELS
    micro, micro_summary = cluster_subset(
        df,
        k=micro_k,
        random_state=random_state,
        min_fraction=min_fraction,
        generic_labels=labels_to_skip,
        mixed_label=mixed_label,
    )
    micro = micro.copy()
    micro["microcluster"] = micro["cluster"]
    micro["microcluster_label"] = micro["cluster_label"]

    micro_ids = sorted(int(x) for x in micro["microcluster"].unique())
    centers_array = np.array(
        [
            (
                float(micro.loc[micro["microcluster"] == micro_id, "UMAP1"].mean()),
                float(micro.loc[micro["microcluster"] == micro_id, "UMAP2"].mean()),
            )
            for micro_id in micro_ids
        ],
        dtype=float,
    )
    if len(micro_ids) > 1:
        nn_count = min(max(2, n_neighbors + 1), len(micro_ids))
        nbrs = NearestNeighbors(n_neighbors=nn_count)
        nbrs.fit(centers_array)
        distances, _neighbors = nbrs.kneighbors(centers_array)
        nonzero_neighbor_distances = distances[:, 1:].ravel()
        nonzero_neighbor_distances = nonzero_neighbor_distances[
            nonzero_neighbor_distances > 0
        ]
        if len(nonzero_neighbor_distances) == 0:
            distance_threshold = 0.0
        else:
            distance_threshold = float(np.median(nonzero_neighbor_distances)) * max_dist_factor
    else:
        distance_threshold = 0.0

    current_clusters: dict[int, dict[str, object]] = {}
    next_cluster_id = max(micro_ids) + 1 if micro_ids else 0
    for micro_id in micro_ids:
        sub = micro[micro["microcluster"] == micro_id]
        current_clusters[micro_id] = {
            "member_microclusters": [micro_id],
            "rows": sub.index.tolist(),
            "centroid": (
                float(sub["UMAP1"].mean()),
                float(sub["UMAP2"].mean()),
            ),
            "taxinfo": _cluster_taxonomy_info(
                sub,
                min_fraction=min_fraction,
                generic_labels=labels_to_skip,
            ),
            "cluster_label": choose_cluster_label(
                sub,
                global_counts=global_counts,
                global_total=global_total,
                min_fraction=min_fraction,
                generic_labels=labels_to_skip,
                mixed_label=mixed_label,
            ),
        }

    merge_rows: list[dict[str, object]] = []
    while len(current_clusters) > 1:
        current_ids = sorted(current_clusters)
        centers = np.array(
            [current_clusters[cid]["centroid"] for cid in current_ids],
            dtype=float,
        )
        nn_count = min(max(2, n_neighbors + 1), len(current_ids))
        nbrs = NearestNeighbors(n_neighbors=nn_count)
        nbrs.fit(centers)
        distances, neighbors = nbrs.kneighbors(centers)

        best_candidate: dict[str, object] | None = None
        seen_pairs: set[tuple[int, int]] = set()
        for pos, cluster_id in enumerate(current_ids):
            for distance, nbr_pos in zip(distances[pos][1:], neighbors[pos][1:]):
                other_id = current_ids[int(nbr_pos)]
                pair = tuple(sorted((cluster_id, other_id)))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                if distance > distance_threshold:
                    continue

                cluster_a = current_clusters[cluster_id]
                cluster_b = current_clusters[other_id]
                combined_microclusters = sorted(
                    set(cluster_a["member_microclusters"]) | set(cluster_b["member_microclusters"])
                )
                combined_sub = micro[micro["microcluster"].isin(combined_microclusters)]
                combined_info = _cluster_taxonomy_info(
                    combined_sub,
                    min_fraction=min_fraction,
                    generic_labels=labels_to_skip,
                )
                consensus_label = combined_info["consensus_label"]
                consensus_depth = int(combined_info["consensus_depth"])
                consensus_fraction = float(combined_info["consensus_fraction"])
                if (
                    consensus_label is None
                    or consensus_depth < min_common_depth
                    or consensus_fraction < min_fraction
                ):
                    continue

                common_label, common_depth = _deepest_common_informative_name(
                    list(cluster_a["taxinfo"]["mode_path"]),
                    list(cluster_b["taxinfo"]["mode_path"]),
                    generic_labels=labels_to_skip,
                )
                label_a = str(cluster_a["cluster_label"])
                label_b = str(cluster_b["cluster_label"])
                candidate = {
                    "cluster_a": cluster_id,
                    "cluster_b": other_id,
                    "distance": float(distance),
                    "consensus_label": str(consensus_label),
                    "consensus_depth": consensus_depth,
                    "consensus_fraction": consensus_fraction,
                    "common_label": common_label,
                    "common_depth": common_depth,
                    "label_a": label_a,
                    "label_b": label_b,
                    "combined_microclusters": combined_microclusters,
                }
                if best_candidate is None:
                    best_candidate = candidate
                else:
                    left = (
                        float(candidate["distance"]),
                        -int(candidate["consensus_depth"]),
                        -float(candidate["consensus_fraction"]),
                    )
                    right = (
                        float(best_candidate["distance"]),
                        -int(best_candidate["consensus_depth"]),
                        -float(best_candidate["consensus_fraction"]),
                    )
                    if left < right:
                        best_candidate = candidate

        if best_candidate is None:
            break

        cluster_a = current_clusters.pop(int(best_candidate["cluster_a"]))
        cluster_b = current_clusters.pop(int(best_candidate["cluster_b"]))
        combined_sub = micro[
            micro["microcluster"].isin(best_candidate["combined_microclusters"])
        ]
        new_label = choose_cluster_label(
            combined_sub,
            global_counts=global_counts,
            global_total=global_total,
            min_fraction=min_fraction,
            generic_labels=labels_to_skip,
            mixed_label=mixed_label,
        )
        current_clusters[next_cluster_id] = {
            "member_microclusters": best_candidate["combined_microclusters"],
            "rows": combined_sub.index.tolist(),
            "centroid": (
                float(combined_sub["UMAP1"].mean()),
                float(combined_sub["UMAP2"].mean()),
            ),
            "taxinfo": _cluster_taxonomy_info(
                combined_sub,
                min_fraction=min_fraction,
                generic_labels=labels_to_skip,
            ),
            "cluster_label": new_label,
        }
        merge_rows.append(
            {
                "new_cluster_id": next_cluster_id,
                "cluster_a": best_candidate["cluster_a"],
                "cluster_b": best_candidate["cluster_b"],
                "distance": best_candidate["distance"],
                "distance_threshold": distance_threshold,
                "label_a": best_candidate["label_a"],
                "label_b": best_candidate["label_b"],
                "merged_label": new_label,
                "merged_consensus_label": best_candidate["consensus_label"],
                "merged_consensus_depth": best_candidate["consensus_depth"],
                "merged_consensus_fraction": best_candidate["consensus_fraction"],
                "common_label_before_merge": best_candidate["common_label"],
                "common_depth_before_merge": best_candidate["common_depth"],
                "member_microclusters": ",".join(
                    str(x) for x in best_candidate["combined_microclusters"]
                ),
            }
        )
        next_cluster_id += 1

    ordered_final_ids = sorted(
        current_clusters,
        key=lambda cid: (
            float(current_clusters[cid]["centroid"][0]),
            float(current_clusters[cid]["centroid"][1]),
        ),
    )
    root_to_cluster = {cid: idx for idx, cid in enumerate(ordered_final_ids)}
    micro_to_cluster: dict[int, int] = {}
    for cid, info in current_clusters.items():
        for micro_id in info["member_microclusters"]:
            micro_to_cluster[int(micro_id)] = root_to_cluster[cid]

    merged = micro.copy()
    merged["cluster"] = merged["microcluster"].map(micro_to_cluster).astype(int)

    summary_rows: list[dict[str, object]] = []
    for cluster_id in sorted(merged["cluster"].unique()):
        sub = merged[merged["cluster"] == cluster_id]
        label = choose_cluster_label(
            sub,
            min_fraction=min_fraction,
            generic_labels=labels_to_skip,
            mixed_label=mixed_label,
        )
        taxinfo = _cluster_taxonomy_info(
            sub,
            min_fraction=min_fraction,
            generic_labels=labels_to_skip,
        )
        member_microclusters = sorted(int(x) for x in sub["microcluster"].unique())
        summary_rows.append(
            {
                "cluster": int(cluster_id),
                "cluster_label": label,
                "n": len(sub),
                "microclusters": ",".join(str(x) for x in member_microclusters),
                "n_microclusters": len(member_microclusters),
                "center_umap1": float(sub["UMAP1"].mean()),
                "center_umap2": float(sub["UMAP2"].mean()),
                "umap1_min": float(sub["UMAP1"].min()),
                "umap1_max": float(sub["UMAP1"].max()),
                "umap2_min": float(sub["UMAP2"].min()),
                "umap2_max": float(sub["UMAP2"].max()),
                "top_depth19": _rank_mode(sub, 19)[0],
                "top_depth19_n": _rank_mode(sub, 19)[1],
                "top_depth20": _rank_mode(sub, 20)[0],
                "top_depth20_n": _rank_mode(sub, 20)[1],
                "top_depth21": _rank_mode(sub, 21)[0],
                "top_depth21_n": _rank_mode(sub, 21)[1],
                "consensus_label": taxinfo["consensus_label"],
                "consensus_depth": taxinfo["consensus_depth"],
                "consensus_fraction": taxinfo["consensus_fraction"],
                "major_signature": _informative_signature(
                    sub,
                    generic_labels=labels_to_skip,
                ),
            }
        )
        merged.loc[sub.index, "cluster_label"] = label

    merged_summary = (
        pd.DataFrame(summary_rows).sort_values("cluster").reset_index(drop=True)
    )
    merges_df = pd.DataFrame(merge_rows)
    if not merges_df.empty:
        merges_df = merges_df.sort_values("new_cluster_id").reset_index(drop=True)
    return merged, merged_summary, micro, micro_summary, merges_df


def _pad_limits(x: pd.Series, y: pd.Series, pad_fraction: float = 0.05) -> tuple[float, float, float, float]:
    x0, x1 = float(x.min()), float(x.max())
    y0, y1 = float(y.min()), float(y.max())
    px = (x1 - x0) * pad_fraction if x1 > x0 else 1.0
    py = (y1 - y0) * pad_fraction if y1 > y0 else 1.0
    return x0 - px, x1 + px, y0 - py, y1 + py


def write_cluster_plot(
    all_df: pd.DataFrame,
    clustered: pd.DataFrame,
    summary: pd.DataFrame,
    out_pdf: Path,
    title: str,
    background_alpha: float,
    subset_label: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    ax0, ax1 = axes
    subset_is_full = len(clustered) == len(all_df)

    ax0.scatter(
        all_df["UMAP1"],
        all_df["UMAP2"],
        s=2.0,
        lw=0,
        alpha=background_alpha,
        color=GREY,
        zorder=1,
    )

    for row in summary.itertuples():
        sub = clustered[clustered["cluster"] == row.cluster]
        color = CLUSTER_COLORS[row.cluster % len(CLUSTER_COLORS)]
        label = f"C{row.cluster}: {row.cluster_label} (n={row.n})"
        for ax in axes:
            ax.scatter(
                sub["UMAP1"],
                sub["UMAP2"],
                s=7.0,
                lw=0,
                alpha=0.85,
                color=color,
                label=label if ax is ax0 else None,
                zorder=2,
            )
            ax.text(
                float(sub["UMAP1"].mean()),
                float(sub["UMAP2"].mean()),
                str(row.cluster),
                fontsize=10,
                ha="center",
                va="center",
                weight="bold",
                color="black",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.65, "pad": 1.5},
                zorder=3,
            )

    ax0.set_title("Full UMAP Context")
    ax1.set_title("Clustered UMAP" if subset_is_full else f"{subset_label} Zoom")

    for ax in axes:
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
        ax.set_aspect("equal", adjustable="box")

    ax0.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        frameon=False,
        fontsize=8,
    )
    ax0.set_title(f"Full UMAP Context\n{title}")

    x0, x1, y0, y1 = _pad_limits(clustered["UMAP1"], clustered["UMAP2"])
    ax1.set_xlim(x0, x1)
    ax1.set_ylim(y0, y1)
    fig.savefig(out_pdf, dpi=300)
    plt.close(fig)


def run(args: argparse.Namespace) -> list[Path]:
    in_path = Path(args.df)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path, sep="\t", index_col=0)
    if "UMAP1" not in df.columns or "UMAP2" not in df.columns:
        raise ValueError("Input dataframe must contain UMAP1 and UMAP2 columns.")

    if args.subset_taxid is None:
        subset = df.copy()
        subset_label = args.subset_label or "All samples"
        subset_slug = args.subset_slug or "all"
    else:
        subset = subset_by_taxid(df, args.subset_taxid)
        if subset.empty:
            raise ValueError(f"No rows matched lineage taxid {args.subset_taxid}.")
        inferred_label = _infer_subset_name_from_taxid(df, args.subset_taxid)
        subset_label = args.subset_label or inferred_label or f"Taxid {args.subset_taxid}"
        subset_slug = (
            args.subset_slug
            or _normalize_slug(subset_label)
            or f"taxid{args.subset_taxid}"
        )

    generic_labels = _generic_labels_for_subset(df, args.subset_taxid)

    outputs: list[Path] = []
    subset_base = out_dir / f"{in_path.stem}.{subset_slug}_subset.tsv"
    subset.to_csv(subset_base, sep="\t")
    outputs.append(subset_base)

    for k in args.ks:
        clustered, summary = cluster_subset(
            subset,
            k=k,
            random_state=args.random_state,
            min_fraction=args.taxon_aware_min_fraction,
            generic_labels=generic_labels,
            mixed_label=args.mixed_label,
        )
        assignments_path = out_dir / f"{in_path.stem}.{subset_slug}.k{k}.assignments.tsv"
        summary_path = out_dir / f"{in_path.stem}.{subset_slug}.k{k}.summary.tsv"
        plot_path = out_dir / f"{in_path.stem}.{subset_slug}.k{k}.pdf"
        clustered.to_csv(assignments_path, sep="\t")
        summary.to_csv(summary_path, sep="\t", index=False)
        write_cluster_plot(
            all_df=df,
            clustered=clustered,
            summary=summary,
            out_pdf=plot_path,
            title=f"{subset_label} candidate clusters (k={k})",
            background_alpha=args.background_alpha,
            subset_label=subset_label,
        )
        outputs.extend([assignments_path, summary_path, plot_path])

    if args.taxon_aware:
        merged, merged_summary, micro, micro_summary, edges_df = taxon_aware_cluster_subset(
            subset,
            micro_k=args.taxon_aware_micro_k,
            random_state=args.random_state,
            min_fraction=args.taxon_aware_min_fraction,
            min_common_depth=args.taxon_aware_min_common_depth,
            n_neighbors=args.taxon_aware_neighbors,
            max_dist_factor=args.taxon_aware_max_dist_factor,
            generic_labels=generic_labels,
            mixed_label=args.mixed_label,
        )
        base = (
            out_dir
            / f"{in_path.stem}.{subset_slug}.taxonaware.microk{args.taxon_aware_micro_k}"
        )
        micro_assignments_path = base.with_suffix(".micro.assignments.tsv")
        micro_summary_path = base.with_suffix(".micro.summary.tsv")
        merged_assignments_path = base.with_suffix(".merged.assignments.tsv")
        merged_summary_path = base.with_suffix(".merged.summary.tsv")
        edges_path = base.with_suffix(".merge_edges.tsv")
        plot_path = base.with_suffix(".merged.pdf")
        micro.to_csv(micro_assignments_path, sep="\t")
        micro_summary.to_csv(micro_summary_path, sep="\t", index=False)
        merged.to_csv(merged_assignments_path, sep="\t")
        merged_summary.to_csv(merged_summary_path, sep="\t", index=False)
        edges_df.to_csv(edges_path, sep="\t", index=False)
        write_cluster_plot(
            all_df=df,
            clustered=merged,
            summary=merged_summary,
            out_pdf=plot_path,
            title=(
                f"{subset_label} taxon-aware clusters "
                f"(micro-k={args.taxon_aware_micro_k})"
            ),
            background_alpha=args.background_alpha,
            subset_label=subset_label,
        )
        outputs.extend(
            [
                micro_assignments_path,
                micro_summary_path,
                merged_assignments_path,
                merged_summary_path,
                edges_path,
                plot_path,
            ]
        )

    return outputs


def main(
    argv: list[str] | None = None,
    *,
    prog: str = "egt umap-taxonomy-clusters",
    description: str | None = None,
    default_subset_taxid: int | None = None,
    default_subset_label: str | None = None,
    default_subset_slug: str | None = None,
    default_mixed_label: str = "Mixed cluster",
) -> int:
    args = parse_args(
        argv,
        prog=prog,
        description=description,
        default_subset_taxid=default_subset_taxid,
        default_subset_label=default_subset_label,
        default_subset_slug=default_subset_slug,
        default_mixed_label=default_mixed_label,
    )
    outputs = run(args)
    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
