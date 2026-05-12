from __future__ import annotations

from collections import Counter

import pandas as pd

from egt.arthropoda_umap_clusters import (
    choose_cluster_label,
    cluster_subset,
    relabel_by_centroid,
    subset_by_taxid,
    taxon_aware_cluster_subset,
)
from egt.umap_taxonomy_clusters import parse_args as parse_generic_args


def test_subset_by_taxid_matches_lineage_string():
    df = pd.DataFrame(
        {
            "sample": ["a", "b", "c"],
            "taxid_list_str": [
                "1;2;6656;10",
                "1;2;7711;20",
                "1;2;6656;30",
            ],
            "UMAP1": [0.0, 1.0, 2.0],
            "UMAP2": [0.0, 1.0, 2.0],
        }
    )
    out = subset_by_taxid(df, 6656)
    assert out["sample"].tolist() == ["a", "c"]


def test_choose_cluster_label_prefers_specific_taxon():
    df = pd.DataFrame(
        {
            "taxid_list_str": [
                "1;2;3;4;5;6;7;8;9;10;11;12;13;6656;14;15;16;17;18;19;7088;9991",
                "1;2;3;4;5;6;7;8;9;10;11;12;13;6656;14;15;16;17;18;19;7088;9992",
                "1;2;3;4;5;6;7;8;9;10;11;12;13;6656;14;15;16;17;18;19;7088;9993",
                "1;2;3;4;5;6;7;8;9;10;11;12;13;6656;14;15;16;17;18;19;7088;9994",
            ],
            "taxname_list_str": [
                "r;a;b;c;d;e;f;g;h;i;j;k;m;Arthropoda;Mandibulata;Pancrustacea;Hexapoda;Insecta;Dicondylia;Amphiesmenoptera;Lepidoptera;SpeciesA",
                "r;a;b;c;d;e;f;g;h;i;j;k;m;Arthropoda;Mandibulata;Pancrustacea;Hexapoda;Insecta;Dicondylia;Amphiesmenoptera;Lepidoptera;SpeciesB",
                "r;a;b;c;d;e;f;g;h;i;j;k;m;Arthropoda;Mandibulata;Pancrustacea;Hexapoda;Insecta;Dicondylia;Amphiesmenoptera;Lepidoptera;SpeciesC",
                "r;a;b;c;d;e;f;g;h;i;j;k;m;Arthropoda;Mandibulata;Pancrustacea;Hexapoda;Insecta;Dicondylia;Amphiesmenoptera;Lepidoptera;SpeciesD",
            ],
            "UMAP1": [0.0, 0.1, 0.2, 0.3],
            "UMAP2": [0.0, 0.1, 0.2, 0.3],
        }
    )
    assert choose_cluster_label(df) == "Lepidoptera"


def test_relabel_by_centroid_orders_left_to_right():
    labels = pd.Series([2, 0, 1, 2, 1, 0]).to_numpy()
    centers = pd.DataFrame(
        {
            "x": [2.0, 1.0, -1.0],
            "y": [0.0, 0.0, 0.0],
        }
    ).to_numpy()
    relabeled = relabel_by_centroid(labels, centers)
    assert relabeled.tolist() == [0, 2, 1, 0, 1, 2]


def test_cluster_subset_returns_summary_columns():
    df = pd.DataFrame(
        {
            "sample": [f"s{i}" for i in range(6)],
            "taxid_list_str": ["1;2;6656;6960;7088"] * 3 + ["1;2;6656;6960;7147"] * 3,
            "taxname_list_str": ["root;Euk;Arthropoda;Hexapoda;Lepidoptera"] * 3
            + ["root;Euk;Arthropoda;Hexapoda;Diptera"] * 3,
            "UMAP1": [0.0, 0.1, 0.2, 5.0, 5.1, 5.2],
            "UMAP2": [0.0, 0.1, 0.2, 5.0, 5.1, 5.2],
        }
    )
    clustered, summary = cluster_subset(df, k=2, random_state=0)
    assert set(clustered["cluster"]) == {0, 1}
    assert {"cluster", "cluster_label", "n", "major_signature"} <= set(summary.columns)


def test_choose_cluster_label_prefers_deeper_enriched_subclade():
    df = pd.DataFrame(
        {
            "taxid_list_str": ["1;2;6656;10;11;12;13;14;15;100"] * 6
            + ["1;2;6656;10;11;12;13;14;16;200"] * 4,
            "taxname_list_str": (
                ["root;Metazoa;Arthropoda;Hexapoda;Amphiesmenoptera;Lepidoptera;Glossata;Obtectomera;Noctuidae;spA"] * 6
                + ["root;Metazoa;Arthropoda;Hexapoda;Amphiesmenoptera;Lepidoptera;Glossata;Obtectomera;Tortricidae;spB"] * 4
            ),
            "UMAP1": [float(i) for i in range(10)],
            "UMAP2": [0.0] * 10,
        }
    )
    global_counts = Counter(
        {
            (7, "Obtectomera"): 500,
            (8, "Noctuidae"): 20,
            (8, "Tortricidae"): 25,
        }
    )
    label = choose_cluster_label(
        df,
        global_counts=global_counts,
        global_total=1000,
        min_fraction=0.55,
    )
    assert label == "Noctuidae-rich"


def test_taxon_aware_cluster_subset_merges_nearby_same_lineage_groups():
    df = pd.DataFrame(
        {
            "sample": [f"s{i}" for i in range(9)],
            "taxid_list_str": (
                ["1;2;6656;6960;8000;7088;8001;9001"] * 3
                + ["1;2;6656;6960;8000;7088;8001;9002"] * 3
                + ["1;2;6656;6960;7147;7148;7149;9003"] * 3
            ),
            "taxname_list_str": (
                ["root;Euk;Arthropoda;Hexapoda;Amphiesmenoptera;Lepidoptera;Glossata;spA"] * 3
                + ["root;Euk;Arthropoda;Hexapoda;Amphiesmenoptera;Lepidoptera;Glossata;spB"] * 3
                + ["root;Euk;Arthropoda;Hexapoda;Diptera;Brachycera;Muscomorpha;spC"] * 3
            ),
            "UMAP1": [0.0, 0.05, 0.1, 0.35, 0.4, 0.45, 3.0, 3.1, 3.2],
            "UMAP2": [0.0, 0.02, 0.04, 0.28, 0.3, 0.33, 3.0, 3.1, 3.2],
        }
    )
    merged, merged_summary, micro, micro_summary, edges = taxon_aware_cluster_subset(
        df,
        micro_k=3,
        random_state=0,
        min_fraction=0.6,
        min_common_depth=5,
        n_neighbors=2,
        max_dist_factor=2.0,
    )
    assert len(micro_summary) == 3
    assert len(merged_summary) < len(micro_summary)
    assert len(edges) >= 1


def test_generic_parser_defaults_to_full_umap():
    args = parse_generic_args(["--df", "in.tsv", "--out-dir", "out"])
    assert args.subset_taxid is None
    assert args.subset_label is None
    assert args.subset_slug is None


def test_choose_cluster_label_generalizes_beyond_arthropoda():
    df = pd.DataFrame(
        {
            "taxid_list_str": ["1;2;7711;40674;9443"] * 4 + ["1;2;7711;40674;9989"] * 2,
            "taxname_list_str": (
                ["root;Metazoa;Chordata;Mammalia;Primates"] * 4
                + ["root;Metazoa;Chordata;Mammalia;Rodentia"] * 2
            ),
            "UMAP1": [float(i) for i in range(6)],
            "UMAP2": [0.0] * 6,
        }
    )
    global_counts = Counter(
        {
            (2, "Chordata"): 1000,
            (3, "Mammalia"): 200,
            (4, "Primates"): 20,
            (4, "Rodentia"): 30,
        }
    )
    label = choose_cluster_label(
        df,
        global_counts=global_counts,
        global_total=1000,
        generic_labels={"root", "Metazoa", "Chordata"},
        min_fraction=0.55,
    )
    assert label == "Primates-rich"
