from __future__ import annotations

import sys
import types
from pathlib import Path

import pandas as pd
import pytest

from egt import phylotreeumap_subsample as pts


class FakeNCBI:
    def __init__(self):
        self.ranks = {
            1: "kingdom",
            10: "phylum",
            20: "class",
            30: "order",
            40: "family",
            41: "family",
            100: "species",
            101: "species",
            102: "species",
        }
        self.names = {
            1: "Animalia",
            10: "Chordata",
            20: "Mammalia",
            30: "Primates",
            40: "Hominidae",
            41: "Muridae",
            100: "Homo sapiens",
            101: "Pan troglodytes",
            102: "Mus musculus",
        }

    def get_rank(self, taxids):
        return {taxid: self.ranks[taxid] for taxid in taxids if taxid in self.ranks}

    def get_taxid_translator(self, taxids):
        return {taxid: self.names[taxid] for taxid in taxids if taxid in self.names}


def test_generate_subsample_priorities_and_validation():
    assert pts.return_kingdom_limited_order()[0] == "kingdom"
    assert pts.rank_sort_full()["family"] > pts.rank_sort_full()["order"]

    priorities = pts.generate_subsample_priorities("genus", "family")
    assert priorities[0][0] == "genus"
    assert priorities[-1][0] == "family"

    assert pts.generate_subsample_priorities("allsamples", "allsamples") == [["allsamples"]]

    with pytest.raises(SystemExit):
        pts.generate_subsample_priorities("family", "genus")


def test_subsample_phylogenetically_priority_and_select_all(monkeypatch):
    monkeypatch.setattr(pts, "NCBITaxa", lambda: FakeNCBI())
    df = pd.DataFrame(
        {
            "sample": ["GCF_human", "GCA_human", "chimp", "mouse"],
            "taxid_list_str": [
                "1;10;20;30;40;100",
                "1;10;20;30;40;100",
                "1;10;20;30;40;101",
                "1;10;20;30;41;102",
            ],
        }
    )

    selected, flat = pts.subsample_phylogenetically(
        df,
        max_per_bucket=2,
        priority=True,
        priority_taxids={100},
        bucket_priority=("family", "order"),
    )
    assert set(flat) <= set(df["sample"])
    assert selected[40]["priority_count"] == 1
    assert "GCF_human" in [x["sample"] for x in selected[40]["chosen"]]

    all_selected, all_flat = pts.subsample_phylogenetically(
        df,
        select_all=True,
        select_all_rank="family",
    )
    assert len(all_flat) == 4
    assert all_selected[40]["final_size"] == 3


def test_report_helpers_render_text(monkeypatch):
    monkeypatch.setattr(pts, "NCBITaxa", lambda: FakeNCBI())
    selected_buckets = {
        40: {
            "rank": "family",
            "name": "Hominidae",
            "chosen": [
                {"sample": "human", "path": [1, 10, 20, 30, 40, 100], "species_taxid": 100},
                {"sample": "chimp", "path": [1, 10, 20, 30, 40, 101], "species_taxid": 101},
            ],
            "original_size": 3,
            "final_size": 2,
            "priority_count": 1,
            "cap_exceeded": False,
        }
    }

    rank_map, name_map = pts._rank_name_maps_full(selected_buckets)
    assert rank_map[40] == "family"
    assert name_map[100] == "Homo sapiens"

    breadcrumbs = pts.make_subsampling_report_breadcrumbs(selected_buckets, root_taxid=1)
    summary = pts.make_subsampling_summary_table(selected_buckets)
    tree = pts.make_subsampling_report_tree(selected_buckets, root_taxid=1)

    assert "Subsampling Report" in breadcrumbs
    assert "Hominidae (family)" in summary
    assert "bucket: Hominidae (family) <2/3>" in tree


def test_generate_subsample_priorities_custom_and_error_paths():
    custom = pts.generate_subsample_priorities(None, None, custom_sample_set=["family", "order", "class"])
    assert custom == [["family", "order", "class"], ["order", "class"], ["class"]]

    allsamples = pts.generate_subsample_priorities("allsamples", "family")
    assert allsamples[0][0] == "allsamples"
    assert allsamples[-1][0] == "family"

    with pytest.raises(SystemExit):
        pts.generate_subsample_priorities(None, "family", custom_sample_set=["family"])
    with pytest.raises(SystemExit):
        pts.generate_subsample_priorities("not_a_rank", "family")
    with pytest.raises(SystemExit):
        pts.generate_subsample_priorities("family", "allsamples")


def test_subsample_phylogenetically_edge_cases_and_main(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(pts, "NCBITaxa", lambda: FakeNCBI())
    df = pd.DataFrame(
        {
            "sample": ["human", "chimp", "mouse"],
            "taxid_list_str": [
                "1;10;20;30;40;100",
                "1;10;20;30;40;101",
                "1;10;20;30;41;102",
            ],
        }
    )

    selected, flat = pts.subsample_phylogenetically(
        df,
        max_per_bucket=0,
        priority=True,
        priority_taxids={100},
        bucket_priority=("family", "order"),
    )
    assert selected[40]["cap_exceeded"] is True
    assert "human" in flat

    rank_selected, rank_flat = pts.subsample_phylogenetically(
        df,
        max_per_bucket=1,
        priority=False,
        bucket_priority=("not-a-rank",),
    )
    assert len(rank_flat) == 1
    assert None in rank_selected

    sampletsv = tmp_path / "sample.tsv"
    df.assign(extra=[1, 2, 3]).to_csv(sampletsv, sep="\t")

    monkeypatch.setattr(
        pts,
        "subsample_phylogenetically",
        lambda _df, max_per_bucket=10, priority=True: (
            {
                40: {
                    "rank": "family",
                    "name": "Hominidae",
                    "chosen": [{"sample": "human", "path": [1, 10, 20, 30, 40, 100], "species_taxid": 100}],
                    "original_size": 2,
                    "final_size": 1,
                    "priority_count": 1,
                    "cap_exceeded": False,
                }
            },
            ["human"],
        ),
    )
    monkeypatch.chdir(tmp_path)
    assert pts.main(["--rbh-dir", str(tmp_path), "--sample-tsv", str(sampletsv), "--max-per-bucket", "1"]) == 0
    outdir = tmp_path / "subsampling_reports"
    assert (outdir / "summary_table.txt").exists()
    assert (outdir / "breadcrumbs.txt").exists()
    assert (outdir / "tree.txt").exists()
    assert (outdir / "selected_samples.txt").read_text().strip() == "human"


def test_subsample_select_all_none_bucket_and_report_toggles(monkeypatch):
    monkeypatch.setattr(pts, "NCBITaxa", lambda: FakeNCBI())
    monkeypatch.setitem(sys.modules, "ete4", types.SimpleNamespace(NCBITaxa=lambda: FakeNCBI()))
    df = pd.DataFrame(
        {
            "sample": ["human", "unknown"],
            "taxid_list_str": ["1;10;20;30;40;100", "1;999999"],
        }
    )

    selected, flat = pts.subsample_phylogenetically(
        df,
        select_all=True,
        select_all_rank="family",
    )
    assert flat == ["human", "unknown"]
    assert None in selected
    assert selected[None]["rank"] == "unranked"
    assert selected[None]["name"] == "Unranked"

    selected_buckets = {
        20: {
            "rank": "class",
            "name": "Mammalia",
            "chosen": [{"sample": "human", "path": [1, 10, 20, 30, 40, 100], "species_taxid": 100}],
            "original_size": 3,
            "final_size": 1,
            "priority_count": 0,
            "cap_exceeded": False,
        },
        40: {
            "rank": "family",
            "name": "Hominidae",
            "chosen": [{"sample": "human", "path": [1, 10, 20, 30, 40, 100], "species_taxid": 100}],
            "original_size": 2,
            "final_size": 1,
            "priority_count": 0,
            "cap_exceeded": False,
        },
        None: {
            "rank": "unranked",
            "name": "Unknown",
            "chosen": [{"sample": "mystery", "path": [1, 999999], "species_taxid": 999999}],
            "original_size": 1,
            "final_size": 1,
            "priority_count": 0,
            "cap_exceeded": False,
        },
    }

    breadcrumbs = pts.make_subsampling_report_breadcrumbs(
        selected_buckets, root_taxid=None, include_unranked=False
    )
    assert "mystery" in breadcrumbs

    tree = pts.make_subsampling_report_tree(
        selected_buckets,
        root_taxid=None,
        include_unranked=False,
        suppress_higher_fallback_labels=False,
    )
    assert "Mammalia [class] <1/3>" in tree
    assert "mystery  [bucket: Unknown (unranked) <1/1>]" in tree
