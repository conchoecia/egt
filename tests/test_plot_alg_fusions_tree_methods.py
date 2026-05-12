from __future__ import annotations

import math
import sys
import threading

import numpy as np
import pandas as pd
import pytest

from egt import plot_alg_fusions as paf


def _tiny_perspchrom():
    return pd.DataFrame(
        {
            "species": ["sp1-10-GCF1.1", "sp1-10-GCA1.1", "sp2-20-GCF1.1"],
            "taxid": [10, 10, 20],
            "taxidstring": ["1;10", "1;10", "1;20"],
            "A": [1, 0, 1],
            "B": [1, 1, 0],
            ("A", "B"): [1, 0, 0],
        }
    )


def test_split_loss_coloc_tree_methods(tmp_path, monkeypatch):
    monkeypatch.setattr(paf, "NCBITaxa", lambda: type("FakeNCBI", (), {"get_taxid_translator": lambda self, taxids: {taxid: f"name_{taxid}" for taxid in taxids}})())
    monkeypatch.setattr(paf.np.random, "choice", lambda values: values[0])
    monkeypatch.setattr(paf.np.random, "random", lambda: 0.0)

    tree = paf.SplitLossColocTree(_tiny_perspchrom(), calibrated_ages={1: 100.0, 10: 0.0, 20: 10.0})
    assert tree.sample_to_taxidlist["sp1-10-GCF1.1"] == [1, 10]
    assert set(tree.leaves) == {"sp1-10-GCF1.1", "sp1-10-GCA1.1", "sp2-20-GCF1.1"}

    tree.add_taxname_to_all_nodes()
    assert tree.G.nodes[1]["taxname"] == "name_1"

    tree.G.edges[1, 10]["num_fusions"] = 2
    tree.G.edges[1, 10]["num_losses"] = 1
    tree.calculate_branch_lengths()
    tree.calculate_event_rates()
    assert tree.G.edges[1, 10]["branch_length_mya"] == pytest.approx(100.0)
    assert tree.G.edges[1, 10]["fusions_per_my"] == pytest.approx(0.02)

    assert tree.get_edges_in_clade(1)

    conserved = tree._conservation_of_colocalizations(tree.perspchrom)
    assert conserved[("A", "B")] >= 0

    sdf = tree.perspchrom.loc[["sp1-10-GCF1.1", "sp2-20-GCF1.1"]]
    pdf = tree.empty_predecessor.copy()
    pdf.index = [1]
    parent = tree._determine_parental_ALG_PresAbs(pdf.copy(), sdf)
    assert set(parent[tree.ALGcols].iloc[0]) <= {0, 1}

    split_parent = tree._determine_parental_ALG_Splits(parent.copy(), sdf.iloc[[0]])
    assert split_parent["A"].iloc[0] in {0, 1}

    filtered = tree._filter_sdf_for_high_quality(tree.perspchrom.loc[["sp1-10-GCF1.1", "sp1-10-GCA1.1"]])
    assert list(filtered.index) == ["sp1-10-GCF1.1"]

    for node in tree.G.nodes():
        if "dataframe" not in tree.G.nodes[node]:
            df = tree.empty_predecessor.copy()
            df.index = [node]
            tree.G.nodes[node]["dataframe"] = df

    outfile = tmp_path / "tree.tsv"
    tree.save_tree_to_df(outfile)
    written = pd.read_csv(outfile, sep="\t", index_col=0)
    assert {"color", "node"} <= set(written.columns)


def test_split_loss_coloc_tree_error_and_helper_branches(monkeypatch):
    monkeypatch.setattr(paf, "NCBITaxa", lambda: type("FakeNCBI", (), {"get_taxid_translator": lambda self, taxids: {taxid: f"name_{taxid}" for taxid in taxids}})())
    monkeypatch.setattr(paf.np.random, "choice", lambda values: values[0])
    monkeypatch.setattr(paf.np.random, "random", lambda: 0.0)

    tree = paf.SplitLossColocTree(_tiny_perspchrom(), calibrated_ages={1: 0.0, 10: 10.0, 20: 1.0})
    tree.calculate_branch_lengths()
    assert tree.G.edges[1, 10]["branch_length_mya"] == 0.0

    tree.G.edges[1, 10]["num_fusions"] = 1
    tree.G.edges[1, 10]["num_losses"] = 2
    tree.calculate_event_rates()
    assert math.isnan(tree.G.edges[1, 10]["fusions_per_my"])
    assert math.isnan(tree.G.edges[1, 10]["losses_per_my"])

    assert tree._parental_probability_log(0, 0.9, 0.1) == pytest.approx(np.log(0.1))
    assert tree._parental_probability_log(1, 0.9, 0.1) == pytest.approx(np.log(0.9))
    assert tree._count_values_ge_1(pd.Series([0, 1, 2])) == 2

    pdf = tree.empty_predecessor.copy()
    pdf.index = [999]
    tree.G.add_node(999)
    with pytest.raises(IOError, match="There are no leaves"):
        tree._determine_ALG_colocalization(pdf, tree.perspchrom.iloc[[0]])

    bad_pdf = tree.empty_predecessor.copy()
    bad_pdf.index = [1]
    with pytest.raises(IOError, match="didn't finish assigning"):
        tree._determine_parental_ALG_Splits(bad_pdf, tree.perspchrom.iloc[[0]])

    good_pdf = tree.empty_predecessor.copy()
    good_pdf.index = [1]
    for col in tree.ALGcols:
        good_pdf[col] = 1
    with pytest.raises(IOError, match="length of 0"):
        tree._determine_parental_ALG_Splits(good_pdf, tree.perspchrom.iloc[0:0])

    internal = pd.DataFrame({"A": [1, 1], "B": [1, 0]}, index=[1, 20])
    assert list(tree._filter_sdf_for_high_quality(internal).index) == [1, 20]

    tree.G.add_edge(99, 10)
    with pytest.raises(Exception, match="one incoming edge"):
        tree.get_edges_in_clade(10)


def test_monitor_progress_and_main_parallel(tmp_path, monkeypatch):
    rbhs = tmp_path / "rbhs"
    rbhs.mkdir()
    (rbhs / "dummy.rbh").write_text("placeholder\n")

    locdf = pd.DataFrame({"sample": ["sp1"]})
    perspchrom = pd.DataFrame(
        {
            "species": ["sp1", "sp2"],
            "taxid": [10, 20],
            "taxidstring": ["1;10", "1;20"],
            "A": [1, 0],
            "B": [0, 1],
            ("A", "B"): [0, 1],
        }
    )

    class FakeTree:
        def __init__(self, df, calibrated_ages=None):
            self.tree_df = pd.DataFrame({"metric": [1], "color": ["#111111"]}, index=["node"])

        def percolate(self):
            return None

        def calculate_branch_lengths(self):
            return None

        def calculate_event_rates(self):
            return None

        def save_tree_to_df(self, path):
            self.tree_df.to_csv(path, sep="\t", compression="gzip")

    class FakePool:
        def __init__(self, _ncores):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def imap_unordered(self, func, worker_args):
            for args in worker_args:
                yield func(args)

    thread_events = []

    class FakeThread:
        def __init__(self, target=None, args=()):
            self.target = target
            self.args = args

        def start(self):
            thread_events.append("start")

        def join(self):
            thread_events.append("join")

    args = type(
        "Args",
        (),
        {
            "directory": str(rbhs),
            "ALG_rbh": str(tmp_path / "ALG.rbh"),
            "minsig": 0.01,
            "ALGname": "ALG",
            "tree_info": str(tmp_path / "tree_info.tsv"),
            "parallel": True,
            "ncores": 2,
        },
    )()

    monkeypatch.setattr(paf, "parse_args", lambda argv=None: args)
    monkeypatch.setattr(paf, "load_calibrated_tree", lambda _path: {"ages": {1: 2.0}})
    monkeypatch.setattr(paf, "rbh_files_to_locdf_and_perspchrom", lambda *args, **kwargs: (locdf.copy(), perspchrom.copy()))
    monkeypatch.setattr(paf, "SplitLossColocTree", FakeTree)
    monkeypatch.setattr(paf, "save_UMAP_plotly", lambda *args, **kwargs: None)
    monkeypatch.setattr(paf, "Pool", FakePool)
    monkeypatch.setattr(paf.threading, "Thread", FakeThread)
    monkeypatch.chdir(tmp_path)

    stop_event = threading.Event()
    monkeypatch.setattr(paf.time, "sleep", lambda _seconds: stop_event.set())
    paf.monitor_progress(str(tmp_path), 1, stop_event)

    assert paf.main([]) == 0
    saved = pd.read_csv(tmp_path / "per_species_ALG_presence_fusions.tsv", sep="\t")
    assert "changestrings" in saved.columns
    assert (tmp_path / "changestring_checkpoints" / "sp1.txt").exists()
    assert thread_events == ["start", "join"]


def test_main_sequential_uses_existing_checkpoint(tmp_path, monkeypatch):
    rbhs = tmp_path / "rbhs"
    rbhs.mkdir()
    (rbhs / "dummy.rbh").write_text("placeholder\n")

    checkpoint_dir = tmp_path / "changestring_checkpoints"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "sp1.txt").write_text("cached-change")

    locdf = pd.DataFrame({"sample": ["sp1"]})
    perspchrom = pd.DataFrame(
        {
            "species": ["sp1"],
            "taxid": [10],
            "taxidstring": ["1;10"],
            "A": [1],
            "B": [0],
            ("A", "B"): [1],
        }
    )

    class FakeTree:
        def __init__(self, df, calibrated_ages=None):
            self.tree_df = pd.DataFrame({"metric": [1], "color": ["#111111"]}, index=["node"])

        def percolate(self):
            return None

        def calculate_branch_lengths(self):
            return None

        def calculate_event_rates(self):
            return None

        def save_tree_to_df(self, path):
            self.tree_df.to_csv(path, sep="\t", compression="gzip")

    args = type(
        "Args",
        (),
        {
            "directory": str(rbhs),
            "ALG_rbh": str(tmp_path / "ALG.rbh"),
            "minsig": 0.01,
            "ALGname": "ALG",
            "tree_info": None,
            "parallel": False,
            "ncores": 1,
        },
    )()

    monkeypatch.setattr(paf, "parse_args", lambda argv=None: args)
    monkeypatch.setattr(paf, "rbh_files_to_locdf_and_perspchrom", lambda *args, **kwargs: (locdf.copy(), perspchrom.copy()))
    monkeypatch.setattr(paf, "SplitLossColocTree", FakeTree)
    monkeypatch.setattr(paf, "save_UMAP_plotly", lambda *args, **kwargs: None)
    monkeypatch.chdir(tmp_path)

    assert paf.main([]) == 0
    saved = pd.read_csv(tmp_path / "per_species_ALG_presence_fusions.tsv", sep="\t")
    assert saved.loc[0, "changestrings"] == "cached-change"


def test_rebuild_lineage_strings_reports_conflicts_and_cycles(monkeypatch):
    monkeypatch.setattr(
        paf,
        "NCBITaxa",
        lambda: type("FakeNCBI", (), {"get_taxid_translator": lambda self, taxids: {taxid: f"name_{taxid}" for taxid in taxids}})(),
    )
    monkeypatch.setattr(paf.np.random, "choice", lambda values: values[0])
    monkeypatch.setattr(paf.np.random, "random", lambda: 0.0)

    tree = paf.SplitLossColocTree(_tiny_perspchrom(), calibrated_ages={1: 100.0, 10: 0.0, 20: 10.0})

    tree.G.add_edge(99, "sp1-10-GCF1.1")
    with pytest.raises(IOError, match="Graph topology conflicts detected"):
        tree._rebuild_lineage_strings_from_topology()

    tree = paf.SplitLossColocTree(_tiny_perspchrom(), calibrated_ages={1: 100.0, 10: 0.0, 20: 10.0})
    tree.G.add_edge("sp1-10-GCF1.1", 1)
    with pytest.raises(IOError, match="Graph topology conflicts detected"):
        tree._rebuild_lineage_strings_from_topology()


def test_get_predecessor_multiple_parent_debug_branch(monkeypatch):
    monkeypatch.setattr(
        paf,
        "NCBITaxa",
        lambda: type(
            "FakeNCBI",
            (),
            {
                "get_taxid_translator": lambda self, taxids: {taxid: f"name_{taxid}" for taxid in taxids},
            },
        )(),
    )
    monkeypatch.setattr(paf.np.random, "choice", lambda values: values[0])
    monkeypatch.setattr(paf.np.random, "random", lambda: 0.0)

    tree = paf.SplitLossColocTree(_tiny_perspchrom(), calibrated_ages={1: 100.0, 10: 0.0, 20: 10.0})
    tree.G.add_edge(99, "sp1-10-GCF1.1")

    class FakeLookupNCBI:
        def get_taxid_translator(self, taxids):
            return {taxid: f"Taxon{taxid}" for taxid in taxids}

    monkeypatch.setattr(sys.modules["ete4"], "NCBITaxa", lambda: FakeLookupNCBI())
    with pytest.raises(IOError, match="There should only be one predecessor"):
        tree._get_predecessor("sp1-10-GCF1.1")
