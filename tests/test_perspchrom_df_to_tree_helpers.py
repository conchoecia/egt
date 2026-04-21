from __future__ import annotations

import os

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import pytest

from egt import perspchrom_df_to_tree as pdt


class FakeNCBI:
    def get_taxid_translator(self, taxids):
        return {taxid: f"name_{taxid}" for taxid in taxids}


def test_rgb_float_to_hex():
    assert pdt.rgb_float_to_hex((1.0, 0.5, 0.0)) == "#ff7f00"


def test_parse_gain_loss_string_and_dataframe_wrapper():
    changes = "1-([('A','B')]|['L1']|['S1'])-2-([]|[]|[])-3"
    df = pdt.parse_gain_loss_string(changes, "sampleA")
    assert list(df["source_taxid"]) == [1, 2]
    assert df.loc[0, "colocalizations"] == [("A", "B")]
    assert df.loc[0, "losses"] == ["L1"]
    assert df.loc[0, "splits"] == ["S1"]

    wrapped = pdt.parse_gain_loss_from_perspchrom_df(
        pd.DataFrame({"changestrings": [changes], "species": ["sampleA"]})
    )
    assert len(wrapped) == 2


def test_safe_get_taxid_translator_adds_custom_negative_taxids():
    names = pdt.safe_get_taxid_translator(FakeNCBI(), [1, -67, -68])
    assert names[1] == "name_1"
    assert names[-67] == "Myriazoa"
    assert names[-68] == "Parahoxozoa"


def test_nodes_in_same_cc_and_colocalize_these_nodes():
    graph = nx.Graph()
    graph.add_nodes_from(["A", "B", "C"])
    assert pdt.nodes_in_same_CC(graph, ["A", "B"]) is False
    graph = pdt.colocalize_these_nodes(graph, ["A", "B", "C"])
    assert pdt.nodes_in_same_CC(graph, ["A", "B", "C"]) is True
    assert graph.has_edge("A", "C")


def test_stats_on_changedf(monkeypatch):
    monkeypatch.setattr(pdt, "NCBITaxa", lambda: FakeNCBI())
    sampledf = pd.DataFrame(
        {
            "taxidstring": ["1;2", "1;3"],
        }
    )
    changedf = pd.DataFrame(
        {
            "source_taxid": [1],
            "target_taxid": [2],
            "colocalizations": [[("A", "B")]],
            "losses": [["L1"]],
            "samplename": ["sampleA"],
            "sample_taxid": [999],
        }
    )

    stats = pdt.stats_on_changedf(sampledf, changedf)

    assert set(stats["change"]) == {("A", "B"), "L1"}
    assert set(stats["target_taxid_name"]) == {"name_2"}
    assert all(stats["counts"] == 1)


def test_size_helpers_and_delete_node_resize(monkeypatch):
    graph = nx.Graph()
    graph.add_node("A", size=2)
    graph.add_node("B", size=3)
    graph.add_edge("A", "B")

    assert pdt.node_size_fraction_of_total_size(graph, "A") == 2 / 5
    assert pdt.node_size_fraction_of_largest(graph, "A") == 2 / 3
    assert pdt.node_size_CC(graph, "A") == 5
    assert pdt.node_size_CC_fraction_of_total_size(graph, "A") == 1.0
    assert pdt.node_size_CC_fraction_of_largest(graph, "A") == 1.0

    monkeypatch.setattr(pdt.random, "choice", lambda seq: seq[0])
    resized = pdt.delete_node_resize(graph, "A")
    assert "A" not in resized.nodes
    assert resized.nodes["B"]["size"] == 5


def test_trace_cache_roundtrip_and_staleness(tmp_path):
    cache = {
        1: {
            ("ALG", "frac", "abs"): {
                "lines": {"bin1": {"num_sim": [10, 20], "value": [0.5, 0.7]}},
                "num_sims": 20,
            }
        }
    }
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    pdt.save_trace_cache_to_file(cache, str(cache_dir))
    loaded = pdt.load_trace_cache_from_file(str(cache_dir))
    assert loaded[1][("ALG", "frac", "abs")]["num_sims"] == 20
    assert loaded[1][("ALG", "frac", "abs")]["lines"]["bin1"]["value"] == [0.5, 0.7]

    sim = tmp_path / "sim.tsv"
    sim.write_text("x\n")
    os.utime(sim, (os.path.getmtime(sim) + 5, os.path.getmtime(sim) + 5))
    assert pdt.is_trace_cache_stale([str(sim)], str(cache_dir)) is True


def test_assert_single_size_abs_condition_and_df_to_obs_exp_dict():
    df = pd.DataFrame(
        {
            "ob_ex": ["observed", "expected"],
            "size_frac": ["frac", "frac"],
            "abs_CC": ["abs", "abs"],
            "bin": ["b1", "b1"],
            "counts": [3, 1],
        }
    )
    assert pdt.assert_single_size_abs_condition(df) is True
    ove_df = pdt.df_to_obs_exp_dict(df)
    assert "b1" in ove_df
    ove_dict = pdt.df_to_obs_exp_dict({"observed": {"b1": 3}, "expected": {"b1": 1}})
    assert ove_dict["b1"] > 0


def test_generate_trace_panel_and_phylo_tree_helpers(tmp_path, monkeypatch):
    fig, ax = plt.subplots()
    trace_cache = {"lines": {"bin1": {"num_sim": [10, 20], "value": [0.1, 0.2]}}, "num_sims": 20}
    pdt.generate_trace_panel(ax, [("1", "2")], [], "ALG", "frac", "abs", trace_cache=trace_cache)
    assert ax.get_xlim()[1] == 20
    plt.close(fig)

    monkeypatch.setattr(pdt, "NCBITaxa", lambda: FakeNCBI())
    tree = pdt.PhyloTree()
    assert tree.add_lineage_string("1;2;3") == 0
    df = pd.DataFrame({"taxidstring": ["1;2;3", "1;4"]})
    assert tree.build_tree_from_per_sp_chrom_df(df) == 0
    tree.add_taxname_to_all_nodes()
    assert tree.G.nodes[1]["taxname"] == "name_1"
    edges = tree.get_edges_in_clade(2)
    assert (1, 2) in edges and (2, 3) in edges

    outfile = tmp_path / "nodes.tsv"
    pdt.generate_node_taxid_file_from_per_sp_chrom_df(df, outfile)
    assert "1\tname_1" in outfile.read_text()


def test_stats_df_to_loss_fusion_dfs_and_trace_cache_builder(tmp_path, monkeypatch):
    algdf = pd.DataFrame(
        {
            "ALGname": ["A", "B", "C"],
            "Color": ["#aa0000", "#00aa00", "#0000aa"],
            "Size": [10, 20, 5],
        }
    )
    perspchrom = pd.DataFrame(
        {
            "species": ["sp1", "sp2"],
            "changestrings": [
                "1-([('A', 'B')]|['C']|[])-2",
                "1-([('A', 'B')]|[]|[])-2",
            ],
        }
    )

    dispersion_df, coloc_df, alg_coloc_df = pdt.stats_df_to_loss_fusion_dfs(perspchrom, algdf, obs_seed=1)
    assert list(dispersion_df["thisloss"]) == ["C"]
    assert list(coloc_df["thiscoloc"]) == [("A", "B")]
    assert list(alg_coloc_df["thiscolor"]) == [("A", "B")]

    def fake_sample(self, frac=1, random_state=None):
        return self.iloc[::-1].reset_index(drop=True)

    monkeypatch.setattr(pd.DataFrame, "sample", fake_sample)
    _, _, randomized_algs = pdt.stats_df_to_loss_fusion_dfs(
        perspchrom.iloc[[0]],
        algdf,
        obs_seed=1,
        randomize_ALGs=True,
    )
    assert len(randomized_algs) == 1

    cached = {2: {("ALG", "frac", "abs"): {"lines": {"x": {"num_sim": [5], "value": [1.0]}}, "num_sims": 5}}}
    monkeypatch.setattr(pdt, "is_trace_cache_stale", lambda files: False)
    monkeypatch.setattr(pdt, "load_trace_cache_from_file", lambda: cached)
    assert pdt.precompute_trace_cache(["dummy.tsv"], type("T", (), {"G": nx.DiGraph()})(), [2]) == cached

    sim_files = []
    for idx in range(3):
        sim = tmp_path / f"sim_{idx}.tsv"
        pd.DataFrame(
            {
                "ALG_num": ["ALG", "ALG"],
                "size_frac": ["frac", "frac"],
                "abs_CC": ["abs", "abs"],
                "branch": ["(1, 2)", "(1, 2)"],
                "bin": ["b1", "b1"],
                "ob_ex": ["observed", "expected"],
                "counts": [2, 1],
                "obs_count": [1, 1],
            }
        ).to_csv(sim, sep="\t", index=False)
        sim_files.append(str(sim))

    class FakeTree:
        def __init__(self):
            self.G = nx.DiGraph()
            self.G.add_node(2)

        def get_edges_in_clade(self, taxid):
            return [(1, 2)] if taxid == 2 else []

    saved = {}
    monkeypatch.setattr(pdt, "is_trace_cache_stale", lambda files: True)
    monkeypatch.setattr(pdt, "save_trace_cache_to_file", lambda cache: saved.setdefault("cache", cache))
    built = pdt.precompute_trace_cache(sim_files, FakeTree(), [2, 99])
    assert ("ALG", "frac", "abs") in built[2]
    assert built[2][("ALG", "frac", "abs")]["num_sims"] == 3
    assert built[2][("ALG", "frac", "abs")]["lines"]["b1"]["value"][-1] > 0
    assert saved["cache"][2][("ALG", "frac", "abs")]["num_sims"] == 3


def test_generate_trace_panel_validation_and_file_fallback(tmp_path):
    fig, ax = plt.subplots()
    with pytest.raises(Exception, match="ALG_num must be either ALG or num"):
        pdt.generate_trace_panel(ax, [], [], "bad", "frac", "abs")
    with pytest.raises(Exception, match="frac_or_size must be either frac or size"):
        pdt.generate_trace_panel(ax, [], [], "ALG", "bad", "abs")
    with pytest.raises(Exception, match="abs_CC must be either abs or CC"):
        pdt.generate_trace_panel(ax, [], [], "ALG", "frac", "bad")
    plt.close(fig)

    sim = tmp_path / "trace.tsv"
    pd.DataFrame(
        {
            "ALG_num": ["ALG", "ALG"],
            "size_frac": ["frac", "frac"],
            "abs_CC": ["abs", "abs"],
            "branch": ["(1, 2)", "(1, 2)"],
            "bin": ["b1", "b1"],
            "ob_ex": ["observed", "expected"],
            "counts": [4, 2],
            "obs_count": [3, 3],
        }
    ).to_csv(sim, sep="\t", index=False)

    fig, ax = plt.subplots()
    pdt.generate_trace_panel(ax, ["(1, 2)"], [str(sim)], "ALG", "frac", "abs", trace_cache=None)
    assert ax.get_xlim()[1] == 3
    plt.close(fig)


def test_simulation_plot_workers_and_heatmap_orchestration(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "simulations").mkdir()
    algdf = pd.DataFrame({"ALGname": ["A"], "Color": ["#aa0000"], "Size": [1]})
    plotdf_sumdf = pd.DataFrame(
        {
            "ALG_num": ["num", "num", "ALG", "ALG"],
            "bin": ["b1", "b1", "b2", "b2"],
            "ob_ex": ["observed", "expected", "observed", "expected"],
            "size_frac": ["frac", "frac", "size", "size"],
            "abs_CC": ["abs", "abs", "abs", "abs"],
            "counts": [2, 1, 3, 2],
        }
    )

    class FakeTree:
        def __init__(self):
            self.G = nx.DiGraph()
            self.G.add_node(2, taxname="Clade")

        def get_edges_in_clade(self, taxid):
            return [(1, 2)]

        def build_tree_from_per_sp_chrom_df(self, _df):
            return 0

        def add_taxname_to_all_nodes(self):
            return 0

    monkeypatch.setattr(pdt, "NCBITaxa", lambda: FakeNCBI())
    monkeypatch.setattr(pdt, "gen_square_ax_and_colorbar", lambda *args, **kwargs: ([0, 0, 0.1, 0.1], [0.11, 0, 0.02, 0.1]))
    monkeypatch.setattr(pdt, "gen_square_ax", lambda *args, **kwargs: [0.2, 0, 0.1, 0.1])
    monkeypatch.setattr(pdt, "generate_mean_counts_panel", lambda ax, cax, *args, **kwargs: (ax, cax))
    monkeypatch.setattr(pdt, "generate_obs_exp_panel", lambda ax, cax, *args, **kwargs: (ax, cax))
    monkeypatch.setattr(pdt, "generate_ALG_mean_counts_panel", lambda ax, cax, *args, **kwargs: (ax, cax))
    monkeypatch.setattr(pdt, "generate_ALG_obs_exp_counts_panel", lambda ax, cax, *args, **kwargs: (ax, cax))
    monkeypatch.setattr(pdt, "generate_trace_panel", lambda ax, *args, **kwargs: ax)

    tree = FakeTree()
    pdt._make_one_simulation_plot(
        algdf,
        plotdf_sumdf,
        2,
        2,
        tree,
        2,
        [],
        str(tmp_path / "heatmap"),
        trace_cache={2: {("num", "frac", "abs"): {"lines": {}, "num_sims": 1}, ("ALG", "size", "abs"): {"lines": {}, "num_sims": 1}}},
    )
    assert any(path.name.startswith("heatmap_2_Clade") for path in tmp_path.iterdir())

    missing_tree = FakeTree()
    missing_tree.G = nx.DiGraph()
    pdt._make_one_simulation_plot(algdf, plotdf_sumdf, 1, 1, missing_tree, 99, [], str(tmp_path / "missing"))
    assert any(path.name.startswith("missing_99_") for path in tmp_path.iterdir())

    sim_file = tmp_path / "sim.tsv"
    sim_file.write_text("x\n")
    monkeypatch.setattr(pdt, "_global_sampledf_path", "sample.tsv")
    monkeypatch.setattr(pdt, "_global_algdf_path", "alg.tsv")
    monkeypatch.setattr(pdt, "run_n_simulations_save_results", lambda *args, **kwargs: open(args[2], "w").write("ok\n"))
    ok = pdt._run_simulation_worker((1, 5, 10, 0.1))
    assert ok["status"] == "success"
    monkeypatch.setattr(pdt, "run_n_simulations_save_results", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")))
    failed = pdt._run_simulation_worker((2, 5, 10, 0.1))
    assert failed["status"] == "failed"

    monkeypatch.setattr(pdt, "_make_one_simulation_plot", lambda *args, **kwargs: None)
    worker_ok = pdt._make_heatmap_worker((algdf, plotdf_sumdf, 1, 1, tree, 2, [], "out", 6, None))
    assert worker_ok["status"] == "success"
    monkeypatch.setattr(pdt, "_make_one_simulation_plot", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("bad heatmap")))
    worker_bad = pdt._make_heatmap_worker((algdf, plotdf_sumdf, 1, 1, tree, 2, [], "out", 6, None))
    assert worker_bad["status"] == "failed"

    class FakeColoc:
        def __init__(self):
            self.plotmatrix_sumdf = pd.DataFrame({"branch": ["(1, 2)"], "x": [1]})
            self.num_observed_observations = 3
            self.num_expected_observations = 4

        def plotmatrix_listoffiles_to_plotmatrix(self, _files):
            return None

    monkeypatch.setattr(pdt, "PhyloTree", FakeTree)
    monkeypatch.setattr(pdt.rbh_tools, "parse_ALG_rbh_to_colordf", lambda _path: algdf)
    monkeypatch.setattr(pdt, "coloc_array", FakeColoc)
    monkeypatch.setattr(pdt, "precompute_trace_cache", lambda *args, **kwargs: {2: {}})
    monkeypatch.setattr(
        pdt,
        "_make_heatmap_worker",
        lambda args: {"taxid": args[5], "taxon_name": "Clade", "status": "success", "message": "", "filename": "x.pdf"},
    )
    monkeypatch.setattr(pdt.odp_plot, "format_matplotlib", lambda: None)
    pdt.read_simulations_and_make_heatmaps([str(sim_file)], pd.DataFrame({"taxidstring": ["1;2"]}), "alg.rbh", "out", [2], num_processes=1, skip_traces=True)


def test_axis_geometry_helpers_and_panel_generators():
    assert pdt.i2f(2, 8) == 0.25
    assert pdt.gen_square_ax(2, 4, 10, 20, 5) == [0.2, 0.2, 0.5, 0.25]
    plot_params, cbar_params = pdt.gen_square_ax_and_colorbar(2, 4, 10, 20, 5)
    assert plot_params == [0.2, 0.2, 0.5, 0.25]
    assert cbar_params[0] > plot_params[0]

    algdf = pd.DataFrame(
        {
            "ALGname": ["A", "B", "C"],
            "Color": ["#aa0000", "#00aa00", "#0000aa"],
            "Size": [1, 2, 3],
        }
    )
    sumdf = pd.DataFrame(
        {
            "ALG_num": ["ALG", "ALG", "ALG", "ALG", "num", "num", "num", "num", "num", "num", "num", "num"],
            "bin": [
                "('A', 'B')",
                "('A', 'B')",
                "('B', 'C')",
                "('B', 'C')",
                "(0.0, 0.5)",
                "(0.0, 0.5)",
                "(0.5, 1.0)",
                "(0.5, 1.0)",
                "(1, 2)",
                "(1, 2)",
                "(2, 3)",
                "(2, 3)",
            ],
            "ob_ex": ["observed", "expected", "observed", "expected", "observed", "expected", "observed", "expected", "observed", "expected", "observed", "expected"],
            "size_frac": ["frac", "frac", "frac", "frac", "frac", "frac", "frac", "frac", "size", "size", "size", "size"],
            "abs_CC": ["abs", "abs", "abs", "abs", "abs", "abs", "abs", "abs", "CC", "CC", "CC", "CC"],
            "counts": [4, 2, 6, 3, 5, 2, 7, 3, 8, 4, 9, 5],
            "count_per_sim": [4, 2, 6, 3, 5, 2, 7, 3, 8, 4, 9, 5],
        }
    )

    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    pdt.generate_ALG_obs_exp_counts_panel(axes[0, 0], axes[0, 1], sumdf, algdf)
    pdt.generate_ALG_mean_counts_panel(axes[0, 2], axes[0, 3], sumdf, algdf)
    pdt.generate_mean_counts_panel(axes[1, 0], axes[1, 1], sumdf, "frac", "abs")
    pdt.generate_obs_exp_panel(axes[1, 2], axes[1, 3], sumdf, "size", "CC")

    assert axes[0, 0].get_xlim()[1] == len(algdf)
    assert axes[0, 2].get_title().startswith("Mean count of fusion events")
    assert "Smaller ALG size" in axes[1, 0].get_xlabel()
    assert "CCs" in axes[1, 2].get_title()
    plt.close(fig)


def test_main_stats_only_mode(tmp_path, monkeypatch):
    perspchrom = tmp_path / "persp.tsv"
    perspchrom.write_text("species\ttaxidstring\nsp\t1;2\n")

    calls = {}
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        pdt,
        "generate_stats_df",
        lambda infile, outfile: calls.setdefault("stats", (infile, outfile)) or 0,
    )

    assert pdt.main([str(perspchrom), "--skip-simulations", "--skip-heatmaps"]) == 0
    assert calls["stats"] == (str(perspchrom), "statsdf.tsv")


def test_main_simulation_and_heatmap_pipeline(tmp_path, monkeypatch):
    perspchrom = tmp_path / "persp.tsv"
    alg_rbh = tmp_path / "alg.rbh"
    for path in [perspchrom, alg_rbh]:
        path.write_text("x\n")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(pdt, "generate_stats_df", lambda infile, outfile: 0)
    monkeypatch.setattr(
        pdt.rbh_tools,
        "parse_ALG_rbh_to_colordf",
        lambda _path: pd.DataFrame({"ALGname": ["A"], "Color": ["#aa0000"], "Size": [1]}),
    )

    class FakePool:
        def __init__(self, processes):
            self.processes = processes

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def imap_unordered(self, func, job_args):
            return iter(
                [
                    {"index": idx, "status": "success", "message": "", "filename": f"sim_results_{idx}_1.tsv"}
                    for idx, *_rest in job_args
                ]
            )

        def close(self):
            return None

        def join(self):
            return None

        def terminate(self):
            return None

    monkeypatch.setattr(pdt, "Pool", FakePool)
    monkeypatch.setattr(
        pdt,
        "glob",
        type(
            "FakeGlob",
            (),
            {
                "glob": staticmethod(
                    lambda pattern: [str(tmp_path / "simulations" / "sim_results_0_1.tsv")]
                    if "sim_results" in pattern
                    else []
                )
            },
        )(),
    )

    heatmap_calls = {}
    monkeypatch.setattr(
        pdt,
        "read_simulations_and_make_heatmaps",
        lambda sim_files, per_sp_df, alg_path, outprefix, clades, num_processes=1, skip_traces=False: heatmap_calls.setdefault(
            "args",
            (sim_files, per_sp_df, alg_path, outprefix, len(clades), num_processes, skip_traces),
        ),
    )

    assert (
        pdt.main(
            [
                str(perspchrom),
                str(alg_rbh),
                "--num-simulations",
                "1",
                "--sims-per-run",
                "1",
                "--num-processes",
                "2",
                "--skip-traces",
            ]
        )
        == 0
    )

    assert heatmap_calls["args"][1] == str(perspchrom)
    assert heatmap_calls["args"][2] == str(alg_rbh)
    assert heatmap_calls["args"][3] == "simulations"
    assert heatmap_calls["args"][5] == 2
    assert heatmap_calls["args"][6] is True
