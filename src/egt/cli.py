"""egt command-line dispatcher."""
from __future__ import annotations

import argparse
import importlib
import sys
from typing import Callable

from .go_subcommands import GO_SUBCOMMANDS

# Registry: subcommand name → (module, help text) OR a nested dict of the
# same shape (umbrella group). Each leaf module must expose
# `def main(argv: list[str] | None = None) -> int | None`.
SUBCOMMANDS: dict[str, object] = {
    "phylotreeumap":            ("egt.phylotreeumap",              "PhyloTreeUMAP — manifold projection of per-species ALG state"),
    "phylotreeumap-subsample":  ("egt.phylotreeumap_subsample",    "Subsample species for UMAP with per-clade caps"),
    "alg-fusions":              ("egt.plot_alg_fusions",           "Plot ALG fusion events across a phylogeny (v3)"),
    "alg-dispersion":           ("egt.plot_alg_dispersion",        "Plot ALG dispersion across species"),
    "perspchrom-df-to-tree":    ("egt.perspchrom_df_to_tree",      "Map perspective-chromosome changes onto a tree (Monte Carlo)"),
    "decay-pairwise":           ("egt.plot_decay_pairwise_steps",  "Pairwise evolutionary-decay analysis"),
    "decay-many-species":       ("egt.plot_decay_many_species",    "Cross-species ALG conservation / decay"),
    "chrom-number-vs-changes":  ("egt.plot_chrom_number_vs_changes","Chromosome number vs rearrangement changes"),
    "branch-stats-vs-time":     ("egt.plot_branch_stats_vs_time",  "Branch statistics against geologic time"),
    "branch-stats-tree":        ("egt.plot_branch_stats_tree",     "Branch statistics laid on a tree"),
    "branch-stats-tree-pair":   ("egt.plot_branch_stats_tree_pair","Paired branch-stats tree plots"),
    "collapsed-tree":           ("egt.plot_collapsed_tree",        "Collapsed-tree visualization"),
    "tree-changes":             ("egt.plot_tree_changes",          "Plot per-branch changes on a tree"),
    "fourier-of-rates":         ("egt.fourier_of_rates",           "Fourier analysis of chromosomal change rates"),
    "fourier-support-vs-time":  ("egt.plot_fourier_support_vs_time","Plot Fourier-support vs time"),
    "count-unique-changes":     ("egt.count_unique_changes_per_branch","Count unique changes per branch"),
    "defining-features":        ("egt.defining_features",          "Identify defining features per clade"),
    "defining-features-plot":   ("egt.defining_features_plot",     "Plot defining features"),
    "defining-features-plotRBH":("egt.defining_features_plotRBH",  "Plot defining features on RBH dataframes"),
    "taxids-to-newick":         ("egt.taxids_to_newick",           "Build a Newick tree from a list of NCBI taxids"),
    "newick-to-common-ancestors":("egt.newick_to_common_ancestors","Compute common-ancestor info and divergence times"),
    "algs-split-across-scaffolds":("egt.algs_split_across_scaffolds","Identify ALGs split across scaffolds"),
    "get-assembly-sizes":       ("egt.get_assembly_sizes",         "Summarize assembly sizes from a directory of genomes"),
    "pull-entries-from-yaml":   ("egt.pull_entries_from_yaml",     "Pull entries from a YAML sample list"),
    "aggregate-filechecker":    ("egt.aggregate_filechecker_benchmarks","Aggregate filechecker benchmarks"),
    "aggregate-filesizes":      ("egt.aggregate_filesizes",        "Aggregate file-size summaries"),
    "join-supplementary-tables":("egt.join_supplementary_tables",  "Join supplementary-table fragments"),
    "phylotreeumap-plotdfs":    ("egt.phylotreeumap_plotdfs",      "PhyloTreeUMAP plotting dataframes"),
    "palette-preview":          ("egt.palette_preview",            "Render a tree colored by the paper palette (sanity-check)"),
    "pigeonhole-check":         ("egt.pigeonhole_check",           "Null-model test for shared ALG fusions on small karyotypes"),
    "divergence-vs-dispersal":  ("egt.divergence_vs_dispersal",    "Correlate protein divergence with ALG dispersal"),
    "build-family-naming-map":  ("egt.build_family_naming_map",    "BCnS ALG family → human gene ID map"),
    "entanglement-browse":      ("egt.entanglement_browse",        "Rank clade-characteristic ALG fusion pairs"),
    "entanglement-go-enrich":   ("egt.entanglement_go_enrich",     "GO enrichment on clade-characteristic ALG genes"),
    # Umbrella group — `egt go <sub>`. Full listing in GO_SUBCOMMANDS.
    "go":                       (GO_SUBCOMMANDS,                   "GO enrichment analysis suite"),
}


def _load(module_path: str) -> Callable[[list[str] | None], int | None]:
    mod = importlib.import_module(module_path)
    fn = getattr(mod, "main", None)
    if fn is None:
        raise SystemExit(f"egt: module {module_path} has no `main(argv)` entrypoint")
    return fn


def _print_help(registry: dict, prog: str) -> None:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Evolutionary Genome Topology analysis toolkit.",
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="<command>")
    for name, val in registry.items():
        hlp = val[1] if isinstance(val, tuple) else "(subgroup — see `--help`)"
        sub.add_parser(name, help=hlp, add_help=False)
    parser.print_help()


def _dispatch(registry: dict, argv: list[str], prog: str) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        _print_help(registry, prog)
        return 0
    cmd, *rest = argv
    if cmd not in registry:
        print(f"{prog}: unknown command: {cmd!r}. "
              f"Run `{prog} --help` for a list.", file=sys.stderr)
        return 2
    entry = registry[cmd]
    # Nested group: recurse with the child registry.
    if isinstance(entry, tuple) and isinstance(entry[0], dict):
        return _dispatch(entry[0], rest, f"{prog} {cmd}")
    # Leaf: (module_path, help).
    module_path, _hlp = entry
    fn = _load(module_path)
    rv = fn(rest)
    return int(rv) if rv is not None else 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    return _dispatch(SUBCOMMANDS, argv, "egt")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
