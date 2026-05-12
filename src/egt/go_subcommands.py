"""Subcommand dispatch registry for `egt go <name>`.

The top-level `egt` CLI resolves `"go"` via `SUBCOMMANDS` to this module.
Keeping all `egt.go.*` command entry points in one registry keeps the
parent `cli.py` free of implementation detail.
"""
from __future__ import annotations

# name → (module_path, help)
GO_SUBCOMMANDS: dict[str, tuple[str, str]] = {
    "sweep":               ("egt.go.sweep",
                            "Per-clade N-sweep GO enrichment."),
    "pair-coenrich":       ("egt.go.pair_coenrich",
                            "Pair-level binomial-null co-enrichment."),
    "plot":                ("egt.go.plots.enrich",
                            "Dotplots / heatmap / tree / emap / cnet / upset."),
    "plot-volcano":        ("egt.go.plots.volcano",
                            "Volcano plots per clade."),
    "plot-pair-distance":  ("egt.go.plots.pair_distance",
                            "Per-clade pair-distance scatter."),
    "plot-pair-coenrich":  ("egt.go.plots.pair_coenrich",
                            "Summary figures for pair co-enrichment."),
    "benchmark":           ("egt.go.benchmark_dispatch",
                            "External-reference benchmarks (--ref …)."),
}
