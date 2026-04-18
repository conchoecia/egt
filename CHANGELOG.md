# Changelog

## 0.2.0

New subcommands:

- `palette-preview`: render a Newick tree colored by a shared paper palette;
  emit FigTree / iTOL-ready colored and collapsed variants.
- `pigeonhole-check`: null-model test of shared ALG fusions under random
  ALG-to-chromosome assortment.
- `divergence-vs-dispersal`: per-clade regression of ALG dispersal on a
  precomputed per-species protein-divergence metric.
- `build-family-naming-map`: map BCnS ALG families to human gene IDs via
  the human per-species RBH (plus an optional HMM-consensus DIAMOND pass).
- `entanglement-browse`: rank clade-characteristic ALG fusion pairs.
- `entanglement-go-enrich`: hypergeometric GO enrichment for clade-
  characteristic ALG gene sets.

New shared module:

- `egt.palette`: canonical clade-color source (`data/paper_palette.yaml`)
  with lineage-aware resolution, shared by all plotting CLIs.

## 0.1.0

Initial release of `egt`, an analysis toolkit for characterizing chromosome
evolution across metazoan genomes using reciprocal-best-hits data from
[`odp`](https://github.com/conchoecia/odp).
