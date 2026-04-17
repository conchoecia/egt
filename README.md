# egt — Evolutionary Genome Topology

`egt` is a Python / Snakemake analysis toolkit for characterizing
chromosome evolution across metazoan genomes. It builds on reciprocal-best-hits
data from [`odp`](https://github.com/conchoecia/odp) and provides tools for:

- ALG (ancestral linkage group) fusion, dispersal, and rate analyses
- PhyloTreeUMAP: manifold projection of per-species ALG state (MGT, MLT, and
  one-dot-one-genome variants)
- perspective-chromosome reconstruction with Monte Carlo support
- branch-wise rate analyses against a calibrated tree
- Fourier-period analysis of rate time series
- phylogenetic subsampling, tree prep, taxonomy utilities

## Getting Started

```sh
git clone https://github.com/conchoecia/egt.git
cd egt
python -m venv .venv && source .venv/bin/activate
pip install -e .

egt --help
bash tests/smoke/test_cli.sh
```

Primary input is a directory of per-species RBH files produced by `odp`
against the BCnS ALG database. From there, most analyses are a single
`egt <subcommand>` call or a Snakefile under `workflows/`.

## Quick Start

### PhyloTreeUMAP — manifold projection of per-species ALG state

```sh
# 1. build per-sample distance matrices + sampledf
egt phylotreeumap build-distances \
    --rbh-dir /path/to/rbh_files \
    --alg-name BCnSSimakov2022 \
    --sampledf-out GTUMAP/sampledf.tsv \
    --distance-dir GTUMAP/distance_matrices/

# 2. index ALG locus pairs
egt phylotreeumap algcomboix \
    --alg-rbh /path/to/LG_db/BCnSSimakov2022/BCnSSimakov2022.rbh \
    --output GTUMAP/alg_combo_to_ix.tsv

# 3. run the UMAP + HTML plot (MGT / MLT / ODOG variants)
egt phylotreeumap mgt-mlt-umap --help
```

### ALG fusion analysis on a calibrated tree

```sh
egt alg-fusions --help
```

### Perspective-chromosome tree mapping + Monte Carlo rates

```sh
egt perspchrom-df-to-tree --help
```

### Rate analyses, Fourier periodicity, branch stats

```sh
egt branch-stats-vs-time    --help
egt fourier-of-rates        --help
egt fourier-support-vs-time --help
egt collapsed-tree          --help
egt tree-changes            --help
egt decay-pairwise          --help
egt decay-many-species      --help
```

### Phylogeny preparation

```sh
egt taxids-to-newick           --help
egt newick-to-common-ancestors --help
```

## Users' Guide

`egt` is a collection of analysis scripts rather than a monolithic pipeline.
Each script is also registered as a subcommand of the `egt` console script:

```sh
egt alg-fusions --help
# equivalent to
python -m egt.plot_alg_fusions --help
```

### Installation

```sh
git clone https://github.com/conchoecia/egt.git
cd egt
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

### Python requirements

Python 3.10 or newer. `pip install -e .` pulls the deps from `pyproject.toml`:

- numpy, pandas, scipy, scikit-learn, matplotlib, networkx, Pillow
- umap-learn[plot] — UMAP + the plotting extras needed by PhyloTreeUMAP
- bokeh — interactive HTML plots
- ete4 — taxonomy trees and NCBI taxid handling
- snakemake (>=7, <9)
- pyyaml

Conda equivalent:

```sh
mamba install -c conda-forge -c bioconda \
      python=3.11 numpy pandas scipy scikit-learn matplotlib networkx pillow \
      "umap-learn" bokeh ete4 "snakemake<9" pyyaml
pip install --no-deps -e .
```

### Upstream tools

`egt` consumes outputs of several companion tools:

- [`odp`](https://github.com/conchoecia/odp) — per-species RBH files, ALG
  databases (BCnSSimakov2022 etc.)
- [`chrombase`](https://github.com/conchoecia/chrombase) — chromosome-scale
  NCBI genome database builder
- [`genbargo`](https://github.com/conchoecia/genbargo) — embargo-aware
  assembly curation
- [`chromsim`](https://github.com/conchoecia/chromsim) — chromosome-evolution
  simulations

### CLI overview

```
phylotreeumap             — UMAP-over-ALG-topology (MGT, MLT, ODOG subcommands)
phylotreeumap-subsample   — subsample species phylogenetically with per-clade caps
alg-fusions               — plot fusion events on a phylogeny (canonical v3)
alg-dispersion            — plot ALG dispersion across species
perspchrom-df-to-tree     — map perspective-chromosome changes onto a tree (Monte Carlo)
decay-pairwise            — pairwise ALG-decay analysis
decay-many-species        — cross-species ALG conservation / decay
chrom-number-vs-changes   — chromosome count vs rearrangement-rate scatter
branch-stats-vs-time      — branch statistics against geologic time
branch-stats-tree         — branch statistics laid out on a tree
branch-stats-tree-pair    — paired branch-stats tree plots
collapsed-tree            — collapsed-tree visualization
tree-changes              — per-branch changes on a tree
fourier-of-rates          — Fourier analysis of chromosomal change rates
fourier-support-vs-time   — Fourier-support-vs-time plots
count-unique-changes      — count unique changes per branch
defining-features         — identify clade-defining features
defining-features-plot    — plot defining features
defining-features-plotRBH — plot defining features on RBH dataframes
taxids-to-newick          — build a Newick tree from NCBI taxids
newick-to-common-ancestors — divergence-time annotation from a timetree
algs-split-across-scaffolds — find ALGs split across scaffolds
get-assembly-sizes        — summarize assembly sizes
pull-entries-from-yaml    — select rows from a YAML sample list
aggregate-filechecker     — aggregate filechecker benchmarks
aggregate-filesizes       — aggregate file-size summaries
join-supplementary-tables — join table fragments
phylotreeumap-plotdfs     — PhyloTreeUMAP plotting dataframe helper
```

### Snakemake workflows

Multi-stage Snakemake definitions live under `workflows/`:

```
workflows/
├── phylotree_umap.smk
├── phylotree_umap_subsampling.smk
├── perspchrom_df_stats_and_mc.smk
├── annotate_sample_df.smk
├── sample_to_num_chromosomes.smk
├── odol_annotate_blast.smk
└── pipeline/
    ├── README.md
    ├── config.template.yaml
    └── run.sh
```

Each workflow is standalone and parameterized via a YAML config.

### Input file formats

- **RBH files** (`.rbh`) — tab-separated reciprocal-best-hits output of `odp`.
  Filenames must embed the NCBI taxid as the second hyphen-separated field,
  e.g. `speciesname-7777-something.rbh`.
- **Sample dataframe** (`sampledf.tsv`) — output of
  `egt phylotreeumap build-distances`; consumed by most downstream commands.
- **ALG database RBH** — e.g. `BCnSSimakov2022.rbh`, from `odp`'s LG_db.
- **Newick trees** — ete4-readable. `egt taxids-to-newick` emits these.
- **Divergence-time tables** — TSV, as accepted by
  `egt newick-to-common-ancestors`.

## Layout

```
egt/
├── src/egt/                    — Python package
│   ├── cli.py                  — argparse dispatcher
│   ├── _vendor/                — vendored, frozen plotting utilities
│   ├── legacy/                 — prior versions of plot_ALG_fusions kept for parity
│   └── *.py                    — one module per subcommand
├── workflows/                  — Snakemake workflows
├── configs/                    — example configs
├── data/                       — small bundled data
├── tests/
│   ├── testdb/                 — mini_hydra + mini_urchin fixtures
│   └── smoke/test_cli.sh       — CLI smoke test
└── docs/
```

## Related tools

- [`odp`](https://github.com/conchoecia/odp)
- [`chrombase`](https://github.com/conchoecia/chrombase)
- [`genbargo`](https://github.com/conchoecia/genbargo)
- [`chromsim`](https://github.com/conchoecia/chromsim)

## Citing egt

If you use this toolkit, please cite:

> Schultz, D.T., Blümel, A., Destanović, D., Sarigol, F., Simakov, O. (2024).
> *Topological mixing and irreversibility in animal chromosome evolution.*
> bioRxiv. [doi:10.1101/2024.07.29.605683](https://doi.org/10.1101/2024.07.29.605683)

For background on the topological framework for comparative genomics, see:

> Schultz, D.T., Simakov, O. (2026).
> *Topological Approaches in Animal Comparative Genomics.*
> Annual Review of Animal Biosciences 14(1), 17–48.
> [doi:10.1146/annurev-animal-030424-084541](https://doi.org/10.1146/annurev-animal-030424-084541)

See also [`CITATION.cff`](CITATION.cff).

## License

MIT — see [`LICENSE`](LICENSE).
