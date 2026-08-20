# egt — Evolutionary Genome Topology

[![CI](https://github.com/conchoecia/egt/actions/workflows/ci.yml/badge.svg)](https://github.com/conchoecia/egt/actions/workflows/ci.yml)
![Coverage](images/coverage-badge.svg)
[![PyPI](https://img.shields.io/pypi/v/egt.svg)](https://pypi.org/project/egt/)
[![Bioconda](https://img.shields.io/conda/vn/bioconda/egt.svg?label=bioconda)](https://anaconda.org/bioconda/egt)
[![Python versions](https://img.shields.io/pypi/pyversions/egt.svg)](https://pypi.org/project/egt/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

See [CHANGELOG.md](CHANGELOG.md) for release notes.

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

## Table of Contents

- [Overview: what egt does and how to read its plots](#overview-what-egt-does-and-how-to-read-its-plots)
  - [The core idea](#the-core-idea)
  - [Reading the UMAP plots](#reading-the-umap-plots)
  - [Where egt fits among comparative-genomics methods](#where-egt-fits-among-comparative-genomics-methods)
- [Getting Started](#getting-started)
- [Quick Start](#quick-start)
  - [PhyloTreeUMAP](#phylotreeumap--manifold-projection-of-per-species-alg-state)
  - [ALG fusion analysis](#alg-fusion-analysis-on-a-calibrated-tree)
  - [Perspective-chromosome mapping](#perspective-chromosome-tree-mapping--monte-carlo-rates)
  - [Rate, Fourier, and branch-stats analyses](#rate-analyses-fourier-periodicity-branch-stats)
  - [Phylogeny preparation](#phylogeny-preparation)
- [Users' Guide](#users-guide)
  - [Installation](#installation)
  - [Python requirements](#python-requirements)
  - [Upstream tools](#upstream-tools)
  - [CLI overview](#cli-overview)
  - [Snakemake workflows](#snakemake-workflows)
  - [Input file formats](#input-file-formats)
- [Layout](#layout)
- [Related tools](#related-tools)
- [Citing egt](#citing-egt)
- [License](#license)

## Overview: what egt does and how to read its plots

### The core idea

egt compares genomes by **where genes sit relative to one another**. For two
gene families (RBH loci) that both occur on the same chromosome in a genome, it
records how far apart they are; collected across thousands of locus pairs and
many genomes, each genome becomes a long vector of pairwise gene-family
distances. `egt phylotreeumap` projects those high-dimensional vectors into 2D
with UMAP so the relationships can be inspected visually.

Because the unit of comparison is a single pairwise distance — not an aligned
block or a fixed gene order — this tolerates rearrangement, missing data, and
lineage-specific loss: a locus pair contributes wherever both members share a
chromosome and is ignored elsewhere (missing values are encoded with a
sentinel).

### Reading the UMAP plots

Every `phylotreeumap` plot is a UMAP embedding:

- **MGT (Multi-Genome Topology)** — each point is a **genome / sample**; genomes
  with similar genome-wide gene-arrangement patterns land near each other.
- **MLT (Multi-Locus Topology)** — each point is an **ALG locus / gene family**;
  loci that keep similar cross-genome distance relationships (e.g. members of the
  same ALG) cluster together.

As with any UMAP, the absolute axis values and the spacing between clusters are
not meaningful — only the relative grouping is. To go the other way and ask
*which gene-family pairs are characteristic of a region* of an embedding, use
`egt inverse-transform`; to rank the features that distinguish a clade, use
`egt defining-features`.

### Where egt fits among comparative-genomics methods

Most genome-comparison tools work through **syntenic blocks** (e.g. MCScanX),
**gene presence/absence and pangenomes**, **rearrangement distances** (e.g.
DCJ / inversion models), or **whole-genome alignment** — they detect discrete
collinear blocks or reduce a comparison to a single number. egt is complementary
and **distance- / embedding-based**: it keeps the full landscape of pairwise
gene-family distances and projects it, so you can see how genomes or loci cluster
and how ancestral linkage structure is retained or broken. Reach for egt for an
exploratory, reference-free overview of arrangement similarity across many
genomes or loci, plus the downstream rate, fusion, and dispersal analyses — not
when you need a specific syntenic-block annotation or an exact rearrangement
count.

Input is reciprocal-best-hits (RBH) data from
[`odp`](https://github.com/conchoecia/odp) against an ALG database (e.g. the
metazoan `BCnSSimakov2022` set); the approach generalizes to any clade for which
an appropriate ortholog / ALG reference is available.

## Getting Started

Install the released package — **bioconda** pulls in every dependency
(umap-learn, bokeh, ete4, …) as conda packages, so it's the least-friction route:

```sh
# conda / mamba (recommended — resolves all deps from conda-forge + bioconda)
mamba install -c conda-forge -c bioconda egt

# or from PyPI
pip install egt

egt --help
```

To develop against a local checkout instead:

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

Released package (pulls in all dependencies):

```sh
mamba install -c conda-forge -c bioconda egt   # conda / mamba
pip install egt                                 # or PyPI
```

From source, for development:

```sh
git clone https://github.com/conchoecia/egt.git
cd egt
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

### Python requirements

Python 3.10 or newer; tested on Python 3.10–3.13 across Linux and macOS in
[CI](https://github.com/conchoecia/egt/actions/workflows/ci.yml). `pip install -e .`
pulls the deps from `pyproject.toml`:

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
alg-dispersal             — plot ALG dispersal across species
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
inverse-transform         — gene-family pairs characteristic of a UMAP embedding region
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

If you use `egt` in your work, please cite:

> Schultz, D.T., Blümel, A., Destanović, D., Sarigol, F., Simakov, O. (2026).
> *Topological mixing and irreversibility in animal chromosome evolution.*
> Science Advances **12**(34), eadz5561.
> [doi:10.1126/sciadv.adz5561](https://doi.org/10.1126/sciadv.adz5561)

For background on the topological framework for comparative genomics, see:

> Schultz, D.T., Simakov, O. (2026).
> *Topological Approaches in Animal Comparative Genomics.*
> Annual Review of Animal Biosciences 14(1), 17–48.
> [doi:10.1146/annurev-animal-030424-084541](https://doi.org/10.1146/annurev-animal-030424-084541)

See also [`CITATION.cff`](CITATION.cff).

## License

MIT — see [`LICENSE`](LICENSE).
