#!/bin/bash
# Regenerate the mini_hydra and mini_urchin test fixtures from a full genome
# database on disk.  The committed fixtures are the output of this script — it
# is provided for reproducibility of the fixture-generation process, not for
# normal use.
#
# Required env var:
#   GENOMES_ROOT — directory containing the source genomes laid out as
#       <accession>/<accession>.chrFilt.pep.gz
#       <accession>/<accession>.chrFilt.chrom.gz
#       <accession>/<accession>.chr.fasta.gz
#
# Example:
#   GENOMES_ROOT=/path/to/odp_ncbi_genome_db/annotated_genomes \
#       bash generate_mini_genomes.sh

set -euo pipefail
: "${GENOMES_ROOT:?GENOMES_ROOT must point to the annotated_genomes directory}"

function analyze {
    python generate_mini_dataset.py \
        -p "${PROTEIN}" \
        -c "${CHROM}" \
        -g "${GENOME}" \
        --prefix "${PREFIX}"
}

# mini_hydra — Hydra vulgaris
ACC=GCF_022113875.1
PROTEIN=${GENOMES_ROOT}/${ACC}/${ACC}.chrFilt.pep.gz
CHROM=${GENOMES_ROOT}/${ACC}/${ACC}.chrFilt.chrom.gz
GENOME=${GENOMES_ROOT}/${ACC}/${ACC}.chr.fasta.gz
PREFIX=mini_hydra/mini_hydra
mkdir -p mini_hydra
analyze

# mini_urchin — Strongylocentrotus purpuratus
ACC=GCF_018143015.1
PROTEIN=${GENOMES_ROOT}/${ACC}/${ACC}.chrFilt.pep.gz
CHROM=${GENOMES_ROOT}/${ACC}/${ACC}.chrFilt.chrom.gz
GENOME=${GENOMES_ROOT}/${ACC}/${ACC}.chr.fasta.gz
PREFIX=mini_urchin/mini_urchin
mkdir -p mini_urchin
analyze
