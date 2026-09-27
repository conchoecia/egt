# Distances in gene-order steps

`egt phylotreeumap build-distances --distance-mode index` replaces positions
with consecutive, one-based ranks **within each chromosome**, then calculates
absolute rank differences between marker pairs on that chromosome. Thus three
retained markers at positions 100, 150 and 9000 receive ranks 1, 2 and 3: their
pair distances are 1, 2 and 1 steps. This follows the index definition in
EGTool's `prep_egt.py` (provided source revision `a27663d`;
https://bitbucket.org/viemet/egtool).

```bash
egt phylotreeumap build-distances \
    --rbh-dir /path/to/rbh_files \
    --alg-name BCnSSimakov2022 \
    --distance-mode index \
    --sampledf-out GTUMAP_index/sampledf.tsv \
    --distance-dir GTUMAP_index/distance_matrices/
```

The remaining MGT/MLT stages use the usual sample table, locus-pair index and
sparse matrix. Keep their outputs in a separate run directory too. Python
callers can pass `distance_mode="index"` to `rbh_to_gb`,
`rbh_to_distance_gbgz`, `rbh_directory_to_distance_matrix`, or
`sampleToRbhFileDict_to_sample_matrix`. The two PhyloTreeUMAP Snakemake
workflows accept `distance_mode: index` in their configuration and select
`GTUMAP_index` automatically.

## What is counted

Ranks count retained RBH markers, not every annotated gene. Genes missing
from the input do not occupy rank positions. EGT keeps its existing marker
selection: index mode does not add EGTool's global all-species orthogroup
intersection or paralog-selection rules. Marker coverage therefore affects
these distances and should be reported.

Only rows with an observed scaffold and coordinate are ranked. Ranks restart
on each chromosome. Positions sort numerically, and ties break by marker ID
in lexical order, making the result reproducible under input-row permutation.
Tied coordinates do not establish a biological order. Retained marker IDs
must be unique within a sample. Input dataframes are not modified.

The default `coordinate` mode continues to subtract the supplied positions;
its units are the input coordinate units (usually base pairs). Index mode
emits `{sample}.index.gb.gz`, records `retained_marker_steps` in the sample
table and distance metadata, and leaves physical `genome_size` unavailable.
Each newly generated distance file has a `.meta.json` sidecar. Reusing a
cache requires matching input and unit metadata and an intact data checksum.
Builders regenerate stale or unverified caches; legacy files remain readable
by the matrix combiner. A known unit mismatch requires a fresh output path.
The matrix combiner rejects mixed or unknown modes
when the sample table declares a `distance_mode` column.

## Ancestral and simulated genomes

Ordered marker lists can be supplied as ordinary RBH rows with chromosome
IDs and integer positions. Use the same mode for extant and simulated
genomes. The conversion does not infer missing gene order: ancestral
chromosome-membership partitions must not be sorted by marker ID and then
presented as biologically inferred order. Sample unresolved order under an
explicit scheme and identify those samples separately.

The existing sparse distance representation does not distinguish an
unobserved pair from a pair on different chromosomes. Index mode does not
change that missing-value policy. Changing observation masks can therefore
affect an MGT embedding; examine coverage and use appropriate controls before
interpreting separation among extant and sampled genomes.
