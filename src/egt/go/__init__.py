"""GO enrichment analysis machinery for clade-specific locus pairs.

Modules:
  stats          — hypergeometric, binomial-tail, BH-FDR.
  io             — NCBI gene2accession / gene2go / family-map / .obo / unique_pairs loaders.
  enrichment     — bag-of-genes hypergeometric enrichment.
  sweep          — per-clade N-sweep driver (stability / closeness / intersection axes).
  pair_coenrich  — pair-level binomial co-enrichment test.
  plots          — visualisation (volcano, dotplot/heatmap/tree/emap/cnet/upset, pair-distance).
  benchmarks     — cross-checks (goatools, Irimia 2012 CAMPs, Simakov 2013 blocks, within-ALG).

Plot-style reference
--------------------
Visual conventions (dotplot layout, colour scheme, colorbar units,
q=0.05 reference line) are inspired by the clusterProfiler/enrichplot
R package's "dotplot for ORA" figure, documented at:

    https://yulab-smu.top/biomedical-knowledge-mining-book/enrichplot.html#fig-dotplot-setup-1

Specifically the dotplot implementation in ``plots.enrich.make_dotplots``
follows that reference for:

  - colour = raw q-value (low = red ≡ significant, high = blue) on a
    LogNorm-scaled ``RdBu`` colormap;
  - dot size = k (number of foreground families annotated to the term);
  - x-axis = log2 fold enrichment;
  - colorbar labelled in scientific notation with a dotted red line at
    q = 0.05 to mark the conventional significance threshold.

The supplementary-table column convention (k / n / K / N /
fold_enrichment / q_value) follows the same book's over-representation
analysis chapter:

    https://yulab-smu.top/biomedical-knowledge-mining-book/02-Enrichment.html
"""
