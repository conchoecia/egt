"""GO enrichment analysis machinery for clade-specific locus pairs.

Modules:
  stats          — hypergeometric, binomial-tail, BH-FDR.
  io             — NCBI gene2accession / gene2go / family-map / .obo / unique_pairs loaders.
  enrichment     — bag-of-genes hypergeometric enrichment.
  sweep          — per-clade N-sweep driver (stability / closeness / intersection axes).
  pair_coenrich  — pair-level binomial co-enrichment test.
  plots          — visualisation (volcano, dotplot/heatmap/tree/emap/cnet/upset, pair-distance).
  benchmarks     — cross-checks (goatools, Irimia 2012 CAMPs, Simakov 2013 blocks, within-ALG).
"""
