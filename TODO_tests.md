# TODO — data-integrity tests for the phylotreeumap pipeline

Prompted by a concrete bug found on 2026-04-19: the per-species
`.gb.gz` file for Hirudonipponia-42736-GCA040113095.1 correctly stores
`distance = 7,391` for pair (genefamily_4381, genefamily_7471), but
the aggregated `allsamples.coo.npz` stores `15,390,028` at the
corresponding (row, col). Positions in the RBH file agree with the
gb.gz; the breakage is between the per-species gb.gz files and the
final COO.

## Tests to add

1. **RBH → gb.gz consistency.** For a given species, for every row in
   `<species>.gb.gz`, assert
   `gb.gz.distance == abs(rbh[<SAMPLE>_pos_x] − rbh[<SAMPLE>_pos_y])`
   when the two `rbh`s are on the same scaffold; no gb.gz rows
   should exist for different-scaffold pairs.

2. **Per-species gb.gz ↔ COO consistency.** For a given sample,
   `COO[row_idx, col_idx]` should equal the gb.gz's `distance` for
   the pair that `col_idx` indexes (via `combo_to_index.txt`). And
   the other direction: every gb.gz pair should appear in the COO at
   the expected cell. Test on many species so we'd catch a
   row-scramble bug in the `concatdf["row_indices"] = idx` step of
   the current COO builder.

3. **COO row alignment to `sampledf.tsv`.** For every sample,
   `COO[sampledf.index.get_loc(sample), :]` should contain exactly
   that sample's gb.gz values (not some other sample's). Run across
   ≥ 20 species stratified by clade.

4. **Run broadly.** Start with a handful of representative species
   across phyla. Promote to ≥ 100. Ideally sweep all 5,821 species
   eventually (takes a while but worth it given how much downstream
   analysis depends on the COO).

## Related tests worth including

- Confirm the semantics of stored zeros in the COO: observed evidence
  suggests they're "pair not on same scaffold" placeholders, not real
  zero-distance observations. `load_coo`'s silent-drop behavior
  (via the `lil.data == 0` mask that's effectively a no-op) relies
  on this. A test should assert zero cells come from a deliberate
  convention, not from accidental NaN-scrubbing.
- Round-trip smoke test: tiny hand-constructed RBH → gb.gz → COO
  gives back the distances the hand-constructed RBH asserts.
- Byte-match `defining_features.process_coo_file` (sparse-native
  rewrite) against the legacy dense path on a fixture.

## Why it matters

Every downstream statistic — z-scores, clade-defining-pair flags,
SupplementaryTable_16, GO enrichment — inherits from the COO. A
row-scramble bug would quietly invalidate many paper figures while
individual per-species distances still look right. These tests should
land before we trust any new SupplementaryTable regenerated from the
202509 data.

## Additional tests worth writing

### A. Input-file schema + invariants

- **RBH file schema**: required columns present (`rbh`,
  `BCnSSimakov2022_gene`, `BCnSSimakov2022_scaf`,
  `BCnSSimakov2022_pos`, `<SAMPLE>_gene`, `<SAMPLE>_scaf`,
  `<SAMPLE>_pos`, plus the `_plotindex`/`_plotpos` if the species is
  in the plotted set). Fail loud with the missing column name.
- **RBH uniqueness**: within an RBH file, `rbh` is unique (each BCnS
  family maps to at most one ortholog in that species). If
  duplicates exist the downstream pair-distance math isn't
  well-defined without an explicit paralog policy.
- **RBH positions are non-negative integers**.
- **Scaffold + plotindex monotonicity**: within each `<SAMPLE>_scaf`
  group, `<SAMPLE>_pos` sorted ascending should give `<SAMPLE>_ix`
  increasing — catches position-column swaps.
- **sampledf schema**: `sample` column unique; `taxid_list` parses
  as a list of ints; `index` is `RangeIndex(0..N-1)` (no gaps, no
  duplicates, no surprise labels — the COO build uses positional
  iteration and silently breaks if the index is anything else).
- **combo_to_index.txt invariants**: values are exactly
  `range(len(file))` (contiguous, 0-based, no duplicates); pair
  keys satisfy `rbh1 < rbh2` lexicographically; every pair appearing
  in any species' gb.gz has an entry.

### B. Distance computation

- **Same-scaffold condition is strict**: if `<SAMPLE>_scaf` differs
  between two rbhs, no distance row is emitted for them. Test with
  a synthetic 2-scaffold RBH where the two families are on
  different scaffolds — gb.gz must be empty.
- **Reflexivity**: `abs(pos_x - pos_y) == 0` iff `pos_x == pos_y`.
  Construct a 2-gene RBH at identical `_pos` values — gb.gz must
  record distance 0. (This, combined with the observed-zeros-are-
  dropped convention, is the entire basis for the "the COO stores 0
  as a placeholder" theory.)
- **Symmetry**: swapping the two `rbh` rows in the input RBH must
  not change the emitted gb.gz distance (only rbh1/rbh2 ordering).
- **Integer output**: gb.gz `distance` column is integer (bp). No
  float drift from intermediate operations.

### C. COO construction

- **Shape**: `coo.shape == (len(sampledf), len(combo_to_index))`.
- **No-row-scramble test**: for a random 20 species, re-read each
  species' gb.gz, union into `expected[(row_idx, col_idx)] = dist`,
  and assert `coo.tocsr()[row_idx, col_idx] == dist` for every
  entry and for every row. This is the direct generalization of
  the Hirudonipponia bug search.
- **Completeness**: every `(species, pair)` row in any gb.gz appears
  exactly once in the COO's stored entries. No duplicate or dropped
  cells.
- **Extraneousness**: no COO cell whose (row, col) does not
  correspond to a gb.gz row for that species.
- **Dtype**: COO values are numeric (int32 / float32 / float64
  consistent with the gb.gz `distance` dtype). Catches accidental
  string conversion.
- **Nullity of gb.gz placeholders**: if the pipeline ever emits 0
  as a placeholder (current convention), a deliberate test should
  assert the total count of stored zeros stays below some expected
  fraction. Spikes would indicate a regression.

### D. defining-features output (per-clade TSVs)

- **Schema**: every emitted per-clade tsv has columns `pair`,
  `notna_in`, `notna_out`, `mean_in`, `sd_in`, `mean_out`, `sd_out`,
  `occupancy_in`, `occupancy_out` in a stable order.
- **Integer counts**: `notna_in`, `notna_out` are non-negative
  integers.
- **Counts bounded by clade size**: `notna_in ≤ n_in_clade`;
  `notna_out ≤ n_out_clade`; `occupancy_in == notna_in / n_in_clade`;
  similarly for _out. Off-by-one errors here would be silent.
- **Means non-negative**: `mean_in`, `mean_out` are ≥ 0 whenever
  `notna > 0`, and NaN when `notna == 0`.
- **Stddev NaN semantics**: `sd_in` is NaN iff `notna_in < 2`; same
  for `sd_out`. pandas default `ddof=1` must be matched.
- **No duplicate pairs**: `pair` column is unique within the file.
- **Global aggregation invariants**: summing `notna_in + notna_out`
  for each `pair` over all clades' rows does *not* exceed the COO's
  structural nnz for that column (clades overlap nontrivially, so
  equality doesn't hold; inequality should).

### E. Sparse-native vs legacy equivalence

- **Byte-match on fixture**: hand-build a small COO (say 10 species
  × 50 pairs with a handful of stored zeros, a few observed
  zero-distances, a few NaN-like rows), run both
  `load_coo + dense_process` (legacy) and `load_coo_sparse +
  sparse_process` (rewrite). Assert the emitted per-clade TSVs
  match within 1e-7 on floats and exactly on ints.
- **Memory budget**: confirm the sparse path peaks below 100 GB on
  the 5,821 × 2.79M production COO (the dense path OOMs at 400 GB).
- **Speedup check**: sparse path finishes in < 30 min on 4 CPUs for
  28 clades; anything > 1 h regresses.

### F. defining_features_plot2.py aggregation + flag derivation

- **Coverage**: every clade listed in the taxid-list argument
  produces rows in the aggregated `unique_pairs.tsv`. Missing
  clades are a sign of a filename-parse glitch.
- **z-score formula**: `sd_in_out_ratio_log_sigma == (x - mean) / std`
  computed over the clade's pairs, where `x = log(sd_in / sd_out)`
  after dropping infinities/NaNs. Synthetic fixture with known σ
  → assert z-score matches hand-calculated value.
- **Flag thresholds**:
  - `close_in_clade == 1` ⇔ `mean_in_out_ratio_log_sigma < -2`
    AND `occupancy_in ≥ 0.5` (paper's threshold).
  - `stable_in_clade == 1` ⇔ `sd_in_out_ratio_log_sigma < -2`
    AND `occupancy_in ≥ 0.5`.
  - `unique_to_clade == 1` ⇔ `freq_outside == 0` (or equivalent:
    `notna_out == 0`).
  - `distant_in_clade`, `unstable_in_clade`: mirror with `> 2`.
- **Column order stability**: the final `unique_pairs.tsv` columns
  are stable across runs. Downstream plotting (go_enrichment_sweep,
  SupplementaryTable_16) relies on names not order, but a stable
  order makes diffs legible.

### G. Round-trip & regression

- **Tiny end-to-end fixture**: `tests/fixtures/tiny/` with 5 fake
  species, 30 pairs, known distances, known clade membership.
  Ship expected outputs (`fixture_expected_coo.npz`,
  `fixture_expected_<clade>.tsv.gz`,
  `fixture_expected_unique_pairs.tsv`). CI runs the full
  `RBH → gb.gz → COO → defining-features → defining_features_plot2`
  chain and diffs against the expected files.
- **Golden SupplementaryTable_16 snapshot**: for a fixed COO +
  fixed clade list + fixed egt commit, the emitted unique_pairs
  TSV hash matches. Changes to the algorithm require updating the
  snapshot with a PR that documents the biological implication.

### H. Edge cases

- **Singleton clade** (n_species_in = 1): skipped cleanly with an
  informative log line, no division by zero, no corrupted row.
- **Whole-dataset clade** (n_species_out = 0): also skipped cleanly
  (currently the case for Metazoa under the paper's species set).
- **Species with empty gb.gz**: must not corrupt COO row ordering.
- **Pair appears in no species**: COO column still exists (for
  stable indexing) but contains nothing; downstream must tolerate.
- **Pair in only one species**: `notna_in + notna_out == 1`,
  `sd_in` or `sd_out` is NaN. Test that downstream flags don't
  mis-derive on these (e.g., no `stable_in_clade == 1` when the
  stddev is NaN).
- **Genuine observed zero** (two orthologs at identical `<SAMPLE>_pos`):
  if the paper's convention is to drop these (current behavior),
  assert they don't show up in the stats; if the convention is to
  keep them, assert they're counted. The test documents the
  decision.
- **Massive floats**: distances up to ~10^8 bp, squared → 10^16,
  summed over 6k species → 10^19. Float32 overflows silently
  (mantissa ~7 digits). Assert the internal aggregation uses
  float64 for sum-of-squares.

### I. Observational / performance regressions

- **nnz of the COO** is in a sensible range (for 5,821 × 2.79M,
  current is ~9.2e8 stored entries). Drastic changes are suspicious.
- **Stored-zero fraction** stays near 0.002% (observed on 202509).
  A spike to 1%+ indicates an upstream change in the same-scaffold
  policy.
- **Per-clade row counts**: for standard clades with many species
  (Vertebrata, Protostomia, Mollusca), n_rows_in_clade_tsv should
  be in the 5k–20k range. Order-of-magnitude changes flag an issue.

### J. Cross-format round-trip

- **CSR/COO equivalence**: `coo.tocsr().tocoo()` preserves every
  cell (dtype and value); `coo.eliminate_zeros()` followed by
  `tocsr()` gives a CSR with the same non-zero cells as the COO
  had non-zero entries.
- **NPZ persistence**: `save_npz` → `load_npz` round-trip preserves
  structure, dtype, and shape exactly. (scipy versions have had
  regressions here historically.)

### K. Round-trip: gb.gz → COO → gb.gz (byte-match)

The tightest correctness check available. The idea is that all
information in the per-species gb.gz files is supposed to be losslessly
represented by the COO — row index for sample, col index for pair,
value for distance — so recovering per-species gb.gz files from the
COO should reproduce the originals exactly.

- **Build phase**: pick N (start with 5 species, later 50+) species;
  run the gb.gz → COO pipeline (`phylotreeumap.py:2695-2770` path)
  to produce a `small_coo.npz` over just those species.
- **Decompose phase**: new test helper (`egt._testing.gb_roundtrip`
  or similar) that takes the small COO + sampledf + combo_to_index
  and emits one gb.gz per species, with the same three columns
  (`rbh1`, `rbh2`, `distance`) that the original gb.gz has. Must
  reverse-lookup `(rbh1, rbh2)` from the pair column index and use
  the sampledf row ordering for the species name.
- **Compare phase**:
  - **gb.gz byte-match**: `zcat decomposed.gb.gz | sort -u` ==
    `zcat original.gb.gz | sort -u`. (Order may differ within a row
    group; sorting makes the comparison robust.)
  - **gb.gz value-match**: tolerant comparison on the distance
    column (int == int; floats within 1e-9 relative), catches dtype
    or precision regressions that a literal byte-diff would flag
    without helping debug.
  - **COO ↔ decomposed gb.gz**: every row in the decomposed gb.gz
    for sample S appears as a stored entry at `COO[row_idx(S), ·]`
    with the matching value; every stored entry in that row of the
    COO appears as a row in the decomposed gb.gz.

- **What this catches**:
  - Row-scramble bugs (the one we just found): decomposed gb.gz
    for Hirudonipponia would contain distances belonging to some
    other species.
  - Silent dtype promotion / truncation (distance 15,390,028.0 vs
    15,390,028).
  - Pair-direction flips (rbh1 ↔ rbh2 swapped during aggregation).
  - Zero-handling regressions (if we ever change the stored-zero
    convention, this test will flag it explicitly rather than
    leaving the effect to drift silently into downstream stats).
  - `combo_to_index.txt` key drift (added/removed pairs changing
    column indices between build runs).

- **Stretch**: run this round-trip for all 5,821 species. If the
  pipeline's COO build step is correct, every species' gb.gz
  decomposes back to its original. Even a single species with
  non-matching bytes blocks the paper until traced. Good candidate
  for a nightly CI job.

This test subsumes several of the earlier ones (C.no-row-scramble,
C.completeness, C.extraneousness, C.dtype) in one end-to-end
assertion; include both the round-trip test and the narrower ones so
the narrower ones can pinpoint where a regression lives if the
round-trip fires.
