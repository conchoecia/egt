# Test suite — state, findings, and remaining work

History: originally a plan listing sections A–K of needed data-integrity tests, prompted by a 2026-04-19 row-scramble bug in `construct_coo_matrix_from_sampledf` (COO stored the wrong species' data at a given row). The bug is fixed; the plan's sections are now implemented (see the section map below). This file is retained as the state-of-the-tests record + a tracker for remaining / stretch items.

Current count: **136 passing tests** in `tests/`.

## Section → implementation map

| Section (original plan) | Status | Test file(s) |
|---|---|---|
| A — Input-file schema + invariants | implemented | `tests/test_rbh_parsing.py` |
| B — Distance computation | implemented | `tests/test_distance_computation.py` |
| C — COO construction | implemented | `tests/test_coo_construction.py` |
| — Grid integrity (mesh-bed-leveling) | implemented | `tests/test_coo_integrity.py` + `src/egt/_testing/grid_verify_coo.py` |
| D — `defining-features` per-clade output | implemented | `tests/test_defining_features_output.py` |
| E — Sparse-native vs legacy equivalence (correctness) | implemented | `tests/test_sparse_vs_legacy.py` |
| F — `defining_features_plot2` z-scores + flags | implemented | `tests/test_defining_features_flags.py` |
| G — Tiny end-to-end fixture | implemented | `tests/test_end_to_end_fixture.py` |
| H — Edge cases | implemented | `tests/test_defining_features_edge_cases.py` |
| I — Observational / performance regressions | documented (manual) | `tests/README_observational.md` |
| J — Cross-format round-trip | implemented | `tests/test_sparse_roundtrip.py` |
| K — `gb.gz → COO → gb.gz` byte-match round-trip | implemented | `tests/test_gb_roundtrip.py` + `src/egt/_testing/gb_roundtrip.py` |

## Real bugs surfaced while writing these tests

- **Row-scramble in `construct_coo_matrix_from_sampledf`** (the prompt for this work). `df["row_indices"] = idx` in the per-species iteration wrote each species' data at its LABEL position, not its POSITIONAL position; any upstream filter that left gaps in `sampledf.index` silently scrambled the COO. Fixed in `src/egt/phylotreeumap.py:2608–2633, 2728–2738` (`reset_index(drop=True)` + positional `pos - 1` assignment). Confirmed on production 202509 COO: 10 % of sampled grid cells mismatched vs. their per-species `gb.gz`.
- **Stale `source.rbh_tools` import in `src/egt/legacy/defining_features_plot2.py:17`.** The module would `ModuleNotFoundError` on any import because `source` is not a valid top-level package in the current layout. Fixed to `from egt import rbh_tools`. This blocked section F from being testable until resolved; previously worked only because the module was never actually imported in the active code path.

## Intentionally not implemented as pytest unit tests

- **I. Observational / performance regressions** (nnz sanity, stored-zero fraction, per-clade row counts). These require production-scale COO baselines (5,821 × 2.79 M) and are not reproducible in `tmp_path`. Documented as a manual checklist in `tests/README_observational.md` to be run against any production COO rebuild.
- **E. Memory budget + speedup bullets.** "Sparse path peaks < 100 GB" and "< 30 min on 4 CPUs for 28 clades" are production-scale regressions, not unit tests. Measured instead during production runs and logged in run outputs.

## Stretch / future items (not required for current trust)

- **G. Golden SupplementaryTable_16 snapshot.** For a fixed COO + clade list + egt commit, the emitted `unique_pairs.tsv` hash matches. Would require versioning the fixture as the code evolves, so deferred until algorithm is frozen for the paper.
- **K. 5,821-species round-trip.** Run the `gb.gz → COO → gb.gz` round-trip across every species, not just the synthetic fixture. Good candidate for a nightly CI job once a small production-size fixture is carved out.
- **F. Kill the inline flag-derivation copy in `defining_features_plot2.main()`.** The extracted helpers (`add_ratio_columns`, `compute_z_scores`, `assign_flags`) mirror the logic but `main()` still runs its own inline version. Future cleanup: call the helpers from `main()` and delete the duplication.

## Related operational notes

- The rebuild of the production 202509 COO (required because the scrambled file invalidated downstream analyses) is tracked in `schultz-et-al-2026/post_analyses/defining_features_pairs/run_rebuild_coo.sh` and uses the new `egt phylotreeumap combine-distances` subcommand.
- Any future COO rebuild should run `grid_verify_coo` against the product before declaring success; a `0/N` fail count is the gate. The production 202509 scramble was caught by this exact check.
