# Changelog

All notable changes to `egt` are documented here.

The format follows [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Pull requests should append entries to `[Unreleased]` under the appropriate section. On release, rename `[Unreleased]` → `[X.Y.Z] - YYYY-MM-DD` and add a fresh empty `[Unreleased]` block at the top.

## [Unreleased]

## [0.2.2] - 2026-05-12

### Added
- `tests/test_dispersal.py` — 9 tests / 20 parametrized cases covering bin coverage gap, bin edge inclusion, `num_algs_detected` invariant, divergence-loader failure modes, presence-fusions meta-column deny-list, `1 − n_alg / 29` default formula, figure-close hygiene, and a regression guard banning the literal `"dispersion"` anywhere in `src/egt/`.
- `tests/test_dispersal_panel_style.py` — 24 cases covering density-alpha math (empty input, clamp, inverse-density, sqrt scaling, reference modes, error cases), `rgba_array` shape, `apply_rc` rcParams, `style_axes` spines, holozoan-constant key consistency, Creolimax pass-through, `--column-width-mm` emit, `--axes-aspect` override, CLI plumbing.
- `egt.dispersal_panel_style` module — one place for the dispersal-figure helpers (`density_alpha`, `rgba_array`, `apply_rc`, `style_axes`, `OKABE_ITO`, `ANIMAL_COLOR`, `HOLOZOAN_COLORS`, `HOLOZOAN_LABELS`, `HOLOZOAN_SPECIES`, `ALG_ORDER`, `alg_colors`).
- `egt decay-pairwise --column-width-mm <int>` — additionally emit `panels_CD_<N>mm.pdf` with total figure width exactly N millimeters; axes width solved from N and the fixed inch-anchored margins. Common values: 90 (single-column), 180 (double-column). Widths too small to fit margins raise `ValueError`.
- `egt decay-pairwise --axes-aspect <float>` — per-panel data-box width / height ratio (default ~1.236 = 0.9621 / 0.7786, the original axes geometry). Locks aspect across column widths so panels never stretch.
- `bump-my-version` tooling — `[tool.bumpversion]` config + dev extra. Future bumps: `bump-my-version bump <patch|minor|major> && git push --follow-tags`.
- `CHANGELOG.md` reformatted to Keep-a-Changelog 1.1.0; pre-existing v0.1.0 and v0.2.0 entries preserved, v0.2.1 placeholder added.

### Changed
- **CLI subcommand renamed**: `egt alg-dispersion` → `egt alg-dispersal`. Module `plot_alg_dispersion.py` → `plot_alg_dispersal.py`. Every internal identifier, output filename (`ALG_dispersion_{algname}.pdf` → `ALG_dispersal_{algname}.pdf`, `A_ALG_dispersion_by_conservation.pdf` → `A_ALG_dispersal_by_conservation.pdf`), output directory, and plot label has been swept. *Breaking — update any pipelines or Snakefiles that invoke the old subcommand name.*
- `plot_pairwise_decay_sp1_vs_all` (`{sp1}_decay_plot_vs_divergence_time.pdf`) now renders the top-row scatter panels with blue Okabe-Ito animal points using density-scaled per-point alpha (counters TimeTree-cluster oversaturation at the 691 / 707 Mya splits), ±20 Mya x-jitter, and diamond markers for holozoan callouts (Capsaspora, Salpingoeca, Creolimax) drawn at exact divergence times. Holozoan rows pass through unmodified — Creolimax `fraction_conserved = 0` across every chromosome is treated as full BCnS scramble, not as missing data.
- Typography across `decay-pairwise` panels: Helvetica with Arial / DejaVu Sans fallback, `pdf.fonttype = 42` (TrueType embedded, text stays editable in Illustrator), 0.4 pt axes/spines, 6 pt ticks, 7 pt body, no minor ticks, no gridlines, no top/right spines, no figure title. "Mya" replaces "MYA"; sentence-case axis labels.
- `egt phylotreeumap` interactive HTML output: right-side dashboard restructured (sticky Active-view banner, tabbed Summary / Legend / Rows readout), structured exploration summary card (n, scope, MRCA, shared lineage, composition), clickable legend chips re-select a color group with per-group copy-samples button, linked-tree leaf hover tooltips + Hide/Show control, custom-taxonomy helper (Myriazoa / Parahoxozoa fake taxids -67 / -68 replace Eumetazoa in lineages), theming improvements, header genome counter.
- `egt phylotreeumap-plotdfs` phyla plot: new layout via `paper_palette.yaml` + `paper_palette_simple.yaml`; clean PDF canvas alignment.
- CI workflow (`.github/workflows/ci.yml`): adds top-level `concurrency` block with `cancel-in-progress: true` so stale runs on rapid pushes terminate (saves CI minutes during force-push iteration on a PR branch).

### Fixed
- Bokeh HTML rendering in `egt phylotreeumap`: inject `<style>` / `<script>` before the actual `</body>` (not the literal one inside DOMPurify's embedded source) via `str.rpartition` instead of `str.replace`.
- Theme font stacks: single-quoted family names so they nest cleanly inside `style="..."`.
- All 8 CustomJS callbacks: extracted the delegated click handler out of `_taxonomy_summary_js()` to dodge the raw-string termination trap that silently broke active-view buttons, lineage search, lasso, summary/legend/rows updates.
- Shadow-DOM click delegation: walk `ev.composedPath()` so clicks inside Bokeh 3.x Shadow DOM resolve against `data-action` / `data-scope-key` / `data-legend-color` attributes.

### Internal
- Dropped a broken `sys.path.insert("../source")` in `plot_alg_dispersal.py` pointing at a nonexistent directory.
- `tests/test_plot_alg_dispersion.py` renamed in lockstep with the module rename.
- Coverage badge bot continues to commit `[skip ci]` updates on push-to-main.

## [0.2.1] - 2026-04-24

Pre-CHANGELOG release. See `git log v0.2.0..v0.2.1` for the commit-level detail.

## [0.2.0]

### Added
- `palette-preview` subcommand — render a Newick tree colored by a shared paper palette; emit FigTree / iTOL-ready colored and collapsed variants.
- `pigeonhole-check` subcommand — null-model test of shared ALG fusions under random ALG-to-chromosome assortment.
- `divergence-vs-dispersal` subcommand — per-clade regression of ALG dispersal on a precomputed per-species protein-divergence metric.
- `build-family-naming-map` subcommand — map BCnS ALG families to human gene IDs via the human per-species RBH (plus an optional HMM-consensus DIAMOND pass).
- `entanglement-browse` subcommand — rank clade-characteristic ALG fusion pairs.
- `entanglement-go-enrich` subcommand — hypergeometric GO enrichment for clade-characteristic ALG gene sets.
- `egt.palette` shared module — canonical clade-color source (`data/paper_palette.yaml`) with lineage-aware resolution, shared by all plotting CLIs.

## [0.1.0]

Initial release of `egt`, an analysis toolkit for characterizing chromosome evolution across metazoan genomes using reciprocal-best-hits data from [`odp`](https://github.com/conchoecia/odp).
