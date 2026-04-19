# Observational / performance sanity checks (TODO_tests.md section I)

These are human-run sanity checks against the production COO after
any rebuild of `allsamples.coo.npz`. They are **not** unit tests --
they depend on the real 5,821-species production data and are too
coarse-grained to live in `pytest`. Run them when a new COO is built
and compare against the recorded 202509 baselines below.

## Baselines (recorded 2026-04 against the 202509 release)

- Shape: 5,821 species x 2,790,000 pairs.
- Stored entries (`coo.nnz`): ~9.2e8.
- Stored-zero fraction: ~0.002%.
- Vertebrata per-clade TSV row count: 5k-20k range.
- Protostomia per-clade TSV row count: 5k-20k range.
- Mollusca per-clade TSV row count: 5k-20k range.

## Check 1: COO `nnz` sanity

```python
from scipy.sparse import load_npz
coo = load_npz("path/to/allsamples.coo.npz")
print(f"shape={coo.shape}  nnz={coo.nnz}")
```

- **Alarm threshold**: `nnz` more than 20% off the 9.2e8 baseline.
  Either the species count changed (sampledf grew/shrank) or the
  `rbh_to_gb` same-scaffold policy changed upstream.

## Check 2: Stored-zero fraction

```python
import numpy as np
from scipy.sparse import load_npz
coo = load_npz("path/to/allsamples.coo.npz")
stored_zeros = int((coo.data == 0).sum())
frac = stored_zeros / coo.nnz
print(f"stored_zeros={stored_zeros:,}  fraction={frac:.5%}")
```

- **Alarm threshold**: fraction > 0.01 (1%). Current is ~2e-5 (0.002%).
  A spike indicates the upstream `rbh_to_gb` or `construct_coo_matrix_from_sampledf`
  pipeline changed how same-scaffold is decided, or a new upstream placeholder
  is leaking through.

## Check 3: Per-clade row counts for well-known clades

```python
import glob, pandas as pd
for clade in ("Vertebrata", "Protostomia", "Mollusca"):
    # The per-clade tsv name is <Nodename>_<taxid>_unique_pair_df.tsv.gz.
    hits = glob.glob(f"{clade}_*_unique_pair_df.tsv.gz")
    if not hits:
        print(f"MISSING: {clade}")
        continue
    df = pd.read_csv(hits[0], sep="\t")
    print(f"{clade}: n_rows={len(df):,}")
```

- **Expected range**: 5k-20k rows per clade (after `keep = notna_in > 0`
  filter). Order-of-magnitude changes (e.g. 500 or 200k) suggest
  either an upstream COO issue or a change to the process_coo_file
  filter semantics.

## When to run

- After every `allsamples.coo.npz` rebuild.
- Before promoting a new COO into the `post_analyses/` pipeline.
- As part of the nightly CI for the 5,821-species round-trip (TODO §K
  stretch goal), if that ever gets wired up.

## Why these aren't unit tests

These checks rely on:

1. The real 202509 production COO being accessible at a well-known
   path -- tests in CI / tmp_path can't materialize a 5k x 2.8M matrix
   cheaply.
2. Baselines that are properties of *this specific dataset*, not of
   the code. A change in species set legitimately shifts these numbers
   without a bug.

So they live here as a human checklist, not as automated coverage.
