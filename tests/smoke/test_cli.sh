#!/bin/bash
# Smoke test: verify the egt CLI dispatcher loads and every subcommand is
# at least discoverable. Does NOT run any analysis.
set -euo pipefail

egt --help > /dev/null
echo "egt --help: ok"

# Each subcommand should at least resolve its import and print its help.
for cmd in phylotreeumap alg-fusions perspchrom-df-to-tree \
           decay-pairwise taxids-to-newick newick-to-common-ancestors; do
    if egt "$cmd" --help > /dev/null 2>&1; then
        echo "egt $cmd --help: ok"
    else
        echo "egt $cmd --help: FAIL (module missing main(argv) or argparse)" >&2
        exit 1
    fi
done

echo "all smoke checks passed"
