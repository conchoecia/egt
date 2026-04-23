"""CLI smoke tests — verifies the egt dispatcher and each registered
subcommand at least resolve their imports and print --help.

Real analysis correctness is not exercised here; see the per-module tests
(to be added) for that.
"""
from __future__ import annotations

import os
import subprocess
import sys

import pytest

from egt import cli


def _run(*argv: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.setdefault("NUMBA_CACHE_DIR", "/tmp/egt-test-numba-cache")
    env.setdefault("MPLCONFIGDIR", "/tmp/egt-test-mpl-cache")
    return subprocess.run(
        [sys.executable, "-m", "egt.cli", *argv],
        capture_output=True,
        text=True,
        env=env,
    )


def test_egt_help_runs():
    """`egt --help` exits 0 and lists subcommands."""
    result = _run("--help")
    assert result.returncode == 0, result.stderr
    assert "usage" in result.stdout.lower()


def test_subcommands_registry_nonempty():
    """The CLI dispatcher registers at least one subcommand."""
    assert cli.SUBCOMMANDS, "SUBCOMMANDS registry is empty"


# Subset known to work at scaffold time; other subcommands may require
# heavy imports (ete4.treeview, datashader, etc.) that aren't sanity
# checks worth blocking CI on. Expand this list as modules stabilize.
SUBCOMMANDS_SMOKE = [
    "phylotreeumap",
    "phylotreeumap-subsample",
    "alg-fusions",
    "alg-dispersion",
    "perspchrom-df-to-tree",
    "decay-pairwise",
    "decay-many-species",
    "taxids-to-newick",
    "newick-to-common-ancestors",
    "algs-split-across-scaffolds",
    "get-assembly-sizes",
    "pull-entries-from-yaml",
    "aggregate-filechecker",
    "aggregate-filesizes",
    "join-supplementary-tables",
    "phylotreeumap-plotdfs",
    "umap-taxonomy-clusters",
    "count-unique-changes",
    "branch-stats-vs-time",
    "branch-stats-tree",
    "collapsed-tree",
    "fourier-of-rates",
    "fourier-support-vs-time",
    "chrom-number-vs-changes",
]


@pytest.mark.parametrize("subcommand", SUBCOMMANDS_SMOKE)
def test_subcommand_help(subcommand: str):
    """Every listed subcommand resolves its import and responds to --help."""
    if subcommand not in cli.SUBCOMMANDS:
        pytest.skip(f"{subcommand!r} not registered in cli.SUBCOMMANDS")
    result = _run(subcommand, "--help")
    assert result.returncode == 0, (
        f"`egt {subcommand} --help` failed:\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )
