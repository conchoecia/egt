"""Visualization modules for GO enrichment outputs.

Every submodule writes matplotlib "Agg" backend PDFs and exposes a
`main(argv) -> int` entry point so it can be registered as a subcommand.
"""
import matplotlib
matplotlib.use("Agg")  # enforce headless backend on import
