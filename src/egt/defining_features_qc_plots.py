"""Per-clade QC plots for ``egt defining-features``.

Ported from ``dev_scripts/defining_features_plot2.py`` on the odp
``decay-branch`` at commit 898101525a8a84e4e9aad3b51504d578d08a3844
(tagged in odp as ``odp-pre-egt-split-20260416``). The original lived
as a standalone post-processor that, for each per-clade
``{nodename}_{taxid}_unique_pair_df.tsv.gz`` file, emitted five
separate single-page PDFs (each a ``make_marginal_plot``: hexbin on
the central panel, marginal histograms, plus Q-Q plots on the
top/right).

Here we consolidate those five figures into a single multi-page PDF
per clade (``*_unique_pairs_qc.pdf``) and call it directly from
``defining_features.process_coo_file`` so every defining-features run
produces QC plots alongside the existing ``*_unique_pair_df.tsv.gz``
output. Plot semantics, axis ranges, hexbin gridsize, Q-Q subsample
size, and sigma-line placement are preserved verbatim from the
original.

The actual plot primitives (``make_marginal_plot``, ``scatter_hist``,
``qq_plot``, ``add_ratio_columns``) are imported from
``egt.legacy.defining_features_plot2`` — reusing them keeps the port
honest and means the unit tests on ``add_ratio_columns`` etc. continue
to guard this code path too.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# matplotlib and the legacy plot helpers are imported lazily inside the
# entry point. This keeps the egt import graph cheap — users who call
# ``defining-features --no-qc-plots`` (or import ``process_coo_file``
# programmatically with qc_plots disabled) don't pay matplotlib startup.


def _sd_cutoff(series: pd.Series, sd_number: int, side: str = "low") -> float:
    """Compute a +/- sigma cutoff line on a pandas Series.

    ``side="low"`` returns ``mean - sd_number*std`` (matches the
    original's ``mean - (sd_number * std)``). ``side="high"`` returns
    ``mean + sd_number*std``.
    """
    mu = series.mean()
    sd = series.std()
    if side == "low":
        return mu - sd_number * sd
    return mu + sd_number * sd


def write_qc_plots(
    per_clade_df: pd.DataFrame,
    out_pdf: str | Path,
    nodename: str,
    taxid: int | str,
    sd_number: int = 2,
) -> Path:
    """Write a multi-page QC PDF for a single clade's defining-features stats.

    The five pages correspond to the five ``make_marginal_plot`` calls
    in the original ``defining_features_plot2.main`` body, in the same
    order and with the same sigma cutoffs / axis ranges:

      1. sd_in_out_ratio_log vs occupancy_in, full y-range [0, 1]
         then the occupancy>=0.5 subset with -/+ sigma vertical lines.
      2. mean_in_out_ratio_log vs occupancy_in, full then >=0.5 subset
         with -sigma vertical line.
      3. mean_in_out_ratio_log vs sd_in_out_ratio_log (no subset).
      4. mean_in vs occupancy_in, full then >=0.5 subset with -sigma
         vertical line.

    (The original emitted the sd/occupancy plot twice in a row — once
    with just the lower line and once with both upper and lower. We
    keep the consolidated "both lines" variant since it's a strict
    superset of the single-line variant.)

    Parameters
    ----------
    per_clade_df
        DataFrame with the columns written by
        ``defining_features.process_coo_file``: ``pair``, ``notna_in``,
        ``notna_out``, ``mean_in``, ``sd_in``, ``mean_out``, ``sd_out``,
        ``occupancy_in``, ``occupancy_out``.
    out_pdf
        Destination path. Will overwrite.
    nodename
        Clade name (e.g. ``Arthropoda``). Used in the PDF title only.
    taxid
        NCBI taxid, used in the PDF title.
    sd_number
        Sigma cutoff used for the vertical lines. Default 2, matching
        the original's ``--sigma 2`` default.

    Returns
    -------
    Path
        The path of the PDF written.
    """
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    from egt.legacy.defining_features_plot2 import (
        add_ratio_columns,
        make_marginal_plot,
    )

    out_pdf = Path(out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    # add_ratio_columns returns a copy with pseudocount-1-shifted
    # mean/sd and the four *_ratio / *_ratio_log columns the plots use.
    # We apply it to the full per-clade df, then drop NaN rows inside
    # each plot (same as the original).
    df = add_ratio_columns(per_clade_df, pseudocount=1)

    with PdfPages(str(out_pdf)) as pdf:
        # --- Page 1/2: sd_in_out_ratio_log vs occupancy_in --------------
        xcol, ycol = "sd_in_out_ratio_log", "occupancy_in"
        plotdf = df[[xcol, ycol]].dropna()
        x = plotdf[xcol].to_list()
        y = plotdf[ycol].to_list()
        fig = make_marginal_plot(x, y, xcol, ycol, yrange=[0, 1])
        fig.suptitle(
            f"{nodename} (taxid {taxid}) — {xcol} vs {ycol}",
            fontsize=10,
        )
        pdf.savefig(fig)
        plt.close(fig)

        # Subset to occupancy_in >= 0.5 (well-represented pairs only)
        # and draw both -sigma and +sigma cutoff lines. The original
        # also emitted a variant with only the -sigma line; this
        # combined variant is strictly more informative.
        plotdf_hi = plotdf[plotdf["occupancy_in"] >= 0.5]
        if len(plotdf_hi) > 0 and plotdf_hi[xcol].std() > 0:
            sd_low = _sd_cutoff(plotdf_hi[xcol], sd_number, side="low")
            sd_high = _sd_cutoff(plotdf_hi[xcol], sd_number, side="high")
            x = plotdf_hi[xcol].to_list()
            y = plotdf_hi[ycol].to_list()
            fig = make_marginal_plot(
                x, y, xcol, ycol,
                yrange=[0.5, 1],
                vertical_lines=[sd_low, sd_high],
                vertical_line_labels=[
                    f"-{sd_number} sigma", f"{sd_number} sigma",
                ],
            )
            fig.suptitle(
                f"{nodename} (taxid {taxid}) — {xcol} vs {ycol} "
                f"(occupancy_in>=0.5)",
                fontsize=10,
            )
            pdf.savefig(fig)
            plt.close(fig)

        # --- Page 3/4: mean_in_out_ratio_log vs occupancy_in ------------
        xcol, ycol = "mean_in_out_ratio_log", "occupancy_in"
        plotdf = df[[xcol, ycol]].dropna()
        x = plotdf[xcol].to_list()
        y = plotdf[ycol].to_list()
        fig = make_marginal_plot(x, y, xcol, ycol, yrange=[0, 1])
        fig.suptitle(
            f"{nodename} (taxid {taxid}) — {xcol} vs {ycol}",
            fontsize=10,
        )
        pdf.savefig(fig)
        plt.close(fig)

        plotdf_hi = plotdf[plotdf["occupancy_in"] >= 0.5]
        if len(plotdf_hi) > 0 and plotdf_hi[xcol].std() > 0:
            sd_low = _sd_cutoff(plotdf_hi[xcol], sd_number, side="low")
            x = plotdf_hi[xcol].to_list()
            y = plotdf_hi[ycol].to_list()
            fig = make_marginal_plot(
                x, y, xcol, ycol,
                yrange=[0.5, 1],
                vertical_lines=[sd_low],
                vertical_line_labels=[f"-{sd_number} sigma"],
            )
            fig.suptitle(
                f"{nodename} (taxid {taxid}) — {xcol} vs {ycol} "
                f"(occupancy_in>=0.5)",
                fontsize=10,
            )
            pdf.savefig(fig)
            plt.close(fig)

        # --- Page 5: mean_in_out_ratio_log vs sd_in_out_ratio_log ------
        # No occupancy subset / no yrange — this plot is only used for
        # sanity-checking that the two ratio axes move together the way
        # we expect.
        xcol, ycol = "mean_in_out_ratio_log", "sd_in_out_ratio_log"
        plotdf = df[[xcol, ycol]].dropna()
        if len(plotdf) > 0:
            x = plotdf[xcol].to_list()
            y = plotdf[ycol].to_list()
            fig = make_marginal_plot(x, y, xcol, ycol)
            fig.suptitle(
                f"{nodename} (taxid {taxid}) — {xcol} vs {ycol}",
                fontsize=10,
            )
            pdf.savefig(fig)
            plt.close(fig)

        # --- Page 6/7: mean_in vs occupancy_in -------------------------
        # This one is on the raw mean_in (in bp) rather than the log
        # ratio. It's the most-used QC plot in practice because it
        # directly shows which pairs are "close and well-represented".
        xcol, ycol = "mean_in", "occupancy_in"
        plotdf = df[[xcol, ycol]].dropna()
        x = plotdf[xcol].to_list()
        y = plotdf[ycol].to_list()
        fig = make_marginal_plot(x, y, xcol, ycol, yrange=[0, 1])
        fig.suptitle(
            f"{nodename} (taxid {taxid}) — {xcol} vs {ycol}",
            fontsize=10,
        )
        pdf.savefig(fig)
        plt.close(fig)

        plotdf_hi = plotdf[plotdf["occupancy_in"] >= 0.5]
        if len(plotdf_hi) > 0 and plotdf_hi[xcol].std() > 0:
            sd_low = _sd_cutoff(plotdf_hi[xcol], sd_number, side="low")
            x = plotdf_hi[xcol].to_list()
            y = plotdf_hi[ycol].to_list()
            fig = make_marginal_plot(
                x, y, xcol, ycol,
                yrange=[0.5, 1],
                vertical_lines=[sd_low],
                vertical_line_labels=[f"-{sd_number} sigma"],
            )
            fig.suptitle(
                f"{nodename} (taxid {taxid}) — {xcol} vs {ycol} "
                f"(occupancy_in>=0.5)",
                fontsize=10,
            )
            pdf.savefig(fig)
            plt.close(fig)

    return out_pdf


def write_qc_plots_from_tsv(
    tsv_path: str | Path,
    out_pdf: Optional[str | Path] = None,
    nodename: Optional[str] = None,
    taxid: Optional[int | str] = None,
    sd_number: int = 2,
) -> Path:
    """Convenience: load a per-clade ``*_unique_pair_df.tsv.gz`` from
    disk and write its QC PDF. If ``out_pdf`` is None, writes
    ``{stem}_qc.pdf`` next to the input. If ``nodename`` / ``taxid``
    are None they are parsed from the filename
    (``{nodename}_{taxid}_unique_pair_df.tsv.gz``).
    """
    tsv_path = Path(tsv_path)
    df = pd.read_csv(tsv_path, sep="\t")

    if nodename is None or taxid is None:
        # Filename format: {nodename}_{taxid}_unique_pair_df.tsv.gz
        stem = tsv_path.name
        for suffix in (".tsv.gz", ".tsv"):
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                break
        # Strip the trailing "_unique_pair_df" marker.
        marker = "_unique_pair_df"
        if stem.endswith(marker):
            stem = stem[: -len(marker)]
        # Split from the right: last component is taxid, rest is nodename.
        parts = stem.rsplit("_", 1)
        if len(parts) == 2:
            parsed_node, parsed_taxid = parts
        else:
            parsed_node, parsed_taxid = stem, "unknown"
        if nodename is None:
            nodename = parsed_node
        if taxid is None:
            taxid = parsed_taxid

    if out_pdf is None:
        # Strip .tsv.gz or .tsv then swap the _unique_pair_df marker for
        # _unique_pairs_qc.pdf so the QC PDF lives alongside the TSV.
        stem = tsv_path.name
        for suffix in (".tsv.gz", ".tsv"):
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                break
        out_pdf = tsv_path.parent / f"{stem}_qc.pdf"

    return write_qc_plots(df, out_pdf, nodename, taxid, sd_number=sd_number)
