"""Tests for the new egt.dispersal_panel_style module (density_alpha + style)
and the Science Advances upgrade to plot_pairwise_decay_sp1_vs_all.

Covers:
  1. density_alpha math:
     - empty input returns empty array
     - clamp to [alpha_min, alpha_max]
     - inverse-density: denser bin → lower alpha than sparse bin
     - sqrt scaling: 4× density → ~½× alpha at base
     - reference="min" gives base_alpha to the sparsest bin
     - reference="median" gives base_alpha to the median bin
     - unknown reference raises ValueError
  2. rgba_array shape + column layout matches alpha input
  3. style.apply_rc applies SA rcParams (font, fonttype, line widths)
  4. style.style_axes hides top/right spines
  5. Holozoan constants:
     - HOLOZOAN_COLORS / HOLOZOAN_LABELS keys are the same set
     - HOLOZOAN_SPECIES equals HOLOZOAN_COLORS keys
     - Creolimax full ID present (regression guard)
  6. plot_pairwise_decay_sp1_vs_all:
     - Creolimax-like all-zero row passes through (NOT filtered)
     - --column-width-mm N flag emits panels_CD_<N>mm.pdf at the requested width
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from egt import plot_decay_pairwise_steps as pdps
from egt.dispersal_panel_style import (
    ANIMAL_COLOR,
    HOLOZOAN_COLORS,
    HOLOZOAN_LABELS,
    HOLOZOAN_SPECIES,
    apply_rc,
    density_alpha,
    rgba_array,
    style_axes,
)


# --------------------------------------------------------------------- #
# 1. density_alpha math
# --------------------------------------------------------------------- #
def test_density_alpha_empty_input_returns_empty():
    out = density_alpha(np.array([]), bin_width=10.0)
    assert out.shape == (0,)


def test_density_alpha_clamps_to_alpha_min_max():
    # 100 points in one bin → very high local density → alpha should
    # bottom out at alpha_min.
    x = np.full(100, 50.0)
    out = density_alpha(x, bin_width=2.0, base_alpha=0.5,
                        alpha_min=0.05, alpha_max=0.5,
                        reference="median")
    assert (out >= 0.05 - 1e-12).all()
    assert (out <= 0.5  + 1e-12).all()


def test_density_alpha_inverse_density():
    """Sparse-bin alpha must be ≥ dense-bin alpha."""
    x = np.concatenate([np.full(20, 100.0), np.array([500.0])])
    out = density_alpha(x, bin_width=10.0, base_alpha=0.5,
                        alpha_min=0.0, alpha_max=1.0,
                        reference="median")
    dense_alpha = out[0]
    sparse_alpha = out[-1]
    assert sparse_alpha > dense_alpha


def test_density_alpha_sqrt_scaling():
    """Doubling local density should reduce alpha by ~sqrt(2)."""
    x_sparse = np.array([0.0, 0.0])                # local count 2
    x_dense  = np.array([100.0, 100.0, 100.0, 100.0,
                         100.0, 100.0, 100.0, 100.0])  # local count 8
    x = np.concatenate([x_sparse, x_dense])
    out = density_alpha(x, bin_width=2.0, base_alpha=0.4,
                        alpha_min=0.0, alpha_max=1.0,
                        reference="median")
    # With median count = 8, alpha[sparse] = 0.4 * sqrt(8/2) = 0.8
    # but clamped to alpha_max=1.0, so 0.8 stands.
    # alpha[dense]  = 0.4 * sqrt(8/8) = 0.4.
    assert out[0] == pytest.approx(0.8, abs=1e-9)
    assert out[2] == pytest.approx(0.4, abs=1e-9)


def test_density_alpha_reference_min_baselines_sparsest_bin():
    """With reference='min' the sparsest point gets base_alpha exactly."""
    x = np.array([0.0, 100.0, 100.0, 100.0])
    out = density_alpha(x, bin_width=2.0, base_alpha=0.3,
                        alpha_min=0.0, alpha_max=1.0,
                        reference="min")
    # sparsest count = 1 (point at 0); ref=1; alpha[0] = 0.3 * sqrt(1/1) = 0.3
    assert out[0] == pytest.approx(0.3, abs=1e-9)


def test_density_alpha_reference_median_baselines_median_bin():
    x = np.array([0.0, 0.0, 0.0, 100.0])
    out = density_alpha(x, bin_width=2.0, base_alpha=0.4,
                        alpha_min=0.0, alpha_max=1.0,
                        reference="median")
    # counts: [3, 3, 3, 1]; median = 3; alpha[dense] = 0.4 * sqrt(3/3) = 0.4
    assert out[0] == pytest.approx(0.4, abs=1e-9)


def test_density_alpha_unknown_reference_raises():
    with pytest.raises(ValueError, match="unknown reference"):
        density_alpha(np.array([1.0, 2.0]), reference="bogus")


# --------------------------------------------------------------------- #
# 2. rgba_array
# --------------------------------------------------------------------- #
def test_rgba_array_shape_and_alpha_channel():
    alphas = np.array([0.1, 0.5, 0.9])
    rgba = rgba_array("#0072B2", alphas)
    assert rgba.shape == (3, 4)
    # alpha column matches input exactly
    assert np.allclose(rgba[:, 3], alphas)
    # rgb triplet identical for every row
    assert np.allclose(rgba[0, :3], rgba[1, :3])
    assert np.allclose(rgba[1, :3], rgba[2, :3])


# --------------------------------------------------------------------- #
# 3. apply_rc
# --------------------------------------------------------------------- #
def test_apply_rc_sets_sa_compliant_rcparams():
    apply_rc()
    assert matplotlib.rcParams["pdf.fonttype"] == 42
    assert matplotlib.rcParams["ps.fonttype"]  == 42
    assert matplotlib.rcParams["svg.fonttype"] == "none"
    assert "Helvetica" in matplotlib.rcParams["font.sans-serif"]
    assert matplotlib.rcParams["axes.linewidth"]   == pytest.approx(0.4)
    assert matplotlib.rcParams["xtick.major.width"] == pytest.approx(0.4)
    assert matplotlib.rcParams["axes.spines.top"]   is False
    assert matplotlib.rcParams["axes.spines.right"] is False
    assert matplotlib.rcParams["legend.frameon"]    is False


# --------------------------------------------------------------------- #
# 4. style_axes
# --------------------------------------------------------------------- #
def test_style_axes_hides_top_and_right_spines():
    fig, ax = plt.subplots()
    style_axes(ax)
    assert ax.spines["top"].get_visible()   is False
    assert ax.spines["right"].get_visible() is False
    assert ax.spines["left"].get_linewidth()   == pytest.approx(0.6)
    assert ax.spines["bottom"].get_linewidth() == pytest.approx(0.6)
    plt.close(fig)


# --------------------------------------------------------------------- #
# 5. Holozoan constants
# --------------------------------------------------------------------- #
def test_holozoan_colors_and_labels_keys_match():
    assert set(HOLOZOAN_COLORS.keys()) == set(HOLOZOAN_LABELS.keys())


def test_holozoan_species_matches_colors_keys():
    assert set(HOLOZOAN_SPECIES) == set(HOLOZOAN_COLORS.keys())


def test_creolimax_full_species_id_present():
    """Regression guard: the manuscript caption keys off the exact ID."""
    assert any("Creolimax" in sp for sp in HOLOZOAN_SPECIES)
    full = "Creolimaxfragrantissima-196028-GCA033442365.1"
    assert full in HOLOZOAN_SPECIES


# --------------------------------------------------------------------- #
# 6. plot_pairwise_decay_sp1_vs_all integration
# --------------------------------------------------------------------- #
def _write_decay_tsv(path: Path, *, divergence_time: float,
                     genes_per_chrom: list[int],
                     conserved_per_chrom: list[int]) -> None:
    """Write a per-chromosome decay TSV in the egt schema."""
    n = len(genes_per_chrom)
    df = pd.DataFrame({
        "sp1_scaf":            [f"chr{i+1}" for i in range(n)],
        "sp1_scaf_genecount":  genes_per_chrom,
        "conserved":           conserved_per_chrom,
        "scattered":           [g - c for g, c in zip(genes_per_chrom, conserved_per_chrom)],
        "divergence_time":     [divergence_time] * n,
        "fraction_conserved":  [c / g if g > 0 else 0.0 for g, c in zip(genes_per_chrom, conserved_per_chrom)],
    })
    df.to_csv(path, sep="\t", index=False)


def test_plot_pairwise_decay_passes_through_zero_fraction_holozoan(tmp_path: Path):
    """A Creolimax-like row with fraction_conserved=0 on every chromosome
    must be plotted (as a holozoan diamond), NOT filtered out."""
    sp1 = "Pectenmaximus-6579-GCF902652985.1"
    animal = "GenericAnimal-1234-GCF000000001.1"
    creolimax = "Creolimaxfragrantissima-196028-GCA033442365.1"

    animal_tsv = tmp_path / "animal.tsv"
    creo_tsv   = tmp_path / "creo.tsv"
    _write_decay_tsv(animal_tsv,
                     divergence_time=500.0,
                     genes_per_chrom=[100, 100],
                     conserved_per_chrom=[80, 60])
    _write_decay_tsv(creo_tsv,
                     divergence_time=900.0,
                     genes_per_chrom=[100, 100],
                     conserved_per_chrom=[0, 0])  # full scramble

    filestruct = {sp1: {animal: str(animal_tsv), creolimax: str(creo_tsv)}}

    outdir = tmp_path / "out"
    pdps.plot_pairwise_decay_sp1_vs_all(sp1, filestruct, outdir=str(outdir))

    assert (outdir / f"{sp1}_decay_plot_vs_divergence_time.pdf").exists()


@pytest.mark.parametrize("width_mm", [90, 180])
def test_plot_pairwise_decay_column_width_emits_named_pdf(tmp_path: Path,
                                                          width_mm: int):
    """column_width_mm=N must additionally emit panels_CD_<N>mm.pdf
    alongside the main 2x2 PDF."""
    sp1 = "Pectenmaximus-6579-GCF902652985.1"
    animal = "GenericAnimal-1234-GCF000000001.1"
    animal_tsv = tmp_path / "animal.tsv"
    _write_decay_tsv(animal_tsv,
                     divergence_time=500.0,
                     genes_per_chrom=[100, 100],
                     conserved_per_chrom=[70, 80])
    filestruct = {sp1: {animal: str(animal_tsv)}}

    outdir = tmp_path / "out"
    pdps.plot_pairwise_decay_sp1_vs_all(sp1, filestruct, outdir=str(outdir),
                                        column_width_mm=width_mm)
    assert (outdir / f"{sp1}_decay_plot_vs_divergence_time.pdf").exists()
    assert (outdir / f"panels_CD_{width_mm}mm.pdf").exists()


def test_parse_args_accepts_column_width_mm_flag(tmp_path: Path):
    """CLI plumbing: --column-width-mm N must set args.column_width_mm = N."""
    cfg = tmp_path / "config.yaml"
    cfg.write_text("species: {}\n")
    args = pdps.parse_args([
        "--config", str(cfg),
        "--directory", str(tmp_path),
        "--column-width-mm", "90",
    ])
    assert args.column_width_mm == 90


def test_plot_pairwise_decay_column_width_too_small_raises(tmp_path: Path):
    """Width that leaves no room for axes raises ValueError."""
    sp1 = "Pectenmaximus-6579-GCF902652985.1"
    animal = "GenericAnimal-1234-GCF000000001.1"
    animal_tsv = tmp_path / "animal.tsv"
    _write_decay_tsv(animal_tsv,
                     divergence_time=500.0,
                     genes_per_chrom=[100, 100],
                     conserved_per_chrom=[70, 80])
    filestruct = {sp1: {animal: str(animal_tsv)}}
    with pytest.raises(ValueError, match="too small"):
        pdps.plot_pairwise_decay_sp1_vs_all(
            sp1, filestruct, outdir=str(tmp_path / "out"),
            column_width_mm=10,
        )


# --------------------------------------------------------------------- #
# 7. _compute_axes_dims — aspect-preserving geometry
# --------------------------------------------------------------------- #
@pytest.mark.parametrize("width_mm", [60, 90, 120, 180])
def test_compute_axes_dims_preserves_aspect_at_default(width_mm: int):
    """At any column width, AX_W / AX_H must equal the default aspect."""
    aspect = pdps._DEFAULT_AXES_ASPECT
    fig_w, fig_h, AX_W, AX_H = pdps._compute_axes_dims(width_mm, aspect)
    assert AX_W / AX_H == pytest.approx(aspect, rel=1e-12)


def test_compute_axes_dims_default_aspect_value():
    """The default aspect must reproduce the original 0.9621/0.7786 ratio."""
    assert pdps._DEFAULT_AXES_ASPECT == pytest.approx(0.9621 / 0.7786, rel=1e-12)


@pytest.mark.parametrize(
    "aspect, expected_relation",
    [
        (1.0, "ax_w_equals_ax_h"),
        (2.0, "ax_w_is_2x_ax_h"),
        (0.5, "ax_h_is_2x_ax_w"),
    ],
)
def test_compute_axes_dims_aspect_override(aspect: float, expected_relation: str):
    _fw, _fh, AX_W, AX_H = pdps._compute_axes_dims(90, aspect)
    if expected_relation == "ax_w_equals_ax_h":
        assert AX_W == pytest.approx(AX_H, rel=1e-12)
    elif expected_relation == "ax_w_is_2x_ax_h":
        assert AX_W == pytest.approx(2 * AX_H, rel=1e-12)
    elif expected_relation == "ax_h_is_2x_ax_w":
        assert AX_H == pytest.approx(2 * AX_W, rel=1e-12)


def test_compute_axes_dims_total_width_matches_input():
    """fig_w must equal the mm input converted to inches exactly."""
    fig_w, _, _, _ = pdps._compute_axes_dims(90, 1.236)
    assert fig_w == pytest.approx(90 / 25.4, rel=1e-12)


def test_compute_axes_dims_negative_aspect_raises():
    with pytest.raises(ValueError, match="axes_aspect"):
        pdps._compute_axes_dims(90, -1.0)


def test_compute_axes_dims_zero_aspect_raises():
    with pytest.raises(ValueError, match="axes_aspect"):
        pdps._compute_axes_dims(90, 0.0)


def test_parse_args_accepts_axes_aspect_flag(tmp_path: Path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("species: {}\n")
    args = pdps.parse_args([
        "--config", str(cfg),
        "--directory", str(tmp_path),
        "--column-width-mm", "90",
        "--axes-aspect", "1.5",
    ])
    assert args.axes_aspect == pytest.approx(1.5)


@pytest.mark.parametrize("aspect", [1.0, 1.5, 2.0])
def test_plot_pairwise_decay_respects_custom_aspect(tmp_path: Path, aspect: float):
    """Different aspect values must succeed and emit the named PDF."""
    sp1 = "Pectenmaximus-6579-GCF902652985.1"
    animal = "GenericAnimal-1234-GCF000000001.1"
    animal_tsv = tmp_path / "animal.tsv"
    _write_decay_tsv(animal_tsv,
                     divergence_time=500.0,
                     genes_per_chrom=[100, 100],
                     conserved_per_chrom=[70, 80])
    filestruct = {sp1: {animal: str(animal_tsv)}}
    outdir = tmp_path / "out"
    pdps.plot_pairwise_decay_sp1_vs_all(
        sp1, filestruct, outdir=str(outdir),
        column_width_mm=90, axes_aspect=aspect,
    )
    assert (outdir / "panels_CD_90mm.pdf").exists()
