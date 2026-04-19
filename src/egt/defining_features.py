#!/usr/bin/env python

import argparse
from collections import Counter
import ete4 as ete3


# get the path of this script, so we know where to look for the plotdfs file
# This block imports fasta-parser as fasta
import os
import sys

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, lil_matrix, csr_matrix, save_npz, load_npz
import time

from egt.phylotreeumap import (algcomboix_file_to_dict)

def parse_args(argv=None):
    """
    Here, we need:
      - a path to a coo file
      - a path to a sample df
      - a path to the coo combination file
      - a list of the taxids for which we want to save the unique pairs tsv file
    """
    parser = argparse.ArgumentParser(description="Define features" )
    parser.add_argument("--coo_path",             type=str, help="Path to the coo file", required=True)
    parser.add_argument("--sample_df_path",       type=str, help="Path to the sample df", required=True)
    parser.add_argument("--coo_combination_path", type=str, help="Path to the coo combination file", required=True)
    parser.add_argument("--taxid_list",           type=str, help="Comma-separated list of taxids for which we want to save the unique pairs tsv file", required=False, default = "")
    # --no-qc-plots suppresses the per-clade QC PDFs that are emitted
    # alongside each *_unique_pair_df.tsv.gz by default. The automated
    # downstream tests pass this flag so they don't spend time rendering
    # matplotlib figures on tiny synthetic fixtures.
    parser.add_argument("--no-qc-plots", dest="qc_plots", action="store_false",
                        default=True,
                        help="Skip the per-clade QC PDF emission.")

    args = parser.parse_args(argv)
    # check that both the coo file and the df file exist. Same with coo combination file
    if not os.path.exists(args.coo_path):
        raise FileNotFoundError(f"{args.coo_path} does not exist")
    if not os.path.exists(args.sample_df_path):
        raise FileNotFoundError(f"{args.sample_df_path} does not exist")
    if not os.path.exists(args.coo_combination_path):
        raise FileNotFoundError(f"{args.coo_combination_path} does not exist")

    # The taxid list will be provided to us as a list of strings.
    # Here, we clean up the comma-separated list of taxids and convert it to a list of integers
    # If the taxid_list is empty, we will return an empty list.
    outlist = []
    if args.taxid_list != "":
        for taxid in args.taxid_list.split(","):
            if not taxid.isdigit():
                raise ValueError(f"taxid {taxid} is not an integer. Exiting.")
            outlist.append(int(taxid))
        args.taxid_list = outlist
    else:
        args.taxid_list = []
    return args

def load_coo(cdf, coofile, ALGcomboix, missing_value_as):
    """
    This loads a coo file and converts the missing values to the missing_value_as.
    Assumes that none of the values in the matrix will be -1, as this matrix should
      only contain positive integers.
    """
    print("loading the coo file")
    lil = load_npz(coofile).tolil()
    # check that the largest row index of the lil matrix is less than the largest index of cdf - 1
    if lil.shape[0] > max(cdf.index) + 1:
        raise ValueError(f"The largest row index of the lil matrix, {lil.shape[0]}, is greater than the largest index of cdf, {max(cdf.index)}. Exiting.")
    # check that the largest value of the ALGcomboix is less than the number of columns of the lil matrix - 1
    if max(ALGcomboix.values()) > lil.shape[1] - 1:
        raise ValueError(f"The largest value of the ALGcomboix, {max(ALGcomboix.values())}, is greater than the number of columns of the lil matrix, {lil.shape[1]}. Exiting.")

    # If the matrix is large, we have to convert the real zeros to -1 before we change to csf
    # we have to flip the values of the lil matrix
    print("setting zeros to -1")
    lil.data[lil.data == 0] = -1
    # We have to convert this to a dense matrix now. There is no way to modify the large values in a sparse matrix.
    print("Converting to a dense matrix. RAM will increase now.")
    # float32 is plenty of precision for inter-locus distances (in bp) and
    # halves the memory vs default float64. For the 202509 release's
    # 5821 species x 2.79M pairs, that's ~65 GB instead of ~130 GB.
    matrix = lil.toarray().astype(np.float32)
    del lil
    # if the missing_values is "large", then we have to convert the 0 to the missing_value_as
    # Here we switch the representation, namely we don't have to access the data with .data now that this
    #  is a dense matrix.
    print(f"setting zeros to {missing_value_as}")
    matrix[matrix == 0] = missing_value_as
    # now we convert the -1s to 0
    print("converting -1s to 0")
    matrix[matrix == -1] = 0
    return matrix

def load_coo_sparse(cdf, coofile, ALGcomboix):
    """Load the COO NPZ and return a CSR with stored zeros removed.

    The original `load_coo` above (dense path) tried to protect
    biologically-real zero-distance pairs via a `lil.data == 0` flip
    that, by inspection of the actual data, never fired the way the
    comments suggested — so in practice every stored-zero cell in the
    dense matrix was converted to NaN and dropped from the stats.

    Direct verification against the per-species RBH files on the
    202509 dataset (post_analyses/defining_features_pairs/
    zero_pairs_chrom_entries_annelida.tsv) shows that the COO's stored
    zeros are NOT biologically-real exact-neighbor observations: of
    185 Annelida stored-zero cells, zero corresponded to two orthologs
    at the same position; most had orthologs on different scaffolds or
    on the same scaffold at real distances of 7 kb to 25 Mb. So the
    stored zeros are upstream placeholders (likely "distance not
    computable" or similar), not observations.

    We therefore strip stored zeros at load time. This matches the
    effective behavior of the original code (zeros dropped) without
    pretending they're observations.
    """
    print("loading the coo file (sparse path)")
    coo = load_npz(coofile)
    if coo.shape[0] > max(cdf.index) + 1:
        raise ValueError(f"COO has more rows ({coo.shape[0]}) than cdf "
                         f"({max(cdf.index) + 1}).")
    if max(ALGcomboix.values()) > coo.shape[1] - 1:
        raise ValueError(f"ALGcomboix max index ({max(ALGcomboix.values())}) "
                         f"> COO cols - 1 ({coo.shape[1] - 1}).")
    # float64 for stable sum-of-squares (distances can be 1e8 bp → sq 1e16
    # → sum across thousands of species 1e19, beyond float32 mantissa).
    print(f"  COO: shape={coo.shape}  nnz={coo.nnz}  "
          f"density={coo.nnz/(coo.shape[0]*coo.shape[1]):.5f}")
    csr = coo.tocsr().astype(np.float64)
    del coo
    # Drop stored zeros — they're upstream placeholders (non-observations),
    # not biologically real zero-distance measurements. After this, every
    # stored entry in `csr` is a real observation with value > 0.
    n_before = csr.nnz
    csr.eliminate_zeros()
    n_after = csr.nnz
    print(f"  dropped {n_before - n_after} stored-zero cells "
          f"({(n_before - n_after) / max(n_before, 1) * 100:.4f}%)")
    return csr


def compute_col_aggregates(csr):
    """Return (notna, sum_v, sumsq_v) per column as numpy arrays.

    Assumes `csr` has already had stored zeros eliminated (see
    `load_coo_sparse`), so every stored entry is a real observation
    with value > 0. No shift is applied; the aggregates are directly
    in the observed (bp) domain.

    notna    = per-column count of stored entries.
    sum_v    = per-column sum of stored values.
    sumsq_v  = per-column sum of squares of stored values.
    """
    notna = np.asarray(csr.getnnz(axis=0)).ravel().astype(np.int64)
    sum_v = np.asarray(csr.sum(axis=0)).ravel().astype(np.float64)
    sumsq_v = np.asarray(csr.multiply(csr).sum(axis=0)).ravel().astype(np.float64)
    return notna, sum_v, sumsq_v


def _mean_std_sample(notna, sum_v, sumsq_v):
    """Return mean and sample stddev (ddof=1) per column, matching pandas.

    NaN where notna==0 for mean, or notna<2 for stddev. Negative variance
    from float noise is clamped to 0.
    """
    n = notna.astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = np.where(n > 0, sum_v / np.maximum(n, 1.0), np.nan)
        # var = (sumsq - n*mean^2) / (n - 1)
        centered = sumsq_v - n * mean * mean
        centered[~np.isfinite(centered)] = 0.0
        centered[centered < 0] = 0.0
        denom = np.where(n > 1, n - 1.0, 1.0)
        var = centered / denom
        std = np.where(n > 1, np.sqrt(var), np.nan)
    return mean, std


def compute_statistics(col, inindex, outindex):
    """
    This function is applied pairwise to the columns of the matrix.
    It computes the statistics for the in and out samples.
    This could probably be done more efficiently in the future by using a DFS, but who knows really without optimizing.
        This is fine for now. :)
    """
    len_inindex = len(inindex)
    len_outindex = len(outindex)
    invals    = col.loc[inindex]
    outvals   = col.loc[outindex]
    notna_in  = invals.notna().sum()
    notna_out = outvals.notna().sum()

    mean_in   = invals.mean()
    sd_in     = invals.std()

    mean_out  = outvals.mean()
    sd_out    = outvals.std()

    return {
        "pair":          col.name,
        "notna_in":      notna_in,
        "notna_out":     notna_out,
        "mean_in":       mean_in,
        "sd_in":         sd_in,
        "mean_out":      mean_out,
        "sd_out":        sd_out,
        "occupancy_in":  notna_in  / len_inindex,
        "occupancy_out": notna_out / len_outindex}

def process_coo_file(sampledffile, ALGcomboixfile, coofile,
                     dfoutfilepath, missing_value_as = np.nan,
                     taxid_list = [], qc_plots = True):
    """
    Handles loading in the coo file and transforms it to a matrix that we can work with.

    Required args:
      - sampledffile: str, path to the sample df
        - ALGcomboixfile: str, path to the coo combination file
        - coofile: str, path to the coo file
        - dfoutfilepath: str, path to the output df file that we will write to
    Optional args:
      - missing_value_as: int, the value that we will use to represent missing values
    """
    ###make sure missing_value_as is an integer
    ## ensure that missing_value_as is an integer or np.nan
    #if not isinstance(missing_value_as, (int, np.nan)):
    #    raise ValueError(f"missing_value_as must be an integer or np.nan. Got {missing_value_as} instead. Exiting.")

    # check that all of the relevant files are actually present
    for filepath in [sampledffile, ALGcomboixfile, coofile]:
        if not os.path.exists(filepath):
            raise ValueError(f"The filepath {filepath} does not exist. Exiting.")

    # check that the file ending for the df outfile is .df
    if not dfoutfilepath.endswith(".df"):
        raise ValueError(f"The dfoutfilepath {dfoutfilepath} does not end with '.df'. Exiting.")

    # check that the type of taxid_list is a list
    if not isinstance(taxid_list, list):
        raise ValueError(f"taxid_list must be a list. Got {taxid_list} instead. Exiting.")

    # Sparse-native path: never densify the matrix. Work on a CSR with
    # values +1-shifted (see load_coo_sparse for why — lets us distinguish
    # "observed exactly-neighboring pair, distance 0" from "not observed
    # at all", matching the original's `lil.data[lil.data == 0] = -1`
    # trick). All per-clade stats then come from four vectorized sparse
    # aggregations on row-sliced views.
    cdf = pd.read_csv(sampledffile, sep="\t", index_col=0)
    # print the columns of cdf
    # Read in the ALGcomboixfile
    ALGcomboix = algcomboix_file_to_dict(ALGcomboixfile)

    csr = load_coo_sparse(cdf, coofile, ALGcomboix)
    print(f"done loading the matrix: shape={csr.shape}  nnz={csr.nnz}")

    # Whole-matrix per-column aggregates, computed once and reused across
    # clades. For each clade we then only need the in-clade slice's
    # aggregates — out-clade falls out by subtraction, no need to
    # materialize another slice per iteration.
    t0 = time.time()
    total_notna, total_sum_v, total_sumsq_v = compute_col_aggregates(csr)
    print(f"  - global aggregates in {time.time() - t0:.1f}s")

    # now find the things that define the species
    # convert the column "taxid_list" to a list of ints. use eval
    cdf["taxid_list"] = cdf["taxid_list"].apply(eval)
    # first collect all of the ncbi taxids from the cdf dict
    all_taxids = set()
    for _, row in cdf.iterrows():
        all_taxids.update(row["taxid_list"])

    # If the user specified some taxids, we will only iterate over those taxids
    # Otherwise, go through all of the taxids in the cdf
    if len(taxid_list) > 0:
        iterate_taxids = list(taxid_list)
        # make sure that all of the taxids that we want to iterate through are in all_taxids.
        for taxid in iterate_taxids:
            if taxid not in all_taxids:
                raise ValueError(f"taxid {taxid} is not in the taxids in the cdf. Exiting.")
    else:
        iterate_taxids = sorted(all_taxids)

    # Per-species membership cached as a set per row of cdf. CSR rows are
    # integer-positional, so the N-th element of this array corresponds to
    # the N-th row of the CSR, same as the N-th row of cdf.
    taxid_sets = cdf["taxid_list"].apply(set).to_numpy()

    # load NCBI now that we know we will use it. Past the "taxid not in dataset" check
    NCBI = ete3.NCBITaxa()
    name_replacements = {" ": "", ",": "", ";": "", "(": "", ")": "",
                         ".": "", "-": "", "_": ""}
    # For each taxid, get the pairs that are unique to this taxid
    for counter, taxid in enumerate(iterate_taxids, start=1):
        nodename_raw = NCBI.get_taxid_translator([taxid])[taxid]
        nodename = nodename_raw
        for k, v in name_replacements.items():
            nodename = nodename.replace(k, v)
        outfile = f"{nodename}_{taxid}_unique_pair_df.tsv.gz"
        if os.path.exists(outfile):
            print(f"[{counter}/{len(iterate_taxids)}] {nodename_raw} -- "
                  f"{outfile} already exists. Skipping.")
            continue

        # Boolean row mask over the full CSR — True for in-clade species.
        in_mask = np.fromiter((taxid in s for s in taxid_sets),
                              count=len(taxid_sets), dtype=bool)
        n_in = int(in_mask.sum())
        n_out = int((~in_mask).sum())
        # if there are at least two samples, we should continue to analyze this
        # if the size of the out-clade is zero, we don't analyze it
        if n_in < 2 or n_out == 0:
            print(f"[{counter}/{len(iterate_taxids)}] {nodename_raw} -- "
                  f"skipped (n_in={n_in}, n_out={n_out})")
            continue

        t1 = time.time()
        csr_in = csr[in_mask, :]
        notna_in, sum_v_in, sumsq_v_in = compute_col_aggregates(csr_in)

        # Out-clade aggregates by subtraction — avoids materializing
        # csr[out_mask, :] at every iteration (which would duplicate ~28×
        # the work across the whole clade loop).
        notna_out_col = total_notna - notna_in
        sum_v_out = total_sum_v - sum_v_in
        sumsq_v_out = total_sumsq_v - sumsq_v_in

        mean_in, sd_in = _mean_std_sample(notna_in, sum_v_in, sumsq_v_in)
        mean_out, sd_out = _mean_std_sample(notna_out_col, sum_v_out, sumsq_v_out)

        # The following distribution is close to normal in log space, if not
        # a little skewed. Because it has this property, for each pair we
        # can measure where it falls in the distribution. If sd_in_out_ratio
        # is nan, that is because the value does not appear in the out
        # samples, meaning this is a new feature for this taxid. If
        # sd_in_out_ratio is 0, this means that the variance was 0 for the
        # in samples. This probably means that this pair was only measured
        # twice? So from this number we can rank the pairs based on that
        # ratio, then filter even more for well-represented pairs. From the
        # other information we can get the things that do not occur in
        # other samples, and that have a small SD or a small mean. Those
        # derived ratios live in the downstream aggregation step
        # (`defining_features_plot2.py`), not here.

        # Keep only columns that have at least one in-clade observation,
        # matching the original code's "drop all-NaN-for-this-taxid"
        # filter (innan_dict → ignore_columns before df.apply).
        keep = notna_in > 0
        pairs = np.arange(csr.shape[1], dtype=np.int64)[keep]

        unique_pair_df = pd.DataFrame({
            "pair": pairs,
            "notna_in": notna_in[keep].astype(np.int64),
            "notna_out": notna_out_col[keep].astype(np.int64),
            "mean_in": mean_in[keep],
            "sd_in": sd_in[keep],
            "mean_out": mean_out[keep],
            "sd_out": sd_out[keep],
            "occupancy_in": notna_in[keep].astype(np.float64) / n_in,
            "occupancy_out": (notna_out_col[keep].astype(np.float64)
                              / n_out),
        })
        # save this so we can play with it
        unique_pair_df.to_csv(outfile, sep="\t", index=False,
                               compression="gzip")
        print(f"[{counter}/{len(iterate_taxids)}] {nodename_raw} -- "
              f"rows={len(unique_pair_df)}  elapsed={time.time() - t1:.1f}s  "
              f"-> {outfile}")

        # Per-clade QC plots — ported from dev_scripts/
        # defining_features_plot2.py on odp's decay-branch. Skipped if
        # the caller passed --no-qc-plots (qc_plots=False) or if the
        # per-clade df has no rows left after the keep-mask.
        if qc_plots and len(unique_pair_df) > 0:
            try:
                # Import lazily so that --no-qc-plots runs don't pay the
                # matplotlib import cost.
                from egt.defining_features_qc_plots import write_qc_plots
                qc_pdf = f"{nodename}_{taxid}_unique_pairs_qc.pdf"
                write_qc_plots(
                    unique_pair_df, qc_pdf,
                    nodename=nodename_raw, taxid=taxid,
                )
                print(f"    QC plots -> {qc_pdf}")
            except Exception as exc:
                # QC plots are advisory — a plotting failure must not
                # lose the already-written per-clade TSV. Log and move on.
                print(f"    WARN: QC plot generation failed for "
                      f"{nodename_raw} ({taxid}): {exc}")

    # Historical: an empty aggregated unique_pairs.tsv.gz was written
    # here. The real aggregation (with nodename / ortholog1 / z-score /
    # close_in_clade / stable_in_clade etc. columns) lives downstream in
    # defining_features_plot2.py, so we skip this no-op write.

def main(argv=None):
    args = parse_args(argv)
    print(args)
    #process_coo_file(sampledffile, ALGcomboixfile, coofile,
    #                 dfoutfilepath, missing_value_as = 9999999999)
    if len(args.taxid_list) > 0:
        process_coo_file(args.sample_df_path, args.coo_combination_path, args.coo_path, "test.df",
                         taxid_list = args.taxid_list, qc_plots = args.qc_plots)
    else:
        process_coo_file(args.sample_df_path, args.coo_combination_path, args.coo_path, "test.df",
                         qc_plots = args.qc_plots)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())