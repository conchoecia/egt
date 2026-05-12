#!/usr/bin/env python

"""
    Filename:   plot_decay_pairwise_steps_test.py
   File type:   python script (.py)
      Author:   darrin t schultz (github: @conchoecia)
Date created:   July 29th, 2023

Description:
  - The purpose of this script is to demonstrate ALG decay by performing pairwise comparisons
     over a tree. For the purposes of demonstration there will be parameters that are hardcoded
     to work with specific species combinations. The end goal is that these analyses will work
     on a phylogenetic tree.
"""

# plotting options
import matplotlib.pyplot as plt
from egt._vendor import odp_plotting_functions as odp_plot
#
import argparse
import numpy as np
import pandas as pd
import os
import pickle
import random
import sys

from egt import rbh_tools
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
import yaml
from math import ceil as ceil
from matplotlib.lines import Line2D
from tqdm import tqdm

# set up argparse method to get the directory of the .tsv files we want to plot
def parse_args(argv=None):
    """
    For the final file we will need a tree file,
    """
    parser = argparse.ArgumentParser(
        description="Plot the pairwise decay of chromosomes over a tree.",
        epilog="TREE-BASED WORKFLOW: Generate divergence times using Newick_to_common_ancestors.py, "
               "then provide the .divergence_times.txt file with --divergence_file. "
               "This eliminates the need for manual divergence_times entries in the config YAML.")
    parser.add_argument("-t", "--tree", help="The tree file (newick) to use for the analysis. The tree must have divergence dates estimated for the nodes.")
    # path to a directory containing the .rbh files with significance values calculated
    parser.add_argument("-d", "--directory", help="The directory containing the .rbh files to use for the analysis.")
    # Path to a config.yaml file that contains the parameters for the analysis we want.
    # Temporary until the tree functionality is added.
    parser.add_argument("-c", "--config", help="The config.yaml file containing the parameters for the analysis. Optional if --divergence_file is provided.")
    # NEW: Divergence times file from Newick_to_common_ancestors.py
    parser.add_argument("--divergence_file", 
        help="TSV file with pairwise divergence times (output from Newick_to_common_ancestors.py). "
             "Format: species1<TAB>species2<TAB>divergence_time_mya. "
             "If provided, this takes precedence over config file divergence_times.")
    # add a chrom file for the species we want to plot. This will help us determine which things are chroms
    parser.add_argument("-C", "--chrom", help = "The .chrom file for the species we want to plot. This will help us determine which things are chromosomes.")
    # Target_species is the species we want to plot
    # We plot this species against everything else
    parser.add_argument("-T", "--target_species", 
        help="The species name to analyze (e.g., 'Pectenmaximus-6579-GCF902652985.1'). "
             "This must match a species name in the config file. "
             "This species will be plotted against all other species.")
    # NEW: Performance and analysis parameters
    parser.add_argument("--cache_dir", default="./cache",
        help="Directory to store cached chromosome sizes (default: ./cache)")
    parser.add_argument("--min_scaf_size", type=int, default=500000,
        help="Minimum scaffold size in bp to include in analysis (default: 500000)")
    parser.add_argument("--fet_threshold", type=float, default=0.05,
        help="Fisher's Exact Test p-value threshold for significance (default: 0.05)")
    parser.add_argument("--bin_size", type=int, default=50,
        help="Time bin size in million years for violin plots (default: 50)")
    parser.add_argument("--ALG_rbh",
        help="Path to ALG database RBH file (e.g., /path/to/LG_db/BCnSSimakov2022/BCnSSimakov2022.rbh)")
    parser.add_argument("--ALGname", default="BCnS",
        help="Name of ALG database (default: BCnS)")
    parser.add_argument("--ALG_rbh_dir",
        help="Directory containing species vs ALG RBH files (e.g., /path/to/rbh_files/) for ALG mapping. "
             "Files should be named like {species}_vs_{ALGname}.rbh")
    parser.add_argument("--column-width-mm", dest="column_width_mm",
        type=int, default=None,
        help="If set, additionally emit a column-width-fitting "
             "side-by-side C+D PDF named panels_CD_<N>mm.pdf where N is "
             "this argument. Total figure width is exactly N "
             "millimeters; axes width is solved from N and fixed "
             "margins. Common journal column widths: 90 (single "
             "column), 180 (double column).")
    args = parser.parse_args(argv)
    # if the length of args is 0 print the help and quit
    if argv is None and len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    if argv is not None and len(argv) == 0:
        parser.print_help(sys.stderr)
        sys.exit(1)

    # check that the tree file actually exists
    for thisfile in [args.tree, args.divergence_file, args.config]:
        if thisfile is not None:
            if not os.path.isfile(thisfile):
                raise Exception("The file you provided does not exist: {}".format(thisfile))
        else:
            pass # no biggie if we didn't specify this file. We probably don't care.
    # check that the directory actually exists
    if not os.path.isdir(args.directory):
        raise Exception("The directory you provided does not exist.")
    return args

def read_yaml_file(file_path):
    """
    from ChatGPT
    """
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data

def load_divergence_times_from_file(divergence_file, config=None):
    """
    Load pairwise divergence times from a TSV file generated by Newick_to_common_ancestors.py.
    
    The file format is three tab-separated columns without a header:
    taxid1<TAB>taxid2<TAB>divergence_time_mya
    
    Parameters:
    -----------
    divergence_file : str
        Path to the divergence_times.txt file with taxids
    config : dict, optional
        Config dictionary with species information. If provided, used to map taxids to species names.
        Expected structure: config['species'][species_name]['taxid'] = taxid
    
    Returns:
    --------
    dict : Nested dictionary {species_name: {species_name: time_mya}}
           Uses species names from config if provided, otherwise uses taxids as strings.
           Symmetric - both (sp1, sp2) and (sp2, sp1) are populated
    
    Example:
    --------
    >>> times = load_divergence_times_from_file("mytree.divergence_times.txt", config)
    >>> print(times["Pectenmaximus-6579-GCF902652985.1"]["Crassostreakumamoto-13970-GCA..."])
    330.0
    """
    print(f"Loading divergence times from {divergence_file}...", file=sys.stderr)
    
    # Build taxid to species name mapping from config
    taxid_to_species = {}
    if config and 'species' in config:
        for species_name, species_info in config['species'].items():
            if 'taxid' in species_info:
                taxid = int(species_info['taxid'])
                taxid_to_species[taxid] = species_name
        print(f"  Found {len(taxid_to_species)} species with taxids in config", file=sys.stderr)
    
    divergence_times = {}
    num_loaded = 0
    num_skipped_no_mapping = 0
    
    with open(divergence_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) != 3:
                print(f"Warning: Skipping malformed line: {line}", file=sys.stderr)
                continue
            
            taxid1_str, taxid2_str, time_str = parts
            try:
                taxid1 = int(taxid1_str)
                taxid2 = int(taxid2_str)
                time_mya = float(time_str)
            except ValueError:
                print(f"Warning: Could not parse line: {line}", file=sys.stderr)
                continue
            
            # Map taxids to species names if config provided
            if taxid_to_species:
                if taxid1 not in taxid_to_species or taxid2 not in taxid_to_species:
                    num_skipped_no_mapping += 1
                    continue
                sp1 = taxid_to_species[taxid1]
                sp2 = taxid_to_species[taxid2]
            else:
                # Fall back to using taxids as strings
                sp1 = str(taxid1)
                sp2 = str(taxid2)
            
            # Create nested dict structure and populate symmetrically
            if sp1 not in divergence_times:
                divergence_times[sp1] = {}
            if sp2 not in divergence_times:
                divergence_times[sp2] = {}
            
            divergence_times[sp1][sp2] = time_mya
            divergence_times[sp2][sp1] = time_mya
            num_loaded += 1
    
    # Count unique species pairs (divide by 2 since symmetric)
    num_species = len(divergence_times)
    num_pairs = sum(len(v) for v in divergence_times.values()) // 2
    print(f"  Loaded {num_pairs} pairwise divergence times for {num_species} species", file=sys.stderr)
    if num_skipped_no_mapping > 0:
        print(f"  Skipped {num_skipped_no_mapping} pairs (species not in config)", file=sys.stderr)
    
    return divergence_times

def parse_config(config_file, directory_of_rbh_files, target_species, divergence_times_dict=None):
    """
    Parse config file and match RBH files to species pairs.
    
    Parameters:
    -----------
    config_file : str or None
        Path to YAML config file. Can be None if divergence_times_dict is provided.
    directory_of_rbh_files : str
        Directory containing .rbh files
    target_species : str or list
        Species to analyze
    divergence_times_dict : dict, optional
        Pre-loaded divergence times from load_divergence_times_from_file().
        If provided, takes precedence over config file's divergence_times field.

    Returns:
    --------
    dict : Configuration dictionary with analyses, analysis_files, and divergence_times
    """
    species_prefix_to_filename = {}
    filelist = os.listdir(directory_of_rbh_files) # get a list of files in the directory
    
    # Load config file if provided
    if config_file is not None:
        config = read_yaml_file(config_file)
    else:
        config = {}
    
    # add the target species to the config if it isn't there already
    if "target_species" not in config:
        if isinstance(target_species, list):
            config["target_species"] = target_species
        else:
            config["target_species"] = [target_species]

    # Use provided divergence_times_dict if available, otherwise fall back to config
    if divergence_times_dict is not None:
        print("  Using divergence times from --divergence_file", file=sys.stderr)
        config["divergence_times"] = divergence_times_dict
    elif "divergence_times" not in config:
        raise IOError("The field 'divergence_times' must be present in the config file, "
                      "or provide --divergence_file with divergence times from Newick_to_common_ancestors.py")

    # Ensure that the target_species is in the divergence_times field
    unseen_target_species = set([x for x in config["target_species"]])
    all_species_set = set()
    for sp1 in config["divergence_times"].keys():
        if sp1 in config["target_species"]:
            # remove this from the set of unseen target species
            # don't raise an error if it isn't in the set
            unseen_target_species.discard(sp1)
        for sp2 in config["divergence_times"][sp1].keys():
            all_species_set.add(sp1)
            all_species_set.add(sp2)
            if sp2 in config["target_species"]:
                # remove this from the set of unseen target species
                unseen_target_species.discard(sp2)
    if len(unseen_target_species) > 0:
        raise IOError("The target_species [{}] is not in the divergence_times field.".format(unseen_target_species))

    # This is not efficient, but go through the loop again to determine which species pairs we need
    #  We need this because later we need to check that all of these species pairs have files
    sp_to_expected_pairs = {x: set() for x in config["target_species"]}
    if "analyses" not in config:
        config["analyses"] = {x:{} for x in config["target_species"]}

    for target in config["analyses"]:
        for sp in all_species_set:
            if sp != target:
                sptup = tuple(sorted((target, sp)))
                sp_to_expected_pairs[target].add(sptup)
                # now we also need to add the divergence time to the analyses field
                if sp not in config["analyses"][target]:
                    s12 = tuple(sorted((target, sp)))
                    config["analyses"][target][sp] = config["divergence_times"][s12[0]][s12[1]]
    #for entry in sorted(config["analyses"]["Pectenmaximus6579"]):
    #    print("{}: {}".format(entry, config["analyses"]["Pectenmaximus6579"][entry]))
    #print(len(config["analyses"]["Pectenmaximus6579"]))

    # FILE PAIRING
    # for each analysis we now need to get the file to look at for the pairwise comparison
    if "analysis_files" not in config:
        config["analysis_files"] = {x:{} for x in config["target_species"]}

    # get the pairs from the files
    pair_to_file = {}
    for thisfile in [x for x in filelist if x.endswith(".rbh")]:
        fields = thisfile.split("_")
        analysis_pair = tuple(sorted((fields[0], fields[1])))
        if analysis_pair not in pair_to_file:
            pair_to_file[analysis_pair] = []
        complete_filepath = os.path.join(directory_of_rbh_files, thisfile)
        pair_to_file[analysis_pair].append(complete_filepath)

    # go through the target species analysis pairs as inferred from the divergence_times field
    #sp_to_expected_pairs = {x: set() for x in config["target_species"]}
    for target in config["target_species"]:
        for analysis_pair in sp_to_expected_pairs[target]:
            if analysis_pair not in pair_to_file:
                raise IOError("The analysis pair {} is not in the filelist.".format(analysis_pair))
            not_target = [x for x in analysis_pair if x != target][0]
            config["analysis_files"][target][not_target] = pair_to_file[analysis_pair][0]

    return config

def decay_of_one_species_pair(rawdf, sp1, sp2, sp_to_chr_to_size, min_scaf_len=500000, fet_threshold=0.05, debug=False):
    """
    This gets the 1:1-ish chromosomes from the vantage point of sp1.
    Returns a dictionary of sp1 chromosomes as keys, and lists of sp2
     chromosomes that are significant matches.

    Returns a decay dataframe that looks like this:

    #  sp1_scaf  sp2_scaf            sp1_scaf_genecount conserved  scattered
    #  PMA1      ['AHY11','AHY13']   590                339        251
    #  PMA10     ['AHY11', 'AHY9']   361                229        132
    #  PMA11     ['AHY9']            423                312        111

    Parameters:
    -----------
    rawdf : pd.DataFrame
        RBH dataframe with gene pair information
    sp1, sp2 : str
        Species names
    sp_to_chr_to_size : dict
        Nested dictionary of chromosome sizes
    min_scaf_len : int
        Minimum scaffold length in bp to include (default: 500000)
    fet_threshold : float
        Fisher's Exact Test p-value threshold (default: 0.05)

    TODO: We need to add all of the SP1 chromosomes, even if they don't have any significant matches.
    """
    # whole_FET is the column where we record the significance using FET between the two columns
    if debug:
        print(f"  [DEBUG {sp1} vs {sp2}] Initial rows in rawdf: {len(rawdf)}", file=sys.stderr)
    df = rawdf.copy()
    if debug:
        print(f"  [DEBUG {sp1} vs {sp2}] After copy: {len(df)}", file=sys.stderr)
    df = df[df["whole_FET"] < fet_threshold]
    if debug:
        print(f"  [DEBUG {sp1} vs {sp2}] After FET filter (p<{fet_threshold}): {len(df)}", file=sys.stderr)
    # At this point we only need the columns that are {sp1|sp2}_scaf and whole_FET.
    #  We have the necessary information for checking for 1:1 relationships between the chromosomes
    df = df[[x for x in df.columns if x.endswith("_scaf") or x == "whole_FET"]]
    if debug:
        print(f"  [DEBUG {sp1} vs {sp2}] After column selection: {len(df)}", file=sys.stderr)
    # filter the {sp1|sp2}_scaf to only contain scaffold IDs with lengths > min_scaf_len in the sp_to_chr_to_size dict
    #  We need to do this because we don't want to plot the small scaffolds
    MIN_SCAF_LEN = min_scaf_len
    sp1_valid_scafs = [x for x in sp_to_chr_to_size[sp1].keys() if sp_to_chr_to_size[sp1][x] > MIN_SCAF_LEN]
    if debug:
        print(f"  [DEBUG {sp1} vs {sp2}] {sp1} scaffolds > {MIN_SCAF_LEN}bp: {len(sp1_valid_scafs)} of {len(sp_to_chr_to_size[sp1])}", file=sys.stderr)
        # Show sample scaffold names from sp_to_chr_to_size
        print(f"  [DEBUG {sp1} vs {sp2}] Sample scaffolds from sp_to_chr_to_size[{sp1}]: {list(sp_to_chr_to_size[sp1].keys())[:3]}", file=sys.stderr)
        # Show unique scaffold names in the dataframe
        df_sp1_scafs = df["{}_scaf".format(sp1)].unique()
        print(f"  [DEBUG {sp1} vs {sp2}] Sample scaffolds from RBH file [{sp1}_scaf]: {list(df_sp1_scafs)[:3]}", file=sys.stderr)
        print(f"  [DEBUG {sp1} vs {sp2}] Total unique scaffolds in RBH file: {len(df_sp1_scafs)}", file=sys.stderr)
    df = df[df["{}_scaf".format(sp1)].isin(sp1_valid_scafs)]
    if debug:
        print(f"  [DEBUG {sp1} vs {sp2}] After sp1 size filter: {len(df)}", file=sys.stderr)
    # do the same thing for sp2
    sp2_valid_scafs = [x for x in sp_to_chr_to_size[sp2].keys() if sp_to_chr_to_size[sp2][x] > MIN_SCAF_LEN]
    if debug:
        print(f"  [DEBUG {sp1} vs {sp2}] {sp2} scaffolds > {MIN_SCAF_LEN}bp: {len(sp2_valid_scafs)} of {len(sp_to_chr_to_size[sp2])}", file=sys.stderr)
        if len(df) > 0:
            # Show sample scaffold names from sp_to_chr_to_size
            print(f"  [DEBUG {sp1} vs {sp2}] Sample scaffolds from sp_to_chr_to_size[{sp2}]: {list(sp_to_chr_to_size[sp2].keys())[:3]}", file=sys.stderr)
            # Show unique scaffold names in the dataframe
            df_sp2_scafs = df["{}_scaf".format(sp2)].unique()
            print(f"  [DEBUG {sp1} vs {sp2}] Sample scaffolds from RBH file [{sp2}_scaf]: {list(df_sp2_scafs)[:3]}", file=sys.stderr)
            print(f"  [DEBUG {sp1} vs {sp2}] Total unique scaffolds in RBH file: {len(df_sp2_scafs)}", file=sys.stderr)
    df = df[df["{}_scaf".format(sp2)].isin(sp2_valid_scafs)]
    if debug:
        print(f"  [DEBUG {sp1} vs {sp2}] After sp2 size filter: {len(df)}", file=sys.stderr)
    # for each chromosome in SP1, get how many significantly related chromosomes there are in SP2
    #  We need to consider 1:1 relationships
    #  We also need to consider 1:many relationships
    gb = df.groupby(["{}_scaf".format(sp1)])
    # Fix tuple bug: extract string from tuple if groupby returns tuple keys
    sp1_scaf_to_sp2_scaf = {(x[0][0] if isinstance(x[0], tuple) else x[0]): x[1]["{}_scaf".format(sp2)].unique().tolist()
                            for x in gb}
    del df
    del gb

    # For each scaf in sp1, get the percent of genes that are conserved in sp2 on the significant scaffolds from sp1_scaf_to_sp2_scaf
    # Fix tuple bug: convert tuple keys to strings
    sp1_scaf_to_total_genes_raw = rawdf.groupby(["{}_scaf".format(sp1)]).size().to_dict()
    sp1_scaf_to_total_genes = {(k[0] if isinstance(k, tuple) else k): v for k, v in sp1_scaf_to_total_genes_raw.items()}
    # groupby both sp1 scafs and sp2 scafs
    sp1gb = rawdf.groupby(["{}_scaf".format(sp1), "{}_scaf".format(sp2)])
    # for each group go through and figure out the number of genes that are on the chromosome pairs in sp1_scaf_to_sp2_scaf
    sp1_scaf_to_conserved_genes = {}
    for k in sp1_scaf_to_sp2_scaf.keys():
        sp1_scaf_to_conserved_genes[k] = 0
        for l in sp1_scaf_to_sp2_scaf[k]:
            if (k, l) in sp1gb.groups:
                sp1_scaf_to_conserved_genes[k] += len(sp1gb.groups[(k, l)])

    # now we can make a decay dataframe
    # The format will be like this:
    #  sp1_scaf  sp2_scaf            sp1_scaf_genecount conserved  scattered
    #  PMA1      ['AHY11','AHY13']   590                339        251
    #  PMA10     ['AHY11', 'AHY9']   361                229        132
    #  PMA11     ['AHY9']            423                312        111
    entries = []
    for sp1_scaf in sp_to_chr_to_size[sp1].keys():
        thisentry = { "sp1_scaf": sp1_scaf,
                      "sp2_scaf": [],
                      "sp1_scaf_genecount": 0,
                      "conserved":          0,
                      "scattered":          0
                      }
        if sp1_scaf in sp1_scaf_to_total_genes.keys():
            thisentry["sp1_scaf_genecount"] = sp1_scaf_to_total_genes[sp1_scaf]
            thisentry["scattered"]          = sp1_scaf_to_total_genes[sp1_scaf]
        if sp1_scaf in sp1_scaf_to_conserved_genes.keys():
            thisentry["conserved"]          = sp1_scaf_to_conserved_genes[sp1_scaf]
        if sp1_scaf in sp1_scaf_to_sp2_scaf.keys():
            thisentry["sp2_scaf"]           = sp1_scaf_to_sp2_scaf[sp1_scaf]
        if (sp1_scaf in sp1_scaf_to_total_genes) and (sp1_scaf in sp1_scaf_to_conserved_genes):
            thisentry["scattered"]          = sp1_scaf_to_total_genes[sp1_scaf] - sp1_scaf_to_conserved_genes[sp1_scaf]
        entries.append(thisentry)
    decaydf = pd.DataFrame(entries)
    if debug:
        print(f"  [DEBUG {sp1} vs {sp2}] Final decay dataframe rows: {len(decaydf)}", file=sys.stderr)
        non_zero_rows = decaydf[decaydf['sp1_scaf_genecount'] > 0]
        print(f"  [DEBUG {sp1} vs {sp2}] Rows with genes: {len(non_zero_rows)}", file=sys.stderr)

    return decaydf

def jitter(iterable, maxjitter):
    """
    Add a little jitter to the iterable using the maxjitter value
      as the maximum or minimum value to add.
    """
    import random
    return [max(0, x) + random.uniform(-maxjitter, maxjitter) for x in iterable]

def rbh_files_to_sp_to_chr_to_size(rbh_filelist):
    """
    Get the chromosome sizes by using the rbh files and getting the max gene indices.
    This function now includes progress bars for better user feedback.
    """
    sp_to_chr_to_size = {}
    sp_to_scaf_to_genecount = {}
    print("Calculating chromosome sizes from RBH files...", file=sys.stderr)
    for thisfile in tqdm(rbh_filelist, desc="Processing RBH files"):
        # load as pandas df
        df = pd.read_csv(thisfile, sep="\t")
        allsp = [x.replace("_gene", "") for x in df.columns if x.endswith("_gene")]
        for sp in allsp:
            spdf = df[[x for x in df.columns if x.startswith(sp)]]
            # groupby {}_scaf
            spgrp = spdf.groupby(["{}_scaf".format(sp)])
            # make a dict of the {}_scaf as the key and the max {}_pos column as the value
            # Note: x[0] returns a tuple when groupby has a list, so extract the first element
            dict_of_maxes = {x[0][0] if isinstance(x[0], tuple) else x[0]: x[1]["{}_pos".format(sp)].max() for x in spgrp}
            # Update sp_to_chr_to_size, making sure the add this species if it doesn't exist.
            #  We also need to make sure that the chromosome is in the dictionary.
            #  Only update if the current max is bigger than the existing max
            if sp not in sp_to_chr_to_size:
                sp_to_chr_to_size[sp] = {}
            for scaf in dict_of_maxes.keys():
                if scaf not in sp_to_chr_to_size[sp]:
                    sp_to_chr_to_size[sp][scaf] = 0
            for k in dict_of_maxes.keys():
                if dict_of_maxes[k] > sp_to_chr_to_size[sp][k]:
                    sp_to_chr_to_size[sp][k] = dict_of_maxes[k]

            # now add the gene list to the sp_to_scaf_to_genecount
            if sp not in sp_to_scaf_to_genecount:
                sp_to_scaf_to_genecount[sp] = {}
            for scaf in df["{}_scaf".format(sp)].unique().tolist():
                if scaf not in sp_to_scaf_to_genecount[sp]:
                    sp_to_scaf_to_genecount[sp][scaf] = set()
                subdf = df[df["{}_scaf".format(sp)] == scaf]
                sp_to_scaf_to_genecount[sp][scaf].update(subdf["{}_gene".format(sp)].tolist())

    # now return the size of the set for sp_to_scaf_to_genecount
    for sp in sp_to_scaf_to_genecount.keys():
        for scaf in sp_to_scaf_to_genecount[sp].keys():
            sp_to_scaf_to_genecount[sp][scaf] = len(sp_to_scaf_to_genecount[sp][scaf])

    return sp_to_chr_to_size, sp_to_scaf_to_genecount

def get_chromosome_sizes_cached(rbh_filelist, cache_dir="./cache"):
    """
    Get chromosome sizes with caching to avoid re-reading RBH files on subsequent runs.
    
    The cache file is stored as a pickle and includes a hash of the RBH file list
    to detect when files have changed.
    
    Parameters:
    -----------
    rbh_filelist : list
        List of paths to RBH files
    cache_dir : str
        Directory to store cache files (default: ./cache)
    
    Returns:
    --------
    tuple : (sp_to_chr_to_size, sp_to_scaf_to_genecount)
    """
    # Safely create cache directory
    os.makedirs(cache_dir, exist_ok=True)
    
    # Create a hash of the file list for cache invalidation
    import hashlib
    file_hash = hashlib.md5(
        '|'.join(sorted(rbh_filelist)).encode('utf-8')
    ).hexdigest()[:16]
    
    cache_file = os.path.join(cache_dir, f"chrom_sizes_{file_hash}.pkl")
    
    # Try to load from cache
    if os.path.exists(cache_file):
        try:
            print(f"Loading cached chromosome sizes from {cache_file}", file=sys.stderr)
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            print("  Successfully loaded cached data", file=sys.stderr)
            return cached_data
        except Exception as e:
            print(f"  Warning: Failed to load cache ({e}), recalculating...", file=sys.stderr)
    
    # Calculate from scratch
    print("Cache not found or invalid, calculating chromosome sizes...", file=sys.stderr)
    sp_to_chr_to_size, sp_to_scaf_to_genecount = rbh_files_to_sp_to_chr_to_size(rbh_filelist)
    
    # Save to cache
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump((sp_to_chr_to_size, sp_to_scaf_to_genecount), f)
        print(f"  Cached chromosome sizes to {cache_file}", file=sys.stderr)
    except Exception as e:
        print(f"  Warning: Failed to save cache ({e})", file=sys.stderr)
    
    return sp_to_chr_to_size, sp_to_scaf_to_genecount

def calculate_pairwise_decay_sp1_vs_many(sp1, config, sp_to_chr_to_size,
                                         sp_to_keepscafs, outdir="./", 
                                         min_scaf_len=500000, fet_threshold=0.05):
    """
    Calculates the pairwise chromosomal decay between two species.
    Saves the decay dataframes to files. Each file is sp1 vs sp2.
    
    Parameters:
    -----------
    min_scaf_len : int
        Minimum scaffold length in bp (default: 500000)
    fet_threshold : float
        Fisher's Exact Test p-value threshold (default: 0.05)
    """
    # just hold onto these until the end to avoid writing only some files
    sp_to_decay_df = {}
    # iterate through the pairs of species
    print(f"Calculating pairwise decay for {sp1} vs {len(config['analyses'][sp1])} species...", file=sys.stderr)
    print(f"[DEBUG] Will show detailed output for first 3 species pairs...", file=sys.stderr)
    species_list = list(config["analyses"][sp1].keys())
    show_debug = True  # Enable debug for first few
    for idx, sp2 in enumerate(species_list):
        if idx == 3:
            print(f"[DEBUG] Disabling detailed output, processing remaining {len(species_list)-3} species...", file=sys.stderr)
            show_debug = False
        if idx % 500 == 0 and idx > 0:
            print(f"[DEBUG] Progress: {idx}/{len(species_list)} species processed", file=sys.stderr)
        
        analysis_pair = tuple(sorted((sp1, sp2)))
        rbhfile = config["analysis_files"][sp1][sp2]
        
        if show_debug:
            print(f"\n[DEBUG] ===== Processing pair {idx+1}: {sp1} vs {sp2} =====", file=sys.stderr)
            print(f"[DEBUG] RBH file: {rbhfile}", file=sys.stderr)
        # read in the rbh file as a pandas df
        rawdf = pd.read_csv(rbhfile, sep="\t")

        # get the corresponding chromosomes
        sp1_sp2_decay = decay_of_one_species_pair(rawdf, sp1, sp2, sp_to_chr_to_size,
                                                   min_scaf_len=min_scaf_len,
                                                   fet_threshold=fet_threshold,
                                                   debug=show_debug)

        # Add the divergence times to the dataframe
        sp1_sp2_decay["divergence_time"] = config["analyses"][sp1][sp2]

        # Add a percent conserved column (handle division by zero)
        sp1_sp2_decay["fraction_conserved"] = sp1_sp2_decay.apply(
            lambda row: row["conserved"] / row["sp1_scaf_genecount"] if row["sp1_scaf_genecount"] > 0 else 0, 
            axis=1)

        if show_debug:
            print(f"  [DEBUG {sp1} vs {sp2}] Before 1% gene filter: {len(sp1_sp2_decay)} rows", file=sys.stderr)
        # only keep the scaffolds for species 1 that we know are valid
        sp1_sp2_decay = sp1_sp2_decay[sp1_sp2_decay["sp1_scaf"].isin(sp_to_keepscafs[sp1])]
        if show_debug:
            print(f"  [DEBUG {sp1} vs {sp2}] After 1% gene filter: {len(sp1_sp2_decay)} rows", file=sys.stderr)
            if len(sp1_sp2_decay) == 0:
                print(f"  [DEBUG {sp1} vs {sp2}] WARNING: EMPTY after 1% filter!", file=sys.stderr)

        # stash this to save to a file later
        sp_to_decay_df[sp2] = sp1_sp2_decay
        #print("\n", sp2, "\n", sp1_sp2_decay, file = sys.stderr)

    # Print summary of processing
    print(f"\n[DEBUG] Processing complete for {sp1}:", file=sys.stderr)
    print(f"  Total species processed: {len(species_list)}", file=sys.stderr)
    non_empty_count = sum(1 for sp2 in sp_to_decay_df if len(sp_to_decay_df[sp2]) > 0)
    print(f"  Non-empty dataframes: {non_empty_count}", file=sys.stderr)
    print(f"  Empty dataframes: {len(species_list) - non_empty_count}", file=sys.stderr)

    # safely make the outdir if it doesn't exist
    os.makedirs(outdir, exist_ok=True)

    sp_to_file_df = {}
    # now save the files and save the paths to a structure
    for sp2 in config["analyses"][sp1].keys():
        outprefix = "{}_vs_{}_chromosomal_decay.tsv".format(sp1, sp2)
        outdir_prefix = os.path.join(outdir, outprefix)
        # save the decay dataframe
        sp_to_decay_df[sp2].to_csv(outdir_prefix, sep="\t", index=False)
        # save the path to the decay dataframe
        sp_to_file_df[sp2] = outdir_prefix

    return {sp1: sp_to_file_df}

def plot_dispersal_by_ALG(sp1, filestruct, alg_rbh_file, algname="BCnS", alg_rbh_dir=None, outdir="./"):
    """
    Plot ALG conservation rates categorized by MEDIAN ALG dispersal levels.
    
    For each species pair:
    1. Calculate conservation % for each ALG
    2. Use MEDIAN ALG conservation to bin species pair (0-20%, 40-60%, 80-100%)
    3. Each datapoint is one ALG's conservation % in one species pair
    4. Plot box-and-whisker distributions with ALGs sorted by size
    """
    if alg_rbh_file is None:
        print(f"WARNING: No ALG RBH file provided, skipping dispersal plot for {sp1}", file=sys.stderr)
        return
    
    # Parse ALG database file to get ALG sizes and colors
    try:
        alg_df = rbh_tools.parse_ALG_rbh_to_colordf(alg_rbh_file)
    except Exception as e:
        print(f"ERROR parsing ALG RBH file {alg_rbh_file}: {e}", file=sys.stderr)
        return
    
    # Sort ALGs by size (smallest to largest) - alg_df is already sorted by size
    sorted_algs = alg_df['ALGname'].tolist()
    alg_sizes = dict(zip(alg_df['ALGname'], alg_df['Size']))
    alg_colors = dict(zip(alg_df['ALGname'], alg_df['Color']))
    
    # Load sp1 vs ALG RBH file to map genes to ALGs
    sp1_alg_mapping = {}  # {gene_id: alg_name}
    if alg_rbh_dir:
        sp1_alg_file = os.path.join(alg_rbh_dir, f"{sp1}_vs_{algname}.rbh")
        if os.path.exists(sp1_alg_file):
            try:
                sp1_alg_df = pd.read_csv(sp1_alg_file, sep="\t")
                # Map sp1 genes to ALGs
                sp1_gene_col = f"{sp1}_gene"
                if sp1_gene_col in sp1_alg_df.columns and 'gene_group' in sp1_alg_df.columns:
                    sp1_alg_mapping = dict(zip(sp1_alg_df[sp1_gene_col], sp1_alg_df['gene_group']))
                    print(f"Loaded {len(sp1_alg_mapping)} gene-to-ALG mappings for {sp1}", file=sys.stderr)
                else:
                    print(f"WARNING: {sp1_alg_file} missing required columns", file=sys.stderr)
            except Exception as e:
                print(f"WARNING: Could not load {sp1_alg_file}: {e}", file=sys.stderr)
        else:
            print(f"WARNING: {sp1_alg_file} not found", file=sys.stderr)
    
    # Define dispersal bins based on MEDIAN ALG conservation
    dispersal_bins = {
        '0%-20% dispersal': (0.0, 0.2),
        '40%-60% dispersal': (0.4, 0.6),
        '80%-100% dispersal': (0.8, 1.0)
    }
    
    # Data structure: {bin_name: {alg: [conservation values]}}
    bin_to_alg_data = {bin_name: {alg: [] for alg in sorted_algs} for bin_name in dispersal_bins.keys()}
    
    # Collect data for each species pair
    num_pairs_processed = 0
    num_pairs_binned = 0
    
    for sp2, filepath in filestruct[sp1].items():
        df = pd.read_csv(filepath, sep="\t")
        if len(df) == 0:
            continue
        
        num_pairs_processed += 1
        
        # Calculate conservation for each ALG in this species pair
        if sp1_alg_mapping:
            # Use gene-level mapping if available
            # This requires reading the gene-level data which isn't in the decay dataframe
            # For now, fall back to chromosome-based approach
            pass
        
        # Fallback: Use chromosome names as proxy for ALGs (simplified heuristic)
        alg_conservation = {}  # {alg: fraction_conserved}
        
        for alg in sorted_algs:
            # Find chromosomes that match this ALG name (simplified heuristic)
            alg_rows = df[df['sp1_scaf'].str.contains(alg, case=False, na=False)]
            
            if len(alg_rows) > 0:
                total_genes = alg_rows['sp1_scaf_genecount'].sum()
                conserved_genes = alg_rows['conserved'].sum()
                if total_genes > 0:
                    alg_conservation[alg] = conserved_genes / total_genes
                else:
                    alg_conservation[alg] = 0.0
            else:
                alg_conservation[alg] = 0.0
        
        # Calculate MEDIAN ALG conservation across all ALGs with data
        if len(alg_conservation) > 0:
            conservation_values = [v for v in alg_conservation.values() if v > 0]
            if len(conservation_values) > 0:
                median_conservation = pd.Series(conservation_values).median()
            else:
                median_conservation = 0.0
        else:
            continue
        
        # Determine which bin this species pair belongs to based on median
        target_bin = None
        for bin_name, (lower, upper) in dispersal_bins.items():
            if lower <= median_conservation <= upper:
                target_bin = bin_name
                break
        
        if target_bin is None:
            continue
        
        num_pairs_binned += 1
        
        # Add each ALG's conservation as a datapoint in the appropriate bin
        for alg in sorted_algs:
            if alg in alg_conservation:
                bin_to_alg_data[target_bin][alg].append(alg_conservation[alg])
    
    print(f"Dispersal plot: processed {num_pairs_processed} pairs, binned {num_pairs_binned}", file=sys.stderr)
    
    # Create figure with 3 panels (one for each dispersal bin)
    fig, axes = plt.subplots(1, 3, figsize=(22.5, 6))
    fig.suptitle(f"{sp1} - Total conserved ALG orthologs on scaffolds with FET p<0.05", fontsize=14)
    
    # Plot each dispersal bin
    for idx, bin_name in enumerate(['0%-20% dispersal', '40%-60% dispersal', '80%-100% dispersal']):
        ax = axes[idx]
        ax.set_title(bin_name)
        ax.set_xlabel(f"{algname} ALG")
        if idx == 0:
            ax.set_ylabel("Fraction of ALG genes conserved")
        
        # Prepare data for box-and-whisker plot
        plot_data = []
        plot_positions = []
        plot_labels = []
        plot_colors = []
        
        for pos, alg in enumerate(sorted_algs):
            data = bin_to_alg_data[bin_name][alg]
            if len(data) > 0:
                plot_data.append(data)
                plot_positions.append(pos)
                plot_labels.append(f"{alg}")
                plot_colors.append(alg_colors.get(alg, '#1f77b4'))
        
        if len(plot_data) > 0:
            # Create box-and-whisker plot
            bp = ax.boxplot(plot_data, positions=plot_positions, 
                           widths=0.6, patch_artist=True,
                           showfliers=True, notch=False)
            
            # Color the boxes
            for patch, color in zip(bp['boxes'], plot_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax.set_xticks(plot_positions)
            ax.set_xticklabels(plot_labels, rotation=90, fontsize=8, ha='center')
        else:
            ax.text(0.5, 0.5, 'No data in this bin', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
        
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(outdir, exist_ok=True)
    outprefix = f"{sp1}_ALG_dispersal_by_conservation"
    outdir_prefix = os.path.join(outdir, outprefix)
    plt.savefig(f"{outdir_prefix}.pdf", format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved dispersal plot to {outdir_prefix}.pdf", file=sys.stderr)
    """
    Plot ALG conservation rates categorized by MEDIAN ALG dispersal levels.
    
    For each species pair:
    1. Calculate conservation % for each ALG
    2. Use MEDIAN ALG conservation to bin species pair (0-20%, 40-60%, 80-100%)
    3. Each datapoint is one ALG's conservation % in one species pair
    4. Plot box-and-whisker distributions with ALGs sorted by size
    """
    if alg_rbh_file is None:
        print(f"WARNING: No ALG RBH file provided, skipping dispersal plot for {sp1}", file=sys.stderr)
        return
    
    # Parse ALG database file to get ALG sizes and colors
    try:
        alg_df = rbh_tools.parse_ALG_rbh_to_colordf(alg_rbh_file)
    except Exception as e:
        print(f"ERROR parsing ALG RBH file {alg_rbh_file}: {e}", file=sys.stderr)
        return
    
    # Sort ALGs by size (smallest to largest) - alg_df is already sorted by size
    sorted_algs = alg_df['ALGname'].tolist()
    alg_sizes = dict(zip(alg_df['ALGname'], alg_df['Size']))
    alg_colors = dict(zip(alg_df['ALGname'], alg_df['Color']))
    
    # Define dispersal bins based on MEDIAN ALG conservation
    dispersal_bins = {
        '0%-20% dispersal': (0.0, 0.2),
        '40%-60% dispersal': (0.4, 0.6),
        '80%-100% dispersal': (0.8, 1.0)
    }
    
    # Data structure: {bin_name: {alg: [conservation values]}}
    bin_to_alg_data = {bin_name: {alg: [] for alg in sorted_algs} for bin_name in dispersal_bins.keys()}
    
    # Collect data for each species pair
    num_pairs_processed = 0
    num_pairs_binned = 0
    
    for sp2, filepath in filestruct[sp1].items():
        df = pd.read_csv(filepath, sep="\t")
        if len(df) == 0:
            continue
        
        num_pairs_processed += 1
        
        # Calculate conservation for each ALG in this species pair
        # Use chromosome names as proxy for ALGs (simplified approach)
        alg_conservation = {}  # {alg: fraction_conserved}
        
        for alg in sorted_algs:
            # Find chromosomes that match this ALG name (simplified heuristic)
            alg_rows = df[df['sp1_scaf'].str.contains(alg, case=False, na=False)]
            
            if len(alg_rows) > 0:
                total_genes = alg_rows['sp1_scaf_genecount'].sum()
                conserved_genes = alg_rows['conserved'].sum()
                if total_genes > 0:
                    alg_conservation[alg] = conserved_genes / total_genes
                else:
                    alg_conservation[alg] = 0.0
            else:
                alg_conservation[alg] = 0.0
        
        # Calculate MEDIAN ALG conservation across all ALGs
        if len(alg_conservation) > 0:
            conservation_values = [v for v in alg_conservation.values() if v > 0]
            if len(conservation_values) > 0:
                median_conservation = pd.Series(conservation_values).median()
            else:
                median_conservation = 0.0
        else:
            continue
        
        # Determine which bin this species pair belongs to based on median
        target_bin = None
        for bin_name, (lower, upper) in dispersal_bins.items():
            if lower <= median_conservation <= upper:
                target_bin = bin_name
                break
        
        if target_bin is None:
            continue
        
        num_pairs_binned += 1
        
        # Add each ALG's conservation as a datapoint in the appropriate bin
        for alg in sorted_algs:
            if alg in alg_conservation:
                bin_to_alg_data[target_bin][alg].append(alg_conservation[alg])
    
    print(f"Dispersal plot: processed {num_pairs_processed} pairs, binned {num_pairs_binned}", file=sys.stderr)
    
    # Create figure with 3 panels (one for each dispersal bin)
    fig, axes = plt.subplots(1, 3, figsize=(22.5, 6))
    fig.suptitle(f"{sp1} - Total conserved ALG orthologs on scaffolds with FET p<0.05", fontsize=14)
    
    # Plot each dispersal bin
    for idx, bin_name in enumerate(['0%-20% dispersal', '40%-60% dispersal', '80%-100% dispersal']):
        ax = axes[idx]
        ax.set_title(bin_name)
        ax.set_xlabel(f"{algname} ALG")
        if idx == 0:
            ax.set_ylabel("Fraction of ALG genes conserved")
        
        # Prepare data for box-and-whisker plot
        plot_data = []
        plot_positions = []
        plot_labels = []
        plot_colors = []
        
        for pos, alg in enumerate(sorted_algs):
            data = bin_to_alg_data[bin_name][alg]
            if len(data) > 0:
                plot_data.append(data)
                plot_positions.append(pos)
                plot_labels.append(f"{alg}")
                plot_colors.append(alg_colors.get(alg, '#1f77b4'))
        
        if len(plot_data) > 0:
            # Create box-and-whisker plot
            bp = ax.boxplot(plot_data, positions=plot_positions, 
                           widths=0.6, patch_artist=True,
                           showfliers=True, notch=False)
            
            # Color the boxes
            for patch, color in zip(bp['boxes'], plot_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax.set_xticks(plot_positions)
            ax.set_xticklabels(plot_labels, rotation=90, fontsize=8, ha='center')
        else:
            ax.text(0.5, 0.5, 'No data in this bin', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
        
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(outdir, exist_ok=True)
    outprefix = f"{sp1}_ALG_dispersal_by_conservation"
    outdir_prefix = os.path.join(outdir, outprefix)
    plt.savefig(f"{outdir_prefix}.pdf", format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved dispersal plot to {outdir_prefix}.pdf", file=sys.stderr)

def get_chromosome_to_dominant_alg(species, alg_rbh_dir, alg_rbh_file, algname):
    """
    For a given species, determine which BCnS ALG dominates each chromosome.
    
    Args:
        species: Species name (e.g., 'Pectenmaximus-6579-GCF902652985.1')
        alg_rbh_dir: Directory containing species vs ALG RBH files
        alg_rbh_file: Path to ALG database RBH file (e.g., BCnSSimakov2022.rbh)
        algname: Name of ALG database (e.g., 'BCnSSimakov2022' or 'BCnS')
    
    Returns:
        dict: chromosome_name -> (dominant_ALG, ALG_color)
        dict: ALG_name -> ALG_color (for all ALGs)
    """
    # Parse ALG database to get colors
    alg_df = rbh_tools.parse_ALG_rbh_to_colordf(alg_rbh_file)
    alg_to_color = dict(zip(alg_df['ALGname'], alg_df['Color']))
    
    # Find species RBH file
    # Try multiple patterns
    patterns = [
        os.path.join(alg_rbh_dir, f"{algname}Simakov2022_{species}_xy_reciprocal_best_hits.plotted.rbh"),
        os.path.join(alg_rbh_dir, f"BCnSSimakov2022_{species}_xy_reciprocal_best_hits.plotted.rbh"),
        os.path.join(alg_rbh_dir, f"{species}_vs_{algname}.rbh")
    ]
    
    rbh_file = None
    for pattern in patterns:
        if os.path.exists(pattern):
            rbh_file = pattern
            break
    
    if not rbh_file:
        print(f"Warning: No ALG RBH file found for {species}, using default colors", file=sys.stderr)
        return {}, alg_to_color
    
    # Parse RBH file to get ALG composition per chromosome
    # File has header with columns including 'gene_group' (ALG name) and '{species}_scaf' (chromosome)
    chrom_alg_counts = {}  # {chrom: {alg: count}}
    
    try:
        # Read RBH file with header
        rbh_df = pd.read_csv(rbh_file, sep='\t')
        
        # Get the column name for species scaffold (chromosome)
        species_scaf_col = f"{species}_scaf"
        
        # Verify required columns exist
        if 'gene_group' not in rbh_df.columns:
            print(f"Warning: 'gene_group' column not found in {rbh_file}", file=sys.stderr)
            return {}, alg_to_color
        if species_scaf_col not in rbh_df.columns:
            print(f"Warning: '{species_scaf_col}' column not found in {rbh_file}", file=sys.stderr)
            return {}, alg_to_color
        
        # Count genes per ALG per chromosome
        for _, row in rbh_df.iterrows():
            chrom = row[species_scaf_col]  # e.g., "NC_047017.1"
            alg = row['gene_group']         # e.g., "A1b"
            
            if pd.isna(chrom) or pd.isna(alg):
                continue
                
            if chrom not in chrom_alg_counts:
                chrom_alg_counts[chrom] = {}
            if alg not in chrom_alg_counts[chrom]:
                chrom_alg_counts[chrom][alg] = 0
            chrom_alg_counts[chrom][alg] += 1
            
    except Exception as e:
        print(f"Warning: Error parsing {rbh_file}: {e}", file=sys.stderr)
        return {}, alg_to_color
    
    # Determine dominant ALG per chromosome
    chrom_to_dominant_alg = {}
    for chrom, alg_counts in chrom_alg_counts.items():
        if not alg_counts:
            continue
        # Find ALG with most genes
        dominant_alg = max(alg_counts.items(), key=lambda x: x[1])[0]
        color = alg_to_color.get(dominant_alg, '#1f77b4')  # Default blue if not found
        chrom_to_dominant_alg[chrom] = (dominant_alg, color)
    
    print(f"  Mapped {len(chrom_to_dominant_alg)} chromosomes to dominant ALGs for {species}", file=sys.stderr)
    return chrom_to_dominant_alg, alg_to_color

def _sa_scatter_panel(ax, animals_df, holos_df, *,
                      point_size=2.0,
                      base_alpha=0.50,
                      alpha_min=0.10,
                      alpha_max=0.55,
                      density_bin=40.0,
                      density_reference="min",
                      x_jitter=20.0,
                      rng_seed=0):
    """Render one Science Advances–style scatter panel.

    Animal points get a density-scaled per-point alpha (so dense
    TimeTree clusters don't dominate) and ±x_jitter Mya jitter to
    reveal density within a single TimeTree split. Holozoan rows are
    drawn as black-outlined diamonds at exact divergence time with
    species-specific colors so a reader can match values back to the
    caption.

    Required columns: ``divergence_time``, ``fraction_conserved``;
    ``holos_df`` also needs ``species``.
    """
    if len(animals_df) > 0:
        x_raw = animals_df["divergence_time"].to_numpy(dtype=float)
        if x_jitter > 0:
            rng = np.random.default_rng(rng_seed)
            x = x_raw + rng.uniform(-x_jitter, x_jitter, size=x_raw.size)
        else:
            x = x_raw
        alphas = density_alpha(x_raw, bin_width=density_bin,
                               base_alpha=base_alpha,
                               alpha_min=alpha_min, alpha_max=alpha_max,
                               reference=density_reference)
        rgba = rgba_array(ANIMAL_COLOR, alphas)
        ax.scatter(x, animals_df["fraction_conserved"], s=point_size,
                   c=rgba, alpha=None, linewidths=0, rasterized=True)

    for sp, sub in holos_df.groupby("species"):
        color = HOLOZOAN_COLORS.get(sp, "#000000")
        label = HOLOZOAN_LABELS.get(sp, sp)
        ax.scatter(sub["divergence_time"], sub["fraction_conserved"],
                   marker="D", s=7, c=color,
                   edgecolor="black", linewidths=0.3, zorder=6,
                   label=label)

    style_axes(ax)


def _emit_cd_column_width(sp1, wg_records, pc_records, outdir, *,
                          width_mm, holozoan_species):
    """Emit a column-width-fitting C+D side-by-side PDF.

    The total figure width is exactly ``width_mm`` millimeters. Common
    journal-column widths: 90 mm (single column), 180 mm (double
    column). Axes width is solved from the target figure width and
    fixed inch-anchored margins; axes height stays at 0.7786 in so the
    data-box aspect ratio matches the original 90 mm preset (0.9621 ×
    0.7786 in axes at width_mm=90). Helvetica, editable text
    (`pdf.fonttype=42`), 0.4 pt axes/spines.
    """
    MM_PER_IN = 25.4
    fig_w = width_mm / MM_PER_IN
    AX_H = 0.7786
    MARGIN_L, MARGIN_R = 0.42, 0.15
    MARGIN_B, MARGIN_T = 0.65, 0.18
    WSPACE_IN = 0.10
    AX_W = (fig_w - MARGIN_L - MARGIN_R - WSPACE_IN) / 2.0
    if AX_W <= 0:
        raise ValueError(
            f"column_width_mm={width_mm} is too small: leaves no room "
            f"for axes after {MARGIN_L + MARGIN_R + WSPACE_IN:.2f} in of "
            f"margins + inter-panel spacing."
        )
    fig_h = MARGIN_B + AX_H + MARGIN_T

    fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_h), sharey=True,
                             gridspec_kw=dict(wspace=WSPACE_IN / AX_W))
    ax_c, ax_d = axes
    fig.subplots_adjust(
        left=MARGIN_L / fig_w,
        right=1 - MARGIN_R / fig_w,
        bottom=MARGIN_B / fig_h,
        top=1 - MARGIN_T / fig_h,
    )

    holozoan_set = set(holozoan_species)
    wg_df = pd.DataFrame(wg_records,
                         columns=["species", "divergence_time", "fraction_conserved"])
    pc_df = pd.DataFrame(pc_records,
                         columns=["species", "divergence_time", "fraction_conserved"])
    wg_animals = wg_df[~wg_df["species"].isin(holozoan_set)]
    wg_holos   = wg_df[ wg_df["species"].isin(holozoan_set)]
    pc_animals = pc_df[~pc_df["species"].isin(holozoan_set)]
    pc_holos   = pc_df[ pc_df["species"].isin(holozoan_set)]

    _sa_scatter_panel(ax_c, wg_animals, wg_holos,
                      point_size=2.0, base_alpha=0.50,
                      alpha_min=0.10, alpha_max=0.55,
                      density_reference="min", x_jitter=20.0, rng_seed=1)
    _sa_scatter_panel(ax_d, pc_animals, pc_holos,
                      point_size=0.9, base_alpha=0.40,
                      alpha_min=0.03, alpha_max=0.55,
                      density_reference="min", x_jitter=20.0, rng_seed=2)

    xmax_candidates = []
    if len(wg_df) > 0:
        xmax_candidates.append(wg_df["divergence_time"].max())
    if len(pc_df) > 0:
        xmax_candidates.append(pc_df["divergence_time"].max())
    xmax = (max(xmax_candidates) + 30) if xmax_candidates else 1100
    for ax in (ax_c, ax_d):
        ax.set_xlim(0, xmax)
        ax.set_ylim(-0.02, 1.02)
        ax.tick_params(axis="both", which="both", labelsize=6)

    ax_c.set_ylabel("Fraction conserved", fontsize=7)
    ax_d.set_ylabel("")

    holo_handles = [
        Line2D([0], [0], marker="D", linestyle="none",
               markerfacecolor=HOLOZOAN_COLORS[sp],
               markeredgecolor="black", markeredgewidth=0.35,
               markersize=3, label=HOLOZOAN_LABELS[sp])
        for sp in HOLOZOAN_COLORS
        if sp in holozoan_set
    ]
    if holo_handles:
        fig.legend(handles=holo_handles, loc="lower center",
                   bbox_to_anchor=(0.5, 0.02), fontsize=6, ncol=3,
                   handlelength=0.6, borderpad=0.1, handletextpad=0.25,
                   columnspacing=0.8, labelspacing=0.18)

    for ax, letter in zip([ax_c, ax_d], "CD"):
        ax.text(0.0, 1.02, letter, transform=ax.transAxes,
                fontsize=9, fontweight="bold", va="bottom", ha="left",
                family="sans-serif")

    fig.text(0.5, (MARGIN_B - 0.20) / fig_h,
             f"Divergence time (Mya) from {sp1}",
             ha="center", va="center", fontsize=7)

    os.makedirs(outdir, exist_ok=True)
    outfile = os.path.join(outdir, f"panels_CD_{int(width_mm)}mm.pdf")
    fig.savefig(outfile, format="pdf", dpi=600)
    plt.close(fig)
    print(f"Saved {width_mm} mm column-width figure to {outfile}", file=sys.stderr)


def plot_pairwise_decay_sp1_vs_all(sp1, filestruct, outdir="./", bin_size=50,
                                    alg_rbh_file=None, alg_rbh_dir=None,
                                    algname="BCnS", column_width_mm=None,
                                    holozoan_species=None):
    """
    Plot pairwise chromosome decay of sp1 versus every other species in
    ``filestruct[sp1]``, in the Science Advances scatter style.

    Top-left  (axes[0][0]) — whole-genome conservation vs divergence time.
    Top-right (axes[0][1]) — per-chromosome conservation vs divergence time.
    Bottom-left  (axes[1][0]) — whole-genome violins per time bin.
    Bottom-right (axes[1][1]) — per-chromosome violins per time bin.

    Scatter panels use blue Okabe-Ito animal points with a density-scaled
    per-point alpha (sparse-bin reference, so dense TimeTree clusters
    don't oversaturate) and ±20 Mya x-jitter, plus diamond markers for
    holozoan callouts drawn at exact divergence times.

    Parameters
    ----------
    sp1, filestruct :
        Focal species + nested dict of per-pair TSV paths.
    outdir : str
        Output directory (created if missing).
    bin_size : int
        Violin-plot time bin width (Mya).
    alg_rbh_file, alg_rbh_dir, algname :
        Retained for backwards-compat. The scatter no longer colors
        chromosomes by dominant ALG (one blue + holozoan callouts), but
        callers still pass these for other code paths.
    column_width_mm : int | None
        If set, additionally emit ``panels_CD_<N>mm.pdf`` — a column-
        width-fitting side-by-side figure with total figure width
        exactly ``column_width_mm`` millimeters. Axes width is solved
        from N and fixed margins; axes height stays 0.7786 in. Common
        values: 90 (single-column journal), 180 (double-column).
    holozoan_species : iterable[str] | None
        Species IDs to draw as holozoan callouts. Defaults to
        :data:`egt.dispersal_panel_style.HOLOZOAN_SPECIES`.
    """
    apply_rc()
    holozoan_set = set(holozoan_species) if holozoan_species is not None else set(HOLOZOAN_SPECIES)

    BIN_SIZE = bin_size
    sp_bins  = {x: [] for x in range(BIN_SIZE, 1500, BIN_SIZE)}
    chr_bins = {x: [] for x in range(BIN_SIZE, 1500, BIN_SIZE)}
    sp1_sort_order = []
    skipped_species = []

    # Records collected during the loop, used by the SA scatter renderer.
    wg_records = []   # (species, divergence_time, fraction_conserved)
    pc_records = []   # one per sp2 × chromosome

    NUMBER_OF_ROWS = 2
    NUMBER_OF_COLS = 2
    fig, axes = plt.subplots(NUMBER_OF_ROWS, NUMBER_OF_COLS,
                             figsize=(7.5 * NUMBER_OF_COLS, 6 * NUMBER_OF_ROWS))

    total_pairs = len(filestruct[sp1].keys())
    processed_count = 0
    print(f"\nPlotting {total_pairs} species pairs...", file=sys.stderr)

    for sp2 in filestruct[sp1].keys():
        processed_count += 1
        print(f"\r  Progress: {processed_count}/{total_pairs} species pairs "
              f"({100*processed_count/total_pairs:.1f}%)          ",
              end='', file=sys.stderr)

        sp1_sp2_decay = pd.read_csv(filestruct[sp1][sp2], sep="\t")
        if len(sp1_sp2_decay) == 0:
            skipped_species.append(sp2)
            continue

        if sp1_sort_order == []:
            sp1_sort_order = sp1_sp2_decay.sort_values(
                by="sp1_scaf_genecount", ascending=True
            )["sp1_scaf"].tolist()

        sp1_sp2_decay = sp1_sp2_decay.set_index("sp1_scaf").reindex(sp1_sort_order).reset_index()

        divergence_time_mode = sp1_sp2_decay["divergence_time"].mode()
        if len(divergence_time_mode) == 0:
            skipped_species.append(sp2)
            continue
        divergence_time = divergence_time_mode[0]
        divergence_time_bin = int(BIN_SIZE * ceil(divergence_time / BIN_SIZE))

        total_genes = sp1_sp2_decay["sp1_scaf_genecount"].sum()
        total_conserved = sp1_sp2_decay["conserved"].sum()
        whole_fraction = total_conserved / total_genes if total_genes > 0 else 0.0

        wg_records.append((sp2, float(divergence_time), float(whole_fraction)))
        for _, row in sp1_sp2_decay.iterrows():
            pc_records.append((sp2, float(row["divergence_time"]),
                               float(row["fraction_conserved"])))

        sp_bins[divergence_time_bin].append(whole_fraction)
        chr_bins[divergence_time_bin].extend(sp1_sp2_decay["fraction_conserved"].tolist())

    print("", file=sys.stderr)

    # Partition records by holozoan membership for the SA scatter.
    wg_df = pd.DataFrame(wg_records,
                         columns=["species", "divergence_time", "fraction_conserved"])
    pc_df = pd.DataFrame(pc_records,
                         columns=["species", "divergence_time", "fraction_conserved"])

    wg_animals = wg_df[~wg_df["species"].isin(holozoan_set)] if len(wg_df) else wg_df
    wg_holos   = wg_df[ wg_df["species"].isin(holozoan_set)] if len(wg_df) else wg_df
    pc_animals = pc_df[~pc_df["species"].isin(holozoan_set)] if len(pc_df) else pc_df
    pc_holos   = pc_df[ pc_df["species"].isin(holozoan_set)] if len(pc_df) else pc_df

    if len(wg_df) > 0:
        _sa_scatter_panel(axes[0][0], wg_animals, wg_holos,
                          point_size=12.0, base_alpha=0.50,
                          alpha_min=0.10, alpha_max=0.55,
                          density_reference="min",
                          x_jitter=20.0, rng_seed=1)
    else:
        axes[0][0].text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=axes[0][0].transAxes)
    axes[0][0].set_xlabel(f"Divergence time (Mya) from {sp1}")
    axes[0][0].set_ylabel("Fraction conserved (whole genome)")

    if len(pc_df) > 0:
        _sa_scatter_panel(axes[0][1], pc_animals, pc_holos,
                          point_size=4.0, base_alpha=0.40,
                          alpha_min=0.03, alpha_max=0.55,
                          density_reference="min",
                          x_jitter=20.0, rng_seed=2)
    else:
        axes[0][1].text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=axes[0][1].transAxes)
    axes[0][1].set_xlabel(f"Divergence time (Mya) from {sp1}")
    axes[0][1].set_ylabel("Fraction conserved (per chromosome)")

    # Drop empty violin bins; recentre to bin midpoint.
    sp_bins  = {x - (BIN_SIZE / 2): sp_bins[x]  for x in sp_bins  if len(sp_bins[x])  > 0}
    chr_bins = {x - (BIN_SIZE / 2): chr_bins[x] for x in chr_bins if len(chr_bins[x]) > 0}

    if len(sp_bins) > 0 and all(len(v) > 0 for v in sp_bins.values()):
        axes[1][0].violinplot(sp_bins.values(), sp_bins.keys(),
                              points=10, widths=20,
                              showmeans=False, showextrema=True, showmedians=True)
    else:
        axes[1][0].text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=axes[1][0].transAxes)
    axes[1][0].set_xlabel(f"Divergence time (Mya) from {sp1}")
    axes[1][0].set_ylabel("Fraction conserved (whole genome)")

    if len(chr_bins) > 0 and all(len(v) > 0 for v in chr_bins.values()):
        axes[1][1].violinplot(chr_bins.values(), chr_bins.keys(),
                              points=5, widths=20,
                              showmeans=False, showextrema=True, showmedians=True)
    else:
        axes[1][1].text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=axes[1][1].transAxes)
    axes[1][1].set_xlabel(f"Divergence time (Mya) from {sp1}")
    axes[1][1].set_ylabel("Fraction conserved (per chromosome)")

    for ax in axes.flat:
        ax.set_ylim(-0.05, 1.05)
        style_axes(ax)

    # Apply consistent x-limits across all 4 panels.
    xlims_candidates = []
    for ax in axes.flat:
        lo, hi = ax.get_xlim()
        if np.isfinite(lo) and np.isfinite(hi):
            xlims_candidates.append((lo, hi))
    if xlims_candidates:
        lo = min(c[0] for c in xlims_candidates)
        hi = max(c[1] for c in xlims_candidates)
        for ax in axes.flat:
            ax.set_xlim(lo, hi)

    # Legend on top-right panel: animal + holozoan callouts.
    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="none",
               markerfacecolor=ANIMAL_COLOR, markeredgecolor="none",
               markersize=4, label="Animal target"),
    ]
    legend_handles += [
        Line2D([0], [0], marker="D", linestyle="none",
               markerfacecolor=HOLOZOAN_COLORS[sp],
               markeredgecolor="black", markeredgewidth=0.4,
               markersize=4, label=HOLOZOAN_LABELS[sp])
        for sp in HOLOZOAN_COLORS
        if sp in holozoan_set
    ]
    axes[0][1].legend(handles=legend_handles, loc="upper right",
                      fontsize=6, handlelength=1.2, borderpad=0.3,
                      handletextpad=0.4, labelspacing=0.25)

    total_species = len(filestruct[sp1])
    plotted_species = total_species - len(skipped_species)
    print(f"Plotting summary for {sp1}:", file=sys.stderr)
    print(f"  Total species pairs: {total_species}", file=sys.stderr)
    print(f"  Successfully plotted: {plotted_species}", file=sys.stderr)
    print(f"  Skipped (no data after filtering): {len(skipped_species)}", file=sys.stderr)

    fig.tight_layout()

    os.makedirs(outdir, exist_ok=True)
    outprefix = "{}_decay_plot_vs_divergence_time".format(sp1)
    outdir_prefix = os.path.join(outdir, outprefix)
    fig.savefig("{}.pdf".format(outdir_prefix), format="pdf", dpi=600)
    plt.close(fig)
    print(f"Saved decay overview to {outdir_prefix}.pdf", file=sys.stderr)

    if column_width_mm is not None:
        _emit_cd_column_width(sp1, wg_records, pc_records, outdir,
                              width_mm=column_width_mm,
                              holozoan_species=holozoan_set)


def plot_decay_twospecies(sp1, sp2, path_to_tsv, scaffolds_to_keep_sp1, outdir):
    """
    This plots the decay of an ALG between number of genes in the main chromosome,
    and the number of genes in smaller chromosomes

    Parameters:
        sp1: One species that is being plotted. This text title should match the text in the tsv file.
        sp2: Another species that is being plotted. This text should match the text of the tsv file.
        path_to_tsv: The path to the tsv file that contains the data to plot. The structure is described below.
        scaffolds_to_keep_sp1: A list of scaffolds to keep for sp1. This is necessary because we don't want to plot the small scaffolds.
        outdir: The directory to which we will save the plot.

    The input is the tsv output by calculate_pairwise_decay_sp1_vs_many:
    For example, here is one df with PMA (scallop) as sp1 and PFI (sponge) as sp2:

        sp1_scaf        sp2_scaf  sp1_scaf_genecount  conserved  scattered  divergence_time  fraction_conserved
     0      PMA1    [PFI8, PFI1]                 552        322        230              800            0.583333
     1     PMA10   [PFI13, PFI1]                 310        182        128              800            0.587097
     2     PMA11          [PFI7]                 376        262        114              800            0.696809

    The output is one figure with two subplots.
    The left subplot is the ranked sizes of the chromosomes in sp1. The right subplot is the actual size of the chromosomes in sp1
    """
    BARS = True
    df = pd.read_csv(path_to_tsv, sep="\t")
    # only keep the scaffolds for species 1 that we know are valid
    df = df[df["sp1_scaf"].isin(scaffolds_to_keep_sp1)]

    # rank the chromosomes based on their size and sort by the rank
    df["sp1_ranked"] = df["sp1_scaf_genecount"].rank(ascending=True, method="first")
    df = df.sort_values(by="sp1_ranked")

    # set up the two panels of the plot
    NUMBER_OF_ROWS = 2
    NUMBER_OF_COLS = 2
    fig, axes = plt.subplots(NUMBER_OF_ROWS, NUMBER_OF_COLS,
                             figsize = (7.5 * NUMBER_OF_COLS, 7*(NUMBER_OF_ROWS - 1)),
                             gridspec_kw={'height_ratios': [1, 2.5]})

    fig.suptitle("{} and {} chromosome conservation vs {} chromosome size".format(sp1, sp2, sp1))

    if BARS:
        # on the left plot the total gene count as red bars
        axes[1][0].bar(df["sp1_ranked"], df["sp1_scaf_genecount"], color = "red", alpha = 1)
    else:
        # plot the chromosome sizes by rank on the left
        axes[1][0].plot(df["sp1_ranked"], df["sp1_scaf_genecount"], "bo")

    # plot the chromosomes by actual size on the right
    axes[1][1].plot(df["sp1_scaf_genecount"], df["sp1_scaf_genecount"], "bo")

    # add some horizontal space between axes[0] and axes[1]
    fig.subplots_adjust(wspace=0.5)

    # make vertical lines on the left and the right plot. Do it by iterating through the dataframe
    for index, row in df.iterrows():
        if BARS:
            pass
        else:
            axes[1][0].plot([row["sp1_ranked"], row["sp1_ranked"]], [0, row["sp1_scaf_genecount"]], "k-", alpha = 0.33)
        axes[1][1].plot([row["sp1_scaf_genecount"], row["sp1_scaf_genecount"]], [0, row["sp1_scaf_genecount"]], "k-", alpha = 0.33)

    # now we plot the blue points
    if BARS:
        # now we plot blue bars for the number of genes conserved
        axes[1][0].bar(df["sp1_ranked"], df["conserved"], color = "blue", alpha = 1)
    else:
        # now we plot blue points for number of genes degraded
        axes[1][0].plot(df["sp1_ranked"], df["scattered"], "ro")
    axes[1][1].plot(df["sp1_scaf_genecount"], df["scattered"], "ro")

    # add some yaxis labels. make the color red to match the dots. Then make the tick labels red too
    color = "blue"
    axes[1][0].set_ylabel("Number of orthologs on chromosome", color=color)
    axes[1][1].set_ylabel("Number of orthologs on chromosome", color=color)
    axes[1][0].tick_params(axis='y', labelcolor=color)
    axes[1][1].tick_params(axis='y', labelcolor=color)

    # add some xaxis labels
    axes[1][0].set_xlabel("Chromosome ranked by ortholog count")
    axes[1][1].set_xlabel("Number of orthologs on chromosome")

    # on the left side we will add x-axis ticks that at the sp1 chromosome names, and rotate everything 45 degrees
    axes[1][0].set_xticks(df["sp1_ranked"])
    axes[1][0].set_xticklabels(df["sp1_scaf"], rotation=90, ha="center")

    # get the y-axis limits for the left plot
    left_ylim = axes[1][0].get_ylim()
    left_xlim = axes[1][0].get_xlim()

    # get the max value of the sp1_scaf_genecount
    left_maxgene = df["sp1_scaf_genecount"].max()
    # print out the ratios of the total limits to the max gene count
    ylim_scale_factor = abs(1 - (left_ylim[1]/left_maxgene))

    # set the y-limits of the left plot to the right plot
    axes[1][0].set_ylim(axes[1][1].get_ylim())

    # now we clone the axes and plot the percent conserved on the y-axes
    axL = axes[0][0]
    axR = axes[0][1]
    ylim_scale_number = 100
    axL.set_ylim( [-5, 100] )
    axL.set_xlim( left_xlim )
    axR.set_ylim( [-5, 100] )

    color = 'black'
    axL.set_ylabel('percent conserved on ALGs', color=color)  # we already handled the x-label with ax1
    axR.set_ylabel('percent conserved on ALGs', color=color)
    # set color of axL and axR yaxis ticks to blue
    axL.tick_params(axis='y', labelcolor=color)
    axR.tick_params(axis='y', labelcolor=color)
    # change the x-axis ticks to be the same positions as the bottom left
    # turn off the tick labels for the top left
    axL.set_xticks(df["sp1_ranked"])
    axL.set_xticklabels(["" for x in df["sp1_ranked"]])
    # now plot the data
    axL.plot(df["sp1_ranked"], 100*(df["conserved"]/df["sp1_scaf_genecount"]), color = color, lw = 1)
    axR.plot(df["sp1_scaf_genecount"], 100*(df["conserved"]/df["sp1_scaf_genecount"]), color = color, lw = 1)

    # turn off the top and right bar of the frames of each axes
    for ax in axes.flatten():
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # CALL THIS TO GET THE VISUAL STYLE WE NEED
    odp_plot.format_matplotlib()

    # adjust the bounding to fit text that went outside the limit of the plot
    plt.tight_layout()
    # safe make the directory
    os.makedirs(outdir, exist_ok=True)
    outprefix = "{}_and_{}_chromosome_conservation".format(sp1, sp2)
    outdir_prefix = os.path.join(outdir, outprefix)
    plt.savefig("{}.pdf".format(outdir_prefix), format='pdf')
    plt.close()

def main(argv=None):
    # parse the arguments
    args = parse_args(argv)

    # Load config first if provided (needed for taxid-to-species-name mapping)
    initial_config = None
    if args.config:
        initial_config = read_yaml_file(args.config)
        print(f"Loaded config from {args.config}", file=sys.stderr)
    elif args.divergence_file and not args.config:
        raise ValueError("When using --divergence_file, you must also provide --config "
                        "to map taxids to species names.")

    # Load divergence times from file if provided
    divergence_times_dict = None
    if args.divergence_file:
        divergence_times_dict = load_divergence_times_from_file(args.divergence_file, config=initial_config)

    # we must parse the config file to get the analysis parameters
    config = parse_config(args.config, args.directory, args.target_species, 
                         divergence_times_dict=divergence_times_dict)

    # build a sp_to_chr_to_size nested dictionary. We wouldn't need this if we finished clink.
    # nested for loop to get all the files for all the species
    rbh_filelist = set()
    for sp1 in config["analysis_files"].keys():
        for sp2 in config["analysis_files"][sp1].keys():
            rbh_filelist.add(config["analysis_files"][sp1][sp2])
    rbh_filelist = list(rbh_filelist)
    
    # Use cached chromosome sizes if available
    sp_to_chr_to_size, sp_to_scaf_to_genecount = get_chromosome_sizes_cached(
        rbh_filelist, cache_dir=args.cache_dir)
    ## print the target genecounts
    #print(sp_to_scaf_to_genecount[args.target_species])

    # For the target species, get a list of scaf names if they have more
    #  than 1% of the total amount of genes in the geneome.
    #  This is to avoid plotting the small scaffolds.
    target_keep_these_scafs_gt_one_percent_genes = {x:set() for x in config["target_species"]}
    for sp in config["target_species"]:
        total_genes = sum([sp_to_scaf_to_genecount[sp][scaf] for scaf in sp_to_scaf_to_genecount[sp]])
        for scaf in sp_to_scaf_to_genecount[sp]:
            if sp_to_scaf_to_genecount[sp][scaf] >= (0.01 * total_genes):
                target_keep_these_scafs_gt_one_percent_genes[sp].add(scaf)
        
        # Print diagnostic info
        print(f"\nScaffold filtering for {sp}:", file=sys.stderr)
        print(f"  Total genes: {total_genes}", file=sys.stderr)
        print(f"  Total scaffolds: {len(sp_to_scaf_to_genecount[sp])}", file=sys.stderr)
        print(f"  Scaffolds with ≥1% of genes: {len(target_keep_these_scafs_gt_one_percent_genes[sp])}", file=sys.stderr)
        print(f"  Kept scaffolds: {sorted(target_keep_these_scafs_gt_one_percent_genes[sp])}", file=sys.stderr)

    # safely make the directory called 'odp_pairwise_decay'
    os.makedirs("odp_pairwise_decay", exist_ok=True)
    # make a plot using the data
    for sp1 in config["target_species"]:
        outdir = os.path.join("odp_pairwise_decay", sp1)
        outdir = os.path.join(outdir, "decay_dataframes")
        # calculate the pairwise decay in chromosomes, save the files, get the list of files
        filestruct = calculate_pairwise_decay_sp1_vs_many(
            sp1, config, sp_to_chr_to_size,
            target_keep_these_scafs_gt_one_percent_genes, outdir,
            min_scaf_len=args.min_scaf_size,
            fet_threshold=args.fet_threshold)

        # make the summary plot of all the chromosomes
        outdir = os.path.join("odp_pairwise_decay", sp1)
        outdir = os.path.join(outdir, "plot_overview_sp_sp")
        plot_pairwise_decay_sp1_vs_all(sp1, filestruct, outdir=outdir, bin_size=args.bin_size,
                                       alg_rbh_file=args.ALG_rbh, alg_rbh_dir=args.ALG_rbh_dir,
                                       algname=args.ALGname,
                                       column_width_mm=args.column_width_mm)
        
        # make the dispersal plots
        outdir = os.path.join("odp_pairwise_decay", sp1)
        outdir = os.path.join(outdir, "plot_ALG_dispersal")
        plot_dispersal_by_ALG(sp1, filestruct, args.ALG_rbh, args.ALGname, args.ALG_rbh_dir, outdir=outdir)
        #
        ### make individual sp-sp scatterplots
        #outdir = os.path.join("odp_pairwise_decay", sp1)
        #outdir = os.path.join(outdir, "plot_individual_sp_sp")
        #for sp2 in sorted(filestruct[sp1].keys()):
        #    plot_decay_twospecies(sp1, sp2, filestruct[sp1][sp2],
        #                          target_keep_these_scafs_gt_one_percent_genes[sp1], outdir)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())