"""Loaders for GO enrichment inputs.

Everything here is pure parsing — no statistics. Parsers accept `.gz` or
plain text transparently. Unit-tested against small fixture files so the
exact column layout they expect is documented by the tests themselves.
"""
from __future__ import annotations

import gzip
import re
from collections import defaultdict
from pathlib import Path
from typing import Mapping

import pandas as pd


# NCBI `gene2go` uses these one-word category names; our internal code
# (and the rest of the ecosystem) uses the two-letter codes.
NCBI_CATEGORY_MAP: dict[str, str] = {
    "Process": "BP",
    "Function": "MF",
    "Component": "CC",
    "biological_process": "BP",
    "molecular_function": "MF",
    "cellular_component": "CC",
}

REFSEQ_PROTEIN_RE = re.compile(r"^(NP|XP|YP)_[0-9]+\.[0-9]+$")
UNIPROT_SP_RE = re.compile(r"^sp\|([^|]+)\|([^|]+)$")


def _opener(path):
    return gzip.open if str(path).endswith(".gz") else open


def parse_gene2accession(path) -> tuple[dict[str, str], dict[str, str]]:
    """Parse NCBI `gene2accession` into (prot_to_gene, gene_to_symbol).

    Columns expected (tab-separated):
      0 tax_id, 1 GeneID, 2 status, 3 RNA_nucleotide_accession.version,
      4 RNA_nucleotide_gi, 5 protein_accession.version, 6 protein_gi,
      7..14 …, 15 Symbol

    Skips rows where `protein_accession.version` or `GeneID` is empty/"-".
    The returned `gene_to_symbol` keeps the first Symbol seen per GeneID
    (NCBI publishes the canonical symbol in the first row it emits).
    """
    prot_to_gene: dict[str, str] = {}
    gene_to_symbol: dict[str, str] = {}
    with _opener(path)(path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 7:
                continue
            prot, gene = f[5], f[1]
            if prot and prot != "-" and gene and gene != "-":
                prot_to_gene[prot] = gene
            if len(f) >= 16 and gene and gene != "-":
                sym = f[15].strip()
                if sym and sym != "-" and gene not in gene_to_symbol:
                    gene_to_symbol[gene] = sym
    return prot_to_gene, gene_to_symbol


def parse_gene2go(path) -> tuple[dict[str, set[str]], dict[str, str]]:
    """Parse NCBI `gene2go` into (gene_to_terms, term_namespace).

    Columns expected (tab-separated):
      0 tax_id, 1 GeneID, 2 GO_ID, 3 Evidence, 4 Qualifier,
      5 GO_term, 6 Pubmed, 7 Category

    Rows with `NOT` in the Qualifier are dropped. `term_namespace` records
    the first Category seen per GO_ID (NCBI is consistent).
    """
    gene_to_terms: dict[str, set[str]] = defaultdict(set)
    term_ns: dict[str, str] = {}
    with _opener(path)(path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 8:
                continue
            gene, go_id, qualifier, category = f[1], f[2], f[4], f[7]
            if "NOT" in qualifier.upper().split("|"):
                continue
            gene_to_terms[gene].add(go_id)
            ns = NCBI_CATEGORY_MAP.get(category)
            if ns and go_id not in term_ns:
                term_ns[go_id] = ns
    return dict(gene_to_terms), term_ns


def parse_family_map(
    path,
    prot_to_gene: Mapping[str, str],
) -> tuple[dict[str, set[str]], dict[str, int]]:
    """Load `bcns_family_to_human_gene.tsv` and resolve each family to GeneIDs.

    Returns (fam_to_genes, stats) where `stats` is a small dict of join
    coverage counts (n_rows, n_empty, n_refseq_hit, n_refseq_miss,
    n_sp_uniprot, n_other, n_families_mapped). RefSeq NP_/XP_/YP_
    accessions are resolved via `prot_to_gene`; Swiss-Prot pass-2 entries
    are counted but not mapped (no known join into Entrez without a
    separate UniProt→GeneID table).
    """
    df = pd.read_csv(path, sep="\t")
    family_col = df.columns[0]
    gene_col = "human_gene" if "human_gene" in df.columns else df.columns[2]
    fam_to_genes: dict[str, set[str]] = defaultdict(set)
    n_rows = 0
    n_empty = 0
    n_refseq_hit = 0
    n_refseq_miss = 0
    n_sp = 0
    n_other = 0
    for _, row in df.iterrows():
        n_rows += 1
        fam = row[family_col]
        val = row[gene_col]
        if not isinstance(val, str) or not val.strip():
            n_empty += 1
            continue
        v = val.strip()
        if REFSEQ_PROTEIN_RE.match(v):
            g = prot_to_gene.get(v)
            if g:
                fam_to_genes[fam].add(g)
                n_refseq_hit += 1
            else:
                n_refseq_miss += 1
        elif UNIPROT_SP_RE.match(v):
            n_sp += 1
        else:
            n_other += 1
    stats = dict(
        n_rows=n_rows,
        n_empty=n_empty,
        n_refseq_hit=n_refseq_hit,
        n_refseq_miss=n_refseq_miss,
        n_sp_uniprot=n_sp,
        n_other=n_other,
        n_families_mapped=len(fam_to_genes),
    )
    return dict(fam_to_genes), stats


def load_obo_names(path) -> tuple[dict[str, str], dict[str, str]]:
    """Parse a `go-basic.obo` file into (id_to_name, id_to_namespace).

    Namespaces are returned in the internal BP/MF/CC codes. Missing file
    or None path returns empty dicts (caller decides whether that's an
    error).
    """
    names: dict[str, str] = {}
    ns: dict[str, str] = {}
    if not path or not Path(path).exists():
        return names, ns
    cur: dict[str, str] | None = None
    with open(path) as fh:
        for line in fh:
            line = line.rstrip()
            if line == "[Term]":
                cur = {}
                continue
            if line == "" and cur is not None:
                if "id" in cur:
                    if "name" in cur:
                        names[cur["id"]] = cur["name"]
                    if "namespace" in cur:
                        ns[cur["id"]] = NCBI_CATEGORY_MAP.get(
                            cur["namespace"], "?"
                        )
                cur = None
                continue
            if cur is None:
                continue
            if ":" in line:
                k, v = line.split(":", 1)
                k = k.strip()
                v = v.strip()
                if k in ("id", "name", "namespace"):
                    cur[k] = v
    return names, ns


# On-disk `significant_terms.tsv` carries the publication-standard
# canonical column set (k/n/K/N annotated with single-letter brackets so
# the hypergeometric parametrisation is unambiguous for human readers).
# Internal consumers in egt.go (plots, benchmarks) were written against
# the short forms; this alias map lets each loader re-expose the short
# names without losing the canonical on-disk schema.
SIGNIFICANT_TERMS_SHORT_ALIASES: dict[str, str] = {
    "foreground_hits_[k]": "k",
    "foreground_size_[n]": "n",
    "background_hits_[K]": "K",
    "background_size_[N]": "N",
    "ratio_in_study_[k/n]": "ratio_in_study",
    "ratio_in_pop_[K/N]": "ratio_in_pop",
    "fold_enrichment": "fold",
    "p_value": "p",
    "q_value": "q",
}


def load_significant_terms(path) -> pd.DataFrame:
    """Read significant_terms.tsv and alias the canonical column names
    to the short single-letter forms used by the egt.go plotters."""
    df = pd.read_csv(path, sep="\t")
    return df.rename(columns=SIGNIFICANT_TERMS_SHORT_ALIASES)


# Per-cell sweep summary (summary.tsv + per_clade/<clade>.tsv) carries
# a different set of hypergeometric names than significant_terms.tsv
# (one row per sweep cell, not per enriched term). Same bracket
# convention; this alias map lets downstream code keep using short
# identifiers without re-writing every accessor.
SWEEP_SUMMARY_SHORT_ALIASES: dict[str, str] = {
    "foreground_size_[n]": "foreground_size",
    "top_term_hits_[k]": "top_term_k",
    "top_term_bg_hits_[K]": "top_term_K",
    "top_term_fold_enrichment": "top_term_fold",
    "top_q_value": "top_q",
}


def load_sweep_summary(path) -> pd.DataFrame:
    """Read summary.tsv (or a per-clade sweep tsv) and alias canonical
    column names to the short forms used by the egt.go plotters and
    benchmarks."""
    df = pd.read_csv(path, sep="\t")
    return df.rename(columns=SWEEP_SUMMARY_SHORT_ALIASES)


def load_unique_pairs(path) -> pd.DataFrame:
    """Load a defining-pair table (.xlsx or TSV / TSV.gz) into a DataFrame.

    Accepts either the Dryad `SupplementaryTable_16.xlsx` (read via
    openpyxl) or a TSV emitted by `egt defining-features`. The returned
    DataFrame is checked for the minimal columns the sweep needs.
    """
    p = str(path).lower()
    if p.endswith(".xlsx") or p.endswith(".xls"):
        df = pd.read_excel(path, engine="openpyxl")
    else:
        df = pd.read_csv(path, sep="\t", compression="infer", low_memory=False)
    required = ("nodename", "ortholog1", "ortholog2", "occupancy_in",
                "sd_in_out_ratio_log_sigma", "mean_in_out_ratio_log_sigma")
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"unique_pairs table missing required columns: {missing}"
        )
    return df


def build_family_gene_annotations(
    fam_to_genes: Mapping[str, set[str]],
    gene_to_terms: Mapping[str, set[str]],
) -> tuple[set[str], dict[str, set[str]]]:
    """Restrict gene→term annotations to the BCnS-family universe.

    Returns (background_gene_ids, background_to_terms).

    - `background_gene_ids`: union of all GeneIDs across the family map.
    - `background_to_terms`: for the subset of those GeneIDs that have
      at least one GO annotation, the annotation set.

    This is the pair-enrichment "annotatable background", used both to
    restrict the foreground and to compute the hypergeometric N_total.
    """
    background = set().union(*fam_to_genes.values()) if fam_to_genes else set()
    background_to_terms: dict[str, set[str]] = {}
    for g in background:
        t = gene_to_terms.get(g)
        if t:
            background_to_terms[g] = t
    return background, background_to_terms
