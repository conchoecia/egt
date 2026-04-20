"""
Block-level benchmark: Simakov 2013 microsyntenic blocks preserved
from the bilaterian ancestor, vs our defining-pair catalog.

Reference: Simakov et al. 2013, Nature 493:526-531 (doi 10.1038/nature11696).
Supplementary Data MOESM7 (file `2012-01-00225C-microsynteny.xls`) lists
every species-specific microsyntenic block with a `Classification` column
whose `bilaterianAnc` tag marks blocks preserved from the bilaterian
ancestor.

Gene-ID pipeline. Simakov 2013's `All genes in region` field uses a
2012-era internal numeric namespace that does not round-trip to modern
NCBI Entrez GeneIDs (verified: only 1.7 % of 3,259 bilaterianAnc Hsa
IDs appear in current gene2accession; NCBI gene_history covers a small
fraction and most of those are retired without replacement).

This script therefore maps via **genomic coordinates** using GENCODE v19
(GRCh37, Dec 2013 — contemporaneous with the Simakov paper):

  Simakov block (chr:start-end, GRCh37)
    → GENCODE v19 gene records overlapping that interval (gene_name = HGNC symbol)
    → Entrez GeneID via NCBI gene2accession (Symbol column, full-human lookup)
    → BCnS family via our family map

Intra-block gene pairs are then enumerated at the BCnS-family level and
looked up against our 28-clade defining-pair union + the
`stable_in_clade` subset.
"""
import argparse
import gzip
import re
from collections import defaultdict
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd


REFSEQ_PROTEIN_RE = re.compile(r"^(NP|XP|YP)_[0-9]+\.[0-9]+$")


def parse_gene2accession(path):
    """Return (prot_to_gene, symbol_to_geneids).

    Full human table, no BCnS restriction.
    """
    opener = gzip.open if str(path).endswith(".gz") else open
    prot_to_gene = {}
    symbol_to_geneids = defaultdict(set)
    with opener(path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 16:
                continue
            gene = f[1]; prot = f[5]; sym = f[15].strip()
            if prot and prot != "-" and gene and gene != "-":
                prot_to_gene[prot] = gene
            if sym and sym != "-" and gene and gene != "-":
                symbol_to_geneids[sym].add(gene)
    return prot_to_gene, dict(symbol_to_geneids)


def parse_family_map(path, prot_to_gene):
    df = pd.read_csv(path, sep="\t")
    family_col = df.columns[0]
    gene_col = "human_gene" if "human_gene" in df.columns else df.columns[2]
    gene_to_family = {}
    for _, row in df.iterrows():
        fam = row[family_col]
        val = row[gene_col]
        if not isinstance(val, str) or not val.strip():
            continue
        v = val.strip()
        if REFSEQ_PROTEIN_RE.match(v):
            g = prot_to_gene.get(v)
            if g:
                gene_to_family[g] = fam
    return gene_to_family


def load_gencode_genes(path):
    """Return DataFrame of GENCODE gene entries: chrom (int str without 'chr'),
    start (0-indexed), end (exclusive), gene_name, gene_type."""
    opener = gzip.open if str(path).endswith(".gz") else open
    rows = []
    name_re = re.compile(r'gene_name "([^"]+)"')
    type_re = re.compile(r'gene_type "([^"]+)"')
    with opener(path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 9 or f[2] != "gene":
                continue
            chrom = f[0].replace("chr", "")
            start = int(f[3]) - 1  # GTF is 1-based inclusive → 0-based
            end = int(f[4])        # GTF end is inclusive → exclusive for half-open
            m_name = name_re.search(f[8])
            m_type = type_re.search(f[8])
            rows.append((chrom, start, end,
                         m_name.group(1) if m_name else "",
                         m_type.group(1) if m_type else ""))
    return pd.DataFrame(rows, columns=["chrom", "start", "end",
                                         "gene_name", "gene_type"])


def index_gencode_by_chrom(gencode_df):
    """Build {chrom: sorted list of (start, end, gene_name)}."""
    idx = defaultdict(list)
    for _, r in gencode_df.iterrows():
        idx[r["chrom"]].append((r["start"], r["end"], r["gene_name"]))
    for c in idx:
        idx[c].sort()
    return dict(idx)


def parse_coord(s):
    """Simakov's 'Chrom:begin-end' → (chrom, begin, end) or None."""
    if not isinstance(s, str):
        return None
    m = re.match(r"^([\w\.\-]+):(\d+)-(\d+)$", s.strip())
    if not m:
        return None
    return (m.group(1).replace("chr", ""), int(m.group(2)), int(m.group(3)))


def genes_overlapping(coord, chrom_idx):
    """Return list of gene_name values whose GENCODE interval overlaps
    the block coord (chrom, begin, end). Half-open interval arithmetic."""
    chrom, begin, end = coord
    if chrom not in chrom_idx:
        return []
    out = []
    for gs, ge, gname in chrom_idx[chrom]:
        if ge <= begin:
            continue
        if gs >= end:
            # sorted by start → safe to break once we're past the block
            break
        if not gname:
            continue
        out.append(gname)
    return out


def load_unique_pairs(path):
    df = pd.read_csv(path, sep="\t", compression="infer", low_memory=False)
    # Normalise stable_in_clade: CSV may load as string "True"/"False".
    def to_bool(v):
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.strip().lower() == "true"
        return bool(v)
    df["stable_bool"] = df["stable_in_clade"].apply(to_bool)
    per_clade = defaultdict(set)
    stable = set()
    for r in df.itertuples(index=False):
        f = frozenset((r.ortholog1, r.ortholog2))
        if len(f) != 2:
            continue
        per_clade[r.nodename].add(f)
        if r.stable_bool:
            stable.add(f)
    union = set().union(*per_clade.values()) if per_clade else set()
    return dict(per_clade), union, stable


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--microsynteny-xls", required=True)
    ap.add_argument("--classification", default="bilaterianAnc")
    ap.add_argument("--species", default="Hsa")
    ap.add_argument("--family-map", required=True)
    ap.add_argument("--gene2accession", required=True)
    ap.add_argument("--gencode-gtf", required=True,
                    help="GENCODE v19 GTF (GRCh37).")
    ap.add_argument("--unique-pairs", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args(argv)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("[load] Simakov microsynteny ...")
    ms = pd.read_excel(args.microsynteny_xls, engine="calamine")
    sub = ms[ms["Species ID"] == args.species].copy()
    sub["classif_str"] = sub["Classification"].fillna("").astype(str)
    tags = [t.strip() for t in args.classification.split(",") if t.strip()]
    sub = sub[sub["classif_str"].apply(
        lambda s: any(t in s for t in tags))]
    print(f"  {args.species} ∩ {args.classification}: {len(sub)} blocks")

    print("[load] gene2accession ...")
    prot_to_gene, symbol_to_geneids = parse_gene2accession(args.gene2accession)
    print(f"  protein→GeneID {len(prot_to_gene):,}  symbol→GeneID {len(symbol_to_geneids):,}")

    print("[load] family map ...")
    gene_to_family = parse_family_map(args.family_map, prot_to_gene)
    print(f"  GeneID→BCnS-family {len(gene_to_family):,}")

    print("[load] GENCODE v19 (GRCh37) gene entries ...")
    gencode = load_gencode_genes(args.gencode_gtf)
    print(f"  GTF gene entries: {len(gencode):,}")
    chrom_idx = index_gencode_by_chrom(gencode)
    print(f"  chromosomes indexed: {len(chrom_idx)}")

    print("[load] our defining pairs ...")
    per_clade, union_pairs, stable_pairs = load_unique_pairs(args.unique_pairs)
    print(f"  defining union: {len(union_pairs):,}  "
          f"stable_in_clade: {len(stable_pairs):,}")

    bcns_pair_universe_size = 2_785_980
    p_pick_union = len(union_pairs) / bcns_pair_universe_size
    p_pick_stable = len(stable_pairs) / bcns_pair_universe_size

    block_records = sub.to_dict("records")
    block_rows = []
    pair_rows = []

    n_blocks_coord_ok = 0
    n_blocks_some_gene = 0
    n_blocks_some_bcns = 0

    for rec in block_records:
        cid = rec["ClusID"]
        classif = rec.get("Classification", "")
        coord_str = rec.get("Chrom:begin-end", "")
        coord = parse_coord(coord_str)
        if coord is None:
            continue
        n_blocks_coord_ok += 1
        gene_names = genes_overlapping(coord, chrom_idx)
        gene_names = sorted(set(g for g in gene_names if g))
        if not gene_names:
            continue
        n_blocks_some_gene += 1
        # Symbols → GeneIDs (expand all matches) → BCnS families.
        fams_per_gene = {}
        for sym in gene_names:
            gids = symbol_to_geneids.get(sym, set())
            fams = {gene_to_family[g] for g in gids if g in gene_to_family}
            if fams:
                fams_per_gene[sym] = fams
        if not fams_per_gene:
            continue
        n_blocks_some_bcns += 1

        # Family set within this block (union across mapped symbols).
        unique_fams = set()
        for fs in fams_per_gene.values():
            unique_fams |= fs
        unique_fams = sorted(unique_fams)

        # Enumerate all intra-block family pairs.
        block_pair_set = set()
        for fa, fb in combinations(unique_fams, 2):
            block_pair_set.add(frozenset((fa, fb)))
        if not block_pair_set:
            continue

        n_pairs_hit_union = sum(1 for fp in block_pair_set if fp in union_pairs)
        n_pairs_hit_stable = sum(1 for fp in block_pair_set if fp in stable_pairs)

        block_rows.append(dict(
            ClusID=cid,
            classification=classif,
            coords=coord_str,
            n_genes_in_region_GENCODE=len(gene_names),
            n_genes_mapped_to_BCnS=len(fams_per_gene),
            n_families=len(unique_fams),
            n_intra_block_pairs=len(block_pair_set),
            n_hits_union=n_pairs_hit_union,
            n_hits_stable=n_pairs_hit_stable,
        ))
        for fp in block_pair_set:
            fa, fb = sorted(fp)
            pair_rows.append(dict(
                ClusID=cid, fam_A=fa, fam_B=fb,
                in_defining_union=fp in union_pairs,
                in_stable=fp in stable_pairs,
            ))

    bdf = pd.DataFrame(block_rows)
    pdf = pd.DataFrame(pair_rows)
    bdf.to_csv(out / "per_block.tsv", sep="\t", index=False)
    pdf.to_csv(out / "per_pair.tsv", sep="\t", index=False)

    n_blocks_total = len(sub)
    n_blocks_any_pair = int((bdf["n_intra_block_pairs"] >= 1).sum()) if len(bdf) else 0
    n_pairs_total = len(pdf)
    unique_pairs_in_blocks = ({frozenset((r["fam_A"], r["fam_B"]))
                                for r in pair_rows})
    n_pairs_unique = len(unique_pairs_in_blocks)
    uniq_hit_union = sum(1 for p in unique_pairs_in_blocks if p in union_pairs)
    uniq_hit_stable = sum(1 for p in unique_pairs_in_blocks if p in stable_pairs)

    expected_union = n_pairs_unique * p_pick_union
    expected_stable = n_pairs_unique * p_pick_stable
    enr_union = uniq_hit_union / expected_union if expected_union else float("nan")
    enr_stable = uniq_hit_stable / expected_stable if expected_stable else float("nan")

    print("\n" + "=" * 70)
    print(f"Simakov 2013 {args.classification} blocks ({args.species}): {n_blocks_total}")
    print(f"  coords parseable: {n_blocks_coord_ok}")
    print(f"  GENCODE intersect produced ≥1 gene: {n_blocks_some_gene}")
    print(f"  ≥1 gene mapping to a BCnS family: {n_blocks_some_bcns}")
    print(f"  ≥1 intra-block BCnS family pair: {n_blocks_any_pair}")
    print(f"  unique intra-block family pairs: {n_pairs_unique}")
    print(f"  hits vs defining-pair union: {uniq_hit_union} / {n_pairs_unique} "
          f"= {100.0*uniq_hit_union/max(n_pairs_unique,1):.1f}%  "
          f"(null {expected_union:.1f}, enrichment {enr_union:.1f}×)")
    print(f"  hits vs stable_in_clade subset: {uniq_hit_stable} / {n_pairs_unique} "
          f"= {100.0*uniq_hit_stable/max(n_pairs_unique,1):.1f}%  "
          f"(null {expected_stable:.1f}, enrichment {enr_stable:.1f}×)")
    print("=" * 70)

    summary_rows = [
        dict(metric="Simakov 2013 blocks analysed (species+classification filter)",
             value=n_blocks_total),
        dict(metric="Blocks with parseable coordinates", value=n_blocks_coord_ok),
        dict(metric="Blocks with ≥1 gene in GENCODE v19 interval",
             value=n_blocks_some_gene),
        dict(metric="Blocks with ≥1 BCnS-mappable gene",
             value=n_blocks_some_bcns),
        dict(metric="Blocks contributing ≥1 intra-block family pair",
             value=n_blocks_any_pair),
        dict(metric="Unique intra-block family pairs", value=n_pairs_unique),
        dict(metric="Hits in defining-pair union",
             value=f"{uniq_hit_union} ({100*uniq_hit_union/max(n_pairs_unique,1):.1f}%)"),
        dict(metric="Hits in stable_in_clade subset",
             value=f"{uniq_hit_stable} ({100*uniq_hit_stable/max(n_pairs_unique,1):.1f}%)"),
        dict(metric="Expected under uniform-random null (defining union)",
             value=f"{expected_union:.1f}"),
        dict(metric="Enrichment (defining union)", value=f"{enr_union:.1f}x"),
        dict(metric="Expected under uniform-random null (stable subset)",
             value=f"{expected_stable:.1f}"),
        dict(metric="Enrichment (stable subset)", value=f"{enr_stable:.1f}x"),
    ]
    pd.DataFrame(summary_rows).to_csv(out / "summary.tsv", sep="\t", index=False)
    print(f"[write] {out}/summary.tsv")

    md = [
        "# Simakov 2013 block-level benchmark (coord-mapped)\n",
        "**Source:** Simakov et al. 2013, *Nature* 493:526-531, "
        "Supplementary Data MOESM7 (`2012-01-00225C-microsynteny.xls`).\n",
        "",
        "## How gene-pair identities were mapped between the two datasets",
        "",
        "Simakov 2013's `All genes in region` field uses a 2012-era internal",
        "numeric ID namespace that does not round-trip to modern NCBI Entrez",
        "GeneIDs (verified: 1.7% direct overlap with current gene2accession,",
        "no useful coverage from gene_history). We therefore map via genomic",
        "coordinates:",
        "",
        "1. **Block coord → GENCODE v19 genes.** Each Simakov block has a",
        "   `Chrom:begin-end` field in GRCh37 coordinates. GENCODE v19",
        "   (release Dec 2013, the canonical GRCh37-frozen annotation and",
        "   contemporaneous with the Simakov paper) is loaded and gene",
        "   entries overlapping each block interval are collected.",
        "2. **GENCODE gene_name → Entrez GeneID.** Each overlapping gene's",
        "   HGNC symbol is resolved against the full human NCBI",
        "   `gene2accession` symbol table (192,184 symbol keys, no BCnS",
        "   pre-filter — same pipeline as our Irimia 2012 benchmark).",
        "3. **GeneID → BCnS family.** Each resolved GeneID is looked up in",
        "   our family map (`bcns_family_to_human_gene.tsv`).",
        "",
        "A block contributes representable pairs only when ≥2 of its",
        "overlapping genes resolve to BCnS families. Intra-block family",
        "pairs are canonicalised as `frozenset({family_A, family_B})` and",
        "tested against our 28-clade defining-pair union and the",
        "`stable_in_clade` subset.",
        "",
        "## Results",
        "",
        "| | value |",
        "|---|---|",
        f"| Simakov 2013 `{args.classification}` blocks ({args.species}) | **{n_blocks_total}** |",
        f"| Blocks with parseable coords | {n_blocks_coord_ok} |",
        f"| Blocks with ≥ 1 GENCODE gene in interval | {n_blocks_some_gene} |",
        f"| Blocks with ≥ 1 BCnS-mappable gene | {n_blocks_some_bcns} |",
        f"| Blocks contributing ≥ 1 intra-block BCnS family pair | {n_blocks_any_pair} |",
        f"| Unique intra-block family pairs | **{n_pairs_unique}** |",
        f"| Hits in our defining-pair union | **{uniq_hit_union}** ({100*uniq_hit_union/max(n_pairs_unique,1):.1f}%) |",
        f"| Hits in `stable_in_clade` subset | **{uniq_hit_stable}** ({100*uniq_hit_stable/max(n_pairs_unique,1):.1f}%) |",
        f"| BCnS pair universe size | {bcns_pair_universe_size:,} |",
        f"| Expected hits under uniform-random null (defining union) | {expected_union:.1f} |",
        f"| **Enrichment (defining union)** | **{enr_union:.1f}×** |",
        f"| Expected hits under uniform-random null (stable subset) | {expected_stable:.1f} |",
        f"| **Enrichment (stable subset)** | **{enr_stable:.1f}×** |",
        "",
        "## Interpretation",
        "",
        "Most Simakov 2013 bilaterianAnc blocks contain multiple genes that",
        "are not in the BCnS marker set (the blocks were defined genome-",
        "wide, whereas our analysis is restricted to the ~2,300 deep-time",
        "ortholog-family markers). What this benchmark tests is: **among",
        "pairs of genes that our marker set CAN see, how many are found in",
        "our defining-pair catalog?** A pair's presence in a Simakov block",
        "is an independent claim of bilaterian-ancestral tight linkage;",
        "presence in our catalog is an independent claim of clade-specific",
        "distance-distribution extremity. An above-null overlap is evidence",
        "that our pair-flagging machinery recovers the same biological",
        "signal Simakov 2013 reported.",
    ]
    (out / "report.md").write_text("\n".join(md))
    print(f"[write] {out}/report.md")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
