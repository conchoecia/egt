"""Helpers for manuscript-specific custom animal topology handling."""

from __future__ import annotations

import warnings

METAZOA_TAXID = 33208
EUMETAZOA_TAXID = 6072
CTENOPHORA_TAXID = 10197
PORIFERA_TAXID = 6040
CNIDARIA_TAXID = 6073
PLACOZOA_TAXID = 10226
BILATERIA_TAXID = 33213
MYRIAZOA_TAXID = -67
PARAHOXOZOA_TAXID = -68

CUSTOM_TAXID_NAMES = {
    MYRIAZOA_TAXID: "Myriazoa",
    PARAHOXOZOA_TAXID: "Parahoxozoa",
}


class CustomTopologyWarning(UserWarning):
    """Warning raised when NCBI taxonomy conflicts with manuscript topology."""


def warn_eumetazoa_replaced(stacklevel=2):
    """Warn that Eumetazoa was replaced by the manuscript custom topology."""
    warnings.warn(
        "Encountered NCBI Eumetazoa (6072) while applying the manuscript custom "
        "animal topology. Eumetazoa is replaced with Myriazoa (-67) and, where "
        "appropriate, Parahoxozoa (-68). Use an NCBI/no-custom-topology path if "
        "you intentionally want Eumetazoa.",
        CustomTopologyWarning,
        stacklevel=stacklevel,
    )


def apply_custom_animal_topology_to_taxid_lineage(lineage, warn=True):
    """Return lineage with manuscript animal topology replacing NCBI Eumetazoa."""
    lineage = [int(tid) for tid in lineage]
    if METAZOA_TAXID not in lineage:
        return lineage

    has_eumetazoa = EUMETAZOA_TAXID in lineage
    metazoa_index = lineage.index(METAZOA_TAXID)
    prefix = lineage[:metazoa_index + 1]

    if CTENOPHORA_TAXID in lineage:
        if warn and has_eumetazoa:
            warn_eumetazoa_replaced(stacklevel=3)
        ctenophora_index = lineage.index(CTENOPHORA_TAXID)
        return prefix + lineage[ctenophora_index:]

    if PORIFERA_TAXID in lineage:
        if warn and has_eumetazoa:
            warn_eumetazoa_replaced(stacklevel=3)
        porifera_index = lineage.index(PORIFERA_TAXID)
        return prefix + [MYRIAZOA_TAXID] + lineage[porifera_index:]

    parahoxozoa_children = [CNIDARIA_TAXID, PLACOZOA_TAXID, BILATERIA_TAXID]
    child_indices = [lineage.index(tid) for tid in parahoxozoa_children if tid in lineage]
    if child_indices:
        if warn and has_eumetazoa:
            warn_eumetazoa_replaced(stacklevel=3)
        child_index = min(child_indices)
        return prefix + [MYRIAZOA_TAXID, PARAHOXOZOA_TAXID] + lineage[child_index:]

    if has_eumetazoa:
        if warn:
            warn_eumetazoa_replaced(stacklevel=3)
        eumetazoa_index = lineage.index(EUMETAZOA_TAXID)
        return prefix + [MYRIAZOA_TAXID, PARAHOXOZOA_TAXID] + lineage[eumetazoa_index + 1:]

    return lineage
