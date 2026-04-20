"""Tests for egt.go.enrichment."""
from __future__ import annotations

import math

from egt.go.enrichment import enrich_for_foreground, reduce_pairs_to_geneset


def test_reduce_pairs_to_geneset_unions():
    fam_to_genes = {
        "A": {"g1", "g2"},
        "B": {"g2", "g3"},
        "C": {"g4"},
    }
    # Not-found families contribute nothing.
    out = reduce_pairs_to_geneset(["A", "B", "Z"], fam_to_genes)
    assert out == {"g1", "g2", "g3"}


def test_reduce_pairs_to_geneset_empty():
    assert reduce_pairs_to_geneset([], {"A": {"g1"}}) == set()


def test_enrich_empty_foreground_returns_empty_namespaces():
    bg = {"g1": {"GO:X"}, "g2": {"GO:Y"}}
    out = enrich_for_foreground([], bg, {"GO:X": "BP", "GO:Y": "MF"})
    assert out == {"all": [], "BP": [], "MF": [], "CC": []}


def test_enrich_empty_background():
    out = enrich_for_foreground(["g1"], {}, {})
    assert out == {"all": [], "BP": [], "MF": [], "CC": []}


def test_enrich_detects_planted_overrepresentation():
    # 100 background genes. 10 of them carry the planted term.
    # Foreground = 20 genes; 8 of them carry the planted term.
    # Expected: hypergeom_sf(k=8, N=100, K=10, n=20) ≈ 0.0001 range,
    # so q for the planted term should be small and fold large.
    bg: dict[str, set[str]] = {}
    for i in range(100):
        gid = f"g{i}"
        terms = {"GO:BASE"}
        if i < 10:
            terms.add("GO:PLANTED")
        bg[gid] = terms
    ns = {"GO:BASE": "BP", "GO:PLANTED": "BP"}
    # Foreground: 8 of the first 10 (planted carriers) + 12 from rest.
    fg = [f"g{i}" for i in range(8)] + [f"g{i}" for i in range(50, 62)]
    res = enrich_for_foreground(fg, bg, ns)
    planted = [r for r in res["all"] if r["go_id"] == "GO:PLANTED"]
    assert len(planted) == 1
    r = planted[0]
    assert r["k"] == 8
    assert r["K"] == 10
    assert r["n"] == 20
    assert r["N"] == 100
    # fold = (8/20) / (10/100) = 4.0
    assert math.isclose(r["fold"], 4.0)
    assert r["p"] < 1e-3
    assert r["q"] < 1e-3
    # The "BP" namespace should contain the same row (all terms are BP).
    assert any(rr["go_id"] == "GO:PLANTED" for rr in res["BP"])
    # MF / CC empty.
    assert res["MF"] == []
    assert res["CC"] == []


def test_enrich_respects_min_term_hits():
    # Two terms: one with 1 foreground hit, one with 3. Default min=2
    # drops the singleton.
    bg = {f"g{i}": {"GO:A", "GO:B"} for i in range(50)}
    bg["g0"] = {"GO:A", "GO:RARE"}   # GO:RARE reached by 1 foreground gene
    bg["g1"] = {"GO:A"}               # not tagged with GO:RARE
    ns = {"GO:A": "BP", "GO:B": "BP", "GO:RARE": "BP"}
    res = enrich_for_foreground(["g0", "g1", "g2"], bg, ns, min_term_hits=2)
    terms = {r["go_id"] for r in res["all"]}
    assert "GO:RARE" not in terms

    res2 = enrich_for_foreground(["g0", "g1", "g2"], bg, ns, min_term_hits=1)
    terms2 = {r["go_id"] for r in res2["all"]}
    assert "GO:RARE" in terms2


def test_enrich_per_namespace_correction():
    # Many MF terms but only a few BP. BH correction operates per
    # namespace, so the same uncorrected p can yield a different q in
    # different namespaces.
    bg: dict[str, set[str]] = {}
    ns: dict[str, str] = {}
    for i in range(200):
        gid = f"g{i}"
        terms = {"GO:BP1"} if i < 60 else set()
        # Spread 30 molecular_function terms across genes so many
        # get k >= min_term_hits in a modest foreground.
        for m in range(30):
            if (i + m) % 7 == 0:
                terms.add(f"GO:MF{m}")
                ns[f"GO:MF{m}"] = "MF"
        terms.add("GO:BP1")
        bg[gid] = terms
    ns["GO:BP1"] = "BP"
    fg = [f"g{i}" for i in range(30)]
    res = enrich_for_foreground(fg, bg, ns)
    # "all" is the union: non-empty.
    assert res["all"]
    # BP and MF are non-empty.
    assert res["BP"]
    assert res["MF"]
    # Per-namespace BH: the BP q-values are independent of MF q-values.
    # Find GO:BP1 under both corrections.
    q_all = next(r["q"] for r in res["all"] if r["go_id"] == "GO:BP1")
    q_bp = next(r["q"] for r in res["BP"] if r["go_id"] == "GO:BP1")
    # The BP-only correction corrects over a smaller set, so its q is
    # generally <= the all-namespace q for the same term.
    assert q_bp <= q_all + 1e-12


def test_enrich_handles_missing_namespace_gracefully():
    # An empty namespace subset short-circuits to no output for that ns.
    bg = {"g1": {"GO:X"}, "g2": {"GO:X"}}
    # Declare term as MF — CC sub-slice is empty.
    ns = {"GO:X": "MF"}
    res = enrich_for_foreground(["g1", "g2"], bg, ns)
    assert res["MF"]
    assert res["CC"] == []
