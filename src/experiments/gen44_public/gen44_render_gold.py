# -*- coding: utf-8 -*-
# gen44_render_gold.py -- **gold-chain empirical convention** renderer (candidate enumeration;
# used to measure the device upper bound)
#
# Owner discipline (clarified 2026-09-26 00:0x):
#   This file is **only for the A2 / gold-chain empirical convention** (device upper-bound
#   measurement). Every reading that cites it must be labelled "includes rendering candidate
#   enumeration; device upper-bound convention". It is **forbidden** for model exam runs (the [8]
#   final test) -- the exam side always uses the deterministic single rendering of
#   `gen44_render_exam.py` (one chain, one rendered result).
#
# Basis: V42_DESIGN §1 family S "candidate order fixed (pre-registered); the gold chain = the
#        first candidate that passes the frozen judging triple" -- that rule was previously
#        **unimplemented** (only a 2-step chain existed); this file is its implementation.
#
# Candidate order (fixed, pre-registered in ERRATUM5 §C):
#   FamS: C1 [v, close] → C2 [close] → C3 [v] → C4 [v_alt1, close] → C5 [v_alt2, close]
#   FamA: C1 [g42_sb_close(norm_num [hyps])]  (single candidate)

VARIANT_ORDER = ("push_cast", "field_simp", "norm_num1")
_V2K = {"push_cast": "push", "field_simp": "fs", "norm_num1": "nn1"}


def candidates(F, rec, got, reduce_variant=None):
    """-> [proof_text, ...] (fixed order, deduplicated preserving order, "skip" / empty removed)."""
    ops = got["chain_ops"]
    pl = dict(got.get("route_payload") or {})
    pl.update(got)
    if got["family"] == F.FAM_SUBST:
        t = F.render_tactic(rec, got["family"], pl, ops)
        return [t] if t and t != "skip" else []
    route = got.get("route")
    rv = reduce_variant or F.reduce_variant
    v = pl.get("var") or rv(rec["concl"]["meta"])
    vname = {"push": "push_cast", "fs": "field_simp", "nn1": "norm_num1"}[v]
    close_op = "g42_s_%s" % route
    specs = [("c1_v_close", vname, ["g42_s_reduce", close_op]),
             ("c2_close", None, [close_op]),
             ("c3_v", vname, ["g42_s_reduce"])]
    alts = [x for x in VARIANT_ORDER if x != vname]
    for i, alt in enumerate(alts):
        specs.append(("c%d_%s_close" % (4 + i, alt), alt, ["g42_s_reduce", close_op]))
    out = []
    for name, vn, chain in specs:
        pl2 = dict(pl)
        if vn is not None:
            pl2["var"] = _V2K[vn]
        t = F.render_tactic(rec, got["family"], pl2, chain)
        if t and t != "skip" and t not in out:
            out.append(t)
    return out
