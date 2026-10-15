# -*- coding: utf-8 -*-
# ccp_unit.py -- unit tests for the CCP three classes / three artifacts (advance team gen12, constitution change 58 step C acceptance)
#
# 8 assertions over 12 examples per class (mixed across families):
#   U1 (renamed in gen15 F-04) bridge_produced = **count of make_problem outputs that passed the
#      built-in roundtrip gate** (semantics = "produced and not blocked by the built-in roundtrip
#      gate", **not** an independent re-verification reading; an independent bridge roundtrip
#      assertion is not implemented in this batch => this field is not evidence for
#      "statement -> params == generation params". See CCP_UNIT_CORRIGENDA.md §F-04)
#   U2 every step's certificate kind is registered AND independent of the generator module AND
#      passes re-verification
#   U3 operator allowlist (positive chains ⊆ PROOF_OPS) AND zero occurrences of banned operators
#   U4 full chain Lean rc=0 AND no sorry
#   U5 eight-tactic floor: 0 closures (above the floor)
#   U6 negative examples fail truly (semantic perturbation -> operator error, not a resource error)
#      [gen15 F-01 fix: n_ok increments only when np_[2] is not None (the operator really errored);
#       before the fix the counter incremented unconditionally => it always equalled n_tot and
#       could never fail]
#   U7 **checker non-vacuity**: perturb the payload of a correct certificate numerically; the
#      independent checker must reject (falsifies that the checker is effective)
#   U8 Valid anchoring: representative problem names of this class in Valid (names only, no
#      statement text)
#      [gen15 F-09 adds technique_anchor: name families anchored by **technique**, unrelated to
#       the mechanical priority chain]
# Writes only scratch/ccp/UNIT_*.json (+ the Lean intermediate directory chosen by --lean-tag).
import argparse
import collections
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ccp_bridge as BR
import ccp_check as CK
import ccp_families as F
import ccp_lean as L
import ccp_ops as OPS

RES = os.environ.get("SCOPE_RESEARCH_ROOT", ".")
CCP = os.path.join(RES, "scratch", "ccp")
MASTER = 20260923
N_PER_CLASS = 12


def perturb(kind, payload):
    """Numeric perturbation of a legal certificate (the checker must reject)."""
    p = json.loads(json.dumps(payload))
    if kind == "ccp_ind_recurrence_cert":
        p["closed"]["value"] = str(int(p["closed"]["value"]) + 1)
    elif kind == "ccp_ring_identity_cert":
        k = sorted(p["lhs"])[0]
        p["lhs"][k] = str(int(p["lhs"][k]) + 1)
    elif kind in ("ccp_ineq_sos_cert", "ccp_ineq_close_cert"):
        key = "input_dom" if kind == "ccp_ineq_sos_cert" else "goal_dom"
        k = sorted(p[key])[0]
        p[key][k] = str(int(p[key][k]) + 1)
    elif kind == "ccp_nt_dvd_cert":
        p["const"] = int(p["const"]) + 1
    elif kind == "ccp_shape_cert":
        p["hyp"] = "zz_nonexistent"           # hypothesis name does not exist => the shape check must reject
        p["m"] = int(p["m"]) + 1              # wrong modulus => the shape check must reject
    else:
        return None
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=N_PER_CLASS)
    ap.add_argument("--skip-lean", action="store_true")
    # gen15 (append-only guard): Lean intermediates go to a unit_*<tag> subdirectory; default "" = original behavior (no overwrite of old intermediates)
    ap.add_argument("--lean-tag", default="", help="Lean intermediate subdirectory suffix (concatenated verbatim, e.g. _gen15 => unit_chain_gen15)")
    a = ap.parse_args()
    rep = {"schema": "ccp-unit/1", "master": MASTER, "per_class": a.per_class, "classes": {}}
    jobs, meta = [], []
    for cls in F.CLASSES:
        fams = list(F.PROBE_FAMS[cls])
        probs, errs = [], collections.Counter()
        seq, fi = 0, 0
        while len(probs) < a.per_class and seq < a.per_class * 80:
            seq += 1
            family = fams[fi % len(fams)]
            fi += 1
            pr, err = F.make_problem(cls, family, seq, MASTER)
            if pr is None:
                errs["%s:%s" % (family, err)] += 1
                continue
            probs.append(pr)
        # U1/U2/U3/U6/U7 (pure CPU)
        u = {"n": len(probs), "gen_fail": dict(errs), "bridge_produced": 0,
             "U2_cert_ok": 0, "U2_cert_steps": 0, "U3_whitelist": 0, "U6_neg_truefail": 0,
             "U7_checker_nonvacuous": 0, "U7_n": 0, "cert_kinds": {}, "fams": {}}
        for pr in probs:
            u["fams"][pr["family"]] = u["fams"].get(pr["family"], 0) + 1
            u["bridge_produced"] += 1     # F-04: make_problem yields only after passing the built-in roundtrip gate (output count)
            for s in pr["chain"]:
                u["U2_cert_steps"] += 1
                if s.get("cert_ok"):
                    u["U2_cert_ok"] += 1
                k = s["cert"]["kind"] if isinstance(s.get("cert"), dict) else None
                u["cert_kinds"][k] = u["cert_kinds"].get(k, 0) + 1
                if k:
                    # U7: perturbed re-verification must reject
                    pj = perturb(k, s["cert"]["payload"])
                    if pj is not None:
                        ok2, _, _ = CK.check(k, pj)
                        u["U7_n"] += 1
                        u["U7_checker_nonvacuous"] += 0 if ok2 else 1
            okwl, _ = OPS.whitelist_check([s["op"] for s in pr["chain"]])
            u["U3_whitelist"] += 1 if okwl else 0
            if not a.skip_lean:
                jobs.append((pr["name"], pr["stmt"], "\n".join(pr["proof"])))
            meta.append(pr)
        rep["classes"][cls] = u
    # U4/U5 (Lean)
    if not a.skip_lean and jobs:
        cres = L.chain_check(jobs, subdir="unit_chain" + a.lean_tag)
        for cls in F.CLASSES:
            rep["classes"][cls]["U4_lean_ok"] = sum(
                1 for (n, _, _) in jobs if n.startswith("ccp_%s_" % cls) and cres.get(n, {}).get("ok"))
            rep["classes"][cls]["U4_n"] = sum(1 for (n, _, _) in jobs if n.startswith("ccp_%s_" % cls))
        fsamp = collections.defaultdict(list)
        for (n, s, _) in jobs:
            for cls in F.CLASSES:
                if n.startswith("ccp_%s_" % cls):
                    fsamp[cls].append((n, s))
        for cls, items in fsamp.items():
            fl = L.floor_probe(items, subdir="unit_floor_" + cls + a.lean_tag)
            rep["classes"][cls]["U5_floor_hard"] = sum(1 for v in fl.values() if not v["closers"])
            rep["classes"][cls]["U5_floor_n"] = len(items)
            rep["classes"][cls]["U5_closers"] = dict(collections.Counter(
                c for v in fl.values() for c in v["closers"]))
    # U6 (gen15 F-01 fix: the true-failure criterion is that the perturbed operator returns a
    #     non-None err, i.e. np_[2] is not None; before the fix n_ok += 1 was unrelated to np_[2],
    #     so the counter always equalled n_tot and could not fail structurally)
    for cls in F.CLASSES:
        n_ok = n_tot = 0
        notfail, unbuildable = [], []
        for pr in [p for p in meta if p["cls"] == cls][:a.per_class]:
            np_ = F_neg(cls, pr)
            if np_ is None:
                unbuildable.append(pr["name"])      # perturbation could not be built (excluded from the denominator)
                continue
            n_tot += 1
            if np_[2] is not None:
                n_ok += 1                           # true failure: the operator really errored
            else:
                notfail.append(pr["name"])          # still solvable after perturbation => not a true failure (in the denominator, not the numerator)
        rep["classes"][cls]["U6_neg_truefail"] = n_ok
        rep["classes"][cls]["U6_neg_n"] = n_tot
        rep["classes"][cls]["U6_neg_notfail"] = notfail
        rep["classes"][cls]["U6_neg_unbuildable"] = unbuildable
        rep["classes"][cls]["U6_truefail_rate"] = round(n_ok / float(n_tot), 4) if n_tot else None
    # U8 Valid anchoring (example names read from the taxonomy JSON)
    tp = os.path.join(CCP, "CCP_TAXONOMY.json")
    if os.path.exists(tp):
        tax = json.load(open(tp, encoding="utf-8"))
        anchors = {}
        for c in tax["classes"]:
            anchors[c["id"]] = c["valid_examples"]
        rep["U8_valid_anchor"] = {k: anchors.get(k, []) for k in
                                  ("induction_rec", "induction_sum", "sos_bound",
                                   "bound_chain", "div_closure", "mod_arith")}
    # F-09: anchor by **technique** (not the mechanical priority chain) -- fills U8_valid_anchor's empty set for induction_rec
    rep["technique_anchor"] = technique_anchor()
    out = os.path.join(CCP, "UNIT_CCP.json")
    rep["md5"] = L.save_json(out, rep)
    print(json.dumps(rep["classes"], ensure_ascii=False, indent=1))
    print("[unit] → %s md5=%s" % (out, rep["md5"]))


def F_neg(cls, pr):
    import ccp_course as C
    return C.neg_probe(cls, pr)


def technique_anchor():
    """F-09: Valid name families anchored by **technique** (not by the mechanical priority chain)
    (names only, no statement text).

    Two routes side by side (both reproducible, neither substitutes for the other):
      (a) by_name_family -- the CCP_PILOT_SPEC.md convention: Valid problems whose name contains
          `induction` (the induction_* name family);
      (b) by_structure   -- the structural convention: those `ccp_taxonomy._feat` reads as
          inductive structure (forall ∧ (hypchain ∨ powvar)).
    Both routes carry `mechanical_label_of_anchored` (the class these problems actually receive
    from the mechanical `classify()` priority chain) -- precisely because the earlier branches
    (sum/choose/dvd/card/prime/mod/analysis) absorb them, the taxonomy's
    `valid_examples[induction_rec]` is empty and the U8 field reads as an empty set; this field is
    the real "anchored by technique" list."""
    out = {"purpose": "F-09: U8_valid_anchor[induction_rec] 为空 = 机械链口径, 非「题库无归纳题」; "
                      "本字段给出按技巧锚定的名单与机械链吸走方向",
           "induction_rec": {"by_name_family": {}, "by_structure": {}}}
    try:
        import ccp_taxonomy as T
        valid = T.load_valid()

        def _collect(pred, rule):
            names, mech = [], collections.Counter()
            for p in valid:
                if pred(p):
                    names.append(p.name)
                    mech[T.classify(p.binders, p.concl, p.name)] += 1
            return {"rule": rule, "n": len(names), "names": names,
                    "mechanical_label_of_anchored": dict(mech)}

        out["induction_rec"]["by_name_family"] = _collect(
            lambda p: bool(re.search("induct", p.name, re.I)),
            "Valid 题名含 'induct' (CCP_PILOT_SPEC.md:7 的 induction_* 题名族口径)")
        out["induction_rec"]["by_structure"] = _collect(
            lambda p: (lambda f: f["forall"] and (f["hypchain"] or f["powvar"]))(
                T._feat(p.binders, p.concl, p.name)),
            "Valid 中 _feat 判为归纳结构 (forall ∧ (hypchain ∨ powvar))")
        out["n"] = out["induction_rec"]["by_name_family"]["n"]
    except Exception as e:                                     # noqa: BLE001
        out["error"] = "%s: %s" % (type(e).__name__, e)
        out["n"] = 0
    return out


if __name__ == "__main__":
    main()
