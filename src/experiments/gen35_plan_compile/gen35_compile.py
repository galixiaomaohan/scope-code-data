# -*- coding: utf-8 -*-
# gen35_compile.py -- **the proof compiler** (core engineering artifact, change 98): operator chain -> proof text
#
# Role (V35_DESIGN §2): the LLM emits only the **operator-name sequence**; the actual arguments of each
#   accepted step are computed on the fly by the **device** from the row's own payload (corpus row
#   steps[].args / exam sidecar sfa) plus the accumulated chain state; proof text = the operator-template
#   tactic of each emitting step concatenated in chain order. **The reference proof text
#   (proof_text / pr["proof"]) is never read by this file.**
#
# Acceptance semantics (pre-registered): step k is accepted only if chain[k] == ops_gold[k]; otherwise
#   OP_MISMATCH (no line emitted); k >= len(ops_gold) => OVER_RUN (no line emitted). Emitting operator
#   set = operators that emit proof.append/extend in the generator (device-only steps fold/delta only
#   update device state, emitting nothing).
#
# Only imports (zero edits): gen32_ops (operator impl + proof templates) . gen32_poly (polynomials) .
#   gen33_device (ell-reconstruction convention).
import os
import sys
from fractions import Fraction

import gen35_common as C

import gen32_ops as OPS                                               # noqa: E402
import gen32_poly as P                                               # noqa: E402
import gen33_device as DEV                                           # noqa: E402

SOS_BLOCK_OPS = ("gen32_sos_block", "gen32_sos_cross")
SOS_TAIL_OPS = ("gen32_sos_tail_zero", "gen32_sos_tail_square", "gen32_sos_tail_other")
PROOF_EMIT = ("gen32_nest_solve", "gen32_nest_close") + SOS_BLOCK_OPS + SOS_TAIL_OPS
DEVICE_ONLY = ("gen32_nest_fold", "gen32_nest_delta")
KNOWN_OPS = PROOF_EMIT + DEVICE_ONLY


# ---------------------------------------------------------------- row -> view
def pv_of_corpus(row):
    meta = row.get("meta") or {}
    return {"name": row["name"], "family": row.get("family"), "cls": row.get("cls"),
            "head": row["head"], "concl": row["concl"], "stmt": row.get("stmt_state"),
            "ops": list(row.get("chain_ops") or []),
            "sfa": [dict(s.get("args") or {}) for s in (row.get("steps") or [])],
            "vars": list(meta.get("vars") or []), "dom_dump": meta.get("dom_dump"),
            "meta": meta, "src": "corpus"}


def pv_of_exam(rec):
    return {"name": rec.get("name"), "family": rec.get("family"), "cls": rec.get("cls"),
            "head": rec.get("head"), "concl": rec.get("concl"), "stmt": rec.get("stmt"),
            "ops": list(rec.get("ops") or []),
            "sfa": [dict(x or {}) for x in (rec.get("sfa") or [])],
            "vars": list(rec.get("vars") or []), "dom_dump": rec.get("dom_dump"),
            "meta": {}, "src": "exam"}


def pv_ops(pv, chain=None):
    return list(chain if chain is not None else pv["ops"])


def pv_dom(pv):
    """Family-C target-domain polynomial: meta.dom_dump -> row dom_dump -> terminal-step args.goal_dom (same convention as gen33_device)."""
    d = (pv.get("meta") or {}).get("dom_dump")
    if d is None:
        d = pv.get("dom_dump")
    if d is None:
        for a in reversed(pv["sfa"]):
            if a.get("goal_dom") is not None:
                d = a["goal_dom"]
                break
    return d


def pv_vars(pv):
    v = list(pv.get("vars") or [])
    if not v:
        for a in pv["sfa"]:
            if a.get("vars"):
                v = list(a["vars"])
                break
    return v


def pv_nv(pv):
    v = pv_vars(pv)
    if v:
        return len(v)
    for a in pv["sfa"]:
        if a.get("nv"):
            return int(a["nv"])
    return 1


# ---------------------------------------------------------------- device argument supply
def dev_args(pv, k, st):
    """**Device execution arguments** of the k-th accepted step (computed from payload; the reference proof text and in-row res are never read)."""
    op = pv["ops"][k]
    a = dict(pv["sfa"][k] or {})
    if op in SOS_BLOCK_OPS:
        vars_ = pv_vars(pv)
        if not vars_:
            raise KeyError("sos 块缺 vars (行 vars / 步载荷 vars 皆无)")
        a["vars"] = vars_
        a["R_in"] = P.dump(st["R"]) if st.get("R") is not None else a.get("R_in")
        a["hint_idx"] = int(st["nblk"]) + 1
    elif op in SOS_TAIL_OPS:
        vars_ = pv_vars(pv)
        if not vars_:
            raise KeyError("sos 尾缺 vars")
        nv = len(vars_)
        a["vars"] = vars_
        a["hints"] = ["h%d" % (i + 1) for i in range(int(st["nblk"]))]
        a["hints_poly"] = list(st["hints_poly"])
        dom = pv_dom(pv)
        a["goal_dom"] = dom if dom is not None else a.get("goal_dom")
        if st.get("R") is not None:
            a["rho"] = str(st["R"].get(tuple([0] * nv), Fraction(0)))
        if op == "gen32_sos_tail_square" and a.get("sq") is None:
            a["sq"] = 2
    return a


# ---------------------------------------------------------------- compilation
def compile_chain(pv, chain, strict=True, keep_recs=True):
    """chain -> {proof_text, steps, ...}. strict=True = pre-registered acceptance semantics (§2.2)."""
    gold = list(pv["ops"])
    st = {"R": None, "nblk": 0, "hints_poly": []}
    dom = pv_dom(pv)
    if dom is not None:
        st["R"] = P.load(dom)
    lines, recs = [], []
    n_emit = n_dev_ok = 0
    for k, op in enumerate(chain):
        rec = {"k": k, "op": op, "gold": gold[k] if k < len(gold) else None}
        if k >= len(gold):
            rec["err"] = "OVER_RUN"
            recs.append(rec)
            continue
        if strict and op != gold[k]:
            rec["err"] = "OP_MISMATCH"
            recs.append(rec)
            continue
        if op not in OPS.OPS:
            rec["err"] = "OP_NOT_REGISTERED"
            recs.append(rec)
            continue
        try:
            args = dev_args(pv, k, st)
        except Exception as e:                                       # noqa: BLE001
            rec["err"] = "ARGS_EXC:%s:%s" % (type(e).__name__, e)
            recs.append(rec)
            continue
        rec["args"] = args if keep_recs else None
        # device-argument self-consistency reading (compared against in-row recorded values; diagnostic only, arguments unchanged)
        for key in ("R_in", "goal_dom", "rho"):
            if key in (pv["sfa"][k] or {}):
                rec["dev_match_" + key] = bool(C.canon(args.get(key)) ==
                                               C.canon(pv["sfa"][k].get(key)))
        try:
            r = OPS.call(op, args)
        except Exception as e:                                       # noqa: BLE001
            rec["err"] = "OP_EXC:%s:%s" % (type(e).__name__, e)
            recs.append(rec)
            continue
        rec["ok"] = bool(r.ok)
        rec["err"] = r.err
        rec["cert_detail"] = (r.cert_detail or "")[:200]
        rec["resid_spec"] = r.resid_spec
        rec["tactic_lines"] = len([l for l in (r.tactic or "").split("\n") if l.strip()])
        rec["emit"] = bool(r.ok and op in PROOF_EMIT)
        if rec["emit"]:
            lines.append(r.tactic)
            n_emit += 1
        if r.ok and op in DEVICE_ONLY:
            n_dev_ok += 1
        if r.ok and op in SOS_BLOCK_OPS:
            ell = DEV.ell_poly(args)
            absorbed = P.pscale(P.pmul(ell, ell), Fraction(args["mult"]))
            st["hints_poly"].append(P.dump(absorbed))
            st["R"] = (P.load(r.value["R_out"]) if isinstance(r.value, dict)
                       and r.value.get("R_out") is not None else P.psub(st["R"], absorbed))
            st["nblk"] += 1
        recs.append(rec)
    proof = "\n".join([l for l in lines if l.strip()])
    err_hist = {}
    for r in recs:
        if r.get("err"):
            err_hist[r["err"]] = err_hist.get(r["err"], 0) + 1
    n_mis = sum(1 for r in recs if r.get("err") == "OP_MISMATCH")
    cmp_n = min(len(chain), len(gold))
    out = {"name": pv.get("name"), "src": pv.get("src"), "family": pv.get("family"),
           "n_chain": len(chain), "n_gold": len(gold),
           "first_hit": bool(chain and gold and chain[0] == gold[0]),
           "exact_hit": bool(list(chain) == gold),
           "prefix_hit": bool(cmp_n and list(chain)[:cmp_n] == gold[:cmp_n]),
           "over_run": bool(len(chain) > len(gold)),
           "n_emit": n_emit, "n_dev_ok": n_dev_ok, "n_mismatch": n_mis,
           "err_hist": err_hist, "proof_text": proof,
           "n_lines": len([l for l in proof.split("\n") if l.strip()]),
           "proof_md5": C.md5s(proof)}
    if keep_recs:
        out["steps"] = recs
    return out


def dev_payload_report(pv):
    """Device-argument self-consistency check (diagnostic only): per-step recomputation vs in-row recorded values."""
    res = compile_chain(pv, list(pv["ops"]))
    n_key = n_match = 0
    bad = []
    for r in res["steps"]:
        for key in ("R_in", "goal_dom", "rho"):
            mk = "dev_match_" + key
            if mk in r:
                n_key += 1
                if r[mk]:
                    n_match += 1
                elif len(bad) < 8:
                    bad.append({"k": r["k"], "op": r["op"], "key": key})
    return {"n_key": n_key, "n_match": n_match,
            "match_rate": round(n_match / max(1, n_key), 6), "bad": bad,
            "device_ok_rate": round(sum(1 for r in res["steps"] if r.get("ok")) /
                                    max(1, len(res["steps"])), 6)}
