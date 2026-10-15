# -*- coding: utf-8 -*-
# bh_env.py -- bridge-hard batch state-machine executor (gen14): shared environment for the rule arm / gates / curriculum
#
# Responsibilities (device side of the change-49 obligation split):
#   * statement text -> arguments = the bridge (bh_bridge), called by this module;
#   * **class / operator name / candidate index are supplied by the caller (model or rule arm)**;
#   * each step's intermediate objects (residual / closed form / body) are computed by the device and exposed as state;
#   * terminal-step / closed-chain arguments are derived by the device from **current state** => "stop early",
#     "drop a block" and "wrong block" are all rejected by the certificate.
# Candidate-set order: the md5 of (problem name, step index, candidate text) decides it (deterministic, not
# "true answer first") -- defeats index shortcuts.
import hashlib
import re
from fractions import Fraction

import bh_bridge as BR
import bh_ops as OPS
import bh_poly as P

NAT = "ℕ"


def stable_order(items, keyf):
    return sorted(items, key=lambda c: hashlib.md5(
        ("%s|%s" % (keyf, _desc(c))).encode("utf-8")).hexdigest())


def _desc(c):
    a = c.get("args") or {}
    parts = [str(c.get("op"))]
    if "ell" in a:
        parts.append(str(sorted(a["ell"].get("coefs", {}).items())))
        parts.append(str(a["ell"].get("shift")))
        parts.append(str(a.get("mult")))
    if "hyp" in a:
        parts.append(str(a["hyp"]))
        parts.append(str(a.get("var")))
    return "|".join(parts)


# ================================================================ IND
def ind_exec(pr, prog, name):
    if len(prog) != 2:
        return False, "prog_len=%d" % len(prog)
    step_cands = []
    for k in range(2):
        cs = pr["cands"][k]
        step_cands.append({c["op"]: c for c in cs})
    r0 = OPS.call(prog[0]["op"], step_cands[0][prog[0]["op"]]["args"]) \
        if prog[0]["op"] in step_cands[0] else None
    if r0 is None:
        return False, "op0_not_in_candidates"
    if not (r0.ok and r0.cert_ok):
        return False, "step0:%s" % (r0.err or "cert_reject")
    r1 = OPS.call(prog[1]["op"], step_cands[1][prog[1]["op"]]["args"]) \
        if prog[1]["op"] in step_cands[1] else None
    if r1 is None:
        return False, "op1_not_in_candidates"
    if not (r1.ok and r1.cert_ok):
        return False, "step1:%s" % (r1.err or "cert_reject")
    return True, "certified"


# ================================================================ INEQ
def ineq_initial(pr):
    m = pr["meta"]
    return dict(m["dom_poly"]), m


def ineq_candidates(pr, k, R):
    """**State-driven** candidate set (not a reference order): blocks whose support variables still occur
    in the residual = viable candidates (possibly several); all other blocks and the perturbed decoys =
    decoys (rejected locally by the purge criterion)."""
    m = pr["meta"]
    vars_, nv = m["vars"], m["nv"]
    idx = {v: i for i, v in enumerate(vars_)}
    live_vars = set()
    for mono, c in R.items():
        if c != 0:
            for i, e in enumerate(mono):
                if e > 0:
                    live_vars.add(vars_[i])
    out = []
    for b in m["blocks"]:
        op = "bh_ineq_cross" if len(b["coefs"]) >= 2 else "bh_ineq_block"
        present = any(v in live_vars for v in b["coefs"])
        out.append({"op": op, "args": {"ell": {"coefs": dict(b["coefs"]),
                                               "shift": b["shift"]}, "mult": b["mult"]},
                    "viable": present})
        out.append({"op": op, "args": {"ell": {"coefs": dict(b["coefs"]),
                                               "shift": b["shift"] + 1},
                                       "mult": b["mult"]}, "viable": False})
        if len(b["coefs"]) == 1:
            v = list(b["coefs"])[0]
            out.append({"op": op, "args": {"ell": {"coefs": {v: b["coefs"][v] * 2},
                                                   "shift": b["shift"]},
                                           "mult": b["mult"]}, "viable": False})
        else:
            out.append({"op": op, "args": {"ell": {"coefs": dict(b["coefs"]),
                                                   "shift": b["shift"] + 2},
                                           "mult": b["mult"] + 1}, "viable": False})
    return stable_order(out, "%s|%d|%s" % (pr["name"], k, P.dump(R)))


def ineq_tail_candidates(pr, R, hints, hints_poly):
    m = pr["meta"]
    out = []
    for cn, key in (("bh_ineq_tail_zero", "zero"), ("bh_ineq_tail_square", "square"),
                    ("bh_ineq_tail_other", "other")):
        a = {"vars": m["vars"], "rho": str(R.get((0,) * m["nv"], Fraction(0))),
             "hints": hints, "goal_dom": m["dom_dump"], "hints_poly": hints_poly}
        if key == "square":
            a["sq"] = m["cls_sq"] if m["cls_sq"] else 2
        out.append({"op": cn, "args": a, "viable": key == m["cls"]})
    return stable_order(out, "%s|tail|%s" % (pr["name"], P.dump(R)))


def ineq_exec(pr, prog, name):
    """prog = [ {op, cand} x n_blk ] + [ {op, cand} (tail) ]."""
    m = pr["meta"]
    vars_, nv = m["vars"], m["nv"]
    R = P.load(m["dom_dump"])
    hints, hints_poly = [], []
    if len(prog) < 1:
        return False, "empty_prog"
    nblk = len(prog) - 1
    if nblk > len(m["blocks"]):
        return False, "too_many_blocks(%d>%d)" % (nblk, len(m["blocks"]))
    for k in range(nblk):
        cands = ineq_candidates(pr, k, R)
        idx = int(prog[k].get("cand") or 0)
        if idx >= len(cands):
            return False, "cand_idx_oob@%d" % k
        cc = cands[idx]
        args = dict(cc["args"])
        args.update({"vars": vars_, "R_in": P.dump(R), "hint_idx": k + 1})
        r = OPS.call(cc["op"], args)
        if not (r.ok and r.cert_ok):
            return False, "block%d:%s" % (k, r.err or "cert_reject")
        hints.append("h%d" % (k + 1))
        ell = OPS._ell_poly(vars_, cc["args"]["ell"])
        hints_poly.append(P.dump(P.pscale(P.pmul(ell, ell), Fraction(cc["args"]["mult"]))))
        R = P.load(r.value["R_out"])
    tcands = ineq_tail_candidates(pr, R, hints, hints_poly)
    tidx = int(prog[-1].get("cand") or 0)
    if tidx >= len(tcands):
        return False, "tail_idx_oob"
    tc = tcands[tidx]
    rt = OPS.call(tc["op"], tc["args"])
    if not (rt.ok and rt.cert_ok):
        return False, "tail:%s" % (rt.err or "cert_reject")
    return True, "certified"


# ================================================================ NT
def nt_state(pr):
    m = pr["meta"]
    return {"body": m["body0"], "expanded": [], "k": 0}


def nt_candidates(pr, st):
    m = pr["meta"]
    out = []
    for d in m["defs"]:
        if _occ(d["lhs"], st["body"]):
            out.append({"op": "bh_nt_rewrite", "args": {"hyp": d["name"], "lhs": d["lhs"]},
                        "viable": True, "key": d["name"]})
    for h in m["hyps"]:
        if _occ(h["var"], st["body"]):
            op = ("bh_nt_expand_v%d" % h["val"] if pr["family"] == "nt_val"
                  else "bh_nt_expand")
            out.append({"op": op, "args": {"m": m["m"], "hyp": h["name"], "var": h["var"],
                                           "modulus": m["m"] ** h["val"]},
                        "viable": True, "key": h["name"]})
    live = [c for c in out if c["viable"]]
    for h in m["hyps"]:
        if not _occ(h["var"], st["body"]):
            out.append({"op": "bh_nt_expand", "args": {"m": m["m"], "hyp": h["name"],
                                                       "var": h["var"]},
                        "viable": False, "key": h["name"]})
    closable = (not live)
    if closable:
        out.append({"op": "bh_nt_close", "args": {}, "viable": True, "key": "close"})
    return stable_order(out, "%s|%d|%s" % (pr["name"], st["k"], st["body"]))


def nt_close_args(pr, st):
    """Device-side derivation of terminal-step arguments: build the witness from the statement terms
    plus the expanded set."""
    m = pr["meta"]
    params = sorted(set(["w_%s" % v for v in st["expanded"]]))
    wit = []
    terms = []
    for t in m["terms"]:
        vs = t["vars"]
        c = Fraction(t["coef"]) * Fraction(m["m"]) ** t["k"] / m["M"]
        if c.denominator != 1 or c < 0:
            return None
        terms.append({"coef": str(t["coef"]), "k": t["k"],
                      "wvars": ["w_%s" % v for v in vs]})
        wit.append({"coef": str(int(c)), "monomial": ["w_%s" % v for v in vs]})
    if m["K"] // m["M"] != 0:
        wit.append({"coef": str(int(m["K"] // m["M"])), "monomial": []})
    return {"m": m["m"], "M": m["M"], "params": params, "terms": terms,
            "const": m["K"], "witness": wit}


def nt_exec(pr, prog, name):
    m = pr["meta"]
    st = nt_state(pr)
    for si, step in enumerate(prog):
        cands = nt_candidates(pr, st)
        idx = int(step.get("cand") or 0)
        if idx >= len(cands):
            return False, "cand_oob@%d" % si
        cc = cands[idx]
        if step.get("op") and step["op"] != cc["op"]:
            return False, "op_mismatch@%d" % si
        if cc["op"] == "bh_nt_close":
            cargs = nt_close_args(pr, st)
            if cargs is None:
                return False, "witness_nonnatural"
            r = OPS.call("bh_nt_close", cargs)
            if not (r.ok and r.cert_ok):
                return False, "close:%s" % (r.err or "cert_reject")
            leftover = _leftover_vars(st["body"], pr)
            if leftover:
                return False, "unclosed_vars=%s" % leftover
            return True, "certified"
        key = cc["key"]
        if cc["op"] == "bh_nt_rewrite":
            d = [x for x in m["defs"] if x["name"] == key][0]
            a = {"hyp": d["name"], "lhs": d["lhs"], "text": _ctx(pr, st["body"]),
                 "body": st["body"]}
            r = OPS.call("bh_nt_rewrite", a)
            if not (r.ok and r.cert_ok):
                return False, "rewrite:%s" % (r.err or "cert_reject")
            st["body"] = BR.subst_var(st["body"], d["lhs"], d["rhs"])
        else:
            h = [x for x in m["hyps"] if x["name"] == key][0]
            if cc["op"] == "bh_nt_expand" and (h["val"] != 1 or pr["family"] == "nt_val"):
                # non-valuation operator used in a valuation family / valuation operator used with v=1 => class-assertion mismatch
                pass
            a = {"m": m["m"], "hyp": h["name"], "var": h["var"], "text": _ctx(pr, st["body"]),
                 "body": st["body"], "modulus": m["m"] ** h["val"]}
            r = OPS.call(cc["op"], a)
            if not (r.ok and r.cert_ok):
                return False, "expand_%s:%s" % (key, r.err or "cert_reject")
            st["body"] = BR.subst_var(st["body"], h["var"],
                                      "%d * w_%s" % (m["m"] ** h["val"], h["var"]))
            st["expanded"].append(h["var"])
        st["k"] += 1
    return False, "no_close(%d steps)" % len(prog)


def _ctx(pr, body):
    m = pr["meta"]
    return "(%s : %s)%s %s ⊢ %d ∣ %s" % (m["binders"], NAT, m["def_txt"],
                                                     m["hyp_txt"], m["M"], body)


def _occ(v, text):
    return bool(re.search(r"(?<![A-Za-z0-9_'])%s(?![A-Za-z0-9_'])" % re.escape(v), text))


def _leftover_vars(body, pr):
    m = pr["meta"]
    names = [h["var"] for h in m["hyps"]] + [d["lhs"] for d in m["defs"]]
    return [v for v in names if _occ(v, body)]


# ================================================================ unified entry point
def run_program(cls, pr, prog, name=None):
    name = name or pr["name"]
    if cls == "ind":
        return ind_exec(pr, prog, name)
    if cls == "ineq":
        return ineq_exec(pr, prog, name)
    return nt_exec(pr, prog, name)


def play_program(cls, pr):
    """Device self-play reference (RB-play): pick the "viable" candidate at each step (first in stable
    order when several are viable)."""
    if cls == "ind":
        prog = []
        for k in range(2):
            cs = pr["cands"][k]
            i = [j for j, c in enumerate(cs) if c.get("viable")][0]
            prog.append({"op": cs[i]["op"], "cand": i})
        return prog
    if cls == "ineq":
        m = pr["meta"]
        R = P.load(m["dom_dump"])
        prog, hints, hp = [], [], []
        for k in range(len(m["blocks"])):
            cs = ineq_candidates(pr, k, R)
            i = [j for j, c in enumerate(cs) if c.get("viable")][0]
            prog.append({"op": cs[i]["op"], "cand": i})
            args = dict(cs[i]["args"])
            args.update({"vars": m["vars"], "R_in": P.dump(R), "hint_idx": k + 1})
            r = OPS.call(cs[i]["op"], args)
            if not r.ok:
                break
            hints.append("h%d" % (k + 1))
            ell = OPS._ell_poly(m["vars"], cs[i]["args"]["ell"])
            hp.append(P.dump(P.pscale(P.pmul(ell, ell), Fraction(cs[i]["args"]["mult"]))))
            R = P.load(r.value["R_out"])
        tc = ineq_tail_candidates(pr, R, hints, hp)
        i = [j for j, c in enumerate(tc) if c.get("viable")][0]
        prog.append({"op": tc[i]["op"], "cand": i})
        return prog
    st = nt_state(pr)
    prog = []
    guard = 0
    while guard < 64:
        guard += 1
        cs = nt_candidates(pr, st)
        live = [j for j, c in enumerate(cs) if c.get("viable")]
        if not live:
            break
        i = live[0]
        prog.append({"op": "bh_nt_close" if cs[i]["op"] == "bh_nt_close" else None,
                     "cand": i})
        cc = cs[i]
        if cc["op"] == "bh_nt_close":
            break
        key = cc["key"]
        if cc["op"] == "bh_nt_rewrite":
            d = [x for x in pr["meta"]["defs"] if x["name"] == key][0]
            st["body"] = BR.subst_var(st["body"], d["lhs"], d["rhs"])
        else:
            h = [x for x in pr["meta"]["hyps"] if x["name"] == key][0]
            st["body"] = BR.subst_var(st["body"], h["var"],
                                      "%d * w_%s" % (pr["meta"]["m"] ** h["val"], h["var"]))
            st["expanded"].append(h["var"])
        st["k"] += 1
    return prog
