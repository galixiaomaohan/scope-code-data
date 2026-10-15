# -*- coding: utf-8 -*-
# bh_families.py -- bridge-hard six-family problem generator (gen14, change-58 curriculum-upgrade batch)
#
# Six families = three classes x 2 implementation mechanisms each:
#   ind_polysum  semantic-degree gate (both incremental and closed-form steps dressed as "same value,
#                different form") + perfect-square gate (a real computation: polynomial square root)
#   ind_geo      degeneracy gate (c1=0 determined by the fixed point c/(1-r); declared form dressed as
#                "same value, different form")
#   ineq_chain   state-dependent block chain + decoy search (residual-driven candidates, variable length)
#   ineq_tail    terminal semantic-class gate (zero / perfect square / generic positive; class-exclusion assertion)
#   nt_closure   definition closure + hypothesis-subset choice (decoy hypotheses; step-type state dispatch)
#   nt_val       m-adic valuation gate (modulus written in non-canonical form; valuation must really be computed)
#
# Obligation split (change-49): statement text -> arguments = the bridge; **class** = the model;
# computation = operators + certificates; Lean = final review.
# Each problem's chain is generated along **the actually executed path**; every step's state carries the
# device-computed intermediate objects (residual / closed form / body) and the candidate set.
import hashlib
import random
import re
from fractions import Fraction

import bh_bridge as BR
import bh_check as CK
import bh_ops as OPS
import bh_poly as P
import bh_tpl as TPL

TPLPICK = TPL.pick

CLASSES = ("ind", "ineq", "nt")
FAM_OF = {"ind": ["ind_polysum", "ind_geo"], "ineq": ["ineq_chain", "ineq_tail"],
          "nt": ["nt_closure", "nt_val"]}
CLASS_ZH = {"ind": "\u5f52\u7eb3\u9012\u63a8", "ineq": "\u4e0d\u7b49\u5f0f\u653e\u7f29",
            "nt": "\u6574\u9664\u95ed\u5305"}
VARS2 = ["x", "y"]
VARS3 = ["x", "y", "z"]


def seeded(master, cls, family, seq):
    return int(hashlib.md5(("bh|%d|%s|%s|%d" % (master, cls, family, seq)).encode("utf-8")
                           ).hexdigest()[:8], 16)


def _rng(master, cls, family, seq):
    return random.Random(seeded(master, cls, family, seq))


# ================================================================ rendering helpers
def render_poly_int(p, var="n"):
    items = sorted([(m, c) for m, c in p.items() if c != 0], key=lambda kv: (-kv[0][0], kv[0]))
    parts = []
    for m, c in items:
        c = Fraction(c)
        e = m[0]
        if e == 0:
            parts.append(("+" if c > 0 else "-", str(int(abs(c)))))
            continue
        body = var if e == 1 else ("%s ^ %d" % (var, e))
        parts.append(("+" if c > 0 else "-", body if abs(c) == 1 else
                      "%d * %s" % (int(abs(c)), body)))
    out = ""
    for i, (sg, b) in enumerate(parts):
        out = (("- " if sg == "-" else "") + b) if i == 0 else \
            out + (" - " if sg == "-" else " + ") + b
    return out or "0"


def horner_text(coefs, var="(n : \u2124)"):
    out = str(int(coefs[-1]))
    for c in reversed(coefs[:-1]):
        out = "%d + %s * (%s)" % (int(c), var, out)
    return out


def _canon_txt(s):
    return re.sub(r"\s+", "", s or "")


def poly_root(p):
    ok, root = CK._poly_sqrt(p)
    return root if ok else None


# ================================================================ IND-1: ind_polysum
def gen_ind_polysum(rng):
    dP = rng.choice([1, 2, 2, 3, 4, 4])
    square = rng.random() < 0.90 and dP % 2 == 0
    if square:
        m_ = dP // 2
        rc = [rng.randint(-3, 3) for _ in range(m_ + 1)]
        if all(c == 0 for c in rc):
            rc[0] = 1
        if rc[-1] == 0:
            rc[-1] = rng.choice([1, 2])
        root = {tuple([i]): Fraction(c) for i, c in enumerate(rc)}
        Pc = P.pmul(root, root)
    else:
        square = False
        Pc = {}
        for i in range(dP + 1):
            c = rng.randint(-6, 6)
            if i == dP and c == 0:
                c = rng.choice([1, 2, 3])
            if c:
                Pc[(i,)] = Fraction(c)
    dP = max(m[0] for m in Pc)
    S = OPS.shift_sub(Pc)
    dS = max([m[0] for m in S] or [0])
    if dS != dP - 1 or dP > 4:
        return None
    if poly_root(Pc) is not None:
        square = True
    return {"family": "ind_polysum", "P": Pc, "S": S, "deg": dS,
            "v0": int(Pc.get((0,), Fraction(0))), "square": square}


def build_ind_polysum(name, g, rng):
    Pc, S, deg, v0 = g["P"], g["S"], g["deg"], g["v0"]
    stpl = rng.choice(["plain", "diff", "scaled"])
    if stpl == "plain":
        step = "f (k + 1) = f k + (%s)" % render_poly_int(S, "k")
    elif stpl == "diff":
        A = {}
        for i in range(deg + 2):
            c = rng.randint(-5, 5)
            if c:
                A[(i,)] = Fraction(c)
        B = P.psub(A, S)
        step = ("f (k + 1) = f k + ((%s) - (%s))"
                % (render_poly_int(A, "k"), render_poly_int(B, "k")))
    else:
        step = "2 * f (k + 1) = 2 * f k + (%s)" % render_poly_int(P.pscale(S, 2), "k")
    coefs = [Pc.get((i,), Fraction(0)) for i in range(deg + 2)]
    ctpl = rng.choice(["plain", "horner", "noise"])
    if ctpl == "plain":
        concl = "f n = %s" % render_poly_int(Pc, "n")
    elif ctpl == "horner":
        concl = "f n = %s" % horner_text(coefs)
    else:
        q = rng.randint(2, 9)
        concl = "f n = %s + (%d ^ 2 - %d * %d)" % (horner_text(coefs), q, q, q)
    stmt = ("theorem %s (f : \u2115 \u2192 \u2124) (h0 : f 0 = %d) (hstep : \u2200 k, %s) :\n"
            "    \u2200 n : \u2115, %s" % (name, v0, step, concl))
    try:
        br = BR.bridge_ind_sum(stmt)
    except Exception as e:                                     # noqa: BLE001
        return None, "bridge:%s" % type(e).__name__
    if br["v0"] != v0 or br["deg"] != deg or P.dump(br["S"]) != P.dump(S):
        return None, "bridge_mismatch"
    Ttxt, Tpoly = BR.bridge_ind_declared(stmt)
    if P.dump(Tpoly) != P.dump(Pc):
        return None, "decl_mismatch"
    r1 = OPS.call("bh_ind_sum_d%d" % deg, {"v0": str(v0), "deg": deg, "S": P.dump(S)})
    if not (r1.ok and r1.cert_ok):
        return None, "solve_%s" % (r1.err or "cert")
    root = poly_root(Pc)
    rdeg = max([m[0] for m in root] or [0]) if root else None
    if root is not None and rdeg in (1, 2):
        rc = [root.get((i,), Fraction(0)) for i in range(rdeg + 1)]
        cls3 = "square2" if rdeg == 1 else "square4"
        cname = {"square2": "bh_ind_close_sq2", "square4": "bh_ind_close_sq4"}[cls3]
        cargs = {"P": P.dump(Pc), "T": P.dump(Tpoly), "cls": cls3,
                 "sq": [str(int(c)) for c in rc]}
    else:
        rdeg = None
        cls3 = "poly"
        cname = "bh_ind_close_poly"
        cargs = {"P": P.dump(Pc), "T": P.dump(Tpoly), "cls": "poly"}
    r2 = OPS.call(cname, cargs)
    if not (r2.ok and r2.cert_ok):
        return None, "close_%s" % (r2.err or "cert")
    closed = P.render_poly(Pc, ["n"])
    canon_step = "f (k + 1) = f k + (%s)" % render_poly_int(S, "k")
    hs_step = None if stpl == "plain" else (canon_step,
                                            "rw_ring" if stpl == "diff" else "linarith")
    solve_tac = OPS.ind_solve_tactic("poly", closed, hs_step)
    close_lines = ["rw [hP]"] + ([] if _canon_txt(Ttxt) == _canon_txt(closed) else ["ring"])
    proof = solve_tac.split("\n") + close_lines
    bind = "(f : \u2115 \u2192 \u2124) (h0 : f 0 = %d) (hstep : \u2200 k, %s)" % (v0, step)
    g1 = "%s \u22a2 \u2200 n : \u2115, %s" % (bind, concl)
    s1 = "%s (n : \u2115) (hP : f n = %s) \u22a2 \u2200 n : \u2115, %s" % (bind, closed, concl)
    s2 = "%s (n : \u2115) (hP : f n = %s) \u22a2 %s = %s" % (bind, closed, closed, Ttxt)
    chain = [{"op": "bh_ind_sum_d%d" % deg, "args": {"v0": v0, "deg": deg, "S": P.dump(S)},
              "sym": "sum_deg%d" % deg, "tactic": solve_tac, "value": r1.value,
              "cert": r1.cert, "cert_ok": True, "cert_detail": r1.cert_detail,
              "resid": r1.resid},
             {"op": cname, "args": cargs, "sym": "close", "tactic": "rw [hP]\nring",
              "value": r2.value, "cert": r2.cert, "cert_ok": True,
              "cert_detail": r2.cert_detail, "resid": r2.resid}]
    sqargs = {}
    for c3, cn3 in (("square2", "bh_ind_close_sq2"), ("square4", "bh_ind_close_sq4")):
        need = {"square2": 2, "square4": 3}[c3]
        sqargs[cn3] = {"P": P.dump(Pc), "T": P.dump(Tpoly), "cls": c3,
                       "sq": [str(int(root.get((i,), Fraction(0))))
                              for i in range(need)] if root else ["0"]}
    cands = [[{"op": "bh_ind_sum_d%d" % d, "args": {"v0": v0, "deg": d, "S": P.dump(S)},
               "viable": d == deg} for d in range(0, 4)],
             [{"op": "bh_ind_close_poly", "args": {"P": P.dump(Pc), "T": P.dump(Tpoly),
                                                   "cls": "poly"},
               "viable": cls3 == "poly"},
              {"op": "bh_ind_close_sq2", "args": sqargs["bh_ind_close_sq2"],
               "viable": cls3 == "square2"},
              {"op": "bh_ind_close_sq4", "args": sqargs["bh_ind_close_sq4"],
               "viable": cls3 == "square4"}]]
    return {"stmt": stmt, "g1": g1, "states": [g1, s1, s2], "chain": chain, "proof": proof,
            "closed": closed, "cands": cands,
            "choice": [deg, {"poly": 0, "square2": 1, "square4": 2}[cls3]],
            "meta": {"deg": deg, "cls3": cls3, "square": root is not None,
                     "step_tpl": stpl,
                     "concl_tpl": ctpl}}, None


# ================================================================ IND-2: ind_geo
def gen_ind_geo(rng):
    r = rng.choice([2, 3, 4, -2, -3])
    q = rng.randint(-4, 4)
    c = (1 - r) * q
    if c == 0:
        c = (1 - r)
    c0 = Fraction(c, 1 - r)
    degen = rng.random() < 0.4 or int(c0) < 0
    if degen:
        v0 = int(c0)
    else:
        v0 = int(c0) + rng.choice([1, 2, -1, -2, 3, -3, 5, -5])
    c1 = Fraction(v0) - c0
    return {"family": "ind_geo", "r": r, "c": c, "v0": v0, "c1": c1, "c0": c0,
            "cls": "const" if c1 == 0 else "pow"}


def build_ind_geo(name, g, rng):
    r, c, v0, c1, c0, cls = g["r"], g["c"], g["v0"], g["c1"], g["c0"], g["cls"]
    stpl = rng.choice(["plain", "moved", "scaled"])
    if stpl == "plain":
        step = "f (k + 1) = %d * f k %s %d" % (r, "+" if c > 0 else "-", abs(c))
    elif stpl == "moved":
        step = "f (k + 1) + f k = %d * f k %s %d" % (r + 1, "+" if c > 0 else "-", abs(c))
    else:
        step = "2 * f (k + 1) = %d * f k %s %d" % (2 * r, "+" if 2 * c > 0 else "-",
                                                   abs(2 * c))
    if cls == "const":
        s0 = int(c0)
        tpl = TPLPICK(cls, s0, r, rng)
        concl = "f n = %s" % TPL.render(tpl, 0, s0, r)
    else:
        tpl = TPLPICK(cls, int(c0), r, rng)
        concl = "f n = %s" % TPL.render(tpl, int(c1), int(c0), r)
    stmt = ("theorem %s (f : \u2115 \u2192 \u2124) (h0 : f 0 = %d) (hstep : \u2200 k, %s) :\n"
            "    \u2200 n : \u2115, %s" % (name, v0, step, concl))
    try:
        br = BR.bridge_ind_geo(stmt)
    except Exception as e:                                     # noqa: BLE001
        return None, "bridge:%s" % type(e).__name__
    if not all(br[k] == g[k] for k in ("v0", "r", "c", "c1", "c0", "cls")):
        return None, "bridge_mismatch"
    r1 = OPS.call("bh_ind_geo", {"v0": v0, "r": r, "c": c})
    if not (r1.ok and r1.cert_ok):
        return None, "geo_%s" % (r1.err or "cert")
    closed = OPS.geo_closed_text(c1, c0, r, "plain")
    Ttxt = concl.split("=", 1)[1].strip()
    if cls == "const":
        cname = "bh_ind_close_const"
        cargs = {"v0": v0, "r": r, "c": c, "cls": "const", "tpl": tpl, "T_text": Ttxt}
    else:
        cname = "bh_ind_close_pow"
        cargs = {"v0": v0, "r": r, "c": c, "cls": "pow", "tpl": tpl, "T_text": Ttxt}
    r2 = OPS.call(cname, cargs)
    if not (r2.ok and r2.cert_ok):
        return None, "close_%s" % (r2.err or "cert")
    kind = "geo_const" if cls == "const" else "geo_pow"
    canon_step = "f (k + 1) = %d * f k %s %d" % (r, "+" if c > 0 else "-", abs(c))
    hs_step = None if stpl == "plain" else (canon_step, "linarith")
    solve_tac = OPS.ind_solve_tactic(kind, closed, hs_step)
    close_lines = ["rw [hP]"] + ([] if _canon_txt(Ttxt) == _canon_txt(closed) else ["ring"])
    proof = solve_tac.split("\n") + close_lines
    bind = "(f : \u2115 \u2192 \u2124) (h0 : f 0 = %d) (hstep : \u2200 k, %s)" % (v0, step)
    g1 = "%s \u22a2 \u2200 n : \u2115, %s" % (bind, concl)
    s1 = ("%s (n : \u2115) (hP : f n = %s) (hc1 : c1 = %s) \u22a2 \u2200 n : \u2115, %s"
          % (bind, closed, int(c1), concl))
    s2 = "%s (n : \u2115) (hP : f n = %s) \u22a2 %s = %s" % (bind, closed, closed, Ttxt)
    chain = [{"op": "bh_ind_geo", "args": {"v0": v0, "r": r, "c": c}, "sym": "geo_solve",
              "tactic": solve_tac, "value": r1.value, "cert": r1.cert, "cert_ok": True,
              "cert_detail": r1.cert_detail, "resid": r1.resid},
             {"op": cname, "args": cargs, "sym": "close", "tactic": "rw [hP]\nring",
              "value": r2.value, "cert": r2.cert, "cert_ok": True,
              "cert_detail": r2.cert_detail, "resid": r2.resid}]
    cands = [[{"op": "bh_ind_geo", "args": {"v0": v0, "r": r, "c": c}, "viable": True}],
             [{"op": "bh_ind_close_const",
               "args": cargs if cls == "const" else {"v0": v0, "r": r, "c": c,
                                                     "cls": "const", "tpl": tpl,
                                                     "T_text": Ttxt},
               "viable": cls == "const"},
              {"op": "bh_ind_close_pow",
               "args": cargs if cls == "pow" else {"v0": v0, "r": r, "c": c, "cls": "pow",
                                                   "tpl": "pow_plain", "T_text": Ttxt},
               "viable": cls == "pow"}]]
    return {"stmt": stmt, "g1": g1, "states": [g1, s1, s2], "chain": chain, "proof": proof,
            "closed": closed, "cands": cands, "choice": [0, 0 if cls == "const" else 1],
            "meta": {"r": r, "c": c, "c1": str(c1), "c0": str(c0), "cls": cls,
                     "step_tpl": stpl, "concl_tpl": tpj(tpl)}}, None


def tpj(t):
    return t


# ================================================================ INEQ
def gen_ineq(rng, cross_ok=True, jrange=(2, 3)):
    nv = rng.choice([2, 3])
    vars_ = (VARS2 if nv == 2 else VARS3)[:]
    J = min(rng.choice(list(jrange)), nv)
    idx = list(range(nv))
    rng.shuffle(idx)
    parts = [idx[j::J] for j in range(J)]
    blocks = []
    for part in parts:
        vs = [vars_[i] for i in part]
        if len(vs) >= 2 and cross_ok:
            signs = [rng.choice([1, -1]) for _ in vs]
            if all(s == signs[0] for s in signs):
                signs[0] = -signs[0]
            blocks.append({"coefs": {v: s for v, s in zip(vs, signs)},
                           "shift": rng.randint(0, 4), "mult": rng.choice([1, 1, 2]),
                           "vars": vs})
        else:
            s = rng.choice([1, -1, 1])
            blocks.append({"coefs": {vs[0]: s}, "shift": rng.randint(0, 5),
                           "mult": rng.choice([1, 1, 1, 2, 3]), "vars": vs})
    cls = rng.choice(["zero", "square", "square", "other", "other"])
    if cls == "zero":
        rho, sq = Fraction(0), None
    elif cls == "square":
        s = rng.randint(2, 7)
        rho, sq = Fraction(s * s), s
    else:
        rho, sq = None, None
        for cand in rng.sample(range(1, 40), 20):
            if CK._isqrt(cand) is None:
                rho = Fraction(cand)
                break
        if rho is None:
            rho = Fraction(2)
    return {"nv": nv, "vars": vars_, "blocks": blocks, "rho": rho, "cls": cls, "sq": sq}


def _ineq_dom(g):
    nv, vars_ = g["nv"], g["vars"]
    dom = {}
    for b in g["blocks"]:
        ell = OPS._ell_poly(vars_, b)
        dom = P.padd(dom, P.pscale(P.pmul(ell, ell), b["mult"]))
    return P.padd(dom, {(0,) * nv: g["rho"]})


def _block_candidates(g, b, R, rng):
    op = "bh_ineq_cross" if len(b["coefs"]) >= 2 else "bh_ineq_block"
    out = [{"op": op, "args": {"ell": {"coefs": dict(b["coefs"]), "shift": b["shift"]},
                               "mult": b["mult"]}, "viable": True},
           {"op": op, "args": {"ell": {"coefs": dict(b["coefs"]), "shift": b["shift"] + 1},
                               "mult": b["mult"]}, "viable": False}]
    if len(b["coefs"]) == 1:
        v = list(b["coefs"])[0]
        out.append({"op": op, "args": {"ell": {"coefs": {v: b["coefs"][v] * 2},
                                               "shift": b["shift"]}, "mult": b["mult"]},
                    "viable": False})
    else:
        vs = list(b["coefs"])
        out.append({"op": op, "args": {"ell": {"coefs": dict(b["coefs"]),
                                               "shift": b["shift"] + 2},
                                       "mult": b["mult"] + 1}, "viable": False})
    order = list(range(len(out)))
    rng.shuffle(order)
    res = [out[i] for i in order]
    for cc in res:
        args = dict(cc["args"])
        args.update({"vars": g["vars"], "R_in": P.dump(R), "hint_idx": 1})
        r = OPS.call(cc["op"], args)
        cc["_ok"] = bool(r.ok and r.cert_ok)
        cc["_err"] = r.err if not cc["_ok"] else None
    if sum(1 for cc in res if cc["_ok"]) != 1:
        return None
    return res


def build_ineq(name, g, rng, famname):
    vars_, nv = g["vars"], g["nv"]
    dom = _ineq_dom(g)
    D = rng.choice([0, 0, rng.randint(-9, 9)])
    LHS = P.padd(dom, {(0,) * nv: Fraction(D)})
    lhs_txt = P.render_poly(LHS, vars_)
    if rng.random() < 0.5:
        concl = "%s \u2265 %d" % (lhs_txt, D)
    else:
        concl = "%d \u2264 %s" % (D, lhs_txt)
    stmt = "theorem %s (%s : \u211a) :\n    %s" % (name, " ".join(vars_), concl)
    try:
        br = BR.bridge_ineq(stmt, vars_)
    except Exception as e:                                     # noqa: BLE001
        return None, "bridge:%s" % type(e).__name__
    if P.dump(br["dom"]) != P.dump(dom):
        return None, "bridge_mismatch"
    bind = "(%s : \u211a)" % " ".join(vars_)
    R = dict(dom)
    chain, proof, states, cands = [], [], [], []
    hints = []
    states.append("%s \u22a2 %s  \u2016 residual: %s" % (bind, concl, P.render_poly(R, vars_)))
    for k, b in enumerate(g["blocks"]):
        R_pre = dict(R)
        cc_list = _block_candidates(g, b, R_pre, rng)
        if cc_list is None:
            return None, "cand_degenerate"
        chosen = [i for i, cc in enumerate(cc_list) if cc["viable"]][0]
        args = dict(cc_list[chosen]["args"])
        args.update({"vars": vars_, "R_in": P.dump(R_pre), "hint_idx": k + 1})
        op = cc_list[chosen]["op"]
        r = OPS.call(op, args)
        if not (r.ok and r.cert_ok):
            return None, "block%d_%s" % (k, r.err or "cert")
        chain.append({"op": op, "args": args, "sym": "block%d" % k, "tactic": r.tactic,
                      "value": r.value, "cert": r.cert, "cert_ok": True,
                      "cert_detail": r.cert_detail, "resid": r.resid})
        proof.append(r.tactic)
        hints.append("h%d" % (k + 1))
        R = P.load(r.value["R_out"])
        ctx = "%s %s" % (bind, " ".join("(h%d : (0:\u211a) \u2264 %s)"
                                        % (i + 1, chain[i]["value"]["hint"])
                                        for i in range(len(chain))))
        states.append("%s \u22a2 %s  \u2016 residual: %s"
                      % (ctx, concl, P.render_poly(R, vars_)))
        cands.append([{"op": cc["op"],
                       "args": dict(cc["args"], vars=vars_, R_in=P.dump(R_pre)),
                       "viable": cc["viable"]} for cc in cc_list])
    tailop = {"zero": "bh_ineq_tail_zero", "square": "bh_ineq_tail_square",
              "other": "bh_ineq_tail_other"}[g["cls"]]
    hints_poly = []
    for k, b in enumerate(g["blocks"]):
        ell = OPS._ell_poly(vars_, b)
        hints_poly.append(P.dump(P.pscale(P.pmul(ell, ell), b["mult"])))
    targs = {"vars": vars_, "rho": str(R.get((0,) * nv, Fraction(0))), "hints": hints,
             "goal_dom": P.dump(dom), "hints_poly": hints_poly}
    if g["cls"] == "square":
        targs["sq"] = g["sq"]
    rt = OPS.call(tailop, targs)
    if not (rt.ok and rt.cert_ok):
        return None, "tail_%s" % (rt.err or "cert")
    chain.append({"op": tailop, "args": targs, "sym": "tail", "tactic": rt.tactic,
                  "value": rt.value, "cert": rt.cert, "cert_ok": True,
                  "cert_detail": rt.cert_detail, "resid": rt.resid})
    proof.extend(rt.tactic.split("\n"))
    tcands = []
    for cn, key in (("bh_ineq_tail_zero", "zero"), ("bh_ineq_tail_square", "square"),
                    ("bh_ineq_tail_other", "other")):
        a = {"vars": vars_, "rho": str(R.get((0,) * nv, Fraction(0))), "hints": hints,
             "goal_dom": P.dump(dom), "hints_poly": hints_poly}
        if key == "square":
            a["sq"] = g["sq"] if g["sq"] else 2
        tcands.append({"op": cn, "args": a, "viable": key == g["cls"]})
    cands.append(tcands)
    states.append("%s \u22a2 %s  \u2016 residual: %s"
                  % (" ".join("(h%d : (0:\u211a) \u2264 %s)" % (i + 1, chain[i]["value"]["hint"])
                              for i in range(len(chain) - 1)) + " " + bind,
                     concl, P.render_poly(R, vars_)))
    g1 = "%s \u22a2 %s" % (bind, concl)
    return {"stmt": stmt, "g1": g1, "states": states, "chain": chain, "proof": proof,
            "cands": cands, "choice": [0] * (len(g["blocks"]) + 1),
            "meta": {"J": len(g["blocks"]), "cls": g["cls"], "nv": nv, "fam": famname,
                     "D": D, "vars": vars_,
                     "blocks": [{k2: v2 for k2, v2 in b.items()} for b in g["blocks"]],
                     "cls_sq": g["sq"], "dom_dump": P.dump(dom),
                     "rho": str(g["rho"])}}, None


# ================================================================ NT
def _mod_text(m, val, style):
    if style == 0:
        return str(m ** val)
    if style == 1:
        return " * ".join([str(m)] * val)
    if style == 2 and val == 2:
        return "%d ^ 2" % m
    if style == 3 and val == 1:
        return "1 * %d" % m
    return " * ".join([str(m)] * val)


def gen_nt(rng, fam):
    m = rng.choice([2, 3, 4, 5])
    if fam == "nt_val":
        k, M = 3, m ** 3
        vals = [1, 2]
        rng.shuffle(vals)
        hyps = [{"name": "ha", "var": "a", "val": vals[0], "style": rng.randint(0, 3)},
                {"name": "hb", "var": "b", "val": vals[1], "style": rng.randint(0, 3)}]
        extra = []
        if rng.random() < 0.6:
            v2 = [1, 2]
            rng.shuffle(v2)
            extra = [{"name": "hc", "var": "c", "val": v2[0], "style": rng.randint(0, 3)},
                     {"name": "hd", "var": "d", "val": v2[1], "style": rng.randint(0, 3)}]
        hyps += extra
        hyps.append({"name": "he", "var": "e", "val": 1, "style": rng.randint(0, 3)})
        terms = [{"coef": 1, "vars": ["a", "b"]}]
        if rng.random() < 0.4:
            terms.append({"coef": rng.choice([2, 3]), "vars": ["a", "b"]})
        plan = [("hyp", "ha"), ("hyp", "hb")]
        return {"family": fam, "m": m, "k": k, "M": M, "hyps": hyps, "terms": terms,
                "K": M * rng.choice([1, 1, 2, 3]), "defs": [], "plan": plan,
                "body_vars": ["a", "b"], "used": ["a", "b"]}
    k = rng.choice([2, 3])
    M = m ** k
    used = rng.sample(["a", "b", "c"], k)
    hyps = [{"name": "h" + v, "var": v, "val": 1, "style": 0} for v in used]
    for v in [v for v in ["a", "b", "c"] if v not in used]:
        hyps.append({"name": "h" + v, "var": v, "val": 1, "style": 0})
    hyps.append({"name": "he", "var": "e", "val": 1, "style": 0})
    defs = [{"name": "hu", "lhs": "u", "rhs_vars": used[:2]}]
    plan = []
    if k == 3:
        defs.append({"name": "hv", "lhs": "v", "rhs": "u", "extra": used[2]})
        plan = [("def", "hv"), ("def", "hu")] + [("hyp", "h" + v) for v in used]
        terms = [{"coef": 1, "use": "v"}]
        if rng.random() < 0.4:
            terms.append({"coef": 1, "use": "u"})
    else:
        plan = [("def", "hu")] + [("hyp", "h" + v) for v in used]
        terms = [{"coef": 1, "use": "u"}]
    return {"family": fam, "m": m, "k": k, "M": M, "hyps": hyps, "terms": terms,
            "K": M * rng.choice([1, 1, 2, 3]), "defs": defs, "plan": plan,
            "body_vars": used, "used": used}


def _def_rhs_text(d, g):
    if "rhs_vars" in d:
        return " * ".join(d["rhs_vars"])
    return "%s * %s" % (d["rhs"], d["extra"])


def _def_expand_vars(lhs, defs, g):
    dmap = {d["lhs"]: d for d in defs}
    if lhs not in dmap:
        return [lhs]
    d = dmap[lhs]
    if "rhs_vars" in d:
        out = []
        for v in d["rhs_vars"]:
            out.extend(_def_expand_vars(v, defs, g))
        return out
    return _def_expand_vars(d["rhs"], defs, g) + [d["extra"]]


def build_nt(name, g, rng):
    m, M, k = g["m"], g["M"], g["k"]
    hyps, defs = g["hyps"], g["defs"]
    hyp_txt = " ".join("(%s : %s \u2223 %s)"
                       % (h["name"], _mod_text(m, h["val"], h["style"]), h["var"])
                       for h in hyps)
    def_txt = "".join(" (%s : %s = %s)" % (d["name"], d["lhs"], _def_rhs_text(d, g))
                      for d in defs)
    def _term_text(t):
        core = t["use"] if "use" in t else " * ".join(t["vars"])
        return core if t["coef"] == 1 else "%s * %s" % (t["coef"], core)

    body = " + ".join([_term_text(t) for t in g["terms"]] + [str(g["K"])])
    binders = sorted(set([h["var"] for h in hyps] + [d["lhs"] for d in defs]
                         + g["body_vars"]))
    stmt = ("theorem %s (%s : \u2115)%s %s :\n    %d \u2223 %s"
            % (name, " ".join(binders), def_txt, hyp_txt, M, body))
    try:
        br = BR.bridge_nt(stmt)
    except Exception as e:                                     # noqa: BLE001
        return None, "bridge:%s" % type(e).__name__
    if (br["M"] != M or br["body_text"] != re.sub(r"\s+", " ", body).strip()
            or len(br["hyps"]) != len(hyps) or len(br["defs"]) != len(defs)):
        return None, "bridge_mismatch"
    val = {h["var"]: h["val"] for h in hyps}
    hyp_by_var = {h["var"]: h for h in hyps}
    defmap = {d["lhs"]: d for d in defs}
    term_list = []
    for t in g["terms"]:
        vs = _def_expand_vars(t["use"], defs, g) if "use" in t else list(t["vars"])
        term_list.append({"coef": t["coef"], "k": sum(val[v] for v in vs), "vars": vs})
    if any(t["k"] < k for t in term_list):
        return None, "insufficient_valuation"
    if g["K"] % M != 0:
        return None, "K_not_divisible"
    # ---- execute along plan (chain = the reference path)
    bind_all = "(%s : \u2115)%s %s" % (" ".join(binders), def_txt, hyp_txt)
    bind_ctx = "(%s : \u2115)" % " ".join(binders)
    cur = re.sub(r"\s+", " ", body).strip()
    chain, proof, states, cands = [], [], [], []
    states.append("%s \u22a2 %d \u2223 %s" % (bind_all, M, cur))
    expanded = []

    def _ctx(body_now):
        return "%s \u22a2 %d \u2223 %s" % (bind_all, M, body_now)

    for si, (kindk, key) in enumerate(g["plan"]):
        # Candidate set (state-driven): definitions currently present + occurring hypothesis variables + decoys + close
        cs = []
        occ_defs = [d for d in defs
                    if re.search(r"(?<![A-Za-z0-9_'])%s(?![A-Za-z0-9_'])" % re.escape(d["lhs"]),
                                 cur)]
        for d in occ_defs:
            cs.append({"op": "bh_nt_rewrite", "args": {"hyp": d["name"], "lhs": d["lhs"]},
                       "viable": kindk == "def" and key == d["name"]})
        occ_vars = BR.body_vars(cur, [h["var"] for h in hyps])
        for h in hyps:
            if h["var"] in occ_vars:
                opn = ("bh_nt_expand_v%d" % h["val"] if g["family"] == "nt_val"
                       else "bh_nt_expand")
                cs.append({"op": opn, "args": {"m": m, "hyp": h["name"], "var": h["var"],
                                               "modulus": m ** h["val"]},
                           "viable": kindk == "hyp" and key == h["name"]})
        dv = [h for h in hyps if h["var"] not in occ_vars]
        if dv:
            h = dv[rng.randrange(len(dv))]
            cs.append({"op": "bh_nt_expand", "args": {"m": m, "hyp": h["name"],
                                                      "var": h["var"]}, "viable": False})
        if not occ_defs and not occ_vars:
            cs.append({"op": "bh_nt_close", "args": {"m": m, "M": M},
                       "viable": kindk == "close"})
        cands.append(cs)
        # Execute this step
        if kindk == "def":
            d = [x for x in defs if x["name"] == key][0]
            a = {"hyp": d["name"], "lhs": d["lhs"], "text": _ctx(cur), "body": cur}
            r = OPS.call("bh_nt_rewrite", a)
            if not (r.ok and r.cert_ok):
                return None, "rewrite_%s" % (r.err or "cert")
            chain.append({"op": "bh_nt_rewrite", "args": a, "sym": "def_%s" % d["lhs"],
                          "tactic": r.tactic, "value": r.value, "cert": r.cert,
                          "cert_ok": True, "cert_detail": r.cert_detail, "resid": r.resid})
            proof.append(r.tactic)
            cur = BR.subst_var(cur, d["lhs"], _def_rhs_text(d, g))
        else:
            h = [x for x in hyps if x["name"] == key][0]
            opn = ("bh_nt_expand_v%d" % h["val"] if g["family"] == "nt_val"
                   else "bh_nt_expand")
            a = {"m": m, "hyp": h["name"], "var": h["var"], "text": _ctx(cur), "body": cur,
                 "modulus": m ** h["val"]}
            r = OPS.call(opn, a)
            if not (r.ok and r.cert_ok):
                return None, "expand_%s_%s" % (key, r.err or "cert")
            chain.append({"op": opn, "args": a, "sym": "exp_%s" % h["var"],
                          "tactic": r.tactic, "value": r.value, "cert": r.cert,
                          "cert_ok": True, "cert_detail": r.cert_detail, "resid": r.resid})
            proof.append(r.tactic)
            cur = BR.subst_var(cur, h["var"], "%d * w_%s" % (m ** h["val"], h["var"]))
            expanded.append(h["var"])
        states.append("%s \u22a2 %d \u2223 %s" % (bind_all, M, cur))
    params = sorted(set(["w_%s" % v for v in expanded]))
    wit = []
    for t in term_list:
        c = Fraction(t["coef"]) * Fraction(m) ** t["k"] / M
        wit.append({"coef": str(int(c)), "monomial": ["w_%s" % v for v in t["vars"]]})
    if g["K"] // M != 0:
        wit.append({"coef": str(int(g["K"] // M)), "monomial": []})
    cargs = {"m": m, "M": M, "params": params,
             "terms": [{"coef": str(t["coef"]), "k": t["k"],
                        "wvars": ["w_%s" % v for v in t["vars"]]} for t in term_list],
             "const": g["K"], "witness": wit}
    # Close-step candidates + execution
    dd = [h for h in hyps if h["var"] not in expanded]
    cs = [{"op": "bh_nt_close", "args": cargs, "viable": True}]
    if dd:
        h = dd[rng.randrange(len(dd))]
        cs.append({"op": "bh_nt_expand", "args": {"m": m, "hyp": h["name"], "var": h["var"],
                                                  "modulus": m ** h["val"]}, "viable": False})
    else:
        cs.append({"op": "bh_nt_close", "args": cargs, "viable": False})
    cands.append(cs)
    rc = OPS.call("bh_nt_close", cargs)
    if not (rc.ok and rc.cert_ok):
        return None, "close_%s" % (rc.err or "cert")
    chain.append({"op": "bh_nt_close", "args": cargs, "sym": "close", "tactic": rc.tactic,
                  "value": rc.value, "cert": rc.cert, "cert_ok": True,
                  "cert_detail": rc.cert_detail, "resid": rc.resid})
    proof.append(rc.tactic)
    states.append("%s \u22a2 \u2203 k, %d * k = %s" % (bind_all, M, cur))
    choice = []
    for cs in cands:
        vv = [i for i, c in enumerate(cs) if c["viable"]]
        choice.append(vv[0] if vv else 0)
    g1 = "%s \u22a2 %d \u2223 %s" % (bind_all, M, re.sub(r"\s+", " ", body).strip())
    return {"stmt": stmt, "g1": g1, "states": states, "chain": chain, "proof": proof,
            "cands": cands, "choice": choice,
            "meta": {"m": m, "k": k, "M": M, "used": g["used"], "cls": g["family"],
                     "binders": " ".join(binders), "hyp_txt": hyp_txt, "def_txt": def_txt,
                     "body0": re.sub(r"\s+", " ", body).strip(), "K": g["K"],
                     "hyps": [{"name": h["name"], "var": h["var"], "val": h["val"],
                               "style": h["style"]} for h in hyps],
                     "defs": [{"name": d2["name"], "lhs": d2["lhs"],
                               "rhs": _def_rhs_text(d2, g)} for d2 in defs],
                     "terms": [{"coef": t["coef"], "k": t["k"], "vars": list(t["vars"])}
                               for t in term_list],
                     "plan": [[a2, b2] for (a2, b2) in g["plan"]]}}, None


# ================================================================ unified entry point
def make_problem(cls, family, seq, master):
    rng = _rng(master, cls, family, seq)
    name = "bh_%s_%s_%06d" % (cls, family.split("_", 1)[1], seq)
    pr = None
    for _ in range(14):
        if family in ("ineq_chain", "ineq_tail"):
            g = gen_ineq(rng, cross_ok=(family == "ineq_chain"),
                         jrange=((2, 3) if family == "ineq_chain" else (1, 2)))
            pr, err = build_ineq(name, g, rng, family)
        elif family == "ind_polysum":
            g = gen_ind_polysum(rng)
            if g is None:
                continue
            pr, err = build_ind_polysum(name, g, rng)
        elif family == "ind_geo":
            g = gen_ind_geo(rng)
            pr, err = build_ind_geo(name, g, rng)
        else:
            g = gen_nt(rng, family)
            pr, err = build_nt(name, g, rng)
        if pr is not None:
            break
    if pr is None:
        return None, err
    pr.update({"name": name, "cls": cls, "family": family, "seq": seq,
               "seed": seeded(master, cls, family, seq)})
    return pr, None
