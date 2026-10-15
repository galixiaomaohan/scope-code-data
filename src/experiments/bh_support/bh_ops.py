# -*- coding: utf-8 -*-
# bh_ops.py -- bridge-hard batch certified operators (gen14, change-58 curriculum-upgrade batch)
#
# Each operator = (a) computation logic + (b) Lean tactic wrapper + (c) certificate (pure data).
# **No self-certification**: all re-verification lives in bh_check (register_checker structurally
# rejects functions from the same module).
#
# Key difference from gen12 (ccp_ops) = **the operator name carries the decision (class assertion)**:
#   every operator name maps to a class assertion that can be judged true or false (semantic degree /
#   perfect square / degeneracy c1=0 / terminal semantic class / m-adic valuation); the certificate
#   machinery judges that class **exclusively**, so a wrong operator name = a genuine failure
#   (CERT_REJECT). The device derives only arguments (args, derivable from the statement); the class is
#   chosen by the model -- this is the mechanism that makes "routing" load-bearing (change-49 obligation split).
import os
import sys
from fractions import Fraction

import bh_check as CK
import bh_poly as P

BANNED_OPS = ("math_evaluate_expr",)
PROOF_OPS = ("bh_ind_sum_d0", "bh_ind_sum_d1", "bh_ind_sum_d2", "bh_ind_sum_d3",
             "bh_ind_geo",
             "bh_ind_close_poly", "bh_ind_close_sq", "bh_ind_close_sq2",
             "bh_ind_close_sq4", "bh_ind_close_const", "bh_ind_close_pow",
             "bh_ineq_block", "bh_ineq_cross",
             "bh_ineq_tail_zero", "bh_ineq_tail_square", "bh_ineq_tail_other",
             "bh_nt_rewrite", "bh_nt_expand", "bh_nt_expand_v1", "bh_nt_expand_v2",
             "bh_nt_close")
UNLIM = Fraction(10 ** 9)


class OpResult(object):
    __slots__ = ("ok", "err", "value", "cert", "cert_ok", "cert_detail", "tactic", "resid")

    def __init__(self, ok, err=None, value=None, cert=None, cert_ok=None,
                 cert_detail="", tactic="", resid=None):
        self.ok, self.err, self.value = ok, err, value
        self.cert, self.cert_ok, self.cert_detail = cert, cert_ok, cert_detail
        self.tactic, self.resid = tactic, resid or [0.0]

    def to_dict(self):
        return {"ok": self.ok, "err": self.err, "value": self.value, "cert": self.cert,
                "cert_ok": self.cert_ok, "cert_detail": self.cert_detail,
                "tactic": self.tactic, "resid": self.resid}


def _finish(ok, err, value, cert, tactic, resid=None, detail=""):
    if not ok:
        return OpResult(False, err=err, value=value, tactic=tactic, resid=resid)
    cok, cr, cdet = CK.check(cert["kind"], cert["payload"])
    ind = CK.__name__ not in CK.independence(cert["kind"])
    msg = (cdet if not detail else detail + " | " + cdet) + " | independent=%s" % ind
    if not (cok and ind):
        return OpResult(False, err="CERT_REJECT", value=value, cert=cert, cert_ok=False,
                        cert_detail=msg, tactic=tactic, resid=cr)
    return OpResult(True, None, value, cert, True, msg, tactic=tactic, resid=cr)


# ================================================================ difference-equation solve (generator algorithm)
def shift_sub(p, var_idx=0):
    """P(n+1) - P(n): via binomial expansion (the generator route)."""
    nv = len(next(iter(p))) if p else 1
    out = {}
    for m, c in p.items():
        e = m[var_idx]
        if e == 0:
            continue
        for j in range(e):
            # C(e, j)*n^j  (coefficient)
            from math import comb
            mm = list(m)
            mm[var_idx] = j
            key = tuple(mm)
            out[key] = out.get(key, Fraction(0)) + c * comb(e, j)
    return {m: c for m, c in out.items() if c != 0}


def solve_difference(S, v0):
    """Solve P(n+1) - P(n) = S(n), P(0)=v0; P = sum p_i n^i (integer/rational coefficients)."""
    d = max([m[0] for m in S] or [0])
    p = {(0,): Fraction(v0)}
    # Unknowns p_1..p_{d+1}; coefficient matching: coeff of n^{i-1} in dP == S_{i-1}
    sol = {}
    for i in range(d + 1, 0, -1):
        from math import comb
        # d(n^i) = sum_{j<i} C(i,j) n^j
        tgt = sum((c for m, c in S.items() if m[0] == i - 1), Fraction(0))
        acc = Fraction(0)
        for j in range(i + 1, d + 2):
            if j in sol:
                acc += sol[j] * comb(j, i - 1)
        sol[i] = (tgt - acc) / i          # coefficient of n^{i-1} in d(n^i) = i
    for i, c in sol.items():
        if c != 0:
            m = [0] * 1
            m[0] = i
            p[tuple(m)] = p.get(tuple(m), Fraction(0)) + c
    return {m: c for m, c in p.items() if c != 0}


def _ind_sum(args, want_deg):
    """args {v0, S (univariate dump, variable order [k])}. Class assertion = S's **exact** degree == want_deg."""
    v0 = Fraction(args["v0"])
    S = P.load(args["S"])
    S = {m: c for m, c in S.items()}
    d = int(args.get("deg", want_deg))
    if d != want_deg:
        return OpResult(False, err="DEG_ARG_MISMATCH",
                        value={"declared": d, "op": want_deg}, tactic="")
    real = max([m[0] for m in S] or [0])
    if real != want_deg:
        return OpResult(False, err="DEG_CLASS_VIOLATION",
                        value={"real_deg": real, "op_deg": want_deg}, tactic="")
    Pc = solve_difference(S, v0)
    for m, c in Pc.items():
        if abs(c) > UNLIM:
            return OpResult(False, err="BOUND", tactic="")
    cert = {"kind": "bh_ind_sum_cert",
            "payload": {"deg": want_deg, "v0": str(v0), "S": P.dump(S), "P": P.dump(Pc)}}
    canon = P.render_poly(Pc, ["n"])
    valo = {"closed": canon, "P": P.dump(Pc), "deg": want_deg}
    return _finish(True, None, valo, cert, tactic="bh:sum_d%d" % want_deg, resid=[0.0],
                   detail="difference_equation_solve")


def op_ind_sum_d0(args):
    return _ind_sum(args, 0)


def op_ind_sum_d1(args):
    return _ind_sum(args, 1)


def op_ind_sum_d2(args):
    return _ind_sum(args, 2)


def op_ind_sum_d3(args):
    return _ind_sum(args, 3)


def ind_solve_tactic(kind, closed, hs_step=None):
    """Lean tactic block for the solve step. kind in {poly, geo_pow, geo_const}.
    A non-empty hs_step means hstep is a "same value, different form" disguise (moved/scaled); first
    normalize it to the canonical step via hs. The normalization tactic depends on the disguise type:
    'linarith' (linear reshaping) / 'rw_ring' (algebraically equal but not linearly decidable)."""
    rw_name = "hstep"
    pre = ""
    if hs_step:
        step_text, how = hs_step
        tac = ("linarith" if how == "linarith" else "rw [h]\n  ring")
        pre = ("have hs : \u2200 k, %s := by\n  intro k\n  have h := hstep k\n  %s\n"
               % (step_text, tac))
        rw_name = "hs"
    if kind == "geo_const":
        body = ("intro n\n%shave hP : f n = %s := by\n  induction n with\n"
                "  | zero => exact h0\n  | succ n ih =>\n    rw [%s, ih]\n    ring"
                % (pre, closed, rw_name))
        return body
    if kind == "geo_pow":
        body = ("intro n\n%shave hP : f n = %s := by\n  induction n with\n"
                "  | zero => norm_num [h0]\n  | succ n ih =>\n"
                "    rw [%s, ih, pow_succ]\n    ring" % (pre, closed, rw_name))
        return body
    return ("intro n\n%shave hP : f n = %s := by\n  induction n with\n"
            "  | zero => norm_num [h0]\n  | succ n ih =>\n"
            "    rw [%s, ih]\n    push_cast\n    ring" % (pre, closed, rw_name))


def op_ind_geo(args):
    """args {v0, r, c}. Class assertion (step-1) = affine/geometric family (r != 1) with an integer
    fixed point. Outputs c1 = v0 - c/(1-r), c0 = c/(1-r); **step-2's class (const/pow) is chosen by the operator name**."""
    v0, r, c = int(args["v0"]), int(args["r"]), int(args["c"])
    if r == 1:
        return OpResult(False, err="DEGENERATE_LINEAR", tactic="")
    if c % (1 - r) != 0:
        return OpResult(False, err="NONINT_FIXEDPOINT", value={"c": c, "1-r": 1 - r},
                        tactic="")
    c0 = Fraction(c, 1 - r)
    c1 = Fraction(v0) - c0
    cert = {"kind": "bh_geo_cert",
            "payload": {"v0": v0, "r": r, "c": c, "c1": str(c1), "c0": str(c0),
                        "cls": "const" if c1 == 0 else "pow"}}
    return _finish(True, None, {"c1": str(c1), "c0": str(c0), "r": r,
                                "cls": "const" if c1 == 0 else "pow"}, cert,
                   tactic="bh:geo_solve", resid=[0.0], detail="fixed_point_affine")


def geo_closed_text(c1, c0, r, style="plain"):
    """Geometric closed-form **canonical** text (for the hP line; same source as bh_tpl's pow_plain/pow_reordered)."""
    import bh_tpl as TPL
    c1, c0 = int(c1), int(c0)
    if c1 == 0:
        return str(c0)
    return TPL.render("pow_reordered" if style == "reordered" else "pow_plain", c1, c0, r)


def _ind_close(args, cls):
    """args {P, T or T_text, sq?}: equivalence between the computed closed form P and the statement-declared
    T, plus class exclusion. The T_text path uses the **verbatim statement text**; if it is not a polynomial
    in n (e.g. contains r^n) => this class is unusable (a genuine failure)."""
    Pd = P.load(args["P"])
    if "T" in args:
        Td = P.load(args["T"])
    else:
        try:
            Td = P.parse_poly(args["T_text"], ["n"])
        except Exception:                                      # noqa: BLE001
            return OpResult(False, err="T_NOT_POLYNOMIAL", value={"cls": cls}, tactic="")
    nv = (len(next(iter(Pd))) if Pd else (len(next(iter(Td))) if Td else 1))
    payload = {"P": P.dump(Pd), "T": P.dump(Td), "cls": cls}
    if cls in ("square", "square2", "square4"):
        if "sq" not in args:
            return OpResult(False, err="MISSING_SQ_ROOT", tactic="")
        payload["sq"] = [str(x) for x in args["sq"]]
    cert = {"kind": "bh_equiv_cert", "payload": payload}
    return _finish(True, None, {"cls": cls, "T": P.dump(Td)}, cert,
                   tactic="rw [hP]\nring", resid=[0.0], detail="closed_form_equiv")


def op_ind_close_poly(args):
    return _ind_close(args, "poly")


def op_ind_close_sq(args):
    return _ind_close(args, "square")


def op_ind_close_sq2(args):
    return _ind_close(args, "square2")


def op_ind_close_sq4(args):
    return _ind_close(args, "square4")


def op_ind_close_const(args):
    """Degenerate-class close: with (v0,r,c,tpl,T_text) given, use the geometric template certificate;
    otherwise use the polynomial equivalence certificate."""
    if "v0" in args:
        return _geo_close(args)
    return _ind_close(args, "const")


def _geo_close(args):
    cert = {"kind": "bh_geo_close_cert",
            "payload": {"v0": int(args["v0"]), "r": int(args["r"]), "c": int(args["c"]),
                        "cls": args["cls"], "tpl": args["tpl"], "text": args["T_text"]}}
    return _finish(True, None, {"tpl": args["tpl"], "cls": args["cls"]}, cert,
                   tactic="rw [hP]\nring", resid=[0.0],
                   detail="geo_close_%s" % args["tpl"])


def op_ind_close_pow(args):
    """args {v0, r, c, T_text, tpl}: **template-level** check of the geometric closed form against the
    declared form (same value, different form). The checker independently recomputes c1/c0 (forward
    simulation) + the declared text must strictly equal the template instance built from the recomputed
    params + class exclusion."""
    return _geo_close(args)


# ================================================================ INEQ family
def _ell_poly(vars_, ell):
    nv = len(vars_)
    idx = {v: i for i, v in enumerate(vars_)}
    out = {}
    for v, cf in ell["coefs"].items():
        if Fraction(cf) == 0:
            continue
        m = [0] * nv
        m[idx[v]] = 1
        out[tuple(m)] = Fraction(cf)
    sh = Fraction(ell.get("shift", 0))
    if sh != 0:
        out[(0,) * nv] = out.get((0,) * nv, Fraction(0)) - sh
    return out


def ell_text(ell):
    parts = []
    for v, cf in ell["coefs"].items():
        cf = Fraction(cf)
        if cf == 0:
            continue
        s = v if abs(cf) == 1 else "%d * %s" % (int(abs(cf)), v)
        parts.append(("+" if cf > 0 else "-", s))
    sh = Fraction(ell.get("shift", 0))
    if sh != 0 or not parts:
        parts.append(("-" if sh > 0 else "+", str(int(abs(sh)))))
    out = ""
    for i, (sg, b) in enumerate(parts):
        if i == 0:
            out = ("- " if sg == "-" else "") + b
        else:
            out += (" - " if sg == "-" else " + ") + b
    return "(%s)" % out


def _ineq_block(args, name):
    """args {vars, R_in(dump), ell:{coefs,shift}, mult, hint_idx}.
    Class assertion = the block type (single-variable completing the square / multi-variable cross)
    plus the certificate's own purge (exact absorption)."""
    vars_ = list(args["vars"])
    nv = len(vars_)
    R_in = P.load(args["R_in"])
    ell = args["ell"]
    supp = [v for v, cf in ell["coefs"].items() if Fraction(cf) != 0]
    if name == "bh_ineq_cross" and len(supp) < 2:
        return OpResult(False, err="CROSS_NEEDS_TWO_VARS", value={"supp": supp}, tactic="")
    if name == "bh_ineq_block" and len(supp) != 1:
        return OpResult(False, err="BLOCK_NEEDS_ONE_VAR", value={"supp": supp}, tactic="")
    try:
        lp = _ell_poly(vars_, ell)
    except Exception as e:                                     # noqa: BLE001
        return OpResult(False, err="ELL_ARGS:%s" % type(e).__name__, tactic="")
    mult = Fraction(args["mult"])
    R_out = P.psub(R_in, P.pscale(P.pmul(lp, lp), mult))
    cert = {"kind": "bh_block_cert",
            "payload": {"vars": vars_, "R_in": P.dump(R_in), "R_out": P.dump(R_out),
                        "ell": P.dump(lp), "mult": str(mult)}}
    hint = "%d * %s ^ 2" % (int(mult), ell_text(ell))
    tact = "have h%d : (0:\u211a) \u2264 %s := by positivity" % (int(args["hint_idx"]), hint)
    valo = {"R_out": P.dump(R_out), "R_out_text": P.render_poly(R_out, vars_),
            "ell": ell, "ell_text": ell_text(ell), "mult": str(mult), "hint": hint}
    return _finish(True, None, valo, cert, tactic=tact, resid=[0.0], detail="sos_block")


def op_ineq_block(args):
    return _ineq_block(args, "bh_ineq_block")


def op_ineq_cross(args):
    return _ineq_block(args, "bh_ineq_cross")


def _ineq_tail(args, cls):
    """args {vars, goal_dom, hints_poly, rho, sq?}: terminal step. Class assertion = the residual's
    semantic class (exclusive); the **closed-chain criterion** = sum(taken blocks) + rho ≡ goal dom
    (verified by the certificate machinery), so stopping early necessarily fails."""
    rho = Fraction(args["rho"])
    payload = {"rho": str(rho), "cls": cls, "vars": list(args["vars"]),
               "goal_dom": dict(args["goal_dom"]),
               "hints_poly": [dict(h) for h in args["hints_poly"]]}
    if cls == "square":
        payload["sq"] = int(args["sq"])
    cert = {"kind": "bh_tail_cert", "payload": payload}
    hs = list(args["hints"])
    if cls == "zero":
        body = ["nlinarith [%s]" % ", ".join(hs)]
    elif cls == "square":
        s = int(args["sq"])
        body = ["have hrho : (0:\u211a) \u2264 (%d : \u211a) ^ 2 := sq_nonneg (%d : \u211a)"
                % (s, s), "nlinarith [%s, hrho]" % ", ".join(hs)]
    else:
        body = ["have hrho : (0:\u211a) \u2264 (%s : \u211a) := by norm_num" % rho,
                "nlinarith [%s, hrho]" % ", ".join(hs)]
    return _finish(True, None, {"rho": str(rho), "cls": cls}, cert, tactic="\n".join(body),
                   resid=[0.0], detail="tail_%s" % cls)


def op_ineq_tail_zero(args):
    return _ineq_tail(args, "zero")


def op_ineq_tail_square(args):
    return _ineq_tail(args, "square")


def op_ineq_tail_other(args):
    return _ineq_tail(args, "other")


# ================================================================ NT family
def op_nt_rewrite(args):
    """args {hyp, lhs, text(full context), body(current body)}: definition step.
    Preconditions = definition shape + current body contains lhs (a decoy fails locally with PRECOND_OCCURS)."""
    cert = {"kind": "bh_shape_cert",
            "payload": {"assert": "def_hyp", "hyp": args["hyp"], "lhs": args["lhs"],
                        "text": args["text"]}}
    r = CK.check(cert["kind"], cert["payload"])
    if not r[0]:
        return OpResult(False, err="CERT_REJECT", cert=cert, cert_ok=False, cert_detail=r[2],
                        tactic="")
    r2 = CK.check("bh_shape_cert", {"assert": "occurs", "var": args["lhs"],
                                    "body": args.get("body"), "text": args["text"]})
    if not r2[0]:
        return OpResult(False, err="PRECOND_OCCURS", cert=cert, cert_ok=False,
                        cert_detail=r2[2], tactic="")
    return _finish(True, None, {"applied": args["hyp"]}, cert,
                   tactic="rw [%s]" % args["hyp"], resid=[0.0], detail="def_rewrite")


def _nt_expand(args, vwant):
    """args {m, hyp, var, text, body, modulus?}: witness step; the nt_val variant additionally asserts
    m-adic valuation == vwant."""
    cert = {"kind": "bh_shape_cert",
            "payload": {"assert": "dvd_hyp", "hyp": args["hyp"], "m": int(args["m"]),
                        "var": args["var"], "text": args["text"],
                        "modulus": int(args.get("modulus", args["m"]))}}
    r = CK.check(cert["kind"], cert["payload"])
    if not r[0]:
        return OpResult(False, err="CERT_REJECT", cert=cert, cert_ok=False, cert_detail=r[2],
                        tactic="")
    occ = CK.check("bh_shape_cert", {"assert": "occurs", "var": args["var"],
                                     "body": args.get("body"), "text": args["text"]})
    if not occ[0]:
        return OpResult(False, err="PRECOND_OCCURS", cert=cert, cert_ok=False,
                        cert_detail=occ[2], tactic="")
    if vwant is not None:
        mv = CK.check("bh_shape_cert", {"assert": "mod_val", "hyp": args["hyp"],
                                        "m": int(args["m"]), "val": vwant,
                                        "modulus": int(args["modulus"]), "text": args["text"]})
        if not mv[0]:
            return OpResult(False, err="CERT_REJECT", cert=cert, cert_ok=False,
                            cert_detail=mv[2], tactic="")
    return _finish(True, None, {"var": args["var"], "val": vwant}, cert,
                   tactic="obtain \u27e8w_%s, rfl\u27e9 := %s" % (args["var"], args["hyp"]),
                   resid=[0.0], detail="nt_expand_v%s" % (vwant if vwant is not None else "any"))


def op_nt_expand(args):
    return _nt_expand(args, None)


def op_nt_expand_v1(args):
    return _nt_expand(args, 1)


def op_nt_expand_v2(args):
    return _nt_expand(args, 2)


def _mono_text(t):
    c = Fraction(t["coef"])
    mons = list(t["monomial"])
    if not mons:
        return str(int(c))
    parts = []
    if c != 1:
        parts.append(str(int(c)))
    parts.extend(mons)
    return " * ".join(parts)


def op_nt_close(args):
    """args {m, M, terms, const, witness, params}: witness identity (polynomials over the naturals)."""
    m, M = int(args["m"]), int(args["M"])
    wit = args["witness"]
    payload = {"m": m, "M": M, "params": list(args["params"]),
               "terms": [{"coef": str(t["coef"]), "k": int(t["k"]),
                          "wvars": list(t["wvars"]), "freevar": t.get("freevar"),
                          "mfac": bool(t.get("mfac"))} for t in args["terms"]],
               "const": int(args["const"]),
               "witness": [{"coef": str(int(Fraction(t["coef"]))),
                            "monomial": list(t["monomial"])} for t in wit]}
    cert = {"kind": "bh_wit_cert", "payload": payload}
    parts = [_mono_text(t) for t in wit if Fraction(t["coef"]) != 0]
    body = " + ".join(parts) if parts else "0"
    return _finish(True, None, {"witness_text": body, "M": M, "m": m}, cert,
                   tactic="exact \u27e8%s, by ring\u27e9" % body, resid=[0.0],
                   detail="nt_close")


OPS = {"bh_ind_sum_d0": op_ind_sum_d0, "bh_ind_sum_d1": op_ind_sum_d1,
       "bh_ind_sum_d2": op_ind_sum_d2, "bh_ind_sum_d3": op_ind_sum_d3,
       "bh_ind_geo": op_ind_geo,
       "bh_ind_close_poly": op_ind_close_poly, "bh_ind_close_sq": op_ind_close_sq,
       "bh_ind_close_sq2": op_ind_close_sq2, "bh_ind_close_sq4": op_ind_close_sq4,
       "bh_ind_close_const": op_ind_close_const, "bh_ind_close_pow": op_ind_close_pow,
       "bh_ineq_block": op_ineq_block, "bh_ineq_cross": op_ineq_cross,
       "bh_ineq_tail_zero": op_ineq_tail_zero, "bh_ineq_tail_square": op_ineq_tail_square,
       "bh_ineq_tail_other": op_ineq_tail_other,
       "bh_nt_rewrite": op_nt_rewrite, "bh_nt_expand": op_nt_expand,
       "bh_nt_expand_v1": op_nt_expand_v1, "bh_nt_expand_v2": op_nt_expand_v2,
       "bh_nt_close": op_nt_close}


def call(op, args):
    fn = OPS.get(op)
    if fn is None:
        return OpResult(False, err="OP_NOT_REGISTERED", tactic="")
    if op in BANNED_OPS:
        return OpResult(False, err="BANNED_OP", tactic="")
    return fn(args)


def whitelist_check(ops_used):
    bad = [o for o in ops_used if o not in PROOF_OPS]
    banned = [o for o in ops_used if o in BANNED_OPS]
    return (not bad) and (not banned), {"non_whitelist": bad, "banned": banned}
