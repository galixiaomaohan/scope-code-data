# -*- coding: utf-8 -*-
# ccp_ops.py -- CCP certified operators for the three pilot classes (advance team gen12, constitution change 58)
#
# Each operator = (a) computation logic (closed-form fit / sum-of-squares decomposition / witness
# construction) + (b) Lean tactic wrapper (tactic text) + (c) certificate (pure-data payload,
# handed to ccp_check for independent re-verification). **No self-certification**: generation
# logic lives in this module, all re-verification lives in ccp_check (whose register_checker
# declares independent_of=("ccp_ops",)).
# Operators outside the allowlist may not enter positive chains; math_evaluate_expr is banned for
# positive supervision across the whole batch (v1's six-loop root cause, sealed off directly).
from fractions import Fraction

import ccp_check as CK

PROOF_OPS = ("ccp_ind_recurrence", "ccp_ring_identity", "ccp_ineq_sos", "ccp_ineq_close",
             "ccp_nt_dvd_witness", "ccp_nt_dvd_combine", "ccp_nt_dvd_prod_combine")
BANNED_OPS = ("math_evaluate_expr",)
UNLIM = Fraction(10 ** 12)


class OpResult(object):
    __slots__ = ("ok", "err", "value", "cert", "cert_ok", "cert_detail", "tactic", "resid")

    def __init__(self, ok, err=None, value=None, cert=None, cert_ok=None,
                 cert_detail="", tactic="", resid=None):
        self.ok = ok
        self.err = err
        self.value = value
        self.cert = cert
        self.cert_ok = cert_ok
        self.cert_detail = cert_detail
        self.tactic = tactic
        self.resid = resid or [0.0]

    def to_dict(self):
        return {"ok": self.ok, "err": self.err, "value": self.value, "cert": self.cert,
                "cert_ok": self.cert_ok, "cert_detail": self.cert_detail,
                "tactic": self.tactic, "resid": self.resid}


def _finish(ok, err, value, cert, tactic, resid=None, detail=""):
    """Pipeline tail: independent certificate re-verification (a certificate that is not
    independent or does not pass is not a solution; mirrors svokernel §13)."""
    if not ok:
        return OpResult(False, err=err, value=value, tactic=tactic, resid=resid)
    cok, cr, cdet = CK.check(cert["kind"], cert["payload"])
    ind = CK.__name__ not in CK.independence(cert["kind"])
    msg = (cdet if not detail else detail + " | " + cdet) + " | independent=%s" % ind
    if not (cok and ind):
        # certificate re-verification failed => **not a solution** (mirrors svokernel §13);
        # step failure state = fail_c
        return OpResult(False, err="CERT_REJECT", value=value, cert=cert, cert_ok=False,
                        cert_detail=msg, tactic=tactic, resid=cr)
    return OpResult(True, None, value, cert, True, msg, tactic=tactic, resid=cr)


def dump_poly(p):
    return {",".join(str(e) for e in m): str(c) for m, c in p.items() if c != 0}


def parse_poly(d):
    return {tuple(int(x) for x in k.split(",")): Fraction(v) for k, v in d.items()}


def padd(p, q):
    out = dict(p)
    for m, c in q.items():
        out[m] = out.get(m, Fraction(0)) + c
    return {m: c for m, c in out.items() if c != 0}


def pscale(p, k):
    k = Fraction(k)
    return {m: c * k for m, c in p.items() if c * k != 0}


def lin_text(coefs, vars_, shift):
    """Lean text of the linear form (Σ coefs_i·x_i − shift) (for hint / `have` lines)."""
    parts = []
    for i, a in enumerate(coefs):
        a = Fraction(a)
        if a == 0:
            continue
        if a == 1:
            parts.append(vars_[i])
        elif a == -1:
            parts.append("-" + vars_[i])
        else:
            parts.append("%s*%s" % (a, vars_[i]))
    s = " + ".join(parts).replace("+ -", "- ")
    sh = Fraction(shift)
    if sh > 0:
        s = "%s - %s" % (s, sh)
    elif sh < 0:
        s = "%s + %s" % (s, -sh)
    return s


def sq_of_lin(nvars, coefs, shift, mult):
    """Expansion of mult·(Σ coefs_i x_i − shift)² (structural convolution path)."""
    zero = (0,) * nvars
    poly = {}
    for i, a in enumerate(coefs):
        if Fraction(a) != 0:
            m = [0] * nvars
            m[i] = 1
            poly[tuple(m)] = Fraction(a)
    poly[zero] = poly.get(zero, Fraction(0)) - Fraction(shift)
    sq = {}
    for m1, c1 in poly.items():
        for m2, c2 in poly.items():
            mm = tuple(a + b for a, b in zip(m1, m2))
            sq[mm] = sq.get(mm, Fraction(0)) + c1 * c2
    return pscale({m: c for m, c in sq.items() if c != 0}, mult)


# ---------------------------------------------------------------- 1. recurrence closed form (induction class)
def op_ind_recurrence(args):
    """args: {form:"lin"|"aff", v0, step, sample_n}. Generator route = **two-point fit**
    (solve the 2x2 from a(0), a(1) for the closed-form coordinates); shares no code with the
    checker's fixed-point / forward-simulation route."""
    form = args["form"]
    v0 = int(args["v0"])
    n = int(args["sample_n"])
    try:
        if form == "lin":
            d = int(args["step"])
            a0, a1 = Fraction(v0), Fraction(v0 + d)
            c1, c0 = a1 - a0, a0
            step_v = d
            val = c1 * Fraction(n) + c0
        elif form == "aff":
            r, c = int(args["step"][0]), int(args["step"][1])
            a0, a1 = Fraction(v0), Fraction(r * v0 + c)
            c1 = (a1 - a0) / Fraction(r - 1)
            c0 = a0 - c1
            step_v = [r, c]
            val = c1 * Fraction(r) ** n + c0
        else:
            return OpResult(False, err="UNSUPPORTED_FORM", tactic="")
    except Exception as e:                                    # noqa: BLE001
        return OpResult(False, err="EXC:%s" % type(e).__name__, tactic="")
    if abs(c1) > UNLIM or abs(c0) > UNLIM or abs(val) > UNLIM:
        return OpResult(False, err="BOUND", tactic="")
    cert = {"kind": "ccp_ind_recurrence_cert",
            "payload": {"form": form, "v0": v0, "step": step_v, "n": n,
                        "closed": {"c1": str(c1), "c0": str(c0), "value": str(val)}}}
    valo = {"form": form, "c1": str(c1), "c0": str(c0), "n": n, "value": str(val)}
    # Note (gen15 F-10): the `tactic` returned by this operator is a **symbolic label**
    # (operator-family name), **not Lean text**; ind's real proof body is rendered by
    # ccp_families.build_ind (this field takes no part in rendering). The field name is kept
    # to avoid breaking the chain (corpus/cache/downstream read the 'tactic' key); only the
    # misreading point is annotated here.
    return _finish(True, None, valo, cert, tactic="induction_closed_form",
                   resid=[0.0], detail="two_point_fit")


# ---------------------------------------------------------------- 2. symbolic ring identity (closed-chain step)
def op_ring_identity(args):
    """args: {nvars, lhs:{mono:coef}, rhs:{mono:coef}, atom, side}. Atom basis = the r^n atom
    of the inductive closed form (the Lean-side atom step is provided by pow_succ); this
    operator verifies the identity **in the atom basis**."""
    try:
        lhs = parse_poly(args["lhs"])
        rhs = parse_poly(args["rhs"])
    except Exception as e:                                    # noqa: BLE001
        return OpResult(False, err="ARGS:%s" % type(e).__name__, tactic="")
    nv = int(args["nvars"])
    side = int(args.get("side", 4))
    for p in (lhs, rhs):
        for m, c in p.items():
            if len(m) != nv or any(e > side - 1 for e in m) or abs(c) > UNLIM:
                return OpResult(False, err="BOUND", tactic="")
    cert = {"kind": "ccp_ring_identity_cert",
            "payload": {"nvars": nv, "side": side, "lhs": dump_poly(lhs), "rhs": dump_poly(rhs)}}
    return _finish(True, None, {"identity": True, "nvars": nv, "atom": args.get("atom", "r^n")},
                   cert, tactic="ring", resid=[0.0], detail="atom_basis_identity")


# ---------------------------------------------------------------- 3. sum of squares (inequality class)
def op_ineq_sos(args):
    """args: {vars, dom:{mono:coef}} -- dom = target LHS − RHS (constant included, derived from
    the statement by the bridge rule). Real solve: (1) negative cross term -> difference of
    squares; (2) diagonal term -> per-variable completion of the square (a>0); (3) the residual
    must be a constant >= 0, otherwise fail_c."""
    nvars = len(args["vars"])
    try:
        P = parse_poly(args["dom"])
    except Exception as e:                                    # noqa: BLE001
        return OpResult(False, err="ARGS:%s" % type(e).__name__, tactic="")
    zero = (0,) * nvars
    P.setdefault(zero, Fraction(0))
    input_dom = dump_poly(P)
    squares = []
    for i in range(nvars):
        for j in range(i + 1, nvars):
            mi = [0] * nvars
            mi[i] = 1
            mj = [0] * nvars
            mj[j] = 1
            cross = tuple(a + b for a, b in zip(mi, mj))
            c = P.get(cross, Fraction(0))
            if c < 0:
                mult = -c / 2
                coefs = [1 if k == i else (-1 if k == j else 0) for k in range(nvars)]
                P = padd(P, pscale(sq_of_lin(nvars, coefs, 0, mult), -1))
                squares.append({"coefs": coefs, "shift": "0", "mult": str(mult)})
    for i in range(nvars):
        sqkey = tuple([2 if k == i else 0 for k in range(nvars)])
        linkey = tuple([1 if k == i else 0 for k in range(nvars)])
        a = P.get(sqkey, Fraction(0))
        if a == 0:
            continue
        if a < 0:
            return OpResult(False, err="UNSUPPORTED_NEG_SQ", tactic="",
                            value={"var": args["vars"][i]})
        ell = P.get(linkey, Fraction(0))
        shift = -ell / (2 * a)
        coefs = [1 if k == i else 0 for k in range(nvars)]
        P = padd(P, pscale(sq_of_lin(nvars, coefs, str(shift), a), -1))
        squares.append({"coefs": coefs, "shift": str(shift), "mult": str(a)})
    rest = {m: c for m, c in P.items() if c != 0}
    if rest and (len(rest) > 1 or zero not in rest):
        return OpResult(False, err="UNSUPPORTED_LEFTOVER", tactic="",
                        value={"leftover": len(rest)})
    rho = rest.get(zero, Fraction(0))
    if rho < 0:
        return OpResult(False, err="RESIDUAL_NEGATIVE", tactic="", value={"residual": str(rho)})
    if not squares:
        return OpResult(False, err="NO_SQUARE_FOUND", tactic="")
    cert = {"kind": "ccp_ineq_sos_cert",
            "payload": {"vars": list(args["vars"]), "squares": squares,
                        "input_dom": input_dom, "residual": str(rho)}}
    hints = [lin_text(sq["coefs"], args["vars"], sq["shift"]) for sq in squares]
    tactic = "nlinarith [%s]" % ", ".join("sq_nonneg (%s)" % h for h in hints)
    valo = {"nvars": nvars, "squares": squares, "residual": str(rho), "hints": hints,
            "input_dom": input_dom}
    return _finish(True, None, valo, cert, tactic=tactic, resid=[float(abs(rho))],
                   detail="sos_solver")


# ---------------------------------------------------------------- 3b. closing step, closed chain (inequality class)
def op_ineq_close(args):
    """args: {vars, squares, residual, text}. Closing-step computation: **re-parse** the target
    polynomial from the statement text and assert target polynomial == Σ squares + ρ with ρ >= 0
    (i.e. the hint combination suffices to derive the goal)."""
    import ccp_bridge as BR
    vars_ = list(args["vars"])
    try:
        stmt = "theorem _ccp_close (%s : ℚ) :\n    %s" % (" ".join(vars_), args["text"])
        br = BR.bridge_ineq(stmt, vars_)
    except Exception as e:                                    # noqa: BLE001
        return OpResult(False, err="GOAL_PARSE:%s" % type(e).__name__, tactic="")
    dom = {",".join(map(str, m)): str(c) for m, c in br["dom"].items()}
    squares = args["squares"]
    rho = Fraction(args["residual"])
    hints = [lin_text(sq["coefs"], vars_, sq["shift"]) for sq in squares]
    cert = {"kind": "ccp_ineq_close_cert",
            "payload": {"vars": vars_, "squares": squares, "goal_dom": dom,
                        "residual": str(rho)}}
    tactic = "nlinarith [%s]" % ", ".join("h%d" % i for i in range(1, len(hints) + 1))
    return _finish(True, None, {"goal_dom": dom, "residual": str(rho), "hints": hints},
                   cert, tactic=tactic, resid=[float(abs(rho))], detail="sos_close")


# ---------------------------------------------------------------- 4. divisibility structural step (witness extraction)
def op_nt_dvd_witness(args):
    """args: {m, hyp, var, text}: structural step -- assert an assumption of the form m ∣ var
    (re-parsed and verified by the checker)."""
    cert = {"kind": "ccp_shape_cert",
            "payload": {"assert": "dvd_hyp", "text": args["text"], "hyp": args["hyp"],
                        "m": int(args["m"])}}
    wit = "w_%s" % args["var"]
    return _finish(True, None, {"var": args["var"], "witness_var": wit, "m": int(args["m"])},
                   cert, tactic="obtain ⟨%s, rfl⟩ := %s" % (wit, args["hyp"]),
                   resid=[0.0], detail="shape_step")


# ---------------------------------------------------------------- 5. divisibility combine step (closed chain)
def op_nt_dvd_combine(args):
    """args: {m_h, M, terms:[{coef,k,vars,freevar,mfac}], const, hypvars:[{var,hyp}]}.
    Computation: after substituting v := m_h·w_v the target equals M·(witness polynomial).
    Witness = Σ (coef·m_h^k / M)·Π w_v [+ (coef/M)·free var] + const/M -- all coefficients must
    be **nonnegative integers** (ℕ-valued witness), otherwise fail_c."""
    m_h, M = int(args["m_h"]), int(args["M"])
    K = int(args["const"])
    if m_h <= 1 or M <= 1 or M % m_h != 0:
        return OpResult(False, err="BAD_MODULUS", tactic="")
    hypvars = [x["var"] for x in args["hypvars"]]
    wof = {v: "w_%s" % v for v in hypvars}
    params = [wof[v] for v in hypvars] + sorted({t["freevar"] for t in args["terms"]
                                                 if t.get("freevar")})
    wit_terms = []
    for t in args["terms"]:
        coef = Fraction(t["coef"])
        if t.get("mfac"):
            wc = coef * Fraction(m_h) / M      # structural term c*m_h*t: target coefficient contribution coef*m_h (unchanged by substitution)
            mon = [t["freevar"]]
        else:
            wc = coef * Fraction(m_h) ** int(t["k"]) / M
            mon = [wof[v] for v in t["vars"]]
        if wc.denominator != 1 or wc < 0:
            return OpResult(False, err="WITNESS_NOT_NATURAL", value={"coef": str(wc)},
                            tactic="")
        wit_terms.append({"coef": str(int(wc)), "monomial": mon})
    if K % M != 0:
        return OpResult(False, err="CONST_NOT_DIVISIBLE", value={"K": K, "M": M}, tactic="")
    if K // M != 0:
        wit_terms.append({"coef": str(K // M), "monomial": []})   # constant witness (aligns with G's constant term)
    cert = {"kind": "ccp_nt_dvd_cert",
            "payload": {"m_h": m_h, "M": M, "params": params,
                        "terms": [{"coef": str(t["coef"]), "k": int(t["k"]),
                                   "wvars": [wof[v] for v in t["vars"]],
                                   "freevar": t.get("freevar"), "mfac": bool(t.get("mfac"))}
                                  for t in args["terms"]],
                        "const": K, "witness": wit_terms}}
    def _mono(t):
        c = Fraction(t["coef"])
        mons = list(t["monomial"])
        if not mons:
            return str(int(c))
        parts = []
        if c != 1:
            parts.append(str(int(c)))
        parts.extend(mons)
        return " * ".join(parts)
    parts = [_mono(t) for t in wit_terms if Fraction(t["coef"]) != 0]
    wit = " + ".join(parts) if parts else "0"
    valo = {"witness": wit, "M": M, "m_h": m_h, "const_wit": K // M,
            "witness_terms": wit_terms, "params": params}
    return _finish(True, None, valo, cert, tactic="exact ⟨%s, by ring⟩" % wit,
                   resid=[0.0], detail="dvd_combine")


def op_nt_dvd_prod_combine(args):
    """Thin shell for the product family (m² ∣ a*b): delegates to the generalized combine step."""
    a = dict(args)
    a.setdefault("terms", [{"coef": 1, "k": 2, "vars": ["a", "b"], "freevar": None,
                            "mfac": False}])
    a.setdefault("const", 0)
    r = op_nt_dvd_combine(a)
    if r.ok:
        r.value["prod"] = True
    return r


OPS = {"ccp_ind_recurrence": op_ind_recurrence,
       "ccp_ring_identity": op_ring_identity,
       "ccp_ineq_sos": op_ineq_sos,
       "ccp_ineq_close": op_ineq_close,
       "ccp_nt_dvd_witness": op_nt_dvd_witness,
       "ccp_nt_dvd_combine": op_nt_dvd_combine,
       "ccp_nt_dvd_prod_combine": op_nt_dvd_prod_combine}


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
