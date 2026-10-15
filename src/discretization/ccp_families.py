# -*- coding: utf-8 -*-
# ccp_families.py -- problem-family generators for the three CCP pilot classes (advance team gen12, constitution change 58)
#
# The three classes = the default candidates of the task book (maximum difference):
#        ind inductive recurrence · ineq inequalities (sum-of-squares / completing the square) ·
#        nt number-theoretic identities (divisibility closure). Per family:
#        params -> statement (Lean text) -> bridge recomputation (roundtrip gate) ->
#        operator chain (with certificates) -> proof body (per-operator tactic lines) ->
#        one-chain-multiple-rows (whole-chain row + per-step rows).
# Discipline: positive-chain operators ⊆ ccp_ops.PROOF_OPS; math_evaluate_expr forbidden; every
# step's certificate independently re-verified.
import hashlib
import re
from fractions import Fraction

import ccp_bridge as BR
import ccp_ops as OPS

CLASSES = ("ind", "ineq", "nt")
CLASS_ZH = {"ind": "归纳递推", "ineq": "不等式放缩", "nt": "数论整除"}


def _cseed(master, cls, family, seq):
    return int(hashlib.md5(("ccp|%d|%s|%s|%d" % (master, cls, family, seq)).encode("utf-8")).hexdigest()[:8], 16)


def sgn_join(terms):
    """[('+', 'x ^ 2'), ('-', '2 * x')] → 'x ^ 2 - 2 * x'"""
    out = ""
    for i, (s, t) in enumerate(terms):
        if i == 0:
            out = ("- " if s == "-" else "") + t
        else:
            out += (" - " if s == "-" else " + ") + t
    return out


def render_poly(p, vars_):
    """{exponent tuple: Fraction} → canonical flat sum (integer coefficients; the generator
    produces integer coefficients only)."""
    zero = (0,) * len(vars_)
    items = sorted(p.items(), key=lambda kv: (-sum(kv[0]), kv[0]))
    parts = []
    for m, c in items:
        c = Fraction(c)
        assert c.denominator == 1, "非整数系数 %s" % c
        ci = int(c)
        if ci == 0:
            continue
        s = "-" if ci < 0 else "+"
        a = abs(ci)
        mono = []
        for i, e in enumerate(m):
            if e == 0:
                continue
            mono.append(vars_[i] if e == 1 else "%s ^ %d" % (vars_[i], e))
        if not mono:
            parts.append((s, str(a)))
            continue
        body = " * ".join(mono)
        parts.append((s, body if a == 1 else "%d * %s" % (a, body)))
    return sgn_join(parts) if parts else "0"


def poly_of_squares(nvars, squares, rho=0):
    p = {}
    zero = (0,) * nvars
    for sq in squares:
        p = OPS.padd(p, OPS.sq_of_lin(nvars, sq["coefs"], sq["shift"], sq["mult"]))
    p = OPS.padd(p, {zero: Fraction(rho)})
    return p


def lint(x):
    return str(int(x))


# ================================================================= IND: inductive recurrence
def gen_ind(family, rng):
    form = {"ind_lin": "lin", "ind_aff2": "aff", "ind_aff3": "aff"}[family]
    v0 = rng.randint(-20, 20)
    if family == "ind_lin":
        d = rng.choice([x for x in range(-12, 13) if x != 0])
        step = d
        v0 = rng.randint(-30, 30)
    elif family == "ind_aff2":
        r = 2
        c = rng.choice([x for x in range(-9, 10) if x != 0])
        step = [r, c]
    else:
        r = 3
        c = rng.choice([x for x in range(-8, 9, 2) if x != 0])
        step = [r, c]
    return {"form": form, "v0": v0, "step": step, "family": family, "ty": "ℤ"}


def _ind_step_text(form, step):
    if form == "lin":
        d = int(step)
        return "f k + %d" % d if d > 0 else "f k - %d" % (-d)
    r, c = int(step[0]), int(step[1])
    base = "f k" if r == 1 else ("%d * f k" % r)
    if c == 0:
        return base
    return base + (" + %d" % c if c > 0 else " - %d" % (-c))


def ind_statement(name, p, c1, c0):
    form = p["form"]
    if form == "lin":
        closed = _lin_text(c1, c0)
        succ = _lin_text(c1, c0, at="n + 1")
    else:
        r = int(p["step"][0])
        closed = _aff_text(c1, c0, r)
        succ = _aff_text(c1, c0, r, at="n + 1", pow_shift=True)
    stmt = ("theorem %s (f : ℕ → ℤ) (h0 : f 0 = %d) (hstep : ∀ k, f (k + 1) = %s) :\n"
            "    ∀ n : ℕ, f n = %s" % (name, p["v0"], _ind_step_text(form, p["step"]), closed))
    return stmt, closed, succ


def _lin_text(c1, c0, at="n"):
    c1, c0 = int(c1), int(c0)
    at_s = "(n : ℤ)" if at == "n" else "((n : ℤ) + 1)"
    t = at_s if abs(c1) == 1 else "%d * %s" % (abs(c1), at_s)
    items = []
    if c1:
        items.append((1 if c1 > 0 else -1, t))
    if c0:
        items.append((1 if c0 > 0 else -1, str(abs(c0))))
    if not items:
        return "0"
    out = ""
    for i, (sg, tt) in enumerate(items):
        if i == 0:
            out = ("- " if sg < 0 else "") + tt
        else:
            out += (" - " if sg < 0 else " + ") + tt
    return out


def _aff_text(c1, c0, r, at="n", pow_shift=False):
    c1, c0 = int(c1), int(c0)
    ex = "n" if not pow_shift else "n + 1"
    pth = "%d ^ %s" % (r, ex) if not pow_shift else "%d ^ (%s)" % (r, ex)
    items = []
    if c1:
        items.append((1 if c1 > 0 else -1, pth if abs(c1) == 1 else "%d * %s" % (abs(c1), pth)))
    if c0:
        items.append((1 if c0 > 0 else -1, str(abs(c0))))
    if not items:
        return "0"
    out = ""
    for i, (sg, tt) in enumerate(items):
        if i == 0:
            out = ("- " if sg < 0 else "") + tt
        else:
            out += (" - " if sg < 0 else " + ") + tt
    return out


def build_ind(family, name, p, sample_n):
    """→ (stmt, chain_steps, proof_lines, states)."""
    a = dict(p)
    a["sample_n"] = sample_n
    r1 = OPS.call("ccp_ind_recurrence", a)
    if not r1.ok or not r1.cert_ok:
        return None, "ind_recurrence_%s" % (r1.err or "cert"), None, None
    c1, c0 = Fraction(r1.value["c1"]), Fraction(r1.value["c0"])
    if c1.denominator != 1 or c0.denominator != 1:
        return None, "ind_nonint_closed", None, None
    stmt, closed, succ = ind_statement(name, p, c1, c0)
    # bridge recomputation (roundtrip gate): statement -> params must equal the generation params
    br = BR.bridge_ind(stmt)
    if (br["form"], br["v0"], br["step"]) != (p["form"], p["v0"], p["step"]):
        return None, "ind_bridge_mismatch", None, None
    decl = BR.bridge_ind_declared(stmt)
    if _canon(decl) != _canon(closed):
        return None, "ind_decl_mismatch", None, None
    # ring identity step (atom basis)
    if p["form"] == "lin":
        lhs = {(0,): c0 + Fraction(p["step"]), (1,): c1}
        rhs = {(0,): c0 + c1, (1,): c1}
        atom = "n"
        rw = "rw [hstep, ih]"
    else:
        r, c = int(p["step"][0]), int(p["step"][1])
        lhs = {(0,): r * c0 + c, (1,): r * c1}
        rhs = {(0,): c0, (1,): c1 * r}
        atom = "%d^n" % r
        # when the closed form is constant (c1=0, degenerate case) the goal shows no r^n,
        # so pow_succ has nothing to rewrite
        rw = "rw [hstep, ih, pow_succ]" if "^" in closed else "rw [hstep, ih]"
    r2 = OPS.call("ccp_ring_identity",
                  {"nvars": 1, "atom": atom, "side": 4,
                   "lhs": {",".join(map(str, k)): str(v) for k, v in lhs.items()},
                   "rhs": {",".join(map(str, k)): str(v) for k, v in rhs.items()}})
    if not r2.ok or not r2.cert_ok:
        return None, "ind_ring_%s" % (r2.err or "cert"), None, None
    open_tac = "\n".join(["intro n", "induction n with", "| zero => simp [h0]",
                          "| succ n ih =>", "  %s" % rw])
    # closing-step tactic lines: the lin form carries the ℕ→ℤ projection (↑(1+n)) and needs
    # push_cast to reach ring normal form; the aff form has no projection
    close_lines = ["  push_cast", "  ring"] if p["form"] == "lin" else ["  ring"]
    chain = [{"idx": 1, "op": "ccp_ind_recurrence", "args": a, "tactic": open_tac,
              "value": r1.value, "cert": r1.cert, "cert_ok": r1.cert_ok,
              "cert_detail": r1.cert_detail, "resid": r1.resid, "sym": "closed_form"},
             {"idx": 2, "op": "ccp_ring_identity", "args": {"nvars": 1, "atom": atom},
              "tactic": "\n".join(x.strip() for x in close_lines), "value": r2.value,
              "cert": r2.cert, "cert_ok": r2.cert_ok, "cert_detail": r2.cert_detail,
              "resid": r2.resid, "sym": "atom_identity"}]
    proof = open_tac.split("\n") + close_lines
    bind = "(f : ℕ → ℤ) (h0 : f 0 = %d) (hstep : ∀ k, f (k + 1) = %s)" % (
        p["v0"], _ind_step_text(p["form"], p["step"]))
    g1 = "%s ⊢ ∀ n : ℕ, f n = %s" % (bind, closed)
    if p["form"] == "lin":
        applied = "(%s) + (%d)" % (closed, int(p["step"]))
    else:
        r, c = int(p["step"][0]), int(p["step"][1])
        applied = "%d * (%s) + (%d)" % (r, closed, c)
    st2 = "%s (n : ℕ) (ih : f n = %s) ⊢ %s = %s" % (bind, closed, applied, succ)
    states = [g1, st2]
    return {"stmt": stmt, "chain": chain, "proof": proof, "states": states,
            "closed": closed, "g1": g1, "ty": "ℤ"}, None, None, None


def _canon(s):
    """Text normalization (strip whitespace / unify symbols) for declared-value consistency comparison."""
    s = re.sub(r"\s+", "", s)
    return s.replace("≤", "<=").replace("≥", ">=")


# ================================================================= INEQ: sum of squares
def gen_ineq(family, rng):
    if family == "ineq_sq1":
        s = rng.randint(-9, 9)
        return {"nvars": 1, "vars": ["x"], "squares": [{"coefs": [1], "shift": str(s), "mult": 1}],
                "rho": 0, "family": family}
    if family == "ineq_diff2":
        return {"nvars": 2, "vars": ["x", "y"],
                "squares": [{"coefs": [1, -1], "shift": "0", "mult": 1}],
                "rho": rng.choice([0, 0, 1, 3, 5]), "family": family}
    if family == "ineq_sq2":
        s1, s2 = rng.randint(-7, 7), rng.randint(-7, 7)
        return {"nvars": 2, "vars": ["x", "y"],
                "squares": [{"coefs": [1, 0], "shift": str(s1), "mult": 1},
                            {"coefs": [0, 1], "shift": str(s2), "mult": 1}],
                "rho": rng.choice([0, 0, 2]), "family": family}
    if family == "ineq_diff_sq":
        s = rng.randint(-6, 6)
        return {"nvars": 3, "vars": ["x", "y", "z"],
                "squares": [{"coefs": [1, -1, 0], "shift": "0", "mult": 1},
                            {"coefs": [0, 0, 1], "shift": str(s), "mult": 1}],
                "rho": 0, "family": family}
    if family == "ineq_sq2_mult":
        s1 = rng.randint(-7, 7)
        m = rng.choice([2, 3])
        return {"nvars": 2, "vars": ["x", "y"],
                "squares": [{"coefs": [1, 0], "shift": str(s1), "mult": m},
                            {"coefs": [0, 1], "shift": "0", "mult": 1}],
                "rho": 0, "family": family}
    raise ValueError(family)


def build_ineq(family, name, p, rng):
    nv = p["nvars"]
    vars_ = p["vars"]
    rho = Fraction(p["rho"])
    S = poly_of_squares(nv, p["squares"], 0)
    zero = (0,) * nv
    S0 = S.get(zero, Fraction(0))
    S1 = {m: c for m, c in S.items() if m != zero}
    D = rng.choice([0, 0, 0, rng.randint(-9, 9)])
    K = S0 + rho + D
    LHS = dict(S1)
    LHS[zero] = LHS.get(zero, Fraction(0)) + K
    lhs_txt = render_poly(LHS, vars_)
    flip = rng.random() < 0.5
    rel = "≥" if not flip else "≤"
    concl = "%s %s %d" % (lhs_txt, rel, D)
    head = "theorem %s (%s : ℚ) :\n    %s" % (name, " ".join(vars_), concl)
    stmt = "%s" % head
    # bridge recomputation
    br = BR.bridge_ineq(stmt, vars_)
    dom_plan = {m: c for m, c in S.items()}
    dom_plan[zero] = dom_plan.get(zero, Fraction(0)) + rho
    dom_plan = {m: c for m, c in dom_plan.items() if c != 0}
    if {m: Fraction(c) for m, c in br["dom"].items()} != dom_plan:
        return None, "ineq_bridge_mismatch", None, None
    dom = {",".join(map(str, m)): str(c) for m, c in br["dom"].items()}
    r1 = OPS.call("ccp_ineq_sos", {"vars": vars_, "dom": dom})
    if not r1.ok or not r1.cert_ok:
        return None, "ineq_sos_%s" % (r1.err or "cert"), None, None
    r2 = OPS.call("ccp_ineq_close", {"vars": vars_, "dom": dom,
                                     "squares": r1.value["squares"],
                                     "residual": r1.value["residual"], "text": concl})
    if not r2.ok or not r2.cert_ok:
        return None, "ineq_close_%s" % (r2.err or "cert"), None, None
    hlines = []
    for i, h in enumerate(r1.value["hints"], 1):
        hlines.append("have h%d : (0:ℚ) ≤ (%s) ^ 2 := sq_nonneg (%s)" % (i, h, h))
    close = "nlinarith [%s]" % ", ".join("h%d" % i for i in range(1, len(hlines) + 1))
    chain = [{"idx": 1, "op": "ccp_ineq_sos", "args": {"vars": vars_, "dom": dom},
              "tactic": "\n".join(hlines), "value": r1.value, "cert": r1.cert,
              "cert_ok": r1.cert_ok, "cert_detail": r1.cert_detail, "resid": r1.resid,
              "sym": "sos_decompose"},
             {"idx": 2, "op": "ccp_ineq_close",
              "args": {"vars": vars_, "squares": r1.value["squares"],
                       "residual": r1.value["residual"]},
              "tactic": close, "value": r2.value, "cert": r2.cert, "cert_ok": r2.cert_ok,
              "cert_detail": r2.cert_detail, "resid": r2.resid, "sym": "closure"}]
    proof = hlines + [close]
    bind = "(%s : ℚ)" % " ".join(vars_)
    g1 = "%s ⊢ %s" % (bind, concl)
    ctx2 = "%s %s" % (bind, " ".join("(h%d : (0:ℚ) ≤ %s ^ 2)" % (i, h)
                                     for i, h in enumerate(r1.value["hints"], 1)))
    st2 = "%s ⊢ %s" % (ctx2, concl)
    return {"stmt": stmt, "chain": chain, "proof": proof, "states": [g1, st2],
            "closed": concl, "g1": g1, "ty": "ℚ"}, None, None, None


# ================================================================= NT: divisibility closure
NT_FAMS = ("nt_dvd_mix", "nt_dvd_prod")          # course families (the control group is not part of the course)
NT_CTRL = ("nt_dvd_lin_ctrl",)                       # floor-instrument positive control (probes/unit tests only)


def gen_nt(family, rng):
    m = rng.choice([2, 3, 4, 5, 6, 7, 8, 9, 10, 12])
    p = {"family": family, "m": m, "struct": rng.random() < 0.5}
    if family == "nt_dvd_prod":
        p.update({"M": m * m, "plan": [("prod", 1, ["a", "b"])]})
        return p
    if family == "nt_dvd_lin_ctrl":
        p.update({"M": m, "plan": [("lin", rng.randint(1, 9), ["a"]),
                                   ("lin", rng.randint(1, 9), ["b"])]})
        return p
    plan = []
    for _ in range(rng.choice([1, 1, 2])):
        plan.append(("prod", rng.randint(1, 6), ["a", "b"]))
    if rng.random() < 0.6:
        plan.append(("lin", rng.randint(1, 9), [rng.choice(["a", "b"])]))
    if p["struct"]:
        plan.append(("struct", rng.randint(1, 5), ["t"]))
    p.update({"M": m, "plan": plan})
    return p


def _nt_term_text(m, kind, coef, vars_):
    if kind == "struct":
        return "%d * %d * %s" % (coef, m, vars_[0])
    body = " * ".join(vars_)
    return body if coef == 1 else "%d * %s" % (coef, body)


def build_nt(family, name, p, rng):
    """Unified construction for the three NT families: nt_dvd_mix (products + linear +
    structural terms) / nt_dvd_prod (m² ∣ a*b) / nt_dvd_lin_ctrl (purely linear, the **floor
    control group**: omega is expected to close it => dropped by the floor filter)."""
    m = p["m"]
    M = p["M"]
    plan = p["plan"]
    need_t = any(kd == "struct" for kd, _, _ in plan)
    hyps = [("ha", "a"), ("hb", "b")]
    binder_nat = ["a", "b"] + (["t"] if need_t else [])
    const = rng.choice([0, 0, M, 2 * M, 3 * M])
    parts = [_nt_term_text(m, kd, c, vs) for kd, c, vs in plan]
    if const:
        parts.append(str(const))
    body = " + ".join(parts)
    hyps_txt = " ".join("(%s : %d ∣ %s)" % (h, m, v) for h, v in hyps)
    stmt = "theorem %s (%s : ℕ) %s :" % (name, " ".join(binder_nat), hyps_txt)
    stmt = stmt + chr(10) + "    %d ∣ %s" % (M, body)
    try:
        br = BR.bridge_nt(stmt)
    except Exception as e:                                    # noqa: BLE001
        return None, "nt_bridge_parse:%s" % type(e).__name__, None, None
    # roundtrip gate: bridge parse results must match the generation plan item by item
    if br["M"] != M or br["m_h"] != m or br["const"] != const or len(br["terms"]) != len(plan):
        return None, "nt_bridge_mismatch", None, None
    for tb, (kd, c, vs) in zip(br["terms"], plan):
        if kd == "struct":
            if not tb.get("mfac") or tb["freevar"] != vs[0] or Fraction(tb["coef"]) != Fraction(c):
                return None, "nt_bridge_term_mismatch", None, None
        else:
            if tb.get("mfac") or sorted(tb["vars"]) != sorted(vs)                     or Fraction(tb["coef"]) != Fraction(c) or int(tb["k"]) != len(vs):
                return None, "nt_bridge_term_mismatch", None, None
    chain, states, proof = [], [], []
    cur_body = body
    for i, (h, v) in enumerate(hyps):
        ctx = "%s %s" % (" ".join("(%s : ℕ)" % x for x in binder_nat),
                         " ".join("(%s : %d ∣ %s)" % (hh, m, vv) for hh, vv in hyps[i:]))
        a = {"m": m, "hyp": h, "var": v, "text": "%s ⊢ %d ∣ %s" % (ctx, M, cur_body)}
        r = OPS.call("ccp_nt_dvd_witness", a)
        if not r.ok or not r.cert_ok:
            return None, "nt_witness_%s" % (r.err or "cert"), None, None
        chain.append({"idx": i + 1, "op": "ccp_nt_dvd_witness", "args": a,
                      "tactic": r.tactic, "value": r.value, "cert": r.cert,
                      "cert_ok": r.cert_ok, "cert_detail": r.cert_detail,
                      "resid": r.resid, "sym": "witness_%s" % v})
        proof.append(r.tactic)
        cur_body = _subst_var(cur_body, v, "%d * w_%s" % (m, v))
        states.append("%s %s ⊢ %d ∣ %s" %
                      (" ".join("(%s : ℕ)" % x for x in binder_nat),
                       " ".join("(%s : %d ∣ %s)" % (hh, m, vv) for hh, vv in hyps[i + 1:]),
                       M, cur_body))
    cargs = {"m_h": m, "M": M, "terms": [{"coef": t["coef"], "k": t["k"], "vars": t["vars"],
                                          "freevar": t["freevar"], "mfac": t["mfac"]}
                                         for t in br["terms"]],
             "const": const, "hypvars": [{"var": "a", "hyp": "ha"}, {"var": "b", "hyp": "hb"}]}
    r2 = OPS.call("ccp_nt_dvd_combine", cargs)
    if not r2.ok or not r2.cert_ok:
        return None, "nt_combine_%s" % (r2.err or "cert"), None, None
    chain.append({"idx": len(hyps) + 1, "op": "ccp_nt_dvd_combine", "args": cargs,
                  "tactic": r2.tactic, "value": r2.value, "cert": r2.cert,
                  "cert_ok": r2.cert_ok, "cert_detail": r2.cert_detail,
                  "resid": r2.resid, "sym": "combine"})
    proof.append(r2.tactic)
    bind = "(%s : ℕ) %s" % (" ".join(binder_nat), hyps_txt)
    g1 = "%s ⊢ %d ∣ %s" % (bind, M, body)
    return {"stmt": stmt, "chain": chain, "proof": proof, "states": [g1] + states,
            "closed": body, "g1": g1, "ty": "ℕ", "nt_body": body}, None, None, None


def _subst_var(body, var, repl):
    """Text replacement of an independent variable factor in a sum body (word boundary)."""
    return re.sub(r"(?<![A-Za-z0-9_'])%s(?![A-Za-z0-9_'])" % re.escape(var),
                  "(%s)" % repl, body)


# ================================================================= unified entry point
FAM_OF = {"ind": ["ind_lin", "ind_aff2", "ind_aff3"],
          "ineq": ["ineq_sq1", "ineq_diff2", "ineq_sq2", "ineq_diff_sq", "ineq_sq2_mult"],
          "nt": list(NT_FAMS)}
PROBE_FAMS = {"ind": list(FAM_OF["ind"]), "ineq": list(FAM_OF["ineq"]),
              "nt": list(NT_FAMS) + list(NT_CTRL)}


def gen_params(cls, family, rng):
    if cls == "ind":
        return gen_ind(family, rng)
    if cls == "ineq":
        return gen_ineq(family, rng)
    return gen_nt(family, rng)


def make_problem(cls, family, seq, master):
    rng = __import__("random").Random(_cseed(master, cls, family, seq))
    name = "ccp_%s_%s_%06d" % (cls, family.split("_", 1)[1], seq)
    p = gen_params(cls, family, rng)
    if p is None:
        return None, "params"
    if cls == "ind":
        r = build_ind(family, name, p,
                      sample_n=rng.randint(1, 10 if family == "ind_aff3" else 30))
    elif cls == "ineq":
        r = build_ineq(family, name, p, rng)
    else:
        r = build_nt(family, name, p, rng)
    pr, err = r[0], r[1]
    if pr is None:
        return None, err
    pr.update({"name": name, "cls": cls, "family": family, "seq": seq,
               "seed": _cseed(master, cls, family, seq)})
    return pr, None
