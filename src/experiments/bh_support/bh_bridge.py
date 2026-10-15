# -*- coding: utf-8 -*-
# bh_bridge.py -- bridge-hard batch bridge rules (pure functions: problem text -> operator parameters; gen14, change-58 curriculum-upgrade batch)
#
# Four bridges:
#   bridge_ind_sum : hstep in linear form (a*f(k+1) + b*f k + d(k) = 0, a = -b) -> {v0, S(k), deg}
#   bridge_ind_geo : same form but a != -b -> {v0, r, c, c1, c0, cls}
#   bridge_ineq    : goal (LHS >= D)/(LHS <= D) -> {vars, dom = non-negative side minus the other side}
#   bridge_nt      : goal (M | body) + hypotheses (h_i : <modulus text> | v) + definitions (h : u = expr) -> structured
# All are **text-level parses** (no generator internal state), used by the round-trip consistency gate to
# recompute (statement text -> params == generation params).
#
# Obligation split (change-49 alignment): the bridge derives only "arguments derivable from the statement"
#   (coefficients / variables / modulus / body text); the **class** (semantic degree / perfect square /
#   degenerate / terminal semantic class / m-adic valuation / step type) is chosen by the model; the bridge
#   can assemble arguments for any candidate class but **never makes the class decision for the model**.
import re
from fractions import Fraction

import bh_poly as P

HSTEP = re.compile(r"\(\s*hstep\s*:\s*\u2200\s*k\s*,", re.S)


def split_concl(text):
    """Statement text -> (header, concl): a line-trailing ':' at bracket depth 0 separates the theorem body."""
    depth, cands, last = 0, [], -1
    for i, ch in enumerate(text):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == ":" and depth == 0:
            last = i
            nl = text.find("\n", i)
            tail = text[i + 1: nl if nl >= 0 else len(text)]
            if not tail.strip():
                cands.append(i)
    pos = cands[-1] if cands else last
    if pos < 0:
        raise ValueError("无结论分隔符")
    return text[:pos].strip(), re.sub(r"\s+", " ", text[pos + 1:]).strip()


def _norm_f(text):
    """Normalize f (k + 1) / f k to the atoms f1 / f0."""
    t = re.sub(r"f\s*\(\s*k\s*\+\s*1\s*\)", "f1", text)
    t = re.sub(r"f\s*\(\s*1\s*\+\s*k\s*\)", "f1", t)
    t = re.sub(r"f\s*\(\s*\(k\s*:\s*\u2115\)\s*\+\s*1\s*\)", "f1", t)
    t = re.sub(r"f\s+k", "f0", t)
    t = re.sub(r"f\s*\(\s*k\s*\)", "f0", t)
    return t


def _hstep_text(stmt):
    m = HSTEP.search(stmt)
    if not m:
        raise ValueError("hstep 未匹配")
    # The hstep binder block ends at the ')' closing the header (which ends with ') :'); that ')' is not part of the step text
    sep = stmt.index(":\n") if ":\n" in stmt else len(stmt)
    head_end = stmt.rindex(")", 0, sep)
    seg = re.sub(r"\s+", " ", stmt[m.end(): head_end]).strip()
    return seg


def _h0_val(stmt):
    m = re.search(r"\(\s*h0\s*:\s*f\s+0\s*=\s*(-?\d+)\s*\)", stmt)
    if not m:
        raise ValueError("h0 未匹配")
    return int(m.group(1))


def bridge_ind_linform(stmt):
    """hstep -> {v0, a, b, d(k) polynomial, kind}; kind in {additive, geometric}.
    Equation form: a*f(k+1) + b*f k + d(k) = 0 (a != 0)."""
    v0 = _h0_val(stmt)
    step = _hstep_text(stmt)
    step = re.sub(r"^\u2200\s*k\s*,\s*", "", step).strip()
    if "=" not in step:
        raise ValueError("步进式无等号: %r" % step)
    i = step.index("=")
    lhs, rhs = _norm_f(step[:i]), _norm_f(step[i + 1:])
    E = P.psub(P.parse_poly(lhs, ["f1", "f0", "k"]), P.parse_poly(rhs, ["f1", "f0", "k"]))
    a = sum((c for m, c in E.items() if m[0] == 1 and m[1] == 0 and m[2] == 0), Fraction(0))
    b = sum((c for m, c in E.items() if m[0] == 0 and m[1] == 1 and m[2] == 0), Fraction(0))
    if a == 0:
        raise ValueError("f(k+1) 系数为 0")
    d = {}
    for m, c in E.items():
        if m[0] > 1 or m[1] > 1 or (m[0] and m[1]):
            raise ValueError("非线性步进 (f1/f0 高次): %s" % (m,))
        if m[0] or m[1]:
            if m[2] != 0:
                raise ValueError("f 系数依赖 k (非本族): %s" % (m,))
            continue
        d[(m[2],)] = d.get((m[2],), Fraction(0)) + c
    d = {m: c for m, c in d.items() if c != 0}
    kind = "additive" if (a + b) == 0 else "geometric"
    return {"v0": v0, "a": a, "b": b, "d": d, "kind": kind, "step_raw": step}


def bridge_ind_sum(stmt):
    """Additive family: f(k+1) = f k + S(k) (may be scaled by c / rearranged). Returns {v0, S, deg}."""
    lf = bridge_ind_linform(stmt)
    if lf["kind"] != "additive":
        raise ValueError("非加法族 (a+b≠0)")
    S = P.pscale(lf["d"], Fraction(-1) / lf["a"])
    deg = max([m[0] for m in S] or [0])
    return {"v0": lf["v0"], "S": S, "deg": deg}


def bridge_ind_geo(stmt):
    """Geometric family: f(k+1) = r*f k + c. Requires constant d and integer r. Returns {v0, r, c, c1, c0, cls}."""
    lf = bridge_ind_linform(stmt)
    if lf["kind"] != "geometric":
        raise ValueError("非几何族 (a+b=0)")
    d = lf["d"]
    if any(m[0] != 0 for m in d):
        raise ValueError("几何族 d 必须为常数")
    dconst = d.get((0,), Fraction(0))
    r = Fraction(-lf["b"]) / lf["a"]
    c = Fraction(-dconst) / lf["a"]
    if r.denominator != 1 or c.denominator != 1:
        raise ValueError("r/c 非整数")
    r, c = int(r), int(c)
    if r == 1:
        raise ValueError("r=1 非本族")
    if c % (1 - r) != 0:
        raise ValueError("不动点非整数")
    c0 = Fraction(c, 1 - r)
    c1 = Fraction(lf["v0"]) - c0
    return {"v0": lf["v0"], "r": r, "c": c, "c1": c1, "c0": c0,
            "cls": "const" if c1 == 0 else "pow"}


def bridge_ind_declared(stmt, vars_=("n",)):
    """Closed-form text declared in the conclusion (verbatim) + normalized polynomial (for the equivalence gate)."""
    _, concl = split_concl(stmt)
    m = re.search(r"\u2200\s*n\s*:\s*\u2115\s*,\s*f\s*n\s*=\s*(.+?)\s*$", concl)
    if not m:
        raise ValueError("结论未匹配")
    txt = m.group(1).strip()
    return txt, P.parse_poly(txt, list(vars_))


def bridge_ineq(stmt, vars_):
    """Goal (LHS >= D) / (LHS <= D) -> {vars, dom}. dom = the non-negative side minus the other side."""
    _, concl = split_concl(stmt)
    mm = re.search(r"\s*(\u2265|\u2264|=)\s*", concl)
    if not mm:
        raise ValueError("无关系符: %r" % concl)
    lhs, rhs = concl[:mm.start()].strip(), concl[mm.end():].strip()
    rel = mm.group(1)
    pl, pr = P.parse_poly(lhs, vars_), P.parse_poly(rhs, vars_)
    if rel == "\u2265":
        dom = P.psub(pl, pr)
    elif rel == "\u2264":
        dom = P.psub(pr, pl)
    else:
        raise ValueError("非序关系")
    return {"vars": list(vars_), "dom": dom, "lhs": pl, "D": pr, "rel": rel}


HYPPAT = re.compile(r"\(\s*(\w+)\s*:\s*([0-9+\-*^() ]+?)\s*\u2223\s*([a-zA-Z][A-Za-z0-9_']*)\s*\)")
DEFPAT = re.compile(r"\(\s*(\w+)\s*:\s*([a-zA-Z][A-Za-z0-9_']*)\s*=\s*([^()]*(?:\([^()]*\)[^()]*)*)\)")


def bridge_nt(stmt):
    """Statement text -> {M, hyps:[{name, mod_text, mod_val, var}], defs:[{name, lhs, rhs_text}],
    binders:[...], body_text, body_poly?}."""
    header, concl = split_concl(stmt)
    m = re.fullmatch(r"(\d+)\s*\u2223\s*(.+)", concl)
    if not m:
        raise ValueError("目标未匹配: %r" % concl)
    M = int(m.group(1))
    body = re.sub(r"\s+", " ", m.group(2)).strip()
    hyps = []
    for mm in HYPPAT.finditer(header):
        hyps.append({"name": mm.group(1), "mod_text": mm.group(2).strip(),
                     "mod_val": None, "var": mm.group(3)})
    defs = []
    for mm in DEFPAT.finditer(header):
        defs.append({"name": mm.group(1), "lhs": mm.group(2),
                     "rhs_text": re.sub(r"\s+", " ", mm.group(3)).strip()})
    binders = []
    for mm in re.finditer(r"\(([^()]*?)\s*:\s*\u2115\)", header):
        for nm in mm.group(1).split():
            if re.fullmatch(r"[a-zA-Z][A-Za-z0-9_']*", nm):
                binders.append(nm)
    return {"M": M, "hyps": hyps, "defs": defs, "binders": binders, "body_text": body,
            "header": header, "concl": concl}


def body_vars(body_text, cand):
    """Candidate variables occurring in the body text (word boundaries)."""
    out = []
    for v in cand:
        if re.search(r"(?<![A-Za-z0-9_'])%s(?![A-Za-z0-9_'])" % re.escape(v), body_text):
            out.append(v)
    return out


def subst_var(body, var, repl):
    return re.sub(r"(?<![A-Za-z0-9_'])%s(?![A-Za-z0-9_'])" % re.escape(var),
                  "(%s)" % repl, body)


def roundtrip_ok(kind, stmt, params, tol_fields):
    """Round-trip gate: the bridge parse must match the generation params field by field
    (tolerance: polynomial / numeric text comparison)."""
    try:
        if kind == "ind_sum":
            br = bridge_ind_sum(stmt)
            return (br["v0"] == params["v0"] and br["deg"] == params["deg"]
                    and P.dump(br["S"]) == P.dump(params["S"]))
        if kind == "ind_geo":
            br = bridge_ind_geo(stmt)
            return all(br[k] == params[k] for k in ("v0", "r", "c", "c1", "c0", "cls"))
        if kind == "ineq":
            br = bridge_ineq(stmt, params["vars"])
            return P.dump(br["dom"]) == P.dump(params["dom"])
        if kind == "nt":
            br = bridge_nt(stmt)
            return (br["M"] == params["M"] and br["body_text"] == params["body_text"]
                    and len(br["hyps"]) == len(params["hyps"])
                    and len(br["defs"]) == len(params["defs"]))
    except Exception as e:                                    # noqa: BLE001
        return "EXC:%s" % type(e).__name__
    return False
