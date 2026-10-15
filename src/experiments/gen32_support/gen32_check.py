# -*- coding: utf-8 -*-
# gen32_check.py -- gen32 independent checker (layer-III long-chain deep-reasoning problem family batch)
#
# Prohibition (structural, mirrors bh_check): this module **must not import gen32_ops / gen32_poly**;
#   register_checker verifies fn.__module__ at registration time => self-certification is structurally rejected.
# The checker consumes only the certificate payload (pure data) and recomputes with an algorithm **different from
#   the generator's**:
#   * generator (gen32_ops): constructive -- difference-equation solving / folding / declared-root re-squaring / forward construction
#   * checker (this module): **grid substitution + forward difference simulation + independent polynomial square-root decision + exact purge absorption**
# Completeness convention: every polynomial identity is recomputed on (bound+1)^nv distinct grid points under the
#   **declared degree bound**; out-of-bound is fail-closed.
import re
from fractions import Fraction

CHECKERS = {}
_INDEP = {}
COEF_BOUND = Fraction(10) ** 18          # coefficient absolute upper bound (fail-closed)


def register_checker(kind, independent_of):
    def deco(fn):
        if fn.__module__ in tuple(independent_of):
            raise ValueError("self-certification violation: checker %s shares module %s with the generator" % (kind, fn.__module__))
        CHECKERS[kind] = fn
        _INDEP[kind] = tuple(independent_of)
        return fn
    return deco


def check(kind, payload):
    fn = CHECKERS.get(kind)
    if fn is None:
        return False, [1.0], "unregistered certificate type %r" % (kind,)
    try:
        return fn(payload)
    except Exception as e:                                          # noqa: BLE001
        return False, [1.0], "checker exception %s: %s" % (type(e).__name__, e)


def independence(kind):
    return _INDEP.get(kind, ())


# ---------------------------------------------------------------- self-owned polynomial path (numeric)
def _fl(d):
    """dump -> {exponent tuple: coefficient}. Integer-valued coefficients **degrade to int** (fast path for grid recomputation: exact int arithmetic)."""
    out = {}
    for k, v in d.items():
        f = Fraction(v)
        out[tuple(int(x) for x in k.split(","))] = (int(f) if f.denominator == 1 else f)
    return out


def _nvar(p):
    return len(next(iter(p))) if p else 0


def _eval(p, vals):
    """Exact evaluation. Integer coefficients + integer grid points => int fast path (exact int arithmetic, same value as Fraction)."""
    s = 0
    for m, c in p.items():
        t = c
        for i, e in enumerate(m):
            if e:
                t = t * (vals[i] ** e)
        s = s + t
    return s


def _deg_ok(p, nv, bound):
    for m in p:
        if len(m) != nv or any(e > bound for e in m):
            return False
    return True


def _coef_ok(*polys):
    for p in polys:
        for c in p.values():
            if abs(Fraction(c)) > COEF_BOUND:
                return False
    return True


def _padd(p, q):
    out = dict(p)
    for m, c in q.items():
        out[m] = out.get(m, Fraction(0)) + c
    return {m: c for m, c in out.items() if c != 0}


def _pscale(p, k):
    f = Fraction(k)
    k = int(f) if f.denominator == 1 else f
    return {m: c * k for m, c in p.items() if c * k != 0}


def _pmul(p, q):
    out = {}
    for m1, c1 in p.items():
        for m2, c2 in q.items():
            if len(m1) != len(m2):
                raise ValueError("dimension mismatch")
            m = tuple(a + b for a, b in zip(m1, m2))
            out[m] = out.get(m, Fraction(0)) + c1 * c2
    return {m: c for m, c in out.items() if c != 0}


def _grid_eq(p, q, nv, bound, side=None):
    """Identical on (bound+1)^nv distinct grid points (both p and q must lie within the bound)."""
    if not (_deg_ok(p, nv, bound) and _deg_ok(q, nv, bound)):
        return False, "degree out of bound (bound=%d) or component count mismatch (nv=%d)" % (bound, nv)
    side = side or (bound + 1)
    pts = list(range(2, 2 + side))
    idx = [0] * max(1, nv)

    def rec(i):
        if i == nv:
            vals = [Fraction(pts[idx[j]]) for j in range(nv)]
            return _eval(p, vals) == _eval(q, vals)
        for k in range(side):
            idx[i] = k
            if not rec(i + 1):
                return False
        return True
    if nv == 0:
        return (True, "identity on the empty variable set") if p == q else (False, "grid mismatch")
    return (True, "grid identity") if rec(0) else (False, "grid mismatch")


def _isqrt(n):
    if n < 0:
        return None
    if n == 0:
        return 0
    r = int(round(n ** 0.5))
    for c in (r - 2, r - 1, r, r + 1, r + 2):
        if c >= 0 and c * c == n:
            return c
    return None


def _fsqrt(v):
    v = Fraction(v)
    if v < 0:
        return None
    rn, rd = _isqrt(abs(v.numerator)), _isqrt(v.denominator)
    if rn is None or rd is None:
        return None
    return Fraction(rn, rd)


def _poly_sqrt_deg1_or_2(T):
    """Univariate perfect-square decision (checker-owned: top-down coefficient matching + final identity recomputation)."""
    if not T:
        return True, {(0,): Fraction(0)}
    D = max(m[0] for m in T if len(m) == 1)
    if D % 2 != 0:
        return False, None
    m_ = D // 2
    top = sum((c for mm, c in T.items() if mm[0] == D), Fraction(0))
    s = _fsqrt(top)
    if s is None:
        return False, None
    root = {m_: s}
    for t in range(1, m_ + 1):
        i = m_ - t
        tgt = sum((c for mm, c in T.items() if mm[0] == D - t), Fraction(0))
        known = Fraction(0)
        for a in range(i, m_ + 1):
            b = D - t - a
            if (a == i and b == m_) or (a == m_ and b == i):
                continue
            if b < 0 or b > m_:
                continue
            if a in root and b in root:
                known += root[a] * root[b]
        root[i] = (tgt - known) / (2 * root[m_])
    cand = {tuple([i]): c for i, c in root.items() if c != 0}
    if _padd(_pmul(cand, cand), _pscale(T, -1)):
        return False, None
    return True, cand


# ================================================================ 1. first-order difference equation
@register_checker("gen32_solve_cert", independent_of=("gen32_ops", "gen32_poly"))
def chk_solve(payload):
    """Certificate {nv, deg, v0, S, P}: (1) the **exact** degree of S == deg and the leading coefficient is nonzero;
    (2) P(0) == v0; (3) telescoping (forward simulation): P(k+1) - P(k) == S(k), k = 0..deg+3;
    (4) degree ladder deg(P) == deg+1; (5) coefficient bound."""
    nv = int(payload["nv"])
    deg = int(payload["deg"])
    v0 = Fraction(payload["v0"])
    S, Pc = _fl(payload["S"]), _fl(payload["P"])
    r = []
    if nv != 1:
        return False, [1.0], "this criterion is 1-variable only (nv=%d)" % nv
    if not _coef_ok(S, Pc):
        return False, [1.0], "coefficient out of bound"
    sd = max([m[0] for m in S] or [0])
    sc = sum((c for m, c in S.items() if m[0] == deg), Fraction(0))
    ok = (sd == deg and sc != 0)
    r.append(0.0 if ok else 1.0)
    if not ok:
        return False, r, "exact degree of S != %d (measured %d, leading coefficient %s)" % (deg, sd, sc)
    p0 = _eval(Pc, [Fraction(0)])
    r.append(0.0 if p0 == v0 else 1.0)
    if p0 != v0:
        return False, r, "P(0)=%s != v0=%s" % (p0, v0)
    bad = []
    for k in range(0, deg + 4):
        if _eval(Pc, [Fraction(k + 1)]) - _eval(Pc, [Fraction(k)]) != _eval(S, [Fraction(k)]):
            bad.append(k)
    r.append(0.0 if not bad else 1.0)
    if bad:
        return False, r, "telescoping identity mismatch at k in %s" % bad
    pd = max([m[0] for m in Pc] or [0])
    r.append(0.0 if pd == deg + 1 else 1.0)
    if pd != deg + 1:
        return False, r, "deg(P)=%d != %d" % (pd, deg + 1)
    return True, r, "exact degree + initial value + telescoping + degree ladder all agree (deg=%d, %d forward simulation points)" % (deg, deg + 4)


# ================================================================ 2. folding (lower closed form -> upper increment)
@register_checker("gen32_fold_cert", independent_of=("gen32_ops", "gen32_poly"))
def chk_fold(payload):
    """Certificate {nv, scale, lower, upper}: (1) upper ≡ scale · lower (grid, bound = maxdeg+1);
    (2) scale != 0 (degenerate folding forbidden); (3) lower is a nonzero polynomial (empty folding forbidden); (4) coefficient bound."""
    nv = int(payload["nv"])
    scale = Fraction(payload["scale"])
    lo, up = _fl(payload["lower"]), _fl(payload["upper"])
    r = []
    if not _coef_ok(lo, up):
        return False, [1.0], "coefficient out of bound"
    if scale == 0:
        return False, [1.0], "degenerate folding scale=0"
    if not lo:
        return False, [1.0], "the lower object is the zero polynomial"
    bnd = max([m[0] for m in list(lo) + list(up)] or [0]) + 1
    ok, msg = _grid_eq(up, _pscale(lo, scale), 1, bnd)
    r.append(0.0 if ok else 1.0)
    if not ok:
        return False, r, "upper ≢ scale·lower (%s)" % msg
    return True, r, "folding identity holds (scale=%s, deg(lower)=%d)" % (
        scale, max([m[0] for m in lo] or [0]))


# ================================================================ 3. telescoping verification (ΔP ≡ S)
@register_checker("gen32_delta_cert", independent_of=("gen32_ops", "gen32_poly"))
def chk_delta(payload):
    """Certificate {nv, upper, inc}: forward difference simulation Δupper(k) == inc(k), k = 0..bnd;
    bnd = max(deg upper, deg inc) + 3; inc must also be a nonzero polynomial (empty verification forbidden)."""
    nv = int(payload["nv"])
    up, inc = _fl(payload["upper"]), _fl(payload["inc"])
    if nv != 1:
        return False, [1.0], "this criterion is 1-variable only"
    if not _coef_ok(up, inc):
        return False, [1.0], "coefficient out of bound"
    if not inc:
        return False, [1.0], "increment is the zero polynomial (empty verification)"
    bnd = int(payload.get("bnd") or (max([m[0] for m in list(up) + list(inc)] or [0]) + 3))
    bad = []
    for k in range(0, bnd + 1):
        if _eval(up, [Fraction(k + 1)]) - _eval(up, [Fraction(k)]) != _eval(inc, [Fraction(k)]):
            bad.append(k)
    if bad:
        return False, [1.0], "Δ mismatch at k in %s" % bad[:5]
    return True, [0.0], "Δ forward difference verification passed (%d points, deg upper=%d)" % (
        bnd + 1, max([m[0] for m in up] or [0]))


# ================================================================ 4. closed-form equivalence + class exclusivity
@register_checker("gen32_close_cert", independent_of=("gen32_ops", "gen32_poly"))
def chk_close(payload):
    """Certificate {nv, P, T, cls, sq?}: (1) P ≡ T (grid, bound = maxdeg+1);
    (2) class exclusivity (real computation): cls='poly' => T is not a perfect square and not constant; cls='sq2' =>
    T is a perfect square with a degree-1 root, and the **declared root re-squared** ≡ T; cls='sq4' => degree-2 root;
    cls='const' => T is constant and P is constant too."""
    Pd, Td = _fl(payload["P"]), _fl(payload["T"])
    cls = payload["cls"]
    nv = int(payload.get("nv") or (_nvar(Pd) or _nvar(Td) or 1))
    if not _coef_ok(Pd, Td):
        return False, [1.0], "coefficient out of bound"
    bnd = max([m[0] for m in list(Pd) + list(Td)] or [0]) + 1
    ok, msg = _grid_eq(Pd, Td, nv, bnd)
    r = [0.0 if ok else 1.0]
    if not ok:
        return False, r, "P ≢ T (%s)" % msg
    isconst = all(all(e == 0 for e in m) for m in Td)
    if cls == "const":
        if not isconst:
            return False, r + [1.0], "cls=const but T is not constant"
    elif cls == "poly":
        if isconst:
            return False, r + [1.0], "cls=poly but T is constant"
        isq, root = _poly_sqrt_deg1_or_2(Td)
        if isq:
            return False, r + [1.0], "cls=poly but T is a perfect square (exclusivity failed)"
    elif cls in ("sq2", "sq4"):
        isq, root = _poly_sqrt_deg1_or_2(Td)
        if not isq:
            return False, r + [1.0], "cls=%s but T is not a perfect square" % cls
        rdeg = max([m[0] for m in root] or [0])
        want = 1 if cls == "sq2" else 2
        if rdeg != want:
            return False, r + [1.0], "cls=%s requires root degree %d (measured %d)" % (cls, want, rdeg)
        if "sq" not in payload:
            return False, r + [1.0], "declared root missing"
        coefs = [Fraction(x) for x in payload["sq"]]
        wantp = {tuple([i]): c for i, c in enumerate(coefs) if c != 0}
        ok2, msg2 = _grid_eq(Td, _pmul(wantp, wantp), 1, max(bnd, len(coefs) * 2))
        if not ok2:
            return False, r + [1.0], "T ≢ square of the declared root (%s)" % msg2
    else:
        return False, r + [1.0], "illegal class %r" % cls
    r.append(0.0)
    return True, r, "P≡T and class exclusivity hold (cls=%s, constant=%s)" % (cls, isconst)


# ================================================================ 5. square-block absorption (sos round)
@register_checker("gen32_block_cert", independent_of=("gen32_ops", "gen32_poly"))
def chk_block(payload):
    """Certificate {vars, R_in, R_out, ell, mult}: (1) R_in ≡ mult·ℓ² + R_out (grid, bound=2:
    the polynomials involved in a block certificate are all sums of squares of linear forms => per-variable degree <=2;
    out-of-bound is fail-closed and rejected);
    (2) mult is a positive integer; (3) purge: R_out contains no nonzero term in any support variable of ℓ (exact
    absorption); (4) ℓ nonzero; (5) coefficient bound."""
    vars_ = list(payload["vars"])
    nv = len(vars_)
    R_in, R_out, ell = _fl(payload["R_in"]), _fl(payload["R_out"]), _fl(payload["ell"])
    _mf = Fraction(payload["mult"])
    mult = int(_mf) if _mf.denominator == 1 else _mf
    if not _coef_ok(R_in, R_out, ell):
        return False, [1.0], "coefficient out of bound"
    coefs = []
    for i in range(nv):
        m = [0] * nv
        m[i] = 1
        coefs.append(ell.get(tuple(m), 0))
    shift = -ell.get((0,) * nv, 0)
    lin = {}
    for i, a in enumerate(coefs):
        if a != 0:
            m = [0] * nv
            m[i] = 1
            lin[tuple(m)] = a
    if shift != 0:
        lin[(0,) * nv] = lin.get((0,) * nv, 0) - shift
    sq = _pmul(lin, lin)
    want = _padd(_pscale(sq, mult), R_out)
    ok, msg = _grid_eq(R_in, want, nv, 2)
    r = [0.0 if ok else 1.0]
    if not ok:
        return False, r, "R_in ≢ m·ℓ²+R_out (%s)" % msg
    if not (mult > 0) or Fraction(mult).denominator != 1:
        return False, r + [1.0], "mult=%s is not a positive integer" % mult
    supp = [i for i, a in enumerate(coefs) if a != 0]
    if not supp:
        return False, r + [1.0], "ℓ is constant (no support)"
    stray = [(",".join(map(str, m)), str(c)) for m, c in R_out.items()
             if c != 0 and any(m[i] > 0 for i in supp)]
    if stray:
        return False, r + [1.0], "purge failed: R_out retains absorbed variables %s" % stray[:3]
    r.append(0.0)
    return True, r, "block identity + positive multiplier + purge all hold (supp=%s mult=%s)" % (supp, mult)


# ================================================================ 6. closing semantic class
@register_checker("gen32_tail_cert", independent_of=("gen32_ops", "gen32_poly"))
def chk_tail(payload):
    """Certificate {rho, cls, sq?, vars, goal_dom, hints_poly}:
    (1) **closed-chain identity**: goal_dom ≡ Σ hints_poly + rho (grid, bound=2 -- goals/blocks are all sums of squares
    of linear forms, per-variable degree <=2; out-of-bound is fail-closed and rejected) => early closing / a missing
    block must fail;
    (2) class exclusivity (real computation): zero <=> rho==0; square <=> rho==sq² (sq>=2); other <=> rho>0 and not a perfect square."""
    rho = Fraction(payload["rho"])
    cls = payload["cls"]
    if not _coef_ok(_fl(payload["goal_dom"]), *[_fl(h) for h in payload["hints_poly"]]):
        return False, [1.0], "coefficient out of bound"
    r = []
    if rho < 0:
        return False, [1.0], "rho=%s < 0" % rho
    rho = int(rho) if rho.denominator == 1 else rho
    vars_ = list(payload["vars"])
    nv = len(vars_)
    acc = {(0,) * nv: rho}
    for h in payload["hints_poly"]:
        acc = _padd(acc, _fl(h))
    ok, msg = _grid_eq(acc, _fl(payload["goal_dom"]), nv, 2)
    r.append(0.0 if ok else 1.0)
    if not ok:
        return False, r, "closed-chain identity failed: Σblocks + rho ≢ goal_dom (%s)" % msg
    if cls == "zero":
        ok, msg = (rho == 0), "class zero requires rho==0"
    elif cls == "square":
        s = int(payload["sq"])
        ok = (rho == Fraction(s * s)) and s >= 2
        msg = "class square requires rho == sq^2 (sq=%d rho=%s)" % (s, rho)
    elif cls == "other":
        ps = _isqrt(int(rho)) if (rho > 0 and Fraction(rho).denominator == 1) else None
        ok = (rho > 0) and (ps is None)
        msg = "class other requires rho>0 and not a perfect square (rho=%s)" % rho
    else:
        return False, [1.0], "illegal class %r" % cls
    if not ok:
        return False, r + [1.0], msg
    r.append(0.0)
    return True, r, "closed-chain identity + closing class %s hold (rho=%s)" % (cls, rho)


# ---------------------------------------------------------------- self-test (non-vacuous truth)
def selftest():
    out = {}
    # S = 1 + 2k, P = n^2 + 3 (P(0)=3): ΔP = 2k+1
    out["solve_pos"] = chk_solve({"nv": 1, "deg": 1, "v0": "3",
                                  "S": {"0": "1", "1": "2"},
                                  "P": {"0": "3", "2": "1"}})[0]
    out["solve_perturbed"] = chk_solve({"nv": 1, "deg": 1, "v0": "3",
                                        "S": {"0": "1", "1": "2"},
                                        "P": {"0": "4", "2": "1"}})[0]
    out["solve_wrongdeg"] = chk_solve({"nv": 1, "deg": 2, "v0": "3",
                                       "S": {"0": "1", "1": "2"},
                                       "P": {"0": "3", "2": "1"}})[0]
    out["fold_pos"] = chk_fold({"nv": 1, "scale": "3", "lower": {"0": "1", "1": "2"},
                                "upper": {"0": "3", "1": "6"}})[0]
    out["fold_neg"] = chk_fold({"nv": 1, "scale": "3", "lower": {"0": "1", "1": "2"},
                                "upper": {"0": "3", "1": "5"}})[0]
    out["fold_zero_scale"] = chk_fold({"nv": 1, "scale": "0", "lower": {"0": "1"},
                                       "upper": {"0": "0"}})[0]
    out["delta_pos"] = chk_delta({"nv": 1, "upper": {"0": "3", "2": "1"},
                                  "inc": {"0": "1", "1": "2"}})[0]
    out["delta_neg"] = chk_delta({"nv": 1, "upper": {"0": "3", "2": "1"},
                                  "inc": {"0": "2", "1": "2"}})[0]
    sqT = {"0": "9", "1": "6", "2": "1"}                    # (n+3)^2
    out["close_sq2_pos"] = chk_close({"nv": 1, "P": sqT, "T": sqT, "cls": "sq2",
                                      "sq": ["3", "1"]})[0]
    out["close_poly_on_square"] = chk_close({"nv": 1, "P": sqT, "T": sqT,
                                             "cls": "poly"})[0]
    out["close_sq2_wrongroot"] = chk_close({"nv": 1, "P": sqT, "T": sqT, "cls": "sq2",
                                            "sq": ["3", "2"]})[0]
    q4 = {"0": "1", "1": "2", "2": "3", "3": "2", "4": "1"}   # (1+n+n^2)^2
    out["close_sq4_pos"] = chk_close({"nv": 1, "P": q4, "T": q4, "cls": "sq4",
                                      "sq": ["1", "1", "1"]})[0]
    out["close_sq2_on_quartic"] = chk_close({"nv": 1, "P": q4, "T": q4, "cls": "sq2",
                                             "sq": ["1", "1"]})[0]
    out["close_const_pos"] = chk_close({"nv": 1, "P": {"0": "7"}, "T": {"0": "7"},
                                        "cls": "const"})[0]
    # block: (x+y)^2 + (z+2)^2 + 5 -> R_in; after taking the block (x+y)^2, R_out = z^2+4z+9
    R_in = {"2,0,0": "1", "1,1,0": "2", "0,2,0": "1", "0,0,2": "1", "0,0,1": "4", "0,0,0": "9"}
    out["block_pos"] = chk_block({"vars": ["x", "y", "z"], "R_in": R_in,
                                  "R_out": {"0,0,2": "1", "0,0,1": "4", "0,0,0": "9"},
                                  "ell": {"1,0,0": "1", "0,1,0": "1"}, "mult": "1"})[0]
    out["block_purge_neg"] = chk_block({"vars": ["x", "y", "z"], "R_in": R_in,
                                        "R_out": {"0,0,2": "1", "0,0,1": "4", "0,0,0": "7",
                                                  "1,0,0": "2", "0,1,0": "1"},
                                        "ell": {"1,0,0": "1", "0,1,0": "1", "0,0,0": "-1"},
                                        "mult": "1"})[0]
    # tail: dom = (x+y)^2 + 25, block = (x+y)^2, rho = 25 (square, sq=5)  [2 vars => key length 2]
    dom = {"2,0": "1", "1,1": "2", "0,2": "1", "0,0": "25"}
    blk = {"2,0": "1", "1,1": "2", "0,2": "1"}
    out["tail_square_pos"] = chk_tail({"vars": ["x", "y"], "rho": "25", "cls": "square",
                                       "sq": 5, "goal_dom": dom, "hints_poly": [blk]})[0]
    out["tail_square_neg"] = chk_tail({"vars": ["x", "y"], "rho": "25", "cls": "square",
                                       "sq": 4, "goal_dom": dom, "hints_poly": [blk]})[0]
    out["tail_other_neg"] = chk_tail({"vars": ["x", "y"], "rho": "25", "cls": "other",
                                      "goal_dom": dom, "hints_poly": [blk]})[0]
    out["tail_missing_block"] = chk_tail({"vars": ["x", "y"], "rho": "25", "cls": "zero",
                                          "goal_dom": dom, "hints_poly": []})[0]
    return out


if __name__ == "__main__":
    import json
    st = selftest()
    print(json.dumps(st, ensure_ascii=False, indent=1))
    n_pass = sum(1 for v in st.values() if v)
    n_neg = sum(1 for k, v in st.items()
                if k.endswith(("neg", "perturbed", "wrongdeg", "wrongroot",
                               "on_square", "on_quartic", "zero_scale", "missing_block")))
    print("SELFTEST pass=%d/%d | expected-negative %d entries all False=%s"
          % (n_pass, len(st), n_neg,
             all(not st[k] for k in st if k.endswith(
                 ("_neg", "perturbed", "wrongdeg", "wrongroot", "on_square",
                  "on_quartic", "zero_scale", "missing_block")))))
