# -*- coding: utf-8 -*-
# ccp_check.py -- CCP independent checker (advance team gen12, constitution change 58)
#
# Ban: this module **must never import ccp_ops** (generator and checker follow separate paths,
#      mirroring the independent_of structural constraint of svokernel.certify). The checker
#      consumes only the certificate payload (pure data) and recomputes with **different
#      algorithms** than the generator:
#   * recurrence class: forward simulation + step-operator action (generator fits two points
#     a(0)/a(1); checker uses simulation / fixed point)
#   * polynomial-identity class: deterministic grid identity test (per-variable degree <= d =>
#     d+1 distinct grid points; all-equal means identity)
#   * divisibility class: pointwise integer substitution + modulus-condition recomputation
#     (generator goes through coefficient construction)
import hashlib
from fractions import Fraction


def register_checker(kind, independent_of):
    def deco(fn):
        if fn.__module__ in independent_of:
            raise ValueError("禁自证违规: 检查器 %s 与生成器同模块 %s" % (kind, fn.__module__))
        CHECKERS[kind] = fn
        _INDEP[kind] = tuple(independent_of)
        return fn
    return deco


CHECKERS = {}
_INDEP = {}


def check(kind, payload):
    fn = CHECKERS.get(kind)
    if fn is None:
        return False, [1.0], "未登记证书类型 %r" % (kind,)
    try:
        return fn(payload)
    except Exception as e:                                   # noqa: BLE001
        return False, [1.0], "检查器异常 %s: %s" % (type(e).__name__, e)


def independence(kind):
    return _INDEP.get(kind, ())


# ---------------------------------------------------------------- polynomial helpers (checker's own path)
def poly_mul(p, q):
    out = {}
    for m1, c1 in p.items():
        for m2, c2 in q.items():
            if len(m1) != len(m2):
                raise ValueError("维度不一致")
            m = tuple(a + b for a, b in zip(m1, m2))
            out[m] = out.get(m, Fraction(0)) + c1 * c2
    return {m: c for m, c in out.items() if c != 0}


def poly_add(p, q):
    out = dict(p)
    for m, c in q.items():
        out[m] = out.get(m, Fraction(0)) + c
    return {m: c for m, c in out.items() if c != 0}


def poly_scale(p, k):
    k = Fraction(k)
    return {m: c * k for m, c in p.items() if c * k != 0}


def poly_eval(p, vals):
    s = Fraction(0)
    for m, c in p.items():
        t = c
        for i, e in enumerate(m):
            if e:
                t *= Fraction(vals[i]) ** e
        s += t
    return s


def deg_ok(p, nvars, bound):
    for m in p:
        if len(m) != nvars or any(e > bound for e in m):
            return False
    return True


def grid_identity_test(p, q, nvars, deg_per_var, side=None):
    """Deterministic grid identity test (two polynomials of degree <= d that agree on d+1
    distinct values per variable are identical). The degree bound is asserted first; out of
    bounds means FAIL (guards against grid-miss false positives)."""
    side = side or (deg_per_var + 1)
    if not (deg_ok(p, nvars, deg_per_var) and deg_ok(q, nvars, deg_per_var)):
        return False, "次数越界 (deg_per_var=%d)" % deg_per_var
    diff = poly_add(p, poly_scale(q, -1))
    pts = list(range(2, 2 + side))
    idx = [0] * nvars

    def rec(i):
        if i == nvars:
            vals = [Fraction(pts[idx[j]]) for j in range(nvars)]
            return poly_eval(diff, vals) == 0
        for k in range(side):
            idx[i] = k
            if not rec(i + 1):
                return False
        return True

    return (True, "网格全等") if rec(0) else (False, "格点失配")


def _fl(p):
    """{'m': coef} dicts in the payload (key = exponent-tuple string 'a,b') -> polynomial dict."""
    return {tuple(int(x) for x in k.split(",")): Fraction(v) for k, v in p.items()}


def _dump(p):
    return {",".join(str(e) for e in m): str(c) for m, c in p.items()}


# ---------------------------------------------------------------- recurrence class
def _sim_forward(form, v0, step, n):
    a = int(v0)
    if form == "lin":
        d = int(step)
        for _ in range(n):
            a += d
        return Fraction(a)
    if form == "aff":
        r, c = int(step[0]), int(step[1])
        for _ in range(n):
            a = r * a + c
        return Fraction(a)
    raise ValueError(form)


def _fixed_point_coords(form, v0, step):
    """Checker-route coordinates: fixed point / difference method (shares no code with the
    generator's two-point fit)."""
    if form == "lin":
        d = int(step)
        return {"c1": Fraction(d), "c0": Fraction(int(v0))}
    if form == "aff":
        r, c = int(step[0]), int(step[1])
        B = Fraction(c, 1 - r)
        return {"c1": Fraction(int(v0)) - B, "c0": B}
    raise ValueError(form)


def _eval_coords(form, coords, step, n):
    if form == "lin":
        return coords["c1"] * Fraction(n) + coords["c0"]
    if form == "aff":
        return coords["c1"] * Fraction(int(step[0])) ** n + coords["c0"]
    raise ValueError(form)


def _step_apply(form, coords, step):
    """Step-operator action: lin (c1,c0)->(c1, c0+c1); aff(r,c) (A,B)->(rA, rB+c)."""
    if form == "lin":
        return {"c1": coords["c1"], "c0": coords["c0"] + coords["c1"]}
    if form == "aff":
        r, c = int(step[0]), int(step[1])
        return {"c1": r * coords["c1"], "c0": r * coords["c0"] + c}
    raise ValueError(form)


@register_checker("ccp_ind_recurrence_cert", independent_of=("ccp_ops",))
def chk_ind_recurrence(payload):
    """Certificate: (1) forward simulation of n steps == declared closed-form value;
    (2) fixed-point coordinates == declared coordinates; (3) one step-operator action ==
    extrapolation at n+1."""
    form, v0, step = payload["form"], int(payload["v0"]), payload["step"]
    n = int(payload["n"])
    closes = payload["closed"]
    step_v = int(step) if form == "lin" else [int(step[0]), int(step[1])]
    r = []
    sim = _sim_forward(form, v0, step_v, n)
    d1 = abs(sim - Fraction(closes["value"]))
    r.append(float(min(1, d1)))
    if d1 != 0:
        return False, r, "前向模拟 %s ≠ 声明 %s" % (sim, closes["value"])
    co = _fixed_point_coords(form, v0, step_v)
    for k in ("c1", "c0"):
        d = abs(co[k] - Fraction(closes[k]))
        r.append(float(min(1, d)))
        if d != 0:
            return False, r, "坐标 %s: 不动点解 %s ≠ 声明 %s" % (k, co[k], closes[k])
    lhs = _eval_coords(form, _step_apply(form, co, step_v), step_v, n)
    rhs = _eval_coords(form, co, step_v, n + 1)
    d3 = abs(lhs - rhs)
    r.append(float(min(1, d3)))
    if d3 != 0:
        return False, r, "步进不自洽: %s ≠ %s" % (lhs, rhs)
    return True, r, "模拟+不动点+步进三重一致 (form=%s n=%d)" % (form, n)


# ---------------------------------------------------------------- SOS
def _expand_squares(squares, nvars):
    tot = {}
    zero = (0,) * nvars
    for sq in squares:
        poly = {}
        for i, a in enumerate(sq["coefs"]):
            if Fraction(a) != 0:
                m = [0] * nvars
                m[i] = 1
                poly[tuple(m)] = Fraction(a)
        poly[zero] = poly.get(zero, Fraction(0)) - Fraction(sq["shift"])
        tot = poly_add(tot, poly_scale(poly_mul(poly, poly), sq["mult"]))
    return tot


@register_checker("ccp_ineq_sos_cert", independent_of=("ccp_ops",))
def chk_ineq_sos(payload):
    """SOS certificate: (1) independently expand Σ m_j·(linear form)² by convolution and check
    input_dom == expansion + ρ (grid identity); (2) all multipliers m_j > 0 (hint legality);
    (3) ρ >= 0."""
    nvars = len(payload["vars"])
    squares = payload["squares"]
    zero = (0,) * nvars
    S = _expand_squares(squares, nvars)
    rho = Fraction(payload["residual"])
    S = poly_add(S, {zero: rho})
    r = []
    ok1, msg1 = grid_identity_test(_fl(payload["input_dom"]), S, nvars, deg_per_var=2, side=3)
    r.append(0.0 if ok1 else 1.0)
    if not ok1:
        return False, r, "input_dom ≠ Σ平方 + ρ (%s)" % msg1
    bad = [sq["mult"] for sq in squares if not (Fraction(sq["mult"]) > 0)]
    r.append(float(len(bad)))
    if bad:
        return False, r, "非正乘子 %s" % bad
    if rho < 0:
        r.append(1.0)
        return False, r, "余项 ρ=%s < 0" % rho
    r.append(0.0)
    return True, r, "SOS 展开=输入多项式+ρ 且乘子全正 (nvars=%d nsq=%d ρ=%s)" % (
        nvars, len(squares), rho)


@register_checker("ccp_ineq_close_cert", independent_of=("ccp_ops",))
def chk_ineq_close(payload):
    """Closed-chain certificate: goal_dom obtained by **re-parsing the statement** == Σ squares + ρ
    (grid identity) and ρ >= 0 (the arithmetic core of the proposition "the hint combination
    suffices to derive the goal")."""
    nvars = len(payload["vars"])
    zero = (0,) * nvars
    S = poly_add(_expand_squares(payload["squares"], nvars), {zero: Fraction(payload["residual"])})
    ok, msg = grid_identity_test(_fl(payload["goal_dom"]), S, nvars, deg_per_var=2, side=3)
    r = [0.0 if ok else 1.0]
    if not ok:
        return False, r, "goal_dom ≠ Σ平方 + ρ (%s)" % msg
    if Fraction(payload["residual"]) < 0:
        r.append(1.0)
        return False, r, "ρ < 0"
    r.append(0.0)
    return True, r, "闭链恒等成立 (nvars=%d, ρ=%s)" % (nvars, payload["residual"])


# ---------------------------------------------------------------- divisibility class
@register_checker("ccp_nt_dvd_cert", independent_of=("ccp_ops",))
def chk_nt_dvd(payload):
    """Generalized divisibility-closure certificate: on the parameter space
    Z = {w_v (one per assumption variable)} ∪ {free variables},
       G(Z) := Σ coef·(m_h^{k}·Π w) + coef·(m_h·free var) + K      (after substituting v := m_h·w_v)
       W(Z) := witness polynomial
    Independent criterion: G ≡ M·W (deterministic grid identity; degree <= 2 per variable =>
    3 points/variable suffice), and all witness coefficients are nonnegative integers.
    The checker **rebuilds G itself** from terms/const (does not reuse the generator's
    construction path)."""
    m_h, M = int(payload["m_h"]), int(payload["M"])
    params = list(payload["params"])
    idx = {n: i for i, n in enumerate(params)}
    nv = len(params)
    zero = (0,) * nv
    if m_h <= 1 or M <= 1 or M % m_h != 0:
        return False, [1.0], "非法模数对 (m_h=%d, M=%d)" % (m_h, M)
    G = {}
    for t in payload["terms"]:
        coef = Fraction(t["coef"])
        if t.get("mfac"):
            mon = [0] * nv
            if t.get("freevar") is None or t["freevar"] not in idx:
                return False, [1.0], "自由变量缺失: %s" % t
            mon[idx[t["freevar"]]] += 1
            fac = Fraction(m_h)
        else:
            mon = [0] * nv
            for wn in t["wvars"]:
                if wn not in idx:
                    return False, [1.0], "见证变量未登记: %s" % wn
                mon[idx[wn]] += 1
            fac = Fraction(m_h) ** int(t["k"])
        key = tuple(mon)
        G[key] = G.get(key, Fraction(0)) + coef * fac
    G[zero] = G.get(zero, Fraction(0)) + Fraction(payload["const"])
    W = {}
    for t in payload["witness"]:
        c = Fraction(t["coef"])
        if c.denominator != 1 or c < 0:
            return False, [1.0], "见证系数非自然数: %s" % c
        mon = [0] * nv
        for wn in t["monomial"]:
            if wn not in idx:
                return False, [1.0], "见证单项式变量未登记: %s" % wn
            mon[idx[wn]] += 1
        key = tuple(mon)
        W[key] = W.get(key, Fraction(0)) + c
    ok, msg = grid_identity_test(G, poly_scale(W, M), nv, deg_per_var=2, side=3)
    return (ok, [0.0 if ok else 1.0],
            "广义整除恒等 G ≡ %d·W %s (m_h=%d M=%d params=%s)" % (M, msg, m_h, M, params))


# ---------------------------------------------------------------- symbolic ring identity
@register_checker("ccp_ring_identity_cert", independent_of=("ccp_ops",))
def chk_ring_identity(payload):
    """Left/right polynomial dicts -> grid identity test."""
    nvars = int(payload["nvars"])
    side = int(payload.get("side", 4))
    ok, msg = grid_identity_test(_fl(payload["lhs"]), _fl(payload["rhs"]), nvars,
                                 deg_per_var=side - 1, side=side)
    return ok, [0.0 if ok else 1.0], "环恒等 %s (%s)" % (ok, msg)


# ---------------------------------------------------------------- structural step (bridge-rule shape)
@register_checker("ccp_shape_cert", independent_of=("ccp_ops",))
def chk_shape(payload):
    """Structural step: the checker re-parses the **text** to verify the shape (does not trust
    the generator's internal state)."""
    import re
    txt = payload["text"]
    kind = payload["assert"]
    if kind == "dvd_hyp":
        pat = r"\(%s\s*:\s*%d\s*∣\s*([a-z][A-Za-z0-9_']*)\)" % (re.escape(payload["hyp"]), int(payload["m"]))
        mm = re.search(pat, txt)
        if not mm:
            return False, [1.0], "假设形状不符: %s" % pat
        return True, [0.0], "假设形状确认 %s" % mm.group(0)
    if kind == "dvd_goal":
        mm = re.search(r"⊢\s*%d\s*∣\s*(.+)$" % int(payload["m"]), txt)
        if not mm:
            return False, [1.0], "目标形状不符"
        return True, [0.0], "目标形状确认 %s" % mm.group(1)[:80]
    if kind == "ineq_goal":
        mm = re.search(r"⊢\s*(.+?)\s*(≥|≤)\s*(.+)$", txt)
        if not mm:
            return False, [1.0], "不等式目标形状不符"
        return True, [0.0], "不等式目标 %s %s %s" % (mm.group(1)[:40], mm.group(2), mm.group(3)[:40])
    if kind == "ind_goal":
        mm = re.search(r"⊢\s*∀\s*\(?([a-z][A-Za-z0-9_']*)\s*:\s*ℕ\)?,", txt)
        if not mm:
            return False, [1.0], "归纳目标形状不符"
        return True, [0.0], "归纳目标 (∀ %s : ℕ)" % mm.group(1)
    return False, [1.0], "未知断言 %r" % kind


def ledger_digest(items):
    h = hashlib.md5()
    for it in sorted(items):
        h.update(("%s|%s\n" % (it[0], it[1])).encode("utf-8"))
    return h.hexdigest()
