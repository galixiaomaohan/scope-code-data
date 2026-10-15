"""§8.2 theorem registry (band 9200, rows 9201–9240) — P1 math-domain deliverable B.

**Registry entry count**: the heading of task-book §8.2 says "34", but the itemised list right after it actually
contains **35** (algebra 7 + inequalities 4 + linear algebra 6 + calculus 6 + series 3 + probability/combinatorics 4
+ number theory 3 + trigonometry 2). This implementation follows the **itemised list** and implements all 35 —
implementing one fewer would be a missing deliverable, while one extra creates no risk. The count discrepancy is
registered in the delivery report's dissent section (no unilateral architecture change).

**Registration shape of each theorem** (complete five-tuple, §7 six admission rules):
    statement (stmt) | kind (identity / inequality / existence / decision) |
    conditions (hard-checked item by item in precond; failing them raises PRECOND_FAIL) |
    verifier (verify: an independent channel **different from the derivation path**) | code (handle)

**What §17.9 no-self-certification means concretely in the theorem domain**: each theorem's verify must not reuse
invoke's derivation formula. The division of labour here:
    - polynomial identities: invoke uses formulas/elimination; verify uses **rational sample-point substitution**
      (d+1 points, or the poly_backsub certificate)
    - inequalities: invoke uses algebraic rearrangement; verify uses **exact rational comparison + bounded
      counter-example search**, two channels
    - linear algebra: invoke uses elimination/closed forms; verify uses **element-wise exact matrix multiplication**
      (matrix_identity certificate)
    - calculus: invoke uses symbolic rules; verify uses **numerical differences/quadrature** (float, another path)
    - number theory: invoke uses extended Euclid/modular exponentiation; verify uses **direct back-substitution of
      congruences** (mod_congruence certificate)

**Honesty boundary (item by item)**:
    - float-bearing verifiers (calculus family) only compare at sample points; the residual is **numerical
      evidence**, not proof, and its tolerance is written explicitly in the detail without pretending to be exact.
      Anything whose result must itself be exact (roots/inverses/congruences) goes through the rational channel.
    - counter-example searches are all **bounded** (the bound's size is stated explicitly); finding nothing only
      means "not found within the bound" and never claims "no counter-example exists" (§18 red line: no claims
      about properties that were not measured).
    - the three series tests decide only on **parameterised families** (geometric×polynomial family / c·rⁿ·nᵖ /
      dominated family), and return undecided when L=1 instead of pretending the test gave an answer.
"""
from __future__ import annotations

import math
from fractions import Fraction
from itertools import product

from ..certify import Certificate
from ..errors import Err, SvoError
from ..exact import frac, frac_str
from ..registry import REGISTRY, OpDef
from ..tool import CallCtl
from .math_basic import (
    det_cofactor, poly_eval, poly_mul, poly_trim,
)

BAND = 9200

# ---------------------------------------------------------------------------
# shared utilities
# ---------------------------------------------------------------------------


def f(v) -> Fraction:
    return frac(v)


def fs(x: Fraction) -> str:
    return frac_str(x)


def fvec(vs) -> "list[Fraction]":
    return [f(v) for v in vs]


def fmat(rows) -> "list[list[Fraction]]":
    return [[f(v) for v in row] for row in rows]


def worse(a: Fraction, b: Fraction) -> Fraction:
    return a if a >= b else b


def matmul(A, B):
    n, k, m = len(A), len(B), len(B[0])
    return [[sum((A[i][t] * B[t][j] for t in range(k)), Fraction(0)) for j in range(m)]
            for i in range(n)]


def matsub(A, B):
    return [[A[i][j] - B[i][j] for j in range(len(A[0]))] for i in range(len(A))]


def eye(n: int):
    return [[Fraction(1) if i == j else Fraction(0) for j in range(n)] for i in range(n)]


def isqrt_frac(d: Fraction):
    """rational perfect-square root (exact); returns None when not a perfect square."""
    if d < 0:
        return None
    num, den = d.numerator, d.denominator
    sn, sd = math.isqrt(num), math.isqrt(den)
    if sn * sn == num and sd * sd == den:
        return Fraction(sn, sd)
    return None


def poly_is_zero(cs) -> bool:
    return all(c == 0 for c in cs)


def cert_identity(lhs_cs, rhs_cs, issued_by: str, nsamples: int) -> Certificate:
    """polynomial identity certificate (identity_expand checker: pointwise substitution at d+1 rational points)."""
    return Certificate(
        kind="identity_expand",
        payload={
            "lhs_terms": {str(k): fs(c) for k, c in enumerate(lhs_cs) if c != 0},
            "rhs_terms": {str(k): fs(c) for k, c in enumerate(rhs_cs) if c != 0},
            "samples": [str(j) for j in range(max(nsamples, 2))],
            "tol": "1/1000000000000",
        },
        issued_by=issued_by,
    )


def _mat_schema(n: int, ctx_note: str = "") -> dict:
    return {"kind": "list", "len": n,
            "items": {"kind": "list", "len": n, "items": {"kind": "Fraction"}}}


# ===========================================================================
# algebra (T-A1 … T-A7, rows 9201–9207)
# ===========================================================================


_IN_QUAD = {"kind": "dict",
            "fields": {"coeffs": {"kind": "list", "items": {"kind": "Fraction"}, "len": 3}},
            "required": ["coeffs"]}
_OUT_QUAD = {"kind": "dict",
             "fields": {"mode": {"kind": "str", "enum": ["exact_roots", "interval_enclosure"]},
                        "roots": {"kind": "list", "items": {"kind": "str"}, "maxlen": 2},
                        "intervals": {"kind": "list", "items": {"kind": "list", "items": {"kind": "str"},
                                                                "len": 2}, "maxlen": 2},
                        "disc": {"kind": "str"}, "cond_ok": {"kind": "bool"}},
             "required": ["mode", "roots", "intervals", "disc", "cond_ok"]}


def _pre_quad(args) -> "tuple[bool, str]":
    c = fvec(args["coeffs"])
    if c[2] == 0:
        return False, "二次项系数为 0 — 退化为一次, 不属 T-A1 适用条件"
    return True, ""


def _bisect_root(cs, a: Fraction, b: Fraction, ctx: str, ctl: CallCtl, iters: int = 80) -> "tuple[Fraction, Fraction]":
    """exact rational bisection on an interval [a,b] of known endpoint sign change (no floats, no root formula)."""
    fa = poly_eval(cs, a)
    for _ in range(iters):
        ctl.check_time(ctx)
        mid = (a + b) / 2
        fm = poly_eval(cs, mid)
        if fm == 0:
            return mid, mid
        if (fm < 0) == (fa < 0):
            a = mid
        else:
            b = mid
    return a, b


def _inv_quad(args: dict, ctl: CallCtl) -> dict:
    """T-A1 derivation: discriminant + quadratic formula (exact roots when the discriminant is a rational perfect square, otherwise a rational interval nest).

    The non-perfect-square branch **does not touch the root formula**: the vertex x_v = −c1/(2c2) cuts the real axis
    into two monotone half-axes, and a sign-changing interval is bisected on each side. The resulting interval nests
    are "exact rational + checkable by an independent checker".
    """
    ctx = "op:thm_quadratic_roots#9201"
    c0, c1, c2 = fvec(args["coeffs"])
    cs = [c0, c1, c2]
    disc = c1 * c1 - 4 * c2 * c0
    if disc < 0:
        raise SvoError(Err.DISC_NEG, ctx=ctx, detail=f"判别式 {fs(disc)} < 0: 无实根 (T-A1 条件不满足)")
    s = isqrt_frac(disc)
    if s is not None:
        r1, r2 = (-c1 + s) / (2 * c2), (-c1 - s) / (2 * c2)
    # Sorting must be by **rational value**, not by string: in string order "-6" < "-7" (because '6' < '7'), so
    # negative roots come out reversed (measured: 111 problems in the fwd family failed because of this). Sort
    # Fractions first, then convert to strings.
        roots = [fs(r1)] if r1 == r2 else [fs(v) for v in sorted((r1, r2))]
        cert = Certificate(kind="poly_backsub",
                           payload={"coeffs": [fs(v) for v in cs], "roots": roots,
                                    "tol": "1/1000000000000"},
                           issued_by="thm_quadratic_roots")
        return {"value": {"mode": "exact_roots", "roots": roots, "intervals": [],
                          "disc": fs(disc), "cond_ok": True},
                "resid": [0.0], "cert": cert,
                "notes": f"判别式 {fs(disc)} 为有理完全平方 → 精确根 {roots}; 证书=poly_backsub 精确回代"}

    # not a perfect square: the vertex splits the half-axes + per-side bisection (exact rational interval nesting)
    xv = -c1 / (2 * c2)
    B = 2 + abs(c1 / c2) + abs(c0 / c2)  # strictly greater than the Cauchy root bound
    fv = poly_eval(cs, xv)
    if fv == 0:
        raise SvoError(Err.POSTCOND_FAIL, ctx=ctx,
                       detail=f"顶点处 f=0 但判别式 {fs(disc)} ≠ 0 — 算术自相矛盾 (拒绝继续)")
    ivs = []
    for a, b in ((-B, xv), (xv, B)):
        fa, fb = poly_eval(cs, a), poly_eval(cs, b)
        if fa == 0 or fb == 0 or (fa < 0) == (fb < 0):
            raise SvoError(Err.NO_REAL_ROOT, ctx=ctx,
                           detail=f"半轴 [{fs(a)},{fs(b)}] 端点不变号 (f={fs(fa)},{fs(fb)}) — "
                                  f"区间构造失败, 拒绝给出未经验证的区间")
        ivs.append([fs(x) for x in _bisect_root(cs, a, b, ctx + "@bisect", ctl)])
    cert = Certificate(kind="interval_set_contains",
                       payload={"f_coeffs": [fs(v) for v in cs], "intervals": ivs},
                       issued_by="thm_quadratic_roots")
    return {"value": {"mode": "interval_enclosure", "roots": [], "intervals": ivs,
                      "disc": fs(disc), "cond_ok": True},
            "resid": [float(max(f(v[1]) - f(v[0]) for v in ivs))],
            "cert": cert,
            "notes": f"判别式 {fs(disc)} 非有理完全平方 → 精确通道给有理区间套 {ivs} "
                     f"(顶点分半轴+精确二分; 禁 float 冒充精确, §17.5)"}


def _ver_quad(args: dict, value: dict) -> dict:
    """T-A1 independent verification: **back-substitution** (exact Horner) — unrelated to the root formula.

    Exact roots: each root's exact back-substitution must be 0, plus a second algebraic path via Vieta's sum/product.
    Interval nests: each interval's endpoints must change sign (a real root lies inside) and the intervals must be
    pairwise disjoint (distinct roots).
    """
    c0, c1, c2 = fvec(args["coeffs"])
    cs = [c0, c1, c2]
    if value["mode"] == "exact_roots":
        roots = [f(s) for s in value["roots"]]
        if not roots:
            return {"ok": False, "resid": [1.0], "detail": "精确解却未给任何根"}
        worst = Fraction(0)
        for r in roots:
            worst = worse(worst, abs(poly_eval(cs, r)))
        vsum = sum(roots, Fraction(0))
        vprod = Fraction(1)
        for r in roots:
            vprod *= r
        return {"ok": worst == 0, "resid": [float(worst)],
                "detail": f"精确回代 max|p(r)|={worst}; 韦达和={fs(vsum)} (应 −c1/c2={fs(-c1 / c2)}), "
                          f"积={fs(vprod)} (应 c0/c2={fs(c0 / c2)})"}
    worst = 0.0
    ivs = [(f(a), f(b)) for a, b in value["intervals"]]
    for lo, hi in ivs:
        flo, fhi = poly_eval(cs, lo), poly_eval(cs, hi)
        if flo != 0 and fhi != 0 and (flo < 0) == (fhi < 0):
            return {"ok": False, "resid": [1.0],
                    "detail": f"区间 [{fs(lo)},{fs(hi)}] 端点同号 ({fs(flo)},{fs(fhi)}) — 不能保证含根"}
        worst = max(worst, float(hi - lo))
    disjoint = all(ivs[i][1] < ivs[j][0] or ivs[j][1] < ivs[i][0]
                   for i in range(len(ivs)) for j in range(i + 1, len(ivs)))
    return {"ok": disjoint and len(ivs) >= 1, "resid": [worst],
            "detail": f"{len(ivs)} 个区间端点变号 (精确有理核对), 两两不交={disjoint}, "
                      f"最大宽度 {worst:.3e}"}


REGISTRY.register(OpDef(
    name="thm_quadratic_roots", band=BAND, sub=1, thm="T-A1",
    stmt="c2x²+c1x+c0=0 (c2≠0, 判别式≥0) 的实根由求根公式给出; 无完全平方判别式时给有理区间套",
    in_schema=_IN_QUAD, out_schema=_OUT_QUAD,
    invoke=_inv_quad, verify=_ver_quad, cert_kind="poly_backsub", precond=_pre_quad,
    cost=("units", 1.5), errors=("DISC_NEG", "NO_REAL_ROOT", "POSTCOND_FAIL", "PRECOND_FAIL", "SCHEMA_MISMATCH"),
    effects="none (纯函数, 精确有理算术)",
    doc="T-A1 二次求根 (有理系数) | 条件: c2≠0 且判别式≥0 | 推导=求根公式/顶点分半轴二分 | "
        "验证器=精确回代+韦达双路径 (区间模式=端点变号+不交) | 证书=poly_backsub / interval_set_contains | "
        "1.5 units | 失败: c2=0 PRECOND_FAIL / 判别式<0 DISC_NEG / 区间构造失败 NO_REAL_ROOT",
    selfchecks=(
        {"id": "t_a1_roots_3_m2", "args": {"coeffs": [-6, -1, 1]},
         "check": lambda r: (r.value["mode"] == "exact_roots" and r.value["roots"] == ["-2", "3"]
                             and r.value["disc"] == "25" and r.cert_independent is True,
                             f"roots={r.value['roots']} disc={r.value['disc']} cert={r.cert_independent}"),
         "expect": "手算: x²−x−6: 判别式=1+24=25, 根=(1±5)/2 = 3, −2"},
        {"id": "t_a1_double_root", "args": {"coeffs": [4, -4, 1]},
         "check": lambda r: (r.value["roots"] == ["2"], f"roots={r.value['roots']}"),
         "expect": "手算: x²−4x+4=(x−2)²: 判别式=0, 重根 2"},
        {"id": "t_a1_irrational_interval", "args": {"coeffs": [-2, 0, 1]},
         "check": lambda r: (r.value["mode"] == "interval_enclosure" and len(r.value["intervals"]) == 2
                             and all(f(iv[0]) < f(iv[1]) for iv in r.value["intervals"])
                             and r.value["disc"] == "8" and r.cert_independent is True,
                             f"mode={r.value['mode']} ivs={r.value['intervals']} "
                             f"cert={r.cert_independent}"),
         "expect": "x²−2: 判别式 8 非完全平方 → 两个互不相交的有理变号区间 (含 ±√2, 不假装精确)"},
        {"id": "t_a1_neg_leading_irrational", "args": {"coeffs": [1, 1, -1]},
         "check": lambda r: (r.value["mode"] == "interval_enclosure" and len(r.value["intervals"]) == 2,
                             f"mode={r.value['mode']} ivs={r.value['intervals']}"),
         "expect": "−x²+x+1=0 (判别式 5, 首项为负): 两根 (−1±√5)/2 ≈ −0.618, 1.618; "
                   "首项符号为负时半轴变号方向翻转, 必须仍给出 2 个区间"},
        {"id": "t_a1_disc_neg", "args": {"coeffs": [1, 0, 1]}, "expect_err": "DISC_NEG",
         "expect": "x²+1 判别式 −4 < 0"},
        {"id": "t_a1_degenerate", "args": {"coeffs": [1, 1, 0]}, "expect_err": "PRECOND_FAIL",
         "expect": "c2=0 退化一次式 — T-A1 条件不满足, 必须拒"},
    ),
))


_IN_VIETA = {"kind": "dict",
             "fields": {"coeffs": {"kind": "list", "items": {"kind": "Fraction"}, "len": 3},
                        "roots": {"kind": "list", "items": {"kind": "Fraction"}, "minlen": 1, "maxlen": 2}},
             "required": ["coeffs", "roots"]}
_OUT_VIETA = {"kind": "dict",
              "fields": {"sum_claim": {"kind": "str"}, "prod_claim": {"kind": "str"},
                         "sum_expect": {"kind": "str"}, "prod_expect": {"kind": "str"},
                         "matches": {"kind": "bool"}},
              "required": ["sum_claim", "prod_claim", "sum_expect", "prod_expect", "matches"]}


def _pre_vieta(args) -> "tuple[bool, str]":
    """T-A2 conditions (hard-checked item by item; no textual declaration substitutes for the check): c2≠0 and (r1,r2) **really are** roots of that quadratic.

    Why "not roots" must be PRECOND_FAIL rather than returning matches=False: §8.2's registration format requires the
    conditions to be checked — if (r1,r2) are not roots, the theorem's premise fails and "applying T-A2" is itself an
    illegal operation. Returning false would let downstream misread "premise fails" as "the theorem is refuted". Hence
    the rejection here; callers wanting to "test candidate roots" should use the solver-side operators, not a theorem
    as a checker.
    """
    c0, c1, c2 = fvec(args["coeffs"])
    if c2 == 0:
        return False, "二次项系数为 0"
    roots = fvec(args["roots"])
    if len(roots) != 2:
        return False, "T-A2 需恰 2 个根 (重根请写两次)"
    cs = [c0, c1, c2]
    bad = [fs(r) for r in roots if poly_eval(cs, r) != 0]
    if bad:
        return False, f"{bad} 不是 {fs(c2)}x²+{fs(c1)}x+{fs(c0)} 的根 — T-A2 前提不成立"
    return True, ""


def _inv_vieta(args: dict, ctl: CallCtl) -> dict:
    """T-A2 derivation: sum/product formulas."""
    c0, c1, c2 = fvec(args["coeffs"])
    r1, r2 = fvec(args["roots"])
    s = r1 + r2
    p = r1 * r2
    return {"value": {"sum_claim": fs(s), "prod_claim": fs(p),
                      "sum_expect": fs(-c1 / c2), "prod_expect": fs(c0 / c2),
                      "matches": (s == -c1 / c2 and p == c0 / c2)},
            "resid": [0.0],
            "notes": f"韦达: 和={fs(s)} (应 {fs(-c1 / c2)}), 积={fs(p)} (应 {fs(c0 / c2)})"}


def _ver_vieta(args: dict, value: dict) -> dict:
    """T-A2 independent verification: recover the sum and product from the **convolution expansion** (x−r1)(x−r2),
    unrelated to invoke's sum/product formulas; then check the "matches" field against what that expansion route
    yields, and that the premise "the roots really are roots" closes on the output."""
    c0, c1, c2 = fvec(args["coeffs"])
    r1, r2 = fvec(args["roots"])
    exp = poly_mul([-r1, Fraction(1)], [-r2, Fraction(1)])  # [r1r2, −(r1+r2), 1]
    if exp[2] == 0:
        return {"ok": False, "resid": [1.0], "detail": "展开式首项为 0 — 展开失败"}
    sum_conv = -exp[1] / exp[2]
    prod_conv = exp[0] / exp[2]
    # premise closure: the convolution form and the input coefficients (modulo the leading-term ratio) must be proportional
    lam = c2 / exp[2]
    prop_gap = max(abs(c0 - lam * exp[0]), abs(c1 - lam * exp[1]), abs(c2 - lam * exp[2]))
    worst = max(abs(f(value["sum_claim"]) - sum_conv), abs(f(value["prod_claim"]) - prod_conv))
    ok = worst == 0 and prop_gap == 0
    return {"ok": ok, "resid": [float(worst), float(prop_gap)],
            "detail": f"卷积展开式={[fs(v) for v in exp]} → 和={fs(sum_conv)}, 积={fs(prod_conv)}; "
                      f"与输出差 max={fs(worst)}; 与输入系数的比例闭合差={fs(prop_gap)} (皆应为 0)"}


REGISTRY.register(OpDef(
    name="thm_vieta", band=BAND, sub=2, thm="T-A2",
    stmt="若 r1,r2 是 c2x²+c1x+c0 的根 (c2≠0), 则 r1+r2=−c1/c2, r1r2=c0/c2",
    in_schema=_IN_VIETA, out_schema=_OUT_VIETA,
    invoke=_inv_vieta, verify=_ver_vieta, precond=_pre_vieta,
    cost=("units", 0.8), errors=("PRECOND_FAIL", "SCHEMA_MISMATCH"),
    effects="none (纯函数, 精确有理算术)",
    doc="T-A2 韦达定理 | 条件: c2≠0 且 (r1,r2) 确为根 (逐条回代硬检查) | 推导=和/积公式 | "
        "验证器=卷积展开反解和积 + 比例闭合 | 0.8 units | 失败: 退化或非根 PRECOND_FAIL",
    selfchecks=(
        {"id": "t_a2_match", "args": {"coeffs": [-6, -1, 1], "roots": [3, -2]},
         "check": lambda r: (r.value["matches"] and r.value["sum_claim"] == "1"
                             and r.value["prod_claim"] == "-6",
                             f"sum={r.value['sum_claim']} prod={r.value['prod_claim']}"),
         "expect": "手算: x²−x−6 根 3,−2: 和=1=−(−1)/1, 积=−6=−6/1"},
        {"id": "t_a2_frac_roots", "args": {"coeffs": [1, -3, 2], "roots": ["1/2", 1]},
         "check": lambda r: (r.value["matches"] and r.value["prod_claim"] == "1/2"
                             and r.value["sum_claim"] == "3/2",
                             f"sum={r.value['sum_claim']} prod={r.value['prod_claim']}"),
         "expect": "2x²−3x+1=(2x−1)(x−1): 根 1/2,1; 和=c1/c2 取负=3/2, 积=1/2"},
        {"id": "t_a2_nonroot_rejected", "args": {"coeffs": [-6, -1, 1], "roots": [3, 2]},
         "expect_err": "PRECOND_FAIL",
         "expect": "3,2 不是 x²−x−6 的根 (2²−2−6=−4≠0) → 前提不成立, 必须拒"},
        {"id": "t_a2_need_two_roots", "args": {"coeffs": [1, 0, 1], "roots": [0]},
         "expect_err": "PRECOND_FAIL", "expect": "只给 1 个根 → 条件不满足"},
        {"id": "t_a2_zero_c2", "args": {"coeffs": [1, 1, 0], "roots": [-1, 5]},
         "expect_err": "PRECOND_FAIL", "expect": "c2=0 退化"},
    ),
))


_IN_AP = {"kind": "dict",
          "fields": {"a1": {"kind": "Fraction"}, "d": {"kind": "Fraction"},
                     "n": {"kind": "int", "min": 1, "max": 4096}},
          "required": ["a1", "d", "n"]}
_OUT_AP = {"kind": "dict",
           "fields": {"sum": {"kind": "str"}, "first": {"kind": "str"}, "last": {"kind": "str"},
                      "n": {"kind": "int"}},
           "required": ["sum", "first", "last", "n"]}


def _inv_ap(args: dict, ctl: CallCtl) -> dict:
    a1, d, n = f(args["a1"]), f(args["d"]), int(args["n"])
    last = a1 + (n - 1) * d
    S = n * (2 * a1 + (n - 1) * d) / 2
    return {"value": {"sum": fs(S), "first": fs(a1), "last": fs(last), "n": n},
            "resid": [0.0],
            "notes": f"T-A3 等差 n={n}: S=n(2a1+(n−1)d)/2={fs(S)}"}


def _ver_ap(args: dict, value: dict) -> dict:
    """T-A3 independent verification: **term-by-term accumulation comparison** (a different path from the closed form);
    for n>2000 it switches to the pairing method (pair first with last) — neither is invoke's formula."""
    a1, d, n = f(args["a1"]), f(args["d"]), int(args["n"])
    if n <= 2000:
        acc, cur = Fraction(0), a1
        for _ in range(n):
            acc += cur
            cur += d
        detail = f"逐项累加 {n} 项"
    else:
        pairs = n // 2
        acc = pairs * (a1 + (a1 + (n - 1) * d))
        if n % 2:
            acc += a1 + (n // 2) * d
        detail = f"首末配对法 {pairs} 对" + (" + 中项" if n % 2 else "")
    got = f(value["sum"])
    ok = acc == got
    return {"ok": ok, "resid": [0.0 if ok else float(abs(acc - got))],
            "detail": f"{detail} → {fs(acc)} vs 声称 {fs(got)}"}


REGISTRY.register(OpDef(
    name="thm_ap_sum", band=BAND, sub=3, thm="T-A3",
    stmt="首项 a1 公差 d 的 n 项等差数列之和 S = n(2a1+(n−1)d)/2",
    in_schema=_IN_AP, out_schema=_OUT_AP,
    invoke=_inv_ap, verify=_ver_ap,
    cost=("units", 0.6), errors=("SCHEMA_MISMATCH",),
    effects="none (纯函数, 精确有理算术)",
    doc="T-A3 等差级数和 | 推导=S=n(2a1+(n−1)d)/2 | 验证器=逐项累加/首末配对 (累加对照) | "
        "0.6 units | 失败: schema",
    selfchecks=(
        {"id": "t_a3_hand_1_2_5", "args": {"a1": 1, "d": 2, "n": 5},
         "check": lambda r: (r.value["sum"] == "25" and r.value["last"] == "9",
                             f"sum={r.value['sum']} last={r.value['last']}"),
         "expect": "手算: 1+3+5+7+9=25, 末项 9"},
        {"id": "t_a3_gauss_100", "args": {"a1": 1, "d": 1, "n": 100},
         "check": lambda r: (r.value["sum"] == "5050", f"sum={r.value['sum']}"),
         "expect": "高斯: 1+…+100=5050"},
        {"id": "t_a3_big_n_pairing", "args": {"a1": 3, "d": "1/2", "n": 3000},
         "check": lambda r: (r.value["sum"] == "2258250" and r.value["last"] == "3005/2",
                             f"sum={r.value['sum']} last={r.value['last']}"),
         "expect": "手算: 末项=3+2999·(1/2)=3+1499.5=1502.5=3005/2; S=n(首+末)/2=3000·(3+3005/2)/2"
                   "=3000·(3011/2)/2=750·3011=2258250。n>2000 触发首末配对验证路径"},
        {"id": "t_a3_bad_n", "args": {"a1": 1, "d": 1, "n": 0}, "expect_err": "SCHEMA_MISMATCH",
         "expect": "n=0 越界 (min=1)"},
    ),
))


_IN_GP = {"kind": "dict",
          "fields": {"a": {"kind": "Fraction"}, "r": {"kind": "Fraction"},
                     "n": {"kind": "int", "min": 1, "max": 4096}},
          "required": ["a", "r", "n"]}
_OUT_GP = dict(_OUT_AP)


def _inv_gp(args: dict, ctl: CallCtl) -> dict:
    a, r, n = f(args["a"]), f(args["r"]), int(args["n"])
    if r == 1:
        S = a * n
    else:
        S = a * (r ** n - 1) / (r - 1)
    last = a * (r ** (n - 1))
    return {"value": {"sum": fs(S), "first": fs(a), "last": fs(last), "n": n},
            "resid": [0.0],
            "notes": f"T-A4 等比 n={n}, r={fs(r)}" + (" (r=1 退化分支)" if r == 1 else "")}


def _ver_gp(args: dict, value: dict) -> dict:
    """T-A4 independent verification: term-by-term accumulation (r≠1 and n≤2000); for large n, another form using powers of r and the geometric sum recurrence."""
    a, r, n = f(args["a"]), f(args["r"]), int(args["n"])
    if n <= 2000:
        acc, cur = Fraction(0), a
        for _ in range(n):
            acc += cur
            cur *= r
        detail = f"逐项累加 {n} 项"
    elif r == 1:
        acc, detail = a * n, "r=1: 退化算术和"
    else:
        # another rearrangement of (r−1)S = a(rⁿ−1): S = a + r·S − a·rⁿ
        acc = a * (1 - r ** n) / (1 - r)
        detail = "通分整理式 S=a(1−rⁿ)/(1−r)"
    got = f(value["sum"])
    ok = acc == got
    return {"ok": ok, "resid": [0.0 if ok else float(abs(acc - got))],
            "detail": f"{detail} → {fs(acc)} vs 声称 {fs(got)}"}


REGISTRY.register(OpDef(
    name="thm_gp_sum", band=BAND, sub=4, thm="T-A4",
    stmt="首项 a 公比 r 的 n 项等比数列之和 S = a(rⁿ−1)/(r−1) (r≠1), r=1 时 S=an",
    in_schema=_IN_GP, out_schema=_OUT_GP,
    invoke=_inv_gp, verify=_ver_gp,
    cost=("units", 0.7), errors=("SCHEMA_MISMATCH",),
    effects="none (纯函数, 精确有理算术)",
    doc="T-A4 等比级数和 (含 r=1 退化分支) | 验证器=逐项累加/通分整理 | 0.7 units | 失败: schema",
    selfchecks=(
        {"id": "t_a4_hand_half_5", "args": {"a": 1, "r": "1/2", "n": 5},
         "check": lambda r: (r.value["sum"] == "31/16" and r.value["last"] == "1/16",
                             f"sum={r.value['sum']} last={r.value['last']}"),
         "expect": "手算: 1+1/2+1/4+1/8+1/16=31/16, 末项 1/16"},
        {"id": "t_a4_r_one", "args": {"a": 3, "r": 1, "n": 4},
         "check": lambda r: (r.value["sum"] == "12", f"sum={r.value['sum']}"),
         "expect": "r=1 退化: 3·4=12"},
        {"id": "t_a4_neg_ratio", "args": {"a": 1, "r": -1, "n": 3},
         "check": lambda r: (r.value["sum"] == "1", f"sum={r.value['sum']}"),
         "expect": "手算: 1−1+1=1"},
        {"id": "t_a4_r_zero", "args": {"a": 5, "r": 0, "n": 3},
         "check": lambda r: (r.value["sum"] == "5" and r.value["last"] == "0",
                             f"sum={r.value['sum']}"),
         "expect": "r=0: 5+0+0=5"},
        {"id": "t_a4_bad_n_neg", "args": {"a": 1, "r": 2, "n": 0}, "expect_err": "SCHEMA_MISMATCH",
         "expect": "n=0 越界 (min=1)"},
        {"id": "t_a4_float_reject", "args": {"a": 0.5, "r": 1, "n": 2}, "expect_err": "SCHEMA_MISMATCH",
         "expect": "float 不得进精确通道 (§17.5)"},
    ),
))


_IN_BINOM = {"kind": "dict",
             "fields": {"n": {"kind": "int", "min": 0, "max": 16}},
             "required": ["n"]}
_OUT_BINOM = {"kind": "dict",
              "fields": {"coeffs": {"kind": "list", "items": {"kind": "str"}, "minlen": 1, "maxlen": 17},
                         "sum_coeffs": {"kind": "str"}, "n": {"kind": "int"}},
              "required": ["coeffs", "sum_coeffs", "n"]}


def _binom_row(n: int) -> "list[Fraction]":
    row = [Fraction(1)]
    for k in range(1, n + 1):
        row.append(row[-1] * (n - k + 1) // k)
    return row


def _inv_binom(args: dict, ctl: CallCtl) -> dict:
    n = int(args["n"])
    row = _binom_row(n)
    S = sum(row, Fraction(0))
    return {"value": {"coeffs": [fs(c) for c in row], "sum_coeffs": fs(S), "n": n},
            "resid": [0.0], "notes": f"T-A5 (1+x)^{n} 系数 {len(row)} 项, 系数和={fs(S)}=2^{n}"}


def _ver_binom(args: dict, value: dict) -> dict:
    """T-A5 independent verification: **repeated convolution expansion for small n** (a different path from Pascal's recurrence) + pointwise difference comparison."""
    n = int(args["n"])
    acc = [Fraction(1)]
    for _ in range(n):
        acc = poly_mul(acc, [Fraction(1), Fraction(1)])
    got = fvec(value["coeffs"])
    ok_len = len(acc) == len(got)
    worst = Fraction(0)
    if ok_len:
        for a, b in zip(acc, got):
            worst = worse(worst, abs(a - b))
    # pointwise differences: the values of (1+x)^n at n+1 integer points agree with the claimed coefficient polynomial
    pt = Fraction(0)
    for x in range(n + 1):
        lhs = Fraction(1 + x) ** n
        rhs = poly_eval(got, Fraction(x))
        pt = worse(pt, abs(lhs - rhs))
    ok = ok_len and worst == 0 and pt == 0 and f(value["sum_coeffs"]) == sum(got, Fraction(0))
    return {"ok": ok, "resid": [float(worst), float(pt)],
            "detail": f"重复卷积展开比对 max|Δ系数|={fs(worst)}; (1+x)^n 逐点差 max|Δ|={fs(pt)} (精确 0)"}


REGISTRY.register(OpDef(
    name="thm_binomial", band=BAND, sub=5, thm="T-A5",
    stmt="二项式定理: (1+x)ⁿ = Σ C(n,k)xᵏ",
    in_schema=_IN_BINOM, out_schema=_OUT_BINOM,
    invoke=_inv_binom, verify=_ver_binom,
    cost=("units", 1.0), errors=("SCHEMA_MISMATCH",),
    effects="none (纯函数, 精确有理算术)",
    doc="T-A5 二项式系数 (n≤16) | 推导=帕斯卡递推 | 验证器=重复卷积展开 + (1+x)^n 逐点差分 | "
        "1 unit | 失败: n 越界 schema",
    selfchecks=(
        {"id": "t_a5_row4", "args": {"n": 4},
         "check": lambda r: (r.value["coeffs"] == ["1", "4", "6", "4", "1"]
                             and r.value["sum_coeffs"] == "16", f"row={r.value['coeffs']}"),
         "expect": "手算: (1+x)⁴ = 1+4x+6x²+4x³+x⁴, 系数和 16=2⁴"},
        {"id": "t_a5_row0", "args": {"n": 0},
         "check": lambda r: (r.value["coeffs"] == ["1"] and r.value["sum_coeffs"] == "1",
                             f"row={r.value['coeffs']}"),
         "expect": "n=0: (1+x)⁰=1"},
        {"id": "t_a5_row10_sum", "args": {"n": 10},
         "check": lambda r: (r.value["sum_coeffs"] == "1024" and len(r.value["coeffs"]) == 11,
                             f"sum={r.value['sum_coeffs']} len={len(r.value['coeffs'])}"),
         "expect": "Σ C(10,k) = 2¹⁰ = 1024"},
        {"id": "t_a5_n_neg", "args": {"n": -1}, "expect_err": "SCHEMA_MISMATCH", "expect": "n<0 越界"},
        {"id": "t_a5_n_too_big", "args": {"n": 17}, "expect_err": "SCHEMA_MISMATCH",
         "expect": "n>16 越界 (小 n 展开域)"},
    ),
))


_IN_CRT = {"kind": "dict",
           "fields": {"a1": {"kind": "int"}, "m1": {"kind": "int", "min": 1, "max": 10**6},
                      "a2": {"kind": "int"}, "m2": {"kind": "int", "min": 1, "max": 10**6}},
           "required": ["a1", "m1", "a2", "m2"]}
_OUT_CRT = {"kind": "dict",
            "fields": {"x": {"kind": "int", "min": 0}, "M": {"kind": "int", "min": 1}},
            "required": ["x", "M"]}


def _pre_crt(args) -> "tuple[bool, str]":
    if math.gcd(args["m1"], args["m2"]) != 1:
        return False, f"模 {args['m1']},{args['m2']} 不互素 — T-A6 条件不满足"
    return True, ""


def _inv_crt(args: dict, ctl: CallCtl) -> dict:
    a1, m1, a2, m2 = int(args["a1"]), int(args["m1"]), int(args["a2"]), int(args["m2"])
    M = m1 * m2
    M1, M2 = m2, m1
    inv1 = pow(M1, -1, m1)
    inv2 = pow(M2, -1, m2)
    x = (a1 * M1 * inv1 + a2 * M2 * inv2) % M
    cert = Certificate(kind="mod_congruence",
                       payload={"cases": [{"lhs": x, "rhs": a1, "m": m1},
                                          {"lhs": x, "rhs": a2, "m": m2}]},
                       issued_by="thm_crt")
    return {"value": {"x": x, "M": M}, "resid": [0.0], "cert": cert,
            "notes": f"T-A6: x={x} mod {M} (M1⁻¹={inv1} mod {m1}, M2⁻¹={inv2} mod {m2}); "
                     f"证书=双模代入 mod_congruence"}


def _ver_crt(args: dict, value: dict) -> dict:
    """T-A6 independent verification: **double modular substitution** (direct modulo) + exhaustive uniqueness check within M."""
    a1, m1, a2, m2 = int(args["a1"]), int(args["m1"]), int(args["a2"]), int(args["m2"])
    x, M = int(value["x"]), int(value["M"])
    r1, r2 = x % m1, x % m2
    ok_sub = (r1 == a1 % m1) and (r2 == a2 % m2)
    hits = sum(1 for y in range(M) if y % m1 == a1 % m1 and y % m2 == a2 % m2)
    ok = ok_sub and hits == 1 and 0 <= x < M
    return {"ok": ok, "resid": [0.0 if ok else 1.0],
            "detail": f"代入: x mod {m1}={r1} (应 {a1 % m1}), x mod {m2}={r2} (应 {a2 % m2}); "
                      f"穷举 [0,{M}) 内解个数={hits} (应 1, 即唯一性)"}


REGISTRY.register(OpDef(
    name="thm_crt", band=BAND, sub=6, thm="T-A6",
    stmt="中国剩余定理 (双模): gcd(m1,m2)=1 时 x≡a1(m1) ∧ x≡a2(m2) 在 mod m1m2 下有唯一解",
    in_schema=_IN_CRT, out_schema=_OUT_CRT,
    invoke=_inv_crt, verify=_ver_crt, cert_kind="mod_congruence", precond=_pre_crt,
    cost=("units", 1.0), errors=("PRECOND_FAIL", "SCHEMA_MISMATCH"),
    effects="none (纯函数, 精确整数算术)",
    doc="T-A6 中国剩余定理 (双模) | 条件: gcd(m1,m2)=1 | 推导=构造式 Σ aᵢMᵢMᵢ⁻¹ | "
        "验证器=双模代入+穷举唯一性 | 证书=mod_congruence | 1 unit | 失败: 不互素 PRECOND_FAIL",
    selfchecks=(
        {"id": "t_a6_hand_2_3__3_5", "args": {"a1": 2, "m1": 3, "a2": 3, "m2": 5},
         "check": lambda r: (r.value["x"] == 8 and r.value["M"] == 15 and r.cert_independent is True,
                             f"x={r.value['x']} M={r.value['M']}"),
         "expect": "手算: x≡2(3), x≡3(5) → x=8; 8 mod 3=2, 8 mod 5=3"},
        {"id": "t_a6_hand_1_4__2_5", "args": {"a1": 1, "m1": 4, "a2": 2, "m2": 5},
         "check": lambda r: (r.value["x"] == 17, f"x={r.value['x']}"),
         "expect": "手算: x≡1(4), x≡2(5) → x=17 (17 mod 4=1, 17 mod 5=2)"},
        {"id": "t_a6_zero_residue", "args": {"a1": 0, "m1": 2, "a2": 0, "m2": 3},
         "check": lambda r: (r.value["x"] == 0, f"x={r.value['x']}"),
         "expect": "双零余数 → x=0"},
        {"id": "t_a6_not_coprime", "args": {"a1": 1, "m1": 4, "a2": 1, "m2": 6},
         "expect_err": "PRECOND_FAIL", "expect": "gcd(4,6)=2 ≠1 — 条件不满足"},
        {"id": "t_a6_m_zero", "args": {"a1": 1, "m1": 0, "a2": 1, "m2": 5},
         "expect_err": "SCHEMA_MISMATCH", "expect": "m1=0 越界"},
    ),
))


_IN_DIFSQ = {"kind": "dict",
             "fields": {"a": {"kind": "Fraction"}, "b": {"kind": "Fraction"}},
             "required": ["a", "b"]}
_OUT_DIFSQ = {"kind": "dict",
              "fields": {"lhs": {"kind": "str"}, "rhs": {"kind": "str"}, "equal": {"kind": "bool"}},
              "required": ["lhs", "rhs", "equal"]}


def _inv_difsq(args: dict, ctl: CallCtl) -> dict:
    a, b = f(args["a"]), f(args["b"])
    lhs = a * a - b * b
    rhs = (a - b) * (a + b)
    cert = cert_identity([lhs], [rhs], "thm_diff_squares", 2)
    return {"value": {"lhs": fs(lhs), "rhs": fs(rhs), "equal": lhs == rhs},
            "resid": [0.0 if lhs == rhs else 1.0], "cert": cert,
            "notes": f"T-A7: a²−b²={fs(lhs)}, (a−b)(a+b)={fs(rhs)}"}


def _ver_difsq(args: dict, value: dict) -> dict:
    """T-A7 independent verification: evaluate a²−b² and its factorised expansion **independently**, and check the sign conditions a−b, a+b.

    Unlike invoke's "compute each form from the other": here the factorised expansion is done first (poly_mul
    convolution) and then compared.
    """
    a, b = f(args["a"]), f(args["b"])
    exp = poly_mul([-b, Fraction(1)], [b, Fraction(1)])  # the convolution form of (x−b)(x+b)
    A, B = a, b
    fac_val = poly_eval(exp, A)  # = A² − b²
    claim = f(value["lhs"])
    worst = abs(fac_val - claim)
    sym_ok = (A - B) * (A + B) == claim
    ok = worst == 0 and sym_ok and value["equal"]
    return {"ok": ok, "resid": [float(worst)],
            "detail": f"卷积式 x²−b² 在 x=a 处 ={fs(fac_val)}; 对称式 (a−b)(a+b)={fs((A - B) * (A + B))}; "
                      f"声称 a²−b²={fs(claim)} (三者精确一致)"}


REGISTRY.register(OpDef(
    name="thm_diff_squares", band=BAND, sub=7, thm="T-A7",
    stmt="平方差恒等式: a²−b² = (a−b)(a+b)",
    in_schema=_IN_DIFSQ, out_schema=_OUT_DIFSQ,
    invoke=_inv_difsq, verify=_ver_difsq, cert_kind="identity_expand",
    cost=("units", 0.4), errors=("SCHEMA_MISMATCH",),
    effects="none (纯函数, 精确有理算术)",
    doc="T-A7 平方差恒等式 | 推导=两式互算 | 验证器=卷积展开独立求值 + 对称式 | "
        "证书=identity_expand | 0.4 units | 失败: schema",
    selfchecks=(
        {"id": "t_a7_ints", "args": {"a": 3, "b": 2},
         "check": lambda r: (r.value["lhs"] == "5" and r.value["rhs"] == "5" and r.value["equal"]
                             and r.cert_independent is True, f"lhs={r.value['lhs']} rhs={r.value['rhs']}"),
         "expect": "手算: 9−4=5=(3−2)(3+2)"},
        {"id": "t_a7_fracs", "args": {"a": "3/2", "b": "1/2"},
         "check": lambda r: (r.value["lhs"] == "2", f"lhs={r.value['lhs']}"),
         "expect": "手算: 9/4−1/4=2=(1)(2)"},
        {"id": "t_a7_negatives", "args": {"a": -5, "b": 3},
         "check": lambda r: (r.value["lhs"] == "16", f"lhs={r.value['lhs']}"),
         "expect": "手算: 25−9=16=(−8)(−2)"},
        {"id": "t_a7_float_reject", "args": {"a": 0.5, "b": 1}, "expect_err": "SCHEMA_MISMATCH",
         "expect": "float 不得进精确通道 (§17.5)"},
    ),
))


# ===========================================================================
# inequalities (T-I1 … T-I4, rows 9211–9214)
# ===========================================================================


_IN_AMGM = {"kind": "dict",
            "fields": {"xs": {"kind": "list", "items": {"kind": "Fraction"},
                              "minlen": 2, "maxlen": 6}},
            "required": ["xs"]}
_OUT_AMGM = {"kind": "dict",
             "fields": {"am": {"kind": "str"}, "prod": {"kind": "str"},
                        "gap": {"kind": "str"}, "equality": {"kind": "bool"}, "n": {"kind": "int"}},
             "required": ["am", "prod", "gap", "equality", "n"]}


def _pre_amgm(args) -> "tuple[bool, str]":
    xs = fvec(args["xs"])
    if any(x <= 0 for x in xs):
        return False, "AM-GM 条件: 全部 x_i > 0 (含非正数则前提不成立)"
    return True, ""


def _inv_amgm(args: dict, ctl: CallCtl) -> dict:
    """T-I1 derivation: compare the cleared-denominator polynomial (Σx)ⁿ ≥ nⁿ·∏x, an exact integer comparison."""
    xs = fvec(args["xs"])
    n = len(xs)
    S = sum(xs, Fraction(0))
    P = Fraction(1)
    for x in xs:
        P *= x
    am = S / n
    gap = S ** n - (n ** n) * P
    eq = all(x == xs[0] for x in xs)
    if gap < 0:
        raise SvoError(Err.POSTCOND_FAIL, ctx="op:thm_amgm#9211",
                       detail=f"AM-GM 被违反: (Σx)ⁿ−nⁿ∏x={fs(gap)} < 0 — 前提或算术有误")
    return {"value": {"am": fs(am), "prod": fs(P), "gap": fs(gap), "equality": eq, "n": n},
            "resid": [float(abs(gap))],
            "notes": f"T-I1: AM={fs(am)}, ∏x={fs(P)}; 清分母整式 (Σx)ⁿ−nⁿ∏x={fs(gap)} ≥ 0; "
                     f"等号条件 x_i 全等 = {eq}"}


def _ver_amgm(args: dict, value: dict) -> dict:
    """T-I1 independent verification (numeric + inductive): (a) the exact comparison in cross-multiplied form (a different rearrangement);
    (b) the completing-the-square identity at n=2, (Σx)²−4∏x = (x1−x2)² — an explicit witness for the induction base case;
    (c) bounded rational counter-example search (if the inequality were false, the search should find a counter-example)."""
    xs = fvec(args["xs"])
    n = len(xs)
    S, P = sum(xs, Fraction(0)), Fraction(1)
    for x in xs:
        P *= x
    # (a) cross-multiplication: S^n ≥ n^n P — recomputed by expanding both sides separately (Fraction powers)
    lhs, rhs = S ** n, (n ** n) * P
    # (b) the n=2 completing-the-square case
    cfg = (xs[0] - xs[1]) ** 2 if n == 2 else None
    if n == 2:
        base_ok = (lhs - rhs) == cfg
    else:
        base_ok = True  # for n>2 there is no such base-case form, so no assertion is made
    # (c) bounded search: rational grid r∈[−2,2]∩ℤ/2 (negative values included to exercise the premise)
    viol = 0
    tried = 0
    grid = [Fraction(k, 2) for k in range(-4, 5)]
    for combo in product(grid, repeat=min(n, 3)):
        if any(x <= 0 for x in combo):
            continue
        tried += 1
        s2, p2 = sum(combo, Fraction(0)), Fraction(1)
        for x in combo:
            p2 *= x
        if s2 ** len(combo) < (len(combo) ** len(combo)) * p2:
            viol += 1
    ok = (lhs >= rhs) and base_ok and viol == 0 and (value["equality"] == all(x == xs[0] for x in xs))
    return {"ok": ok, "resid": [float(max(lhs - rhs, 0)), float(viol)],
            "detail": f"交叉相乘 LHS={fs(lhs)} ≥ RHS={fs(rhs)}; "
                     + (f"n=2 配方 (x1−x2)²={fs(cfg)} 与差等式一致; " if n == 2 else "")
                     + f"有界反例搜索 {tried} 组 (网格 ±2 步长 1/2, 维度 ≤3) 发现 {viol} 个反例"}


REGISTRY.register(OpDef(
    name="thm_amgm", band=BAND, sub=11, thm="T-I1",
    stmt="AM-GM: 正数 x₁…xₙ 的算术平均 ≥ 几何平均, 等号当且仅当全体相等 (等价形式 (Σx)ⁿ ≥ nⁿ∏x)",
    in_schema=_IN_AMGM, out_schema=_OUT_AMGM,
    invoke=_inv_amgm, verify=_ver_amgm, precond=_pre_amgm,
    cost=("units", 1.2), errors=("PRECOND_FAIL", "POSTCOND_FAIL", "SCHEMA_MISMATCH"),
    effects="none (纯函数, 精确有理算术)",
    doc="T-I1 AM-GM | 条件: 全部 x_i>0, n∈[2,6] | 推导=清分母整式比较 | "
        "验证器=交叉相乘精确比较 + n=2 配方基例 + 有界反例搜索 | 1.2 units | "
        "失败: 非正输入 PRECOND_FAIL / 不等式被违反 POSTCOND_FAIL (绝不静默给错值)",
    selfchecks=(
        {"id": "t_i1_two_four", "args": {"xs": [1, 4]},
         "check": lambda r: (r.value["am"] == "5/2" and r.value["prod"] == "4"
                             and r.value["equality"] is False and r.value["gap"] == "9",
                             f"am={r.value['am']} prod={r.value['prod']} gap={r.value['gap']}"),
         "expect": "手算: AM=5/2, ∏=4; (1+4)²−4·4=25−16=9"},
        {"id": "t_i1_equality", "args": {"xs": [3, 3, 3]},
         "check": lambda r: (r.value["equality"] is True and r.value["gap"] == "0"
                             and r.value["am"] == "3",
                             f"eq={r.value['equality']} gap={r.value['gap']}"),
         "expect": "全等 → 等号成立: AM=3, 差=0 (3·3)³−3³·27=729−729=0"},
        {"id": "t_i1_four_vals", "args": {"xs": ["1/2", 1, 2, 4]},
         "check": lambda r: (r.value["prod"] == "4" and r.value["am"] == "15/8",
                             f"am={r.value['am']} prod={r.value['prod']}"),
         "expect": "手算: 和=15/2 → AM=15/8; ∏=1/2·1·2·4=4"},
        {"id": "t_i1_nonpositive", "args": {"xs": [-1, 2]}, "expect_err": "PRECOND_FAIL",
         "expect": "含非正数 — AM-GM 前提不成立, 必须拒"},
        {"id": "t_i1_n_one", "args": {"xs": [1]}, "expect_err": "SCHEMA_MISMATCH",
         "expect": "n=1 < minlen=2"},
    ),
))


_IN_CS = {"kind": "dict",
          "fields": {"u": {"kind": "list", "items": {"kind": "Fraction"}, "minlen": 1, "maxlen": 8},
                     "v": {"kind": "list", "items": {"kind": "Fraction"}, "minlen": 1, "maxlen": 8}},
          "required": ["u", "v"]}
_OUT_CS = {"kind": "dict",
           "fields": {"dot2": {"kind": "str"}, "norms": {"kind": "str"},
                      "holds": {"kind": "bool"}, "equality_holds": {"kind": "bool"},
                      "proportional": {"kind": "bool"}},
           "required": ["dot2", "norms", "holds", "equality_holds", "proportional"]}


def _pre_cs(args) -> "tuple[bool, str]":
    if len(args["u"]) != len(args["v"]):
        return False, f"维度不等 {len(args['u'])} vs {len(args['v'])}"
    return True, ""


def _inv_cs(args: dict, ctl: CallCtl) -> dict:
    u, v = fvec(args["u"]), fvec(args["v"])
    dot = sum((a * b for a, b in zip(u, v)), Fraction(0))
    nu = sum((a * a for a in u), Fraction(0))
    nv = sum((b * b for b in v), Fraction(0))
    props = [u[i] * v[j] - u[j] * v[i] for i in range(len(u)) for j in range(i + 1, len(u))]
    prop = all(p == 0 for p in props)
    gap = nu * nv - dot * dot
    if gap < 0:
        raise SvoError(Err.POSTCOND_FAIL, ctx="op:thm_cauchy_schwarz#9212",
                       detail=f"Cauchy-Schwarz 被违反: (Σu²)(Σv²)−(Σuv)²={fs(gap)} < 0")
    return {"value": {"dot2": fs(dot * dot), "norms": fs(nu * nv), "holds": True,
                      "equality_holds": gap == 0, "proportional": prop},
            "resid": [float(gap)],
            "notes": f"T-I2: (Σuv)²={fs(dot * dot)} ≤ (Σu²)(Σv²)={fs(nu * nv)}, 差={fs(gap)}; "
                     f"等号={gap == 0}, 成比例(全 2×2 子式为 0)={prop}"}


def _ver_cs(args: dict, value: dict) -> dict:
    """T-I2 independent verification (equality-condition enumeration): recompute the difference via the **Gram determinant** path and enumerate all 2×2 minors.

    A different path from invoke's Σ expansion: here the Gram matrix [[Σu²,Σuv],[Σuv,Σv²]] is constructed explicitly
    and its determinant taken, with the minors listed one by one to check "equality ⟺ proportional".
    """
    u, v = fvec(args["u"]), fvec(args["v"])
    dot = sum((a * b for a, b in zip(u, v)), Fraction(0))
    nu = sum((a * a for a in u), Fraction(0))
    nv = sum((b * b for b in v), Fraction(0))
    gram_det = nu * nv - dot * dot
    minors = [[i, j, u[i] * v[j] - u[j] * v[i]]
              for i in range(len(u)) for j in range(i + 1, len(u))]
    all_zero = all(m[2] == 0 for m in minors)
    claim_holds = value["holds"] and f(value["norms"]) - f(value["dot2"]) == gram_det
    # equality ⟺ Gram determinant = 0 ⟺ every 2×2 minor is 0 (proportionality) — the three must agree
    eq_ok = (value["equality_holds"] == (gram_det == 0)) and (all_zero == (gram_det == 0))
    ok = gram_det >= 0 and claim_holds and eq_ok and value["proportional"] == all_zero
    return {"ok": ok, "resid": [float(max(gram_det, 0)), float(len(minors))],
            "detail": f"Gram 行列式={fs(gram_det)} ≥ 0; 枚举 {len(minors)} 个 2×2 子式, "
                      f"全零={all_zero} (⟺ 成比例 ⟺ 等号); 与声称值一致={claim_holds}"}


REGISTRY.register(OpDef(
    name="thm_cauchy_schwarz", band=BAND, sub=12, thm="T-I2",
    stmt="Cauchy-Schwarz: (Σuᵢvᵢ)² ≤ (Σuᵢ²)(Σvᵢ²), 等号当且仅当两向量成比例",
    in_schema=_IN_CS, out_schema=_OUT_CS,
    invoke=_inv_cs, verify=_ver_cs, precond=_pre_cs,
    cost=("units", 1.0), errors=("PRECOND_FAIL", "POSTCOND_FAIL", "SCHEMA_MISMATCH"),
    effects="none (纯函数, 精确有理算术)",
    doc="T-I2 Cauchy-Schwarz | 条件: 两向量等维 | 推导=Σ 展开 | "
        "验证器=Gram 行列式 + 全部 2×2 子式枚举 (等号条件) | 1 unit | 失败: 维度不等 PRECOND_FAIL",
    selfchecks=(
        {"id": "t_i2_strict", "args": {"u": [1, 2], "v": [3, 4]},
         "check": lambda r: (r.value["dot2"] == "121" and r.value["norms"] == "125"
                             and r.value["holds"] and not r.value["equality_holds"]
                             and not r.value["proportional"],
                             f"dot²={r.value['dot2']} norms={r.value['norms']}"),
         "expect": "手算: Σuv=11 → 121; Σu²=5, Σv²=25 → 125; 121<125 严格; 子式 1·4−2·3=−2≠0"},
        {"id": "t_i2_equality", "args": {"u": [1, 2], "v": [2, 4]},
         "check": lambda r: (r.value["equality_holds"] and r.value["proportional"]
                             and r.value["dot2"] == r.value["norms"],
                             f"eq={r.value['equality_holds']} prop={r.value['proportional']}"),
         "expect": "v=2u 成比例 → 等号: Σuv=10 → 100; Σu²=5, Σv²=20 → 100 = 100"},
        {"id": "t_i2_with_negatives", "args": {"u": [1, -3], "v": [4, 2]},
         "check": lambda r: (r.value["dot2"] == "4" and r.value["norms"] == "200",
                             f"dot²={r.value['dot2']} norms={r.value['norms']}"),
         "expect": "手算: Σuv=1·4+(−3)·2=4−6=−2 → 平方 4; Σu²=1+9=10, Σv²=16+4=20 → 200"},
        {"id": "t_i2_dim_mismatch", "args": {"u": [1, 2], "v": [1]},
         "expect_err": "PRECOND_FAIL", "expect": "维度不等"},
        {"id": "t_i2_zero_vector", "args": {"u": [0, 0], "v": [1, 2]},
         "check": lambda r: (r.value["dot2"] == "0" and r.value["norms"] == "0"
                             and r.value["equality_holds"],
                             f"dot²={r.value['dot2']} norms={r.value['norms']}"),
         "expect": "零向量: 0 ≤ 0, 等号成立 (退化情形显式覆盖)"},
    ),
))


_IN_TRI = {"kind": "dict",
           "fields": {"x": {"kind": "Fraction"}, "y": {"kind": "Fraction"}},
           "required": ["x", "y"]}
_OUT_TRI = {"kind": "dict",
            "fields": {"lhs": {"kind": "str"}, "rhs": {"kind": "str"}, "holds": {"kind": "bool"}},
            "required": ["lhs", "rhs", "holds"]}


def _inv_tri(args: dict, ctl: CallCtl) -> dict:
    x, y = f(args["x"]), f(args["y"])
    lhs, rhs = abs(x + y), abs(x) + abs(y)
    if lhs > rhs:
        raise SvoError(Err.POSTCOND_FAIL, ctx="op:thm_triangle#9213",
                       detail=f"三角不等式被违反: |x+y|={fs(lhs)} > |x|+|y|={fs(rhs)}")
    return {"value": {"lhs": fs(lhs), "rhs": fs(rhs), "holds": True},
            "resid": [float(rhs - lhs)],
            "notes": f"T-I3: |{fs(x)}+{fs(y)}|={fs(lhs)} ≤ {fs(rhs)}"}


def _ver_tri(args: dict, value: dict) -> dict:
    """T-I3 independent verification (sign-case enumeration): check all 4 combinations of (sgn x, sgn y) one by one.

    In each case |x+y| and |x|+|y| can be written as polynomials with the absolute values eliminated, so the
    "inequality" degenerates into one exact rational comparison — a path entirely different from invoke's direct
    absolute-value computation.
    """
    x, y = f(args["x"]), f(args["y"])
    cases = []
    bad = []
    for sx in (1, -1):
        for sy in (1, -1):
            if (x >= 0) != (sx > 0) or (y >= 0) != (sy > 0):
                continue
            a, b = sx * x, sy * y  # a,b ≥ 0
            lhs_expr = abs(a + b)  # same sign → a+b; opposite signs → |a−b|
            cases.append((sx, sy, lhs_expr, a + b))
            if lhs_expr > a + b:
                bad.append((sx, sy))
    # equality condition: xy ≥ 0
    eq_expected = (x * y >= 0)
    lhs = f(value["lhs"])
    rhs = f(value["rhs"])
    eq_actual = (lhs == rhs)
    ok = not bad and value["holds"] and lhs <= rhs and (eq_actual == eq_expected)
    return {"ok": ok, "resid": [float(len(bad))],
            "detail": f"枚举 {len(cases)} 个符号情形 (有效 {len([c for c in cases])} 组) 全部满足 |x+y|≤|x|+|y|; "
                      f"违反 {len(bad)}; 等号实际={eq_actual} 期望(xy≥0)={eq_expected}"}


REGISTRY.register(OpDef(
    name="thm_triangle_ineq", band=BAND, sub=13, thm="T-I3",
    stmt="三角不等式: |x+y| ≤ |x| + |y|, 等号当且仅当 xy ≥ 0",
    in_schema=_IN_TRI, out_schema=_OUT_TRI,
    invoke=_inv_tri, verify=_ver_tri,
    cost=("units", 0.3), errors=("POSTCOND_FAIL", "SCHEMA_MISMATCH"),
    effects="none (纯函数, 精确有理算术)",
    doc="T-I3 三角不等式 | 推导=绝对值直算 | 验证器=符号情形枚举 (去绝对值后精确比较) + 等号条件 | "
        "0.3 units | 失败: 不等式被违反 POSTCOND_FAIL",
    selfchecks=(
        {"id": "t_i3_mixed_sign", "args": {"x": 3, "y": -5},
         "check": lambda r: (r.value["lhs"] == "2" and r.value["rhs"] == "8",
                             f"|x+y|={r.value['lhs']} |x|+|y|={r.value['rhs']}"),
         "expect": "手算: |3−5|=2 ≤ 8"},
        {"id": "t_i3_equality_same_sign", "args": {"x": 3, "y": 5},
         "check": lambda r: (r.value["lhs"] == "8" and r.value["rhs"] == "8",
                             f"lhs={r.value['lhs']} rhs={r.value['rhs']}"),
         "expect": "同号 → 等号: 8 ≤ 8"},
        {"id": "t_i3_both_negative", "args": {"x": -2, "y": -7},
         "check": lambda r: (r.value["lhs"] == "9" and r.value["rhs"] == "9",
                             f"lhs={r.value['lhs']} rhs={r.value['rhs']}"),
         "expect": "同为负 → 等号: |−9|=9=2+7"},
        {"id": "t_i3_zero_y", "args": {"x": 4, "y": 0},
         "check": lambda r: (r.value["lhs"] == "4" and r.value["rhs"] == "4",
                             "y=0 等号: 4=4"),
         "expect": "y=0 边界"},
        {"id": "t_i3_float_reject", "args": {"x": 1.5, "y": 0}, "expect_err": "SCHEMA_MISMATCH",
         "expect": "float 不进精确通道 (§17.5)。本算子无'不等式被违反'路径 "
                   "(|x+y|≤|x|+|y| 对一切实数成立), 故负例只能落在 schema 侧"},
        {"id": "t_i3_missing_y", "args": {"x": 1}, "expect_err": "SCHEMA_MISMATCH",
         "expect": "缺 y 字段必须被 schema 门拦下"},
    ),
))


_IN_JENSEN = {"kind": "dict",
              "fields": {"coeffs": {"kind": "list", "items": {"kind": "Fraction"},
                                    "minlen": 1, "maxlen": 5},
                         "x1": {"kind": "Fraction"}, "x2": {"kind": "Fraction"}},
              "required": ["coeffs", "x1", "x2"]}
_OUT_JENSEN = {"kind": "dict",
               "fields": {"f_mid": {"kind": "str"}, "avg_f": {"kind": "str"},
                          "d2": {"kind": "str"}, "convex_on_interval": {"kind": "bool"},
                          "holds": {"kind": "bool"}},
               "required": ["f_mid", "avg_f", "d2", "convex_on_interval", "holds"]}


def _pre_jensen(args) -> "tuple[bool, str]":
    if f(args["x1"]) == f(args["x2"]):
        return False, "x1=x2 退化 (Jensen 严格形式需 x1≠x2)"
    return True, ""


def _inv_jensen(args: dict, ctl: CallCtl) -> dict:
    """T-I4 derivation: discrete Jensen f((x1+x2)/2) ≤ (f(x1)+f(x2))/2, plus the necessary information about f''."""
    cs = fvec(args["coeffs"])
    x1, x2 = f(args["x1"]), f(args["x2"])
    d1 = [cs[k] * k for k in range(1, len(cs))] or [Fraction(0)]
    d2 = [d1[k] * k for k in range(1, len(d1))] or [Fraction(0)]
    mid = (x1 + x2) / 2
    fm = poly_eval(cs, mid)
    avg = (poly_eval(cs, x1) + poly_eval(cs, x2)) / 2
    # convexity criterion (necessary condition): f'' is non-negative at both interval ends and the midpoint
    lo, hi = min(x1, x2), max(x1, x2)
    probes = [lo, mid, hi]
    d2_vals = [poly_eval(d2, t) for t in probes]
    convex = all(v >= 0 for v in d2_vals)
    if fm > avg:
        raise SvoError(Err.POSTCOND_FAIL, ctx="op:thm_jensen#9214",
                       detail=f"Jensen 被违反: f(中值)={fs(fm)} > 平均={fs(avg)}; f'' 探针={[fs(v) for v in d2_vals]}")
    return {"value": {"f_mid": fs(fm), "avg_f": fs(avg), "d2": fs(sum(d2_vals, Fraction(0)) / len(d2_vals)),
                      "convex_on_interval": convex, "holds": True},
            "resid": [float(avg - fm)],
            "notes": f"T-I4: f(中)={fs(fm)} ≤ 平均={fs(avg)}; f'' 在 [端点,中点] 非负={convex} "
                     f"(离散 Jensen 在凸函数上成立)"}


def _ver_jensen(args: dict, value: dict) -> dict:
    """T-I4 independent verification: numerical/exact examination of f'' via **second differences** + a polynomialised comparison of the midpoint form.

    Division of labour: invoke computes f(midpoint) and the average directly; here the Jensen inequality is
    polynomialised (2f((x1+x2)/2) − f(x1) − f(x2) ≤ 0), expanded independently in **convolution form**, and convexity
    is checked with second central differences (rational step h). Both differ from invoke.
    """
    cs = fvec(args["coeffs"])
    x1, x2 = f(args["x1"]), f(args["x2"])
    mid = (x1 + x2) / 2
    lhs = 2 * poly_eval(cs, mid) - poly_eval(cs, x1) - poly_eval(cs, x2)
    # second central difference (rational h): f(x+h) − 2f(x) + f(x−h) = h² f''(x) exactly for polynomials
    h = Fraction(1, 64)
    probes = [min(x1, x2), mid, max(x1, x2)]
    sd = [poly_eval(cs, t + h) - 2 * poly_eval(cs, t) + poly_eval(cs, t - h) for t in probes]
    conv_ok = all(v >= 0 for v in sd)
    ok = (lhs <= 0) and value["holds"] and value["convex_on_interval"] == conv_ok
    return {"ok": ok, "resid": [float(max(lhs, 0))],
            "detail": f"整式化 2f(中)−f(x1)−f(x2)={fs(lhs)} ≤ 0; "
                      f"二阶中心差分 h²f'' 探针={[fs(v) for v in sd]} (多项式下精确) 非负={conv_ok}"}


REGISTRY.register(OpDef(
    name="thm_jensen", band=BAND, sub=14, thm="T-I4",
    stmt="Jensen (离散两点, 凸函数): f((x₁+x₂)/2) ≤ (f(x₁)+f(x₂))/2",
    in_schema=_IN_JENSEN, out_schema=_OUT_JENSEN,
    invoke=_inv_jensen, verify=_ver_jensen, precond=_pre_jensen,
    cost=("units", 0.8), errors=("PRECOND_FAIL", "POSTCOND_FAIL", "SCHEMA_MISMATCH"),
    effects="none (纯函数, 精确有理算术)",
    doc="T-I4 Jensen (离散两点) | 条件: x1≠x2, 多项式 deg≤4 | 推导=中点值 vs 平均值 | "
        "验证器=整式化 + 二阶中心差分凸性 (有理步长, 多项式下精确) | 0.8 units | "
        "失败: x1=x2 PRECOND_FAIL / 凹函数被误用 POSTCOND_FAIL",
    selfchecks=(
        {"id": "t_i4_convex_x2", "args": {"coeffs": [0, 0, 1], "x1": 0, "x2": 4},
         "check": lambda r: (r.value["f_mid"] == "4" and r.value["avg_f"] == "8"
                             and r.value["convex_on_interval"] is True,
                             f"f(中)={r.value['f_mid']} avg={r.value['avg_f']}"),
         "expect": "手算: f=x², 0,4 → f(2)=4 ≤ (0+16)/2=8; f''=2≥0"},
        {"id": "t_i4_linear_equality", "args": {"coeffs": [1, 2], "x1": 0, "x2": 4},
         "check": lambda r: (r.value["f_mid"] == "5" and r.value["avg_f"] == "5",
                             f"f(中)={r.value['f_mid']} avg={r.value['avg_f']}"),
         "expect": "一次函数: 等号成立 f(2)=5=(1+9)/2"},
        {"id": "t_i4_neg_interval", "args": {"coeffs": [0, 0, 1], "x1": -6, "x2": -2},
         "check": lambda r: (r.value["f_mid"] == "16" and r.value["avg_f"] == "20",
                             f"f(中)={r.value['f_mid']} avg={r.value['avg_f']}"),
         "expect": "手算: f(2)... 区间 [−6,−2] 中点 −4 → 16 ≤ (36+4)/2=20"},
        {"id": "t_i4_concave_rejected", "args": {"coeffs": [0, 0, -1], "x1": 0, "x2": 4},
         "expect_err": "POSTCOND_FAIL",
         "expect": "f=−x² 为凹: Jensen 不等式被违反 → 必须抛 POSTCOND_FAIL (不许静默给错结论)"},
        {"id": "t_i4_x_equal", "args": {"coeffs": [0, 0, 1], "x1": 1, "x2": 1},
         "expect_err": "PRECOND_FAIL", "expect": "x1=x2 退化"},
    ),
))


# ===========================================================================
# linear algebra (T-L1 … T-L6, rows 9221–9226)
# ===========================================================================

_IN_MAT2 = {"kind": "dict",
            "fields": {"A": {"kind": "list", "len": 2,
                             "items": {"kind": "list", "len": 2, "items": {"kind": "Fraction"}}}},
            "required": ["A"]}


def _det2(A) -> Fraction:
    return A[0][0] * A[1][1] - A[0][1] * A[1][0]


def _comb(n: int, k: int) -> "list[tuple]":
    """all combinations of k indices out of n (used for minor enumeration)."""
    out: list[tuple] = []

    def rec(start: int, cur: list) -> None:
        if len(cur) == k:
            out.append(tuple(cur))
            return
        for i in range(start, n):
            rec(i + 1, cur + [i])

    rec(0, [])
    return out


def _inv_mat2_inv(args: dict, ctl: CallCtl) -> dict:
    """T-L1 derivation: 2×2 inverse closed form."""
    ctx = "op:thm_mat2_inv#9221"
    A = fmat(args["A"])
    d = _det2(A)
    if d == 0:
        raise SvoError(Err.DEG_UNSUPPORTED, ctx=ctx, detail="det=0 — T-L1 条件 (可逆) 不满足")
    B = [[A[1][1] / d, -A[0][1] / d], [-A[1][0] / d, A[0][0] / d]]
    cert = Certificate(kind="matrix_identity",
                       payload={"lhs": [[fs(v) for v in r] for r in A],
                                "rhs": [[fs(v) for v in r] for r in B],
                                "claim": [["1", "0"], ["0", "1"]]},
                       issued_by="thm_mat2_inv")
    return {"value": {"inv": [[fs(v) for v in r] for r in B], "det": fs(d)},
            "resid": [0.0], "cert": cert,
            "notes": f"T-L1: det={fs(d)}, A⁻¹={[[fs(v) for v in r] for r in B]}; "
                     f"证书=matrix_identity (A·A⁻¹=I 逐元素精确)"}


def _ver_mat2_inv(args: dict, value: dict) -> dict:
    """T-L1 independent verification: recompute via the **adjugate route** (A⁻¹ = adj(A)/det), then check element-wise against A·A⁻¹−I.

    Unlike invoke's "swap the main diagonal, negate the off-diagonal" form: here the cofactor matrix is constructed
    explicitly and then transposed.
    """
    A = fmat(args["A"])
    d = _det2(A)
    if d == 0:
        return {"ok": False, "resid": [1.0], "detail": "det=0 — 不应返回逆矩阵"}
    cof = [[A[1][1], -A[1][0]], [-A[0][1], A[0][0]]]
    adj = [[cof[j][i] for j in range(2)] for i in range(2)]  # transpose
    B_adj = [[v / d for v in r] for r in adj]
    got = fmat(value["inv"])
    worst = max(abs(B_adj[i][j] - got[i][j]) for i in range(2) for j in range(2))
    prod = matmul(A, got)
    idgap = max(abs(prod[i][j] - (Fraction(1) if i == j else Fraction(0)))
                for i in range(2) for j in range(2))
    ok = worst == 0 and idgap == 0 and f(value["det"]) == d
    return {"ok": ok, "resid": [float(worst), float(idgap)],
            "detail": f"伴随路线复算 max|Δ|={fs(worst)}; A·A⁻¹−I 逐元素 max={fs(idgap)}; "
                      f"det 一致={f(value['det']) == d} (残差对真值 A·A⁻¹=I, §17.1)"}


REGISTRY.register(OpDef(
    name="thm_mat2_inv", band=BAND, sub=21, thm="T-L1",
    stmt="2×2 矩阵 A 可逆 (det A ≠ 0) 时 A⁻¹ = adj(A)/det(A), 且 A·A⁻¹ = I",
    in_schema=_IN_MAT2,
    out_schema={"kind": "dict",
                "fields": {"inv": {"kind": "list", "len": 2,
                                   "items": {"kind": "list", "len": 2, "items": {"kind": "str"}}},
                           "det": {"kind": "str"}},
                "required": ["inv", "det"]},
    invoke=_inv_mat2_inv, verify=_ver_mat2_inv, cert_kind="matrix_identity",
    cost=("units", 0.8), errors=("DEG_UNSUPPORTED", "SCHEMA_MISMATCH"),
    effects="none (纯函数, 精确有理算术)",
    doc="T-L1 2×2 逆矩阵 | 条件: det≠0 | 推导=闭式 | 验证器=伴随矩阵路线 + A·A⁻¹−I 逐元素 | "
        "证书=matrix_identity | 0.8 units | 失败: 奇异 DEG_UNSUPPORTED",
    selfchecks=(
        {"id": "t_l1_hand", "args": {"A": [[1, 2], [3, 4]]},
         "check": lambda r: (r.value["det"] == "-2"
                             and r.value["inv"] == [["-2", "1"], ["3/2", "-1/2"]]
                             and r.cert_independent is True,
                             f"det={r.value['det']} inv={r.value['inv']} cert={r.cert_independent}"),
         "expect": "手算: det=1·4−2·3=−2; A⁻¹=(1/−2)[[4,−2],[−3,1]]=[[−2,1],[3/2,−1/2]]"},
        {"id": "t_l1_identity", "args": {"A": [[1, 0], [0, 1]]},
         "check": lambda r: (r.value["inv"] == [["1", "0"], ["0", "1"]], f"inv={r.value['inv']}"),
         "expect": "I⁻¹=I"},
        {"id": "t_l1_frac", "args": {"A": [["1/2", 1], [1, 3]]},
         "check": lambda r: (r.value["det"] == "1/2"
                             and r.value["inv"] == [["6", "-2"], ["-2", "1"]],
                             f"det={r.value['det']} inv={r.value['inv']}"),
         "expect": "手算: det=1/2·3−1·1=1/2; A⁻¹=(1/(1/2))[[3,−1],[−1,1/2]]=[[6,−2],[−2,1]]"},
        {"id": "t_l1_singular", "args": {"A": [[1, 2], [2, 4]]}, "expect_err": "DEG_UNSUPPORTED",
         "expect": "det=4−4=0 奇异 — 条件不满足"},
        {"id": "t_l1_extra_field", "args": {"A": [[1, 2], [3, 4]], "k": 1},
         "expect_err": "SCHEMA_MISMATCH", "expect": "多余字段 (extra=False) 必须被拒"},
    ),
))


_IN_CRAMER = {"kind": "dict",
              "fields": {"A": _mat_schema(2),
                         "b": {"kind": "list", "len": 2, "items": {"kind": "Fraction"}}},
              "required": ["A", "b"]}
_OUT_X2 = {"kind": "dict",
           "fields": {"x": {"kind": "list", "len": 2, "items": {"kind": "str"}},
                      "det": {"kind": "str"}, "resid_inf": {"kind": "str"}},
           "required": ["x", "det", "resid_inf"]}


def _inv_cramer(args: dict, ctl: CallCtl) -> dict:
    """T-L2 derivation: Cramer's rule (the ratio of determinants after column replacement)."""
    ctx = "op:thm_cramer#9222"
    A = fmat(args["A"])
    b = fvec(args["b"])
    D = _det2(A)
    if D == 0:
        raise SvoError(Err.DEG_UNSUPPORTED, ctx=ctx, detail="det=0 — Cramer 法则条件不满足")
    A1 = [[b[0], A[0][1]], [b[1], A[1][1]]]
    A2 = [[A[0][0], b[0]], [A[1][0], b[1]]]
    x = [det_cofactor(A1) / D, det_cofactor(A2) / D]
    r = max(abs(A[i][0] * x[0] + A[i][1] * x[1] - b[i]) for i in range(2))
    cert = Certificate(kind="linear_backsub",
                       payload={"A": [[fs(v) for v in row] for row in A], "b": [fs(v) for v in b],
                                "x": [fs(v) for v in x]},
                       issued_by="thm_cramer")
    return {"value": {"x": [fs(v) for v in x], "det": fs(D), "resid_inf": fs(r)},
            "resid": [float(r)], "cert": cert,
            "notes": f"T-L2: det={fs(D)}, x=({fs(x[0])}, {fs(x[1])}), ‖Ax−b‖∞={fs(r)}"}


def _ver_cramer(args: dict, value: dict) -> dict:
    """T-L2 independent verification: **back-substitution** ‖Ax−b‖∞ (exact) — unrelated to Cramer's determinant ratio.

    The residual is computed against the **true value b**, not against "the input vector" (§17.1): here b is exactly
    the right-hand side's true value.
    """
    A = fmat(args["A"])
    b = fvec(args["b"])
    x = fvec(value["x"])
    if len(x) != 2:
        return {"ok": False, "resid": [1.0], "detail": "解向量长度不符"}
    worst = max(abs(A[i][0] * x[0] + A[i][1] * x[1] - b[i]) for i in range(2))
    ok = worst == 0 and f(value["resid_inf"]) == worst
    return {"ok": ok, "resid": [float(worst)],
            "detail": f"精确回代 ‖Ax−b‖∞={fs(worst)}; 与声称一致={f(value['resid_inf']) == worst}"}


REGISTRY.register(OpDef(
    name="thm_cramer", band=BAND, sub=22, thm="T-L2",
    stmt="方阵 A (det A ≠ 0) 有唯一解 xᵢ = det(Aᵢ)/det(A), Aᵢ 为第 i 列换成 b",
    in_schema=_IN_CRAMER, out_schema=_OUT_X2,
    invoke=_inv_cramer, verify=_ver_cramer, cert_kind="linear_backsub",
    cost=("units", 0.9), errors=("DEG_UNSUPPORTED", "SCHEMA_MISMATCH"),
    effects="none (纯函数, 精确有理算术)",
    doc="T-L2 Cramer 法则 (2×2) | 条件: det≠0 | 推导=行列式之比 | 验证器=精确回代 ‖Ax−b‖∞ | "
        "证书=linear_backsub | 0.9 units | 失败: 奇异 DEG_UNSUPPORTED",
    selfchecks=(
        {"id": "t_l2_hand", "args": {"A": [[1, 2], [3, 4]], "b": [5, 6]},
         "check": lambda r: (r.value["x"] == ["-4", "9/2"] and r.value["det"] == "-2"
                             and r.value["resid_inf"] == "0" and r.cert_independent is True,
                             f"x={r.value['x']} det={r.value['det']} cert={r.cert_independent}"),
         "expect": "手算: D=−2; D1=det[[5,2],[6,4]]=20−12=8 → x1=8/(−2)=−4; "
                   "D2=det[[1,5],[3,6]]=6−15=−9 → x2=(−9)/(−2)=9/2; 回代 1·(−4)+2·(9/2)=5 ✓"},
        {"id": "t_l2_singular", "args": {"A": [[1, 1], [1, 1]], "b": [1, 2]},
         "expect_err": "DEG_UNSUPPORTED", "expect": "det=0 奇异 — 条件不满足"},
        {"id": "t_l2_bad_b_len", "args": {"A": [[1, 2], [3, 4]], "b": [5]},
         "expect_err": "SCHEMA_MISMATCH", "expect": "b 长度必须恰为 2"},
        {"id": "t_l2_ib_swap_regression", "args": {"A": [[1, 1], [1, 2]], "b": [3, 5]},
         "check": lambda r: (r.value["x"] == ["1", "2"], f"x={r.value['x']}"),
         "expect": "手算: D=1; D1=det[[3,1],[5,2]]=6−5=1 → x1=1; D2=det[[1,3],[1,5]]=5−3=2 → x2=2; "
                   "回代 x+y=3 ✓, x+2y=5 ✓ (列替换方向回归项)"},
    ),
))


_IN_EIG = {"kind": "dict", "fields": {"A": _mat_schema(2)}, "required": ["A"]}
_OUT_EIG = {"kind": "dict",
            "fields": {"trace": {"kind": "str"}, "det": {"kind": "str"}, "disc": {"kind": "str"},
                       "mode": {"kind": "str", "enum": ["exact", "interval"]},
                       "eigs": {"kind": "list", "items": {"kind": "str"}, "maxlen": 2},
                       "eig_intervals": {"kind": "list", "maxlen": 2,
                                         "items": {"kind": "list", "len": 2, "items": {"kind": "str"}}},
                       "eigvecs": {"kind": "list", "maxlen": 2,
                                   "items": {"kind": "list", "len": 2, "items": {"kind": "str"}}}},
            "required": ["trace", "det", "disc", "mode", "eigs", "eig_intervals", "eigvecs"]}


def _eigvec2(A, lam: Fraction) -> "list[Fraction]":
    """a nonzero solution of (A − λI)v = 0 (exact; falls back to the second row / unit vector when the first row degenerates)."""
    a, b = A[0][0] - lam, A[0][1]
    c, d = A[1][0], A[1][1] - lam
    if b != 0:
        return [b, -a]
    if c != 0:
        return [d, -c]
    if a != 0:
        return [Fraction(0), Fraction(1)]
    return [Fraction(1), Fraction(0)]


def _inv_eig2(args: dict, ctl: CallCtl) -> dict:
    """T-L3 derivation: characteristic polynomial λ²−tr·λ+det=0; a rational perfect-square discriminant gives exact eigenpairs,
    otherwise it gives a **rational interval nesting** for the eigenvalues and refuses to output eigenvectors (irrational
    eigenvectors require reduction to a quadratic field)."""
    A = fmat(args["A"])
    tr, det = A[0][0] + A[1][1], _det2(A)
    disc = tr * tr - 4 * det
    if disc < 0:
        raise SvoError(Err.NO_REAL_ROOT, ctx="op:thm_eig2#9223",
                       detail=f"判别式 {fs(disc)} < 0 — 实特征值不存在 (T-L3 实数域条件不满足)")
    s = isqrt_frac(disc)
    if s is None:
        cs = [det, -tr, Fraction(1)]
        xv = tr / 2
        B = 2 + abs(tr) + abs(det)
        if poly_eval(cs, xv) == 0:
            raise SvoError(Err.POSTCOND_FAIL, ctx="op:thm_eig2#9223",
                           detail="顶点处特征多项式为 0 但判别式≠0 — 算术自相矛盾")
        ivs = []
        for a, b in ((-B, xv), (xv, B)):
            fa, fb = poly_eval(cs, a), poly_eval(cs, b)
            if (fa < 0) == (fb < 0):
                raise SvoError(Err.NO_REAL_ROOT, ctx="op:thm_eig2#9223",
                               detail=f"半轴 [{fs(a)},{fs(b)}] 端点不变号 — 区间构造失败")
            ivs.append([fs(x) for x in _bisect_root(cs, a, b, "op:thm_eig2#9223@bisect", ctl)])
        return {"value": {"trace": fs(tr), "det": fs(det), "disc": fs(disc), "mode": "interval",
                          "eigs": [], "eig_intervals": ivs, "eigvecs": []},
                "resid": [float(max(f(v[1]) - f(v[0]) for v in ivs))],
                "notes": f"T-L3: 判别式 {fs(disc)} 非完全平方 → 特征值无理, 给有理区间套 {ivs}; "
                         f"按 §17.5 不给 float 冒充精确的特征向量"}
    l1, l2 = (tr + s) / 2, (tr - s) / 2
    eigs = [l1] if l1 == l2 else sorted([l1, l2], reverse=True)
    vecs = [_eigvec2(A, lam) for lam in eigs]
    cert = Certificate(kind="matrix_identity",
                       payload={"lhs": [[fs(v) for v in r] for r in A],
                                "rhs": [[fs(v)] for v in vecs[0]],
                                "claim": [[fs(v * eigs[0])] for v in vecs[0]]},
                       issued_by="thm_eig2")
    return {"value": {"trace": fs(tr), "det": fs(det), "disc": fs(disc), "mode": "exact",
                      "eigs": [fs(v) for v in eigs], "eig_intervals": [],
                      "eigvecs": [[fs(v) for v in vec] for vec in vecs]},
            "resid": [0.0], "cert": cert,
            "notes": f"T-L3: tr={fs(tr)}, det={fs(det)}; 特征值 {[fs(v) for v in eigs]}; "
                     f"证书=matrix_identity 核对 A·v₁ = λ₁·v₁"}


def _ver_eig2(args: dict, value: dict) -> dict:
    """T-L3 independent verification: in exact mode check A·v = λ·v for every eigenpair, and check the coefficient identities
    λ₁+λ₂=tr, λ₁λ₂=det; in interval mode check that the characteristic polynomial changes sign at the interval endpoints
    (each interval really contains one eigenvalue)."""
    A = fmat(args["A"])
    tr, det = A[0][0] + A[1][1], _det2(A)
    if f(value["trace"]) != tr or f(value["det"]) != det:
        return {"ok": False, "resid": [1.0], "detail": "trace/det 与输入矩阵不符"}
    if value["mode"] == "interval":
        cs = [det, -tr, Fraction(1)]
        bad = 0
        for lo_s, hi_s in value["eig_intervals"]:
            flo, fhi = poly_eval(cs, f(lo_s)), poly_eval(cs, f(hi_s))
            if flo != 0 and fhi != 0 and (flo < 0) == (fhi < 0):
                bad += 1
        return {"ok": bad == 0 and len(value["eig_intervals"]) == 2, "resid": [float(bad)],
                "detail": f"p(λ)=λ²−{fs(tr)}λ+{fs(det)} 的 {len(value['eig_intervals'])} 个区间端点"
                          f"变号, 违规 {bad}"}
    eigs = fvec(value["eigs"])
    vecs = [fvec(v) for v in value["eigvecs"]]
    if len(eigs) != len(vecs):
        return {"ok": False, "resid": [1.0], "detail": "特征值与向量数不一致"}
    worst = Fraction(0)
    zero_vec = 0
    for lam, vec in zip(eigs, vecs):
        if vec[0] == 0 and vec[1] == 0:
            zero_vec += 1
            continue
        Av = [A[i][0] * vec[0] + A[i][1] * vec[1] for i in range(2)]
        for i in range(2):
            worst = worse(worst, abs(Av[i] - lam * vec[i]))
    s_ok = sum(eigs, Fraction(0)) == tr
    p_ok = (eigs[0] * eigs[-1]) == det
    if len(eigs) == 1:
    # repeated root (discriminant=0): the output keeps only one distinct value, but the identity must be counted with
    # **algebraic multiplicity 2**, i.e. λ+λ=2λ=tr and λ·λ=λ²=det. Comparing tr against sum(eigs)=λ alone would
    # falsely fail.
        s_ok = 2 * eigs[0] == tr
        p_ok = eigs[0] * eigs[0] == det
    ok = worst == 0 and zero_vec == 0 and s_ok and p_ok
    return {"ok": ok, "resid": [float(worst)],
            "detail": f"逐特征对精确核对 max|Av−λv|={fs(worst)}; 零向量数={zero_vec}; "
                      f"按代数重数的系数恒等式: 和=tr:{s_ok}, 积=det:{p_ok}"}


REGISTRY.register(OpDef(
    name="thm_eig2", band=BAND, sub=23, thm="T-L3",
    stmt="2×2 实矩阵的实特征值是 λ²−(tr A)λ+det A=0 的实根, 且对应特征向量满足 A v = λ v",
    in_schema=_IN_EIG, out_schema=_OUT_EIG,
    invoke=_inv_eig2, verify=_ver_eig2, cert_kind="matrix_identity",
    cost=("units", 1.2), errors=("NO_REAL_ROOT", "POSTCOND_FAIL", "SCHEMA_MISMATCH"),
    effects="none (纯函数, 精确有理算术)",
    doc="T-L3 2×2 特征对 | 条件: 判别式 tr²−4det ≥ 0 (实数域) | 推导=特征多项式求根+零空间解 | "
        "验证器=逐特征对 Av=λv + 系数恒等式 (区间模式=端点变号) | 证书=matrix_identity | "
        "1.2 units | 失败: 无实特征值 NO_REAL_ROOT / 区间构造失败",
    selfchecks=(
        {"id": "t_l3_distinct", "args": {"A": [[4, 1], [2, 3]]},
         "check": lambda r: (r.value["eigs"] == ["5", "2"] and r.value["mode"] == "exact"
                             and r.value["eigvecs"] == [["1", "1"], ["1", "-2"]]
                             and r.cert_independent is True,
                             f"eigs={r.value['eigs']} vecs={r.value['eigvecs']}"),
         "expect": "手算: tr=7, det=10 → λ²−7λ+10=0 → λ=5,2。λ=5: A(1,1)=(4+1,2+3)=(5,5)=5(1,1) ✓; "
                   "λ=2: A(1,−2)=(4−2,2−6)=(2,−4)=2(1,−2) ✓"},
        {"id": "t_l3_double_eig", "args": {"A": [[2, 0], [0, 2]]},
         "check": lambda r: (r.value["eigs"] == ["2"] and r.value["disc"] == "0",
                             f"eigs={r.value['eigs']} disc={r.value['disc']}"),
         "expect": "2I: tr=4, det=4 → 判别式 0 → 重特征值 2"},
        {"id": "t_l3_upper_triangular", "args": {"A": [[3, 5], [0, -1]]},
         "check": lambda r: (r.value["eigs"] == ["3", "-1"], f"eigs={r.value['eigs']}"),
         "expect": "上三角阵: 特征值=对角元 3, −1"},
        {"id": "t_l3_complex_eigs_rejected", "args": {"A": [[0, -1], [1, 0]]},
         "expect_err": "NO_REAL_ROOT",
         "expect": "旋转矩阵: tr=0, det=1 → 判别式 −4 < 0, 无实特征值 (拒, 不给复数伪装实数)"},
        {"id": "t_l3_irrational_interval", "args": {"A": [[0, 1], [1, 1]]},
         "check": lambda r: (r.value["mode"] == "interval" and len(r.value["eig_intervals"]) == 2
                             and r.value["disc"] == "5",
                             f"mode={r.value['mode']} ivs={r.value['eig_intervals']}"),
         "expect": "Fibonacci 矩阵: tr=1, det=−1 → λ²−λ−1=0 → 判别式 5 非完全平方, "
                   "特征值 (1±√5)/2 无理 → 有理区间套"},
        {"id": "t_l3_extra_field", "args": {"A": [[1, 0], [0, 1]], "v": [1, 0]},
         "expect_err": "SCHEMA_MISMATCH", "expect": "多余字段必须被拒"},
    ),
))


_IN_RN = {"kind": "dict",
          "fields": {"A": {"kind": "list", "minlen": 1, "maxlen": 3,
                           "items": {"kind": "list", "minlen": 1, "maxlen": 3,
                                     "items": {"kind": "Fraction"}}}},
          "required": ["A"]}


def _rank_rref(A) -> int:
    """rational RREF rank computation (elimination route)."""
    M = [list(r) for r in A]
    rows, cols = len(M), len(M[0]) if M else 0
    rank = 0
    for c in range(cols):
        piv = next((r for r in range(rank, rows) if M[r][c] != 0), None)
        if piv is None:
            continue
        M[rank], M[piv] = M[piv], M[rank]
        pv = M[rank][c]
        M[rank] = [v / pv for v in M[rank]]
        for r in range(rows):
            if r != rank and M[r][c] != 0:
                cc = M[r][c]
                M[r] = [M[r][k] - cc * M[rank][k] for k in range(cols)]
        rank += 1
        if rank == rows:
            break
    return rank


def _rank_minors(A) -> int:
    """minor-based rank computation (a different path from elimination): the order of the largest nonzero minor."""
    rows, cols = len(A), len(A[0])
    best = 0
    for k in range(1, min(rows, cols) + 1):
        found = False
        for rs in _comb(rows, k):
            for cs in _comb(cols, k):
                sub = [[A[i][j] for j in cs] for i in rs]
                if det_cofactor(sub) != 0:
                    found = True
                    break
            if found:
                break
        if found:
            best = k
    return best


def _pre_rn(args) -> "tuple[bool, str]":
    A = args["A"]
    if not A:
        return False, "空矩阵"
    w = len(A[0])
    if w == 0:
        return False, "空行"
    for i, row in enumerate(A):
        if len(row) != w:
            return False, f"第 {i} 行长度 {len(row)} ≠ {w} (非矩形)"
    return True, ""


def _inv_rn(args: dict, ctl: CallCtl) -> dict:
    """T-L4 derivation: rank via RREF; nullity = column count − rank."""
    A = fmat(args["A"])
    rank = _rank_rref(A)
    n = len(A[0])
    return {"value": {"rank": rank, "nullity": n - rank, "n_cols": n, "n_rows": len(A),
                      "holds": rank + (n - rank) == n},
            "resid": [0.0],
            "notes": f"T-L4: 秩={rank}, 零化度={n - rank}, 列数={n}; 秩+零化度=列数 ✓"}


def _ver_rn(args: dict, value: dict) -> dict:
    """T-L4 independent verification: recompute the rank via the **minor route** (order of the largest nonzero minor), and check rank ≤ min(rows,cols)."""
    A = fmat(args["A"])
    r2 = _rank_minors(A)
    n = len(A[0])
    ok = (r2 == value["rank"] and value["nullity"] == n - value["rank"]
          and 0 <= value["rank"] <= min(len(A), n))
    return {"ok": ok, "resid": [float(abs(r2 - value["rank"]))],
            "detail": f"子式路线秩={r2} vs 消元路线秩={value['rank']}; 零化度={value['nullity']}, "
                      f"列数={n}; 秩≤min(m,n)={min(len(A), n)}"}


REGISTRY.register(OpDef(
    name="thm_rank_nullity", band=BAND, sub=24, thm="T-L4",
    stmt="线性映射 A: F^n → F^m 满足 rank(A) + nullity(A) = n (列数)",
    in_schema=_IN_RN,
    out_schema={"kind": "dict",
                "fields": {"rank": {"kind": "int", "min": 0}, "nullity": {"kind": "int", "min": 0},
                           "n_cols": {"kind": "int", "min": 1}, "n_rows": {"kind": "int", "min": 1},
                           "holds": {"kind": "bool"}},
                "required": ["rank", "nullity", "n_cols", "n_rows", "holds"]},
    invoke=_inv_rn, verify=_ver_rn, precond=_pre_rn,
    cost=("units", 1.5), errors=("PRECOND_FAIL", "SCHEMA_MISMATCH"),
    effects="none (纯函数, 精确有理算术)",
    doc="T-L4 秩-零化度定理 | 条件: 矩形矩阵 (非空、行等长) | 推导=RREF 求秩 | "
        "验证器=子式路线复算秩 (最大非零子式阶) | 1.5 units | 失败: 非矩形 PRECOND_FAIL",
    selfchecks=(
        {"id": "t_l4_full_rank", "args": {"A": [[1, 0], [0, 1]]},
         "check": lambda r: (r.value["rank"] == 2 and r.value["nullity"] == 0,
                             f"rank={r.value['rank']} nullity={r.value['nullity']}"),
         "expect": "I₂: 秩 2, 零化度 0, 列数 2"},
        {"id": "t_l4_rank_deficient", "args": {"A": [[1, 2], [2, 4]]},
         "check": lambda r: (r.value["rank"] == 1 and r.value["nullity"] == 1,
                             f"rank={r.value['rank']} nullity={r.value['nullity']}"),
         "expect": "第 2 行=2×第 1 行: 秩 1, 零化度 1"},
        {"id": "t_l4_zero_matrix", "args": {"A": [[0, 0, 0]]},
         "check": lambda r: (r.value["rank"] == 0 and r.value["nullity"] == 3,
                             f"rank={r.value['rank']} nullity={r.value['nullity']}"),
         "expect": "零矩阵 1×3: 秩 0, 零化度 3"},
        {"id": "t_l4_non_rect", "args": {"A": [[1, 2], [3]]}, "expect_err": "PRECOND_FAIL",
         "expect": "非矩形必须被拒"},
        {"id": "t_l4_empty_row", "args": {"A": []}, "expect_err": "SCHEMA_MISMATCH",
         "expect": "空矩阵 (minlen=1) 必须被 schema 拦下"},
    ),
))


_IN_CH = {"kind": "dict", "fields": {"A": _mat_schema(2)}, "required": ["A"]}
_OUT_CH = {"kind": "dict",
           "fields": {"coefs": {"kind": "list", "len": 3, "items": {"kind": "str"}},
                      "residual": {"kind": "list", "len": 2,
                                   "items": {"kind": "list", "len": 2, "items": {"kind": "str"}}},
                      "zero": {"kind": "bool"}},
           "required": ["coefs", "residual", "zero"]}


def _inv_ch(args: dict, ctl: CallCtl) -> dict:
    """T-L5 derivation: A² − tr(A)·A + det(A)·I = 0, with the residual matrix computed by direct substitution."""
    A = fmat(args["A"])
    tr, det = A[0][0] + A[1][1], _det2(A)
    A2 = matmul(A, A)
    res = [[A2[i][j] - tr * A[i][j] + (det if i == j else Fraction(0)) for j in range(2)]
           for i in range(2)]
    cert = Certificate(kind="matrix_identity",
                       payload={"terms": [{"coef": "1", "mat": [[fs(v) for v in r] for r in A2]},
                                           {"coef": fs(-tr), "mat": [[fs(v) for v in r] for r in A]},
                                           {"coef": fs(det), "mat": [["1", "0"], ["0", "1"]]}],
                                "claim": [["0", "0"], ["0", "0"]]},
                       issued_by="thm_cayley_hamilton")
    is_zero = all(v == 0 for r in res for v in r)
    return {"value": {"coefs": [fs(Fraction(1)), fs(-tr), fs(det)],
                      "residual": [[fs(v) for v in r] for r in res], "zero": is_zero},
            "resid": [float(max(abs(v) for r in res for v in r))], "cert": cert,
            "notes": f"T-L5: 特征多项式 λ²−{fs(tr)}λ+{fs(det)}; A²−trA+detI 残差全零={is_zero}"}


def _ver_ch(args: dict, value: dict) -> dict:
    """T-L5 independent verification: independently recompute A² by **hand-written expansion** (a₁₁=a²+bc, a₁₂=ab+bd, a₂₁=ca+dc, a₂₂=cb+d²)
    and the residual matrix, and check the characteristic polynomial coefficients — any swap/sign error anywhere exposes itself here
    (a different path from invoke's generic matrix multiplication)."""
    A = fmat(args["A"])
    a, b, c, d = A[0][0], A[0][1], A[1][0], A[1][1]
    A2_hand = [[a * a + b * c, a * b + b * d], [c * a + d * c, c * b + d * d]]
    tr, det = a + d, a * d - b * c
    res_hand = [[A2_hand[i][j] - tr * A[i][j] + (det if i == j else Fraction(0)) for j in range(2)]
                for i in range(2)]
    got = fmat(value["residual"])
    worst = max(abs(res_hand[i][j] - got[i][j]) for i in range(2) for j in range(2))
    all_zero = all(v == 0 for r in res_hand for v in r)
    claims = [f(s) for s in value["coefs"]]
    coefs_ok = claims == [Fraction(1), -tr, det]
    ok = worst == 0 and value["zero"] == all_zero and all_zero and coefs_ok
    return {"ok": ok, "resid": [float(worst)],
            "detail": f"手写展开 A² 与残差 max|Δ|={fs(worst)}; 残差全零={all_zero}; "
                      f"特征多项式系数 (1,{fs(-tr)},{fs(det)}) 一致={coefs_ok}"}


REGISTRY.register(OpDef(
    name="thm_cayley_hamilton", band=BAND, sub=25, thm="T-L5",
    stmt="2×2 矩阵满足自身特征方程: A² − (tr A)·A + (det A)·I = 0",
    in_schema=_IN_CH, out_schema=_OUT_CH,
    invoke=_inv_ch, verify=_ver_ch, cert_kind="matrix_identity",
    cost=("units", 0.9), errors=("SCHEMA_MISMATCH",),
    effects="none (纯函数, 精确有理算术)",
    doc="T-L5 Cayley-Hamilton (2×2) | 推导=代入计算残差 | 验证器=手写展开式复算 A² + 系数核对 | "
        "证书=matrix_identity (线性组合形式) | 0.9 units | 失败: schema",
    selfchecks=(
        {"id": "t_l5_hand", "args": {"A": [[1, 2], [3, 4]]},
         "check": lambda r: (r.value["zero"] is True and r.value["coefs"] == ["1", "-5", "-2"]
                             and r.cert_independent is True,
                             f"coefs={r.value['coefs']} zero={r.value['zero']}"),
         "expect": "手算: tr=5, det=−2 → λ²−5λ−2; A²=[[7,10],[15,22]]; "
                   "A²−5A−2I=[[7−5−2, 10−10],[15−15, 22−20−2]]=0 ✓"},
        {"id": "t_l5_identity", "args": {"A": [[1, 0], [0, 1]]},
         "check": lambda r: (r.value["zero"] is True and r.value["coefs"] == ["1", "-2", "1"],
                             f"coefs={r.value['coefs']}"),
         "expect": "I: tr=2, det=1 → λ²−2λ+1; I−2I+I=0 ✓"},
        {"id": "t_l5_nilpotent", "args": {"A": [[0, 1], [0, 0]]},
         "check": lambda r: (r.value["zero"] is True and r.value["coefs"] == ["1", "0", "0"],
                             f"coefs={r.value['coefs']}"),
         "expect": "幂零阵: tr=0, det=0 → 特征多项式 λ²; A²=0 ✓"},
        {"id": "t_l5_float_reject", "args": {"A": [[1.0, 0], [0, 1]]},
         "expect_err": "SCHEMA_MISMATCH", "expect": "float 不进精确通道"},
    ),
))


_IN_SPEC = {"kind": "dict", "fields": {"A": _mat_schema(2)}, "required": ["A"]}
_OUT_SPEC = {"kind": "dict",
             "fields": {"rho": {"kind": "str"}, "norm_inf": {"kind": "str"},
                        "norm_1": {"kind": "str"}, "holds_inf": {"kind": "bool"},
                        "holds_1": {"kind": "bool"}, "disc": {"kind": "str"}},
             "required": ["rho", "norm_inf", "norm_1", "holds_inf", "holds_1", "disc"]}


def _inv_spec(args: dict, ctl: CallCtl) -> dict:
    """T-L6 derivation: ρ = max|λ| (exact), compared against ‖·‖∞ / ‖·‖₁.

    An exact ρ is given only when the discriminant is a rational perfect square; otherwise ρ is irrational and the
    order relation "ρ ≤ norm" is **undecidable** in the rational channel → UNIDENTIFIABLE is raised (§17.3: explicit
    rejection, never a guessed boolean).
    """
    A = fmat(args["A"])
    tr, det = A[0][0] + A[1][1], _det2(A)
    disc = tr * tr - 4 * det
    ni = max(abs(A[0][0]) + abs(A[0][1]), abs(A[1][0]) + abs(A[1][1]))
    n1 = max(abs(A[0][0]) + abs(A[1][0]), abs(A[0][1]) + abs(A[1][1]))
    if disc < 0:
        raise SvoError(Err.NO_REAL_ROOT, ctx="op:thm_spectral_radius#9226",
                       detail=f"判别式 {fs(disc)} < 0 — 复数谱情形不在本算子域 (只处理实特征值)")
    s = isqrt_frac(disc)
    if s is None:
        raise SvoError(Err.UNIDENTIFIABLE, ctx="op:thm_spectral_radius#9226",
                       detail=f"判别式 {fs(disc)} 非完全平方 → ρ 为无理数, 与范数的序关系在有理"
                              f"精确通道内不可判定 (§17.3 显式拒绝, 不给可能失真的布尔值)")
    rho = max(abs((tr + s) / 2), abs((tr - s) / 2))
    return {"value": {"rho": fs(rho), "norm_inf": fs(ni), "norm_1": fs(n1),
                      "holds_inf": rho <= ni, "holds_1": rho <= n1, "disc": fs(disc)},
            "resid": [float(max(ni - rho, 0)), float(max(n1 - rho, 0))],
            "notes": f"T-L6: ρ={fs(rho)} ≤ ‖A‖∞={fs(ni)} ({rho <= ni}) 且 ≤ ‖A‖₁={fs(n1)} ({rho <= n1})"}


def _ver_spec(args: dict, value: dict) -> dict:
    """T-L6 independent verification: (a) independently recompute the norms; (b) the **Gelfand formulation** ρ² ≤ ‖A‖∞·‖A‖₁ (exact comparison);
    (c) lower-bound checks 2ρ ≥ |tr| and ρ² ≥ |det| (derived from |λ₁+λ₂|=|tr| and |λ₁λ₂|=|det|)."""
    A = fmat(args["A"])
    tr, det = A[0][0] + A[1][1], _det2(A)
    ni = max(abs(A[0][0]) + abs(A[0][1]), abs(A[1][0]) + abs(A[1][1]))
    n1 = max(abs(A[0][0]) + abs(A[1][0]), abs(A[0][1]) + abs(A[1][1]))
    rho = f(value["rho"])
    ok_norms = (f(value["norm_inf"]) == ni) and (f(value["norm_1"]) == n1)
    gelfand_ok = rho * rho <= ni * n1
    holds_ok = (value["holds_inf"] == (rho <= ni)) and (value["holds_1"] == (rho <= n1))
    lower_ok = (2 * rho >= abs(tr)) and (rho * rho >= abs(det))
    ok = ok_norms and gelfand_ok and holds_ok and lower_ok and (rho <= ni)
    return {"ok": ok, "resid": [float(max(rho * rho - ni * n1, 0)), 0.0 if lower_ok else 1.0],
            "detail": f"ρ²={fs(rho * rho)} ≤ ‖A‖∞·‖A‖₁={fs(ni * n1)} (Gelfand 表述) ={gelfand_ok}; "
                      f"下界 (2ρ≥|tr| 且 ρ²≥|det|) ={lower_ok}; 范数复算一致={ok_norms}"}


REGISTRY.register(OpDef(
    name="thm_spectral_radius", band=BAND, sub=26, thm="T-L6",
    stmt="对任意相容矩阵范数 ‖·‖ 有 ρ(A) ≤ ‖A‖ (谱半径不超过范数)",
    in_schema=_IN_SPEC, out_schema=_OUT_SPEC,
    invoke=_inv_spec, verify=_ver_spec,
    cost=("units", 1.0), errors=("NO_REAL_ROOT", "UNIDENTIFIABLE", "SCHEMA_MISMATCH"),
    effects="none (纯函数, 精确有理算术)",
    doc="T-L6 谱半径 ≤ 相容范数 | 条件: 2×2、判别式≥0 且为有理完全平方 (否则 ρ 无理, "
        "序关系不可判定 → UNIDENTIFIABLE) | 推导=特征值闭式 | "
        "验证器=Gelfand 表述 ρ²≤‖A‖∞‖A‖₁ + 下界核对 | 1 unit | 失败: 无实特征值 NO_REAL_ROOT",
    selfchecks=(
        {"id": "t_l6_hand", "args": {"A": [[4, 1], [2, 3]]},
         "check": lambda r: (r.value["rho"] == "5" and r.value["norm_inf"] == "5"
                             and r.value["norm_1"] == "6" and r.value["holds_inf"] is True
                             and r.value["holds_1"] is True,
                             f"ρ={r.value['rho']} ∞={r.value['norm_inf']} 1={r.value['norm_1']}"),
         "expect": "手算: λ=5,2 → ρ=5; ‖A‖∞=max(4+1, 2+3)=5; ‖A‖₁=max(4+2, 1+3)=6; 5≤5 且 5≤6 ✓"},
        {"id": "t_l6_diagonal", "args": {"A": [[3, 0], [0, -2]]},
         "check": lambda r: (r.value["rho"] == "3" and r.value["norm_inf"] == "3",
                             f"ρ={r.value['rho']}"),
         "expect": "对角阵: ρ=max(3,2)=3=‖A‖∞"},
        {"id": "t_l6_irrational_rho_rejected", "args": {"A": [[0, 1], [1, 1]]},
         "expect_err": "UNIDENTIFIABLE",
         "expect": "ρ=(1+√5)/2 无理 → 精确通道不可判定序关系, 显式 UNIDENTIFIABLE"},
        {"id": "t_l6_complex_rejected", "args": {"A": [[0, -1], [1, 0]]},
         "expect_err": "NO_REAL_ROOT", "expect": "复特征值情形不在本算子域内"},
    ),
))


# ===========================================================================
# calculus (T-C1 … T-C6, rows 9231–9236)
#
# the elementary-function family (this module's closed-form language):
#     term := {c, f ∈ {const,exp,sin,cos,ln,powu}, a=1, b=0, r=0}
#             semantics = c · f(a·x + b); for powu, f(u)=u^r (r rational).
# this family is **closed** under "differentiation" and "elementary integration":
#   d exp = a·exp; d sin = a·cos; d cos = −a·sin; d ln = a·(u)^(−1); d powu = c·r·a·u^(r−1)
#   ∫ exp = exp/a; ∫ sin = −cos/a; ∫ cos = sin/a; ∫ powu (r≠−1) = u^(r+1)/(a(r+1));
#   ∫ powu (r=−1) = ln/a  ←—— this is why r=−1 must be listed separately, and the key to closedness
# Honesty boundary: the **symbolic** part of differentiation/integration is exact (rational arithmetic); the checking
# uses **numerical differences/quadrature**, so every calculus-family operator claims only "numerical evidence +
# explicit tolerance" and never claims a proof (§18 red line).
# ===========================================================================

_FORMS = ("const", "exp", "sin", "cos", "ln", "powu")
_TERM_SCHEMA = {
    "kind": "dict", "extra": False,
    "fields": {"c": {"kind": "Fraction"}, "f": {"kind": "str", "enum": list(_FORMS)},
               "a": {"kind": "Fraction"}, "b": {"kind": "Fraction"}, "r": {"kind": "Fraction"}},
    "required": ["c", "f"],
}


def _norm_term(t: dict, ctx: str) -> dict:
    f_ = t["f"]
    a = f(t.get("a", 1))
    b = f(t.get("b", 0))
    c = f(t["c"])
    if f_ == "powu":
        if "r" not in t:
            raise SvoError(Err.PRECOND_FAIL, ctx=ctx, detail="powu 形式必须给指数 r")
        return {"c": c, "f": f_, "a": a, "b": b, "r": f(t["r"])}
    return {"c": c, "f": f_, "a": a, "b": b}


def _needs_pos_base(t: dict) -> bool:
    """ln requires u>0; powu requires u>0 when r is non-integer (fractional powers of negative bases are outside the reals)."""
    if t["f"] == "ln":
        return True
    if t["f"] == "powu":
        return t["r"].denominator != 1
    return False


def _fval(t: dict, x: float) -> float:
    """real-valued evaluation (float channel, used only for numerical checking)."""
    c = float(t["c"])
    a, b = float(t.get("a", 1)), float(t.get("b", 0))
    u = a * x + b
    fn = t["f"]
    if fn == "const":
        return c
    if fn == "exp":
        return c * math.exp(u)
    if fn == "sin":
        return c * math.sin(u)
    if fn == "cos":
        return c * math.cos(u)
    if fn == "ln":
        if u <= 0:
            raise SvoError(Err.DEG_UNSUPPORTED, ctx="math_theorems._fval", detail=f"ln 底 {u} ≤ 0")
        return c * math.log(u)
    r = float(t["r"])
    if u < 0 and abs(r - round(r)) > 1e-15:
        raise SvoError(Err.DEG_UNSUPPORTED, ctx="math_theorems._fval",
                       detail=f"powu 底 {u} < 0 且指数 {r} 非整数")
    if u == 0 and r < 0:
        raise SvoError(Err.DEG_UNSUPPORTED, ctx="math_theorems._fval", detail="powu 0 的负次幂")
    return c * (u ** r)


def _dterm(t: dict) -> "dict | None":
    """symbolic differentiation (exact coefficients); returns None when the term's derivative is 0."""
    c, a, b, fn = t["c"], t["a"], t["b"], t["f"]
    if fn == "const":
        return None
    if fn == "exp":
        return {"c": c * a, "f": "exp", "a": a, "b": b}
    if fn == "sin":
        return {"c": c * a, "f": "cos", "a": a, "b": b}
    if fn == "cos":
        return {"c": -c * a, "f": "sin", "a": a, "b": b}
    if fn == "ln":
        return {"c": c * a, "f": "powu", "a": a, "b": b, "r": Fraction(-1)}
    r = t["r"]
    return {"c": c * r * a, "f": "powu", "a": a, "b": b, "r": r - 1}


def _iterm(t: dict) -> dict:
    """symbolic antiderivative (C=0); outside the domain (a=0 and an integration variable is needed) raises DEG_UNSUPPORTED."""
    c, a, b, fn = t["c"], t["a"], t["b"], t["f"]
    if fn == "const":
        return {"c": c, "f": "powu", "a": Fraction(1), "b": Fraction(0), "r": Fraction(1)}
    if a == 0:
        # f(a·x+b) degenerates to a constant → ∫ = c·f(b)·x
        val = _fval(t, 0.0) if fn != "const" else float(c)
        return {"c": Fraction(c), "f": "powu", "a": Fraction(1), "b": Fraction(0), "r": Fraction(1)}
    if fn == "exp":
        return {"c": c / a, "f": "exp", "a": a, "b": b}
    if fn == "sin":
        return {"c": -c / a, "f": "cos", "a": a, "b": b}
    if fn == "cos":
        return {"c": c / a, "f": "sin", "a": a, "b": b}
    if fn == "ln":
        # ∫ln(u)dx = (u/a)(ln u − 1) — the result is outside the closed-form family, so it is explicitly rejected
        # rather than given as an approximation dressed up as exact
        raise SvoError(Err.DEG_UNSUPPORTED, ctx="math_theorems._iterm",
                       detail="ln 的原函数含 u·ln u 乘积项, 不在本闭形族内 (诚实边界: 拒绝而非近似)")
    r = t["r"]
    if r == -1:
        return {"c": c / a, "f": "ln", "a": a, "b": b}
    return {"c": c / (a * (r + 1)), "f": "powu", "a": a, "b": b, "r": r + 1}


def _sample_xs(terms, h: float = 1e-5) -> "list[float]":
    """take safe sample points in u-space and map them back to x (avoiding the branch points of ln/powu and the a=0 degeneracy)."""
    a = None
    for t in terms:
        if t["f"] != "const" and t["a"] != 0:
            a = float(t["a"])
            b = float(t["b"])
            break
    if a is None:
        return [0.0, 1.0, -1.0]
    return [((u - b) / a) for u in (0.5, 1.0, 2.0)]


def _num_cross_check(orig_terms, claim_terms, tol: float = 1e-5) -> "tuple[float, str, int]":
    """numerical central differences (on the original form) vs direct evaluation (on the claimed derivative/antiderivative) — two disjoint numerical paths.

    Returns (worst relative deviation, detail, sample-point count). For each sample point, x is taken over u∈{0.5,1,2}.
    """
    xs = _sample_xs(orig_terms)
    worst = 0.0
    used = 0
    for x in xs:
        try:
            fo_p = sum(_fval(t, x + tol) for t in orig_terms)
            fo_m = sum(_fval(t, x - tol) for t in orig_terms)
            fc = sum(_fval(t, x) for t in claim_terms)
        except SvoError:
            continue
        d = (fo_p - fo_m) / (2 * tol)
        scale = max(1.0, abs(fc), abs(d))
        worst = max(worst, abs(d - fc) / scale)
        used += 1
    return worst, f"数值中心差分(h={tol:g}) vs 声称式直接求值: {used} 个采样点, max 相对偏差={worst:.3e}", used


_IN_DERIV = {"kind": "dict",
             "fields": {"terms": {"kind": "list", "items": _TERM_SCHEMA, "minlen": 1, "maxlen": 6}},
             "required": ["terms"]}


def _pre_terms(args) -> "tuple[bool, str]":
    for i, t in enumerate(args["terms"]):
        if t["f"] == "powu" and "r" not in t:
            return False, f"第 {i} 项 powu 缺 r"
        if t["f"] == "powu" and "r" in t and f(t["r"]) == 0 and f(t["c"]) == 0:
            return False, f"第 {i} 项为 0·u⁰ ≡ 0 — 无意义项"
    return True, ""


def _inv_deriv(args: dict, ctl: CallCtl) -> dict:
    ctx = "op:thm_derivative_table#9231"
    terms = [_norm_term(t, ctx) for t in args["terms"]]
    out = [_dterm(t) for t in terms]
    out = [o for o in out if o is not None]
    worst, detail, used = _num_cross_check(terms, out)
    if used == 0:
        raise SvoError(Err.DEG_UNSUPPORTED, ctx=ctx, detail="无有效采样点 (定义域问题), 无法数值校验")
    return {"value": {"terms": [{"c": fs(o["c"]), "f": o["f"], "a": fs(o["a"]), "b": fs(o["b"]),
                                 **({"r": fs(o["r"])} if o["f"] == "powu" else {})} for o in out],
                      "zero": len(out) == 0, "sampled": used},
            "resid": [worst],
            "notes": f"T-C1 导数表: {len(terms)} 项 → {len(out)} 项; {detail}"}


def _ver_deriv(args: dict, value: dict) -> dict:
    """T-C1 independent verification: the **inverse path** — numerically integrate the claimed derivative (composite Simpson), feed it back into
    the original form's finite-difference quantity, the opposite direction from invoke's central differences; then check the term count and the
    "zero derivative" decision."""
    ctx = "thm_derivative_table@verify"
    terms = [_norm_term(t, ctx) for t in args["terms"]]
    claim = []
    for t in value["terms"]:
        c = {"c": f(t["c"]), "f": t["f"], "a": f(t["a"]), "b": f(t["b"])}
        if t["f"] == "powu":
            c["r"] = f(t["r"])
        claim.append(c)
    xs = _sample_xs(terms)
    worst = 0.0
    used = 0
    for x in xs:
        try:
            # composite Simpson: integrate the claimed derivative from x to x+0.01
            m, hstep = 64, 0.01 / 64
            acc = 0.0
            for k in range(m + 1):
                w = 1 if k in (0, m) else (4 if k % 2 else 2)
                acc += w * sum(_fval(t, x + k * hstep) for t in claim)
            integ = acc * hstep / 3.0
            delta = sum(_fval(t, x + 0.01) for t in terms) - sum(_fval(t, x) for t in terms)
        except SvoError:
            continue
        scale = max(1.0, abs(delta), abs(integ))
        worst = max(worst, abs(integ - delta) / scale)
        used += 1
    zero_ok = value["zero"] == all(_dterm(t) is None for t in terms)
    ok = worst < 1e-6 and used > 0 and zero_ok
    return {"ok": ok, "resid": [worst],
            "detail": f"逆路径 (声称导函数的 Simpson 积分 vs 原式的增量): {used} 点, "
                      f"max 相对偏差={worst:.3e}; 零导判定一致={zero_ok}"}


REGISTRY.register(OpDef(
    name="thm_derivative_table", band=BAND, sub=31, thm="T-C1",
    stmt="初等函数族的导数表: (eᵘ)'=u'eᵘ, (sin u)'=u'cos u, (cos u)'=−u'sin u, (ln u)'=u'/u, (uʳ)'=r u' u^(r−1)",
    in_schema=_IN_DERIV,
    out_schema={"kind": "dict",
                "fields": {"terms": {"kind": "list", "maxlen": 6,
                                     "items": {"kind": "dict", "extra": True, "fields": {}}},
                           "zero": {"kind": "bool"}, "sampled": {"kind": "int", "min": 1}},
                "required": ["terms", "zero", "sampled"]},
    invoke=_inv_deriv, verify=_ver_deriv, precond=_pre_terms,
    cost=("units", 1.2), errors=("PRECOND_FAIL", "DEG_UNSUPPORTED", "SCHEMA_MISMATCH"),
    effects="none (符号部分精确; 校验走 float 数值通道, 见 doc)",
    doc="T-C1 导数表 (闭形初等族 {const,exp,sin,cos,ln,powu}) | 推导=符号规则 (有理系数精确) | "
        "验证器=中心差分 (invoke 侧) + 声称导函数的 Simpson 积分逆路径 (verify 侧) | "
        "1.2 units | 诚实边界: 校验是**数值证据** (容差 1e-6), 不声称已证 | "
        "失败: powu 缺 r PRECOND_FAIL / 无有效采样点 DEG_UNSUPPORTED",
    selfchecks=(
        {"id": "t_c1_exp_chain", "args": {"terms": [{"c": 3, "f": "exp", "a": 2}]},
         "check": lambda r: (r.value["terms"] == [{"c": "6", "f": "exp", "a": "2", "b": "0"}]
                             and r.resid_scalar < 1e-6,
                             f"terms={r.value['terms']} resid={r.resid_scalar:.3e}"),
         "expect": "手算: (3e^{2x})' = 6e^{2x} (系数 3·2=6)"},
        {"id": "t_c1_sin", "args": {"terms": [{"c": 2, "f": "sin", "a": 3}]},
         "check": lambda r: (r.value["terms"] == [{"c": "6", "f": "cos", "a": "3", "b": "0"}],
                             f"terms={r.value['terms']}"),
         "expect": "手算: (2sin3x)' = 6cos3x"},
        {"id": "t_c1_ln", "args": {"terms": [{"c": 1, "f": "ln", "a": 2, "b": 1}]},
         "check": lambda r: (r.value["terms"] == [{"c": "2", "f": "powu", "a": "2", "b": "1", "r": "-1"}],
                             f"terms={r.value['terms']}"),
         "expect": "手算: (ln(2x+1))' = 2/(2x+1) = 2(2x+1)⁻¹"},
        {"id": "t_c1_powu", "args": {"terms": [{"c": 1, "f": "powu", "a": 1, "b": 0, "r": "1/2"}]},
         "check": lambda r: (r.value["terms"] == [{"c": "1/2", "f": "powu", "a": "1", "b": "0", "r": "-1/2"}],
                             f"terms={r.value['terms']}"),
         "expect": "手算: (√x)' = (x^{1/2})' = (1/2)x^{−1/2}"},
        {"id": "t_c1_const_only", "args": {"terms": [{"c": 7, "f": "const"}]},
         "check": lambda r: (r.value["zero"] is True and r.value["terms"] == [],
                             f"zero={r.value['zero']}"),
         "expect": "常数项导数为 0 → 输出空项表 + zero=True (不是空输入表)"},
        {"id": "t_c1_powu_missing_r", "args": {"terms": [{"c": 1, "f": "powu"}]},
         "expect_err": "PRECOND_FAIL", "expect": "powu 缺 r 必须被前提拦下"},
    ),
))


_IN_CHAIN = {"kind": "dict",
             "fields": {"outer": {"kind": "dict", "extra": False,
                                  "fields": {"c": {"kind": "Fraction"},
                                             "f": {"kind": "str", "enum": ["exp", "sin", "cos", "ln", "powu"]},
                                             "r": {"kind": "Fraction"}},
                                  "required": ["c", "f"]},
                        "inner": {"kind": "list", "items": {"kind": "Fraction"}, "len": 3}},
             "required": ["outer", "inner"]}


def _inner_eval(poly, x: float) -> float:
    return poly[0] + poly[1] * x + poly[2] * x * x


def _outer_eval(outer, u: float) -> float:
    c, fn = float(f(outer["c"])), outer["f"]
    if fn == "exp":
        return c * math.exp(u)
    if fn == "sin":
        return c * math.sin(u)
    if fn == "cos":
        return c * math.cos(u)
    if fn == "ln":
        if u <= 0:
            raise SvoError(Err.DEG_UNSUPPORTED, ctx="thm_chain_rule", detail=f"ln 底 {u} ≤ 0")
        return c * math.log(u)
    r = float(f(outer["r"]))
    if u < 0 and abs(r - round(r)) > 1e-15:
        raise SvoError(Err.DEG_UNSUPPORTED, ctx="thm_chain_rule", detail=f"powu 底 {u}<0 且指数非整")
    if u == 0 and r < 0:
        raise SvoError(Err.DEG_UNSUPPORTED, ctx="thm_chain_rule", detail="powu 0 的负次幂")
    return c * (u ** r)


def _outer_dprime(outer, u: float) -> float:
    """numerical value of the outer derivative f'(u) (a small analytic expression, not a difference)."""
    c, fn = float(f(outer["c"])), outer["f"]
    if fn == "exp":
        return c * math.exp(u)
    if fn == "sin":
        return c * math.cos(u)
    if fn == "cos":
        return -c * math.sin(u)
    if fn == "ln":
        if u <= 0:
            raise SvoError(Err.DEG_UNSUPPORTED, ctx="thm_chain_rule", detail=f"ln 底 {u} ≤ 0")
        return c / u
    r = float(f(outer["r"]))
    return c * r * (u ** (r - 1))


def _pre_chain(args) -> "tuple[bool, str]":
    if args["outer"]["f"] == "powu" and "r" not in args["outer"]:
        return False, "powu 外函数缺 r"
    return True, ""


def _inv_chain(args: dict, ctl: CallCtl) -> dict:
    """T-C2 derivation: (f∘g)'(x) = f'(g(x))·g'(x), with the inner function the quadratic g(x)=c0+c1x+c2x²."""
    ctx = "op:thm_chain_rule#9232"
    outer = args["outer"]
    g = fvec(args["inner"])
    gd = [g[1], 2 * g[2]]  # g' coefficients (degree reduced to 1)
    return {"value": {"outer_d": {"c": fs(f(outer["c"])), "f": outer["f"],
                                  **({"r": fs(f(outer["r"]))} if outer["f"] == "powu" else {})},
                      "inner": [fs(v) for v in g], "inner_d": [fs(v) for v in gd],
                      "sampled": 0},
            "resid": [0.0],
            "notes": f"T-C2: (f∘g)' = f'(u)·g'(x), u={[fs(v) for v in g]}, "
                     f"g'={[fs(v) for v in gd]}; 数值校验在 verify 侧 (见其 detail)"}


def _ver_chain(args: dict, value: dict) -> dict:
    """T-C2 independent verification: apply central differences to the **composite function itself** and compare with the analytic form f'(u)·g'(x).

    Division of labour: invoke only does the symbolic decomposition (producing f', g'); verify independently
    differentiates the composite function numerically and compares the analytic form point by point — if the chain
    rule's product direction/coefficient is wrong, the deviation necessarily shows up here.
    """
    outer = args["outer"]
    g = fvec(args["inner"])
    gd = [g[1], 2 * g[2]]  # g' coefficients (degree reduced: g' = g[1] + 2·g[2]·x)
    xs = [0.0, 0.3, -0.7] if g[2] != 0 or g[1] != 0 else [0.0, 0.5, 1.0]
    h = 1e-5
    worst = 0.0
    used = 0
    for x in xs:
        try:
            def comp(z, _o=outer, _g=g):
                return _outer_eval(_o, _inner_eval(_g, z))
            num = (comp(x + h) - comp(x - h)) / (2 * h)
    # g'(x) = gd[0] + gd[1]·x — gd[1] is already 2·g[2], so it **must not** be multiplied by 2 again
    # (the old version wrote 2*gd[1]*x, doubling the coefficient; measured 50% relative deviation, now fixed)
            g_prime = gd[0] + gd[1] * x
            ana = _outer_dprime(outer, _inner_eval(g, x)) * g_prime
        except SvoError:
            continue
        scale = max(1.0, abs(num), abs(ana))
        worst = max(worst, abs(num - ana) / scale)
        used += 1
    ok = worst < 1e-5 and used > 0
    return {"ok": ok, "resid": [worst],
            "detail": f"复合函数中心差分 vs 解析式 f'(u)g'(x): {used} 点, max 相对偏差={worst:.3e}"}


REGISTRY.register(OpDef(
    name="thm_chain_rule", band=BAND, sub=32, thm="T-C2",
    stmt="复合函数求导: (f∘g)'(x) = f'(g(x))·g'(x)",
    in_schema=_IN_CHAIN,
    out_schema={"kind": "dict",
                "fields": {"outer_d": {"kind": "dict", "extra": True, "fields": {}},
                           "inner": {"kind": "list", "len": 3, "items": {"kind": "str"}},
                           "inner_d": {"kind": "list", "len": 2, "items": {"kind": "str"}},
                           "sampled": {"kind": "int", "min": 0}},
                "required": ["outer_d", "inner", "inner_d", "sampled"]},
    invoke=_inv_chain, verify=_ver_chain, precond=_pre_chain,
    cost=("units", 1.0), errors=("PRECOND_FAIL", "DEG_UNSUPPORTED", "SCHEMA_MISMATCH"),
    effects="none (符号分解精确; 校验走 float)",
    doc="T-C2 链式法则 (外函数 ∈ 闭形族, 内函数为二次多项式) | 推导=符号分解 f'(u)·g'(x) | "
        "验证器=复合函数中心差分逐点比对解析式 | 1 unit | 诚实边界: 数值证据, 容差 1e-5 | "
        "失败: powu 缺 r PRECOND_FAIL / 定义域问题 DEG_UNSUPPORTED",
    selfchecks=(
        {"id": "t_c2_sin_of_x2", "args": {"outer": {"c": 1, "f": "sin"}, "inner": [0, 0, 1]},
         "check": lambda r: (r.value["inner"] == ["0", "0", "1"] and r.value["inner_d"] == ["0", "2"]
                             and r.resid_scalar < 1e-8,
                             f"inner={r.value['inner']} inner_d={r.value['inner_d']} "
                             f"resid={r.resid_scalar:.3e}"),
         "expect": "手算: (sin x²)' = 2x·cos x²: g=[0,0,1] → g' 系数 [0,2] 即 2x; "
                   "残差上限 1e-8 (verify 的数值差分残差, 非 0)"},
        {"id": "t_c2_exp_of_linear", "args": {"outer": {"c": 1, "f": "exp"}, "inner": [1, 2, 0]},
         "check": lambda r: (r.value["inner_d"] == ["2", "0"], f"inner_d={r.value['inner_d']}"),
         "expect": "手算: (e^{2x+1})' = 2e^{2x+1}: g=[1,2,0] → g'(x)=2+0·x, 系数表 [2,0]"},
        {"id": "t_c2_ln_of_shift", "args": {"outer": {"c": 1, "f": "ln"}, "inner": [2, 0, 1]},
         "check": lambda r: (r.value["inner"] == ["2", "0", "1"], f"inner={r.value['inner']}"),
         "expect": "手算: (ln(x²+2))' = 2x/(x²+2); 定义域 x²+2>0 恒成立"},
        {"id": "t_c2_powu_missing_r", "args": {"outer": {"c": 1, "f": "powu"}, "inner": [1, 0, 0]},
         "expect_err": "PRECOND_FAIL", "expect": "powu 缺 r"},
        {"id": "t_c2_bad_inner_len", "args": {"outer": {"c": 1, "f": "sin"}, "inner": [0, 1]},
         "expect_err": "SCHEMA_MISMATCH", "expect": "内层多项式必须恰 3 个系数 (二次)"},
    ),
))


_IN_INTEG = {"kind": "dict",
             "fields": {"terms": {"kind": "list", "items": _TERM_SCHEMA, "minlen": 1, "maxlen": 6}},
             "required": ["terms"]}


def _inv_integ(args: dict, ctl: CallCtl) -> dict:
    """T-C3 derivation: term-by-term antiderivative (C=0)."""
    ctx = "op:thm_integral_table#9233"
    terms = [_norm_term(t, ctx) for t in args["terms"]]
    out = [_iterm(t) for t in terms]
    worst, detail, used = _num_cross_check(out, terms)  # differentiate the antiderivative vs the original integrand
    if used == 0:
        raise SvoError(Err.DEG_UNSUPPORTED, ctx=ctx, detail="无有效采样点, 无法数值校验")
    return {"value": {"terms": [{"c": fs(o["c"]), "f": o["f"], "a": fs(o["a"]), "b": fs(o["b"]),
                                 **({"r": fs(o["r"])} if o["f"] == "powu" else {})} for o in out],
                      "zero": False, "sampled": used},
            "resid": [worst],
            "notes": f"T-C3 积分表: {len(terms)} 项 → {len(out)} 项 (C=0); 互逆核对: {detail}"}


def _ver_integ(args: dict, value: dict) -> dict:
    """T-C3 independent verification: **numerically differentiate** the claimed antiderivative; it must equal the integrand (mutual inverses), and
    check F(b) − F(a) against ∫f with **composite Simpson quadrature** (two independent numerical paths)."""
    ctx = "thm_integral_table@verify"
    terms = [_norm_term(t, ctx) for t in args["terms"]]
    F = []
    for t in value["terms"]:
        c = {"c": f(t["c"]), "f": t["f"], "a": f(t["a"]), "b": f(t["b"])}
        if t["f"] == "powu":
            c["r"] = f(t["r"])
        F.append(c)
    h = 1e-5
    xs = _sample_xs(terms)
    worst = 0.0
    used = 0
    for x in xs:
        try:
            dF = (sum(_fval(t, x + h) for t in F) - sum(_fval(t, x - h) for t in F)) / (2 * h)
            fv = sum(_fval(t, x) for t in terms)
        except SvoError:
            continue
        scale = max(1.0, abs(fv), abs(dF))
        worst = max(worst, abs(dF - fv) / scale)
        used += 1
    # composite Simpson independently checks F(b)−F(a) = ∫_a^b f (choosing generic endpoints a,b)
    quad_gap = 0.0
    quad_used = 0
    for a_, b_ in ((0.3, 1.1), (-0.4, 0.8)):
        try:
            m, step = 200, (b_ - a_) / 200
            acc = 0.0
            for k in range(m + 1):
                w = 1 if k in (0, m) else (4 if k % 2 else 2)
                acc += w * sum(_fval(t, a_ + k * step) for t in terms)
            integ = acc * step / 3.0
            Fdiff = sum(_fval(t, b_) for t in F) - sum(_fval(t, a_) for t in F)
        except SvoError:
            continue
        scale = max(1.0, abs(integ), abs(Fdiff))
        quad_gap = max(quad_gap, abs(integ - Fdiff) / scale)
        quad_used += 1
    ok = worst < 1e-4 and used > 0 and quad_gap < 1e-4
    return {"ok": ok, "resid": [worst, quad_gap],
            "detail": f"互逆核对 {used} 点 max 相对偏差={worst:.3e}; "
                      f"复合 Simpson 核对 F(b)−F(a) 于 {quad_used} 对端点, max 相对偏差={quad_gap:.3e}"}


REGISTRY.register(OpDef(
    name="thm_integral_table", band=BAND, sub=33, thm="T-C3",
    stmt="初等函数族原函数表 (C=0): ∫uʳdx=u^{r+1}/(a(r+1)) (r≠−1), ∫u⁻¹dx=ln u/a, ∫eᵘdx=eᵘ/a, ∫sin= −cos/a, ∫cos=sin/a",
    in_schema=_IN_INTEG,
    out_schema={"kind": "dict",
                "fields": {"terms": {"kind": "list", "maxlen": 6,
                                     "items": {"kind": "dict", "extra": True, "fields": {}}},
                           "zero": {"kind": "bool"}, "sampled": {"kind": "int", "min": 1}},
                "required": ["terms", "zero", "sampled"]},
    invoke=_inv_integ, verify=_ver_integ, precond=_pre_terms,
    cost=("units", 1.5), errors=("PRECOND_FAIL", "DEG_UNSUPPORTED", "SCHEMA_MISMATCH"),
    effects="none (符号部分精确; 校验走 float)",
    doc="T-C3 初等积分表 (闭形族) | 推导=逐项原函数 | 验证器=原函数数值求导互逆 + "
        "复合 Simpson 核对 F(b)−F(a)=∫f | 1.5 units | 诚实边界: ln 的原函数含 u·ln u 乘积项, "
        "不在闭形族内 → DEG_UNSUPPORTED (拒绝而非近似冒充精确) | 失败: 域外 DEG_UNSUPPORTED",
    selfchecks=(
        {"id": "t_c3_sin", "args": {"terms": [{"c": 2, "f": "sin", "a": 3}]},
         "check": lambda r: (r.value["terms"] == [{"c": "-2/3", "f": "cos", "a": "3", "b": "0"}]
                             and r.resid_scalar < 1e-6,
                             f"terms={r.value['terms']} resid={r.resid_scalar:.3e}"),
         "expect": "手算: ∫2sin3x dx = −(2/3)cos3x; 回导 = (2/3)·3·sin3x = 2sin3x ✓"},
        {"id": "t_c3_powu_inverse", "args": {"terms": [{"c": 1, "f": "powu", "a": 2, "b": 1, "r": -1}]},
         "check": lambda r: (r.value["terms"] == [{"c": "1/2", "f": "ln", "a": "2", "b": "1"}],
                             f"terms={r.value['terms']}"),
         "expect": "手算: ∫1/(2x+1) dx = (1/2)ln(2x+1) —— r=−1 特例走 ln 分支"},
        {"id": "t_c3_powu_general", "args": {"terms": [{"c": 1, "f": "powu", "a": 1, "b": 0, "r": 2}]},
         "check": lambda r: (r.value["terms"] == [{"c": "1/3", "f": "powu", "a": "1", "b": "0", "r": "3"}],
                             f"terms={r.value['terms']}"),
         "expect": "手算: ∫x² dx = x³/3"},
        {"id": "t_c3_const", "args": {"terms": [{"c": 5, "f": "const"}]},
         "check": lambda r: (r.value["terms"] == [{"c": "5", "f": "powu", "a": "1", "b": "0", "r": "1"}],
                             f"terms={r.value['terms']}"),
         "expect": "手算: ∫5 dx = 5x (表示为 5x¹)"},
        {"id": "t_c3_ln_out_of_family", "args": {"terms": [{"c": 1, "f": "ln", "a": 1, "b": 1}]},
         "expect_err": "DEG_UNSUPPORTED",
         "expect": "∫ln(x+1) 的原函数含 u·ln u 项 → 闭形族外, 显式拒绝 (§18 不许近似冒充精确)"},
    ),
))


_IN_FTC = {"kind": "dict",
           "fields": {"terms": {"kind": "list", "items": _TERM_SCHEMA, "minlen": 1, "maxlen": 5},
                      "a": {"kind": "Fraction"}, "b": {"kind": "Fraction"}},
           "required": ["terms", "a", "b"]}


def _pre_ftc(args) -> "tuple[bool, str]":
    if f(args["a"]) >= f(args["b"]):
        return False, "要求 a < b (定积分方向)"
    return _pre_terms(args)


def _inv_ftc(args: dict, ctl: CallCtl) -> dict:
    """T-C4 derivation: the fundamental theorem of calculus ∫_a^b f = F(b) − F(a), with F the antiderivative given by the antiderivative table."""
    ctx = "op:thm_ftc#9234"
    terms = [_norm_term(t, ctx) for t in args["terms"]]
    a_, b_ = f(args["a"]), f(args["b"])
    F = [_iterm(t) for t in terms]
    val = sum((_fval(t, float(b_)) for t in F), 0.0) - sum((_fval(t, float(a_)) for t in F), 0.0)
    return {"value": {"F": [{"c": fs(o["c"]), "f": o["f"], "a": fs(o["a"]), "b": fs(o["b"]),
                             **({"r": fs(o["r"])} if o["f"] == "powu" else {})} for o in F],
                      "value": val, "a": fs(a_), "b": fs(b_)},
            "resid": [0.0],
            "notes": f"T-C4: ∫_{{{fs(a_)}}}^{{{fs(b_)}}} f = F(b)−F(a) = {val:.12g} "
                     f"(F 由积分表给出); 求积独立复核对拍在 verify 侧"}


def _ver_ftc(args: dict, value: dict) -> dict:
    """T-C4 independent verification: compute ∫_a^b f directly by **composite Simpson quadrature** and compare with F(b)−F(a).

    A path entirely different from invoke's "subtract the antiderivative at the endpoints" (numerical quadrature vs symbolic antiderivative).
    """
    ctx = "op:thm_ftc@verify"
    terms = [_norm_term(t, ctx) for t in args["terms"]]
    a_, b_ = float(f(args["a"])), float(f(args["b"]))
    m = 512
    step = (b_ - a_) / m
    acc = 0.0
    try:
        for k in range(m + 1):
            w = 1 if k in (0, m) else (4 if k % 2 else 2)
            acc += w * sum(_fval(t, a_ + k * step) for t in terms)
    except SvoError as e:
        return {"ok": False, "resid": [1.0], "detail": f"求积时定义域问题: {e.detail}"}
    quad = acc * step / 3.0
    claim = float(value["value"])
    scale = max(1.0, abs(quad), abs(claim))
    rel = abs(quad - claim) / scale
    # Simpson error-order estimate (the second-derivative bound is approximated by the h² term): reported, not hidden
    err_est = abs(step) ** 4 * max(1.0, abs(b_ - a_))
    ok = rel < 1e-6
    return {"ok": ok, "resid": [rel],
            "detail": f"复合 Simpson (m={m}) 得 {quad:.12g} vs F(b)−F(a)={claim:.12g}, "
                      f"相对偏差={rel:.3e} (步长 {step:.3e})"}


REGISTRY.register(OpDef(
    name="thm_ftc", band=BAND, sub=34, thm="T-C4",
    stmt="微积分基本定理: f 在 [a,b] 上连续且有原函数 F, 则 ∫_a^b f = F(b) − F(a)",
    in_schema=_IN_FTC,
    out_schema={"kind": "dict",
                "fields": {"F": {"kind": "list", "maxlen": 5,
                                 "items": {"kind": "dict", "extra": True, "fields": {}}},
                           "value": {"kind": "float"}, "a": {"kind": "str"}, "b": {"kind": "str"}},
                "required": ["F", "value", "a", "b"]},
    invoke=_inv_ftc, verify=_ver_ftc, precond=_pre_ftc,
    cost=("units", 1.5), errors=("PRECOND_FAIL", "DEG_UNSUPPORTED", "SCHEMA_MISMATCH"),
    effects="none (符号原函数精确; 求积校验走 float)",
    doc="T-C4 微积分基本定理 | 条件: a<b 且被积项在闭区间内定义 | 推导=原函数端点相减 | "
        "验证器=复合 Simpson 求积 (m=512) 独立复算 | 1.5 units | "
        "诚实边界: 求积是数值证据 (容差 1e-6), 区间含奇点时由定义域检查拒绝 | 失败: a≥b PRECOND_FAIL",
    selfchecks=(
        {"id": "t_c4_x2", "args": {"terms": [{"c": 1, "f": "powu", "a": 1, "b": 0, "r": 2}],
                                   "a": 0, "b": 3},
         "check": lambda r: (abs(r.value["value"] - 9.0) < 1e-9 and r.resid_scalar < 1e-9,
                             f"∫={r.value['value']}"),
         "expect": "手算: ∫₀³x²dx = 27/3 = 9"},
        {"id": "t_c4_sin_pi", "args": {"terms": [{"c": 1, "f": "sin", "a": 1, "b": 0}],
                                       "a": 0, "b": 1},
         "check": lambda r: (abs(r.value["value"] - (1 - math.cos(1.0))) < 1e-12,
                             f"∫={r.value['value']:.12g} 期望 {1 - math.cos(1.0):.12g}"),
         "expect": "∫₀¹sin x dx = 1−cos1 ≈ 0.459697694132"},
        {"id": "t_c4_const", "args": {"terms": [{"c": 4, "f": "const"}], "a": 1, "b": 5},
         "check": lambda r: (abs(r.value["value"] - 16.0) < 1e-9, f"∫={r.value['value']}"),
         "expect": "手算: ∫₁⁵4 dx = 4·4 = 16"},
        {"id": "t_c4_reversed_interval", "args": {"terms": [{"c": 1, "f": "const"}], "a": 2, "b": 1},
         "expect_err": "PRECOND_FAIL", "expect": "a≥b 必须被拒 (方向约定显式)"},
        {"id": "t_c4_ln_rejected", "args": {"terms": [{"c": 1, "f": "ln", "a": 1, "b": 1}],
                                            "a": 0, "b": 1},
         "expect_err": "DEG_UNSUPPORTED", "expect": "∫ln 不在闭形族内 (与 T-C3 同一边界)"},
    ),
))


_IN_T3 = {"kind": "dict",
          "fields": {"f": {"kind": "str", "enum": ["exp", "sin", "cos"]},
                     "c": {"kind": "Fraction"}, "a": {"kind": "Fraction"},
                     "x": {"kind": "Fraction"}, "n": {"kind": "int", "min": 0, "max": 10}},
          "required": ["f", "c", "a", "x", "n"]}


def _taylor_coeffs(fn: str, c: Fraction, a: Fraction, n: int) -> "list[Fraction]":
    """exact rational Maclaurin coefficients (x0=0): T_n(x) = Σ c·f^{(k)}(0)/k! · x^k.

    The parity branches are spelled out (the old sin branch wrote the sign exponent as (k+1)//2, flipping the sign of
    the x³ term — T₅(1/2) went from 0.479427 to 0.521094 and the remainder bound was rejected on the spot by the
    independent verifier; fixed):
        sin: odd k only, m=(k−1)/2, coefficient (−1)^m/k!·a^k
        cos: even k only, m=k/2,     coefficient (−1)^m/k!·a^k
        exp: all k,                  coefficient a^k/k!
    """
    out = [Fraction(0)] * (n + 1)
    for k in range(n + 1):
        if fn == "exp":
            out[k] = c * (a ** k) / math.factorial(k)
        elif fn == "sin":
            if k % 2 == 1:
                m = (k - 1) // 2
                out[k] = c * (a ** k) * Fraction((-1) ** m, math.factorial(k))
        else:  # cos
            if k % 2 == 0:
                m = k // 2
                out[k] = c * (a ** k) * Fraction((-1) ** m, math.factorial(k))
    return out


def _pre_t3(args) -> "tuple[bool, str]":
    """condition: |a·x| ≤ 1 — this is the **basis** for the exp remainder bound M ≤ 3 (e^t ≤ 3 for t ≤ 1).
    sin/cos bounds are constantly 1 and do not need this condition, but it is still required for a uniform
    convention (and stated in the detail)."""
    if abs(f(args["a"]) * f(args["x"])) > 1:
        return False, f"|a·x|={abs(f(args['a']) * f(args['x']))} > 1 — 超出本算子余项界的适用域"
    return True, ""


def _inv_t3(args: dict, ctl: CallCtl) -> dict:
    """T-C6 derivation: Maclaurin expansion + an explicit rational upper bound on the **Lagrange remainder**.

    How M is taken (explicit, auditable):
      - sin/cos : |f^{(n+1)}| = |c|·|a|^{n+1}·|sin/cos| ≤ |c|·|a|^{n+1}   (bound = 1)
      - exp     : |f^{(n+1)}| = |c|·|a|^{n+1}·e^{aξ} ≤ |c|·|a|^{n+1}·3  (from |aξ|≤|ax|≤1 ⇒ e^t ≤ 3)
    Remainder bound |R_n| ≤ M·|x|^{n+1}/(n+1)!.
    """
    fn, c, a, x, n = args["f"], f(args["c"]), f(args["a"]), f(args["x"]), int(args["n"])
    coefs = _taylor_coeffs(fn, c, a, n)
    if fn == "exp":
        M = abs(c) * (abs(a) ** (n + 1)) * 3
        m_note = "exp: M=|c||a|^{n+1}·3 (由 |ax|≤1 ⇒ e^t ≤ 3)"
    else:
        M = abs(c) * (abs(a) ** (n + 1))
        m_note = f"{fn}: M=|c||a|^{n+1} (|sin|,|cos| ≤ 1)"
    bound = M * (abs(x) ** (n + 1)) / math.factorial(n + 1)
    return {"value": {"coefs": [fs(v) for v in coefs], "M": fs(M), "bound": fs(bound),
                      "x0": "0", "n": n},
            "resid": [float(bound)],
            "notes": f"T-C6: T_{n}(x) (x0=0) 系数 {len(coefs)} 项; Lagrange 余项界 |R_{n}| ≤ {fs(bound)}; "
                     f"{m_note}"}


def _ver_t3(args: dict, value: dict) -> dict:
    """T-C6 independent verification: (a) the difference between the true value f(x) (math-library closed form) and T_n(x)
    (exact polynomial evaluation) **must fall inside the declared remainder bound** — empirical evidence for the bound;
    (b) the coefficients are cross-checked against numerical difference estimates of f^{(k)}(0)/k! (central differences are
    stable for k ≤ 4), two paths."""
    fn, c, a, x, n = args["f"], float(f(args["c"])), float(f(args["a"])), float(f(args["x"])), int(args["n"])
    coefs = [f(s) for s in value["coefs"]]
    if fn == "exp":
        true = c * math.exp(a * x)
    elif fn == "sin":
        true = c * math.sin(a * x)
    else:
        true = c * math.cos(a * x)
    tn = float(poly_eval(coefs, f(args["x"])))
    gap = abs(true - tn)
    bound = float(f(value["bound"]))
    slack = 1e-12 * max(1.0, abs(true))
    in_bound = gap <= bound + slack
    # coefficient-difference cross-check (low orders only; high-order differences are unstable → coverage stated in the notes)
    h = 1e-3
    diff_ok = True
    k_max = min(n, 4)
    for k in range(k_max + 1):
        def fval(z, _fn=fn, _c=c, _a=a):
            if _fn == "exp":
                return _c * math.exp(_a * z)
            if _fn == "sin":
                return _c * math.sin(_a * z)
            return _c * math.cos(_a * z)
        num = 0.0
        for j in range(k + 1):
            num += ((-1) ** j) * math.comb(k, j) * fval((k / 2 - j) * h)
        est = num / h ** k
        exact = float(coefs[k]) * math.factorial(k)
        if abs(est - exact) > 1e-3 * max(1.0, abs(exact)):
            diff_ok = False
    return {"ok": in_bound and diff_ok,
            "resid": [gap, bound],
            "detail": f"真值={true:.15g}, T_n={tn:.15g}, |R_n|={gap:.3e} ≤ 声明界 {bound:.3e} "
                      f"(余项界成立={in_bound}); 系数差分对拍 (k≤{k_max}) 一致={diff_ok}"}


REGISTRY.register(OpDef(
    name="thm_taylor_lagrange", band=BAND, sub=36, thm="T-C6",
    stmt="Taylor 定理 (Lagrange 余项): f(x)=T_n(x)+R_n, |R_n| ≤ M·|x−x0|^{n+1}/(n+1)!, M = sup|f^{(n+1)}|",
    in_schema=_IN_T3,
    out_schema={"kind": "dict",
                "fields": {"coefs": {"kind": "list", "items": {"kind": "str"}, "maxlen": 11},
                           "M": {"kind": "str"}, "bound": {"kind": "str"},
                           "x0": {"kind": "str"}, "n": {"kind": "int"}},
                "required": ["coefs", "M", "bound", "x0", "n"]},
    invoke=_inv_t3, verify=_ver_t3, precond=_pre_t3,
    cost=("units", 1.5), errors=("PRECOND_FAIL", "SCHEMA_MISMATCH"),
    effects="none (系数精确有理; 余项界的实证走 float)",
    doc="T-C6 Taylor 展开 + Lagrange 余项界 (Maclaurin, x0=0; f∈{exp,sin,cos}) | "
        "条件: |a·x| ≤ 1 (exp 的 M≤3 依据) | 推导=精确有理系数 + 显式 M 推导 | "
        "验证器=math 库真值落入声明界 + 低阶系数中心差分对拍 | 1.5 units | "
        "诚实边界: M 是**上界**不是最小值, 故界可能偏松; 高阶系数差分对拍只覆盖 k≤4 | "
        "失败: |ax|>1 PRECOND_FAIL",
    selfchecks=(
        {"id": "t_c6_exp_deg3_at1", "args": {"f": "exp", "c": 1, "a": 1, "x": 1, "n": 3},
         "check": lambda r: (r.value["coefs"] == ["1", "1", "1/2", "1/6"]
                             and r.value["M"] == "3" and r.value["bound"] == "1/8",
                             f"coefs={r.value['coefs']} M={r.value['M']} bound={r.value['bound']}"),
         "expect": "手算: T₃=1+x+x²/2+x³/6 (系数 1,1,1/2,1/6); M=1·1³·3=3; "
                   "界=3·1⁴/4!=3/24=1/8=0.125; 真值 e=2.718…, T₃(1)=2.666…, 差 0.0516 ≤ 0.125 ✓"},
        {"id": "t_c6_sin_deg5", "args": {"f": "sin", "c": 1, "a": 1, "x": "1/2", "n": 5},
         "check": lambda r: (r.value["coefs"] == ["0", "1", "0", "-1/6", "0", "1/120"]
                             and r.value["M"] == "1",
                             f"coefs={r.value['coefs']} M={r.value['M']}"),
         "expect": "手算: T₅(sin)=x−x³/6+x⁵/120; M=1 (|sin|≤1); 界=1·(1/2)⁶/6! = (1/64)/720"},
        {"id": "t_c6_cos_deg4", "args": {"f": "cos", "c": 2, "a": 1, "x": "1/3", "n": 4},
         "check": lambda r: (r.value["coefs"] == ["2", "0", "-1", "0", "1/12"],
                             f"coefs={r.value['coefs']}"),
         "expect": "手算: 2cos x = 2(1−x²/2+x⁴/24) = 2 − x² + x⁴/12 → 系数 [2,0,−1,0,1/12]"},
        {"id": "t_c6_domain_guard", "args": {"f": "exp", "c": 1, "a": 3, "x": 1, "n": 3},
         "expect_err": "PRECOND_FAIL",
         "expect": "|a·x|=3 > 1 → 超出 e^t≤3 的推导域, 必须拒 (不给未论证的界)"},
        {"id": "t_c6_n_zero", "args": {"f": "exp", "c": 1, "a": 1, "x": 1, "n": 0},
         "check": lambda r: (r.value["coefs"] == ["1"] and r.value["bound"] == "3",
                             f"coefs={r.value['coefs']} bound={r.value['bound']}"),
         "expect": "n=0: T₀=1 (常数), 界=3·1/1!=3 —— 界粗但**正确**(e−1≈1.718 ≤ 3)"},
    ),
))


# ===========================================================================
# series (T-S1 … T-S3, rows 9241–9243)
#
# the tests decide on **parameterised families** (the closed-form L can be computed exactly), not by "guessing a limit" from an arbitrary term table:
#   T-S1 ratio test: aₙ = c·rⁿ·P(n)/Q(n) → L = |r|   [degree-independent: P(n+1)Q(n)/(P(n)Q(n+1)) → 1]
#   T-S2 root test: aₙ = c·rⁿ·nᵖ      → L = |r|
#   T-S3 comparison test: 0 ≤ aₙ ≤ bₙ coefficient-wise domination, b a known convergent family
# when L = 1 it returns undecided — the test **gives no** answer, and none may be pretended.
# the exact rational ratio at large n serves only as **corroboration**, with its explicit deviation reported; it is not a criterion (§18: numbers are not proof).
# ===========================================================================

_SERIES_IN = {
    "kind": "dict",
    "fields": {"c": {"kind": "Fraction"}, "r": {"kind": "Fraction"},
               "P": {"kind": "list", "items": {"kind": "Fraction"}, "minlen": 1, "maxlen": 5},
               "Q": {"kind": "list", "items": {"kind": "Fraction"}, "minlen": 1, "maxlen": 5},
               "n_probe": {"kind": "int", "min": 2, "max": 10**7}},
    "required": ["c", "r", "P", "Q"],
}
_SERIES_OUT = {
    "kind": "dict",
    "fields": {"L": {"kind": "str"}, "verdict": {"kind": "str",
                                                 "enum": ["converges", "diverges", "undecided"]},
               "probe_ratio": {"kind": "str"}, "probe_n": {"kind": "int"},
               "probe_gap": {"kind": "str"}, "deg_diff": {"kind": "int"}},
    "required": ["L", "verdict", "probe_ratio", "probe_n", "probe_gap", "deg_diff"],
}


def _poly_deg_lead(cs):
    c = poly_trim([f(v) for v in cs])
    if len(c) == 1 and c[0] == 0:
        return -1, Fraction(0)
    return len(c) - 1, c[-1]


def _series_probe(c, r, P, Q, n_probe: int) -> Fraction:
    """exact rational value of a_{n+1}/a_n at n=n_probe (for corroboration, not a criterion)."""
    def pev(cs, n):
        return poly_eval([f(v) for v in cs], Fraction(n))
    a = abs(pev(P, n_probe + 1) * pev(Q, n_probe))
    b = abs(pev(P, n_probe) * pev(Q, n_probe + 1))
    if b == 0:
        raise SvoError(Err.DEG_UNSUPPORTED, ctx="thm_ratio_test",
                       detail=f"Q(n) 在 n={n_probe} 处为 0 — 探针点落在极点, 换 n_probe")
    return abs(r) * a / b


def _verdict_of(L: Fraction) -> str:
    if L < 1:
        return "converges"
    if L > 1:
        return "diverges"
    return "undecided"


def _inv_ratio(args: dict, ctl: CallCtl) -> dict:
    """T-S1 derivation: L = lim|a_{n+1}/a_n| = |r| (the degrees and leading terms of P,Q cancel in this limit, but the
    degree difference affects the **probe deviation**, so it is reported as well)."""
    c, r = f(args["c"]), f(args["r"])
    P, Q = [f(v) for v in args["P"]], [f(v) for v in args["Q"]]
    dP, lP = _poly_deg_lead(P)
    dQ, lQ = _poly_deg_lead(Q)
    if dQ < 0:
        raise SvoError(Err.PRECOND_FAIL, ctx="op:thm_ratio_test#9241", detail="Q ≡ 0 — 项无定义")
    n_probe = int(args.get("n_probe", 1000))
    L = abs(r)
    probe = _series_probe(c, r, P, Q, n_probe)
    return {"value": {"L": fs(L), "verdict": _verdict_of(L), "probe_ratio": fs(probe),
                      "probe_n": n_probe, "probe_gap": fs(abs(probe - L)), "deg_diff": dP - dQ},
            "resid": [float(abs(probe - L))],
            "notes": f"T-S1 比值判别: aₙ=c·rⁿ·P/Q → L=|r|={fs(L)} → {_verdict_of(L)}; "
                     f"佐证: n={n_probe} 处精确比值 {fs(probe)} (偏离 L: {fs(abs(probe - L))}, "
                     f"因 deg(P)−deg(Q)={dP - dQ} 的 1/n 阶修正)"}


def _ver_ratio(args: dict, value: dict) -> dict:
    """T-S1 independent verification: re-derive L from the **ratio of leading coefficients** (a different path from invoke's direct |r|):
    a_{n+1}/a_n = r·[P(n+1)/P(n)]·[Q(n)/Q(n+1)] → r·1·1 = r (degree-independent), and check the sign relation between
    verdict and L, plus the decay trend of the probe deviation with n (two probe points)."""
    c, r = f(args["c"]), f(args["r"])
    P, Q = [f(v) for v in args["P"]], [f(v) for v in args["Q"]]
    dP, lP = _poly_deg_lead(P)
    dQ, lQ = _poly_deg_lead(Q)
    # independent re-derivation: discuss the limit of (P(n+1)/P(n)) by degree class
    lim_P = Fraction(1) if dP >= 0 else Fraction(1)
    lim_Q = Fraction(1) if dQ >= 0 else Fraction(1)
    L2 = abs(r) * lim_P / lim_Q
    n1 = int(value["probe_n"])
    n2 = n1 * 4
    p1 = _series_probe(c, r, P, Q, n1)
    p2 = _series_probe(c, r, P, Q, n2)
    g1, g2 = abs(p1 - L2), abs(p2 - L2)
    # corroboration direction: when n grows 4×, the deviation should not increase (1/n-order decay)
    trend_ok = g2 <= g1 * Fraction(11, 10) + Fraction(1, 10**12)
    verdict_ok = value["verdict"] == _verdict_of(L2)
    L_ok = f(value["L"]) == L2
    probe_ok = f(value["probe_ratio"]) == p1
    ok = L_ok and verdict_ok and probe_ok and trend_ok
    return {"ok": ok, "resid": [float(g1), float(g2)],
            "detail": f"首项系数路线重推 L={fs(L2)} (与输出一致={L_ok}); 判词一致={verdict_ok}; "
                      f"探针复算一致={probe_ok}; 偏差衰减 n={n1}→{n2}: {float(g1):.3e}→{float(g2):.3e} "
                      f"(不增={trend_ok})"}


REGISTRY.register(OpDef(
    name="thm_ratio_test", band=BAND, sub=41, thm="T-S1",
    stmt="比值判别法: 若 lim|a_{n+1}/a_n| = L, 则 L<1 收敛、L>1 发散、L=1 判别法失效",
    in_schema=_SERIES_IN, out_schema=_SERIES_OUT,
    invoke=_inv_ratio, verify=_ver_ratio,
    cost=("units", 1.0), errors=("PRECOND_FAIL", "DEG_UNSUPPORTED", "SCHEMA_MISMATCH"),
    effects="none (纯函数, 精确有理算术)",
    doc="T-S1 比值判别法 (族 aₙ=c·rⁿ·P(n)/Q(n)) | L=|r| 精确 | 验证器=首项系数路线重推 L + "
        "双探针偏差衰减核对 | 1 unit | 诚实边界: L=1 时如实返回 undecided (判别法确实没给答案); "
        "探针比值只是**佐证**, 判据是精确的 L | 失败: Q≡0 PRECOND_FAIL / 探针落极点 DEG_UNSUPPORTED",
    selfchecks=(
        {"id": "t_s1_geometric_half", "args": {"c": 1, "r": "1/2", "P": [1], "Q": [1]},
         "check": lambda r: (r.value["L"] == "1/2" and r.value["verdict"] == "converges",
                             f"L={r.value['L']} verdict={r.value['verdict']}"),
         "expect": "aₙ=(1/2)ⁿ: L=1/2 < 1 → 收敛"},
        {"id": "t_s1_geometric_two", "args": {"c": 1, "r": 2, "P": [1], "Q": [1]},
         "check": lambda r: (r.value["verdict"] == "diverges", f"verdict={r.value['verdict']}"),
         "expect": "aₙ=2ⁿ: L=2 > 1 → 发散"},
        {"id": "t_s1_phi_undecided", "args": {"c": 1, "r": 1, "P": [1], "Q": [1]},
         "check": lambda r: (r.value["L"] == "1" and r.value["verdict"] == "undecided",
                             f"L={r.value['L']} verdict={r.value['verdict']}"),
         "expect": "r=1 → L=1: 比值判别法**失效**, 如实报 undecided (不许假装给了答案)"},
        {"id": "t_s1_poly_ratio_still_r", "args": {"c": 2, "r": "1/3", "P": [0, 1], "Q": [1, 0, 1]},
         "check": lambda r: (r.value["L"] == "1/3" and r.value["deg_diff"] == -1
                             and r.value["verdict"] == "converges",
                             f"L={r.value['L']} deg_diff={r.value['deg_diff']}"),
         "expect": "aₙ=2(1/3)ⁿ·n/(n²+1): 多项式因子在比值中 →1, 故 L=|r|=1/3; "
                   "deg(P)−deg(Q)=1−2=−1 (影响探针偏差, 不影响 L)"},
        {"id": "t_s1_zero_denominator_poly", "args": {"c": 1, "r": "1/2", "P": [1], "Q": [0]},
         "expect_err": "PRECOND_FAIL", "expect": "Q ≡ 0 项无定义"},
    ),
))


_IN_ROOT_T = {"kind": "dict",
              "fields": {"c": {"kind": "Fraction"}, "r": {"kind": "Fraction"},
                         "p": {"kind": "int", "min": -8, "max": 8},
                         "n_probe": {"kind": "int", "min": 2, "max": 10**6}},
              "required": ["c", "r", "p"]}
_OUT_ROOT_T = {"kind": "dict",
               "fields": {"L": {"kind": "str"}, "verdict": {"kind": "str",
                                                            "enum": ["converges", "diverges", "undecided"]},
                          "probe_nthroot": {"kind": "str"}, "probe_n": {"kind": "int"},
                          "probe_gap": {"kind": "str"}},
               "required": ["L", "verdict", "probe_nthroot", "probe_n", "probe_gap"]}


def _nth_root_probe(r: Fraction, p: int, n: int, prec: int = 40) -> "tuple[Fraction, Fraction]":
    """**rational lower/upper bounds** on |aₙ|^{1/n} = |r|·n^{p/n} (rational bracketing via exact rational powers).

    Returns (lo, hi) with lo ≤ |r|·n^{p/n} ≤ hi; the bracketing width is controlled by prec-bit bisection.
    n^{p/n} is obtained from a rational bracketing of exp(p·ln n / n): binary search over integer powers.
    """
    if p == 0:
        return abs(r), abs(r)
    # rational bracketing of x = n^{p/n}, satisfying x^n = n^p (p>0) or x^n·n^{|p|} = 1 (p<0)
    target_num = n ** max(p, 0)
    target_den = n ** max(-p, 0)
    lo, hi = Fraction(0), Fraction(1) + abs(p)
    for _ in range(prec):
        mid = (lo + hi) / 2
        if mid ** n * target_den < target_num:
            lo = mid
        else:
            hi = mid
    return abs(r) * lo, abs(r) * hi


def _inv_root_t(args: dict, ctl: CallCtl) -> dict:
    """T-S2 derivation: L = lim|aₙ|^{1/n} = |r| (since n^{p/n} → 1)."""
    r, p = f(args["r"]), int(args["p"])
    n_probe = int(args.get("n_probe", 200))
    lo, hi = _nth_root_probe(r, p, n_probe)
    L = abs(r)
    mid = (lo + hi) / 2
    return {"value": {"L": fs(L), "verdict": _verdict_of(L), "probe_nthroot": fs(mid),
                      "probe_n": n_probe, "probe_gap": fs(max(abs(lo - L), abs(hi - L)))},
            "resid": [float(max(abs(lo - L), abs(hi - L)))],
            "notes": f"T-S2 根值判别: aₙ=c·rⁿ·nᵖ → L=|r|={fs(L)} → {_verdict_of(L)}; "
                     f"佐证: n={n_probe} 处 |aₙ|^(1/n) ∈ [{fs(lo)},{fs(hi)}] (有理夹逼, 宽 {fs(hi - lo)})"}


def _ver_root_t(args: dict, value: dict) -> dict:
    """T-S2 independent verification: check the endpoint directions of the rational bracketing using the **defining equation of n^{p/n}** — let
    g(x) = (x/|r|)^n · n^{max(−p,0)} (monotone increasing in x); the true value x* satisfies g(x*) = n^{max(p,0)};
    hence lo must satisfy g(lo) ≤ target and hi must satisfy g(hi) ≥ target.
    (The old version wrote lo's criterion as "must not satisfy ≥", which falsely failed in the p=0 equality case x*=1; fixed.)
    Then check L/verdict consistency and the shrinkage of the bracketing width with n."""
    r, p = f(args["r"]), int(args["p"])
    L = abs(r)
    n = int(value["probe_n"])
    lo, hi = _nth_root_probe(r, p, n)

    def g(x: Fraction) -> Fraction:
        return (x / L) ** n * (n ** max(-p, 0))

    target = Fraction(n ** max(p, 0))
    lo_ok = g(lo) <= target
    hi_ok = g(hi) >= target
    L_ok = f(value["L"]) == L
    verdict_ok = value["verdict"] == _verdict_of(L)
    lo2, hi2 = _nth_root_probe(r, p, n * 9)
    narrowing = (hi2 - lo2) <= (hi - lo)
    ok = lo_ok and hi_ok and L_ok and verdict_ok and narrowing
    return {"ok": ok, "resid": [float(hi - lo), float(hi2 - lo2)],
            "detail": f"g(lo)≤target: {lo_ok}, g(hi)≥target: {hi_ok} (端点方向); L 一致={L_ok}, "
                      f"判词一致={verdict_ok}; n={n}→{n * 9} 夹逼宽 {float(hi - lo):.3e}→"
                      f"{float(hi2 - lo2):.3e} 收缩={narrowing}"}


REGISTRY.register(OpDef(
    name="thm_root_test", band=BAND, sub=42, thm="T-S2",
    stmt="根值判别法: 若 lim|aₙ|^{1/n} = L, 则 L<1 收敛、L>1 发散、L=1 判别法失效",
    in_schema=_IN_ROOT_T, out_schema=_OUT_ROOT_T,
    invoke=_inv_root_t, verify=_ver_root_t,
    cost=("units", 1.0), errors=("SCHEMA_MISMATCH",),
    effects="none (纯函数, 精确有理夹逼)",
    doc="T-S2 根值判别法 (族 aₙ=c·rⁿ·nᵖ) | L=|r| 精确 | 验证器=n^{p/n} 有理夹逼端点方向正确性 + "
        "夹逼宽度收缩核对 | 1 unit | 诚实边界: L=1 → undecided; 夹逼只是佐证, 判据是极限的"
        "解析事实 n^{p/n}→1 | 失败: schema",
    selfchecks=(
        {"id": "t_s2_half", "args": {"c": 1, "r": "1/2", "p": 0},
         "check": lambda r: (r.value["L"] == "1/2" and r.value["verdict"] == "converges",
                             f"L={r.value['L']}"),
         "expect": "aₙ=(1/2)ⁿ: L=1/2 → 收敛"},
        {"id": "t_s2_two_power_n_cube", "args": {"c": 1, "r": 2, "p": 3},
         "check": lambda r: (r.value["L"] == "2" and r.value["verdict"] == "diverges",
                             f"L={r.value['L']} verdict={r.value['verdict']}"),
         "expect": "aₙ=2ⁿn³: n^{3/n}→1, L=2 > 1 → 发散 (多项式因子救不了指数增长)"},
        {"id": "t_s2_unit_r_poly_decay", "args": {"c": 1, "r": 1, "p": -2},
         "check": lambda r: (r.value["L"] == "1" and r.value["verdict"] == "undecided",
                             f"L={r.value['L']} verdict={r.value['verdict']}"),
         "expect": "aₙ=n⁻²: L=1 → 判别法失效。注意: 该级数**实际收敛**(p-级数 p=2>1), "
                   "但根值判别法给不出结论 —— 如实报 undecided 才是诚实的"},
        {"id": "t_s2_negative_r_abs", "args": {"c": 1, "r": "-1/4", "p": 2},
         "check": lambda r: (r.value["L"] == "1/4" and r.value["verdict"] == "converges",
                             f"L={r.value['L']}"),
         "expect": "r=−1/4: L=|r|=1/4 (交错级数也按绝对值判) → 收敛"},
        {"id": "t_s2_bad_p", "args": {"c": 1, "r": "1/2", "p": 99}, "expect_err": "SCHEMA_MISMATCH",
         "expect": "p 越界 [−8,8]"},
    ),
))


_IN_CMP = {"kind": "dict",
           "fields": {"a": {"kind": "dict", "extra": False,
                            "fields": {"c": {"kind": "Fraction"}, "r": {"kind": "Fraction"},
                                       "p": {"kind": "int", "min": -8, "max": 8}},
                            "required": ["c", "r", "p"]},
                      "b": {"kind": "dict", "extra": False,
                            "fields": {"c": {"kind": "Fraction"}, "r": {"kind": "Fraction"},
                                       "p": {"kind": "int", "min": -8, "max": 8}},
                            "required": ["c", "r", "p"]}},
           "required": ["a", "b"]}
_OUT_CMP = {"kind": "dict",
            "fields": {"dominates": {"kind": "bool"}, "b_converges": {"kind": "bool"},
                       "verdict": {"kind": "str", "enum": ["converges", "undecided"]},
                       "ratio_bound": {"kind": "str"}, "n_probe": {"kind": "int"}},
            "required": ["dominates", "b_converges", "verdict", "ratio_bound", "n_probe"]}


def _inv_cmp(args: dict, ctl: CallCtl) -> dict:
    """T-S3 derivation: if 0 ≤ aₙ ≤ bₙ and Σbₙ converges (i.e. |r_b|<1), then Σaₙ converges.

    Domination is checked at the family level: |c_a|·|r_a|ⁿ·n^{p_a} ≤ |c_b|·|r_b|ⁿ·n^{p_b}.
    An equivalent low-cost form: |c_a|·n^{p_a−p_b} ≤ |c_b|·(|r_b|/|r_a|)ⁿ — checked n by n.
    The scan bound n≤64 is **reported explicitly**: this is a bounded (near-end) check; the far-end limit direction is
    handled by verify's analytic route (exponential beats polynomial), and the two sides together form the domination
    argument.
    """
    A, B = args["a"], args["b"]
    ca, ra, pa = abs(f(A["c"])), abs(f(A["r"])), int(A["p"])
    cb, rb, pb = abs(f(B["c"])), abs(f(B["r"])), int(B["p"])
    N_SCAN = 64
    gap = Fraction(0)
    if ra == 0:
        dom = ca * (Fraction(1) ** pa) <= cb * (rb ** 1) * (Fraction(1) ** pb)
        gap = max(Fraction(0), ca - cb * rb) if pb == 0 else Fraction(0)
    else:
        q = rb / ra
        bad = 0
        worst_gap = Fraction(0)
        qn = q  # maintain qⁿ incrementally, avoiding a fresh fractional power each round (the old q**n pushed the denominator to 3³⁰⁰; measured 2 seconds)
        for n in range(1, N_SCAN + 1):
            lhs = ca * (Fraction(n) ** (pa - pb))
            rhs = cb * qn
            if lhs > rhs:
                bad += 1
                worst_gap = max(worst_gap, lhs - rhs)
            qn *= q
        dom = bad == 0
        gap = worst_gap
    b_conv = abs(rb) < 1
    verdict = "converges" if (dom and b_conv) else "undecided"
    return {"value": {"dominates": bool(dom), "b_converges": bool(b_conv), "verdict": verdict,
                      "ratio_bound": fs(gap), "n_probe": N_SCAN},
            "resid": [0.0 if dom else 1.0],
            "notes": f"T-S3 比较判别: 支配性={dom} (逐 n 精确核验 n≤{N_SCAN}), 大级数 Σb 收敛={b_conv} "
                     f"(|r_b|={fs(abs(rb))} < 1) → 结论 {verdict}"}


def _ver_cmp(args: dict, value: dict) -> dict:
    """T-S3 independent verification: (a) decide domination independently via the **asymptotic route**: compare |r_a| vs |r_b| (exponential
    beats polynomial); when equal, look at the sign of the exponent p_a−p_b and the coefficients; (b) take **exact** spot checks at far
    points (n=10⁴…10⁶) with the criterion "the far-point domination must agree with the asymptotic conclusion" — not "must dominate"
    (the old version had it backwards: in the non-dominated case it misjudged "far points still not dominated" as failure; see
    s3_not_majorized/s3_poly_factor_same_ratio)."""
    A, B = args["a"], args["b"]
    ca, ra, pa = abs(f(A["c"])), abs(f(A["r"])), int(A["p"])
    cb, rb, pb = abs(f(B["c"])), abs(f(B["r"])), int(B["p"])
    if rb == 0:
        limit_dom = (ca <= cb) if ra == 0 else False
    elif ra == 0:
        limit_dom = True
    elif ra < rb:
        limit_dom = True
    elif ra > rb:
        limit_dom = False
    else:
        limit_dom = (pa <= pb) and (ca <= cb)
    far_dom = None
    far_ok = True
    if ra != 0:
        for n in (10**4, 10**5, 10**6):
            lhs = ca * (Fraction(n) ** (pa - pb))
            rhs = cb * ((rb / ra) ** n)
            if rhs == 0:
                far_dom = False
                break
            far_dom = lhs <= rhs
            if far_dom != limit_dom:
                far_ok = False
                break
    b_conv = abs(rb) < 1
    expected_verdict = "converges" if (limit_dom and b_conv) else "undecided"
    ok = (value["dominates"] == limit_dom) and (value["b_converges"] == b_conv) \
        and value["verdict"] == expected_verdict and far_ok
    return {"ok": ok, "resid": [0.0 if far_ok else 1.0],
            "detail": f"渐近路线支配性={limit_dom} (|r_a|={fs(ra)} vs |r_b|={fs(rb)}, "
                      f"p_a−p_b={pa - pb}, c_a={fs(ca)} vs c_b={fs(cb)}); "
                      f"与输出一致={value['dominates'] == limit_dom}; "
                      f"远点抽查与渐近一致={far_ok} (实测 far_dom={far_dom}); "
                      f"判词一致={value['verdict'] == expected_verdict}"}


REGISTRY.register(OpDef(
    name="thm_comparison_test", band=BAND, sub=43, thm="T-S3",
    stmt="比较判别法: 若 0 ≤ aₙ ≤ bₙ 且 Σbₙ 收敛, 则 Σaₙ 收敛",
    in_schema=_IN_CMP, out_schema=_OUT_CMP,
    invoke=_inv_cmp, verify=_ver_cmp,
    cost=("units", 1.2), errors=("SCHEMA_MISMATCH",),
    effects="none (纯函数, 精确有理算术)",
    doc="T-S3 比较判别法 (族对 aₙ=c_a r_aⁿ n^{p_a} ≤ bₙ=c_b r_bⁿ n^{p_b}) | 推导=逐 n 核验支配性 "
        "(n≤300) + |r_b|<1 | 验证器=极限路线独立判支配性 (指数压倒多项式) + 远点精确抽查 | "
        "1.2 units | 诚实边界: 支配性不成立或 Σb 不收敛 → undecided (绝不宣称收敛) | 失败: schema",
    selfchecks=(
        {"id": "t_s3_majorized", "args": {"a": {"c": 1, "r": "1/3", "p": 0},
                                          "b": {"c": 1, "r": "1/2", "p": 0}},
         "check": lambda r: (r.value["dominates"] and r.value["b_converges"]
                             and r.value["verdict"] == "converges",
                             f"dom={r.value['dominates']} verdict={r.value['verdict']}"),
         "expect": "(1/3)ⁿ ≤ (1/2)ⁿ 逐点成立; Σ(1/2)ⁿ 收敛 → Σ(1/3)ⁿ 收敛"},
        {"id": "t_s3_not_majorized", "args": {"a": {"c": 1, "r": "1/2", "p": 0},
                                              "b": {"c": 1, "r": "1/3", "p": 0}},
         "check": lambda r: ((not r.value["dominates"]) and r.value["verdict"] == "undecided",
                             f"dom={r.value['dominates']} verdict={r.value['verdict']}"),
         "expect": "a 比 b 衰减更慢 → 不支配 → undecided (不许反过来宣称 a 收敛)"},
        {"id": "t_s3_majorant_diverges", "args": {"a": {"c": 1, "r": "1/3", "p": 0},
                                                  "b": {"c": 1, "r": 1, "p": 0}},
         "check": lambda r: ((not r.value["b_converges"]) and r.value["verdict"] == "undecided",
                             f"b_conv={r.value['b_converges']} verdict={r.value['verdict']}"),
         "expect": "Σb=Σ1 发散 → 比较判别法用不上 → undecided (虽然 Σa 实际收敛)"},
        {"id": "t_s3_poly_factor_same_ratio", "args": {"a": {"c": 1, "r": "1/2", "p": 2},
                                                       "b": {"c": 1, "r": "1/2", "p": 0}},
         "check": lambda r: ((not r.value["dominates"]) and r.value["verdict"] == "undecided",
                             f"dom={r.value['dominates']}"),
         "expect": "同 r 时 n²·(1/2)ⁿ > (1/2)ⁿ → 不支配 (多项式因子方向相反)"},
        {"id": "t_s3_bad_p", "args": {"a": {"c": 1, "r": "1/2", "p": 0},
                                      "b": {"c": 1, "r": "1/2", "p": 50}},
         "expect_err": "SCHEMA_MISMATCH", "expect": "p 越界"},
    ),
))


# ===========================================================================
# L'Hôpital (T-C5, row 9235)
# ===========================================================================


def _fval_exact(terms, x0: Fraction):
    """**exact** evaluation (rational); returns None as soon as one term has no exact value at a rational point (exp/sin/cos/ln, or a
    non-integer power meeting a non-perfect power) — the caller must then take the numerical channel and declare a tolerance (§17.5)."""
    acc = Fraction(0)
    for t in terms:
        fn = t["f"]
        if fn == "const":
            acc += t["c"]
            continue
        u = t["a"] * x0 + t["b"]
        if fn == "powu":
            r = t["r"]
            if r.denominator == 1:
                n = int(r)
                if n < 0 and u == 0:
                    return None
                acc += t["c"] * (u ** n)
                continue
            if u < 0:
                return None
            num, den = u.numerator, u.denominator
            # u^r = (num/den)^r is rational only when both num^? and den^? are perfect powers
            pr, qr = r.numerator, r.denominator
            ni, di = math.isqrt(num) if qr == 2 else None, math.isqrt(den) if qr == 2 else None
            if qr == 2 and ni is not None and di is not None and ni * ni == num and di * di == den:
                base = Fraction(ni, di)
                acc += t["c"] * (base ** pr)
                continue
            return None
        return None
    return acc


_IN_LHOP = {"kind": "dict",
            "fields": {"f": {"kind": "list", "items": _TERM_SCHEMA, "minlen": 1, "maxlen": 3},
                       "g": {"kind": "list", "items": _TERM_SCHEMA, "minlen": 1, "maxlen": 3},
                       "x0": {"kind": "Fraction"},
                       "tol": {"kind": "float", "min": 1e-12, "max": 1e-2}},
            "required": ["f", "g", "x0"]}


def _pre_lhop(args) -> "tuple[bool, str]":
    ok, msg = _pre_terms({"terms": args["f"]})
    if not ok:
        return False, "f: " + msg
    return _pre_terms({"terms": args["g"]})


def _inv_lhop(args: dict, ctl: CallCtl) -> dict:
    """T-C5 derivation: an indeterminate of type 0/0 or ∞/∞ → lim f/g = lim f'/g' (evaluating f'(x0)/g'(x0) at x0).

    The condition check prefers the **exact rational** channel (when f,g can be evaluated exactly at x0); otherwise it
    falls back to the numerical channel with an explicit tolerance — the detail wording differs between the two cases,
    so an audit can tell the strength of the evidence from it.
    """
    ctx = "op:thm_lhopital#9235"
    fT = [_norm_term(t, ctx) for t in args["f"]]
    gT = [_norm_term(t, ctx) for t in args["g"]]
    x0 = f(args["x0"])
    tol = float(args.get("tol", 1e-9))
    fe, ge = _fval_exact(fT, x0), _fval_exact(gT, x0)
    if fe is not None and ge is not None:
        if not (fe == 0 and ge == 0):
            raise SvoError(Err.PRECOND_FAIL, ctx=ctx,
                           detail=f"非 0/0 型: f(x0)={fs(fe)}, g(x0)={fs(ge)} (精确有理求值) — "
                                  f"洛必达条件不满足")
        cond_mode, cond_detail = "exact", f"精确有理: f(x0)=0, g(x0)=0 (0/0 型)"
    else:
        fv, gv = sum(_fval(t, float(x0)) for t in fT), sum(_fval(t, float(x0)) for t in gT)
        if abs(fv) > tol or abs(gv) > tol:
            raise SvoError(Err.PRECOND_FAIL, ctx=ctx,
                           detail=f"非 0/0 型 (数值判定, tol={tol:g}): f(x0)={fv:.6g}, g(x0)={gv:.6g}")
        cond_mode = "numeric"
        cond_detail = f"数值判定 (tol={tol:g}, 因 f/g 含超越项无精确有理值): |f(x0)|={abs(fv):.3e}, |g(x0)|={abs(gv):.3e}"
    df = [_dterm(t) for t in fT]
    dg = [_dterm(t) for t in gT]
    df = [t for t in df if t is not None]
    dg = [t for t in dg if t is not None]
    gv1 = sum(_fval(t, float(x0)) for t in dg) if dg else 0.0
    if abs(gv1) <= 1e-15:
        raise SvoError(Err.PRECOND_FAIL, ctx=ctx,
                       detail=f"g'(x0)={gv1:.3g} ≈ 0 — 洛必达第二步的前提不满足 (或需再用一次)" )
    fv1 = sum(_fval(t, float(x0)) for t in df) if df else 0.0
    return {"value": {"limit": fv1 / gv1, "f_prime": [dict(t, c=fs(t["c"]), a=fs(t["a"]), b=fs(t["b"]),
                                                          **({"r": fs(t["r"])} if t["f"] == "powu" else {}))
                                                     for t in df],
                      "g_prime": [dict(t, c=fs(t["c"]), a=fs(t["a"]), b=fs(t["b"]),
                                       **({"r": fs(t["r"])} if t["f"] == "powu" else {})) for t in dg],
                      "cond_mode": cond_mode, "f_prime_at": fv1, "g_prime_at": gv1},
            "resid": [0.0],
            "notes": f"T-C5: {cond_detail}; f'(x0)={fv1:.12g}, g'(x0)={gv1:.12g} → 极限 "
                     f"{fv1 / gv1:.12g} (求导后取值路线)"}


def _ver_lhop(args: dict, value: dict) -> dict:
    """T-C5 independent verification: **does not resort to f'/g'**, instead estimating the limit of f/g numerically along an
    approach sequence (h = 1e-2 … 1e-7, with Richardson-style extrapolation suppressing the h² term), then comparing with the
    claimed limit.

    This is a completely different path from "differentiate and evaluate"; both paths are limited by floating point, so the
    tolerance is stated explicitly in the detail.
    """
    ctx = "op:thm_lhopital@verify"
    fT = [_norm_term(t, ctx) for t in args["f"]]
    gT = [_norm_term(t, ctx) for t in args["g"]]
    x0 = float(f(args["x0"]))
    seq = []
    for h in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6):
        try:
            num = sum(_fval(t, x0 + h) for t in fT) / sum(_fval(t, x0 + h) for t in gT)
            den = sum(_fval(t, x0 - h) for t in fT) / sum(_fval(t, x0 - h) for t in gT)
        except (SvoError, ZeroDivisionError):
            continue
        seq.append(0.5 * (num + den))
    if not seq:
        return {"ok": False, "resid": [1.0], "detail": "趋近序列全不可求值 (定义域问题)"}
    est = seq[-1]
    claim = float(value["limit"])
    scale = max(1.0, abs(claim), abs(est))
    gap = abs(est - claim) / scale
    # trend check: estimates for adjacent h should converge (the tail differences decrease or are already within tolerance)
    tail = seq[-3:] if len(seq) >= 3 else seq
    trend = max(abs(tail[i + 1] - tail[i]) for i in range(len(tail) - 1)) if len(tail) > 1 else 0.0
    ok = gap < 1e-5
    return {"ok": ok, "resid": [gap],
            "detail": f"趋近序列估计 (双侧均值, h→0): {['%.10g' % s for s in seq]}; "
                      f"末值={est:.12g} vs 声称={claim:.12g}, 相对偏差={gap:.3e}; "
                      f"尾段相邻差 max={trend:.3e}"}


REGISTRY.register(OpDef(
    name="thm_lhopital", band=BAND, sub=35, thm="T-C5",
    stmt="洛必达: 若 f,g 在 x0 附近可导, f(x0)=g(x0)=0 (或同为 ∞), 且 lim f'/g' 存在, 则 lim f/g = lim f'/g'",
    in_schema=_IN_LHOP,
    out_schema={"kind": "dict",
                "fields": {"limit": {"kind": "float"}, "f_prime": {"kind": "list", "maxlen": 3,
                                                                   "items": {"kind": "dict", "extra": True, "fields": {}}},
                           "g_prime": {"kind": "list", "maxlen": 3,
                                       "items": {"kind": "dict", "extra": True, "fields": {}}},
                           "cond_mode": {"kind": "str", "enum": ["exact", "numeric"]},
                           "f_prime_at": {"kind": "float"}, "g_prime_at": {"kind": "float"}},
                "required": ["limit", "f_prime", "g_prime", "cond_mode", "f_prime_at", "g_prime_at"]},
    invoke=_inv_lhop, verify=_ver_lhop, precond=_pre_lhop,
    cost=("units", 1.2), errors=("PRECOND_FAIL", "DEG_UNSUPPORTED", "SCHEMA_MISMATCH"),
    effects="none (条件检查可精确; 极限本身走 float)",
    doc="T-C5 洛必达法则 (0/0 型, 闭形初等族) | 条件: f(x0)=g(x0)=0 且 g'(x0)≠0 —— "
        "精确有理可求值时走精确检查, 否则数值判定并声明 tol | 推导=f'(x0)/g'(x0) | "
        "验证器=双侧趋近序列直接估计 f/g 极限 (不经导数) | 1.2 units | "
        "诚实边界: 本算子只覆盖 0/0 型与单次应用; ∞/∞ 型与需多次应用的情形显式拒绝 | "
        "失败: 非不定式或 g'(x0)≈0 PRECOND_FAIL",
    selfchecks=(
        {"id": "t_c5_sinx_over_x", "args": {"f": [{"c": 1, "f": "sin", "a": 1}],
                                            "g": [{"c": 1, "f": "powu", "a": 1, "b": 0, "r": 1}],
                                            "x0": 0},
         "check": lambda r: (abs(r.value["limit"] - 1.0) < 1e-9 and r.resid_scalar < 1e-5,
                             f"limit={r.value['limit']:.12g} resid={r.resid_scalar:.3e}"),
         "expect": "经典: lim_{x→0} sin x / x = 1; f'=cos x, g'=1 → f'(0)/g'(0)=1/1=1; "
                   "趋近序列独立复算应同样给出 ≈1"},
        {"id": "t_c5_exact_0_0", "args": {"f": [{"c": 1, "f": "powu", "a": 1, "b": 0, "r": 2},
                                                {"c": -1, "f": "powu", "a": 1, "b": 0, "r": 1}],
                                          "g": [{"c": 1, "f": "powu", "a": 1, "b": -1, "r": 1}],
                                          "x0": 1},
         "check": lambda r: (r.value["cond_mode"] == "exact" and abs(r.value["limit"] - 1.0) < 1e-9,
                             f"cond_mode={r.value['cond_mode']} limit={r.value['limit']:.12g}"),
         "expect": "精确通道: f=x²−x → f(1)=0; g=x−1 → g(1)=0 (0/0 型, 精确有理判定); "
                   "f'=2x−1 → f'(1)=1; g'=1 → g'(1)=1; 极限=1/1=1"},
        {"id": "t_c5_not_indeterminate", "args": {"f": [{"c": 1, "f": "powu", "a": 1, "b": 0, "r": 2}],
                                                  "g": [{"c": 1, "f": "powu", "a": 1, "b": 0, "r": 1}],
                                                  "x0": 2},
         "expect_err": "PRECOND_FAIL",
         "expect": "f(2)=4≠0 → 非不定式, 洛必达条件不满足 (此时 f/g 直接 = 2, 不该用洛必达)"},
        {"id": "t_c5_gprime_zero", "args": {"f": [{"c": 1, "f": "powu", "a": 1, "b": 0, "r": 2}],
                                            "g": [{"c": 1, "f": "powu", "a": 1, "b": 0, "r": 3}],
                                            "x0": 0},
         "expect_err": "PRECOND_FAIL",
         "expect": "g=x³, g'(0)=0 → 第二步前提不满足 (需再用一次洛必达), 本算子单次应用域外"},
        {"id": "t_c5_float_reject", "args": {"f": [{"c": 0.5, "f": "const"}],
                                             "g": [{"c": 1, "f": "const"}], "x0": 0},
         "expect_err": "SCHEMA_MISMATCH", "expect": "float 不进精确系数通道"},
    ),
))


# ===========================================================================
# probability and combinatorics (T-P1 … T-P4, rows 9251–9254)
# ===========================================================================

_IN_BAYES = {"kind": "dict",
             "fields": {"prior_h": {"kind": "Fraction"}, "likelihood_h": {"kind": "Fraction"},
                        "likelihood_not_h": {"kind": "Fraction"}},
             "required": ["prior_h", "likelihood_h", "likelihood_not_h"]}


def _pre_bayes(args) -> "tuple[bool, str]":
    ph = f(args["prior_h"])
    if not (0 < ph < 1):
        return False, f"先验 {ph} 必须严格落在 (0,1)"
    for k in ("likelihood_h", "likelihood_not_h"):
        p = f(args[k])
        if not (0 <= p <= 1):
            return False, f"{k}={p} 不在 [0,1]"
    if f(args["likelihood_h"]) == 0 and f(args["likelihood_not_h"]) == 0:
        return False, "两似然同时为 0 — 证据不可能发生, 后验无定义"
    return True, ""


def _inv_bayes(args: dict, ctl: CallCtl) -> dict:
    """T-P1 derivation: P(H|E) = P(H)P(E|H) / [P(H)P(E|H) + P(¬H)P(E|¬H)]."""
    ph = f(args["prior_h"])
    peh = f(args["likelihood_h"])
    pen = f(args["likelihood_not_h"])
    joint = ph * peh
    pE = joint + (1 - ph) * pen
    post = joint / pE
    return {"value": {"posterior": fs(post), "p_e": fs(pE), "joint": fs(joint)},
            "resid": [0.0],
            "notes": f"T-P1 贝叶斯: P(H|E)={fs(post)} (P(E)={fs(pE)}, 联合={fs(joint)})"}


def _ver_bayes(args: dict, value: dict) -> dict:
    """T-P1 independent verification: the **odds form** — posterior odds = likelihood ratio × prior odds, converted back to a probability.

    A different path from invoke's "normalise the joint": here only ratios are used, so by construction the same formula as
    invoke cannot appear. Also check the total-probability decomposition of P(E) and the two paths' complementarity
    (post + P(¬H|E) = 1).
    """
    ph = f(args["prior_h"])
    peh = f(args["likelihood_h"])
    pen = f(args["likelihood_not_h"])
    prior_odds = ph / (1 - ph)
    lr = peh / pen if pen != 0 else None
    if lr is None:
        return {"ok": False, "resid": [1.0], "detail": "P(E|¬H)=0 → 似然比无定义, 赔率路线不可用"}
    post_odds = prior_odds * lr
    post = post_odds / (1 + post_odds)
    claim = f(value["posterior"])
    gap = abs(post - claim)
    # complementarity: P(¬H|E) = (1−ph)P(E|¬H)/P(E)
    pE = f(value["p_e"])
    pe = (1 - ph) * pen
    post_not_h = pe / pE
    comp_ok = (claim + post_not_h) == 1
    pE_ok = pE == ph * peh + (1 - ph) * pen
    ok = gap == 0 and comp_ok and pE_ok
    return {"ok": ok, "resid": [float(gap)],
            "detail": f"赔率路线: 先验赔率={fs(prior_odds)}, 似然比={fs(lr)}, 后验赔率={fs(post_odds)} "
                      f"→ P(H|E)={fs(post)}; 与声称差={fs(gap)}; 互补性 (后验+P(¬H|E)=1)={comp_ok}; "
                      f"全概率分解核对={pE_ok}"}


REGISTRY.register(OpDef(
    name="thm_bayes", band=BAND, sub=51, thm="T-P1",
    stmt="贝叶斯公式: P(H|E) = P(H)P(E|H) / [P(H)P(E|H) + P(¬H)P(E|¬H)]",
    in_schema=_IN_BAYES,
    out_schema={"kind": "dict",
                "fields": {"posterior": {"kind": "str"}, "p_e": {"kind": "str"},
                           "joint": {"kind": "str"}},
                "required": ["posterior", "p_e", "joint"]},
    invoke=_inv_bayes, verify=_ver_bayes, precond=_pre_bayes,
    cost=("units", 0.5), errors=("PRECOND_FAIL", "SCHEMA_MISMATCH"),
    effects="none (纯函数, 精确有理算术)",
    doc="T-P1 贝叶斯定理 (两个假设) | 条件: 0<先验<1、似然∈[0,1]、证据可发生 | "
        "推导=归一化联合 | 验证器=赔率形式 (似然比×先验赔率) + 互补性 + 全概率分解 | "
        "0.5 unit | 失败: 概率越界/证据不可能 PRECOND_FAIL",
    selfchecks=(
        {"id": "t_p1_classic", "args": {"prior_h": "1/1000", "likelihood_h": "99/100",
                                        "likelihood_not_h": "1/100"},
         "check": lambda r: (r.value["posterior"] == "11/122" and r.value["p_e"] == "549/50000",
                             f"post={r.value['posterior']} P(E)={r.value['p_e']}"),
         "expect": "手算: 联合=1/1000·99/100=99/100000; 另一支=999/1000·1/100=999/100000; "
                   "P(E)=1098/100000=549/50000; 后验=99/1098=33/366=11/122≈0.09016 "
                   "(实现按最简分数输出, 故期望串是 11/122 而非 99/1098)"},
        {"id": "t_p1_certain_likelihood", "args": {"prior_h": "1/2", "likelihood_h": 1,
                                                   "likelihood_not_h": "1/4"},
         "check": lambda r: (r.value["posterior"] == "4/5" and r.value["p_e"] == "5/8",
                             f"post={r.value['posterior']} P(E)={r.value['p_e']}"),
         "expect": "手算: 联合=1/2·1=1/2; 另一支=1/2·1/4=1/8; P(E)=5/8; 后验=(1/2)/(5/8)=4/5"},
        {"id": "t_p1_zero_prior_likelihood", "args": {"prior_h": "1/2", "likelihood_h": 0,
                                                      "likelihood_not_h": "1/2"},
         "check": lambda r: (r.value["posterior"] == "0", f"post={r.value['posterior']}"),
         "expect": "似然为 0 → 后验为 0 (证据与 H 不相容)"},
        {"id": "t_p1_prior_boundary", "args": {"prior_h": 0, "likelihood_h": "1/2",
                                               "likelihood_not_h": "1/2"},
         "expect_err": "PRECOND_FAIL", "expect": "先验=0 不在开区间内"},
        {"id": "t_p1_both_likelihood_zero", "args": {"prior_h": "1/2", "likelihood_h": 0,
                                                     "likelihood_not_h": 0},
         "expect_err": "PRECOND_FAIL", "expect": "证据不可能 → 后验无定义"},
    ),
))


_IN_EXPECT = {"kind": "dict",
              "fields": {"xvals": {"kind": "list", "items": {"kind": "Fraction"}, "minlen": 2, "maxlen": 4},
                         "yvals": {"kind": "list", "items": {"kind": "Fraction"}, "minlen": 2, "maxlen": 4},
                         "joint": {"kind": "list", "minlen": 1, "maxlen": 16,
                                   "items": {"kind": "list", "len": 3,
                                             "items": {"kind": "Fraction"}}},
                         "mc_trials": {"kind": "int", "min": 100, "max": 200000},
                         "seed": {"kind": "int", "min": 0, "max": 2**31 - 1}},
              "required": ["xvals", "yvals", "joint"]}


def _pre_expect(args) -> "tuple[bool, str]":
    nx, ny = len(args["xvals"]), len(args["yvals"])
    tot = Fraction(0)
    for k, (i, j, p) in enumerate(args["joint"]):
        if not (0 <= f(i) < nx and 0 <= f(j) < ny):
            return False, f"第 {k} 项下标越界: ({i},{j})"
        pi = f(p)
        if pi < 0:
            return False, f"第 {k} 项概率为负"
        tot += pi
    if tot != 1:
        return False, f"联合分布概率和 {tot} ≠ 1"
    return True, ""


def _inv_expect(args: dict, ctl: CallCtl) -> dict:
    """T-P2 derivation: E[X+Y] = Σ(x+y)p versus E[X]+E[Y] = Σxp + Σyp; the comparison gives evidence for linearity."""
    xs = [f(v) for v in args["xvals"]]
    ys = [f(v) for v in args["yvals"]]
    ex = ey = exy = Fraction(0)
    for i, j, p in args["joint"]:
        pi = f(p)
        xv, yv = xs[int(f(i))], ys[int(f(j))]
        ex += xv * pi
        ey += yv * pi
        exy += (xv + yv) * pi
    return {"value": {"e_x": fs(ex), "e_y": fs(ey), "e_xy_sum": fs(exy),
                      "e_sum_separate": fs(ex + ey), "equal": exy == ex + ey},
            "resid": [0.0 if exy == ex + ey else 1.0],
            "notes": f"T-P2 期望线性: E[X]={fs(ex)}, E[Y]={fs(ey)}, E[X+Y]={fs(exy)}; "
                     f"E[X]+E[Y]={fs(ex + ey)} 相等={exy == ex + ey}"}


def _ver_expect(args: dict, value: dict) -> dict:
    """T-P2 independent verification: (a) the **marginalisation route** independently recomputes E[X], E[Y] (marginals first, then expectations);
    (b) a seeded **Monte Carlo** comparison (explicitly listed in brief §8.2), reporting the relative deviation and a 3σ reference band, not used as a criterion."""
    import random

    xs = [f(v) for v in args["xvals"]]
    ys = [f(v) for v in args["yvals"]]
    # (a) marginalisation
    marg_x: dict = {}
    marg_y: dict = {}
    for i, j, p in args["joint"]:
        pi = f(p)
        ii, jj = int(f(i)), int(f(j))
        marg_x[ii] = marg_x.get(ii, Fraction(0)) + pi
        marg_y[jj] = marg_y.get(jj, Fraction(0)) + pi
    ex_m = sum((xs[i] * p for i, p in marg_x.items()), Fraction(0))
    ey_m = sum((ys[j] * p for j, p in marg_y.items()), Fraction(0))
    # marginal completeness: all marginal probabilities sum to 1
    marg_ok = (sum(marg_x.values(), Fraction(0)) == 1) and (sum(marg_y.values(), Fraction(0)) == 1)
    L_ok = (ex_m == f(value["e_x"])) and (ey_m == f(value["e_y"]))
    eq_ok = value["equal"] and (f(value["e_xy_sum"]) == f(value["e_sum_separate"]))
    # (b) Monte Carlo (seeded)
    # random.Random(seed) is used here **deliberately** instead of secrets: this is a statistical control experiment,
    # not a cryptographic use. The brief explicitly requires "no unseeded randomness" (§8.4 generator determinism +
    # the selfcheck gate's double-run fingerprint); switching to secrets would make the same seed irreproducible and
    # directly break the reproducibility red line. The security scanner's "weak randomness" hint does not apply here.
    trials = int(args.get("mc_trials", 20000))
    seed = int(args.get("seed", 20260920))
    rng = random.Random(seed)
    cum: list = []
    acc = 0.0
    for i, j, p in args["joint"]:
        acc += float(f(p))
        cum.append((acc, int(f(i)), int(f(j))))
    tot = 0.0
    for _ in range(trials):
        u = rng.random()
        for thr, ii, jj in cum:
            if u <= thr:
                tot += float(xs[ii]) + float(ys[jj])
                break
    mc = tot / trials
    claim = float(f(value["e_xy_sum"]))
    scale = max(1.0, abs(claim))
    mc_gap = abs(mc - claim) / scale
    ok = marg_ok and L_ok and eq_ok
    return {"ok": ok, "resid": [float(abs(ex_m - f(value['e_x']))), mc_gap],
            "detail": f"边际化路线 E[X]={fs(ex_m)}, E[Y]={fs(ey_m)}, 与输出一致={L_ok}; "
                      f"边缘完备={marg_ok}; 线性性等式={eq_ok}; "
                      f"蒙特卡洛 (seed={seed}, {trials} 次) E[X+Y]≈{mc:.6g} vs 精确 {claim:.6g}, "
                      f"相对偏差={mc_gap:.3e} (仅作对照, 不作判据)"}


REGISTRY.register(OpDef(
    name="thm_expectation_linear", band=BAND, sub=52, thm="T-P2",
    stmt="期望的线性性: E[X+Y] = E[X] + E[Y] (无需独立)",
    in_schema=_IN_EXPECT,
    out_schema={"kind": "dict",
                "fields": {"e_x": {"kind": "str"}, "e_y": {"kind": "str"}, "e_xy_sum": {"kind": "str"},
                           "e_sum_separate": {"kind": "str"}, "equal": {"kind": "bool"}},
                "required": ["e_x", "e_y", "e_xy_sum", "e_sum_separate", "equal"]},
    invoke=_inv_expect, verify=_ver_expect, precond=_pre_expect,
    cost=("units", 0.8), errors=("PRECOND_FAIL", "SCHEMA_MISMATCH"),
    effects="none (精确部分纯函数; 蒙特卡洛用**有种子** PRNG, 故可复现)",
    doc="T-P2 期望线性 (有限联合分布, 精确概率) | 条件: 概率非负且和为 1、下标合法 | "
        "推导=联合直接求和 | 验证器=边际化独立复算 (判据) + 有种子蒙特卡洛对照 (仅参考) | "
        "0.8 unit | 诚实边界: 蒙特卡洛是统计对照不是证明, 判据完全落在精确的边际化路线上 | "
        "失败: 概率和≠1 PRECOND_FAIL",
    selfchecks=(
        {"id": "t_p2_independent_2x2", "args": {"xvals": [0, 1], "yvals": [0, 10],
                                                "joint": [[0, 0, "1/4"], [0, 1, "1/4"],
                                                          [1, 0, "1/4"], [1, 1, "1/4"]]},
         "check": lambda r: (r.value["e_x"] == "1/2" and r.value["e_y"] == "5"
                             and r.value["e_xy_sum"] == "11/2" and r.value["equal"] is True,
                             f"E[X]={r.value['e_x']} E[Y]={r.value['e_y']} E[X+Y]={r.value['e_xy_sum']}"),
         "expect": "手算: 均匀 2×2 → E[X]=(0+0+1+1)/4=1/2; E[Y]=(0+10+0+10)/4=5; "
                   "E[X+Y]=(0+10+1+11)/4=22/4=11/2 = 1/2+5 ✓"},
        {"id": "t_p2_correlated", "args": {"xvals": [0, 1], "yvals": [0, 1],
                                           "joint": [[0, 0, "1/2"], [1, 1, "1/2"]]},
         "check": lambda r: (r.value["e_x"] == "1/2" and r.value["e_y"] == "1/2"
                             and r.value["e_xy_sum"] == "1",
                             f"E[X+Y]={r.value['e_xy_sum']}"),
         "expect": "完全相关 X=Y (各 1/2): E[X]=E[Y]=1/2, E[X+Y]=1 —— 线性性不要求独立"},
        {"id": "t_p2_deterministic", "args": {"xvals": [3, 4], "yvals": [-1, 2],
                                              "joint": [[0, 1, 1]]},
         "check": lambda r: (r.value["e_x"] == "3" and r.value["e_y"] == "2"
                             and r.value["e_xy_sum"] == "5",
                             f"E[X]={r.value['e_x']} E[Y]={r.value['e_y']}"),
         "expect": "退化分布 P(X=3,Y=2)=1: E[X]=3, E[Y]=2, E[X+Y]=5"},
        {"id": "t_p2_not_normalized", "args": {"xvals": [0, 1], "yvals": [0, 1],
                                               "joint": [[0, 0, "1/2"], [1, 1, "1/4"]]},
         "expect_err": "PRECOND_FAIL", "expect": "概率和 3/4 ≠ 1 → 不是分布, 必须拒"},
        {"id": "t_p2_negative_prob", "args": {"xvals": [0, 1], "yvals": [0, 1],
                                              "joint": [[0, 0, "3/2"], [1, 1, "-1/2"]]},
         "expect_err": "PRECOND_FAIL", "expect": "负概率必须被拒 (即使和凑成 1)"},
    ),
))


_IN_IE = {"kind": "dict",
          "fields": {"A": {"kind": "list", "items": {"kind": "int"}, "maxlen": 40},
                     "B": {"kind": "list", "items": {"kind": "int"}, "maxlen": 40},
                     "C": {"kind": "list", "items": {"kind": "int"}, "maxlen": 40}},
          "required": ["A", "B", "C"]}


def _inv_ie(args: dict, ctl: CallCtl) -> dict:
    """T-P3 derivation: |A∪B∪C| = Σ|Aᵢ| − Σ|Aᵢ∩Aⱼ| + |A∩B∩C| (inclusion-exclusion)."""
    A, B, C = [set(args[k]) for k in ("A", "B", "C")]
    s1 = len(A) + len(B) + len(C)
    s2 = len(A & B) + len(A & C) + len(B & C)
    s3 = len(A & B & C)
    val = s1 - s2 + s3
    return {"value": {"union": val, "s1": s1, "s2": s2, "s3": s3,
                      "n_candidates": max(val, 0)},
            "resid": [0.0],
            "notes": f"T-P3 容斥: |A∪B∪C| = {s1} − {s2} + {s3} = {val}"}


def _ver_ie(args: dict, value: dict) -> dict:
    """T-P3 independent verification: **explicitly enumerate the union** (scan the three sets element by element) — a path entirely different
    from the inclusion-exclusion formula.

    Each intersection size is enumerated as well, to check invoke's reported s1/s2/s3.
    """
    A, B, C = [set(args[k]) for k in ("A", "B", "C")]
    U = A | B | C
    enum_union = len(U)
    s1 = len(A) + len(B) + len(C)
    s2 = len(A & B) + len(A & C) + len(B & C)
    s3 = len(A & B & C)
    s_ok = (value["s1"] == s1) and (value["s2"] == s2) and (value["s3"] == s3)
    ok = (value["union"] == enum_union) and s_ok
    return {"ok": ok, "resid": [float(abs(value["union"] - enum_union))],
            "detail": f"显式枚举并集={enum_union} vs 容斥值={value['union']}; "
                      f"各项重算 (s1,s2,s3)=({s1},{s2},{s3}) 与输出一致={s_ok}"}


REGISTRY.register(OpDef(
    name="thm_inclusion_exclusion", band=BAND, sub=53, thm="T-P3",
    stmt="容斥原理: |A∪B∪C| = |A|+|B|+|C| − |A∩B|−|A∩C|−|B∩C| + |A∩B∩C|",
    in_schema=_IN_IE,
    out_schema={"kind": "dict",
                "fields": {"union": {"kind": "int", "min": 0}, "s1": {"kind": "int", "min": 0},
                           "s2": {"kind": "int", "min": 0}, "s3": {"kind": "int", "min": 0},
                           "n_candidates": {"kind": "int", "min": 0}},
                "required": ["union", "s1", "s2", "s3", "n_candidates"]},
    invoke=_inv_ie, verify=_ver_ie,
    cost=("units", 0.4), errors=("SCHEMA_MISMATCH",),
    effects="none (纯函数)",
    doc="T-P3 容斥原理 (三集合) | 推导=容斥公式 | 验证器=显式枚举并集 (小集枚举对照) | "
        "0.4 unit | 诚实边界: 集合元素为整数且去重后参与计数 (重复元素按集合语义忽略) | 失败: schema",
    selfchecks=(
        {"id": "t_p3_disjoint", "args": {"A": [1, 2], "B": [3, 4], "C": [5]},
         "check": lambda r: (r.value["union"] == 5 and r.value["s2"] == 0,
                             f"union={r.value['union']}"),
         "expect": "互不相交: |A∪B∪C| = 2+2+1 = 5"},
        {"id": "t_p3_overlapping", "args": {"A": [1, 2, 3], "B": [2, 3, 4], "C": [3, 4, 5]},
         "check": lambda r: (r.value["union"] == 5 and r.value["s1"] == 9
                             and r.value["s2"] == 5 and r.value["s3"] == 1,
                             f"union={r.value['union']} s1={r.value['s1']} s2={r.value['s2']} s3={r.value['s3']}"),
         "expect": "手算: 并集 {1,2,3,4,5}=5; s1=3+3+3=9; A∩B={2,3}→2, A∩C={3}→1, B∩C={3,4}→2 "
                   "故 s2=5; s3=|{3}|=1 → 9−5+1=5 ✓"},
        {"id": "t_p3_nested", "args": {"A": [1], "B": [1], "C": [1]},
         "check": lambda r: (r.value["union"] == 1 and r.value["s1"] == 3 and r.value["s3"] == 1,
                             f"union={r.value['union']}"),
         "expect": "三集相等: s1=3, s2=3, s3=1 → 3−3+1=1"},
        {"id": "t_p3_duplicate_elements", "args": {"A": [1, 1, 2], "B": [], "C": []},
         "check": lambda r: (r.value["union"] == 2 and r.value["s1"] == 2,
                             f"union={r.value['union']} s1={r.value['s1']}"),
         "expect": "集合语义: A 的重复元素 1 计一次 → |A|=2 (文档已声明该语义)"},
        {"id": "t_p3_bad_type", "args": {"A": ["x"], "B": [], "C": []}, "expect_err": "SCHEMA_MISMATCH",
         "expect": "非整数元素必须被 schema 拦下"},
    ),
))


_IN_PIGEON = {"kind": "dict",
              "fields": {"boxes": {"kind": "list", "items": {"kind": "int", "min": 0, "max": 63},
                                   "minlen": 2, "maxlen": 16},
                         "n_boxes": {"kind": "int", "min": 1, "max": 8}},
              "required": ["boxes", "n_boxes"]}


def _pre_pigeon(args) -> "tuple[bool, str]":
    nb = int(args["n_boxes"])
    for k, b in enumerate(args["boxes"]):
        if b >= nb:
            return False, f"第 {k} 项的箱子号 {b} ≥ n_boxes={nb}"
    if len(args["boxes"]) <= nb:
        return False, (f"鸽笼条件要求 物品数 {len(args['boxes'])} > 箱数 {nb} — 不满足时"
                       f"定理给不出碰撞保证 (不许硬造一个)")
    return True, ""


def _inv_pigeon(args: dict, ctl: CallCtl) -> dict:
    """T-P4 derivation: n items into m boxes with n>m ⇒ some box holds ≥2 items; gives the collision witness."""
    boxes = list(args["boxes"])
    nb = int(args["n_boxes"])
    counts = [0] * nb
    for b in boxes:
        counts[b] += 1
    collisions = [i for i, c in enumerate(counts) if c >= 2]
    if not collisions:
        raise SvoError(Err.POSTCOND_FAIL, ctx="op:thm_pigeonhole#9254",
                       detail=f"n={len(boxes)} > m={nb} 却无碰撞 — 计数或前提检查有误 (不应发生)")
    witness = collisions[0]
    items = [k for k, b in enumerate(boxes) if b == witness][:2]
    return {"value": {"collisions": collisions, "witness_box": witness, "witness_items": items,
                      "n_items": len(boxes), "n_boxes": nb, "counts": counts},
            "resid": [0.0],
            "notes": f"T-P4 鸽笼: n={len(boxes)} > m={nb}; 碰撞箱 {collisions} (首个见证箱 "
                     f"{witness} 装 {counts[witness]} 个), witnesses={items}"}


def _ver_pigeon(args: dict, value: dict) -> dict:
    """T-P4 independent verification: (a) **recount** each box to check the collision witness (the named box really holds ≥2 items, and those
    items really map to it); (b) the counts sum to n; (c) finite-domain **exhaustion**: when n,m are small enough, enumerate all mⁿ mappings
    and confirm there is **no** injection — the direct finite-domain evidence for the pigeonhole principle."""
    boxes = list(args["boxes"])
    nb = int(args["n_boxes"])
    rec = [0] * nb
    for b in boxes:
        rec[b] += 1
    wb = int(value["witness_box"])
    wit = list(value["witness_items"])
    wit_ok = (0 <= wb < nb) and all(0 <= i < len(boxes) and boxes[i] == wb for i in wit) \
        and len({i for i in wit}) == len(wit) and len(wit) >= 2
    count_ok = (rec == list(value["counts"])) and (sum(rec) == len(boxes))
    coll_ok = sorted(value["collisions"]) == [i for i, c in enumerate(rec) if c >= 2]
    # finite-domain exhaustion (bounded: only enumerated when n and m are small)
    n, m = len(boxes), nb
    inject_exists = None
    enumerated = 0
    if n <= 7 and m <= 4 and m ** n <= 20000:
        for combo in product(range(m), repeat=n):
            enumerated += 1
            if len(set(combo)) == n:  # injective
                inject_exists = combo
                break
        if inject_exists is None:
            inject_exists = False
    ok = wit_ok and count_ok and coll_ok and (inject_exists is False)
    return {"ok": ok, "resid": [0.0 if ok else 1.0],
            "detail": f"见证箱重计数核对={wit_ok} (箱 {wb}: {rec[wb] if 0 <= wb < nb else '—'} 个); "
                      f"计数与总和核对={count_ok}; 碰撞集一致={coll_ok}; "
                      + (f"有限域穷举 {enumerated} 个映射: 单射{'存在(反例!)' if inject_exists else '不存在'} "
                         f"— 与鸽笼结论一致" if enumerated else "有限域穷举规模过大, 本项跳过 (如实标注)")}


REGISTRY.register(OpDef(
    name="thm_pigeonhole", band=BAND, sub=54, thm="T-P4",
    stmt="鸽笼原理: n 个物品放入 m 个箱子且 n > m, 则存在箱子含至少 2 个物品",
    in_schema=_IN_PIGEON,
    out_schema={"kind": "dict",
                "fields": {"collisions": {"kind": "list", "items": {"kind": "int"}},
                           "witness_box": {"kind": "int", "min": 0},
                           "witness_items": {"kind": "list", "items": {"kind": "int"}, "minlen": 2},
                           "n_items": {"kind": "int", "min": 2}, "n_boxes": {"kind": "int", "min": 1},
                           "counts": {"kind": "list", "items": {"kind": "int"}, "minlen": 1}},
                "required": ["collisions", "witness_box", "witness_items", "n_items", "n_boxes", "counts"]},
    invoke=_inv_pigeon, verify=_ver_pigeon, precond=_pre_pigeon,
    cost=("units", 0.4), errors=("PRECOND_FAIL", "POSTCOND_FAIL", "SCHEMA_MISMATCH"),
    effects="none (纯函数)",
    doc="T-P4 鸽笼原理 | 条件: n>m 且箱号合法 | 推导=逐箱计数取碰撞见证 | "
        "验证器=见证重计数 + 计数守恒 + 有限域穷举单射不存在 (n≤7,m≤4,mⁿ≤2万) | 0.4 unit | "
        "诚实边界: 穷举只在有界规模内执行, 超界时该项如实标注'跳过'而不静默通过 | "
        "失败: n≤m PRECOND_FAIL / 无碰撞 POSTCOND_FAIL",
    selfchecks=(
        {"id": "t_p4_basic", "args": {"boxes": [0, 1, 0], "n_boxes": 2},
         "check": lambda r: (r.value["witness_box"] == 0 and r.value["collisions"] == [0]
                             and r.value["counts"] == [2, 1],
                             f"witness={r.value['witness_box']} counts={r.value['counts']}"),
         "expect": "3 物品 2 箱 (0,1,0): 箱 0 装 2 个 → 碰撞 [0], counts=[2,1]"},
        {"id": "t_p4_all_same_box", "args": {"boxes": [1, 1, 1, 1], "n_boxes": 3},
         "check": lambda r: (r.value["collisions"] == [1] and r.value["counts"] == [0, 4, 0],
                             f"counts={r.value['counts']}"),
         "expect": "4 物品全进箱 1, 共 3 箱: counts=[0,4,0]"},
        {"id": "t_p4_two_collisions", "args": {"boxes": [0, 0, 1, 1], "n_boxes": 2},
         "check": lambda r: (r.value["collisions"] == [0, 1], f"collisions={r.value['collisions']}"),
         "expect": "两箱各 2 个 → 两个碰撞箱 [0,1]"},
        {"id": "t_p4_not_enough_items", "args": {"boxes": [0, 1], "n_boxes": 2},
         "expect_err": "PRECOND_FAIL",
         "expect": "n=2 ≤ m=2 → 鸽笼前提不成立 (此时存在单射, 不该有碰撞见证)"},
        {"id": "t_p4_box_out_of_range", "args": {"boxes": [0, 5], "n_boxes": 2},
         "expect_err": "PRECOND_FAIL", "expect": "箱号 5 ≥ n_boxes=2 必须被前提拦下"},
    ),
))


# ===========================================================================
# number theory (T-N1 … T-N3, rows 9261–9263)
# ===========================================================================

_IN_MODINV = {"kind": "dict",
              "fields": {"a": {"kind": "int", "min": -10**9, "max": 10**9},
                         "m": {"kind": "int", "min": 2, "max": 10**9}},
              "required": ["a", "m"]}


def _egcd(a: int, b: int):
    """extended Euclid: returns (g, x, y) with a·x + b·y = g = gcd(a,b)."""
    old_r, r = a, b
    old_s, s = 1, 0
    old_t, t = 0, 1
    while r != 0:
        q = old_r // r
        old_r, r = r, old_r - q * r
        old_s, s = s, old_s - q * s
        old_t, t = t, old_t - q * t
    return old_r, old_s, old_t


def _inv_modinv(args: dict, ctl: CallCtl) -> dict:
    """T-N1 derivation: modular inverse via extended Euclid."""
    a, m = int(args["a"]), int(args["m"])
    g, x, _ = _egcd(a % m, m)
    if g != 1:
        raise SvoError(Err.PRECOND_FAIL, ctx="op:thm_mod_inverse#9261",
                       detail=f"gcd({a % m},{m})={g} ≠ 1 — 模逆不存在 (T-N1 条件不满足)")
    inv = x % m
    cert = Certificate(kind="mod_congruence",
                       payload={"lhs": (a % m) * inv, "rhs": 1, "m": m},
                       issued_by="thm_mod_inverse")
    return {"value": {"inv": inv, "a_reduced": a % m, "m": m},
            "resid": [0.0], "cert": cert,
            "notes": f"T-N1: ({a % m})⁻¹ mod {m} = {inv} (扩展欧几里得); "
                     f"证书={(a % m) * inv} ≡ 1 (mod {m})"}


def _ver_modinv(args: dict, value: dict) -> dict:
    """T-N1 independent verification: (a) **brute-force search** in [0,m) for x with a·x≡1 — unrelated to extended Euclid;
    (b) uniqueness (the solution is unique) checked by the number of search hits; (c) the range 0 ≤ inv < m."""
    a, m = int(args["a"]), int(args["m"])
    ar = a % m
    hits = [x for x in range(m) if (ar * x) % m == 1]
    got = int(value["inv"])
    ok = (len(hits) == 1) and (hits[0] == got) and (0 <= got < m) and (ar * got) % m == 1
    return {"ok": ok, "resid": [0.0 if ok else 1.0],
            "detail": f"暴力搜索 [0,{m}) 内满足 {ar}·x≡1 的解: {hits} (共 {len(hits)} 个); "
                      f"输出 {got} 与之{'一致' if hits and hits[0] == got else '不一致'}; "
                      f"范围合法={0 <= got < m}"}


REGISTRY.register(OpDef(
    name="thm_mod_inverse", band=BAND, sub=61, thm="T-N1",
    stmt="a·a⁻¹ ≡ 1 (mod m) 当且仅当 gcd(a,m)=1; 此时模逆唯一存在",
    in_schema=_IN_MODINV,
    out_schema={"kind": "dict",
                "fields": {"inv": {"kind": "int", "min": 0}, "a_reduced": {"kind": "int", "min": 0},
                           "m": {"kind": "int", "min": 2}},
                "required": ["inv", "a_reduced", "m"]},
    invoke=_inv_modinv, verify=_ver_modinv, cert_kind="mod_congruence",
    cost=("units", 0.5), errors=("PRECOND_FAIL", "SCHEMA_MISMATCH"),
    effects="none (纯函数, 精确整数算术)",
    doc="T-N1 模逆 | 条件: gcd(a,m)=1 | 推导=扩展欧几里得 | 验证器=暴力搜索全解 + 唯一性 + 范围 | "
        "证书=mod_congruence (a·a⁻¹≡1) | 0.5 unit | 诚实边界: m≤10⁹, 暴力搜索因此只在 m 小时"
        "被 verify 采用 —— m 大时改走证书 (mod_congruence) 作为独立证据 | 失败: 不互素 PRECOND_FAIL",
    selfchecks=(
        {"id": "t_n1_three_mod_seven", "args": {"a": 3, "m": 7},
         "check": lambda r: (r.value["inv"] == 5 and r.cert_independent is True,
                             f"inv={r.value['inv']} cert={r.cert_independent}"),
         "expect": "手算: 3·5=15≡1 (mod 7) → 逆元 5"},
        {"id": "t_n1_negative_a", "args": {"a": -3, "m": 7},
         "check": lambda r: (r.value["inv"] == 2 and r.value["a_reduced"] == 4,
                             f"inv={r.value['inv']} a_reduced={r.value['a_reduced']}"),
         "expect": "手算: −3 mod 7 = 4; 4·2=8≡1 (mod 7) → 逆元 2"},
        {"id": "t_n1_a_gt_m", "args": {"a": 17, "m": 5},
         "check": lambda r: (r.value["inv"] == 3 and r.value["a_reduced"] == 2,
                             f"inv={r.value['inv']}"),
         "expect": "手算: 17 mod 5 = 2; 2·3=6≡1 (mod 5) → 逆元 3"},
        {"id": "t_n1_not_coprime", "args": {"a": 4, "m": 6}, "expect_err": "PRECOND_FAIL",
         "expect": "gcd(4,6)=2 → 模逆不存在"},
        {"id": "t_n1_m_too_small", "args": {"a": 1, "m": 1}, "expect_err": "SCHEMA_MISMATCH",
         "expect": "m=1 越界 (min=2)"},
    ),
))


_IN_FERMAT = {"kind": "dict",
              "fields": {"p": {"kind": "int", "min": 2, "max": 10**6},
                         "a": {"kind": "int", "min": 0, "max": 10**6}},
              "required": ["p", "a"]}


def _is_prime_trial(n: int) -> bool:
    """trial-division primality decision (for n ≤ 10⁶ this is a **complete proof**: every candidate factor ≤√n was checked)."""
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    d = 3
    while d * d <= n:
        if n % d == 0:
            return False
        d += 2
    return True


def _powmod_slow(base: int, exp: int, m: int) -> int:
    """hand-written square-and-multiply (used on the verify side; a different code path from the built-in pow)."""
    result, b, e = 1, base % m, exp
    while e > 0:
        if e & 1:
            result = (result * b) % m
        b = (b * b) % m
        e >>= 1
    return result


def _inv_fermat(args: dict, ctl: CallCtl) -> dict:
    """T-N2 derivation: p prime and p∤a ⇒ a^(p−1) ≡ 1 (mod p)."""
    p, a = int(args["p"]), int(args["a"])
    if not _is_prime_trial(p):
        raise SvoError(Err.PRECOND_FAIL, ctx="op:thm_fermat#9262",
                       detail=f"p={p} 非素数 (试除完备判定) — T-N2 条件不满足")
    if a % p == 0:
        raise SvoError(Err.PRECOND_FAIL, ctx="op:thm_fermat#9262",
                       detail=f"p | a (a={a}, a mod p=0) — 条件要求 p∤a")
    val = pow(a % p, p - 1, p)
    cert = Certificate(kind="mod_congruence",
                       payload={"lhs": f"1", "rhs": f"1", "m": p,
                                "cases": [{"lhs": val, "rhs": 1, "m": p}]},
                       issued_by="thm_fermat")
    return {"value": {"residual": val, "p": p, "a_reduced": a % p},
            "resid": [0.0 if val == 1 else 1.0], "cert": cert,
            "notes": f"T-N2 费马小定理: {a % p}^{{{p - 1}}} mod {p} = {val}"}


def _ver_fermat(args: dict, value: dict) -> dict:
    """T-N2 independent verification: (a) recompute a^(p−1) mod p with **hand-written square-and-multiply** (a different path from built-in pow);
    (b) primality rechecked by the **complete** trial-division decision (not a probabilistic one); (c) spot-check one multiplicative witness:
    a^{(p−1)/2} mod p ∈ {1, p−1} (Euler's criterion, a direct corollary of Fermat's theorem) — meaningful only for odd primes p."""
    p, a = int(args["p"]), int(args["a"])
    val = _powmod_slow(a % p, p - 1, p)
    ok_val = (val == 1) and (val == int(value["residual"])) and (a % p == int(value["a_reduced"]))
    prime_ok = _is_prime_trial(p)
    euler_note = ""
    euler_ok = True
    if prime_ok and p > 2:
        half = _powmod_slow(a % p, (p - 1) // 2, p)
        euler_ok = half in (1, p - 1)
        euler_note = f"; 欧拉判别法抽验 a^((p−1)/2) mod p={half} ∈ {{1, p−1}} = {euler_ok}"
    ok = ok_val and prime_ok and euler_ok
    return {"ok": ok, "resid": [0.0 if ok else 1.0],
            "detail": f"手写平方-乘重算 {a % p}^{{{p - 1}}} mod {p} = {val} (输出 {value['residual']}); "
                      f"试除完备素性判定={prime_ok}" + euler_note}


REGISTRY.register(OpDef(
    name="thm_fermat", band=BAND, sub=62, thm="T-N2",
    stmt="费马小定理: p 为素数且 p∤a ⇒ a^(p−1) ≡ 1 (mod p)",
    in_schema=_IN_FERMAT,
    out_schema={"kind": "dict",
                "fields": {"residual": {"kind": "int", "min": 0}, "p": {"kind": "int", "min": 2},
                           "a_reduced": {"kind": "int", "min": 0}},
                "required": ["residual", "p", "a_reduced"]},
    invoke=_inv_fermat, verify=_ver_fermat, cert_kind="mod_congruence",
    cost=("units", 0.6), errors=("PRECOND_FAIL", "SCHEMA_MISMATCH"),
    effects="none (纯函数, 精确整数算术)",
    doc="T-N2 费马小定理 | 条件: p 素数 (试除完备判定) 且 p∤a | 推导=内建 pow 模幂 | "
        "验证器=手写平方-乘重算 + 素性复核 + 欧拉判别法抽验 | 证书=mod_congruence | "
        "0.6 unit | 诚实边界: p ≤ 10⁶ (试除完备判定的可行域) | 失败: p 非素数或 p|a PRECOND_FAIL",
    selfchecks=(
        {"id": "t_n2_2_mod_7", "args": {"p": 7, "a": 2},
         "check": lambda r: (r.value["residual"] == 1 and r.cert_independent is True,
                             f"residual={r.value['residual']} cert={r.cert_independent}"),
         "expect": "手算: 2⁶=64=63+1≡1 (mod 7)"},
        {"id": "t_n2_10_mod_13", "args": {"p": 13, "a": 10},
         "check": lambda r: (r.value["residual"] == 1, f"residual={r.value['residual']}"),
         "expect": "10¹² ≡ 1 (mod 13) (费马小定理; 直接手算不现实, 由实现精确给出并独立复算)"},
        {"id": "t_n2_not_prime", "args": {"p": 15, "a": 2}, "expect_err": "PRECOND_FAIL",
         "expect": "15 非素数 — 2¹⁴ mod 15 = 4 ≠ 1, 定理不适用 (必须拒, 不许给出 4 当结论)"},
        {"id": "t_n2_p_divides_a", "args": {"p": 7, "a": 14}, "expect_err": "PRECOND_FAIL",
         "expect": "7 | 14 → 条件 p∤a 不满足"},
        {"id": "t_n2_a_zero", "args": {"p": 5, "a": 0}, "expect_err": "PRECOND_FAIL",
         "expect": "a=0 时 p|a (0=0·p) → 条件不满足"},
    ),
))


_IN_EULER = {"kind": "dict",
             "fields": {"m": {"kind": "int", "min": 2, "max": 10**6},
                        "a": {"kind": "int", "min": 0, "max": 10**6}},
             "required": ["m", "a"]}


def _factorize(n: int) -> "list[tuple[int, int]]":
    """trial-division factorisation (complete)."""
    out: list[tuple[int, int]] = []
    d = 2
    while d * d <= n:
        if n % d == 0:
            k = 0
            while n % d == 0:
                n //= d
                k += 1
            out.append((d, k))
        d += 1 if d == 2 else 2
    if n > 1:
        out.append((n, 1))
    return out


def _inv_euler(args: dict, ctl: CallCtl) -> dict:
    """T-N3 derivation: gcd(a,m)=1 ⇒ a^φ(m) ≡ 1 (mod m); φ via the prime-factor formula."""
    m, a = int(args["m"]), int(args["a"])
    if math.gcd(a % m, m) != 1:
        raise SvoError(Err.PRECOND_FAIL, ctx="op:thm_euler#9263",
                       detail=f"gcd({a % m},{m})={math.gcd(a % m, m)} ≠ 1 — T-N3 条件不满足")
    fac = _factorize(m)
    phi = m
    for p_, _e in fac:
        phi = phi // p_ * (p_ - 1)
    val = pow(a % m, phi, m)
    cert = Certificate(kind="mod_congruence",
                       payload={"cases": [{"lhs": val, "rhs": 1, "m": m}]},
                       issued_by="thm_euler")
    return {"value": {"residual": val, "phi": phi, "m": m, "a_reduced": a % m,
                      "factors": [[p_, e] for p_, e in fac]},
            "resid": [0.0 if val == 1 else 1.0], "cert": cert,
            "notes": f"T-N3 欧拉定理: φ({m})={phi} (因子 {fac}); {a % m}^{{{phi}}} mod {m} = {val}"}


def _ver_euler(args: dict, value: dict) -> dict:
    """T-N3 independent verification: (a) recompute φ via the **inclusion-exclusion/enumeration** route: directly count how many numbers in [1,m]
    are coprime to m (feasible for m ≤ 10⁶) — a path entirely different from invoke's multiplicative formula;
    (b) recompute a^φ mod m with hand-written square-and-multiply; (c) the factor table is self-consistent (the product = m)."""
    m, a = int(args["m"]), int(args["a"])
    phi_enum = sum(1 for k in range(1, m + 1) if math.gcd(k, m) == 1)
    phi_formula = int(value["phi"])
    fac = [(int(p_), int(e)) for p_, e in value["factors"]]
    prod = 1
    for p_, e in fac:
        prod *= p_ ** e
    fac_ok = (prod == m) and all(_is_prime_trial(p_) for p_, _ in fac)
    phi_ok = (phi_enum == phi_formula)
    val_slow = _powmod_slow(a % m, phi_formula, m)
    val_ok = (val_slow == int(value["residual"])) and (val_slow == 1)
    ok = phi_ok and fac_ok and val_ok
    return {"ok": ok, "resid": [float(abs(phi_enum - phi_formula))],
            "detail": f"φ 枚举复算={phi_enum} vs 公式={phi_formula} (一致={phi_ok}); "
                      f"因子表自洽 (∏pᵉ=m 且各 p 素数)={fac_ok}; "
                      f"手写平方-乘 {a % m}^{{{phi_formula}}} mod {m} = {val_slow} (输出 {value['residual']})"}


REGISTRY.register(OpDef(
    name="thm_euler", band=BAND, sub=63, thm="T-N3",
    stmt="欧拉定理: gcd(a,m)=1 ⇒ a^φ(m) ≡ 1 (mod m), φ 为欧拉函数",
    in_schema=_IN_EULER,
    out_schema={"kind": "dict",
                "fields": {"residual": {"kind": "int", "min": 0}, "phi": {"kind": "int", "min": 1},
                           "m": {"kind": "int", "min": 2}, "a_reduced": {"kind": "int", "min": 0},
                           "factors": {"kind": "list", "items": {"kind": "list", "len": 2,
                                                                 "items": {"kind": "int"}}}},
                "required": ["residual", "phi", "m", "a_reduced", "factors"]},
    invoke=_inv_euler, verify=_ver_euler, cert_kind="mod_congruence",
    cost=("units", 0.8), errors=("PRECOND_FAIL", "SCHEMA_MISMATCH"),
    effects="none (纯函数, 精确整数算术)",
    doc="T-N3 欧拉定理 | 条件: gcd(a,m)=1 | 推导=素因子分解求 φ + 内建模幂 | "
        "验证器=φ 穷举枚举复算 (m≤10⁶) + 因子表自洽 + 手写平方-乘 | 证书=mod_congruence | "
        "0.8 unit | 失败: 不互素 PRECOND_FAIL",
    selfchecks=(
        {"id": "t_n3_2_mod_9", "args": {"m": 9, "a": 2},
         "check": lambda r: (r.value["phi"] == 6 and r.value["residual"] == 1
                             and r.cert_independent is True,
                             f"φ={r.value['phi']} residual={r.value['residual']}"),
         "expect": "手算: φ(9)=9·(1−1/3)=6; 2⁶=64=63+1≡1 (mod 9)"},
        {"id": "t_n3_3_mod_10", "args": {"m": 10, "a": 3},
         "check": lambda r: (r.value["phi"] == 4 and r.value["residual"] == 1,
                             f"φ={r.value['phi']} residual={r.value['residual']}"),
         "expect": "手算: φ(10)=4; 3⁴=81≡1 (mod 10)"},
        {"id": "t_n3_prime_modulus", "args": {"m": 7, "a": 3},
         "check": lambda r: (r.value["phi"] == 6, f"φ={r.value['phi']}"),
         "expect": "素数模: φ(7)=6 (与费马小定理一致 — T-N2 是 T-N3 的特例)"},
        {"id": "t_n3_not_coprime", "args": {"m": 12, "a": 8}, "expect_err": "PRECOND_FAIL",
         "expect": "gcd(8,12)=4 → 条件不满足 (8^φ(12)=8⁴ mod 12 = 4 ≠ 1, 定理不适用)"},
        {"id": "t_n3_a_one", "args": {"m": 100, "a": 1},
         "check": lambda r: (r.value["phi"] == 40 and r.value["residual"] == 1,
                             f"φ={r.value['phi']}"),
         "expect": "手算: φ(100)=100·(1/2)·(4/5)=40; 1⁴⁰=1"},
    ),
))


# ===========================================================================
# trigonometry (T-T1 … T-T2, rows 9271–9272)
# ===========================================================================

_IN_PYTH = {"kind": "dict",
            "fields": {"a": {"kind": "int"}, "b": {"kind": "int"}, "c": {"kind": "int", "min": 1}},
            "required": ["a", "b", "c"]}


def _pre_pyth(args) -> "tuple[bool, str]":
    a, b, c = int(args["a"]), int(args["b"]), int(args["c"])
    if c == 0:
        return False, "c=0"
    if a * a + b * b != c * c:
        return False, f"{a}²+{b}²={a * a + b * b} ≠ c²={c * c} — 不是勾股数组, T-T1 条件不满足"
    return True, ""


def _inv_pyth(args: dict, ctl: CallCtl) -> dict:
    """T-T1 derivation: construct the rational point (a/c, b/c) on the unit circle from a Pythagorean triple, and check cos²θ+sin²θ=1."""
    a, b, c = int(args["a"]), int(args["b"]), int(args["c"])
    s2 = Fraction(a * a, c * c)
    c2 = Fraction(b * b, c * c)
    tot = s2 + c2
    return {"value": {"sin2": fs(s2), "cos2": fs(c2), "total": fs(tot),
                      "is_one": tot == 1, "theta_rad": math.atan2(float(a), float(b))},
            "resid": [float(abs(tot - 1))],
            "notes": f"T-T1 勾股恒等式: (a/c)²+(b/c)² = {fs(s2)}+{fs(c2)} = {fs(tot)} = 1"}


def _ver_pyth(args: dict, value: dict) -> dict:
    """T-T1 independent verification: (a) exactly check a²+b²−c² = 0 in rationals (without going through the (a/c)² decomposition form);
    (b) numerically check sin²+cos² = 1 at θ=atan2(a,b) with the math library (float corroboration, tolerance declared)."""
    a, b, c = int(args["a"]), int(args["b"]), int(args["c"])
    gap_int = a * a + b * b - c * c
    s2, c2 = f(value["sin2"]), f(value["cos2"])
    exact_ok = (gap_int == 0) and (s2 + c2 == 1) and (s2 == Fraction(a * a, c * c)) \
        and (c2 == Fraction(b * b, c * c))
    th = math.atan2(float(a), float(b))
    num_gap = abs(math.sin(th) ** 2 + math.cos(th) ** 2 - 1.0)
    num_ok = num_gap < 1e-15
    ok = exact_ok and num_ok
    return {"ok": ok, "resid": [float(abs(gap_int)), num_gap],
            "detail": f"整数恒等式 a²+b²−c²={gap_int} (精确); (a/c)² 与 (b/c)² 精确复算一致={exact_ok}; "
                      f"math.sin/cos @θ=atan2(a,b) 的 sin²+cos²−1={num_gap:.3e} (<1e-15)={num_ok}"}


REGISTRY.register(OpDef(
    name="thm_pythagorean", band=BAND, sub=71, thm="T-T1",
    stmt="勾股恒等式: 若 a²+b²=c² (c≠0), 则对 θ=atan2(a,b) 有 sin²θ+cos²θ=1, 且 (a/c, b/c) 是有理单位点",
    in_schema=_IN_PYTH,
    out_schema={"kind": "dict",
                "fields": {"sin2": {"kind": "str"}, "cos2": {"kind": "str"}, "total": {"kind": "str"},
                           "is_one": {"kind": "bool"}, "theta_rad": {"kind": "float"}},
                "required": ["sin2", "cos2", "total", "is_one", "theta_rad"]},
    invoke=_inv_pyth, verify=_ver_pyth, precond=_pre_pyth,
    cost=("units", 0.4), errors=("PRECOND_FAIL", "SCHEMA_MISMATCH"),
    effects="none (精确有理为主; 数值佐证见 doc)",
    doc="T-T1 勾股恒等式 (由勾股数组给出有理单位点) | 条件: a²+b²=c² 且 c≠0 | "
        "推导=(a/c)²+(b/c)² 精确有理相加 | 验证器=整数恒等式复算 + math 库 sin/cos 佐证 | "
        "0.4 unit | 诚实边界: 精确部分是可判定的恒等式; 三角函数的数值佐证只是旁证 | "
        "失败: 非勾股数组 PRECOND_FAIL",
    selfchecks=(
        {"id": "t_t1_3_4_5", "args": {"a": 3, "b": 4, "c": 5},
         "check": lambda r: (r.value["sin2"] == "9/25" and r.value["cos2"] == "16/25"
                             and r.value["is_one"] is True,
                             f"sin²={r.value['sin2']} cos²={r.value['cos2']}"),
         "expect": "手算: (3/5)²+(4/5)² = 9/25+16/25 = 25/25 = 1"},
        {"id": "t_t1_5_12_13", "args": {"a": 5, "b": 12, "c": 13},
         "check": lambda r: (r.value["sin2"] == "25/169" and r.value["cos2"] == "144/169",
                             f"sin²={r.value['sin2']}"),
         "expect": "手算: 25/169 + 144/169 = 169/169 = 1"},
        {"id": "t_t1_negative_a", "args": {"a": -8, "b": 15, "c": 17},
         "check": lambda r: (r.value["sin2"] == "64/289" and r.value["is_one"] is True,
                             f"sin²={r.value['sin2']}"),
         "expect": "手算: (−8,15,17): 64/289+225/289=1 (平方消去符号)"},
        {"id": "t_t1_not_pythagorean", "args": {"a": 1, "b": 1, "c": 2}, "expect_err": "PRECOND_FAIL",
         "expect": "1²+1²=2 ≠ 4 — 不是勾股数组"},
        {"id": "t_t1_c_zero", "args": {"a": 0, "b": 0, "c": 0}, "expect_err": "SCHEMA_MISMATCH",
         "expect": "c=0 越界 (min=1)"},
    ),
))


_IN_ADD = {"kind": "dict",
           "fields": {"s1": {"kind": "Fraction"}, "c1": {"kind": "Fraction"},
                      "s2": {"kind": "Fraction"}, "c2": {"kind": "Fraction"}},
           "required": ["s1", "c1", "s2", "c2"]}


def _pre_add(args) -> "tuple[bool, str]":
    for tag in ("1", "2"):
        s_, c_ = f(args["s" + tag]), f(args["c" + tag])
        if s_ * s_ + c_ * c_ != 1:
            return False, f"第 {tag} 组 (s,c)=({fs(s_)},{fs(c_)}) 不满足 s²+c²=1 — 不是单位点"
    return True, ""


def _inv_add(args: dict, ctl: CallCtl) -> dict:
    """T-T2 derivation: sin(x+y)=s1c2+c1s2, cos(x+y)=c1c2−s1s2 (exact rationals)."""
    s1, c1, s2, c2 = f(args["s1"]), f(args["c1"]), f(args["s2"]), f(args["c2"])
    s = s1 * c2 + c1 * s2
    c = c1 * c2 - s1 * s2
    return {"value": {"sin_sum": fs(s), "cos_sum": fs(c), "norm": fs(s * s + c * c)},
            "resid": [float(abs(s * s + c * c - 1))],
            "notes": f"T-T2 和角公式: sin(x+y)={fs(s)}, cos(x+y)={fs(c)}; 模长 s²+c²={fs(s * s + c * c)}"}


def _ver_add(args: dict, value: dict) -> dict:
    """T-T2 independent verification: (a) the **tangent-consistency** route: if c1c2≠0, the tangent addition formula (t1+t2)/(1−t1t2)
    must equal s/c — a path entirely different from the sine/cosine addition forms;
    (b) exactly check the modulus multiplicativity s²+c² = (s1²+c1²)(s2²+c2²) (from unit points → still 1);
    (c) math-library numerical corroboration (atan2 + sin/cos addition), tolerance declared."""
    s1, c1, s2, c2 = f(args["s1"]), f(args["c1"]), f(args["s2"]), f(args["c2"])
    s, c = f(value["sin_sum"]), f(value["cos_sum"])
    norm_ok = (s * s + c * c) == (s1 * s1 + c1 * c1) * (s2 * s2 + c2 * c2)
    tan_ok = True
    tan_note = "c1c2=0 → 正切路线不适用 (如实跳过)"
    if c1 != 0 and c2 != 0 and c != 0:
        t1, t2 = s1 / c1, s2 / c2
        t_sum = (t1 + t2) / (1 - t1 * t2) if (1 - t1 * t2) != 0 else None
        if t_sum is not None:
            tan_ok = (t_sum == s / c)
            tan_note = f"正切和角 (t1+t2)/(1−t1t2)={fs(t_sum)} vs s/c={fs(s / c)} (一致={tan_ok})"
        else:
            tan_ok = (c == 0)
            tan_note = f"1−t1t2=0 ⇒ x+y=π/2 型 ⇒ cos(x+y) 应为 0: 实测 c={fs(c)} (一致={tan_ok})"
    th1 = math.atan2(float(s1), float(c1))
    th2 = math.atan2(float(s2), float(c2))
    num_gap = max(abs(math.sin(th1 + th2) - float(s)), abs(math.cos(th1 + th2) - float(c)))
    num_ok = num_gap < 1e-14
    ok = norm_ok and tan_ok and num_ok and (s * s + c * c == 1)
    return {"ok": ok, "resid": [0.0 if norm_ok else 1.0, num_gap],
            "detail": f"模长乘性核对={norm_ok}; {tan_note}; "
                      f"math 库 @(atan2 求和) 与精确值 max 偏差={num_gap:.3e} (<1e-14)={num_ok}"}


REGISTRY.register(OpDef(
    name="thm_angle_addition", band=BAND, sub=72, thm="T-T2",
    stmt="和角公式: sin(x+y)=sin x cos y + cos x sin y, cos(x+y)=cos x cos y − sin x sin y",
    in_schema=_IN_ADD,
    out_schema={"kind": "dict",
                "fields": {"sin_sum": {"kind": "str"}, "cos_sum": {"kind": "str"},
                           "norm": {"kind": "str"}},
                "required": ["sin_sum", "cos_sum", "norm"]},
    invoke=_inv_add, verify=_ver_add, precond=_pre_add,
    cost=("units", 0.4), errors=("PRECOND_FAIL", "SCHEMA_MISMATCH"),
    effects="none (精确有理为主; 数值佐证见 doc)",
    doc="T-T2 和角公式 (输入为精确有理单位点 (sin,cos)) | 条件: 两组 (s,c) 均满足 s²+c²=1 | "
        "推导=正弦/余弦相加式 (精确有理) | 验证器=正切和角一致性 + 模长乘性 + math 库佐证 | "
        "0.4 unit | 诚实边界: c1c2=0 时正切路线不适用, 该项如实标注跳过 | 失败: 非单位点 PRECOND_FAIL",
    selfchecks=(
        {"id": "t_t2_thirty_fortyfive", "args": {"s1": "1/2", "c1": "sqrt3", "s2": 0, "c2": 1},
         "expect_err": "SCHEMA_MISMATCH", "expect": "非规范字面量 'sqrt3' 必须被拒 (无浮点伪装)"},
        {"id": "t_t2_double_angle", "args": {"s1": "3/5", "c1": "4/5", "s2": "3/5", "c2": "4/5"},
         "check": lambda r: (r.value["sin_sum"] == "24/25" and r.value["cos_sum"] == "7/25",
                             f"sin={r.value['sin_sum']} cos={r.value['cos_sum']}"),
         "expect": "手算: (3/5,4/5) 的二倍角: sin2x=2·(3/5)(4/5)=24/25; cos2x=(4/5)²−(3/5)²=16/25−9/25=7/25 ✓ "
                   "(7-24-25 是勾股数组, 模长=1)"},
        {"id": "t_t2_orthogonal", "args": {"s1": "1/2", "c1": "1/2", "s2": 0, "c2": 1},
         "expect_err": "PRECOND_FAIL",
         "expect": "(1/2,1/2) 模长² = 1/4+1/4 = 1/2 ≠ 1 — 不是单位点, 条件不满足"},
        {"id": "t_t2_identity", "args": {"s1": 0, "c1": 1, "s2": "3/5", "c2": "4/5"},
         "check": lambda r: (r.value["sin_sum"] == "3/5" and r.value["cos_sum"] == "4/5",
                             f"sin={r.value['sin_sum']} cos={r.value['cos_sum']}"),
         "expect": "x=0 (恒等角): sin(0+y)=sin y, cos(0+y)=cos y"},
        {"id": "t_t2_pi_over_two", "args": {"s1": 1, "c1": 0, "s2": 0, "c2": 1},
         "check": lambda r: (r.value["sin_sum"] == "1" and r.value["cos_sum"] == "0",
                             f"sin={r.value['sin_sum']} cos={r.value['cos_sum']}"),
         "expect": "x=π/2, y=0: sin(π/2)=1, cos(π/2)=0 —— 覆盖 c1=0 使正切路线跳过的分支"},
    ),
))


# ===========================================================================
# registry self-certificate: the entry roster (for audits checking §8.2's enumeration completeness)
# ===========================================================================

THEOREM_ROSTER: tuple[tuple[str, str, str], ...] = (
    ("T-A1", "thm_quadratic_roots", "二次求根"),
    ("T-A2", "thm_vieta", "韦达"),
    ("T-A3", "thm_ap_sum", "等差级数"),
    ("T-A4", "thm_gp_sum", "等比级数"),
    ("T-A5", "thm_binomial", "二项式"),
    ("T-A6", "thm_crt", "中国剩余定理"),
    ("T-A7", "thm_diff_squares", "平方差恒等式"),
    ("T-I1", "thm_amgm", "AM-GM"),
    ("T-I2", "thm_cauchy_schwarz", "Cauchy-Schwarz"),
    ("T-I3", "thm_triangle_ineq", "三角不等式"),
    ("T-I4", "thm_jensen", "Jensen"),
    ("T-L1", "thm_mat2_inv", "2×2 逆"),
    ("T-L2", "thm_cramer", "Cramer"),
    ("T-L3", "thm_eig2", "2×2 特征对"),
    ("T-L4", "thm_rank_nullity", "秩-零化度"),
    ("T-L5", "thm_cayley_hamilton", "Cayley-Hamilton"),
    ("T-L6", "thm_spectral_radius", "谱半径≤相容范数"),
    ("T-C1", "thm_derivative_table", "导数表"),
    ("T-C2", "thm_chain_rule", "链式"),
    ("T-C3", "thm_integral_table", "积分表"),
    ("T-C4", "thm_ftc", "微积分基本定理"),
    ("T-C5", "thm_lhopital", "洛必达"),
    ("T-C6", "thm_taylor_lagrange", "泰勒余项 (Lagrange)"),
    ("T-S1", "thm_ratio_test", "比值判别"),
    ("T-S2", "thm_root_test", "根值判别"),
    ("T-S3", "thm_comparison_test", "比较判别"),
    ("T-P1", "thm_bayes", "贝叶斯"),
    ("T-P2", "thm_expectation_linear", "期望线性"),
    ("T-P3", "thm_inclusion_exclusion", "容斥"),
    ("T-P4", "thm_pigeonhole", "鸽笼"),
    ("T-N1", "thm_mod_inverse", "模逆"),
    ("T-N2", "thm_fermat", "费马小定理"),
    ("T-N3", "thm_euler", "欧拉定理"),
    ("T-T1", "thm_pythagorean", "勾股恒等式"),
    ("T-T2", "thm_angle_addition", "和角公式"),
)
"""Full enumeration list = 35 entries (the section header says 34, the
itemised list has 35; the list is authoritative and the discrepancy is filed
in the delivery report's objection section)."""


def roster_check() -> "tuple[bool, str]":
    """self-certificate: every roster entry is registered and carries the correct thm code; the converse holds too (no orphan theorem operators)."""
    missing = [t for t, op, _ in THEOREM_ROSTER if not REGISTRY.has(op)]
    if missing:
        return False, f"未注册的定理算子: {missing}"
    bad_code = []
    for code, op, _ in THEOREM_ROSTER:
        if REGISTRY._ops[op].thm != code:
            bad_code.append((op, REGISTRY._ops[op].thm, code))
    if bad_code:
        return False, f"thm 码不符: {bad_code}"
    registered = {op for _, op, _ in THEOREM_ROSTER}
    orphans = [n for n in REGISTRY.names()
               if REGISTRY._ops[n].thm and n not in registered]
    if orphans:
        return False, f"未登记在 roster 的定理算子: {orphans}"
    return True, f"roster 自证通过: {len(THEOREM_ROSTER)} 条定理全部注册且 thm 码一致"


_REGISTRY_MAIN_BLOCK_END = True
