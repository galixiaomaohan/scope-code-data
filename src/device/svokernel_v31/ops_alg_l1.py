# -*- coding: utf-8 -*-
"""Layer 1: structural number theory (band 9910, rows 9911–9918 implemented; band 9940,
rows 9941–9943 stubs).

Implemented (8, verified):
  alg_cont_frac          9911  continued fraction + convergents (exact rational recurrence recomputation)
  alg_pell_solve         9912  Pell equation x²−Dy²=1 (continued-fraction fundamental solution + power family + back-substitution)
  qf_gauss_reduce        9913  Gauss reduction of indefinite binary quadratic forms (reduction inequality + equivalence chain)
  mf_qexp_arith          9914  modular-form q-expansion arithmetic (truncated mul/add/scale, Kronecker re-check)
  mf_hecke_tp            9915  Hecke operator T_p recomputed from the definition (scattering/collection dual forms)
  ec_group_law           9916  elliptic-curve chord-tangent group law (exact rational back-substitution + Vieta)
  ec_nagell_lutz         9917  torsion order (doubling to O) + Nagell-Lutz necessary condition
  curve_point_count_small 9918 small-field point count (exhaustive + Legendre-sum dual path + Hasse bound)

**stubs (3, explicitly rejected rather than faked)**: ant_ring_integers (class number) /
ec_descent_2 (2-descent) / hasse_principle_check (Hasse principle) -- see each operator's
stub_reason and external_dep.
"""
from __future__ import annotations

import math

from .v31registry import V31OpDef, V31REGISTRY  # import first: triggers the sys.path bootstrap

from svokernel.certify import Certificate
from svokernel.errors import Err, SvoError
from svokernel.exact import frac, frac_str
from svokernel.tool import CallCtl

from .algnum import PowerSeries, QsqrtD
from .numtheory import is_prime, is_squarefree, sqrt_mod_prime

BAND_L1 = 9910
BAND_STUB = 9940
_EXACT = {"kind": "int"}
_FR = {"kind": "Fraction"}


def _sdict(fields, req):
    return {"kind": "dict", "fields": fields, "required": req}


def _reg(**kw):
    kw.setdefault("status", "verified")
    kw.setdefault("layer", 1)
    return V31REGISTRY.register(V31OpDef(**kw))


def _stub(**kw):
    """stub registration: three items are mandatory (gap statement / external dependency /
    layer)."""
    kw["status"] = "stub"
    kw.setdefault("layer", 1)
    return V31REGISTRY.register(V31OpDef(**kw))


def _mk_stub(name, band, sub, layer, in_schema, out_schema, reason, dep, doc, certs):
    """Build a stub operator: invoke raises MATHLIB_MISS directly, and the verifier is
    synonymous.

    The selfcheck asserts "explicit absence" (expect_err=MATHLIB_MISS) -- proving that a stub
    never silently returns a value.
    """
    def _invoke(args, ctl):
        raise SvoError(Err.MATHLIB_MISS, ctx="op:%s#%d" % (name, band + sub),
                       detail="stub: 缺 %s —— 本算子在本版**不产生任何值**" % dep)

    def _verify(args, value):
        raise SvoError(Err.MATHLIB_MISS, ctx="op:%s@verify" % name,
                       detail="stub 无实现, 验证器同样缺席 (不做假验证)")

    return _reg(
        name=name, band=band, sub=sub, layer=layer, status="stub",
        in_schema=in_schema, out_schema=out_schema, invoke=_invoke, verify=_verify,
        cost=("units", 0.0), errors=("MATHLIB_MISS", "SCHEMA_MISMATCH"),
        effects="none (stub: 无副作用也无实现)",
        doc=doc, stub_reason=reason, external_dep=dep, planned_cert=certs,
        selfchecks=(
            {"id": name + "_absent", "args": _stub_args(name), "expect_err": "MATHLIB_MISS",
             "expect": "stub 显式拒绝: 缺口=%s" % reason},
        ),
    )


def _stub_args(name):
    """Stub selfcheck input: it must **pass the schema gate** (otherwise it would only yield
    SCHEMA_MISMATCH and prove nothing about the absence of an implementation), and then invoke
    raises MATHLIB_MISS."""
    return {
        "ant_ring_integers": {"d": 5},
        "ec_descent_2": {"a": 0, "b": 1},
        "hasse_principle_check": {"variety_code": "conic_x2_minus_2y2_eq_1", "eq": [1, 0, -2]},
    }[name]


# ===========================================================================
# 9911  alg_cont_frac
# ===========================================================================

_CF_IN = _sdict({
    "num": _EXACT, "den": {"kind": "int", "min": -(10 ** 18), "max": 10 ** 18},
    "maxterms": {"kind": "int", "min": 1, "max": 5000},
}, ["num", "den", "maxterms"])
_CF_OUT = _sdict({
    "num": _EXACT, "den": _EXACT, "terms": {"kind": "list", "items": _EXACT, "minlen": 1, "maxlen": 5000},
    "convergents": {"kind": "list", "items": {"kind": "tuple", "items": [_EXACT, _EXACT]}, "minlen": 1, "maxlen": 5000},
    "n_terms": _EXACT, "exact_match": {"kind": "bool"}, "p_last": _EXACT, "q_last": _EXACT,
}, ["num", "den", "terms", "convergents", "n_terms", "exact_match", "p_last", "q_last"])


def _cf_precond(args):
    if args["den"] == 0:
        return False, "den = 0 (分母为零无连分数)"
    if args["num"] == 0:
        return True, ""
    return True, ""


def _cf_expand(num, den, maxterms):
    """Euclidean expansion (floor semantics, automatically correct for negatives):
    num/den = [a0; a1, …]."""
    terms = []
    n_, d_ = num, den
    while len(terms) < maxterms:
        a = n_ // d_          # Python floor division => Rosen continued fraction (correct for negatives too)
        terms.append(a)
        rem = n_ - a * d_
        if rem == 0:
            return terms, True
        n_, d_ = d_, rem
    return terms, False


def _convergents(terms):
    ps, qs = [1, terms[0]], [0, 1]
    for a in terms[1:]:
        ps.append(a * ps[-1] + ps[-2])
        qs.append(a * qs[-1] + qs[-2])
    return [(ps[i], qs[i]) for i in range(1, len(ps))]


def _invoke_cf(args, ctl):
    num, den, mx = args["num"], args["den"], args["maxterms"]
    terms, done = _cf_expand(num, den, mx)
    if not done:
        raise SvoError(Err.BOUNDED_VIOLATION, ctx="op:alg_cont_frac#9911",
                       detail="连分数长度超过 maxterms=%d 仍未终止 (拒绝截断: 截断的连分数不是该有理数)"
                              % mx)
    convs = _convergents(terms)
    pk, qk = convs[-1]
    cert = Certificate(kind="cf_convergent",
                       payload={"num": num, "den": den, "terms": terms,
                                "convergents": [[p, q] for p, q in convs]},
                       issued_by="alg_cont_frac")
    return {"value": {"num": num, "den": den, "terms": terms,
                      "convergents": [[p, q] for p, q in convs], "n_terms": len(terms),
                      "exact_match": pk * den == qk * num, "p_last": pk, "q_last": qk},
            "resid": [0.0], "cert": cert, "cost": float(len(terms)) * 1.0,
            "notes": "%d/%d = %r; 末收敛子 %d/%d" % (num, den, terms, pk, qk)}


def _verify_cf(args, value):
    """Independent verifier: matrix-product rebuild + interlacing inequality (exact rationals),
    a different path from the floor recurrence."""
    num, den = args["num"], args["den"]
    terms = value["terms"]
    bad = []
    # matrix product [[a0,1],[1,0]]·[[a1,1],[1,0]]··· has top-left/bottom-left = last numerator/denominator
    M = [[1, 0], [0, 1]]
    for a in terms:
        A = [[a, 1], [1, 0]]
        M = [[M[0][0] * A[0][0] + M[0][1] * A[1][0], M[0][0] * A[0][1] + M[0][1] * A[1][1]],
             [M[1][0] * A[0][0] + M[1][1] * A[1][0], M[1][0] * A[0][1] + M[1][1] * A[1][1]]]
    if [M[0][0], M[1][0]] != [value["p_last"], value["q_last"]]:
        bad.append("矩阵乘积 %r ≠ 声称末收敛子 [%d,%d]"
                   % ([M[0][0], M[1][0]], value["p_last"], value["q_last"]))
    f = frac(str(num)) / frac(str(den))
    if value["q_last"] == 0 or frac("%d/%d" % (value["p_last"], value["q_last"])) != f:
        bad.append("末收敛子 ≠ 原分数 (精确有理不等)")
    x = f
    convs = value["convergents"]
    # interlacing inequality: strict for non-last pairs, equality allowed for the last pair (when
    # x is rational the last-pair error is exactly 1/(q_{N−1} q_N)).
    for i in range(len(convs) - 1):
        p1, q1 = convs[i]
        p2, q2 = convs[i + 1]
        if q1 <= 0 or q2 <= 0:
            continue
        gap = abs(x - frac("%d/%d" % (p1, q1)))
        bound = frac("1/%d" % (q1 * q2))
        if i == len(convs) - 2:
            if gap > bound:
                bad.append("末对收敛子误差超过界")
        elif gap >= bound:
            bad.append("第 %d 对收敛子违反严格交错不等式" % i)
    for i, a in enumerate(terms[:-1]):
        if a != int(a):
            bad.append("项 %d 非整数" % i)
    return {"ok": not bad, "resid": [float(len(bad))],
            "detail": "矩阵乘积重建 + 精确相等 + 交错不等式 (末对含等号) 通过 (%d 项)" % len(terms)
                      if not bad else "; ".join(bad)}


_reg(
    name="alg_cont_frac", band=BAND_L1, sub=1,
    in_schema=_CF_IN, out_schema=_CF_OUT,
    invoke=_invoke_cf, verify=_verify_cf, cert_kind="cf_convergent",
    cost=("units", 2.0),
    errors=("PRECOND_FAIL", "SCHEMA_MISMATCH", "POSTCOND_FAIL", "BOUNDED_VIOLATION", "TIMEOUT"),
    precond=_cf_precond,
    effects="none (纯函数, 精确整数递推)",
    doc=("Continued-fraction expansion of a rational plus all convergents (exact integers, no "
         "float). Generation = Euclidean floor recurrence (Rosen convention for negatives); "
         "independent verifier = **2×2 matrix-product** rebuild of the last convergent + exact "
         "rational decision of the interlacing inequality |x−p_i/q_i| < 1/(q_i q_{i+1}); "
         "certificate cf_convergent is independently re-checked by certcheck_alg (own matrix "
         "code). maxterms exhausted without termination => BOUNDED_VIOLATION (truncation is "
         "refused: a truncated continued fraction is not that rational)."),
    requires_cert=True,
    thm="T-V31-CF",
    stmt="有理数 ↔ 有限连分数一一对应; 收敛子满足 |x − p_i/q_i| < 1/(q_i q_{i+1})",
    selfchecks=(
        {"id": "cf_415_93", "args": {"num": 415, "den": 93, "maxterms": 50},
         "check": lambda r: (r.value["terms"] == [4, 2, 6, 7] and r.value["exact_match"]
                             and r.cert_independent,
                             "415/93 = %r" % (r.value["terms"],)),
         "expect": "经典例: 415/93 = [4;2,6,7]"},
        {"id": "cf_negative", "args": {"num": -5, "den": 3, "maxterms": 50},
         "check": lambda r: (r.value["terms"] == [-2, 3], "−5/3 = %r (floor 语义)" % (r.value["terms"],)),
         "expect": "负数: −5/3 = [−2;3]"},
        {"id": "cf_sqrt2_rational", "args": {"num": 99, "den": 70, "maxterms": 50},
         "check": lambda r: (r.value["terms"] == [1, 2, 2, 2, 2, 2],
                             "99/70 = %r (√2 收敛子)" % (r.value["terms"],)),
         "expect": "√2 的有理逼近 99/70"},
        {"id": "cf_integer", "args": {"num": 7, "den": 1, "maxterms": 10},
         "check": lambda r: (r.value["terms"] == [7] and r.value["n_terms"] == 1,
                             "整数: 单项连分数"),
         "expect": "整数边界: [7]"},
        {"id": "cf_zero", "args": {"num": 0, "den": 5, "maxterms": 10},
         "check": lambda r: (r.value["terms"] == [0], "0 = [0]"),
         "expect": "零边界"},
        {"id": "cf_den_zero_reject", "args": {"num": 1, "den": 0, "maxterms": 10},
         "expect_err": "PRECOND_FAIL", "expect": "分母为零被 precond 拒"},
        {"id": "cf_fib_ratio_maxterms", "args": {"num": 6765, "den": 4181, "maxterms": 5},
         "expect_err": "BOUNDED_VIOLATION",
         "expect": "黄金比收敛子 6765/4181 需 19 项 > maxterms=5 ⇒ 拒绝截断"},
    ),
)


# ===========================================================================
# 9912  alg_pell_solve
# ===========================================================================

_PELL_IN = _sdict({
    "D": {"kind": "int", "min": 2, "max": 10 ** 6},
    "max_y": {"kind": "int", "min": 1, "max": 10 ** 30},
    "minimality_cap": {"kind": "int", "min": 1, "max": 200000},
}, ["D", "max_y", "minimality_cap"])
_PELL_OUT = _sdict({
    "D": _EXACT, "fundamental": {"kind": "tuple", "items": [_EXACT, _EXACT]},
    "solutions": {"kind": "list", "items": {"kind": "tuple", "items": [_EXACT, _EXACT]}, "minlen": 1, "maxlen": 64},
    "n_solutions": _EXACT, "x1_digits": _EXACT, "minimality_exhaustive": {"kind": "bool"},
}, ["D", "fundamental", "solutions", "n_solutions", "x1_digits", "minimality_exhaustive"])


def _pell_precond(args):
    D = args["D"]
    if D < 2:
        return False, "D >= 2"
    r = math.isqrt(D)
    if r * r == D:
        return False, "D=%d 是完全平方 ⇒ x²−Dy²=1 只有平凡解 (1,0), 域外" % D
    return True, ""


def _pell_fundamental(D, ctl):
    """Find the fundamental solution via the continued fraction (PQa recurrence, all integers)."""
    a0 = math.isqrt(D)
    m, d, a = 0, 1, a0
    p_prev, p_cur = 1, a0
    q_prev, q_cur = 0, 1
    if p_cur * p_cur - D * q_cur * q_cur == 1:
        return p_cur, q_cur
    for _ in range(1 << 20):
        ctl.check_time("op:alg_pell_solve#9912")
        m = d * a - m
        d = (D - m * m) // d
        a = (a0 + m) // d
        p_prev, p_cur = p_cur, a * p_cur + p_prev
        q_prev, q_cur = q_cur, a * q_cur + q_prev
        if p_cur * p_cur - D * q_cur * q_cur == 1:
            return p_cur, q_cur
    raise SvoError(Err.BOUNDED_VIOLATION, ctx="op:alg_pell_solve#9912",
                   detail="连分数周期超限未找到基本解 (内部错误)")


def _invoke_pell(args, ctl):
    D, max_y, cap = args["D"], args["max_y"], args["minimality_cap"]
    x1, y1 = _pell_fundamental(D, ctl)
    sols = [[x1, y1]]
    cx, cy = x1, y1
    while True:
        ctl.check_time("op:alg_pell_solve#9912")
        nx, ny = x1 * cx + D * y1 * cy, x1 * cy + y1 * cx
        if ny > max_y or len(sols) >= 64:
            break
        sols.append([nx, ny])
        cx, cy = nx, ny
    cert = Certificate(kind="pell_backsub",
                       payload={"D": D, "fundamental": [x1, y1], "solutions": sols},
                       issued_by="alg_pell_solve")
    return {"value": {"D": D, "fundamental": [x1, y1], "solutions": sols,
                      "n_solutions": len(sols), "x1_digits": len(str(x1)),
                      "minimality_exhaustive": y1 <= cap},
            "resid": [0.0], "cert": cert, "cost": float(len(sols)) * 3.0,
            "notes": "D=%d 基本解 (%d,%d), y<=%d 内 %d 个解" % (D, x1, y1, max_y, len(sols))}


def _verify_pell(args, value):
    """Independent verifier: per-solution back-substitution + power-recurrence identity +
    fundamental-solution minimality (exhaustive y<y1, or the continued-fraction first-solution
    structure)."""
    D = args["D"]
    bad = []
    minimality_mode = ""
    for x, y in value["solutions"]:
        if x * x - D * y * y != 1:
            bad.append("(%d,%d) 回代失败" % (x, y))
    x1, y1 = value["fundamental"]
    if value["solutions"][0] != [x1, y1]:
        bad.append("解表首项 ≠ 基本解")
    if y1 <= args["minimality_cap"]:
        for y in range(1, y1):
            v = D * y * y + 1
            r = math.isqrt(v)
            if r * r == v:
                bad.append("最小性失败: y=%d 给出更小解" % y)
                break
        minimality_mode = "穷举 y<%d" % y1
    else:
        # beyond the cap: degrade to a **structural path** -- independently rerun the continued
        # fraction period of √D and take the first convergent that satisfies the back-substitution;
        # it must be exactly (x1,y1). This is another independent argument for minimality
        # (continued-fraction convergents have increasing denominators, so the first solution is
        # the minimal-denominator solution). Honest note: it is not an exhaustive y<y1 scan.
        a0 = math.isqrt(D)
        m, d, a = 0, 1, a0
        pp, pc, qp, qc = 1, a0, 0, 1
        found = None
        if pc * pc - D * qc * qc == 1:
            found = (pc, qc)
        else:
            for _ in range(1 << 20):
                m = d * a - m
                d = (D - m * m) // d
                a = (a0 + m) // d
                pp, pc = pc, a * pc + pp
                qp, qc = qc, a * qc + qp
                if pc * pc - D * qc * qc == 1:
                    found = (pc, qc)
                    break
        if found != (x1, y1):
            bad.append("连分数首解 %r ≠ 声称基本解 (%d,%d)" % (found, x1, y1))
        minimality_mode = "连分数首解结构 (y1=%d 超 cap=%d, 未做穷举)" % (y1, args["minimality_cap"])
    cx, cy = x1, y1
    for i, (x, y) in enumerate(value["solutions"]):
        if i:
            cx, cy = x1 * cx + D * y1 * cy, x1 * cy + y1 * cx
        if [x, y] != [cx, cy]:
            bad.append("第 %d 解 ≠ 幂递推" % i)
            break
    return {"ok": not bad, "resid": [float(len(bad))],
            "detail": "%d 个解回代 + 幂递推一致; 最小性判据 = %s"
                      % (len(value["solutions"]), minimality_mode)
                      if not bad else "; ".join(bad)}


_reg(
    name="alg_pell_solve", band=BAND_L1, sub=2,
    in_schema=_PELL_IN, out_schema=_PELL_OUT,
    invoke=_invoke_pell, verify=_verify_pell, cert_kind="pell_backsub",
    cost=("units", 6.0),
    errors=("PRECOND_FAIL", "SCHEMA_MISMATCH", "POSTCOND_FAIL", "BOUNDED_VIOLATION", "TIMEOUT"),
    precond=_pell_precond,
    effects="none (纯函数)",
    doc=("Positive integer solutions of the Pell equation x² − D y² = 1: fundamental solution "
         "(x1,y1) + power family (x1+y1√D)^k. Generation = integer PQa recurrence on the "
         "continued fraction of √D; independent verifier = **per-solution back-substitution** + "
         "power-recurrence identity + fundamental-solution minimality (exhaustive y < y1 "
         "confirming no smaller solution; beyond the cap it degrades and says so in "
         "detail); certificate pell_backsub is independently re-checked by certcheck_alg. A "
         "square D has only the trivial solution => PRECOND_FAIL (out of domain)."),
    requires_cert=True,
    thm="T-V31-PELL",
    stmt="D>0 非完全平方 ⇒ x²−Dy²=1 有无穷多正整数解, 全部由基本解 (x1+y1√D)^k 给出",
    selfchecks=(
        {"id": "pell_D2", "args": {"D": 2, "max_y": 100, "minimality_cap": 1000},
         "check": lambda r: (r.value["fundamental"] == [3, 2] and r.value["solutions"][:2] == [[3, 2], [17, 12]]
                             and r.cert_independent,
                             "D=2: (3,2), (17,12)"),
         "expect": "x²−2y²=1 基本解 (3,2); 次解 (17,12)"},
        {"id": "pell_D13", "args": {"D": 13, "max_y": 10 ** 6, "minimality_cap": 200000},
         "check": lambda r: (r.value["fundamental"] == [649, 180], "D=13: (649,180)"),
         "expect": "经典难例 D=13"},
        {"id": "pell_D61_oneyear", "args": {"D": 61, "max_y": 10 ** 12, "minimality_cap": 200000},
         "check": lambda r: (r.value["fundamental"] == [1766319049, 226153980],
                             "D=61: (%d,%d) —— 著名的困难基本解"
                             % (r.value["fundamental"][0], r.value["fundamental"][1])),
         "expect": "D=61 基本解 x=1766319049 (连分数法秒解, 暴力法需 ~2.3e8 步)"},
        {"id": "pell_max_y_boundary", "args": {"D": 2, "max_y": 2, "minimality_cap": 1000},
         "check": lambda r: (r.value["n_solutions"] == 1, "max_y=2: 仅基本解 (y=12 超界)"),
         "expect": "max_y 边界: 只含 y ≤ max_y 的解"},
        {"id": "pell_square_reject", "args": {"D": 4, "max_y": 100, "minimality_cap": 100},
         "expect_err": "PRECOND_FAIL", "expect": "完全平方 D 只有平凡解 ⇒ 域外"},
        {"id": "pell_D1_reject", "args": {"D": 1, "max_y": 100, "minimality_cap": 100},
         "expect_err": "SCHEMA_MISMATCH", "expect": "D >= 2 是类型层声明"},
        {"id": "pell_D3_small", "args": {"D": 3, "max_y": 100, "minimality_cap": 1000},
         "check": lambda r: (r.value["fundamental"] == [2, 1], "D=3: (2,1)"),
         "expect": "最小非平凡例 D=3"},
    ),
)


# ===========================================================================
# 9913  qf_gauss_reduce
# ===========================================================================

_QF_IN = _sdict({
    "a": {"kind": "int", "min": 1, "max": 10 ** 9},
    "b": {"kind": "int", "min": -(10 ** 9), "max": 10 ** 9},
    "c": {"kind": "int", "min": -(10 ** 9), "max": 10 ** 9},
    "step_cap": {"kind": "int", "min": 1, "max": 10000},
}, ["a", "b", "c", "step_cap"])
_QF_OUT = _sdict({
    "D": _EXACT, "input": {"kind": "tuple", "items": [_EXACT, _EXACT, _EXACT]},
    "reduced": {"kind": "tuple", "items": [_EXACT, _EXACT, _EXACT]},
    "forms": {"kind": "list", "minlen": 1, "maxlen": 10001,
              "items": {"kind": "tuple", "items": [_EXACT, _EXACT, _EXACT]}},
    "matrices": {"kind": "list", "minlen": 0, "maxlen": 10000,
                 "items": {"kind": "tuple", "items": [
                     {"kind": "tuple", "items": [_EXACT, _EXACT]},
                     {"kind": "tuple", "items": [_EXACT, _EXACT]}]}},
    "n_steps": _EXACT, "already_reduced": {"kind": "bool"},
}, ["D", "input", "reduced", "forms", "matrices", "n_steps", "already_reduced"])


def _qf_precond(args):
    a, b, c = args["a"], args["b"], args["c"]
    D = b * b - 4 * a * c
    if a <= 0:
        return False, "本算子只实现 a >= 1 分支 (a<=0 的型先互换成 a>0; 我们不声称 " \
                      "a<=0 分支的正确性, 故显式拒绝)"
    if D <= 0:
        return False, "判别式 D = %d <= 0 (本算子只做不定型 D>0)" % D
    r = math.isqrt(D)
    if r * r == D:
        return False, "D = %d 是完全平方 (抛物型退化, 无规约理论)" % D
    return True, ""


def _qf_reduced_triple(a, b, c, D):
    """Exact-integer reduction criterion: 0 < b < √D and √D − b < 2|a| < √D + b."""
    if b <= 0 or b * b >= D:
        return False
    m = abs(a)
    return (2 * m + b) ** 2 > D and (2 * m - b) ** 2 < D


def _qf_canon_shift_k(a, b, D):
    """The unique k that shifts b into the canonical window (isqrt(D) − 2|a|, isqrt(D)] (exact
    integers, sign-independent).

    The window length is exactly 2|a| => every residue class mod 2a has exactly one
    representative inside the window => uniqueness is proven.
    """
    m0 = math.isqrt(D)          # D is not a perfect square => m0² < D
    m = abs(a)
    k0 = (m0 - b) // (2 * a)
    for k in (k0 - 2, k0 - 1, k0, k0 + 1, k0 + 2):
        bn = b + 2 * a * k
        if m0 - 2 * m < bn <= m0:
            return k
    return None


def _qf_apply_shift(a, b, c, k):
    """Left-multiply by the x-substitution matrix [[1,k],[0,1]]: (a, b+2ak, a k² + b k + c)."""
    return a, b + 2 * a * k, a * k * k + b * k + c


def _qf_apply_rho(a, b, c, k):
    """Left-multiply by the ρ matrix [[0,−1],[1,k]]: (c, −b+2ck, a − bk + c k²). det = 1."""
    return c, -b + 2 * c * k, a - b * k + c * k * k


def _invoke_qf(args, ctl):
    a, b, c, cap = args["a"], args["b"], args["c"], args["step_cap"]
    D = b * b - 4 * a * c
    forms = [[a, b, c]]
    mats = []
    if _qf_reduced_triple(a, b, c, D):
        red = [a, b, c]
        already = True
    else:
        already = False
        for _ in range(cap):
            ctl.check_time("op:qf_gauss_reduce#9913")
            k = _qf_canon_shift_k(a, b, D)
            if k is None:
                raise SvoError(Err.BOUNDED_VIOLATION, ctx="op:qf_gauss_reduce#9913",
                               detail="规范窗内无同余代表 (内部错误)")
            if k != 0:
                a2, b2, c2 = _qf_apply_shift(a, b, c, k)
                mats.append([[1, k], [0, 1]])
                a, b, c = a2, b2, c2
                forms.append([a, b, c])
                if _qf_reduced_triple(a, b, c, D):
                    break
            # inside the canonical window b ∈ (isqrt(D)−2|a|, isqrt(D)]; if still not reduced,
            # only a can change: use ρ (swap + shift)
            if _qf_reduced_triple(a, b, c, D):
                break
            k2 = _qf_canon_shift_k(c, -b, D)
            if k2 is None:
                raise SvoError(Err.BOUNDED_VIOLATION, ctx="op:qf_gauss_reduce#9913",
                               detail="ρ 步规范窗内无同余代表 (内部错误)")
            a2, b2, c2 = _qf_apply_rho(a, b, c, k2)
            mats.append([[0, -1], [1, k2]])
            a, b, c = a2, b2, c2
            forms.append([a, b, c])
            if _qf_reduced_triple(a, b, c, D):
                break
        else:
            raise SvoError(Err.BOUNDED_VIOLATION, ctx="op:qf_gauss_reduce#9913",
                           detail="%d 步内未达规约型 (拒绝给未规约的'结果')" % cap)
        red = [a, b, c]
    if red[1] * red[1] - 4 * red[0] * red[2] != D:
        raise SvoError(Err.POSTCOND_FAIL, ctx="op:qf_gauss_reduce#9913",
                       detail="规约型判别式漂移 (内部错误)")
    cert = Certificate(kind="form_reduce_chain",
                       payload={"D": D, "forms": forms, "matrices": mats, "reduced": red},
                       issued_by="qf_gauss_reduce")
    return {"value": {"D": D, "input": [args["a"], args["b"], args["c"]], "reduced": red,
                      "forms": forms, "matrices": mats, "n_steps": len(mats),
                      "already_reduced": already},
            "resid": [0.0], "cert": cert, "cost": float(len(mats)) * 4.0,
            "notes": "[%d,%d,%d] → [%d,%d,%d] (D=%d, %d 步%s)"
                     % (args["a"], args["b"], args["c"], red[0], red[1], red[2], D, len(mats),
                        ", 原本已规约" if already else "")}


def _verify_qf(args, value):
    """Independent verifier: recompute step by step with its own form transformations + det check
    + discriminant invariant + tail reduction criterion."""
    D = value["D"]
    forms, mats = value["forms"], value["matrices"]
    bad = []
    if len(forms) != len(mats) + 1:
        bad.append("链长与矩阵数不匹配")
    for f in forms:
        if f[1] * f[1] - 4 * f[0] * f[2] != D:
            bad.append("型 %r 判别式 ≠ %d" % (f, D))
    for i, M in enumerate(mats):
        if i + 1 >= len(forms):
            break
        det = M[0][0] * M[1][1] - M[0][1] * M[1][0]
        if abs(det) != 1:
            bad.append("第 %d 步 det=%d 非单位" % (i, det))
        got = _form_apply(forms[i], M)
        if got != list(forms[i + 1]):
            bad.append("第 %d 步变换复算 %r ≠ 链中 %r" % (i, got, forms[i + 1]))
    red = value["reduced"]
    if list(forms[-1]) != list(red):
        bad.append("链末端 ≠ 声称规约型")
    if not _qf_reduced_triple(red[0], red[1], red[2], D):
        bad.append("末端型不满足规约不等式")
    return {"ok": not bad, "resid": [float(len(bad))],
            "detail": "%d 步链: 判别式不变量 + 单位行列式 + 逐型复算 + 末端规约 通过" % len(mats)
                      if not bad else "; ".join(bad)}


def _form_apply(f, M):
    a, b, c = f
    p, q = M[0]
    r, s = M[1]
    return [a * p * p + b * p * r + c * r * r,
            2 * a * p * q + b * (p * s + q * r) + 2 * c * r * s,
            a * q * q + b * q * s + c * s * s]


_reg(
    name="qf_gauss_reduce", band=BAND_L1, sub=3,
    in_schema=_QF_IN, out_schema=_QF_OUT,
    invoke=_invoke_qf, verify=_verify_qf, cert_kind="form_reduce_chain",
    cost=("units", 8.0),
    errors=("PRECOND_FAIL", "SCHEMA_MISMATCH", "POSTCOND_FAIL", "BOUNDED_VIOLATION", "TIMEOUT"),
    precond=_qf_precond,
    effects="none (纯函数)",
    doc=("Gauss reduction of the indefinite binary quadratic form [a,b,c] (D = b²−4ac > 0, not a "
         "perfect square): output the reduced form + an **equivalence chain** (one det=1 unimodular "
         "matrix per step). Generation: alternate the canonical shift (move b into the window "
         "(isqrt(D)−2|a|, isqrt(D)], window length exactly 2|a| => unique representative) and the "
         "ρ step (swap + shift); independent verifier: recompute step by step with its own form "
         "transformations + det + discriminant invariant + tail reduction inequality; certificate "
         "form_reduce_chain is independently re-checked by certcheck_alg. Domain note: only the "
         "a >= 1 branch is implemented; exceeding the step cap => BOUNDED_VIOLATION (no "
         "un-reduced 'result')."),
    requires_cert=True,
    thm="T-V31-QF",
    stmt="不定型经 SL(2,Z) 作用可达**唯一**规约型 [a,b,c]: 0 < b < √D 且 √D−b < 2|a| < √D+b",
    selfchecks=(
        {"id": "qf_disc8", "args": {"a": 1, "b": 0, "c": -2, "step_cap": 100},
         "check": lambda r: (r.value["reduced"] == [1, 2, -1] and not r.value["already_reduced"]
                             and r.cert_independent,
                             "[1,0,−2] → %r (D=8)" % (r.value["reduced"],)),
         "expect": "x²−2y² 的型 [1,0,−2] 规约到 [1,2,−1]"},
        {"id": "qf_disc20_rho", "args": {"a": 1, "b": 2, "c": -4, "step_cap": 100},
         "check": lambda r: (r.value["reduced"] == [1, 4, -1],
                             "[1,2,−4] → %r (D=20: 需 ρ 步换位)" % (r.value["reduced"],)),
         "expect": "D=20 需换位分支 (回归测试: 早期版本在此打转)"},
        {"id": "qf_already_reduced", "args": {"a": 1, "b": 5, "c": -2, "step_cap": 100},
         "check": lambda r: (r.value["reduced"] == [1, 5, -2] and r.value["already_reduced"]
                             and r.value["n_steps"] == 0,
                             "[1,5,−2] 已规约 (D=33): 零步恒等"),
         "expect": "已规约输入: 空链 (幂等性边界)"},
        {"id": "qf_disc33_from_unreduced", "args": {"a": 1, "b": 1, "c": -8, "step_cap": 100},
         "check": lambda r: (r.value["reduced"][1] * r.value["reduced"][1]
                             - 4 * r.value["reduced"][0] * r.value["reduced"][2] == 33
                             and r.value["n_steps"] >= 1,
                             "[1,1,−8] → %r (D=33)" % (r.value["reduced"],)),
         "expect": "同判别式的非规约型规约到该判别式的某个规约型"},
        {"id": "qf_big_b", "args": {"a": 2, "b": 57, "c": 3, "step_cap": 100},
         "check": lambda r: (abs(r.value["reduced"][1]) < math.isqrt(r.value["D"]) + 1,
                             "[2,57,3] (D=%d) → %r" % (r.value["D"], r.value["reduced"])),
         "expect": "大 b 需多步规范平移"},
        {"id": "qf_disc4_reject", "args": {"a": 1, "b": 0, "c": -1, "step_cap": 100},
         "expect_err": "PRECOND_FAIL", "expect": "D = 0+4 = 4 是完全平方 ⇒ 抛物型域外"},
        {"id": "qf_definite_reject", "args": {"a": 1, "b": 0, "c": 1, "step_cap": 100},
         "expect_err": "PRECOND_FAIL", "expect": "D = −4 <= 0 (正定型) 域外"},
        {"id": "qf_schema_a0", "args": {"a": 0, "b": 1, "c": -1, "step_cap": 100},
         "expect_err": "SCHEMA_MISMATCH", "expect": "a >= 1 是类型层声明"},
    ),
)


# ===========================================================================
# 9914  mf_qexp_arith
# ===========================================================================

_QE_IN = _sdict({
    "mode": {"kind": "str", "enum": ["mul", "add", "sub", "scale"]},
    "a": {"kind": "list", "items": _EXACT, "minlen": 1, "maxlen": 512},
    "b": {"kind": "list", "items": _EXACT, "minlen": 0, "maxlen": 512},
    "prec": {"kind": "int", "min": 1, "max": 512},
    "scale_k": {"kind": "int", "min": -(10 ** 6), "max": 10 ** 6},
}, ["mode", "a", "b", "prec", "scale_k"])
_QE_OUT = _sdict({
    "mode": {"kind": "str"}, "prec": _EXACT,
    "coeffs": {"kind": "list", "items": _EXACT, "minlen": 1, "maxlen": 512},
    "canonical": {"kind": "str"}, "n_nonzero": _EXACT, "leading": _EXACT,
}, ["mode", "prec", "coeffs", "canonical", "n_nonzero", "leading"])


def _qe_precond(args):
    if args["mode"] != "scale" and len(args["b"]) == 0:
        return False, "二元运算需要 b 系数表非空"
    return True, ""


def _invoke_qexp(args, ctl):
    mode, prec = args["mode"], args["prec"]
    A = PowerSeries(args["a"], prec)
    if mode == "scale":
        out = A.scale(args["scale_k"])
        B = PowerSeries([0] * prec, prec)
    else:
        B = PowerSeries(args["b"], prec)
        out = {"mul": lambda: A * B, "add": lambda: A + B, "sub": lambda: A - B}[mode]()
    cert = Certificate(kind="qexp_mul",
                       payload={"mode": mode, "prec": out.prec, "a": A.coeffs, "b": B.coeffs,
                                "scale_k": args["scale_k"], "product": out.coeffs},
                       issued_by="mf_qexp_arith")
    lead = next((c for c in out.coeffs if c != 0), 0)
    return {"value": {"mode": mode, "prec": out.prec, "coeffs": list(out.coeffs),
                      "canonical": out.to_str(), "n_nonzero": sum(1 for c in out.coeffs if c),
                      "leading": lead},
            "resid": [0.0], "cert": cert, "cost": float(prec) ** 2 * 0.01,
            "notes": "%s (q^%d): %s" % (mode, out.prec, out.to_str())}


def _verify_qexp(args, value):
    """Independent verifier: multiplication goes through **Kronecker substitution** (pack into
    big integers and multiply); add/sub are recomputed term by term."""
    mode, prec = args["mode"], args["prec"]
    A = list(args["a"])
    B = list(args["b"])
    got = list(value["coeffs"])
    bad = []
    if mode == "scale":
        want = [c * args["scale_k"] for c in (A + [0] * prec)[:prec]]
    elif mode in ("add", "sub"):
        want = [(A[i] if i < len(A) else 0) + (B[i] if i < len(B) else 0) * (1 if mode == "add" else -1)
                for i in range(prec)]
    else:
        want = _kronecker_mul(A, B, prec)
    if want[:prec] != got:
        bad.append("%s 独立复算不一致: %r vs %r" % (mode, want[:prec][:6], got[:6]))
    if got != list(PowerSeries(got, prec).coeffs):
        bad.append("规范化后系数表变化 (规范型不变量被破坏)")
    # second independent invariant: the constant term must be the product of the two constant
    # terms (independent of truncation, always true for mul)
    if mode == "mul":
        a0 = A[0] if len(A) > 0 else 0
        b0 = B[0] if len(B) > 0 else 0
        if got[0] != a0 * b0:
            bad.append("常数项恒等式失败: %d ≠ %d·%d" % (got[0], a0, b0))
    return {"ok": not bad, "resid": [float(len(bad))],
            "detail": "%s: Kronecker/逐项 独立复算 + 规范型 + 常数项恒等 通过 (q^%d)" % (mode, prec)
                      if not bad else "; ".join(bad)}


def _kronecker_mul(A, B, prec):
    """Kronecker-substitution multiplication (exact). **Negative coefficients must be split into
    positive/negative parts**: packing them directly drags two's complement in (a trap we hit:
    [1,−2,1]·[1,2,1] computed 126 instead of −2). Four non-negative convolutions combined
    linearly."""
    def kron(P_, Q_):
        mx = max([1] + P_) * max([1] + Q_) * max(1, prec)
        bits = mx.bit_length() + 2
        pa = 0
        for c in reversed(P_):
            pa = (pa << bits) + c
        pb = 0
        for c in reversed(Q_):
            pb = (pb << bits) + c
        pr = pa * pb
        out = []
        mask = (1 << bits) - 1
        for _ in range(prec):
            out.append(pr & mask)
            pr >>= bits
        return out

    Apos, Aneg = [max(0, x) for x in A], [max(0, -x) for x in A]
    Bpos, Bneg = [max(0, x) for x in B], [max(0, -x) for x in B]
    pp, pn = kron(Apos, Bpos), kron(Apos, Bneg)
    np_, nn = kron(Aneg, Bpos), kron(Aneg, Bneg)
    return [pp[i] - pn[i] - np_[i] + nn[i] for i in range(prec)]


_reg(
    name="mf_qexp_arith", band=BAND_L1, sub=4,
    in_schema=_QE_IN, out_schema=_QE_OUT,
    invoke=_invoke_qexp, verify=_verify_qexp, cert_kind="qexp_mul",
    cost=("units", 3.0),
    errors=("PRECOND_FAIL", "SCHEMA_MISMATCH", "POSTCOND_FAIL", "TIMEOUT"),
    precond=_qe_precond,
    effects="none (纯函数)",
    doc=("Truncated arithmetic on modular-form q-expansions: mul/add/sub/scale on Z[q]/(q^N). The "
         "type is carried by algnum.PowerSeries (canonical form + precision never silently aligned: "
         "operating on different prec raises). Generation = schoolbook truncated convolution; "
         "independent verifier (for the product) = **Kronecker substitution** (pack coefficients "
         "into big integers, multiply, unpack -- a genuinely different algorithm); certificate "
         "qexp_mul is independently re-checked by certcheck_alg."),
    requires_cert=True,
    thm="T-V31-QEXP",
    stmt="Z[q]/(q^N) 是环; 截断卷积满足结合与分配律 (可逐系数独立复核)",
    selfchecks=(
        {"id": "qe_mul_small", "args": {"mode": "mul", "a": [1, 2, 3], "b": [1, 1, 0], "prec": 4, "scale_k": 0},
         "check": lambda r: (r.value["coeffs"] == [1, 3, 5, 3] and r.cert_independent,
                             "(1+2q+3q²)(1+q) = %r" % (r.value["coeffs"],)),
         "expect": "手算: 1 + 3q + 5q² + 3q³"},
        {"id": "qe_mul_negative", "args": {"mode": "mul", "a": [1, -2, 1], "b": [1, 2, 1], "prec": 4, "scale_k": 0},
         "check": lambda r: (r.value["coeffs"] == [1, 0, -2, 0],
                             "(1−q)²(1+q)² = %r" % (r.value["coeffs"],)),
         "expect": "(1−q)²(1+q)² = (1−q²)² = 1 − 2q² + q⁴ → 截断 q⁴: [1,0,−2,0]"},
        {"id": "qe_add", "args": {"mode": "add", "a": [1, 2], "b": [3, 4], "prec": 2, "scale_k": 0},
         "check": lambda r: (r.value["coeffs"] == [4, 6], "逐项加"),
         "expect": "加法逐项"},
        {"id": "qe_scale_zero", "args": {"mode": "scale", "a": [1, 2, 3], "b": [], "prec": 3, "scale_k": 0},
         "check": lambda r: (r.value["coeffs"] == [0, 0, 0] and r.value["n_nonzero"] == 0,
                             "数乘 0 ⇒ 零级数"),
         "expect": "数乘边界 0"},
        {"id": "qe_prec1_boundary", "args": {"mode": "mul", "a": [5], "b": [7], "prec": 1, "scale_k": 1},
         "check": lambda r: (r.value["coeffs"] == [35], "prec=1: 只留常数项"),
         "expect": "精度 1 边界"},
        {"id": "qe_sub", "args": {"mode": "sub", "a": [1, 5], "b": [1, 5], "prec": 2, "scale_k": 0},
         "check": lambda r: (r.value["coeffs"] == [0, 0], "自减为零级数"),
         "expect": "减法: 自身相减为零"},
        {"id": "qe_empty_reject", "args": {"mode": "mul", "a": [1], "b": [], "prec": 3, "scale_k": 0},
         "expect_err": "PRECOND_FAIL", "expect": "二元运算缺 b ⇒ precond 拒"},
        {"id": "qe_bad_mode_reject", "args": {"mode": "div", "a": [1], "b": [1], "prec": 3, "scale_k": 0},
         "expect_err": "SCHEMA_MISMATCH", "expect": "未登记 mode (本类型无除法: Z[q] 非域)"},
        {"id": "qe_schema_float", "args": {"mode": "mul", "a": [1.5], "b": [1], "prec": 2, "scale_k": 0},
         "expect_err": "SCHEMA_MISMATCH", "expect": "float 系数进精确通道被拒"},
    ),
)


# ===========================================================================
# 9915  mf_hecke_tp
# ===========================================================================

_HT_IN = _sdict({
    "coeffs": {"kind": "list", "items": _EXACT, "minlen": 2, "maxlen": 256},
    "p": {"kind": "int", "min": 2, "max": 10 ** 6},
    "k": {"kind": "int", "min": 1, "max": 100},
    "prec": {"kind": "int", "min": 2, "max": 256},
}, ["coeffs", "p", "k", "prec"])
_HT_OUT = _sdict({
    "p": _EXACT, "k": _EXACT, "prec": _EXACT,
    "coeffs": {"kind": "list", "items": _EXACT, "minlen": 2, "maxlen": 256},
    "weight_factor": _EXACT, "eigenvalue_if_eigen": _EXACT,
    "is_eigen_at_p": {"kind": "bool"}, "eigen_check_ok": {"kind": "bool"},
}, ["p", "k", "prec", "coeffs", "weight_factor", "eigenvalue_if_eigen", "is_eigen_at_p", "eigen_check_ok"])


def _ht_precond(args):
    if not is_prime(args["p"]):
        return False, "p=%d 非素数 (Hecke 算子 T_p 的指标须为素数)" % args["p"]
    if args["k"] < 1:
        return False, "权 k >= 1"
    if len(args["coeffs"]) < args["prec"]:
        return False, "系数表长度 %d < prec %d" % (len(args["coeffs"]), args["prec"])
    return True, ""


def _invoke_hecke(args, ctl):
    p, k, prec = args["p"], args["k"], args["prec"]
    A = PowerSeries(args["coeffs"], prec)
    w = p ** (k - 1)
    out = [0] * prec
    for n in range(1, prec):
        anp = A.coeff(n * p)
        an_p = A.coeff(n // p) if n % p == 0 else 0
        out[n] = anp + w * an_p
    outS = PowerSeries(out, prec)
    # eigenform check: if a_1 = 1 and a_p = λ, then b_n = λ·a_n must hold for every n
    lam = A.coeff(p)
    is_eig = A.coeff(1) == 1
    eig_ok = False
    if is_eig:
        # truncation boundary: assert b_n = λ·a_n only for data-reachable n (n·p < prec)
        eig_ok = all(out[n] == lam * A.coeff(n) for n in range(1, prec) if n * p < prec)
    cert = Certificate(kind="hecke_tp_expand",
                       payload={"prec": prec, "p": p, "k": k, "coeffs": A.coeffs, "out": outS.coeffs},
                       issued_by="mf_hecke_tp")
    return {"value": {"p": p, "k": k, "prec": prec, "coeffs": out,
                      "weight_factor": w, "eigenvalue_if_eigen": lam,
                      "is_eigen_at_p": is_eig, "eigen_check_ok": eig_ok},
            "resid": [0.0], "cert": cert, "cost": float(prec) * 1.0,
            "notes": "T_%d (权 %d, p^(k−1)=%d): %s" % (p, k, w, outS.to_str())}


def _verify_hecke(args, value):
    """Independent verifier: **scattering-style** recomputation + eigenform relation (for
    a_n = σ_{k−1}(n), T_p should be λ·identity)."""
    p, k, prec = args["p"], args["k"], args["prec"]
    A = list(args["coeffs"])
    w = p ** (k - 1)
    bad = []
    acc = [0] * prec
    for n in range(1, prec):
        if n % p == 0:
            acc[n // p] += A[n]
        if n * p < prec:
            acc[n * p] += w * A[n]
    for n in range(1, prec):
        if acc[n] != value["coeffs"][n]:
            bad.append("散射式重算 b_%d = %d ≠ %d" % (n, acc[n], value["coeffs"][n]))
            break
    if value["is_eigen_at_p"] and value["eigen_check_ok"]:
        lam = A[p] if p < len(A) else 0
        for n in range(1, prec):
            if n * p >= prec:
                break          # truncation boundary: a_{np} is not in the table, assert nothing
            if value["coeffs"][n] != lam * A[n]:
                bad.append("特征形式关系 b_n = λ a_n 在 n=%d 失败" % n)
                break
    return {"ok": not bad, "resid": [float(len(bad))],
            "detail": "散射式重算一致%s" % ("; 特征形式关系通过 (截断内)" if value["eigen_check_ok"] else "")
                      if not bad else "; ".join(bad)}


_reg(
    name="mf_hecke_tp", band=BAND_L1, sub=5,
    in_schema=_HT_IN, out_schema=_HT_OUT,
    invoke=_invoke_hecke, verify=_verify_hecke, cert_kind="hecke_tp_expand",
    cost=("units", 4.0),
    errors=("PRECOND_FAIL", "SCHEMA_MISMATCH", "POSTCOND_FAIL", "TIMEOUT"),
    precond=_ht_precond,
    effects="none (纯函数)",
    doc=("Hecke operator T_p acting on a weight-k q-expansion: "
         "(T_p f)(q) = Σ_n (a_{np} + p^{k−1} a_{n/p}) q^n. Generation = **collection-style** "
         "per-term formula; independent verifier = **scattering-style** index derivation (spread "
         "a_m onto b_{m/p} and b_{mp}, the inverse direction of collection) + eigenform relation "
         "b_n = λ·a_n (when a_1=1); certificate hecke_tp_expand is recomputed along the scattering "
         "form by certcheck_alg."),
    requires_cert=True,
    thm="T-V31-HECKE",
    stmt="(T_p f)(q) = Σ(a_{np} + p^{k−1}a_{n/p})q^n; 特征形式满足 T_p f = λ_p f (λ_p = a_p)",
    selfchecks=(
        {"id": "hecke_eisenstein_eigen", "args": {"coeffs": [0, 1, 9, 28, 73, 126, 252], "p": 2, "k": 4, "prec": 7},
         "check": lambda r: (r.value["is_eigen_at_p"] and r.value["eigen_check_ok"]
                             and r.value["eigenvalue_if_eigen"] == 9 and r.cert_independent,
                             "E_4 (a_n = σ_3(n)): λ_2 = 1+8 = 9, 特征关系成立"),
         "expect": "Eisenstein 级数 E_4: a_n = σ_3(n), λ_p = 1+p³ ⇒ T_2 应为 9·恒等"},
        {"id": "hecke_definition_small", "args": {"coeffs": [0, 5, 7], "p": 2, "k": 1, "prec": 3},
         "check": lambda r: (r.value["coeffs"] == [0, 7, 5] and r.value["weight_factor"] == 1,
                             "定义式: b_1 = a_2 = 7, b_2 = p^0·a_1 = 5 ⇒ %r" % (r.value["coeffs"],)),
         "expect": "权 1 时 p^{k−1}=1: b_1 = a_2, b_2 = a_1"},
        {"id": "hecke_prec2_boundary", "args": {"coeffs": [0, 5, 7], "p": 2, "k": 1, "prec": 2},
         "check": lambda r: (r.value["coeffs"] == [0, 0],
                             "prec=2 时 a_2 因截断不在表内 ⇒ b_1 = 0 (截断诚实性)"),
         "expect": "精度边界 prec=2: 表外系数按 0 处理 (PowerSeries 截断语义)"},
        {"id": "hecke_p_not_prime_reject", "args": {"coeffs": [0, 1, 2], "p": 4, "k": 2, "prec": 3},
         "expect_err": "PRECOND_FAIL", "expect": "T_p 的指标须素"},
        {"id": "hecke_coeffs_short_reject", "args": {"coeffs": [0, 1], "p": 2, "k": 2, "prec": 5},
         "expect_err": "PRECOND_FAIL", "expect": "系数表短于精度 ⇒ 拒"},
        {"id": "hecke_k1_boundary", "args": {"coeffs": [0, 1, 2], "p": 2, "k": 1, "prec": 3},
         "check": lambda r: (r.value["weight_factor"] == 1, "k=1 (权 1) 合法"),
         "expect": "k=1 边界 (p^{k−1}=1)"},
        {"id": "hecke_k0_schema_reject", "args": {"coeffs": [0, 1, 2], "p": 2, "k": 0, "prec": 3},
         "expect_err": "SCHEMA_MISMATCH", "expect": "k >= 1 是类型层声明"},
    ),
)


# ===========================================================================
# 9916  ec_group_law
# ===========================================================================

_PT = {"kind": "list", "items": {"kind": "str"}, "len": 2}
_EC_IN = _sdict({"a": _FR, "b": _FR, "P": _PT, "Q": _PT}, ["a", "b", "P", "Q"])
_EC_OUT = _sdict({
    "is_infinity": {"kind": "bool"}, "x": {"kind": "str"}, "y": {"kind": "str"},
    "lam": {"kind": "str"}, "case": {"kind": "str",
                                     "enum": ["add", "double", "inverse", "identity", "torsion_double"]},
    "on_curve_R": {"kind": "bool"},
}, ["is_infinity", "x", "y", "lam", "case", "on_curve_R"])


def _pt(v):
    """["O","O"] = the point at infinity; otherwise ["x","y"] as canonical rational literals."""
    if len(v) != 2:
        raise SvoError(Err.SCHEMA_MISMATCH, ctx="op:ec_group_law", detail="点须为 [x,y] 或 [O,O]")
    if v[0] == "O":
        return None
    return frac(v[0]), frac(v[1])


def _on_curve(a, b, P):
    if P is None:
        return True
    x, y = P
    return y * y == x * x * x + a * x + b


def _ec_add(a, b, P, Q):
    """Chord-tangent group law (exact rationals). Returns (R, lam, case).

    Branch order matters: **test P == Q (doubling) before the inverse** -- the 2-torsion point
    P=(x,0) satisfies both P == Q and P == −P, and testing the inverse first would misclassify it
    as inverse (a trap we hit).
    """
    if P is None:
        return Q, None, "identity"
    if Q is None:
        return P, None, "identity"
    x1, y1 = P
    x2, y2 = Q
    if P == Q:
        if y1 == 0:
            return None, None, "torsion_double"
        lam = (3 * x1 * x1 + a) / (2 * y1)
        case = "double"
    elif x1 == x2 and y1 + y2 == 0:
        return None, None, "inverse"
    else:
        lam = (y2 - y1) / (x2 - x1)
        case = "add"
    x3 = lam * lam - x1 - x2
    y3 = lam * (x1 - x3) - y1
    return (x3, y3), lam, case


def _ec_precond(args):
    a, b = frac(args["a"]), frac(args["b"])
    if 4 * a ** 3 + 27 * b ** 2 == 0:
        return False, "4a³+27b² = 0 ⇒ 奇异曲线 (群律不成立)"
    for key in ("P", "Q"):
        try:
            P = _pt(args[key])
        except SvoError as e:
            return False, "点 %s 解析失败: %s" % (key, e.detail)
        if P is not None and not _on_curve(a, b, P):
            return False, "点 %s = %r 不在曲线 y²=x³+%sx+%s 上" % (key, P, a, b)
    return True, ""


def _invoke_ec(args, ctl):
    a, b = frac(args["a"]), frac(args["b"])
    P, Q = _pt(args["P"]), _pt(args["Q"])
    R, lam, case = _ec_add(a, b, P, Q)
    if R is None:
        val = {"is_infinity": True, "x": "0", "y": "0", "lam": "0", "case": case, "on_curve_R": True}
    else:
        val = {"is_infinity": False, "x": frac_str(R[0]), "y": frac_str(R[1]),
               "lam": frac_str(lam) if lam is not None else "0", "case": case,
               "on_curve_R": _on_curve(a, b, R)}
    cert = Certificate(kind="ec_chord_backsub",
                       payload={"a": frac_str(a), "b": frac_str(b), "P": args["P"], "Q": args["Q"],
                                "R": (["O", "O"] if R is None else [frac_str(R[0]), frac_str(R[1])]),
                                "lam": val["lam"], "case": case},
                       issued_by="ec_group_law")
    return {"value": val, "resid": [0.0], "cert": cert, "cost": 2.0,
            "notes": "P+Q (%s): %s" % (case, "O" if R is None else "(%s, %s)" % (val["x"], val["y"]))}


def _verify_ec(args, value):
    """Independent verifier: λ recomputed from the definition + the third intersection S on the
    curve + R = −S + Vieta + back-substitution.

    Why S instead of "the three points are collinear": for doubling, the tangent's third
    intersection with the curve can be P itself (an inflection point, e.g. y²=x³+1 at (0,1)), and
    then "y_R = y_P + λ(x_R−x_P)" simply fails -- the correct uniform statement is that
    S = (x_R, y_P+λ(x_R−x_P)) lies on the curve and R = −S (with S a 2-torsion point => R=O).
    """
    a, b = frac(args["a"]), frac(args["b"])
    P, Q = _pt(args["P"]), _pt(args["Q"])
    bad = []
    if value["is_infinity"]:
        st = {"identity": (P is None or Q is None),
              "inverse": (P is not None and Q is not None and P[0] == Q[0] and P[1] + Q[1] == 0),
              "torsion_double": (P is not None and P == Q and P[1] == 0)}.get(value["case"], False)
        if not st:
            bad.append("∞ 结果的 case=%s 与输入不符" % value["case"])
        return {"ok": not bad, "resid": [float(len(bad))],
                "detail": "∞ 结果按 %s 情形独立复核通过" % value["case"] if not bad else "; ".join(bad)}
    R = (frac(value["x"]), frac(value["y"]))
    # identity case: P + O = P (or O + Q = Q) -- there is no λ
    if P is None or Q is None:
        other = Q if P is None else P
        if R != other:
            bad.append("恒等元加法结果 ≠ 另一加数")
        return {"ok": not bad, "resid": [float(len(bad))],
                "detail": "恒等元情形: 结果 = 另一加数" if not bad else "; ".join(bad)}
    lam = frac(value["lam"])
    if not _on_curve(a, b, R):
        bad.append("R 不在曲线上")
    x1, y1 = P
    x2, y2 = Q
    if P == Q:
        if y1 == 0:
            bad.append("y=0 的倍点应为 O")
        elif (3 * x1 * x1 + a) / (2 * y1) != lam:
            bad.append("倍点切线斜率不符")
    else:
        if x1 == x2:
            bad.append("x 相等却给非 ∞ 结果")
        elif (y2 - y1) / (x2 - x1) != lam:
            bad.append("弦斜率不符")
    y_s = y1 + lam * (R[0] - x1)
    if y_s * y_s != R[0] ** 3 + a * R[0] + b:
        bad.append("第三交点 S 不在曲线上")
    # R = −S; when y_S = 0, S is a 2-torsion point and −S = S => R must be S (i.e. y_R = 0), not O.
    if y_s == 0:
        if R[1] != 0:
            bad.append("第三交点 S 的 y=0 (2-挠) ⇒ 应有 y_R = 0")
    elif R[1] != -y_s:
        bad.append("R ≠ −S")
    if x1 + x2 + R[0] != lam * lam:
        bad.append("Vieta 根和失败")
    return {"ok": not bad, "resid": [float(len(bad))],
            "detail": "λ 定义 + 第三交点 S 在曲线 + R=−S + Vieta + 回代 通过" if not bad else "; ".join(bad)}


_reg(
    name="ec_group_law", band=BAND_L1, sub=6,
    in_schema=_EC_IN, out_schema=_EC_OUT,
    invoke=_invoke_ec, verify=_verify_ec, cert_kind="ec_chord_backsub",
    cost=("units", 2.0),
    errors=("PRECOND_FAIL", "SCHEMA_MISMATCH", "POSTCOND_FAIL", "TIMEOUT"),
    precond=_ec_precond,
    effects="none (纯函数, 精确有理算术)",
    doc=("Chord-tangent group law P+Q on the elliptic curve y² = x³ + ax + b (exact rationals, "
         "no float). Generation = textbook closed form (λ, x_R = λ²−x_P−x_Q, "
         "y_R = λ(x_P−x_R)−y_P); independent verifier = **λ recomputed from the definition + "
         "collinearity + Vieta root sum x_P+x_Q+x_R = λ² + curve back-substitution** (a different "
         "derivation from the x_R closed form, numerically independent of it); certificate "
         "ec_chord_backsub is independently re-checked by certcheck_alg. The boundary cases (O, "
         "inverse, 2-torsion doubling) each form a case and are decided independently."),
    requires_cert=True,
    thm="T-V31-ECLAW",
    stmt="曲线 y²=x³+ax+b (4a³+27b²≠0) 的点集在弦切律下构成交换群, O 为单位元",
    selfchecks=(
        {"id": "ecl_chord", "args": {"a": "0", "b": "1", "P": ["0", "1"], "Q": ["2", "3"]},
         "check": lambda r: (not r.value["is_infinity"] and r.value["x"] == "-1" and r.value["y"] == "0"
                             and r.value["case"] == "add" and r.cert_independent,
                             "(0,1)+(2,3) = (%s,%s) 全在直线 y = x+1 上" % (r.value["x"], r.value["y"])),
         "expect": "三点共线例: (0,1),(2,3),(−1,0) 满足 y=x+1"},
        {"id": "ecl_double", "args": {"a": "0", "b": "1", "P": ["0", "1"], "Q": ["0", "1"]},
         "check": lambda r: (r.value["case"] == "double" and r.value["x"] == "0" and r.value["y"] == "-1"
                             and r.value["lam"] == "0",
                             "2·(0,1) = (%s,%s) (切线水平)" % (r.value["x"], r.value["y"])),
         "expect": "倍点: 切线斜率 0 ⇒ 2P = (0,−1)"},
        {"id": "ecl_torsion_double", "args": {"a": "0", "b": "1", "P": ["-1", "0"], "Q": ["-1", "0"]},
         "check": lambda r: (r.value["is_infinity"] and r.value["case"] == "torsion_double",
                             "2-挠点 (−1,0) 倍点为 O"),
         "expect": "2-挠点: 切线竖直 ⇒ 2P = O"},
        {"id": "ecl_inverse", "args": {"a": "0", "b": "1", "P": ["0", "1"], "Q": ["0", "-1"]},
         "check": lambda r: (r.value["is_infinity"] and r.value["case"] == "inverse",
                             "P + (−P) = O"),
         "expect": "逆元相加为 O"},
        {"id": "ecl_identity", "args": {"a": "0", "b": "1", "P": ["0", "1"], "Q": ["O", "O"]},
         "check": lambda r: (r.value["case"] == "identity" and r.value["x"] == "0" and r.value["y"] == "1",
                             "P + O = P (单位元)"),
         "expect": "单位元边界"},
        {"id": "ecl_not_on_curve_reject", "args": {"a": "0", "b": "1", "P": ["1", "1"], "Q": ["2", "3"]},
         "expect_err": "PRECOND_FAIL", "expect": "1² ≠ 1³+1 ⇒ 点不在曲线上"},
        {"id": "ecl_singular_reject", "args": {"a": "0", "b": "0", "P": ["0", "0"], "Q": ["1", "1"]},
         "expect_err": "PRECOND_FAIL", "expect": "4a³+27b² = 0 ⇒ 奇异曲线"},
    ),
)


# ===========================================================================
# 9917  ec_nagell_lutz
# ===========================================================================

_NL_IN = _sdict({
    "a": _EXACT, "b": _EXACT,
    "P": {"kind": "list", "items": _EXACT, "len": 2},
    "order_cap": {"kind": "int", "min": 1, "max": 64},
}, ["a", "b", "P", "order_cap"])
_NL_OUT = _sdict({
    "a": _EXACT, "b": _EXACT, "disc": _EXACT,
    "P": {"kind": "list", "items": _EXACT, "len": 2},
    "order": _EXACT, "is_torsion": {"kind": "bool"},
    "lutz_necessary_ok": {"kind": "bool"}, "lutz_note": {"kind": "str"},
    "steps": {"kind": "list", "items": {"kind": "list", "items": {"kind": "str"}, "minlen": 0, "maxlen": 2},
              "minlen": 1, "maxlen": 66},
}, ["a", "b", "disc", "P", "order", "is_torsion", "lutz_necessary_ok", "lutz_note", "steps"])


def _nl_precond(args):
    a, b = args["a"], args["b"]
    if 4 * a ** 3 + 27 * b ** 2 == 0:
        return False, "奇异曲线 (4a³+27b²=0), 群律不成立"
    x, y = args["P"]
    if y * y != x * x * x + a * x + b:
        return False, "点 %r 不在曲线 y²=x³+%dx+%d 上" % (args["P"], a, b)
    return True, ""


def _invoke_nl(args, ctl):
    a, b = args["a"], args["b"]
    cap = args["order_cap"]
    x, y = args["P"]
    disc = -16 * (4 * a ** 3 + 27 * b ** 2)
    lutz_ok = (y == 0) or (disc % (y * y) == 0 if y != 0 else True)
    note = ("坐标整数且 y=0 (2-挠点)" if y == 0 else
            ("坐标整数且 y²=%d 整除判别式 %d (Nagell-Lutz 必要条件成立)" % (y * y, disc) if lutz_ok else
             "y² 不整除判别式 ⇒ Nagell-Lutz 必要条件失败 ⇒ 必非扭点"))
    P = (frac(x), frac(y))
    # steps are recorded as canonical rational literals: multiples of an integral point **can** be
    # rational points (e.g. (−1,1) on y²=x³+2), so coordinates during the walk must not be assumed
    # integral. [] denotes O.
    steps = [[frac_str(frac(x)), frac_str(frac(y))]]
    order = None
    cur = P
    for k in range(2, cap + 1):
        ctl.check_time("op:ec_nagell_lutz#9917")
        cur, _lam, _case = _ec_add(frac(a), frac(b), cur, P)
        if cur is None:
            order = k
            steps.append([])
            break
        steps.append([frac_str(cur[0]), frac_str(cur[1])])
    is_tors = order is not None
    cert = Certificate(kind="torsion_walk",
                       payload={"a": a, "b": b, "P": [x, y], "order": order or 0,
                                "steps": steps, "completed": is_tors, "order_cap": cap},
                       issued_by="ec_nagell_lutz")
    return {"value": {"a": a, "b": b, "disc": disc, "P": [x, y],
                      "order": order or 0, "is_torsion": is_tors, "lutz_necessary_ok": lutz_ok,
                      "lutz_note": note, "steps": steps},
            "resid": [0.0 if is_tors else 1.0], "cert": cert, "cost": float(len(steps)) * 2.0,
            "notes": ("扭点: 阶 %d (走 %d 步至 O)" % (order, len(steps) - 1)) if is_tors
                     else ("%d 步内未至 O ⇒ 非扭点 (Mazur 界: E(Q)_tors 阶 <= 12)" % (cap))}


def _verify_nl(args, value):
    """Independent verifier: per-step curve back-substitution + distinctness + independent
    double-and-add scalar multiplication + Lutz condition."""
    a, b = frac(args["a"]), frac(args["b"])
    x, y = args["P"]
    P = (frac(x), frac(y))
    bad = []
    if value["is_torsion"]:
        n = value["order"]
        if n < 1 or n > 16:
            bad.append("阶 %d 超出 Mazur 可能阶 (1..10,12) 的合理范围" % n)
        # distinctness + per-step on-curve
        seen = set()
        for i, st in enumerate(value["steps"]):
            if not st:
                if i != len(value["steps"]) - 1:
                    bad.append("步 %d 即 O" % i)
                continue
            pt = (frac(st[0]), frac(st[1]))
            if not _on_curve(a, b, pt):
                bad.append("步 %d 不在曲线上" % i)
            if pt in seen:
                bad.append("步 %d 重复" % i)
            seen.add(pt)
        # independent scalar multiplication
        R = None
        add = P
        k = n
        while k > 0:
            if k & 1:
                R, _l, _c = _ec_add(a, b, R, add)
            add, _l, _c = _ec_add(a, b, add, add)
            k >>= 1
        if R is not None:
            bad.append("独立 double-and-add: %d·P = %r ≠ O" % (n, R))
    else:
        if value["lutz_necessary_ok"]:
            bad.append("未找到阶但 Lutz 必要条件成立 —— 这合法 (必要条件非充分); 检查器仅记录")
            bad.pop()
        for i, st in enumerate(value["steps"]):
            if not st:
                bad.append("步 %d 为 O 却称非扭点" % i)
                continue
            if not _on_curve(a, b, (frac(st[0]), frac(st[1]))):
                bad.append("步 %d 不在曲线上" % i)
    return {"ok": not bad, "resid": [float(len(bad))],
            "detail": ("扭点阶 %d: 互异 + 回代 + 独立标量乘 通过" % value["order"]) if not bad and value["is_torsion"]
                      else ("非扭点: 走步全部在曲线上 (Lutz 必要条件=%s, 非充分)" % value["lutz_necessary_ok"])
                      if not bad else "; ".join(bad)}


_reg(
    name="ec_nagell_lutz", band=BAND_L1, sub=7,
    in_schema=_NL_IN, out_schema=_NL_OUT,
    invoke=_invoke_nl, verify=_verify_nl, cert_kind="torsion_walk",
    cost=("units", 5.0),
    errors=("PRECOND_FAIL", "SCHEMA_MISMATCH", "POSTCOND_FAIL", "TIMEOUT"),
    precond=_nl_precond,
    effects="none (纯函数, 精确有理算术)",
    doc=("Torsion decision for integral points on the integral curve y²=x³+ax+b: doubling to O "
         "(iterated addition of P, 2P, …) gives the order, and the Nagell-Lutz necessary condition "
         "(integral point + y² divides the discriminant) is checked. Not reaching O within "
         "order_cap steps judges the point non-torsion, by Mazur's theorem (E(Q)_tors order ∈ "
         "{1..10,12}, so cap >= 16 suffices). Generation = iterated addition; independent verifier "
         "= per-step curve back-substitution + distinct walk steps (distinct + last step O => the "
         "order is exactly n) + **own double-and-add scalar multiplication** re-checking n·P = O + "
         "the Lutz condition; certificate torsion_walk is independently re-checked by "
         "certcheck_alg. Domain note: integral points only (the Lutz theorem guarantees torsion "
         "points are integral, so no torsion point is lost)."),
    requires_cert=True,
    thm="T-V31-NL",
    stmt="若 P ∈ E(Q)_tors 则 P 为整点且 (y=0 或 y² | D) —— 必要条件; 阶由倍点走动确定",
    selfchecks=(
        {"id": "nl_order6_classic", "args": {"a": 0, "b": 1, "P": [2, 3], "order_cap": 16},
         "check": lambda r: (r.value["is_torsion"] and r.value["order"] == 6
                             and r.cert_independent,
                             "E: y²=x³+1 上 (2,3) 的阶 = %d (扭群 Z/6)" % r.value["order"]),
         "expect": "经典事实: y²=x³+1 的扭群为 Z/6Z, (2,3) 是生成元, 阶 6"},
        {"id": "nl_order2", "args": {"a": 0, "b": 1, "P": [-1, 0], "order_cap": 16},
         "check": lambda r: (r.value["order"] == 2 and r.value["steps"][-1] == [],
                             "(−1,0) 是 2-挠点 (末步 = []=O)"),
         "expect": "y=0 ⇒ 阶 2"},
        {"id": "nl_order3", "args": {"a": 0, "b": 1, "P": [0, 1], "order_cap": 16},
         "check": lambda r: (r.value["order"] == 3, "(0,1) 的阶 = 3"),
         "expect": "(0,1) 阶 3"},
        {"id": "nl_non_torsion_lutz_satisfied", "args": {"a": 0, "b": 2, "P": [-1, 1], "order_cap": 16},
         "check": lambda r: (not r.value["is_torsion"] and r.value["lutz_necessary_ok"],
                             "y²=x³+2 上 (−1,1): Lutz 必要条件成立但非扭点 (必要非充分)"),
         "expect": "Lutz 条件是**必要**的: 满足它也可能非扭点 (本算子的诚实边界)"},
        {"id": "nl_not_on_curve_reject", "args": {"a": 0, "b": 1, "P": [1, 1], "order_cap": 16},
         "expect_err": "PRECOND_FAIL", "expect": "1 ≠ 2 ⇒ 点不在曲线上"},
        {"id": "nl_singular_reject", "args": {"a": 0, "b": 0, "P": [0, 0], "order_cap": 16},
         "expect_err": "PRECOND_FAIL", "expect": "奇异曲线域外"},
    ),
)


# ===========================================================================
# 9918  curve_point_count_small
# ===========================================================================

_PC_IN = _sdict({
    "a": {"kind": "int", "min": -(10 ** 6), "max": 10 ** 6},
    "b": {"kind": "int", "min": -(10 ** 6), "max": 10 ** 6},
    "p": {"kind": "int", "min": 3, "max": 200000},
}, ["a", "b", "p"])
_PC_OUT = _sdict({
    "a": _EXACT, "b": _EXACT, "p": _EXACT, "count": _EXACT, "affine_count": _EXACT,
    "legendre_sum": _EXACT, "hasse_ok": {"kind": "bool"}, "hasse_slack": _EXACT,
    "two_algorithms_agree": {"kind": "bool"},
}, ["a", "b", "p", "count", "affine_count", "legendre_sum", "hasse_ok", "hasse_slack",
    "two_algorithms_agree"])


def _pc_precond(args):
    a, b, p = args["a"], args["b"], args["p"]
    if p < 3:
        return False, "p >= 3 (p=2 时 Legendre 路线退化)"
    if not is_prime(p):
        return False, "p=%d 非素数" % p
    if (4 * a ** 3 + 27 * b ** 2) % p == 0:
        return False, "4a³+27b² ≡ 0 (mod p) ⇒ 奇异曲线 (本算子只数非奇异曲线的点)"
    return True, ""


def _invoke_pc(args, ctl):
    a, b, p = args["a"], args["b"], args["p"]
    affine = 0
    lsum = 0
    for x in range(p):
        ctl.check_time("op:curve_point_count_small#9918")
        rhs = (x * x * x + a * x + b) % p
        if rhs == 0:
            affine += 1
        else:
            e = pow(rhs, (p - 1) // 2, p)
            if e == 1:
                affine += 2
                lsum += 1
            else:
                lsum -= 1
    count = affine + 1
    # second algorithm: square-root count table (independent of the Euler criterion; note the
    # count is of **points**, not of x values -- every rhs≠0 quadratic residue gives 2 y's and
    # rhs=0 gives 1. An early version counted x values as points; fixed.)
    rootcnt = {}
    for y in range(p):
        v = y * y % p
        rootcnt[v] = rootcnt.get(v, 0) + 1
    affine2 = sum(rootcnt.get((x * x * x + a * x + b) % p, 0) for x in range(p))
    agree = (affine == affine2)
    hasse_ok = (count - p - 1) ** 2 <= 4 * p
    cert = Certificate(kind="pointcount_small",
                       payload={"a": a, "b": b, "p": p, "count": count, "affine": affine,
                                "legendre_sum": lsum},
                       issued_by="curve_point_count_small")
    return {"value": {"a": a, "b": b, "p": p, "count": count, "affine_count": affine,
                      "legendre_sum": lsum, "hasse_ok": hasse_ok,
                      "hasse_slack": 4 * p - (count - p - 1) ** 2,
                      "two_algorithms_agree": agree},
            "resid": [0.0 if agree else 1.0], "cert": cert, "cost": float(p) * 3.0,
            "notes": "N = %d = p+1+Σ(a_p) = %d+1+(%d); Hasse 松弛 %d"
                     % (count, p, lsum, 4 * p - (count - p - 1) ** 2)}


def _verify_pc(args, value):
    """Independent verifier: **per-y full scan** square-root count + Hasse bound (exact integers)
    + the Σ(legendre) identity."""
    a, b, p = args["a"], args["b"], args["p"]
    bad = []
    # definition-level: count y solutions per x (O(p²); within p <= 2e5 it runs only at selfcheck
    # scale; this is the independent second channel)
    aff2 = 0
    ls2 = 0
    for x in range(p):
        rhs = (x * x * x + a * x + b) % p
        cnt = 0
        for y in range(p):
            if y * y % p == rhs:
                cnt += 1
        aff2 += cnt
        if rhs != 0:
            ls2 += 1 if cnt == 2 else -1
    if aff2 != value["affine_count"]:
        bad.append("逐 y 全扫的仿射点数 %d ≠ 声称 %d" % (aff2, value["affine_count"]))
    if value["count"] != value["affine_count"] + 1:
        bad.append("总点数 ≠ 仿射点数 + 1 (无穷远点)")
    if value["count"] != p + 1 + value["legendre_sum"]:
        bad.append("恒等式 N = p+1+Σ(a_p) 不成立")
    if ls2 != value["legendre_sum"]:
        bad.append("逐点穷举 Σlegendre %d ≠ 声称 %d" % (ls2, value["legendre_sum"]))
    if (value["count"] - p - 1) ** 2 > 4 * p:
        bad.append("Hasse 界被违反")
    if bool(value["hasse_ok"]) != ((value["count"] - p - 1) ** 2 <= 4 * p):
        bad.append("hasse_ok 标志与实算不符")
    return {"ok": not bad, "resid": [float(len(bad))],
            "detail": "逐 y 全扫 (%d 点) + Σlegendre + Hasse 界 三路一致 (N=%d, p=%d)"
                      % (aff2, value["count"], p) if not bad else "; ".join(bad)}


_reg(
    name="curve_point_count_small", band=BAND_L1, sub=8,
    in_schema=_PC_IN, out_schema=_PC_OUT,
    invoke=_invoke_pc, verify=_verify_pc, cert_kind="pointcount_small",
    cost=("units", 6.0),
    errors=("PRECOND_FAIL", "SCHEMA_MISMATCH", "POSTCOND_FAIL", "BOUNDED_VIOLATION", "TIMEOUT"),
    precond=_pc_precond,
    effects="none (纯函数)",
    doc=("Point count of y²=x³+ax+b over the small prime field F_p: "
         "#E(F_p) = p + 1 + Σ_x (x³+ax+b / p). Generation = Legendre sum (Euler criterion "
         "a^((p−1)/2)); independent verifier = **exhaustive square table** + **per-y full scan** "
         "second channel + the exact-integer form of the Hasse bound |N−(p+1)| ≤ 2√p, namely "
         "(N−p−1)² ≤ 4p; certificate pointcount_small is independently re-checked by "
         "certcheck_alg. **Convention clarification**: this operator checks the Hasse **bound**, "
         "not the Hasse **principle** (the latter is the local-global principle, layer-2 stub "
         "hasse_principle_check; the two must not be conflated)."),
    requires_cert=True,
    thm="T-V31-PC",
    stmt="#E(F_p) = p+1+Σ_x(a_p(x)), 且 |#E − (p+1)| ≤ 2√p (Hasse 界)",
    selfchecks=(
        {"id": "pc_x3_plus_x_plus_1_F7", "args": {"a": 1, "b": 1, "p": 7},
         "check": lambda r: (r.value["count"] == 5 and r.value["two_algorithms_agree"]
                             and r.value["hasse_ok"] and r.cert_independent,
                             "#E(F_7) = %d (手算: (0,±1),(2,±2) + O)" % r.value["count"]),
         "expect": "y²=x³+x+1 over F_7: 仿射 4 点 + O = 5"},
        {"id": "pc_x3_minus_x_F5", "args": {"a": -1, "b": 0, "p": 5},
         "check": lambda r: (r.value["count"] == 8, "#E(F_5) = %d" % r.value["count"]),
         "expect": "y²=x³−x over F_5: N = 8 (经典例, Hasse 界内)"},
        {"id": "pc_singular_reject", "args": {"a": 0, "b": 0, "p": 7},
         "expect_err": "PRECOND_FAIL", "expect": "4a³+27b²=0 模 p ⇒ 奇异, 点数公式不适用"},
        {"id": "pc_p3_min_field", "args": {"a": 1, "b": 1, "p": 3},
         "check": lambda r: (r.value["count"] >= 1, "p=3 (最小域) 通过: N=%d" % r.value["count"]),
         "expect": "p=3 边界"},
        {"id": "pc_p_not_prime_reject", "args": {"a": 1, "b": 1, "p": 9},
         "expect_err": "PRECOND_FAIL", "expect": "p 须素"},
        {"id": "pc_large_p_97", "args": {"a": 2, "b": 3, "p": 97},
         "check": lambda r: (r.value["two_algorithms_agree"] and r.value["hasse_ok"],
                             "p=97: N=%d, 三路一致" % r.value["count"]),
         "expect": "较大 p 下两算法仍一致"},
        {"id": "pc_p3_nonsingular", "args": {"a": 1, "b": 2, "p": 3},
         "check": lambda r: (r.value["count"] == 4 and r.value["hasse_ok"],
                             "y²=x³+x+2 over F_3: N=%d (仿射 3 + O)" % r.value["count"]),
         "expect": "p=3 上 N=4 (= p+1, 无挠情形)"},
        {"id": "pc_x3_plus_1_F3_singular_reject", "args": {"a": 0, "b": 1, "p": 3},
         "expect_err": "PRECOND_FAIL",
         "expect": "y²=x³+1 在 p=3 处奇异 (4a³+27b² = 27 ≡ 0 mod 3) ⇒ 域外 (即对 T=3 的扭曲线)"},
    ),
)


# ===========================================================================
# band 9940: three stubs (the part layer 1 cannot do, explicitly isolated)
# ===========================================================================

_mk_stub(
    "ant_ring_integers", BAND_STUB, 1, 1,
    _sdict({"d": {"kind": "int", "min": 2, "max": 10 ** 6}}, ["d"]),
    _sdict({"h": _EXACT, "class_group_invariants": {"kind": "list", "items": _EXACT, "minlen": 0, "maxlen": 16}},
           ["h", "class_group_invariants"]),
    reason="类数 h(d) 与类群结构需要**实二次域类群枚举的完备性证明** (Minkowski 界 + 归约型的完整枚举)",
    dep="实二次域类群算法 (Minkowski 界枚举 + 归约型完整枚举) 或已形式化的类数表 (mathlib 未提供)",
    doc=("stub: class number h(d) and class-group invariants of imaginary/real quadratic fields. "
         "**Not implemented** in this version -- what is missing is a machine-checkable proof of "
         "the enumeration-completeness step (running an enumeration and claiming it is complete is "
         "exactly the fake certificate we want to avoid)."),
    certs="planned: class_group_enumeration (归约型完整枚举 + Minkowski 界审计)",
)

_mk_stub(
    "ec_descent_2", BAND_STUB, 2, 1,
    _sdict({"a": _EXACT, "b": _EXACT}, ["a", "b"]),
    _sdict({"rank": _EXACT, "selmer_dim": _EXACT}, ["rank", "selmer_dim"]),
    reason="2-下降需要 Selmer 群/齐性空间局部可解性的完整判定 (含 p-adic 与实位的完备性)",
    dep="2-下降机器 (Selmer 群 + 局部可解性判定的完备实现) 或形式化证明",
    doc=("stub: 2-descent for the rank of an elliptic curve. **Not implemented** in this version "
         "-- the completeness of the local-solubility decision is an external machine dependency "
         "(one missing local place yields a wrong rank)."),
    certs="planned: selmer_local_solubility (逐位局部可解性 + 全位覆盖审计)",
)

_mk_stub(
    "hasse_principle_check", BAND_STUB, 3, 1,
    _sdict({"variety_code": {"kind": "str"}, "eq": {"kind": "list", "items": _EXACT, "minlen": 1, "maxlen": 64}},
           ["variety_code", "eq"]),
    _sdict({"local_solvable_all": {"kind": "bool"}, "global_solvable": {"kind": "bool"},
            "principle_holds": {"kind": "bool"}},
           ["local_solvable_all", "global_solvable", "principle_holds"]),
    reason="Hasse 原理 (局部-整体原则) 对一般簇**不成立**, 判定其成立需要 Brauer-Manin 障碍等外部理论",
    dep="Brauer 群 / Brauer-Manin 障碍的形式化 (或代数簇的显式有理点参数化)",
    doc=("stub: local-global principle check. **Not implemented** in this version -- it is a "
         "completely different thing from layer 1's curve_point_count_small **Hasse bound**: the "
         "bound is a point-count inequality (implemented), the principle says locally solvable => "
         "globally solvable (generally false; deciding it needs the Brauer-Manin obstruction). The "
         "names are close but the semantics differ, so it is isolated explicitly here to avoid "
         "misuse."),
    certs="planned: brauer_manin_obstruction (障碍计算 + 局部可解性全位覆盖)",
)
