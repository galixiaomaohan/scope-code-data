# -*- coding: utf-8 -*-
"""Layer 2: research grade (band 9920: 3 implemented, rows 9921–9923; band 9940: 5 stubs,
rows 9944–9948).

**The honesty discipline of this layer is stricter than the previous two**: most research-grade
propositions are not "just compute it and it is proven". This layer does only the three things
that are **computationally self-contained**; everything else is a stub, and the gaps and external
dependencies are written out item by item in STATUS/DESIGN.

Implemented (3, verified):
  ec_frey_construct     9921  Frey curve construction + **algebraic identity** checks on invariants/discriminant/conductor
  psi_explicit_estimate 9922  **strict rational interval** for ψ(x) (sieve over prime powers + log enclosure) + comparison against the registered bound
  trans_gelfond_apply   9923  Gelfond-Schneider **instantiation certificate** (exact algebraic re-check of the hypotheses)

stubs (5): mf_level_lowering / ls_match_certificate / ant_zero_free_region /
          circle_method_waring / p_adic_l_precision
"""
from __future__ import annotations

import math
from fractions import Fraction

from .v31registry import V31OpDef, V31REGISTRY

from svokernel.certify import Certificate
from svokernel.errors import Err, SvoError
from svokernel.exact import frac_str
from svokernel.tool import CallCtl

from .algnum import QsqrtD
from .numtheory import (factorize, is_prime, log_bracket, log_bracket_fixed,
                        log_bracket_int, sieve_primes)

BAND_L2 = 9920
BAND_STUB = 9940
_EXACT = {"kind": "int"}
_FR = {"kind": "Fraction"}

#: Fixed-point denominator for the ψ interval (generator side). The verifier/checker side uses a
#: finer denominator (a narrower enclosure) for the nesting test.
PSI_DEN = 10 ** 18
PSI_DEN_FINE = 10 ** 24


def _psi_ref_highprec(x: int) -> Fraction:
    """**High-precision reference value** of ψ(x) = Σ_{p^k ≤ x} log p (50-digit Decimal,
    independent of the generator's fixed-point enclosure route).

    Purpose: selfcheck anchor. The generator returns a strict rational enclosure [lo, hi]
    (~4.5e-14 wide at x=100); this function accumulates directly with decimal.ln() at 50-digit
    precision (error ~1e-48, far below the enclosure width) => the result must lie **strictly
    inside** [lo, hi]. The two routes are independent: one is rational fixed point + atanh series
    + geometric tail bound, the other is high-precision Decimal floating point. A purely rational
    anchor cannot be written down -- ψ(x) is irrational (a sum of logs of distinct primes).
    """
    from decimal import Decimal, localcontext
    with localcontext() as ctx:
        ctx.prec = 50
        ref = Decimal(0)
        for p in sieve_primes(x):
            pk = p
            while pk <= x:
                ref += Decimal(p).ln()
                pk *= p
        return Fraction(ref)


def _sdict(fields, req):
    return {"kind": "dict", "fields": fields, "required": req}


def _reg(**kw):
    kw.setdefault("status", "verified")
    kw.setdefault("layer", 2)
    return V31REGISTRY.register(V31OpDef(**kw))


def _mk_stub(name, band, sub, in_schema, out_schema, reason, dep, doc, certs, layer=2):
    def _invoke(args, ctl):
        raise SvoError(Err.MATHLIB_MISS, ctx="op:%s#%d" % (name, band + sub),
                       detail="stub: 缺 %s —— 本算子在本版不产生任何值" % dep)

    def _verify(args, value):
        raise SvoError(Err.MATHLIB_MISS, ctx="op:%s@verify" % name,
                       detail="stub 无实现, 验证器同样缺席 (不做假验证)")

    return _reg(
        name=name, band=band, sub=sub, layer=layer, status="stub",
        in_schema=in_schema, out_schema=out_schema, invoke=_invoke, verify=_verify,
        cost=("units", 0.0), errors=("MATHLIB_MISS", "SCHEMA_MISMATCH"),
        effects="none (stub: 无副作用也无实现)", doc=doc, stub_reason=reason,
        external_dep=dep, planned_cert=certs,
        selfchecks=(
            {"id": name + "_absent", "args": _STUB_ARGS[name], "expect_err": "MATHLIB_MISS",
             "expect": "stub 显式拒绝: 缺口=%s" % reason},
        ),
    )


# ===========================================================================
# 9921  ec_frey_construct
# ===========================================================================

_FREY_IN = _sdict({
    "a": {"kind": "int", "min": 1, "max": 10 ** 12},
    "b": {"kind": "int", "min": 1, "max": 10 ** 12},
    "p": {"kind": "int", "min": 1, "max": 64},
}, ["a", "b", "p"])
_FREY_OUT = _sdict({
    "a": _EXACT, "b": _EXACT, "p": _EXACT, "A": _EXACT, "B": _EXACT, "C": _EXACT, "c": _EXACT,
    "curve": {"kind": "dict", "fields": {"a2": _EXACT, "a4": _EXACT, "a6": _EXACT},
              "required": ["a2", "a4", "a6"]},
    "disc": _EXACT, "cond": _EXACT, "c4": _EXACT,
    "j_num": {"kind": "str"}, "j_den": {"kind": "str"},
    "disc_bits": _EXACT, "semistable_claim": {"kind": "str"},
}, ["a", "b", "p", "A", "B", "C", "c", "curve", "disc", "cond", "c4", "j_num", "j_den",
    "disc_bits", "semistable_claim"])


def _iroot(n: int, k: int) -> int:
    """Integer k-th root (floor)."""
    if n < 0 or k < 1:
        raise ValueError("iroot 域")
    lo, hi = 0, 1
    while hi ** k <= n:
        hi *= 2
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if mid ** k <= n:
            lo = mid
        else:
            hi = mid
    return lo


def _frey_precond(args):
    a, b, p = args["a"], args["b"], args["p"]
    if a < 1 or b < 1:
        return False, "要求 a, b >= 1 (Frey 构造的两项均非零)"
    if a == b:
        return False, "a = b ⇒ A = B ⇒ x(x−A)(x+B) 退化 (判别式 0)"
    if math.gcd(a, b) != 1:
        return False, "要求 gcd(a,b) = 1 (Frey 曲线半稳定性的前提)"
    C = a ** p + b ** p
    c = _iroot(C, p)
    if c ** p != C:
        return False, ("a^p + b^p = %d 不是完全 p 次幂 ⇒ 没有 c 使 A+B=C 成 p 次幂形式; "
                       "Frey 构造的前提 (Fermat 型解) 不成立" % C)
    if (a ** p).bit_length() > 4096:
        return False, "A 位数超过 4096 bit (规模上限)"
    return True, ""


def _invoke_frey(args, ctl):
    a, b, p = args["a"], args["b"], args["p"]
    A, B = a ** p, b ** p
    C = A + B
    c = _iroot(C, p)
    a2, a4, a6 = B - A, -A * B, 0
    disc = 16 * (A * B * C) ** 2
    c4 = 16 * (A * A + A * B + B * B)
    cond = 1
    for q, _e in factorize(A * B * C):
        cond *= q
    jn, jd = c4 ** 3, disc
    g = math.gcd(jn, jd) or 1
    jn, jd = jn // g, jd // g
    cert = Certificate(kind="frey_invariants",
                       payload={"a": a, "b": b, "p": p, "A": A, "B": B, "C": C, "c": c,
                                "curve": {"a2": a2, "a4": a4, "a6": a6},
                                "disc": disc, "cond": cond, "c4": c4,
                                "j_num": frac_str(Fraction(jn)), "j_den": frac_str(Fraction(jd))},
                       issued_by="ec_frey_construct")
    return {"value": {"a": a, "b": b, "p": p, "A": A, "B": B, "C": C, "c": c,
                      "curve": {"a2": a2, "a4": a4, "a6": a6}, "disc": disc, "cond": cond,
                      "c4": c4, "j_num": frac_str(Fraction(jn)), "j_den": frac_str(Fraction(jd)),
                      "disc_bits": disc.bit_length(),
                      "semistable_claim": "本算子不声称半稳定性 (属外部定理); 只登记 N=rad(ABC)"},
            "resid": [0.0], "cert": cert, "cost": float(p) * 20.0,
            "notes": "Frey: E: y² = x(x−%d)(x+%d) = x³%+dx²%+dx; Δ=16(ABC)², "
                     "N=rad(ABC)=%d, c4=%d" % (A, B, a2, a4, cond, c4)}


def _verify_frey(args, value):
    """Independent verifier: A+B=C + discriminant via the standard Weierstrass invariants +
    rad recomputation + exact rational j."""
    bad = []
    A, B, C = value["A"], value["B"], value["C"]
    p = args["p"]
    if A != args["a"] ** p or B != args["b"] ** p or A + B != C:
        bad.append("A/B/C 或 A+B=C 复算失败")
    if value["c"] ** p != C:
        bad.append("c^p ≠ C")
    cur = value["curve"]
    b2, b4, b6 = 4 * cur["a2"], 2 * cur["a4"], 4 * cur["a6"]
    b8 = 4 * cur["a2"] * cur["a6"] - cur["a4"] ** 2
    d_std = -b2 * b2 * b8 - 8 * b4 ** 3 - 27 * b6 ** 2 + 9 * b2 * b4 * b6
    if d_std != value["disc"]:
        bad.append("Weierstrass 不变量判别式 %d ≠ 声称 %d" % (d_std, value["disc"]))
    if value["disc"] != 16 * (A * B * C) ** 2:
        bad.append("判别式 ≠ 16(ABC)²")
    if b2 * b2 - 24 * b4 != value["c4"]:
        bad.append("c4 标准式不符")
    r = 1
    for q, _e in factorize(A * B * C):
        r *= q
    if r != value["cond"]:
        bad.append("rad(ABC) = %d ≠ 声称 %d" % (r, value["cond"]))
    jn, jd = Fraction(value["j_num"]), Fraction(value["j_den"])
    if jd * value["c4"] ** 3 != jn * value["disc"]:
        bad.append("j = c4³/Δ 精确有理等式失败")
    if Fraction(jn, jd) != Fraction(value["c4"] ** 3, value["disc"]):
        bad.append("j 数值不等")
    return {"ok": not bad, "resid": [float(len(bad))],
            "detail": "A+B=C + Weierstrass 不变量 + rad 复算 + j 精确有理 通过" if not bad
                      else "; ".join(bad)}


_reg(
    name="ec_frey_construct", band=BAND_L2, sub=1,
    in_schema=_FREY_IN, out_schema=_FREY_OUT,
    invoke=_invoke_frey, verify=_verify_frey, cert_kind="frey_invariants",
    cost=("units", 40.0),
    errors=("PRECOND_FAIL", "SCHEMA_MISMATCH", "POSTCOND_FAIL", "BOUNDED_VIOLATION", "TIMEOUT"),
    precond=_frey_precond,
    effects="none (纯函数)",
    doc=("Frey curve construction (Fermat-type data (a,b,p) -> the curve "
         "y² = x(x−A)(x+B), A=a^p, B=b^p): output the curve coefficients and (Δ, N, c4, j) and "
         "check the **algebraic identities**. Generation = root-product discriminant; independent "
         "verifier = **standard Weierstrass invariants (b2,b4,b6,b8) recomputing the "
         "discriminant** (a different derivation from the root-product form) + independent rad(ABC) "
         "factorization + exact rational j = c4³/Δ; certificate frey_invariants is independently "
         "re-checked by certcheck_res. **Honesty boundary**: this operator checks algebraic "
         "identities; it does not claim geometric/arithmetic facts such as \"the curve is "
         "semistable\" or \"the conductor = rad\" -- those depend on external theorems and are "
         "marked as unproven in the semistable_claim field. For p >= 3 positive instances cannot "
         "exist (FLT), so the positive selfchecks use an algebraic triple (p=1) and a Pythagorean "
         "triple (p=2, not for FLT use) to exercise the machine."),
    requires_cert=True,
    thm="T-V31-FREY",
    stmt="A=a^p, B=b^p, C=A+B 为 p 次幂 ⇒ E: y²=x(x−A)(x+B) 的 Δ=16(ABC)², N=rad(ABC), c4=16(A²+AB+B²)",
    selfchecks=(
        {"id": "frey_p1_algebraic", "args": {"a": 2, "b": 3, "p": 1},
         "check": lambda r: (r.value["A"] == 2 and r.value["B"] == 3 and r.value["C"] == 5
                             and r.value["disc"] == 14400 and r.value["cond"] == 30
                             and r.value["c4"] == 304 and r.cert_independent,
                             "p=1: Δ=16·(2·3·5)²=%d, N=rad(30)=%d, c4=%d"
                             % (r.value["disc"], r.value["cond"], r.value["c4"])),
         "expect": "代数三元组 (2,3,5): Δ=14400, N=30, c4=304"},
        {"id": "frey_p2_pythagorean", "args": {"a": 3, "b": 4, "p": 2},
         "check": lambda r: (r.value["A"] == 9 and r.value["B"] == 16 and r.value["C"] == 25
                             and r.value["disc"] == 16 * 3600 ** 2 and r.value["c4"] == 7696,
                             "勾股 triple (3,4,5): A=9,B=16,C=25, Δ=%d, c4=%d"
                             % (r.value["disc"], r.value["c4"])),
         "expect": "p=2 勾股三元组 (p=2 时 Frey 曲线非 FLT 用途, 仅验机器)"},
        {"id": "frey_p3_no_flt_solution", "args": {"a": 3, "b": 4, "p": 3},
         "expect_err": "PRECOND_FAIL",
         "expect": "27 + 64 = 91 非立方 ⇒ 前提缺失 (FLT n=3 无解, 故这类正例不存在是**数学事实**)"},
        {"id": "frey_gcd_reject", "args": {"a": 2, "b": 4, "p": 1},
         "expect_err": "PRECOND_FAIL", "expect": "gcd(a,b) ≠ 1 域外 (半稳定性前提)"},
        {"id": "frey_a_eq_b_reject", "args": {"a": 3, "b": 3, "p": 1},
         "expect_err": "PRECOND_FAIL", "expect": "a = b ⇒ 判别式为 0"},
        {"id": "frey_big_frey", "args": {"a": 5, "b": 7, "p": 1},
         # B11 fix: the conductor N is the **radical** rad(420) = 210 of a·b·(a+b), not the
         # product 420. (The original criterion compared 5·7·12 = 420 against cond -- while its
         #  own expect line already said rad(420)=210, so criterion and comment contradicted each
         #  other; the real run gives cond = 210.)
         "check": lambda r: (r.value["cond"] == 210 and r.value["disc_bits"] > 0,
                             "p=1 大数: N=rad(5·7·12)=rad(420)=%d" % r.value["cond"]),
         "expect": "rad(420) = 2·3·5·7 = 210"},
    ),
)


# ===========================================================================
# 9922  psi_explicit_estimate
# ===========================================================================

_PSI_IN = _sdict({
    "x": {"kind": "int", "min": 1, "max": 10 ** 6},
    "terms": {"kind": "int", "min": 10, "max": 80},
    "bound_name": {"kind": "str", "enum": ["psi_ge_log2", "psi_le_xlogx"]},
}, ["x", "terms", "bound_name"])
_PSI_OUT = _sdict({
    "x": _EXACT, "terms": _EXACT, "n_prime_powers": _EXACT,
    "psi_lo": {"kind": "str"}, "psi_hi": {"kind": "str"}, "width": {"kind": "str"},
    "ratio_lo": {"kind": "str"}, "ratio_hi": {"kind": "str"},
    "bound_name": {"kind": "str"}, "bound_ok": {"kind": "bool"}, "bound_rhs": {"kind": "str"},
    "honesty_note": {"kind": "str"},
}, ["x", "terms", "n_prime_powers", "psi_lo", "psi_hi", "width", "ratio_lo", "ratio_hi",
    "bound_name", "bound_ok", "bound_rhs", "honesty_note"])


def _psi_precond(args):
    if args["x"] < 1:
        return False, "x >= 1"
    if args["x"] > 10 ** 6:
        return False, "x 超出筛法上限 1e6 (拒绝给未证的上界)"
    if args["terms"] < 10:
        return False, "terms >= 10 (包络宽度需要足够的级数项)"
    return True, ""


def _prime_powers(x: int, ctl) -> list:
    """Prime-power table: enumerate p^k ≤ x per prime p (a different path from the checker's
    sieve)."""
    out = []
    for p in sieve_primes(x):
        pk = p
        while pk <= x:
            ctl.check_time("op:psi_explicit_estimate#9922")
            out.append(p)
            pk *= p
    return out


def _invoke_psi(args, ctl):
    x, terms, bname = args["x"], args["terms"], args["bound_name"]
    pps = _prime_powers(x, ctl)
    # fixed-point interval accumulation (outward rounding): see numtheory.log_bracket_fixed
    lo_i = 0
    hi_i = 0
    for p in pps:
        a, b = log_bracket_int(p, terms, PSI_DEN)
        lo_i += a
        hi_i += b
    lo, hi = Fraction(lo_i, PSI_DEN), Fraction(hi_i, PSI_DEN)
    xx = Fraction(x)
    ratio_lo = lo / xx if x > 0 else Fraction(0)
    ratio_hi = hi / xx if x > 0 else Fraction(0)
    if bname == "psi_ge_log2":
        # B8 fix: the criterion becomes a **necessary-and-sufficient combined criterion**, no
        # longer an interval comparison.
        # Why interval comparison was wrong: log 2 is irrational => any rational lower bound is
        # strictly < log 2; and at x=2, ψ(2) = log 2 **with equality** => "lo >= (lower bound of
        # log 2)" is always false => a valid conclusion is misjudged as a failure.
        # Necessary-and-sufficient criterion: ψ(x) = Σ_{p^k≤x} log p >= log 2 <=> the sum is
        # non-empty <=> there exists p^k <= x (every term log p >= log 2 and the term count is
        # non-negative, so a term implies it; no term means ψ = 0 < log 2).
        rhs = log_bracket(2, 40)[0]
        ok = len(pps) >= 1
        rhs_s = ("%s  (仅展示 log 2 的有理下界; 判定用组合判据: 存在 p^k ≤ x)" % rhs)
        bound_rule = "exists_prime_power_le_x"
    else:
        if x >= 2:
            xl = log_bracket(x, 40)
            rhs = x * xl[1]
        else:
            rhs = Fraction(0)
        ok = hi <= rhs
        rhs_s = str(rhs)
        bound_rule = "interval_le_xlogx"
    cert = Certificate(kind="psi_interval",
                       payload={"x": x, "terms": terms, "n_prime_powers": len(pps),
                                "psi_lo": frac_str(lo), "psi_hi": frac_str(hi),
                                "bound_name": bname, "bound_ok": ok,
                                "bound_rule": bound_rule},
                       issued_by="psi_explicit_estimate")
    return {"value": {"x": x, "terms": terms, "n_prime_powers": len(pps),
                      "psi_lo": frac_str(lo), "psi_hi": frac_str(hi),
                      "width": frac_str(hi - lo), "ratio_lo": frac_str(ratio_lo),
                      "ratio_hi": frac_str(ratio_hi), "bound_name": bname, "bound_ok": ok,
                      "bound_rhs": rhs_s,
                      "honesty_note": "登记界仅含可一行证明的基本界; 深层显式估计 (如 "
                                      "Rosser-Schoenfeld/Dusart 常数) 属外部文献结果, "
                                      "本算子不声称其证明"},
            "resid": [float(hi - lo)], "cert": cert, "cost": float(len(pps)) * 3.0,
            "notes": "ψ(%d) ∈ [%s, %s] (宽 %s), %d 个素数幂, 界 %s:%s"
                     % (x, frac_str(lo)[:24], frac_str(hi)[:24], str(hi - lo)[:24],
                        len(pps), bname, ok)}


def _verify_psi(args, value):
    """Independent verifier: **sieve** recomputation of the prime-power count + nesting test with
    a narrower enclosure + exact rational comparison of the bound."""
    x, terms = args["x"], args["terms"]
    bad = []
    # sieve path (different from the generator's per-prime power enumeration)
    spf = list(range(x + 1))
    d = 2
    while d * d <= x:
        if spf[d] == d:
            for m in range(d * d, x + 1, d):
                if spf[m] == m:
                    spf[m] = d
        d += 1
    n_pp = 0
    for n in range(2, x + 1):
        q = spf[n]
        m = n
        while m % q == 0:
            m //= q
        if m == 1:
            n_pp += 1
    if n_pp != value["n_prime_powers"]:
        bad.append("筛法数出素数幂 %d ≠ %d" % (n_pp, value["n_prime_powers"]))
    lo, hi = Fraction(value["psi_lo"]), Fraction(value["psi_hi"])
    if hi < lo:
        bad.append("区间颠倒")
    # a narrower enclosure (doubled term count + finer fixed-point denominator) must fall inside
    # the certified interval
    lo2_i = 0
    hi2_i = 0
    for p in sieve_primes(x):
        pk = p
        while pk <= x:
            a, b = log_bracket_int(p, terms * 2 + 24, PSI_DEN_FINE)
            lo2_i += a
            hi2_i += b
            pk *= p
    lo2, hi2 = Fraction(lo2_i, PSI_DEN_FINE), Fraction(hi2_i, PSI_DEN_FINE)
    if not (lo <= lo2 and hi2 <= hi):
        bad.append("嵌套检验失败: 更窄包络不在证书区间内")
    # B8 fix: the bound criterion matches the generator's but is **independently recomputed** --
    # psi_ge_log2 uses the necessary-and-sufficient combined criterion (the number of prime
    # powers this function sieves out itself is n_pp >= 1), psi_le_xlogx uses a rational interval
    # comparison.
    if args["bound_name"] == "psi_ge_log2":
        want_bound = n_pp >= 1
    else:
        want_bound = hi <= (x * log_bracket(x, 40)[1] if x >= 2 else 0)
    if bool(value["bound_ok"]) != want_bound:
        bad.append("bound_ok 标志与实算不符 (实算 %s, 声称 %s)"
                   % (want_bound, value["bound_ok"]))
    return {"ok": not bad, "resid": [float(len(bad))],
            "detail": "筛法素数幂计数一致 + 更窄包络嵌套 + 界比较 通过 (x=%d)" % x
                      if not bad else "; ".join(bad)}


_reg(
    name="psi_explicit_estimate", band=BAND_L2, sub=2,
    in_schema=_PSI_IN, out_schema=_PSI_OUT,
    invoke=_invoke_psi, verify=_verify_psi, cert_kind="psi_interval",
    cost=("units", 30.0),
    errors=("PRECOND_FAIL", "SCHEMA_MISMATCH", "POSTCOND_FAIL", "BOUNDED_VIOLATION", "TIMEOUT"),
    precond=_psi_precond,
    effects="none (纯函数)",
    doc=("**Strict rational interval** for Chebyshev ψ(x) = Σ_{p^k ≤ x} log p (interval "
         "arithmetic, reusing the cert_interval idea): prime powers are exhaustively enumerated by "
         "a sieve, and each log p is enclosed by an atanh series + an explicit geometric tail "
         "bound giving a strict rational enclosure [lo, hi] ⊂ Q. Generation = per-prime power "
         "enumeration; independent verifier = **sieve** (smallest-prime-factor array) recomputing "
         "the prime-power count + **nesting test with a narrower enclosure (doubled term count)** "
         "(the narrow interval must be contained in the wide one -- this is the core criterion); "
         "certificate psi_interval is independently re-checked by certcheck_res. **Registered "
         "bounds** are only the two basic bounds provable in one line (ψ >= log 2; ψ <= x·log x); "
         "deeper explicit estimates (Rosser-Schoenfeld/Dusart constants) are external literature "
         "results whose proof this operator **does not claim**, noting so in honesty_note."),
    requires_cert=True,
    thm="T-V31-PSI",
    stmt="ψ(x) = Σ_{p^k≤x} log p; log 2 ≤ ψ(x) ≤ x·log x (基本界); 深层显式估计属外部结果",
    selfchecks=(
        {"id": "psi_100_known_value", "args": {"x": 100, "terms": 30, "bound_name": "psi_ge_log2"},
         # B9 fix: the original anchor Fraction(940453, 10000) = 94.0453 is the **truncated
         # display value** of ψ(100) (true value 94.0453112293573922...), below the enclosure
         # lower bound 94.045311229357377973 => the criterion was always false.
         # The anchor is changed to _psi_ref_highprec(100): an independent route accumulating
         # Σ ln p directly at 50-digit Decimal precision (error ~1e-48, far below the enclosure
         # width 4.5e-14) => its value must lie strictly inside [lo, hi]. This cross-checks two
         # independent routes, "fixed-point rational enclosure vs high-precision float" -- a
         # purely rational anchor cannot be written down because ψ(100) is irrational.
         # B10 fix: the true prime-power count is **35** (25 primes + 10 genuine prime powers
         # 4,8,9,16,25,27,32,49,64,81); the original criterion said 34, and the "34" in its
         # message was a **hard-coded string** rather than a formatted value.
         "check": lambda r: (Fraction(r.value["psi_lo"]) <= _psi_ref_highprec(100)
                             <= Fraction(r.value["psi_hi"])
                             and Fraction(r.value["width"]) < Fraction(1, 10 ** 12)
                             and r.value["n_prime_powers"] == 35
                             and r.cert_independent,
                             "ψ(100) ∈ [%s, %s], 宽 %s, %d 个素数幂"
                             % (r.value["psi_lo"][:26], r.value["psi_hi"][:26],
                                r.value["width"][:26], r.value["n_prime_powers"])),
         "expect": "ψ(100) ≈ 94.0453112293573922 (25 素数 + 10 个真素数幂 = 35 项); "
                   "锚点走 50 位 Decimal 独立路线并严格落在包络内; 包络宽 < 1e-12"},
        {"id": "psi_1_zero", "args": {"x": 1, "terms": 20, "bound_name": "psi_ge_log2"},
         "check": lambda r: (r.value["psi_lo"] == "0" and r.value["psi_hi"] == "0"
                             and r.value["n_prime_powers"] == 0 and not r.value["bound_ok"],
                             "x=1: ψ=0, 空和 ⇒ ψ>=log2 不成立 (bound_ok=False 如实报告)"),
         "expect": "边界 x=1: ψ = 0 (空和), 且界不成立 —— 拒绝也要如实登记"},
        {"id": "psi_2_log2", "args": {"x": 2, "terms": 30, "bound_name": "psi_ge_log2"},
         "check": lambda r: (r.value["n_prime_powers"] == 1 and r.value["bound_ok"],
                             "x=2: ψ = log 2, 一个素数幂"),
         "expect": "x=2: ψ(2) = log 2"},
        {"id": "psi_xlogx_upper", "args": {"x": 1000, "terms": 20, "bound_name": "psi_le_xlogx"},
         "check": lambda r: (r.value["bound_ok"], "ψ(1000) ≤ 1000·log 1000 (基本界)"),
         "expect": "基本上界成立 (log(x!) ≤ x log x 路线)"},
        {"id": "psi_1e6_sieve", "args": {"x": 1000000, "terms": 12, "bound_name": "psi_le_xlogx"},
         "check": lambda r: (r.value["n_prime_powers"] == 78734 and r.value["bound_ok"],
                             "x=1e6: 素数幂 %d 个 (78498 素数 + 236 个高次幂)"
                             % r.value["n_prime_powers"]),
         "expect": "π(1e6) = 78498; 高次幂 (p^k ≤ 1e6, k>=2) 共 236 个 ⇒ 78734"},
        {"id": "psi_x0_schema_reject", "args": {"x": 0, "terms": 20, "bound_name": "psi_ge_log2"},
         "expect_err": "SCHEMA_MISMATCH", "expect": "x >= 1 是类型层声明"},
        {"id": "psi_over_cap_reject", "args": {"x": 2000000, "terms": 20, "bound_name": "psi_ge_log2"},
         "expect_err": "SCHEMA_MISMATCH", "expect": "x ≤ 1e6 是类型层声明 (筛法上限)"},
        {"id": "psi_bad_bound_reject", "args": {"x": 100, "terms": 20, "bound_name": "pnt_asymptotic"},
         "expect_err": "SCHEMA_MISMATCH", "expect": "未登记界名被 enum 拦下"},
    ),
)


# ===========================================================================
# 9923  trans_gelfond_apply
# ===========================================================================

_GEL_IN = _sdict({
    "alpha": {"kind": "str"},
    "beta": {"kind": "str"},
    "thm_code": {"kind": "str", "enum": ["T-V31-GS"]},
}, ["alpha", "beta", "thm_code"])
_GEL_OUT = _sdict({
    "alpha": {"kind": "str"}, "beta": {"kind": "str"}, "thm_code": {"kind": "str"},
    "hypotheses": {"kind": "dict", "fields": {
        "alpha_algebraic": {"kind": "bool"}, "alpha_ne_0": {"kind": "bool"},
        "alpha_ne_1": {"kind": "bool"}, "beta_algebraic": {"kind": "bool"},
        "beta_irrational": {"kind": "bool"}}, "required": ["alpha_algebraic", "alpha_ne_0",
                                                           "alpha_ne_1", "beta_algebraic",
                                                           "beta_irrational"]},
    "conclusion": {"kind": "str"}, "value_kind": {"kind": "str", "enum": ["transcendental"]},
    "n_hypotheses_verified": _EXACT, "scope_note": {"kind": "str"},
}, ["alpha", "beta", "thm_code", "hypotheses", "conclusion", "value_kind",
    "n_hypotheses_verified", "scope_note"])

#: Registered theorem (this operator only **instantiates** it; it does not prove the theorem).
GS_THEOREM = {
    "T-V31-GS": {
        "name": "Gelfond–Schneider",
        "hypotheses": ["alpha_algebraic", "alpha_ne_0", "alpha_ne_1", "beta_algebraic",
                       "beta_irrational"],
        "conclusion": "transcendental(alpha^beta)",
        "source": "外部文献结果 (Gelfond 1934 / Schneider 1934); 本包不含其证明",
    },
}


def _gel_precond(args):
    try:
        al = QsqrtD_from(args["alpha"])
        be = QsqrtD_from(args["beta"])
    except SvoError as e:
        return False, "规范型解析失败: %s" % e.detail
    if al.a == 0 and al.b == 0:
        return False, "α = 0 (Gelfond–Schneider 要求 α ≠ 0)"
    if al.a == 1 and al.b == 0:
        return False, "α = 1 (定理要求 α ≠ 1)"
    if be.b == 0:
        return False, "β ∈ Q (定理要求 β 为无理代数数)"
    if args["thm_code"] not in GS_THEOREM:
        return False, "未登记定理码 %r" % args["thm_code"]
    return True, ""


def QsqrtD_from(s: str) -> QsqrtD:
    """Turn a canonical string into a QsqrtD (a pure rational string is treated as b=0, but the
    domain must be declared explicitly)."""
    from .algnum import parse_qsqrtd
    if "√" not in s:
        # rational: carried with d=2 (b=0 => the element is the rational itself); documented here
        return QsqrtD(2, s, 0)
    return parse_qsqrtd(s)


def _invoke_gel(args, ctl):
    al = QsqrtD_from(args["alpha"])
    be = QsqrtD_from(args["beta"])
    hyps = {
        "alpha_algebraic": True,          # a parseable canonical form => α ∈ Q(√d) or Q => algebraic
        "alpha_ne_0": not (al.a == 0 and al.b == 0),
        "alpha_ne_1": not (al.a == 1 and al.b == 0),
        "beta_algebraic": True,
        "beta_irrational": be.b != 0,
    }
    thm = GS_THEOREM[args["thm_code"]]
    # certificate and echo both store the **canonical** string (QsqrtD.to_str): the checker
    # (certcheck_res) only needs to recognize the canonical form, and generator/verifier use the
    # same string for the same element -- redundant leading '+' etc. is already normalized at the
    # type-layer entry by _slot.
    a_norm, b_norm = al.to_str(), be.to_str()
    cert = Certificate(kind="gelfond_apply",
                       payload={"alpha": a_norm, "beta": b_norm,
                                "thm_code": args["thm_code"], "conclusion": thm["conclusion"],
                                "hypotheses": hyps},
                       issued_by="trans_gelfond_apply")
    return {"value": {"alpha": a_norm, "beta": b_norm, "thm_code": args["thm_code"],
                      "hypotheses": hyps, "conclusion": thm["conclusion"],
                      "value_kind": "transcendental",
                      "n_hypotheses_verified": sum(1 for v in hyps.values() if v),
                      "scope_note": "实例化证书: 假设经验证成立后按登记定理结论签发; "
                                    "定理本身 (%s) 的证明不在本算子内 (%s)"
                                    % (thm["name"], thm["source"])},
            "resid": [0.0], "cert": cert, "cost": 2.0,
            "notes": "%s^%s 超越 (按 %s); 假设 5/5 精确复核; alpha/beta 已规范化为 %s / %s"
                     % (args["alpha"], args["beta"], thm["name"], a_norm, b_norm)}


def _verify_gel(args, value):
    """Independent verifier: re-check every hypothesis with **exact algebraic arithmetic** +
    align the conclusion word for word with the registered theorem + coverage."""
    bad = []
    al = QsqrtD_from(args["alpha"])
    be = QsqrtD_from(args["beta"])
    if not (al.a == 0 and al.b == 0) != value["hypotheses"]["alpha_ne_0"]:
        bad.append("alpha_ne_0 标志与实算不符")
    if value["hypotheses"]["alpha_ne_1"] != (not (al.a == 1 and al.b == 0)):
        bad.append("alpha_ne_1 标志与实算不符")
    if value["hypotheses"]["beta_irrational"] != (be.b != 0):
        bad.append("beta_irrational 标志与实算不符")
    thm = GS_THEOREM[args["thm_code"]]
    if value["conclusion"] != thm["conclusion"]:
        bad.append("结论与登记定理不符")
    for h in thm["hypotheses"]:
        if not value["hypotheses"].get(h):
            bad.append("登记假设 %r 未成立" % h)
    # independent confirmation of algebraicity: the **field element** α satisfies its own minimal
    # polynomial x² − 2a·x + N = 0 (evaluated exactly inside Q(√d)).
    # B12 fix: the original evaluated at the **rational part** al.a (= N − a², generally ≠ 0) --
    # the root of the minimal polynomial is the element a+b√d itself, not its rational part a.
    # With b=0 (α rational) the mistake happened to be invisible; α = 1 − (1/2)√2 exposes it.
    mp = al.min_poly()
    _val = (QsqrtD(al.d, mp[0], 0) + QsqrtD(al.d, mp[1], 0) * al
            + QsqrtD(al.d, mp[2], 0) * (al * al))
    if not (_val.a == 0 and _val.b == 0):
        bad.append("α 不满足其极小多项式 (代数性复核失败)")
    return {"ok": not bad, "resid": [float(len(bad))],
            "detail": "5 条假设精确复核 + 极小多项式回代 + 结论对齐 + 覆盖性 通过"
                      if not bad else "; ".join(bad)}


_reg(
    name="trans_gelfond_apply", band=BAND_L2, sub=3,
    in_schema=_GEL_IN, out_schema=_GEL_OUT,
    invoke=_invoke_gel, verify=_verify_gel, cert_kind="gelfond_apply",
    cost=("units", 3.0),
    errors=("PRECOND_FAIL", "SCHEMA_MISMATCH", "POSTCOND_FAIL", "MATHLIB_MISS", "TIMEOUT"),
    precond=_gel_precond,
    effects="none (纯函数)",
    doc=("**Instantiation** of the Gelfond-Schneider theorem: given algebraic α ≠ 0,1 and an "
         "irrational algebraic β (both as Q(√d) canonical strings), re-check the theorem's "
         "hypotheses exactly and then issue 'α^β is transcendental' per the conclusion. "
         "Generation = canonical parsing + hypothesis decision; independent verifier = exact "
         "algebraic re-check (including the minimal-polynomial back-substitution "
         "x²−2a·x+N=0); certificate gelfond_apply is independently re-checked by certcheck_res "
         "(own canonical-string parser + own square-free test + **hypothesis coverage** check: "
         "every hypothesis the registered theorem requires must have a verified instance). "
         "**Honesty boundary (critical)**: this operator only instantiates; it does **not prove** "
         "Gelfond-Schneider -- the theorem itself is an external literature result "
         "(Gelfond/Schneider 1934), so the certificate's semantics is 'the instantiation is "
         "correct', not 'transcendence is proven' -- stated in scope_note and in the certificate "
         "detail."),
    requires_cert=True,
    thm="T-V31-GS",
    stmt="α 代数, α ≠ 0, α ≠ 1, β 代数且无理 ⇒ α^β 超越 (Gelfond–Schneider)",
    selfchecks=(
        {"id": "gel_classic_2_sqrt2", "args": {"alpha": "2", "beta": "0+1√2", "thm_code": "T-V31-GS"},
         "check": lambda r: (r.value["value_kind"] == "transcendental"
                             and r.value["n_hypotheses_verified"] == 5 and r.cert_independent,
                             "2^√2 超越 (Gelfond 常数, 经典例)"),
         "expect": "2^√2 超越 —— 定理最著名的实例"},
        {"id": "gel_3_sqrt7", "args": {"alpha": "3", "beta": "0+1√7", "thm_code": "T-V31-GS"},
         "check": lambda r: (r.value["value_kind"] == "transcendental", "3^√7 超越"),
         "expect": "另一实例: 3^√7"},
        {"id": "gel_frac_alpha", "args": {"alpha": "1-1/2√2", "beta": "0+1√3", "thm_code": "T-V31-GS"},
         "check": lambda r: (r.value["hypotheses"]["alpha_algebraic"]
                             and r.value["hypotheses"]["alpha_ne_1"],
                             "α = 1 − (1/2)√2 (代数, ≠ 0,1), β = √3"),
         "expect": "有理系数代数数作底"},
        {"id": "gel_rational_beta_reject", "args": {"alpha": "2", "beta": "3", "thm_code": "T-V31-GS"},
         "expect_err": "PRECOND_FAIL", "expect": "β = 3 ∈ Q ⇒ 假设失败 (2^3 = 8 当然不超越)"},
        {"id": "gel_alpha_one_reject", "args": {"alpha": "1", "beta": "0+1√2", "thm_code": "T-V31-GS"},
         "expect_err": "PRECOND_FAIL", "expect": "α = 1 ⇒ 1^β = 1 不超越, 定理前提排除"},
        {"id": "gel_alpha_zero_reject", "args": {"alpha": "0", "beta": "0+1√2", "thm_code": "T-V31-GS"},
         "expect_err": "PRECOND_FAIL", "expect": "α = 0 域外"},
        {"id": "gel_nonsquarefree_d_reject", "args": {"alpha": "2", "beta": "0+1√4", "thm_code": "T-V31-GS"},
         "expect_err": "PRECOND_FAIL",
         "expect": "d=4 非无平方 ⇒ Q(√4)=Q 退化, algnum 类型层拒绝 (β 其实是 2 ∈ Q)"},
        {"id": "gel_bad_thm_code", "args": {"alpha": "2", "beta": "0+1√2", "thm_code": "T-NOPE"},
         "expect_err": "SCHEMA_MISMATCH", "expect": "未登记定理码被 enum 拦下"},
    ),
)


# ===========================================================================
# band 9940: layer 2 stubs ×5 (rows 9944–9948)
# ===========================================================================

_STUB_ARGS = {
    "mf_level_lowering": {"f_coeffs": [0, 1, -24], "N": 11, "M": 1, "prec": 6},
    "ls_match_certificate": {"L_coeffs": [0, 1, 1], "s": 1, "prec": 6, "eps_den": 10 ** 6},
    "ant_zero_free_region": {"sigma": "1/2", "T": 100, "region_code": "classical_3_5_1"},
    "circle_method_waring": {"k": 3, "s": 8, "N": 1000},
    "p_adic_l_precision": {"p": 5, "k": 6, "form_level": 1, "prec": 8},
}

_mk_stub(
    "mf_level_lowering", BAND_STUB, 4,
    _sdict({"f_coeffs": {"kind": "list", "items": _EXACT, "minlen": 1, "maxlen": 64},
            "N": {"kind": "int", "min": 1, "max": 1000},
            "M": {"kind": "int", "min": 1, "max": 1000},
            "prec": {"kind": "int", "min": 2, "max": 64}}, ["f_coeffs", "N", "M", "prec"]),
    _sdict({"g_coeffs": {"kind": "list", "items": _EXACT, "minlen": 1, "maxlen": 64},
            "trace_formula_ok": {"kind": "bool"}}, ["g_coeffs", "trace_formula_ok"]),
    reason="水平下降需要模符号/Atkin-Lehner 理论下的**算子迹公式**与维数公式的完备实现",
    dep="模符号 (modular symbols) 机器或已形式化的模形式维数/迹公式 (mathlib 无)",
    doc=("stub: modular-form level lowering Γ_0(NM) -> Γ_0(N). **Not implemented** in this "
         "version -- the completeness of the trace formula and dimension formula is an external "
         "machine dependency; merely shuffling coefficients and claiming 'this is the q-expansion "
         "of some modular form' is exactly the fake certificate this package avoids."),
    certs="planned: level_lowering_trace (迹公式逐项 + 维数匹配 + 新形式判别)",
)

_mk_stub(
    "ls_match_certificate", BAND_STUB, 5,
    _sdict({"L_coeffs": {"kind": "list", "items": _EXACT, "minlen": 1, "maxlen": 64},
            "s": {"kind": "int", "min": 0, "max": 100},
            "prec": {"kind": "int", "min": 2, "max": 64},
            "eps_den": {"kind": "int", "min": 1, "max": 10 ** 12}}, ["L_coeffs", "s", "prec", "eps_den"]),
    _sdict({"match_ok": {"kind": "bool"}, "L_value_interval": {"kind": "list", "items": {"kind": "str"},
                                                              "minlen": 2, "maxlen": 2}},
           ["match_ok", "L_value_interval"]),
    reason="L 函数匹配需要解析延拓/函数方程的**完备性证书**与 Γ 因子的严格区间算术 (含 Archimedean 位)",
    dep="L 函数解析延拓与函数方程的形式化 (或已证的匹配定理库)",
    doc=("stub: L-function matching certificate (matching some Dirichlet series against a proven "
         "L-function). **Not implemented** in this version -- what is missing is a "
         "machine-checkable certificate of analytic continuation and the functional equation; "
         "comparing finitely many coefficients alone is not a match."),
    certs="planned: l_function_match (函数方程残差 + Γ 因子区间 + 局部因子逐素核对)",
)

_mk_stub(
    "ant_zero_free_region", BAND_STUB, 6,
    _sdict({"sigma": {"kind": "str"}, "T": {"kind": "int", "min": 1, "max": 10 ** 9},
            "region_code": {"kind": "str", "enum": ["classical_3_5_1", "vinogradov_korobov"]}},
           ["sigma", "T", "region_code"]),
    _sdict({"bound_exponent": {"kind": "str"}, "region_nonempty": {"kind": "bool"}},
           ["bound_exponent", "region_nonempty"]),
    reason="零点无区域需要显式零点验证的**完备区间证明** (零点计数 + 有限区间计算 + 渐近项常数)",
    dep="Riemann-Siegel/零点计数与显式常数的形式化, 或已验证的零点表 + 区间算术证书",
    doc=("stub: zero-free region of the ζ function. **Not implemented** in this version -- the "
         "certificate for such a proposition is the complete assembly of 'finite-range computation "
         "+ asymptotic-term constants', and missing any piece makes it a fake certificate. "
         "**Note**: this is a different thing from the Hasse bound and the ψ(x) interval already "
         "implemented in layers 1/2 (those need only elementary bounds)."),
    certs="planned: zero_free_region (零点计数下界 + 有限区间逐段验证 + 渐近常数显式核对)",
)

_mk_stub(
    "circle_method_waring", BAND_STUB, 7,
    _sdict({"k": {"kind": "int", "min": 2, "max": 20}, "s": {"kind": "int", "min": 2, "max": 64},
            "N": {"kind": "int", "min": 1, "max": 10 ** 12}}, ["k", "s", "N"]),
    _sdict({"g_k_upper": _EXACT, "all_n_representable": {"kind": "bool"}},
           ["g_k_upper", "all_n_representable"]),
    reason="圆法 (Hardy-Littlewood) 需要优弧/劣弧估计的显式常数与 Weyl 和界的形式化",
    dep="Weyl 和估计与奇点级数的显式常数形式化 (或已证的 Waring 问题定理库)",
    doc=("stub: circle method / Waring's problem. **Not implemented** in this version -- the "
         "circle-method certificate is an assembly of two explicit estimates, and this package has "
         "no checkable Weyl-sum bound, so no 'representable' conclusion is given (computing "
         "representation counts for small N verifies instances only and cannot support an "
         "'all n' claim)."),
    certs="planned: circle_method_bound (优弧主项 + 劣弧 Weyl 界 + 显式常数核对)",
)

_mk_stub(
    "p_adic_l_precision", BAND_STUB, 8,
    _sdict({"p": {"kind": "int", "min": 2, "max": 10 ** 4}, "k": {"kind": "int", "min": 1, "max": 64},
            "form_level": {"kind": "int", "min": 1, "max": 100},
            "prec": {"kind": "int", "min": 2, "max": 64}}, ["p", "k", "form_level", "prec"]),
    _sdict({"l_p_value_mod_pk": {"kind": "str"}, "interpolation_ok": {"kind": "bool"}},
           ["l_p_value_mod_pk", "interpolation_ok"]),
    reason="p-adic L 函数精度需要 Iwasawa 代数上的插值性质与 p-adic 测度的完备构造",
    dep="p-adic L 函数 (Kubota-Leopoldt) 的构造与插值定理的形式化",
    doc=("stub: precision of p-adic L-functions. **Not implemented** in this version -- the "
         "interpolation property is a constructive theorem; without it, a 'value mod p^k' cannot "
         "be upgraded to a 'precision statement about the p-adic L-value'."),
    certs="planned: padic_l_interpolation (插值点逐点核对 + p-adic 测度矩 + 精度界)",
)
