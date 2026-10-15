# -*- coding: utf-8 -*-
"""Layer 2 research grade -- **independent checkers** for the certificates (3 kinds: Frey
invariants / ψ interval / Gelfond instantiation).

Independence discipline as before: no svokernel_v31 module is imported; everything uses a local
implementation built on the standard library.
"""
from __future__ import annotations

import functools
import math
import re
from fractions import Fraction

from svokernel.certify import register_checker

__all__ = ["CHECKER_KINDS"]

_RAT = re.compile(r"^-?\d+(/\d+)?$")


def _safe(fn):
    @functools.wraps(fn)
    def wrapped(payload):
        try:
            return fn(payload)
        except Exception as e:  # noqa: BLE001
            return False, [], "检查器拒收: payload 非法或检查过程异常 (%s: %s)" % (type(e).__name__, e)

    return wrapped


def _rat(v) -> Fraction:
    if isinstance(v, bool):
        raise ValueError("bool 不是有理数")
    if isinstance(v, int):
        return Fraction(v)
    if isinstance(v, Fraction):
        return v
    if isinstance(v, str):
        if not _RAT.match(v):
            raise ValueError("非规范有理字面量 %r" % (v,))
        return Fraction(v)
    raise ValueError("拒绝 %s 进精确通道" % type(v).__name__)


def _rad(n: int, cap: int = 1000000) -> int:
    """Radical rad(n) = Π distinct prime factors (trial division; a cofactor >1 is included when
    <= cap² or when it passes MR)."""
    m = abs(int(n))
    if m <= 1:
        return 1
    out = 1
    d = 2
    while d <= cap and d * d <= m:
        if m % d == 0:
            out *= d
            while m % d == 0:
                m //= d
        d += 1 if d == 2 else 2
    if m > 1:
        if m <= cap * cap or _mr(m):
            out *= m
        else:
            raise ValueError("rad: 余因子 %d 未证素 (拒绝给不完整根号)" % m)
    return out


def _mr(n: int) -> bool:
    if n < 2:
        return False
    for q in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % q == 0:
            return n == q
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


# ---------------------------------------------------------------------------
# 1) Frey curve invariant certificate
# ---------------------------------------------------------------------------


@register_checker("frey_invariants", independent_of=("ec_frey_construct",))
@_safe
def _check_frey_invariants(payload):
    """Frey curve: A+B=C reconciliation + discriminant recomputed via the **standard
    Weierstrass invariants** (b2,b4,b6,b8) + conductor = radical recomputation of rad(ABC) +
    exact rational check of j = c4³/Δ."""
    p = int(payload["p"])
    A, B, C = int(payload["A"]), int(payload["B"]), int(payload["C"])
    cur = payload["curve"]
    a2, a4, a6 = int(cur["a2"]), int(cur["a4"]), int(cur["a6"])
    disc, cond, c4 = int(payload["disc"]), int(payload["cond"]), int(payload["c4"])
    jn, jd = _rat(payload["j_num"]), _rat(payload["j_den"])
    bad = []
    if A + B != C:
        bad.append("A + B ≠ C (Frey 构造前提缺失)")
    if A <= 0 or B <= 0:
        bad.append("A,B 必须为正")
    if a2 != B - A:
        bad.append("a2 ≠ B − A (曲线系数不按 y²=x(x−A)(x+B) 展开)")
    if a4 != -A * B or a6 != 0:
        bad.append("a4/a6 与 y²=x³+(B−A)x²−ABx 不符")
    # standard Weierstrass invariants (a1=a3=0): b2=4a2, b4=2a4, b6=4a6, b8=4a2a6−a4²
    b2, b4, b6 = 4 * a2, 2 * a4, 4 * a6
    b8 = 4 * a2 * a6 - a4 * a4
    d_std = -b2 * b2 * b8 - 8 * b4 ** 3 - 27 * b6 ** 2 + 9 * b2 * b4 * b6
    if d_std != disc:
        bad.append("Weierstrass 不变量复算的判别式 %d ≠ 证书 %d" % (d_std, disc))
    if disc != 16 * (A * B * C) ** 2:
        bad.append("判别式 ≠ 16(ABC)² (根乘积形式)")
    r = _rad(A * B * C)
    if r != cond:
        bad.append("独立 rad(ABC) = %d ≠ 证书导子 %d" % (r, cond))
    c4_std = b2 * b2 - 24 * b4
    if c4_std != c4:
        bad.append("c4 标准式复算 %d ≠ 证书 %d" % (c4_std, c4))
    if disc == 0:
        bad.append("判别式为 0 (奇异退化, 不是椭圆曲线)")
    elif jn * disc != jd * c4 ** 3:
        bad.append("j 不满足 c4³/Δ 的精确有理等式")
    return (not bad), [float(len(bad))], (
        "A+B=C + Weierstrass 不变量判别式 + rad 复算 + c4/j 精确有理 全通过"
        if not bad else "; ".join(bad))


# ---------------------------------------------------------------------------
# 2) ψ(x) interval certificate
# ---------------------------------------------------------------------------


def _log_bracket_indep(p: int, terms: int):
    """Independent strict log enclosure. It uses the **same binary reduction** as the generator
    (only t' ≤ 1/3 squeezes the bound quickly), but the code is independent and the term count
    may differ; the actual criterion is the **nesting relation** (a narrower enclosure must be
    contained in a wider one), so if the generator understates its width the nesting check fails
    immediately.
    """
    if p <= 0:
        raise ValueError("log 仅对正数")
    if p == 1:
        return Fraction(0), Fraction(0)
    if terms < 1:
        raise ValueError("terms >= 1")
    j = p.bit_length() - 1
    q = 1 << j

    def at(t, K):
        acc = Fraction(0)
        for k in range(1, K + 1, 2):
            acc += t ** k / k
        lo = 2 * acc
        tail = 2 * t ** (K + 1) / ((K + 1) * (1 - t * t))
        return lo, lo + tail

    if q == p:
        a, b = at(Fraction(1, 3), terms)
        return j * a, j * b
    a, b = at(Fraction(1, 3), terms)
    c, d = at(Fraction(p - q, p + q), terms)
    return j * a + c, j * b + d


def _psi_interval_indep(x: int, terms: int, den: int):
    """Independent recomputation of the **fixed-point** rational enclosure of ψ(x): the
    prime-power table is generated by a **sieve** (not by enumerating powers per prime), and the
    accumulation uses a fixed denominator with outward rounding (inclusion guaranteed by
    floor/ceil, same recipe as the generator but independent code)."""
    if x < 1:
        raise ValueError("x >= 1")
    spf = list(range(x + 1))
    d = 2
    while d * d <= x:
        if spf[d] == d:
            for m in range(d * d, x + 1, d):
                if spf[m] == m:
                    spf[m] = d
        d += 1
    lo_i = 0
    hi_i = 0
    n_pp = 0
    for n in range(2, x + 1):
        q = spf[n]
        m = n
        while m % q == 0:
            m //= q
        if m == 1:
            n_pp += 1
            a, b = _log_bracket_int_indep(q, terms, den)
            lo_i += a
            hi_i += b
    return Fraction(lo_i, den), Fraction(hi_i, den), n_pp


def _log_bracket_indep_fixed(p: int, terms: int, den: int):
    lo, hi = _log_bracket_indep(p, terms)
    return int((lo * den).__floor__()), int(-((-hi * den).__floor__()))


def _log_bracket_int_indep(p: int, terms: int, den: int):
    """The checker's **own** all-integer fixed-point log enclosure (same mathematics as the
    generator, independent code; used on the fast path for large x)."""
    if p <= 0:
        raise ValueError("log 仅对正数")
    if p == 1:
        return 0, 0
    if terms % 2 == 0:
        terms -= 1
    K = max(1, terms)
    L = 1
    for k in range(1, K + 1, 2):
        L = L // math.gcd(L, k) * k

    def at(num, dn):
        S = 0
        for k in range(1, K + 1, 2):
            S += 2 * (L // k) * (num ** k) * (dn ** (K - k))
        dnm = L * (dn ** K)
        lo_i = (S * den) // dnm
        tn = 2 * (num ** (K + 1)) * (dn * dn) * den
        td = (K + 1) * (dn ** (K + 1)) * (dn * dn - num * num)
        return int(lo_i), int(lo_i + (-((-tn) // td)) + 2)

    j = p.bit_length() - 1
    q = 1 << j
    a2, b2 = at(1, 3)
    lo_i, hi_i = j * a2, j * b2
    if q != p:
        a, b = at(p - q, p + q)
        lo_i += a
        hi_i += b
    return int(lo_i), int(hi_i)


@register_checker("psi_interval", independent_of=("psi_explicit_estimate",))
@_safe
def _check_psi_interval(payload):
    """ψ(x) interval: independent sieve enumeration of prime powers + a **nesting test** for the
    narrower enclosure + comparison against the registered bound."""
    x = int(payload["x"])
    terms = int(payload["terms"])
    lo, hi = _rat(payload["psi_lo"]), _rat(payload["psi_hi"])
    bad = []
    if hi < lo:
        bad.append("区间上下界颠倒")
    lo2, hi2, n_pp = _psi_interval_indep(x, max(terms, 12) + 24, 10 ** 24)
    if n_pp != int(payload["n_prime_powers"]):
        bad.append("独立筛法数出素数幂 %d 个 ≠ 证书 %d" % (n_pp, int(payload["n_prime_powers"])))
    # nesting: the independent (narrower) enclosure must be contained in the certified interval
    # -- this is the core of the criterion
    if not (lo <= lo2 and hi2 <= hi):
        bad.append("嵌套检验失败: 独立包络 [%s, %s] 不落在证书区间 [%s, %s] 内"
                   % (str(lo2)[:24], str(hi2)[:24], str(lo)[:24], str(hi)[:24]))
    name = payload.get("bound_name", "")
    if name == "psi_ge_log2":
        # B8 fix: the criterion must be **independently recomputed**, and must not compare
        # intervals -- ψ(x) >= log 2 <=> the sum is non-empty <=> the number of prime powers the
        # checker **sieve-s out itself** is >= 1.
        # (log 2 is irrational => any rational lower bound is strictly < log 2 => "lo >= bound"
        #  is always false at the equality case x=2, which would misjudge a valid conclusion as a
        #  failure. The generator and verifier were both changed to this necessary-and-sufficient
        #  criterion.)
        want_bound = n_pp >= 1
        if bool(payload.get("bound_ok")) != want_bound:
            bad.append("bound_ok 标志与独立判据不符 (独立判据 %s, 声称 %s)"
                       % (want_bound, bool(payload.get("bound_ok"))))
        if not want_bound and not (lo == 0 and hi == 0):
            bad.append("无素数幂却给出非零 ψ 区间")
    elif name == "psi_le_xlogx":
        xl = _log_bracket_indep(x, 40) if x >= 2 else (Fraction(0), Fraction(0))
        rhs = x * xl[1] if x >= 2 else Fraction(0)
        if hi > rhs:
            bad.append("ψ(x) <= x·log x 的区间比较失败")
        if bool(payload.get("bound_ok")) != (hi <= rhs):
            bad.append("bound_ok 标志与实算不符")
    else:
        bad.append("未登记界名 %r" % name)
    if x == 1 and (lo != 0 or hi != 0):
        bad.append("x=1 时 ψ 应为 0")
    return (not bad), [float(len(bad))], (
        "独立筛法 %d 个素数幂 + 嵌套检验 + 登记界比较 全通过" % n_pp
        if not bad else "; ".join(bad))


# ---------------------------------------------------------------------------
# 3) Gelfond-Schneider instantiation certificate
# ---------------------------------------------------------------------------


#: Registered theorem table (same table as the generator side; the checker keeps its own copy,
#: used for the **coverage** check).
_GS_THEOREM = {
    "T-V31-GS": {
        "hypotheses": ["alpha_algebraic", "alpha_ne_0", "alpha_ne_1",
                       "beta_algebraic", "beta_irrational"],
        "conclusion": "transcendental(alpha^beta)",
    },
}

_QSQRTD = re.compile(r"^(-?\d+(/\d+)?)?([+-]\d+(/\d+)?)√(\d+)$")


def _parse_q(a, b, d):
    return (Fraction(a) if a else Fraction(0)), (Fraction(b) if b else Fraction(0)), int(d)


def _is_sqfree(n: int) -> bool:
    if n <= 1:
        return n == 1
    d, m = 2, n
    while d * d * d <= m:
        if m % d == 0:
            m //= d
            if m % d == 0:
                return False
        d += 1 if d == 2 else 2
    r = math.isqrt(m)
    return r * r != m


def _parse_canonical(s: str):
    """The checker's own canonical-string parser: "a+b√d" -> (a, b, d). Same syntax as algnum,
    independent code."""
    if not isinstance(s, str):
        raise ValueError("非字符串")
    if "√" not in s:
        # pure rational (no √) -- treated as a rational with b=0 (d takes the placeholder 2)
        # slot rule: one redundant leading '+' is allowed ("+1" ≡ "1"); after stripping it the
        # strict grammar still applies.
        t = s[1:] if isinstance(s, str) and len(s) > 1 and s[0] == "+" else s
        if not _RAT.match(t):
            raise ValueError("非规范有理字面量 %r" % (s,))
        return Fraction(t), Fraction(0), None
    head, _, tail = s.partition("√")
    d = int(tail)
    idx = -1
    for i in range(len(head) - 1, 0, -1):
        if head[i] in "+-" and head[i - 1] != "/":
            idx = i
            break
    if idx < 0:
        a_s, b_s = head, "0"
    else:
        a_s, b_s = head[:idx], head[idx:]
    if a_s in ("", "+"):
        a_s = "0"
    # the b slot carries an **explicit sign separator** ("0+1√2" => b_s = "+1"; "1-1/2√2" =>
    # b_s = "-1/2"). The separator is not part of the literal: strip one leading '+' and then
    # validate against the strict grammar ('-' is accepted by the grammar's own -?).
    # Slot syntax != rational-literal syntax, deliberately kept apart.
    if b_s.startswith("+"):
        b_s = b_s[1:]
    if not _RAT.match(a_s) or not _RAT.match(b_s):
        raise ValueError("非规范有理系数: %r / %r" % (a_s, b_s))
    return Fraction(a_s), Fraction(b_s), d


@register_checker("gelfond_apply", independent_of=("trans_gelfond_apply",))
@_safe
def _check_gelfond_apply(payload):
    """Instantiation certificate: independently re-check every hypothesis (exact algebraic
    arithmetic) + align the conclusion word for word with the registered theorem +
    **hypothesis coverage** (every hypothesis the registered theorem requires must have a
    verified instance)."""
    alpha_s = payload["alpha"]
    beta_s = payload["beta"]
    code = payload["thm_code"]
    concl = payload["conclusion"]
    hyps = payload.get("hypotheses", {})
    bad = []
    thm = _GS_THEOREM.get(code)
    if thm is None:
        return False, [], "未登记定理码 %r" % code
    if concl != thm["conclusion"]:
        bad.append("结论 %r ≠ 登记定理结论 %r" % (concl, thm["conclusion"]))
    aa, ab, ad = _parse_canonical(alpha_s)
    ba, bb, bd = _parse_canonical(beta_s)
    # algebraicity of alpha: a parseable canonical form means ∈ Q(√d) (or Q), hence algebraic;
    # d must be square-free and > 1 when non-rational
    if ad is not None and not (ad > 1 and _is_sqfree(ad)):
        bad.append("α 的 d=%r 不是无平方且 >1 的整数 ⇒ 不构成 Q(√d) 元素" % (ad,))
    if bd is not None and not (bd > 1 and _is_sqfree(bd)):
        bad.append("β 的 d=%r 不是无平方且 >1 的整数" % (bd,))
    if (aa, ab) == (0, 0):
        bad.append("α = 0 (假设 α ≠ 0 失败)")
    if (aa, ab) == (1, 0):
        bad.append("α = 1 (假设 α ≠ 1 失败)")
    beta_irr = (bb != 0)
    if not beta_irr:
        bad.append("β ∈ Q (假设 'β 无理' 失败)")
    # hypothesis coverage + per-item consistency (declared must agree with independent re-check)
    for h in thm["hypotheses"]:
        if not hyps.get(h):
            bad.append("登记定理要求的假设 %r 未在证书中声称成立" % h)
    if hyps.get("alpha_ne_0") and (aa, ab) == (0, 0):
        bad.append("声称 alpha_ne_0 但独立复核 α=0")
    if hyps.get("alpha_ne_1") and (aa, ab) == (1, 0):
        bad.append("声称 alpha_ne_1 但独立复核 α=1")
    if hyps.get("beta_irrational") and not beta_irr:
        bad.append("声称 beta_irrational 但独立复核 β ∈ Q")
    return (not bad), [float(len(bad))], (
        "实例化证书: 假设逐条独立复核 + 覆盖登记定理 %s 全部 %d 条假设 + 结论对齐"
        % (code, len(thm["hypotheses"])) if not bad else "; ".join(bad))


CHECKER_KINDS = ("frey_invariants", "psi_interval", "gelfond_apply")
