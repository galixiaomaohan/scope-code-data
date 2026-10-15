# -*- coding: utf-8 -*-
"""Layer 1 structural number theory -- **independent checkers** for the certificates (9 kinds).

Independence discipline as in certcheck_nt: this module imports **no** svokernel_v31 module (it
does not reuse the generator's algnum/numtheory); every decision has its own implementation using
only the standard library (Fraction/int/math). Each checker consumes pure payload data and returns
(ok, resid, detail); all exceptions are folded into "rejected".
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
    """Canonical rational literal -> Fraction. **Decimal notation is rejected** (that belongs to
    the float channel)."""
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


def _isfree_sq(n: int) -> bool:
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


# ---------------------------------------------------------------------------
# 1) continued-fraction convergent certificate
# ---------------------------------------------------------------------------


@register_checker("cf_convergent", independent_of=("alg_cont_frac",))
@_safe
def _check_cf_convergent(payload):
    """Continued fraction: 2×2 matrix-product rebuild + interlacing inequality for the
    convergents (all exact rationals)."""
    num, den = int(payload["num"]), int(payload["den"])
    terms = [int(t) for t in payload["terms"]]
    convs = [[int(p), int(q)] for p, q in payload["convergents"]]
    if den == 0:
        return False, [], "den = 0"
    if not terms:
        return False, [], "空项表"
    bad = []
    # matrix-product rebuild: [[a0,1],[1,0]]·[[a1,1],[1,0]]··· = [[p_k, p_{k-1}],[q_k, q_{k-1}]]
    mp, mq = (1, 0), (0, 1)   # previous column (p_{-1}, q_{-1}) = (1, 0)
    cur_p, cur_q = 1, 0
    prev_p, prev_q = 0, 1
    for i, a in enumerate(terms):
        cur_p, prev_p = a * cur_p + prev_p, cur_p
        cur_q, prev_q = a * cur_q + prev_q, cur_q
        if i < len(convs):
            if [cur_p, cur_q] != convs[i]:
                bad.append("第 %d 个收敛子矩阵重算 [%d,%d] ≠ 证书 %r" % (i, cur_p, cur_q, convs[i]))
                break
    if len(convs) != len(terms):
        bad.append("收敛子个数 %d ≠ 项数 %d" % (len(convs), len(terms)))
    x = Fraction(num, den)
    if convs:
        pk, qk = convs[-1]
        if qk == 0:
            bad.append("末收敛子分母为 0")
        elif Fraction(pk, qk) != x:
            bad.append("末收敛子 %s ≠ 真值 %s (精确有理不等)" % (Fraction(pk, qk), x))
    # interlacing inequality |x − p_i/q_i| < 1/(q_i q_{i+1}): strict for **all but the last
    # pair**; the last pair (i = last) has |x − p_{N−1}/q_{N−1}| = 1/(q_{N−1} q_N) (equality)
    # because x is exactly rational here, so the criterion is "equality allowed for the last
    # pair, strict elsewhere" -- that is the exact boundary, not a relaxation.
    for i in range(len(convs) - 1):
        p1, q1 = convs[i]
        p2, q2 = convs[i + 1]
        if q1 <= 0 or q2 <= 0:
            bad.append("第 %d 个收敛子分母非正" % i)
            continue
        gap = abs(x - Fraction(p1, q1))
        bound = Fraction(1, q1 * q2)
        if i == len(convs) - 2:
            if gap > bound:
                bad.append("末对收敛子误差 %s > 界 %s" % (gap, bound))
        elif gap >= bound:
            bad.append("第 %d 个收敛子违反严格交错不等式 (%s >= %s)" % (i, gap, bound))
    return (not bad), [float(len(bad))], (
        "矩阵乘积重建 %d 项 + 精确相等 + 交错不等式全通过" % len(terms) if not bad else "; ".join(bad[:4]))


# ---------------------------------------------------------------------------
# 2) Pell equation back-substitution certificate
# ---------------------------------------------------------------------------


@register_checker("pell_backsub", independent_of=("alg_pell_solve",))
@_safe
def _check_pell_backsub(payload):
    """Pell: back-substitute every solution into x²−Dy²=1 + fundamental-solution minimality
    (exhaustive y<y1) + power recurrence identity."""
    D = int(payload["D"])
    fund = [int(payload["fundamental"][0]), int(payload["fundamental"][1])]
    sols = [[int(x), int(y)] for x, y in payload["solutions"]]
    bad = []
    if D < 2 or _is_square(D):
        bad.append("D=%d 必须是 >=2 的非完全平方" % D)
    for x, y in sols:
        if x * x - D * y * y != 1:
            bad.append("(x,y)=(%d,%d) 回代 %d ≠ 1" % (x, y, x * x - D * y * y))
    if sols and sols[0] != fund:
        bad.append("解表首项与基本解不符")
    x1, y1 = fund
    if x1 * x1 - D * y1 * y1 != 1:
        bad.append("基本解回代失败")
    if x1 <= 0 or y1 <= 0:
        bad.append("基本解非正")
    if y1 <= 200000:
        for y in range(1, y1):
            v = D * y * y + 1
            r = math.isqrt(v)
            if r * r == v:
                bad.append("最小性失败: y=%d 也是解 (x=%d)" % (y, r))
                break
    # power recurrence: (x_k + y_k√D) = (x1 + y1√D)^k
    cx, cy = x1, y1
    for i, (x, y) in enumerate(sols):
        if i > 0:
            cx, cy = x1 * cx + D * y1 * cy, x1 * cy + y1 * cx
        if (x, y) != (cx, cy):
            bad.append("第 %d 个解 %r ≠ 幂递推 (%d,%d)" % (i, [x, y], cx, cy))
            break
    return (not bad), [float(len(bad))], (
        "%d 个解全部回代 + 幂递推一致%s" % (len(sols), "" if y1 > 200000 else "; 最小性穷举 y<%d 通过" % y1)
        if not bad else "; ".join(bad[:4]))


def _is_square(n: int) -> bool:
    r = math.isqrt(n) if n >= 0 else -1
    return r >= 0 and r * r == n


# ---------------------------------------------------------------------------
# 3) binary quadratic form reduction-chain certificate
# ---------------------------------------------------------------------------


@register_checker("form_reduce_chain", independent_of=("qf_gauss_reduce",))
@_safe
def _check_form_reduce_chain(payload):
    """Reduction chain: recompute every matrix action + discriminant invariant + determinant = 1
    + reduced inequality at the tail (exact integers)."""
    D = int(payload["D"])
    chain = [[int(a), int(b), int(c)] for a, b, c in payload["forms"]]
    mats = [[[int(m[0][0]), int(m[0][1])], [int(m[1][0]), int(m[1][1])]] for m in payload["matrices"]]
    red = [int(payload["reduced"][0]), int(payload["reduced"][1]), int(payload["reduced"][2])]
    bad = []
    if len(chain) != len(mats) + 1:
        bad.append("链长 %d ≠ 矩阵数 %d + 1" % (len(chain), len(mats)))
    for i, (a, b, c) in enumerate(chain):
        if b * b - 4 * a * c != D:
            bad.append("第 %d 个型判别式 %d ≠ %d" % (i, b * b - 4 * a * c, D))
    for i, M in enumerate(mats):
        if i + 1 >= len(chain):
            break
        det = M[0][0] * M[1][1] - M[0][1] * M[1][0]
        if abs(det) != 1:
            bad.append("第 %d 个矩阵 det=%d 不是单位" % (i, det))
        got = list(_apply_form(chain[i], M))
        if got != list(chain[i + 1]):
            bad.append("第 %d 步矩阵作用得 %r ≠ 链中 %r" % (i, tuple(got), tuple(chain[i + 1])))
    if chain and chain[-1] != red:
        bad.append("链末端 %r ≠ 声称规约型 %r" % (chain[-1], red))
    if not _form_reduced(red, D):
        bad.append("末端型 %r 不满足规约不等式 (|√D−2|a|| < b < √D)" % (red,))
    return (not bad), [float(len(bad))], (
        "链 %d 步: 判别式不变 + 单位矩阵 + 逐步作用复算 + 末端规约 全通过" % len(mats)
        if not bad else "; ".join(bad[:4]))


def _apply_form(f, M):
    """Recompute form f=(a,b,c) under the substitution M=[[p,q],[r,s]] (the discriminant stays
    invariant when det = ±1)."""
    a, b, c = f
    p, q = M[0]
    r, s = M[1]
    return (a * p * p + b * p * r + c * r * r,
            2 * a * p * q + b * (p * s + q * r) + 2 * c * r * s,
            a * q * q + b * q * s + c * s * s)


def _form_reduced(f, D) -> bool:
    """Reduction criterion for indefinite forms (exact-integer version):
    0 < b < √D and √D − b < 2|a| < √D + b."""
    a, b, c = f
    if b <= 0 or b * b >= D:
        return False
    m = abs(a)
    return (2 * m + b) ** 2 > D and (2 * m - b) ** 2 < D


# ---------------------------------------------------------------------------
# 4) q-expansion multiplication certificate (Kronecker substitution: big-integer
# multiplication, a different algorithm from schoolbook)
# ---------------------------------------------------------------------------


@register_checker("qexp_mul", independent_of=("mf_qexp_arith",))
@_safe
def _check_qexp_mul(payload):
    """q-expansion arithmetic: multiplication via Kronecker substitution (pack into big
    integers, multiply, unpack); the other modes are recomputed term by term."""
    mode = payload.get("mode", "mul")
    prec = int(payload["prec"])
    A = [int(x) for x in payload["a"]]
    B = [int(x) for x in payload["b"]]
    P = [int(x) for x in payload["product"]]
    if prec < 1 or len(A) != prec or len(B) != prec or len(P) != prec:
        return False, [], "精度与系数表长度不一致"
    if mode == "add":
        want = [A[i] + B[i] for i in range(prec)]
    elif mode == "sub":
        want = [A[i] - B[i] for i in range(prec)]
    elif mode == "scale":
        k = int(payload.get("scale_k", 0))
        want = [c * k for c in A]
    else:
        # packing width: guarantees no carry crosstalk (|coefficient| bound × number of terms).
        # Negative coefficients drag two's complement into the packing (a trap we hit): so A,B
        # are first split into positive − negative parts, the four non-negative convolutions are
        # each computed by Kronecker substitution, and the results are combined linearly --
        # still a different algorithm from schoolbook, but no longer affected by negatives.
        def kron(P_, Q_):
            mx = max([1] + P_) * max([1] + Q_) * prec
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

        Apos = [max(0, x) for x in A]
        Aneg = [max(0, -x) for x in A]
        Bpos = [max(0, x) for x in B]
        Bneg = [max(0, -x) for x in B]
        pp = kron(Apos, Bpos)
        pn = kron(Apos, Bneg)
        np_ = kron(Aneg, Bpos)
        nn = kron(Aneg, Bneg)
        want = [pp[i] - pn[i] - np_[i] + nn[i] for i in range(prec)]
    for i in range(prec):
        if want[i] != P[i]:
            return False, [1.0], "%s 独立复算第 %d 项 = %d ≠ 证书 %d" % (mode, i, want[i], P[i])
    return True, [0.0], "%s: 独立复算 %d 项 (q^%d) 一致" % (mode, prec, prec)


# ---------------------------------------------------------------------------
# 5) Hecke operator T_p certificate (scattering-style index derivation, the inverse
# direction of the generator's collection style)
# ---------------------------------------------------------------------------


@register_checker("hecke_tp_expand", independent_of=("mf_hecke_tp",))
@_safe
def _check_hecke_tp(payload):
    """T_p: independently recompute via the scattering form
    (T_p f)(q) = Σ a_{np}q^n + p^{k−1}Σ a_n q^{np}."""
    prec = int(payload["prec"])
    p = int(payload["p"])
    k = int(payload["k"])
    A = [int(x) for x in payload["coeffs"]]
    OUT = [int(x) for x in payload["out"]]
    if prec < 1 or len(A) != prec or len(OUT) != prec:
        return False, [], "精度与系数表长度不一致"
    if p < 2 or not _mr_small(p):
        return False, [], "p=%d 非素数" % p
    if k < 1:
        return False, [], "权 k >= 1"
    # scattering: start from all zeros and spread a_m onto b_{m/p} (times p^{k−1}) and b_{mp} (times 1)
    acc = [0] * (prec + 1)
    w = p ** (k - 1)
    for n in range(1, prec):
        if n % p == 0:
            acc[n // p] += A[n]
        m = n * p
        if m < prec:
            acc[m] += w * A[n]
    for n in range(1, prec):
        if acc[n] != OUT[n]:
            return False, [1.0], "散射式重算 b_%d = %d ≠ 证书 %d" % (n, acc[n], OUT[n])
    return True, [0.0], "散射式 (双 coset 定义) 重算 %d 项一致" % (prec - 1)


def _mr_small(n: int) -> bool:
    if n < 2:
        return False
    for q in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % q == 0:
            return n == q
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in (2, 3, 5, 7, 11):
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
# 6) elliptic-curve chord-tangent group-law certificate
# ---------------------------------------------------------------------------


def _inf(P) -> bool:
    """∞ representation: [] / ["O","O"] / ["O", ...] are all treated as the point at infinity
    (the generator uses [], older notations stay compatible)."""
    if isinstance(P, (list, tuple)):
        if len(P) == 0:
            return True
        if len(P) == 2 and str(P[0]) == "O":
            return True
    return False


@register_checker("ec_chord_backsub", independent_of=("ec_group_law",))
@_safe
def _check_ec_chord_backsub(payload):
    """Group law: three collinear points + Vieta root sum = λ² + curve back-substitution
    (a derivation different from the x_R formula)."""
    a, b = _rat(payload["a"]), _rat(payload["b"])
    P, Q, R = payload["P"], payload["Q"], payload["R"]
    lam = payload["lam"]
    case = payload.get("case", "")
    bad = []
    # the identity cases must be handled before parsing the points (the other addend may be
    # written ["O","O"], which _rat would reject)
    if case == "identity":
        if _inf(P) and not _inf(Q):
            ok = _same_pt(R, Q)
        elif _inf(Q) and not _inf(P):
            ok = _same_pt(R, P)
        else:
            ok = False
        return ok, [0.0 if ok else 1.0], (
            "恒等元加法: 结果 = 另一加数" if ok else "恒等元加法被证伪 (结果 ≠ 另一加数)")
    if _inf(R):
        if case == "inverse":
            xp, yp = _rat(P[0]), _rat(P[1])
            xq, yq = _rat(Q[0]), _rat(Q[1])
            ok = xp == xq and yp == -yq and _on_curve(a, b, xp, yp)
            return ok, [0.0 if ok else 1.0], "P + (−P) = O 且 P 在曲线上" if ok else "逆元加法被证伪"
        if case == "torsion_double":
            xp, yp = _rat(P[0]), _rat(P[1])
            ok = yp == 0 and _on_curve(a, b, xp, yp) and _same_pt(P, Q)
            return ok, [0.0 if ok else 1.0], "2·(2-挠点) = O (切线竖直)" if ok else "2-挠点倍点被证伪"
        return False, [], "∞ 结果但 case=%r 非法" % case
    xp, yp = _rat(P[0]), _rat(P[1])
    xq, yq = _rat(Q[0]), _rat(Q[1])
    xr, yr = _rat(R[0]), _rat(R[1])
    lb = _rat(lam) if lam not in (None, "") else None
    if not _on_curve(a, b, xr, yr):
        bad.append("R 不在曲线上 (y² ≠ x³+ax+b)")
    if lb is None:
        bad.append("非 ∞ 结果必须给出 λ")
    else:
        if _same_pt(P, Q):
            want = (3 * xp * xp + a) / (2 * yp) if yp != 0 else None
            if want is None or want != lb:
                bad.append("倍点切线斜率定义不符")
        else:
            if xp == xq:
                bad.append("x 相等却给非 ∞ 结果 (应为逆元或同点)")
            elif (yq - yp) / (xq - xp) != lb:
                bad.append("弦斜率定义不符")
        # the third intersection S = (x_R, y_P + λ(x_R − x_P)) must lie on the curve, and R = −S.
        # This single form covers chord, tangent, **inflection** (the tangent at (0,1) on
        # y²=x³+1 meets the curve with multiplicity three) and 2-torsion (S has y=0 => R = O),
        # so no per-case branching is needed.
        y_s = yp + lb * (xr - xp)
        if y_s * y_s != xr * xr * xr + a * xr + b:
            bad.append("第三交点 S=(x_R, y_line(x_R)) 不在曲线上")
        # R = −S; when y_S = 0, S is a 2-torsion point and −S = S => R = S (i.e. y_R = 0), not O.
        if y_s == 0:
            if yr != 0:
                bad.append("S 为 2-挠点 (y_S=0) ⇒ 应有 y_R = 0")
        elif yr != -y_s:
            bad.append("R ≠ −S (y_R ≠ −y_S)")
        if xp + xq + xr != lb * lb:
            bad.append("Vieta 根和失败: x_P+x_Q+x_R ≠ λ²")
    return (not bad), [float(len(bad))], (
        "λ 定义 + 共线 + Vieta 根和 + 曲线回代 全通过" if not bad else "; ".join(bad))


def _on_curve(a, b, x, y) -> bool:
    return y * y == x * x * x + a * x + b


def _same_pt(P, Q) -> bool:
    if _inf(P) or _inf(Q):
        return _inf(P) and _inf(Q)
    return _rat(P[0]) == _rat(Q[0]) and _rat(P[1]) == _rat(Q[1])


# ---------------------------------------------------------------------------
# 7) Nagell-Lutz torsion-walk certificate
# ---------------------------------------------------------------------------


@register_checker("torsion_walk", independent_of=("ec_nagell_lutz",))
@_safe
def _check_torsion_walk(payload):
    """Torsion walk: every step on the curve + distinct steps + n·P = O + Lutz necessary
    condition (integral coordinates, y²|D).

    Two assertions (distinguished by payload["completed"]) are checked separately:
      completed=True  : the walk ends at O and the first n−1 steps are distinct => the order is
                        exactly n, and n·P=O is re-checked with an own double-and-add;
      completed=False : no O within cap steps => a non-torsion claim (steps must still lie on the
                        curve; this is a "bounded negation", not a complete proof that the order
                        is infinite -- stated in detail).
    """
    from fractions import Fraction as Fr

    a, b = int(payload["a"]), int(payload["b"])
    P = payload["P"]
    order = int(payload["order"])
    steps = payload["steps"]
    completed = bool(payload.get("completed", order >= 1))
    bad = []
    Dc = -16 * (4 * a ** 3 + 27 * b ** 2)
    if Dc == 0:
        bad.append("判别式为 0 (奇异曲线, 群律不成立)")
    xp, yp = _rat(P[0]), _rat(P[1])
    if xp.denominator != 1 or yp.denominator != 1:
        bad.append("Nagell-Lutz: 扭点必须整点")
    else:
        y2 = int(yp) ** 2
        if y2 != 0 and Dc % y2 != 0:
            bad.append("Nagell-Lutz: y²=%d 不整除判别式 %d" % (y2, Dc))
    if not completed:
        cap = int(payload.get("order_cap", len(steps)))
        if len(steps) != cap:
            bad.append("非扭点走步数 %d ≠ cap %d (有界否定必须走满)" % (len(steps), cap))
        for i, St in enumerate(steps):
            if _inf(St):
                bad.append("第 %d 步即为 O ⇒ 其实是扭点, 与 non-torsion 主张矛盾" % (i + 1))
                continue
            if not _on_curve(Fr(a), Fr(b), _rat(St[0]), _rat(St[1])):
                bad.append("第 %d 步不在曲线上" % (i + 1))
        return (not bad), [float(len(bad))], (
            "非扭点 (有界否定): %d 步内无 O, 逐步在曲线, Lutz 必要条件=%s"
            % (len(steps), not any("Lutz" in x for x in bad)) if not bad else "; ".join(bad[:4]))
    if len(steps) != order:
        bad.append("步数 %d ≠ 声称阶 %d" % (len(steps), order))
    seen = set()
    for i, St in enumerate(steps):
        if _inf(St):
            if i != len(steps) - 1:
                bad.append("第 %d 步即为 O (阶应更小)" % (i + 1))
            continue
        x, y = _rat(St[0]), _rat(St[1])
        if not _on_curve(Fr(a), Fr(b), x, y):
            bad.append("第 %d 步不在曲线上" % (i + 1))
        k = (x, y)
        if k in seen:
            bad.append("第 %d 步与更早步重复 ⇒ 阶 < %d" % (i + 1, order))
        seen.add(k)
    if steps and not _inf(steps[-1]):
        bad.append("末步非 O ⇒ 阶 ≥ %d 未被证" % order)
    got = _scalar_mul(a, b, (xp, yp), order)
    if got is not None:
        bad.append("自有 double-and-add 算得 %d·P = %r ≠ O" % (order, got))
    return (not bad), [float(len(bad))], (
        "阶 %d: 逐步在曲线 + 互异 + %d·P = O + Lutz 必要条件 通过" % (order, order)
        if not bad else "; ".join(bad[:4]))


def _scalar_mul(a, b, P, n):
    """Own double-and-add scalar multiplication (returns None for O)."""
    from fractions import Fraction as Fr

    res = None
    add = P
    while n > 0:
        if n & 1:
            res = _ec_add(a, b, res, add)
        add = _ec_add(a, b, add, add)
        n >>= 1
    return res


def _ec_add(a, b, P, Q):
    from fractions import Fraction as Fr

    if P is None:
        return Q
    if Q is None:
        return P
    x1, y1 = P
    x2, y2 = Q
    if x1 == x2 and y1 == -y2:
        return None
    if P == Q:
        if y1 == 0:
            return None
        lam = (3 * x1 * x1 + a) / (2 * y1)
    else:
        lam = (y2 - y1) / (x2 - x1)
    x3 = lam * lam - x1 - x2
    return (x3, lam * (x1 - x3) - y1)


# ---------------------------------------------------------------------------
# 8) small-field point-count certificate
# ---------------------------------------------------------------------------


@register_checker("pointcount_small", independent_of=("curve_point_count_small",))
@_safe
def _check_pointcount_small(payload):
    """Point count: own Legendre-sum recomputation (Euler criterion) + Hasse bound (exact
    integers) + non-singularity."""
    a, b, p = int(payload["a"]), int(payload["b"]), int(payload["p"])
    count = int(payload["count"])
    bad = []
    if p < 3 or not _mr_small(p):
        bad.append("p=%d 必须是奇素数" % p)
    if (4 * a ** 3 + 27 * b ** 2) % p == 0:
        bad.append("4a³+27b² ≡ 0 (mod p) ⇒ 奇异曲线")
    affine = 0
    legendre_sum = 0
    for x in range(p):
        rhs = (x * x * x + a * x + b) % p
        if rhs == 0:
            affine += 1
            ls = 0
        else:
            e = pow(rhs, (p - 1) // 2, p)
            ls = 1 if e == 1 else -1
            affine += 2 if ls == 1 else 0
        legendre_sum += ls
    if affine + 1 != count:
        bad.append("仿射点数重算 +1 = %d ≠ 证书 %d" % (affine + 1, count))
    if payload.get("legendre_sum") is not None and int(payload["legendre_sum"]) != legendre_sum:
        bad.append("Legendre 和重算 %d ≠ 证书 %d" % (legendre_sum, int(payload["legendre_sum"])))
    if (count - p - 1) ** 2 > 4 * p:
        bad.append("Hasse 界被违反: |N − (p+1)|² = %d > 4p = %d" % ((count - p - 1) ** 2, 4 * p))
    return (not bad), [float(len(bad))], (
        "Legendre 和 %d, N=%d (p+1%s%d), Hasse 界内" % (legendre_sum, count, "+" if count >= p + 1 else "-", abs(count - p - 1))
        if not bad else "; ".join(bad))


# ---------------------------------------------------------------------------
# 9) Q(√d) arithmetic certificate (regular representation = embedding into the 2×2 matrix
# ring, a different path from the direct formula)
# ---------------------------------------------------------------------------


@register_checker("qsqrtd_arith", independent_of=("v31_qsqrtd_arith",))
@_safe
def _check_qsqrtd_arith(payload):
    """Q(√d): recompute with the **regular representation** [[a, bd],[b, a]] as matrix operations
    (a different path from expanding (a+bi)(c+ei))."""
    d = int(payload["d"])
    op = payload["op"]
    bad = []
    if d <= 1 or not _isfree_sq(d):
        bad.append("d=%d 必须无平方且 > 1" % d)
    ops = payload["operands"]

    def reg(x):
        """Regular representation [[a, bd],[b, a]] (multiplication matrix of element x)."""
        return [[x[0], x[1] * d], [x[1], x[0]]]

    def from_reg(M):
        return (M[0][0], M[1][0])

    res = payload["result"]
    rr = (_rat(res[0]), _rat(res[1]))
    if op in ("mul", "add", "sub"):
        if len(ops) != 2:
            return False, [], "二元运算需要两个操作数"
        A, B = (_rat(ops[0][0]), _rat(ops[0][1])), (_rat(ops[1][0]), _rat(ops[1][1]))
        if op == "add":
            want = (A[0] + B[0], A[1] + B[1])
        elif op == "sub":
            want = (A[0] - B[0], A[1] - B[1])
        else:
            MA, MB = reg(A), reg(B)
            MC = [[MA[0][0] * MB[0][0] + MA[0][1] * MB[1][0],
                   MA[0][0] * MB[0][1] + MA[0][1] * MB[1][1]],
                  [MA[1][0] * MB[0][0] + MA[1][1] * MB[1][0],
                   MA[1][0] * MB[0][1] + MA[1][1] * MB[1][1]]]
            if MC[0][1] != MC[1][0] * d:
                bad.append("矩阵乘法结果不是 Q(√d) 的型 (表示不一致)")
            want = from_reg(MC)
        if want != rr:
            bad.append("%s 矩阵/向量复算 %r ≠ 证书 %r" % (op, want, rr))
    elif op == "norm":
        if len(ops) != 1:
            return False, [], "norm 需要一个操作数"
        A = (_rat(ops[0][0]), _rat(ops[0][1]))
        want = A[0] * A[0] - d * A[1] * A[1]
        if want != rr[0] or rr[1] != 0:
            bad.append("范数复算 %s ≠ 证书 %r" % (want, rr))
    elif op == "inv":
        if len(ops) != 1:
            return False, [], "inv 需要一个操作数"
        A = (_rat(ops[0][0]), _rat(ops[0][1]))
        n = A[0] * A[0] - d * A[1] * A[1]
        if n == 0:
            return False, [], "范数为 0 不可逆"
        want = (A[0] / n, -A[1] / n)
        if want != rr:
            bad.append("逆元复算 %r ≠ 证书 %r" % (want, rr))
        chk = (A[0] * rr[0] + d * A[1] * rr[1], A[0] * rr[1] + A[1] * rr[0])
        if chk != (1, 0):
            bad.append("逆元回乘 ≠ 1")
    else:
        return False, [], "未知 op %r" % op
    return (not bad), [float(len(bad))], (
        "Q(√%d) 正规表示复算一致 (op=%s)" % (d, op) if not bad else "; ".join(bad))


CHECKER_KINDS = (
    "cf_convergent", "pell_backsub", "form_reduce_chain", "qexp_mul", "hecke_tp_expand",
    "ec_chord_backsub", "torsion_walk", "pointcount_small", "qsqrtd_arith",
)
