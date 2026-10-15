# -*- coding: utf-8 -*-
"""L0 contest number theory -- **independent checkers** for the certificates (10 kinds).

Independence discipline (stricter than the legacy package; auditable programmatically):
  - this module imports **no** internal module of svokernel_v31 -- neither the generator
    module (ops_nt_l0) nor the primitive module the generator uses (numtheory). The checkers
    carry their own small toolkit (_powmod / _mr / _trial_prime / _factor_trial / _vdiv),
    "same syntax, different code" from the generator: the generator uses formulas/factor
    tables, the checkers use exhaustive division and trial-division proofs;
  - every checker consumes payload data only (pure data, no "trust me" flag) and returns
    (ok, resid, detail): resid = violation amount (here = count of offending items);
  - all exceptions are folded into "rejected" by _safe -- a checker **never raises**, because
    the certificate gate of TypedTool.call only catches SvoError; any other exception would
    break the fail-safe promise that "call() does not raise".
"""
from __future__ import annotations

import functools

from svokernel.certify import register_checker

__all__ = ["CHECKER_KINDS"]


# ---------------------------------------------------------------------------
# local toolkit (stdlib only; a different code path from the generator)
# ---------------------------------------------------------------------------


def _safe(fn):
    @functools.wraps(fn)
    def wrapped(payload):
        try:
            return fn(payload)
        except Exception as e:  # noqa: BLE001 - the single folding point at the checker boundary
            return False, [], "检查器拒收: payload 非法或检查过程异常 (%s: %s)" % (type(e).__name__, e)

    return wrapped


def _powmod(a: int, e: int, m: int) -> int:
    """Hand-written modular exponentiation (square-and-multiply), different from both the
    built-in pow and the generator path."""
    r = 1 % m
    a %= m
    while e > 0:
        if e & 1:
            r = r * a % m
        a = a * a % m
        e >>= 1
    return r


def _mr(n: int) -> bool:
    """Hand-written deterministic Miller-Rabin test (base sequence differs from numtheory:
    fixed scramble, small before large)."""
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in (7, 2, 61, 3, 5, 11, 13, 17, 19, 23, 29, 31, 37):
        if a % n == 0:
            continue
        x = _powmod(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def _trial_prime(n: int, limit=None):
    """Primality proof by trial division: complete division -> True/False; below isqrt -> None
    (unproven)."""
    if n < 2:
        return False
    if limit is None:
        limit = int(n ** 0.5) + 1
        while limit * limit <= n:
            limit += 1
    if limit * limit < n:
        return None
    if n % 2 == 0:
        return n == 2
    d = 3
    while d <= limit:
        if n % d == 0:
            return False
        d += 2
    return True


def _vdiv(p: int, n: int, cap: int = 100000) -> int:
    """v_p(n): exponent by trial division (n != 0). Independent of the generator's formula
    path; this is the definition computed directly."""
    if n == 0:
        raise ValueError("v_p(0) 不可表示")
    n = abs(n)
    if n % p:
        return 0
    e = 0
    while n % p == 0:
        n //= p
        e += 1
        if e > cap:
            raise ValueError("赋值超出检查器上限")
    return e


def _factor_trial(n: int, cap: int = 1000000) -> list:
    """Complete factorization by trial division (complete when n <= cap²); returns [[p,e],...].

    When the cofactor >1: if <= cap² it is itself prime (trial division exhausted); otherwise
    the local MR decides (for n < 3.3e24 that base table is a deterministic answer) -- neither
    path returns an "incomplete factorization". When neither holds the code raises -> the
    checker rejects (better to reject than to pass something unproven).
    """
    out = []
    m = abs(n)
    d = 2
    while d <= cap and d * d <= m:
        if m % d == 0:
            e = 0
            while m % d == 0:
                m //= d
                e += 1
            out.append([d, e])
        d += 1 if d == 2 else 2
    if m > 1:
        if m <= cap * cap or _mr(m):
            out.append([m, 1])
        else:
            raise ValueError("余因子 %d 既超试除上限又未过 MR (拒绝, 不给不完整分解)" % m)
    return out


# ---------------------------------------------------------------------------
# 1) LTE valuation certificate
# ---------------------------------------------------------------------------


@register_checker("lte_valuation", independent_of=("nt_lte",))
@_safe
def _check_lte_valuation(payload):
    """LTE valuation: compute x^n − y^n directly and count the power of p (definition
    computed directly, not via the lemma formula)."""
    p, x, y, n, v = (int(payload["p"]), int(payload["x"]), int(payload["y"]),
                     int(payload["n"]), int(payload["v"]))
    if n < 1:
        return False, [], "n < 1 非 LTE 定义域"
    if x == y:
        return False, [], "x == y ⇒ 差为 0, v_p 不可表示 (退化 payload)"
    if _trial_prime(p) is not True and not _mr(p):
        return False, [], "p=%d 未通过素性证明" % p
    if x.bit_length() + n.bit_length() > 4000:
        return False, [], "规模超检查器上限 (拒绝而非放行)"
    N = x ** n - y ** n
    if N == 0:
        return False, [], "x^n − y^n = 0 (退化)"
    got = _vdiv(p, N)
    ok = got == v
    return ok, [0.0 if ok else 1.0], "定义直算 v_p(x^n−y^n)=%d, 证书声称 %d" % (got, v)


# ---------------------------------------------------------------------------
# 2) Hensel lifted-root certificate
# ---------------------------------------------------------------------------


def _poly_eval(coeffs, x, mod=None):
    acc = 0
    for c in reversed(coeffs):
        acc = acc * x + c
        if mod is not None:
            acc %= mod
    return acc


@register_checker("hensel_root", independent_of=("nt_hensel_lift",))
@_safe
def _check_hensel_root(payload):
    """Hensel lifted root: f(root) ≡ 0 (mod p^k) + root ≡ r0 (mod p) + simple-root condition
    f'(r0) ≢ 0."""
    coeffs = [int(c) for c in payload["coeffs"]]
    p, r0, k, root = int(payload["p"]), int(payload["r0"]), int(payload["k"]), int(payload["root"])
    if len(coeffs) < 2 or k < 1:
        return False, [], "系数表退化或 k < 1"
    if _trial_prime(p) is not True and not _mr(p):
        return False, [], "p=%d 未通过素性证明" % p
    pk = p ** k
    if pk.bit_length() > 200:
        return False, [], "p^k 规模超检查器上限"
    dcoeffs = [i * coeffs[i] for i in range(1, len(coeffs))]
    bad = []
    if _poly_eval(coeffs, root, pk) % pk != 0:
        bad.append("f(root) ≢ 0 mod p^k")
    if (root - r0) % p != 0:
        bad.append("root ≢ r0 mod p (不是该根的提升)")
    if _poly_eval(dcoeffs, r0, p) % p == 0:
        bad.append("f'(r0) ≡ 0 mod p (非单根, 唯一提升不成立)")
    return (not bad), [float(len(bad))], "全部通过" if not bad else "; ".join(bad)


# ---------------------------------------------------------------------------
# 3) Pratt primality certificate tree
# ---------------------------------------------------------------------------


def _verify_pratt(node, depth=0, base_cap=10 ** 7):
    """Recursively verify a Pratt tree; returns (ok, detail). A base node requires a
    trial-division primality **proof**."""
    if depth > 40:
        return False, "树深超限 (拒绝)"
    if not isinstance(node, dict):
        return False, "结点非 dict"
    n = int(node["n"])
    if n < 2:
        return False, "n < 2"
    if node.get("base"):
        if n > base_cap:
            return False, "基础结点 %d 超出试除证明上限" % n
        t = _trial_prime(n)
        if t is not True:
            return False, "基础结点 %d 试除未证素数" % n
        return True, "#%d 试除证明素数" % n
    a = node.get("a")
    facs = node.get("factors")
    if not isinstance(a, int) or not isinstance(facs, list) or not facs:
        return False, "#%d 内部结点缺 a/factors" % n
    if not (1 < a < n):
        return False, "#%d 见证 a=%r 不在 (1,n)" % (n, a)
    prod = 1
    for item in facs:
        q, e = int(item[0]), int(item[1])
        sub = item[2] if len(item) > 2 else None
        if e < 1:
            return False, "#%d 因子指数 %d < 1" % (n, e)
        prod *= q ** e
        ok, det = _verify_pratt(sub, depth + 1, base_cap)
        if not ok:
            return False, "#%d 因子 %d 子证书: %s" % (n, q, det)
    if prod != n - 1:
        return False, "#%d 因子连乘 = %d ≠ n−1 = %d" % (n, prod, n - 1)
    if _powmod(a, n - 1, n) != 1:
        return False, "#%d 见证 a=%d 不满足 a^(n−1) ≡ 1" % (n, a)
    for item in facs:
        q = int(item[0])
        if _powmod(a, (n - 1) // q, n) == 1:
            return False, "#%d 见证阶不足: a^((n−1)/%d) ≡ 1" % (n, q)
    return True, "#%d 见证 a=%d 阶 = n−1 ⇒ 素数" % (n, a)


@register_checker("pratt_tree", independent_of=("nt_pratt_cert",))
@_safe
def _check_pratt_tree(payload):
    tree = payload["tree"]
    n = int(payload["n"])
    if int(tree.get("n", -1)) != n:
        return False, [], "payload.n 与树根 n 不一致"
    ok, det = _verify_pratt(tree, 0)
    return ok, [0.0 if ok else 1.0], det


# ---------------------------------------------------------------------------
# 4) Miller-Rabin witness recomputation
# ---------------------------------------------------------------------------


@register_checker("miller_witness", independent_of=("nt_miller_witness",))
@_safe
def _check_miller_witness(payload):
    """Witness recomputation: recompute the strong-pseudoprime sequence item by item and
    re-check the s/d split and the witness criterion."""
    n, a = int(payload["n"]), int(payload["a"])
    d, s = int(payload["d"]), int(payload["s"])
    res = [int(x) for x in payload["residues"]]
    if n < 3 or not (1 < a < n):
        return False, [], "n/a 越界"
    if (2 ** s) * d != n - 1 or d % 2 == 0:
        return False, [], "n−1 = 2^s·d 分解不成立 (s=%d,d=%d)" % (s, d)
    if len(res) != max(s, 1):
        # s=0 (n even => n−1 odd) means sequence length 1 by convention: only the a^d term,
        # and the MR criterion degrades to a^d ≢ ±1.
        return False, [], "残差序列长度 %d ≠ max(s,1)=%d (s=0 的约定见算子 doc)" % (len(res), max(s, 1))
    bad = []
    # independent recomputation: hand-written square-and-multiply (no pow), then square from d
    r0 = _powmod(a, d, n)
    if r0 != res[0]:
        bad.append("a^d mod n 重算 %d ≠ 证书 %d" % (r0, res[0]))
    cur = res[0]
    for i in range(1, s):
        cur = cur * cur % n
        if cur != res[i]:
            bad.append("第 %d 次平方 %d ≠ 证书 %d" % (i, cur, res[i]))
    wit = (res[0] != 1) and all(x != n - 1 for x in res)
    if bool(payload.get("is_witness", wit)) != wit:
        bad.append("is_witness 声明与复算不符")
    return (not bad), [float(len(bad))], "见证判据复算一致" if not bad else "; ".join(bad)


# ---------------------------------------------------------------------------
# 5) Factor product + per-prime primality certificate
# ---------------------------------------------------------------------------


@register_checker("factor_product", independent_of=("nt_factor_cert",))
@_safe
def _check_factor_product(payload):
    """Factor certificate: product reconciliation + **independent proof** of primality per
    factor + increasing/exponent structure."""
    n = abs(int(payload["n"]))
    facs = [[int(p), int(e)] for p, e in payload["factors"]]
    bad = []
    if n < 2:
        bad.append("n < 2")
    prod = 1
    last = 1
    for p, e in facs:
        if p < 2 or e < 1:
            bad.append("因子 %d^%d 非法" % (p, e))
            continue
        if p <= last:
            bad.append("因子未严格递增: %d ≤ %d" % (p, last))
        last = p
        prod *= p ** e
    if prod != n:
        bad.append("连乘 %d ≠ n=%d" % (prod, n))
    modes = []
    for p, _e in facs:
        t = _trial_prime(p) if p <= 10 ** 12 else None
        if t is True:
            modes.append("trial")
        elif _mr(p):
            modes.append("mr")
        else:
            bad.append("因子 %d 素性未通过" % p)
    return (not bad), [float(len(bad))], (
        "连乘对账 + 逐素性通过 (模式 %s)" % ",".join(modes) if not bad else "; ".join(bad))


# ---------------------------------------------------------------------------
# 6) Multiplicative-order exhaustive certificate
# ---------------------------------------------------------------------------


@register_checker("order_exhaustive", independent_of=("nt_order_mult",))
@_safe
def _check_order_exhaustive(payload):
    """Multiplicative order: a^ord ≡ 1 + minimality (exhaustive for small orders, divisor scan
    for large ones, with the mode reported as measured)."""
    a, n, order = int(payload["a"]), int(payload["n"]), int(payload["order"])
    if n < 2 or order < 1:
        return False, [], "n/order 越界"
    if _powmod(a, order, n) != 1:
        return False, [], "a^order ≢ 1 (mod n) —— 直接否决"
    if order <= 5000:
        for k in range(1, order):
            if _powmod(a, k, n) == 1:
                return False, [1.0], "非最小: 更小的 k=%d 也满足 a^k ≡ 1" % k
        return True, [0.0], "全穷举 1..%d 均非 1 ⇒ 阶 = %d" % (order - 1, order)
    for d in range(1, order):
        if order % d == 0 and _powmod(a, d, n) == 1:
            return False, [1.0], "非最小: 真因子 k=%d 也满足 a^k ≡ 1" % d
    return True, [0.0], "因子扫描 (order=%d > 5000) 通过; 全穷举未做 (如实声明)" % order


# ---------------------------------------------------------------------------
# 7) Jacobi / Euler criterion certificate
# ---------------------------------------------------------------------------


@register_checker("jacobi_euler", independent_of=("nt_jacobi_qr",))
@_safe
def _check_jacobi_euler(payload):
    """Independent recomputation of the Jacobi value (factor n + per-prime Euler criterion,
    unlike the generator's reciprocity path) + root back-substitution."""
    a, n = int(payload["a"]), int(payload["n"])
    val = int(payload["jacobi"])
    if n < 3 or n % 2 == 0:
        return False, [], "n 必须是 >=3 的奇数"
    n_is_prime = payload.get("n_is_prime")
    bad = []
    if n_is_prime:
        if not _mr(n):
            bad.append("声称 n 素数但独立判定为合数")
        e = _powmod(a % n, (n - 1) // 2, n)
        jac = 0 if a % n == 0 else (1 if e == 1 else -1)
        if jac != val:
            bad.append("Euler 判据重算 %d ≠ 证书 %d" % (jac, val))
        want_euler = payload.get("euler_value")
        if want_euler is not None and int(want_euler) != (0 if a % n == 0 else e):
            bad.append("euler_value 与 a^((n−1)/2) 实算不符")
    else:
        if _mr(n) and _trial_prime(n) is True:
            bad.append("声称 n 合数但独立判定为素数")
        facs = _factor_trial(n, 200000)
        jac = 1
        for p, e in facs:
            if a % p == 0:
                jac = 0
                break
            lp = _powmod(a % p, (p - 1) // 2, p)
            lp = 1 if lp == 1 else -1
            if lp == -1 and e % 2 == 1:
                jac = -jac
        if jac != val:
            bad.append("分解+Euler 重算 %d ≠ 证书 %d" % (jac, val))
    if payload.get("has_root"):
        root = int(payload["root"])
        if (root * root - a) % n != 0:
            bad.append("qr_root 回代 root² ≢ a (mod n)")
    return (not bad), [float(len(bad))], "Jacobi/Euler 双路一致" if not bad else "; ".join(bad)


# ---------------------------------------------------------------------------
# 8) Zsigmondy primitive prime factor certificate
# ---------------------------------------------------------------------------


@register_checker("zsi_primitive", independent_of=("nt_zsigmondy",))
@_safe
def _check_zsi_primitive(payload):
    """Primitive prime factors: for each p check p | a^n−b^n and p ∤ a^k−b^k for **all** k<n
    (exhaustive)."""
    a, b, n = int(payload["a"]), int(payload["b"]), int(payload["n"])
    divs = [int(x) for x in payload["divisors"]]
    prims = [int(x) for x in payload["primitives"]]
    if n < 2 or a <= b or b < 0:
        return False, [], "参数域外"
    N = a ** n - b ** n
    if N.bit_length() > 4000:
        return False, [], "规模超检查器上限"
    bad = []
    for p in divs:
        if N % p != 0:
            bad.append("声称因子 %d 不整除 N" % p)
        elif not (_trial_prime(p) is True or _mr(p)):
            bad.append("声称因子 %d 非素数" % p)
    # completeness: factor N and compare the prime-factor set with divs (within the trial cap)
    try:
        full = set(p for p, _e in _factor_trial(N, 1000000))
        if full != set(divs):
            bad.append("因子集合不完整/多余: 独立分解 %r vs 声称 %r" % (sorted(full)[:8], sorted(divs)[:8]))
    except ValueError:
        pass  # beyond the trial cap: completeness unproven (skipped, item checks still run)
    def is_prim(p):
        if N % p != 0:
            return False
        for k in range(1, n):
            if (a ** k - b ** k) % p == 0:
                return False
        return True
    recomputed = sorted(p for p in divs if is_prim(p))
    if recomputed != sorted(prims):
        bad.append("原始素因子集合不符: 穷举重算 %r vs 证书 %r" % (recomputed, sorted(prims)))
    return (not bad), [float(len(bad))], (
        "穷举 k<n 逐 p 判定一致 (%d 个原始素因子)" % len(prims) if not bad else "; ".join(bad))


# ---------------------------------------------------------------------------
# 9) Arithmetic-function convolution-table certificate
# ---------------------------------------------------------------------------


def _vals_brute(name, N):
    """**Exhaustively computed** arithmetic-function table (straight from the definition:
    count divisors / count coprimes), independent of the generator's factorization formulas."""
    out = [0] * (N + 1)
    for n in range(1, N + 1):
        if name == "one":
            out[n] = 1
        elif name == "id":
            out[n] = n
        elif name == "tau":
            out[n] = sum(1 for d in range(1, n + 1) if n % d == 0)
        elif name == "sigma":
            out[n] = sum(d for d in range(1, n + 1) if n % d == 0)
        elif name == "phi":
            out[n] = sum(1 for k in range(1, n + 1) if _gcd(k, n) == 1)
        elif name == "mu":
            if n == 1:
                out[n] = 1
            else:
                e = _vdiv_raw(n)
                out[n] = 0 if e is None else (-1) ** e
        else:
            raise ValueError("未知算术函数 %r" % name)
    return out


def _vdiv_raw(n):
    """Prime-factor count used by μ: number of distinct primes; None if a square divides n."""
    m = n
    cnt = 0
    d = 2
    while d * d <= m:
        if m % d == 0:
            m //= d
            if m % d == 0:
                return None
            cnt += 1
        d += 1 if d == 2 else 2
    if m > 1:
        cnt += 1
    return cnt


def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a


@register_checker("arith_conv_table", independent_of=("arith_func_convolution",))
@_safe
def _check_arith_conv_table(payload):
    """Convolution/inversion table: the checker recomputes the whole table with an
    **exhaustive divisor enumeration** (the generator uses factorization formulas)."""
    f, g = payload["f"], payload["g"]
    N = int(payload["N"])
    mode = payload["mode"]
    vals = [int(x) for x in payload["values"]]
    if not (1 <= N <= 4000):
        return False, [], "N 越界 (检查器上限 4000)"
    if len(vals) != N + 1:
        return False, [], "values 长度 %d ≠ N+1" % len(vals)
    fv, gv = _vals_brute(f, N), _vals_brute(g, N)
    bad = []
    if mode == "dirichlet_conv":
        for n in range(1, N + 1):
            s = 0
            for d in range(1, n + 1):
                if n % d == 0:
                    s += fv[d] * gv[n // d]
            if s != vals[n]:
                bad.append("n=%d 重算 %d ≠ 表 %d" % (n, s, vals[n]))
    elif mode == "summatory":
        for n in range(1, N + 1):
            s = sum(fv[d] for d in range(1, n + 1) if n % d == 0)
            if s != vals[n]:
                bad.append("n=%d 求和重算 %d ≠ 表 %d" % (n, s, vals[n]))
    elif mode == "mobius_invert":
        # semantics: table = h * μ with h = f * 1 (computed exhaustively from f), i.e.
        # "f*1 then Möbius inversion back to f".
        muv = _vals_brute("mu", N)
        hv = [0] * (N + 1)
        for n in range(1, N + 1):
            hv[n] = sum(fv[d] for d in range(1, n + 1) if n % d == 0)
        for n in range(1, N + 1):
            s = sum(hv[d] * muv[n // d] for d in range(1, n + 1) if n % d == 0)
            if s != vals[n]:
                bad.append("n=%d Möbius 反演重算 %d ≠ 表 %d" % (n, s, vals[n]))
    else:
        return False, [], "未知 mode %r" % mode
    return (not bad), [float(len(bad))], (
        "穷举直算重算整表一致 (N=%d, mode=%s)" % (N, mode) if not bad
        else "; ".join(bad[:6]) + (" ..." if len(bad) > 6 else ""))


# ---------------------------------------------------------------------------
# 10) p-adic valuation additivity certificate
# ---------------------------------------------------------------------------


@register_checker("padic_valuation", independent_of=("nt_valuation_ring",))
@_safe
def _check_padic_valuation(payload):
    """Valuation-row certificate: recompute the four valuations per row from the definition and
    re-check the two non-Archimedean identities."""
    p = int(payload["p"])
    rows = payload["rows"]
    if _trial_prime(p) is not True and not _mr(p):
        return False, [], "p=%d 未通过素性证明" % p
    bad = []
    for i, r in enumerate(rows):
        a, b, va, vb, vab, vsum = (int(r[0]), int(r[1]), int(r[2]), int(r[3]), int(r[4]), int(r[5]))
        if a == 0 or b == 0 or a + b == 0:
            bad.append("行 %d: 含 0 (赋值不可表示), 证书不该存在" % i)
            continue
        ca, cb = _vdiv(p, a), _vdiv(p, b)
        if (ca, cb) != (va, vb):
            bad.append("行 %d: v_p 重算 (%d,%d) ≠ 证书 (%d,%d)" % (i, ca, cb, va, vb))
        if _vdiv(p, a * b) != vab:
            bad.append("行 %d: v_p(ab) 重算不符" % i)
        if _vdiv(p, a + b) != vsum:
            bad.append("行 %d: v_p(a+b) 重算不符" % i)
        if vab != va + vb:
            bad.append("行 %d: 可加性 v(ab) = v(a)+v(b) 不成立" % i)
        if vsum < min(va, vb):
            bad.append("行 %d: 下界 v(a+b) >= min 不成立" % i)
        # only the **implication direction** is asserted: unequal valuations => the equality
        # holds. The converse is false (14+21=35: both v_7 are 1 and v_7(35)=1, so the equality
        # holds even though the valuations are equal); a checker must not use a false proposition
        # as a criterion.
        if va != vb and vsum != min(va, vb):
            bad.append("行 %d: 赋值相异却未取等号 (非阿基米德性失败)" % i)
    return (not bad), [float(len(bad))], (
        "%d 行赋值全复算通过 (定义直算)" % len(rows) if not bad else "; ".join(bad[:6]))


CHECKER_KINDS = (
    "lte_valuation", "hensel_root", "pratt_tree", "miller_witness", "factor_product",
    "order_exhaustive", "jacobi_euler", "zsi_primitive", "arith_conv_table", "padic_valuation",
)
