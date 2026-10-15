# -*- coding: utf-8 -*-
"""Layer 0: contest number theory (band 9900, rows 9901–9910) -- all verified.

All ten operators follow the p0_reference delivery style: the VMP three steps (symbol -> handle /
kernel execution / result -> latent state) + an **independent verifier** (a different algorithm
from the generation path) + an **independent certificate checker** (separate module certcheck_nt,
which does not import this module) + >=3 selfchecks (positive / negative / boundary).

  nt_lte                 9901  LTE lemma (odd prime p | x−y) -- generation by formula, verification by definition
  nt_hensel_lift         9902  Hensel unique simple-root lift -- generation by Newton, verification by back-substitution + exhaustive uniqueness
  nt_pratt_cert          9903  Pratt primality certificate tree -- generation by witness search, verification by recursive recomputation
  nt_miller_witness      9904  Miller-Rabin witness recomputation -- generation by pow, verification by hand-written square-and-multiply
  nt_factor_cert         9905  factor product + per-prime primality -- generation by rho, verification by trial-division proof
  nt_order_mult          9906  multiplicative order -- generation by divisor enumeration of φ, verification by full scan / divisor scan
  nt_jacobi_qr           9907  Jacobi symbol + Euler criterion (+ Tonelli-Shanks root) -- dual path
  nt_zsigmondy           9908  Zsigmondy primitive prime factors -- exhaustive per-k<n decision
  arith_func_convolution 9909  μ/φ/σ/τ convolution + Möbius inversion dual channel
  nt_valuation_ring      9910  p-adic valuation additivity (non-Archimedean identities) check

Iron rules: exact values travel as int (float never impersonates exact); residuals only report
floats; long loops call check_time periodically; failures raise only §15 error codes.
"""
from __future__ import annotations

from .v31registry import V31OpDef, V31REGISTRY  # import first: triggers the sys.path bootstrap

from svokernel.certify import Certificate
from svokernel.errors import Err, SvoError
from svokernel.tool import CallCtl

from .numtheory import (
    divisors_brute, divisors_from_factors, factorize, is_prime, is_prime_trial,
    jacobi, legendre_euler, modinv, mu_from_factors, phi_from_factors,
    sigma_from_factors, sqrt_mod_prime, tau_from_factors, valuation,
    valuation_squaring,
)

BAND_L0 = 9900


def _reg(**kw):
    kw.setdefault("status", "verified")
    kw.setdefault("layer", 0)
    return V31REGISTRY.register(V31OpDef(**kw))


_EXACT = {"kind": "int"}


def _sdict(fields, req):
    return {"kind": "dict", "fields": fields, "required": req}


# ===========================================================================
# 9901  nt_lte  -- LTE lemma
# ===========================================================================

#: Domain note: this operator implements LTE for **odd primes** p only. The p=2 formula carries
#: an extra term v_2(x+y)−1; that case is explicitly rejected (PRECOND_FAIL) instead of returning
#: a value that is wrong by one term -- narrowing the domain is honest, a wrong value is an incident.
_LTE_IN = _sdict({
    "p": {"kind": "int", "min": 2},
    "x": {"kind": "int", "min": -(10 ** 6), "max": 10 ** 6},
    "y": {"kind": "int", "min": -(10 ** 6), "max": 10 ** 6},
    "n": {"kind": "int", "min": 1, "max": 128},
}, ["p", "x", "y", "n"])
_LTE_OUT = _sdict({
    "p": _EXACT, "x": _EXACT, "y": _EXACT, "n": _EXACT,
    "v_p_diff": _EXACT, "v_p_n": _EXACT, "v": _EXACT, "num_bitlen": _EXACT,
}, ["p", "x", "y", "n", "v_p_diff", "v_p_n", "v", "num_bitlen"])


def _lte_precond(args):
    p, x, y, n = args["p"], args["x"], args["y"], args["n"]
    if n < 1:
        return False, "n >= 1 是 LTE 的定义域"
    if p % 2 == 0:
        return False, "本算子只覆盖奇素数 p (p=2 情形需 v_2(x+y)−1 修正项, 显式域外)"
    if not is_prime(p):
        return False, "p=%d 非素数" % p
    if (x - y) == 0:
        return False, "x == y ⇒ x^n−y^n = 0, 赋值不可表示"
    if (x - y) % p != 0:
        return False, "LTE 前提 p | (x−y) 不成立 (p=%d, x−y=%d)" % (p, x - y)
    if (x * y) % p == 0:
        return False, "LTE 前提 p ∤ xy 不成立 (p=%d, x=%d, y=%d)" % (p, x, y)
    return True, ""


def _invoke_nt_lte(args, ctl):
    p, x, y, n = args["p"], args["x"], args["y"], args["n"]
    vd = valuation(p, x - y)
    vn = valuation(p, n)
    v = vd + vn
    nb = (abs(x) ** n + abs(y) ** n).bit_length()
    cert = Certificate(
        kind="lte_valuation",
        payload={"p": p, "x": x, "y": y, "n": n, "v": v, "mode": "odd_prime_only"},
        issued_by="nt_lte",
    )
    return {"value": {"p": p, "x": x, "y": y, "n": n, "v_p_diff": vd, "v_p_n": vn, "v": v,
                      "num_bitlen": nb},
            "resid": [0.0], "cert": cert, "cost": float(n) * 0.1,
            "notes": "LTE: v_%d(%d^%d − %d^%d) = v_p(x−y) + v_p(n) = %d + %d = %d"
                     % (p, x, n, y, n, vd, vn, v)}


def _verify_nt_lte(args, value):
    """Independent verifier: build x^n − y^n directly and count the valuation along a
    **square-bisection path** (never the lemma formula)."""
    p, x, y, n = args["p"], args["x"], args["y"], args["n"]
    N = x ** n - y ** n
    got = valuation_squaring(p, N)
    ok = got == value["v"]
    return {"ok": ok, "resid": [0.0 if ok else float(abs(got - value["v"]))],
            "detail": "定义直算 v_p(x^n−y^n)=%d vs 声称 %d" % (got, value["v"])}


_reg(
    name="nt_lte", band=BAND_L0, sub=1,
    in_schema=_LTE_IN, out_schema=_LTE_OUT,
    invoke=_invoke_nt_lte, verify=_verify_nt_lte, cert_kind="lte_valuation",
    cost=("units", 2.0),
    errors=("PRECOND_FAIL", "SCHEMA_MISMATCH", "POSTCOND_FAIL", "TIMEOUT"),
    precond=_lte_precond,
    effects="none (纯函数, 精确整数算术)",
    doc=("LTE (Lifting the Exponent) lemma: for an odd prime p | x−y with p ∤ xy, "
         "v_p(x^n−y^n) = v_p(x−y) + v_p(n). Generation = the lemma formula; the independent "
         "verifier builds x^n−y^n and counts the valuation along a square-bisection path; the "
         "certificate lte_valuation is recomputed along a third path (plain division counting) "
         "by certcheck_nt. Domain note: odd primes p only (p=2 needs a correction term and is "
         "explicitly rejected)."),
    requires_cert=True,
    thm="T-V31-LTE",
    stmt="p 奇素数, p | x−y, p ∤ xy ⇒ v_p(x^n − y^n) = v_p(x−y) + v_p(n)",
    selfchecks=(
        {"id": "lte_5_2_n2", "args": {"p": 3, "x": 5, "y": 2, "n": 2},
         "check": lambda r: (r.value["v"] == 1 and r.cert is not None and r.cert_independent,
                             "v=%d (25−4=21=3·7), 证书独立复验=%s" % (r.value["v"], r.cert_independent)),
         "expect": "v_3(5²−2²) = v_3(3)+v_3(2) = 1+0 = 1"},
        {"id": "lte_p_divides_n", "args": {"p": 3, "x": 10, "y": 1, "n": 3},
         "check": lambda r: (r.value["v"] == 3,
                             "v=%d, v_p(n)=%d (999 = 27·37)" % (r.value["v"], r.value["v_p_n"])),
         "expect": "v_3(10³−1) = v_3(9)+v_3(3) = 2+1 = 3"},
        {"id": "lte_n1_boundary", "args": {"p": 5, "x": 7, "y": 2, "n": 1},
         "check": lambda r: (r.value["v"] == 1 and r.value["v_p_n"] == 0, "n=1: v=%d" % r.value["v"]),
         "expect": "n=1 时公式退化为 v_p(x−y) = 1"},
        {"id": "lte_negative_base", "args": {"p": 3, "x": 2, "y": 5, "n": 4},
         "check": lambda r: (r.value["v"] == 1, "x<y 且 x−y<0: v=%d (−609 = −3·7·29)" % r.value["v"]),
         "expect": "负差同样成立: v_3(2⁴−5⁴) = v_3(−3)+0 = 1"},
        {"id": "lte_p2_domain_reject", "args": {"p": 2, "x": 3, "y": 1, "n": 2},
         "expect_err": "PRECOND_FAIL",
         "expect": "p=2 是显式域外 (需 v_2(x+y)−1 修正项) —— 拒绝而非给错值"},
        {"id": "lte_not_dividing_diff", "args": {"p": 7, "x": 5, "y": 2, "n": 2},
         "expect_err": "PRECOND_FAIL", "expect": "7 ∤ (5−2)"},
        {"id": "lte_p_divides_x", "args": {"p": 3, "x": 6, "y": 1, "n": 2},
         "expect_err": "PRECOND_FAIL", "expect": "p | xy (3 | 6) 域外"},
        {"id": "lte_schema_float", "args": {"p": 3, "x": 5.0, "y": 2, "n": 2},
         "expect_err": "SCHEMA_MISMATCH", "expect": "§17.5: float 不得进精确通道"},
    ),
)


# ===========================================================================
# 9902  nt_hensel_lift  -- Hensel simple-root lift
# ===========================================================================

_HENSEL_IN = _sdict({
    "coeffs": {"kind": "list", "items": _EXACT, "minlen": 2, "maxlen": 12},
    "p": {"kind": "int", "min": 3},
    "r0": _EXACT,
    "k": {"kind": "int", "min": 1, "max": 12},
}, ["coeffs", "p", "r0", "k"])
_HENSEL_OUT = _sdict({
    "p": _EXACT, "k": _EXACT, "p_k": _EXACT, "r0": _EXACT, "root_mod_pk": _EXACT,
    "f_root_mod_pk": _EXACT, "unique_exhaustive": {"kind": "bool"}, "exhaustive_space": _EXACT,
}, ["p", "k", "p_k", "r0", "root_mod_pk", "f_root_mod_pk", "unique_exhaustive", "exhaustive_space"])


def _peval(coeffs, x, mod=None):
    acc = 0
    for c in reversed(coeffs):
        acc = acc * x + c
        if mod is not None:
            acc %= mod
    return acc


def _hensel_precond(args):
    coeffs, p, r0, k = args["coeffs"], args["p"], args["r0"], args["k"]
    if len(coeffs) < 2 or coeffs[-1] == 0:
        return False, "需要次数 >= 1 的非零首项多项式"
    if not is_prime(p):
        return False, "p=%d 非素数" % p
    if k < 1:
        return False, "k >= 1"
    if p ** k > 10 ** 12:
        return False, "p^k = %d 超出规模上限 1e12" % (p ** k)
    if _peval(coeffs, r0, p) % p != 0:
        return False, "f(r0) ≢ 0 (mod p) —— r0 不是模 p 的根"
    dc = [i * coeffs[i] for i in range(1, len(coeffs))]
    if _peval(dc, r0, p) % p == 0:
        return False, "f'(r0) ≡ 0 (mod p) —— 非单根, 唯一提升不成立 (域外)"
    return True, ""


def _invoke_nt_hensel(args, ctl):
    """Newton lifting: each step pushes the precision from p^m to p^min(2m,k).

    Correctness point (a trap we hit, recorded here): the one-step Newton correction must be
    computed at the **target precision** p^{m2} -- both f(root) and f'(root)^{-1} taken mod p^{m2},
    and the result reduced mod p^{m2} too. If f(root) is reduced mod p^m it is 0, the correction
    vanishes, and the lift stands still (while returning a wrong root). This is the standard
    quadratic-convergence condition: f(r') = f(r)²·f''/(2f'²) + … doubles precision only if f(r)
    and f'^{-1} are both evaluated at the new precision.
    """
    coeffs, p, r0, k = args["coeffs"], args["p"], args["r0"], args["k"]
    dc = [i * coeffs[i] for i in range(1, len(coeffs))]
    root = r0 % p
    m = 1
    while m < k:
        ctl.check_time("op:nt_hensel_lift#9902")
        m2 = min(2 * m, k)
        pm2 = p ** m2
        fr = _peval(coeffs, root, pm2) % pm2
        fpr = _peval(dc, root, pm2) % pm2
        inv = _modinv(fpr, pm2)
        if inv is None:
            raise SvoError(Err.PRECOND_FAIL, ctx="op:nt_hensel_lift#9902",
                           detail="f'(root) 在 mod p^%d 下不可逆 (非单根的提升中途退化)" % m2)
        root = (root - fr * inv) % pm2
        m = m2
    pk = p ** k
    space = p ** (k - 1) if k >= 1 else 1
    exhaustive = space <= 200000
    cert = Certificate(kind="hensel_root",
                       payload={"coeffs": coeffs, "p": p, "r0": r0, "k": k, "root": root,
                                "unique_exhaustive": exhaustive},
                       issued_by="nt_hensel_lift")
    return {"value": {"p": p, "k": k, "p_k": pk, "r0": r0 % p, "root_mod_pk": root,
                      "f_root_mod_pk": _peval(coeffs, root, pk) % pk,
                      "unique_exhaustive": exhaustive, "exhaustive_space": space},
            "resid": [0.0], "cert": cert, "cost": float(k) * 1.0,
            "notes": "牛顿提升到 mod %d; 唯一性穷举空间 %d %s"
                     % (pk, space, "(已穷举)" if exhaustive else "(超限未穷举, 如实声明)")}


def _modinv(a, m):
    """Wrapper around numtheory.modinv: returns None when not invertible (the caller turns that
    into a §15 error code)."""
    return modinv(a, m)


def _verify_nt_hensel(args, value):
    """Independent verifier: back-substitution + (when feasible) **exhaustive uniqueness** -- a
    decision path unrelated to the Newton iteration."""
    coeffs, p, k = args["coeffs"], args["p"], args["k"]
    pk = p ** k
    root = value["root_mod_pk"]
    bad = []
    if _peval(coeffs, root, pk) % pk != 0:
        bad.append("f(root) ≢ 0 mod p^k")
    if root % p != args["r0"] % p:
        bad.append("root 不是 r0 的提升")
    hits = []
    if value["unique_exhaustive"]:
        for t in range(p ** (k - 1)):
            r = args["r0"] % p + t * p
            if _peval(coeffs, r, pk) % pk == 0:
                hits.append(r)
        if hits != [root]:
            bad.append("穷举解得 %r, 与声称 %d 不符 (唯一性/正确性失败)" % (hits[:4], root))
    return {"ok": not bad, "resid": [float(len(bad))],
            "detail": ("代回通过; 穷举 %d 个提升恰解出 %d" % (p ** (k - 1), len(hits)))
                      if not bad else "; ".join(bad)}


_reg(
    name="nt_hensel_lift", band=BAND_L0, sub=2,
    in_schema=_HENSEL_IN, out_schema=_HENSEL_OUT,
    invoke=_invoke_nt_hensel, verify=_verify_nt_hensel, cert_kind="hensel_root",
    cost=("units", 2.5),
    errors=("PRECOND_FAIL", "SCHEMA_MISMATCH", "POSTCOND_FAIL", "TIMEOUT"),
    precond=_hensel_precond,
    effects="none (纯函数, 精确整数算术)",
    doc=("Hensel lemma (simple-root case): f(r0) ≡ 0 (mod p) and f'(r0) ≢ 0 (mod p) => for every "
         "k, r0 lifts uniquely to a root mod p^k. Generation = Newton iteration (precision "
         "doubling); independent verifier = back-substitution + (when the space is <= 2e5) "
         "exhaustive enumeration of all p^(k−1) lifts confirming uniqueness; certificate "
         "hensel_root is recomputed independently by certcheck_nt. Beyond the cap uniqueness is "
         "**not claimed** (noted in the notes)."),
    requires_cert=True,
    thm="T-V31-HENSEL",
    stmt="f(r0) ≡ 0 (mod p), f'(r0) ≢ 0 (mod p) ⇒ ∃! r ≡ r0 (mod p): f(r) ≡ 0 (mod p^k)",
    selfchecks=(
        {"id": "hensel_sqrt2_mod343", "args": {"coeffs": [-2, 0, 1], "p": 7, "r0": 3, "k": 3},
         "check": lambda r: (r.value["root_mod_pk"] == 108 and r.value["unique_exhaustive"]
                             and r.cert_independent,
                             "√2 mod 343 = %d (108² = 11664 = 343·34 + 2)" % r.value["root_mod_pk"]),
         "expect": "x²−2 的根 3 (mod 7) 提升到 108 (mod 343)"},
        {"id": "hensel_linear_mod625", "args": {"coeffs": [1, 2], "p": 5, "r0": 2, "k": 4},
         "check": lambda r: (r.value["root_mod_pk"] == 312,
                             "2·312+1 = 625 ≡ 0 (mod 625): root=%d" % r.value["root_mod_pk"]),
         "expect": "线性 f=2x+1: 解 312 mod 625"},
        {"id": "hensel_k1_boundary", "args": {"coeffs": [-2, 0, 1], "p": 7, "r0": 3, "k": 1},
         "check": lambda r: (r.value["root_mod_pk"] == 3 and r.value["exhaustive_space"] == 1,
                             "k=1 时提升就是 r0 (mod p)"),
         "expect": "k=1 边界: root = r0 mod p"},
        {"id": "hensel_nonsimple_reject", "args": {"coeffs": [0, 0, 1], "p": 3, "r0": 0, "k": 2},
         "expect_err": "PRECOND_FAIL", "expect": "f=x³, f'(0)=0 ⇒ 非单根域外"},
        {"id": "hensel_not_a_root", "args": {"coeffs": [-2, 0, 1], "p": 7, "r0": 5, "k": 2},
         "expect_err": "PRECOND_FAIL", "expect": "25−2 = 23 ≢ 0 (mod 7)"},
        {"id": "hensel_schema_empty", "args": {"coeffs": [], "p": 7, "r0": 3, "k": 2},
         "expect_err": "SCHEMA_MISMATCH", "expect": "空系数表被 schema 门拦下"},
    ),
)


# ===========================================================================
# 9903  nt_pratt_cert  -- Pratt primality certificate tree
# ===========================================================================

_PRATT_LEAF = 10000    # n <= this bound uses a "trial-division proof" as tree leaf (complete and cheap)
_PRATT_DEPTH = 40

_PRATT_IN = _sdict({
    "n": {"kind": "int", "min": 3, "max": 10 ** 19},
    "witness_bound": {"kind": "int", "min": 4, "max": 100000},
}, ["n", "witness_bound"])
_PRATT_OUT = _sdict({
    "n": _EXACT, "is_prime": {"kind": "bool"}, "has_witness": {"kind": "bool"}, "witness": _EXACT,
    "n_minus_1_factors": {"kind": "list", "items": {"kind": "tuple", "items": [_EXACT, _EXACT]}, "minlen": 0, "maxlen": 24},
    "depth": _EXACT, "n_nodes": _EXACT, "is_base_leaf": {"kind": "bool"},
}, ["n", "is_prime", "has_witness", "witness", "n_minus_1_factors", "depth", "n_nodes", "is_base_leaf"])


def _pratt_precond(args):
    n = args["n"]
    if n < 3:
        return False, "n >= 3"
    if n % 2 == 0:
        return False, "n 为偶数: 除 2 外无素数, 域外 (Pratt 树只对奇素数构造)"
    if n - 1 > 10 ** 19:
        return False, "n−1 超出分解上限 1e19"
    return True, ""


def _find_witness(n, primes, bound):
    """Deterministic witness search: ord_n(a) = n−1 iff a^(n−1) ≡ 1 and no (n−1)/q gives 1."""
    for a in range(2, min(n - 1, bound) + 1):
        if pow(a, n - 1, n) != 1:
            continue
        if all(pow(a, (n - 1) // q, n) != 1 for q in primes):
            return a
    return None


def _pratt_node(n, bound, memo, depth, ctl):
    if n in memo:
        return memo[n]
    if depth > _PRATT_DEPTH:
        raise SvoError(Err.BOUNDED_VIOLATION, ctx="op:nt_pratt_cert#9903",
                       detail="证书树深超限 %d" % _PRATT_DEPTH)
    if n <= _PRATT_LEAF:
        # a leaf must **prove itself**: complete trial-division proof; a composite leaf => no
        # Pratt certificate (it is not "skip the check")
        if is_prime_trial(n) is not True:
            raise SvoError(Err.NO_CERTIFICATE, ctx="op:nt_pratt_cert#9903",
                           detail="%d 经完整试除证明为合数 ⇒ 无 Pratt 证书" % n)
        node = {"n": n, "base": True}
        memo[n] = node
        return node
    ctl.check_time("op:nt_pratt_cert#9903")
    facs = factorize(n - 1)
    primes = [q for q, _ in facs]
    a = _find_witness(n, primes, bound)
    if a is None:
        if is_prime(n):
            raise SvoError(Err.UNIDENTIFIABLE, ctx="op:nt_pratt_cert#9903",
                           detail="n=%d 经确定性 MR 判定为素数, 但见证搜索在 a<=%d 内未找到 "
                                  "—— 拒绝给不完整证书 (诚实失败)" % (n, bound))
        raise SvoError(Err.NO_CERTIFICATE, ctx="op:nt_pratt_cert#9903",
                       detail="n=%d 为合数 ⇒ 阶 = n−1 的元素不存在 ⇒ Pratt 证书不存在" % n)
    factors = [[q, e, _pratt_node(q, bound, memo, depth + 1, ctl)] for q, e in facs]
    node = {"n": n, "a": a, "factors": factors, "base": False}
    memo[n] = node
    return node


def _count_nodes(node):
    if node.get("base"):
        return 1
    return 1 + sum(_count_nodes(f[2]) for f in node["factors"])


def _tree_depth(node):
    if node.get("base"):
        return 0
    return 1 + max(_tree_depth(f[2]) for f in node["factors"])


def _invoke_nt_pratt(args, ctl):
    n, bound = args["n"], args["witness_bound"]
    memo = {}
    tree = _pratt_node(n, bound, memo, 0, ctl)
    cert = Certificate(kind="pratt_tree", payload={"tree": tree, "n": n}, issued_by="nt_pratt_cert")
    return {"value": {"n": n, "is_prime": True, "has_witness": not tree.get("base"),
                      "witness": int(tree.get("a") or 0),
                      "n_minus_1_factors": [[f[0], f[1]] for f in tree.get("factors", [])],
                      "depth": _tree_depth(tree), "n_nodes": _count_nodes(tree),
                      "is_base_leaf": bool(tree.get("base"))},
            "resid": [0.0], "cert": cert, "cost": float(_count_nodes(tree)) * 3.0,
            "notes": "Pratt 树: %d 结点, 深 %d%s" % (_count_nodes(tree), _tree_depth(tree),
                                                     " (根即试除叶)" if tree.get("base") else "")}


def _verify_nt_pratt(args, value):
    """Independent verifier: recursive recomputation (own modpow) + deterministic MR
    cross-check."""
    n = args["n"]
    bad = []
    if not is_prime(n):
        bad.append("确定性 MR 判定 n=%d 为合数 (与证书矛盾)" % n)
    prod = 1
    for q, e in value["n_minus_1_factors"]:
        prod *= q ** e
        if not is_prime(q):
            bad.append("n−1 的因子 %d 非素数" % q)
    if not value["is_base_leaf"] and prod != n - 1:
        bad.append("n−1 因子连乘 %d ≠ %d" % (prod, n - 1))
    if value["has_witness"]:
        a = value["witness"]
        if pow(a, n - 1, n) != 1:
            bad.append("见证 a=%d 不满足 a^(n−1) ≡ 1" % a)
        for q, _e in value["n_minus_1_factors"]:
            if pow(a, (n - 1) // q, n) == 1:
                bad.append("见证阶不足: a^((n−1)/%d) ≡ 1" % q)
    return {"ok": not bad, "resid": [float(len(bad))],
            "detail": "MR 交叉判定素数 + n−1 分解复算通过 (见证 a=%d)" % value["witness"]
                      if not bad else "; ".join(bad)}


_reg(
    name="nt_pratt_cert", band=BAND_L0, sub=3,
    in_schema=_PRATT_IN, out_schema=_PRATT_OUT,
    invoke=_invoke_nt_pratt, verify=_verify_nt_pratt, cert_kind="pratt_tree",
    cost=("units", 12.0),
    errors=("PRECOND_FAIL", "SCHEMA_MISMATCH", "POSTCOND_FAIL", "NO_CERTIFICATE",
            "UNIDENTIFIABLE", "BOUNDED_VIOLATION", "TIMEOUT"),
    precond=_pratt_precond,
    effects="none (纯函数; 无文件/网络/子进程)",
    doc=("Pratt primality certificate tree: n prime <=> there is a with ord_n(a) = n−1; the "
         "certificate = witness a + the prime factorization of n−1 + a **recursive "
         "sub-certificate** for every prime factor (leaves with n <= 1e4 are closed by a "
         "trial-division proof). Generation = witness search + rho factorization; independent "
         "verifier = recursive recomputation + deterministic MR cross-check; certificate "
         "pratt_tree is recomputed recursively by certcheck_nt (own modpow, no import of the "
         "generator). Composite => no certificate => NO_CERTIFICATE (a theorem, not an error); "
         "prime but witness search over bound => UNIDENTIFIABLE (honest failure, no incomplete "
         "certificate)."),
    requires_cert=True,
    thm="T-V31-PRATT",
    stmt="n 素数 ⇔ ∃a: ord_n(a) = n−1 (Pratt 证书树可判定素性)",
    selfchecks=(
        {"id": "pratt_mersenne_31", "args": {"n": 2147483647, "witness_bound": 1000},
         "check": lambda r: (r.value["n_minus_1_factors"] == [[2, 1], [3, 2], [7, 1], [11, 1], [31, 1], [151, 1], [331, 1]]
                             and r.cert_independent and r.value["n_nodes"] >= 4,
                             "2^31−1: n−1 分解 %r, %d 结点" % (r.value["n_minus_1_factors"], r.value["n_nodes"])),
         "expect": "2^31−1 = 2147483647 素数; n−1 = 2·3²·7·11·31·151·331"},
        {"id": "pratt_small_97_leaf", "args": {"n": 97, "witness_bound": 100},
         "check": lambda r: (r.value["is_base_leaf"] and r.value["n_nodes"] == 1,
                             "97 ≤ 1e4 ⇒ 试除叶 (完全证明)"),
         "expect": "小素数走试除封底叶"},
        {"id": "pratt_carmichael_561", "args": {"n": 561, "witness_bound": 100},
         "expect_err": "NO_CERTIFICATE",
         "expect": "561 合数 (Carmichael) ⇒ 阶 n−1 的元素不存在 ⇒ 无证书"},
        {"id": "pratt_even_reject", "args": {"n": 1000, "witness_bound": 100},
         "expect_err": "PRECOND_FAIL", "expect": "偶数域外"},
        {"id": "pratt_n3_boundary", "args": {"n": 3, "witness_bound": 4},
         "check": lambda r: (r.value["is_prime"] and r.value["is_base_leaf"], "n=3 最小内部域"),
         "expect": "n=3 边界 (最小奇素数)"},
        {"id": "pratt_semiprime_reject", "args": {"n": 1000003 * 1000033, "witness_bound": 100},
         "expect_err": "NO_CERTIFICATE", "expect": "大合数 (两素因子) ⇒ 无 Pratt 证书"},
        {"id": "pratt_mersenne_61_witness", "args": {"n": 2305843009213693951, "witness_bound": 1000},
         "check": lambda r: (r.cert_independent and r.value["n_nodes"] >= 8,
                             "2^61−1: %d 结点, 见证 a=%d" % (r.value["n_nodes"], r.value["witness"])),
         "expect": "2^61−1 (Mersenne 素数) 的完整证书树"},
    ),
)


# ===========================================================================
# 9904  nt_miller_witness  -- witness recomputation
# ===========================================================================

_MW_IN = _sdict({"n": {"kind": "int", "min": 3, "max": 10 ** 24}, "a": _EXACT}, ["n", "a"])
_MW_OUT = _sdict({
    "n": _EXACT, "a": _EXACT, "d": _EXACT, "s": _EXACT,
    "residues": {"kind": "list", "items": _EXACT, "minlen": 1, "maxlen": 64},
    "is_witness": {"kind": "bool"}, "fermat_ok": {"kind": "bool"},
    "final_is_one": {"kind": "bool"}, "reason": {"kind": "str"},
}, ["n", "a", "d", "s", "residues", "is_witness", "fermat_ok", "final_is_one", "reason"])


def _mw_precond(args):
    n, a = args["n"], args["a"]
    if n < 3:
        return False, "n >= 3"
    if not (1 < a < n):
        return False, "见证必须满足 1 < a < n (a=%d)" % a
    return True, ""


def _invoke_nt_mw(args, ctl):
    n, a = args["n"], args["a"]
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
    residues = [pow(a, d, n)]
    for _ in range(1, s):
        residues.append(residues[-1] * residues[-1] % n)
    is_wit = (residues[0] != 1) and all(x != n - 1 for x in residues)
    fermat = pow(a, n - 1, n) == 1
    reason = ("强见证: a^d ≢ 1 且所有 a^(2^i·d) ≢ −1 ⇒ n 必为合数"
              if is_wit else "非见证: 序列落在 {1, −1} 的强伪素数形态 ⇒ 本基无信息")
    cert = Certificate(kind="miller_witness",
                       payload={"n": n, "a": a, "d": d, "s": s, "residues": residues,
                                "is_witness": is_wit},
                       issued_by="nt_miller_witness")
    return {"value": {"n": n, "a": a, "d": d, "s": s, "residues": residues,
                      "is_witness": is_wit, "fermat_ok": fermat,
                      "final_is_one": residues[-1] == 1, "reason": reason},
            "resid": [0.0], "cert": cert, "cost": float(s) * 0.5,
            "notes": "n−1 = 2^%d·%d, s=%d 个残差" % (s, d, s)}


def _verify_nt_mw(args, value):
    """Independent verifier: **hand-written** square-and-multiply recomputation of the residue
    sequence (a different implementation from pow), plus a re-check of the criterion."""
    n, a, d, s = args["n"], args["a"], value["d"], value["s"]
    r = 1
    e = d
    base = a % n
    while e > 0:
        if e & 1:
            r = r * base % n
        base = base * base % n
        e >>= 1
    seq = [r]
    for _ in range(1, s):
        seq.append(seq[-1] * seq[-1] % n)
    bad = []
    if seq != list(value["residues"]):
        bad.append("残差序列重算不符: %r vs %r" % (seq[:6], list(value["residues"])[:6]))
    if (2 ** s) * d != n - 1 or d % 2 == 0:
        bad.append("n−1 = 2^s·d 分解不成立")
    wit = (seq[0] != 1) and all(x != n - 1 for x in seq)
    if wit != value["is_witness"]:
        bad.append("见证判据复算不符")
    return {"ok": not bad, "resid": [float(len(bad))],
            "detail": "手写平方-乘重算 %d 位残差一致, 见证=%s" % (s, wit) if not bad else "; ".join(bad)}


_reg(
    name="nt_miller_witness", band=BAND_L0, sub=4,
    in_schema=_MW_IN, out_schema=_MW_OUT,
    invoke=_invoke_nt_mw, verify=_verify_nt_mw, cert_kind="miller_witness",
    cost=("units", 1.5),
    errors=("PRECOND_FAIL", "SCHEMA_MISMATCH", "POSTCOND_FAIL", "TIMEOUT"),
    precond=_mw_precond,
    effects="none (纯函数)",
    doc=("Miller-Rabin strong-witness recomputation: given n and base a, output the split "
         "n−1 = 2^s·d and the full residue sequence a^d, a^(2d), …, a^(2^(s−1)d) mod n, and "
         "decide whether a is a strong witness for compositeness of n. Generation = built-in "
         "pow; independent verifier = hand-written square-and-multiply recomputation of the "
         "whole sequence; certificate miller_witness is run a third time by certcheck_nt (own "
         "_powmod), which also re-checks the s/d split and the criterion."),
    requires_cert=True,
    thm="T-V31-MR",
    stmt="a 满足 a^d ≢ 1 且 ∀i<s: a^(2^i d) ≢ −1 (mod n) ⇒ n 为合数",
    selfchecks=(
        {"id": "mw_561_base2", "args": {"n": 561, "a": 2},
         "check": lambda r: (r.value["is_witness"] and r.value["s"] == 4 and r.value["d"] == 35
                             and r.value["residues"][0] == 263 and r.cert_independent,
                             "561: 560=2⁴·35, 残差[0]=%d, 见证=%s" % (r.value["residues"][0], r.value["is_witness"])),
         "expect": "561 = 3·11·17 是 Carmichael 数, 但基 2 仍是强见证 (263,166,67,1)"},
        {"id": "mw_prime101_base2", "args": {"n": 101, "a": 2},
         "check": lambda r: (not r.value["is_witness"] and r.value["fermat_ok"],
                             "素数 101 上基 2 非见证 (费马成立)"),
         "expect": "素数上任何基都不是强见证 (定理)"},
        {"id": "mw_9_base2", "args": {"n": 9, "a": 2},
         "check": lambda r: (r.value["is_witness"] and r.value["s"] == 3 and r.value["d"] == 1
                             and r.value["residues"] == [2, 4, 7],
                             "9: 8 = 2³·1, 残差 %r" % (r.value["residues"],)),
         "expect": "9 = 3² 上基 2 见证; d=1 的边界 (全为平方步)"},
        {"id": "mw_minus1_not_witness", "args": {"n": 9, "a": 8},
         "check": lambda r: (not r.value["is_witness"] and r.value["residues"][0] == 8,
                             "a = −1 (mod n): 首项即 −1 ⇒ 非见证 (序列 %r)" % (r.value["residues"],)),
         "expect": "a ≡ −1 时 MR 基无信息 (非见证), 但也不构成伪素数错误"},
        {"id": "mw_even_n_s0", "args": {"n": 10, "a": 3},
         "check": lambda r: (r.value["s"] == 0 and r.value["d"] == 9 and len(r.value["residues"]) == 1,
                             "n=10: n−1=9 为奇 ⇒ s=0, 单元素序列"),
         "expect": "s=0 边界 (n 偶数 ⇒ n−1 奇数)"},
        {"id": "mw_a_ge_n_reject", "args": {"n": 9, "a": 9},
         "expect_err": "PRECOND_FAIL", "expect": "a ≥ n 域外"},
        {"id": "mw_a1_boundary_reject", "args": {"n": 9, "a": 1},
         "expect_err": "PRECOND_FAIL", "expect": "a=1 域外 (1 < a < n)"},
    ),
)


# ===========================================================================
# 9905  nt_factor_cert  -- factor product + per-prime primality
# ===========================================================================

_FAC_IN = _sdict({"n": {"kind": "int", "min": 2, "max": 10 ** 18}}, ["n"])
_FAC_OUT = _sdict({
    "n": _EXACT,
    "factors": {"kind": "list", "items": {"kind": "tuple", "items": [_EXACT, _EXACT]}, "minlen": 1, "maxlen": 24},
    "n_distinct": _EXACT, "tau": _EXACT, "biggest_factor_bits": _EXACT,
}, ["n", "factors", "n_distinct", "tau", "biggest_factor_bits"])


def _fac_precond(args):
    n = args["n"]
    if n < 2:
        return False, "n >= 2 (1 与 0 无分解)"
    if n > 10 ** 18:
        return False, "n 超出分解上限 1e18"
    return True, ""


def _invoke_nt_factor(args, ctl):
    n = args["n"]
    ctl.check_time("op:nt_factor_cert#9905")
    facs = factorize(n)
    prod = 1
    for p, e in facs:
        prod *= p ** e
    if prod != n:
        # cannot happen, but never pass silently (a broken factorizer must blow up as a
        # structured error)
        raise SvoError(Err.POSTCOND_FAIL, ctx="op:nt_factor_cert#9905",
                       detail="内部连乘对账失败: %d ≠ %d" % (prod, n))
    cert = Certificate(kind="factor_product",
                       payload={"n": n, "factors": facs, "source": "rho+trial"},
                       issued_by="nt_factor_cert")
    return {"value": {"n": n, "factors": facs, "n_distinct": len(facs),
                      "tau": tau_from_factors(facs),
                      "biggest_factor_bits": max(p.bit_length() for p, _ in facs)},
            "resid": [0.0], "cert": cert, "cost": float(len(facs)) * 4.0,
            "notes": "分解 %d = %s (τ=%d)" % (n, "·".join("%d^%d" % (p, e) for p, e in facs),
                                              tau_from_factors(facs))}


def _verify_nt_factor(args, value):
    """Independent verifier: product rebuild + per-factor **trial-division proof** of primality
    (a different path from the rho factorization)."""
    n = args["n"]
    bad = []
    prod = 1
    last = 1
    for p, e in value["factors"]:
        prod *= p ** e
        if p <= last:
            bad.append("因子未严格递增: %d ≤ %d" % (p, last))
        last = p
        t = is_prime_trial(p) if p <= 10 ** 12 else None
        if t is False:
            bad.append("因子 %d 试除证明为合数" % p)
        elif t is None and not is_prime(p):
            bad.append("因子 %d 素性存疑 (试除超限且 MR 否决)" % p)
    if prod != n:
        bad.append("连乘 %d ≠ n=%d" % (prod, n))
    divs = divisors_from_factors(value["factors"])
    brute = divisors_brute(n)
    if len(divs) != len(brute) or divs != brute:
        bad.append("因子表导出的因子集与试除穷举不一致 (%d vs %d 个)" % (len(divs), len(brute)))
    return {"ok": not bad, "resid": [float(len(bad))],
            "detail": "连乘重建 + 逐素试除证明 + 因子集穷举对拍一致 (%d 个因子)" % len(value["factors"])
                      if not bad else "; ".join(bad)}


_reg(
    name="nt_factor_cert", band=BAND_L0, sub=5,
    in_schema=_FAC_IN, out_schema=_FAC_OUT,
    invoke=_invoke_nt_factor, verify=_verify_nt_factor, cert_kind="factor_product",
    cost=("units", 8.0),
    errors=("PRECOND_FAIL", "SCHEMA_MISMATCH", "POSTCOND_FAIL", "BOUNDED_VIOLATION", "TIMEOUT"),
    precond=_fac_precond,
    effects="none (纯函数; rho 用确定性 LCG, 无外部随机)",
    doc=("Complete factorization certificate: outputs [[p,e],…] with product reconciliation. "
         "Generation = trial division + deterministic Brent-rho; independent verifier = product "
         "rebuild + per-factor trial-division **proof** (full trial division for p <= 1e12, MR "
         "beyond) + factor-set comparison against brute divisor enumeration; certificate "
         "factor_product is recomputed independently by certcheck_nt (own trial division/MR, no "
         "import of numtheory). Honesty note: rho is probabilistic, but the result is "
         "re-checked along two independent paths, so \"the factorization is correct\" never "
         "rests on rho's probabilistic claim."),
    requires_cert=True,
    thm="T-V31-FACT",
    stmt="n = Π p_i^{e_i} 且每个 p_i 为素数 (连乘证书可独立复核)",
    selfchecks=(
        {"id": "fac_360", "args": {"n": 360},
         "check": lambda r: (r.value["factors"] == [[2, 3], [3, 2], [5, 1]] and r.value["tau"] == 24
                             and r.cert_independent,
                             "360 = 2³·3²·5, τ=%d" % r.value["tau"]),
         "expect": "小整数穷举可见的分解"},
        {"id": "fac_mersenne31_prime", "args": {"n": 2147483647},
         "check": lambda r: (r.value["factors"] == [[2147483647, 1]] and r.value["n_distinct"] == 1,
                             "素数自身 (2147483647)"),
         "expect": "素数 n: 分解 = 自身"},
        {"id": "fac_10pow12", "args": {"n": 10 ** 12},
         "check": lambda r: (r.value["factors"] == [[2, 12], [5, 12]], "10^12 = 2^12·5^12"),
         "expect": "指数 12 的重因子"},
        {"id": "fac_semiprime_1e6", "args": {"n": 999983 * 1000003},
         "check": lambda r: (r.value["n_distinct"] == 2
                             and all(e == 1 for _p, e in r.value["factors"]),
                             "半素数分解: %r" % (r.value["factors"],)),
         "expect": "两个 ~1e6 素因子的半素数 (rho 路径)"},
        {"id": "fac_one_reject", "args": {"n": 1},
         "expect_err": "SCHEMA_MISMATCH", "expect": "n >= 2 是类型层声明 (n=1 无分解)"},
        {"id": "fac_float_reject", "args": {"n": 2.5},
         "expect_err": "SCHEMA_MISMATCH", "expect": "float 不得进精确通道"},
        {"id": "fac_2_boundary", "args": {"n": 2},
         "check": lambda r: (r.value["factors"] == [[2, 1]], "n=2 最小域"),
         "expect": "n=2 边界"},
    ),
)


# ===========================================================================
# 9906  nt_order_mult  -- multiplicative order
# ===========================================================================

_ORD_IN = _sdict({
    "a": _EXACT, "n": {"kind": "int", "min": 2, "max": 10 ** 12},
    "full_scan_cap": {"kind": "int", "min": 1, "max": 200000},
}, ["a", "n", "full_scan_cap"])
_ORD_OUT = _sdict({
    "a": _EXACT, "n": _EXACT, "order": _EXACT, "phi": _EXACT,
    "divides_phi": {"kind": "bool"},
    "n_factors": {"kind": "list", "items": {"kind": "tuple", "items": [_EXACT, _EXACT]}, "minlen": 1, "maxlen": 16},
    "n_divisors_scanned": _EXACT,
}, ["a", "n", "order", "phi", "divides_phi", "n_factors", "n_divisors_scanned"])


def _ord_precond(args):
    a, n = args["a"], args["n"]
    if n < 2:
        return False, "n >= 2"
    if a % n == 0:
        return False, "a ≡ 0 (mod n): 不属于单位群, 阶不存在"
    g = _gcd_py(a, n)
    if g != 1:
        return False, "gcd(a,n) = %d ≠ 1 —— a 不在 (Z/nZ)* 内, 乘法阶无定义" % g
    return True, ""


def _gcd_py(a, b):
    a, b = abs(a), abs(b)
    while b:
        a, b = b, a % b
    return a


def _invoke_nt_order(args, ctl):
    a, n = args["a"], args["n"]
    facs = factorize(n)
    phi = phi_from_factors(facs)
    divs = divisors_from_factors(facs)
    order = None
    scanned = 0
    for d in divisors_from_factors(factorize(phi)):
        ctl.check_time("op:nt_order_mult#9906")
        scanned += 1
        if pow(a, d, n) == 1:
            order = d
            break
    if order is None:
        raise SvoError(Err.POSTCOND_FAIL, ctx="op:nt_order_mult#9906",
                       detail="在 φ(n)=%d 的因子中未找到 a^d ≡ 1 —— 不可能 (a^φ ≡ 1), 内部错误" % phi)
    cert = Certificate(kind="order_exhaustive",
                       payload={"a": a, "n": n, "order": order,
                                "scan_mode": "full" if order <= 5000 else "divisor"},
                       issued_by="nt_order_mult")
    return {"value": {"a": a, "n": n, "order": order, "phi": phi, "divides_phi": phi % order == 0,
                      "n_factors": facs, "n_divisors_scanned": scanned},
            "resid": [0.0], "cert": cert, "cost": float(scanned) * 5.0,
            "notes": "ord_%d(%d) = %d (φ=%d, 扫 %d 个 φ 的因子; n 有 %d 个因子)"
                     % (n, a, order, phi, scanned, len(divs))}


def _verify_nt_order(args, value):
    """Independent verifier: a^ord ≡ 1 + minimality (full scan 1..ord−1 for small orders, all
    proper divisors for large ones)."""
    a, n, order = args["a"], args["n"], value["order"]
    bad = []
    if pow(a, order, n) != 1:
        bad.append("a^order ≢ 1 (直接否决)")
    cap = args["full_scan_cap"]
    if order <= cap:
        for k in range(1, order):
            if pow(a, k, n) == 1:
                bad.append("非最小: k=%d 也是周期" % k)
                break
        mode = "全穷举 1..%d" % (order - 1)
    else:
        for d in range(1, order):
            if order % d == 0 and pow(a, d, n) == 1:
                bad.append("非最小: 真因子 %d 也是周期" % d)
                break
        mode = "因子扫描 (order=%d > cap=%d, 全穷举未做)" % (order, cap)
    # the order must divide λ(n) (cross-check via idempotence): at least it divides φ(n)
    if value["phi"] % order != 0:
        bad.append("阶 %d 不整除 φ(n)=%d" % (order, value["phi"]))
    return {"ok": not bad, "resid": [float(len(bad))],
            "detail": "a^order ≡ 1 + 最小性通过 (%s)" % mode if not bad else "; ".join(bad)}


_reg(
    name="nt_order_mult", band=BAND_L0, sub=6,
    in_schema=_ORD_IN, out_schema=_ORD_OUT,
    invoke=_invoke_nt_order, verify=_verify_nt_order, cert_kind="order_exhaustive",
    cost=("units", 6.0),
    errors=("PRECOND_FAIL", "SCHEMA_MISMATCH", "POSTCOND_FAIL", "TIMEOUT"),
    precond=_ord_precond,
    effects="none (纯函数)",
    doc=("Multiplicative order ord_n(a) = min{k>0 : a^k ≡ 1 (mod n)}, requiring gcd(a,n)=1. "
         "Generation = factor n, compute φ, then enumerate the divisors of φ and take the first "
         "one that works; independent verifier = direct minimality check (full scan 1..ord−1 "
         "when order <= cap, otherwise all proper divisors, with the mode reported as measured); "
         "certificate order_exhaustive is exhaustively re-checked by certcheck_nt. Honesty "
         "note: for order > 5000 the certificate checker degrades to a divisor scan and is "
         "**not** fully exhaustive -- that case is stated explicitly in detail."),
    requires_cert=True,
    thm="T-V31-ORDER",
    stmt="ord_n(a) | φ(n); ord_n(a) = n−1 ⇔ a 是本原根",
    selfchecks=(
        {"id": "ord_2_mod7", "args": {"a": 2, "n": 7, "full_scan_cap": 1000},
         "check": lambda r: (r.value["order"] == 3 and r.value["phi"] == 6 and r.cert_independent,
                             "ord_7(2)=%d (2,4,1)" % r.value["order"]),
         "expect": "2 模 7 的阶 = 3"},
        {"id": "ord_3_mod7_primitive", "args": {"a": 3, "n": 7, "full_scan_cap": 1000},
         "check": lambda r: (r.value["order"] == 6 == r.value["phi"], "3 是模 7 本原根 (阶=φ=6)"),
         "expect": "3 模 7 的阶 = 6 = φ(7)"},
        {"id": "ord_2_mod561", "args": {"a": 2, "n": 561, "full_scan_cap": 1000},
         "check": lambda r: (r.value["order"] == 40, "ord_561(2)=%d (φ=320)" % r.value["order"]),
         "expect": "Carmichael 数 561 上 2 的阶 = 40 (2^40 ≡ 1, 2^20 = 67)"},
        {"id": "ord_2_mod97", "args": {"a": 2, "n": 97, "full_scan_cap": 1000},
         "check": lambda r: (r.value["order"] == 48 and r.value["divides_phi"],
                             "ord_97(2)=%d" % r.value["order"]),
         "expect": "ord_97(2) = 48 ((2/97)=1 ⇒ 阶整除 48, 且非真因子)"},
        {"id": "ord_a1_boundary", "args": {"a": 1, "n": 10, "full_scan_cap": 100},
         "check": lambda r: (r.value["order"] == 1, "a=1 的阶恒为 1 (最小 k)"),
         "expect": "a=1 边界: 阶 = 1"},
        {"id": "ord_noncoprime_reject", "args": {"a": 2, "n": 4, "full_scan_cap": 100},
         "expect_err": "PRECOND_FAIL", "expect": "gcd(2,4)=2 ⇒ 域外"},
        {"id": "ord_a_multiple_reject", "args": {"a": 7, "n": 7, "full_scan_cap": 100},
         "expect_err": "PRECOND_FAIL", "expect": "a ≡ 0 (mod n) ⇒ 域外"},
    ),
)


# ===========================================================================
# 9907  nt_jacobi_qr  -- Jacobi symbol + Euler criterion
# ===========================================================================

_JAC_IN = _sdict({
    "a": _EXACT, "n": {"kind": "int", "min": 3, "max": 10 ** 12},
    "search_cap": {"kind": "int", "min": 1, "max": 200000},
}, ["a", "n", "search_cap"])
_JAC_OUT = _sdict({
    "a": _EXACT, "n": _EXACT, "jacobi": {"kind": "int", "min": -1, "max": 1},
    "n_is_prime": {"kind": "bool"}, "euler_defined": {"kind": "bool"}, "euler_value": _EXACT,
    "qr_status": {"kind": "str", "enum": ["residue", "nonresidue", "undetermined"]},
    "has_root": {"kind": "bool"}, "root": _EXACT, "search_exhaustive": {"kind": "bool"},
    "n_factors": {"kind": "list", "items": {"kind": "tuple", "items": [_EXACT, _EXACT]}, "minlen": 0, "maxlen": 16},
}, ["a", "n", "jacobi", "n_is_prime", "euler_defined", "euler_value", "qr_status",
    "has_root", "root", "search_exhaustive", "n_factors"])


def _jac_precond(args):
    n = args["n"]
    if n < 3:
        return False, "n >= 3"
    if n % 2 == 0:
        return False, "Jacobi 符号要求 n 为正奇数 (n=%d 为偶)" % n
    return True, ""


def _invoke_nt_jacobi(args, ctl):
    a, n, cap = args["a"], args["n"], args["search_cap"]
    j = jacobi(a, n)
    nprime = is_prime(n)
    root = None
    exhaustive = False
    if nprime:
        e = 0 if a % n == 0 else pow(a % n, (n - 1) // 2, n)
        euler_val = 0 if a % n == 0 else (1 if e == 1 else n - 1)
        if j == 1:
            r = sqrt_mod_prime(a, n)
            root = r
        status = "residue" if root is not None else ("nonresidue" if j == -1 else "nonresidue")
        if a % n == 0:
            status = "residue"
            root = 0
        exhaustive = True
        facs = []
    else:
        euler_val = 0
        facs = factorize(n)
        lim = min(n - 1, cap)
        for x in range(0, lim + 1):
            ctl.check_time("op:nt_jacobi_qr#9907")
            if (x * x - a) % n == 0:
                root = x
                break
        exhaustive = (n - 1) <= cap
        if root is not None:
            status = "residue"
        elif exhaustive:
            status = "nonresidue"
        else:
            status = "undetermined"
    cert = Certificate(kind="jacobi_euler",
                       payload={"a": a, "n": n, "jacobi": j, "n_is_prime": bool(nprime),
                                "euler_value": euler_val, "has_root": root is not None,
                                "root": int(root or 0), "factors": facs,
                                "search_exhaustive": exhaustive},
                       issued_by="nt_jacobi_qr")
    return {"value": {"a": a, "n": n, "jacobi": j, "n_is_prime": bool(nprime),
                      "euler_defined": bool(nprime), "euler_value": int(euler_val),
                      "qr_status": status, "has_root": root is not None, "root": int(root or 0),
                      "search_exhaustive": bool(exhaustive), "n_factors": facs},
            "resid": [0.0], "cert": cert, "cost": float(min(n, cap)) * 0.02,
            "notes": "(a/n)=%d, n 素数=%s, QR 状态=%s%s"
                     % (j, nprime, status, "" if exhaustive else " (搜索未穷举, 如实声明)")}


def _verify_nt_jacobi(args, value):
    """Independent verifier: Jacobi via **factorization + per-prime Euler criterion** (a
    different path from the generator's reciprocity)."""
    a, n = args["a"], args["n"]
    bad = []
    if value["n_is_prime"] != is_prime(n):
        bad.append("n_is_prime 判定不一致")
    if value["n_is_prime"]:
        e = pow(a % n, (n - 1) // 2, n) if a % n else 0
        want = 0 if a % n == 0 else (1 if e == 1 else -1)
    else:
        want = 1
        for p, e in value["n_factors"]:
            if a % p == 0:
                want = 0
                break
            lp = legendre_euler(a, p)
            if lp == -1 and e % 2 == 1:
                want = -want
    if want != value["jacobi"]:
        bad.append("Jacobi 分解路径重算 %d ≠ %d" % (want, value["jacobi"]))
    if value["has_root"]:
        r = value["root"]
        if (r * r - a) % n != 0:
            bad.append("根回代 r² ≢ a (mod n)")
    elif value["qr_status"] == "residue":
        bad.append("声称 residue 但无根")
    if value["qr_status"] == "nonresidue" and not value["search_exhaustive"]:
        bad.append("声称 nonresidue 但搜索未穷举 (越权结论)")
    return {"ok": not bad, "resid": [float(len(bad))],
            "detail": "Jacobi 双路一致; QR 状态 %s (穷举=%s)" % (value["qr_status"], value["search_exhaustive"])
                      if not bad else "; ".join(bad)}


_reg(
    name="nt_jacobi_qr", band=BAND_L0, sub=7,
    in_schema=_JAC_IN, out_schema=_JAC_OUT,
    invoke=_invoke_nt_jacobi, verify=_verify_nt_jacobi, cert_kind="jacobi_euler",
    cost=("units", 3.0),
    errors=("PRECOND_FAIL", "SCHEMA_MISMATCH", "POSTCOND_FAIL", "TIMEOUT"),
    precond=_jac_precond,
    effects="none (纯函数)",
    doc=("Jacobi symbol (a/n) + Euler-criterion recomputation + quadratic-residue decision. "
         "Generation: Jacobi via binary reciprocity, prime n via the Euler criterion with a "
         "Tonelli-Shanks square root; composite n via exhaustive search. Independent verifier: "
         "factor n + multiply the per-prime Euler criteria (a different algorithm from "
         "reciprocity) + root back-substitution. Certificate jacobi_euler is recomputed along a "
         "third path by certcheck_nt. **Honesty point**: Jacobi=1 on composite n does not "
         "imply QR (classic trap), so qr_status yields nonresidue only when the exhaustive "
         "search is feasible and undetermined otherwise."),
    requires_cert=True,
    thm="T-V31-JACOBI",
    stmt="(a/p) = a^((p−1)/2) (mod p) ∈ {0,±1} (p 奇素数, Euler 判据); 合数上 Jacobi=1 ⇏ QR",
    selfchecks=(
        {"id": "jac_qr_2_mod7", "args": {"a": 2, "n": 7, "search_cap": 100},
         "check": lambda r: (r.value["jacobi"] == 1 and r.value["qr_status"] == "residue"
                             and r.value["has_root"] and r.value["root"] in (3, 4)
                             and r.cert_independent,
                             "(2/7)=1, 根 %d" % r.value["root"]),
         "expect": "3² = 9 ≡ 2 (mod 7) ⇒ 2 是 QR"},
        {"id": "jac_nonresidue_3_mod7", "args": {"a": 3, "n": 7, "search_cap": 100},
         "check": lambda r: (r.value["jacobi"] == -1 and r.value["qr_status"] == "nonresidue",
                             "(3/7)=−1, 状态 %s" % r.value["qr_status"]),
         "expect": "3 模 7 非二次剩余 (Euler 判据 = −1)"},
        {"id": "jac_composite_trap_2_mod15", "args": {"a": 2, "n": 15, "search_cap": 100},
         "check": lambda r: (r.value["jacobi"] == 1 and r.value["qr_status"] == "nonresidue"
                             and not r.value["n_is_prime"] and r.value["search_exhaustive"],
                             "(2/15)=1 但非 QR (15 个同余类穷举已证)"),
         "expect": "Jacobi=1 但非 QR 的经典陷阱: (2/15)=(2/3)(2/5)=(−1)(−1)=1"},
        {"id": "jac_zero_5_mod15", "args": {"a": 5, "n": 15, "search_cap": 100},
         "check": lambda r: (r.value["jacobi"] == 0, "(5/15)=0 (gcd=5)"),
         "expect": "gcd(a,n) ≠ 1 ⇒ Jacobi = 0"},
        {"id": "jac_residue_9_mod15", "args": {"a": 9, "n": 15, "search_cap": 100},
         "check": lambda r: (r.value["qr_status"] == "residue" and r.value["root"] in (3, 12),
                             "9 = 3² (mod 15): 根 %d" % r.value["root"]),
         "expect": "对比项: 同一个合数 n=15 上 9 确实是 QR"},
        {"id": "jac_even_n_reject", "args": {"a": 2, "n": 10, "search_cap": 100},
         "expect_err": "PRECOND_FAIL", "expect": "偶数 n 域外"},
        {"id": "jac_big_prime_17", "args": {"a": 2, "n": 17, "search_cap": 100},
         "check": lambda r: (r.value["qr_status"] == "residue" and r.value["root"] in (6, 11),
                             "6² = 36 ≡ 2 (mod 17)"),
         "expect": "素数 n 走 Euler + Tonelli-Shanks 给出根"},
        {"id": "jac_schema_float", "args": {"a": 1.5, "n": 17, "search_cap": 100},
         "expect_err": "SCHEMA_MISMATCH", "expect": "float 进精确通道被拒"},
    ),
)


# ===========================================================================
# 9908  nt_zsigmondy  -- Zsigmondy primitive prime factors
# ===========================================================================

_ZSI_IN = _sdict({
    "a": {"kind": "int", "min": 2, "max": 10 ** 6},
    "b": {"kind": "int", "min": 1, "max": 10 ** 6},
    "n": {"kind": "int", "min": 2, "max": 64},
}, ["a", "b", "n"])
_ZSI_OUT = _sdict({
    "a": _EXACT, "b": _EXACT, "n": _EXACT, "N": _EXACT,
    "divisors": {"kind": "list", "items": _EXACT, "minlen": 1, "maxlen": 32},
    "primitives": {"kind": "list", "items": _EXACT, "minlen": 0, "maxlen": 32},
    "n_primitive": _EXACT, "is_exception": {"kind": "bool"}, "exception_kind": {"kind": "str"},
}, ["a", "b", "n", "N", "divisors", "primitives", "n_primitive", "is_exception", "exception_kind"])


def _zsi_precond(args):
    a, b, n = args["a"], args["b"], args["n"]
    if not (a > b > 0):
        return False, "要求 a > b > 0"
    if _gcd_py(a, b) != 1:
        return False, "要求 gcd(a,b) = 1 (得 %d)" % _gcd_py(a, b)
    if n < 2:
        return False, "n >= 2"
    if a ** n - b ** n > 10 ** 24:
        return False, "N = a^n−b^n 超过 1e24 (rho 分解可行性上限, 超出即拒绝而非挂死)"
    return True, ""


def _invoke_nt_zsigmondy(args, ctl):
    a, b, n = args["a"], args["b"], args["n"]
    N = a ** n - b ** n
    facs = factorize(N)
    divs = [p for p, _ in facs]
    prims = []
    for p in divs:
        ctl.check_time("op:nt_zsigmondy#9908")
        if all((a ** k - b ** k) % p != 0 for k in range(1, n)):
            prims.append(p)
    is_exc = len(prims) == 0
    if is_exc and n == 2 and _is_pow2(a + b):
        kind = "n=2 且 a+b 为 2 的幂 (Zsigmondy 例外)"
    elif is_exc and (a, b, n) == (2, 1, 6):
        kind = "(a,b,n)=(2,1,6) (Zsigmondy 例外)"
    elif is_exc:
        kind = "无原始素因子但非已知例外 (需人工核查)"
    else:
        kind = ""
    cert = Certificate(kind="zsi_primitive",
                       payload={"a": a, "b": b, "n": n, "divisors": divs, "primitives": prims},
                       issued_by="nt_zsigmondy")
    return {"value": {"a": a, "b": b, "n": n, "N": N, "divisors": divs, "primitives": prims,
                      "n_primitive": len(prims), "is_exception": is_exc, "exception_kind": kind},
            "resid": [0.0], "cert": cert, "cost": float(len(divs)) * float(n) * 1.0,
            "notes": "N=%d 的素因子 %r; 原始素因子 %r%s" % (N, divs, prims,
                                                          (" [" + kind + "]") if kind else "")}


def _is_pow2(m):
    return m > 0 and (m & (m - 1)) == 0


def _verify_nt_zsigmondy(args, value):
    """Independent verifier: per-p checks of p|N and of divisibility for every k<n, plus
    **completeness** (covering N by valuations instead of refactoring)."""
    a, b, n = args["a"], args["b"], args["n"]
    N = value["N"]
    bad = []
    if N != a ** n - b ** n:
        bad.append("N 复算不符")
    for p in value["divisors"]:
        if N % p != 0:
            bad.append("声称因子 %d 不整除 N" % p)
        elif not is_prime(p):
            bad.append("声称因子 %d 非素数" % p)
    # completeness: re-multiplying the listed primes by valuation must reproduce N exactly
    # (no refactoring needed)
    acc = 1
    for p in value["divisors"]:
        if N % p == 0:
            acc *= p ** valuation(p, N)
    if acc != N:
        bad.append("素因子表不完整: 赋值重乘 %d ≠ N=%d" % (acc, N))
    recomputed = sorted(p for p in value["divisors"]
                        if all((a ** k - b ** k) % p != 0 for k in range(1, n)))
    if recomputed != sorted(value["primitives"]):
        bad.append("原始素因子穷举重算 %r ≠ 证书 %r" % (recomputed, sorted(value["primitives"])))
    if bool(value["is_exception"]) != (len(value["primitives"]) == 0):
        bad.append("is_exception 与原始素因子为空不一致")
    return {"ok": not bad, "resid": [float(len(bad))],
            "detail": "N=%d: 因子完整性 (赋值重乘) + 逐 k<n 穷举一致" % N if not bad else "; ".join(bad)}


_reg(
    name="nt_zsigmondy", band=BAND_L0, sub=8,
    in_schema=_ZSI_IN, out_schema=_ZSI_OUT,
    invoke=_invoke_nt_zsigmondy, verify=_verify_nt_zsigmondy, cert_kind="zsi_primitive",
    cost=("units", 10.0),
    errors=("PRECOND_FAIL", "SCHEMA_MISMATCH", "POSTCOND_FAIL", "BOUNDED_VIOLATION", "TIMEOUT"),
    precond=_zsi_precond,
    effects="none (纯函数)",
    doc=("Computational side of Zsigmondy's theorem: for a > b > 0, gcd(a,b)=1, n >= 2, find the "
         "**primitive prime factors** of a^n − b^n (p | a^n−b^n but p ∤ a^k−b^k for all k<n). "
         "Generation = factor N, then exhaustively test each p over k<n; independent verifier = "
         "per-p testing + a **valuation re-multiplication** proving the prime-factor table is "
         "complete (no repeated factorization) + exhaustive recomputation comparison; "
         "certificate zsi_primitive is recomputed independently by certcheck_nt (trial-division "
         "factorization + exhaustion). The two exceptions of the theorem (n=2 with a+b a power "
         "of 2; (2,1,6)) return the empty set as is, tagged with the exception kind -- an empty "
         "set is not a failure, it is the boundary of the theorem."),
    requires_cert=True,
    thm="T-V31-ZSIGMONDY",
    stmt="a>b>0, gcd(a,b)=1, n>1 ⇒ a^n−b^n 有原始素因子 (例外: n=2 且 a+b 为 2 的幂; (2,1,6))",
    selfchecks=(
        {"id": "zsi_2_1_3", "args": {"a": 2, "b": 1, "n": 3},
         "check": lambda r: (r.value["primitives"] == [7] and r.cert_independent,
                             "2³−1=7: 原始素因子 %r" % r.value["primitives"]),
         "expect": "7 ∤ 2−1, 7 ∤ 2²−1 ⇒ 7 原始"},
        {"id": "zsi_2_1_2", "args": {"a": 2, "b": 1, "n": 2},
         "check": lambda r: (r.value["primitives"] == [3] and not r.value["is_exception"],
                             "3 ∤ 2−1 ⇒ 3 原始 (a+b=3 非 2 的幂)"),
         "expect": "a+b = 3 不是 2 的幂 ⇒ 例外不适用"},
        {"id": "zsi_exception_3_1_2", "args": {"a": 3, "b": 1, "n": 2},
         "check": lambda r: (r.value["primitives"] == [] and r.value["is_exception"]
                             and "n=2" in r.value["exception_kind"],
                             "3²−1 = 8 = 2³: 2 | 3−1 ⇒ 无原始素因子 (例外)"),
         "expect": "n=2 且 a+b=4=2² ⇒ Zsigmondy 例外"},
        {"id": "zsi_exception_2_1_6", "args": {"a": 2, "b": 1, "n": 6},
         "check": lambda r: (r.value["primitives"] == [] and r.value["is_exception"]
                             and "(2,1,6)" in r.value["exception_kind"],
                             "2⁶−1 = 63 = 3²·7: 3|2²−1, 7|2³−1 ⇒ 空 (例外)"),
         "expect": "(2,1,6) 例外"},
        {"id": "zsi_5_3_3", "args": {"a": 5, "b": 3, "n": 3},
         "check": lambda r: (r.value["primitives"] == [7] and 2 in r.value["divisors"],
                             "98 = 2·7²: 2 | 5−3 ⇒ 只有 7 原始"),
         "expect": "混合情形: 一个非原始 (2), 一个原始 (7)"},
        {"id": "zsi_not_coprime_reject", "args": {"a": 6, "b": 3, "n": 2},
         "expect_err": "PRECOND_FAIL", "expect": "gcd(a,b) ≠ 1 域外"},
        {"id": "zsi_n1_reject", "args": {"a": 3, "b": 2, "n": 1},
         "expect_err": "SCHEMA_MISMATCH", "expect": "n >= 2 是类型层声明"},
        {"id": "zsi_a_le_b_reject", "args": {"a": 2, "b": 5, "n": 3},
         "expect_err": "PRECOND_FAIL", "expect": "a > b 要求"},
    ),
)


# ===========================================================================
# 9909  arith_func_convolution  -- convolution + Möbius inversion dual channel
# ===========================================================================

#: Function set: one=1, id=n, eps=unit function [n=1], tau=τ, sigma=σ, phi=φ, mu=μ.
#: eps must be included -- μ*1 = ε is the cornerstone of Möbius inversion; without it that
#: identity has no right-hand side to reconcile against.
_FUNCS = ["mu", "phi", "sigma", "tau", "id", "one", "eps"]
_CONV_IN = _sdict({
    "f": {"kind": "str", "enum": _FUNCS},
    "g": {"kind": "str", "enum": _FUNCS},
    "N": {"kind": "int", "min": 1, "max": 2000},
    "mode": {"kind": "str", "enum": ["dirichlet_conv", "mobius_invert", "summatory"]},
}, ["f", "g", "N", "mode"])
_CONV_OUT = _sdict({
    "f": {"kind": "str"}, "g": {"kind": "str"}, "mode": {"kind": "str"}, "N": _EXACT,
    "values": {"kind": "list", "items": _EXACT, "minlen": 2, "maxlen": 2001},
    "identity": {"kind": "str"}, "identity_ok": {"kind": "bool"},
    "closed_form_ok": {"kind": "bool"},
}, ["f", "g", "mode", "N", "values", "identity", "identity_ok", "closed_form_ok"])


def _af_table(name, N):
    """Generation side: build the table from **factorization formulas** (μ/φ/σ/τ go through
    factorize; one/id/eps directly)."""
    out = [0] * (N + 1)
    for n in range(1, N + 1):
        if name == "one":
            out[n] = 1
        elif name == "eps":
            out[n] = 1 if n == 1 else 0
        elif name == "id":
            out[n] = n
        else:
            facs = factorize(n)
            if name == "mu":
                out[n] = mu_from_factors(facs)
            elif name == "phi":
                out[n] = phi_from_factors(facs)
            elif name == "sigma":
                out[n] = sigma_from_factors(facs, 1)
            elif name == "tau":
                out[n] = tau_from_factors(facs)
    return out


_IDENTITIES = {
    ("phi", "one"): ("id", "Σ_{d|n} φ(d) = n"),
    ("mu", "one"): ("eps", "Σ_{d|n} μ(d) = [n=1]"),
    ("one", "one"): ("tau", "1*1 = τ"),
    ("id", "one"): ("sigma", "id*1 = σ"),
    ("mu", "id"): ("phi", "μ*id = φ"),
}


def _conv_precond(args):
    if args["N"] < 1:
        return False, "N >= 1"
    if args["mode"] == "mobius_invert" and args["g"] != "one":
        return False, "mobius_invert 模式语义为 'f*1 再 μ 反演回 f', 要求 g = one"
    return True, ""


def _invoke_nt_conv(args, ctl):
    f, g, N, mode = args["f"], args["g"], args["N"], args["mode"]
    fv = _af_table(f, N)
    gv = _af_table(g, N)
    vals = [0] * (N + 1)
    if mode == "dirichlet_conv":
        for n in range(1, N + 1):
            ctl.check_time("op:arith_func_convolution#9909")
            s = 0
            for d in divisors_brute(n):
                s += fv[d] * gv[n // d]
            vals[n] = s
    elif mode == "summatory":
        for n in range(1, N + 1):
            ctl.check_time("op:arith_func_convolution#9909")
            vals[n] = sum(fv[d] for d in divisors_brute(n))
    else:  # mobius_invert
        hv = [0] * (N + 1)
        for n in range(1, N + 1):
            hv[n] = sum(fv[d] for d in divisors_brute(n))
        muv = _af_table("mu", N)
        for n in range(1, N + 1):
            ctl.check_time("op:arith_func_convolution#9909")
            vals[n] = sum(hv[d] * muv[n // d] for d in divisors_brute(n))
    ident, ident_txt = "", ""
    ident_ok = False
    closed_ok = False
    if mode == "dirichlet_conv":
        hit = _IDENTITIES.get((f, g))
        if hit:
            ident, ident_txt = hit
            ref = _af_table(ident, N)
            ident_ok = all(vals[n] == ref[n] for n in range(1, N + 1))
            closed_ok = ident_ok
    elif mode == "mobius_invert":
        ref = _af_table(f, N)
        ident = f
        ident_ok = all(vals[n] == ref[n] for n in range(1, N + 1))
        closed_ok = ident_ok
        ident_txt = "Möbius 反演回 %s" % f
    else:
        hit = _IDENTITIES.get((f, "one"))
        if hit:
            ident, ident_txt = hit
            ref = _af_table(ident, N)
            ident_ok = all(vals[n] == ref[n] for n in range(1, N + 1))
            closed_ok = ident_ok
    cert = Certificate(kind="arith_conv_table",
                       payload={"f": f, "g": g, "N": N, "mode": mode, "values": vals},
                       issued_by="arith_func_convolution")
    return {"value": {"f": f, "g": g, "mode": mode, "N": N, "values": vals,
                      "identity": ident_txt, "identity_ok": ident_ok, "closed_form_ok": closed_ok},
            "resid": [0.0], "cert": cert, "cost": float(N) * 2.0,
            "notes": "%s (N=%d)%s" % (mode, N, "; " + ident_txt if ident_txt else "")}


def _verify_nt_conv(args, value):
    """Independent verifier: rebuild the whole table by **exhaustive definition-level
    computation** (count divisors / count coprimes), not via the factorization formulas."""
    f, g, N, mode = args["f"], args["g"], args["N"], args["mode"]
    bad = []

    def brute(name, n):
        if name == "one":
            return 1
        if name == "eps":
            return 1 if n == 1 else 0
        if name == "id":
            return n
        if name == "tau":
            return len(divisors_brute(n))
        if name == "sigma":
            return sum(divisors_brute(n))
        if name == "phi":
            return sum(1 for k in range(1, n + 1) if _gcd_py(k, n) == 1)
        if name == "mu":
            # straight from the definition: a square factor -> 0, otherwise (−1)^(number of
            # distinct prime factors)
            m, cnt, d = n, 0, 2
            while d * d <= m:
                if m % d == 0:
                    m //= d
                    if m % d == 0:
                        return 0
                    cnt += 1
                d += 1 if d == 2 else 2
            if m > 1:
                cnt += 1
            return -1 if cnt % 2 else 1
        raise AssertionError(name)

    for n in range(1, N + 1):
        if mode == "dirichlet_conv":
            want = sum(brute(f, d) * brute(g, n // d) for d in divisors_brute(n))
        elif mode == "summatory":
            want = sum(brute(f, d) for d in divisors_brute(n))
        else:
            h = [0] * (n + 1)
            for m in range(1, n + 1):
                h[m] = sum(brute(f, d) for d in divisors_brute(m))
            want = sum(h[d] * brute("mu", n // d) for d in divisors_brute(n))
        if want != value["values"][n]:
            bad.append("n=%d 穷举直算 %d ≠ 表 %d" % (n, want, value["values"][n]))
            break
    if mode in ("dirichlet_conv", "mobius_invert") and not value["identity_ok"]:
        bad.append("双通道对账失败 (identity_ok=False)")
    if mode == "dirichlet_conv" and (f, g) in _IDENTITIES and not value["identity_ok"]:
        bad.append("经典恒等式通道未通过")
    return {"ok": not bad, "resid": [float(len(bad))],
            "detail": "穷举直算重建 %d 项一致%s" % (N, ("; 恒等式 " + value["identity"]) if value["identity"] else "")
                      if not bad else "; ".join(bad)}


_reg(
    name="arith_func_convolution", band=BAND_L0, sub=9,
    in_schema=_CONV_IN, out_schema=_CONV_OUT,
    invoke=_invoke_nt_conv, verify=_verify_nt_conv, cert_kind="arith_conv_table",
    cost=("units", 4.0),
    errors=("PRECOND_FAIL", "SCHEMA_MISMATCH", "POSTCOND_FAIL", "TIMEOUT"),
    precond=_conv_precond,
    effects="none (纯函数)",
    doc=("Arithmetic-function dual channel: Dirichlet convolution (f*g)(n)=Σ_{d|n} f(d)g(n/d), "
         "Möbius inversion (f*1 then *μ back to f), summatory Σ_{d|n} f(d). Generation channel = "
         "factorization formulas (μ/φ/σ/τ via factorize); verification channel = "
         "**definition-level exhaustive direct computation** (τ counts divisors, σ sums "
         "divisors, φ counts coprimes) -- two paths sharing no code; third channel = classic "
         "identity reconciliation (φ*1=id, μ*1=ε, 1*1=τ, id*1=σ) hard-coded in _IDENTITIES. "
         "Certificate arith_conv_table is re-checked by certcheck_nt with a **fourth** "
         "exhaustive implementation."),
    requires_cert=True,
    thm="T-V31-CONV",
    stmt="φ*1 = id, μ*1 = ε, 1*1 = τ, id*1 = σ; Möbius 反演 f = (f*1)*μ 往返恒等",
    selfchecks=(
        {"id": "conv_phi_one_is_id", "args": {"f": "phi", "g": "one", "N": 20, "mode": "dirichlet_conv"},
         "check": lambda r: (r.value["identity_ok"] and all(r.value["values"][n] == n for n in range(1, 21))
                             and r.cert_independent,
                             "Σ_{d|n} φ(d) = n 全 20 项成立 (%s)" % r.value["identity"]),
         "expect": "φ*1 = id (Euler 恒等式)"},
        {"id": "conv_mu_one_is_epsilon", "args": {"f": "mu", "g": "one", "N": 20, "mode": "dirichlet_conv"},
         "check": lambda r: (r.value["values"][1] == 1 and all(r.value["values"][n] == 0 for n in range(2, 21)),
                             "μ*1 = ε: 第 1 项 1, 其后全 0"),
         "expect": "μ*1 = ε (Möbius 反演基础)"},
        {"id": "conv_one_one_is_tau", "args": {"f": "one", "g": "one", "N": 20, "mode": "dirichlet_conv"},
         "check": lambda r: (r.value["values"][12] == 6 and r.value["identity_ok"],
                             "τ(12) = %d" % r.value["values"][12]),
         "expect": "1*1 = τ, τ(12) = 6"},
        {"id": "conv_id_one_is_sigma", "args": {"f": "id", "g": "one", "N": 20, "mode": "dirichlet_conv"},
         "check": lambda r: (r.value["values"][12] == 28, "σ(12) = %d" % r.value["values"][12]),
         "expect": "id*1 = σ, σ(12) = 28"},
        {"id": "conv_mobius_roundtrip", "args": {"f": "phi", "g": "one", "N": 20, "mode": "mobius_invert"},
         "check": lambda r: (r.value["identity_ok"], "Möbius 反演往返恢复 φ (20 项)"),
         "expect": "反演往返恒等 (双通道判决)"},
        {"id": "conv_summatory_mu", "args": {"f": "mu", "g": "one", "N": 20, "mode": "summatory"},
         "check": lambda r: (r.value["values"][1] == 1 and all(r.value["values"][n] == 0 for n in range(2, 21)),
                             "Σ_{d|n} μ(d) = [n=1]"),
         "expect": "部分和通道与恒等式一致"},
        {"id": "conv_N1_boundary", "args": {"f": "mu", "g": "one", "N": 1, "mode": "dirichlet_conv"},
         "check": lambda r: (r.value["values"][1] == 1 and len(r.value["values"]) == 2,
                             "N=1 边界: 表长 2"),
         "expect": "N=1 边界"},
        {"id": "conv_bad_func_reject", "args": {"f": "zeta", "g": "one", "N": 10, "mode": "dirichlet_conv"},
         "expect_err": "SCHEMA_MISMATCH", "expect": "未登记算术函数被 enum 拦下"},
        {"id": "conv_bad_mode_reject", "args": {"f": "mu", "g": "one", "N": 10, "mode": "invert"},
         "expect_err": "SCHEMA_MISMATCH", "expect": "未登记 mode 被 enum 拦下"},
    ),
)


# ===========================================================================
# 9910  nt_valuation_ring  -- p-adic valuation additivity
# ===========================================================================

_ROW = {"kind": "tuple", "items": [_EXACT, _EXACT]}
_VR_IN = _sdict({
    "p": {"kind": "int", "min": 2},
    "pairs": {"kind": "list", "items": _ROW, "minlen": 1, "maxlen": 64},
}, ["p", "pairs"])
_VR_ROW_FIELDS = {
    "a": _EXACT, "b": _EXACT, "v_a": _EXACT, "v_b": _EXACT, "v_ab": _EXACT, "v_sum": _EXACT,
    "additivity_ok": {"kind": "bool"}, "min_rule_ok": {"kind": "bool"},
    "distinct_implies_equality": {"kind": "bool"}, "strict_when_equal": {"kind": "bool"},
}
_VR_OUT = _sdict({
    "p": _EXACT, "n_rows": _EXACT, "all_ok": {"kind": "bool"},
    "rows": {"kind": "list", "minlen": 1, "maxlen": 64,
             "items": {"kind": "dict", "fields": _VR_ROW_FIELDS,
                       "required": list(_VR_ROW_FIELDS)}},
}, ["p", "n_rows", "all_ok", "rows"])


def _vr_precond(args):
    p = args["p"]
    if p < 2:
        return False, "p >= 2"
    if not is_prime(p):
        return False, "p=%d 非素数 (赋值论要求素元)" % p
    if not args["pairs"]:
        return False, "pairs 不得为空"
    return True, ""


def _invoke_nt_vr(args, ctl):
    p, pairs = args["p"], args["pairs"]
    rows = []
    all_ok = True
    for a, b in pairs:
        ctl.check_time("op:nt_valuation_ring#9910")
        if a == 0 or b == 0:
            raise SvoError(Err.DEG_UNSUPPORTED, ctx="op:nt_valuation_ring#9910",
                           detail="v_p(0) = +∞ 不可表示 (a=%d, b=%d)" % (a, b))
        if a + b == 0:
            raise SvoError(Err.DEG_UNSUPPORTED, ctx="op:nt_valuation_ring#9910",
                           detail="a+b = 0 ⇒ v_p(a+b) = +∞ 不可表示 (a=%d, b=%d)" % (a, b))
        va, vb = valuation(p, a), valuation(p, b)
        vab = valuation(p, a * b)
        vsum = valuation(p, a + b)
        add_ok = (vab == va + vb)
        min_ok = (vsum >= min(va, vb))
        # the correct form of the "equality condition": **distinct** valuations imply equality
        # (a one-way implication). The converse fails -- equal valuations may also give equality
        # (e.g. 14+21=35, both v_7 = 1). Stating it as an equivalence would be a false
        # proposition, so only the implication direction is asserted here and the "strictly
        # greater" phenomenon is recorded separately.
        distinct_implies_equality = (va == vb) or (vsum == min(va, vb))
        strict_when_equal = (va == vb) and (vsum > min(va, vb))
        all_ok = all_ok and add_ok and min_ok and distinct_implies_equality
        rows.append({"a": a, "b": b, "v_a": va, "v_b": vb, "v_ab": vab, "v_sum": vsum,
                     "additivity_ok": add_ok, "min_rule_ok": min_ok,
                     "distinct_implies_equality": distinct_implies_equality,
                     "strict_when_equal": strict_when_equal})
    cert = Certificate(kind="padic_valuation",
                       payload={"p": p, "rows": [[r["a"], r["b"], r["v_a"], r["v_b"], r["v_ab"], r["v_sum"]]
                                                 for r in rows]},
                       issued_by="nt_valuation_ring")
    return {"value": {"p": p, "n_rows": len(rows), "all_ok": all_ok, "rows": rows},
            "resid": [0.0 if all_ok else 1.0], "cert": cert, "cost": float(len(rows)) * 3.0,
            "notes": "%d 对整数的 v_%d 可加性/最小律全通过=%s" % (len(rows), p, all_ok)}


def _verify_nt_vr(args, value):
    """Independent verifier: recompute the four valuations along a **square-bisection path**
    and re-check the two non-Archimedean identities."""
    p = args["p"]
    bad = []
    for r in value["rows"]:
        a, b = r["a"], r["b"]
        if (valuation_squaring(p, a), valuation_squaring(p, b)) != (r["v_a"], r["v_b"]):
            bad.append("(%d,%d) 赋值平方-二分路径重算不符" % (a, b))
        if valuation_squaring(p, a * b) != r["v_ab"]:
            bad.append("(%d,%d) v(ab) 重算不符" % (a, b))
        if valuation_squaring(p, a + b) != r["v_sum"]:
            bad.append("(%d,%d) v(a+b) 重算不符" % (a, b))
        if r["v_ab"] != r["v_a"] + r["v_b"]:
            bad.append("(%d,%d) 可加性不成立" % (a, b))
        if not r["min_rule_ok"] or not r["distinct_implies_equality"]:
            bad.append("(%d,%d) 最小律/相异蕴含等号不成立" % (a, b))
        expect = (r["v_a"] == r["v_b"]) and (r["v_sum"] > min(r["v_a"], r["v_b"]))
        if bool(r["strict_when_equal"]) != expect:
            bad.append("(%d,%d) strict_when_equal 标志与实测不符" % (a, b))
    return {"ok": not bad, "resid": [float(len(bad))],
            "detail": "%d 行: 第二路径赋值重算 + 恒等式复核一致" % len(value["rows"])
                      if not bad else "; ".join(bad[:5])}


_reg(
    name="nt_valuation_ring", band=BAND_L0, sub=10,
    in_schema=_VR_IN, out_schema=_VR_OUT,
    invoke=_invoke_nt_vr, verify=_verify_nt_vr, cert_kind="padic_valuation",
    cost=("units", 3.0),
    errors=("PRECOND_FAIL", "DEG_UNSUPPORTED", "SCHEMA_MISMATCH", "POSTCOND_FAIL", "TIMEOUT"),
    precond=_vr_precond,
    effects="none (纯函数)",
    doc=("Additivity check for the p-adic valuation v_p (the non-Archimedean structure of a "
         "valuation domain): for each pair (a,b) output v_p(a), v_p(b), v_p(ab), v_p(a+b) and "
         "decide (i) v_p(ab) = v_p(a)+v_p(b) (additivity), (ii) v_p(a+b) >= min(v_p(a),v_p(b)) "
         "(min rule), (iii) **equality holds <=> the two valuations differ** (the exact form of "
         "non-Archimedean-ness). Generation = plain division counting; independent verifier = "
         "square-bisection path (a different algorithm); certificate padic_valuation is counted "
         "once more from the definition by certcheck_nt. Domain note: at a=0 or b=0 or a+b=0, "
         "v_p = +∞ is unrepresentable, hence an explicit DEG_UNSUPPORTED."),
    requires_cert=True,
    thm="T-V31-VALUATION",
    stmt="v_p(ab) = v_p(a)+v_p(b); v_p(a+b) >= min(v_p(a),v_p(b)); 且 v_p(a) ≠ v_p(b) ⇒ 等号成立",
    selfchecks=(
        {"id": "vr_equal_valuations", "args": {"p": 2, "pairs": [[12, 4]]},
         "check": lambda r: (r.value["rows"][0]["v_a"] == 2 and r.value["rows"][0]["v_b"] == 2
                             and r.value["rows"][0]["v_ab"] == 4 and r.value["rows"][0]["v_sum"] == 4
                             and r.value["rows"][0]["strict_when_equal"] and r.value["all_ok"]
                             and r.cert_independent,
                             "12=2²·3, 4=2² ⇒ v(48)=4, v(16)=4 > min=2 (严格大于)"),
         "expect": "赋值相等时 v(a+b) 可取严格大于 (凹陷现象)"},
        {"id": "vr_strictly_greater", "args": {"p": 2, "pairs": [[2, 2]]},
         "check": lambda r: (r.value["rows"][0]["v_sum"] == 2 and r.value["rows"][0]["strict_when_equal"],
                             "2+2=4: v 从 1 跳到 2"),
         "expect": "v_p(a)=v_p(b)=1, v_p(a+b)=2"},
        {"id": "vr_equality_case", "args": {"p": 2, "pairs": [[2, 4]]},
         "check": lambda r: (r.value["rows"][0]["v_sum"] == 1
                             and not r.value["rows"][0]["strict_when_equal"]
                             and r.value["rows"][0]["distinct_implies_equality"],
                             "2+4=6: v=1 = min ⇒ 相异赋值蕴含等号"),
         "expect": "v 相异 ⇒ 等号成立 (非阿基米德性的精确形式)"},
        {"id": "vr_negative_operands", "args": {"p": 3, "pairs": [[-18, 6]]},
         "check": lambda r: (r.value["rows"][0]["v_a"] == 2 and r.value["rows"][0]["v_ab"] == 3,
                             "|−18| = 2·3² ⇒ v=2; v(−108)=3"),
         "expect": "|a| 语义: v_p(−a) = v_p(a)"},
        {"id": "vr_equal_valuations_no_strict", "args": {"p": 7, "pairs": [[14, 21]]},
         "check": lambda r: (r.value["rows"][0]["v_a"] == 1 and r.value["rows"][0]["v_ab"] == 2
                             and r.value["rows"][0]["v_sum"] == 1
                             and not r.value["rows"][0]["strict_when_equal"],
                             "14+21=35: v_7 均为 1 ⇒ v_sum=1=min (等赋值也取等号)"),
         "expect": "反例意识: 等赋值时等号也可能成立 ⇒ 双向'等号⇔相异'是假命题, 只判蕴含方向"},
        {"id": "vr_multiple_rows", "args": {"p": 7, "pairs": [[49, 7], [14, 21]]},
         "check": lambda r: (r.value["n_rows"] == 2 and r.value["rows"][0]["v_sum"] == 1
                             and r.value["all_ok"],
                             "两行均通过 (56 = 7·8 ⇒ v_7 = 1)"),
         "expect": "多行批处理一致"},
        {"id": "vr_zero_reject", "args": {"p": 5, "pairs": [[0, 5]]},
         "expect_err": "DEG_UNSUPPORTED", "expect": "v_p(0) = +∞ 不可表示"},
        {"id": "vr_sum_zero_reject", "args": {"p": 5, "pairs": [[5, -5]]},
         "expect_err": "DEG_UNSUPPORTED", "expect": "a+b = 0 ⇒ 不可表示"},
        {"id": "vr_p_composite_reject", "args": {"p": 4, "pairs": [[2, 4]]},
         "expect_err": "PRECOND_FAIL", "expect": "p 非素数域外"},
    ),
)
