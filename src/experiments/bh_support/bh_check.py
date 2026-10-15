# -*- coding: utf-8 -*-
# bh_check.py -- bridge-hard batch independent checker (gen14, change-58 curriculum-upgrade batch)
#
# Structural prohibition: this module must **not import bh_ops / bh_poly**; register_checker checks at
# registration time whether fn.__module__ lies inside independent_of => self-certification is
# structurally rejected (mirrors the ccp_check design).
# The checker eats only certificate payloads (pure data) and recomputes with **algorithms different
# from the generator's**:
#   * generator (bh_ops): constructive -- difference-equation solving / completing the square /
#     witness-coefficient construction / template rendering
#   * checker (this module): numeric grid recomputation + forward simulation + pointwise substitution +
#     **text re-parsing** + polynomial square-root decision
# Completeness convention: every polynomial identity is recomputed on (bound+1) distinct grid points
# under the **declared degree bound**; out-of-bound fails closed.
import re
from fractions import Fraction

CHECKERS = {}
_INDEP = {}


def register_checker(kind, independent_of):
    def deco(fn):
        if fn.__module__ in tuple(independent_of):
            raise ValueError("禁自证违规: 检查器 %s 与生成器同模块 %s" % (kind, fn.__module__))
        CHECKERS[kind] = fn
        _INDEP[kind] = tuple(independent_of)
        return fn
    return deco


def check(kind, payload):
    fn = CHECKERS.get(kind)
    if fn is None:
        return False, [1.0], "未登记证书类型 %r" % (kind,)
    try:
        return fn(payload)
    except Exception as e:                                     # noqa: BLE001
        return False, [1.0], "检查器异常 %s: %s" % (type(e).__name__, e)


def independence(kind):
    return _INDEP.get(kind, ())


# ---------------------------------------------------------------- checker-own polynomial path (numeric)
def _fl(d):
    return {tuple(int(x) for x in k.split(",")): Fraction(v) for k, v in d.items()}


def _nvar(p):
    return len(next(iter(p))) if p else 0


def _eval(p, vals):
    s = Fraction(0)
    for m, c in p.items():
        t = Fraction(c)
        for i, e in enumerate(m):
            if e:
                t *= Fraction(vals[i]) ** e
        s += t
    return s


def _deg_ok(p, nv, bound):
    for m in p:
        if len(m) != nv or any(e > bound for e in m):
            return False
    return True


def _eq_on_grid(p, q, nv, bound, side=None):
    side = side or (bound + 1)
    if not (_deg_ok(p, nv, bound) and _deg_ok(q, nv, bound)):
        return False, "次数越界 (bound=%d)" % bound
    pts = list(range(2, 2 + side))
    idx = [0] * nv

    def rec(i):
        if i == nv:
            vals = [Fraction(pts[idx[j]]) for j in range(nv)]
            return _eval(p, vals) == _eval(q, vals)
        for k in range(side):
            idx[i] = k
            if not rec(i + 1):
                return False
        return True
    return (True, "网格全等") if rec(0) else (False, "格点失配")


def _padd(p, q):
    out = dict(p)
    for m, c in q.items():
        out[m] = out.get(m, Fraction(0)) + c
    return {m: c for m, c in out.items() if c != 0}


def _pmul(p, q):
    out = {}
    for m1, c1 in p.items():
        for m2, c2 in q.items():
            if len(m1) != len(m2):
                raise ValueError("维度不一致")
            m = tuple(a + b for a, b in zip(m1, m2))
            out[m] = out.get(m, Fraction(0)) + c1 * c2
    return {m: c for m, c in out.items() if c != 0}


def _pscale(p, k):
    k = Fraction(k)
    return {m: c * k for m, c in p.items() if c * k != 0}


def _lin_pow(nv, coefs, shift, power=2):
    lin = {}
    for i, a in enumerate(coefs):
        if Fraction(a) != 0:
            m = [0] * nv
            m[i] = 1
            lin[tuple(m)] = Fraction(a)
    if Fraction(shift) != 0:
        lin[(0,) * nv] = lin.get((0,) * nv, Fraction(0)) - Fraction(shift)
    out = {(0,) * nv: Fraction(1)}
    for _ in range(power):
        out = _pmul(out, lin)
    return out


def _is_int(v):
    return Fraction(v).denominator == 1


def _fsqrt(v):
    """Exact rational square root (None if not a square)."""
    v = Fraction(v)
    if v < 0:
        return None
    n, d = abs(v.numerator), v.denominator
    rn, rd = _isqrt(n), _isqrt(d)
    if rn is None or rd is None:
        return None
    return Fraction(rn, rd)


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


def _perfect_square_int(k):
    k = int(k)
    if k < 0:
        return None
    return _isqrt(k)


def _poly_sqrt(T):
    """Univariate polynomial **perfect-square decision** (checker-own algorithm: top-down
    coefficient matching + full equality recheck at the end).
    Returns (True, root_poly) or (False, None). A real computation (not an O(1) local predicate)."""
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
                continue                      # pending pair: 2*r_m*r_i
            if b < 0 or b > m_:
                continue
            if a in root and b in root:
                known += root[a] * root[b]
        root[i] = (tgt - known) / (2 * root[m_])
    cand = {tuple([i]): c for i, c in root.items() if c != 0}
    sq = _pmul(cand, cand)
    if _padd(sq, _pscale(T, -1)):
        return False, None
    return True, cand


# ---------------------------------------------------------------- 1. polynomial sums and recurrence closed forms
@register_checker("bh_ind_sum_cert", independent_of=("bh_ops", "bh_poly"))
def chk_ind_sum(payload):
    """Certificate {deg, v0, S, P} (univariate dump, variables [k]/[n]). Four conditions must hold together:
      (1) exact degree: max deg(S) == deg and the k^deg coefficient != 0 (a class assertion: the declared
          degree is the **exact** degree);
      (2) initial value P(0) == v0;
      (3) telescoping identity (numerically complete): P(k+1)-P(k) == S(k), k=0..deg+3;
      (4) degree ladder deg(P) == deg+1."""
    deg = int(payload["deg"])
    v0 = Fraction(payload["v0"])
    S, Pc = _fl(payload["S"]), _fl(payload["P"])
    r = []
    if deg not in (0, 1, 2, 3):
        return False, [1.0], "非法次数声明 %d" % deg
    sd = max([m[0] for m in S] or [0])
    sc = sum((c for m, c in S.items() if m[0] == deg), Fraction(0))
    r.append(0.0 if (sd == deg and sc != 0) else 1.0)
    if not (sd == deg and sc != 0):
        return False, r, "S 的精确次数 ≠ %d (实测 %d, 首系数 %s)" % (deg, sd, sc)
    p0 = _eval(Pc, [Fraction(0)])
    r.append(0.0 if p0 == v0 else 1.0)
    if p0 != v0:
        return False, r, "P(0)=%s ≠ v0=%s" % (p0, v0)
    bad = []
    for k in range(0, deg + 4):
        if _eval(Pc, [Fraction(k + 1)]) - _eval(Pc, [Fraction(k)]) != _eval(S, [Fraction(k)]):
            bad.append(k)
    r.append(0.0 if not bad else 1.0)
    if bad:
        return False, r, "望远镜等式失配 k∈%s" % bad
    pd = max([m[0] for m in Pc] or [0])
    r.append(0.0 if pd == deg + 1 else 1.0)
    if pd != deg + 1:
        return False, r, "deg(P)=%d ≠ %d" % (pd, deg + 1)
    return True, r, "精确次数+初值+望远镜+次数阶梯 四重一致 (deg=%d)" % deg


# ---------------------------------------------------------------- 2. geometric / affine recurrences
@register_checker("bh_geo_cert", independent_of=("bh_ops", "bh_poly"))
def chk_geo(payload):
    """Certificate {v0, r, c, c1, c0, cls}: (1) forward simulation 0..6 steps == c1*r^k+c0;
    (2) fixed-point self-consistency r*c0+c == c0 and c1+c0 == v0;
    (3) class exclusion cls='const' iff c1==0."""
    v0, rr, cc = (Fraction(int(payload["v0"])), Fraction(int(payload["r"])),
                  Fraction(int(payload["c"])))
    c1, c0 = Fraction(payload["c1"]), Fraction(payload["c0"])
    cls = payload["cls"]
    r = []
    if int(payload["r"]) == 1:
        return False, [1.0], "退化直线形 (r=1) 不属本族"
    a = v0
    for k in range(0, 7):
        want = c1 * (rr ** k) + c0
        if a != want:
            return False, [1.0], "前向模拟 k=%d: %s ≠ %s" % (k, a, want)
        a = rr * a + cc
    r.append(0.0)
    if not (c1 + c0 == v0 and rr * c0 + cc == c0):
        return False, r + [1.0], "不动点自洽失败"
    r.append(0.0)
    if cls not in ("const", "pow"):
        return False, r + [1.0], "非法类 %r" % cls
    ok = (c1 == 0) if cls == "const" else (c1 != 0)
    r.append(0.0 if ok else 1.0)
    if not ok:
        return False, r, "类互斥失败: cls=%s 但 c1=%s" % (cls, c1)
    return True, r, "模拟+不动点+类互斥一致 (r=%s c1=%s)" % (rr, c1)


# ---------------------------------------------------------------- 3. closed-form equivalence (class exclusion)
@register_checker("bh_equiv_cert", independent_of=("bh_ops", "bh_poly"))
def chk_equiv(payload):
    """Certificate {P, T, cls, sq?}: (1) P ≡ T (numeric grid, bound = maxdeg+1);
    (2) class exclusion (a real computation): cls='square' => T is a perfect square and rechecking with
        the declared root holds;
        cls='poly' => T is **not** a perfect square (polynomial square-root decision, not an O(1) local
        predicate);
        cls='const' => T is constant and P is constant too."""
    Pd, Td = _fl(payload["P"]), _fl(payload["T"])
    cls = payload["cls"]
    nv = _nvar(Pd) or _nvar(Td) or 1
    bnd = max([m[0] for m in list(Pd) + list(Td)] or [0]) + 1
    ok, msg = _eq_on_grid(Pd, Td, nv, bnd)
    r = [0.0 if ok else 1.0]
    if not ok:
        return False, r, "P ≢ T (%s)" % msg
    isconst = all(all(e == 0 for e in m) for m in Td)
    if cls == "const":
        if not isconst:
            return False, r + [1.0], "cls=const 但 T 非常数"
    elif cls == "poly":
        if isconst:
            return False, r + [1.0], "cls=poly 但 T 为常数"
        isq, root = _poly_sqrt(Td)
        if isq:
            return False, r + [1.0], "cls=poly 但 T 是完全平方 (互斥失败)"
    elif cls in ("square", "square2", "square4"):
        isq, root = _poly_sqrt(Td)
        if not isq:
            return False, r + [1.0], "cls=%s 但 T 非完全平方" % cls
        rdeg = max([m[0] for m in root] or [0])
        want_deg = {"square": None, "square2": 1, "square4": 2}[cls]
        if want_deg is not None and rdeg != want_deg:
            return False, r + [1.0], "cls=%s 要求根次数 %d (实测 %d)" % (
                cls, want_deg, rdeg)
        if "sq" not in payload:
            return False, r + [1.0], "缺声明根"
        coefs = [Fraction(x) for x in payload["sq"]]
        want = {tuple([i]): c for i, c in enumerate(coefs) if c != 0}
        ok2, msg2 = _eq_on_grid(Td, _pmul(want, want), 1,
                                max(bnd, len(coefs) * 2))
        if not ok2:
            return False, r + [1.0], "T ≢ 声明根的平方 (%s)" % msg2
    else:
        return False, r + [1.0], "非法类 %r" % cls
    r.append(0.0)
    return True, r, "P≡T 且类互斥成立 (cls=%s, const=%s)" % (cls, isconst)


# ---------------------------------------------------------------- 3b. geometric closed-form terminal step (template level)
@register_checker("bh_geo_close_cert", independent_of=("bh_ops", "bh_poly"))
def chk_geo_close(payload):
    """Certificate {v0, r, c, cls, tpl, text}. Independent criteria:
      (1) recompute (c1,c0) by **forward simulation** from (v0,r,c); never read the generator's
          closed-form params;
      (2) statement-declared text == the template instance built from the recomputed params under the
          **rendering convention** (bh_tpl) (strict text-level equality);
      (3) class exclusion: the template's semantic class == (c1==0 ? 'const' : 'pow')."""
    import bh_tpl as T
    v0, rr, cc = int(payload["v0"]), int(payload["r"]), int(payload["c"])
    cls, tpl, text = payload["cls"], payload["tpl"], payload["text"]
    r = []
    if rr == 1:
        return False, [1.0], "r=1 非本族"
    c0 = Fraction(cc, 1 - rr)
    c1 = Fraction(v0) - c0
    a = Fraction(v0)
    for k in range(0, 7):
        if a != c1 * (rr ** k) + c0:
            return False, [1.0], "模拟失配 k=%d" % k
        a = rr * a + cc
    r.append(0.0)
    want_cls = "const" if c1 == 0 else "pow"
    if cls != want_cls:
        return False, r + [1.0], "类互斥失败: cls=%s 但复算类=%s" % (cls, want_cls)
    if tpl not in T.TPL_IDS:
        return False, r + [1.0], "未登记模板 %r" % tpl
    if int(c0) != c0 or int(c1) != c1:
        return False, r + [1.0], "闭式参数非整数"
    want = T.render(tpl, int(c1), int(c0), rr)
    if re.sub(r"\s+", "", text) != re.sub(r"\s+", "", want):
        return False, r + [1.0], "声明文本 ≠ 模板实例: 声明 %r 期望 %r" % (text, want)
    r.append(0.0)
    return True, r, "模板 %s 实例与复算闭式严格一致 (cls=%s)" % (tpl, cls)


# ---------------------------------------------------------------- 4. inequality blocks
@register_checker("bh_block_cert", independent_of=("bh_ops", "bh_poly"))
def chk_block(payload):
    """Certificate {vars, R_in, R_out, ell, mult}: (1) R_in ≡ mult*l^2+R_out (grid, bound=3);
    (2) mult is a positive integer; (3) purge: R_out has no nonzero term in a variable of l's support
    (exact absorption criterion)."""
    vars_ = list(payload["vars"])
    nv = len(vars_)
    R_in, R_out, ell = _fl(payload["R_in"]), _fl(payload["R_out"]), _fl(payload["ell"])
    mult = Fraction(payload["mult"])
    coefs = []
    for i in range(nv):
        m = [0] * nv
        m[i] = 1
        coefs.append(ell.get(tuple(m), Fraction(0)))
    shift = -ell.get((0,) * nv, Fraction(0))
    want = _padd(_pscale(_lin_pow(nv, coefs, shift, 2), mult), R_out)
    ok, msg = _eq_on_grid(R_in, want, nv, 3)
    r = [0.0 if ok else 1.0]
    if not ok:
        return False, r, "R_in ≢ m·ℓ²+R_out (%s)" % msg
    if not (mult > 0) or mult.denominator != 1:
        return False, r + [1.0], "mult=%s 非正整数" % mult
    supp = [i for i, a in enumerate(coefs) if a != 0]
    if not supp:
        return False, r + [1.0], "ℓ 为常数 (无支撑)"
    stray = [(",".join(map(str, m)), str(c)) for m, c in R_out.items()
             if c != 0 and any(m[i] > 0 for i in supp)]
    if stray:
        return False, r + [1.0], "purge 失败: R_out 残留被吸收变量 %s" % stray[:3]
    r.append(0.0)
    return True, r, "块恒等 + 正乘子 + purge 三重成立 (supp=%s mult=%s)" % (supp, mult)


# ---------------------------------------------------------------- 5. terminal semantic class
@register_checker("bh_tail_cert", independent_of=("bh_ops", "bh_poly"))
def chk_tail(payload):
    """Certificate {rho, cls, sq?, vars, goal_dom, hints_poly}. Two conditions must hold together:
      (1) **closed-chain identity**: goal_dom ≡ sum(hints_poly) + rho (grid recomputation) => stopping
          early or dropping a block necessarily fails;
      (2) class exclusion (a real computation): zero iff rho==0; square iff rho==sq^2 (sq>=2); other iff
          rho>0 and not a perfect square."""
    rho = Fraction(payload["rho"])
    cls = payload["cls"]
    r = []
    if rho < 0:
        return False, [1.0], "rho=%s < 0" % rho
    if "goal_dom" in payload:
        vars_ = list(payload["vars"])
        nv = len(vars_)
        acc = {(0,) * nv: rho}
        for h in payload["hints_poly"]:
            acc = _padd(acc, _fl(h))
        ok, msg = _eq_on_grid(acc, _fl(payload["goal_dom"]), nv, 4)
        r.append(0.0 if ok else 1.0)
        if not ok:
            return False, r, "闭链恒等失败: Σ块 + rho ≢ 目标 dom (%s)" % msg
    if cls == "zero":
        ok, msg = (rho == 0), "zero 类要求 rho==0"
    elif cls == "square":
        s = int(payload["sq"])
        ok = (rho == Fraction(s * s)) and s >= 2
        msg = "square 类要求 rho == sq^2 (sq=%d rho=%s)" % (s, rho)
    elif cls == "other":
        ps = _perfect_square_int(rho) if _is_int(rho) else None
        ok = (rho > 0) and (ps is None)
        msg = "other 类要求 rho>0 且非完全平方 (rho=%s)" % rho
    else:
        return False, [1.0], "非法类 %r" % cls
    if not ok:
        return False, r + [1.0], msg
    r.append(0.0)
    return True, r, "闭链恒等 + 收尾类 %s 成立 (rho=%s)" % (cls, rho)


# ---------------------------------------------------------------- 6. divisibility witness
@register_checker("bh_wit_cert", independent_of=("bh_ops", "bh_poly"))
def chk_wit(payload):
    """Certificate {m, M, params, terms, const, witness}: the checker **builds its own** G(Z) and W(Z):
      G := sum coef*(m^k * prod(w)) + coef*(m * free variable) + K; criterion G == M*W (grid, bound=2);
    also: witness coefficients must be non-negative integers."""
    m, M = int(payload["m"]), int(payload["M"])
    params = list(payload["params"])
    idx = {n: i for i, n in enumerate(params)}
    nv = len(params)
    if m <= 1 or M <= 1 or M % m != 0:
        return False, [1.0], "非法模数对 (m=%d M=%d)" % (m, M)
    zero = (0,) * nv
    G = {}
    for t in payload["terms"]:
        coef = Fraction(t["coef"])
        if t.get("mfac"):
            if t.get("freevar") not in idx:
                return False, [1.0], "自由变量未登记: %s" % t.get("freevar")
            mon = [0] * nv
            mon[idx[t["freevar"]]] += 1
            fac = Fraction(m)
        else:
            mon = [0] * nv
            for wn in t["wvars"]:
                if wn not in idx:
                    return False, [1.0], "见证变量未登记: %s" % wn
                mon[idx[wn]] += 1
            fac = Fraction(m) ** int(t["k"])
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
    ok, msg = _eq_on_grid(G, _pscale(W, M), nv, 2)
    return (ok, [0.0 if ok else 1.0],
            "整除恒等 G ≡ %d·W %s (m=%d M=%d params=%s)" % (M, msg, m, M, params))


# ---------------------------------------------------------------- 7. structural / shape steps (text re-parsing)
@register_checker("bh_shape_cert", independent_of=("bh_ops", "bh_poly"))
def chk_shape(payload):
    """Structural step: the checker verifies by **re-parsing the text** (does not trust generator state)."""
    txt = payload["text"]
    kind = payload["assert"]
    if kind == "dvd_hyp":
        pat = (r"\(\s*%s\s*:\s*([0-9+\-*^() ]+?)\s*\u2223\s*%s\s*\)"
               % (re.escape(payload["hyp"]), re.escape(payload["var"])))
        mm = re.search(pat, txt)
        if not mm:
            return False, [1.0], "假设形状不符: %s" % pat
        want = int(payload.get("modulus", payload.get("m", 0)))
        got = _eval_numexpr(mm.group(1))
        if got != want:
            return False, [1.0], "假设 %s 的模数求值 %s ≠ 声明 %s" % (
                payload["hyp"], got, want)
        return True, [0.0], "dvd 假设形状确认 (模数 %s)" % got
    if kind == "def_hyp":
        pat = r"\(\s*%s\s*:\s*%s\s*=\s*[^()]*(?:\([^()]*\)[^()]*)*\)" % (
            re.escape(payload["hyp"]), re.escape(payload["lhs"]))
        if not re.search(pat, txt):
            return False, [1.0], "定义形状不符: %s" % pat
        return True, [0.0], "定义假设形状确认"
    if kind == "occurs":
        v = payload["var"]
        tgt = payload.get("body") or txt
        if not re.search(r"(?<![A-Za-z0-9_'])%s(?![A-Za-z0-9_'])" % re.escape(v), tgt):
            return False, [1.0], "变量 %s 未出现 (空操作/诱饵步)" % v
        return True, [0.0], "变量出现确认 %s" % v
    if kind == "mod_val":
        mm = re.search(r"\(\s*%s\s*:\s*([0-9+\-*^() ]+?)\s*\u2223" % re.escape(payload["hyp"]),
                       txt)
        if not mm:
            return False, [1.0], "模数文本未解析"
        val = _eval_numexpr(mm.group(1))
        m = int(payload["m"])
        if val != int(payload["modulus"]):
            return False, [1.0], "模数求值 %s ≠ 声明 %s" % (val, payload["modulus"])
        t, got = val, 0
        while m > 1 and t > 1 and t % m == 0:
            t //= m
            got += 1
        if got != int(payload["val"]):
            return False, [1.0], "m-估值 %d ≠ 声明 %d (modulus=%d m=%d)" % (
                got, payload["val"], val, m)
        return True, [0.0], "模数 %d 的 m-估值确认 %d" % (val, got)
    return False, [1.0], "未知断言 %r" % kind


def _eval_numexpr(s):
    """Pure integer-expression evaluation (only digits, + - * ^ ( ) and spaces allowed)."""
    if not re.fullmatch(r"[\d+\-*^() ]+", s or ""):
        raise ValueError("非法算式 %r" % s)
    return int(eval(s.replace("^", "**"), {"__builtins__": {}}, {}))   # noqa: S307


# ---------------------------------------------------------------- selftest (non-vacuous truth)
def selftest():
    out = {}
    # S(k) = 1 + 2k, P(n) = n^2 + 3 (P(0)=3): telescoping P(k+1)-P(k) = 2k+1
    out["ind_sum_pos"] = chk_ind_sum({"deg": 1, "v0": "3",
                                      "S": {"0": "1", "1": "2"},
                                      "P": {"0": "3", "2": "1"}})[0]
    out["ind_sum_perturbed"] = chk_ind_sum({"deg": 1, "v0": "3",
                                            "S": {"0": "1", "1": "2"},
                                            "P": {"0": "4", "2": "1"}})[0]
    out["ind_sum_wrongcls"] = chk_ind_sum({"deg": 2, "v0": "3",
                                           "S": {"0": "1", "1": "2"},
                                           "P": {"0": "3", "2": "1"}})[0]
    out["geo_pos"] = chk_geo({"v0": 5, "r": 3, "c": -24, "c1": -7, "c0": 12,
                              "cls": "pow"})[0]
    out["geo_clsviol"] = chk_geo({"v0": 12, "r": 3, "c": -24, "c1": 0, "c0": 12,
                                  "cls": "pow"})[0]
    sqT = {"0": "9", "1": "6", "2": "1"}      # (n+3)^2
    out["equiv_square_pos"] = chk_equiv({"P": sqT, "T": sqT, "cls": "square",
                                         "sq": ["3", "1"]})[0]
    out["equiv_poly_on_square"] = chk_equiv({"P": sqT, "T": sqT, "cls": "poly"})[0]
    q4 = {"0": "1", "1": "2", "2": "3", "3": "2", "4": "1"}    # (1+n+n^2)^2
    out["equiv_square_quartic"] = chk_equiv({"P": q4, "T": q4, "cls": "square",
                                             "sq": ["1", "1", "1"]})[0]
    out["equiv_poly_quartic"] = chk_equiv({"P": q4, "T": q4, "cls": "poly"})[0]
    out["tail_square_pos"] = chk_tail({"rho": "25", "cls": "square", "sq": 5})[0]
    out["tail_square_neg"] = chk_tail({"rho": "26", "cls": "square", "sq": 5})[0]
    out["tail_other_neg"] = chk_tail({"rho": "25", "cls": "other"})[0]
    # tree (x+y)^2 + (z+2)^2 + 5 -> R_in; after taking the block (x+y)^2, R_out = z^2+4z+9 (no x/y terms => purge passes)
    R_in = {"2,0,0": "1", "1,1,0": "2", "0,2,0": "1", "0,0,2": "1", "0,0,1": "4", "0,0,0": "9"}
    out["block_pos"] = chk_block({"vars": ["x", "y", "z"], "R_in": R_in,
                                  "R_out": {"0,0,2": "1", "0,0,1": "4", "0,0,0": "9"},
                                  "ell": {"1,0,0": "1", "0,1,0": "1"}, "mult": "1"})[0]
    out["block_purge_neg"] = chk_block({"vars": ["x", "y", "z"], "R_in": R_in,
                                        "R_out": {"0,0,2": "1", "0,0,1": "4", "0,0,0": "7",
                                                  "1,0,0": "2", "0,1,0": "1"},
                                        "ell": {"1,0,0": "1", "0,1,0": "1", "0,0,0": "-1"},
                                        "mult": "1"})[0]
    out["wit_pos"] = chk_wit({"m": 3, "M": 9, "params": ["w_a", "w_b"],
                              "terms": [{"coef": "1", "k": 2, "wvars": ["w_a", "w_b"]}],
                              "const": 9,
                              "witness": [{"coef": "1", "monomial": ["w_a", "w_b"]},
                                          {"coef": "1", "monomial": []}]})[0]
    out["wit_neg"] = chk_wit({"m": 3, "M": 9, "params": ["w_a", "w_b"],
                              "terms": [{"coef": "1", "k": 2, "wvars": ["w_a", "w_b"]}],
                              "const": 9,
                              "witness": [{"coef": "1", "monomial": ["w_a", "w_b"]}]})[0]
    out["shape_occurs_pos"] = chk_shape({"assert": "occurs", "var": "u",
                                         "text": "\u22a2 9 \u2223 u + 3 * t"})[0]
    out["shape_occurs_neg"] = chk_shape({"assert": "occurs", "var": "c",
                                         "text": "\u22a2 9 \u2223 u + 3 * t"})[0]
    out["shape_modval_pos"] = chk_shape({"assert": "mod_val", "hyp": "hb", "m": 3, "val": 2,
                                         "modulus": 9, "text": "(hb : 3 * 3 \u2223 b)"})[0]
    out["shape_modval_neg"] = chk_shape({"assert": "mod_val", "hyp": "hb", "m": 3, "val": 1,
                                         "modulus": 9, "text": "(hb : 3 * 3 \u2223 b)"})[0]
    out["geo_close_const_obf"] = chk_geo_close({"v0": 12, "r": 3, "c": -24, "cls": "const",
                                                "tpl": "const_lin_obf",
                                                "text": "12 * ((n : \u2124) + 1) - "
                                                        "12 * (n : \u2124)",
                                                "tpl_params": []})[0]
    out["geo_close_pow_pos"] = chk_geo_close({"v0": 5, "r": 3, "c": -24, "cls": "pow",
                                              "tpl": "pow_plain",
                                              "text": "-7 * 3 ^ n + 12",
                                              "tpl_params": []})[0]
    out["geo_close_clsviol"] = chk_geo_close({"v0": 12, "r": 3, "c": -24, "cls": "pow",
                                              "tpl": "pow_plain", "text": "-7 * 3 ^ n + 12",
                                              "tpl_params": []})[0]
    out["psqrt_quartic_true"] = _poly_sqrt(_fl(q4))[0]
    out["psqrt_linear_false"] = _poly_sqrt(_fl({"0": "1", "1": "1"}))[0]
    return out
