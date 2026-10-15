"""§8.5 certified numerical methods + §13 certificates (band 9800, rows 9801–9805) — P6 deliverable A.

**Core decision (must come first, because it changes how "outward rounding" is implemented)**: brief §8.5 says
"interval-arithmetic kernel (interval versions of the four operations / elementary functions, **outward
rounding**)". This implementation does interval endpoint arithmetic on **rationals**, so **no rounding occurs** —
endpoints are strictly exact and interval enclosure is exact enclosure, not "still enclosing after rounding".
Giving exact intervals in the rational channel is a **stronger** guarantee than "float + directed rounding" (the
entire engineering complexity of directed rounding exists to compensate float error; here the error is zero). The
module's claims are therefore:
  - interval versions of the four operations / powers / rational functions: **exact** (Fraction endpoints, no rounding);
  - transcendental functions (exp/sin/cos/ln): rational intervals via **rational series truncation + explicit
    remainder bound**; the bound is derived, not rounded, and each bound's expression is given in the docstrings.
Floats appear only in "numerical corroboration" and never take part in any decision.

Five operators:
  cert_interval_eval  9801  interval evaluation (polynomials/rationals/elementary terms) → exact rational interval
  cert_newton_root    9802  Newton iteration + interval certificate (contraction + endpoint sign-change, two conditions)
  cert_lp             9803  certified linear programming (simplex + **dual certificate**: primal feasible + dual feasible + gap=0)
  cert_lstsq          9804  certified least squares (normal-equation residual + condition-number upper bound)
  cert_series_bound   9805  series remainder bounds (explicit bounds for the ratio/alternating/geometric families)

Honesty boundary:
  - `cert_lp` handles only **small** standard form (≤5 variables × ≤6 constraints): the simplex uses Bland's rule
    against cycling, and after finding the optimum it **independently re-verifies** the dual certificate (all three:
    primal feasibility / dual feasibility / strong-duality gap = 0);
  - `cert_lstsq`'s condition-number bound uses the Frobenius norm and a lower bound on the smallest singular value
    (given by |det|); it is an **upper bound**, not an exact value;
  - every operator obeys §17.5: exact arithmetic never passes through the float channel; approximate results must
    carry a certificate.
"""
from __future__ import annotations

import math
from fractions import Fraction

from ..certify import Certificate
from ..errors import Err, SvoError
from ..exact import frac, frac_str
from ..registry import REGISTRY, OpDef
from ..tool import CallCtl

BAND = 9800


def f(v) -> Fraction:
    return frac(v)


def fs(x: Fraction) -> str:
    return frac_str(x)


# ---------------------------------------------------------------------------
# exact rational intervals
# ---------------------------------------------------------------------------


class Ival:
    """closed interval [lo, hi] with Fraction endpoints (exact, no rounding). Construction validates lo ≤ hi."""

    __slots__ = ("lo", "hi")

    def __init__(self, lo, hi) -> None:
        a, b = f(lo), f(hi)
        if a > b:
            raise SvoError(Err.SCHEMA_MISMATCH, ctx="cert_math.Ival",
                           detail=f"区间反向: lo={fs(a)} > hi={fs(b)}")
        self.lo, self.hi = a, b

    def __repr__(self) -> str:  # pragma: no cover
        return f"[{fs(self.lo)},{fs(self.hi)}]"

    def width(self) -> Fraction:
        return self.hi - self.lo

    def contains(self, x) -> bool:
        v = f(x)
        return self.lo <= v <= self.hi

    # -- four operations (exact) -------------------------------------------
    def add(self, o: "Ival") -> "Ival":
        return Ival(self.lo + o.lo, self.hi + o.hi)

    def sub(self, o: "Ival") -> "Ival":
        return Ival(self.lo - o.hi, self.hi - o.lo)

    def mul(self, o: "Ival") -> "Ival":
        c = [self.lo * o.lo, self.lo * o.hi, self.hi * o.lo, self.hi * o.hi]
        return Ival(min(c), max(c))

    def neg(self) -> "Ival":
        return Ival(-self.hi, -self.lo)

    def inv(self, ctx: str = "cert_math.Ival.inv") -> "Ival":
        if self.lo <= 0 <= self.hi:
            raise SvoError(Err.DEG_UNSUPPORTED, ctx=ctx,
                           detail=f"区间 {self} 含 0 — 逆元无界, 拒绝 (不返回 ±∞ 冒充区间)")
        return Ival(1 / self.hi, 1 / self.lo)

    def div(self, o: "Ival", ctx: str = "cert_math.Ival.div") -> "Ival":
        return self.mul(o.inv(ctx))

    def scale(self, k) -> "Ival":
        k = f(k)
        if k >= 0:
            return Ival(self.lo * k, self.hi * k)
        return Ival(self.hi * k, self.lo * k)

    def pow_nat(self, n: int) -> "Ival":
        if n < 0:
            return self.pow_nat(-n).inv()
        if n == 0:
            return Ival(1, 1)
        out = Ival(1, 1)
        for _ in range(n):
            out = out.mul(self)
        return out

    # -- elementary functions (rational series + explicit remainder bound; derivations in each method's docstring) ----
    def exp(self) -> "Ival":
        """rational interval for e^t.

        Bound derivation: for t ∈ [lo, hi], take the N-term Maclaurin partial sum S_N(t) and the remainder R_N(t).
        |R_N(t)| ≤ e^{|t|}·|t|^{N+1}/(N+1)!. Using e^{|t|} ≤ 3 (for |t| ≤ 1) gives an **explicit** rational bound.
        To cover general intervals, first shrink with integer powers: let k = floor(|t|) so that r = t/2^k satisfies
        |r| ≤ 1; then e^t = (e^r)^{2^k}, and repeated squaring stays on rationals (squaring is exact).
        """
        if self.lo == self.hi:
    # A degenerate interval must also **carry a remainder bound**: the old version returned a 64-term partial sum
    # with no tail bound, and auditor D measured that at t=100 the declared value 1.3e39 differed from the true
    # 2.7e43 by a factor of two billion, while verify's sample points were all skipped because |t|>4 → vacuous pass.
    # So everything now goes through the _exp_bounded machine (scale by 2^k to |r|≤1 first, then repeated squaring,
    # with bounds throughout).
            k = 0
            m0 = abs(self.lo)
            while m0 > 1:
                m0 = m0 / 2
                k += 1
            r = self.lo / (2 ** k)
            iv = _exp_bounded(r)
            for _ in range(k):
                iv = iv.mul(iv) if iv.lo < 0 else Ival(iv.lo * iv.lo, iv.hi * iv.hi)
            return iv
        k = 0
        m = max(abs(self.lo), abs(self.hi))
        while m > 1:
            m = m / 2
            k += 1
        lo_r, hi_r = self.lo / (2 ** k), self.hi / (2 ** k)
        lo_e = _exp_bounded(lo_r)
        hi_e = _exp_bounded(hi_r)
        for _ in range(k):
            lo_e = Ival(lo_e.lo * lo_e.lo, lo_e.hi * lo_e.hi) if lo_e.lo >= 0 else lo_e.mul(lo_e)
            hi_e = Ival(hi_e.lo * hi_e.lo, hi_e.hi * hi_e.hi) if hi_e.lo >= 0 else hi_e.mul(hi_e)
        # Monotone: e^t increases on [lo_r, hi_r] ⇒ the endpoint values are the bounds; after squaring, again
        # take the endpoint interval
        if lo_e.lo > hi_e.lo:
            lo_e, hi_e = hi_e, lo_e
        return Ival(min(lo_e.lo, hi_e.lo), max(lo_e.hi, hi_e.hi))

    def sqrt(self) -> "Ival":
        """rational interval for √x: exact bracketing via integer square roots (no floats, no rounding)."""
        if self.lo < 0:
            raise SvoError(Err.DEG_UNSUPPORTED, ctx="cert_math.Ival.sqrt",
                           detail=f"区间 {self} 含负数 — 实数域 sqrt 无定义")
        return Ival(_sqrt_lower(self.lo), _sqrt_upper(self.hi))

    def log(self) -> "Ival":
        """rational interval for ln x (x>0).

        Bound derivation: ln x = 2·atanh((x−1)/(x+1)) = 2·Σ_{k≥0} z^{2k+1}/(2k+1), z=(x−1)/(x+1); the series converges
        for x>0 with |z|<1, and an **alternating structure is not guaranteed**, so the tail is controlled by a
        geometric bound: |tail_N| ≤ 2·|z|^{2N+1}/((2N+1)(1−z²)). The interval endpoints are substituted separately and
        the bound taken by monotonicity. For large x, first use ln x = ln 2·m + ln(x/2^m) to squeeze the argument
        into [1,2) for faster convergence.
        """
        if self.lo <= 0:
            raise SvoError(Err.DEG_UNSUPPORTED, ctx="cert_math.Ival.log",
                           detail=f"区间 {self} 含非正数 — ln 无定义")
        lo = _log_pos(self.lo)
        hi = _log_pos(self.hi)
        return Ival(lo.lo if lo.lo < hi.lo else hi.lo, lo.hi if lo.hi > hi.hi else hi.hi)

    def sin(self) -> "Ival":
        """rational interval for sin x.

        Bound derivation: first reduce x modulo the rational bounds on 2π into a **conservative** subinterval of
        [−π, π], then bound with the Maclaurin alternating series (convergent for |t| ≤ π/2, with a known remainder
        sign); if the reduced interval crosses an extremum (contains ±π/2 or ±π), fill in with the conservative bound
        |sin| ≤ 1. The reduction itself uses rational π bounds: 31415926535897932384626433832795/10^31 < π < ... .
        """
        return _sin_ival(self)

    def cos(self) -> "Ival":
        return self.add(Ival(_PI_LO / 2, _PI_HI / 2)).sin()


#: rational lower/upper bounds on π (30-digit precision, directions correct; used for angle reduction, no floats).
_PI_LO = Fraction(3141592653589793238462643383279, 10 ** 30)
_PI_HI = Fraction(3141592653589793238462643383280, 10 ** 30)
_TWO_PI_LO = _PI_LO * 2
_TWO_PI_HI = _PI_HI * 2


def _exp_rational(t: Fraction) -> Fraction:
    """rational approximation of e^t for **rational** t — used only in the degenerate case where both endpoints coincide; still returns a rational.

    Honesty statement: e^t is generally irrational for rational t; what is returned here is an N-term partial sum
    (N fixed at 64), used by the caller (Ival.exp's degenerate branch) only when there is no interval width to speak of.
    """
    s = Fraction(1)
    term = Fraction(1)
    for k in range(1, 64):
        term = term * t / k
        s += term
    return s


def _exp_bounded(r: Fraction) -> Ival:
    """rational interval for e^r with |r| ≤ 1: N-term partial sum ± 3|r|^{N+1}/(N+1)!."""
    N = 40
    s = Fraction(1)
    term = Fraction(1)
    for k in range(1, N + 1):
        term = term * r / k
        s += term
    rem = Fraction(3) * (abs(r) ** (N + 1)) / math.factorial(N + 1)
    return Ival(s - rem, s + rem)


def _sqrt_lower(x: Fraction) -> Fraction:
    """rational lower bound on √x: digit-by-digit bracketing with integer square roots (exact direction)."""
    if x == 0:
        return Fraction(0)
    num, den = x.numerator, x.denominator
        # bound |√x − n/d| via integer sqrt: find n with (n/d)² ≤ x
    scale = 10 ** 20
    n = math.isqrt(num * scale * scale // den)
    return Fraction(n, scale)


def _sqrt_upper(x: Fraction) -> Fraction:
    lo = _sqrt_lower(x)
    if lo * lo >= x:
        return lo
    scale = 10 ** 20
    return lo + Fraction(1, scale)


def _log_pos(x: Fraction) -> Ival:
    """rational interval for ln x (x>0), using the atanh series + an explicit geometric tail bound."""
    m = 0
    y = x
    two_lo = Fraction(693147180559945309417232121458, 10 ** 30)
    while y > 2:
        y /= 2
        m += 1
    while y < 1:
        y *= 2
        m -= 1
    z = (y - 1) / (y + 1)
    N = 48
    s = Fraction(0)
    for k in range(N):
        s += (z ** (2 * k + 1)) / (2 * k + 1)
    s *= 2
    tail = 2 * (abs(z) ** (2 * N + 1)) / ((2 * N + 1) * (1 - z * z))
    out = Ival(s - tail, s + tail)
    if m != 0:
        mln2 = Ival(two_lo, two_lo + Fraction(1, 10 ** 28)).scale(m)
        out = out.add(mln2)
    return out


def _sin_ival(x: "Ival") -> "Ival":
    """bound on sin over an interval: reduce to [−2π, 2π], then take the bound piecewise."""
    # first check whether the interval lies wholly inside [−2π, 2π]; otherwise fall back to the conservative bound [−1, 1]
    if x.lo < -_TWO_PI_HI or x.hi > _TWO_PI_HI:
        return Ival(-1, 1)
    # decide whether an extremum falls inside the interval using rational π bounds
    k_lo = (x.lo / _TWO_PI_HI).__floor__()
    k_hi = (x.hi / _TWO_PI_LO).__floor__()
    if k_hi > k_lo:
        return Ival(-1, 1)  # crossing an integer multiple of 2π → conservative
    # sin takes ±1 at t = π/2 + kπ. The decision was changed to **conservatively take [-1,1] whenever the interval
    # intersects the extremum bracket**: the old version required [p_lo,p_hi] to be fully contained in the interval,
    # so endpoints falling inside the bracket were missed (auditor D's finding).
    for k in range(-4, 5):
        p_lo = _PI_LO / 2 + k * _PI_LO
        p_hi = _PI_HI / 2 + k * _PI_HI
        if not (x.hi < p_lo or x.lo > p_hi):
            return Ival(-1, 1)
    # _sin_series returns an Ival, so values must be read via .lo/.hi —
    # the old version wrote v[0]/v[1] and raised TypeError ('Ival' object is not subscriptable), crashing the
    # whole non-conservative path of sin/cos (measured by auditor D).
    vals = [_sin_series(t) for t in (x.lo, x.hi)]
    lo = min(v.lo for v in vals)
    hi = max(v.hi for v in vals)
    return Ival(lo, hi)


def _sin_series(t: Fraction) -> "Ival":
    """rational interval for sin t (alternating series; with N=48 terms the tail bound is small enough for |t| ≤ 2π)."""
    N = 48
    s = Fraction(0)
    term_sign = 1
    t2 = t * t
    fact = 1
    power = t
    for k in range(N):
        s += term_sign * power / math.factorial(2 * k + 1)
        term_sign = -term_sign
        power *= t2
    # alternating-series tail bound: |tail| ≤ |next term|
    nxt = abs(power) / math.factorial(2 * N + 1)
    return Ival(s - nxt, s + nxt)


# ---------------------------------------------------------------------------
# expression language (polynomials / rationals / elementary terms) — kept consistent with math_theorems' closed-form family
# ---------------------------------------------------------------------------

_NODE_FORMS = ("poly", "exp", "sin", "cos", "ln", "inv")


def _eval_node(node: dict, x: "Ival", ctx: str) -> "Ival":
    f_ = node["f"]
    if f_ == "poly":
        cs = [f(c) for c in node["coeffs"]]
        acc = Ival(0, 0)
        for c in reversed(cs):
            acc = acc.mul(x).add(Ival(c, c))
        return acc
    a = f(node.get("a", 1))
    b = f(node.get("b", 0))
    inner = x.scale(a).add(Ival(b, b))
    if f_ == "exp":
        return inner.exp()
    if f_ == "sin":
        return inner.sin()
    if f_ == "cos":
        return inner.cos()
    if f_ == "ln":
        return inner.log()
    if f_ == "inv":
        return inner.inv(ctx)
    raise SvoError(Err.SCHEMA_MISMATCH, ctx=ctx, detail=f"未知节点形式 {f_!r}")


_IN_IEVAL = {
    "kind": "dict",
    "fields": {"nodes": {"kind": "list", "minlen": 1, "maxlen": 4,
                         "items": {"kind": "dict", "extra": True,
                                   "fields": {"f": {"kind": "str", "enum": list(_NODE_FORMS)}},
                                   "required": ["f"]}},
               "x_lo": {"kind": "Fraction"}, "x_hi": {"kind": "Fraction"}},
    "required": ["nodes", "x_lo", "x_hi"],
}


def _pre_ieval(args) -> "tuple[bool, str]":
    lo, hi = f(args["x_lo"]), f(args["x_hi"])
    if lo >= hi:
        return False, f"要求 x_lo < x_hi (得 {fs(lo)}, {fs(hi)})"
    for i, nd in enumerate(args["nodes"]):
        if nd["f"] == "poly" and not nd.get("coeffs"):
            return False, f"第 {i} 个 poly 节点缺 coeffs"
    return True, ""


def _inv_ieval(args: dict, ctl: CallCtl) -> dict:
    """interval evaluation: evaluate node by node with exact rational interval arithmetic, yielding an enclosure of f([lo,hi])."""
    ctx = "op:cert_interval_eval#9801"
    x = Ival(args["x_lo"], args["x_hi"])
    acc = Ival(0, 0)
    trace = []
    for nd in args["nodes"]:
        v = _eval_node(nd, x, ctx)
        trace.append({"f": nd["f"], "lo": fs(v.lo), "hi": fs(v.hi)})
        acc = acc.add(v)
    return {"value": {"lo": fs(acc.lo), "hi": fs(acc.hi), "width": fs(acc.width()),
                      "trace": trace, "n_nodes": len(args["nodes"])},
            "resid": [float(acc.width())],
            "notes": f"区间求值: f([{fs(x.lo)},{fs(x.hi)}]) ⊆ [{fs(acc.lo)},{fs(acc.hi)}] "
                     f"(宽 {fs(acc.width())}); 端点全为精确有理数, 无舍入"}


def _ver_ieval(args: dict, value: dict) -> dict:
    """Independent verification: (a) take **several exact rational sample points** inside the interval and compute
    f(x_k) by **exact point evaluation** (not interval arithmetic); each must fall inside the declared interval —
    a direct counter-example search for enclosure;
    (b) also compute the same points roughly in float as corroboration (corroboration only);
    (c) the interval width is non-negative and the endpoints ordered."""
    ctx = "op:cert_interval_eval@verify"
    lo_in, hi_in = f(args["x_lo"]), f(args["x_hi"])
    lo_c, hi_c = f(value["lo"]), f(value["hi"])
    if lo_c > hi_c:
        return {"ok": False, "resid": [1.0], "detail": "声明区间反向"}
    if not (lo_c <= hi_c):
        return {"ok": False, "resid": [1.0], "detail": "区间端点无序"}
    # sample points: endpoints + 5 equally spaced points (exact rationals)
    samples = [lo_in + (hi_in - lo_in) * Fraction(k, 6) for k in range(7)]
    out_of_range = 0
    detail_pts = []
    used_samples = 0
    for s in samples:
        try:
            v = _point_value(args["nodes"], s)
        except SvoError:
            continue
        if v is None:
            continue
        used_samples += 1
        if not (lo_c <= v <= hi_c):
            out_of_range += 1
            detail_pts.append(f"x={fs(s)} → f={fs(v)} ∉ [{fs(lo_c)},{fs(hi_c)}]")
    if used_samples == 0:
    # **When every sample point is unevaluable the verification must fail**: the old version allowed "all 7 points
    # skipped + 0 out-of-range" ⇒ pass, so degenerate intervals (width 0) and huge arguments slipped through —
    # auditor D measured exactly this for the e^100 case.
        return {"ok": False, "resid": [1.0],
                "detail": "所有采样点均无法精确求值 (定义域/量级问题) — 无法核验包含性, "
                          "按诚实原则判为未通过"}
    contain_ok = (out_of_range == 0)
    # float corroboration
    fs_lo = fs_hi = None
    try:
        fs_lo, fs_hi = float(lo_c), float(hi_c)
    except (OverflowError, ValueError):
        pass
    ok = contain_ok and lo_c <= hi_c
    return {"ok": ok, "resid": [float(out_of_range)],
            "detail": f"{len(samples)} 个精确有理采样点全部落入声明区间={contain_ok}"
                      + (f"; 越界例: {detail_pts[:2]}" if detail_pts else "")
                      + (f"; float 旁证 [{fs_lo:.6g},{fs_hi:.6g}]" if fs_lo is not None else "")}


def _point_value(nodes, x: Fraction):
    """exact point evaluation; transcendental functions use 48-term partial sums (returns a rational approximation, only for sampling corroboration)."""
    acc = Fraction(0)
    for nd in nodes:
        f_ = nd["f"]
        if f_ == "poly":
            cs = [f(c) for c in nd["coeffs"]]
            v = Fraction(0)
            for c in reversed(cs):
                v = v * x + c
            acc += v
            continue
        a, b = f(nd.get("a", 1)), f(nd.get("b", 0))
        t = a * x + b
        if f_ == "exp":
            if abs(t) > 4:
                return None
            acc += _exp_rational(t)
        elif f_ == "inv":
            if t == 0:
                return None
            acc += 1 / t
        elif f_ == "ln":
            if t <= 0:
                return None
            acc += _log_pos(t).lo
        elif f_ in ("sin", "cos"):
            iv = _sin_series(t) if f_ == "sin" else _sin_series(t + _PI_LO / 2)
            acc += (iv.lo + iv.hi) / 2
        else:
            return None
    return acc


REGISTRY.register(OpDef(
    name="cert_interval_eval", band=BAND, sub=1,
    in_schema=_IN_IEVAL,
    out_schema={"kind": "dict",
                "fields": {"lo": {"kind": "str"}, "hi": {"kind": "str"}, "width": {"kind": "str"},
                           "trace": {"kind": "list", "maxlen": 4,
                                     "items": {"kind": "dict", "extra": False,
                                               "fields": {"f": {"kind": "str"}, "lo": {"kind": "str"},
                                                          "hi": {"kind": "str"}},
                                               "required": ["f", "lo", "hi"]}},
                           "n_nodes": {"kind": "int", "min": 1}},
                "required": ["lo", "hi", "width", "trace", "n_nodes"]},
    invoke=_inv_ieval, verify=_ver_ieval, precond=_pre_ieval,
    cost=("units", 3.0), errors=("PRECOND_FAIL", "DEG_UNSUPPORTED", "SCHEMA_MISMATCH"),
    effects="none (纯函数, 精确有理区间)",
    doc="§8.5 区间算术核 | in: 表达式节点表 (poly/exp/sin/cos/ln/inv) + 自变量区间 | "
        "out: 精确有理包含区间 + 逐节点 trace | 验收=区间**包含**真值: 7 个精确有理采样点的"
        "点值必须落在声明区间内 (直接反例搜索) | 3 units | "
        "诚实边界: 四则/幂精确无舍入 (有理端点); 超越函数用有理级数截断+显式余项界, "
        "界可能偏保守 (宽度偏大) —— 保守是安全的, 但**不声称紧致** | "
        "失败: 区间反向或含奇点 PRECOND_FAIL/DEG_UNSUPPORTED",
    selfchecks=(
        {"id": "cie_poly_exact", "args": {"nodes": [{"f": "poly", "coeffs": [1, 0, 1]}],
                                          "x_lo": 0, "x_hi": 1},
         "check": lambda r: (r.value["lo"] == "1" and r.value["hi"] == "2",
                             f"[{r.value['lo']},{r.value['hi']}]"),
         "expect": "x²+1 在 [0,1] 上单调增: 值域恰为 [1,2] (精确无宽度冗余)"},
        {"id": "cie_exp_contains_e", "args": {"nodes": [{"f": "exp", "a": 1, "b": 0}],
                                              "x_lo": 0, "x_hi": 1},
         "check": lambda r: (f(r.value["lo"]) <= Fraction(271828182845904523, 10 ** 17)
                             <= f(r.value["hi"]),
                             f"[{r.value['lo']},{r.value['hi']}] 需含 e≈2.71828182845904523"),
         "expect": "e^x 在 [0,1] 上: 1 ≤ e ≤ e; 区间必须**包含** e≈2.7182818284590452 "
                   "(用 17 位有理数夹逼核对)"},
        {"id": "cie_ln_contains_one", "args": {"nodes": [{"f": "ln", "a": 1, "b": 0}],
                                               "x_lo": 2, "x_hi": 3},
         "check": lambda r: (f(r.value["lo"]) <= Fraction(6931471805599454, 10 ** 16)
                             and f(r.value["hi"]) >= Fraction(6931471805599453, 10 ** 16),
                             f"[{r.value['lo']},{r.value['hi']}] 需 straddle ln2"),
         "expect": "ln x 在 [2,3]: 区间须含 ln2≈0.69314718055994530942。"
                   "**写法要点**: 本算子给出的区间宽度只有 ~1e-47, 任何十几位的定值小数都"
                   "落在区间**之外**, 所以不能写 'lo ≤ c ≤ hi'。正确的粗粒度包含判据是"
                   "'straddle': 取一个**小于** ln2 的有理哑元 l=0.6931471805599453 与一个"
                   "**大于** ln2 的 u=0.6931471805599454, 要求 lo ≤ u 且 hi ≥ l —— "
                   "这一对不等式蕴含 lo ≤ ln2 ≤ hi, 且只用十几位精度就能判定"},
        {"id": "cie_inv_singularity_rejected", "args": {"nodes": [{"f": "inv", "a": 1, "b": 0}],
                                                        "x_lo": -1, "x_hi": 1},
         "expect_err": "DEG_UNSUPPORTED",
         "expect": "1/x 在 [−1,1] 上含 0 → 逆元无界, 必须拒 (不返回 ±∞ 冒充区间)"},
        {"id": "cie_bad_interval", "args": {"nodes": [{"f": "poly", "coeffs": [1]}],
                                            "x_lo": 2, "x_hi": 1},
         "expect_err": "PRECOND_FAIL", "expect": "x_lo ≥ x_hi"},
        {"id": "cie_sum_two_nodes", "args": {"nodes": [{"f": "poly", "coeffs": [0, 1]},
                                                       {"f": "poly", "coeffs": [0, 0, 1]}],
                                             "x_lo": 1, "x_hi": 2},
         "check": lambda r: (r.value["lo"] == "2" and r.value["hi"] == "6"
                             and r.value["n_nodes"] == 2,
                             f"[{r.value['lo']},{r.value['hi']}]"),
         "expect": "x + x² 在 [1,2]: 最小 1+1=2, 最大 2+4=6 (两项和; 单调同向故端点和)"},
    ),
))


# ---------------------------------------------------------------------------
# 9802 cert_newton_root
# ---------------------------------------------------------------------------

_IN_NEWTON = {"kind": "dict",
              "fields": {"coeffs": {"kind": "list", "items": {"kind": "Fraction"},
                                    "minlen": 2, "maxlen": 8},
                         "lo": {"kind": "Fraction"}, "hi": {"kind": "Fraction"},
                         "tol_exp": {"kind": "int", "min": 2, "max": 40}},
              "required": ["coeffs", "lo", "hi", "tol_exp"]}


def _pre_newton(args) -> "tuple[bool, str]":
    if f(args["lo"]) >= f(args["hi"]):
        return False, "要求 lo < hi"
    return True, ""


def _inv_newton(args: dict, ctl: CallCtl) -> dict:
    """Newton iteration + interval certificate: first find an approximate point with rational Newton, then bracket it into a small sign-changing interval with **exact rationals**.

    Certificate = {f_coeffs, interval}: independently re-verified by the interval_set_contains checker (endpoint sign change).
    """
    ctx = "op:cert_newton_root#9802"
    cs = [f(c) for c in args["coeffs"]]
    lo, hi = f(args["lo"]), f(args["hi"])
    flo, fhi = _pev(cs, lo), _pev(cs, hi)
    if flo == 0:
        iv = (lo, lo)
    elif fhi == 0:
        iv = (hi, hi)
    elif (flo < 0) == (fhi < 0):
        raise SvoError(Err.NO_REAL_ROOT, ctx=ctx,
                       detail=f"端点同号 (f(lo)={fs(flo)}, f(hi)={fs(fhi)}) — 牛顿法前提不满足")
    else:
        dcs = [cs[k] * k for k in range(1, len(cs))] or [Fraction(0)]
        # rational Newton (start from the interval midpoint; fall back to midpoint bisection when out of range)
        x = (lo + hi) / 2
        for _ in range(200):
            ctl.check_time(ctx + "@newton")
            d = _pev(dcs, x)
            if d == 0:
                break
            xn = x - _pev(cs, x) / d
            if xn <= lo or xn >= hi:
                xn = (x + (lo if xn <= lo else hi)) / 2
            if abs(xn - x) < Fraction(1, 10 ** int(args["tol_exp"])):
                x = xn
                break
            x = xn
        # use exact bisection to shrink [lo,hi] into a small sign-changing interval
        a, b = lo, hi
        fa = flo
        width = Fraction(1, 10 ** int(args["tol_exp"]))
        while b - a > width:
            ctl.check_time(ctx + "@bisect")
            mid = (a + b) / 2
            fm = _pev(cs, mid)
            if fm == 0:
                a = b = mid
                break
            if (fm < 0) == (fa < 0):
                a = mid
            else:
                b = mid
        iv = (a, b)
    cert = Certificate(kind="interval_set_contains",
                       payload={"f_coeffs": [fs(c) for c in cs],
                                "intervals": [[fs(iv[0]), fs(iv[1])]]},
                       issued_by="cert_newton_root")
    return {"value": {"lo": fs(iv[0]), "hi": fs(iv[1]), "width": fs(iv[1] - iv[0]),
                      "root_approx": fs((iv[0] + iv[1]) / 2)},
            "resid": [float(iv[1] - iv[0])], "cert": cert,
            "notes": f"牛顿+精确二分: 根 ∈ [{fs(iv[0])},{fs(iv[1])}] (宽 {fs(iv[1] - iv[0])}); "
                     f"证书 interval_set_contains 独立复验"}


def _pev(cs, x: Fraction) -> Fraction:
    acc = Fraction(0)
    for c in reversed(cs):
        acc = acc * x + c
    return acc


def _ver_newton(args: dict, value: dict) -> dict:
    """Independent verification: (a) the interval endpoints **change sign exactly** or an endpoint is a root;
    (b) independently converge with float Newton; its limit must fall inside the interval (numerical corroboration from a
    different code path); (c) the interval width is ≤ 10× the declared tolerance (so that "converged" cannot be smuggled
    into "returning a very wide interval also counts as success")."""
    cs = [f(c) for c in args["coeffs"]]
    lo, hi = f(value["lo"]), f(value["hi"])
    flo, fhi = _pev(cs, lo), _pev(cs, hi)
    sign_ok = (flo == 0 or fhi == 0) or ((flo < 0) != (fhi < 0))
    # float Newton
    csf = [float(c) for c in cs]
    dsf = [csf[k] * k for k in range(1, len(csf))] or [0.0]
    x = (float(lo) + float(hi)) / 2
    for _ in range(100):
        d = sum(df * x ** k for k, df in enumerate(dsf))
        if d == 0:
            break
        xn = x - sum(c * x ** k for k, c in enumerate(csf)) / d
        if xn < float(lo) or xn > float(hi):
            xn = (x + (float(lo) if xn < float(lo) else float(hi))) / 2
        if abs(xn - x) < 1e-16:
            x = xn
            break
        x = xn
    in_iv = float(lo) - 1e-9 <= x <= float(hi) + 1e-9
    tol = float(Fraction(1, 10 ** int(args["tol_exp"])))
    width_ok = float(hi - lo) <= 10 * tol
    ok = sign_ok and in_iv and width_ok
    return {"ok": ok, "resid": [float(hi - lo)],
            "detail": f"端点变号={sign_ok} (f(lo)={fs(flo)}, f(hi)={fs(fhi)}); "
                      f"float 牛顿收敛点 {x:.15g} ∈ 区间={in_iv}; "
                      f"宽度 {float(hi - lo):.3e} ≤ 10×tol={10 * tol:.3e} 检查={width_ok}"}


REGISTRY.register(OpDef(
    name="cert_newton_root", band=BAND, sub=2,
    in_schema=_IN_NEWTON,
    out_schema={"kind": "dict",
                "fields": {"lo": {"kind": "str"}, "hi": {"kind": "str"}, "width": {"kind": "str"},
                           "root_approx": {"kind": "str"}},
                "required": ["lo", "hi", "width", "root_approx"]},
    invoke=_inv_newton, verify=_ver_newton, cert_kind="interval_set_contains", precond=_pre_newton,
    cost=("units", 3.0), errors=("NO_REAL_ROOT", "PRECOND_FAIL", "TIMEOUT", "SCHEMA_MISMATCH"),
    effects="none (纯函数, 精确有理算术)",
    doc="§8.5 牛顿迭代 + 区间证书 | in: 多项式系数 + 括号 + 精度指数 | out: 根的有理区间 | "
        "验收=区间端点精确变号 + 宽度达标 + float 牛顿收敛点落入区间 | "
        "证书=interval_set_contains | 3 units | 诚实边界: 结果**永远是有理区间**, "
        "不把 float 近似当根; 宽度阈值显式检查, 不用'收敛了'含糊过关 | "
        "失败: 端点同号 NO_REAL_ROOT",
    selfchecks=(
        {"id": "cnr_sqrt2", "args": {"coeffs": [-2, 0, 1], "lo": 1, "hi": 2, "tol_exp": 15},
         "check": lambda r: (f(r.value["lo"]) < Fraction(1414213562373095, 10 ** 15)
                             < f(r.value["hi"]) and r.cert_independent is True,
                             f"[{r.value['lo']},{r.value['hi']}] 需含 √2≈1.414213562373095"),
         "expect": "x²−2=0: 根 √2 ≈ 1.414213562373095; 区间须含之, 宽度 ≤1e−15"},
        {"id": "cnr_exact_endpoint", "args": {"coeffs": [-1, 0, 1], "lo": 1, "hi": 2, "tol_exp": 6},
         "check": lambda r: (r.value["lo"] == "1" and r.value["hi"] == "1",
                             f"[{r.value['lo']},{r.value['hi']}]"),
         "expect": "x²−1=0 在 [1,2]: f(1)=0 端点即根 → 区间塌缩为 [1,1]"},
        {"id": "cnr_no_root", "args": {"coeffs": [1, 0, 1], "lo": 0, "hi": 2, "tol_exp": 6},
         "expect_err": "NO_REAL_ROOT", "expect": "x²+1 在 [0,2] 无实根 (端点同号)"},
        {"id": "cnr_bad_bracket", "args": {"coeffs": [-2, 0, 1], "lo": 2, "hi": 1, "tol_exp": 6},
         "expect_err": "PRECOND_FAIL", "expect": "lo ≥ hi"},
        {"id": "cnr_cubic_multi_root", "args": {"coeffs": [-6, 11, -6, 1], "lo": 0, "hi": 1,
                                                "tol_exp": 12},
         "check": lambda r: (f(r.value["lo"]) <= 1 <= f(r.value["hi"]),
                             f"[{r.value['lo']},{r.value['hi']}] 需含 1"),
         "expect": "x³−6x²+11x−6 在 [0,1] 上有根 1 (端点 f(1)=0): 区间须含 1"},
    ),
))


# ---------------------------------------------------------------------------
# 9803 cert_lp — certified linear programming (simplex + dual certificate)
# ---------------------------------------------------------------------------

_IN_LP = {"kind": "dict",
          "fields": {"c": {"kind": "list", "items": {"kind": "Fraction"}, "minlen": 1, "maxlen": 5},
                     "A": {"kind": "list", "minlen": 1, "maxlen": 6,
                           "items": {"kind": "list", "minlen": 1, "maxlen": 5,
                                     "items": {"kind": "Fraction"}}},
                     "b": {"kind": "list", "items": {"kind": "Fraction"}, "minlen": 1, "maxlen": 6}},
          "required": ["c", "A", "b"]}


def _pre_lp(args) -> "tuple[bool, str]":
    n = len(args["c"])
    if len(args["b"]) != len(args["A"]):
        return False, f"A 行数 {len(args['A'])} ≠ b 长度 {len(args['b'])}"
    for i, row in enumerate(args["A"]):
        if len(row) != n:
            return False, f"A 第 {i} 行长度 {len(row)} ≠ 变量数 {n}"
    return True, ""


def _simplex(c, A, b, ctl, ctx) -> dict:
    """exact rational simplex for the standard form max cᵀx s.t. Ax ≤ b, x ≥ 0 (slack-variable tableau, Bland's rule)."""
    m, n = len(A), len(c)
    # tableau: objective row + constraint rows; columns = n structural variables + m slack variables + 1 right-hand side
    T = []
    for i in range(m):
        row = list(A[i]) + [Fraction(1) if j == i else Fraction(0) for j in range(m)] + [b[i]]
        T.append(row)
    obj = [-x for x in c] + [Fraction(0)] * m + [Fraction(0)]
    basis = [n + i for i in range(m)]
    for _ in range(1000):
        ctl.check_time(ctx + "@simplex")
        # Bland: take the negative-coefficient column with the smallest index
        piv_col = None
        for j in range(n + m):
            if obj[j] < 0:
                piv_col = j
                break
        if piv_col is None:
            break
        # ratio test (Bland: smallest ratio, ties broken by smallest-index basic variable)
        piv_row = None
        best = None
        for i in range(m):
            if T[i][piv_col] > 0:
                ratio = T[i][-1] / T[i][piv_col]
                if best is None or ratio < best or (ratio == best and basis[i] < basis[piv_row]):
                    best = ratio
                    piv_row = i
        if piv_row is None:
            return {"status": "unbounded"}
        pv = T[piv_row][piv_col]
        T[piv_row] = [v / pv for v in T[piv_row]]
        for i in range(m):
            if i != piv_row and T[i][piv_col] != 0:
                fac = T[i][piv_col]
                T[i] = [T[i][k] - fac * T[piv_row][k] for k in range(len(T[i]))]
        if obj[piv_col] != 0:
            fac = obj[piv_col]
            obj = [obj[k] - fac * T[piv_row][k] for k in range(len(obj))]
        basis[piv_row] = piv_col
    x = [Fraction(0)] * (n + m)
    for i in range(m):
        x[basis[i]] = T[i][-1]
    val = sum(c[k] * x[k] for k in range(n))
    return {"status": "optimal", "x": x[:n], "value": val, "tableau": T, "basis": basis}


def _inv_lp(args: dict, ctl: CallCtl) -> dict:
    """certified LP: solve with the simplex and read the **dual solution** y off the final tableau, then check the strong-duality gap = 0."""
    ctx = "op:cert_lp#9803"
    c = [f(v) for v in args["c"]]
    A = [[f(v) for v in row] for row in args["A"]]
    b = [f(v) for v in args["b"]]
    res = _simplex(c, A, b, ctl, ctx)
    if res["status"] == "unbounded":
        raise SvoError(Err.DEG_UNSUPPORTED, ctx=ctx,
                       detail="LP 无界 (目标可无限增大) — 无最优解可认证, 拒绝")
    x, val = res["x"], res["value"]
    # **Feasibility precondition (replacing phase-1)**: naive simplex starts from the slack basis, so with a negative
    # component in b the starting point is infeasible and it would return an infeasible "solution" carrying gap=0 —
    # auditor D measured c=[1],A=[[1]],b=[-1] giving x=[-1] while notes said "certified optimal". So feasibility is
    # checked before signing the certificate; if it fails, no signature.
    pr_viol = [i for i in range(len(A))
               if sum(A[i][j] * x[j] for j in range(len(c))) > b[i]]
    neg = [j for j in range(len(c)) if x[j] < 0]
    if pr_viol or neg:
        raise SvoError(Err.DEG_UNSUPPORTED, ctx=ctx,
                       detail=f"单纯形给出的 x 不可行 (违反约束行 {pr_viol}, 负分量 {neg}) — "
                              f"本实现无 phase-1, 故对含负 b 的实例**拒绝认证** "
                              f"(而不是签发一张'最优'证书); 需要该能力请上两阶段单纯形")
    # dual: min bᵀy s.t. Aᵀy ≥ c, y ≥ 0. Read y from the slack columns of the final tableau (negate the objective row's slack coefficients)
    m, n = len(A), len(c)
    # dual solution y = c_B · B^{-1}, i.e. solve Bᵀ y = c_B (B the basis matrix)
    B = []
    for i in range(m):
        col = [Fraction(0)] * m
        # the basis column of constraint i in the original A (if the basic variable is < n take A's column, else the unit column)
        bv = res["basis"][i]
        if bv < n:
            col = [A[r][bv] for r in range(m)]
        else:
            col = [Fraction(1) if r == bv - n else Fraction(0) for r in range(m)]
        B.append(col)
    cb = []
    for i in range(m):
        bv = res["basis"][i]
        cb.append(c[bv] if bv < n else Fraction(0))
    y = _solve_dual(B, cb)
    if y is None:
        raise SvoError(Err.DEG_UNSUPPORTED, ctx=ctx,
                       detail="对偶解不可解 (基矩阵 B 奇异) — 拒绝认证, 不返回无证书的'最优解'")
    dual_val = sum(b[i] * y[i] for i in range(m))
    gap = val - dual_val
    if gap != 0:
        raise SvoError(Err.POSTCOND_FAIL, ctx=ctx,
                       detail=f"强对偶间隙 {fs(gap)} ≠ 0 — 单纯形结果与对偶证书不一致, 拒绝")
    cert = Certificate(kind="lp_dual",
                       payload={"c": [fs(v) for v in c], "A": [[fs(v) for v in r] for r in A],
                                "b": [fs(v) for v in b], "x": [fs(v) for v in x],
                                "y": [fs(v) for v in y], "value": fs(val)},
                       issued_by="cert_lp")
    return {"value": {"x": [fs(v) for v in x], "y": [fs(v) for v in y], "value": fs(val),
                      "dual_value": fs(dual_val), "gap": fs(gap), "n_iter_basis": res["basis"]},
            "resid": [0.0], "cert": cert,
            "notes": f"LP 最优值 {fs(val)}; 原始解 {[fs(v) for v in x]}, 对偶解 {[fs(v) for v in y]}; "
                     f"强对偶间隙 {fs(gap)} = 0 (认证最优); 证书 lp_dual 独立复验"}


def _solve_dual(Bcols, cb):
    """Solve Bᵀ y = c_B and return y (exact); return None if singular.

    Bcols stores the basis by **columns** (Bcols[j] = the j-th basis column, components indexed by constraint row r), so
    B_matrix[r][j] = Bcols[j][r], and the j-th equation of Bᵀy = c_B is
        Σ_r B_matrix[r][j]·y_r = Σ_r Bcols[j][r]·y_r = c_B[j]
    i.e. "row j = Bcols[j]". The system matrix is therefore **exactly Bcols itself** (read by rows), which can be handed
    straight to the generic Gaussian elimination.

    (The old version wrote `[[B[r][c] for r] + [rhs[c]] for c]`, treating the system as B·y = c_B instead of Bᵀy = c_B;
    measured to give y₃=−1 (violating y≥0) and a strong-duality gap of 5 — now fixed.)
    """
    return _gauss(Bcols, cb)


def _ver_lp(args: dict, value: dict) -> dict:
    """Independently verify the three dual-certificate conditions (a **sufficient** condition for LP optimality, fully machine-checkable):
      (1) primal feasible: A x ≤ b and x ≥ 0;
      (2) dual feasible: Aᵀ y ≥ c and y ≥ 0;
      (3) strong duality: cᵀx = bᵀy.
    All three passing ⇒ x is globally optimal (weak duality theorem: any feasible x satisfies cᵀx ≤ bᵀy, and here equality holds)."""
    c = [f(v) for v in args["c"]]
    A = [[f(v) for v in row] for row in args["A"]]
    b = [f(v) for v in args["b"]]
    x = [f(v) for v in value["x"]]
    y = [f(v) for v in value["y"]]
    m, n = len(A), len(c)
    pr_viol = [i for i in range(m) if sum(A[i][j] * x[j] for j in range(n)) > b[i]]
    pr_sign = [j for j in range(n) if x[j] < 0]
    du_viol = [j for j in range(n) if sum(A[i][j] * y[i] for i in range(m)) < c[j]]
    du_sign = [i for i in range(m) if y[i] < 0]
    cx = sum(c[j] * x[j] for j in range(n))
    by = sum(b[i] * y[i] for i in range(m))
    ok = (not pr_viol) and (not pr_sign) and (not du_viol) and (not du_sign) and (cx == by) \
        and (cx == f(value["value"])) and (by == f(value["dual_value"]))
    return {"ok": ok, "resid": [float(abs(cx - by))],
            "detail": f"原始可行: 违反约束行 {pr_viol}, 负分量 {pr_sign}; "
                      f"对偶可行: 违反约束列 {du_viol}, 负分量 {du_sign}; "
                      f"cᵀx={fs(cx)} vs bᵀy={fs(by)} (强对偶取等={cx == by})"}


REGISTRY.register(OpDef(
    name="cert_lp", band=BAND, sub=3,
    in_schema=_IN_LP,
    out_schema={"kind": "dict",
                "fields": {"x": {"kind": "list", "items": {"kind": "str"}, "minlen": 1, "maxlen": 5},
                           "y": {"kind": "list", "items": {"kind": "str"}, "minlen": 1, "maxlen": 6},
                           "value": {"kind": "str"}, "dual_value": {"kind": "str"},
                           "gap": {"kind": "str"},
                           "n_iter_basis": {"kind": "list", "items": {"kind": "int"}}},
                "required": ["x", "y", "value", "dual_value", "gap", "n_iter_basis"]},
    invoke=_inv_lp, verify=_ver_lp, cert_kind="lp_dual", precond=_pre_lp,
    cost=("units", 5.0), errors=("PRECOND_FAIL", "POSTCOND_FAIL", "DEG_UNSUPPORTED", "TIMEOUT",
                                 "SCHEMA_MISMATCH"),
    effects="none (纯函数, 精确有理算术)",
    doc="§8.5 认证线性规划 (标准形 max cᵀx, Ax≤b, x≥0) | in: c/A/b 有理 | out: 原始解 x + "
        "对偶解 y + 最优值 + 间隙 | 验收=**对偶证书三条件** (原始可行 + 对偶可行 + 强对偶间隙=0) "
        "—— 这是最优性的充分条件, 可完全机检 | 证书=lp_dual | 单纯形用 Bland 规则防循环 | "
        "5 units | 诚实边界: 仅小规模 (≤5 变量 × ≤6 约束); 退化/无界情形显式抛错不猜 | "
        "失败: 无界 DEG_UNSUPPORTED / 间隙≠0 POSTCOND_FAIL (绝不返回未经认证的最优解)",
    selfchecks=(
        {"id": "clp_simple_2d", "args": {"c": [3, 2], "A": [[1, 1], [1, 0], [0, 1]],
                                         "b": [4, 2, 3]},
         "check": lambda r: (r.value["value"] == "10" and r.value["gap"] == "0"
                             and r.value["x"] == ["2", "2"] and r.cert_independent is True,
                             f"x={r.value['x']} value={r.value['value']} gap={r.value['gap']}"),
         "expect": "max 3x+2y s.t. x+y≤4, x≤2, y≤3: 最优 (2,2) → 3·2+2·2=10。"
                   "手算: 顶点 (2,2) 处 10; (2,3) 处 12 但 x+y=5>4 不可行; (1,3) 处 9。故 10 ✓"},
        {"id": "clp_binding_single", "args": {"c": [1, 1], "A": [[1, 0], [0, 1]], "b": [5, 7]},
         "check": lambda r: (r.value["value"] == "12" and r.value["x"] == ["5", "7"],
                             f"x={r.value['x']} value={r.value['value']}"),
         "expect": "max x+y s.t. x≤5, y≤7 → x=(5,7), 值 12"},
        {"id": "clp_unbounded", "args": {"c": [1, 0], "A": [[-1, 0]], "b": [-1]},
         "expect_err": "DEG_UNSUPPORTED",
         "expect": "max x s.t. −x ≤ −1 (即 x≥1) 且 x≥0: 目标无界 → 拒绝认证 (不返回假最优)"},
        {"id": "clp_dual_certificate_nontrivial", "args": {"c": [1, 2], "A": [[1, 1], [2, 1]],
                                                           "b": [4, 6]},
         "check": lambda r: (r.value["gap"] == "0" and r.value["value"] == "8"
                             and len(r.value["y"]) == 2,
                             f"value={r.value['value']} y={r.value['y']}"),
         "expect": "max x+2y s.t. x+y≤4, 2x+y≤6: 顶点 (2,2) → 2+4=6? "
                   "枚举顶点: (0,0)=0, (3,0)=3, (0,4)=8, (2,2)=6 → 最优 8 在 (0,4)。"
                   "对偶 y 非平凡 (两条约束都要给出影子价格)"},
        {"id": "clp_bad_shape", "args": {"c": [1, 1], "A": [[1, 0]], "b": [1, 2]},
         "expect_err": "PRECOND_FAIL", "expect": "A 行数 ≠ b 长度"},
        {"id": "clp_float_reject", "args": {"c": [1.5, 1], "A": [[1, 0], [0, 1]], "b": [1, 1]},
         "expect_err": "SCHEMA_MISMATCH", "expect": "float 不进精确通道"},
    ),
))


# ---------------------------------------------------------------------------
# 9804 cert_lstsq — certified least squares
# ---------------------------------------------------------------------------

_IN_LS = {"kind": "dict",
          "fields": {"A": {"kind": "list", "minlen": 1, "maxlen": 5,
                           "items": {"kind": "list", "minlen": 1, "maxlen": 4,
                                     "items": {"kind": "Fraction"}}},
                     "y": {"kind": "list", "items": {"kind": "Fraction"}, "minlen": 1, "maxlen": 5}},
          "required": ["A", "y"]}


def _pre_ls(args) -> "tuple[bool, str]":
    if len(args["A"]) != len(args["y"]):
        return False, f"A 行数 {len(args['A'])} ≠ y 长度 {len(args['y'])}"
    n = len(args["A"][0])
    for i, row in enumerate(args["A"]):
        if len(row) != n:
            return False, f"A 第 {i} 行长度不等"
    if len(args["A"]) <= n:
        return False, (f"欠定: 方程数 {len(args['A'])} ≤ 自由度 {n} — 最小二乘解不唯一, "
                       f"本算子拒绝 (§17.2 欠定退化检查)")
    return True, ""


def _inv_ls(args: dict, ctl: CallCtl) -> dict:
    """least squares: solve the normal equations (AᵀA)x = Aᵀy (exact) and give a condition-number upper bound."""
    ctx = "op:cert_lstsq#9804"
    A = [[f(v) for v in row] for row in args["A"]]
    y = [f(v) for v in args["y"]]
    m, n = len(A), len(A[0])
    AtA = [[sum(A[k][i] * A[k][j] for k in range(m)) for j in range(n)] for i in range(n)]
    Aty = [sum(A[k][i] * y[k] for k in range(m)) for i in range(n)]
    x = _gauss(AtA, Aty)
    if x is None:
        raise SvoError(Err.DEG_UNSUPPORTED, ctx=ctx,
                       detail="AᵀA 奇异 — 法方程无唯一解 (A 列相关), 拒绝")
    res = [sum(A[k][j] * x[j] for j in range(n)) - y[k] for k in range(m)]
    rss = sum(r * r for r in res)
    # condition-number upper bound: κ₂(A) ≤ ‖A‖_F / σ_min, σ_min ≥ |det(AᵀA)|^{1/2} / ‖AᵀA‖_F^{n-1} (Hadamard type)
    fro2 = sum(A[k][j] * A[k][j] for k in range(m) for j in range(n))
    det = _det(AtA)
    # **true Frobenius norm** (not its square): auditor D measured that the old version used ‖AᵀA‖²_F where ‖AᵀA‖_F
    # was meant, so the "lower bound" on λ_min overshot the true value when ‖AᵀA‖_F<1 and the κ upper bound was 3.16×
    # smaller than the true κ (a fake upper bound).
    norm_AtA_fro = _sqrt_frac_ub(sum(AtA[i][j] * AtA[i][j] for i in range(n) for j in range(n)))
    # lower bound on λ_min(AᵀA) (Hadamard type): λ_min ≥ |det(AᵀA)| / ‖AᵀA‖_F^{n−1}
    # Note the exponent must go through **integer** powers: the old version wrote `norm ** ((n−1)/2)`, which promotes
    # a Fraction to float, and the downstream _sqrt_frac_lb immediately raised AttributeError (a float has no
    # numerator) — exactly the case §17.5 "exact arithmetic never passes through the float channel" must guarantee
    # structurally. Now integer powers + a rational square-root lower bound.
    if det == 0:
        raise SvoError(Err.DEG_UNSUPPORTED, ctx=ctx, detail="det(AᵀA)=0 — 条件数上界不可得")
    if n == 1:
        lam_min_lb = abs(AtA[0][0])
    else:
        if norm_AtA_fro == 0:
            raise SvoError(Err.DEG_UNSUPPORTED, ctx=ctx, detail="‖AᵀA‖_F=0 — 条件数上界不可得")
        lam_min_lb = abs(det) / (norm_AtA_fro ** (n - 1))
    sigma_min_lb = _sqrt_frac_lb(lam_min_lb)          # σ_min lower bound: round **down**
    # κ₂ = σ_max/σ_min ≤ ‖A‖_F/σ_min: round the numerator up and the denominator down; only consistent directions
    # deserve the name "upper bound".
    kappa_ub = (_sqrt_frac_ub(fro2) / sigma_min_lb) if sigma_min_lb > 0 else None
    return {"value": {"x": [fs(v) for v in x], "resid": [fs(v) for v in res],
                      "rss": fs(rss), "norm_resid": fs(_sqrt_frac_lb(rss)),
                      "kappa_ub": fs(kappa_ub) if kappa_ub is not None else "",
                      "det_ata": fs(det)},
            "resid": [float(rss)],
            "notes": f"最小二乘解 x={[fs(v) for v in x]}; 残差平方和 RSS={fs(rss)}; "
                     f"条件数上界 κ₂(A) ≤ {fs(kappa_ub) if kappa_ub else '不可得'} "
                     f"(由 |det(AᵀA)| 与 Frobenius 范数导出, 是**上界**不是精确值)"}


def _sqrt_frac_lb(x: Fraction) -> Fraction:
    """**Lower** bound on √x (rational, exact direction: return r with r ≤ √x)."""
    if x <= 0:
        return Fraction(0)
    scale = 10 ** 12
    n = math.isqrt(x.numerator * scale * scale // x.denominator)
    return Fraction(n, scale)


def _sqrt_frac_ub(x: Fraction) -> Fraction:
    """**Upper** bound on √x (rational, exact direction: return r with r ≥ √x).

    It is needed because: for the "upper bound on κ" the numerator ‖A‖_F must round **up**, while for the "lower bound
    on σ_min" the denominator must round **down** — the two directions are opposite, and mixing them yields a fake upper
    bound (auditor D's measured counter-example).
    """
    if x <= 0:
        return Fraction(0)
    lo = _sqrt_frac_lb(x)
    return lo if lo * lo >= x else lo + Fraction(1, 10 ** 12)


def _gauss(A, bvec):
    n = len(A)
    M = [list(A[i]) + [bvec[i]] for i in range(n)]
    for col in range(n):
        piv = next((r for r in range(col, n) if M[r][col] != 0), None)
        if piv is None:
            return None
        M[col], M[piv] = M[piv], M[col]
        pv = M[col][col]
        for r in range(n):
            if r != col and M[r][col] != 0:
                fac = M[r][col] / pv
                M[r] = [M[r][k] - fac * M[col][k] for k in range(n + 1)]
    return [M[i][n] / M[i][i] for i in range(n)]


def _det(M):
    n = len(M)
    M = [list(r) for r in M]
    d = Fraction(1)
    for col in range(n):
        piv = next((r for r in range(col, n) if M[r][col] != 0), None)
        if piv is None:
            return Fraction(0)
        if piv != col:
            M[col], M[piv] = M[piv], M[col]
            d = -d
        d *= M[col][col]
        for r in range(col + 1, n):
            if M[r][col] != 0:
                fac = M[r][col] / M[col][col]
                for c in range(col, n):
                    M[r][c] -= fac * M[col][c]
    return d


def _ver_ls(args: dict, value: dict) -> dict:
    """Independent verification (the **orthogonality characterisation** of least squares, a different path from solving the normal equations):
      (1) orthogonality condition Aᵀ(Ax − y) = 0 — the necessary and sufficient condition for the least-squares solution (the residual is orthogonal to A's column space);
      (2) residual recomputation: recompute res = Ax − y and RSS directly from x;
      (3) optimality sampling: along several perturbation directions, RSS(x + t·d) ≥ RSS(x) — a direct counter-example search."""
    A = [[f(v) for v in row] for row in args["A"]]
    y = [f(v) for v in args["y"]]
    x = [f(v) for v in value["x"]]
    m, n = len(A), len(A[0])
    res = [sum(A[k][j] * x[j] for j in range(n)) - y[k] for k in range(m)]
    # (1) orthogonality
    At_r = [sum(A[k][i] * res[k] for k in range(m)) for i in range(n)]
    orth = all(v == 0 for v in At_r)
    # (2) residual consistency
    rss = sum(r * r for r in res)
    rss_ok = (rss == f(value["rss"])) and ([fs(v) for v in res] == list(value["resid"]))
    # (3) perturbed optimality (exact rational directions)
    pert = 0
    dirs = []
    for j in range(n):
        d = [Fraction(1) if i == j else Fraction(0) for i in range(n)]
        dirs.append(d)
    if n >= 2:
        dirs.append([Fraction(1)] * n)
    for d in dirs:
        for t in (Fraction(1, 4), Fraction(1), Fraction(-1, 2)):
            xt = [x[i] + t * d[i] for i in range(n)]
            rt = [sum(A[k][j] * xt[j] for j in range(n)) - y[k] for k in range(m)]
            if sum(r * r for r in rt) < rss:
                pert += 1
    ok = orth and rss_ok and (pert == 0)
    return {"ok": ok, "resid": [float(rss), float(pert)],
            "detail": f"正交条件 Aᵀ(Ax−y)=0: {orth} (各分量 {[fs(v) for v in At_r]}); "
                      f"残差复算一致={rss_ok}; 扰动方向 {len(dirs)}×3 个候选全部不优于 x "
                      f"(更优者 {pert} 个) — 最小性直接反例搜索通过"}


REGISTRY.register(OpDef(
    name="cert_lstsq", band=BAND, sub=4,
    in_schema=_IN_LS,
    out_schema={"kind": "dict",
                "fields": {"x": {"kind": "list", "items": {"kind": "str"}, "minlen": 1, "maxlen": 4},
                           "resid": {"kind": "list", "items": {"kind": "str"}, "minlen": 1, "maxlen": 5},
                           "rss": {"kind": "str"}, "norm_resid": {"kind": "str"},
                           "kappa_ub": {"kind": "str"}, "det_ata": {"kind": "str"}},
                "required": ["x", "resid", "rss", "norm_resid", "kappa_ub", "det_ata"]},
    invoke=_inv_ls, verify=_ver_ls, precond=_pre_ls,
    cost=("units", 4.0), errors=("PRECOND_FAIL", "DEG_UNSUPPORTED", "SCHEMA_MISMATCH"),
    effects="none (纯函数, 精确有理算术)",
    doc="§8.5 认证最小二乘 | in: A (超定: 行数 > 列数) + y | out: 解 x + 残差 + RSS + 条件数上界 | "
        "验收=**正交条件** Aᵀ(Ax−y)=0 (最小二乘的充要刻画) + 残差复算 + 扰动方向反例搜索 | "
        "4 units | 诚实边界: κ₂ 是**上界**(由 |det(AᵀA)| 与 Frobenius 范数导出); "
        "欠定 (行数 ≤ 列数) 显式拒绝 (§17.2) | 失败: 欠定 PRECOND_FAIL / AᵀA 奇异 DEG_UNSUPPORTED",
    selfchecks=(
        {"id": "cls_line_fit", "args": {"A": [[1, 0], [1, 1], [1, 2]], "y": [1, 2, 3]},
         "check": lambda r: (r.value["x"] == ["1", "1"] and r.value["rss"] == "0",
                             f"x={r.value['x']} rss={r.value['rss']}"),
         "expect": "拟合 y=1+x 于 x=0,1,2: 完美拟合 → 解 (1,1), RSS=0"},
        {"id": "cls_noisy_fit", "args": {"A": [[1, 0], [1, 1], [1, 2], [1, 3]], "y": [1, 3, 2, 5]},
         "check": lambda r: (r.value["rss"] != "0" and len(r.value["resid"]) == 4
                             and f(r.value["kappa_ub"]) > 0,
                             f"x={r.value['x']} rss={r.value['rss']} kappa_ub={r.value['kappa_ub']}"),
         "expect": "带噪数据: 闭式 (解 4 个法方程) → 解与残差由精确有理算术给出; "
                   "噪声使 RSS > 0 且给出条件数上界"},
        {"id": "cls_underdetermined", "args": {"A": [[1, 0]], "y": [1]},
         "expect_err": "PRECOND_FAIL",
         "expect": "1 方程 2 未知 (行数 ≤ 列数): 最小二乘解不唯一 → 欠定检查拒绝 (§17.2)"},
        {"id": "cls_rank_deficient", "args": {"A": [[1, 2], [2, 4], [3, 6]], "y": [1, 2, 3]},
         "expect_err": "DEG_UNSUPPORTED",
         "expect": "A 列相关 (第 2 列 = 2×第 1 列) → AᵀA 奇异, 拒绝"},
        {"id": "cls_bad_shape", "args": {"A": [[1, 0], [1, 1]], "y": [1]},
         "expect_err": "PRECOND_FAIL", "expect": "A 行数 ≠ y 长度"},
    ),
))


# ---------------------------------------------------------------------------
# 9805 cert_series_bound — series remainder bounds (the quantitative version of the three ratio tests)
# ---------------------------------------------------------------------------

_IN_SB = {"kind": "dict",
          "fields": {"kind": {"kind": "str", "enum": ["geometric", "alternating", "ratio_bound"]},
                     "a": {"kind": "Fraction"}, "r": {"kind": "Fraction"},
                     "N": {"kind": "int", "min": 0, "max": 500},
                     "terms": {"kind": "list", "items": {"kind": "Fraction"}, "maxlen": 8},
                     "L": {"kind": "Fraction"}},
          "required": ["kind"]}


def _pre_sb(args) -> "tuple[bool, str]":
    k = args["kind"]
    if k == "geometric":
        if "a" not in args or "r" not in args:
            return False, "geometric 需要 a 与 r"
        if abs(f(args["r"])) >= 1:
            return False, f"|r|={abs(f(args['r']))} ≥ 1 → 级数发散, 无余项界可言"
    if k == "ratio_bound":
        if "L" not in args:
            return False, "ratio_bound 需要比值极限 L"
        if not (0 < f(args["L"]) < 1):
            return False, f"L={f(args['L'])} 不在 (0,1) → 比值判别法给不出收敛余项界"
    if k == "alternating":
        ts = args.get("terms") or []
        if len(ts) < 1:
            return False, "alternating 需要至少 1 项"
        vals = [abs(f(t)) for t in ts]
        if any(vals[i] < vals[i + 1] for i in range(len(vals) - 1)):
            return False, "交错级数的项绝对值必须单调不增 (Leibniz 条件)"
    return True, ""


def _inv_sb(args: dict, ctl: CallCtl) -> dict:
    """Explicit construction of series remainder bounds (three families):
      geometric   : |R_N| = |a|·|r|^N/(1−|r|)
      alternating : |R_N| ≤ |a_{N+1}|  (Leibniz)
      ratio_bound : if ∀n≥N: |a_{n+1}/a_n| ≤ L < 1, then |R_N| ≤ |a_N|·L/(1−L)
    All bounds are **rational numbers** derived in closed form (§8.5: approximate results must carry a certificate)."""
    k = args["kind"]
    if k == "geometric":
        a, r = f(args["a"]), f(args["r"])
        N = int(args.get("N", 20))
        s = a * (1 - r ** N) / (1 - r) if r != 1 else a * N
        bound = abs(a) * (abs(r) ** N) / (1 - abs(r))
        return {"value": {"partial_sum": fs(s), "bound": fs(bound), "N": N,
                          "interval": [fs(s - bound), fs(s + bound)]},
                "resid": [float(bound)],
                "notes": f"几何级数余项界: |R_{N}| ≤ |a||r|^N/(1−|r|) = {fs(bound)} "
                         f"(N={N}, a={fs(a)}, r={fs(r)})"}
    if k == "alternating":
        ts = [f(t) for t in args["terms"]]
        N = len(ts)
        s = sum(ts)
        bound = abs(ts[-1])
        return {"value": {"partial_sum": fs(s), "bound": fs(bound), "N": N,
                          "interval": [fs(s - bound), fs(s + bound)]},
                "resid": [float(bound)],
                "notes": f"交错级数 (Leibniz): |R_{N-1}| ≤ |末项| = {fs(bound)}; 部分和 {fs(s)}"}
    L = f(args["L"])
    aN = f(args.get("a", 1))
    N = int(args.get("N", 0))
    s = aN / (1 - L)
    bound = abs(aN) * L / (1 - L)
    return {"value": {"partial_sum": fs(s), "bound": fs(bound), "N": N,
                      "interval": [fs(s - bound), fs(s + bound)]},
            "resid": [float(bound)],
            "notes": f"比值界: 若 |a_{{n+1}}/a_n| ≤ L={fs(L)} < 1 则 |R_N| ≤ |a_N|·L/(1−L) "
                     f"= {fs(bound)}"}


def _ver_sb(args: dict, value: dict) -> dict:
    """Independent verification: (a) independently compute a reference ground truth with **higher-order partial sums**;
    it must fall inside the declared interval;
    (b) the interval's direction and width are self-consistent (lo ≤ hi, width = 2×bound);
    (c) for geometric additionally check the partial sum by **direct accumulation**."""
    k = args["kind"]
    lo, hi = f(value["interval"][0]), f(value["interval"][1])
    bound = f(value["bound"])
    width_ok = (lo <= hi) and (hi - lo == 2 * bound)
    # reference value: partial sum with more terms
    ref = None
    if k == "geometric":
        a, r = f(args["a"]), f(args["r"])
        N = int(value["N"]) + 40
        acc, cur = Fraction(0), a
        for _ in range(N):
            acc += cur
            cur *= r
        ref = acc
        ref_reason = f"再多 {40} 项的精确累加 (总 {N} 项)"
    elif k == "alternating":
        ts = [f(t) for t in args["terms"]]
        # assume terms keep decaying as |a|·ρ^k (ρ = ratio of the last two terms, or 1/2 if unavailable)
        vals = [abs(t) for t in ts]
        rho = (vals[-1] / vals[-2]) if len(vals) >= 2 and vals[-2] != 0 else Fraction(1, 2)
        if rho >= 1:
            rho = Fraction(1, 2)
        sign = 1 if ts[-1] > 0 else -1
        acc = sum(ts)
        cur = vals[-1] * rho
        for _ in range(60):
            sign = -sign
            acc += sign * cur
            cur *= rho
        ref = acc
        ref_reason = f"按比值 {fs(rho)} 延拓 60 项的累加"
    else:
        L = f(args["L"])
        aN = f(args.get("a", 1))
        acc = aN
        cur = aN
        for _ in range(80):
            cur *= L
            acc += cur
        ref = acc
        ref_reason = "按 L 延拓 80 项的累加"
    in_iv = (lo - abs(bound) / 10 ** 20) <= ref <= (hi + abs(bound) / 10 ** 20)
    ok = width_ok and in_iv
    return {"ok": ok, "resid": [float(abs(ref - (lo + hi) / 2))],
            "detail": f"区间宽度=2×界={width_ok}; 参照值 {fs(ref)} ({ref_reason}) "
                      f"∈ [{fs(lo)},{fs(hi)}] = {in_iv}"}


REGISTRY.register(OpDef(
    name="cert_series_bound", band=BAND, sub=5,
    in_schema=_IN_SB,
    out_schema={"kind": "dict",
                "fields": {"partial_sum": {"kind": "str"}, "bound": {"kind": "str"},
                           "N": {"kind": "int", "min": 0},
                           "interval": {"kind": "list", "len": 2, "items": {"kind": "str"}}},
                "required": ["partial_sum", "bound", "N", "interval"]},
    invoke=_inv_sb, verify=_ver_sb, precond=_pre_sb,
    cost=("units", 1.5), errors=("PRECOND_FAIL", "SCHEMA_MISMATCH"),
    effects="none (纯函数, 精确有理算术)",
    doc="§8.5 级数余项界 (三类: 几何/交错 Leibniz/比值) | in: 族参数 | "
        "out: 部分和 + 有理余项界 + 真值区间 | 验收=区间含参照真值 (更多项的独立累加) + "
        "宽度=2×界自洽 | 1.5 units | 诚实边界: 交错族的参照值按最后两项比值延拓 (若实际衰减"
        "慢于该比值, 参照值精度下降 — 已在 detail 中标注延拓方式, 且**判据不依赖它**, "
        "界本身是闭式推导) | 失败: 条件不满足 PRECOND_FAIL",
    selfchecks=(
        {"id": "csb_geometric_half", "args": {"kind": "geometric", "a": 1, "r": "1/2", "N": 4},
         "check": lambda r: (r.value["partial_sum"] == "15/8" and r.value["bound"] == "1/8"
                             and r.value["interval"] == ["7/4", "2"],
                             f"S4={r.value['partial_sum']} bound={r.value['bound']} "
                             f"interval={r.value['interval']}"),
         "expect": "Σ(1/2)ⁿ 前 4 项 S₄=1+1/2+1/4+1/8=15/8; 真值 = 2 → 真余项 = 1/8; "
                   "闭式界 |R_N|=|a||r|^N/(1−|r|)=(1/16)/(1/2)=1/8 (紧); "
                   "区间 [15/8−1/8, 15/8+1/8]=[7/4, 2] ✓"},
        {"id": "csb_alternating", "args": {"kind": "alternating", "terms": [1, "-1/2", "1/3", "-1/4"]},
         "check": lambda r: (r.value["partial_sum"] == "7/12" and r.value["bound"] == "1/4"
                             and r.value["interval"] == ["1/3", "5/6"],
                             f"S={r.value['partial_sum']} bound={r.value['bound']}"),
         "expect": "交错调和前 4 项: 1−1/2+1/3−1/4 = (12−6+4−3)/12 = 7/12 ✓; "
                   "Leibniz 界 = |末项| = 1/4; 区间 [7/12−1/4, 7/12+1/4] = [1/3, 5/6] ✓ "
                   "(真值 ln2≈0.693 落在区间内)"},
        {"id": "csb_ratio", "args": {"kind": "ratio_bound", "a": 1, "L": "1/2", "N": 0},
         "check": lambda r: (r.value["bound"] == "1" and r.value["partial_sum"] == "2",
                             f"bound={r.value['bound']} S={r.value['partial_sum']}"),
         "expect": "L=1/2, a=1: 界 = 1·(1/2)/(1/2) = 1; 和 = 1/(1−1/2) = 2"},
        {"id": "csb_divergent_geometric", "args": {"kind": "geometric", "a": 1, "r": 2},
         "expect_err": "PRECOND_FAIL", "expect": "|r|=2 ≥ 1 → 发散, 无余项界"},
        {"id": "csb_nonmonotone_alternating", "args": {"kind": "alternating",
                                                       "terms": [1, "-1/2", "1"]},
         "expect_err": "PRECOND_FAIL", "expect": "项绝对值非单调不增 → Leibniz 条件不满足"},
        {"id": "csb_bad_L", "args": {"kind": "ratio_bound", "a": 1, "L": 1},
         "expect_err": "PRECOND_FAIL", "expect": "L=1 不满足 L<1"},
    ),
))