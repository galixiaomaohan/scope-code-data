# -*- coding: utf-8 -*-
"""V31 new type system: algebraic numbers Q(sqrt d) and formal power series
(coefficient table + canonical form + equality certificates).

Why a "type" belongs in the operator library:
  The layer 1/2 operators (Pell / binary quadratic forms / modular-form q-expansion / Frey
  curves / transcendence tests) operate not on integers and rationals but on **extension-field
  elements** and **truncated power series**. If each built its own representation, the layers
  could not compose (layer 3 lemma instantiation must move objects across layers). This module
  therefore defines two canonical forms + normalized equality criteria, and for each type a
  **equality certificate** data shape (canonical string + structural checks), so "two objects
  are equal" can be verified by an independent checker instead of relying on Python's `==`.

Honesty boundaries (same discipline as the codebook):
  - Q(sqrt d) covers only real quadratic-field elements with **d square-free and > 1** (a
    square d degenerates the "field" to Q; this type **explicitly rejects** instead of quietly
    treating it as rational); in a + b√d, a,b ∈ Q exactly.
  - PowerSeries is a **truncated** power series (Z[q]/(q^N)); it has no notion of convergence,
    only "equal mod q^N", so the equality criterion must carry N -- this type stores N inside
    the object and raises on comparing objects with different N (equality at different
    precisions is a different proposition; silent alignment is the most common hidden
    wrong-value source).
"""
from __future__ import annotations

import math
from fractions import Fraction

from svokernel.errors import Err, SvoError
from svokernel.exact import frac, frac_str

__all__ = ["QsqrtD", "PowerSeries", "is_squarefree_int", "parse_qsqrtd",
           "parse_qsqrtd_lenient", "parse_series"]


def is_squarefree_int(n: int) -> bool:
    """Square-free test (n >= 1). Implementation independent of numtheory (kept local so the
    type invariants can be checked here)."""
    if n <= 0:
        return False
    if n == 1:
        return True
    d = 2
    m = n
    while d * d * d <= m:
        if m % d == 0:
            m //= d
            if m % d == 0:
                return False
        d += 1 if d == 2 else 2
    r = math.isqrt(m)
    return r * r != m


def _slot(v):
    """Normalize a **coefficient slot** literal in a Q(sqrt d) canonical string.

    Rule: a slot may carry one **leading '+' as an explicit sign separator** (b slot of
    "0+1√2" = "+1", b slot of "1+2√2" = "+2"); that '+' is stripped before the strict grammar
    of exact.frac (namely -?\\d+(/\\d+)? ) validates it -- **the legacy exact.py contract is
    untouched**; the relaxation applies only to this type's own serialization slots (slot
    syntax != rational-literal syntax, deliberately kept apart).
    Semantically +x == x, so stripping the sign changes no value; "-" is accepted by the
    strict grammar's own sign rule.
    """
    if isinstance(v, str) and len(v) > 1 and v[0] == "+":
        return v[1:]
    return v


class QsqrtD:
    """a + b√d (a, b ∈ Q, d square-free and > 1). Canonical form: field element <-> (d, a, b)
    is a bijection.

    Equality certificate (canonical form): {"d": d, "a": "p/q", "b": "p/q"} -- a checker only
    has to group by d and compare (a,b) as vectors of Q²; no method of this class is needed.
    """

    __slots__ = ("d", "a", "b")

    def __init__(self, d: int, a, b=0) -> None:
        d = int(d)
        if d <= 1 or not is_squarefree_int(d):
            raise SvoError(Err.PRECOND_FAIL, ctx="algnum.QsqrtD",
                           detail="要求 d 无平方且 d > 1 (得 d=%d); 完全平方 d 会让 Q(√d) 退化为 Q, "
                                  "本类型显式拒绝而不静默降级" % d)
        self.d = d
        self.a = frac(_slot(a), "algnum.QsqrtD.a")
        self.b = frac(_slot(b), "algnum.QsqrtD.b")

    # -- field arithmetic --------------------------------------------------
    def __add__(self, o: "QsqrtD") -> "QsqrtD":
        self._same_field(o)
        return QsqrtD(self.d, self.a + o.a, self.b + o.b)

    def __sub__(self, o: "QsqrtD") -> "QsqrtD":
        self._same_field(o)
        return QsqrtD(self.d, self.a - o.a, self.b - o.b)

    def __neg__(self) -> "QsqrtD":
        return QsqrtD(self.d, -self.a, -self.b)

    def __mul__(self, o: "QsqrtD") -> "QsqrtD":
        self._same_field(o)
        # (a+b√d)(c+e√d) = (ac + be·d) + (ae + bc)√d
        return QsqrtD(self.d, self.a * o.a + self.b * o.b * self.d, self.a * o.b + self.b * o.a)

    def norm(self) -> Fraction:
        """N(a+b√d) = a² − d b² (rational, exact)."""
        return self.a * self.a - self.d * self.b * self.b

    def inv(self) -> "QsqrtD":
        n = self.norm()
        if n == 0:
            raise SvoError(Err.DEG_UNSUPPORTED, ctx="algnum.QsqrtD.inv",
                           detail="范数为 0 ⇒ 零因子, 不可逆")
        return QsqrtD(self.d, self.a / n, -self.b / n)

    def __truediv__(self, o: "QsqrtD") -> "QsqrtD":
        return self * o.inv()

    def is_rational(self) -> bool:
        return self.b == 0

    def conjugate(self) -> "QsqrtD":
        return QsqrtD(self.d, self.a, -self.b)

    def min_poly(self) -> tuple:
        """Minimal polynomial coefficients (ascending powers, rational):
        x² − 2a·x + (a² − d b²)."""
        return (self.norm(), -2 * self.a, Fraction(1))

    # -- serialization -----------------------------------------------------
    def to_str(self) -> str:
        """Canonical string: b=0 -> pure rational ("5/6"); b!=0 -> "a±b√d" (**b slot carries
        an explicit sign**, e.g. "1+2√2").

        Fix (bug from a previous agent): the original used "%s%+s√%d" -- for strings and
        numbers alike it produces no '+' (measured), so "1+2√2" came out as "12√2": the
        square-free slot was concatenated into the a slot => on read-back a=12, b=0 (silent
        wrong value). The separator is now built explicitly; printf sign flags are no longer
        trusted. b=0 omits the √ tail -- parse_qsqrtd_lenient carries that convention back.
        """
        if self.b == 0:
            return frac_str(self.a)
        bs = frac_str(self.b)
        if not bs.startswith("-"):
            bs = "+" + bs
        return "%s%s√%d" % (frac_str(self.a), bs, self.d)

    def cert_dict(self) -> dict:
        return {"d": self.d, "a": frac_str(self.a), "b": frac_str(self.b)}

    def _same_field(self, o: "QsqrtD") -> None:
        if not isinstance(o, QsqrtD):
            raise SvoError(Err.SCHEMA_MISMATCH, ctx="algnum.QsqrtD",
                           detail="跨类型运算被拒 (得 %s)" % type(o).__name__)
        if o.d != self.d:
            raise SvoError(Err.SCHEMA_MISMATCH, ctx="algnum.QsqrtD",
                           detail="跨域运算被拒: Q(√%d) vs Q(√%d)" % (self.d, o.d))

    def __eq__(self, o) -> bool:
        return isinstance(o, QsqrtD) and o.d == self.d and o.a == self.a and o.b == self.b

    def __hash__(self) -> int:
        return hash((self.d, self.a, self.b))

    def __repr__(self) -> str:
        return "QsqrtD(%s)" % self.to_str()


def parse_qsqrtd(s: str) -> QsqrtD:
    """Parse the canonical string "a+b√d" / "a-b√d" (a,b are canonical rational literals;
    empty means 0)."""
    if not isinstance(s, str) or "√" not in s:
        raise SvoError(Err.SCHEMA_MISMATCH, ctx="algnum.parse_qsqrtd",
                       detail="非 Q(√d) 规范串: %r" % (s,))
    head, _, tail = s.partition("√")
    d = int(tail)
    # head looks like "3/2+1/2" or "-1" or "1-1/2"; split at the last +/- (not at position 0)
    idx = -1
    for i in range(len(head) - 1, 0, -1):
        if head[i] in "+-" and head[i - 1] != "/":
            idx = i
            break
    if idx < 0:
        a_str, b_str = head, "0"
    else:
        a_str, b_str = head[:idx], head[idx:]
    if a_str in ("", "+"):
        a_str = "0"
    return QsqrtD(d, a_str, b_str)


# ---------------------------------------------------------------------------
# formal power series (truncated)
# ---------------------------------------------------------------------------


class PowerSeries:
    """Element of Z[q]/(q^N): coefficient table + precision N (canonical form: trailing zeros
    dropped but the precision marker kept).

    Equality certificate: {"prec": N, "coeffs": [...]} -- two objects are equal <=> same
    precision and equal coefficient tables; different precisions raise (equality at different
    precisions is a different proposition, see the module docstring).
    """

    __slots__ = ("prec", "coeffs")

    def __init__(self, coeffs, prec=None) -> None:
        cs = [int(c) if not isinstance(c, Fraction) else c for c in coeffs]
        for c in cs:
            if isinstance(c, bool) or not isinstance(c, (int, Fraction)):
                raise SvoError(Err.SCHEMA_MISMATCH, ctx="algnum.PowerSeries",
                               detail="系数必须为整数 (得 %r)" % (c,))
        if prec is None:
            prec = len(cs)
        prec = int(prec)
        if prec < 1:
            raise SvoError(Err.SCHEMA_MISMATCH, ctx="algnum.PowerSeries", detail="精度 N >= 1")
        if len(cs) < prec:
            cs = cs + [0] * (prec - len(cs))
        elif len(cs) > prec:
            cs = cs[:prec]
        self.prec = prec
        self.coeffs = cs

    def __add__(self, o: "PowerSeries") -> "PowerSeries":
        self._same_prec(o)
        return PowerSeries([x + y for x, y in zip(self.coeffs, o.coeffs)], self.prec)

    def __sub__(self, o: "PowerSeries") -> "PowerSeries":
        self._same_prec(o)
        return PowerSeries([x - y for x, y in zip(self.coeffs, o.coeffs)], self.prec)

    def __mul__(self, o: "PowerSeries") -> "PowerSeries":
        """Truncated convolution (schoolbook). Generator-side path; the checker uses Kronecker
        substitution (big-integer multiplication)."""
        self._same_prec(o)
        N = self.prec
        out = [0] * N
        for i, x in enumerate(self.coeffs):
            if x == 0:
                continue
            for j, y in enumerate(o.coeffs[: N - i]):
                out[i + j] += x * y
        return PowerSeries(out, N)

    def scale(self, k: int) -> "PowerSeries":
        return PowerSeries([c * k for c in self.coeffs], self.prec)

    def q_power_shift(self, m: int) -> "PowerSeries":
        """Multiply by q^m (insert m zero coefficients; precision unchanged)."""
        return PowerSeries([0] * m + self.coeffs, self.prec)

    def coeff(self, n: int) -> int:
        return self.coeffs[n] if 0 <= n < self.prec else 0

    def to_str(self) -> str:
        return "[" + ",".join(str(c) for c in self.coeffs) + "](q^%d)" % self.prec

    def cert_dict(self) -> dict:
        return {"prec": self.prec, "coeffs": list(self.coeffs)}

    def _same_prec(self, o: "PowerSeries") -> None:
        if not isinstance(o, PowerSeries):
            raise SvoError(Err.SCHEMA_MISMATCH, ctx="algnum.PowerSeries",
                           detail="跨类型运算被拒 (得 %s)" % type(o).__name__)
        if o.prec != self.prec:
            raise SvoError(Err.SCHEMA_MISMATCH, ctx="algnum.PowerSeries",
                           detail="精度不同不可直接运算: q^%d vs q^%d (显式拒绝而非静默对齐)"
                                  % (self.prec, o.prec))

    def __eq__(self, o) -> bool:
        return isinstance(o, PowerSeries) and o.prec == self.prec and o.coeffs == self.coeffs

    def __repr__(self) -> str:
        return "PowerSeries(%s)" % self.to_str()


def parse_series(coeffs, prec) -> PowerSeries:
    return PowerSeries(list(coeffs), int(prec))


def parse_qsqrtd_lenient(s: str, d_default: int = 2) -> QsqrtD:
    """Canonical string -> QsqrtD, where a **pure rational string (no '√') denotes an element
    with b=0** (d takes the d_default placeholder).

    Why needed: to_str serializes field elements with b=0 to a pure rational string ("5/6"),
    because a+0√d is just a; the read-back side must accept that form, otherwise "reading back
    a string one wrote" fails. d is only a placeholder -- callers comparing (a, b) pairs are
    independent of d; callers that need d to mean something must **not** use this function
    (use parse_qsqrtd instead).
    """
    if isinstance(s, str) and "√" not in s:
        return QsqrtD(d_default, s, 0)
    return parse_qsqrtd(s)
