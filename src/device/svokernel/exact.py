r"""Exact rational channel.

**Why it exists**: exact arithmetic never passes through the float channel (§17.5). JSON and
literals have no rational type, so this package defines a **canonical rational literal** as the
serialized form of the exact channel: `"3"`, `"-2"`, `"1/2"`, `"-7/3"`. Rules:
  - only `-?\d+(/\d+)?` is allowed — decimal notation ("1.5") and scientific notation are
    rejected outright (they are semantically float; admitting them would open a back door for float);
  - the denominator must not be 0;
  - output is always normalized (reduced; integers written as bare numerator with denominator 1).

Constructing from float, e.g. `Fraction(1.5)`, is forbidden inside this package: `frac()`
explicitly rejects float inputs.
"""
from __future__ import annotations

import re
from fractions import Fraction

from .errors import Err, SvoError

_RAT = re.compile(r"^-?\d+(/\d+)?$")


def is_exact_literal(s: str) -> bool:
    if not isinstance(s, str):
        return False
    return bool(_RAT.match(s))


def frac(v: "int | Fraction | str", ctx: str = "exact.frac") -> Fraction:
    """Exact Fraction constructor: accepts int / Fraction / canonical rational literal string; rejects float."""
    if isinstance(v, bool):
        raise SvoError(Err.SCHEMA_MISMATCH, ctx=ctx, detail="bool 不是有理数")
    if isinstance(v, Fraction):
        return v
    if isinstance(v, int):
        return Fraction(v)
    if isinstance(v, str):
        if not is_exact_literal(v):
            raise SvoError(
                Err.SCHEMA_MISMATCH,
                ctx=ctx,
                detail=f"非规范有理字面量 {v!r} (只允许 -?\\d+(/\\d+)?; 小数写法属 float 通道, 禁用)",
            )
        num, _, den = v.partition("/")
        d = int(den) if den else 1
        if d == 0:
            raise SvoError(Err.DEG_UNSUPPORTED, ctx=ctx, detail=f"分母为 0: {v!r}")
        return Fraction(int(num), d)
    raise SvoError(
        Err.SCHEMA_MISMATCH,
        ctx=ctx,
        detail=f"拒绝 {type(v).__name__} 进入精确通道 (float 不得伪装精确: {v!r})",
    )


def frac_str(f: Fraction) -> str:
    """Canonical serialization: 3 → '3'; 1/2 → '1/2'; -4/2 → '-2'."""
    fr = Fraction(f)
    return str(fr.numerator) if fr.denominator == 1 else f"{fr.numerator}/{fr.denominator}"


def frac_list(vs, ctx: str = "exact.frac_list") -> list[Fraction]:
    return [frac(v, ctx) for v in vs]


def frac_list_str(fs) -> list[str]:
    return [frac_str(f) for f in fs]
