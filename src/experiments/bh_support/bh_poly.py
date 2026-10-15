# -*- coding: utf-8 -*-
# bh_poly.py -- bridge-hard batch polynomial engine (generator side; gen14, change-58 curriculum-upgrade batch)
#
# Serves the generators only (bh_ops/bh_bridge/bh_families). **The checker (bh_check) must not import
# this module**: two-path independence is a structural constraint of the certificate machinery
# (mirrors the ccp_ops/ccp_check independent_of design).
# Supports: integer/fraction coefficients, + - * ^(numeric exponent), nested-parenthesis multiplication
# (i.e. real expansion), type-annotation stripping.
import re
from fractions import Fraction

_CAST = re.compile(r":\s*(?:\u2124|\u211a|\u2115|\u211d)")


def strip_cast(s):
    """Strip Lean type annotations (`(n : Z)` -> `(n)`); safe because the generator emits controlled text only."""
    return _CAST.sub("", s or "")


def _mk(vars_):
    idx = {v: i for i, v in enumerate(vars_)}
    n = len(vars_)
    return idx, n


def pzero(n):
    return {}


def padd(p, q):
    out = dict(p)
    for m, c in q.items():
        out[m] = out.get(m, Fraction(0)) + c
    return {m: c for m, c in out.items() if c != 0}


def psub(p, q):
    return padd(p, {m: -c for m, c in q.items()})


def pscale(p, k):
    k = Fraction(k)
    if k == 0:
        return {}
    return {m: c * k for m, c in p.items()}


def pmul(p, q):
    if not p or not q:
        return {}
    out = {}
    for m1, c1 in p.items():
        for m2, c2 in q.items():
            m = tuple(a + b for a, b in zip(m1, m2))
            out[m] = out.get(m, Fraction(0)) + c1 * c2
    return {m: c for m, c in out.items() if c != 0}


def ppow(p, k):
    out = {tuple([0] * len(next(iter(p)))): Fraction(1)} if p else {}
    for _ in range(k):
        out = pmul(out, p)
    return out


def peval(p, vals):
    s = Fraction(0)
    for m, c in p.items():
        t = c
        for i, e in enumerate(m):
            if e:
                t *= Fraction(vals[i]) ** e
        s += t
    return s


def total_deg(p):
    return max([sum(m) for m in p] or [0])


def deg_in(p, i):
    return max([m[i] for m in p] or [0])


def lead_coef(p, i):
    d = deg_in(p, i)
    for m, c in p.items():
        if m[i] == d and all(e == 0 for j, e in enumerate(m) if j != i):
            return c
    # With mixed terms, sum those of highest degree in this variable (this batch's generators only compare single-variable powers)
    return sum((c for m, c in p.items() if m[i] == d), Fraction(0))


class ParseError(ValueError):
    pass


def parse_poly(text, vars_):
    """Text -> {exponent tuple: Fraction}. vars_ fixes the order; an unregistered identifier raises ParseError."""
    idx, n = _mk(vars_)
    s = strip_cast(text)
    s = s.replace("\u00b7", "*").replace("\u2212", "-")
    toks = _tok(s)
    pos = [0]

    def peek():
        return toks[pos[0]] if pos[0] < len(toks) else None

    def take():
        t = peek()
        pos[0] += 1
        return t

    def atom():
        t = take()
        if t is None:
            raise ParseError("意外结束")
        if t == "(":
            p = expr()
            if take() != ")":
                raise ParseError("缺右括号")
            return p
        if t == "-":
            return pscale(atom(), -1)
        if t == "+":
            return atom()
        if re.fullmatch(r"\d+", t):
            return {tuple([0] * n): Fraction(int(t))}
        if t in idx:
            m = [0] * n
            m[idx[t]] = 1
            return {tuple(m): Fraction(1)}
        raise ParseError("未知记号 %r" % t)

    def factor():
        base = atom()
        while peek() == "^":
            take()
            e = take()
            if not re.fullmatch(r"\d+", e or ""):
                raise ParseError("非数字指数 %r" % e)
            base = ppow(base, int(e))
        return base

    def term():
        p = factor()
        while peek() == "*":
            take()
            p = pmul(p, factor())
        return p

    def expr():
        p = term()
        while peek() in ("+", "-"):
            op = take()
            q = term()
            p = padd(p, q if op == "+" else pscale(q, -1))
        return p

    out = expr()
    if pos[0] != len(toks):
        raise ParseError("尾部残留 %r" % toks[pos[0]:])
    return out


def _tok(s):
    out, cur = [], ""
    for ch in s:
        if ch.isspace():
            if cur:
                out.append(cur)
                cur = ""
            continue
        if ch in "+-*^()":
            if cur:
                out.append(cur)
                cur = ""
            out.append(ch)
            continue
        cur += ch
    if cur:
        out.append(cur)
    return out


def render_poly(p, vars_, var_first=True):
    """Canonical-form text: monomials in descending order (by total degree, then lexicographic),
    for state display and Lean re-parsing."""
    items = sorted([(m, c) for m, c in p.items() if c != 0],
                   key=lambda kv: (-sum(kv[0]), kv[0]))
    parts = []
    for m, c in items:
        c = Fraction(c)
        neg = c < 0
        a = abs(c)
        mono = []
        for i, e in enumerate(m):
            if e == 0:
                continue
            mono.append(vars_[i] if e == 1 else "%s ^ %d" % (vars_[i], e))
        if not mono:
            body = str(a) if a.denominator == 1 else str(a)
        else:
            b = " * ".join(mono)
            if a == 1:
                body = b
            else:
                body = "%s * %s" % (a if a.denominator == 1 else a, b)
        parts.append(("-" if neg else "+", body))
    if not parts:
        return "0"
    out = ""
    for i, (sg, b) in enumerate(parts):
        if i == 0:
            out = ("- " if sg == "-" else "") + b
        else:
            out += (" - " if sg == "-" else " + ") + b
    return out


def dump(p):
    return {",".join(str(e) for e in m): str(c) for m, c in p.items() if c != 0}


def load(d):
    return {tuple(int(x) for x in k.split(",")): Fraction(v) for k, v in d.items()}
