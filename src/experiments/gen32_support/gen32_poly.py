# -*- coding: utf-8 -*-
# gen32_poly.py -- gen32 polynomial engine (generator side; reuses the dump/load/parse/render conventions of the frozen bh_poly)
#
# Division of labor with bh_check / gen32_check: this module **serves the generator and operators only**;
# the checker (gen32_check) **must not** import this module (register_checker structurally rejects same-module self-certification).
#
# New in this module (relative to bh_poly): the resid8 spec (§2 pre-registered) -- the step state is the exact value of the
# primary operation polynomial at 8 fixed integer grid points; the nonzero criterion is fail-closed.
from fractions import Fraction

import bh_poly as P                                                   # read-only reuse


def padd(p, q):
    return P.padd(p, q)


def psub(p, q):
    return P.psub(p, q)


def pscale(p, k):
    return P.pscale(p, k)


def pmul(p, q):
    return P.pmul(p, q)


def ppow(p, k):
    return P.ppow(p, k)


def peval(p, vals):
    return P.peval(p, vals)


def dump(p):
    return P.dump(p)


def load(d):
    return P.load(d)


def render_poly(p, vars_):
    return P.render_poly(p, vars_)


def parse_poly(text, vars_):
    return P.parse_poly(text, vars_)


def total_deg(p):
    return P.total_deg(p)


def shift_sub(p, var_idx=0):
    """Δp = p(n+1) − p(n) (binomial expansion; isomorphic to bh_ops.shift_sub, kept locally in this module)."""
    from math import comb
    out = {}
    for m, c in p.items():
        e = m[var_idx]
        if e == 0:
            continue
        for j in range(e):
            mm = list(m)
            mm[var_idx] = j
            key = tuple(mm)
            out[key] = out.get(key, Fraction(0)) + c * comb(e, j)
    return {m: c for m, c in out.items() if c != 0}


def solve_difference(S, v0):
    """Solve P(n+1) − P(n) = S(n), P(0) = v0 (rational coefficients; constructive generator algorithm)."""
    from math import comb
    d = max([m[0] for m in S] or [0])
    p = {(0,): Fraction(v0)}
    sol = {}
    for i in range(d + 1, 0, -1):
        tgt = sum((c for m, c in S.items() if m[0] == i - 1), Fraction(0))
        acc = Fraction(0)
        for j in range(i + 1, d + 2):
            if j in sol:
                acc += sol[j] * comb(j, i - 1)
        sol[i] = (tgt - acc) / i
    for i, c in sol.items():
        if c != 0:
            p[(i,)] = p.get((i,), Fraction(0)) + c
    return {m: c for m, c in p.items() if c != 0}


def integral(P_, v0):
    """Closed form of Σ_{k<n} P_(k) (i.e. solve ΔQ = P_, Q(0)=v0)."""
    return solve_difference(P_, v0)


def cumsum0(P_):
    """Q(n) = Σ_{k<n} P_(k) with Q(0) = 0."""
    return solve_difference(P_, 0)


def const_term(p, nv=1):
    return p.get(tuple([0] * nv), Fraction(0))


# ---------------------------------------------------------------- resid spec (§2)
def poly_points(nv):
    """8 fixed integer grid points P_j = (j+2, j+3, ..., j+2+nv−1), j = 0..7."""
    return [tuple(j + 2 + i for i in range(nv)) for j in range(8)]


RES_DIM = 8


def resid_vals(p, nv):
    """Exact values (list of Fraction, length 8). p may be None (the zero polynomial)."""
    pts = poly_points(nv)
    if p is None:
        p = {}
    return [P.peval(p, pt) for pt in pts]


def resid_is_nonzero(p, nv):
    return any(v != 0 for v in resid_vals(p, nv))


def resid_pack(p, nv, spec):
    """-> {res, res_exact, res_spec}. res is the user-side frame value (int, always 8-dim); res_exact is the exact string form."""
    vals = resid_vals(p, nv)
    out = []
    for v in vals:
        v = Fraction(v)
        out.append(int(v) if v.denominator == 1 else float(v))
    return {"res": out, "res_exact": [str(v) for v in vals], "res_spec": spec}


def resid_ok(pack):
    """Nonzero criterion (fail-closed): at least one of the 8 dimensions is nonzero."""
    return any(x != 0 for x in pack["res"])
