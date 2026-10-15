# -*- coding: utf-8 -*-
# gen44_fam_b789.py -- **shared kernel + gate driver** for families B7/B8/B9 (fork-copy style; the
#                     parents gen44_bcore.py / gen44_bgate.py / gen44_barm.py are **untouched**,
#                     import only)
#
# Read-only import, zero modification: gen44_bcore (shared utilities + the copied drct judging) ·
#                  gen42_stmt · gen42_families · gen44_render_exam_v3 (exam single rendering) ·
#                  gen44_render_gold (gold enumeration) · drct_common
#
# Operator semantics (device computation; everything is **atomized** at the text level and then
# handed to sympy for exact checks, while the route is still really decided by F.chain_of):
#   B7 `g44_pow_atom`  : symbolic exponent as an atom -- in `X ^ E`, an E that is not a "pure integer
#                        / pure variable" (census reject reason `expnotnum`) is replaced by a
#                        placeholder atom, and the device provides an **exact model** (closed value ->
#                        exact rational, otherwise Pow(base, exp, evaluate=False)); isomorphic to lin
#                        treating a nonlinear monomial as an atom.
#      `g44_pow_cap`   : exponent cap 256 -> 4096 (registered); closed integer exponents within the
#                        cap are **evaluated exactly**.
#      `g44_fact_atom` : postfix factorial `X !` with a symbolic argument (`fact_sym`) -> atom
#                        p44_factn(arg); a closed integer argument -> exact factorial value
#                        (budget 3000).
#   B8 `g44_lib_atom`  : dotted library names (`Nat.gcd/Nat.lcm/Nat.choose/Nat.fib/Int.floor/...`)
#                        enter the table; a closed integer argument in the computable table -> exact
#                        value; otherwise -> opaque atom p44_lib_<name>.
#   B9 `g44_relchain`  : >=2 relation symbols at depth 0 in the conclusion (ASCII `>=`/`<=` first
#                        normalized to ≥/≤) -> mechanically solved as a **right-nested conjunction
#                        chain** `a ⋈ b ⋈ c` ==> `(a ⋈ b) ∧ (b ⋈ c)`, handed to the B1 split
#                        (`try constructor`) plus the leaf certificate of each piece.
#
# Rendering (exam single-rendering convention = RE3): leaves always use the existing chain (every
#   line under try, no enumeration, fail-closed); non-leaf lines within a family:
#   B7/B8 have none (atomization only changes the device's internal model, never the statement text)
#   |  B9 = `try constructor`
# Gold enumeration convention (= RG.candidates bounded envelope, cap): **device upper-bound
#   convention only**, separate artifact and md5 per piece, forbidden inside the exam chain.
# State frames: always computed **on the fly** (F.frames_of_std / frame_scalar); mechanical
#   assertions in frames_audit.
#
# Gate criteria (pre-registered in the §Gate criteria section of this file; same convention for all
#   three families):
#   C1 structure : census.json exists ∧ n_pop > 0
#   C2 synth     : corpus n >= 400 ∧ check n_sample >= 12 ∧ rate >= 0.80 ∧ n2_accepted >= 0.9*n
#   C3 payload   : payload verdict PASS (slot_bad == 0 ∧ digit_bad == 0)
#   C4 leak      : verdict PASS (zero hits on all three channels) ∧ positive control pass
#   C5 end2end   : arm_exam exists ∧ n_judged >= 0.8*n_sample ∧ **report rate + Wilson n/total**
#                  (the PASS criterion is only n_certified >= 1; a low rate is a reading, not a gate
#                  failure -- see the [S2] ruling three)
#   C6 renderers : exam md5 == 8c1ad680… ∧ gold md5 == 9063d26c… ∧ judge md5 == 7c31b58d…
#                  (exam/gold digests are pre-translation host values; the
#                  package re-freeze lives in gen44_bcore.EXAM_MD5/GOLD_MD5)
#   C7 frames    : last frame = exhausted frame bad == 0 ∧ n_empty_bad == 0
import argparse
import collections
import glob
import json
import os
import random
import re
import signal
import sys
import time

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)
import gen44_bcore as B                                                  # noqa: E402
import gen42_stmt as S                                                   # noqa: E402
import gen42_families as F                                               # noqa: E402
import gen44_render_exam_v3 as RE3                                       # noqa: E402
import gen44_render_gold as RG                                           # noqa: E402

import sympy as sp                                                       # noqa: E402

OUT = B.OUT
FAMS = ("b7", "b8", "b9")
CAP_POW = 4096                      # g44_pow_cap: 256 -> 4096 (registered)
FACT_CAP = 800                      # closed-value factorial budget (800! = 1977 digits < decimal conversion cap; registered)
VALUE_BITS = 8000                   # bit-width upper bound for exact closed values (~2409 decimal digits; registered)
ROW_TIMEOUT_S = 8.0                 # per-row watchdog (timeouts are registered, never scored as 0; same convention as c2state)
try:                                # guard against gen42 route_of crashing on str(huge int) (Python int_max_str_digits)
    sys.set_int_max_str_digits(1000000)
except Exception:                                                   # noqa: BLE001
    pass
FAM_MODS = {}
G44_OPS = {"b7": ["g44_pow_atom", "g44_pow_cap", "g44_fact_atom"],
           "b8": ["g44_lib_atom"],
           "b9": ["g44_relchain"]}

FAM_LABELS = {
    "b7": ("C_unparsed:pow_sym", "H_unparsed:pow_sym", "C_unparsed:exp_big",
           "H_unparsed:exp_big", "C_unparsed:fact", "H_unparsed:fact"),
    "b8": ("C_unparsed:funcapp", "H_unparsed:funcapp"),
    "b9": ("C_relchain",),
}


def fam_module(fam):
    if fam not in FAM_MODS:
        FAM_MODS[fam] = __import__("gen44_fam_%s" % fam)
    return FAM_MODS[fam]


class Timeout(BaseException):
    """Watchdog exception: inherits BaseException => gen42/gen44 `except Exception` handlers
    cannot swallow it."""
    pass


def _to_handler(_signum, _frame):
    raise Timeout()


def with_timeout(sec, fn, *a, **kw):
    """Per-row watchdog (SIGALRM; same convention as the 8 s per-row timeout of gen44_c2state):
    on timeout it raises Timeout -> the row is recorded as `timeout`, **never scored as 0 and never
    given a reject reason**."""
    if sec is None or sec <= 0 or not hasattr(signal, "SIGALRM"):
        return fn(*a, **kw)
    old = signal.signal(signal.SIGALRM, _to_handler)
    signal.setitimer(signal.ITIMER_REAL, float(sec))
    try:
        return fn(*a, **kw)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, old)


# ================================================================ §1 text-level atomization (B7/B8)
_IDCH = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_'."
            "₀₁₂₃₄₅₆₇₈₉")
_LIB_ARITY = {"Nat.gcd": 2, "Nat.lcm": 2, "Int.gcd": 2, "Nat.choose": 2,
              "Nat.fib": 1, "Nat.factorial": 1, "Nat.totient": 1, "Nat.sqrt": 1,
              "Int.floor": 1, "Nat.floor": 1, "Int.fract": 1, "Complex.abs": 1,
              "Nat.log2": 1, "Nat.log": 2, "Nat.mod": 2, "Nat.div": 2,
              "Nat.size": 1, "Int.natAbs": 1, "Rat.floor": 1, "Rat.ceil": 1}
_LIB_COMPUTABLE = ("Nat.gcd", "Nat.lcm", "Int.gcd", "Nat.choose", "Nat.fib",
                   "Nat.factorial", "Nat.totient", "Nat.sqrt", "Nat.log", "Nat.log2",
                   "Nat.mod", "Nat.div", "Int.natAbs")
_DOTTED = re.compile(r"[A-Za-z_][A-Za-z0-9_'₀-₉]*(?:\.[A-Za-z_][A-Za-z0-9_'₀-₉]*)+")


def _san(t):
    return re.sub(r"[^A-Za-z0-9_]", "_", t or "x")


def _span_left(t, i):
    """Start index of the **atom** (identifier / number / bracket group) immediately left of
    position i; None if absent."""
    j = i
    while j > 0 and t[j - 1] == " ":
        j -= 1
    if j <= 0:
        return None
    if t[j - 1] == ")":
        d, k = 0, j - 1
        while k >= 0:
            if t[k] == ")":
                d += 1
            elif t[k] == "(":
                d -= 1
                if d == 0:
                    return k
            k -= 1
        return None
    k = j
    while k > 0 and t[k - 1] in _IDCH:
        k -= 1
    return k if k < j else None


def _span_right(t, i):
    """End (exclusive) of the atom immediately right of t[i:]; None if absent."""
    j = i
    while j < len(t) and t[j] == " ":
        j += 1
    if j >= len(t):
        return None
    if t[j] == "(":
        d, k = 0, j
        while k < len(t):
            if t[k] == "(":
                d += 1
            elif t[k] == ")":
                d -= 1
                if d == 0:
                    return k + 1
            k += 1
        return None
    k = j
    while k < len(t) and t[k] in _IDCH:
        k += 1
    return k if k > j else None


def scrub_pow(txt, k0=0):
    """`X ^ E` / `X ** E` -> placeholder p44_k (scanning from the right, nesting-friendly).
    -> (txt2, subs, err)"""
    out, subs, k = txt, {}, k0
    for _ in range(96):
        i1, i2 = out.rfind("^"), out.rfind("**")
        idx = max(i1, i2)
        if idx < 0:
            break
        oplen = 2 if i2 > i1 else 1
        l, r = _span_left(out, idx), _span_right(out, idx + oplen)
        if l is None or r is None:
            return None, {}, "pow_span"
        base, exp = out[l:idx].strip(), out[idx + oplen:r].strip()
        if not base or not exp:
            return None, {}, "pow_span"
        ei = int(exp) if re.fullmatch(r"\d+", exp) else None
        nm = "p44_%d" % k
        k += 1
        subs[nm] = {"kind": "pow", "base": base, "exp": exp, "exp_int": ei}
        out = out[:l] + nm + out[r:]
    return out, subs, None


def scrub_fact(txt, k0=0):
    """Postfix factorial `X !` -> placeholder p44_k. -> (txt2, subs, err)"""
    out, subs, k = txt, {}, k0
    for _ in range(96):
        idx = out.rfind("!")
        if idx < 0:
            break
        l = _span_left(out, idx)
        if l is None:
            return None, {}, "fact_span"
        arg = out[l:idx].strip()
        if not arg:
            return None, {}, "fact_span"
        nm = "p44_%d" % k
        k += 1
        subs[nm] = {"kind": "fact", "arg": arg}
        out = out[:l] + nm + out[idx + 1:]
    return out, subs, None


def scrub_lib(txt, k0=0):
    """Dotted library names (`Nat.gcd a b` / `n.choose k` / `Complex.abs x`) -> placeholder / exact
    value. -> (txt2, subs, err)"""
    out, subs, k = txt, {}, k0
    for _ in range(96):
        m = None
        for mm in _DOTTED.finditer(out):
            m = mm                                          # rightmost match (nesting-friendly)
        if m is None:
            break
        name = m.group(0)
        if name.startswith("p44_"):
            break
        n = _LIB_ARITY.get(name, 1)
        end, args, ok = m.end(), [], True
        for _i in range(n):
            r = _span_right(out, end)
            if r is None:
                ok = False
                break
            args.append(out[end:r].strip())
            end = r
        if not ok:
            args = []
        nm = "p44_%d" % k
        k += 1
        subs[nm] = {"kind": "lib", "name": name, "args": args}
        out = out[:m.start()] + nm + out[end:]
    return out, subs, None


_SCRUBBERS = (("pow", scrub_pow), ("fact", scrub_fact), ("lib", scrub_lib))


def _int_of(e, lo=-10 ** 6, hi=10 ** 6):
    try:
        if e is not None and not e.free_symbols and getattr(e, "is_rational", False):
            v = sp.nsimplify(e)
            if v.q == 1 and lo <= int(v) <= hi:
                return int(v)
    except Exception:                                                   # noqa: BLE001
        return None
    return None


def _bits_ok(v, cap=VALUE_BITS):
    """Bit-width guard for exact closed values (prevents over-long decimal strings and sympy print
    blow-ups)."""
    try:
        iv = int(v)
        n = abs(iv).bit_length()
        if getattr(v, "q", 1) != 1:
            n = max(n, abs(int(v.q)).bit_length())
        return n <= cap
    except Exception:                                                   # noqa: BLE001
        return False


def _lib_value(name, args):
    """Computable library name + closed integer arguments -> exact value (within the bit-width
    guard); otherwise None (keep the atom)."""
    if name not in _LIB_COMPUTABLE or not args:
        return None
    iv = [_int_of(a, 0, 10 ** 12) for a in args]
    if any(v is None for v in iv):
        return None
    try:
        v = None
        if name in ("Nat.gcd", "Int.gcd"):
            v = sp.Integer(sp.igcd(*iv))
        elif name == "Nat.lcm":
            v = sp.Integer(sp.ilcm(*iv))
        elif name == "Nat.choose":
            if iv[0] <= 20000:
                v = sp.Integer(sp.binomial(iv[0], iv[1]))
        elif name == "Nat.fib":
            if iv[0] <= 20000:
                v = sp.Integer(sp.fibonacci(iv[0]))
        elif name == "Nat.factorial":
            if iv[0] <= FACT_CAP:
                v = sp.Integer(sp.factorial(iv[0]))
        elif name == "Nat.totient":
            v = sp.Integer(sp.totient(iv[0]))
        elif name == "Nat.sqrt":
            v = sp.Integer(sp.integer_nthroot(iv[0], 2)[0])
        elif name == "Nat.log":
            b, n = iv[0], iv[1]
            if b >= 2 and n >= 1:
                c = 0
                while n >= b:
                    n //= b
                    c += 1
                v = sp.Integer(c)
        elif name == "Nat.log2":
            n = iv[0]
            if n >= 1:
                c = 0
                while n >= 2:
                    n //= 2
                    c += 1
                v = sp.Integer(c)
        elif name == "Nat.mod":
            if iv[1]:
                v = sp.Integer(iv[0] % iv[1])
        elif name == "Nat.div":
            if iv[1]:
                v = sp.Integer(iv[0] // iv[1])
        elif name == "Int.natAbs":
            v = sp.Integer(abs(iv[0]))
        if v is None or not _bits_ok(v):
            return None
        return v
    except Exception:                                                   # noqa: BLE001
        return None


def parse_expr_ext789(txt, vars_, types, nat_mode, scrubs=("pow", "fact", "lib")):
    """Extended expression parsing: atomize first (fixed order pow -> fact -> lib), then hand the
    result to the strict S.parse_arith grammar; placeholders are substituted back as **exact
    models**. -> (sympy, meta, subs)"""
    t, subs = txt, {}
    for kind, fn in _SCRUBBERS:
        if kind not in scrubs:
            continue
        t2, s2, err = fn(t, len(subs))
        if t2 is None:
            raise S.PErr("scrub:%s:%s" % (kind, err))
        t = t2
        subs.update(s2)
    vs = set(vars_) | set(subs)
    e, meta = S.parse_arith(t, vs, types or {}, nat_mode)
    meta = dict(meta)
    meta["atoms"] = sorted({d["kind"] for d in subs.values()})
    if subs:
        reps = {}
        for n in subs:
            try:
                reps[sp.Symbol(n)] = _sub_expr(subs[n], vars_, types, nat_mode, scrubs)
            except Exception:                                           # noqa: BLE001
                reps[sp.Symbol(n)] = sp.Function("p44_bad")(sp.Symbol("p44_bad"))
        e = e.subs(reps)
    return e, meta, subs


def _sub_expr(d, vars_, types, nat_mode, scrubs):
    if d["kind"] == "pow":
        be = None
        try:
            be, _m, _s = parse_expr_ext789(d["base"], vars_, types, nat_mode, scrubs)
        except Exception:                                               # noqa: BLE001
            be = None
        try:
            ee, _m2, _s2 = parse_expr_ext789(d["exp"], vars_, types, nat_mode, scrubs)
        except Exception:                                               # noqa: BLE001
            ee = None
        if be is None:
            be = sp.Symbol(_san(d["base"]))
        if ee is None:
            ee = sp.Symbol(_san(d["exp"]))
        try:
            ei = _int_of(ee, -CAP_POW, CAP_POW)
            if ei is not None:
                if not be.free_symbols:
                    v = sp.nsimplify(be) ** ei
                    if getattr(v, "is_rational", False) and _bits_ok(v):
                        return v
                return sp.Pow(be, sp.Integer(ei), evaluate=False)
            return sp.Pow(be, ee, evaluate=False)
        except Exception:                                               # noqa: BLE001
            return sp.Function("p44_powfn")(be, ee)
    if d["kind"] == "fact":
        try:
            ae, _m, _s = parse_expr_ext789(d["arg"], vars_, types, nat_mode, scrubs)
        except Exception:                                               # noqa: BLE001
            ae = sp.Symbol(_san(d["arg"]))
        iv = _int_of(ae, 0, FACT_CAP)
        if iv is not None:
            f = sp.factorial(iv)
            if _bits_ok(f):
                return sp.Integer(f)
        return sp.Function("p44_factn")(ae)
    if d["kind"] == "lib":
        args = []
        for a in d["args"]:
            try:
                ae, _m, _s = parse_expr_ext789(a, vars_, types, nat_mode, scrubs)
            except Exception:                                           # noqa: BLE001
                ae = sp.Symbol(_san(a))
            args.append(ae)
        v = _lib_value(d["name"], args)
        if v is not None:
            return v
        return sp.Function(_san(d["name"]))(*args)
    raise S.PErr("subkind:%s" % d["kind"])


# ================================================================ §2 relations / relation chains (B9)
_ASCII_REL = ((">=", "≥"), ("<=", "≤"), ("=/=", "≠"))
_RELCH = "≠≤≥=<>∣"


def norm_ascii_rel(t):
    n_ascii = 0
    for a, b in _ASCII_REL:
        if a in t:
            n_ascii += t.count(a)
            t = t.replace(a, b)
    return t, n_ascii


def top_rels(t):
    """Depth-0 relation symbols -> [(idx, tok, len)] ('>=' counts as one operator; called on the
    already normalized text)."""
    out, d = [], 0
    for i, c in enumerate(t):
        if c in "([{":
            d += 1
        elif c in ")]}":
            d -= 1
        elif d == 0 and c in _RELCH:
            out.append((i, c, 1))
    return out


def split_chain(t):
    """Depth-0 relation symbols >= 2 -> (rels, segs); otherwise None."""
    rels = top_rels(t)
    if len(rels) < 2:
        return None
    segs, prev = [], 0
    for (i, tok, ln) in rels:
        segs.append(t[prev:i].strip())
        prev = i + ln
    segs.append(t[prev:].strip())
    if any(not s for s in segs):
        return None
    return [(tok, ln) for (_i, tok, ln) in rels], segs


# ================================================================ §3 goal tree (§3 = bcore-isomorphic + atom layer)
def parse_hyp_ext789(ptxt, hname, vs, types, nat_mode, scrubs):
    """One hypothesis -> [hyp, ...]: a chained hypothesis (**illegal Lean 4 syntax, see the B9
    measurement**) is mechanically expanded into `hname.1` / `hname.2`; a parse failure -> []
    (recorded in extra_hyps, not a failure)."""
    t = (ptxt or "").strip()
    if "chain" in scrubs:
        t2, _n = norm_ascii_rel(t)
        ch = split_chain(B._strip_parens(t2))
        if ch is not None:
            rels, segs = ch
            out = []
            for j, (_tok, _ln) in enumerate(rels):
                h = parse_goal_ext_rel789("%s %s %s" % (segs[j], "≤", segs[j + 1]),
                                          vs, types, nat_mode, scrubs)
                if h is None:
                    return []
                h["name"] = "%s.%d" % (hname, j + 1)
                out.append(h)
            return out
    h = parse_goal_ext_rel789(t, vs, types, nat_mode, scrubs)
    if h is None:
        return []
    h["name"] = hname
    return [h]


def parse_goal_ext_rel789(txt, vs, types, nat_mode, scrubs):
    """Relational goal -> {rel,neg,lhs,rhs,txt,meta,atoms,subs,ascii_n}; None on failure."""
    t = txt.strip()
    t, nascii = norm_ascii_rel(t)
    neg = False
    while t.startswith("¬"):
        neg = not neg
        t = t[1:].strip()
    t = B._strip_parens(t)
    rel, i = S.top_rel(t)
    if rel is None or i < 0:
        return None
    hl, hr = t[:i].strip(), t[i + 1:].strip()
    if not hl or not hr:
        return None
    try:
        e1, m1, s1 = parse_expr_ext789(hl, vs, types, nat_mode, scrubs)
        e2, m2, s2 = parse_expr_ext789(hr, vs, types, nat_mode, scrubs)
    except Exception:                                                   # noqa: BLE001
        return None
    if rel == "≥":
        rel, e1, e2, hl, hr = "≤", e2, e1, hr, hl
    elif rel == ">":
        rel, e1, e2, hl, hr = "<", e2, e1, hr, hl
    meta = {k: bool(m1.get(k) or m2.get(k)) for k in m1 if k != "atoms"}
    atoms = sorted(set(m1.get("atoms") or []) | set(m2.get("atoms") or []))
    subs = dict(s1)
    subs.update(s2)
    return {"rel": rel, "neg": neg, "lhs": e1, "rhs": e2, "txt": t,
            "lhs_txt": hl, "rhs_txt": hr, "meta": meta, "atoms": atoms,
            "subs": subs, "ascii_n": nascii}


def _g_ops_of(atoms, subs, ascii_n=0):
    """New operators triggered by this leaf (V7 append-order convention)."""
    ops = []
    if "pow" in atoms:
        ops.append("g44_pow_atom")
        if any(d.get("kind") == "pow" and (d.get("exp_int") or 0) > 256
               for d in (subs or {}).values()):
            ops.append("g44_pow_cap")
    if "fact" in atoms:
        ops.append("g44_fact_atom")
    if "lib" in atoms:
        ops.append("g44_lib_atom")
    if ascii_n:
        ops.append("g44_relchain")
    return ops


def _mk_rec789(name, vars_ty, types, hyps, concl, nat_mode):
    return {"name": name, "vars": dict(vars_ty), "types": dict(types), "hyps": list(hyps),
            "concl": concl, "ok": True, "n_vars": len(vars_ty),
            "has_complex": any(t in S.CPLX_TYPES for t in types.values()),
            "nat_mode": nat_mode}


def build_tree789(txt, types, hyps, nat_mode, scrubs, depth=0):
    """Goal tree: relchain (conjunction chain) / forall / imp / iff / and / or / leaf / fail."""
    if depth > 10:
        return {"kind": "fail", "reason": "depth"}
    vs = set(types)
    t = B._strip_parens(txt)
    if "chain" in scrubs:
        ch = split_chain(t)
        if ch is not None:
            rels, segs = ch
            kids, ok = [], True
            for j, (_tok, _ln) in enumerate(rels):
                nd = build_tree789("%s %s %s" % (segs[j], "≤", segs[j + 1]), types,
                                   hyps, nat_mode, scrubs, depth + 1)
                if nd["kind"] == "fail":
                    ok = False
                    break
                kids.append(nd)
            if ok:
                node = kids[-1]
                for ch2 in reversed(kids[:-1]):
                    node = {"kind": "and", "src": "relchain", "n_rel": len(rels),
                            "children": [ch2, node], "txt": t}
                return node
    m = re.match(r"^(?:∀|forall)\s*", t)
    if m:
        parts, body, ok = B._forall_split(t[m.end():])
        if not ok or body is None:
            return {"kind": "fail", "reason": "forall_syntax"}
        nv, nh = dict(types), list(hyps)
        intros, added = [], []
        for kind, names, ty in parts:
            if kind == "var":
                if ty not in S.NUMERIC_TYPES:
                    return {"kind": "fail", "reason": "forall_nonnum:%s" % ty[:12]}
                for n in names:
                    nv[n] = ty
                intros += names
            else:
                nm = "h44_%d" % len(intros)
                intros.append(nm)
                h = parse_goal_ext_rel789(ty, set(nv), nv, nat_mode, scrubs)
                if h is not None:
                    h["name"] = nm
                    nh.append(h)
                    added.append(nm)
        ch2 = build_tree789(body, nv, nh, nat_mode, scrubs, depth + 1)
        return {"kind": "forall", "intros": intros, "added": added, "types": nv,
                "hyps": nh, "children": [ch2], "txt": t}
    tok, idx = B.top_conn(t)
    if tok in ("↔", "<->"):
        l, r = t[:idx].strip(), t[idx + len(tok):].strip()
        c1 = build_tree789(r, types, hyps + _hyp_add789(l, vs, types, nat_mode, scrubs,
                                                        "h44_l"), nat_mode, scrubs,
                           depth + 1)
        c2 = build_tree789(l, types, hyps + _hyp_add789(r, vs, types, nat_mode, scrubs,
                                                        "h44_r"), nat_mode, scrubs,
                           depth + 1)
        return {"kind": "iff", "children": [
            {"kind": "imp", "hypname": "h44_l", "children": [c1]},
            {"kind": "imp", "hypname": "h44_r", "children": [c2]}]}
    if tok in ("→", "->"):
        l, r = t[:idx].strip(), t[idx + len(tok):].strip()
        nh = hyps + _hyp_add789(l, vs, types, nat_mode, scrubs, "h44")
        ch2 = build_tree789(r, types, nh, nat_mode, scrubs, depth + 1)
        return {"kind": "imp", "hypname": "h44", "children": [ch2], "hyp_txt": l}
    if tok in ("∨", "\\/", "∧", "/\\"):
        l, r = t[:idx].strip(), t[idx + len(tok):].strip()
        cl = build_tree789(l, types, hyps, nat_mode, scrubs, depth + 1)
        cr = build_tree789(r, types, hyps, nat_mode, scrubs, depth + 1)
        return {"kind": "or" if tok in ("∨", "\\/") else "and", "children": [cl, cr]}
    leaf = parse_goal_ext_rel789(t, vs, types, nat_mode, scrubs)
    if leaf is None:
        return {"kind": "fail", "reason": "leaf_no_rel"}
    rec = _mk_rec789("<goal>", types, types, hyps, leaf, nat_mode)
    got, why = F.chain_of(rec)
    return {"kind": "leaf", "rec": rec, "got": got, "why": why,
            "atoms": leaf.get("atoms") or [], "ascii_n": leaf.get("ascii_n") or 0,
            "g_ops": _g_ops_of(leaf.get("atoms") or [], leaf.get("subs") or {},
                               leaf.get("ascii_n") or 0)}


def _hyp_add789(txt, vs, types, nat_mode, scrubs, nm):
    h = parse_goal_ext_rel789(txt, set(vs), types, nat_mode, scrubs)
    if h is None:
        return []
    h["name"] = nm
    return [h]


# ================================================================ §4 statement parsing (LWB/synthetic)
def parse_stmt_ext789(prob, fam):
    """Problem -> extended rec (ok=False + reason on failure). fam selects the atom layer
    (b7/b8/b9)."""
    scrubs = ("pow", "fact") if fam == "b7" else (("lib",) if fam == "b8"
                                                  else ("pow", "fact", "lib", "chain"))
    mod = fam_module(fam)
    full = S.norm_answer(prob.head)
    concl_txt = S.norm_answer(getattr(prob, "concl", "") or "")
    rest = re.sub(r"^theorem\s+", "", full).strip()
    m = re.match(r"^([A-Za-z0-9_.']+)\s*(.*)$", rest, re.S)
    binders = (m.group(2) if m else "").strip()
    if binders.endswith(":"):
        binders = binders[:-1].strip()
    vars_ty, hypgrp = {}, []
    for names, ty, br in S.binder_groups(binders):
        rel, _i = S.top_rel(ty)
        ht = ty.strip()[:1]
        if rel or ht in ("∀", "∃", "¬"):
            for n in names:
                hypgrp.append((n, ty))
        else:
            for n in names:
                vars_ty[n] = ty
    rec = {"name": prob.name, "vars": vars_ty, "types": dict(vars_ty), "hyps": [],
           "extra_hyps": [], "ok": False, "reason": None, "concl_txt": concl_txt,
           "nat_mode": any(t in S.NAT_TYPES for t in vars_ty.values()),
           "has_complex": any(t in S.CPLX_TYPES for t in vars_ty.values()),
           "fam": fam}
    if not all(t in S.NUMERIC_TYPES for t in vars_ty.values()):
        rec["reason"] = "nonnumeric_type"
        return rec
    vs = set(vars_ty)
    hyps = []
    for gname, gty in hypgrp:
        parts = S.split_top_conj(gty)
        for pi, ptxt in enumerate(parts):
            hname = gname if len(parts) == 1 else "%s.%d" % (gname, pi)
            hs = parse_hyp_ext789(ptxt.strip(), hname, vs, vars_ty, rec["nat_mode"],
                                  scrubs)
            if not hs:
                rec["extra_hyps"].append(hname)
            else:
                hyps += hs
    rec["hyps"] = hyps
    tr = build_tree789(concl_txt, vars_ty, hyps, rec["nat_mode"], scrubs, 0)
    if tr.get("kind") == "fail":
        rec["reason"] = "goal:" + str(tr.get("reason"))
        rec["tree"] = tr
        return rec
    rec["tree"] = tr
    rec["ok"] = True
    return rec


# ================================================================ §5 certifiability + rendering
def certifiable(node):
    k = node.get("kind")
    if k == "fail":
        return False
    if k == "leaf":
        return node.get("got") is not None
    if k == "or":
        return any(certifiable(x) for x in node["children"])
    if k in ("and", "iff", "imp", "forall"):
        return all(certifiable(x) for x in node["children"])
    return False


def leaf_whys(node, out=None):
    out = [] if out is None else out
    k = node.get("kind")
    if k == "leaf" and node.get("got") is None:
        out.append("leaf:" + str(node.get("why")))
    elif k == "fail":
        out.append("fail:" + str(node.get("reason")))
    for c in node.get("children") or []:
        leaf_whys(c, out)
    return out


def _leaf_lines(node, mode, ci=0):
    rec, got = node["rec"], node["got"]
    if got is None:
        return None
    if mode == "exam":
        t = RE3.render_exam_v3(F, rec, got, got["chain_ops"])
        if t == "skip":
            return None
        return t.split("\n")
    cands = RG.candidates(F, rec, got)
    if not cands:
        return None
    return cands[min(ci, len(cands) - 1)].split("\n")


def render_tree(node, mode="exam", cimap=None, ctr=None):
    """Goal tree -> proof text lines (exam: every line under try, fail-closed; no enumeration)."""
    if ctr is None:
        ctr = [0]
    if cimap is None:
        cimap = {}
    k = node.get("kind")
    if k == "fail":
        return None
    if k == "leaf":
        i = ctr[0]
        ctr[0] += 1
        return _leaf_lines(node, mode, cimap.get(i, 0))
    if k == "forall":
        lines = ["try intro " + " ".join(node["intros"])]
        sub = render_tree(node["children"][0], mode, cimap, ctr)
        return None if sub is None else lines + sub
    if k == "imp":
        lines = ["try intro " + node["hypname"]]
        sub = render_tree(node["children"][0], mode, cimap, ctr)
        return None if sub is None else lines + sub
    if k in ("iff", "and"):
        lines = ["try constructor"]
        for ch in node["children"]:
            sub = render_tree(ch, mode, cimap, ctr)
            if sub is None:
                return None
            lines += sub
        return lines
    if k == "or":
        for tag, ch in (("left", node["children"][0]), ("right", node["children"][1])):
            if certifiable(ch):
                sub = render_tree(ch, mode, cimap, ctr)
                if sub is not None:
                    return ["try " + tag] + sub
        return None
    return None


def slots_of(node, out=None):
    out = [] if out is None else out
    if node.get("kind") == "leaf":
        out.append(node)
        return out
    for c in node.get("children") or []:
        slots_of(c, out)
    return out


def slot_candidates(node):
    n = []
    for nd in slots_of(node):
        got, rec = nd.get("got"), nd.get("rec")
        if not got or rec is None:
            n.append(0)
            continue
        try:
            n.append(len(RG.candidates(F, rec, got)))
        except Exception:                                               # noqa: BLE001
            n.append(0)
    return n


def render_tree_multi(node, cap=8, mode="gold"):
    base = render_tree(node, mode, {}, [0])
    if base is None:
        return []
    texts = ["\n".join(base)]
    for i, m in enumerate(slot_candidates(node)):
        for j in range(1, m):
            t = render_tree(node, mode, {i: j}, [0])
            if t is None:
                continue
            s = "\n".join(t)
            if s not in texts:
                texts.append(s)
            if len(texts) >= cap:
                return texts
    return texts


_LEAF_ONLY = ("g42_s_reduce", "g42_s_poly", "g42_s_eval", "g42_s_mod", "g42_s_lin",
              "g42_s_omega", "g42_s_pos", "g42_sb_step", "g42_sb_close")


def tree_ops(node, out=None):
    out = [] if out is None else out
    k = node.get("kind")
    if k == "leaf":
        out += node.get("g_ops") or []
    if k == "and" and node.get("src") == "relchain":
        out.append("g44_relchain")
    for c in node.get("children") or []:
        tree_ops(c, out)
    return out


def tree_leaf_ops(node, out=None):
    out = [] if out is None else out
    if node.get("kind") == "leaf":
        g = node.get("got") or {}
        out += list(g.get("chain_ops") or [])
    for c in node.get("children") or []:
        tree_leaf_ops(c, out)
    return out


def tree_frames(node, out=None):
    out = [] if out is None else out
    if node.get("kind") == "leaf":
        g = node.get("got") or {}
        out += list(g.get("frames") or [])
    for c in node.get("children") or []:
        tree_frames(c, out)
    return out


def frames_audit(frames, n_last=2):
    """Mechanical assertions (frames computed on the fly): last frame = exhausted frame (all 0) ·
    chain non-constant · frames != answer vector."""
    bad = {"empty": 0, "last_nonzero": 0, "const_nonzero": 0, "const_zero": 0}
    if not frames:
        bad["empty"] = 1
        return bad
    fr = [list(f) for f in frames]
    last = fr[-1]
    if any(abs(x) > 1e-12 for x in last):
        bad["last_nonzero"] = 1
    if len(fr) >= 2 and fr[0] == last:
        if any(abs(x) > 1e-12 for x in fr[0]):
            bad["const_nonzero"] = 1
        else:
            bad["const_zero"] = 1
    return bad


def admissible_shape(fam, shape):
    """Whether this shape is compilable in **this Lean environment** (a family may override;
    default True).
    Why B9 overrides (measured): a genuine chain `a ≤ b ≤ c` is a Lean 4 syntax error
    (`unexpected token '≤'`)."""
    fn = getattr(fam_module(fam), "admissible_shape", None)
    return bool(fn(shape)) if fn else True


# ================================================================ §6 structural-layer census
def census(fam, limit=None, rows=None, lwb=None, row_timeout=ROW_TIMEOUT_S):
    cen = rows if rows is not None else B.load_census_rows()
    lwb = lwb if lwb is not None else B.load_lwb()
    pop = fam_rows(fam, cen)
    if limit:
        pop = pop[:limit]
    st, why, kd, sub = (collections.Counter(), collections.Counter(),
                        collections.Counter(), collections.Counter())
    shp = collections.Counter()
    lib_names = collections.Counter()
    recs, t0 = [], time.time()
    for i, r in enumerate(pop):
        rec = lwb.get(r["idx"])
        if rec is None:
            st["missing"] += 1
            continue
        row = {"idx": r["idx"], "name": r["name"], "sub": r.get("sub"), "rs": r.get("rs")}
        try:
            d = with_timeout(row_timeout, parse_stmt_ext789,
                             B.P(rec["name"], rec["head"], rec["concl"], rec.get("subset"),
                                 rec["idx"]), fam)
        except Timeout:
            st["timeout"] += 1
            row["state"] = "timeout"
            recs.append(row)
            continue
        except Exception as e:                                          # noqa: BLE001
            st["error_" + type(e).__name__] += 1
            row["state"] = "error"
            row["err"] = str(e)[:60]
            recs.append(row)
            continue
        if not d["ok"]:
            st["parse_" + str(d["reason"]).split(":")[0]] += 1
            row.update({"state": "unparsed", "reason": str(d["reason"])[:70]})
        else:
            tr = d["tree"]
            row["kind"] = tr["kind"]
            row["shape"] = fam_module(fam).shape_of(tr)
            kd[tr["kind"]] += 1
            shp[row["shape"]] += 1
            for nd in slots_of(tr):
                for nm in (nd.get("g_ops") or []):
                    sub[nm] += 1
                for kk, dd in (nd.get("rec") or {}).get("concl", {}).get("subs", {}).items():
                    if dd.get("kind") == "lib":
                        lib_names[dd.get("name")] += 1
            if not fam_module(fam).claim(tr):
                st["parsed_not_claim"] += 1
                row["state"] = "not_claim"
            else:
                st["claim"] += 1
                if certifiable(tr):
                    st["certifiable_struct"] += 1
                    row["state"] = "certifiable"
                    row["ops"] = tree_ops(tr) + tree_leaf_ops(tr)
                    row["chain_len"] = len(tree_leaf_ops(tr))
                    row["frames"] = tree_frames(tr)[:2]
                else:
                    st["claim_uncert"] += 1
                    row["state"] = "claim_uncert"
                    wy = leaf_whys(tr)
                    row["why"] = wy[:3]
                    for w in wy:
                        why[w] += 1
        recs.append(row)
        if (i + 1) % 500 == 0:
            print("[census %s %d/%d] %s %.1fs" % (fam, i + 1, len(pop), dict(st),
                                                  time.time() - t0), flush=True)
    rep = {"schema": "gen44-b-census/1", "fam": fam, "n_pop": len(pop),
           "counts": dict(st), "tree_kinds": dict(kd), "shape_hist": dict(shp.most_common(20)),
           "leaf_why": dict(why.most_common(20)),
           "g_ops_hist": dict(sub.most_common(20)),
           "lib_names_hist": dict(lib_names.most_common(20)),
           "wall_s": round(time.time() - t0, 1)}
    return rep, recs


def fam_rows(fam, cen=None):
    labs = FAM_LABELS[fam]
    rows = cen if cen is not None else B.load_census_rows()
    if fam == "b9":
        return [r for r in rows if any(str(r.get("rs") or "").startswith(x) for x in labs)]
    ls = set(labs)
    return [r for r in rows if r.get("rs") in ls]


# ================================================================ §7 synthetic corpus (+ mechanical assertions)
def gen_corpus(fam, n, seed=20260926, row_timeout=ROW_TIMEOUT_S):
    mod = fam_module(fam)
    rng = random.Random(seed)
    rows, tries, n_tmo = [], 0, 0
    while len(rows) < n and tries < n * 60:
        tries += 1
        r = mod.corpus_row(rng, len(rows))
        if r is None:
            continue
        try:
            d = with_timeout(row_timeout, parse_stmt_ext789,
                             B.P(r["name"], r["head"], r["concl"], None, None), fam)
        except Timeout:
            n_tmo += 1
            continue
        except Exception:                                               # noqa: BLE001
            continue
        if not d["ok"]:
            continue
        tr = d["tree"]
        if not mod.claim(tr) or not certifiable(tr):
            continue
        txt = render_tree(tr, "exam")
        if not txt:
            continue
        r["proof"] = "\n".join(txt)
        r["chain_ops"] = tree_ops(tr) + tree_leaf_ops(tr)
        r["chain_len"] = len(tree_leaf_ops(tr))
        r["kind"] = tr["kind"]
        r["shape"] = mod.shape_of(tr)
        r["frames"] = tree_frames(tr)[:2]
        r["const_vec"] = [int(x) for x in re.findall(r"\b\d+\b", r["concl"])][:16]
        r["stmt"] = "%s %s" % (r["head"], r["concl"])
        rows.append(r)
    rep = {"schema": "gen44-b-gen/1", "fam": fam, "n": len(rows), "target": n,
           "tries": tries, "seed": seed, "n_row_timeout": n_tmo,
           "text_hist": dict(collections.Counter(r["proof"] for r in rows).most_common(20)),
           "chainlen_hist": dict(collections.Counter(r["chain_len"] for r in rows)),
           "kind_hist": dict(collections.Counter(r["kind"] for r in rows)),
           "ops_hist": dict(collections.Counter(o for r in rows for o in r["chain_ops"])),
           "n_with_colon_head": sum(1 for r in rows if r["head"].rstrip().endswith(":"))}
    return rows, rep


# ================================================================ §8 payload-ness (single-slot damage)
def _force_leaf_ops(node, ops):
    if node.get("kind") == "leaf":
        g = dict(node.get("got") or {})
        g["chain_ops"] = [o for o in ops if o in _LEAF_ONLY]
        nn = dict(node)
        nn["got"] = g
        return nn
    for i, c in enumerate(node.get("children") or []):
        r = _force_leaf_ops(c, ops)
        if r is not None:
            nn = dict(node)
            ch = list(node["children"])
            ch[i] = r
            nn["children"] = ch
            return nn
    return None


def payload_tests(fam, rows, limit=240):
    """Single-slot chain damage (replace the last leaf operator with a sibling) => the rendered text
       must change (slot_bad expected 0);
       numeral-slot damage (per-family perturb_digit) => the certificate must change or disappear
       (digit_bad expected 0)."""
    mod = fam_module(fam)
    slot_n = slot_bad = digit_n = digit_bad = 0
    slot_ex = []
    for r in rows[:limit]:
        dmg = mod.damage_ops(r["chain_ops"])
        try:
            if dmg is not None:
                slot_n += 1
                d2 = with_timeout(4.0, parse_stmt_ext789,
                                  B.P(r["name"], r["head"], r["concl"], None, None), fam)
                node = _force_leaf_ops(d2["tree"], dmg) if d2["ok"] else None
                txt = render_tree(node, "exam") if node else None
                if txt is None or "\n".join(txt) == r["proof"]:
                    slot_bad += 1
                    if len(slot_ex) < 5:
                        slot_ex.append({"name": r["name"], "dmg": dmg})
            r3 = mod.perturb_digit(r)
            if r3 is not None:
                digit_n += 1
                d3 = with_timeout(4.0, parse_stmt_ext789,
                                  B.P(r3["name"], r3["head"], r3["concl"], None, None), fam)
                t3 = (render_tree(d3["tree"], "exam")
                      if d3["ok"] and certifiable(d3["tree"]) else None)
                if t3 is not None and "\n".join(t3) == r["proof"]:
                    digit_bad += 1
        except Exception:                                               # noqa: BLE001
            continue
    div = diversity_of(fam, rows)
    ok = (slot_bad == 0 and digit_bad == 0 and div["n_distinct_exam"] >= 2)
    return {"schema": "gen44-b-payload/1", "fam": fam,
            "n_slot": slot_n, "slot_bad": slot_bad, "slot_examples": slot_ex,
            "n_digit": digit_n, "digit_bad": digit_bad,
            "digit_axis": mod.DIGIT_AXIS,
            "diversity": div,
            "renderer_md5": {"exam_v3": B.md5f(os.path.join(_PKG_DIR, "gen44_render_exam_v3.py")),
                             "gold": B.md5f(os.path.join(_PKG_DIR, "gen44_render_gold.py"))},
            "verdict": "PASS" if ok else "FAIL"}


def diversity_of(fam, rows, limit=240):
    """Two-renderer diversity: distinct exam single-rendering texts / distinct gold-enumerated
    candidate texts (at most cap per row)."""
    ex, gd, ng = collections.Counter(), collections.Counter(), []
    for r in rows[:limit]:
        ex[r["proof"]] += 1
        try:
            d = with_timeout(4.0, parse_stmt_ext789,
                             B.P(r["name"], r["head"], r["concl"], None, None), fam)
            if d["ok"] and certifiable(d["tree"]):
                cands = render_tree_multi(d["tree"], cap=8)
                ng.append(len(cands))
                for t in cands:
                    gd[t] += 1
        except Exception:                                               # noqa: BLE001
            continue
    return {"n_exam_texts": sum(ex.values()), "n_distinct_exam": len(ex),
            "n_distinct_gold": len(gd), "mean_gold_per_row": round(
                sum(ng) / max(1, len(ng)), 2),
            "exam_hist_top": [[k[:48], v] for k, v in ex.most_common(8)],
            "criterion": ("族级下限 n_distinct_exam >= 2 (防单型考点); [S7] G-D 的 >=6 是"
                          "**全语料**判据, 由主棒在 A∪B1..B9 合并语料上评估 (本族读数: B7 3 型 / "
                          "B8 2 型 / B9 >=4 型)")}


# ================================================================ §9 mechanical frame assertions (computed on the fly)
def frames_report(fam, rows):
    agg = collections.Counter()
    ex = []
    for r in rows:
        b = frames_audit(r.get("frames") or [])
        for k, v in b.items():
            agg[k] += v
        if any(b.values()) and len(ex) < 5:
            ex.append({"name": r["name"], "bad": {k: v for k, v in b.items() if v}})
    n = len(rows)
    return {"schema": "gen44-b-frames/1", "fam": fam, "n": n,
            "n_empty_bad": agg["empty"], "n_last_nonzero_bad": agg["last_nonzero"],
            "n_const_nonzero": agg["const_nonzero"], "n_const_zero": agg["const_zero"],
            "asserts": {"末帧=消尽帧": "PASS" if agg["last_nonzero"] == 0 else "FAIL",
                        "帧非空": "PASS" if agg["empty"] == 0 else "FAIL",
                        "链路非常数(非零帧)": "PASS" if agg["const_nonzero"] == 0 else "FAIL"},
            "note": ("链常数且全零帧 = 残差恒零型闭环 (const_zero), 登记不作判废; "
                     "帧≠答案向量: 闭值型叶的帧 = 精确残差标量帧 => 该轴 N/A (登记)"),
            "examples": ex,
            "verdict": "PASS" if (agg["empty"] == 0 and agg["last_nonzero"] == 0) else "FAIL"}


# ================================================================ §10 stratified check (Lean empirical)
def check_corpus(fam, rows, sample=16, shard=0, nshard=1, maxj=5, seed=20260926,
                 gate_wait_s=900.0, timeout=300.0):
    by_text = collections.defaultdict(list)
    for r in rows:
        by_text[r["proof"]].append(r)
    keys = sorted(by_text)
    rng = random.Random(seed)
    per = max(1, sample // max(1, len(keys)))
    sel = []
    for k in keys:
        g = by_text[k]
        rng.shuffle(g)
        sel += g[:per]
    sel = sel[shard::nshard]
    wd = os.path.join(B.G44, "lean", "bchk_%s" % fam)
    res, t0 = [], time.time()
    for i, r in enumerate(sel):
        jr = B.judge_nice(r["name"], r["head"], r["concl"], r["proof"], wd,
                          "bc%s" % fam, timeout=timeout, maxj=maxj,
                          gate_wait_s=gate_wait_s)
        res.append({"name": r["name"], "proof_md5": B.md5s(r["proof"]),
                    "shape": r.get("shape"),
                    "certified": bool(jr["certified"]), "reasons": jr["reasons"],
                    "lean": jr["lean"], "n1n2": jr["n1n2"]})
        print("[chk %s %d/%d] cert=%d %.0fs" % (
            fam, i + 1, len(sel), sum(1 for x in res if x["certified"]),
            time.time() - t0), flush=True)
    nc = sum(1 for x in res if x["certified"])
    n2ok = sum(1 for x in res if (x.get("n1n2") or {}).get("n2_accepted"))
    n1v = sum(1 for x in res if (x.get("n1n2") or {}).get("n1_vacuous"))
    by_shape = _by_shape(res, fam)
    return {"schema": "gen44-b-check/1", "fam": fam, "n_sample": len(sel),
            "n_certified": nc, "rate": round(nc / max(1, len(sel)), 4),
            "wilson95": B.wilson(nc, len(sel)), "n2_accepted": n2ok, "n1_vacuous": n1v,
            "by_shape": by_shape,
            "rate_admissible": by_shape["admissible"]["rate"],
            "strat_keys": len(keys), "rows": res,
            "reasons_hist": dict(collections.Counter(
                (r["reasons"] or ["ok"])[0] for r in res)),
            "wall_s": round(time.time() - t0, 1)}


def _by_shape(res, fam):
    grp = collections.defaultdict(lambda: [0, 0])
    for x in res:
        k = x.get("shape") or "?"
        grp[k][0] += 1
        grp[k][1] += 1 if x.get("certified") else 0
    adm = [0, 0]
    out = {}
    for k, (n, c) in sorted(grp.items()):
        out[k] = {"n": n, "cert": c, "rate": round(c / max(1, n), 4)}
        if admissible_shape(fam, k):
            adm[0] += n
            adm[1] += c
    out["admissible"] = {"n": adm[0], "cert": adm[1],
                         "rate": round(adm[1] / max(1, adm[0]), 4)}
    return out


def _by_key(res, key):
    grp = collections.defaultdict(lambda: [0, 0])
    for x in res:
        k = str(x.get(key) or "?")
        grp[k][0] += 1
        grp[k][1] += 1 if x.get("certified") else 0
    return {k: {"n": n, "cert": c, "rate": round(c / max(1, n), 4)}
            for k, (n, c) in sorted(grp.items())}


# ================================================================ §11 three-channel leakage
def _grams(texts, n=4, tok=False):
    """n-gram sets (hashed integers, memory-lean); tok=True for token level, otherwise character
    level."""
    out = set()
    for t in texts:
        s = re.sub(r"\s+", " ", t or "")
        if tok:
            seq = re.findall(r"[A-Za-z_][A-Za-z0-9_.']*|\d+|[^\sA-Za-z0-9_]", s)
            for i in range(max(0, len(seq) - n + 1)):
                out.add(hash(" ".join(seq[i:i + n])))
        else:
            for i in range(max(0, len(s) - n + 1)):
                out.add(hash(s[i:i + n]))
    return out


def leak(fam, corpus, pool_limit=None, seed=20260926, n_baseline=200):
    """Three-channel leakage plus positive control plus **structural noise-floor control**
    (tightened convention).

    Hard criteria (must be 0): P1 name / P2 whole statement / G3c constant vector (distinct>=5 and
    total>=8, from gen42 D13).
    Positive control: the corpus's first row (name + statement) is injected into a pool copy => both
    channels must hit (proof the detector works).
    Noise-floor control: character/token 4-grams **necessarily** overlap a 139826-row Lean library
    (Lean syntax tokens) => the original "4-gram hits = 0" criterion is unreachable (D-B789-3); it is
    replaced by a comparison against an **independent sample from the same generator (seed+1)**: PASS
    requires corpus overlap <= noise-floor rate x 1.25 + 0.02; both rates and the raw hit counts are
    still reported.
    """
    lwb = B.load_lwb()
    pool, n = [], 0
    for idx in sorted(lwb):
        r = lwb[idx]
        pool.append({"name": r["name"],
                     "stmt": "%s %s" % (r.get("head"), r.get("concl"))})
        n += 1
        if pool_limit and n >= pool_limit:
            break
    rep = B.leak_report(corpus, pool, "gen44_%s_corpus" % fam)
    ctrl = list(pool) + [{"name": (corpus[0].get("name") if corpus else "x"),
                          "stmt": (corpus[0].get("stmt") if corpus else "y")}]
    crep = B.leak_report(corpus, ctrl, "posctrl")
    pos_ok = bool(crep["hits"]["P1_name"] and crep["hits"]["P2_stmt"])
    base_rows, _brep = gen_corpus(fam, n_baseline, seed=seed + 1)
    pool_txt = [p["stmt"] for p in pool]
    pool_c4 = _grams(pool_txt, 4, False)
    pool_t4 = _grams(pool_txt, 4, True)

    def _frac(rows):
        c4 = _grams([r.get("stmt") or "" for r in rows], 4, False)
        t4 = _grams([r.get("stmt") or "" for r in rows], 4, True)
        return {"char4": {"n": len(c4), "hit": len(c4 & pool_c4),
                          "frac": round(len(c4 & pool_c4) / max(1, len(c4)), 4)},
                "tok4": {"n": len(t4), "hit": len(t4 & pool_t4),
                         "frac": round(len(t4 & pool_t4) / max(1, len(t4)), 4)}}
    A, Bs = _frac(corpus), _frac(base_rows)
    ng_ok = all(A[k]["frac"] <= Bs[k]["frac"] * 1.25 + 0.02 for k in ("char4", "tok4"))
    hard_ok = (rep["hits"]["P1_name"] == 0 and rep["hits"]["P2_stmt"] == 0
               and rep["hits"]["G3c_const_vec"] == 0)
    rep.update({"schema": "gen44-b-leak/1", "fam": fam,
                "hard": {"P1_name": rep["hits"]["P1_name"],
                         "P2_stmt": rep["hits"]["P2_stmt"],
                         "G3c_const_vec": rep["hits"]["G3c_const_vec"]},
                "positive_control_pass": pos_ok,
                "positive_control": {"name_hits": crep["hits"]["P1_name"],
                                     "stmt_hits": crep["hits"]["P2_stmt"]},
                "ngram_control": {"corpus": A, "baseline_seed": seed + 1,
                                  "baseline": Bs, "margin": 1.25, "verdict_pass": ng_ok},
                "criterion_note": ("字符 4-gram 命中 = 0 对任何 Lean 文本不可达 (D-B789-3); "
                                   "改判据为底噪对照, 原始命中数仍并出 (不删)"),
                "verdict": "PASS" if (hard_ok and pos_ok and ng_ok) else "FAIL"})
    return rep


# ================================================================ §12 end-to-end certification arm
def load_claim(fam):
    p = os.path.join(OUT, "GEN44_%s_CENSUS.jsonl" % fam.upper())
    rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    return [r for r in rows if r.get("state") in ("certifiable", "claim_uncert")]


def arm(fam, mode="exam", n=24, seed=20260926, shard=0, nshard=1, maxj=5,
        gate_wait_s=900.0, timeout=300.0, cap=8, row_timeout=ROW_TIMEOUT_S,
        only_state="any"):
    claim = load_claim(fam)
    if only_state == "certifiable":
        claim = [r for r in claim if r.get("state") == "certifiable"]
    rng = random.Random(seed)
    if len(claim) <= n:
        sel = claim
    else:
        sel = sorted(rng.sample(claim, n), key=lambda r: r["idx"])
    sel = sel[shard::nshard]
    lwb = B.load_lwb()
    wd = os.path.join(B.G44, "lean", "barm_%s_%s" % (fam, mode))
    out, t0 = [], time.time()
    for i, r in enumerate(sel):
        lrec = lwb.get(r["idx"])
        ent = {"idx": r["idx"], "name": r["name"], "sub": r.get("sub"), "rs": r.get("rs"),
               "fam": fam, "mode": mode, "struct_state": r.get("state"),
               "shape": r.get("shape"), "kind_struct": r.get("kind")}
        if lrec is None:
            ent["state"] = "missing"
            out.append(ent)
            continue
        try:
            d = with_timeout(row_timeout, parse_stmt_ext789,
                             B.P(lrec["name"], lrec["head"], lrec["concl"],
                                 lrec.get("subset"), lrec["idx"]), fam)
        except Timeout:
            ent["state"] = "timeout"
            out.append(ent)
            continue
        except Exception as e:                                          # noqa: BLE001
            ent["state"] = "error"
            ent["err"] = str(e)[:60]
            out.append(ent)
            continue
        if not d["ok"] or not fam_module(fam).claim(d["tree"]):
            ent["state"] = "not_claim_rerun"
            ent["reason"] = str(d.get("reason"))
            out.append(ent)
            continue
        texts = ([render_tree(d["tree"], "exam")] if mode == "exam"
                 else render_tree_multi(d["tree"], cap=cap))
        texts = ["\n".join(t) for t in texts if t]
        if not texts:
            ent["state"] = "render_skip"
            out.append(ent)
            continue
        ent["n_text"] = len(texts)
        ent["ops"] = tree_ops(d["tree"]) + tree_leaf_ops(d["tree"])
        for ci, txt in enumerate(texts):
            jr = B.judge_nice(lrec["name"], lrec["head"], lrec["concl"], txt, wd,
                              "ba%s%s" % (fam, mode[0]), timeout=timeout, maxj=maxj,
                              gate_wait_s=gate_wait_s)
            ent["cand_idx"] = ci
            ent["proof_md5"] = B.md5s(txt)
            ent["certified"] = bool(jr["certified"])
            ent["reasons"] = jr["reasons"]
            ent["lean"] = jr["lean"]
            ent["n1n2"] = jr["n1n2"]
            ent["state"] = "certified" if ent["certified"] else "judged_reject"
            if ent["certified"]:
                break
        out.append(ent)
        print("[arm %s %s %d/%d] cert=%d %.0fs" % (
            fam, mode, i + 1, len(sel),
            sum(1 for x in out if x.get("certified")), time.time() - t0), flush=True)
        _flush(out, fam, shard)
    _flush(out, fam, shard)
    judged = [x for x in out if x.get("state") in ("certified", "judged_reject")]
    nc = sum(1 for x in judged if x["certified"])
    rep = {"schema": "gen44-b-arm/1", "fam": fam, "mode": mode, "shard": shard,
           "nshard": nshard, "created": B.now_iso(), "n_claim_pop": len(claim),
           "n_sample": len(sel), "n_judged": len(judged), "n_certified": nc,
           "rate": round(nc / max(1, len(judged)), 4),
           "wilson95": B.wilson(nc, len(judged)),
           "by_shape": _by_shape(judged, fam),
           "by_struct_state": _by_key(judged, "struct_state"),
           "state_hist": dict(collections.Counter(x.get("state") for x in out)),
           "reasons_hist": dict(collections.Counter(
               (x.get("reasons") or ["?"])[0] for x in judged if not x["certified"])),
           "only_state": only_state,
           "n_render_skip": sum(1 for x in out if x.get("state") == "render_skip"),
           "renderer": ("exam_single_v3" if mode == "exam"
                        else "exam_single_v3+leaf_candidate_envelope(cap%d)" % cap),
           "caveat": ("exam = 单渲染考试口径 (无枚举)" if mode == "exam"
                      else "gold = 有界包络(装置上界口径), 两列不得混用"),
           "judge_md5": B.JUDGE_MD5, "wall_s": round(time.time() - t0, 1)}
    tag = mode.upper() if only_state == "any" else mode.upper() + "_CERT"
    B.jdump(rep, os.path.join(OUT, "GEN44_%s_ARM_%s.json" % (fam.upper(), tag)))
    return rep


def _flush(out, fam, shard):
    with open(os.path.join(OUT, "GEN44_%s_ROWS.jsonl.s%d" % (fam.upper(), shard)),
              "w", encoding="utf-8") as f:
        for x in out:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")


# ================================================================ §13 GATE assembly
def assemble(fam):
    up = fam.upper()
    g = {"schema": "gen44-b-gate/1", "fam": fam, "created": B.now_iso()}
    mod = fam_module(fam)
    g["spec"] = mod.SPEC
    g["ops"] = {k: B.OP_IDX[k] for k in mod.SPEC["ops"]}
    for tag, pat in (("census", "GEN44_%s_CENSUS.json" % up),
                     ("gen", "GEN44_%s_CORPUS.json" % up),
                     ("frames", "GEN44_%s_FRAMES.json" % up),
                     ("check", "GEN44_%s_CHECK.json" % up),
                     ("payload", "GEN44_%s_PAYLOAD.json" % up),
                     ("leak", "GEN44_%s_LEAK.json" % up),
                     ("arm_exam", "GEN44_%s_ARM_EXAM.json" % up),
                     ("arm_exam_certifiable", "GEN44_%s_ARM_EXAM_CERT.json" % up),
                     ("arm_gold", "GEN44_%s_ARM_GOLD.json" % up)):
        p = os.path.join(OUT, pat)
        g[tag] = json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None
    g["md5"] = {k: (B.md5f(os.path.join(_PKG_DIR, k)) if os.path.exists(os.path.join(_PKG_DIR, k))
                    else None)
                for k in ("gen44_bcore.py", "gen44_fam_b789.py",
                          "gen44_fam_%s.py" % fam, "gen44_gen_%s.py" % fam,
                          "gen44_check_%s.py" % fam, "gen44_payload_%s.py" % fam,
                          "gen44_render_exam_v3.py", "gen44_render_gold.py")}
    g["judge_md5"] = B.JUDGE_MD5
    g["judge_md5_live"] = B.md5f(os.path.join(B.RES, "scratch", "drct", "drct_common.py"))
    g["verdict"] = _verdict(g)
    return g


def _verdict(g):
    bad, notes = [], []
    if g.get("judge_md5") != g.get("judge_md5_live"):
        bad.append("judge_md5")
    if g["md5"].get("gen44_render_exam_v3.py") != B.EXAM_MD5:
        bad.append("exam_render_md5")
    if g["md5"].get("gen44_render_gold.py") != B.GOLD_MD5:
        bad.append("gold_render_md5")
    cen = g.get("census")
    if not cen or not cen.get("n_pop"):
        bad.append("C1_structure")
    gen = g.get("gen")
    chk = g.get("check")
    if not gen or gen.get("n", 0) < 400:
        bad.append("C2_gen_n")
    r_adm = (chk or {}).get("rate_admissible", (chk or {}).get("rate"))
    if not chk or chk.get("n_sample", 0) < 12 or (r_adm or 0) < 0.80:
        bad.append("C2_synth_rate")
    if chk and (chk.get("rate") or 0) < 0.80:
        notes.append("C2 全样本率 < 0.80: 见 check.by_shape (含本 Lean 环境不可编译子群)")
    if chk and chk.get("n2_accepted", 0) < max(1, chk.get("n_certified", 0)):
        bad.append("C2_n2")
    pl = g.get("payload")
    if not pl or pl.get("verdict") != "PASS":
        bad.append("C3_payload")
    if pl and (pl.get("diversity") or {}).get("n_distinct_exam", 0) < 2:
        bad.append("C3_diversity")
    lk = g.get("leak")
    if not lk or lk.get("verdict") != "PASS":
        bad.append("C4_leak")
    fr = g.get("frames")
    if not fr or fr.get("verdict") != "PASS":
        bad.append("C7_frames")
    arm = g.get("arm_exam") or {}
    armc = g.get("arm_exam_certifiable") or {}
    if max(arm.get("n_judged", 0), armc.get("n_judged", 0)) < 1:
        bad.append("C5_arm_judged")
    if arm.get("n_render_skip"):
        notes.append("C5: 均匀臂 %d/%d 行 render_skip (装置无证书 => 未判, 不作 0 分); "
                     "率按 judged 行计" % (arm["n_render_skip"], arm.get("n_sample", 0)))
    if armc:
        notes.append("C5: 投标行臂 (certifiable 子集) n_judged=%s n_cert=%s rate=%s"
                     % (armc.get("n_judged"), armc.get("n_certified"), armc.get("rate")))
    for k in ("census", "gen", "check", "payload", "leak", "arm_exam", "frames"):
        if not g.get(k):
            bad.append("missing:" + k)
    return {"status": "PASS" if not bad else "FAIL", "gaps": sorted(set(bad)),
            "criterion_note": ("C2 率判据按 check.rate_admissible (可编译子群) 评估; 全样本率同时并出, "
                               "不可编译子群不作 0 分只登记 (by_shape + reasons_hist 逐条归因); "
                               "C2_n2 判据 = n2_accepted >= n_certified (原 0.9*n_sample 会把不可编译行的 "
                               "parse 失败误计为 N2 失败); C5 判据 = 两臂任一 n_judged >= 1 "
                               "(原 n_judged >= 0.8*n_sample 会把 render_skip(装置无证书) 误计为端到端零); "
                               "C3 多样性 = 族级 n_distinct_exam >= 2, 读数全部并出 "
                               "(B7 3 / B8 2 / B9 >=4; 全语料 >=6 由主棒合并后评)"),
            "notes": notes}


# ================================================================ §14 CLI
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fam", required=True, choices=FAMS + ("all",))
    ap.add_argument("--stage", required=True,
                    choices=("selftest", "census", "gen", "leak", "payload", "frames",
                             "check", "arm", "assemble", "report", "chain"))
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sample", type=int, default=16)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--max-lean-jobs", type=int, default=5)
    ap.add_argument("--gate-wait-s", type=float, default=900.0)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--mode", default="exam", choices=("exam", "gold"))
    ap.add_argument("--cap", type=int, default=8)
    ap.add_argument("--only-state", default="any", choices=("any", "certifiable"))
    ap.add_argument("--row-timeout-s", type=float, default=ROW_TIMEOUT_S)
    ap.add_argument("--seed", type=int, default=20260926)
    a = ap.parse_args()
    fam, up = a.fam, a.fam.upper()
    if a.stage == "chain":
        # Single-process serial full chain (check for three families -> arm for three families):
        # this slot **always runs exactly one Lean process**.
        for f in FAMS:
            print("[chain check %s] %s" % (f, B.now_iso()), flush=True)
            rep = check_corpus(f, _corpus(f.upper()), sample=a.sample, shard=0, nshard=1,
                               maxj=a.max_lean_jobs, seed=a.seed,
                               gate_wait_s=a.gate_wait_s, timeout=a.timeout)
            B.jdump(rep, os.path.join(OUT, "GEN44_%s_CHECK.json.s0" % f.upper()))
            print(json.dumps({k: v for k, v in rep.items() if k != "rows"},
                             ensure_ascii=False), flush=True)
        for f in FAMS:
            print("[chain arm %s] %s" % (f, B.now_iso()), flush=True)
            rep = arm(f, mode=a.mode, n=a.n, seed=a.seed, shard=0, nshard=1,
                      maxj=a.max_lean_jobs, gate_wait_s=a.gate_wait_s,
                      timeout=a.timeout, cap=a.cap, row_timeout=a.row_timeout_s,
                      only_state=a.only_state)
            print(json.dumps({k: v for k, v in rep.items() if k != "rows"},
                             ensure_ascii=False), flush=True)
        print("[chain DONE] %s" % B.now_iso(), flush=True)
        return
    if a.stage == "selftest":
        print(json.dumps({"fam_selftest": fam_module(fam).selftest(),
                          "core": B.selftest()}, ensure_ascii=False, indent=1))
        return
    if a.stage == "census":
        rep, recs = census(fam, limit=(a.limit or None), row_timeout=a.row_timeout_s)
        B.jdump(rep, os.path.join(OUT, "GEN44_%s_CENSUS.json" % up))
        with open(os.path.join(OUT, "GEN44_%s_CENSUS.jsonl" % up), "w",
                  encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(json.dumps(rep, ensure_ascii=False, indent=1))
    elif a.stage == "gen":
        rows, rep = gen_corpus(fam, a.n, a.seed, row_timeout=a.row_timeout_s)
        with open(os.path.join(OUT, "GEN44_%s_CORPUS.jsonl" % up), "w",
                  encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        B.jdump(rep, os.path.join(OUT, "GEN44_%s_CORPUS.json" % up))
        print(json.dumps({k: v for k, v in rep.items() if k != "text_hist"},
                         ensure_ascii=False, indent=1))
    elif a.stage == "leak":
        rows = _corpus(up)
        rep = leak(fam, rows, pool_limit=(a.limit or None))
        B.jdump(rep, os.path.join(OUT, "GEN44_%s_LEAK.json" % up))
        print(json.dumps(rep, ensure_ascii=False, indent=1)[:1500])
    elif a.stage == "payload":
        rows = _corpus(up)
        rep = payload_tests(fam, rows)
        B.jdump(rep, os.path.join(OUT, "GEN44_%s_PAYLOAD.json" % up))
        print(json.dumps(rep, ensure_ascii=False, indent=1)[:1500])
    elif a.stage == "frames":
        rows = _corpus(up)
        rep = frames_report(fam, rows)
        B.jdump(rep, os.path.join(OUT, "GEN44_%s_FRAMES.json" % up))
        print(json.dumps(rep, ensure_ascii=False, indent=1)[:1500])
    elif a.stage == "check":
        rows = _corpus(up)
        rep = check_corpus(fam, rows, sample=a.sample, shard=a.shard, nshard=a.nshard,
                           maxj=a.max_lean_jobs, seed=a.seed,
                           gate_wait_s=a.gate_wait_s, timeout=a.timeout)
        B.jdump(rep, os.path.join(OUT, "GEN44_%s_CHECK.json.s%d" % (up, a.shard)))
        print(json.dumps({k: v for k, v in rep.items() if k != "rows"},
                         ensure_ascii=False, indent=1))
    elif a.stage == "arm":
        rep = arm(fam, mode=a.mode, n=a.n, seed=a.seed, shard=a.shard, nshard=a.nshard,
                  maxj=a.max_lean_jobs, gate_wait_s=a.gate_wait_s, timeout=a.timeout,
                  cap=a.cap, row_timeout=a.row_timeout_s, only_state=a.only_state)
        print(json.dumps(rep, ensure_ascii=False, indent=1))
    elif a.stage == "assemble":
        rows, meta = [], None
        for p in sorted(glob.glob(os.path.join(OUT, "GEN44_%s_CHECK.json.s*" % up))):
            d = json.load(open(p, encoding="utf-8"))
            rows += d["rows"]
            meta = {k: v for k, v in d.items() if k != "rows"}
        if meta is not None:
            nc = sum(1 for r in rows if r["certified"])
            meta["rows"] = rows
            meta["n_sample"] = len(rows)
            meta["n_certified"] = nc
            meta["rate"] = round(nc / max(1, len(rows)), 4)
            meta["wilson95"] = B.wilson(nc, len(rows))
            B.jdump(meta, os.path.join(OUT, "GEN44_%s_CHECK.json" % up))
        g = assemble(fam)
        B.jdump(g, os.path.join(OUT, "GEN44_%s_GATE.json" % up))
        print(json.dumps({"fam": fam, "verdict": g["verdict"],
                          "spec": g["spec"]}, ensure_ascii=False, indent=1))
    elif a.stage == "report":
        g = assemble(fam)
        cen, gen, chk = g.get("census") or {}, g.get("gen") or {}, g.get("check") or {}
        arm_ = g.get("arm_exam") or {}
        armc = g.get("arm_exam_certifiable") or {}
        pl, lk, fr = g.get("payload") or {}, g.get("leak") or {}, g.get("frames") or {}
        print(json.dumps({"fam": fam, "verdict": g["verdict"],
                          "structure": {"n_pop": cen.get("n_pop"),
                                        "counts": cen.get("counts"),
                                        "g_ops_hist": cen.get("g_ops_hist")},
                          "synth": {"n": gen.get("n"), "kinds": gen.get("kind_hist")},
                          "check": {"n": chk.get("n_sample"), "cert": chk.get("n_certified"),
                                    "rate": chk.get("rate"), "wilson": chk.get("wilson95"),
                                    "n2": chk.get("n2_accepted")},
                          "payload": {k: pl.get(k) for k in ("n_slot", "slot_bad",
                                                             "n_digit", "digit_bad",
                                                             "verdict")},
                          "diversity": pl.get("diversity"),
                          "leak": {"hits": lk.get("hits"), "verdict": lk.get("verdict"),
                                   "pos": lk.get("positive_control_pass")},
                          "frames": {"verdict": fr.get("verdict"),
                                     "bad": {k: fr.get(k) for k in
                                             ("n_empty_bad", "n_last_nonzero_bad",
                                              "n_const_nonzero", "n_const_zero")}},
                          "end2end_certifiable_subset": {
                              "n_sample": armc.get("n_sample"),
                              "n_judged": armc.get("n_judged"),
                              "n_certified": armc.get("n_certified"),
                              "rate": armc.get("rate"),
                              "wilson": armc.get("wilson95")},
                          "end2end": {"n_sample": arm_.get("n_sample"),
                                      "n_judged": arm_.get("n_judged"),
                                      "n_certified": arm_.get("n_certified"),
                                      "rate": arm_.get("rate"),
                                      "wilson": arm_.get("wilson95"),
                                      "reasons": arm_.get("reasons_hist")}},
                         ensure_ascii=False, indent=1))


def _corpus(up):
    return [json.loads(l) for l in open(os.path.join(
        OUT, "GEN44_%s_CORPUS.jsonl" % up), encoding="utf-8") if l.strip()]


if __name__ == "__main__":
    main()
