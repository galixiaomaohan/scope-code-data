# -*- coding: utf-8 -*-
# gen42_stmt.py -- miniF2F statement-structure parsing + sympy exact kernel (device-layer shared module; gen42)
#
# Conventions:
#   * relation layer: outermost relation symbol in {=, ≤, <, ≠, ≥, >, ∣}; conjunctions (∧) of hypotheses are split mechanically;
#     ¬ prefix (including ∤) sets neg=True.
#   * expression layer: restricted grammar -> sympy exact expressions (rationals/polynomials/rational functions).
#     Allowed: numeric literals · binder variables · + - * / ^(numeric exponent) · % · !(closed-value factorial)
#           · |x| (Abs) · ↑ (cast, marked) · Nat.factorial/Nat.gcd/Nat.lcm (closed values) · type annotations
#     Forbidden: any function-symbol term and non-numeric atom (√/log/sin/Finset/Set/ZMod/Complex/...)
#   * type semantics: binary `-` over ℕ means truncated subtraction (Max(0,a-b)) · `/` over ℕ/ℤ means integer division
#     (evaluable only when closed); once used => set the nat_risk flag (only the closed-form evaluation route is
#     allowed, ring/lin routes are forbidden).
#
# Read-only: probes/lean_exam.py.txt (loading the frozen artifact)
import os
import re
import sys

RES = os.environ.get("SCOPE_RESEARCH_ROOT", ".")
G42 = os.path.join(RES, "scratch", "ccp", "gen42")
PROBES = os.path.join(RES, "probes")
MINIF2F_ROOT = os.environ.get(
    "SCOPE_MINIF2F_ROOT", os.path.join(RES, os.pardir, "lean", "miniF2F-main"))

import importlib.machinery as _im                                       # noqa: E402
import importlib.util as _iu                                            # noqa: E402


def _load(name, path):
    ld = _im.SourceFileLoader(name, path)
    m = _iu.module_from_spec(_iu.spec_from_loader(name, ld))
    sys.modules[name] = m
    ld.exec_module(m)
    return m


import sympy as sp                                                     # noqa: E402

SUB = "₀₁₂₃₄₅₆₇₈₉"
IDRE = r"[A-Za-z_][A-Za-z0-9_'%s]*" % SUB
NUMERIC_TYPES = {"ℚ", "ℝ", "ℤ", "ℕ", "ℂ", "Rat", "Real", "Int", "Nat", "Q", "R", "Z", "N",
                 "Complex"}
CPLX_TYPES = {"ℂ", "Complex"}
RAT_TYPES = {"ℚ", "ℝ", "Rat", "Real", "Q", "R"}
INT_TYPES = {"ℤ", "Int", "Z"}
NAT_TYPES = {"ℕ", "Nat", "N"}
RELS = ("≠", "≤", "≥", "=", "<", ">", "∣")


class PErr(Exception):
    pass


# ================================================================ structural scan
def top_groups(s):
    out, depth, start = [], 0, None
    for i, ch in enumerate(s):
        if ch in "([{":
            if depth == 0:
                start = i + 1
            depth += 1
        elif ch in ")]}":
            depth -= 1
            if depth == 0 and start is not None:
                out.append((s[start - 1], s[start:i]))
                start = None
    return out


def binder_groups(txt):
    out = []
    for br, g in top_groups(txt):
        idx, depth = -1, 0
        for i, ch in enumerate(g):
            if ch in "([{":
                depth += 1
            elif ch in ")]}":
                depth -= 1
            elif ch == ":" and depth == 0:
                idx = i
                break
        if idx < 0:
            continue
        names = [n for n in g[:idx].split() if re.fullmatch(IDRE, n)]
        out.append((names, g[idx + 1:].strip(), br))
    return out


def top_rel(s):
    d = 0
    for i, ch in enumerate(s):
        if ch in "([{":
            d += 1
        elif ch in ")]}":
            d -= 1
        elif d == 0 and ch in RELS:
            return ch, i
    return None, -1


def split_top_conj(s):
    """Split on top-level ∧."""
    parts, d, cur = [], 0, []
    i = 0
    while i < len(s):
        ch = s[i]
        if ch in "([{":
            d += 1
        elif ch in ")]}":
            d -= 1
        if d == 0 and ch == "∧":
            parts.append("".join(cur).strip())
            cur = []
            i += 1
            continue
        cur.append(ch)
        i += 1
    parts.append("".join(cur).strip())
    return [p for p in parts if p]


# ================================================================ finite sum/product preprocessing
_SUMHDR = re.compile(r"[∑∏]\s*([A-Za-z_][A-Za-z0-9_'%s]*)\s*(?:∈|in)\s*" % SUB)


def _group_end(s, start):
    """Find the end of the **current** group starting at start (the ancestor group's closing bracket is where depth returns to -1)."""
    d, i = 0, start
    while i < len(s):
        if s[i] in "([{":
            d += 1
        elif s[i] in ")]}":
            if d == 0:
                return i
            d -= 1
        i += 1
    return len(s)


def rewrite_sumprod(s, vars_, types, nat_mode, depth=0):
    """Exactly evaluate ∑/∏ over Finset.Icc/Ico/range (symbolic index, numeric bounds) into a constant."""
    if depth > 6 or ("∑" not in s and "∏" not in s):
        return s, None
    out, flags = s, {"uses_sumprod": True}
    for _ in range(16):
        i = next((j for j, ch in enumerate(out) if ch in "∑∏"), None)
        if i is None:
            break
        op = out[i]
        m = _SUMHDR.match(out, i)
        if not m:
            raise PErr("sumparse")
        kv, p = m.group(1), m.end()
        mset = re.match(r"(Finset\.(?:Icc|Ico|range)|Set\.Icc)\s*", out[p:])
        if not mset:
            raise PErr("sumset")
        kind = mset.group(1)
        p += mset.end()
        # bounds: numeric literals
        nums = []
        while len(nums) < 2:
            mn = re.match(r"(\d+)\s*", out[p:])
            if not mn:
                break
            nums.append(int(mn.group(1)))
            p += mn.end()
        if kind.endswith("Icc"):
            if len(nums) < 2:
                raise PErr("sumbounds")
            lo, hi = nums[0], nums[1]
        elif kind.endswith("Ico"):
            if len(nums) < 2:
                raise PErr("sumbounds")
            lo, hi = nums[0], nums[1] - 1
        else:
            if len(nums) < 1:
                raise PErr("sumbounds")
            lo, hi = 0, nums[0] - 1
        mp = re.match(r"\s*,\s*", out[p:])
        if not mp:
            raise PErr("sumcomma")
        p += mp.end()
        end = _group_end(out, p)
        body = out[p:end]
        # evaluate inner ∑/∏ recursively first
        body2, _f2 = rewrite_sumprod(body, vars_, types, nat_mode, depth + 1)
        kk = sp.Symbol(kv)
        bv, mb = parse_arith(body2, set(vars_) | {kv}, types, nat_mode)
        acc = None
        n = hi - lo + 1
        if n < 0:
            raise PErr("sumbadrange")
        if n > 400:
            raise PErr("sumtooloong")
        for t in range(lo, hi + 1):
            v = bv.subs({kk: sp.Integer(t)})
            if v.free_symbols:
                raise PErr("summand_not_closed")
            acc = v if acc is None else (acc * v if op == "∏" else acc + v)
        if acc is None:
            raise PErr("sumbadrange")
        acc = sp.nsimplify(acc)
        txt = ("(%s)" % acc) if acc.q != 1 else ("(%s)" % acc.p)
        out = out[:i] + txt + out[end:]
    return out, flags


# ================================================================ expression core (tokenize)
_TOK = re.compile(r"\s*(?:(\d+)|([A-Za-z_][A-Za-z0-9_.'%s]*)|(\*\*|\^|//|[+\-*/()%%↑!|]|≤|≥|≠)|(ℚ|ℝ|ℤ|ℕ|→)|(.))" % SUB,
                  re.S)


class _P:
    def __init__(self, toks, vars_, types, nat_mode):
        self.t, self.i, self.vars, self.types, self.nat = toks, 0, set(vars_), types, nat_mode
        self.uses_div = self.uses_sub = self.uses_mod = self.uses_cast = False
        self.uses_abs = self.uses_fact = self.uses_sympow = False
        self.nat_risk = False

    def pk(self):
        return self.t[self.i] if self.i < len(self.t) else (None, None)

    def take(self):
        v = self.t[self.i]
        self.i += 1
        return v

    def atom(self):
        k, v = self.pk()
        if k == "num":
            self.take()
            return sp.Integer(int(v))
        if k == "op" and v == "(":
            self.take()
            e = self.expr()
            k2, v2 = self.pk()
            if k2 != "op" or v2 != ")":
                raise PErr("rparen")
            self.take()
            # postfix factorial?
            if self.pk() == ("op", "!"):
                self.take()
                if e.free_symbols:
                    raise PErr("fact_sym")
                self.uses_fact = True
                return sp.factorial(int(e))
            return e
        if k == "op" and v == "-":
            self.take()
            return -self.atom()
        if k == "op" and v == "|":
            self.take()
            e = self.expr()
            if self.pk() != ("op", "|"):
                raise PErr("absclose")
            self.take()
            self.uses_abs = True
            return sp.Abs(e)
        if k == "op" and v == "↑":
            self.take()
            self.uses_cast = True
            return self.atom()
        if k == "id":
            if v in self.vars:
                self.take()
                # postfix factorial
                if self.pk() == ("op", "!"):
                    self.take()
                    raise PErr("fact_sym")
                return sp.Symbol(v)
            if v == "Nat" and self.pk()[1] == "Nat":
                pass
            raise PErr("fvar:%s" % v)
        raise PErr("atom:%r" % (v,))

    def factor(self):
        b = self.atom()
        # postfix factorial (closed value)
        while self.pk() == ("op", "!"):
            self.take()
            if b.free_symbols:
                raise PErr("fact_sym")
            if b.q == 1:
                b = sp.factorial(int(b))
            else:
                raise PErr("fact_nonint")
            self.uses_fact = True
        k, v = self.pk()
        while k == "op" and v in ("^", "**"):
            self.take()
            k2, v2 = self.pk()
            if k2 == "num":
                self.take()
                e = int(v2)
                if e > 256:
                    raise PErr("expbig")
                b = b ** e
            elif k2 == "id" and v2 in self.vars:
                self.take()
                self.uses_sympow = True
                b = sp.Pow(b, sp.Symbol(v2), evaluate=False)
            else:
                raise PErr("expnotnum")
            k, v = self.pk()
        return b

    def powchain(self):
        """Handle dotted names Nat.factorial / Nat.gcd etc. (the dot is folded into the id token)."""
        k, v = self.pk()
        if k == "id" and v in ("Nat.factorial", "Nat.gcd", "Nat.lcm", "Nat.choose"):
            self.take()
            args = []
            while self.pk() == ("op", "("):
                self.take()
                args.append(self.expr())
                if self.pk() != ("op", ")"):
                    raise PErr("callclose")
                self.take()
            if any(a.free_symbols for a in args):
                raise PErr("call_sym")
            if v == "Nat.factorial":
                return sp.factorial(int(args[0]))
            if v == "Nat.gcd":
                return sp.Integer(sp.igcd(*[int(a) for a in args]))
            if v == "Nat.lcm":
                return sp.Integer(sp.ilcm(*[int(a) for a in args]))
            return sp.binomial(int(args[0]), int(args[1]))
        return self.factor()

    def term(self):
        e = self.powchain()
        while True:
            k, v = self.pk()
            if k == "op" and v in ("*", "/", "//", "%"):
                self.take()
                f = self.powchain()
                if v == "*":
                    e = e * f
                elif v == "%":
                    self.uses_mod = True
                    e = sp.Mod(e, f)
                else:
                    self.uses_div = True
                    if v == "//":
                        self.nat_risk = True
                    e = e / f
            else:
                return e

    def expr(self):
        e = self.term()
        while True:
            k, v = self.pk()
            if k == "op" and v in ("+", "-"):
                self.take()
                t = self.term()
                if v == "-":
                    self.uses_sub = True
                    if self.nat:
                        self.nat_risk = True
                        e = sp.Max(0, e - t)
                    else:
                        e = e - t
                else:
                    e = e + t
            else:
                return e


def parse_arith(s, vars_, types=None, nat_mode=False):
    """Lean arithmetic text -> (sympy, meta). Raises PErr on failure."""
    s = s.strip()
    s = re.sub(r":\s*(ℚ|ℝ|ℤ|ℕ|Rat|Real|Int|Nat|Q|R|Z|N)\b", "", s)
    s = re.sub(r"\s+", " ", s)
    s = s.replace("Complex.I", "CI")
    vars2 = set(vars_) | ({"CI"} if "CI" in s else set())
    extra = {}
    if "∑" in s or "∏" in s:
        s, fl = rewrite_sumprod(s, vars2, types or {}, nat_mode)
        extra.update(fl or {})
    toks = []
    i = 0
    while i < len(s):
        mm = _TOK.match(s, i)
        if not mm:
            raise PErr("lex@%d %r" % (i, s[i:i + 12]))
        if mm.group(1) is not None:
            toks.append(("num", mm.group(1)))
        elif mm.group(2) is not None:
            toks.append(("id", mm.group(2)))
        elif mm.group(3) is not None:
            toks.append(("op", mm.group(3)))
        else:
            raise PErr("chr %r" % (mm.group(5) or mm.group(4),))
        i = mm.end()
    for k, v in toks:
        if k == "id" and v not in vars2:
            raise PErr("fvar:%s" % v)
    p = _P(toks, vars2, types or {}, nat_mode)
    e = p.expr()
    if p.i != len(p.t):
        raise PErr("trailing:%r" % (p.t[p.i:],))
    if "CI" in vars2:
        e = e.subs({sp.Symbol("CI"): sp.I})
    meta = {"uses_div": p.uses_div, "uses_sub": p.uses_sub, "uses_mod": p.uses_mod,
            "uses_cast": p.uses_cast, "uses_abs": p.uses_abs, "uses_fact": p.uses_fact,
            "uses_sympow": p.uses_sympow, "nat_risk": p.nat_risk}
    meta.update(extra)
    return e, meta


# ================================================================ statement parsing
def norm_answer(s):
    s = re.sub(r"answer\(([^()]*(?:\([^()]*\)[^()]*)*)\)", r"(\1)", s)
    return re.sub(r"\s+", " ", s or "").strip()


def _classify_group(names, ty):
    rel, _i = top_rel(ty)
    ht = ty.strip()[:1]
    return bool(rel) or ht in ("∀", "∃", "¬", "(") and bool(top_rel(ty))


def parse_stmt(prob):
    """Problem -> structured dict (with per-item parse results; any item failing => ok=False + reason)."""
    full = norm_answer(prob.head)
    concl_txt = norm_answer(getattr(prob, "concl", "") or "")
    rest = re.sub(r"^theorem\s+", "", full).strip()
    m = re.match(r"^([A-Za-z0-9_.']+)\s*(.*)$", rest, re.S)
    binders = (m.group(2) if m else "").strip()
    if binders.endswith(":"):
        binders = binders[:-1].strip()
    groups = binder_groups(binders)
    vars_ty, hypgroups = {}, []
    for names, ty, br in groups:
        rel, _i = top_rel(ty)
        ht = ty.strip()[:1]
        if rel or ht in ("∀", "∃", "¬"):
            for n in names:
                hypgroups.append((n, ty))
        else:
            for n in names:
                vars_ty[n] = ty
    rec = {"name": prob.name, "binders": binders, "vars": vars_ty,
           "n_vars": len(vars_ty), "concl_txt": concl_txt,
           "concl": None, "hyps": [], "ok": False, "reason": None,
           "n_hyps": len(hypgroups)}
    vs = set(vars_ty)
    nat_mode = any(t in NAT_TYPES for t in vars_ty.values())
    rec["nat_mode"] = nat_mode
    rec["has_complex"] = any(t in CPLX_TYPES for t in vars_ty.values())
    if not all(t in NUMERIC_TYPES for t in vars_ty.values()):
        rec["reason"] = "nonnumeric_type"
        return rec
    # conclusion
    neg = False
    ct = concl_txt
    while ct.startswith("¬"):
        neg = not neg
        ct = ct[1:].strip()
    if ct.startswith("(") and ct.endswith(")") and top_rel(ct[1:-1])[0] is not None:
        ct = ct[1:-1].strip()
    crel, ci = top_rel(ct)
    if crel is None:
        rec["reason"] = "concl_no_rel"
        return rec
    cl, cr = ct[:ci].strip(), ct[ci + 1:].strip()
    try:
        el, ml = parse_arith(cl, vs, vars_ty, nat_mode)
        er, mr = parse_arith(cr, vs, vars_ty, nat_mode)
    except PErr as ex:
        rec["reason"] = "concl_unparsed:%s" % ex
        return rec
    rec["concl"] = {"rel": crel, "neg": neg, "lhs": el, "rhs": er,
                    "lhs_txt": cl, "rhs_txt": cr,
                    "meta": {k: bool(ml[k] or mr[k]) for k in ml}}
    # hypotheses (split ∧)
    for gname, gty in hypgroups:
        parts = split_top_conj(gty)
        if len(parts) > 1 and not gname:
            parts = ["%s.%d" % (gname, i) for i, _ in enumerate(parts)]
        for pi, ptxt in enumerate(parts):
            hname = gname if len(parts) == 1 else "%s.%d" % (gname, pi)
            ptxt = ptxt.strip()
            neg = False
            while ptxt.startswith("¬"):
                neg = not neg
                ptxt = ptxt[1:].strip()
            if ptxt.startswith("(") and ptxt.endswith(")") and \
                    top_rel(ptxt[1:-1])[0] is not None:
                ptxt = ptxt[1:-1].strip()
            hrel, hi = top_rel(ptxt)
            if hrel is None:
                rec["reason"] = "hyp_no_rel:%s" % hname
                return rec
            hl, hr = ptxt[:hi].strip(), ptxt[hi + 1:].strip()
            try:
                e1, m1 = parse_arith(hl, vs, vars_ty, nat_mode)
                e2, m2 = parse_arith(hr, vs, vars_ty, nat_mode)
            except PErr as ex:
                rec["reason"] = "hyp_unparsed:%s:%s" % (hname, ex)
                return rec
            rec["hyps"].append({"name": hname, "rel": hrel, "neg": neg,
                                "lhs": e1, "rhs": e2, "txt": ptxt,
                                "meta": {k: bool(m1[k] or m2[k]) for k in m1}})
    rec["ok"] = True
    return rec


# ================================================================ route decision (live computation)
def linform(e):
    """Linear form -> ({sym: coef}, const) or None."""
    try:
        num, den = sp.fraction(sp.together(e))
        if den.free_symbols:
            return None
        e = sp.expand(num)
        c, const = {}, sp.Integer(0)
        for t in (e.args if e.is_Add else (e,)):
            fs = t.free_symbols
            if not fs:
                const += t
            elif len(fs) == 1:
                s = next(iter(fs))
                if sp.degree(t, s) > 1:
                    return None
                c[s] = c.get(s, sp.Integer(0)) + sp.expand(t / s)
            else:
                return None
        return {k: sp.nsimplify(v) for k, v in c.items()}, const
    except Exception:                                                  # noqa: BLE001
        return None


def is_poly(e):
    try:
        num, den = sp.fraction(sp.together(e))
        ident = sp.Symbol("__x__")
        return not den.free_symbols
    except Exception:                                                  # noqa: BLE001
        return False


def expand0(e):
    """Is the polynomial/rational expression identically 0 (live computation)."""
    try:
        num, den = sp.fraction(sp.together(e))
        if den.free_symbols:
            return None
        return sp.expand(num) == 0
    except Exception:                                                  # noqa: BLE001
        return None


def closed(e):
    return not e.free_symbols


def exact_int(e):
    if not closed(e):
        return None
    v = sp.nsimplify(e)
    if v.q != 1:
        return None
    return int(v)


def eval_closed(e):
    if not closed(e):
        return None
    return sp.nsimplify(e)


def rel_truth(rel, a, b, neg=False):
    """Truth of a relation for closed values; returns True/False/None. ∣ = divisibility (b = a·c)."""
    try:
        if rel == "=":
            t = (a == b)
        elif rel == "≤":
            t = (a <= b)
        elif rel == "<":
            t = (a < b)
        elif rel == "≥":
            t = (a >= b)
        elif rel == ">":
            t = (a > b)
        elif rel == "≠":
            t = (a != b)
        elif rel == "∣":
            ai, bi = int(sp.nsimplify(a)), int(sp.nsimplify(b))
            t = (bi == 0) if ai == 0 else (bi % ai == 0)
        else:
            return None
        if t is True or t is False:
            return (not t) if neg else t
        return None
    except Exception:                                                  # noqa: BLE001
        return None


def farkas(hyps, concl):
    """Exact Fourier-Motzkin: hyps => concl (form (coeffs, const, strict) denotes Σc·x+c <=/< 0)."""
    def neg(f):
        c, k, st = f
        return ({v: -x for v, x in c.items()}, -k, not st)

    sysm = [(dict(c), k, st) for c, k, st in hyps] + [neg(concl)]

    def fm(system, vs):
        if not vs:
            return not all((k <= 0) if not st else (k < 0) for _c, k, st in system)
        v = vs[0]
        pos, neg_, zer = [], [], []
        for c, k, st in system:
            a = c.get(v, 0)
            (pos if a > 0 else neg_ if a < 0 else zer).append((c, k, st))
        new = list(zer)
        for cp, kp, sp_ in pos:
            for cn, kn, sn in neg_:
                ap, an = cp.get(v, 0), cn.get(v, 0)
                c = {u: cp.get(u, 0) * (-an) + cn.get(u, 0) * ap
                     for u in set(list(cp.keys()) + list(cn.keys())) if u != v}
                c = {u: x for u, x in c.items() if x != 0}
                new.append((c, kp * (-an) + kn * ap, sp_ or sn))
        return fm(new, vs[1:])

    vs = sorted({v for c, _k, _s in sysm for v in c}, key=str)
    if len(vs) > 4:
        return None
    return fm(sysm, vs)


def linear_entail(hyps, concl_rel, lhs, rhs):
    """hyps => (lhs rel rhs)?  hyps: [dict(rel,lhs,rhs,neg)]. Returns True/False/None."""
    def form(l, r, rel):
        lf = linform(l - r)
        if lf is None:
            return None
        c, k = lf
        if rel == "≤":
            return (c, k, False)
        if rel == "<":
            return (c, k, True)
        return None
    hf = []
    for h in hyps:
        forms = []
        if h["rel"] == "=":
            f1, f2 = form(h["lhs"], h["rhs"], "≤"), form(h["rhs"], h["lhs"], "≤")
            if f1 is None or f2 is None:
                return None
            forms = [f1, f2]
        elif h["rel"] in ("≤", "<", "≥", ">"):
            rel = h["rel"]
            l, r = h["lhs"], h["rhs"]
            if rel == "≥":
                rel, l, r = "≤", r, l
            elif rel == ">":
                rel, l, r = "<", r, l
            f = form(l, r, rel)
            if f is None:
                return None
            forms = [f]
        elif h["rel"] == "≠":
            continue
        else:
            return None
        if h["neg"]:
            return None
        hf += forms
    if len(hf) > 24:
        return None
    if concl_rel == "=":
        v1, v2 = farkas(hf, form(lhs, rhs, "≤")), farkas(hf, form(rhs, lhs, "≤"))
        if v1 is None or v2 is None:
            return None
        return bool(v1 and v2)
    if concl_rel in ("≤", "<"):
        c = form(lhs, rhs, concl_rel)
        if c is None:
            return None
        return farkas(hf, c)
    return None


def linearize(e):
    """Write an expanded expression as an **atomic linear form**: constant part + {atom: coefficient}.
    An atom = a monomial (symbol/product/power) -- consistent with Lean `linarith` treating nonlinear monomials as
    atoms. This is a **relaxation** of the original system (true algebraic relations between atoms are dropped) =>
    if the implication holds it is certainly true (sound)."""
    try:
        e = sp.expand(e)
    except Exception:                                                  # noqa: BLE001
        return None
    atoms, const = {}, sp.Integer(0)
    for t in (e.args if e.is_Add else (e,)):
        try:
            c, rest = t.as_coeff_Mul()
        except Exception:                                              # noqa: BLE001
            return None
        if not rest.free_symbols:
            const += c * rest
        else:
            atoms[rest] = atoms.get(rest, sp.Integer(0)) + c
    atoms = {k: v for k, v in atoms.items() if v != 0}
    return atoms, const


def farkas_atoms(hyps, concl):
    """Exact Fourier-Motzkin over atomic linear forms. Form: (atoms, const, strict) denotes Σc·a + const <=/< 0."""
    def neg(f):
        a, k, st = f
        return ({v: -x for v, x in a.items()}, -k, not st)

    sysm = [(dict(a), k, st) for a, k, st in hyps] + [neg(concl)]

    def fm(system, vs):
        if not vs:
            feas = all((k <= 0) if not st else (k < 0) for _a, k, st in system)
            if os.environ.get("G42_FMDBG"):
                sys.stderr.write("FM base feas=%s | %s\n" % (feas, [(str(k), st) for _a, k, st in system]))
            return not feas                 # refutation: system infeasible <=> the implication holds
        v = vs[0]
        pos, neg_, zer = [], [], []
        for a, k, st in system:
            c = a.get(v, 0)
            (pos if c > 0 else neg_ if c < 0 else zer).append((a, k, st))
        new = list(zer)
        for ap_, kp, sp_ in pos:
            for an_, kn, sn in neg_:
                cp, cn = ap_.get(v, 0), an_.get(v, 0)
                a = {u: ap_.get(u, 0) * (-cn) + an_.get(u, 0) * cp
                     for u in set(list(ap_.keys()) + list(an_.keys())) if u != v}
                a = {u: x for u, x in a.items() if x != 0}
                new.append((a, kp * (-cn) + kn * cp, sp_ or sn))
        if os.environ.get("G42_FMDBG"):
            sys.stderr.write("FM v=%s pos=%d neg=%d zer=%d new=%d\n"
                             % (v, len(pos), len(neg_), len(zer), len(new)))
        if len(new) > 900:
            return None
        return fm(new, vs[1:])

    vs = sorted({v for a, _k, _s in sysm for v in a}, key=sp.srepr)
    if len(vs) > 7:
        return None
    return fm(sysm, vs)


def atom_entail(hyps, concl_rel, lhs, rhs):
    """Implication check over atomic linear forms (sound relaxation; completeness not guaranteed => Lean adjudicates in the end)."""
    def form(l, r, rel):
        lf = linearize(l - r)
        if lf is None:
            return None
        a, k = lf
        if rel == "≤":
            return (a, k, False)
        if rel == "<":
            return (a, k, True)
        return None
    hf = []
    for h in hyps:
        if h["neg"]:
            return None
        rel = h["rel"]
        l, r = h["lhs"], h["rhs"]
        if rel == "=":
            f1, f2 = form(l, r, "≤"), form(r, l, "≤")
            if f1 is None or f2 is None:
                return None
            hf += [f1, f2]
        elif rel in ("≤", "<", "≥", ">"):
            if rel == "≥":
                rel, l, r = "≤", r, l
            elif rel == ">":
                rel, l, r = "<", r, l
            f = form(l, r, rel)
            if f is None:
                return None
            hf.append(f)
        elif rel == "≠":
            continue
        else:
            return None
    if len(hf) > 24:
        return None
    if concl_rel == "=":
        v1, v2 = farkas_atoms(hf, form(lhs, rhs, "≤")), farkas_atoms(hf, form(rhs, lhs, "≤"))
        if v1 is None or v2 is None:
            return None
        return bool(v1 and v2)
    if concl_rel in ("≤", "<"):
        c = form(lhs, rhs, concl_rel)
        if c is None:
            return None
        return farkas_atoms(hf, c)
    return None


# ---------------------------------------------------------------- Presburger-fragment decision
def presburger_ok(e):
    """Is e in the Presburger fragment (variables · integer constants · + - · x integer constant · % integer constant)."""
    if e is None:
        return False
    e = sp.expand(e)
    if not e.free_symbols:
        return True
    if e.is_Symbol:
        return True
    if e.is_Add:
        return all(presburger_ok(a) for a in e.args)
    if e.is_Mul:
        num = [a for a in e.args if not a.free_symbols]
        rest = [a for a in e.args if a.free_symbols]
        return bool(num) and len(rest) == 1 and presburger_ok(rest[0])
    if e.is_Pow:
        return getattr(e.exp, "is_Integer", False) and e.exp == 1
    if isinstance(e, sp.Mod):
        return (not e.args[1].free_symbols) and presburger_ok(e.args[0])
    return False


def nat_arith_ok(e):
    """Is e in the ℕ/ℤ +/x-constant/% fragment and **free of** truncated subtraction (the range omega can handle)."""
    if e is None:
        return False
    if not e.free_symbols:
        return True
    if e.is_Symbol:
        return True
    if isinstance(e, sp.Max):
        return False
    if e.is_Add:
        return all(nat_arith_ok(a) for a in e.args)
    if e.is_Mul:
        num = [a for a in e.args if not a.free_symbols]
        rest = [a for a in e.args if a.free_symbols]
        return bool(num) and len(rest) == 1 and nat_arith_ok(rest[0])
    if isinstance(e, sp.Mod):
        return (not e.args[1].free_symbols) and nat_arith_ok(e.args[0])
    return False


def omega_profile(rec):
    """Device profile of the Presburger/omega route: extract coefficient-matrix features of the goal and hypotheses (8-dim state frame)."""
    c = rec["concl"]
    if not (nat_arith_ok(c["lhs"]) and nat_arith_ok(c["rhs"])):
        return None
    for h in rec["hyps"]:
        if h["neg"] or h["rel"] not in ("=", "≤", "<", "≥", ">"):
            return None
        if not (nat_arith_ok(h["lhs"]) and nat_arith_ok(h["rhs"])):
            return None
    atoms, coef = set(), []
    def add(e):
        lf = linearize(e)
        if lf is None:
            return False
        a, k = lf
        atoms.update(a.keys())
        coef.extend(abs(v) for v in a.values())
        coef.append(abs(k))
        return True
    if not add(c["lhs"] - c["rhs"]):
        return None
    for h in rec["hyps"]:
        if not add(h["lhs"] - h["rhs"]):
            return None
    A = sorted(atoms, key=sp.srepr)
    fr = [float(len(A)), float(len(coef)),
          float(sum(1 for h in rec["hyps"] if h["rel"] == "=")),
          float(sum(1 for h in rec["hyps"] if h["rel"] in ("≤", "<"))),
          float(max([abs(v) for v in coef] or [0])),
          float(sum(abs(v) for v in coef)),
          1.0 if any(isinstance(x, sp.Mod) for x in sp.preorder_traversal(c["lhs"] - c["rhs"]))
          else 0.0,
          1.0 if any("Nat" in str(t) or t in ("ℕ", "ℤ") for t in rec["vars"].values())
          else 0.0]
    return {"n_atoms": len(A), "frame": fr}


def pos_ok(e, hyps, strict=True, depth=0):
    """Structural positivity decision: e > 0 (strict) / e >= 0. Uses only structure + sign conditions from the hypotheses."""
    if depth > 8 or e is None:
        return None
    e = sp.expand(e)
    if not e.free_symbols:
        v = sp.nsimplify(e)
        if v.is_number:
            return bool(v > 0) if strict else bool(v >= 0)
        return None
    if e.is_Symbol:
        for h in hyps:
            if h["neg"] or h["lhs"] != 0:
                continue
            if h["rhs"] == e:
                if h["rel"] == "<":
                    return True
                if h["rel"] == "≤":
                    return True if not strict else None
        return None
    if e.is_Pow:
        b, n = e.base, e.exp
        if getattr(n, "is_Integer", False) and int(n) % 2 == 0 and int(n) > 0:
            return True                              # even power >= 0 (holds when strict=False)
        if strict and pos_ok(b, hyps, True, depth + 1):
            return True
        return None
    if e.is_Mul:
        vals = [pos_ok(a, hyps, strict, depth + 1) for a in e.args]
        if any(v is False for v in vals):
            return False
        return True if all(v is True for v in vals) else None
    if e.is_Add:
        vals = [pos_ok(a, hyps, strict, depth + 1) for a in e.args]
        if any(v is False for v in vals):
            return False
        return True if all(v is True for v in vals) else None
    return None


LE = _load("lean_exam_g42s", os.path.join(PROBES, "lean_exam.py.txt"))
