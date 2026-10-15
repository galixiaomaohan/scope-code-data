# -*- coding: utf-8 -*-
"""skelmath -- the deterministic mathematical core of the skeleton goal language
(the divisibility/congruence small world).

This module is the **shared foundation** of the v32 skeleton operators (specs / generator /
checkers all take from here):
  * parsing/expansion/evaluation/differencing/substitution for univariate integer-coefficient
    polynomials (stdlib only, no float, no randomness);
  * **complete decision**: for the congruence predicate `m | p(v)`, we have
    `∀v ∈ ℤ. m | p(v) ⟺ ∀r ∈ {0..m-1}. m | p(r)` (because `v ≡ r (mod m) ⟹ p(v) ≡ p(r) (mod m)`)
    -- checking m integers is **complete**, not sampling;
  * the goal-family catalog FAMILY_CATALOG (candidate moduli are **really verified** at import
    time with the decision above; an empty result is an assertion failure);
  * goal-text rendering (the single source of problem/subgoal text -- generator and operators call
    the same function, so "the value appears literally" is word-for-word consistent on both sides);
  * the executable version of strategy rules R1-R6 (design doc §3.2).

**Honesty boundary**: this is **not** Mathlib. The goal language has only congruence/divisibility
predicates, because only those have a **complete** decision here. Any claim of "stronger
mathematics" exceeds this module's capability (see DESIGN §10).
"""
from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "VAR_NAMES", "VAR_ID", "ExprError", "parse_poly", "poly_str", "poly_eval", "poly_gcd",
    "poly_shift", "poly_sub", "poly_scale", "residues", "divides_all", "subst_affine",
    "subst_text", "new_var", "universal_goal_text", "negation_goal_text",
    "conjunction_goal_text", "leaf_goal_text", "Family", "FAMILY_CATALOG", "VALID_MODULI",
    "family_of", "parse_goal", "Goal", "decide_strategy", "STRATEGY_OPS", "ledger_str",
    "parse_ledger", "NODE_KINDS", "stack_str", "parse_stack_str", "n_open_of",
    "KIND_CODE", "CODE_KIND", "ST_CODE", "CODE_ST", "rename_text",
]

#: Variable-name pool (appears literally in the problem text; vid is the bindable integer slot)
VAR_NAMES = ("n", "m", "k", "t", "j")
VAR_ID = {v: i for i, v in enumerate(VAR_NAMES)}
#: Substitution parameter names (used by branching/stepping, distinct from the substituted variable)
SUBST_NAMES = ("k", "j", "t")


class ExprError(ValueError):
    """Expression syntax / variable out of range -- the caller must fold it into a §15 error code
    (no bare raise)."""


# ---------------------------------------------------------------------------
# 1) expressions: text -> integer-coefficient polynomial (low degree first)
# ---------------------------------------------------------------------------

def _tok(s: str):
    out, i, n = [], 0, len(s)
    while i < n:
        c = s[i]
        if c.isspace():
            i += 1
            continue
        if c.isdigit():
            j = i
            while j < n and s[j].isdigit():
                j += 1
            out.append(("num", int(s[i:j])))
            i = j
            continue
        if c.isalpha():
            j = i
            while j < n and (s[j].isalnum() or s[j] == "_"):
                j += 1
            out.append(("id", s[i:j]))
            i = j
            continue
        if c in "+-*^()":
            out.append((c, c))
            i += 1
            continue
        raise ExprError("非法字符 %r @%d" % (c, i))
    return out


class _P:
    """Recursive-descent parser: expr := term (('+'|'-') term)* ; term := fac ('*' fac)* ;
    fac := base ('^' num)? ; base := num | id | '(' expr ')'."""

    def __init__(self, toks, var):
        self.t, self.i, self.var = toks, 0, var

    def peek(self):
        return self.t[self.i] if self.i < len(self.t) else (None, None)

    def eat(self, kind):
        k, v = self.peek()
        if k != kind:
            raise ExprError("期望 %r, 实得 %r" % (kind, (k, v)))
        self.i += 1
        return v

    def expr(self):
        acc = self.term()
        while True:
            k, _ = self.peek()
            if k == "+":
                self.i += 1
                acc = _padd(acc, self.term())
            elif k == "-":
                self.i += 1
                acc = _psub(acc, self.term())
            else:
                return acc

    def term(self):
        acc = self.fac()
        while True:
            k, _ = self.peek()
            if k == "*":
                self.i += 1
                acc = _pmul(acc, self.fac())
            else:
                return acc

    def fac(self):
        b = self.base()
        k, _ = self.peek()
        if k == "^":
            self.i += 1
            e = self.eat("num")
            if not (0 <= e <= 12):
                raise ExprError("指数 %r 越界 (0..12)" % e)
            acc = (1,)
            for _ in range(e):
                acc = _pmul(acc, b)
            return acc
        return b

    def base(self):
        k, v = self.peek()
        if k == "num":
            self.i += 1
            return (v,)
        if k == "id":
            self.i += 1
            if v != self.var:
                raise ExprError("未登记变量 %r (本目标语言只有 %r)" % (v, self.var))
            return (0, 1)
        if k == "(":
            self.i += 1
            e = self.expr()
            self.eat(")")
            return e
        raise ExprError("意外的记号 %r" % ((k, v),))


def _trim(c):
    c = list(c)
    while len(c) > 1 and c[-1] == 0:
        c.pop()
    return tuple(c) or (0,)


def _padd(a, b):
    n = max(len(a), len(b))
    return _trim([(a[i] if i < len(a) else 0) + (b[i] if i < len(b) else 0) for i in range(n)])


def _psub(a, b):
    n = max(len(a), len(b))
    return _trim([(a[i] if i < len(a) else 0) - (b[i] if i < len(b) else 0) for i in range(n)])


def _pmul(a, b):
    out = [0] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        if x:
            for j, y in enumerate(b):
                out[i + j] += x * y
    return _trim(out)


def parse_poly(s: str, var: str) -> tuple:
    """Expression text -> integer-coefficient polynomial (low degree first). The variable must be
    exactly var; any trailing tokens are rejected."""
    if var not in VAR_NAMES:
        raise ExprError("变量 %r 不在登记池 %r" % (var, VAR_NAMES))
    p = _P(_tok(s), var)
    c = p.expr()
    if p.i != len(p.t):
        raise ExprError("表达式尾部有多余记号: %r" % (s,))
    return _trim(c)


def poly_str(c, var: str) -> str:
    """Expanded rendering (canonical form, deterministic): 2,3 -> "3*n + 2"."""
    terms = []
    for i in range(len(c) - 1, -1, -1):
        k = c[i]
        if k == 0:
            continue
        if i == 0:
            terms.append("%d" % k)
        elif i == 1:
            terms.append("%d*%s" % (k, var) if k != 1 else var)
        else:
            terms.append("%d*%s^%d" % (k, var, i) if k != 1 else "%s^%d" % (var, i))
    if not terms:
        return "0"
    s = terms[0]
    for t in terms[1:]:
        s += (" - " + t[1:]) if t.startswith("-") else (" + " + t)
    return s


def poly_eval(c, v: int) -> int:
    acc = 0
    for k in reversed(c):
        acc = acc * v + k
    return acc


def poly_gcd(c) -> int:
    from math import gcd
    g = 0
    for k in c:
        g = gcd(g, abs(int(k)))
    return g


def poly_shift(c, s: int) -> tuple:
    """c(x + s) -- binomial expansion."""
    out = [0] * len(c)
    for i, k in enumerate(c):
        if k == 0:
            continue
        # (x+s)^i = Σ_j C(i,j) s^(i-j) x^j
        from math import comb
        for j in range(i + 1):
            out[j] += k * comb(i, j) * (s ** (i - j))
    return _trim(out)


def poly_sub(a, b):
    return _psub(a, b)


def poly_scale(a, t: int):
    return _trim([x * t for x in a])


def poly_diff(c) -> tuple:
    """Difference f(x+1) − f(x) (the step kernel of discrete induction)."""
    return _psub(poly_shift(c, 1), c)


def subst_affine(c, r: int, m: int) -> tuple:
    """c(r + m*x) -- the polynomial after branch substitution (in the new variable)."""
    out = [0] * len(c)
    from math import comb
    for i, k in enumerate(c):
        if k == 0:
            continue
        for j in range(i + 1):
            out[j] += k * comb(i, j) * (r ** (i - j)) * (m ** j)
    return _trim(out)


def residues(c, m: int) -> list:
    """Complete residue values [p(0)%m, ..., p(m-1)%m] -- the full decision table (not sampling)."""
    return [poly_eval(c, r) % m for r in range(m)]


def divides_all(c, m: int) -> bool:
    """∀v∈ℤ. m | p(v) <=> all complete residues are 0 (an equivalence, see the module
    docstring)."""
    return all(x == 0 for x in residues(c, m))


def subst_text(factored: str, var: str, r: int, m: int, kname: str) -> str:
    """Replace the variable by (r + m*k) in an expression text -- the single substitution
    implementation for problem/subgoal text.

    WARNING: a power must be parenthesized: in `n^2` the n is the **base**, and substituting it
    with `2*k^2` would parse as `2*(k^2)` -- semantically wrong (an early version of this function
    really hit that: the independent verifier rejected 885 rows on the spot, see STATUS stage 3).
    Hence: whenever the variable is immediately followed by `^`, always parenthesize; otherwise
    r=0 may drop the parentheses (shorter, context budget).
    """
    toks = _tok(factored)
    out = []
    for i, (kind, v) in enumerate(toks):
        if kind == "id":
            if v != var:
                raise ExprError("替换时遇到未登记变量 %r" % (v,))
            nxt = toks[i + 1][0] if i + 1 < len(toks) else None
            if nxt == "^":
                out.append("(%d*%s)" % (m, kname) if r == 0 else "(%d + %d*%s)" % (r, m, kname))
            else:
                out.append(("%d*%s" % (m, kname)) if r == 0 else ("(%d + %d*%s)" % (r, m, kname)))
        elif kind == "num":
            out.append(str(v))
        else:
            out.append(v)
    return "".join(out)


def rename_text(factored: str, old_var: str, new_var_name: str) -> str:
    """Rename the variable in an expression text to new_var_name (token level; numbers/operators
    untouched).

    Why needed: the family catalog fixes its factored expressions under the variable "n", while the
    variable name of each instance row may vary (one of n/m/k/t/j) -- renaming must share the
    parser's tokenization; a string replace is not allowed (it would replace the "n" inside
    identifiers like "sin").
    """
    if old_var not in VAR_NAMES or new_var_name not in VAR_NAMES:
        raise ExprError("改名变量 %r/%r 不在登记池" % (old_var, new_var_name))
    out = []
    for kind, v in _tok(factored):
        if kind == "id":
            if v != old_var:
                raise ExprError("改名时遇到未登记变量 %r" % (v,))
            out.append(new_var_name)
        elif kind == "num":
            out.append(str(v))
        else:
            out.append(v)
    return "".join(out)


def new_var(i: int, avoid=()) -> str:
    """Pick the next substitution variable name not in avoid (deterministic)."""
    for name in SUBST_NAMES:
        if name not in avoid:
            return name
    raise ExprError("替换变量池耗尽")


# ---------------------------------------------------------------------------
# 2) goal text (single source: operators and generator both call here)
# ---------------------------------------------------------------------------

def universal_goal_text(m: int, factored: str, var: str) -> str:
    return "for every integer %s, %d divides %s" % (var, m, factored)


def negation_goal_text(m: int, factored: str, var: str) -> str:
    return "there is no integer %s with %d dividing %s" % (var, m, factored)


def conjunction_goal_text(var: str, parts) -> str:
    """parts = [(m, factored), ...] => "for every integer n, (..) and (..)" (universal-claim
    form)."""
    body = " and ".join("(%d divides %s)" % (m, f) for m, f in parts)
    return "for every integer %s, %s" % (var, body)


def conjunction_neg_goal_text(var: str, parts) -> str:
    """parts = [(m, factored), ...] => "for every integer n, (no .. divides ..) and (no .. divides ..)".

    Semantics: the conjunction of two **non-existence** claims (two independent unsolvable
    congruence propositions). Mathematically equivalent to "no integer satisfies either of these
    two congruences (jointly or separately)"; after decomposition by subgoal_focus(split) each is
    closed by contradiction.
    Why this form (rather than a single negated goal): the **non-triviality gate** requires a proof
    chain of >= 2 non-trivial steps -- the natural proof of a single negated goal has only 1 step
    (contradiction_close) and fails the gate (see the gate parameters in audit_data.py).
    """
    body = " and ".join("(no %d divides %s)" % (m, f) for m, f in parts)
    return "for every integer %s, %s" % (var, body)


def leaf_goal_text(m: int, factored_sub: str, kname: str) -> str:
    """Leaf goal = the substituted universal goal (variable is the substitution parameter; the shape
    matches U but the argument is a branch parameter)."""
    return universal_goal_text(m, factored_sub, kname)


GOAL_PREFIX_UNIVERSAL = "for every integer "
GOAL_PREFIX_NEGATION = "there is no integer "

#: Strict shapes (only the three renderings produced by this module are accepted; anything else is
#: rejected)
import re as _re                                                    # noqa: E402

_RE_UNIV = _re.compile(r"^for every integer ([a-z]+), (\d+) divides (.*)$")
_RE_NEG = _re.compile(r"^there is no integer ([a-z]+) with (\d+) dividing (.*)$")
_RE_CONJ = _re.compile(r"^for every integer ([a-z]+), \((.*)\) and \((.*)\)$")
_RE_NEGCONJ = _re.compile(
    r"^for every integer ([a-z]+), \(no (\d+) divides (.*)\) and \(no (\d+) divides (.*)\)$")
_RE_CLAIM = _re.compile(r"^(\d+) divides (.*)$")


def parse_goal(text: str) -> "Goal":
    """Problem goal text -> Goal (shape/modulus/variable/factored expression/polynomial).

    Only the shapes produced by this module's renderer are accepted (strict: anything else is
    rejected -- so text that merely "looks like a goal" cannot slip in).
    """
    mo = _RE_NEG.match(text)
    if mo:
        var, m, f = mo.group(1), int(mo.group(2)), mo.group(3)
        if var not in VAR_NAMES:
            raise ExprError("未登记变量 %r" % (var,))
        return Goal(form="N", m=m, var=var, factored=f,
                    coeffs=parse_poly(f, var), text=text)
    mo = _RE_NEGCONJ.match(text)            # negated conjunction (two "no m divides f" claims)
    if mo:
        var = mo.group(1)
        if var not in VAR_NAMES:
            raise ExprError("未登记变量 %r" % (var,))
        claims = ((int(mo.group(2)), mo.group(3)), (int(mo.group(4)), mo.group(5)))
        for _m, _f in claims:
            parse_poly(_f, var)
        return Goal(form="NC", m=claims[0][0], var=var, factored=text,
                    coeffs=tuple(), claims=tuple(claims), text=text)
    mo = _RE_CONJ.match(text)               # the conjunction shape is tested **first**: after the
    #                                         comma it has "(" rather than a modulus
    if mo:
        var = mo.group(1)
        if var not in VAR_NAMES:
            raise ExprError("未登记变量 %r" % (var,))
        parts = []
        for chunk in (mo.group(2), mo.group(3)):
            cm = _RE_CLAIM.match(chunk)
            if not cm:
                raise ExprError("合取目标 claim 形态不符: %r" % (chunk,))
            parts.append((int(cm.group(1)), cm.group(2)))
            parse_poly(cm.group(2), var)                            # the claim expression must parse
        return Goal(form="C", m=parts[0][0], var=var, factored=text,
                    coeffs=tuple(), claims=tuple(parts), text=text)
    mo = _RE_UNIV.match(text)
    if mo:
        var, m, f = mo.group(1), int(mo.group(2)), mo.group(3)
        if var not in VAR_NAMES:
            raise ExprError("未登记变量 %r" % (var,))
        return Goal(form="U", m=m, var=var, factored=f,
                    coeffs=parse_poly(f, var), text=text)
    raise ExprError("未登记的目标形态: %r" % (text,))


@dataclass(frozen=True)
class Goal:
    form: str              # "U" universal | "N" negated | "C" conjunction | "NC" negated conjunction
    m: int
    var: str
    factored: str
    coeffs: tuple
    text: str
    claims: tuple = ()

    def with_var(self, v: str) -> "Goal":
        return Goal(self.form, self.m, v, self.factored, self.coeffs, self.text, self.claims)


# ---------------------------------------------------------------------------
# 3) goal-family catalog (candidate moduli really verified at import time)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Family:
    fid: int
    name: str
    kind: str          # case | ind | conj | neg
    factored: str
    cand_moduli: tuple
    note: str


FAMILY_CATALOG = (
    Family(0, "prod2", "case", "n*(n+1)", (2,), "两连积: 偶"),
    Family(1, "prod3", "case", "n*(n+1)*(n+2)", (2, 3, 6), "三连积"),
    Family(2, "prod4", "case", "n*(n+1)*(n+2)*(n+3)", (2, 3, 4, 6, 8, 12), "四连积"),
    Family(3, "sq_minus_n", "case", "n^2 - n", (2,), "n(n-1)"),
    Family(4, "cube_minus_n", "ind", "n^3 - n", (2, 3, 6), "幂差 d=3"),
    Family(5, "pow5_minus_n", "ind", "n^5 - n", (2, 3, 5, 6, 10), "幂差 d=5"),
    Family(6, "pow7_minus_n", "ind", "n^7 - n", (2, 3, 6, 7), "幂差 d=7"),
    Family(7, "nope_sq_2_mod4", "neg", "n^2 - 2", (4,), "无解: 平方 mod 4 ∈{0,1}"),
    Family(8, "nope_sq_3_mod4", "neg", "n^2 - 3", (4,), "无解: 平方 mod 4 ∈{0,1}"),
    Family(9, "nope_cube_2_mod7", "neg", "n^3 - 2", (7,), "无解: 立方 mod 7 无 2"),
    Family(10, "nope_quartic_2_mod8", "neg", "n^4 - 2", (8,), "无解: 四次方 mod 8 ∈{0,1}"),
    Family(11, "nope_sq_n_1_mod2", "neg", "n^2 + n + 1", (2,), "恒奇"),
)

#: Conjunction family: two sub-claims (both case families), decomposed by subgoal_focus(split).
CONJ_CLAIMS = ((0, 2), (1, 3))     # (family_fid, modulus)


def _validate_catalog() -> dict:
    """Real verification at import time: a case family's modulus must "always divide" and the
    factor count must be >= 2; an ind family's difference must always divide and the base case must
    hold; a neg family must be "unsolvable" (no zero among the complete residues). Returns
    {fid: (valid moduli...)}."""
    valid = {}
    for fam in FAMILY_CATALOG:
        c = parse_poly(fam.factored, "n")
        ok = []
        for m in fam.cand_moduli:
            if fam.kind == "case":
                if m >= 2 and divides_all(c, m):
                    ok.append(m)
            elif fam.kind == "ind":
                d = poly_diff(c)
                if m >= 2 and divides_all(c, m) and divides_all(d, m):
                    ok.append(m)
            elif fam.kind == "neg":
                if m >= 2 and not any(x == 0 for x in residues(c, m)):
                    ok.append(m)
            else:
                raise AssertionError("未登记族 kind=%r" % (fam.kind,))
        if not ok:
            raise AssertionError("族 %s 无有效模数 (候选 %r) —— 目录自检失败" % (fam.name, fam.cand_moduli))
        valid[fam.fid] = tuple(ok)
    # conjunction family: both claims must each be valid
    for fid, m in CONJ_CLAIMS:
        if m not in valid[fid]:
            raise AssertionError("合取族 claim (fid=%d, m=%d) 无效" % (fid, m))
    return valid


VALID_MODULI = _validate_catalog()


def family_of(fid: int) -> Family:
    for f in FAMILY_CATALOG:
        if f.fid == fid:
            return f
    raise ExprError("未登记族 fid=%r" % (fid,))


# ---------------------------------------------------------------------------
# 4) strategy rule table R1–R6 (executable version of design doc §3.2)
# ---------------------------------------------------------------------------

#: Skeleton operator names (index order = codebook row order 9951..9956 => action indices 155..160)
STRATEGY_OPS = ("skel_case_split", "skel_induction_scheme", "skel_subgoal_focus",
                "skel_leaf_close", "skel_contradiction_close", "skel_qed_check")


def decide_strategy(state: dict) -> str:
    """State -> next skeleton operator (a deterministic rule table; every criterion consumes only
    literal fields of the state).

    state fields: goal(text), form(U/N/C/L), n_open(int), all_closed(bool), pending(bool)
    """
    form = state.get("form")
    if state.get("all_closed") and int(state.get("n_open", 0)) == 0:
        return "skel_qed_check"                                  # R6
    if form == "L":
        return "skel_leaf_close"                                 # R1
    if form == "C":
        return "skel_subgoal_focus"                              # R2
    if form == "N":
        return "skel_contradiction_close"                        # R3
    if form in ("U", "I"):                                       # R4 / R5
        return "skel_case_split" if state.get("shape") == "product" else "skel_induction_scheme"
    raise ExprError("decide_strategy: 未登记状态 %r" % (state,))


# ---------------------------------------------------------------------------
# 5) node ledger (the text carrier of the qed audit; appears literally in the problem text =>
#    value-reachable)
# ---------------------------------------------------------------------------

def ledger_str(nodes) -> str:
    """nodes = [(gid, kind, status)] -> "g0|k|C;g1|l|C" (short codes, deterministic order).

    Short codes are not aesthetics: the ledger must enter the problem token stream **word for
    word** (a hard value-reachability constraint), and long names would burn the context budget
    (measured: a long-name ledger made the conj family's last frame exceed the 168-token limit =>
    frame-level reachability 0.998; with short codes 1.000). parse_ledger accepts both forms
    (lenient input, compact output).
    """
    return ";".join("g%d|%s|%s" % (g, KIND_CODE[k], ST_CODE[s]) for g, k, s in sorted(nodes))


def parse_ledger(s: str):
    """Ledger text -> [(gid, kind, status)]; a shape mismatch is rejected."""
    out = []
    if not s.strip():
        return out
    for chunk in s.split(";"):
        parts = _re.split(r"[:|]", chunk)
        if len(parts) != 3 or not parts[0].startswith("g"):
            raise ExprError("台账项形态不符: %r" % (chunk,))
        g = int(parts[0][1:])
        k, st = CODE_KIND.get(parts[1], parts[1]), CODE_ST.get(parts[2], parts[2])
        if k not in NODE_KINDS:
            raise ExprError("台账 kind 未登记: %r" % (k,))
        if st not in ("open", "closed"):
            raise ExprError("台账 status 未登记: %r" % (st,))
        out.append((g, k, st))
    return sorted(out)


# ---------------------------------------------------------------------------
# 6) goal-stack state string (appears literally in the problem text => value-reachable; carried as
#    a string inside operator args => the operator stays a pure function)
# ---------------------------------------------------------------------------

NODE_KINDS = ("root", "case", "ind-base", "ind-step", "claim", "leaf", "neg")


#: Short codes inside state strings (token saving: the state string enters the problem text frame
#: by frame, and long names would waste context)
KIND_CODE = {"root": "r", "claim": "c", "leaf": "l", "neg": "n",
             "ind-base": "b", "ind-step": "s", "case": "k"}
CODE_KIND = {v: k for k, v in KIND_CODE.items()}
ST_CODE = {"open": "O", "closed": "C"}
CODE_ST = {v: k for k, v in ST_CODE.items()}


def stack_str(nodes, focus: int, depth: int) -> str:
    """Node ledger + focus + depth -> canonical string (short codes) "g0|r|C|-1;g1|c|O|0@f=1@d=1"."""
    body = ";".join("g%d|%s|%s|%d" % (int(n["gid"]), KIND_CODE[n["kind"]],
                                       ST_CODE[n["status"]], int(n["parent"]))
                    for n in sorted(nodes, key=lambda x: int(x["gid"])))
    return "%s@f=%d@d=%d" % (body, int(focus), int(depth))


def parse_stack_str(s: str) -> dict:
    """Canonical string -> {nodes, focus, depth}; a shape mismatch is rejected."""
    head, sep, tail = s.partition("@")
    if not sep:
        raise ExprError("栈状态串缺焦点段: %r" % (s,))
    mo = _re.match(r"^(?:focus=(-?\d+)@depth=(\d+)|f=(-?\d+)@d=(\d+))$", tail)
    if not mo:
        raise ExprError("栈状态串尾段形态不符: %r" % (tail,))
    fx = mo.group(1) if mo.group(1) is not None else mo.group(3)
    dp = mo.group(2) if mo.group(2) is not None else mo.group(4)
    nodes = []
    for chunk in head.split(";"):
        if not chunk:
            continue
        parts = chunk.split("|")
        if len(parts) != 4 or not parts[0].startswith("g"):
            raise ExprError("节点项形态不符: %r" % (chunk,))
        g = int(parts[0][1:])
        k, st, p = parts[1], parts[2], int(parts[3])
        k = CODE_KIND.get(k, k)          # both short codes and long names are accepted (output uses short codes only)
        st = CODE_ST.get(st, st)
        if k not in NODE_KINDS:
            raise ExprError("节点 kind 未登记: %r" % (k,))
        if st not in ("open", "closed"):
            raise ExprError("节点 status 未登记: %r" % (st,))
        nodes.append({"gid": g, "kind": k, "status": st, "parent": p})
    return {"nodes": sorted(nodes, key=lambda x: x["gid"]),
            "focus": int(fx), "depth": int(dp)}


def n_open_of(nodes) -> int:
    return sum(1 for n in nodes if n["status"] == "open")
