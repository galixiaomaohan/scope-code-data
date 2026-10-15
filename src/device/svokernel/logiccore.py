"""Logic trust root (§10.1) — a small hand-written kernel that decides only, never searches, and can be audited line by line.

Contains: terms and substitution / unification / propositional formulas and CNF / resolution /
Hilbert proof kernel / SAT assignment checking.

**Trust-root principle** (§5): the verification chain ultimately collapses onto a small, auditable
checker. This module is that checker: all functions are pure decision functions, with no search
heuristics (search lives on the operator side), and every failure returns an explicit signal
(False/None) instead of silently passing.

Conventions:
  - terms: ("v", name) variable | ("c", name) constant | ("f", fn, (arg,...)) application
  - variables are written "?x"; sigma keys are variable names without '?'
  - propositional formula AST: ("atom", name) | ("not", f) | ("and", f, g) | ("or", f, g) | ("imp", f, g)
  - Hilbert axiom schemas (metavariables A/B/C):
      A1: A → (B → A)
      A2: (A → (B → C)) → ((A → B) → (A → C))
      A3: (¬A → ¬B) → (B → A)
    The only derivation rule is MP. The kernel does **not trust** the caller's claim "this is an axiom
    instance" — it does the pattern matching itself.
"""
from __future__ import annotations

import re
from itertools import combinations
from typing import Iterable

from .errors import Err, SvoError

# ---------------------------------------------------------------------------
# terms (first-order)
# ---------------------------------------------------------------------------

Term = tuple


def V(name: str) -> Term:
    return ("v", name.lstrip("?"))


def C(name: str) -> Term:
    return ("c", name)


def F(fn: str, *args: Term) -> Term:
    return ("f", fn, tuple(args))


_TERM_TOKEN = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*|\?)\s*")


def parse_term(s: str) -> Term:  # noqa: C901 - hand-written recursive descent, kept single-exit for clarity
    """Parse a term string: "f(?x, g(a, ?y))" / "?x" / "a"."""
    pos = 0
    n = len(s)

    def skip() -> None:
        nonlocal pos
        while pos < n and s[pos].isspace():
            pos += 1

    def ident() -> str:
        nonlocal pos
        skip()
        m = re.match(r"[A-Za-z_][A-Za-z0-9_]*", s[pos:])
        if not m:
            raise SvoError(Err.UNIFY_FAIL, ctx="parse_term", detail=f"位置 {pos} 处期望标识符: {s!r}")
        pos += m.end()
        return m.group(0)

    def term() -> Term:
        nonlocal pos
        skip()
        if pos < n and s[pos] == "?":
            pos += 1
            return V(ident())
        name = ident()
        skip()
        if pos < n and s[pos] == "(":
            pos += 1
            args = [term()]
            skip()
            while pos < n and s[pos] == ",":
                pos += 1
                args.append(term())
                skip()
            if pos >= n or s[pos] != ")":
                raise SvoError(Err.UNIFY_FAIL, ctx="parse_term", detail=f"缺右括号: {s!r}")
            pos += 1
            return F(name, *args)
        return C(name)

    t = term()
    skip()
    if pos != n:
        raise SvoError(Err.UNIFY_FAIL, ctx="parse_term", detail=f"位置 {pos} 起有残余: {s!r}")
    return t


def term_str(t: Term) -> str:
    if t[0] == "v":
        return "?" + t[1]
    if t[0] == "c":
        return t[1]
    return f"{t[1]}(" + ", ".join(term_str(a) for a in t[2]) + ")"


def walk_vars(t: Term) -> set[str]:
    if t[0] == "v":
        return {t[1]}
    if t[0] == "c":
        return set()
    out: set[str] = set()
    for a in t[2]:
        out |= walk_vars(a)
    return out


def apply_subst(t: Term, sigma: dict[str, Term]) -> Term:
    """Substitution σ applied to a term. Deep-copy semantics (returns a new structure); variables bound in σ are not substituted again (simultaneous substitution)."""
    if t[0] == "v":
        return sigma.get(t[1], t)
    if t[0] == "c":
        return t
    return ("f", t[1], tuple(apply_subst(a, sigma) for a in t[2]))


def _occurs(v: str, t: Term) -> bool:
    return v in walk_vars(t)


def unify(a: Term, b: Term, sigma: "dict[str, Term] | None" = None, depth: int = 0) -> "dict[str, Term] | None":
    """Unification (with occurs check). Returns the MGU or None (not unifiable).

    Failure returns None instead of raising — a decision function should not express the normal
    conclusion "not unifiable" through an exception; the caller (the unify operator) maps None to UNIFY_FAIL.
    """
    if depth > 512:
        return None
    s = dict(sigma or {})
    stack: list[tuple[Term, Term]] = [(a, b)]
    while stack:
        x, y = stack.pop()
        x = apply_subst(x, s)
        y = apply_subst(y, s)
        if x == y:
            continue
        if x[0] == "v":
            if _occurs(x[1], y):
                return None
            s[x[1]] = y
            continue
        if y[0] == "v":
            if _occurs(y[1], x):
                return None
            s[y[1]] = x
            continue
        if x[0] == "c" or y[0] == "c":
            if x != y:
                return None
            continue
        if x[1] != y[1] or len(x[2]) != len(y[2]):
            return None
        stack.extend(zip(x[2], y[2]))
    return s


# ---------------------------------------------------------------------------
# propositional formulas / CNF / resolution
# ---------------------------------------------------------------------------

_ATOM = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def parse_formula(s: str) -> tuple:
    """Propositional formula recursive descent: ~, &, |, -> (right-associative), parentheses, atom names."""
    toks = _tokenize(s)
    pos = 0

    def peek() -> "str | None":
        return toks[pos] if pos < len(toks) else None

    def eat(t: str) -> bool:
        nonlocal pos
        if peek() == t:
            pos += 1
            return True
        return False

    def atom_or_paren() -> tuple:
        nonlocal pos
        t = peek()
        if t is None:
            raise SvoError(Err.SCHEMA_MISMATCH, ctx="parse_formula", detail=f"表达式意外结束: {s!r}")
        if t == "(":
            pos += 1
            f = imp()
            if not eat(")"):
                raise SvoError(Err.SCHEMA_MISMATCH, ctx="parse_formula", detail=f"缺右括号: {s!r}")
            return f
        pos += 1
        return ("atom", t)

    def unary() -> tuple:
        nonlocal pos
        if eat("~"):
            return ("not", unary())
        return atom_or_paren()

    def conj() -> tuple:
        nonlocal pos  # [P1 contractor minimal-skeleton fix 2026-09-20] conj/disj/imp lacked `nonlocal`,
        f = unary()  # so parsing any formula with & | -> raised UnboundLocalError; see delivery-report dissent section
        while peek() == "&":
            pos += 1
            f = ("and", f, unary())
        return f

    def disj() -> tuple:
        nonlocal pos
        f = conj()
        while peek() == "|":
            pos += 1
            f = ("or", f, conj())
        return f

    def imp() -> tuple:
        nonlocal pos
        f = disj()
        if peek() == "-" and pos + 1 < len(toks) and toks[pos + 1] == ">":
            pos += 2
            return ("imp", f, imp())
        return f

    res = imp()
    if pos != len(toks):
        raise SvoError(Err.SCHEMA_MISMATCH, ctx="parse_formula", detail=f"位置 {pos} 起有残余 token: {toks[pos:]}")
    return res


def _tokenize(s: str) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(s):
        c = s[i]
        if c.isspace():
            i += 1
            continue
        if c in "()~&|":
            out.append(c)
            i += 1
            continue
        if s.startswith("->", i):
            out.append("-")
            out.append(">")
            i += 2
            continue
        m = _ATOM.match(s, i)
        if not m:
            raise SvoError(Err.SCHEMA_MISMATCH, ctx="tokenize", detail=f"位置 {i} 处非法字符 {c!r}: {s!r}")
        out.append(m.group(0))
        i = m.end()
    return out


def formula_str(f: tuple) -> str:
    k = f[0]
    if k == "atom":
        return f[1]
    if k == "not":
        inner = formula_str(f[1])
        return f"~{inner}" if f[1][0] in ("atom", "not") else f"~({inner})"
    op = {"and": "&", "or": "|", "imp": "->"}[k]
    return f"({formula_str(f[1])} {op} {formula_str(f[2])})"


def to_cnf(f: tuple, max_clauses: int = 4096) -> tuple[list[frozenset[int]], dict[int, str], dict[str, int]]:
    """Formula → CNF (clause list + variable number↔name). Exceeding the cap raises BOUNDED_VIOLATION (no silent truncation)."""
    # 1) NNF (push ~ down to atoms)
    def nnf(x: tuple, neg: bool) -> tuple:
        k = x[0]
        if k == "atom":
            return ("not", x) if neg else x
        if k == "not":
            return nnf(x[1], not neg)
        if k == "and":
            a, b = nnf(x[1], neg), nnf(x[2], neg)
            return ("or" if neg else "and", a, b)
        if k == "or":
            a, b = nnf(x[1], neg), nnf(x[2], neg)
            return ("and" if neg else "or", a, b)
        # imp: a→b ≡ ~a|b
        a, b = nnf(x[1], not neg), nnf(x[2], neg)
        return ("and" if neg else "or", a, b)

    n = nnf(f, False)

    # 2) distribute to CNF (set form: or = literal set, and = clause set)
    def dist(x: tuple) -> list[frozenset[tuple]]:
        k = x[0]
        if k == "atom":
            return [frozenset([("pos", x[1])])]
        if k == "not":
            return [frozenset([("neg", x[1][1])])]
        if k == "or":
            left, right = dist(x[1]), dist(x[2])
            if len(left) * len(right) > max_clauses:
                raise SvoError(Err.BOUNDED_VIOLATION, ctx="to_cnf", detail=f"CNF 展开超上限 {max_clauses}")
            return [a | b for a in left for b in right]
        left, right = dist(x[1]), dist(x[2])
        if len(left) + len(right) > max_clauses:
            raise SvoError(Err.BOUNDED_VIOLATION, ctx="to_cnf", detail=f"CNF 子句数超上限 {max_clauses}")
        return left + right

    clauses_raw = dist(n)
    if len(clauses_raw) > max_clauses:
        raise SvoError(Err.BOUNDED_VIOLATION, ctx="to_cnf", detail=f"CNF 子句数 {len(clauses_raw)} 超上限 {max_clauses}")

    name2int: dict[str, int] = {}
    int2name: dict[int, str] = {}

    def vid(name: str) -> int:
        if name not in name2int:
            v = len(name2int) + 1
            name2int[name] = v
            int2name[v] = name
        return name2int[name]

    cnf: list[frozenset[int]] = []
    for cl in clauses_raw:
        lits = set()
        for sign, name in cl:
            v = vid(name)
            lits.add(v if sign == "pos" else -v)
        cnf.append(frozenset(lits))
    return cnf, int2name, name2int


def clause_str(cl: Iterable[int], int2name: dict[int, str]) -> str:
    if not cl:
        return "⊥"
    return " ∨ ".join(("" if l > 0 else "¬") + int2name.get(abs(l), str(abs(l))) for l in cl)


def resolve(c1: frozenset[int], c2: frozenset[int]) -> "list[frozenset[int]]":
    """Resolution: for each complementary literal pair produce one resolvent (§10.1 definition)."""
    out: list[frozenset[int]] = []
    for l in c1:
        if -l in c2:
            r = (c1 - {l}) | (c2 - {-l})
            if r not in out:
                out.append(r)
    return out


def unit_propagate(cnf: list[frozenset[int]], assign: dict[int, bool]) -> "tuple[bool, dict[int, bool]]":
    """Unit propagation. Returns (conflict?, assignment closure)."""
    a = dict(assign)
    changed = True
    while changed:
        changed = False
        for cl in cnf:
            sat = False
            unassigned: list[int] = []
            for l in cl:
                v = abs(l)
                if v in a:
                    if a[v] == (l > 0):
                        sat = True
                        break
                else:
                    unassigned.append(l)
            if sat:
                continue
            if not unassigned:
                return True, a  # empty clause → conflict
            if len(unassigned) == 1:
                l = unassigned[0]
                val = l > 0
                v = abs(l)
                if v in a and a[v] != val:
                    return True, a
                if v not in a:
                    a[v] = val
                    changed = True
    return False, a


def check_sat_assignment(cnf: Iterable[Iterable[int]], assign: dict[int, bool]) -> bool:
    """§10.6 verbatim: assignment re-verification, O(total literals)."""
    for cl in cnf:
        if not any(assign.get(abs(l), None) == (l > 0) for l in cl):
            return False
    return True


def resolution_refute(cnf: list[frozenset[int]], max_steps: int = 20000) -> tuple[bool, int, str]:
    """Resolution refutation (decides UNSAT): saturate resolution until the empty clause. Returns (found_empty, steps, detail).

    Only for **tiny** CNFs (selfchecks / small exam generation); large instances go through the DRAT certificate path.
    """
    work = [frozenset(c) for c in cnf]
    seen = set(work)
    steps = 0
    while steps < max_steps:
        new: list[frozenset[int]] = []
        for c1, c2 in combinations(work, 2):
            for r in resolve(c1, c2):
                steps += 1
                if len(r) == 0:
                    return True, steps, "导出空子句"
                if r not in seen:
                    seen.add(r)
                    new.append(r)
        if not new:
            return False, steps, "饱和且无空子句 (可满足)"
        work.extend(new)
    return False, steps, f"未在 {max_steps} 步内判定 (资源上限, 不宣称结论)"


# ---------------------------------------------------------------------------
# Hilbert proof kernel (§10.1) — decides only, never searches
# ---------------------------------------------------------------------------

_A1: tuple = ("imp", ("mv", "A"), ("imp", ("mv", "B"), ("mv", "A")))
_A2: tuple = (
    "imp",
    ("imp", ("mv", "A"), ("imp", ("mv", "B"), ("mv", "C"))),
    ("imp", ("imp", ("mv", "A"), ("mv", "B")), ("imp", ("mv", "A"), ("mv", "C"))),
)
_A3: tuple = ("imp", ("imp", ("not", ("mv", "A")), ("not", ("mv", "B"))), ("imp", ("mv", "B"), ("mv", "A")))
AXIOM_SCHEMAS: dict[str, tuple] = {"A1": _A1, "A2": _A2, "A3": _A3}


def match_schema(pattern: tuple, f: tuple, binding: "dict[str, tuple] | None" = None) -> "dict[str, tuple] | None":
    """Pattern matching (metavariables "mv" may bind any subformula; the same metavariable must bind the same subformula)."""
    b = dict(binding or {})
    stack: list[tuple[tuple, tuple]] = [(pattern, f)]
    while stack:
        p, x = stack.pop()
        if p[0] == "mv":
            name = p[1]
            if name in b:
                if b[name] != x:
                    return None
            else:
                b[name] = x
            continue
        if p[0] != x[0]:
            return None
        if p[0] in ("atom",):
            if p[1] != x[1]:
                return None
        elif p[0] == "not":
            stack.append((p[1], x[1]))
        else:
            stack.append((p[1], x[1]))
            stack.append((p[2], x[2]))
    return b


def is_axiom_instance(f: tuple) -> "str | None":
    """Return the matched axiom name (A1/A2/A3) or None."""
    for name, schema in AXIOM_SCHEMAS.items():
        if match_schema(schema, f) is not None:
            return name
    return None


def check_proof(hyps: list[str], steps: list[dict], goal: str) -> tuple[bool, int, str]:
    """Hilbert proof re-verification (§10.1): each step decided in O(1), decides only, never searches.

    steps[i] ∈
      {"rule":"hyp",   "idx":k, "formula":s}   s must == hyps[k]
      {"rule":"axiom", "formula":s}            s must be an instance of A1/A2/A3
      {"rule":"mp", "from":[i,j], "formula":s} {derived[i],derived[j]} = {p→q, p}, s == q
    goal: must appear in derived.
    Returns (ok, index of the first failing step, explanation).
    """
    hyps_f = []
    for i, h in enumerate(hyps):
        try:
            hyps_f.append(parse_formula(h))
        except SvoError as e:
            return False, -1, f"假设 {i} 解析失败: {e.detail}"
    derived: list[tuple] = []
    for i, st in enumerate(steps):
        rule = st.get("rule")
        try:
            f = parse_formula(st["formula"])
        except (SvoError, KeyError) as e:
            return False, i, f"步 {i} formula 非法: {e}"
        if rule == "hyp":
            k = st.get("idx")
            if not isinstance(k, int) or not (0 <= k < len(hyps_f)):
                return False, i, f"步 {i} hyp 索引越界: {k!r}"
            if f != hyps_f[k]:
                return False, i, f"步 {i} 声称假设 {k}, 但公式不等 ({formula_str(f)} ≠ {formula_str(hyps_f[k])})"
        elif rule == "axiom":
            ax = is_axiom_instance(f)
            if ax is None:
                return False, i, f"步 {i} 声称公理, 但非 A1/A2/A3 的实例: {formula_str(f)}"
        elif rule == "mp":
            fr = st.get("from")
            if not (isinstance(fr, list) and len(fr) == 2 and all(isinstance(j, int) for j in fr)):
                return False, i, f"步 {i} mp.from 非法: {fr!r}"
            j1, j2 = fr
            if not (0 <= j1 < i and 0 <= j2 < i):
                return False, i, f"步 {i} mp 引用非前置步 {fr!r} (禁前向引用/自引用)"
            p, q = derived[j1], derived[j2]
            ok = False
            cands = [(p, q), (q, p)]
            for ante, cons in cands:
                if ante[0] == "imp" and ante[2] == f and cons == ante[1]:
                    ok = True
                    break
            if not ok:
                return False, i, f"步 {i} MP 不成立: 由 {fr} 得不到 {formula_str(f)}"
        else:
            return False, i, f"步 {i} 未知规则 {rule!r} (只许 hyp/axiom/mp)"
        derived.append(f)
    if not derived:
        return False, -1, "证明为空 (无步骤不算证明)"
    g = parse_formula(goal)
    if g not in derived:
        return False, len(steps), f"目标 {formula_str(g)} 不在已推导公式中"
    return True, len(steps), f"{len(steps)} 步全部过核, 目标已导出"


# ---------------------------------------------------------------------------
# convenience: generate a SAT assignment from CNF (exhaustive, tiny instances)
# ---------------------------------------------------------------------------


def dpll_small(cnf: list[frozenset[int]], max_nodes: int = 200000) -> "dict[int, bool] | None":
    """Exhaustive solver for tiny instances (selfchecks and exam generation only; large instances go through the DRAT path)."""
    nvars = max((abs(l) for cl in cnf for l in cl), default=0)
    nodes = 0

    def rec(assign: dict[int, bool]) -> "dict[int, bool] | None":
        nonlocal nodes
        nodes += 1
        if nodes > max_nodes:
            raise SvoError(Err.BOUNDED_VIOLATION, ctx="dpll_small", detail=f"超节点上限 {max_nodes}")
        conflict, a = unit_propagate(cnf, assign)
        if conflict:
            return None
        unassigned = [v for v in range(1, nvars + 1) if v not in a]
        if not unassigned:
            return a if check_sat_assignment(cnf, a) else None
        v = unassigned[0]
        for val in (True, False):
            a2 = dict(a)
            a2[v] = val
            r = rec(a2)
            if r is not None:
                return r
        return None

    return rec({})
