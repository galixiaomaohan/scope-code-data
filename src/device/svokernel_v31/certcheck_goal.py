# -*- coding: utf-8 -*-
"""Layer 3 invocation mechanism -- **independent checkers** for the certificates (4 kinds).

The objects of this layer are terms/formulas rather than numbers, so the independence requirement
has a special shape: the checkers carry **their own syntactic-structure handling** (substitution /
evaluation / pattern matching) and import neither svokernel's logiccore nor the generator module.
The syntax (one grammar) is shared, the code is not -- if the two sides disagree on some input,
logiccore is the root of trust (the logical root of trust of this package is svokernel.logiccore;
the checkers are merely the party that **independently recomputes**).

Terms/formulas in payloads are passed as **structures** (nested lists) and the checkers do no
grammar parsing -- so there is no risk surface of "two parsers disagreeing"; independence is
concentrated on the three semantic operations: substitution, evaluation, matching.
"""
from __future__ import annotations

import functools
from fractions import Fraction

from svokernel.certify import register_checker

__all__ = ["CHECKER_KINDS"]


def _safe(fn):
    @functools.wraps(fn)
    def wrapped(payload):
        try:
            return fn(payload)
        except Exception as e:  # noqa: BLE001
            return False, [], "检查器拒收: payload 非法或检查过程异常 (%s: %s)" % (type(e).__name__, e)

    return wrapped


# ---------------------------------------------------------------------------
# independent structural handling of terms (first-order) and formulas (propositions)
# ---------------------------------------------------------------------------


def _t(x):
    """list -> tuple normalization (payload structures arrive as lists; semantically tuples)."""
    if isinstance(x, (list, tuple)):
        return tuple(_t(i) for i in x)
    return x


def _term_subst(t, sigma):
    """Independently implemented term substitution: ("v",n) looks up σ; ("c",n) unchanged;
    ("f",fn,args) recurses."""
    if t[0] == "v":
        return sigma.get(t[1], t)
    if t[0] == "c":
        return t
    return ("f", t[1], tuple(_term_subst(a, sigma) for a in t[2]))


def _term_vars(t):
    if t[0] == "v":
        return {t[1]}
    if t[0] == "c":
        return set()
    out = set()
    for a in t[2]:
        out |= _term_vars(a)
    return out


def _walk_t(t, sigma):
    """One-step walk of a term under σ (returns a **term** structure, not a variable name)."""
    t = _t(t)
    while t[0] == "v" and t[1] in sigma:
        nt = _t(sigma[t[1]])
        if nt == t:
            break
        t = nt
    return t


def _resolve(t, sigma, depth=0):
    """**Recursively unfold** a term under σ (deep-copy semantics, σ is never modified); the
    depth cap guards against pathological self-reference explosion."""
    t = _t(t)
    if depth > 64:
        return t
    if t[0] == "v":
        return _resolve(sigma[t[1]], sigma, depth + 1) if t[1] in sigma else t
    if t[0] in ("c", "atom"):
        return t
    if t[0] == "not":
        return ("not", _resolve(t[1], sigma, depth + 1))
    if t[0] in ("and", "or", "imp"):
        return (t[0], _resolve(t[1], sigma, depth + 1), _resolve(t[2], sigma, depth + 1))
    if t[0] == "f":
        return ("f", t[1], tuple(_resolve(a, sigma, depth + 1) for a in t[2]))
    return t


def _bind(var_t, t, sigma):
    """Variable binding (with occurs check).

    B17 fix: the occurs check must run on the **unfolded** term. Counterexample (selfcheck
    ga_occurs_check): σ1 = {thm0_x ↦ g(?y)}; then bind ?y to g(?thm0_x) -- without unfolding the
    variable set of t is only {thm0_x} and y stays invisible; after unfolding t = g(g(?y)) which
    contains y => the real cycle is caught (agreeing with logiccore).
    """
    var = var_t[1]
    t = _resolve(_walk_t(t, sigma), sigma)
    cur = _walk_t(("v", var), sigma)
    if cur != ("v", var):
        return _unify_t(cur, t, sigma)
    if t == ("v", var):
        return sigma
    if var in _term_vars(t):
        return None                      # occurs check (decided on the unfolded term)
    out = dict(sigma)
    out[var] = t
    return out


def _unify_t(a, b, sigma):
    """The checker's **own** first-order unification (occurs-check version): returns σ or None.

    Why the checker needs its own unifier: "the conclusion does **not** unify with the goal" is a
    **universally quantified negative proposition** -- only by really running unification once can
    it be independently re-checked; seeing an empty σ in the certificate cannot distinguish "it
    really does not unify" from "the generator was lazy and never tried". This implementation does
    structural unification only (tuple shapes of terms/formulas) and **parses no grammar**, so its
    code is independent of logiccore's unifier (syntax shared, code not shared -- see the module
    docstring).
    """
    a, b = _t(a), _t(b)
    if not a or not b:
        return None
    if a[0] == "v":
        return _bind(a, b, sigma)
    if b[0] == "v":
        return _bind(b, a, sigma)
    if a[0] in ("c", "atom") or b[0] in ("c", "atom"):
        return sigma if a == b else None
    if a[0] != b[0]:
        return None
    if a[0] == "f":
        if a[1] != b[1] or len(a[2]) != len(b[2]):
            return None
        pa, pb = a[2], b[2]
    elif a[0] == "not":
        pa, pb = (a[1],), (b[1],)
    elif a[0] in ("and", "or", "imp"):
        pa, pb = (a[1], a[2]), (b[1], b[2])
    else:
        return None
    cur = sigma
    for x, y in zip(pa, pb):
        cur = _unify_t(x, y, cur)
        if cur is None:
            return None
    return cur


def _pat_vars(pat):
    """Variable-name set of a schema-library pattern (pattern variables look like ("v", "?A"))."""
    pat = _t(pat)
    if pat[0] == "v":
        return {pat[1]}
    if pat[0] in ("c", "atom"):
        return set()
    if pat[0] == "not":
        return _pat_vars(pat[1])
    return _pat_vars(pat[1]) | _pat_vars(pat[2])


def _formula_atoms(f):
    k = f[0]
    if k == "atom":
        return {f[1]}
    if k == "not":
        return _formula_atoms(f[1])
    return _formula_atoms(f[1]) | _formula_atoms(f[2])


def _eval_formula(f, assign):
    """Independently implemented proposition evaluation (bool semantics, no short-circuit
    reliance)."""
    k = f[0]
    if k == "atom":
        return bool(assign[f[1]])
    if k == "not":
        return not _eval_formula(f[1], assign)
    a = _eval_formula(f[1], assign)
    b = _eval_formula(f[2], assign)
    if k == "and":
        return a and b
    if k == "or":
        return a or b
    if k == "imp":
        return (not a) or b
    raise ValueError("未知公式结点 %r" % (k,))


def _equiv_exhaustive(forms, atoms, cap=12):
    """Enumerate all assignments and check that all formulas are semantically equivalent;
    returns (ok, n_rows, first counterexample)."""
    atoms = sorted(atoms)
    if len(atoms) > cap:
        raise ValueError("原子数 %d 超检查器穷举上限 %d" % (len(atoms), cap))
    n = len(atoms)
    for mask in range(1 << n):
        assign = {a: bool((mask >> i) & 1) for i, a in enumerate(atoms)}
        vals = [_eval_formula(f, assign) for f in forms]
        if any(v != vals[0] for v in vals):
            return False, 1 << n, assign
    return True, 1 << n, None


def _unify_pat(pat, concrete, sigma):
    """The checker's **structural match** (used only for the "lemma shape ⊆ pattern library"
    check, not a general unifier): pattern variables look like ("v", "?A"); matching concrete
    against them records A ↦ subterm."""
    pat = _t(pat)
    concrete = _t(concrete)
    if pat[0] == "v":
        nm = pat[1]
        if nm in sigma:
            return sigma[nm] == concrete
        sigma[nm] = concrete
        return True
    if pat[0] != concrete[0]:
        return False
    if pat[0] in ("c",):
        return pat[1] == concrete[1]
    if pat[0] == "atom":
        return pat[1] == concrete[1]
    if pat[0] == "not":
        return _unify_pat(pat[1], concrete[1], sigma)
    if pat[0] in ("and", "or", "imp"):
        return _unify_pat(pat[1], concrete[1], sigma) and _unify_pat(pat[2], concrete[2], sigma)
    if pat[0] == "f":
        if pat[1] != concrete[1] or len(pat[2]) != len(concrete[2]):
            return False
        return all(_unify_pat(a, b, sigma) for a, b in zip(pat[2], concrete[2]))
    return False


# ---------------------------------------------------------------------------
# 1) goal-theorem application certificate
# ---------------------------------------------------------------------------


@register_checker("thm_apply_witness", independent_of=("goal_apply_theorem",))
@_safe
def _check_thm_apply_witness(payload):
    """Theorem application: the checker applies σ to conclusion/premises itself and compares goal
    and subgoals symbol by symbol."""
    goal = payload["goal"]
    concl = payload["conclusion"]
    prem = payload["premises"]
    subg = payload["subgoals"]
    sigma_raw = payload["sigma"]
    bad = []
    sigma = {}
    for k, v in sigma_raw.items():
        sigma[k] = v          # values are already structures (list/tuple)
    # the scope of σ: theorem variables or goal variables only
    allowed = _term_vars(concl) | _term_vars(goal)
    for v in prem:
        allowed |= _term_vars(v)
    # B15 fix: two legitimate conclusions -- applied (σ(conclusion) = goal) and not_applicable
    # (conclusion and goal do not unify).
    if payload.get("mode", "applied") == "not_applicable":
        if sigma or subg:
            bad.append("not_applicable 却给出 σ/子目标 (越权声称)")
        # independently re-check the universal negative: really run the checker's **own** unifier
        if _unify_t(concl, goal, {}) is not None:
            bad.append("not_applicable 不成立: 检查器自带合一器给出了 σ ⇒ 结论与目标其实合一")
        return (not bad), [float(len(bad))], (
            "not_applicable 独立复核: 自带合一器确认结论与目标不合一, 且未越权给出 σ/子目标"
            if not bad else "; ".join(bad))
    for k in sigma:
        if k not in allowed:
            bad.append("σ 的定义域含未知变量 %r (不在定理/目标变量集内)" % (k,))
    if _term_subst(concl, sigma) != goal:
        bad.append("σ(结论) ≠ 目标 —— 应用不成立")
    if len(subg) != len(prem):
        bad.append("子目标数 %d ≠ 前提数 %d" % (len(subg), len(prem)))
    for i, pr in enumerate(prem):
        want = _term_subst(pr, sigma)
        if i < len(subg) and want != subg[i]:
            bad.append("子目标 %d ≠ σ(前提) (%r vs %r)" % (i, subg[i], want))
    # σ idempotence (self-consistency on its range): σ(t) == t for t ∈ range(σ)
    for k, v in sigma.items():
        if _term_subst(v, sigma) != v:
            bad.append("σ 非幂等: σ(%s 的值) 再作用仍有变化" % k)
    return (not bad), [float(len(bad))], (
        "σ(结论) = 目标 逐符号成立; %d 个子目标 = σ(前提); σ 幂等" % len(subg)
        if not bad else "; ".join(bad))


# ---------------------------------------------------------------------------
# 2) lemma instantiation certificate
# ---------------------------------------------------------------------------


@register_checker("lemma_instance", independent_of=("lemma_instantiate",))
@_safe
def _check_lemma_instance(payload):
    """Lemma instantiation: independently compute σ(statement), the residual free variables, and
    the legality of the σ keys."""
    stmt = payload["statement"]
    inst = payload["instance"]
    residual = payload["residual_vars"]
    sigma = payload["sigma"]
    bad = []
    declared = _term_vars(stmt)
    for k in sigma:
        if k not in declared:
            bad.append("σ 键 %r 不是引理的自由变量 (实例化越权)" % (k,))
    got = _term_subst(stmt, sigma)
    if got != inst:
        bad.append("σ(statement) ≠ 实例 (%r vs %r)" % (inst, got))
    want_res = sorted(declared - set(sigma))
    if sorted(residual) != want_res:
        bad.append("残余自由变量 %r ≠ 独立计算 %r" % (sorted(residual), want_res))
    if _term_vars(inst) != set(residual):
        bad.append("实例的自由变量集 ≠ 残余变量集")
    return (not bad), [float(len(bad))], (
        "σ(statement) 逐符号一致; 残余变量 %r 独立复算一致" % (residual,)
        if not bad else "; ".join(bad))


# ---------------------------------------------------------------------------
# 3) goal transformation certificate (semantic equivalence = exhaustive truth table)
# ---------------------------------------------------------------------------


@register_checker("equiv_truth_table", independent_of=("goal_transform",))
@_safe
def _check_equiv_truth_table(payload):
    """Goal transformation: the checker's own evaluator + exhaustive assignment enumeration
    verifying goal <=> AND(subgoals). It also performs a **structural check** per mode (the
    transformation shape must really be that mode's variant)."""
    goal = _t(payload["goal"])
    subs = [_t(s) for s in payload["subgoals"]]
    mode = payload["mode"]
    bad = []
    if not subs:
        bad.append("空子目标集 (变换必须给出至少一个子目标)")
    # semantics: goal <=> (sub1 ∧ sub2 ∧ …), compared assignment by assignment
    atoms = _formula_atoms(goal)
    for s in subs:
        atoms |= _formula_atoms(s)
    conj = None
    for s in subs:
        conj = s if conj is None else ("and", conj, s)
    ok, rows, cex = _equiv_exhaustive([goal, conj], atoms)
    if not ok:
        bad.append("语义等价失败 (反例指派 %r)" % (cex,))
    if rows != (1 << len(atoms)):
        bad.append("穷举行数 %d ≠ 2^%d" % (rows, len(atoms)))
    # structural check
    if mode == "contrapositive":
        if goal[0] != "imp" or len(subs) != 1:
            bad.append("contrapositive 要求目标是蕴含且恰一子目标")
        elif subs[0] != ("imp", ("not", goal[2]), ("not", goal[1])):
            bad.append("子目标不是逆否式 ¬B → ¬A")
    elif mode == "iff_split":
        if subs != [("imp", goal[1][1], goal[1][2]), ("imp", goal[2][1], goal[2][2])]:
            bad.append("iff_split 的子目标不是两个方向 (A→B, B→A)")
    elif mode == "de_morgan_dual":
        if goal[0] != "not" or goal[1][0] not in ("and", "or"):
            bad.append("de_morgan_dual 要求 ¬(A∘B) 形状")
        else:
            g = goal[1]
            want = ("or", ("not", g[1]), ("not", g[2])) if g[0] == "and" else \
                   ("and", ("not", g[1]), ("not", g[2]))
            if subs != [want]:
                bad.append("对偶式不是 De Morgan 那一款")
    elif mode == "conj_split":
        if goal[0] != "and" or subs != [goal[1], goal[2]]:
            bad.append("conj_split 要求目标是合取且子目标 = 两个合取项")
    else:
        bad.append("未登记 mode %r" % mode)
    return (not bad), [float(len(bad))], (
        "真值表 %d 行 (原子 %r) 语义等价 + mode=%s 结构核对 通过" % (rows, sorted(atoms), mode)
        if not bad else "; ".join(bad))


# ---------------------------------------------------------------------------
# 4) lemma discovery certificate
# ---------------------------------------------------------------------------


#: The checker's own **pattern-library copy** (same table as the generator, independent code):
#: used for the structural check.
_SCHEMA_PATTERNS = {
    "S1_or_intro_left": ("imp", ("v", "?A"), ("or", ("v", "?A"), ("v", "?B"))),
    "S2_and_elim_left": ("imp", ("and", ("v", "?A"), ("v", "?B")), ("v", "?A")),
    "S3_and_elim_right": ("imp", ("and", ("v", "?A"), ("v", "?B")), ("v", "?B")),
    "S4_weakening": ("imp", ("v", "?A"), ("imp", ("v", "?B"), ("v", "?A"))),
    "S5_hyp_syllogism": ("imp", ("imp", ("v", "?A"), ("v", "?B")),
                         ("imp", ("imp", ("v", "?B"), ("v", "?C")),
                          ("imp", ("v", "?A"), ("v", "?C")))),
    "S6_modus_ponens": ("imp", ("and", ("imp", ("v", "?A"), ("v", "?B")), ("v", "?A")), ("v", "?B")),
    "S7_de_morgan_and": ("imp", ("not", ("and", ("v", "?A"), ("v", "?B"))),
                         ("or", ("not", ("v", "?A")), ("not", ("v", "?B")))),
    "S9_contrapositive": ("imp", ("imp", ("v", "?A"), ("v", "?B")),
                          ("imp", ("not", ("v", "?B")), ("not", ("v", "?A")))),
}


@register_checker("lemma_validity", independent_of=("lemma_discover",))
@_safe
def _check_lemma_validity(payload):
    """Lemma discovery: independently verify item by item (i) validity (exhaustive truth table),
    (ii) the shape belongs to the pattern library, (iii) usefulness (atoms ⊆ goal atoms, and
    non-trivial)."""
    goal = _t(payload["goal"])
    lemmas = [_t(x) for x in payload["lemmas"]]
    goal_atoms = _formula_atoms(goal)
    bad = []
    # B14 fix: n_tried == 0 is not necessarily wrong -- when the goal's atom count is below the
    # **shortest pattern's arity** in the library, no pattern can be instantiated and 0 is the
    # only legal answer (an empty result, not an error; selfcheck ld_atom_only_goal is exactly
    # that case). Conversely, claiming 0 tries when enumeration was possible is a "false empty
    # result" => must be rejected. The criterion is **independently recomputed** by the checker
    # (using its own pattern-library copy to count the minimum variable count, reading nothing
    # from the generator).
    if _SCHEMA_PATTERNS:
        min_arity = min(len(_pat_vars(pat)) for pat in _SCHEMA_PATTERNS.values())
    else:
        min_arity = 0
    n_tried = int(payload.get("n_tried", -1))
    if n_tried < 0:
        bad.append("未记录候选尝试数")
    elif n_tried == 0 and len(goal_atoms) >= min_arity:
        bad.append("目标原子数 %d >= 模式最短元数 %d, 却声称 0 次候选 (谎报空结果)"
                   % (len(goal_atoms), min_arity))
    for f in lemmas:
        atoms = _formula_atoms(f)
        # validity: f is true under every assignment
        f_atoms = sorted(atoms)
        if len(f_atoms) > 12:
            bad.append("引理原子数超限: %r" % (f_atoms,))
            continue
        valid = True
        for mask in range(1 << len(f_atoms)):
            assign = {a: bool((mask >> i) & 1) for i, a in enumerate(f_atoms)}
            if not _eval_formula(f, assign):
                valid = False
                break
        if not valid:
            bad.append("引理非恒真: %r" % (f,))
        if not atoms <= goal_atoms:
            bad.append("引理原子集 %r ⊄ 目标原子集 %r (无用)" % (sorted(atoms), sorted(goal_atoms)))
        if f == goal:
            bad.append("引理与目标逐字相同 (不构成辅助引理)")
        if f[0] == "atom" or (f[0] == "not" and f[1][0] == "atom"):
            bad.append("引理是单个文字 (平凡, 无用)")
        # the shape must match one library pattern (structural match, own pattern copy)
        hit = None
        for sid, pat in _SCHEMA_PATTERNS.items():
            if _unify_pat(pat, f, {}):
                hit = sid
                break
        if hit is None:
            bad.append("引理形状不在模式库内: %r" % (f,))
    if payload.get("n_useful") != len(lemmas):
        bad.append("n_useful %r ≠ 实际引理数 %d" % (payload.get("n_useful"), len(lemmas)))
    return (not bad), [float(len(bad))], (
        "%d 条引理: 恒真 + 形状属模式库 + 有用性 全部独立复核" % len(lemmas)
        if not bad else "; ".join(bad[:4]))


CHECKER_KINDS = ("thm_apply_witness", "lemma_instance", "equiv_truth_table", "lemma_validity")
