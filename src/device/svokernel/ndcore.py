"""ND kernel and logic-domain certificate checkers — the **structural** (not declarative) realisation of §17.9 "no self-certification".

**Why a separate module**: the selfcheck gate (svokernel.selfcheck) checks "is the certificate
checker's module part of the generator set". The two checkers `nd_proof` and `cex_witness` used to
live in `ops/logic_ops.py` — the very same module as the tactic engine / counter-example search that
generates them. Even though the checking functions themselves take the kernel-decision path, the fact
"same module" already degrades "independence" into a self-declaration. Moving the kernel and checkers
here turns independence into a filesystem fact: generators in `ops/logic_ops.py`, checkers in
`svokernel/ndcore.py`, and this module **does not import** any generator operator.

Three things:
  1. **ND kernel (trust root)**: `infer(node)` recursively decides a natural-deduction proof term and
     returns (conclusion formula, undischarged assumption set); failure raises POSTCOND_FAIL. The kernel
     **decides only, never searches**.
  2. **Certificate checkers**: `nd_proof` (proof term passes the kernel) and `cex_witness` (assignment
     satisfies all hypotheses and refutes the goal). Both consume payload data only and never touch any
     generator state.
  3. **Formula utilities**: canon/imp/not/subst/contains — re-imported by logic_ops, guaranteeing that
     "the part shared between generator and checker" is only formula syntax, not decision logic.

ND rule set (classical propositional logic; ⊥ is the reserved atom "FALSE_"):
  assume / axiom(A1|A2|A3, reusing the existing trust root logiccore.is_axiom_instance) /
  imp_i(discharged) / imp_e / and_i / and_e1 / and_e2 / or_i1 / or_i2 /
  or_e(two discharged) / neg_i(discharged) / neg_e / raa(discharged) / efq /
  cases(atomic LEM — the propositional "case split") / replace(propositional replacement theorem)

Soundness argument: every rule is a valid inference of classical propositional logic. cases = P∨¬P
holds for atoms; replace = if ⊢src→dst and ⊢dst→src, then G[src:=dst] and G are equivalent (the
propositional replacement theorem). Honesty boundary: this kernel is **propositional**; universal
generalisation / first-order instantiation are not part of it (those go through §8.6 Lean or the
§10.1 logiccore Hilbert kernel + universal generalisation).
"""
from __future__ import annotations

from .certify import CHECKERS
from .errors import Err, SvoError
from .logiccore import formula_str, is_axiom_instance, parse_formula

FALSE_ATOM = "FALSE_"
"""Reserved atom name for ⊥ (rejected if it collides with a user atom — see the reserved-word check in check_nd)."""

ND_RULES: tuple[str, ...] = (
    "assume", "axiom", "imp_i", "imp_e", "and_i", "and_e1", "and_e2", "or_i1", "or_i2",
    "or_e", "neg_i", "neg_e", "raa", "efq", "cases", "replace",
)


# ---------------------------------------------------------------------------
# formula-syntax utilities (shared by generators and checkers; syntax only, no decision)
# ---------------------------------------------------------------------------


def canon(f: tuple) -> str:
    """Formula → canonical string (= logiccore.formula_str, with **atom names preserved verbatim**).

    Why not a compact canon that "renames atoms in order of first occurrence": that canon is
    **non-injective** — atoms a and b both map to the same string, so the set comparison "undischarged
    assumptions ⊆ given hypotheses" in check_nd would mistake a's assumption for b's, and a proof about a
    could impersonate a proof about b. That is a **silent wrong value**, exactly what §7.6 fail-safe
    forbids. Hence we reuse the trust root's name-preserving implementation directly (the parser already
    rejects illegal characters outside reserved names, so injectivity is guaranteed by the literal form).
    """
    return formula_str(f)


def canon_str(s: str, ctx: str) -> str:
    return canon(parse_formula(s))


def imp_formula(a: tuple, b: tuple) -> tuple:
    return ("imp", a, b)


def not_formula(a: tuple) -> tuple:
    return ("not", a)


def subst_formula(f: tuple, src: tuple, dst: tuple) -> tuple:
    """G[src := dst]: replace every subformula of G isomorphic to src by dst wholesale."""
    if f == src:
        return dst
    k = f[0]
    if k in ("atom",):
        return f
    if k == "not":
        return ("not", subst_formula(f[1], src, dst))
    return (k, subst_formula(f[1], src, dst), subst_formula(f[2], src, dst))


def contains(f: tuple, sub: tuple) -> bool:
    if f == sub:
        return True
    k = f[0]
    if k == "atom":
        return False
    if k == "not":
        return contains(f[1], sub)
    return contains(f[1], sub) or contains(f[2], sub)


def atoms_of(f: tuple) -> set:
    """Set of atom names appearing in the formula (excluding the reserved atom ⊥)."""
    out: set = set()

    def walk(x: tuple) -> None:
        k = x[0]
        if k == "atom":
            if x[1] != FALSE_ATOM:
                out.add(x[1])
            return
        if k == "not":
            walk(x[1])
            return
        walk(x[1])
        walk(x[2])

    walk(f)
    return out


# ---------------------------------------------------------------------------
# 1) ND kernel (trust root: decides only, never searches)
# ---------------------------------------------------------------------------


def infer(node: dict) -> "tuple[tuple, frozenset]":
    """Node → (conclusion formula AST, set of undischarged assumptions).

    Decides only, never searches: the caller supplies the whole proof-term structure, and the kernel
    only checks rule by rule whether each step is legal. Any structural or rule mismatch raises
    POSTCOND_FAIL (never a silent accept).
    """
    if not isinstance(node, dict) or "rule" not in node:
        raise SvoError(Err.POSTCOND_FAIL, ctx="nd_kernel", detail=f"非法证明节点: {node!r}")
    r = node["rule"]

    def P(s):
        return parse_formula(s)

    def C(f):
        return canon(f)

    if r == "assume":
        f = P(node["f"])
        return f, frozenset({C(f)})
    if r == "axiom":
        f = P(node["f"])
        if is_axiom_instance(f) is None:
            raise SvoError(Err.POSTCOND_FAIL, ctx="nd_kernel",
                           detail=f"axiom 节点非 A1/A2/A3 实例: {C(f)}")
        return f, frozenset()
    if r == "imp_i":
        f = P(node["f"])
        d = P(node["discharge"])
        if f[0] != "imp" or f[1] != d:
            raise SvoError(Err.POSTCOND_FAIL, ctx="nd_kernel",
                           detail=f"imp_i: f ≠ ({C(d)} -> _)")
        sub, as_ = infer(node["sub"])
        if sub != f[2]:
            raise SvoError(Err.POSTCOND_FAIL, ctx="nd_kernel",
                           detail=f"imp_i: 子证明结论 {C(sub)} ≠ {C(f[2])}")
        return f, as_ - {C(d)}
    if r == "imp_e":
        f = P(node["f"])
        maj, a1 = infer(node["maj"])
        mino, a2 = infer(node["min"])
        if maj[0] != "imp" or maj[1] != mino:
            raise SvoError(Err.POSTCOND_FAIL, ctx="nd_kernel",
                           detail=f"imp_e: maj={C(maj)} 的前件 ≠ mino={C(mino)}")
        if f != maj[2]:
            raise SvoError(Err.POSTCOND_FAIL, ctx="nd_kernel",
                           detail=f"imp_e: 声称结论 {C(f)} ≠ 实际结论 {C(maj[2])}")
        return maj[2], a1 | a2
    if r == "and_i":
        f = P(node["f"])
        l, a1 = infer(node["l"])
        rr, a2 = infer(node["r"])
        if f[0] != "and" or f[1] != l or f[2] != rr:
            raise SvoError(Err.POSTCOND_FAIL, ctx="nd_kernel", detail="and_i: 子结论与 f 不匹配")
        return f, a1 | a2
    if r in ("and_e1", "and_e2"):
        f = P(node["f"])
        of, a = infer(node["of"])
        if of[0] != "and":
            raise SvoError(Err.POSTCOND_FAIL, ctx="nd_kernel", detail=f"{r}: of 非合取")
        want = of[1] if r == "and_e1" else of[2]
        if f != want:
            raise SvoError(Err.POSTCOND_FAIL, ctx="nd_kernel", detail=f"{r}: f ≠ 目标分量")
        return f, a
    if r in ("or_i1", "or_i2"):
        f = P(node["f"])
        of, a = infer(node["of"])
        if f[0] != "or":
            raise SvoError(Err.POSTCOND_FAIL, ctx="nd_kernel", detail=f"{r}: f 非析取")
        want = f[1] if r == "or_i1" else f[2]
        if of != want:
            raise SvoError(Err.POSTCOND_FAIL, ctx="nd_kernel", detail=f"{r}: of ≠ 析取分量")
        return f, a
    if r == "or_e":
        f = P(node["f"])
        dj, ad = infer(node["disj"])
        if dj[0] != "or":
            raise SvoError(Err.POSTCOND_FAIL, ctx="nd_kernel", detail="or_e: disj 非析取")
        d1 = P(node["d1"])
        d2 = P(node["d2"])
        if dj[1] != d1 or dj[2] != d2:
            raise SvoError(Err.POSTCOND_FAIL, ctx="nd_kernel", detail="or_e: 分情况公式与析取式不符")
        s1, a1 = infer(node["sub1"])
        s2, a2 = infer(node["sub2"])
        if s1 != f or s2 != f:
            raise SvoError(Err.POSTCOND_FAIL, ctx="nd_kernel", detail="or_e: 两支结论 ≠ f")
        return f, ad | (a1 - {C(d1)}) | (a2 - {C(d2)})
    if r == "neg_i":
        f = P(node["f"])
        d = P(node["discharge"])
        if f[0] != "not" or f[1] != d:
            raise SvoError(Err.POSTCOND_FAIL, ctx="nd_kernel", detail="neg_i: f ≠ ~discharge")
        sub, a = infer(node["sub"])
        if sub != ("atom", FALSE_ATOM):
            raise SvoError(Err.POSTCOND_FAIL, ctx="nd_kernel", detail="neg_i: 子证明未导出 ⊥")
        return f, a - {C(d)}
    if r == "neg_e":
        maj, a1 = infer(node["maj"])
        mino, a2 = infer(node["min"])
        if maj[0] != "not" or maj[1] != mino:
            raise SvoError(Err.POSTCOND_FAIL, ctx="nd_kernel", detail="neg_e: maj ≠ ~mino")
        if P(node["f"]) != ("atom", FALSE_ATOM):
            raise SvoError(Err.POSTCOND_FAIL, ctx="nd_kernel", detail="neg_e: 声称结论非 ⊥")
        return ("atom", FALSE_ATOM), a1 | a2
    if r == "raa":
        f = P(node["f"])
        d = P(node["discharge"])
        if d[0] != "not" or d[1] != f:
            raise SvoError(Err.POSTCOND_FAIL, ctx="nd_kernel", detail="raa: discharge ≠ ~f")
        sub, a = infer(node["sub"])
        if sub != ("atom", FALSE_ATOM):
            raise SvoError(Err.POSTCOND_FAIL, ctx="nd_kernel", detail="raa: 子证明未导出 ⊥")
        return f, a - {C(d)}
    if r == "efq":
        f = P(node["f"])
        of, a = infer(node["of"])
        if of != ("atom", FALSE_ATOM):
            raise SvoError(Err.POSTCOND_FAIL, ctx="nd_kernel", detail="efq: of 不是 ⊥")
        return f, a
    if r == "cases":
        f = P(node["f"])
        p = P(node["atom"])
        if p[0] != "atom" or p[1] == FALSE_ATOM:
            raise SvoError(Err.POSTCOND_FAIL, ctx="nd_kernel", detail="cases: atom 须为原子")
        pos, a1 = infer(node["pos"])
        neg, a2 = infer(node["neg"])
        if pos != f or neg != f:
            raise SvoError(Err.POSTCOND_FAIL, ctx="nd_kernel", detail="cases: 两支结论 ≠ f")
        return f, (a1 - {C(p)}) | (a2 - {C(("not", p))})
    if r == "replace":
        f = P(node["f"])
        src = P(node["src"])
        dst = P(node["dst"])
        fwd, a1 = infer(node["eq_fwd"])
        bwd, a2 = infer(node["eq_bwd"])
        of, a3 = infer(node["of"])
        if fwd != imp_formula(src, dst) or bwd != imp_formula(dst, src):
            raise SvoError(Err.POSTCOND_FAIL, ctx="nd_kernel",
                           detail="replace: eq_fwd/eq_bwd 不是 src→dst / dst→src")
        if subst_formula(of, src, dst) != f:
            raise SvoError(Err.POSTCOND_FAIL, ctx="nd_kernel",
                           detail="replace: G[src:=dst] ≠ f (替换定理前提不满足)")
        return f, a1 | a2 | a3
    raise SvoError(Err.POSTCOND_FAIL, ctx="nd_kernel", detail=f"未知 ND 规则 {r!r}")


def check_nd(term: dict, goal: str, hyps: "list[str]") -> "tuple[bool, str]":
    """Top-level ND kernel decision: term ⊢ goal, and every undischarged assumption ⊆ hyps."""
    concl, assumptions = infer(term)
    goal_f = parse_formula(goal)
    if concl != goal_f:
        return False, f"证明项结论 {canon(concl)} ≠ 目标 {canon(goal_f)}"
    reserved = sorted(a for a in assumptions if FALSE_ATOM in a)
    allowed = {canon_str(h, "nd") for h in hyps}
    extra = sorted(assumptions - allowed)
    if extra:
        return False, f"未消解假设 {extra} ⊄ 给定假设"
    if reserved:
        return False, f"证明项依赖保留原子 ⊥ 但未消解: {reserved}"
    return True, f"ND 核通过: 结论 {canon(concl)}, 假设闭合 ({len(allowed)} 条)"


# ---------------------------------------------------------------------------
# 2) semantic evaluator (for the cex_witness checker; a third implementation, distinct from
#    both logic_ops paths)
# ---------------------------------------------------------------------------


def ev_shortcircuit(f: tuple, env: dict) -> bool:
    """Short-circuit evaluation: node-by-node boolean operations, unlike either the recursive or the
    postfix-stack implementation in logic_ops.

    The reserved atom ⊥ is constantly false (⊥ has no model in the assignment domain).
    """
    k = f[0]
    if k == "atom":
        if f[1] == FALSE_ATOM:
            return False
        if f[1] not in env:
            raise SvoError(Err.SCHEMA_MISMATCH, ctx="ndcore.ev",
                           detail=f"指派缺原子 {f[1]!r} (求值不可偏定)")
        return bool(env[f[1]])
    if k == "not":
        return not ev_shortcircuit(f[1], env)
    if k == "and":
        return ev_shortcircuit(f[1], env) and ev_shortcircuit(f[2], env)
    if k == "or":
        return ev_shortcircuit(f[1], env) or ev_shortcircuit(f[2], env)
    if k == "imp":
        # material interpretation: (¬A) ∨ B — a different path from the node-style implementation
        return (not ev_shortcircuit(f[1], env)) or ev_shortcircuit(f[2], env)
    raise SvoError(Err.SCHEMA_MISMATCH, ctx="ndcore.ev", detail=f"未知连接词 {k!r}")


def falsifies(hyps: "list[str]", goal: str, assignment: dict) -> "tuple[bool, str]":
    """Is the assignment a counter-example: all hypotheses true and the goal false."""
    env = {str(k): bool(v) for k, v in assignment.items()}
    try:
        hs = [ev_shortcircuit(parse_formula(h), env) for h in hyps]
        g = ev_shortcircuit(parse_formula(goal), env)
    except SvoError as e:
        return False, f"指派不适定: {e.detail}"
    ok = all(hs) and not g
    return ok, (f"指派满足全部 {len(hyps)} 条假设且反驳目标" if ok
                else f"假设真值表 {hs}, 目标 {g} —— 不构成反例")


# ---------------------------------------------------------------------------
# 3) certificate-checker registration (this module ≠ generator module → §17.9 independence is
#    a filesystem fact)
# ---------------------------------------------------------------------------


def _register_nd_checker() -> None:
    if CHECKERS.has("nd_proof"):
        return

    def _check(payload: dict) -> "tuple[bool, list, str]":
        ok, detail = check_nd(payload["term"], payload["goal"], payload.get("hyps", []))
        return ok, [0.0 if ok else 1.0], "nd_proof: " + detail

    CHECKERS.register("nd_proof", _check,
                      independent_of=("logic_tactic_step", "logic_proof_search"),
                      module=__name__)


def _register_cex_checker() -> None:
    if CHECKERS.has("cex_witness"):
        return

    def _check(payload: dict) -> "tuple[bool, list, str]":
        ok, detail = falsifies(payload["hyps"], payload["goal"], payload["assignment"])
        return ok, [0.0 if ok else 1.0], "cex_witness: " + detail

    CHECKERS.register("cex_witness", _check,
                      independent_of=("logic_counterexample",), module=__name__)


_register_nd_checker()
_register_cex_checker()
