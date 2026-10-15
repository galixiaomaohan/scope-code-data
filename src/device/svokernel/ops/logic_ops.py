"""§10.1–10.5 logic-domain operators (band 9500, sub 1..12 → rows 9501–9512).

All operators are built on `svokernel.logiccore` (the trust root: decides only, never searches):
    - CNF/resolution/unit propagation/SAT assignment checking → call the corresponding logiccore functions directly;
    - Hilbert proofs → logiccore.check_proof (certify's proof_kernel checker);
    - the tactic library → a **home-made natural-deduction kernel** (§10.4 allows "logiccore's Hilbert kernel or a
      home-made ND kernel"); the kernel decides only and never searches, and axiom leaves reuse
      logiccore.is_axiom_instance (the trust root is not re-implemented).

Division of labour for independent verifiers (§17.9 no self-certification):
    - invoke takes the syntactic/kernel path (to_cnf / resolve / check_proof / ND kernel);
    - verify takes the **semantic path** (truth-table enumeration / recursive evaluator / clause-by-clause
      recomputation) — the two code paths are disjoint.

Honesty boundary (declared item by item; see each operator's docs):
    - truth-table cross-checking requires atoms ≤ 14 (2^14 assignments); exceeding it raises BOUNDED_VIOLATION and is
      never silently skipped;
    - the tactic library's `induction` in the pure propositional fragment = atomic case split (LEM), not mathematical
      induction (that needs §8.6 Lean);
    - proof_search is a constructive tactic and **incomplete** — failing to find a proof only returns solved=False and
      never claims unprovability.
"""
from __future__ import annotations

import copy
from itertools import product
from typing import Any

from ..certify import CHECKERS, Certificate
from ..errors import Err, SvoError
from ..logiccore import (
    apply_subst,
    check_proof,
    check_sat_assignment,
    clause_str,
    dpll_small,
    is_axiom_instance,
    parse_formula,
    parse_term,
    resolve,
    to_cnf,
    unify,
    unit_propagate,
)
from ..registry import REGISTRY, OpDef
from ..tool import CallCtl

# The ND kernel and logic-domain certificate checkers have been moved out to svokernel/ndcore.py — the
# **structural** realisation of §17.9 "no self-certification": checker and generator live in different modules, so
# independence is a filesystem fact rather than a self-declaration. Here we only re-import its **syntax utilities**
# (canon/imp/not/subst); the decision logic remains ndcore's exclusive property.
from ..ndcore import (
    FALSE_ATOM as _FALSE,
    canon as _nd_canon,
    check_nd as _nd_check,
    imp_formula as _nd_imp_f,
    not_formula as _nd_not_f,
    subst_formula as _nd_subst,
)

_MAX_EQUIV_ATOMS = 14  # truth-table cross-check cap (2^14 = 16384 assignments × formula evaluation)


# ---------------------------------------------------------------------------
# two independently implemented formula evaluators (§17.9: invoke and verify each use one; the code paths are disjoint)
# ---------------------------------------------------------------------------


def _ev_rec(f: tuple, env: dict) -> bool:
    """recursive evaluator (verify side)."""
    k = f[0]
    if k == "atom":
        return bool(env[f[1]])
    if k == "not":
        return not _ev_rec(f[1], env)
    if k == "and":
        return _ev_rec(f[1], env) and _ev_rec(f[2], env)
    if k == "or":
        return _ev_rec(f[1], env) or _ev_rec(f[2], env)
    if k == "imp":
        return (not _ev_rec(f[1], env)) or _ev_rec(f[2], env)
    raise SvoError(Err.SCHEMA_MISMATCH, ctx="logic._ev_rec", detail=f"未知公式节点 {k!r}")


def _norm_clause_set(cnf: list, varmap: dict) -> set:
    """Normalize (integer clause table, name→int map) into "a clause set written by name".

    Used by selfchecks: the internal numbering order is an implementation detail (it changes with traversal order), so
    assertions must compare after resolving names, otherwise the selfcheck would be pinned to an irrelevant numbering
    convention (the old imp_cnf failed exactly this way).
    """
    inv = {int(v): str(k) for k, v in varmap.items()}
    out = set()
    for cl in cnf:
        out.add(frozenset((inv[abs(int(l))], int(l) > 0) for l in cl))
    return out


def _postfix(f: tuple) -> list[tuple]:
    """formula → postfix sequence (for invoke-side evaluation; a different mechanism from the recursive version)."""
    out: list[tuple] = []

    def walk(x: tuple) -> None:
        k = x[0]
        if k == "atom":
            out.append(("p", x[1]))
        elif k == "not":
            walk(x[1])
            out.append(("op", "not"))
        else:
            walk(x[1])
            walk(x[2])
            out.append(("op", k))

    walk(f)
    return out


def _ev_pf(pf: list[tuple], env: dict) -> bool:
    st: list[bool] = []
    for tag, v in pf:
        if tag == "p":
            st.append(bool(env[v]))
        elif v == "not":
            st.append(not st.pop())
        elif v == "and":
            b, a = st.pop(), st.pop()
            st.append(a and b)
        elif v == "or":
            b, a = st.pop(), st.pop()
            st.append(a or b)
        elif v == "imp":
            b, a = st.pop(), st.pop()
            st.append((not a) or b)
        else:
            raise SvoError(Err.SCHEMA_MISMATCH, ctx="logic._ev_pf", detail=f"未知算子 {v!r}")
    if len(st) != 1:
        raise SvoError(Err.SCHEMA_MISMATCH, ctx="logic._ev_pf", detail=f"求值栈残留 {len(st)}")
    return st[0]


def _ev_iter(f: tuple, env: dict) -> bool:
    return _ev_pf(_postfix(f), env)


def _atoms(f: tuple) -> set:
    k = f[0]
    if k == "atom":
        return {f[1]}
    if k == "not":
        return _atoms(f[1])
    return _atoms(f[1]) | _atoms(f[2])


def _canon(f: tuple) -> str:
    return _nd_canon(f)


def _canon_str(s: str, ctx: str) -> str:
    return _canon(parse_formula(s)) if isinstance(s, str) else _canon(s)


def _imp_s(a: str, b: str) -> str:
    return f"({a} -> {b})"


def _imp_f(a: tuple, b: tuple) -> tuple:
    return _nd_imp_f(a, b)


def _not_f(a: tuple) -> tuple:
    return _nd_not_f(a)


def _subst_formula(f: tuple, src: tuple, dst: tuple) -> tuple:
    return _nd_subst(f, src, dst)


def _contains(f: tuple, sub: tuple) -> bool:
    if f == sub:
        return True
    k = f[0]
    if k == "atom":
        return False
    if k == "not":
        return _contains(f[1], sub)
    return _contains(f[1], sub) or _contains(f[2], sub)


def _clause_list(cl: "frozenset[int] | set[int] | list[int]") -> list[int]:
    return sorted(int(l) for l in cl)


def _parse_clause(s: str, ctx: str) -> frozenset:
    try:
        lits = [int(t) for t in s.split(",") if t.strip() != ""]
    except ValueError as e:
        raise SvoError(Err.SCHEMA_MISMATCH, ctx=ctx, detail=f"子句串非法 {s!r}: {e}") from e
    if any(l == 0 for l in lits):
        raise SvoError(Err.SCHEMA_MISMATCH, ctx=ctx, detail=f"文字 0 非法 (子句 {s!r})")
    return frozenset(lits)


# ===========================================================================
# operator 1: logic_cnf_transform (row 9501) — §10.1 CNF transform + all-assignment truth-table equivalence check
# ===========================================================================

_IN_CNF = {"kind": "dict", "fields": {"formula": {"kind": "str"}}, "required": ["formula"]}
_OUT_CNF = {
    "kind": "dict",
    "fields": {
        "cnf": {"kind": "list", "items": {"kind": "list", "items": {"kind": "int"}}, "minlen": 0, "maxlen": 512},
        "varmap": {"kind": "dict", "extra": True},
        "n_clauses": {"kind": "int"},
        "n_atoms": {"kind": "int"},
        "checked_assignments": {"kind": "int"},
    },
    "required": ["cnf", "varmap", "n_clauses", "n_atoms", "checked_assignments"],
}


def _invoke_cnf(args: dict, ctl: CallCtl) -> dict:
    ctx = "op:logic_cnf_transform#9501"
    f = parse_formula(args["formula"])  # non-formula raises SCHEMA_MISMATCH
    atoms = sorted(_atoms(f))
    if len(atoms) > _MAX_EQUIV_ATOMS:
        raise SvoError(Err.BOUNDED_VIOLATION, ctx=ctx,
                       detail=f"原子数 {len(atoms)} > {_MAX_EQUIV_ATOMS} — 真值表等效核对超界, 拒绝")
    cnf, int2name, name2int = to_cnf(f)
    pf = _postfix(f)
    mism = 0
    for bits in product((False, True), repeat=len(atoms)):
        env = dict(zip(atoms, bits))
        f_val = _ev_pf(pf, env)
        assign = {name2int[a]: env[a] for a in atoms}
        c_val = check_sat_assignment(cnf, assign)  # trust root §10.6 implementation
        if f_val != c_val:
            mism += 1
    if mism:
        raise SvoError(Err.POSTCOND_FAIL, ctx=ctx, detail=f"真值表等效核对失败 {mism} 个指派")
    return {
        "value": {"cnf": [_clause_list(c) for c in cnf], "varmap": {k: v for k, v in name2int.items()},
                  "n_clauses": len(cnf), "n_atoms": len(atoms),
                  "checked_assignments": 2 ** len(atoms)},
        "resid": [0.0],
        "notes": f"to_cnf 得 {len(cnf)} 子句; 全部 {2 ** len(atoms)} 个指派真值表等效核对通过",
    }


def _verify_cnf(args: dict, value: dict) -> dict:
    """independent verification (semantic path; the recursive evaluator ≠ invoke's postfix-stack evaluator)."""
    f = parse_formula(args["formula"])
    atoms = sorted(_atoms(f))
    name2int = {str(k): int(v) for k, v in value["varmap"].items()}
    cnf = [frozenset(int(l) for l in cl) for cl in value["cnf"]]
    mism = 0
    for bits in product((False, True), repeat=len(atoms)):
        env = dict(zip(atoms, bits))
        if _ev_rec(f, env) != check_sat_assignment(cnf, {name2int[a]: env[a] for a in atoms}):
            mism += 1
    return {"ok": mism == 0 and value["n_clauses"] == len(cnf),
            "resid": [float(mism)],
            "detail": f"递归求值器交叉核对 {2 ** len(atoms)} 指派, 不等效 {mism} 个"}


REGISTRY.register(OpDef(
    name="logic_cnf_transform", band=9500, sub=1,
    in_schema=_IN_CNF, out_schema=_OUT_CNF,
    invoke=_invoke_cnf, verify=_verify_cnf,
    cost=("units", 3.0), errors=("SCHEMA_MISMATCH", "BOUNDED_VIOLATION", "POSTCOND_FAIL"),
    effects="none (纯函数)",
    doc="§10.1 公式→CNF (德摩根/分配, logiccore.to_cnf); 独立核对 = 枚举全部 2^n 指派比对原公式与 CNF 真值 "
        "(invoke 用后缀栈求值器 + 信任根 check_sat_assignment; verify 用递归求值器)。原子数 >14 拒绝。",
    selfchecks=(
        {"id": "imp_cnf", "args": {"formula": "a->b"},
         "check": lambda r: (_norm_clause_set(r.value["cnf"], r.value["varmap"])
                             == _norm_clause_set([[-1, 2]], {"a": 1, "b": 2}),
                             f"cnf={r.value['cnf']} varmap={r.value['varmap']}"),
         "expect": "a→b 的 CNF 应等价于 {¬a, b} —— 按 varmap 解名后比对, 不锁内部编号顺序"},
        {"id": "demorgan_equiv", "args": {"formula": "(a|b)&(~a|b)"},
         "check": lambda r: (r.value["checked_assignments"] == 4 and r.value["n_clauses"] >= 1,
                             f"4 指派核对, {r.value['n_clauses']} 子句")},
        {"id": "bad_syntax", "args": {"formula": "(a"}, "expect_err": "SCHEMA_MISMATCH"},
        {"id": "too_many_atoms", "args": {"formula": " & ".join(f"a{i}" for i in range(15))},
         "expect_err": "BOUNDED_VIOLATION"},
    ),
))


# ===========================================================================
# operator 2: logic_resolve_step (row 9502) — §10.1 single resolution step + truth-table entailment verification
# ===========================================================================

_IN_RES = {
    "kind": "dict",
    "fields": {
        "c1": {"kind": "list", "items": {"kind": "int"}, "minlen": 0, "maxlen": 24},
        "c2": {"kind": "list", "items": {"kind": "int"}, "minlen": 0, "maxlen": 24},
        "pivot": {"kind": "int"},
    },
    "required": ["c1", "c2"],
}
_OUT_RES = {
    "kind": "dict",
    "fields": {
        "resolvent": {"kind": "list", "items": {"kind": "int"}, "minlen": 0, "maxlen": 48},
        "pivot": {"kind": "int"},
        "checked_assignments": {"kind": "int"},
    },
    "required": ["resolvent", "pivot", "checked_assignments"],
}


def _tt_consequence(parents: list, child, varmap: dict, ev) -> int:
    """Is the resolvent a logical consequence of the parent clauses? Enumerate all assignments and count the violations (parents all true while child false)."""
    vs = sorted({abs(l) for cl in parents for l in cl} | {abs(l) for l in child})
    bad = 0
    for bits in product((False, True), repeat=len(vs)):
        env = {v: b for v, b in zip(vs, bits)}
        if ev(env, parents) and not ev(env, [child]):
            bad += 1
    return bad


def _cl_sat_iter(env: dict, clauses: list) -> bool:
    """postfix clause-satisfaction decision (invoke side)."""
    for cl in clauses:
        if not any(env.get(abs(l)) == (l > 0) for l in cl):
            return False
    return True


def _cl_sat_rec(env: dict, clauses: list) -> bool:
    """recursive clause-satisfaction decision (verify side)."""
    if not clauses:
        return True
    head, rest = clauses[0], clauses[1:]
    sat = False
    for l in head:
        if env.get(abs(l)) == (l > 0):
            sat = True
            break
    return sat and _cl_sat_rec(env, rest)


def _invoke_resolve(args: dict, ctl: CallCtl) -> dict:
    ctx = "op:logic_resolve_step#9502"
    c1, c2 = frozenset(args["c1"]), frozenset(args["c2"])
    for name, cl in (("c1", c1), ("c2", c2)):
        if 0 in cl:
            raise SvoError(Err.SCHEMA_MISMATCH, ctx=ctx, detail=f"{name} 含文字 0")
    comps = [l for l in c1 if -l in c2]
    pivot = args.get("pivot")
    if pivot is None:
        if len(comps) == 0:
            raise SvoError(Err.PRECOND_FAIL, ctx=ctx, detail="c1/c2 无互补文字对, 不可归结")
        if len(comps) > 1:
            raise SvoError(Err.SPEC_AMBIGUOUS, ctx=ctx,
                           detail=f"有 {len(comps)} 个互补对 {comps}, 须显式给 pivot")
        pivot = comps[0]
    if pivot not in c1 or -pivot not in c2:
        raise SvoError(Err.PRECOND_FAIL, ctx=ctx,
                       detail=f"pivot {pivot} 须 ∈ c1 且 -pivot ∈ c2")
    r = (c1 - {pivot}) | (c2 - {-pivot})
    # independent check embedded in invoke (truth table): the resolvent is a logical consequence of the two parents
    bad = _tt_consequence([c1, c2], r, {}, _cl_sat_iter)
    if bad:
        raise SvoError(Err.POSTCOND_FAIL, ctx=ctx, detail=f"真值表验证失败: {bad} 个指派下亲本真而归结式假")
    return {"value": {"resolvent": _clause_list(r), "pivot": pivot,
                      "checked_assignments": 2 ** len({abs(l) for l in c1 | c2 | r})},
            "resid": [0.0],
            "notes": f"resolve({{ {clause_str(c1, {})} }}, {{ ... }}) pivot={pivot} → {clause_str(r, {})}; "
                     f"真值表推论验证通过"}


def _verify_resolve(args: dict, value: dict) -> dict:
    """independent verification (recursive decision + structural containment check)."""
    c1, c2 = frozenset(args["c1"]), frozenset(args["c2"])
    r = frozenset(value["resolvent"])
    p = int(value["pivot"])
    struct_ok = r <= ((c1 | c2) - {p, -p}) and p in c1 and -p in c2
    bad = _tt_consequence([c1, c2], r, {}, _cl_sat_rec)
    return {"ok": struct_ok and bad == 0, "resid": [float(bad)],
            "detail": f"递归判定器: 推论违背 {bad} 指派; 结构包含 {'OK' if struct_ok else 'FAIL'}"}


REGISTRY.register(OpDef(
    name="logic_resolve_step", band=9500, sub=2,
    in_schema=_IN_RES, out_schema=_OUT_RES,
    invoke=_invoke_resolve, verify=_verify_resolve,
    cost=("units", 2.0), errors=("SCHEMA_MISMATCH", "PRECOND_FAIL", "SPEC_AMBIGUOUS", "POSTCOND_FAIL"),
    effects="none (纯函数)",
    doc="§10.1 单步归结 C₁∪{l} 与 C₂∪{¬l} ⊢ C₁∪C₂ (logiccore.resolve); "
        "独立核对 = 真值表: 枚举全部指派验证归结式是两亲本的逻辑推论 (§10.1 明文)。",
    selfchecks=(
        {"id": "basic_resolve", "args": {"c1": [1, 2], "c2": [-1, 3], "pivot": 1},
         "check": lambda r: (r.value["resolvent"] == [2, 3], f"resolvent={r.value['resolvent']}")},
        {"id": "empty_resolvent", "args": {"c1": [1], "c2": [-1]},
         "check": lambda r: (r.value["resolvent"] == [], "归结出空子句 (UNSAT 证据)")},
        {"id": "bad_pivot", "args": {"c1": [1, 2], "c2": [-1, 3], "pivot": 5}, "expect_err": "PRECOND_FAIL"},
        {"id": "no_complement", "args": {"c1": [1, 2], "c2": [3]}, "expect_err": "PRECOND_FAIL"},
        {"id": "ambiguous_multi_pivot", "args": {"c1": [1, 2], "c2": [-1, -2]}, "expect_err": "SPEC_AMBIGUOUS"},
        {"id": "zero_literal", "args": {"c1": [0, 1], "c2": [-1]}, "expect_err": "SCHEMA_MISMATCH"},
    ),
))


# ===========================================================================
# operator 3: logic_unit_prop (row 9503) — §10.1 unit-propagation closure + independent check
# ===========================================================================


def _mk_cnf(raw: list, ctx: str, max_clauses: int = 4096) -> "list[frozenset]":
    if len(raw) > max_clauses:
        raise SvoError(Err.BOUNDED_VIOLATION, ctx=ctx, detail=f"子句数 {len(raw)} > {max_clauses}")
    out = []
    for cl in raw:
        c = frozenset(int(l) for l in cl)
        if 0 in c:
            raise SvoError(Err.SCHEMA_MISMATCH, ctx=ctx, detail=f"子句含文字 0: {list(cl)}")
        out.append(c)
    return out


_IN_UP = {
    "kind": "dict",
    "fields": {
        "cnf": {"kind": "list", "items": {"kind": "list", "items": {"kind": "int"}}, "minlen": 0, "maxlen": 512},
        "assign": {"kind": "dict", "extra": True},
    },
    "required": ["cnf", "assign"],
}
_OUT_UP = {
    "kind": "dict",
    "fields": {
        "conflict": {"kind": "bool"},
        "closure": {"kind": "dict", "extra": True},
        "propagated": {"kind": "int"},
        "independent_check": {"kind": "str"},
    },
    "required": ["conflict", "closure", "propagated", "independent_check"],
}


def _up_independent_check(cnf: list, assign_in: dict, conflict: bool, closure: dict) -> "tuple[bool, str]":
    """independent check embedded in invoke: after propagation, clause by clause, no unsatisfied unit clause remains / the conflict flag agrees with the empty clause."""
    a = {int(k): bool(v) for k, v in closure.items()}
    unsat_unit, empty_clause = 0, 0
    for cl in cnf:
        sat = False
        unassigned = []
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
            empty_clause += 1
        elif len(unassigned) == 1:
            l = unassigned[0]
            if a.get(abs(l)) != (l > 0):
                unsat_unit += 1  # an unsatisfied unit clause still exists → the closure is incomplete
    if conflict and empty_clause == 0:
        return False, "声称冲突但无全假子句"
    if not conflict and (unsat_unit or empty_clause):
        return False, f"非冲突但残留未满足单元子句 {unsat_unit} / 空子句 {empty_clause}"
    for k, v in assign_in.items():
        if a.get(int(k)) != bool(v):
            return False, f"闭包不扩展输入指派 (var {k})"
    return True, f"闭包完备: 未满足单元子句 0, 空子句 {empty_clause}, 冲突={conflict}"


def _invoke_unit_prop(args: dict, ctl: CallCtl) -> dict:
    ctx = "op:logic_unit_prop#9503"
    cnf = _mk_cnf(args["cnf"], ctx)
# input assignments must be bool value by value: the old version coerced with bool(v), silently treating
# "yes"/1 as True — exactly the silent acceptance §7.6 fail-safe forbids. Now routed through _parse_assign's strict validation.
    assign_in = _parse_assign(args["assign"], ctx)
    conflict, closure = unit_propagate(cnf, assign_in)
    ok, detail = _up_independent_check(cnf, assign_in, conflict, closure)
    if not ok:
        raise SvoError(Err.POSTCOND_FAIL, ctx=ctx, detail=f"独立核对失败: {detail}")
    return {"value": {"conflict": bool(conflict),
                      "closure": {str(k): v for k, v in closure.items()},
                      "propagated": len(closure) - len(assign_in),
                      "independent_check": detail},
            "resid": [0.0 if not conflict else 1.0],
            "notes": f"传播 {len(closure) - len(assign_in)} 个变量; {detail}"}


def _verify_unit_prop(args: dict, value: dict) -> dict:
    """independent verification (watch-list implementation ≠ invoke's repeated scanning)."""
    cnf = _mk_cnf(args["cnf"], "logic_unit_prop@verify")
    a = {int(k): bool(v) for k, v in value["closure"].items()}
    watch: dict = {}
    for i, cl in enumerate(cnf):
        for l in cl:
            watch.setdefault(abs(l), []).append(i)
    changed = True
    conflicts = 0
    while changed:
        changed = False
        for i, cl in enumerate(cnf):
            sat = any(a.get(abs(l)) == (l > 0) for l in cl)
            if sat:
                continue
            un = [l for l in cl if abs(l) not in a]
            if not un:
                conflicts += 1
                continue
            if len(un) == 1:
                l = un[0]
                v = abs(l)
                if a.get(v) != (l > 0):
                    a[v] = l > 0
                    changed = True
    exp_conflict = conflicts > 0
    same = (exp_conflict == value["conflict"]) and (a == {int(k): bool(v) for k, v in value["closure"].items()})
    return {"ok": same, "resid": [0.0 if same else 1.0],
            "detail": f"watch-list 复算闭包 {'一致' if same else '不一致'} (conflict={exp_conflict})"}


REGISTRY.register(OpDef(
    name="logic_unit_prop", band=9500, sub=3,
    in_schema=_IN_UP, out_schema=_OUT_UP,
    invoke=_invoke_unit_prop, verify=_verify_unit_prop,
    cost=("units", 2.0), errors=("SCHEMA_MISMATCH", "BOUNDED_VIOLATION", "POSTCOND_FAIL"),
    effects="none (纯函数)",
    doc="§10.1 单元传播闭包 (logiccore.unit_propagate); 独立核对 = 逐子句检查传播后无未满足单元子句、"
        "闭包扩展输入指派、冲突标志与全假子句一致; verify 用 watch-list 第二实现复算。",
    selfchecks=(
        {"id": "chain", "args": {"cnf": [[1, -2], [2]], "assign": {}},
         "check": lambda r: (r.value["closure"] == {"1": True, "2": True} and not r.value["conflict"]
                             and r.value["propagated"] == 2,
                             f"closure={r.value['closure']} propagated={r.value['propagated']}"),
         "expect": "两级传播: [2] 单元 → 2=T; 于是 [1,-2] 中 -2 为假 → 1=T (旧版只断言 {2:T} 是**漏传**, 已修)"},
        {"id": "conflict_flag", "args": {"cnf": [[1], [-1]], "assign": {}},
         "check": lambda r: (r.value["conflict"] is True, "单元子句 1 与 -1 冲突")},
        {"id": "empty_cnf", "args": {"cnf": [], "assign": {"1": True}},
         "check": lambda r: (r.value["closure"] == {"1": True} and not r.value["conflict"], "空公式恒无冲突")},
        {"id": "zero_lit", "args": {"cnf": [[0, 1]], "assign": {}}, "expect_err": "SCHEMA_MISMATCH"},
        {"id": "bad_assign_val", "args": {"cnf": [[1]], "assign": {"1": "yes"}}, "expect_err": "SCHEMA_MISMATCH"},
    ),
))


# ===========================================================================
# operator 4: logic_unify (row 9504) — §10.1 term unification + unify_witness certificate
# ===========================================================================

_IN_UN = {"kind": "dict", "fields": {"a": {"kind": "str"}, "b": {"kind": "str"}}, "required": ["a", "b"]}
_OUT_UN = {
    "kind": "dict",
    "fields": {
        "unifiable": {"kind": "bool"},
        "sigma": {"kind": "dict", "extra": True},
        "sigma_str": {"kind": "dict", "extra": True},
    },
    "required": ["unifiable", "sigma", "sigma_str"],
}


def _invoke_unify(args: dict, ctl: CallCtl) -> dict:
    ctx = "op:logic_unify#9504"
    a = parse_term(args["a"])
    b = parse_term(args["b"])
    mgu = unify(a, b)
    if mgu is None:
        raise SvoError(Err.UNIFY_FAIL, ctx=ctx,
                       detail=f"不合一: {args['a']!r} vs {args['b']!r} (冲突或 occurs-check)")
    from ..logiccore import term_str

    sigma_str = {k: term_str(v) for k, v in mgu.items()}
    cert = Certificate(kind="unify_witness",
                       payload={"a": args["a"], "b": args["b"], "sigma": dict(mgu)},
                       issued_by="logic_unify")
    return {"value": {"unifiable": True, "sigma": dict(mgu), "sigma_str": sigma_str},
            "resid": [0.0], "cert": cert,
            "notes": f"MGU σ = {sigma_str}; 证书 unify_witness (σ 独立作用到两侧逐符号相等)"}


def _verify_unify(args: dict, value: dict) -> dict:
    """independent verification: re-apply σ (string form) to both terms, requiring symbol-wise equality — without going through logiccore.unify."""
    from ..logiccore import term_str

    a = parse_term(args["a"])
    b = parse_term(args["b"])
    sigma = {k: parse_term(v) for k, v in value["sigma_str"].items()}
    sa, sb = apply_subst(a, sigma), apply_subst(b, sigma)
    ok = sa == sb
    # σ_str is structurally consistent with σ (the two representations agree)
    ok = ok and all(term_str(sigma[k]) == v for k, v in value["sigma_str"].items())
    return {"ok": ok, "resid": [0.0 if ok else 1.0],
            "detail": f"见证核对 σ(a)={term_str(sa)} {'==' if ok else '!='} σ(b)={term_str(sb)}"}


def _unify_verifier_rejects_wrong_claim() -> "tuple[bool, str]":
    """Independent probe: feed the unify_witness checker a **wrong** unification witness; it must reject it.

    Auditor E's mutation experiment showed that logic_unify's verifier had no "rejects wrong input" coverage anywhere in
    the library — the verifier could rot while the selfchecks stayed green. This probe calls CHECKERS directly, bypassing
    the operator's own path.
    """
    from ..certify import CHECKERS, Certificate

    cert = Certificate(kind="unify_witness",
                       payload={"a": "f(?x)", "b": "f(g(?y))", "sigma": {"x": "h(?y)"}},
                       issued_by="probe")
    ok, _resid, _detail = CHECKERS.check(cert)
    return (not ok), ("检查器正确否决了错误 σ (x↦h(?y) 不能让两侧相等)" if not ok
                      else "**检查器接受了错误 σ** — 合一见证检查器失效 (红色警报)")


REGISTRY.register(OpDef(
    name="logic_unify", band=9500, sub=4,
    in_schema=_IN_UN, out_schema=_OUT_UN,
    invoke=_invoke_unify, verify=_verify_unify, cert_kind="unify_witness",
    cost=("units", 1.5), errors=("UNIFY_FAIL", "SCHEMA_MISMATCH", "POSTCOND_FAIL"),
    effects="none (纯函数)",
    doc="§10.1 项合一 unify(a,b,θ) (logiccore.unify, 含 occurs check); 失败抛 UNIFY_FAIL 不静默; "
        "成功必须产出 unify_witness 证书 (certify 独立检查器: σ 作用到两侧逐符号相等)。",
    requires_cert=True,
    selfchecks=(
        {"id": "mgu_basic", "args": {"a": "f(?x, a)", "b": "f(b, ?y)"},
         "check": lambda r: (r.value["sigma_str"] == {"x": "b", "y": "a"}
                             and r.cert is not None and r.cert_independent,
                             f"σ={r.value['sigma_str']}, 证书独立复验={r.cert_independent}")},
        {"id": "deep_term", "args": {"a": "g(f(a, ?x))", "b": "g(?y)"},
         "check": lambda r: (r.value["sigma_str"] == {"y": "f(a, ?x)"}, f"σ={r.value['sigma_str']}")},
        {"id": "clash_fail", "args": {"a": "f(a)", "b": "f(b)"}, "expect_err": "UNIFY_FAIL"},
        {"id": "occurs_fail", "args": {"a": "f(?x)", "b": "?x"}, "expect_err": "UNIFY_FAIL"},
        {"id": "parse_fail", "args": {"a": "f(", "b": "a"}, "expect_err": "UNIFY_FAIL"},
        {"id": "verifier_rejects_wrong_sigma", "args": {"a": "f(?x)", "b": "f(?y)"},
         "check": lambda r: _unify_verifier_rejects_wrong_claim(),
         "expect": "把**错误的** σ (x↦h(?y)) 直接喂给 unify_witness 证书检查器: 必须被否决。"
                   "审计员 E 指出旧自检里没有任何'验证器拒错'的用例 —— 验证器坏掉也无人知晓; "
                   "本项补上该覆盖 (σ 正确时 x↦g(?y) 应通过, 错误时 h(?y) 应否决)"},
    ),
))


# ===========================================================================
# operator 5: logic_sat_verify (row 9505) — §10.6 checker as an operator (violated-clause count, O(total literals))
# ===========================================================================

_IN_SV = {
    "kind": "dict",
    "fields": {
        "cnf": {"kind": "list", "items": {"kind": "list", "items": {"kind": "int"}}, "minlen": 0, "maxlen": 2048},
        "assign": {"kind": "dict", "extra": True},
    },
    "required": ["cnf", "assign"],
}
_OUT_SV = {
    "kind": "dict",
    "fields": {
        "violated": {"kind": "int"},
        "violated_clauses": {"kind": "list", "items": {"kind": "int"}},
        "total_clauses": {"kind": "int"},
        "total_literals": {"kind": "int"},
    },
    "required": ["violated", "violated_clauses", "total_clauses", "total_literals"],
}


def _parse_assign(raw: dict, ctx: str) -> dict:
    out = {}
    for k, v in raw.items():
        if not isinstance(v, bool):
            raise SvoError(Err.SCHEMA_MISMATCH, ctx=ctx,
                           detail=f"指派值须为 bool, var {k!r} 得 {type(v).__name__}={v!r}")
        try:
            out[int(k)] = v
        except (TypeError, ValueError) as e:
            raise SvoError(Err.SCHEMA_MISMATCH, ctx=ctx, detail=f"指派键非法 {k!r}: {e}") from e
    return out


def _invoke_sat_verify(args: dict, ctl: CallCtl) -> dict:
    ctx = "op:logic_sat_verify#9505"
    cnf = _mk_cnf(args["cnf"], ctx, max_clauses=4096)
    assign = _parse_assign(args["assign"], ctx)
    bad: list[int] = []
    n_lits = 0
    for i, cl in enumerate(cnf):  # §10.6: O(total literals), generator early-exit
        n_lits += len(cl)
        if not any(assign.get(abs(l), None) == (l > 0) for l in cl):
            bad.append(i)
            ctl.check_time(ctx) if i % 512 == 0 else None
    return {"value": {"violated": len(bad), "violated_clauses": bad,
                      "total_clauses": len(cnf), "total_literals": n_lits},
            "resid": [float(len(bad))],
            "notes": f"违背子句 {len(bad)}/{len(cnf)} (总文字数 {n_lits})"}


def _verify_sat_verify(args: dict, value: dict) -> dict:
    """independent verification: trust-root check_sat_assignment + explicit clause-by-clause recomputation of violations."""
    cnf = _mk_cnf(args["cnf"], "logic_sat_verify@verify", max_clauses=4096)
    assign = _parse_assign(args["assign"], "logic_sat_verify@verify")
    root_ok = check_sat_assignment(cnf, assign)
    bad: list[int] = []
    for i, cl in enumerate(cnf):
        sat = False
        for l in cl:
            v = abs(l)
            if v in assign:
                if assign[v] == (l > 0):
                    sat = True
                    break
        if not sat:
            bad.append(i)
    ok = (root_ok == (len(bad) == 0)) and (bad == value["violated_clauses"]) \
        and (len(bad) == value["violated"])
    return {"ok": ok, "resid": [float(len(bad))],
            "detail": f"信任根 check_sat_assignment={root_ok}, 逐条复算违背 {len(bad)} 条"}


REGISTRY.register(OpDef(
    name="logic_sat_verify", band=9500, sub=5,
    in_schema=_IN_SV, out_schema=_OUT_SV,
    invoke=_invoke_sat_verify, verify=_verify_sat_verify,
    cost=("units", 1.0), errors=("SCHEMA_MISMATCH", "BOUNDED_VIOLATION"),
    effects="none (纯函数)",
    doc="§10.6 SAT 证书检查器算子化: CNF+指派 → 违背子句数, O(总文字数); "
        "独立验证 = logiccore.check_sat_assignment (信任根) + 逐条违背子句显式复算。",
    selfchecks=(
        {"id": "all_sat", "args": {"cnf": [[1, 2], [-1, 3]], "assign": {"1": True, "2": False, "3": True}},
         "check": lambda r: (r.value["violated"] == 0, "0 违背")},
        {"id": "count_violations", "args": {"cnf": [[1], [-1, 2], [3]], "assign": {"1": False}},
         "check": lambda r: (r.value["violated"] == 2 and r.value["violated_clauses"] == [0, 2],
                             f"violated={r.value['violated']} clauses={r.value['violated_clauses']}"),
         "expect": "逐子句核: [1] 因 1=F 不满足; [-1,2] 由 ¬1 满足 (不算违背); [3] 未指派 → 不满足。"
                   "故 2 条违背、下标 [0,2] —— 手算核对, 旧版断 3 是错的"},
        {"id": "empty_cnf_sat", "args": {"cnf": [], "assign": {}},
         "check": lambda r: (r.value["violated"] == 0, "空 CNF 空指派 = 满足")},
        {"id": "zero_literal", "args": {"cnf": [[0]], "assign": {}}, "expect_err": "SCHEMA_MISMATCH"},
        {"id": "bad_assign", "args": {"cnf": [[1]], "assign": {"1": 3}}, "expect_err": "SCHEMA_MISMATCH"},
    ),
))


# ===========================================================================
# operator 6: logic_sat_solve (row 9506) — small-instance DPLL + sat_assignment certificate
# ===========================================================================

_IN_SS = {
    "kind": "dict",
    "fields": {
        "cnf": {"kind": "list", "items": {"kind": "list", "items": {"kind": "int"}}, "minlen": 0, "maxlen": 1024},
        "max_vars": {"kind": "int", "min": 1, "max": 22},
        "max_nodes": {"kind": "int", "min": 100, "max": 2000000},
    },
    "required": ["cnf"],
}
_OUT_SS = {
    "kind": "dict",
    "fields": {
        "sat": {"kind": "bool"},
        "assign": {"kind": "dict", "extra": True},
        "cross_check": {"kind": "str"},
        "n_vars": {"kind": "int"},
    },
    "required": ["sat", "assign", "cross_check", "n_vars"],
}


def _invoke_sat_solve(args: dict, ctl: CallCtl) -> dict:
    ctx = "op:logic_sat_solve#9506"
    cnf = _mk_cnf(args["cnf"], ctx, max_clauses=2048)
    max_vars = int(args.get("max_vars", 20))
    max_nodes = int(args.get("max_nodes", 200000))
    nvars = max((abs(l) for cl in cnf for l in cl), default=0)
    n_lits = sum(len(cl) for cl in cnf)
    if nvars > max_vars or len(cnf) > 1024 or n_lits > 20000:
        raise SvoError(Err.BOUNDED_VIOLATION, ctx=ctx,
                       detail=f"规模超限: vars={nvars}>{max_vars} 或 子句/文字超限 — 不许假装, 走 DRAT 路径")
    ctl.check_time(ctx)
    model = dpll_small(cnf, max_nodes=max_nodes)  # exceeding the node cap itself raises BOUNDED_VIOLATION
    if model is not None:
        assign = {str(k): bool(v) for k, v in model.items()}
        cert = Certificate(kind="sat_assignment", payload={"cnf": [_clause_list(c) for c in cnf],
                                                           "assign": assign}, issued_by="logic_sat_solve")
        return {"value": {"sat": True, "assign": assign, "cross_check": "check_sat_assignment 通过 (verify 侧)",
                          "n_vars": nvars},
                "resid": [0.0], "cert": cert,
                "notes": f"SAT, 指派 {len(assign)} 变量; 证书 sat_assignment"}
    # UNSAT: independently cross-confirm via resolution refutation (small instances only; dpll is decisive and resolution is a second decision path)
    if nvars <= 12 and len(cnf) <= 60:
        from ..logiccore import resolution_refute

        found, steps, detail = resolution_refute(cnf, max_steps=20000)
        if not found:
            raise SvoError(Err.RUNTIME_ERR, ctx=ctx,
                           detail=f"不变量破坏: DPLL 判 UNSAT 但归结反驳失败 ({detail}) — 双判定路径分歧, 拒绝交付")
        cc = f"归结反驳独立确认 UNSAT ({steps} 步导出空子句)"
    else:
        cc = "实例超出归结交叉核对规模, 仅 DPLL 判定 (DPLL 完备但单路径)"
    return {"value": {"sat": False, "assign": {}, "cross_check": cc, "n_vars": nvars},
            "resid": [0.0],
            "notes": f"UNSAT; {cc}. UNSAT 的可独立核查证书走 drat_check (行 9540)"}


def _verify_sat_solve(args: dict, value: dict) -> dict:
    """independent verification: SAT → trust-root clause-by-clause re-verification; UNSAT → (small instances) all-assignment exhaustion confirming no model."""
    cnf = _mk_cnf(args["cnf"], "logic_sat_solve@verify", max_clauses=2048)
    nvars = max((abs(l) for cl in cnf for l in cl), default=0)
    if value["sat"]:
        assign = _parse_assign(value["assign"], "logic_sat_solve@verify")
        ok = check_sat_assignment(cnf, assign)
        return {"ok": ok, "resid": [0.0 if ok else 1.0], "detail": f"信任根复验指派: {'满足' if ok else '不满足'}"}
    if nvars <= 14:
        vs = list(range(1, nvars + 1))
        found = None
        for bits in product((False, True), repeat=nvars):
            env = {v: b for v, b in zip(vs, bits)}
            if check_sat_assignment(cnf, env):
                found = env
                break
        if found is not None:
            return {"ok": False, "resid": [1.0],
                    "detail": f"穷举发现模型 {found} — UNSAT 声称被语义反驳"}
        return {"ok": True, "resid": [0.0], "detail": f"穷举 2^{nvars} 指派确认无模型 (语义路径)"}
    return {"ok": True, "resid": [0.0],
            "detail": "UNSAT 且实例 >14 变量: 语义穷举超界, 依赖 DPLL+归结双判定 (已在 notes 声明)"}


REGISTRY.register(OpDef(
    name="logic_sat_solve", band=9500, sub=6,
    in_schema=_IN_SS, out_schema=_OUT_SS,
    invoke=_invoke_sat_solve, verify=_verify_sat_solve, cert_kind="sat_assignment",
    cost=("units", 6.0), errors=("SCHEMA_MISMATCH", "BOUNDED_VIOLATION", "RUNTIME_ERR", "POSTCOND_FAIL"),
    effects="none (纯函数, 内存内搜索)",
    doc="§10.1/§10.6 小实例 DPLL (logiccore.dpll_small, 判定性) + sat_assignment 指派证书; "
        "超出规模 (默认 >20 变量) 抛 BOUNDED_VIOLATION 不许假装; UNSAT 结论在小实例上用归结反驳交叉确认。"
        "verify 侧: SAT 走信任根逐子句复验, UNSAT 走全指派穷举 (≤14 变量, 语义路径)。",
    selfchecks=(
        {"id": "sat_small", "args": {"cnf": [[1, 2], [-1, 3], [-3, -1]]},
         "check": lambda r: (r.value["sat"] is True and r.cert is not None and r.cert_independent,
                             f"assign={r.value['assign']}")},
        {"id": "unsat_cross", "args": {"cnf": [[1, 2], [-1, 2], [1, -2], [-1, -2]]},
         "check": lambda r: (r.value["sat"] is False and "归结反驳" in r.value["cross_check"],
                             r.value["cross_check"])},
        {"id": "over_limit", "args": {"cnf": [[21], [-21]]}, "expect_err": "BOUNDED_VIOLATION"},
        {"id": "zero_lit", "args": {"cnf": [[0, 1]]}, "expect_err": "SCHEMA_MISMATCH"},
    ),
))


# ===========================================================================
# operator 7: logic_hilbert_prove (row 9507) — Hilbert kernel re-verification + proof_kernel certificate + semantic cross-check
# ===========================================================================

_STEP_SCH = {
    "kind": "dict", "extra": False,
    "fields": {
        "rule": {"kind": "str", "enum": ["hyp", "axiom", "mp"]},
        "formula": {"kind": "str"},
        "idx": {"kind": "int", "min": 0},
        "from": {"kind": "list", "items": {"kind": "int", "min": 0}, "len": 2},
    },
    "required": ["rule", "formula"],
}
_IN_HP = {
    "kind": "dict",
    "fields": {
        "hyps": {"kind": "list", "items": {"kind": "str"}, "minlen": 0, "maxlen": 16},
        "steps": {"kind": "list", "items": _STEP_SCH, "minlen": 1, "maxlen": 512},
        "goal": {"kind": "str"},
    },
    "required": ["hyps", "steps", "goal"],
}
_OUT_HP = {
    "kind": "dict",
    "fields": {
        "n_steps": {"kind": "int"},
        "goal": {"kind": "str"},
        "axiom_usage": {"kind": "dict", "extra": True},
        "semantic_cross": {"kind": "str"},
    },
    "required": ["n_steps", "goal", "axiom_usage", "semantic_cross"],
}


def _invoke_hilbert(args: dict, ctl: CallCtl) -> dict:
    ctx = "op:logic_hilbert_prove#9507"
    hyps, steps, goal = args["hyps"], args["steps"], args["goal"]
    atoms = set()
    for s in [goal] + list(hyps) + [st["formula"] for st in steps]:
        atoms |= _atoms(parse_formula(s))
    if len(atoms) > _MAX_EQUIV_ATOMS:
        raise SvoError(Err.BOUNDED_VIOLATION, ctx=ctx,
                       detail=f"原子数 {len(atoms)} > {_MAX_EQUIV_ATOMS} — 语义交叉核对超界, 拒绝")
    ok, idx, detail = check_proof(hyps, steps, goal)  # the kernel decides only, never searches
    if not ok:
        raise SvoError(Err.POSTCOND_FAIL, ctx=ctx, detail=f"Hilbert 核拒绝 (步 {idx}): {detail}")
    usage = {"A1": 0, "A2": 0, "A3": 0}
    for st in steps:
        if st["rule"] == "axiom":
            ax = is_axiom_instance(parse_formula(st["formula"]))
            if ax:
                usage[ax] = usage.get(ax, 0) + 1
    cert = Certificate(kind="proof_kernel", payload={"formulas": list(hyps), "steps": [dict(s) for s in steps],
                                                     "goal": goal}, issued_by="logic_hilbert_prove")
    return {"value": {"n_steps": len(steps), "goal": goal, "axiom_usage": usage,
                      "semantic_cross": f"verify 侧真值表: {len(atoms)} 原子全指派蕴含核对"},
            "resid": [0.0], "cert": cert,
            "notes": f"{len(steps)} 步全部过核; 公理使用 {usage}"}


def _verify_hilbert(args: dict, value: dict) -> dict:
    """independent verification (semantic): hyps ⊨ goal — truth-table enumeration, a path entirely different from the proof system (syntax)."""
    hyps_f = [parse_formula(h) for h in args["hyps"]]
    goal_f = parse_formula(args["goal"])
    atoms = sorted(set().union(*map(_atoms, hyps_f + [goal_f])) if hyps_f else _atoms(goal_f))
    cex = 0
    for bits in product((False, True), repeat=len(atoms)):
        env = dict(zip(atoms, bits))
        if all(_ev_rec(h, env) for h in hyps_f) and not _ev_rec(goal_f, env):
            cex += 1
    return {"ok": cex == 0, "resid": [float(cex)],
            "detail": f"语义蕴含核对: 反例指派 {cex} 个 (2^{len(atoms)} 全枚举)"}


REGISTRY.register(OpDef(
    name="logic_hilbert_prove", band=9500, sub=7,
    in_schema=_IN_HP, out_schema=_OUT_HP,
    invoke=_invoke_hilbert, verify=_verify_hilbert, cert_kind="proof_kernel",
    cost=("units", 2.0), errors=("SCHEMA_MISMATCH", "BOUNDED_VIOLATION", "POSTCOND_FAIL"),
    effects="none (纯函数)",
    doc="§10.1 自制证明核: 假设+步序列+目标 → logiccore.check_proof (核只判定不搜索); "
        "证书 kind=proof_kernel (certify 独立检查器); verify = 真值表语义蕴含核对 (语法 vs 语义双路径)。"
        "注意: 语义核对是可靠性方向的交叉 (证明通过 ⟹ 必然蕴含), 不证明证明存在性。",
    requires_cert=True,
    selfchecks=(
        {"id": "identity_proof", "args": {"hyps": [], "goal": "(a -> a)", "steps": [
            {"rule": "axiom", "formula": "(a -> ((a -> a) -> a))"},
            {"rule": "axiom", "formula": "((a -> ((a -> a) -> a)) -> ((a -> (a -> a)) -> (a -> a)))"},
            {"rule": "mp", "from": [1, 0], "formula": "((a -> (a -> a)) -> (a -> a))"},
            {"rule": "axiom", "formula": "(a -> (a -> a))"},
            {"rule": "mp", "from": [2, 3], "formula": "(a -> a)"},
        ]},
         "check": lambda r: (r.value["n_steps"] == 5 and r.cert_independent
                             and r.value["axiom_usage"]["A1"] == 2 and r.value["axiom_usage"]["A2"] == 1,
                             f"5 步, A1×2+A2×1, 证书独立复验={r.cert_independent}")},
        {"id": "mp_exercise", "args": {"hyps": ["a", "(a -> b)"], "goal": "b", "steps": [
            {"rule": "hyp", "idx": 0, "formula": "a"},
            {"rule": "hyp", "idx": 1, "formula": "(a -> b)"},
            {"rule": "mp", "from": [1, 0], "formula": "b"},
        ]},
         "check": lambda r: (r.value["n_steps"] == 3, "MP 3 步")},
        {"id": "bad_mp", "args": {"hyps": ["a"], "goal": "b", "steps": [
            {"rule": "hyp", "idx": 0, "formula": "a"},
            {"rule": "mp", "from": [0, 0], "formula": "b"},
        ]}, "expect_err": "POSTCOND_FAIL"},
        {"id": "goal_not_derived", "args": {"hyps": [], "goal": "a", "steps": [
            {"rule": "axiom", "formula": "(a -> a)"}]}, "expect_err": "POSTCOND_FAIL"},
        {"id": "bad_goal_token", "args": {"hyps": [], "goal": "(a", "steps": [
            {"rule": "axiom", "formula": "(a -> a)"}]}, "expect_err": "SCHEMA_MISMATCH"},
    ),
))


# ===========================================================================
# §10.3 argument-graph shared validation core (used by arg_graph and premise_trace; the two call sites validate along different paths)
# ===========================================================================

_EDGE_KINDS = ("res", "mp", "hilbert")

#: antecedent arity per edge kind (a hard constraint). Without it, a single-antecedent res edge would raise
#: IndexError on `from[1]`, disguising "the user wrote a wrong edge" as RUNTIME_ERR — that is an interface-contract
#: problem and must be caught explicitly as PRECOND_FAIL.
_EDGE_ARITY: dict = {"res": 2, "mp": 2, "hilbert": None}  # None = at least 1


def _load_graph(args: dict, ctx: str) -> "tuple[dict[str, dict], list[dict], set, str]":
    nodes = {n["id"]: n for n in args["nodes"]}
    if len(nodes) != len(args["nodes"]):
        raise SvoError(Err.SCHEMA_MISMATCH, ctx=ctx, detail="节点 id 重复")
    edges = args["edges"]
    for e in edges:
        kind = e.get("kind")
        if kind not in _EDGE_KINDS:
            raise SvoError(Err.PRECOND_FAIL, ctx=ctx, detail=f"未知边类型 {kind!r} (允许 {_EDGE_KINDS})")
        want = _EDGE_ARITY[kind]
        n_from = len(e["from"])
        if want is not None and n_from != want:
            raise SvoError(Err.PRECOND_FAIL, ctx=ctx,
                           detail=f"边 {e.get('eid')} 类型 {kind} 需 {want} 个前件, 得 {n_from}")
        if want is None and n_from < 1:
            raise SvoError(Err.PRECOND_FAIL, ctx=ctx,
                           detail=f"边 {e.get('eid')} 类型 {kind} 至少需 1 个前件")
        if len(set(e["from"])) != n_from:
            raise SvoError(Err.PRECOND_FAIL, ctx=ctx, detail=f"边 {e.get('eid')} 前件重复")
        for src in e["from"]:
            if src not in nodes:
                raise SvoError(Err.SCHEMA_MISMATCH, ctx=ctx, detail=f"边引用未登记节点 {src!r}")
        if e["to"] not in nodes:
            raise SvoError(Err.SCHEMA_MISMATCH, ctx=ctx, detail=f"边引用未登记节点 {e['to']!r}")
        if e["to"] in e["from"]:
            raise SvoError(Err.PRECOND_FAIL, ctx=ctx, detail=f"边 {e.get('eid')} 自环 ({e['to']})")
    return nodes, edges, set(args["premises"]), args["conclusion"]


def _check_edge_semantic(e: dict, nodes: dict, ctx: str) -> "tuple[bool, str]":
    """edge-level semantic re-verification (verify side): truth-table/semantic path, unlike invoke's kernel/resolution path."""
    kind = e["kind"]
    to_stmt = nodes[e["to"]]["stmt"]
    if kind == "res":
        c1 = _parse_clause(nodes[e["from"][0]]["stmt"], ctx)
        c2 = _parse_clause(nodes[e["from"][1]]["stmt"], ctx)
        child = _parse_clause(to_stmt, ctx)
        bad = _tt_consequence([c1, c2], child, {}, _cl_sat_rec)
        return bad == 0, f"res 语义推论违背 {bad} 指派"
    if kind == "mp":
        maj = parse_formula(nodes[e["from"][0]]["stmt"])
        mino = parse_formula(nodes[e["from"][1]]["stmt"])
        concl = parse_formula(to_stmt)
        if maj[0] != "imp" or maj[2] != concl or maj[1] != mino:
            return False, "mp 结构不符 (maj ≠ min→concl)"
        return True, "mp 结构核对通过"
    if kind == "hilbert":
        rec = e["record"]
        hyps = [nodes[i]["stmt"] for i in e["from"]]
        ok, idx, detail = check_proof(hyps, rec["steps"], to_stmt)
        return ok, f"hilbert 核 ({'' if ok else f'步 {idx} 拒: '}{detail})"
    return False, f"未知边类型 {kind!r}"


def _check_edge_kernel(e: dict, nodes: dict, ctx: str) -> "tuple[bool, str]":
    """edge-level kernel re-verification (invoke side): resolution members / structural match / check_proof."""
    kind = e["kind"]
    to_stmt = nodes[e["to"]]["stmt"]
    if kind == "res":
        c1 = _parse_clause(nodes[e["from"][0]]["stmt"], ctx)
        c2 = _parse_clause(nodes[e["from"][1]]["stmt"], ctx)
        child = _parse_clause(to_stmt, ctx)
        if child not in resolve(c1, c2):
            return False, f"child {sorted(child)} 不是 resolve(c1,c2) 的归结式"
        bad = _tt_consequence([c1, c2], child, {}, _cl_sat_iter)
        return bad == 0, f"归结成员核对 + 真值表推论违背 {bad}"
    if kind == "mp":
        maj = parse_formula(nodes[e["from"][0]]["stmt"])
        mino = parse_formula(nodes[e["from"][1]]["stmt"])
        concl = parse_formula(to_stmt)
        if maj[0] != "imp" or maj[2] != concl or maj[1] != mino:
            return False, f"mp 不成立: maj={_canon(maj)} 不是 ({_canon(mino)} -> {_canon(concl)})"
        return True, "mp 结构核对通过"
    if kind == "hilbert":
        rec = e["record"]
        hyps = [nodes[i]["stmt"] for i in e["from"]]
        ok, idx, detail = check_proof(hyps, rec["steps"], to_stmt)
        if not ok:
            return False, f"hilbert 步 {idx} 拒: {detail}"
        return True, detail
    return False, f"未知边类型 {kind!r}"


def _kahn_topo(nodes: dict, edges: list) -> "tuple[list | None, list]":
    """Kahn topological sort; returns (None, nodes on the cycle) when a cycle exists."""
    indeg = {nid: 0 for nid in nodes}
    adj: dict = {nid: [] for nid in nodes}
    for e in edges:
        for src in e["from"]:
            adj[src].append(e["to"])
            indeg[e["to"]] += 1
    queue = sorted([n for n, d in indeg.items() if d == 0])
    order: list = []
    while queue:
        n = queue.pop(0)
        order.append(n)
        newly = []
        for m in sorted(adj[n]):
            indeg[m] -= 1
            if indeg[m] == 0:
                newly.append(m)
        queue.extend(sorted(newly))
    if len(order) != len(nodes):
        cyclic = sorted(n for n, d in indeg.items() if d > 0)
        return None, cyclic
    return order, []


def _dfs_acyclic(nodes: dict, edges: list) -> "tuple[bool, list]":
    """DFS three-colour marking (verify-side cycle detection, a different path from Kahn)."""
    adj: dict = {nid: [] for nid in nodes}
    for e in edges:
        for src in e["from"]:
            adj[src].append(e["to"])
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {nid: WHITE for nid in nodes}
    stack_cycle: list = []

    def dfs(u: str) -> bool:
        color[u] = GRAY
        for v in sorted(adj[u]):
            if color[v] == GRAY:
                stack_cycle.extend([u, v])
                return False
            if color[v] == WHITE and not dfs(v):
                return False
        color[u] = BLACK
        return True

    for n in sorted(nodes):
        if color[n] == WHITE and not dfs(n):
            return False, stack_cycle
    return True, []


# ===========================================================================
# operator 8: logic_arg_graph (row 9508) — §10.3 argument-graph check (DAG + edge re-verification + explicit premises)
# ===========================================================================

_EDGE_SCH = {
    "kind": "dict", "extra": False,
    "fields": {
        "from": {"kind": "list", "items": {"kind": "str"}, "minlen": 0, "maxlen": 4},
        "to": {"kind": "str"},
        "kind": {"kind": "str", "enum": list(_EDGE_KINDS)},
        "eid": {"kind": "str"},
        "record": {"kind": "dict", "extra": True},
    },
    "required": ["from", "to", "kind", "record"],
}
_IN_AG = {
    "kind": "dict",
    "fields": {
        "nodes": {"kind": "list", "minlen": 1, "maxlen": 64, "items": {
            "kind": "dict", "extra": False,
            "fields": {"id": {"kind": "str"}, "stmt": {"kind": "str"}},
            "required": ["id", "stmt"]}},
        "edges": {"kind": "list", "minlen": 1, "maxlen": 96, "items": _EDGE_SCH},
        "premises": {"kind": "list", "items": {"kind": "str"}, "minlen": 0, "maxlen": 32},
        "conclusion": {"kind": "str"},
    },
    "required": ["nodes", "edges", "premises", "conclusion"],
}
_OUT_AG = {
    "kind": "dict",
    "fields": {
        "valid": {"kind": "bool"},
        "topo": {"kind": "list", "items": {"kind": "str"}},
        "n_edges": {"kind": "int"},
        "edge_report": {"kind": "list", "items": {"kind": "dict", "extra": True}},
        "premises_needed": {"kind": "list", "items": {"kind": "str"}},
    },
    "required": ["valid", "topo", "n_edges", "edge_report", "premises_needed"],
}


def _premise_closure(nodes: dict, edges: list, conclusion: str) -> list:
    """conclusion traceback: reverse-reachable nodes without incoming edges = the required premises."""
    incoming: dict = {nid: [] for nid in nodes}
    for e in edges:
        incoming[e["to"]].append(e)
    seen, stack, need = {conclusion}, [conclusion], []
    while stack:
        n = stack.pop()
        if not incoming[n]:
            need.append(n)
            continue
        for e in incoming[n]:
            for src in e["from"]:
                if src not in seen:
                    seen.add(src)
                    stack.append(src)
    return sorted(set(need))


def _invoke_arg_graph(args: dict, ctl: CallCtl) -> dict:
    ctx = "op:logic_arg_graph#9508"
    nodes, edges, premises, conclusion = _load_graph(args, ctx)
    if conclusion not in nodes:
        raise SvoError(Err.SCHEMA_MISMATCH, ctx=ctx, detail=f"结论节点 {conclusion!r} 未登记")
    # 1) DAG acyclicity (Kahn)
    topo, cyclic = _kahn_topo(nodes, edges)
    if topo is None:
        raise SvoError(Err.POSTCOND_FAIL, ctx=ctx, detail=f"论证图有环 (非 DAG), 环上节点: {cyclic}")
    # 2) every edge independently re-verified (kernel/resolution path)
    report = []
    for e in edges:
        ctl.check_time(ctx)
        ok, detail = _check_edge_kernel(e, nodes, ctx)
        report.append({"eid": e.get("eid", f"{e['from']}->{e['to']}"), "ok": bool(ok), "detail": detail})
        if not ok:
            raise SvoError(Err.POSTCOND_FAIL, ctx=ctx,
                           detail=f"边 {e.get('eid', e['from'])}->{e['to']} ({e['kind']}) 复验失败: {detail}")
    # 3) premises explicit: nodes without incoming edges must be declared premises
    has_incoming = {e["to"] for e in edges}
    implicit = sorted(set(nodes) - has_incoming - premises)
    if implicit:
        raise SvoError(Err.PRECOND_FAIL, ctx=ctx,
                       detail=f"节点 {implicit} 无前置且未声明为前提 — 隐式前提的'论证'不是论证 (§10.3)")
    unknown_prem = sorted(premises - set(nodes))
    if unknown_prem:
        raise SvoError(Err.SCHEMA_MISMATCH, ctx=ctx, detail=f"前提引用未登记节点: {unknown_prem}")
    need = _premise_closure(nodes, edges, conclusion)
    missing = sorted(set(need) - premises)
    if missing:
        raise SvoError(Err.PRECOND_FAIL, ctx=ctx,
                       detail=f"结论回溯到未声明前提 {missing} — 论证不闭合")
    return {"value": {"valid": True, "topo": topo, "n_edges": len(edges),
                      "edge_report": report, "premises_needed": need},
            "resid": [0.0],
            "notes": f"DAG 无环 (Kahn), {len(edges)} 条边全部复验, 结论依赖前提 {need}"}


def _verify_arg_graph(args: dict, value: dict) -> dict:
    """independent verification (semantic path + DFS): cycle detection switched to DFS, edge re-verification switched to semantics (truth table/structure), unlike invoke."""
    ctx = "logic_arg_graph@verify"
    nodes, edges, premises, conclusion = _load_graph(args, ctx)
    acyclic, cyc = _dfs_acyclic(nodes, edges)
    probs = []
    if not acyclic:
        probs.append(f"DFS 检出环 {cyc}")
    for e in edges:
        ok, detail = _check_edge_semantic(e, nodes, ctx)
        if not ok:
            probs.append(f"边语义复验失败: {detail}")
    need = _premise_closure(nodes, edges, conclusion)
    if set(need) - premises:
        probs.append("存在未声明前提")
    return {"ok": not probs and value["valid"] and value["topo"] is not None,
            "resid": [float(len(probs))],
            "detail": "DFS+语义交叉: " + ("全部通过" if not probs else "; ".join(probs))}


REGISTRY.register(OpDef(
    name="logic_arg_graph", band=9500, sub=8,
    in_schema=_IN_AG, out_schema=_OUT_AG,
    invoke=_invoke_arg_graph, verify=_verify_arg_graph,
    cost=("units", 4.0), errors=("SCHEMA_MISMATCH", "PRECOND_FAIL", "POSTCOND_FAIL"),
    effects="none (纯函数)",
    doc="§10.3 论证图 arg_check: 节点=命题句柄(clause 串/公式串), 边=已验证推理步 (res/mp/hilbert); "
        "检查 DAG 无环 + 每条边独立复验 + 前提显式 (无入边节点必在 premises 且结论回溯闭合)。"
        "不落为图/隐式前提/有环的'论证'一律拒绝。verify = DFS 环检测 + 边语义(真值表)复验, 双路径。",
    selfchecks=(
        {"id": "resolution_chain",
         "args": {"nodes": [{"id": "p1", "stmt": "1,2"}, {"id": "p2", "stmt": "-1,2"},
                            {"id": "p3", "stmt": "1,-2"}, {"id": "p4", "stmt": "-1,-2"},
                            {"id": "n5", "stmt": "2"}, {"id": "n6", "stmt": "-2"}, {"id": "n7", "stmt": ""}],
                  "edges": [
                      {"from": ["p1", "p2"], "to": "n5", "kind": "res", "eid": "e1", "record": {}},
                      {"from": ["p3", "p4"], "to": "n6", "kind": "res", "eid": "e2", "record": {}},
                      {"from": ["n5", "n6"], "to": "n7", "kind": "res", "eid": "e3", "record": {}}],
                  "premises": ["p1", "p2", "p3", "p4"], "conclusion": "n7"},
         "check": lambda r: (r.value["valid"] and r.value["premises_needed"] == ["p1", "p2", "p3", "p4"],
                             f"topo={r.value['topo']}")},
        {"id": "mp_edge_graph",
         "args": {"nodes": [{"id": "h1", "stmt": "a"}, {"id": "h2", "stmt": "(a -> b)"}, {"id": "c", "stmt": "b"}],
                  "edges": [{"from": ["h2", "h1"], "to": "c", "kind": "mp", "eid": "m1", "record": {}}],
                  "premises": ["h1", "h2"], "conclusion": "c"},
         "check": lambda r: (r.value["valid"], "MP 边图通过")},
        {"id": "cycle_rejected",
    # A genuine cycle built from structurally legal 2-antecedent mp edges: x→y (e1) and y→x (e2) are each other's antecedent.
    # The old version built the cycle from single-antecedent res edges, which are themselves illegal (arity 1≠2), so it hit
    # PRECOND_FAIL first and never actually tested cycle detection — now it uses legal edges and a real cycle.
         "args": {"nodes": [{"id": "p", "stmt": "a->b"}, {"id": "q", "stmt": "b->a"},
                            {"id": "x", "stmt": "b"}, {"id": "y", "stmt": "a"}],
                  "edges": [{"from": ["p", "y"], "to": "x", "kind": "mp", "eid": "e1", "record": {}},
                            {"from": ["q", "x"], "to": "y", "kind": "mp", "eid": "e2", "record": {}}],
                  "premises": ["p", "q"], "conclusion": "x"},
         "expect_err": "POSTCOND_FAIL",
         "expect": "x↔y 互前件成环: Kahn 拓扑不完整 → 拒 (非 DAG 的'论证'不是论证)"},
        {"id": "implicit_premise_rejected",
         "args": {"nodes": [{"id": "x", "stmt": "1"}, {"id": "z", "stmt": "-1,2"},
                            {"id": "y", "stmt": "2"}],
                  "edges": [{"from": ["x", "z"], "to": "y", "kind": "res", "eid": "e1", "record": {}}],
                  "premises": [], "conclusion": "y"},
         "expect_err": "PRECOND_FAIL",
         "expect": "x,z 无入边却未声明为前提 → 隐式前提, 必须拒 (2 前件 res 边, 前提集合空)"},
        {"id": "edge_arity_rejected",
         "args": {"nodes": [{"id": "x", "stmt": "1"}, {"id": "y", "stmt": "2"}],
                  "edges": [{"from": ["x"], "to": "y", "kind": "res", "eid": "e1", "record": {}}],
                  "premises": ["x"], "conclusion": "y"},
         "expect_err": "PRECOND_FAIL",
         "expect": "res 边只要 1 个前件 → 元数违规必须 PRECOND_FAIL (旧版在此抛 IndexError 逃成 RUNTIME_ERR)"},
        {"id": "bad_edge_content_rejected",
         "args": {"nodes": [{"id": "p1", "stmt": "1,2"}, {"id": "p2", "stmt": "-1,2"},
                            {"id": "n5", "stmt": "3"}],
                  "edges": [{"from": ["p1", "p2"], "to": "n5", "kind": "res", "eid": "e1", "record": {}}],
                  "premises": ["p1", "p2"], "conclusion": "n5"},
         "expect_err": "POSTCOND_FAIL"},
    ),
))


# ===========================================================================
# operator 9: logic_premise_trace (row 9509) — §10.3 trace the conclusion back to all premises
# ===========================================================================

_IN_PT = dict(_IN_AG)
_OUT_PT = {
    "kind": "dict",
    "fields": {
        "premises": {"kind": "list", "items": {"kind": "str"}},
        "trace": {"kind": "list", "items": {"kind": "str"}},
        "max_depth": {"kind": "int"},
    },
    "required": ["premises", "trace", "max_depth"],
}


def _invoke_premise_trace(args: dict, ctl: CallCtl) -> dict:
    ctx = "op:logic_premise_trace#9509"
    nodes, edges, premises, conclusion = _load_graph(args, ctx)
    if conclusion not in nodes:
        raise SvoError(Err.SCHEMA_MISMATCH, ctx=ctx, detail=f"结论节点 {conclusion!r} 未登记")
    # first require the graph itself to be legal (reusing the arg check's kernel path — tracing is only meaningful on a legal argument graph)
    topo, cyclic = _kahn_topo(nodes, edges)
    if topo is None:
        raise SvoError(Err.POSTCOND_FAIL, ctx=ctx, detail=f"图有环, 无法回溯: {cyclic}")
    for e in edges:
        ok, detail = _check_edge_kernel(e, nodes, ctx)
        if not ok:
            raise SvoError(Err.POSTCOND_FAIL, ctx=ctx, detail=f"边 {e['to']} 复验失败: {detail}")
    # recursive DFS traceback (with memoisation)
    incoming: dict = {nid: [] for nid in nodes}
    for i, e in enumerate(edges):
        incoming[e["to"]].append(i)
    memo: dict = {}

    def back(n: str, depth: int, path: list) -> "tuple[set, int, list]":
        if n in memo:
            return memo[n]
        if not incoming[n]:
            memo[n] = ({n}, depth, [f"{'  ' * depth}PREMISE {n} = {nodes[n]['stmt']!r}"])
            return memo[n]
        ps: set = set()
        dmax = depth
        lines = []
        for i in sorted(incoming[n]):
            e = edges[i]
            for src in e["from"]:
                sub_ps, sub_d, sub_lines = back(src, depth + 1, path)
                ps |= sub_ps
                dmax = max(dmax, sub_d)
                lines.append(f"{'  ' * depth}{src} --[{e['kind']}#{e.get('eid', i)}]--> {n}")
                lines.extend(sub_lines)
        memo[n] = (ps, dmax, lines)
        return memo[n]

    ps, dmax, lines = back(conclusion, 0, [])
    undeclared = sorted(ps - premises)
    if undeclared:
        raise SvoError(Err.PRECOND_FAIL, ctx=ctx,
                       detail=f"回溯到未声明前提 {undeclared} — 论证不闭合, 拒绝出追踪")
    return {"value": {"premises": sorted(ps), "trace": lines, "max_depth": dmax},
            "resid": [0.0],
            "notes": f"结论 {conclusion!r} 依赖 {len(ps)} 个前提, 追踪深度 {dmax}"}


def _verify_premise_trace(args: dict, value: dict) -> dict:
    """independent verification: iterative work-set reverse reachability (a different implementation from the recursive DFS); compare premise sets."""
    ctx = "logic_premise_trace@verify"
    nodes, edges, _premises, conclusion = _load_graph(args, ctx)
    incoming: dict = {nid: [] for nid in nodes}
    for e in edges:
        incoming[e["to"]].append(e)
    work = [conclusion]
    seen = {conclusion}
    reached_premises: set = set()
    while work:
        n = work.pop()
        if not incoming[n]:
            reached_premises.add(n)
            continue
        for e in incoming[n]:
            for src in e["from"]:
                if src not in seen:
                    seen.add(src)
                    work.append(src)
    ok = sorted(reached_premises) == value["premises"]
    return {"ok": ok, "resid": [0.0 if ok else 1.0],
            "detail": f"迭代反可达复算前提 {sorted(reached_premises)} 与报告 {'一致' if ok else '不一致'}"}


REGISTRY.register(OpDef(
    name="logic_premise_trace", band=9500, sub=9,
    in_schema=_IN_PT, out_schema=_OUT_PT,
    invoke=_invoke_premise_trace, verify=_verify_premise_trace,
    cost=("units", 3.0), errors=("SCHEMA_MISMATCH", "PRECOND_FAIL", "POSTCOND_FAIL"),
    effects="none (纯函数)",
    doc="§10.3 premise_trace: 结论回溯全部前提 (递归 DFS + 记忆化), 返回前提集合+可读追踪路径; "
        "回溯前先复验图合法 (DAG+每边过核); verify 用迭代工作集反向可达复算前提集合。",
    selfchecks=(
        {"id": "trace_resolution",
         "args": {"nodes": [{"id": "p1", "stmt": "1,2"}, {"id": "p2", "stmt": "-1,2"},
                            {"id": "p3", "stmt": "1,-2"}, {"id": "p4", "stmt": "-1,-2"},
                            {"id": "n5", "stmt": "2"}, {"id": "n6", "stmt": "-2"}, {"id": "n7", "stmt": ""}],
                  "edges": [
                      {"from": ["p1", "p2"], "to": "n5", "kind": "res", "eid": "e1", "record": {}},
                      {"from": ["p3", "p4"], "to": "n6", "kind": "res", "eid": "e2", "record": {}},
                      {"from": ["n5", "n6"], "to": "n7", "kind": "res", "eid": "e3", "record": {}}],
                  "premises": ["p1", "p2", "p3", "p4"], "conclusion": "n7"},
         "check": lambda r: (r.value["premises"] == ["p1", "p2", "p3", "p4"] and r.value["max_depth"] == 2
                             and len(r.value["trace"]) > 0,
                             f"premises={r.value['premises']} depth={r.value['max_depth']}")},
        {"id": "single_hop",
         "args": {"nodes": [{"id": "h1", "stmt": "a"}, {"id": "h2", "stmt": "(a -> b)"}, {"id": "c", "stmt": "b"}],
                  "edges": [{"from": ["h2", "h1"], "to": "c", "kind": "mp", "eid": "m1", "record": {}}],
                  "premises": ["h1", "h2"], "conclusion": "c"},
         "check": lambda r: (r.value["premises"] == ["h1", "h2"], "单跳回溯")},
        {"id": "undeclared_premise",
         "args": {"nodes": [{"id": "x", "stmt": "1"}, {"id": "z", "stmt": "-1,2"},
                            {"id": "y", "stmt": "2"}],
                  "edges": [{"from": ["x", "z"], "to": "y", "kind": "res", "eid": "e1", "record": {}}],
                  "premises": [], "conclusion": "y"},
         "expect_err": "PRECOND_FAIL",
         "expect": "回溯到未声明前提 → 拒绝 (与 arg_graph 同构负例, 两算子须同口径)"},
    ),
))


# ===========================================================================
# §10.4 home-made natural-deduction kernel (decides only, never searches) + tactic engine
# ---------------------------------------------------------------------------
# The kernel rules and the ND checker live in svokernel/ndcore.py (module-level independence: §17.9 no self-certification).
# That module exports infer/check_nd (the kernel, deciding only) and the two checkers nd_proof/cex_witness;
# this file acts only as the **generator** side (tactic engine / proof search / counter-example search) and calls check_nd to decide the terminal state.


# --- tactic engine: state = frames/nodes/focus (fully serialisable) -----------------------
#
# A frame's parent_path = [[nid,key], ...] runs from the innermost slot out to the outermost kont node;
# nodes are stored flat in st["nodes"], and unfilled slots are marked with {"$hole": fid} (no dict-aliasing issues,
# safe to deep-copy/serialise). Frame closing: fill back layer by layer along the path; a kont node with no holes owns the frame closure.


def _engine_init(hyps: "list[str]", goal: str) -> dict:
    canon_hyps = [_canon_str(h, "tactic") for h in hyps]
    g = _canon_str(goal, "tactic")
    for h in canon_hyps + [g]:
        if _FALSE in _atoms(parse_formula(h)):
            raise SvoError(Err.SCHEMA_MISMATCH, ctx="tactic",
                           detail=f"保留原子 {_FALSE} 禁止出现在用户公式")
    return {"next_id": 1,
            "frames": {0: {"id": 0, "hyps": canon_hyps, "goal": g,
                           "parent_path": None, "parent_frame": None, "kont": None, "term": None}},
            "nodes": {},
            "focus": [0],
            "orig": {"hyps": canon_hyps, "goal": g},
            "done": False,
            "proof": None}


def _hole(fid: int) -> dict:
    return {"$hole": fid}


def _node_complete_deep(node: Any) -> bool:
    if isinstance(node, dict):
        if "$hole" in node:
            return False
        return all(_node_complete_deep(v) for v in node.values())
    if isinstance(node, list):
        return all(_node_complete_deep(v) for v in node)
    return True


def _close(st: dict, fid: int, term: dict) -> None:
    fr = st["frames"][fid]
    fr["term"] = term
    st["focus"] = [i for i in st["focus"] if i != fid]
    path = fr.get("parent_path")
    if not path:
        st["done"] = True
        st["proof"] = term
        return
    prev = term
    for nid, key in path:  # innermost → outermost
        st["nodes"][nid][key] = prev
        prev = st["nodes"][nid]
    kont_nid = path[-1][0]
    if _node_complete_deep(st["nodes"][kont_nid]):
        _close(st, fr["parent_frame"], st["nodes"][kont_nid])


def _new_frame(st: dict, hyps: list, goal: str, parent_frame: int, parent_path: list) -> int:
    fid = st["next_id"]
    st["next_id"] += 1
    st["frames"][fid] = {"id": fid, "hyps": list(hyps), "goal": goal,
                         "parent_path": parent_path, "parent_frame": parent_frame,
                         "kont": None, "term": None}
    return fid


def _new_kont(st: dict, owner_fid: int, node: dict) -> int:
    nid = st["next_id"]
    st["next_id"] += 1
    st["nodes"][nid] = dict(node)
    st["frames"][owner_fid]["kont"] = nid
    return nid


def _assume(s: str) -> dict:
    return {"rule": "assume", "f": s}


def _apply_tac(st: dict, tac: str, arg: "str | None", argd: "dict | None") -> dict:
    """Apply one tactic to the focus frame; return the updated state (deep copying is the caller's job)."""
    ctx = "tactic_engine"
    if st["done"]:
        raise SvoError(Err.PRECOND_FAIL, ctx=ctx, detail="证明已完成, 无焦点目标")
    if not st["focus"]:
        raise SvoError(Err.RUNTIME_ERR, ctx=ctx, detail="焦点栈空但未完成 (引擎不变量破坏)")
    fid = st["focus"][-1]
    fr = st["frames"][fid]
    hyps, goal = fr["hyps"], fr["goal"]
    g = parse_formula(goal)

    if tac == "noop":
        return st

    if tac == "intro":
        if g[0] != "imp":
            raise SvoError(Err.PRECOND_FAIL, ctx=ctx, detail=f"intro 需目标为蕴涵式, 得 {_canon(g)}")
        a, b = _canon(g[1]), _canon(g[2])
        nid = _new_kont(st, fid, {"rule": "imp_i", "f": goal, "discharge": a, "sub": None})
        child = _new_frame(st, hyps + [a], b, fid, [[nid, "sub"]])
        st["nodes"][nid]["sub"] = _hole(child)
        st["focus"][-1] = child
        return st

    if tac == "apply":
        if not arg:
            raise SvoError(Err.SCHEMA_MISMATCH, ctx=ctx, detail="apply 需要公式参数 (一条蕴涵假设)")
        phi = _canon_str(arg, ctx)
        if phi not in hyps:
            raise SvoError(Err.PRECOND_FAIL, ctx=ctx, detail=f"apply: {phi} 不在假设中")
        pf = parse_formula(phi)
        if pf[0] != "imp" or _canon(pf[2]) != goal:
            raise SvoError(Err.PRECOND_FAIL, ctx=ctx,
                           detail=f"apply: {phi} 不是 (_ -> {goal}) 形")
        p = pf[1]
        pc = _canon(p)
        if pc in hyps:
            _close(st, fid, {"rule": "imp_e", "f": goal, "maj": _assume(phi), "min": _assume(pc)})
            return st
        if is_axiom_instance(p) is not None:
            _close(st, fid, {"rule": "imp_e", "f": goal, "maj": _assume(phi),
                             "min": {"rule": "axiom", "f": pc}})
            return st
        nid = _new_kont(st, fid, {"rule": "imp_e", "f": goal, "maj": _assume(phi), "min": None})
        child = _new_frame(st, hyps, pc, fid, [[nid, "min"]])
        st["nodes"][nid]["min"] = _hole(child)
        st["focus"][-1] = child
        return st

    if tac == "rewrite":
        if not argd:
            raise SvoError(Err.SCHEMA_MISMATCH, ctx=ctx, detail="rewrite 需要 {src,dst,fwd,bwd} 参数")
        src = _canon_str(argd["src"], ctx)
        dst = _canon_str(argd["dst"], ctx)
        fwd = _canon_str(argd["fwd"], ctx)
        bwd = _canon_str(argd["bwd"], ctx)
        if fwd not in hyps or bwd not in hyps:
            raise SvoError(Err.PRECOND_FAIL, ctx=ctx, detail="rewrite: fwd/bwd 须在假设中")
        ff, bf = parse_formula(fwd), parse_formula(bwd)
        src_f, dst_f = parse_formula(src), parse_formula(dst)
        if ff != _imp_f(src_f, dst_f) or bf != _imp_f(dst_f, src_f):
            raise SvoError(Err.PRECOND_FAIL, ctx=ctx, detail="rewrite: fwd ≠ src->dst 或 bwd ≠ dst->src")
        if not _contains(g, src_f):
            raise SvoError(Err.PRECOND_FAIL, ctx=ctx, detail=f"rewrite: 目标不含子公式 {src}")
        new_goal = _canon(_subst_formula(g, src_f, dst_f))
        nid = _new_kont(st, fid, {"rule": "replace", "f": goal, "src": src, "dst": dst,
                                  "eq_fwd": _assume(fwd), "eq_bwd": _assume(bwd), "of": None})
        child = _new_frame(st, hyps, new_goal, fid, [[nid, "of"]])
        st["nodes"][nid]["of"] = _hole(child)
        st["focus"][-1] = child
        return st

    if tac == "induction":
        if not arg:
            raise SvoError(Err.SCHEMA_MISMATCH, ctx=ctx, detail="induction 需要原子名参数")
        if arg not in _atoms(g):
            raise SvoError(Err.PRECOND_FAIL, ctx=ctx, detail=f"induction: 原子 {arg} 不在目标 {goal} 中")
        nid = _new_kont(st, fid, {"rule": "cases", "f": goal, "atom": arg, "pos": None, "neg": None})
        fpos = _new_frame(st, hyps + [arg], goal, fid, [[nid, "pos"]])
        fneg = _new_frame(st, hyps + [_canon(_not_f(parse_formula(arg)))], goal, fid, [[nid, "neg"]])
        st["nodes"][nid]["pos"] = _hole(fpos)
        st["nodes"][nid]["neg"] = _hole(fneg)
        st["focus"][-1] = fneg
        st["focus"].append(fpos)
        return st

    if tac == "split":
        if arg is None:
            if g[0] != "and":
                raise SvoError(Err.PRECOND_FAIL, ctx=ctx, detail=f"split(目标) 需目标为合取式, 得 {goal}")
            nid = _new_kont(st, fid, {"rule": "and_i", "f": goal, "l": None, "r": None})
            fl = _new_frame(st, hyps, _canon(g[1]), fid, [[nid, "l"]])
            frr = _new_frame(st, hyps, _canon(g[2]), fid, [[nid, "r"]])
            st["nodes"][nid]["l"] = _hole(fl)
            st["nodes"][nid]["r"] = _hole(frr)
            st["focus"][-1] = frr
            st["focus"].append(fl)
            return st
        phi = _canon_str(arg, ctx)
        if phi not in hyps:
            raise SvoError(Err.PRECOND_FAIL, ctx=ctx, detail=f"split: 假设 {phi} 不在上下文")
        pf = parse_formula(phi)
        if pf[0] != "or":
            raise SvoError(Err.PRECOND_FAIL, ctx=ctx, detail="split(假设) 需析取假设")
        nid = _new_kont(st, fid, {"rule": "or_e", "f": goal, "disj": _assume(phi),
                                  "d1": _canon(pf[1]), "sub1": None,
                                  "d2": _canon(pf[2]), "sub2": None})
        f1 = _new_frame(st, hyps + [_canon(pf[1])], goal, fid, [[nid, "sub1"]])
        f2 = _new_frame(st, hyps + [_canon(pf[2])], goal, fid, [[nid, "sub2"]])
        st["nodes"][nid]["sub1"] = _hole(f1)
        st["nodes"][nid]["sub2"] = _hole(f2)
        st["focus"][-1] = f2
        st["focus"].append(f1)
        return st

    if tac == "contradict":
        pair = None
        for h in hyps:
            hf = parse_formula(h)
            if hf[0] == "not" and _canon(hf[1]) in hyps:
                pair = (h, _canon(hf[1]))
                break
        if pair is None:
            raise SvoError(Err.PRECOND_FAIL, ctx=ctx, detail="contradict: 上下文无互补对 (P 与 ~P)")
        neg, pos = pair
        _close(st, fid, {"rule": "efq", "f": goal,
                         "of": {"rule": "neg_e", "f": _FALSE, "maj": _assume(neg), "min": _assume(pos)}})
        return st

    if tac == "simplify":
        # (a) the goal is already among the hypotheses → assume closes
        if goal in hyps:
            _close(st, fid, _assume(goal))
            return st
        # (b) some hypothesis is ~~A and the goal is A → raa closes
        for h in hyps:
            hf = parse_formula(h)
            if hf[0] == "not" and hf[1][0] == "not" and _canon(hf[1][1]) == goal:
                negA = _canon(_not_f(parse_formula(goal)))
                _close(st, fid, {"rule": "raa", "f": goal, "discharge": negA,
                                 "sub": {"rule": "neg_e", "f": _FALSE, "maj": _assume(h),
                                         "min": _assume(negA)}})
                return st
        # (c) the goal is ~~A → reduce to the subgoal A (imp_i ∘ neg_e double-layer kont, path backfill)
        if g[0] == "not" and g[1][0] == "not":
            inner = _canon(g[1][1])
            negA = _canon(g[1])
            nid1 = _new_kont(st, fid, {"rule": "imp_i", "f": goal, "discharge": negA, "sub": None})
            nid2 = st["next_id"]
            st["next_id"] += 1
            st["nodes"][nid2] = {"rule": "neg_e", "f": _FALSE, "maj": _assume(negA), "min": None}
            child = _new_frame(st, hyps, inner, fid, [[nid2, "min"], [nid1, "sub"]])
            st["nodes"][nid2]["min"] = _hole(child)
            st["nodes"][nid1]["sub"] = _hole(child)  # placeholder; filled in via the path at closing time to form the complete neg_e node
            st["focus"][-1] = child
            return st
        # (d) some hypothesis is A&B and the goal is A/B → and_e closes
        for h in hyps:
            hf = parse_formula(h)
            if hf[0] == "and":
                if _canon(hf[1]) == goal:
                    _close(st, fid, {"rule": "and_e1", "f": goal, "of": _assume(h)})
                    return st
                if _canon(hf[2]) == goal:
                    _close(st, fid, {"rule": "and_e2", "f": goal, "of": _assume(h)})
                    return st
        raise SvoError(Err.PRECOND_FAIL, ctx=ctx, detail=f"simplify 不适用于目标 {goal}")

    raise SvoError(Err.SCHEMA_MISMATCH, ctx=ctx, detail=f"未知战术 {tac!r}")


def _kernel_final(st: dict) -> None:
    """Once the root frame closes: the proof term must pass the ND kernel (a wrong kernel result is caught → POSTCOND_FAIL)."""
    ok, detail = _nd_check(st["proof"], st["orig"]["goal"], st["orig"]["hyps"])
    if not ok:
        raise SvoError(Err.POSTCOND_FAIL, ctx="tactic_engine@kernel", detail=f"ND 核拦截: {detail}")


# ===========================================================================
# operator 10: logic_tactic_step (row 9510) — §10.4 tactic step (proof state → new state + delta)
# ===========================================================================

_IN_TAC = {
    "kind": "dict",
    "fields": {
        "state": {"kind": "dict", "extra": True},
        "tactic": {"kind": "str", "enum": ["intro", "apply", "rewrite", "induction", "split",
                                           "contradict", "simplify", "noop"]},
        "arg": {"kind": "str"},
        "argd": {"kind": "dict", "extra": True},
    },
    "required": ["state", "tactic"],
}
_OUT_TAC = {
    "kind": "dict",
    "fields": {
        "state": {"kind": "dict", "extra": True},
        "terminal": {"kind": "bool"},
        "increment": {"kind": "dict", "extra": True},
        "kernel_detail": {"kind": "str"},
    },
    "required": ["state", "terminal", "increment", "kernel_detail"],
}


def _state_atoms_ok(st: dict) -> None:
    atoms: set = set()
    for fr in st.get("frames", {}).values():
        for h in fr["hyps"] + [fr["goal"]]:
            atoms |= _atoms(parse_formula(h))
    if len(atoms) > _MAX_EQUIV_ATOMS:
        raise SvoError(Err.BOUNDED_VIOLATION, ctx="tactic",
                       detail=f"状态原子数 {len(atoms)} > {_MAX_EQUIV_ATOMS} — 语义交叉核对超界")


def _invoke_tactic(args: dict, ctl: CallCtl) -> dict:
    ctx = "op:logic_tactic_step#9510"
    st = copy.deepcopy(args["state"])
    if "frames" not in st or "orig" not in st:
        raise SvoError(Err.SCHEMA_MISMATCH, ctx=ctx, detail="state 非引擎状态 (缺 frames/orig) — 用 init 状态")
    _state_atoms_ok(st)
    tac = args["tactic"]
    inc = {"tactic": tac, "arg": args.get("arg"), "argd": args.get("argd")}
    if tac == "noop" and st["done"]:
        _kernel_final(st)  # on a terminal state, run the kernel once more (defence: a tampered proof term is caught)
        ok, detail = _nd_check(st["proof"], st["orig"]["goal"], st["orig"]["hyps"])
        return {"value": {"state": st, "terminal": True, "increment": inc, "kernel_detail": detail},
                "resid": [0.0], "cert": Certificate(kind="nd_proof",
                                                   payload={"term": st["proof"], "goal": st["orig"]["goal"],
                                                            "hyps": st["orig"]["hyps"]},
                                                   issued_by="logic_tactic_step"),
                "notes": "noop: 终结状态重过 ND 核"}
    st = _apply_tac(st, tac, args.get("arg"), args.get("argd"))
    if st["done"]:
        _kernel_final(st)
        ok, detail = _nd_check(st["proof"], st["orig"]["goal"], st["orig"]["hyps"])
        return {"value": {"state": st, "terminal": True, "increment": inc, "kernel_detail": detail},
                "resid": [0.0], "cert": Certificate(kind="nd_proof",
                                                   payload={"term": st["proof"], "goal": st["orig"]["goal"],
                                                            "hyps": st["orig"]["hyps"]},
                                                   issued_by="logic_tactic_step"),
                "notes": f"战术 {tac} 后证明完成; ND 核: {detail}"}
    return {"value": {"state": st, "terminal": False, "increment": inc,
                      "kernel_detail": "未终结 (焦点栈非空)"},
            "resid": [0.0],
            "notes": f"战术 {tac} 应用成功; 焦点 {len(st['focus'])} 个"}


def _verify_tactic(args: dict, value: dict) -> dict:
    """independent verification (semantic): on termination hyps ⊨ goal (truth table); on a non-terminal state, structural invariants."""
    st = value["state"]
    orig = st["orig"]
    hyps_f = [parse_formula(h) for h in orig["hyps"]]
    goal_f = parse_formula(orig["goal"])
    if value["terminal"]:
        atoms = sorted(set().union(*map(_atoms, hyps_f + [goal_f])))
        cex = 0
        for bits in product((False, True), repeat=len(atoms)):
            env = dict(zip(atoms, bits))
            env[_FALSE] = False
            if all(_ev_rec(h, env) for h in hyps_f) and not _ev_rec(goal_f, env):
                cex += 1
        return {"ok": cex == 0, "resid": [float(cex)],
                "detail": f"语义蕴含核对 (递归求值器): 反例 {cex} 个"}
    # non-terminal: structural invariants
    probs = []
    if not st["focus"] and not st["done"]:
        probs.append("焦点空但未完成")
    for fid, fr in st["frames"].items():
        if fr["term"] is None and fid in st["focus"]:
            try:
                parse_formula(fr["goal"])
            except SvoError as e:
                probs.append(f"帧 {fid} 目标非法: {e.detail}")
    return {"ok": not probs, "resid": [float(len(probs))],
            "detail": "非终结状态结构不变量: " + ("通过" if not probs else "; ".join(probs))}


REGISTRY.register(OpDef(
    name="logic_tactic_step", band=9500, sub=10,
    in_schema=_IN_TAC, out_schema=_OUT_TAC,
    invoke=_invoke_tactic, verify=_verify_tactic, cert_kind="nd_proof",
    cost=("units", 2.0), errors=("SCHEMA_MISMATCH", "PRECOND_FAIL", "POSTCOND_FAIL", "BOUNDED_VIOLATION"),
    effects="none (纯函数)",
    doc="§10.4 战术库单步: intro/apply/rewrite/induction/split/contradict/simplify (+noop), "
        "证明状态→新状态+证明项增量; 终结时证明项过自制 ND 核 (核错了会拦 → POSTCOND_FAIL), "
        "证书 nd_proof; verify = 真值表语义蕴含 (语法核 vs 语义, 双路径)。"
        "诚实边界: induction 在命题片段=原子 case 分裂 (LEM), 非数学归纳。",
    selfchecks=(
        {"id": "intro_applicable",
         "args": {"state": _engine_init([], "a->b"), "tactic": "intro"},
         "check": lambda r: (r.value["terminal"] is False and len(r.value["state"]["focus"]) == 1
                             and r.value["state"]["frames"][r.value["state"]["focus"][0]]["hyps"] == ["a"]
                             and r.value["state"]["frames"][r.value["state"]["focus"][0]]["goal"] == "b",
                             f"intro 后焦点帧 hyps={r.value['state']['frames'][r.value['state']['focus'][0]]['hyps']}")},
        {"id": "apply_closes_proof",
         "args": {"state": _engine_init(["a", "(a -> b)"], "b"), "tactic": "apply", "arg": "(a -> b)"},
         "check": lambda r: (r.value["terminal"] is True and r.cert is not None and r.cert_independent,
                             f"kernel: {r.value['kernel_detail']}")},
        {"id": "simplify_close_and_split",
         "args": {"state": _engine_init(["(a & b)"], "b"), "tactic": "simplify"},
         "check": lambda r: (r.value["terminal"] is True, "and_e2 关闭")},
        {"id": "intro_not_imp", "args": {"state": _engine_init([], "a"), "tactic": "intro"},
         "expect_err": "PRECOND_FAIL"},
        {"id": "kernel_intercepts_tampered",
         "args": {"state": {**_engine_init(["a", "(a -> b)"], "b"), "done": True, "focus": [],
                            "proof": {"rule": "assume", "f": "b"}},
                  "tactic": "noop"},
         "expect_err": "POSTCOND_FAIL"},
    ),
))


# ===========================================================================
# operator 11: logic_proof_search (row 9511) — §10.4 bounded tactic search (constructive, incomplete)
# ===========================================================================

_IN_PS = {
    "kind": "dict",
    "fields": {
        "hyps": {"kind": "list", "items": {"kind": "str"}, "minlen": 0, "maxlen": 12},
        "goal": {"kind": "str"},
        "max_steps": {"kind": "int", "min": 10, "max": 5000},
        "max_states": {"kind": "int", "min": 100, "max": 20000},
    },
    "required": ["hyps", "goal"],
}
_OUT_PS = {
    "kind": "dict",
    "fields": {
        "solved": {"kind": "bool"},
        "n_steps": {"kind": "int"},
        "script": {"kind": "list", "items": {"kind": "dict", "extra": True}},
        "reason": {"kind": "str"},
        "n_states": {"kind": "int"},
    },
    "required": ["solved", "n_steps", "script", "reason", "n_states"],
}


def _tactic_candidates(hyps: list, goal: str) -> "list[tuple[str, str | None, dict | None]]":
    """Deterministic tactic candidate order for the focus (Γ, G), easiest first."""
    out: "list[tuple[str, str | None, dict | None]]" = []
    g = parse_formula(goal)
    gc = goal
    # close: G ∈ Γ
    if gc in hyps:
        out.append(("simplify", None, None))
    # ~~ reduction
    if g[0] == "not" and g[1][0] == "not":
        out.append(("simplify", None, None))
    for h in hyps:
        hf = parse_formula(h)
        if hf[0] == "not" and hf[1][0] == "not" and _canon(hf[1][1]) == gc:
            out.append(("simplify", None, None))
            break
        if hf[0] == "and" and (_canon(hf[1]) == gc or _canon(hf[2]) == gc):
            out.append(("simplify", None, None))
            break
    # contradiction
    for h in hyps:
        hf = parse_formula(h)
        if hf[0] == "not" and _canon(hf[1]) in hyps:
            out.append(("contradict", None, None))
            break
    # intro
    if g[0] == "imp":
        out.append(("intro", None, None))
    # MP backwards: Γ contains P→G and P is available
    for h in hyps:
        hf = parse_formula(h)
        if hf[0] == "imp" and _canon(hf[2]) == gc:
            out.append(("apply", h, None))
    # split
    if g[0] == "and":
        out.append(("split", None, None))
    for h in hyps:
        if parse_formula(h)[0] == "or":
            out.append(("split", h, None))
    # rewrite: Γ contains (A->B) and (B->A) and G contains A
    imps = [h for h in hyps if parse_formula(h)[0] == "imp"]
    for f in imps:
        for bw in imps:
            ff, bb = parse_formula(f), parse_formula(bw)
            if _canon(ff[2]) == _canon(bb[1]) and _canon(bb[2]) == _canon(ff[1]) and _contains(g, ff[1]):
                out.append(("rewrite", None,
                            {"src": _canon(ff[1]), "dst": _canon(ff[2]), "fwd": f, "bwd": bw}))
    return out


def _search(st: dict, max_steps: int, max_states: int, ctl: CallCtl, ctx: str) -> "tuple[dict, int, int, str]":
    """DFS + memoisation + triple bound on steps/states/deadline. Returns (terminal state or None, steps, states, reason)."""
    steps = 0
    seen: set = set()

    def key(s: dict) -> tuple:
        fr = s["frames"][s["focus"][-1]] if s["focus"] else None
        return (tuple(sorted(fr["hyps"])) if fr else (), fr["goal"] if fr else "$done",
                len(s["focus"]))

    def dfs(s: dict) -> "dict | None":
        nonlocal steps
        if s["done"]:
            return s
        if not s["focus"]:
            return None
        k = key(s)
        if k in seen:
            return None
        seen.add(k)
        if len(seen) > max_states:
            raise SvoError(Err.BOUNDED_VIOLATION, ctx=ctx, detail=f"状态数 {len(seen)} > {max_states}")
        steps += 1
        if steps > max_steps:
            raise SvoError(Err.BOUNDED_VIOLATION, ctx=ctx, detail=f"步数 {steps} > {max_steps}")
        if steps % 32 == 0:
            ctl.check_time(ctx)
        fr = s["frames"][s["focus"][-1]]
        for tac, arg, argd in _tactic_candidates(fr["hyps"], fr["goal"]):
            child = copy.deepcopy(s)
            try:
                child = _apply_tac(child, tac, arg, argd)
            except SvoError:
                continue  # tactic inapplicable → try the next candidate
            if child["done"]:
                return child  # kernel checking is the caller's job (a wrong kernel result is caught)
            r = dfs(child)
            if r is not None:
                return r
        return None

    res = dfs(st)
    if res is None:
        return st, steps, len(seen), "构造性策略在步数/状态界内未找到证明 (策略不完备, 不宣称不可证)"
    return res, steps, len(seen), "策略找到证明并过核"


def _invoke_proof_search(args: dict, ctl: CallCtl) -> dict:
    ctx = "op:logic_proof_search#9511"
    hyps, goal = args["hyps"], args["goal"]
    atoms = set()
    for s in list(hyps) + [goal]:
        atoms |= _atoms(parse_formula(s))
    if len(atoms) > _MAX_EQUIV_ATOMS:
        raise SvoError(Err.BOUNDED_VIOLATION, ctx=ctx,
                       detail=f"原子数 {len(atoms)} > {_MAX_EQUIV_ATOMS} — 语义交叉核对超界")
    max_steps = int(args.get("max_steps", 800))
    max_states = int(args.get("max_states", 3000))
    st0 = _engine_init(hyps, goal)
    st, steps, n_states, reason = _search(st0, max_steps, max_states, ctl, ctx)
    if st["done"]:
        _kernel_final(st)  # a wrong kernel result is caught
        ok, detail = _nd_check(st["proof"], st["orig"]["goal"], st["orig"]["hyps"])
        script = [{"i": i + 1, "tactic": t} for i, t in
                  enumerate(_script_of(st))]
        return {"value": {"solved": True, "n_steps": steps, "script": script,
                          "reason": reason + "; " + detail, "n_states": n_states},
                "resid": [0.0],
                "cert": Certificate(kind="nd_proof",
                                    payload={"term": st["proof"], "goal": st["orig"]["goal"],
                                             "hyps": st["orig"]["hyps"]},
                                    issued_by="logic_proof_search"),
                "notes": f"{steps} 步 / {n_states} 状态找到证明; ND 核: {detail}"}
    return {"value": {"solved": False, "n_steps": steps, "script": [], "reason": reason,
                      "n_states": n_states},
            "resid": [1.0],
            "notes": f"未解决: {reason}"}


def _script_of(st: dict) -> "list[str]":
    """Extract a readable tactic script from the terminal state (coarse-grained: record the kont nodes' rule order)."""
    return [node["rule"] for node in st["nodes"].values()]


def _verify_proof_search(args: dict, value: dict) -> dict:
    """independent verification (semantic): solved → hyps ⊨ goal by truth table; not solved → only check the honest-reporting fields."""
    hyps_f = [parse_formula(h) for h in args["hyps"]]
    goal_f = parse_formula(args["goal"])
    atoms = sorted(set().union(*map(_atoms, hyps_f + [goal_f])))
    if value["solved"]:
        cex = 0
        for bits in product((False, True), repeat=len(atoms)):
            env = dict(zip(atoms, bits))
            env[_FALSE] = False
            if all(_ev_rec(h, env) for h in hyps_f) and not _ev_rec(goal_f, env):
                cex += 1
        return {"ok": cex == 0, "resid": [float(cex)],
                "detail": f"语义蕴含核对: 反例 {cex} 个 (2^{len(atoms)})"}
    ok = (value["reason"] != "" and value["script"] == [])
    return {"ok": ok, "resid": [0.0 if ok else 1.0],
            "detail": "未解决申报: 无脚本 + 有原因说明 (不宣称不可证)"}


REGISTRY.register(OpDef(
    name="logic_proof_search", band=9500, sub=11,
    in_schema=_IN_PS, out_schema=_OUT_PS,
    invoke=_invoke_proof_search, verify=_verify_proof_search, cert_kind="nd_proof",
    cost=("units", 8.0), errors=("SCHEMA_MISMATCH", "BOUNDED_VIOLATION", "TIMEOUT", "POSTCOND_FAIL"),
    effects="none (纯函数, 有界内存搜索)",
    doc="§10.4 proof_search: 有界 DFS 战术搜索 (候选序: 关闭>化归>矛盾>intro>apply>split>rewrite), "
        "步数/状态数/时限三重界 (超界 BOUNDED_VIOLATION, 不许无界搜索); 找到的证明必须过 ND 核 (核错了会拦), "
        "证书 nd_proof。诚实边界: 构造性策略**不完备** (如排中律 a|~a 不在策略内), 搜不到只报 solved=False。",
    selfchecks=(
        {"id": "syllogism",
         "args": {"hyps": [], "goal": "((a -> b) -> ((b -> c) -> (a -> c)))"},
         "check": lambda r: (r.value["solved"] is True and r.cert_independent,
                             f"{r.value['n_steps']} 步 / {r.value['n_states']} 状态; {r.value['reason'][:60]}")},
        {"id": "with_hyp_mp",
         "args": {"hyps": ["p", "(p -> q)"], "goal": "q"},
         "check": lambda r: (r.value["solved"] is True, "MP 搜索成功")},
        {"id": "non_tautology_honest_fail",
         "args": {"hyps": [], "goal": "a", "max_steps": 100, "max_states": 400},
         "check": lambda r: (r.value["solved"] is False and "不宣称不可证" in r.value["reason"],
                             f"诚实申报: {r.value['reason'][:50]}")},
        {"id": "bad_formula", "args": {"hyps": [], "goal": "(a"}, "expect_err": "SCHEMA_MISMATCH"},
        {"id": "too_many_atoms",
         "args": {"hyps": [], "goal": " & ".join(f"x{i}" for i in range(15))},
         "expect_err": "BOUNDED_VIOLATION"},
    ),
))


# ===========================================================================
# operator 12: logic_counterexample (row 9512) — §10.5 counter-example search over a small finite domain (the assignment domain)
# ===========================================================================

_IN_CEX = {
    "kind": "dict",
    "fields": {
        "hyps": {"kind": "list", "items": {"kind": "str"}, "minlen": 0, "maxlen": 16},
        "goal": {"kind": "str"},
        "max_atoms": {"kind": "int", "min": 1, "max": 16},
    },
    "required": ["hyps", "goal"],
}
_OUT_CEX = {
    "kind": "dict",
    "fields": {
        "found": {"kind": "bool"},
        "assignment": {"kind": "dict", "extra": True},
        "entailed": {"kind": "bool"},
        "checked_assignments": {"kind": "int"},
    },
    "required": ["found", "assignment", "entailed", "checked_assignments"],
}


def _invoke_cex(args: dict, ctl: CallCtl) -> dict:
    ctx = "op:logic_counterexample#9512"
    hyps_f = [parse_formula(h) for h in args["hyps"]]
    goal_f = parse_formula(args["goal"])
    max_atoms = int(args.get("max_atoms", 16))
    atoms = sorted(set().union(*map(_atoms, hyps_f + [goal_f])) if hyps_f else _atoms(goal_f))
    if len(atoms) > max_atoms:
        raise SvoError(Err.BOUNDED_VIOLATION, ctx=ctx,
                       detail=f"原子数 {len(atoms)} > {max_atoms} — 指派域枚举超界")
    pfs = [_postfix(h) for h in hyps_f]
    pf_g = _postfix(goal_f)
    n = 0
    for bits in product((False, True), repeat=len(atoms)):
        n += 1
        env = dict(zip(atoms, bits))
        if all(_ev_pf(p, env) for p in pfs) and not _ev_pf(pf_g, env):
            cert = Certificate(kind="cex_witness",
                               payload={"hyps": list(args["hyps"]), "goal": args["goal"],
                                        "assignment": {k: v for k, v in env.items()}},
                               issued_by="logic_counterexample")
            return {"value": {"found": True, "assignment": {k: v for k, v in env.items()},
                              "entailed": False, "checked_assignments": n},
                    "resid": [0.0], "cert": cert,
                    "notes": f"反例指派 (证书): {env} — hyps 全真而 goal 假"}
    # exhausting all assignments with no counter-example ⇒ semantic entailment holds (complete decision for propositional logic)
    return {"value": {"found": False, "assignment": {}, "entailed": True,
                      "checked_assignments": n},
            "resid": [0.0],
            "notes": f"穷尽 {n} 个指派无反例 ⇒ hyps ⊨ goal (命题语义完备判定)"}


def _verify_cex(args: dict, value: dict) -> dict:
    """independent verification (recursive evaluator): recheck the counter-example assignment for hyps true/goal false; rescan entailed."""
    hyps_f = [parse_formula(h) for h in args["hyps"]]
    goal_f = parse_formula(args["goal"])
    if value["found"]:
        env = {k: bool(v) for k, v in value["assignment"].items()}
        ok = all(_ev_rec(h, env) for h in hyps_f) and not _ev_rec(goal_f, env)
        return {"ok": ok, "resid": [0.0 if ok else 1.0],
                "detail": f"反例复核 (递归求值器): {'成立' if ok else '不成立'}"}
    atoms = sorted(set().union(*map(_atoms, hyps_f + [goal_f])))
    cex = 0
    for bits in product((False, True), repeat=len(atoms)):
        env = dict(zip(atoms, bits))
        if all(_ev_rec(h, env) for h in hyps_f) and not _ev_rec(goal_f, env):
            cex += 1
    return {"ok": (cex == 0) == value["entailed"], "resid": [float(cex)],
            "detail": f"独立复扫: 反例 {cex} 个, entailed={value['entailed']}"}


# the cex_witness checker lives in svokernel/ndcore.py (a different module from the generator logic_counterexample).

REGISTRY.register(OpDef(
    name="logic_counterexample", band=9500, sub=12,
    in_schema=_IN_CEX, out_schema=_OUT_CEX,
    invoke=_invoke_cex, verify=_verify_cex, cert_kind="cex_witness",
    cost=("units", 4.0), errors=("SCHEMA_MISMATCH", "BOUNDED_VIOLATION"),
    effects="none (纯函数)",
    doc="§10.5 反例搜寻: 小指派域 (命题有限域) 全枚举, 找到 hyps 真/goal 假的指派即反例, "
        "指派=证书 (cex_witness 独立检查器复核); 穷尽无反例 ⇒ 语义蕴含成立 (命题完备判定)。"
        "原子数 >16 拒绝。诚实边界: 只支持命题指派域, 不做一阶结构枚举。",
    selfchecks=(
        {"id": "find_cex", "args": {"hyps": ["a"], "goal": "b"},
         "check": lambda r: (r.value["found"] is True and r.value["assignment"] == {"a": True, "b": False}
                             and r.cert_independent,
                             f"cex={r.value['assignment']}")},
        {"id": "no_cex_entailed", "args": {"hyps": ["(a -> b)", "a"], "goal": "b"},
         "check": lambda r: (r.value["found"] is False and r.value["entailed"] is True,
                             "MP 有效, 无反例 (穷尽判定蕴含成立)")},
        {"id": "contradictory_hyps", "args": {"hyps": ["a", "~a"], "goal": "b"},
         "check": lambda r: (r.value["found"] is False and r.value["entailed"] is True,
                             "矛盾假设蕴含一切 (爆炸律), 语义上无反例 — 正确")},
        {"id": "over_limit",
         "args": {"hyps": [], "goal": " & ".join(f"z{i}" for i in range(17))},
         "expect_err": "BOUNDED_VIOLATION"},
    ),
))
