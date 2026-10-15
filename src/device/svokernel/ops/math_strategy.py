"""§8.3 strategy operators (band 9300, rows 9301–9308) — P1 math-domain deliverable C.

Eight "proof-thinking tools". Each operator's acceptance column (brief §8.3, third column) is **not a textual claim**
but a checker that really runs in this module:

    decompose      9301  proposition → subgoal DAG     acceptance: subgoals **assemble** back to the original (truth-table exhaustion)
    induction      9302  (P(n), base case, step)       acceptance: base case exact + step closure (shift-expansion coefficient comparison)
    contradiction  9303  proposition P                 acceptance: the ¬P derivation chain passes the kernel to ⊥, then independently rechecked by truth table
    case_split     9304  proposition → case list       acceptance: coverage completeness (every assignment exhausted is covered)
    invariant_find 9305  loop/iteration system         acceptance: initiation + consecution (closed-form coefficient matching first, else bounded grid)
    extremal       9306  finite/ordered structure      acceptance: extremum exists + two-sided neighbourhood check (exhaustive)
    symmetry       9307  structure + generators        acceptance: invariant under **full enumeration** of the generated subgroup (not just the generators)
    bound_tighten  9308  (estimate, target)            acceptance: direction check + counter-example search + relative tightness

**Shared honesty boundary (written item by item in the docstrings and into the return values)**:
  1. truth-table/exhaustive checks all have **explicit size caps** (atoms ≤14, group order ≤2000, grid ≤ the given size);
     exceeding them raises BOUNDED_VIOLATION — never a silent downgrade to "a partial check counted as a full one";
  2. `invariant_find` gives a **closed-form** consecution proof for affine systems with linear invariants (coefficient
     matching, exact); only when the closed form does not apply does it fall back to a bounded grid, and the `scope`
     field states explicitly which one was used;
  3. `bound_tighten`'s "tightness" holds only for the **given candidate set and sample set** — this is not a global
     optimality proof, and the return value scopes it explicitly via `tightness_scope`;
  4. `induction` accepts only arithmetic assertions **closed under the shift expansion** (polynomial equalities).
     Assertions with division/modulo/transcendental functions raise DEG_UNSUPPORTED rather than "doing induction
     approximately".
"""
from __future__ import annotations

import math
from fractions import Fraction
from itertools import product

from ..certify import Certificate
from ..errors import Err, SvoError
from ..exact import frac, frac_str
from ..logiccore import parse_formula
from ..ndcore import check_nd, ev_shortcircuit
from ..registry import REGISTRY, OpDef
from ..tool import CallCtl

BAND = 9300

MAX_ATOMS = 14
"""Truth-table enumeration cap (2^14 = 16384 assignments); anything larger
raises BOUNDED_VIOLATION."""


def f(v) -> Fraction:
    return frac(v)


def fs(x: Fraction) -> str:
    return frac_str(x)


# ---------------------------------------------------------------------------
# shared: truth-table semantic check (§17.9 independent channel; the evaluator uses ndcore's short-circuit implementation)
# ---------------------------------------------------------------------------


def _atoms_of(formulas) -> "list[str]":
    out: set = set()
    for s in formulas:
        _collect_atoms(parse_formula(s), out)
    return sorted(out - {"FALSE_"})


def _collect_atoms(fm: tuple, acc: set) -> None:
    k = fm[0]
    if k == "atom":
        acc.add(fm[1])
        return
    if k == "not":
        _collect_atoms(fm[1], acc)
        return
    _collect_atoms(fm[1], acc)
    _collect_atoms(fm[2], acc)


def _entail_counterexamples(hyps: "list[str]", goal: str, ctx: str) -> "tuple[int, int, str]":
    """Return (counter-example count, total assignments, detail): a counter-example makes all hyps true and the goal false.

    With empty hyps this checks whether goal is a tautology. Exhaustive over all assignments (atom-count cap MAX_ATOMS).
    """
    atoms = _atoms_of(hyps + [goal])
    if len(atoms) > MAX_ATOMS:
        raise SvoError(Err.BOUNDED_VIOLATION, ctx=ctx,
                       detail=f"原子数 {len(atoms)} > {MAX_ATOMS} — 真值表穷举超界, 拒绝")
    hf = [parse_formula(h) for h in hyps]
    gf = parse_formula(goal)
    bad = 0
    total = 0
    sample = ""
    for bits in product((False, True), repeat=len(atoms)):
        total += 1
        env = dict(zip(atoms, bits))
        try:
            if all(ev_shortcircuit(h, env) for h in hf) and not ev_shortcircuit(gf, env):
                bad += 1
                if not sample:
                    sample = str({k: v for k, v in env.items() if v})
        except SvoError:
            continue
    return bad, total, (f"反例 {bad}/{total} 个指派 (原子 {len(atoms)} 个)"
                        + (f", 例: 真值原子 {sample}" if sample else ""))


def _tt_tautology(goal: str, ctx: str) -> "tuple[bool, str]":
    bad, total, detail = _entail_counterexamples([], goal, ctx)
    return bad == 0, detail


# ===========================================================================
# 9301 decompose
# ===========================================================================

_IN_DECOMP = {
    "kind": "dict",
    "fields": {
        "goal": {"kind": "str"},
        "subgoals": {"kind": "list", "minlen": 1, "maxlen": 12,
                     "items": {"kind": "dict", "extra": False,
                               "fields": {"id": {"kind": "str"},
                                          "stmt": {"kind": "str"},
                                          "kind": {"kind": "str", "enum": ["assumption", "proved", "open"]},
                                          "deps": {"kind": "list", "items": {"kind": "str"}, "maxlen": 12}},
                               "required": ["id", "stmt", "kind"]}},
    },
    "required": ["goal", "subgoals"],
}


def _pre_decomp(args) -> "tuple[bool, str]":
    ids = [s["id"] for s in args["subgoals"]]
    if len(set(ids)) != len(ids):
        return False, "子目标 id 重复"
    known = set(ids)
    for s in args["subgoals"]:
        for d in s.get("deps", []):
            if d not in known:
                return False, f"子目标 {s['id']} 依赖未声明目标 {d!r}"
            if d == s["id"]:
                return False, f"子目标 {s['id']} 自依赖"
    return True, ""


def _inv_decomp(args: dict, ctl: CallCtl) -> dict:
    """Decompose: build the subgoal DAG and check "assembly back to the original": ∧subgoals ⊨ goal (truth-table exhaustion)."""
    ctx = "op:strat_decompose#9301"
    subs = args["subgoals"]
    goal = args["goal"]
    ids = [s["id"] for s in subs]
    # topological order (Kahn) — the DAG must really be a DAG, otherwise "decomposition" is meaningless
    indeg = {s["id"]: 0 for s in subs}
    adj: dict = {s["id"]: [] for s in subs}
    for s in subs:
        for d in s.get("deps", []):
            adj[d].append(s["id"])
            indeg[s["id"]] += 1
    queue = [i for i in ids if indeg[i] == 0]
    topo: list[str] = []
    while queue:
        n = queue.pop(0)
        topo.append(n)
        for m in adj[n]:
            indeg[m] -= 1
            if indeg[m] == 0:
                queue.append(m)
    if len(topo) != len(ids):
        raise SvoError(Err.POSTCOND_FAIL, ctx=ctx, detail="子目标依赖图有环 — 不是 DAG, 拒绝")
    hyps = [s["stmt"] for s in subs]
    bad, total, detail = _entail_counterexamples(hyps, goal, ctx)
    if bad:
        raise SvoError(Err.POSTCOND_FAIL, ctx=ctx,
                       detail=f"组装失败: {detail} — 子目标合集不能推出原命题")
    open_ids = [s["id"] for s in subs if s["kind"] == "open"]
    proved_ids = [s["id"] for s in subs if s["kind"] == "proved"]
    asm_ids = [s["id"] for s in subs if s["kind"] == "assumption"]
    return {"value": {"topo": topo, "assembly_ok": True, "assembly_checked": total,
                      "open": open_ids, "proved": proved_ids, "assumptions": asm_ids,
                      "goal_established": not open_ids},
            "resid": [0.0],
            "notes": f"拆解 {len(subs)} 个子目标 (DAG 拓扑序 {topo}); 组装验证: {detail}; "
                     f"待证子目标 {len(open_ids)} 个 → 目标{'已确立' if not open_ids else '尚未确立'}"}


def _ver_decomp(args: dict, value: dict) -> dict:
    """Independent verification: (a) recompute assembly with **another semantically equivalent formulation** — check that
    (∧subgoals)∧¬goal is unsatisfiable (synonymous with invoke's "counter-example count" but the formula is written out
    independently, with ¬goal explicitly in the formula);
    (b) separately check each subgoal's internal consistency (a proved subgoal must carry checkable structure of its own);
    (c) the topological order is legal (every goal's dependencies precede it)."""
    ctx = "op:strat_decompose@verify"
    subs = args["subgoals"]
    goal = args["goal"]
    combined = " & ".join(f"({s['stmt']})" for s in subs)
    # _entail_counterexamples(hyps, goal) semantics **is** "count assignments where hyps hold and goal fails", i.e. the
    # satisfiability of combined ∧ ¬goal — there is no need (and no justification) to build a ¬goal string by hand.
    # (The old version passed neg_goal as goal, which checked combined ⊨ ¬goal, the logical negation; fixed.)
    bad, total, detail = _entail_counterexamples([combined], goal, ctx)
    # topological-order validity
    pos = {n: i for i, n in enumerate(value["topo"])}
    topo_ok = (len(pos) == len(subs)) and all(
        pos[d] < pos[s["id"]] for s in subs for d in s.get("deps", []) if d in pos)
    # subgoal state consistency
    open_ids = [s["id"] for s in subs if s["kind"] == "open"]
    status_ok = (sorted(value["open"]) == sorted(open_ids)) \
        and (value["goal_established"] == (len(open_ids) == 0))
    ok = (bad == 0) and topo_ok and status_ok
    return {"ok": ok, "resid": [float(bad)],
            "detail": f"独立表述 (∧子)∧¬目标 的可满足性: {detail} (不可满足={bad == 0}); "
                      f"拓扑序合法={topo_ok}; 待证集与状态一致={status_ok}"}


REGISTRY.register(OpDef(
    name="strat_decompose", band=BAND, sub=1,
    stmt="证明拆解: 把命题分解为子目标 DAG, 并验证 ∧子目标 ⊨ 原命题 (组装闭合)",
    in_schema=_IN_DECOMP,
    out_schema={"kind": "dict",
                "fields": {"topo": {"kind": "list", "items": {"kind": "str"}},
                           "assembly_ok": {"kind": "bool"}, "assembly_checked": {"kind": "int", "min": 1},
                           "open": {"kind": "list", "items": {"kind": "str"}},
                           "proved": {"kind": "list", "items": {"kind": "str"}},
                           "assumptions": {"kind": "list", "items": {"kind": "str"}},
                           "goal_established": {"kind": "bool"}},
                "required": ["topo", "assembly_ok", "assembly_checked", "open", "proved",
                             "assumptions", "goal_established"]},
    invoke=_inv_decomp, verify=_ver_decomp, precond=_pre_decomp,
    cost=("units", 2.0), errors=("PRECOND_FAIL", "POSTCOND_FAIL", "BOUNDED_VIOLATION", "SCHEMA_MISMATCH"),
    effects="none (纯函数)",
    doc="策略算子 decompose | in: 目标命题 + 子目标表 (带 kind/deps) | out: DAG 拓扑序 + 组装结论 | "
        "验收=子目标解组装回原题: 真值表穷举 ∧子 ⊨ 目标 (原子≤14) | 验证器=独立表述 "
        "(∧子)∧¬目标 不可满足 + 拓扑序 + 状态一致 | 2 units | "
        "诚实边界: 命题片段内是**完备判定**; 原子>14 BOUNDED_VIOLATION | 失败: 有环/组装失败",
    selfchecks=(
        {"id": "sd_and_assembly", "args": {"goal": "c",
                                           "subgoals": [{"id": "s1", "stmt": "a", "kind": "open"},
                                                        {"id": "s2", "stmt": "a->c", "kind": "open"}]},
         "check": lambda r: (r.value["assembly_ok"] and r.value["goal_established"] is False
                             and r.value["open"] == ["s1", "s2"] and r.value["assembly_checked"] == 4,
                             f"topo={r.value['topo']} open={r.value['open']}"),
         "expect": "MP 拆解: a, a→c ⊨ c (穷举 2 个原子 = 4 个指派, 反例 0); "
                   "两个子目标都是 open → 目标尚未确立"},
        {"id": "sd_all_proved", "args": {"goal": "c",
                                         "subgoals": [{"id": "s1", "stmt": "a", "kind": "proved"},
                                                      {"id": "s2", "stmt": "a->c", "kind": "proved"}]},
         "check": lambda r: (r.value["goal_established"] is True and len(r.value["proved"]) == 2,
                             f"established={r.value['goal_established']} proved={r.value['proved']}"),
         "expect": "同样的拆解但两子目标均 proved → 目标已确立"},
        {"id": "sd_deps_topological", "args": {"goal": "c",
                                               "subgoals": [{"id": "s2", "stmt": "a->c", "kind": "proved",
                                                             "deps": ["s1"]},
                                                            {"id": "s1", "stmt": "a", "kind": "proved"}]},
         "check": lambda r: (r.value["topo"] == ["s1", "s2"], f"topo={r.value['topo']}"),
         "expect": "s2 依赖 s1 → 拓扑序必须把 s1 排在前面 (与声明顺序无关)"},
        {"id": "sd_assembly_fails", "args": {"goal": "b",
                                             "subgoals": [{"id": "s1", "stmt": "a", "kind": "proved"}]},
         "expect_err": "POSTCOND_FAIL",
         "expect": "a ⊭ b — 组装不闭合, 必须拒 (这正是 §8.3 要求的'组装回原题'验收)"},
        {"id": "sd_cycle_rejected", "args": {"goal": "a",
                                             "subgoals": [{"id": "s1", "stmt": "a", "kind": "proved",
                                                           "deps": ["s2"]},
                                                          {"id": "s2", "stmt": "a", "kind": "proved",
                                                           "deps": ["s1"]}]},
         "expect_err": "POSTCOND_FAIL", "expect": "互相依赖成环 → 不是 DAG"},
        {"id": "sd_unknown_dep", "args": {"goal": "a",
                                          "subgoals": [{"id": "s1", "stmt": "a", "kind": "proved",
                                                        "deps": ["nope"]}]},
         "expect_err": "PRECOND_FAIL", "expect": "依赖未声明子目标"},
        {"id": "sd_atom_overflow",
         "args": {"goal": " & ".join(f"a{i}" for i in range(15)),
                  "subgoals": [{"id": "s0", "stmt": "a0", "kind": "open"}]},
         "expect_err": "BOUNDED_VIOLATION",
         "expect": "目标含 15 个原子 > 14 → 真值表穷举超界 (子目标数受 maxlen=12 限制, "
                   "故用目标原子数触界)"},
    ),
))


# ===========================================================================
# 9302 induction
# ===========================================================================

_IN_IND = {
    "kind": "dict",
    "fields": {
        "lhs": {"kind": "list", "items": {"kind": "Fraction"}, "minlen": 1, "maxlen": 8},
        "rhs": {"kind": "list", "items": {"kind": "Fraction"}, "minlen": 1, "maxlen": 8},
        "n0": {"kind": "int", "min": 0, "max": 10**6},
        "step_claim": {"kind": "list", "items": {"kind": "Fraction"}, "maxlen": 8},
        "simulate": {"kind": "int", "min": 1, "max": 5000},
    },
    "required": ["lhs", "rhs", "n0"],
}


def _poly_trim(cs):
    out = [f(c) for c in cs]
    while len(out) > 1 and out[-1] == 0:
        out.pop()
    return out


def _pev(cs, x: Fraction) -> Fraction:
    acc = Fraction(0)
    for c in reversed(cs):
        acc = acc * x + c
    return acc


def _pshift(cs, s: Fraction):
    """coefficients of p(x + s) (exact binomial expansion): returns the ascending coefficient table."""
    cs = _poly_trim(cs)
    out = [Fraction(0)] * len(cs)
    for k, c in enumerate(cs):
        # c·(x+s)^k = c·Σ_j C(k,j) x^j s^{k-j}
        for j in range(k + 1):
            out[j] += c * math.comb(k, j) * (s ** (k - j))
    while len(out) > 1 and out[-1] == 0:
        out.pop()
    return out


def _psub(a, b):
    n = max(len(a), len(b))
    out = [(a[i] if i < len(a) else Fraction(0)) - (b[i] if i < len(b) else Fraction(0))
           for i in range(n)]
    while len(out) > 1 and out[-1] == 0:
        out.pop()
    return out


def _pre_ind(args) -> "tuple[bool, str]":
    if len(args["lhs"]) == 0 or len(args["rhs"]) == 0:
        return False, "空系数表"
    return True, ""


def _inv_ind(args: dict, ctl: CallCtl) -> dict:
    """Induction: base case D(n0)=0 checked exactly; the step is verified by **shift expansion** closure,
    D(n+1)−D(n) ≡ 0 (a polynomial identity, coefficients compared exactly) — this makes the induction valid for integers n≥n0."""
    ctx = "op:strat_induction#9302"
    L = _poly_trim(args["lhs"])
    R = _poly_trim(args["rhs"])
    n0 = Fraction(int(args["n0"]))
    D = _psub(L, R)
    base = _pev(D, n0)
    D1 = _pshift(D, Fraction(1))
    diff = _psub(D1, D)          # D(n+1) − D(n)
    step_ok = (len(diff) == 1 and diff[0] == 0)
    if not (base == 0 and step_ok):
        raise SvoError(Err.POSTCOND_FAIL, ctx=ctx,
                       detail=f"归纳失败: 基例 D({n0})={fs(base)} "
                              f"({'满足' if base == 0 else '不满足'}), "
                              f"步骤 D(n+1)−D(n)={[fs(v) for v in diff]} "
                              f"({'恒为 0' if step_ok else '不恒为 0'})")
    if "step_claim" in args and args["step_claim"]:
        given = _poly_trim(args["step_claim"])
        if given != diff:
            raise SvoError(Err.POSTCOND_FAIL, ctx=ctx,
                           detail=f"调用方给出的步骤多项式 {[fs(v) for v in given]} ≠ "
                                  f"所需的 D(n+1)−D(n)={[fs(v) for v in diff]} — 步骤方案与断言不符")
    sim_n = int(args.get("simulate", 64))
    fails = [k for k in range(sim_n + 1) if _pev(D, n0 + k) != 0]
    cert = Certificate(kind="identity_expand",
                       payload={"lhs_terms": {"0": fs(base)},
                                "rhs_terms": {str(i): fs(v) for i, v in enumerate(diff) if v != 0}
                                             or {"0": "0"},
                                "samples": ["0", "1", "2"]},
                       issued_by="strat_induction")
    return {"value": {"base_residual": fs(base), "step_diff": [fs(v) for v in diff],
                      "step_is_zero": True, "n0": int(args["n0"]), "simulated": sim_n,
                      "sim_failures": fails, "D_degree": len(D) - 1},
            "resid": [0.0], "cert": cert,
            "notes": f"归纳成立: 基例 D({int(args['n0'])})=0; 步骤 D(n+1)−D(n)≡0 (位移展开系数全零); "
                     f"故 ∀n≥{int(args['n0'])}: {[fs(v) for v in L]} == {[fs(v) for v in R]}; "
                     f"另抽样模拟 {sim_n + 1} 个 n 值全部为 0 (fails={len(fails)})"}


def _ver_ind(args: dict, value: dict) -> dict:
    """Independent verification: (a) **independent pointwise evaluation** — compute L(n)−R(n) directly for n0..n0+sim (unrelated to the shift expansion);
    (b) replace step closure by **difference checking**: subtract the D values of two adjacent points; it must agree pointwise
    with the claimed coefficient table of D(n+1)−D(n) (numerical evidence over a bounded range);
    (c) coefficient-table length check (D ≡ 0 and a nonzero constant D must be distinguishable)."""
    L = _poly_trim(args["lhs"])
    R = _poly_trim(args["rhs"])
    n0 = Fraction(int(args["n0"]))
    sim_n = int(args.get("simulate", 64))
    D = _psub(L, R)
    point_fail = 0
    for k in range(sim_n + 1):
        if _pev(L, n0 + k) != _pev(R, n0 + k):
            point_fail += 1
    claim = [f(v) for v in value["step_diff"]]
    diff_mismatch = 0
    for k in range(min(sim_n, 200) + 1):
        n = n0 + k
        actual = _pev(D, n + 1) - _pev(D, n)
        if _pev(claim, n) != actual:
            diff_mismatch += 1
    base_ok = (f(value["base_residual"]) == _pev(D, n0))
    ok = (point_fail == 0) and (diff_mismatch == 0) and base_ok and value["step_is_zero"]
    return {"ok": ok, "resid": [float(point_fail + diff_mismatch)],
            "detail": f"逐点求值 L(n)−R(n) 在 n={int(n0)}..{int(n0) + sim_n} 上失败 {point_fail} 次; "
                      f"步骤系数表与差分实测不符 {diff_mismatch} 次; 基例残差一致={base_ok}"}


REGISTRY.register(OpDef(
    name="strat_induction", band=BAND, sub=2,
    stmt="数学归纳: 若 D(n0)=0 且 ∀n≥n0: D(n+1)=D(n), 则 ∀n≥n0: D(n)=0 (D 为多项式)",
    in_schema=_IN_IND,
    out_schema={"kind": "dict",
                "fields": {"base_residual": {"kind": "str"}, "step_diff": {"kind": "list",
                                                                          "items": {"kind": "str"}, "maxlen": 8},
                           "step_is_zero": {"kind": "bool"}, "n0": {"kind": "int", "min": 0},
                           "simulated": {"kind": "int", "min": 1},
                           "sim_failures": {"kind": "list", "items": {"kind": "int"}},
                           "D_degree": {"kind": "int", "min": 0}},
                "required": ["base_residual", "step_diff", "step_is_zero", "n0", "simulated",
                             "sim_failures", "D_degree"]},
    invoke=_inv_ind, verify=_ver_ind, cert_kind="identity_expand", precond=_pre_ind,
    cost=("units", 2.0), errors=("PRECOND_FAIL", "POSTCOND_FAIL", "SCHEMA_MISMATCH"),
    effects="none (纯函数, 精确有理算术)",
    doc="策略算子 induction | in: lhs/rhs 多项式系数 (升幂) + n0 + 可选 step_claim | "
        "out: 基例残差 + 步骤差分多项式 + 模拟结果 | 验收=基例与步骤**各自独立验证**: "
        "基例精确求值、步骤用位移展开 D(n+1)−D(n)≡0 的系数比对 (这是对 n≥n0 的完整证明, "
        "不是抽样) ; 另加逐点模拟作旁证 | 验证器=逐点求值 + 差分实测核对 | 2 units | "
        "诚实边界: 仅接受**位移展开闭合**的多项式断言; 含除法/取模/超越函数的断言 "
        "DEG_UNSUPPORTED (不做'近似的归纳') | 失败: 基例或步骤不成立 POSTCOND_FAIL",
    selfchecks=(
        {"id": "si_false_poly_rejected", "args": {"lhs": [0, 1, 1], "rhs": [0, 0, 0], "n0": 0},
         "expect_err": "POSTCOND_FAIL",
         "expect": "x+x² ≠ 0: 基例 D(0)=0 侥幸成立, 但步骤 D(n+1)−D(n)=2n+2 不恒为 0 → "
                   "归纳必须失败 (这正是'基例不够、步骤才是关键'的检验)"},
        {"id": "si_shift_identity", "args": {"lhs": [1, -1], "rhs": [0, 1, -1, 1], "n0": 0},
         "expect_err": "POSTCOND_FAIL",
         "expect": "1−x 与 x²−x³+x 不恒等 (前者一次、后者三次) → 拒"},
        {"id": "si_identity_polys", "args": {"lhs": [0, 0, 1], "rhs": [0, 0, 1], "n0": 0,
                                             "simulate": 16},
         "check": lambda r: (r.value["step_is_zero"] is True and r.value["D_degree"] == 0
                             and r.value["sim_failures"] == [],
                             f"D_degree={r.value['D_degree']} step={r.value['step_diff']}"),
         "expect": "lhs=rhs=x² → D≡0 (次数 0 的零多项式), 步骤差分恒 0; n0=0 基例 0=0 ✓"},
        {"id": "si_false_claim_rejected", "args": {"lhs": [0, 1, 1], "rhs": [0, 0, 0], "n0": 0},
         "expect_err": "POSTCOND_FAIL",
         "expect": "x+x² ≡ 0 是假断言: D 非常数 → D(n+1)−D(n)≠0 → 步骤不闭合, 必须拒"},
        {"id": "si_base_fails", "args": {"lhs": [5], "rhs": [3], "n0": 0},
         "expect_err": "POSTCOND_FAIL",
         "expect": "常数 5 与 3: D=2 非零 → 基例 D(0)=2≠0, 归纳在基例即失败"},
        {"id": "si_shifted_start", "args": {"lhs": [0, 1], "rhs": [0, 1], "n0": 7, "simulate": 8},
         "check": lambda r: (r.value["n0"] == 7 and r.value["sim_failures"] == [],
                             f"n0={r.value['n0']}"),
         "expect": "恒等断言从 n0=7 起也成立 (基例在 7 处 =0)"},
        {"id": "si_wrong_step_claim", "args": {"lhs": [0, 0, 1], "rhs": [0, 0, 1], "n0": 0,
                                               "step_claim": [1]},
         "expect_err": "POSTCOND_FAIL",
         "expect": "调用方声明的步骤多项式 [1] ≠ 实际所需的 [0] → 步骤方案与断言不符, 拒"},
    ),
))


# ===========================================================================
# 9303 contradiction
# ===========================================================================

_IN_CONTRA = {
    "kind": "dict",
    "fields": {"goal": {"kind": "str"},
               "term": {"kind": "dict", "extra": True,
                        "fields": {"rule": {"kind": "str"}}, "required": ["rule"]}},
    "required": ["goal", "term"],
}
"""A term is a natural-deduction proof **tree** (rule set: ND_RULES in
svokernel/ndcore.py).

The schema holds only the top-level shape (a rule must be present); the deep
structure is judged recursively by the ndcore kernel, which is exactly the
trust root's job: the kernel judges and never searches, and any rule or
premise mismatch raises POSTCOND_FAIL. The schema cannot express recursive
shapes (this package's controlled subset deliberately excludes recursion),
so structural correctness is guaranteed by the kernel, not the schema. That
is a design choice, not an omission.
"""


def _pre_contra(args) -> "tuple[bool, str]":
    if not args["goal"].strip():
        return False, "空目标命题"
    if not isinstance(args["term"], dict) or not args["term"].get("rule"):
        return False, "证明项顶层缺 rule"
    return True, ""


def _inv_contra(args: dict, ctl: CallCtl) -> dict:
    """Proof by contradiction: an ND proof term starting from ¬P **passes the kernel step by step** and derives ⊥, hence P holds.

    Why the ND kernel rather than §10.1's Hilbert kernel (A1/A2/A3 + MP): the Hilbert system is only the **positive
    implication** fragment with ⊥ as an atom, which can **never be derived** (atoms enter only via hyp or inside the
    compound formulas of axiom instances), so "ending in ⊥" is inexpressible in that kernel. The ND kernel has the three
    rules neg_e / raa / efq that deal with ⊥ directly, which is what makes "proof by contradiction" meaningful. This
    boundary must be recorded explicitly; it is not an implementation preference.
    """
    ctx = "op:strat_contradiction#9303"
    goal = args["goal"]
    hyp = f"~({goal})"
    ok, detail = check_nd(args["term"], "FALSE_", [hyp])
    if not ok:
        raise SvoError(Err.POSTCOND_FAIL, ctx=ctx,
                       detail=f"¬({goal}) 的推导链未过 ND 核: {detail}")
    n_rules = _count_nodes(args["term"])
    cert = Certificate(kind="nd_proof",
                       payload={"term": args["term"], "goal": "FALSE_", "hyps": [hyp]},
                       issued_by="strat_contradiction")
    return {"value": {"derived": "FALSE_", "hypothesis_used": hyp, "n_nodes": n_rules,
                      "kernel_detail": detail},
            "resid": [0.0], "cert": cert,
            "notes": f"反证成立: 假设 ¬({goal}) 经 {n_rules} 个证明节点逐步过 ND 核推出 ⊥; "
                     f"故 P 成立。核报告: {detail}"}


def _count_nodes(term) -> int:
    if not isinstance(term, dict):
        return 0
    n = 1
    for v in term.values():
        if isinstance(v, dict):
            n += _count_nodes(v)
        elif isinstance(v, list):
            n += sum(_count_nodes(x) for x in v)
    return n


def _ver_contra(args: dict, value: dict) -> dict:
    """Independent verification: **ignoring the proof term entirely**, decide independently by truth-table exhaustion whether P is a tautology.

    The two paths are disjoint: invoke takes the **syntactic** path (proof term → kernel), verify takes the **semantic**
    path (semantics → exhaustive truth table). An error on either side surfaces here as a mismatch (proof passes the
    kernel but P is not a tautology, or vice versa). Within the propositional fragment tautologyhood is **decidable
    completely**, so this independent path is decisive.
    """
    ctx = "op:strat_contradiction@verify"
    taut, detail = _tt_tautology(args["goal"], ctx)
    derived_ok = (value["derived"] == "FALSE_") and (int(value["n_nodes"]) == _count_nodes(args["term"]))
    ok = taut and derived_ok
    return {"ok": ok, "resid": [0.0 if taut else 1.0],
            "detail": f"真值表独立判定 P 为重言式: {detail} (重言={taut}); 输出字段一致={derived_ok}"}


REGISTRY.register(OpDef(
    name="strat_contradiction", band=BAND, sub=3,
    stmt="反证法: 若由 ¬P 可推出矛盾 ⊥, 则 P 成立",
    in_schema=_IN_CONTRA,
    out_schema={"kind": "dict",
                "fields": {"derived": {"kind": "str", "enum": ["FALSE_"]},
                           "hypothesis_used": {"kind": "str"}, "n_nodes": {"kind": "int", "min": 1},
                           "kernel_detail": {"kind": "str"}},
                "required": ["derived", "hypothesis_used", "n_nodes", "kernel_detail"]},
    invoke=_inv_contra, verify=_ver_contra, precond=_pre_contra, cert_kind="nd_proof",
    cost=("units", 2.5), errors=("PRECOND_FAIL", "POSTCOND_FAIL", "BOUNDED_VIOLATION", "SCHEMA_MISMATCH"),
    effects="none (纯函数)",
    doc="策略算子 contradiction | in: 目标 P + 由 ¬P 导出 ⊥ 的 **ND 证明项树** | "
        "out: 推出 ⊥ 的确认 (核报告 + 节点数) | 验收=¬P 推导链**逐步过 ND 核**且终得 ⊥ | "
        "证书=nd_proof (payload: term 在 hyps=[¬P] 下导出 FALSE_, 由 ndcore 的独立检查器复验) | "
        "验证器=真值表独立判定 P 为重言式 (完全不看证明项); 两条路径一语法一语义 | 2.5 units | "
        "诚实边界: 必须用 ND 核 —— Hilbert 核 (A1/A2/A3+MP) 是正蕴含片段, 无法导出原子 ⊥, "
        "本算子在 Hilbert 核下不可表达 (已在 docstring 显式记录) | 失败: 证明项未过核 POSTCOND_FAIL",
    selfchecks=(
        {"id": "sc_identity_axiom", "args": {
            "goal": "(a -> a)",
            "term": {"rule": "neg_e", "f": "FALSE_",
                     "maj": {"rule": "assume", "f": "~(a -> a)"},
                     "min": {"rule": "imp_e", "f": "(a -> a)",
                             "maj": {"rule": "imp_e", "f": "((a -> (a -> a)) -> (a -> a))",
                                     "maj": {"rule": "axiom",
                                             "f": "((a -> ((a -> a) -> a)) -> ((a -> (a -> a)) -> (a -> a)))"},
                                     "min": {"rule": "axiom", "f": "(a -> ((a -> a) -> a))"}},
                             "min": {"rule": "axiom", "f": "(a -> (a -> a))"}}}},
         "check": lambda r: (r.value["derived"] == "FALSE_" and r.value["n_nodes"] == 7
                             and r.cert_independent is True,
                             f"derived={r.value['derived']} nodes={r.value['n_nodes']} "
                             f"cert={r.cert_independent}"),
         "expect": "P=(a→a): 假设 ¬(a→a); 用 A1/A2 实例 + 两次 imp_e 导出 (a→a); "
                   "再由 neg_e 得 ⊥ (7 个节点: neg_e + assume + 2×imp_e + 3×axiom)。真值表独立判定 a→a 是重言式 ✓; "
                   "证书 nd_proof 由 ndcore 独立检查器复验"},
        {"id": "sc_non_tautology_rejected", "args": {
            "goal": "a",
            "term": {"rule": "assume", "f": "a"}},
         "expect_err": "POSTCOND_FAIL",
         "expect": "P=a 不是重言式: 该证明项结论是 a 而非 ⊥ → 核拒绝 (不得声称推出 ⊥)"},
        {"id": "sc_bad_rule_rejected", "args": {
            "goal": "(a -> a)",
            "term": {"rule": "magic", "f": "FALSE_"}},
         "expect_err": "POSTCOND_FAIL",
         "expect": "未知规则 magic → 核拒 (只判定不搜索, 表外规则一律拒)"},
        {"id": "sc_wrong_min_rejected", "args": {
            "goal": "(a -> a)",
            "term": {"rule": "neg_e", "f": "FALSE_",
                     "maj": {"rule": "assume", "f": "~(a -> a)"},
                     "min": {"rule": "axiom", "f": "(a -> a)"}}},
         "expect_err": "POSTCOND_FAIL",
         "expect": "(a→a) 不是 A1/A2/A3 的实例 → axiom 节点被核拒 (不能把结论当公理)"},
        {"id": "sc_unresolved_hyp_used", "args": {
            "goal": "(a -> a)",
            "term": {"rule": "neg_e", "f": "FALSE_",
                     "maj": {"rule": "assume", "f": "~(b -> b)"},
                     "min": {"rule": "imp_e", "f": "(b -> b)",
                             "maj": {"rule": "imp_e", "f": "((b -> (b -> b)) -> (b -> b))",
                                     "maj": {"rule": "axiom",
                                             "f": "((b -> ((b -> b) -> b)) -> ((b -> (b -> b)) -> (b -> b)))"},
                                     "min": {"rule": "axiom", "f": "(b -> ((b -> b) -> b))"}},
                             "min": {"rule": "axiom", "f": "(b -> (b -> b))"}}}},
         "expect_err": "POSTCOND_FAIL",
         "expect": "该证明项导出 ⊥ 但依赖未消解假设 ¬(b→b), 而给定假设是 ¬(a→a) → "
                   "假设闭合检查拒绝 (反证的假设必须是**目标**的否定)"},
        {"id": "sc_empty_goal", "args": {"goal": "  ", "term": {"rule": "assume", "f": "a"}},
         "expect_err": "PRECOND_FAIL", "expect": "空目标命题"},
    ),
))


# ===========================================================================
# 9304 case_split
# ===========================================================================

_IN_CASE = {
    "kind": "dict",
    "fields": {"goal": {"kind": "str"},
               "hyps": {"kind": "list", "items": {"kind": "str"}, "maxlen": 12},
               "cases": {"kind": "list", "minlen": 2, "maxlen": 12,
                         "items": {"kind": "dict", "extra": False,
                                   "fields": {"id": {"kind": "str"}, "cond": {"kind": "str"},
                                              "proved": {"kind": "bool"}},
                                   "required": ["id", "cond", "proved"]}},
    },
    "required": ["goal", "cases"],
}


def _pre_case(args) -> "tuple[bool, str]":
    ids = [c["id"] for c in args["cases"]]
    if len(set(ids)) != len(ids):
        return False, "情形 id 重复"
    if any(not c["cond"].strip() for c in args["cases"]):
        return False, "空情形条件"
    return True, ""


def _inv_case(args: dict, ctl: CallCtl) -> dict:
    """Case split: under the hypothesis set H, check (a) every branch H∧cond_i ⊨ goal; (b) **coverage completeness** H ⊨ ∨cond_i
    (i.e. every assignment satisfying H is covered by at least one branch).

    Including H is not a convenience but a **necessary condition**: without hypotheses, "every branch implies the goal"
    and "coverage is complete" holding together **is equivalent to the goal itself being a tautology** (completeness
    forces ∨cond_i ≡ ⊤, and each branch ⊆ goal then forces goal ≡ ⊤). Case splitting would then degenerate into "proving
    a tautology" and be useless. A genuinely useful case split always proceeds "under a hypothesis H" — H supplies the
    information on which the case analysis rests.
    """
    ctx = "op:strat_case_split#9304"
    cases = args["cases"]
    goal = args["goal"]
    hyps = list(args.get("hyps", []))
    per_case_bad = []
    atoms = _atoms_of(hyps + [c["cond"] for c in cases] + [goal])
    if len(atoms) > MAX_ATOMS:
        raise SvoError(Err.BOUNDED_VIOLATION, ctx=ctx,
                       detail=f"原子数 {len(atoms)} > {MAX_ATOMS} — 覆盖性穷举超界")
    for c in cases:
        bad, _tot, detail = _entail_counterexamples(hyps + [c["cond"]], goal, ctx)
        if bad:
            per_case_bad.append({"id": c["id"], "counterexamples": bad, "detail": detail})
    if per_case_bad:
        raise SvoError(Err.POSTCOND_FAIL, ctx=ctx,
                       detail=f"情形不蕴含目标: {per_case_bad[0]} — 分情形论证无效")
    hf = [parse_formula(h) for h in hyps]
    cfs = [parse_formula(c["cond"]) for c in cases]
    total = covered = 0
    uncovered_sample = ""
    for bits in product((False, True), repeat=len(atoms)):
        env = dict(zip(atoms, bits))
        if not all(ev_shortcircuit(h, env) for h in hf):
            continue
        total += 1
        if any(ev_shortcircuit(cf, env) for cf in cfs):
            covered += 1
        elif not uncovered_sample:
            uncovered_sample = str({k: v for k, v in env.items() if v})
    if covered != total:
        raise SvoError(Err.POSTCOND_FAIL, ctx=ctx,
                       detail=f"覆盖不完备: 假设下 {covered}/{total} 个指派被覆盖, "
                              f"未覆盖例: 真值原子 {uncovered_sample} — 必须补情形")
    unproved = [c["id"] for c in cases if not c["proved"]]
    return {"value": {"covered": covered, "total": total, "n_cases": len(cases),
                      "n_hyps": len(hyps), "per_case_ok": True, "unproved": unproved,
                      "goal_established": not unproved},
            "resid": [0.0],
            "notes": f"分情形 {len(cases)} 支 (假设 {len(hyps)} 条): 每支在假设下都蕴含目标; "
                     f"覆盖 {covered}/{total} 个假设成立的指派 (完备); 未证明支 {unproved} → "
                     f"目标{'已确立' if not unproved else '尚未确立'}"}


def _ver_case(args: dict, value: dict) -> dict:
    """Independent verification: (a) recheck coverage by **converse counting** — the number of assignments with all hypotheses true
    and every case false must be 0 (complementary to invoke's "at least one branch true" count, with the formula written out
    independently);
    (b) independently recheck each branch's H∧cond_i ∧ ¬goal unsatisfiability;
    (c) the unproved-branch list agrees with the proved flags."""
    ctx = "op:strat_case_split@verify"
    cases = args["cases"]
    goal = args["goal"]
    hyps = list(args.get("hyps", []))
    atoms = _atoms_of(hyps + [c["cond"] for c in cases] + [goal])
    if len(atoms) > MAX_ATOMS:
        raise SvoError(Err.BOUNDED_VIOLATION, ctx=ctx, detail="原子数超界")
    hf = [parse_formula(h) for h in hyps]
    cfs = [parse_formula(c["cond"]) for c in cases]
    all_false = 0
    for bits in product((False, True), repeat=len(atoms)):
        env = dict(zip(atoms, bits))
        if not all(ev_shortcircuit(h, env) for h in hf):
            continue
        if not any(ev_shortcircuit(cf, env) for cf in cfs):
            all_false += 1
    bad_case = 0
    for c in cases:
        bad, _, _ = _entail_counterexamples(hyps + [c["cond"]], goal, ctx + f"#{c['id']}")
        bad_case += bad
    unproved = [c["id"] for c in cases if not c["proved"]]
    status_ok = (sorted(value["unproved"]) == sorted(unproved))         and (value["goal_established"] == (len(unproved) == 0))
    ok = (all_false == 0) and (bad_case == 0) and status_ok         and value["covered"] == value["total"] and int(value["n_hyps"]) == len(hyps)
    return {"ok": ok, "resid": [float(all_false + bad_case)],
            "detail": f"反向计数: 假设成立且全情形皆假的指派 {all_false} 个 (需为 0); "
                      f"逐支 H∧cond∧¬目标 的反例累计 {bad_case}; 未证明支一致={status_ok}"}


REGISTRY.register(OpDef(
    name="strat_case_split", band=BAND, sub=4,
    stmt="分情形论证: 若情形条件之并是重言式 (覆盖完备) 且每支都蕴含目标, 则目标成立",
    in_schema=_IN_CASE,
    out_schema={"kind": "dict",
                "fields": {"covered": {"kind": "int", "min": 1}, "total": {"kind": "int", "min": 1},
                           "n_cases": {"kind": "int", "min": 2}, "per_case_ok": {"kind": "bool"},
                           "n_hyps": {"kind": "int", "min": 0},
                           "unproved": {"kind": "list", "items": {"kind": "str"}},
                           "goal_established": {"kind": "bool"}},
                "required": ["covered", "total", "n_cases", "per_case_ok", "n_hyps", "unproved",
                             "goal_established"]},
    invoke=_inv_case, verify=_ver_case, precond=_pre_case,
    cost=("units", 2.0), errors=("PRECOND_FAIL", "POSTCOND_FAIL", "BOUNDED_VIOLATION", "SCHEMA_MISMATCH"),
    effects="none (纯函数)",
    doc="策略算子 case_split | in: 目标 + 情形表 {id,cond,proved} | out: 覆盖计数 + 每支蕴含性 + 待证支 | "
        "验收=覆盖完备性 (**情形并集=全集**, 穷举全部指派) + 每支蕴含目标 | "
        "验证器=反向计数(全假指派数=0) + 逐支 cond∧¬目标 不可满足 | 2 units | "
        "诚实边界: 命题片段内完备判定, 原子≤14 | 失败: 覆盖不全或某支不蕴含目标 POSTCOND_FAIL",
    selfchecks=(
        {"id": "scs_real_split", "args": {"goal": "c",
                                          "hyps": ["(a -> c) & (~a -> c)"],
                                          "cases": [{"id": "c1", "cond": "a", "proved": True},
                                                    {"id": "c2", "cond": "~a", "proved": True}]},
         "check": lambda r: (r.value["goal_established"] is True and r.value["per_case_ok"] is True
                             and r.value["covered"] == r.value["total"] and r.value["n_hyps"] == 1,
                             f"covered={r.value['covered']}/{r.value['total']} hyps={r.value['n_hyps']}"),
         "expect": "**真正的分情形**: 假设 (a→c)&(¬a→c); 情形 a 与 ¬a。"
                   "覆盖: 假设成立的 4 个指派(2 原子)全被覆盖; 每支在假设下都推出 c → 目标 c 成立"},
        {"id": "scs_tautology_goal", "args": {"goal": "a | ~a",
                                              "cases": [{"id": "c1", "cond": "a", "proved": True},
                                                        {"id": "c2", "cond": "~a", "proved": True}]},
         "check": lambda r: (r.value["goal_established"] is True and r.value["total"] == 2,
                             f"covered={r.value['covered']}/{r.value['total']}"),
         "expect": "无假设 + 目标为重言式 a|¬a: 两支各蕴含它且覆盖完备 (total=2 个指派) → 成立。"
                   "这正是文档里说明的等价性: 无假设时该算子的成功必然对应目标为重言式"},
        {"id": "scs_incomplete_coverage", "args": {"goal": "c",
                                                   "hyps": ["(a -> c)"],
                                                   "cases": [{"id": "c1", "cond": "a & b", "proved": True},
                                                             {"id": "c2", "cond": "a & ~b", "proved": True}]},
         "expect_err": "POSTCOND_FAIL",
         "expect": "假设 a→c 在 ¬a 时仍可能成立, 而两支都要求 a → 覆盖不完备, 必须拒"},
        {"id": "scs_case_not_entailing", "args": {"goal": "d",
                                                  "hyps": ["a"],
                                                  "cases": [{"id": "c1", "cond": "a", "proved": True},
                                                            {"id": "c2", "cond": "b", "proved": True}]},
         "expect_err": "POSTCOND_FAIL",
         "expect": "情形 a 不蕴含 d → 逐支蕴含检查失败, 必须拒"},
        {"id": "scs_dup_ids", "args": {"goal": "a | ~a",
                                       "cases": [{"id": "x", "cond": "a", "proved": True},
                                                 {"id": "x", "cond": "~a", "proved": True}]},
         "expect_err": "PRECOND_FAIL", "expect": "情形 id 重复"},
        {"id": "scs_single_case", "args": {"goal": "a | ~a",
                                           "cases": [{"id": "x", "cond": "a", "proved": True}]},
         "expect_err": "SCHEMA_MISMATCH", "expect": "情形数 < 2 (minlen=2) — 分情形需要至少两支"},
    ),
))


# ===========================================================================
# 9305 invariant_find
# ===========================================================================

_IN_INV = {
    "kind": "dict",
    "fields": {
        "A": {"kind": "list", "len": 2, "items": {"kind": "list", "len": 2, "items": {"kind": "Fraction"}}},
        "bvec": {"kind": "list", "len": 2, "items": {"kind": "Fraction"}},
        "x0": {"kind": "list", "len": 2, "items": {"kind": "Fraction"}},
        "invariant": {"kind": "dict", "extra": False,
                      "fields": {"l": {"kind": "list", "len": 2, "items": {"kind": "Fraction"}},
                                 "c": {"kind": "Fraction"},
                                 "rel": {"kind": "str", "enum": ["eq", "le", "ge"]}},
                      "required": ["l", "c", "rel"]},
        "simulate": {"kind": "int", "min": 1, "max": 2000},
    },
    "required": ["A", "bvec", "x0", "invariant"],
}


def _affine_step(A, bv, x):
    return [A[i][0] * x[0] + A[i][1] * x[1] + bv[i] for i in range(2)]


def _holds(l, c, rel, x) -> bool:
    v = l[0] * x[0] + l[1] * x[1]
    if rel == "eq":
        return v == c
    if rel == "le":
        return v <= c
    return v >= c


def _inv_inv(args: dict, ctl: CallCtl) -> dict:
    """Invariant: initiation I(x0) and consecution I(x) ⇒ I(F(x)).

    **Closed-form proof of consecution** (exact, not sampled). Write ℓ(F(x)) = (ℓ·A)x + ℓ·b.
    If ℓ·A = μ·ℓ (the parallel condition) holds, then ℓ(F(x)) = μ·ℓ(x) + ℓ·b, so for the region {ℓ(x) ≤ c}:
        sup_{ℓ(x) ≤ c} ℓ(F(x)) = μc + ℓ·b        (for μ ≥ 0; for μ<0 the supremum is +∞ and the invariant cannot hold)
    Hence consecution with rel=le ⟺ μ ≥ 0 and μc + ℓ·b ≤ c; rel=ge is symmetric (μ ≥ 0 and μc + ℓ·b ≥ c);
    rel=eq ⟺ μc + ℓ·b = c.
    (The old version wrote the condition as "g(x)=ℓ(F(x))−c is proportional to ℓ(x)−c", which is **wrong** — it is
     stronger and would misjudge correct invariants like y←y/2 preserving y≤1 as "closed form inapplicable"; rewritten
     per the derivation above.) When the parallel condition fails it falls back to a bounded rational grid and
     labelled in the scope field.
    """
    ctx = "op:strat_invariant_find#9305"
    A = [[f(v) for v in row] for row in args["A"]]
    bv = [f(v) for v in args["bvec"]]
    x0 = [f(v) for v in args["x0"]]
    inv = args["invariant"]
    l = [f(v) for v in inv["l"]]
    c = f(inv["c"])
    rel = inv["rel"]
    if l[0] == 0 and l[1] == 0:
        raise SvoError(Err.PRECOND_FAIL, ctx=ctx, detail="零线性型 ℓ ≡ 0 — 不构成有意义的不变量")
    lx0 = l[0] * x0[0] + l[1] * x0[1]
    if not _holds(l, c, rel, x0):
        raise SvoError(Err.POSTCOND_FAIL, ctx=ctx,
                       detail=f"起始条件不成立: ℓ(x0)={fs(lx0)} 不满足 {rel} {fs(c)}")
    la = [l[0] * A[0][0] + l[1] * A[1][0], l[0] * A[0][1] + l[1] * A[1][1]]
    lb = l[0] * bv[0] + l[1] * bv[1]
    # parallel condition: ℓ·A = μℓ
    mu = None
    if l[0] != 0:
        cand = la[0] / l[0]
        if la[1] == cand * l[1]:
            mu = cand
    elif l[1] != 0:
        cand = la[1] / l[1]
        if la[0] == 0:
            mu = cand
    scope = "closed_form" if mu is not None else "bounded_grid"
    sim = int(args.get("simulate", 64))
    x = list(x0)
    sim_fail = 0
    for _ in range(sim):
        x = _affine_step(A, bv, x)
        ctl.check_time(ctx + "@sim")
        if not _holds(l, c, rel, x):
            sim_fail += 1
    reason = ""
    if scope == "closed_form":
        lim = mu * c + lb
        if rel == "le":
            cons = (mu >= 0) and (lim <= c)
            reason = (f"μ={fs(mu)} ≥ 0 且 μc+ℓ·b={fs(lim)} ≤ c={fs(c)}" if cons
                      else (f"μ={fs(mu)} < 0 → 上确界无界" if mu < 0
                            else f"μ={fs(mu)} ≥ 0 但 μc+ℓ·b={fs(lim)} > c={fs(c)}"))
        elif rel == "ge":
            cons = (mu >= 0) and (lim >= c)
            reason = (f"μ={fs(mu)} ≥ 0 且 μc+ℓ·b={fs(lim)} ≥ c={fs(c)}" if cons
                      else f"μ={fs(mu)} 或界不满足 (μc+ℓ·b={fs(lim)} vs c={fs(c)})")
        else:
            cons = (lim == c)
            reason = f"μc+ℓ·b={fs(lim)} 需等于 c={fs(c)}"
        if not cons:
            raise SvoError(Err.POSTCOND_FAIL, ctx=ctx,
                           detail=f"连续性闭式判定失败: {reason} — 该线性型在迭代下不保持")
    else:
        K = 12
        grid = [Fraction(k, 2) for k in range(-K, K + 1)]
        checked = viol = 0
        for gx in grid:
            for gy in grid:
                pt = [gx, gy]
                if not _holds(l, c, rel, pt):
                    continue
                checked += 1
                if not _holds(l, c, rel, _affine_step(A, bv, pt)):
                    viol += 1
        if viol:
            raise SvoError(Err.POSTCOND_FAIL, ctx=ctx,
                           detail=f"连续性在有界网格上被违反 {viol}/{checked} 点 — 该线性型不是不变量")
        reason = f"ℓ·A 与 ℓ 不平行 → 有界有理网格核验 {checked} 点全部保持"
    return {"value": {"scope": scope, "init_holds": True, "mu": fs(mu) if mu is not None else "",
                      "la": [fs(v) for v in la], "lb": fs(lb),
                      "simulated": sim, "sim_failures": sim_fail,
                      "invariant": {"l": [fs(v) for v in l], "c": fs(c), "rel": rel}},
            "resid": [float(sim_fail)],
            "notes": f"不变量成立: 起始 ℓ(x0)={fs(lx0)} 满足 {rel} {fs(c)}; 连续性 scope={scope} "
                     f"({reason}); 精确模拟 {sim} 步失败 {sim_fail} 次"}


def _ver_inv(args: dict, value: dict) -> dict:
    """Independent verification: (a) **step-by-step exact simulation** rerun independently (catching "the simulation disagrees with the claim");
    (b) **coefficient-level recheck** of the parallel relation ℓ·A = μℓ (not copied from invoke's conclusion);
    (c) **closed-form condition re-derivation**: from μ, c, ℓ·b decide again the relation between μc+ℓ·b and c; it must agree with scope/the conclusion."""
    A = [[f(v) for v in row] for row in args["A"]]
    bv = [f(v) for v in args["bvec"]]
    x0 = [f(v) for v in args["x0"]]
    inv = args["invariant"]
    l = [f(v) for v in inv["l"]]
    c = f(inv["c"])
    rel = inv["rel"]
    sim = int(args.get("simulate", 64))
    x = list(x0)
    fail = 0
    for _ in range(sim):
        x = _affine_step(A, bv, x)
        if not _holds(l, c, rel, x):
            fail += 1
    la = [l[0] * A[0][0] + l[1] * A[1][0], l[0] * A[0][1] + l[1] * A[1][1]]
    la_ok = (la == [f(v) for v in value["la"]])
    lb = l[0] * bv[0] + l[1] * bv[1]
    lb_ok = (lb == f(value["lb"]))
    # independently re-derive the parallel relation and the closed-form condition
    mu_ind = None
    if l[0] != 0 and la[1] == (la[0] / l[0]) * l[1]:
        mu_ind = la[0] / l[0]
    elif l[0] == 0 and l[1] != 0 and la[0] == 0:
        mu_ind = la[1] / l[1]
    scope_ok = (value["scope"] == "closed_form") == (mu_ind is not None)
    mu_claim = f(value["mu"]) if value["mu"] else None
    mu_ok = (mu_claim == mu_ind)
    cons_ok = True
    if mu_ind is not None:
        lim = mu_ind * c + lb
        if rel == "le":
            cons_ok = (mu_ind >= 0) and (lim <= c)
        elif rel == "ge":
            cons_ok = (mu_ind >= 0) and (lim >= c)
        else:
            cons_ok = (lim == c)
    ok = (fail == 0) and (int(value["sim_failures"]) == 0) and la_ok and lb_ok \
        and scope_ok and mu_ok and cons_ok
    return {"ok": ok, "resid": [float(fail)],
            "detail": f"独立重跑 {sim} 步模拟失败 {fail} 次; ℓ·A 系数复核={la_ok}, ℓ·b={lb_ok}; "
                      f"scope 与平行性一致={scope_ok}; μ 复核={mu_ok}; "
                      f"闭式连续条件重推={cons_ok} (μ={value['mu'] or '—'}, c={fs(c)}, rel={rel})"}


REGISTRY.register(OpDef(
    name="strat_invariant_find", band=BAND, sub=5,
    stmt="循环不变量: 若 I(x0) 成立且 I(x) ⇒ I(F(x)) (F 为仿射迭代), 则 I 在任意步成立",
    in_schema=_IN_INV,
    out_schema={"kind": "dict",
                "fields": {"scope": {"kind": "str", "enum": ["closed_form", "bounded_grid"]},
                           "init_holds": {"kind": "bool"}, "mu": {"kind": "str"},
                           "la": {"kind": "list", "len": 2, "items": {"kind": "str"}},
                           "lb": {"kind": "str"}, "simulated": {"kind": "int", "min": 1},
                           "sim_failures": {"kind": "int", "min": 0},
                           "invariant": {"kind": "dict", "extra": True, "fields": {}}},
                "required": ["scope", "init_holds", "mu", "la", "lb", "simulated",
                             "sim_failures", "invariant"]},
    invoke=_inv_inv, verify=_ver_inv, precond=None,
    cost=("units", 2.0), errors=("PRECOND_FAIL", "POSTCOND_FAIL", "TIMEOUT", "SCHEMA_MISMATCH"),
    effects="none (纯函数, 精确有理算术)",
    doc="策略算子 invariant_find | in: 2 维仿射迭代 (A,b,x0) + 线性不变量 (ℓ,c,rel∈{eq,le,ge}) | "
        "out: 起始/连续性结论 + scope | 验收=不变量**任意步**成立: 闭式路线 (ℓ∘A=μℓ 且 μ≥0 for "
        "le/ge) 给出完整证明; 闭式不适用时有界有理网格核验并在 scope 标注 | "
        "验证器=独立重跑模拟 + ℓ∘A 系数级复核 + μ 符号条件 | 2 units | "
        "诚实边界: bounded_grid 只是**有界**证据, 不声称任意步 (scope 字段是审计入口) | "
        "失败: 起始不成立或连续被违反 POSTCOND_FAIL",
    selfchecks=(
        {"id": "sinv_true_invariant", "args": {"A": [[1, 0], [0, "1/2"]], "bvec": [0, 0],
                                               "x0": [0, 1], "invariant": {"l": [0, 1], "c": 1,
                                                                           "rel": "le"},
                                               "simulate": 20},
         "check": lambda r: (r.value["scope"] == "closed_form" and r.value["mu"] == "1/2"
                             and r.value["sim_failures"] == 0 and r.value["init_holds"] is True,
                             f"scope={r.value['scope']} mu={r.value['mu']} sim_fail={r.value['sim_failures']}"),
         "expect": "y 每次减半: ℓ(y)=y, ℓ·A=(1/2)ℓ → μ=1/2 ≥ 0 且 μc+ℓ·b=1/2 ≤ c=1 ✓ "
                   "→ y ≤ 1 是不变量 (闭式证明, 20 步模拟同样通过)"},
        {"id": "sinv_sum_preserved", "args": {"A": [[1, 1], [0, 1]], "bvec": [0, 0],
                                              "x0": [0, 0], "invariant": {"l": [0, 1], "c": 0,
                                                                          "rel": "eq"},
                                              "simulate": 30},
         "check": lambda r: (r.value["scope"] == "closed_form" and r.value["mu"] == "1"
                             and r.value["sim_failures"] == 0,
                             f"scope={r.value['scope']} mu={r.value['mu']}"),
         "expect": "上三角 (x+y, y): y 分量不变 → 不变量 y=0 (eq), μ=1, μc+ℓ·b=0=c ✓ 任意步成立"},
        {"id": "sinv_ge_direction", "args": {"A": [[1, 0], [0, 1]], "bvec": [0, 1],
                                             "x0": [0, 5], "invariant": {"l": [0, 1], "c": 1,
                                                                         "rel": "ge"},
                                             "simulate": 10},
         "check": lambda r: (r.value["scope"] == "closed_form" and r.value["mu"] == "1"
                             and r.value["lb"] == "1",
                             f"scope={r.value['scope']} mu={r.value['mu']} lb={r.value['lb']}"),
         "expect": "y ← y+1 (下界方向): ℓ·A=ℓ → μ=1 ≥ 0 且 μc+ℓ·b=1+1=2 ≥ c=1 ✓ → y ≥ 1 保持"},
        {"id": "sinv_init_fails", "args": {"A": [[1, 0], [0, "1/2"]], "bvec": [0, 0],
                                           "x0": [0, 1], "invariant": {"l": [0, 1], "c": "1/2",
                                                                       "rel": "le"}},
         "expect_err": "POSTCOND_FAIL",
         "expect": "x0 的 y=1 > 1/2 → 起始条件即不成立, 必须拒 (不变量必须从初值起成立)"},
        {"id": "sinv_not_invariant", "args": {"A": [[2, 0], [0, 1]], "bvec": [0, 0], "x0": [1, 0],
                                              "invariant": {"l": [1, 0], "c": 1, "rel": "le"},
                                              "simulate": 5},
         "expect_err": "POSTCOND_FAIL",
         "expect": "x 以 2 倍扩张, 断言 x ≤ 1: μ=2 ≥ 0 但 μc+ℓ·b=2 > c=1 → 闭式判定**不保持**, "
                   "拒 (这正是闭式路线优于抽样的地方: 不靠模拟碰巧通过)"},
        {"id": "sinv_zero_form", "args": {"A": [[1, 0], [0, 1]], "bvec": [0, 0], "x0": [0, 0],
                                          "invariant": {"l": [0, 0], "c": 0, "rel": "eq"}},
         "expect_err": "PRECOND_FAIL", "expect": "零线性型无意义"},
    ),
))


# ===========================================================================
# 9306 extremal
# ===========================================================================

_IN_EXT = {
    "kind": "dict",
    "fields": {"points": {"kind": "list", "minlen": 1, "maxlen": 64,
                          "items": {"kind": "dict", "extra": False,
                                    "fields": {"key": {"kind": "Fraction"}, "val": {"kind": "Fraction"}},
                                    "required": ["key", "val"]}},
               "direction": {"kind": "str", "enum": ["max", "min"]},
               "strict": {"kind": "bool"}},
    "required": ["points", "direction"],
}


def _pre_ext(args) -> "tuple[bool, str]":
    keys = [f(p["key"]) for p in args["points"]]
    if len(set(keys)) != len(keys):
        return False, "键重复 — 本项目要求键严格递增的唯一序 (有序结构的基本要求)"
    return True, ""


def _inv_ext(args: dict, ctl: CallCtl) -> dict:
    """Extremum: find argmax/argmin on a finite ordered structure, with exhaustive existence of the extremum + a two-sided neighbourhood check."""
    pts = sorted(((f(p["key"]), f(p["val"])) for p in args["points"]), key=lambda kv: kv[0])
    direction = args["direction"]
    strict = bool(args.get("strict", False))
    best = (max if direction == "max" else min)(v for _k, v in pts)
    winners = [k for k, v in pts if v == best]
    idx = [i for i, (_k, v) in enumerate(pts) if v == best][0]
    kbest = pts[idx][0]
    # exhaustive existence of the extremum: no better value in the whole table
    viol = sum(1 for _k, v in pts if (v > best if direction == "max" else v < best))
    # two-sided neighbourhood: strict mode requires neighbours to be strictly worse than the extremum
    left = pts[idx - 1] if idx > 0 else None
    right = pts[idx + 1] if idx + 1 < len(pts) else None
    def nb_ok(nb) -> bool:
        if nb is None:
            return True
        return (nb[1] < best) if (direction == "max" and strict) else \
               ((nb[1] <= best) if direction == "max" else
                ((nb[1] > best) if strict else (nb[1] >= best)))
    ok_nb = nb_ok(left) and nb_ok(right)
    if viol or (strict and not ok_nb):
        raise SvoError(Err.POSTCOND_FAIL, ctx="op:strat_extremal#9306",
                       detail=f"极值检查失败: 更优值出现 {viol} 次; 双侧邻域检查={ok_nb} "
                              f"(严格={strict}); 该点不是有效极值")
    return {"value": {"extremum": fs(best), "arg_keys": [fs(k) for k in winners],
                      "direction": direction, "n_points": len(pts),
                      "left_val": fs(left[1]) if left else "", "right_val": fs(right[1]) if right else "",
                      "strict": strict},
            "resid": [0.0],
            "notes": f"极值 {direction}={fs(best)} 在键 {[fs(k) for k in winners]} 处取到; "
                     f"全表 {len(pts)} 点无更优值; 双侧邻域 (左 {fs(left[1]) if left else '—'}, "
                     f"右 {fs(right[1]) if right else '—'}) 检查通过 (严格={strict})"}


def _ver_ext(args: dict, value: dict) -> dict:
    """Independent verification: (a) confirm the extremum independently via **the first/last element after sorting the whole table**
    (a different path from invoke's built-in max/min reduction); (b) key-order strictness; (c) independently re-derive the
    two-sided neighbourhood condition in strict mode."""
    pts = sorted(((f(p["key"]), f(p["val"])) for p in args["points"]), key=lambda kv: kv[0])
    direction = args["direction"]
    strict = bool(args.get("strict", False))
    sorted_vals = sorted(v for _k, v in pts)
    expect = sorted_vals[-1] if direction == "max" else sorted_vals[0]
    ext_ok = (f(value["extremum"]) == expect)
    keys_ok = all(pts[i][0] < pts[i + 1][0] for i in range(len(pts) - 1))
    win_ok = sorted(value["arg_keys"]) == sorted(fs(k) for k, v in pts if v == expect)
    nb_ok = True
    if strict:
        idx = next(i for i, (_k, v) in enumerate(pts) if v == expect)
        for j in (idx - 1, idx + 1):
            if 0 <= j < len(pts):
                if direction == "max" and not (pts[j][1] < expect):
                    nb_ok = False
                if direction == "min" and not (pts[j][1] > expect):
                    nb_ok = False
    ok = ext_ok and keys_ok and win_ok and nb_ok
    return {"ok": ok, "resid": [0.0 if ext_ok else 1.0],
            "detail": f"排序首末独立确认极值={fs(expect)} 与输出一致={ext_ok}; 键序严格={keys_ok}; "
                      f"取极值键集合一致={win_ok}; 严格邻域条件={nb_ok}"}


REGISTRY.register(OpDef(
    name="strat_extremal", band=BAND, sub=6,
    stmt="极值原理: 非空有限全序结构上的目标函数必取到极值; 内点极值处两侧邻域不优于它",
    in_schema=_IN_EXT,
    out_schema={"kind": "dict",
                "fields": {"extremum": {"kind": "str"}, "arg_keys": {"kind": "list",
                                                                     "items": {"kind": "str"}, "minlen": 1},
                           "direction": {"kind": "str", "enum": ["max", "min"]},
                           "n_points": {"kind": "int", "min": 1}, "left_val": {"kind": "str"},
                           "right_val": {"kind": "str"}, "strict": {"kind": "bool"}},
                "required": ["extremum", "arg_keys", "direction", "n_points", "left_val",
                             "right_val", "strict"]},
    invoke=_inv_ext, verify=_ver_ext, precond=_pre_ext,
    cost=("units", 0.6), errors=("PRECOND_FAIL", "POSTCOND_FAIL", "SCHEMA_MISMATCH"),
    effects="none (纯函数, 精确有理算术)",
    doc="策略算子 extremal | in: 有限有序点集 (键唯一) + 方向 + 是否严格 | "
        "out: 极值 + 取极值的键 + 双侧邻域值 | 验收=极值存在 (全表穷举无更优值) + "
        "两侧邻域检查 (严格模式要求邻居严格劣) | 验证器=排序首末独立确认 + 键序严格性 + 邻域重推 | "
        "0.6 unit | 诚实边界: 键必须唯一 (重复键抛 PRECOND_FAIL, 不静默取其一) | "
        "失败: 极值检查或邻域检查不通过 POSTCOND_FAIL",
    selfchecks=(
        {"id": "se_max_interior", "args": {"points": [{"key": 0, "val": 1}, {"key": 1, "val": 5},
                                                      {"key": 2, "val": 3}],
                                           "direction": "max"},
         "check": lambda r: (r.value["extremum"] == "5" and r.value["arg_keys"] == ["1"]
                             and r.value["left_val"] == "1" and r.value["right_val"] == "3",
                             f"extremum={r.value['extremum']} keys={r.value['arg_keys']}"),
         "expect": "内点极大: 键 1 处 val=5, 两侧 1 与 3 均不优于 5 → 通过"},
        {"id": "se_min_endpoint", "args": {"points": [{"key": 0, "val": -2}, {"key": 1, "val": 4}],
                                           "direction": "min"},
         "check": lambda r: (r.value["extremum"] == "-2" and r.value["left_val"] == "",
                             f"extremum={r.value['extremum']} left='{r.value['left_val']}'"),
         "expect": "端点极小: 左侧无邻居 (left_val 为空串表示端点), 右侧 4 不更小"},
        {"id": "se_tie_two_winners", "args": {"points": [{"key": 0, "val": 7}, {"key": 1, "val": 7}],
                                              "direction": "max"},
         "check": lambda r: (r.value["arg_keys"] == ["0", "1"], f"arg_keys={r.value['arg_keys']}"),
         "expect": "并列极值: 两个键都上报 (不静默只取第一个)"},
        {"id": "se_strict_neighbor_violation", "args": {"points": [{"key": 0, "val": 5},
                                                                   {"key": 1, "val": 5}],
                                                        "direction": "max", "strict": True},
         "expect_err": "POSTCOND_FAIL",
         "expect": "严格模式: 邻居与极值相等 (非严格劣) → 邻域检查失败, 拒"},
        {"id": "se_duplicate_key", "args": {"points": [{"key": 1, "val": 1}, {"key": 1, "val": 2}],
                                            "direction": "max"},
         "expect_err": "PRECOND_FAIL", "expect": "键重复 → 不是有序结构, 拒"},
        {"id": "se_bad_direction", "args": {"points": [{"key": 0, "val": 1}], "direction": "up"},
         "expect_err": "SCHEMA_MISMATCH", "expect": "方向枚举外取值"},
    ),
))


# ===========================================================================
# 9307 symmetry
# ===========================================================================

_IN_SYM = {
    "kind": "dict",
    "fields": {"values": {"kind": "list", "items": {"kind": "int"}, "minlen": 2, "maxlen": 12},
               "generators": {"kind": "list", "minlen": 1, "maxlen": 4,
                              "items": {"kind": "list", "items": {"kind": "int"}}},
               "orbit_of": {"kind": "int", "min": 0, "max": 11},
               "max_group": {"kind": "int", "min": 2, "max": 5000}},
    "required": ["values", "generators"],
}

_GROUP_CAP = 2000
"""Generated-subgroup enumeration cap (bound on the order). Anything above
raises BOUNDED_VIOLATION; no silent truncation."""


def _perm_check(p, n: int) -> bool:
    return (len(p) == n) and (sorted(p) == list(range(n)))


def _compose(p, q):
    """(p∘q)(i) = p[q[i]]."""
    return [p[q[i]] for i in range(len(p))]


def _inv_sym(args: dict, ctl: CallCtl) -> dict:
    """Symmetry: enumerate the **subgroup** generated by the generators (its closure), check that the structure is invariant under it, and give the orbit partition."""
    ctx = "op:strat_symmetry#9307"
    vals = [int(v) for v in args["values"]]
    n = len(vals)
    gens = [[int(x) for x in g] for g in args["generators"]]
    for i, g in enumerate(gens):
        if not _perm_check(g, n):
            raise SvoError(Err.PRECOND_FAIL, ctx=ctx,
                           detail=f"生成元 {i}={g} 不是 {n} 元置换 (须恰为 0..{n - 1} 的排列)")
    cap = min(int(args.get("max_group", _GROUP_CAP)), _GROUP_CAP)
    ident = list(range(n))
    group = {tuple(ident)}
    frontier = [ident]
    while frontier:
        ctl.check_time(ctx + "@closure")
        nxt: list = []
        for p in frontier:
            for g in gens:
                q = _compose(g, p)
                t = tuple(q)
                if t not in group:
                    group.add(t)
                    nxt.append(q)
                    if len(group) > cap:
                        raise SvoError(Err.BOUNDED_VIOLATION, ctx=ctx,
                                       detail=f"生成子群阶 > {cap} — 枚举超界, 拒绝 (不静默截断)")
        frontier = nxt
    # invariance: check **every element** of the generated subgroup one by one (not only the generators)
    bad: list = []
    for p in group:
        for i in range(n):
            if vals[p[i]] != vals[i]:
                bad.append((list(p), i))
                break
        if bad:
            break
    # orbit partition (under the generated subgroup)
    seen: set = set()
    orbits: list = []
    for i in range(n):
        if i in seen:
            continue
        orb = sorted({p[i] for p in group})
        seen.update(orb)
        orbits.append(orb)
    if bad:
        raise SvoError(Err.POSTCOND_FAIL, ctx=ctx,
                       detail=f"结构在该群作用下**不变性失败**: 置换 {bad[0][0]} 把下标 {bad[0][1]} "
                              f"映到值不同的位置 — 该结构不具此对称性")
    orbit_of = int(args.get("orbit_of", 0))
    if orbit_of >= n:
        raise SvoError(Err.PRECOND_FAIL, ctx=ctx, detail=f"orbit_of={orbit_of} 越界 (n={n})")
    my_orb = next(o for o in orbits if orbit_of in o)
    return {"value": {"group_order": len(group), "invariant": True, "n_orbits": len(orbits),
                      "orbits": orbits, "orbit_of": my_orb,
                      "n_distinct_values": len(set(vals))},
            "resid": [0.0],
            "notes": f"对称性成立: 生成子群阶 {len(group)} (由 {len(gens)} 个生成元张成), "
                     f"**全部**群元素下结构不变; 轨道 {len(orbits)} 条: {orbits}; "
                     f"元素 {orbit_of} 的轨道 {my_orb}"}


def _ver_sym(args: dict, value: dict) -> dict:
    """Independent verification: (a) **equivalence of orbit partition and invariance** independently rechecked: the structure is
    invariant under the group ⟺ its value is constant on every orbit ⟺ values agree within an orbit (a different path from
    invoke's "scan every permutation");
    (b) the orbits must form a partition of [0,n) (pairwise disjoint with full union);
    (c) group-order lower-bound check: the product of the sizes of the independent-generator orbits is at least the largest orbit length."""
    vals = [int(v) for v in args["values"]]
    n = len(vals)
    orbits = [list(map(int, o)) for o in value["orbits"]]
    flat = [i for o in orbits for i in o]
    partition_ok = (sorted(flat) == list(range(n)))
    orbit_const = all(len({vals[i] for i in o}) == 1 for o in orbits)
    # orbit sizes sum to n, and the group order is at least the largest orbit length (orbit-stabiliser)
    size_ok = (len(flat) == n) and (int(value["group_order"]) >= max(len(o) for o in orbits))
    my_orb = list(map(int, value["orbit_of"]))
    orb_consist = my_orb in orbits
    ok = partition_ok and orbit_const and size_ok and orb_consist and value["invariant"]
    return {"ok": ok, "resid": [0.0 if partition_ok else 1.0],
            "detail": f"轨道剖分合法 (并集=[0,{n}) 且互不相交)={partition_ok}; "
                      f"每个轨道上取值恒定 (⟺ 群不变性)={orbit_const}; "
                      f"群阶 {value['group_order']} ≥ 最大轨道长={size_ok}; 指定轨道在划分中={orb_consist}"}


REGISTRY.register(OpDef(
    name="strat_symmetry", band=BAND, sub=7,
    stmt="对称性论证: 若结构在群 G 的每个元素下不变, 则它在 G 的每条轨道上取常值, 轨道可作代表元化简",
    in_schema=_IN_SYM,
    out_schema={"kind": "dict",
                "fields": {"group_order": {"kind": "int", "min": 1}, "invariant": {"kind": "bool"},
                           "n_orbits": {"kind": "int", "min": 1},
                           "orbits": {"kind": "list", "minlen": 1,
                                      "items": {"kind": "list", "items": {"kind": "int"}, "minlen": 1}},
                           "orbit_of": {"kind": "list", "items": {"kind": "int"}, "minlen": 1},
                           "n_distinct_values": {"kind": "int", "min": 1}},
                "required": ["group_order", "invariant", "n_orbits", "orbits", "orbit_of",
                             "n_distinct_values"]},
    invoke=_inv_sym, verify=_ver_sym,
    cost=("units", 3.0), errors=("PRECOND_FAIL", "POSTCOND_FAIL", "BOUNDED_VIOLATION", "TIMEOUT",
                                 "SCHEMA_MISMATCH"),
    effects="none (纯函数)",
    doc="策略算子 symmetry | in: 结构取值表 + 置换生成元 (+ 指定轨道元素) | "
        "out: 生成子群阶 + 轨道划分 + 不变性结论 | 验收=**群作用轨道枚举下不变**: 枚举生成子群"
        "的全体元素 (非只查生成元), 逐个核对结构不变, 并给出轨道 | 验证器=轨道剖分合法性 + "
        "轨道上取值恒定 (与逐置换扫描等价的另一路径) + 轨道-稳定化子下界 | 3 units | "
        "诚实边界: 需要有限置换群且阶 ≤2000, 超界 BOUNDED_VIOLATION | "
        "失败: 不变性失败 POSTCOND_FAIL / 非置换生成元 PRECOND_FAIL",
    selfchecks=(
        {"id": "ssy_transposition_sym", "args": {"values": [1, 1, 2, 2], "generators": [[1, 0, 2, 3],
                                                                                        [0, 1, 3, 2]],
                                                 "orbit_of": 0},
         "check": lambda r: (r.value["invariant"] is True and r.value["group_order"] == 4
                             and r.value["orbits"] == [[0, 1], [2, 3]],
                             f"order={r.value['group_order']} orbits={r.value['orbits']}"),
         "expect": "值 [1,1,2,2] 在对换 (0 1) 与 (2 3) 下均不变 → 生成群阶 4 (Klein 四元群), "
                   "轨道 {0,1} 与 {2,3}; 元素 0 的轨道 [0,1] "
                   "(旧版写 [1,2,2,1] 是错的: (0 1) 会把它变成 [2,1,2,1])"},
        {"id": "ssy_full_swap_pair", "args": {"values": [5, 5], "generators": [[1, 0]], "orbit_of": 1},
         "check": lambda r: (r.value["group_order"] == 2 and r.value["orbits"] == [[0, 1]]
                             and r.value["orbit_of"] == [0, 1],
                             f"orbits={r.value['orbits']}"),
         "expect": "值 [5,5] 在 S₂ 下不变 → 单一轨道 [0,1]; 元素 1 的轨道同为 [0,1]"},
        {"id": "ssy_cyclic3", "args": {"values": [7, 7, 7], "generators": [[1, 2, 0]], "orbit_of": 2},
         "check": lambda r: (r.value["group_order"] == 3 and r.value["orbits"] == [[0, 1, 2]],
                             f"order={r.value['group_order']} orbits={r.value['orbits']}"),
         "expect": "3-循环生成 C₃ (阶 3): 常值结构不变, 单轨道 [0,1,2]"},
        {"id": "ssy_broken_symmetry", "args": {"values": [1, 2], "generators": [[1, 0]]},
         "expect_err": "POSTCOND_FAIL",
         "expect": "值 [1,2] 在对换下变成 [2,1] ≠ [1,2] → 不变性失败, 必须拒 (对称性不是假设)"},
        {"id": "ssy_not_permutation", "args": {"values": [1, 1], "generators": [[0, 0]]},
         "expect_err": "PRECOND_FAIL", "expect": "[0,0] 不是置换 (元素重复)"},
        {"id": "ssy_group_cap", "args": {"values": [1, 1, 1, 1, 1, 1, 1, 1],
                                         "generators": [[1, 0, 2, 3, 4, 5, 6, 7],
                                                        [2, 3, 4, 5, 6, 7, 0, 1],
                                                        [3, 2, 1, 0, 7, 6, 5, 4]],
                                         "max_group": 2000},
         "check": lambda r: (r.value["group_order"] > 8 and r.value["invariant"] is True,
                             f"order={r.value['group_order']}"),
         "expect": "3 个生成元张成较大的群 (常值结构恒不变): 用于检验闭包枚举与上限保护"},
    ),
))


# ===========================================================================
# 9308 bound_tighten
# ===========================================================================

_IN_BT = {
    "kind": "dict",
    "fields": {"candidates": {"kind": "list", "minlen": 1, "maxlen": 32,
                              "items": {"kind": "Fraction"}},
               "samples": {"kind": "list", "minlen": 1, "maxlen": 64,
                           "items": {"kind": "Fraction"}},
               "direction": {"kind": "str", "enum": ["upper", "lower"]},
               "goal": {"kind": "Fraction"}},
    "required": ["candidates", "samples", "direction"],
}


def _inv_bt(args: dict, ctl: CallCtl) -> dict:
    """Tighten a bound: pick from the candidate set the bound that is **tightest relative to the sample set** and still valid, then search for counter-examples among tighter candidates."""
    cands = [f(v) for v in args["candidates"]]
    samples = [f(v) for v in args["samples"]]
    direction = args["direction"]
    goal = f(args["goal"]) if "goal" in args and args["goal"] is not None else None
    # validity criterion: an upper bound must be ≥ all samples; a lower bound must be ≤ all samples
    def valid(b: Fraction) -> bool:
        return all(b >= s for s in samples) if direction == "upper" else all(b <= s for s in samples)
    feasible = [c for c in cands if valid(c)]
    if not feasible:
        raise SvoError(Err.POSTCOND_FAIL, ctx="op:strat_bound_tighten#9308",
                       detail=f"候选集 {[fs(c) for c in cands]} 中无一个满足{direction}界条件 "
                              f"(样本 {[fs(s) for s in samples]}) — 需补充候选")
    best = min(feasible) if direction == "upper" else max(feasible)
    # counter-example search: for each candidate tighter than best, look for a counter-example in the sample set — exhaustive over the sample set
    tighter = [c for c in cands
               if (c < best if direction == "upper" else c > best)]
    cex: list = []
    for c in tighter:
        bad = [s for s in samples if (s > c if direction == "upper" else s < c)]
        if bad:
            cex.append({"candidate": fs(c), "counterexample": fs(bad[0])})
    # direction check: the direction relative to the target must be correct (when goal is given)
    dir_ok = True
    goal_gap = ""
    if goal is not None:
        if direction == "upper":
            dir_ok = all(s <= goal for s in samples)
            goal_gap = fs(goal - best)
        else:
            dir_ok = all(s >= goal for s in samples)
            goal_gap = fs(best - goal)
    return {"value": {"best": fs(best), "n_feasible": len(feasible), "n_candidates": len(cands),
                      "tighter_refuted": cex, "direction_ok": bool(dir_ok),
                      "gap_to_goal": goal_gap,
                      "tightness_scope": "relative_to_given_candidates_and_samples"},
            "resid": [0.0],
            "notes": f"收紧结果: {direction} 界 = {fs(best)} (候选 {len(cands)} 个中 {len(feasible)} 个有效); "
                     f"更紧的 {len(tighter)} 个候选均被样本反例否决 {cex[:3]}; "
                     f"方向检查={dir_ok}"
                     + (f", 与目标距离={goal_gap}" if goal is not None else "")}


def _ver_bt(args: dict, value: dict) -> dict:
    """Independent verification: (a) independently compute the theoretical tightest bound from the **sample extremes**
    (upper → max(samples), lower → min(samples)) and check that the output best is the largest candidate **not exceeding
    that theoretical bound** (i.e. optimal within the candidate set);
    (b) recheck one by one that tighter candidates really have counter-examples (exhaustive over samples);
    (c) direction consistency (all samples lie on the correct side of best)."""
    cands = [f(v) for v in args["candidates"]]
    samples = [f(v) for v in args["samples"]]
    direction = args["direction"]
    best = f(value["best"])
    theo = max(samples) if direction == "upper" else min(samples)
    # best must be the extremum within the candidate set satisfying the direction condition (upper: smallest feasible, lower: largest feasible)
    feasible = sorted((c for c in cands if (c >= theo if direction == "upper" else c <= theo)))
    best_ok = bool(feasible) and (best == (feasible[0] if direction == "upper" else feasible[-1]))
    side_ok = all(best >= s for s in samples) if direction == "upper" else all(best <= s for s in samples)
    # counter-example recheck
    cex_ok = True
    for item in value["tighter_refuted"]:
        c = f(item["candidate"])
        s = f(item["counterexample"])
        if not ((s > c) if direction == "upper" else (s < c)):
            cex_ok = False
    # relation between the theoretical tightest bound and best: best is not tighter than the theoretical bound (upper: best ≥ theo)
    theo_ok = (best >= theo) if direction == "upper" else (best <= theo)
    ok = best_ok and side_ok and cex_ok and theo_ok
    return {"ok": ok, "resid": [0.0 if ok else 1.0],
            "detail": f"样本极值给出的理论最紧界={fs(theo)}; 候选集内最优={fs(feasible[-1] if feasible and direction == 'lower' else (feasible[0] if feasible else Fraction(0)))} "
                      f"与输出一致={best_ok}; 样本全在正确一侧={side_ok}; "
                      f"更紧候选的反例复查={cex_ok}; best 不紧于理论界={theo_ok}"}


REGISTRY.register(OpDef(
    name="strat_bound_tighten", band=BAND, sub=8,
    stmt="界收紧: 在满足样本约束的候选中取最紧者; 任何更紧的候选都必须给出反例或被反例否决",
    in_schema=_IN_BT,
    out_schema={"kind": "dict",
                "fields": {"best": {"kind": "str"}, "n_feasible": {"kind": "int", "min": 1},
                           "n_candidates": {"kind": "int", "min": 1},
                           "tighter_refuted": {"kind": "list", "maxlen": 32,
                                               "items": {"kind": "dict", "extra": False,
                                                         "fields": {"candidate": {"kind": "str"},
                                                                    "counterexample": {"kind": "str"}},
                                                         "required": ["candidate", "counterexample"]}},
                           "direction_ok": {"kind": "bool"}, "gap_to_goal": {"kind": "str"},
                           "tightness_scope": {"kind": "str"}},
                "required": ["best", "n_feasible", "n_candidates", "tighter_refuted",
                             "direction_ok", "gap_to_goal", "tightness_scope"]},
    invoke=_inv_bt, verify=_ver_bt,
    cost=("units", 1.0), errors=("POSTCOND_FAIL", "SCHEMA_MISMATCH"),
    effects="none (纯函数, 精确有理算术)",
    doc="策略算子 bound_tighten | in: 候选界集合 + 样本集 + 方向 (+ 可选目标) | "
        "out: 最紧可行界 + 被否决的更紧候选及其反例 | 验收=方向检查 (样本全在正确一侧) + "
        "反例搜索 (穷举样本集否决每个更紧候选) | 验证器=样本极值独立算理论界 + 候选集内最优核对 + "
        "反例复查 | 1 unit | 诚实边界: tightness_scope 固定为 "
        "'relative_to_given_candidates_and_samples' —— 本算子**不**声称对真实约束集的全局最优性 | "
        "失败: 无可行候选 POSTCOND_FAIL",
    selfchecks=(
        {"id": "sbt_upper", "args": {"candidates": [10, 7, 5, 4], "samples": [1, 3, 5],
                                     "direction": "upper"},
         "check": lambda r: (r.value["best"] == "5" and r.value["n_feasible"] == 3
                             and len(r.value["tighter_refuted"]) == 1,
                             f"best={r.value['best']} feasible={r.value['n_feasible']} "
                             f"refuted={len(r.value['tighter_refuted'])}"),
         "expect": "样本最大值 5: 候选 10、7、5 三个都 ≥5 (有效数 3), 4 无效; 最紧 = 5; "
                   "比 5 更紧的候选 4 被样本 5 否决 (反例 = 5)"},
        {"id": "sbt_lower", "args": {"candidates": [0, 2, 3, 6], "samples": [1, 4, 7],
                                     "direction": "lower"},
         "check": lambda r: (r.value["best"] == "0" and r.value["n_feasible"] == 1
                             and len(r.value["tighter_refuted"]) == 3,
                             f"best={r.value['best']} feasible={r.value['n_feasible']} "
                             f"refuted={len(r.value['tighter_refuted'])}"),
         "expect": "样本最小值 1: lower 界须 ≤ 全部样本, 故仅候选 0 有效 (0≤1); 2、3、6 都 >1 无效; "
                   "lower 取最大可行 = 0; 比 0 更紧的三个候选 (2,3,6) 全部被样本反例否决"},
        {"id": "sbt_goal_direction", "args": {"candidates": [10, 7, 6], "samples": [1, 3, 5],
                                              "direction": "upper", "goal": 6},
         "check": lambda r: (r.value["best"] == "6" and r.value["direction_ok"] is True
                             and r.value["gap_to_goal"] == "0"
                             and r.value["tightness_scope"] == "relative_to_given_candidates_and_samples",
                             f"best={r.value['best']} gap={r.value['gap_to_goal']}"),
         "expect": "目标 6: 样本 ≤6 成立 → direction_ok; 最紧可行界 6 与目标重合 (gap=0); "
                   "tightness_scope 显式限定为相对候选集与样本集"},
        {"id": "sbt_goal_direction_fails", "args": {"candidates": [10, 7], "samples": [1, 9],
                                                    "direction": "upper", "goal": 6},
         "check": lambda r: (r.value["direction_ok"] is False,
                             f"direction_ok={r.value['direction_ok']} best={r.value['best']}"),
         "expect": "目标 6 但样本含 9: 上界不能收紧到目标以下 → direction_ok=False 如实上报"},
        {"id": "sbt_no_feasible", "args": {"candidates": [1, 2], "samples": [5], "direction": "upper"},
         "expect_err": "POSTCOND_FAIL",
         "expect": "上界候选 1、2 都 < 样本 5 → 无可行候选, 必须拒 (不许返回一个错界)"},
        {"id": "sbt_bad_direction", "args": {"candidates": [1], "samples": [1], "direction": "mid"},
         "expect_err": "SCHEMA_MISMATCH", "expect": "方向枚举外取值"},
    ),
))