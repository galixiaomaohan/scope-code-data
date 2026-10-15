"""§10.7 logic exam generator (curriculum/gen_L3.py).

Three families of **answer-known** logic problems (explicitly listed in brief §10.7):

  ksat         random k-SAT (phase-transition region n=20–50): **satisfiability known** — guaranteed
               by construction (build a model first, then generate clauses)
  hilbert      Hilbert proof problems (template instantiation): the template ships its step sequence;
               truth = that proof passes the kernel
  arg_repair   argument-graph repair (drop an edge, complete it): the answer is the dropped edge

**How "satisfiability known" is implemented (this must be spelled out, because it decides how much
the problems can be trusted)**: the naive route is "generate random k-SAT and let a solver decide
satisfiability" — but that ties the generator to the solver's strength (this library's naive DPLL is
weak in the phase-transition region; only 0.333 of capacity was observed). This generator instead uses
**construction**:
  1. pick a random "hidden model" σ ∈ {0,1}ⁿ;
  2. when generating clauses, guarantee every clause is **true** under σ (rejection sampling: resample
     any random clause that is false under σ).
So "this instance is satisfiable" is a **constructive fact** (the witness is σ), independent of solver
strength. Inverse problems (UNSAT construction) are **not produced** by this generator — random UNSAT
needs DRAT-grade certificates, which belongs to §10.2; forcing UNSAT problems without a certificate
amounts to asking people to trust the problem statement.

**Determinism**: only random.Random(seed); same (n, seed, families) → bit-identical exam set.
"""
from __future__ import annotations

import random

from ..errors import Err, SvoError
from ..logiccore import check_proof, dpll_small

#: Hilbert proof templates (propositional tautologies + step sequences that pass the kernel check)
#: Each template: (goal, hypotheses, step sequence, description) — the step sequence is built from
#: logiccore's axiom schemas A1/A2/A3 + MP
_HILBERT: tuple[tuple, ...] = (
    ("(a -> a)", [], [
        {"rule": "axiom", "formula": "(a -> ((a -> a) -> a))"},
        {"rule": "axiom", "formula": "((a -> ((a -> a) -> a)) -> ((a -> (a -> a)) -> (a -> a)))"},
        {"rule": "mp", "from": [1, 0], "formula": "((a -> (a -> a)) -> (a -> a))"},
        {"rule": "axiom", "formula": "(a -> (a -> a))"},
        {"rule": "mp", "from": [2, 3], "formula": "(a -> a)"}],
     "同一律 a→a (标准 3 步 MP 构造)"),
    ("(a -> b)", ["a", "b"], [
        {"rule": "hyp", "idx": 1, "formula": "b"},
        {"rule": "axiom", "formula": "(b -> (a -> b))"},
        {"rule": "mp", "from": [1, 0], "formula": "(a -> b)"}],
     "由假设直接得到蕴含 (A1 + MP)"),
    ("(a -> a)", ["a"], [
        {"rule": "hyp", "idx": 0, "formula": "a"},
        {"rule": "axiom", "formula": "(a -> (a -> a))"},
        {"rule": "mp", "from": [1, 0], "formula": "(a -> a)"}],
     "带假设同一律 (A1 + MP)"),
)


def _gen_ksat(rng: random.Random, idx: int) -> dict:
    """Constructively satisfiable 3-SAT: take a hidden model first, then sample only clauses true under it."""
    nv = rng.randint(20, 50)
    ratio = 4.26
    m = int(round(ratio * nv))
    model = {v: bool(rng.getrandbits(1)) for v in range(1, nv + 1)}
    clauses: list = []
    guard = 0
    while len(clauses) < m and guard < m * 200:
        guard += 1
        vs = rng.sample(range(1, nv + 1), 3)
        cl = [v if rng.getrandbits(1) else -v for v in vs]
        # Rejection sampling: accept only clauses true under the hidden model → satisfiability is constructive
        if any((l > 0) == model[abs(l)] for l in cl):
            clauses.append(sorted(cl))
    if len(clauses) < m:
        raise SvoError(Err.PRECOND_FAIL, ctx="gen_L3.ksat",
                       detail=f"拒绝采样未能生成 {m} 条子句 (得 {len(clauses)}) — 拒绝产题")
    return {"id": f"L3-sat-{idx:05d}", "family": "ksat", "n_vars": nv, "n_clauses": m,
            "cnf": clauses,
            "model_witness": {str(k): v for k, v in sorted(model.items())},
            "statement": f"判定该 {nv} 变量 3-SAT ({m} 子句) 的可满足性; 若可满足给出指派",
            "check": {"op": "logic_sat_verify",
                      "args": {"cnf": clauses,
                               "assign": {str(k): v for k, v in sorted(model.items())}},
                      "assert": {"violated": 0}},
            "identifiability": ("满足性**由构造保证**: 隐藏模型 σ 在生成时参与拒样, "
                                "故 σ 满足全部子句 (verify 会逐子句复验 violated=0)。"
                                f"相变区 m/n≈{ratio} 使实例对求解器有难度, 但答案不依赖求解器")}


def _gen_hilbert(rng: random.Random, idx: int) -> dict:
    """Hilbert proof problem: template instantiation; the answer = that step sequence passing the kernel (truth comes from the kernel decision)."""
    goal, hyps, steps, desc = _HILBERT[rng.randrange(len(_HILBERT))]
    ok, idx_fail, detail = check_proof(hyps, steps, goal)
    if not ok:
        raise SvoError(Err.PRECOND_FAIL, ctx="gen_L3.hilbert",
                       detail=f"模板自检失败 (步 {idx_fail}): {detail} — 出题器自身的模板必须先过核")
    return {"id": f"L3-hil-{idx:05d}", "family": "hilbert", "goal": goal, "hyps": hyps,
            "steps": steps, "n_steps": len(steps),
            "statement": f"给定假设 {hyps or '(无)'}, 用 Hilbert 公理 (A1/A2/A3) 与 MP 证明 {goal}",
            "check": {"op": "logic_hilbert_prove",
                      "args": {"hyps": hyps, "steps": steps, "goal": goal},
                      "assert": {"n_steps": len(steps)}},
            "identifiability": (f"答案由**核判定**给出而非人评: 步序列逐步 O(1) 复验 "
                                f"(模板自检已在生成时真跑过: {detail})。{desc}")}


def _gen_arg_repair(rng: random.Random, idx: int) -> dict:
    """Argument-graph repair: drop one edge from a complete graph; the answer is that edge (it must re-verify)."""
    # Complete graph: p1=(1,2), p2=(−1,2) ⊢ n=(2)  (resolution)
    nodes = [{"id": "p1", "stmt": "1,2"}, {"id": "p2", "stmt": "-1,2"}, {"id": "n", "stmt": "2"},
             {"id": "p3", "stmt": "-2,3"}, {"id": "m", "stmt": "3"}]
    edges_all = [
        {"from": ["p1", "p2"], "to": "n", "kind": "res", "eid": "e1", "record": {}},
        {"from": ["n", "p3"], "to": "m", "kind": "res", "eid": "e2", "record": {}},
    ]
    drop = rng.randrange(len(edges_all))
    missing = edges_all[drop]
    remaining = [e for i, e in enumerate(edges_all) if i != drop]
    premises = ["p1", "p2", "p3"]
    return {"id": f"L3-arg-{idx:05d}", "family": "arg_repair", "missing_eid": missing["eid"],
            "missing_edge": missing,
            "nodes": nodes, "edges": remaining, "premises": premises, "conclusion": "m",
            "statement": f"论证图缺一条边 (结论 {missing['to']} 无法回溯到前提); 补全该边",
            "check": {"op": "logic_arg_graph",
                      "args": {"nodes": nodes, "edges": edges_all, "premises": premises,
                               "conclusion": "m"},
                      "assert": {"valid": True, "n_edges": len(edges_all)}},
            "identifiability": (f"修补目标**唯一确定到边身份**: 缺边 eid={missing['eid']} "
                                f"({missing['from']} → {missing['to']}), 补回后整图 DAG 无环、"
                                f"每条边过归结核、结论回溯闭合 —— 由 arg_check 独立复验; "
                                f"若存在多条可补边, 判据是'补后验收通过', 不要求与真值边相同")}


_GENS = {"ksat": _gen_ksat, "hilbert": _gen_hilbert, "arg_repair": _gen_arg_repair}
_CYCLE = ("ksat", "hilbert", "arg_repair")


def generate(n: int, seed: int = 20260920, families=None) -> "list[dict]":
    if n < 0:
        raise SvoError(Err.SCHEMA_MISMATCH, ctx="gen_L3.generate", detail=f"n={n} 为负")
    fams = list(families) if families else list(_CYCLE)
    for f_ in fams:
        if f_ not in _GENS:
            raise SvoError(Err.SCHEMA_MISMATCH, ctx="gen_L3.generate", detail=f"未知族 {f_!r}")
    rng = random.Random(seed)
    out: list = []
    for i in range(n):
        item = _GENS[fams[i % len(fams)]](rng, i)
        if not item.get("identifiability"):
            raise SvoError(Err.UNIDENTIFIABLE, ctx="gen_L3.generate",
                           detail=f"题 {item['id']} 缺可辨识性论证 — 不得入集")
        out.append(item)
    return out


def verify_one(item: dict) -> "tuple[bool, str]":
    """Actually run the problem's acceptance operator and compare assertions; for the ksat family also
    re-verify independently against the **hidden model**."""
    from ..registry import REGISTRY

    spec = item["check"]
    if not REGISTRY.has(spec["op"]):
        return False, f"验收算子 {spec['op']!r} 未注册"
    r = REGISTRY.tool(spec["op"]).call(spec["args"])
    if not r.ok:
        return False, f"{spec['op']} 调用失败: {r.err} {r.notes[:120]}"
    bad = [f"{k}: 期望 {v!r} 实得 {r.value.get(k)!r}" for k, v in spec["assert"].items()
           if r.value.get(k) != v]
    if bad:
        return False, f"{spec['op']} 断言不符: {bad}"
    if item["family"] == "ksat":
        # Independent re-verification: the hidden model satisfies every clause (a third check,
        # independent of the operator path)
        model = {int(k): bool(v) for k, v in item["model_witness"].items()}
        for ci, cl in enumerate(item["cnf"]):
            if not any((l > 0) == model[abs(l)] for l in cl):
                return False, f"隐藏模型不满足第 {ci} 条子句 — 出题器构造有误"
        # Then try the solver once more (corroboration only, not a criterion)
        dpll_small([frozenset(c) for c in item["cnf"]], max_nodes=2_000_000)
    return True, f"校验通过: {spec['op']}"


def verify_all(items: "list[dict]", limit: "int | None" = None) -> dict:
    n = len(items) if limit is None else min(limit, len(items))
    failures: list = []
    passed = 0
    for it in items[:n]:
        ok, detail = verify_one(it)
        if ok:
            passed += 1
        else:
            failures.append({"id": it["id"], "family": it["family"], "detail": detail})
    return {"total": n, "passed": passed, "failed": n - passed,
            "rate": (passed / n) if n else 1.0, "failures": failures[:20],
            "unsat_instances": "**不生成** (随机构造 UNSAT 需要 DRAT 级证书, 属 §10.2 职责)"}


def manifest(items: "list[dict]") -> dict:
    from . import fingerprint
    fams: dict = {}
    nv = []
    for it in items:
        fams[it["family"]] = fams.get(it["family"], 0) + 1
        if "n_vars" in it:
            nv.append(it["n_vars"])
    return {"n": len(items), "by_family": dict(sorted(fams.items())),
            "ksat_n_range": [min(nv), max(nv)] if nv else [],
            "fingerprint": fingerprint(items)}
