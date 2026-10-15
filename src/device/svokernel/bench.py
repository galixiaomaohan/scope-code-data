"""§16.7 benchmark harness (eval-only, contamination-proof).

**Anti-contamination measures (checkable item by item)**:
  1. **eval-only**: this module takes part in no operator training/registration flow (it is not in the ops
     MANIFEST and is imported by no operator); it only **reads** operators and aggregates results;
  2. **offline snapshot**: all three domains (math/code/logic) use **locally generated synthetic exam sets**,
     with no network access; an exam set is fully determined by (n, seed), and its fingerprint lands in the
     report — anyone can reproduce the same set with the same seed;
  3. **no mixing of public benchmarks**: this harness contains no problem content from public benchmarks
     (HumanEval etc.) — those need offline snapshot files that are not present in this container; the report
     **explicitly marks "no public benchmark introduced"** instead of pretending they were run.

**Unified metrics (listed in §16.7)**:
  SuccessRate          problems passing acceptance / total problems
  C_beam               average **acceptance steps** (in this implementation = operator calls per problem until acceptance)
  esc_rate             escalation trigger rate (fraction of problems where even the minimum residual exceeds the threshold, so the run escalates/abandons)
  certificate verify % among certificate-carrying problems, the fraction whose certificate was re-verified by an **independent checker**

Four domains:
  math   : §8.4 synthetic curriculum (gen_L1) — each problem's check spec actually runs
  code   : §9.7 defect-injection tasks (locally constructed) — three-gate acceptance
  logic  : §10.7 random k-SAT (phase-transition region) — assignment/certificate verification
  agent  : §11 multi-tool orchestration — **currently not implemented, the report explicitly marks it not covered**

Honesty boundary: this module runs only problems that **machines can accept**; anything requiring human
review or an external benchmark is labelled not covered rather than filled in with "it looks like it ran"
(§18 red line).
"""
from __future__ import annotations

import random
import time

from .errors import SvoError
from .registry import REGISTRY

#: Default evaluation sizes (overridable by the caller; all of them land in the report)
DEFAULTS = {"math_n": 200, "code_n": 40, "logic_n": 60, "seed": 20260920}


def _verdict(op: str, args: dict):
    """Call one operator and return a structured ok/err/resid result (unified reading convention)."""
    if not REGISTRY.has(op):
        return {"ok": False, "err": "OP_MISSING", "resid": float("nan"), "cost": 0.0,
                "cert_ok": None, "steps": 1}
    r = REGISTRY.tool(op).call(args)
    return {"ok": r.ok, "err": r.err, "resid": r.resid_scalar, "cost": r.cost_units,
            "cert_ok": r.cert_independent, "steps": len(r.stages)}


# ---------------------------------------------------------------------------
# math domain
# ---------------------------------------------------------------------------


def bench_math(n: int, seed: int) -> dict:
    """Math: the gen_L1 synthetic curriculum. Each problem actually runs one operator per its check spec; counts success/steps/cert."""
    from .curriculum import gen_L1

    items = gen_L1.generate(n, seed=seed)
    ok = 0
    steps_total = 0
    cert_checked = 0
    cert_ok = 0
    esc = 0
    failures: list = []
    t0 = time.perf_counter()
    for it in items:
        spec = it["check"]
        v = _verdict(spec["op"], spec["args"])
        steps_total += v["steps"]
        if not v["ok"]:
            esc += 1
            failures.append({"id": it["id"], "err": v["err"]})
            continue
        r = REGISTRY.tool(spec["op"]).call(spec["args"])
        bad = [k for k, want in spec["assert"].items() if r.value.get(k) != want]
        if bad:
            esc += 1
            failures.append({"id": it["id"], "mismatch": bad})
            continue
        ok += 1
        if r.cert is not None:
            cert_checked += 1
            if r.cert_independent is True:
                cert_ok += 1
    dt = time.perf_counter() - t0
    return {"domain": "math", "n": len(items), "success": ok,
            "success_rate": ok / max(1, len(items)),
            "c_beam": steps_total / max(1, len(items)), "esc_rate": esc / max(1, len(items)),
            "cert_rate": (cert_ok / cert_checked) if cert_checked else None,
            "cert_checked": cert_checked, "elapsed_s": round(dt, 3),
            "fingerprint": gen_L1.manifest(items)["fingerprint"],
            "by_family": gen_L1.manifest(items)["by_family"],
            "failures": failures[:10]}


# ---------------------------------------------------------------------------
# code domain (locally injected defect tasks)
# ---------------------------------------------------------------------------


#: Seed program templates for the code domain (correct programs, test input sets). **Module-level
#: constants**: the two injection conventions (`_inject_tasks` reversible filter / `_inject_tasks_all`
#: full) must use the same template set, otherwise the difference between the two conventions is
#: polluted by "different problems" instead of being purely a difference in injection difficulty.
_CODE_TEMPLATES: tuple = (
    # (correct program, test input set)
    ([{"t": "assign", "var": "v0", "e": {"t": "bin", "op": "+", "l": {"t": "var", "n": "a0"},
                                        "r": {"t": "var", "n": "a1"}}}],
     [[i, j] for i in range(0, 5) for j in range(0, 5)]),
    ([{"t": "assign", "var": "v0", "e": {"t": "bin", "op": "*", "l": {"t": "var", "n": "a0"},
                                        "r": {"t": "lit", "v": 2}}}],
     [[i] for i in range(0, 8)]),
    ([{"t": "assign", "var": "v0", "e": {"t": "bin", "op": "-", "l": {"t": "var", "n": "a0"},
                                        "r": {"t": "lit", "v": 3}}}],
     [[i] for i in range(0, 9)]),
    ([{"t": "if", "c": {"t": "bin", "op": "<", "l": {"t": "var", "n": "a0"},
                        "r": {"t": "lit", "v": 0}},
       "th": [{"t": "assign", "var": "v0", "e": {"t": "lit", "v": -1}}],
       "el": [{"t": "assign", "var": "v0", "e": {"t": "lit", "v": 1}}]}],
     [[i] for i in range(-5, 6)]),
)


def _inject_tasks(n: int, seed: int) -> list:
    """Build n defect-injection tasks (off-by-one / wrong constant / wrong operator / statement deletion), each with train + held-out tests.

    Every task mutates a **correct seed program** (ground truth known): records (buggy, tests, ground_truth_kind).
    """
    from .ops import code_repair as R

    templates = list(_CODE_TEMPLATES)
    rng = random.Random(seed)
    out: list = []
    for k in range(n):
        base, inputs = templates[k % len(templates)]
        # compute the expectations with the *correct* program
        import copy
        good = copy.deepcopy(base)
        tests = []
        for inp in inputs:
            args = {a: 0 for a in ("a0", "a1", "a2", "a3")}
            for i, v in enumerate(inp):
                args[("a0", "a1", "a2", "a3")[i]] = v
            from .ops.code_core import interp
            r = interp(good, args, 200_000)
            if r["error"] is not None:
                continue
            tests.append({"args": list(inp), "expect": int(r["state"].get("v0", 0))})
        if len(tests) < 4:
            continue
        # Defect injection: pick only from **reversible classes** (the defect and its fix must both lie in
        # the patch search space). Why this constraint: injecting a defect like "lit 10→0" while the patch
        # space can only set literals to 0/1/±1 makes the "correct" 10 unreachable — that is not the
        # operator failing, it is the **benchmark itself being unsolvable** (measured code-domain pass rate
        # 0.325 comes mostly from this). A benchmark must contain only solvable instances, otherwise it
        # measures the problems, not the operator.
        ready = [m for m in R._patch_candidates(good, 64)
                 if m[0] in ("arith_flip", "cond_negate", "boundary_pm1", "stmt_delete")]
        if not ready:
            continue
        kind, desc, buggy = ready[rng.randrange(len(ready))]
        if kind == "stmt_delete":
            continue  # the "inverse" of statement deletion is not in the search space (it requires rebuilding a statement) — likewise unsolvable
        out.append({"buggy": buggy, "tests": tests, "gt_kind": kind, "gt_desc": desc,
                    "template": k % len(templates)})
    return out


def _inject_tasks_all(n: int, seed: int) -> list:
    """Injection under the **full-scope measurement convention**: pick defects from **all** mutation classes (no reversibility filter).

    Why both conventions must be reported (auditor F): injecting only reversible classes = "testing only
    problems we can solve", so the success rate is necessarily inflated (measured 1.0000 under the filtered
    convention vs about 0.325–0.5 under the full convention). The honest benchmark practice is to **report
    both conventions**, letting readers see where the operator's real solving boundary lies.
    """
    import copy
    import random as _r

    from .ops import code_repair as R
    from .ops.code_core import interp

    rng = _r.Random(seed + 7717)
    out: list = []
    for k in range(n):
        base, inputs = _CODE_TEMPLATES[k % len(_CODE_TEMPLATES)]
        good = copy.deepcopy(base)
        tests = []
        for inp in inputs:
            args = {a: 0 for a in ("a0", "a1", "a2", "a3")}
            for i, v in enumerate(inp):
                args[("a0", "a1", "a2", "a3")[i]] = v
            r = interp(good, args, 200_000)
            if r["error"] is not None:
                continue
            tests.append({"args": list(inp), "expect": int(r["state"].get("v0", 0))})
        if len(tests) < 4:
            continue
        muts = R._patch_candidates(good, 64)          # no filter: all classes are present
        if not muts:
            continue
        kind, desc, buggy = muts[rng.randrange(len(muts))]
        out.append({"buggy": buggy, "tests": tests, "gt_kind": kind, "gt_desc": desc,
                    "template": k % len(_CODE_TEMPLATES)})
    return out


def _rate_for(tasks, n_want: int) -> dict:
    ok = 0
    esc = 0
    failures: list = []
    for i, tk in enumerate(tasks):
        r = REGISTRY.tool("rep_patch_gen").call(
            {"program": tk["buggy"], "tests": tk["tests"], "max_candidates": 32, "check_box": True})
        if not r.ok:
            esc += 1
            failures.append({"i": i, "gt_kind": tk["gt_kind"], "err": r.err})
            continue
        ok += 1
    return {"n": len(tasks), "success": ok, "success_rate": ok / max(1, len(tasks)),
            "esc_rate": esc / max(1, len(tasks)), "failures": failures[:8],
            "by_gt_kind": _group_by_kind(tasks, failures)}


def _group_by_kind(tasks, failures) -> dict:
    bad_idx = {f["i"] for f in failures}
    agg: dict = {}
    for i, tk in enumerate(tasks):
        cur = agg.setdefault(tk["gt_kind"], [0, 0])
        cur[1] += 1
        if i not in bad_idx:
            cur[0] += 1
    return {k: f"{v[0]}/{v[1]}" for k, v in sorted(agg.items())}


def bench_code(n: int, seed: int) -> dict:
    """Code: acceptance per problem = the three gates (rep_patch_gen). **Both injection conventions are reported.**"""
    from .ops import code_repair as R

    tasks = _inject_tasks(n, seed)
    ok = 0
    steps_total = 0
    cert_checked = 0
    cert_ok = 0
    esc = 0
    failures: list = []
    t0 = time.perf_counter()
    for i, tk in enumerate(tasks):
        try:
            r = REGISTRY.tool("rep_patch_gen").call(
                {"program": tk["buggy"], "tests": tk["tests"], "max_candidates": 32,
                 "check_box": True})
        except SvoError as e:  # pragma: no cover - call() does not collapse; fallback here
            esc += 1
            failures.append({"i": i, "err": str(e)[:80]})
            continue
        steps_total += len(r.stages)
        if not r.ok:
            esc += 1
            failures.append({"i": i, "err": r.err, "note": r.notes[:80]})
            continue
        ok += 1
        if r.cert is not None:
            cert_checked += 1
            if r.cert_independent is True:
                cert_ok += 1
    dt = time.perf_counter() - t0
    # **Second convention**: full injection (no reversibility filter), exposing the overestimate caused
    # by "testing only solvable problems".
    full = _rate_for(_inject_tasks_all(n, seed), n)
    return {"domain": "code", "n": len(tasks), "success": ok,
            "success_rate": ok / max(1, len(tasks)),
            "c_beam": steps_total / max(1, len(tasks)), "esc_rate": esc / max(1, len(tasks)),
            "cert_rate": (cert_ok / cert_checked) if cert_checked else None,
            "cert_checked": cert_checked, "elapsed_s": round(dt, 3),
            "rate_reversible_injection": ok / max(1, len(tasks)),
            "rate_full_injection": full["success_rate"],
            "full_injection_detail": {"n": full["n"], "success": full["success"],
                                      "by_gt_kind": full["by_gt_kind"],
                                      "failures": full["failures"][:5]},
            "note": "验收=三重门 (原测试全绿 + held-out 全绿 + 变异体杀死); 每题由正确程序变异而来, "
                    "真值已知。**两个口径都报**: rate_reversible_injection = 只注入"
                    "「补丁搜索空间内可逆」的缺陷类 (主口径, 与 rep_patch_gen 的能力面一致); "
                    "rate_full_injection = **不过滤**注入全部变异类 (含 lit_set/stmt_delete 等"
                    "不可逆类) —— 后者才是「什么都得修」口径, 数值明显更低 (审计员 F 指出"
                    "只报前者会高估能力)",
            "failures": failures[:10]}


# ---------------------------------------------------------------------------
# logic domain (random k-SAT)
# ---------------------------------------------------------------------------


def _rand_ksat(nv: int, k: int, m: int, rng: random.Random):
    clauses = []
    for _ in range(m):
        vs = rng.sample(range(1, nv + 1), min(k, nv))
        clauses.append(frozenset(v if rng.random() < 0.5 else -v for v in vs))
    return clauses


def bench_logic(n: int, seed: int) -> dict:
    """Logic: **certificate verification rate** on phase-transition random k-SAT, reported per the operator's **declared capacity cap**.

    Why the split columns: `logic_sat_solve`'s schema caps max_vars at 22 (naive DPLL is exponential; the
    cap is a **deliberate fail-safe**, not a defect). The brief's phase-transition region is n=20–50 —
    above the cap the operator is **already outside its declared capability**. Mixing the two into one
    success rate would hide the real capability boundary, so both are reported here:
      rate_all          : success rate over the whole range (n=20–50) — low is expected, since most problems are out of range
      rate_within_cap   : success rate within the operator's declared capacity (n ≤ max_vars_cap) — the operator's real capability
      out_of_capacity   : number of problems rejected for being out of range
    """
    rng = random.Random(seed)
    MAX_VARS_CAP = 22  # matches ops/logic_ops.py _IN_SS.max_vars (hard-coded here and recorded in the report)
    ok = 0
    steps_total = 0
    cert_checked = 0
    cert_ok = 0
    esc = 0
    ooc = 0
    within_n = within_ok = 0
    failures: list = []
    t0 = time.perf_counter()
    for i in range(n):
        nv = rng.randint(20, 50)
        k = 3
        m = int(round(4.26 * nv))
        cnf = _rand_ksat(nv, k, m, rng)
        in_cap = nv <= MAX_VARS_CAP
        if in_cap:
            within_n += 1
        r = REGISTRY.tool("logic_sat_solve").call({"cnf": [sorted(c) for c in cnf]})
        steps_total += len(r.stages)
        if not r.ok:
            esc += 1
            if not in_cap:
                ooc += 1
            failures.append({"i": i, "nv": nv, "m": m, "in_cap": in_cap, "err": r.err})
            continue
        assign = r.value.get("assign") if isinstance(r.value, dict) else None
        if not assign:
            esc += 1
            failures.append({"i": i, "nv": nv, "m": m, "in_cap": in_cap, "err": "NO_ASSIGN"})
            continue
        v = REGISTRY.tool("logic_sat_verify").call(
            {"cnf": [sorted(c) for c in cnf], "assign": assign})
        if not v.ok:
            esc += 1
            failures.append({"i": i, "nv": nv, "m": m, "in_cap": in_cap, "err": "VERIFY_FAIL"})
            continue
        ok += 1
        if in_cap:
            within_ok += 1
        if r.cert is not None:
            cert_checked += 1
            if r.cert_independent is True:
                cert_ok += 1
    dt = time.perf_counter() - t0
    return {"domain": "logic", "n": n, "success": ok,
            "success_rate": ok / max(1, n), "c_beam": steps_total / max(1, n),
            "esc_rate": esc / max(1, n),
            "cert_rate": (cert_ok / cert_checked) if cert_checked else None,
            "cert_checked": cert_checked, "elapsed_s": round(dt, 3),
            "rate_within_capacity": (within_ok / within_n) if within_n else None,
            "n_within_capacity": within_n, "n_out_of_capacity": ooc,
            "max_vars_cap": MAX_VARS_CAP,
            "note": f"相变区 3-SAT (n=20–50, m/n≈4.26); 算子声明容量 max_vars≤{MAX_VARS_CAP} "
                    f"(朴素 DPLL 指数级, 该界是刻意的失效安全)。"
                    f"rate_all 覆盖全区间; rate_within_capacity 才是算子在其声明能力内的真实成功率; "
                    f"越界题计入 out_of_capacity 并**不**算作能力不足。"
                    f"已解实例必须通过独立指派复验; 本 harness 不声称 UNSAT (无 DRAT 证书链)",
            "failures": failures[:6]}


# ---------------------------------------------------------------------------
# agent domain (not implemented, explicitly labelled)
# ---------------------------------------------------------------------------


def bench_agent(n: int, seed: int) -> dict:
    """Agent domain (§16.7's "multi-tool orchestration: cost/steps/acceptance rate").

    Each task = an orchestration **that requires backtracking**: the candidate list puts doomed calls
    first (unregistered tool / out-of-domain arguments) and acceptable calls after; `agent_call_loop`
    tries them one by one until acceptance passes. The three metrics then carry real meaning:
      success_rate fraction of tasks passing acceptance (does the orchestration converge to a verified solution)
      c_beam       average **number of attempts** (backtracking steps — smaller means better prior ordering)
      esc_rate     fraction where no candidate passed acceptance (the escalation / next-candidate trigger rate)
      cert_rate    fraction of accepted calls carrying an **independent re-verification certificate**
                   (a key link in the combined anchor)
Also attached are the results of the two sub-items replay and committee (trust anchors of the agent
domain, which must not be absent).
    """
    import random

    from .ops.agent_ops import _mock_search

    rng = random.Random(seed)
    trials = max(4, n)
    ok = 0
    steps_total = 0
    esc = 0
    cert_checked = 0
    cert_ok = 0
    costs = 0.0
    failures: list = []
    t0 = time.perf_counter()
    for i in range(trials):
        a = rng.randint(-6, 6)
        b = rng.randint(-6, 6)  # a,b build the "doomed candidates" that make backtracking necessary
        cands = [
            {"tool": "no_such_tool", "args": {}},                       # bind failure → backtrack
            {"tool": "code_exec",
             "args": {"program": [{"t": "assign", "var": "v0",
                                   "e": {"t": "bin", "op": "div", "l": {"t": "var", "n": "a0"},
                                         "r": {"t": "var", "n": "a1"}}}],
                      "inputs": {"a0": a, "a1": b}, "step_limit": 200}},  # structurally illegal → backtrack
            # An acceptable candidate: exact root-finding **with a certificate** (poly_backsub), so that
            # cert_rate is measurable. (Earlier it used math_evaluate_expr — verifier only, no certificate,
            # which kept cert_rate at null.)
            {"tool": "solve_poly_exact", "args": {"p": -1, "q": -6}}
        ]
        r = REGISTRY.tool("agent_call_loop").call(
            {"candidates": cands, "accept_resid_max": 1e-9, "budget_units": 50.0})
        steps_total += len(r.stages)
        if not r.ok:
            esc += 1
            failures.append({"i": i, "err": r.err})
            continue
        v = r.value
        ok += 1
        costs += float(v["budget_spent"])
        acc = v["accepted"]
        if acc.get("cert_kind") is not None:
            cert_checked += 1
            if acc.get("cert_independent") is True:
                cert_ok += 1
    # sub-items: deterministic replay + committee (the two trust anchors of the agent domain)
    rep = REGISTRY.tool("agent_replay_audit").call(
        {"calls": [{"tool": "solve_poly", "args": {"p": -1.0, "q": -6.0}}], "seed": seed, "runs": 2})
    comm = REGISTRY.tool("agent_committee").call(
        {"question": "roots", "agree_field": "roots",
         "members": [{"name": "a", "op": "solve_poly", "args": {"p": -1.0, "q": -6.0}},
                     {"name": "b", "op": "solve_poly", "args": {"p": -1.0, "q": -6.0}}]})
    # real measurement of the agent domain: n is the number of tasks (concrete instances set by seed)
    n_real = trials
    dt = time.perf_counter() - t0
    return {"domain": "agent", "n": n_real, "success": ok,
            "success_rate": ok / max(1, n_real),
            "c_beam": steps_total / max(1, n_real), "esc_rate": esc / max(1, n_real),
            "cert_rate": (cert_ok / cert_checked) if cert_checked else None,
            "cert_checked": cert_checked, "elapsed_s": round(dt, 3), "covered": True,
            "total_cost_units": round(costs, 4),
            "sub_anchors": {"replay_equal": bool(rep.ok and rep.value.get("equal")),
                            "committee_unanimous": bool(comm.ok and comm.value.get("unanimous")),
                            "mock_corpus_hits": len(_mock_search("quadratic", 2))},
            "note": "每任务 = 一段需要回溯的编排 (前面放必然失败的候选: 未注册工具 + 非法程序), "
                    "由 agent_call_loop 回溯到能过验收的调用。c_beam 在此读作**平均尝试次数** "
                    "(越小说明先验序越好); cert_rate 统计被接受调用的独立复验证书比例。"
                    "子项 replay/committee 是 agent 域的信任锚, 一并测量 (mock_search 只读固定语料, 无网)"}


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------


def run_all(math_n: int = None, code_n: int = None, logic_n: int = None,
            seed: int = None) -> dict:
    from . import ops  # noqa: F401  load operators (import registers)

    mn = DEFAULTS["math_n"] if math_n is None else int(math_n)
    cn = DEFAULTS["code_n"] if code_n is None else int(code_n)
    ln = DEFAULTS["logic_n"] if logic_n is None else int(logic_n)
    sd = DEFAULTS["seed"] if seed is None else int(seed)
    domains = [bench_math(mn, sd), bench_code(cn, sd), bench_logic(ln, sd), bench_agent(0, sd)]
    covered = [d for d in domains if d.get("success_rate") is not None]
    agg = {
        "success_rate": (sum(d["success_rate"] * d["n"] for d in covered)
                         / max(1, sum(d["n"] for d in covered))) if covered else None,
        "c_beam": (sum(d["c_beam"] * d["n"] for d in covered)
                   / max(1, sum(d["n"] for d in covered))) if covered else None,
        "esc_rate": (sum(d["esc_rate"] * d["n"] for d in covered)
                     / max(1, sum(d["n"] for d in covered))) if covered else None,
        "cert_rate": (sum(d["cert_rate"] * d["cert_checked"] for d in covered if d["cert_rate"])
                      / max(1, sum(d["cert_checked"] for d in covered if d["cert_rate"])))
        if any(d["cert_rate"] for d in covered) else None,
        "n_total": sum(d["n"] for d in covered),
        "caveat": ("**本汇总混入了两种口径, 不可当成单一分数**: code 域报的是"
                   "「可逆注入」主口径 (rate_full_injection 另列于该域), agent 域只有 4 个"
                   "同构任务 (证据强度弱)。逐域读数请以 domains 表与各域 note 为准"
                   "—— 审计员 F 指出汇总值有虚高之嫌, 故在此显式声明"),
        "domains_covered": [d["domain"] for d in covered],
        "domains_not_covered": [d["domain"] for d in domains if not d.get("covered", True)],
    }
    return {"config": {"math_n": mn, "code_n": cn, "logic_n": ln, "seed": sd},
            "domains": domains, "aggregate": agg,
            "public_benchmarks": "**未引入** (容器内无公开基准离线快照; 全部题目为本地合成, "
                                 "题集指纹随报告落盘可复现)",
            "eval_only": "本 harness 不参与训练/注册, 仅读取算子做测量"}
