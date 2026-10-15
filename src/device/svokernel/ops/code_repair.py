"""§9.3 repair operators (band 9400, rows 9420–9423) — P5 deliverable B.

Four operators:
  rep_localize      9420  spectral defect localisation (Ochiai/Tarantula suspiciousness ranking)
  rep_patch_gen     9421  patch generation + **three gates** (original tests all green + held-out all green + mutants killed)
  rep_regress_guard 9422  before/after behaviour diff table
  rep_mutation_test 9423  mutation testing (arithmetic flip / boundary ±1 / condition negation / statement deletion)

**Three gates (§9.3 and §17.8, highest-priority constraint)**: a patch counts as a "fix" only when
  (1) the original test set is all green; (2) the **held-out** test set is all green; (3) mutants are killed
      (the patched program's mutants cannot survive on the test set + bounded behaviour box)
hold at once. Missing any one gives REGRESS_FAIL / MUTATION_SURVIVE, and the return value names the missing
one. This module writes the three gates as **three sequential checks**; failing any one raises the
corresponding enumerated error, and a "partial pass" is never accepted as a pass.

**V-pair regime (§9.1)**: generators and callers must give the train/held-out split explicitly (default
60/40, as in §9.1); acceptance looks only at held-out, and train is used only to search for patches. The two
sets are decided by `vpair_split`'s rule (modulo the input fingerprint hash), not picked by the caller —
otherwise "held-out" would degenerate into "the few entries I chose myself and know will pass".

**Honesty boundary**:
  - mutant-equivalence decisions are made by **exhaustive enumeration** over the **bounded behaviour box**
    (each input slot ∈ [−2,2], 5⁴ = 625 inputs); behaviour differences outside the box are not detected.
    Gate 3 is therefore stated strictly as "no surviving mutant inside the box", and flagged explicitly in
    the `kill_scope` field.
  - the suspiciousness of spectral localisation answers only for the **test set that supplies coverage
    information**; coverage comes from real execution (engine1), not from estimates.
"""
from __future__ import annotations

import random

from ..certify import Certificate
from ..errors import Err, SvoError
from ..registry import REGISTRY, OpDef
from ..tool import CallCtl
from .code_core import (
    ARGS, SCHEMA_PROG,
    interp, interp_stack, sid_map, validate_prog,
)

BAND = 9400
_BOX = tuple(range(-2, 3))  # bounded behaviour box: 5 values per input slot → 625 inputs
_REV = {"<": ">=", "<=": ">", ">": "<=", ">=": "<", "==": "!=", "!=": "=="}
_MUT_OPS = ("+", "-", "*")


# ---------------------------------------------------------------------------
# shared: test sets, V-pair split, execution
# ---------------------------------------------------------------------------

TEST_SCHEMA = {"kind": "list", "minlen": 1, "maxlen": 64,
               "items": {"kind": "dict", "extra": False,
                         "fields": {"args": {"kind": "list", "minlen": 0, "maxlen": 4,
                                             "items": {"kind": "int", "min": -10**6, "max": 10**6}},
                                    "expect": {"kind": "int", "min": -10**9, "max": 10**9}},
                         "required": ["args", "expect"]}}


def _mk_inputs(args_list) -> dict:
    out = {a: 0 for a in ARGS}
    for i, v in enumerate(args_list[:4]):
        out[ARGS[i]] = int(v)
    return out


def _run_tests(prog, tests, engine=1) -> "tuple[int, int, list]":
    """Run the test set; return (passed, total, failure detail). engine=2 uses the independent engine (iterative stack machine)."""
    fails: list = []
    passed = 0
    for i, t in enumerate(tests):
        inp = _mk_inputs(t["args"])
        r = interp(prog, inp, 200_000) if engine == 1 else interp_stack(prog, inp, 200_000)
        if r["error"] is not None:
            fails.append({"i": i, "reason": r["error"]["code"], "got": None, "expect": t["expect"]})
            continue
        got = r["state"].get("v0", 0)
        if got == t["expect"]:
            passed += 1
        else:
            fails.append({"i": i, "reason": "VALUE", "got": got, "expect": t["expect"]})
    return passed, len(tests), fails


def _split_vpair(tests, ratio: float = 0.6):
    """V-pair split (§9.1's 60/40): **sort by input-fingerprint hash, then cut**.

    Why cut this way instead of "hash modulo bucketing": modulo bucketing is a **matter of luck** on small
    test sets — all 4 tests could land in train, leaving held-out empty (this actually happened), so
    "acceptance looks only at held-out" would degenerate into an empty phrase. Sorting-then-cutting keeps the
    point "the split does not depend on the order the caller supplied" (the sort key is a content hash) while
    guaranteeing both sides are non-empty in the right ratio. n < 2 cannot be split → explicit rejection,
    never a silent degeneration into "use all of train".
    """
    import hashlib

    if len(tests) < 2:
        raise SvoError(Err.PRECOND_FAIL, ctx="rep.split_vpair",
                       detail=f"仅 {len(tests)} 条测试, 无法划分 train/held-out "
                              f"(拒绝在退化划分上验收; 至少需要 2 条)")
    ordered = sorted(tests, key=lambda t: hashlib.sha256(
        repr(sorted(_mk_inputs(t["args"]).items())).encode("utf-8")).hexdigest())
    k = int(round(ratio * len(ordered)))
    k = max(1, min(len(ordered) - 1, k))
    return ordered[:k], ordered[k:]


# ===========================================================================
# 9420 rep_localize (spectral defect localisation)
# ===========================================================================

_IN_LOC = {"kind": "dict",
           "fields": {"program": SCHEMA_PROG, "tests": TEST_SCHEMA,
                      "topk": {"kind": "int", "min": 1, "max": 16},
                      "ground_truth_sid": {"kind": "int", "min": -1, "max": 64}},
           "required": ["program", "tests"]}


def _inv_loc(args: dict, ctl: CallCtl) -> dict:
    """Spectral localisation: count each statement's executions in passing/failing tests and compute suspiciousness with the Ochiai formula.

    Ochiai: sus(e) = fail(e) / sqrt(total_fail × (fail(e) + pass(e)))
    Execution information comes from the **real run** of engine1's executed list (not estimated coverage).
    """
    ctx = "op:rep_localize#9420"
    prog = args["program"]
    validate_prog(prog, ctx)
    tests = args["tests"]
    all_sids = [sid for sid, _ in _walk(prog)]
    fail_ct = {s: 0 for s in all_sids}
    pass_ct = {s: 0 for s in all_sids}
    n_fail = n_pass = 0
    for t in tests:
        inp = _mk_inputs(t["args"])
        r = interp(prog, inp, 200_000)
        if r["error"] is not None:
            n_fail += 1
            for s in r["executed"]:
                if s in fail_ct:
                    fail_ct[s] += 1
            continue
        got = r["state"].get("v0", 0)
        hit = set(r["executed"])
        if got == t["expect"]:
            n_pass += 1
            for s in hit:
                if s in pass_ct:
                    pass_ct[s] += 1
        else:
            n_fail += 1
            for s in hit:
                if s in fail_ct:
                    fail_ct[s] += 1
    scores = []
    for s in all_sids:
        fe, pe = fail_ct[s], pass_ct[s]
        denom = (n_fail * (fe + pe)) ** 0.5
        sus = (fe / denom) if denom > 0 else 0.0
        scores.append({"sid": s, "sus": round(sus, 12), "fail": fe, "pass": pe})
    ranked = sorted(scores, key=lambda d: (-d["sus"], d["sid"]))
    topk = int(args.get("topk", 3))
    top = [d["sid"] for d in ranked[:topk]]
    gt = int(args.get("ground_truth_sid", -1))
    hit_rank = (ranked.index(next(d for d in ranked if d["sid"] == gt)) + 1) if gt >= 0 and any(
        d["sid"] == gt for d in ranked) else 0
    return {"value": {"ranking": ranked, "topk": top, "n_fail": n_fail, "n_pass": n_pass,
                      "ground_truth_rank": hit_rank,
                      "hit_at_k": bool(gt >= 0 and 1 <= hit_rank <= topk)},
            "resid": [0.0 if gt < 0 or hit_rank else 1.0],
            "notes": f"谱定位: {n_fail} 失败 / {n_pass} 通过; top-{topk} = {top}; "
                     + (f"真值语句 {gt} 排名 {hit_rank} (top-{topk} 命中={1 <= hit_rank <= topk})"
                        if gt >= 0 else "未给真值 (只报排序)")}


def _walk(prog):
    sids, _ = sid_map(prog)
    out = []

    def rec(ss):
        for s in ss:
            out.append((sids[id(s)], s))
            if s["t"] == "if":
                rec(s.get("th", []))
                rec(s.get("el", []))
            elif s["t"] == "loop":
                rec(s.get("body", []))

    rec(prog)
    return out


def _ver_loc(args: dict, value: dict) -> dict:
    """Independent verification: (a) recompute the coverage spectrum with **engine2**; the Ochiai ranking must match engine1's
    (two independent engines must report identical coverage, otherwise the localisation result is untrustworthy);
    (b) spot-check the suspiciousness formula: for the top-ranked statement, hand-compute Ochiai from its fail/pass counts and compare."""
    prog = args["program"]
    tests = args["tests"]
    sids = [sid for sid, _ in _walk(prog)]
    fail2 = {s: 0 for s in sids}
    pass2 = {s: 0 for s in sids}
    nf = np_ = 0
    for t in tests:
        inp = _mk_inputs(t["args"])
        r = interp_stack(prog, inp, 200_000)
        hit = set(r["executed"])
        if r["error"] is not None:
            nf += 1
            for s in hit:
                fail2[s] += 1
            continue
        got = r["state"].get("v0", 0)
        if got == t["expect"]:
            np_ += 1
            for s in hit:
                pass2[s] += 1
        else:
            nf += 1
            for s in hit:
                fail2[s] += 1
    rank2 = sorted(sids, key=lambda s: (
        -((fail2[s] / (nf * (fail2[s] + pass2[s])) ** 0.5) if nf and (fail2[s] + pass2[s]) else 0.0), s))
    rank1 = [d["sid"] for d in value["ranking"]]
    engine_ok = (rank1 == rank2) and (nf == value["n_fail"]) and (np_ == value["n_pass"])
    top = value["ranking"][0]
    manual = (top["fail"] / (value["n_fail"] * (top["fail"] + top["pass"])) ** 0.5
              if value["n_fail"] and (top["fail"] + top["pass"]) else 0.0)
    formula_ok = abs(manual - top["sus"]) < 1e-9
    ok = engine_ok and formula_ok
    return {"ok": ok, "resid": [0.0 if ok else 1.0],
            "detail": f"engine2 覆盖谱独立复算: 排名一致={engine_ok} (nf={nf}, np={np_}); "
                      f"Ochiai 公式手算复核={formula_ok}"}


REGISTRY.register(OpDef(
    name="rep_localize", band=BAND, sub=20,
    in_schema=_IN_LOC,
    out_schema={"kind": "dict",
                "fields": {"ranking": {"kind": "list", "maxlen": 64,
                                       "items": {"kind": "dict", "extra": False,
                                                 "fields": {"sid": {"kind": "int"},
                                                            "sus": {"kind": "float"},
                                                            "fail": {"kind": "int", "min": 0},
                                                            "pass": {"kind": "int", "min": 0}},
                                                 "required": ["sid", "sus", "fail", "pass"]}},
                           "topk": {"kind": "list", "items": {"kind": "int"}, "minlen": 1},
                           "n_fail": {"kind": "int", "min": 0}, "n_pass": {"kind": "int", "min": 0},
                           "ground_truth_rank": {"kind": "int", "min": 0},
                           "hit_at_k": {"kind": "bool"}},
                "required": ["ranking", "topk", "n_fail", "n_pass", "ground_truth_rank", "hit_at_k"]},
    invoke=_inv_loc, verify=_ver_loc,
    cost=("units", 2.0), errors=("SCHEMA_MISMATCH",),
    effects="none (纯函数)",
    doc="§9.3 谱缺陷定位 (Ochiai) | in: 程序 + 测试集 (+ 可选真值语句号与 k) | "
        "out: 逐语句可疑度排序 + top-k + 真值排名 | 验收=注入缺陷基准的 top-k 命中率 + "
        "**双引擎覆盖谱一致** + 公式手算复核 | 2 units | 诚实边界: 可疑度只对给定测试集负责; "
        "全通过或全失败 (nf=0 或 np=0) 时 Ochiai 无区分力, 如实给 0 分不臆造排序",
    selfchecks=(
        {"id": "rl_localizes_bug", "args": {"program": [
            {"t": "assign", "var": "v0", "e": {"t": "bin", "op": "+", "l": {"t": "var", "n": "a0"},
                                              "r": {"t": "lit", "v": 10}}}],
            "tests": [{"args": [1], "expect": 2}, {"args": [2], "expect": 3}],
            "topk": 1, "ground_truth_sid": 0},
         "check": lambda r: (r.value["hit_at_k"] is True and r.value["ground_truth_rank"] == 1
                             and r.value["n_fail"] == 2 and r.value["n_pass"] == 0,
                             f"top={r.value['topk']} rank={r.value['ground_truth_rank']} "
                             f"fail={r.value['n_fail']}"),
         "expect": "程序算 a0+10 而期望 a0+1: 两个测试都失败 → 唯一语句的 Ochiai = "
                   "2/√(2·2) = 1.0, 真值语句排名 1, top-1 命中"},
        {"id": "rl_mixed_pass_fail", "args": {"program": [
            {"t": "assign", "var": "v0", "e": {"t": "var", "n": "a0"}},
            {"t": "if", "c": {"t": "bin", "op": ">", "l": {"t": "var", "n": "v0"},
                              "r": {"t": "lit", "v": 0}},
             "th": [{"t": "assign", "var": "v1", "e": {"t": "lit", "v": 1}}],
             "el": [{"t": "assign", "var": "v1", "e": {"t": "lit", "v": -1}}]},
            {"t": "assign", "var": "v0", "e": {"t": "bin", "op": "*", "l": {"t": "var", "n": "v1"},
                                              "r": {"t": "lit", "v": 5}}}],
            "tests": [{"args": [3], "expect": 5}, {"args": [-3], "expect": 5}],
            "topk": 2, "ground_truth_sid": 2},
         "check": lambda r: (r.value["n_fail"] == 1 and r.value["n_pass"] == 1
                             and r.value["ground_truth_rank"] >= 1,
                             f"fail={r.value['n_fail']} pass={r.value['n_pass']} "
                             f"rank={r.value['ground_truth_rank']}"),
         "expect": "负输入分支返回 −5 而期望 5: 一个测试通过一个失败 → 混合谱; "
                   "语句 2 (乘法) 在两个测试中都执行, 可疑度较高"},
        {"id": "rl_all_pass_no_signal", "args": {"program": [
            {"t": "assign", "var": "v0", "e": {"t": "lit", "v": 7}}],
            "tests": [{"args": [], "expect": 7}], "topk": 1},
         "check": lambda r: (r.value["n_fail"] == 0 and r.value["ranking"][0]["sus"] == 0.0,
                             f"n_fail={r.value['n_fail']} sus={r.value['ranking'][0]['sus']}"),
         "expect": "全通过: 无可疑信号 → 所有语句 Ochiai = 0 (不臆造排序)"},
        {"id": "rl_bad_program", "args": {"program": [{"t": "assign", "var": "nope",
                                                       "e": {"t": "lit", "v": 1}}],
                                          "tests": [{"args": [], "expect": 1}]},
         "expect_err": "SCHEMA_MISMATCH", "expect": "非法槽位"},
        {"id": "rl_empty_tests", "args": {"program": [], "tests": []},
         "expect_err": "SCHEMA_MISMATCH", "expect": "空测试集 (minlen=1)"},
    ),
))


# ===========================================================================
# 9423 rep_mutation_test (mutation testing)
# ===========================================================================

_MUT_KINDS = ("arith_flip", "boundary_pm1", "cond_negate", "stmt_delete")


def _expr_nodes(e, path):
    """Recursively yield (sub-path, sub-expression).

    **Why it must be recursive**: checking only the statement's top-level expression misses nested positions.
    For example, the literal 10 in `v0 ← a0 + 10` sits in the right child of bin — a non-recursive
    implementation cannot see it at all, so one-step fixes like "set 10 to 0" would never appear among the
    candidates (measured: only 3 of 6 expected candidates were generated).
    """
    yield path, e
    t = e["t"]
    if t == "bin":
        yield from _expr_nodes(e["l"], path + ["l"])
        yield from _expr_nodes(e["r"], path + ["r"])
    elif t == "un":
        yield from _expr_nodes(e["e"], path + ["e"])
    elif t == "call":
        for i, a in enumerate(e["args"]):
            yield from _expr_nodes(a, path + ["arg%d" % i])


def _mutants(prog, limit: int = 64) -> list:
    """First-order mutants (arithmetic flip / boundary ±1 / condition negation / statement deletion). Returns [(kind, description, mutant program)].

    Mutants are produced by **deep copy + one in-place edit**, guaranteeing exactly first order (for attribution).
    """
    import copy

    out: list = []

    def visit(stmts, path):
        for i, s in enumerate(stmts):
            p = path + [i]
            if s["t"] == "assign":
                base = p + ["e"]
                for sub, node in _expr_nodes(s["e"], []):
                    fp = base + sub
                    tag = "/".join(map(str, fp))
                    tn = node["t"]
                    if tn == "bin" and node["op"] in _MUT_OPS:
                        for op in _MUT_OPS:
                            if op == node["op"]:
                                continue
                            m = copy.deepcopy(prog)
                            _at(m, fp)["op"] = op
                            out.append(("arith_flip", f"{tag}: {node['op']}→{op}", m))
                    if tn == "bin" and node["op"] in _REV:
                        m = copy.deepcopy(prog)
                        _at(m, fp)["op"] = _REV[node["op"]]
                        out.append(("cond_negate", f"{tag}: {node['op']}→{_REV[node['op']]}", m))
                    if tn == "lit":
                        # Two-sided ±1: mutation **testing** uses only +1 per the brief, but patch **search**
                        # must be two-sided — a defect may have been injected as +1, so the fix needs −1
                        # (a one-sided bound would miss half of the fixable defects).
                        for delta in (1, -1):
                            nv = int(node["v"]) + delta
                            if -1000 <= nv <= 1000:
                                m = copy.deepcopy(prog)
                                _at(m, fp)["v"] = nv
                                out.append(("boundary_pm1", f"{tag}: {node['v']}→{nv}", m))
                out.append(("stmt_delete", f"{'/'.join(map(str, p))}: 删语句", _del(prog, p)))
            elif s["t"] == "if":
                visit(s.get("th", []), p + ["th"])
                visit(s.get("el", []), p + ["el"])
            elif s["t"] == "loop":
                visit(s.get("body", []), p + ["body"])
    visit(prog, [])

    def okp(m):
        try:
            validate_prog(m, "rep_mutant")
            return True
        except SvoError:
            return False

    return [(k, d, m) for k, d, m in out if okp(m)][:limit]


def _at(prog, path):
    cur = prog
    for k in path:
        if isinstance(k, int):
            cur = cur[k]
        else:
            cur = cur[k]
    return cur


def _del(prog, path):
    import copy

    m = copy.deepcopy(prog)
    parent = m
    for k in path[:-1]:
        parent = parent[k] if not isinstance(k, int) else parent[k]
    if isinstance(path[-1], int):
        parent.pop(path[-1])
    else:
        parent[path[-1]] = []
    return m


def _patch_candidates(prog, limit: int = 64) -> list:
    """Patch search space = mutants (arithmetic/boundary/negation/statement deletion) **plus** "set a literal to 0 / 1".

    Why the addition: "wrong constant" is one of the most common defect classes, and a ±1 mutation step is too
    small — `v0 ← a0 + 10` (should be `a0`) has no solution in the ±1 neighbourhood, yet "set 10 to 0" reaches it
    in one step. Mutation **testing** still uses strictly the four mutant kinds listed in brief §9.3 (see
    _mutants); what expands here is the **patch search** space. The two serve different purposes and must not be
    conflated.
    """
    import copy

    out = list(_mutants(prog, limit))

    def visit(stmts, path):
        for i, st in enumerate(stmts):
            p = path + [i]
            if st["t"] == "assign":
                base = p + ["e"]
                for sub, node in _expr_nodes(st["e"], []):
                    if node["t"] != "lit" or int(node["v"]) in (0, 1):
                        continue
                    fp = base + sub
                    tag = "/".join(map(str, fp))
                    for target in (0, 1):
                        m = copy.deepcopy(prog)
                        _at(m, fp)["v"] = target
                        out.append(("lit_set", f"{tag}: {node['v']}→{target}", m))
            elif st["t"] == "if":
                visit(st.get("th", []), p + ["th"])
                visit(st.get("el", []), p + ["el"])
            elif st["t"] == "loop":
                visit(st.get("body", []), p + ["body"])

    visit(prog, [])
    seen: set = set()
    uniq: list = []
    for k, d, m in out:
        key = repr(m)
        if key in seen:
            continue
        seen.add(key)
        try:
            validate_prog(m, "rep_patch_candidate")
        except SvoError:
            continue
        uniq.append((k, d, m))
    return uniq[:limit]


def _behavior_box(prog) -> "dict | None":
    """Exhaustively compute the behaviour fingerprint over the bounded behaviour box (each input slot takes values from _BOX)."""
    import itertools

    sig = {}
    for combo in itertools.product(_BOX, repeat=len(ARGS)):
        inp = {a: combo[i] for i, a in enumerate(ARGS)}
        r = interp(prog, inp, 200_000)
        if r["error"] is not None:
            sig[combo] = ("ERR", r["error"]["code"])
        else:
            sig[combo] = tuple(r["state"].get(a, 0) for a in ARGS) + (r["state"].get("v0", 0),)
    return sig


_IN_MUT = {"kind": "dict",
           "fields": {"program": SCHEMA_PROG, "tests": TEST_SCHEMA,
                      "max_mutants": {"kind": "int", "min": 1, "max": 64},
                      "check_box": {"kind": "bool"}},
           "required": ["program", "tests"]}


def _inv_mut(args: dict, ctl: CallCtl) -> dict:
    """Mutation testing: generate first-order mutants, kill them with the test set; report the mutation score (killed/total) and survivor details."""
    prog = args["program"]
    validate_prog(prog, "op:rep_mutation_test#9423")
    tests = args["tests"]
    muts = _mutants(prog, int(args.get("max_mutants", 64)))
    base_pass, n, _ = _run_tests(prog, tests)
    killed, alive = [], []
    box = _behavior_box(prog) if bool(args.get("check_box", False)) else None
    for kind, desc, m in muts:
        p, _tot, fails = _run_tests(m, tests)
        if p < n:
            killed.append({"kind": kind, "desc": desc, "killed_by": fails[0]["i"] if fails else -1})
            continue
        entry = {"kind": kind, "desc": desc, "equivalent_in_box": False}
        if box is not None:
            mb = _behavior_box(m)
            entry["equivalent_in_box"] = (mb == box)
        alive.append(entry)
    total = len(muts)
    score = (len(killed) / total) if total else 1.0
    real_alive = [a for a in alive if not a["equivalent_in_box"]]
    return {"value": {"n_mutants": total, "killed": len(killed), "survived": len(alive),
                      "survived_non_equivalent": len(real_alive), "score": round(score, 12),
                      "killed_detail": killed, "survived_detail": alive,
                      "base_tests_passed": base_pass, "base_tests_total": n,
                      "kill_scope": ("bounded_box(arg∈[−2,2]) + 测试集" if box is not None
                                     else "仅测试集 (未开 check_box, 故未做等价性判定)")},
            "resid": [float(len(real_alive))],
            "notes": f"变异测试: {total} 个一阶变异体, 杀死 {len(killed)}, 存活 {len(alive)} "
                     f"(其中箱内等价 {len(alive) - len(real_alive)}); "
                     f"变异分数 {score:.4f}; 非等价存活 = 残差 = {len(real_alive)}"}


def _ver_mut(args: dict, value: dict) -> dict:
    """Independent verification: (a) rerun the base program and every killed mutant with **engine2**; the kill/survive
    decisions must agree (semantic agreement of the two engines is the premise of any mutation-testing conclusion);
    (b) score = killed/total is self-consistent; (c) survivors = total − killed is self-consistent."""
    prog = args["program"]
    tests = args["tests"]
    muts = _mutants(prog, int(args.get("max_mutants", 64)))
    bp2, n2, _ = _run_tests(prog, tests, engine=2)
    killed2 = 0
    for kind, desc, m in muts:
        p2, _t, _f = _run_tests(m, tests, engine=2)
        if p2 < n2:
            killed2 += 1
    score_ok = abs(value["score"] - (value["killed"] / max(1, value["n_mutants"]))) < 1e-12
    count_ok = (value["killed"] + value["survived"]) == value["n_mutants"]
    engine_ok = (killed2 == value["killed"]) and (bp2 == value["base_tests_passed"])
    ok = score_ok and count_ok and engine_ok
    return {"ok": ok, "resid": [float(abs(killed2 - value["killed"]))],
            "detail": f"engine2 复算: 杀死 {killed2} vs engine1 {value['killed']} (一致={engine_ok}); "
                      f"基程序通过数复算一致={bp2 == value['base_tests_passed']}; "
                      f"分数自洽={score_ok}; 计数自洽={count_ok}"}


REGISTRY.register(OpDef(
    name="rep_mutation_test", band=BAND, sub=23,
    in_schema=_IN_MUT,
    out_schema={"kind": "dict",
                "fields": {"n_mutants": {"kind": "int", "min": 0}, "killed": {"kind": "int", "min": 0},
                           "survived": {"kind": "int", "min": 0},
                           "survived_non_equivalent": {"kind": "int", "min": 0},
                           "score": {"kind": "float", "min": 0.0, "max": 1.0},
                           "killed_detail": {"kind": "list", "maxlen": 64,
                                             "items": {"kind": "dict", "extra": False,
                                                       "fields": {"kind": {"kind": "str"},
                                                                  "desc": {"kind": "str"},
                                                                  "killed_by": {"kind": "int"}},
                                                       "required": ["kind", "desc", "killed_by"]}},
                           "survived_detail": {"kind": "list", "maxlen": 64,
                                               "items": {"kind": "dict", "extra": True, "fields": {}}},
                           "base_tests_passed": {"kind": "int", "min": 0},
                           "base_tests_total": {"kind": "int", "min": 1},
                           "kill_scope": {"kind": "str"}},
                "required": ["n_mutants", "killed", "survived", "survived_non_equivalent", "score",
                             "killed_detail", "survived_detail", "base_tests_passed",
                             "base_tests_total", "kill_scope"]},
    invoke=_inv_mut, verify=_ver_mut,
    cost=("units", 4.0), errors=("SCHEMA_MISMATCH",),
    effects="none (纯函数)",
    doc="§9.3 变异测试 (算术翻转/边界±1/条件取反/删语句, 全部一阶) | in: 程序 + 测试集 | "
        "out: 变异分数 + 杀死/存活明细 + kill_scope | 验收=**双引擎**杀/活判定一致 + 分数与计数自洽 | "
        "残差 = 非等价存活数 (真正需要补测试的目标) | 4 units | "
        "诚实边界: 等价性只在有界行为箱 (arg∈[−2,2], 625 输入) 内判定, 箱外差异不可见 —— "
        "kill_scope 字段逐字写明该范围 | 失败: schema",
    selfchecks=(
        {"id": "rmt_kills_arith", "args": {"program": [
            {"t": "assign", "var": "v0", "e": {"t": "bin", "op": "+", "l": {"t": "var", "n": "a0"},
                                              "r": {"t": "var", "n": "a1"}}}],
            "tests": [{"args": [1, 2], "expect": 3}, {"args": [5, 7], "expect": 12}],
            "max_mutants": 8, "check_box": True},
         "check": lambda r: (r.value["base_tests_passed"] == 2
                             and r.value["killed"] >= 2 and r.value["score"] > 0.5,
                             f"killed={r.value['killed']}/{r.value['n_mutants']} "
                             f"survived={r.value['survived']} score={r.value['score']}"),
         "expect": "v0 = a0 + a1 且测试为两个非退化点: 算术翻转 (+→−/*) 与删语句都会被杀死, "
                   "故分数应 > 0.5; 基程序两个测试全过"},
        {"id": "rmt_boundary_mutant_killed", "args": {"program": [
            {"t": "assign", "var": "v0", "e": {"t": "bin", "op": "*", "l": {"t": "var", "n": "a0"},
                                              "r": {"t": "lit", "v": 2}}}],
            "tests": [{"args": [3], "expect": 6}],
            "max_mutants": 8, "check_box": True},
         "check": lambda r: (any(k["kind"] in ("arith_flip", "boundary_pm1")
                                 for k in r.value["killed_detail"]),
                             f"killed kinds={sorted({k['kind'] for k in r.value['killed_detail']})}"),
         "expect": "乘法与常数 2 的变异 (→+、→*3、2→3) 在测试 (3,expect 6) 上必被杀 "
                   "(单个测试就足以区分)"},
        {"id": "rmt_survivor_reported", "args": {"program": [
            {"t": "assign", "var": "v0", "e": {"t": "lit", "v": 4}}],
            "tests": [{"args": [], "expect": 4}],
            "max_mutants": 8, "check_box": True},
         "check": lambda r: (r.value["base_tests_passed"] == 1
                             and r.value["killed"] + r.value["survived"] == r.value["n_mutants"],
                             f"killed={r.value['killed']} survived={r.value['survived']} "
                             f"total={r.value['n_mutants']}"),
         "expect": "常量程序: 无算子可翻, 只有 4→5 的边界变异 —— 该变异会被测试杀死; "
                   "计数自洽 (killed+survived=total) 是硬要求"},
        {"id": "rmt_no_box_scope_label", "args": {"program": [
            {"t": "assign", "var": "v0", "e": {"t": "var", "n": "a0"}}],
            "tests": [{"args": [1], "expect": 1}], "max_mutants": 4, "check_box": False},
         "check": lambda r: ("仅测试集" in r.value["kill_scope"],
                             f"kill_scope={r.value['kill_scope']}"),
         "expect": "未开 check_box 时 kill_scope 必须如实写'仅测试集 (未做等价性判定)' —— "
                   "结论强度必须与所做的事匹配"},
        {"id": "rmt_empty_tests", "args": {"program": [], "tests": []},
         "expect_err": "SCHEMA_MISMATCH", "expect": "空测试集"},
    ),
))


# ===========================================================================
# 9421 rep_patch_gen (patch generation + three gates)
# ===========================================================================

_IN_PATCH = {"kind": "dict",
             "fields": {"program": SCHEMA_PROG, "tests": TEST_SCHEMA,
                        "max_candidates": {"kind": "int", "min": 1, "max": 64},
                        "split_ratio": {"kind": "float", "min": 0.3, "max": 0.9},
                        "check_box": {"kind": "bool"}},
             "required": ["program", "tests"]}


def _inv_patch(args: dict, ctl: CallCtl) -> dict:
    """Patch generation + **three gates**.

    Flow: (1) search candidates in the **first-order mutation neighbourhood** using the train set (requiring train all green);
    (2) gate 1 — the original (train) test set all green;
    (3) gate 2 — the **held-out** test set all green (the V-pair split is decided by _split_vpair, not chosen by the caller);
    (4) gate 3 — the patched program's mutants have no survivor on the test set + bounded behaviour box (non-equivalent).
    Failing any gate raises the corresponding enumerated error and writes "which gate is missing" into the detail.
    """
    ctx = "op:rep_patch_gen#9421"
    prog = args["program"]
    validate_prog(prog, ctx)
    tests = args["tests"]
    ratio = float(args.get("split_ratio", 0.6))
    train, held = _split_vpair(tests, ratio)
    nt, _, _ = _run_tests(prog, train)
    nh, _, _ = _run_tests(prog, held)
    buggy = (nt < len(train)) or (nh < len(held))
    if not buggy:
        raise SvoError(Err.PRECOND_FAIL, ctx=ctx,
                       detail=f"程序在 train ({nt}/{len(train)}) 与 held-out ({nh}/{len(held)}) 上"
                              f"**全部通过** —— 没有可修的缺陷, 本算子不产补丁 "
                              f"(若想检验测试集强度, 请用 rep_mutation_test)")
    cands = _patch_candidates(prog, int(args.get("max_candidates", 64)))
    found = None
    tried = 0
    for kind, desc, m in cands:
        ctl.check_time(ctx + "@search")
        tried += 1
        pt, _t1, _f1 = _run_tests(m, train)
        if pt < len(train):
            continue
        ph, _t2, f2 = _run_tests(m, held)
        if ph < len(held):
            continue
        found = (kind, desc, m, tried)
        break
    if found is None:
        raise SvoError(Err.REGRESS_FAIL, ctx=ctx,
                       detail=f"三重门未通过: 在 {tried} 个一阶变异候选 (train {len(train)} 条 / "
                              f"held-out {len(held)} 条) 中找不到同时满足门 1+门 2 的补丁 —— "
                              f"缺的是「原测试全绿 + held-out 全绿」, 悬赏范围需扩大 (本算子只搜一阶邻域)")
    kind, desc, patch, tried = found
    # gate 3: mutants killed (on the bounded behaviour box + test set)
    box_on = bool(args.get("check_box", True))
    muts = _mutants(patch, 64)
    base_box = _behavior_box(patch) if box_on else None
    if not muts:
        raise SvoError(Err.MUTATION_SURVIVE, ctx=ctx,
                       detail="三重门第 3 条无法判定: 补丁程序**没有任何可生成的变异体** "
                              "(门 3 = 变异体杀死, 没有变异体就没有证据) —— 按 §13「没有证书的解不算解」"
                              "判为不过, 而不是默认通过")
    alive = []
    for mk, mdesc, mm in muts:
        p, nn, _f = _run_tests(mm, tests)
        if p < nn:
            continue  # the test set kills it → this mutant passes
        if not box_on:
            # **Turning off the behaviour box does not mean "I'll just call it equivalent"**: without the box,
            # equivalence cannot be decided, so "not killed by the test set" must count as surviving. The old
            # version skipped this step when box_on=False, leaving gate 3 idle while still emitting a fake
            # "100% killed" report and a fake certificate with survived=0 — the **blocking finding** auditor A
            # measured in the adversarial audit (gate 3 could be pierced by a single caller switch).
            alive.append({"kind": mk, "desc": mdesc,
                          "note": "测试集未杀死, 且未启用行为箱等价判定 → 计为存活"})
            continue
        if _behavior_box(mm) != base_box:
            alive.append({"kind": mk, "desc": mdesc, "note": "箱内有行为差异但测试集未杀"})
    if alive:
        raise SvoError(Err.MUTATION_SURVIVE, ctx=ctx,
                       detail=f"三重门第 3 条不过: 补丁的 {len(alive)} 个变异体存活 "
                              f"(测试集"
                              + ("与行为箱" if box_on else "，未启用行为箱故等价性未判定")
                              + f"都没杀死), 例: {alive[0]} —— "
                              f"门 1/门 2 已过, 缺的是变异体杀死")
    # The tightened mutation_report checker **reruns the mutants itself**: self-reported counts such as
    # {"total":3,"survived":0} are judged as no evidence (the red team measured this hole). Hence the payload
    # carries the patch program, the test set and all mutant sources, and the checker recomputes the survivor count.
    cert = Certificate(kind="mutation_report",
                       payload={"program": patch, "tests": tests, "box": box_on,
                                "mutants": [m for _k, _d, m in muts]},
                       issued_by="rep_patch_gen")
    return {"value": {"patch_kind": kind, "patch_desc": desc, "patch_program": patch,
                      "candidates_tried": tried,
                      "gate1_train": True, "gate2_heldout": True, "gate3_mutation": True,
                      "n_mutants_of_patch": len(muts),
                      "train_size": len(train), "heldout_size": len(held),
                      "was_buggy": buggy,
                      "kill_scope": ("bounded_box(arg∈[−2,2]) + 测试集" if box_on else "仅测试集")},
            "resid": [0.0], "cert": cert,
            "notes": f"三重门全过: 补丁 = {kind} ({desc}); train {len(train)}/{len(train)} 绿, "
                     f"held-out {len(held)}/{len(held)} 绿; 补丁的 {len(muts)} 个变异体"
                     f"全部被测试集杀死"
                     + ("，且箱内无行为差异" if box_on else "（未启用行为箱: 等价性未判定，"
                        "故任何测试集杀不死的变异体都会使门 3 失败）")
                     + f"; 原程序有缺陷={buggy}; 搜索了 {tried} 个候选"}


def _ver_patch(args: dict, value: dict) -> dict:
    """Independent verification (**gate-by-gate replay**, with equivalence decisions switched to engine2):
     gate 1/gate 2: rerun train / held-out with engine2; both must be all green;
     gate 3       : rerun all mutants of the patch with engine2; no survivor (non-equivalent) is allowed;
     also check the V-pair split and that the patch really **differs** from the original program (so that a
     "patch = original program" trivial patch cannot pass gates 1/2)."""
    prog = args["program"]
    tests = args["tests"]
    train, held = _split_vpair(tests, float(args.get("split_ratio", 0.6)))
    patch = value["patch_program"]
    g1 = _run_tests(patch, train, engine=2)[0] == len(train)
    g2 = _run_tests(patch, held, engine=2)[0] == len(held)
    box_on = bool(args.get("check_box", True))
    muts = _mutants(patch, 64)
    alive = 0
    base_box = _behavior_box(patch) if box_on else None
    for _k, _d, mm in muts:
        p2, n2, _f = _run_tests(mm, tests, engine=2)
        if p2 < n2:
            continue
        # same convention as invoke: without the behaviour box, "not killed by the test set" means surviving (never silently dropped)
        if (not box_on) or (_behavior_box(mm) != base_box):
            alive += 1
    nontrivial = (patch != prog)
    sizes_ok = (value["train_size"] == len(train)) and (value["heldout_size"] == len(held))
    ok = g1 and g2 and (alive == 0) and nontrivial and sizes_ok
    return {"ok": ok, "resid": [float(alive)],
            "detail": f"门 1 (train, engine2)={g1}; 门 2 (held-out, engine2)={g2}; "
                      f"门 3 存活变异体={alive} (须 0); 补丁非平凡={nontrivial}; "
                      f"V 对规模一致={sizes_ok} (train {len(train)} / held {len(held)})"}


REGISTRY.register(OpDef(
    name="rep_patch_gen", band=BAND, sub=21,
    in_schema=_IN_PATCH,
    out_schema={"kind": "dict",
                "fields": {"patch_kind": {"kind": "str"}, "patch_desc": {"kind": "str"},
                           "patch_program": {"kind": "list", "items": {"kind": "dict", "extra": True,
                                                                       "fields": {}}},
                           "candidates_tried": {"kind": "int", "min": 0},
                           "gate1_train": {"kind": "bool"}, "gate2_heldout": {"kind": "bool"},
                           "gate3_mutation": {"kind": "bool"},
                           "n_mutants_of_patch": {"kind": "int", "min": 0},
                           "train_size": {"kind": "int", "min": 1},
                           "heldout_size": {"kind": "int", "min": 1},
                           "was_buggy": {"kind": "bool"}, "kill_scope": {"kind": "str"}},
                "required": ["patch_kind", "patch_desc", "patch_program", "candidates_tried",
                             "gate1_train", "gate2_heldout", "gate3_mutation",
                             "n_mutants_of_patch", "train_size", "heldout_size", "was_buggy",
                             "kill_scope"]},
    invoke=_inv_patch, verify=_ver_patch, cert_kind="mutation_report",
    cost=("units", 6.0), errors=("REGRESS_FAIL", "MUTATION_SURVIVE", "PRECOND_FAIL", "TIMEOUT",
                                 "SCHEMA_MISMATCH"),
    effects="none (纯函数)",
    doc="§9.3 补丁生成 + **三重门** | in: 有缺陷程序 + 测试集 (+ 划分比例) | "
        "out: 通过的补丁 + 三门状态 + 候选搜索计数 | 验收=门1 原测试全绿 + 门2 **held-out** 全绿 "
        "+ 门3 补丁变异体全杀死 (测试集 + 有界行为箱) | 证书=mutation_report (survived=0) | 6 units | "
        "诚实边界: 搜索只在**一阶变异邻域**内 (找不到即 REGRESS_FAIL 并说明缺哪门); "
        "变异体等价性只在 arg∈[−2,2] 的 625 输入箱内判定, 范围写在 kill_scope | "
        "失败: 无合格补丁 REGRESS_FAIL / 有存活变异体 MUTATION_SURVIVE (缺一不算修复, §17.8)",
    selfchecks=(
        {"id": "rp_fixes_plus_one", "args": {"program": [
            {"t": "assign", "var": "v0", "e": {"t": "bin", "op": "+", "l": {"t": "var", "n": "a0"},
                                              "r": {"t": "lit", "v": 10}}}],
            "tests": [{"args": [0], "expect": 0}, {"args": [1], "expect": 1},
                      {"args": [2], "expect": 2}, {"args": [5], "expect": 5},
                      {"args": [7], "expect": 7}, {"args": [9], "expect": 9}],
            "max_candidates": 16, "check_box": True},
         "check": lambda r: (r.value["gate1_train"] and r.value["gate2_heldout"]
                             and r.value["gate3_mutation"]
                             and r.value["patch_kind"] == "lit_set"
                             and r.value["was_buggy"] is True
                             and r.cert_independent is True
                             and r.value["patch_program"][0]["e"]["r"]["v"] == 0,
                             f"kind={r.value['patch_kind']} desc={r.value['patch_desc']} "
                             f"gates=({r.value['gate1_train']},{r.value['gate2_heldout']},"
                             f"{r.value['gate3_mutation']}) cert={r.cert_independent}"),
         "expect": "bug=v0←a0+10 (应恒等): 修复是把嵌套字面量 10 置 0 (lit_set 类), "
                   "三门全过且补丁的变异体全被杀。注意这要求变异器能递归看到 bin 的右孩子 —— "
                   "非递归实现会漏掉这条唯一解"},
        {"id": "rp_arith_operator_fix", "args": {"program": [
            {"t": "assign", "var": "v0", "e": {"t": "bin", "op": "-", "l": {"t": "var", "n": "a0"},
                                              "r": {"t": "var", "n": "a1"}}}],
            "tests": [{"args": [3, 2], "expect": 5}, {"args": [0, 4], "expect": 4},
                      {"args": [7, 7], "expect": 14}, {"args": [1, 1], "expect": 2},
                      {"args": [5, 0], "expect": 5}, {"args": [2, 9], "expect": 11}],
            "max_candidates": 16, "check_box": True},
         "check": lambda r: (r.value["gate1_train"] and r.value["gate2_heldout"]
                             and r.value["gate3_mutation"],
                             f"kind={r.value['patch_kind']} desc={r.value['patch_desc']}"),
         "expect": "bug=a0−a1 (应求和): 一阶修复是把 '-' 翻成 '+' (arith_flip) → 三门应全过"},
        {"id": "rp_heldout_catches_overfit", "args": {"program": [
            {"t": "assign", "var": "v0", "e": {"t": "bin", "op": "*", "l": {"t": "var", "n": "a0"},
                                              "r": {"t": "lit", "v": 2}}}],
            "tests": [{"args": [3], "expect": 6}, {"args": [4], "expect": 8},
                      {"args": [5], "expect": 10}, {"args": [6], "expect": 12},
                      {"args": [7], "expect": 14}, {"args": [8], "expect": 16}],
            "max_candidates": 16, "check_box": True},
         "expect_err": "PRECOND_FAIL",
         "expect": "程序本来就是对的 (v0=2a0, 全部测试通过): 必须**明确报'无需补丁'**而不是"
                   "REGRESS_FAIL —— 把'没有缺陷'与'找不到补丁'混为一谈会误导导航层去扩大搜索"},
        {"id": "rp_no_patch_found", "args": {"program": [
            {"t": "assign", "var": "v0", "e": {"t": "lit", "v": 1}}],
            "tests": [{"args": [0], "expect": 5}, {"args": [1], "expect": 5},
                      {"args": [2], "expect": 5}, {"args": [3], "expect": 5},
                      {"args": [4], "expect": 5}],
            "max_candidates": 4, "check_box": False},
         "expect_err": "REGRESS_FAIL",
         "expect": "期望恒为 5 而程序恒为 1: 邻域内 (1→2 / 置0 / 置1 / 删语句) 无一步能到常量 5 → "
                   "必须 REGRESS_FAIL 并说明'缺的是门 1+门 2', 不许给一个没修的补丁"},
        {"id": "rp_no_mutant_evidence", "args": {"program": [
            {"t": "assign", "var": "v0", "e": {"t": "bin", "op": "+", "l": {"t": "var", "n": "a0"},
                                              "r": {"t": "lit", "v": 1}}}],
            "tests": [{"args": [1], "expect": 0}, {"args": [2], "expect": 0},
                      {"args": [3], "expect": 0}, {"args": [4], "expect": 0}],
            "max_candidates": 16, "check_box": True},
         "expect_err": "MUTATION_SURVIVE",
         "expect": "期望恒为 0 而程序算 a0+1: 一步邻域里 **只有删掉那条语句** 能全过 "
                   "(空程序恒 0); 但空程序生成不出任何变异体 → 门 3 无证据, "
                   "按 §13「没有证书的解不算解」判不过, 而不是默认通过"},
        {"id": "rp_vpair_degenerate", "args": {"program": [
            {"t": "assign", "var": "v0", "e": {"t": "lit", "v": 1}}],
            "tests": [{"args": [0], "expect": 1}], "split_ratio": 0.6},
         "expect_err": "PRECOND_FAIL",
         "expect": "单条测试无法分成 train/held-out 两侧 → 划分退化, 拒绝在退化划分上验收"},
    ),
))


# ===========================================================================
# 9422 rep_regress_guard (before/after behaviour diff table)
# ===========================================================================

_IN_REG = {"kind": "dict",
           "fields": {"before": SCHEMA_PROG, "after": SCHEMA_PROG,
                      "tests": TEST_SCHEMA,
                      "extra_inputs": {"kind": "int", "min": 0, "max": 200},
                      "seed": {"kind": "int", "min": 0, "max": 2**31 - 1}},
           "required": ["before", "after", "tests"]}


def _inv_reg(args: dict, ctl: CallCtl) -> dict:
    """Behaviour diff: compare before/after outputs item by item on the "test set + extra random inputs" and list the difference table.

    The extra random inputs come from a **seeded** PRNG (reproducible), to catch behaviour drift the test set does not cover.
    """
    ctx = "op:rep_regress_guard#9422"
    before, after = args["before"], args["after"]
    validate_prog(before, ctx + ".before")
    validate_prog(after, ctx + ".after")
    tests = args["tests"]
    extra = int(args.get("extra_inputs", 0))
    rng = random.Random(int(args.get("seed", 41)))
    cases = [(_mk_inputs(t["args"]), t["expect"], "test") for t in tests]
    for _ in range(extra):
        inp = {a: rng.randint(-9, 9) for a in ARGS}
        cases.append((inp, None, "random"))
    diffs: list = []
    same = 0
    for inp, exp, src in cases:
        rb = interp(before, inp, 200_000)
        ra = interp(after, inp, 200_000)
        vb = ("ERR", rb["error"]["code"]) if rb["error"] else rb["state"].get("v0", 0)
        va = ("ERR", ra["error"]["code"]) if ra["error"] else ra["state"].get("v0", 0)
        if vb == va:
            same += 1
        else:
            diffs.append({"source": src, "args": [inp[a] for a in ARGS],
                          "before": vb, "after": va,
                          "test_expect": exp,
                          "after_still_correct": (exp is None) or (va == exp)})
    regressed = [d for d in diffs if d["source"] == "test" and not d["after_still_correct"]]
    return {"value": {"n_cases": len(cases), "n_same": same, "n_diff": len(diffs),
                      "diff_table": diffs[:64],
                      "n_regressed_tests": len(regressed),
                      "extra_random_inputs": extra},
            "resid": [float(len(regressed))],
            "notes": f"行为差分: {len(cases)} 个输入 (测试 {len(tests)} + 随机 {extra}), "
                     f"相同 {same}, 不同 {len(diffs)}; 其中**测试回归** {len(regressed)} 条 "
                     f"(原测试期望被破坏); 残差 = 回归条数"}


def _ver_reg(args: dict, value: dict) -> dict:
    """Independent verification: (a) rerun every input pair with **engine2**; every row of the diff table must be reproduced
    (differences are semantic facts, both engines must agree); (b) same/different counts are self-consistent;
    (c) regression decisions agree with the test expectations (does the after value satisfy expect)."""
    before, after = args["before"], args["after"]
    tests = args["tests"]
    extra = int(args.get("extra_inputs", 0))
    rng = random.Random(int(args.get("seed", 41)))
    cases = [(_mk_inputs(t["args"]), t["expect"], "test") for t in tests]
    for _ in range(extra):
        cases.append(({a: rng.randint(-9, 9) for a in ARGS}, None, "random"))
    diff2 = 0
    reg2 = 0
    inconsistent = 0
    listed = {(d["source"], tuple(d["args"])): d for d in value["diff_table"]}
    for inp, exp, src in cases:
        rb = interp_stack(before, inp, 200_000)
        ra = interp_stack(after, inp, 200_000)
        vb = ("ERR", rb["error"]["code"]) if rb["error"] else rb["state"].get("v0", 0)
        va = ("ERR", ra["error"]["code"]) if ra["error"] else ra["state"].get("v0", 0)
        if vb != va:
            diff2 += 1
            key = (src, tuple(inp[a] for a in ARGS))
            if key in listed:
                if listed[key]["before"] != vb or listed[key]["after"] != va:
                    inconsistent += 1
            if src == "test" and va != exp:
                reg2 += 1
    counts_ok = (value["n_diff"] == diff2) and (value["n_same"] + value["n_diff"] == value["n_cases"])
    reg_ok = (value["n_regressed_tests"] == reg2)
    ok = counts_ok and reg_ok and inconsistent == 0
    return {"ok": ok, "resid": [float(inconsistent)],
            "detail": f"engine2 复算: 差异 {diff2} (engine1 报 {value['n_diff']}), 回归 {reg2} "
                      f"(报 {value['n_regressed_tests']}); 计数自洽={counts_ok}; "
                      f"差异表明细与复算不符 {inconsistent} 条"}


REGISTRY.register(OpDef(
    name="rep_regress_guard", band=BAND, sub=22,
    in_schema=_IN_REG,
    out_schema={"kind": "dict",
                "fields": {"n_cases": {"kind": "int", "min": 1}, "n_same": {"kind": "int", "min": 0},
                           "n_diff": {"kind": "int", "min": 0},
                           "diff_table": {"kind": "list", "maxlen": 64,
                                          "items": {"kind": "dict", "extra": True, "fields": {}}},
                           "n_regressed_tests": {"kind": "int", "min": 0},
                           "extra_random_inputs": {"kind": "int", "min": 0}},
                "required": ["n_cases", "n_same", "n_diff", "diff_table", "n_regressed_tests",
                             "extra_random_inputs"]},
    invoke=_inv_reg, verify=_ver_reg,
    cost=("units", 2.0), errors=("SCHEMA_MISMATCH",),
    effects="none (纯函数)",
    doc="§9.3 回归护栏 (前后行为差分表) | in: 改前/改后程序 + 测试集 (+ 额外随机输入) | "
        "out: 逐输入差异表 + 测试回归条数 | 验收=**engine2 复现**全部差异 + 计数自洽 + "
        "回归判定与期望一致 | 残差 = 测试回归条数 | 2 units | "
        "诚实边界: 额外随机输入是有种子的**有界抽样**, 只声称覆盖所抽的那些输入 (extra_random_inputs 可查)",
    selfchecks=(
        {"id": "rrg_detects_regression", "args": {"before": [
            {"t": "assign", "var": "v0", "e": {"t": "bin", "op": "+", "l": {"t": "var", "n": "a0"},
                                              "r": {"t": "var", "n": "a1"}}}],
            "after": [
            {"t": "assign", "var": "v0", "e": {"t": "bin", "op": "-", "l": {"t": "var", "n": "a0"},
                                              "r": {"t": "var", "n": "a1"}}}],
            "tests": [{"args": [3, 2], "expect": 5}, {"args": [1, 1], "expect": 2}],
            "extra_inputs": 6, "seed": 5},
         "check": lambda r: (r.value["n_regressed_tests"] == 2 and r.value["n_diff"] >= 2,
                             f"regressed={r.value['n_regressed_tests']} diff={r.value['n_diff']}"),
         "expect": "+ 改成 −: 两个测试都回归 (3−2=1≠5, 1−1=0≠2); 差异表至少 2 条"},
        {"id": "rrg_no_diff_when_same", "args": {"before": [
            {"t": "assign", "var": "v0", "e": {"t": "lit", "v": 5}}],
            "after": [
            {"t": "assign", "var": "v0", "e": {"t": "lit", "v": 5}}],
            "tests": [{"args": [0], "expect": 5}], "extra_inputs": 4, "seed": 6},
         "check": lambda r: (r.value["n_diff"] == 0 and r.value["n_regressed_tests"] == 0
                             and r.value["n_same"] == r.value["n_cases"],
                             f"diff={r.value['n_diff']} same={r.value['n_same']}"),
         "expect": "两程序逐字相同 → 差异 0 (差分表空)"},
        {"id": "rrg_extra_inputs_find_drift", "args": {"before": [
            {"t": "assign", "var": "v0", "e": {"t": "var", "n": "a0"}}],
            "after": [
            {"t": "assign", "var": "v0", "e": {"t": "bin", "op": "+", "l": {"t": "var", "n": "a0"},
                                              "r": {"t": "lit", "v": 1}}}],
            "tests": [{"args": [0], "expect": 1}], "extra_inputs": 8, "seed": 7},
         "check": lambda r: (r.value["n_regressed_tests"] == 0 and r.value["n_diff"] >= 1,
                             f"regressed={r.value['n_regressed_tests']} diff={r.value['n_diff']}"),
         "expect": "测试集只有 (0 → 1) 恰好通过, 但随机输入会暴露漂移 (a0→a0+1) → "
                   "测试回归 0 条而差异 >= 1 条, 正说明差分表能抓到测试集看不见的变化"},
        {"id": "rrg_bad_after", "args": {"before": [], "after": [{"t": "assign", "var": "zz",
                                                                 "e": {"t": "lit", "v": 1}}],
                                         "tests": [{"args": [], "expect": 0}]},
         "expect_err": "SCHEMA_MISMATCH", "expect": "改后程序非法必须被拦下"},
    ),
))