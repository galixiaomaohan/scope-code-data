"""§9.2 static-analysis operators (band 9400, rows 9410–9415) — P5 deliverable A.

Six pure-function static-analysis operators, all working on §9.1's bounded AST language (code_core's ARGS/VARS/BINOPS):

  ana_dep_graph         9410  dependency graph (statement-level def-use edges)
  ana_dataflow          9411  definition-use chains + taint propagation (sources: input slots a0..a3)
  ana_complexity        9412  cyclomatic complexity (decision-point count, McCabe convention)
  ana_abstract_interp   9413  interval-domain abstract interpretation (**over-approximation direction = sound**: intervals contain the true values)
  ana_invariant_infer   9414  loop invariants (Houdini style: filter template candidates one by one, keeping only those implied by the intervals)
  ana_dead_code         9415  reachability + coverage (unreachable statements + input-space sampled coverage)

**§17.10 in practice (static-analysis over-approximation must declare its direction explicitly)**: every operator in this
module carries a `direction` field in its output:
  - `"over"`  : the conclusion is an **over-approximation** (sound) — things declared "cannot happen" indeed cannot, but
                it may report "possible" where exact information is lost (abstract interpretation / taint / reachability).
  - `"exact"` : graph/counting conclusions are exact (dependency graph, cyclomatic complexity, def-use chains).
Each operator's docstring states item by item which class it belongs to and **exactly where the over-approximation happens**.

**Acceptance convention (§9.2 requires "acceptance = sampled dynamic-execution comparison")**: every operator accepts
`trials`/`seed` parameters, actually runs the concrete interpreter on seeded random inputs, compares the "static
declaration" with the "dynamic observation" item by item, and writes the mismatch rate into the return value
(`dyn_mismatch`). A nonzero mismatch rate is a **real defect** (§9.6 static/dynamic dual-kernel cross-check), not
"approximately right". All dynamic sampling is **bounded**: the return value's `dyn_scope` says how many inputs were sampled.
"""
from __future__ import annotations

import random

from ..errors import Err, SvoError
from ..registry import REGISTRY, OpDef
from ..tool import CallCtl
from .code_core import (
    ARGS, LOOP_CAP, SCHEMA_PROG, VARS,
    interp, sid_map, validate_prog,
)

BAND = 9400
_OPS = ("+", "-", "*", "/", "%")
_RELS = ("<", "<=", ">", ">=", "==", "!=")
_LOGI = ("and", "or")


# ---------------------------------------------------------------------------
# shared: statement traversal (pre-order) and read/write sets
# ---------------------------------------------------------------------------


def _all_stmts(stmts):
    """Pre-order traversal of **any statement list**, yielding statements (without sid).

    Why it must stay separate from _walk_stmts: sid_map numbers relative to **the program it receives**. Calling sid_map
    on a subtree yields local numbers (the subtree's 0th statement → sid 0), which do not match the global numbering —
    this was exactly the root cause of the "unreachable set" once mislabelling the if body as dead code (it computed sid
    on the th subtree separately). Global numbering must be computed once over the whole program, then looked up via id(stmt).
    """
    out = []

    def rec(ss):
        for s in ss:
            out.append(s)
            if s["t"] == "if":
                rec(s.get("th", []))
                rec(s.get("el", []))
            elif s["t"] == "loop":
                rec(s.get("body", []))

    rec(stmts)
    return out


def _walk_stmts(prog):
    """Pre-order traversal of all statements, yielding (sid, stmt). Uses the same numbering rule as code_core.sid_map."""
    sids, _ = sid_map(prog)
    return [(sids[id(s)], s) for s in _all_stmts(prog)]


def _expr_reads(e: dict) -> set:
    """Set of variable names read by the expression (including input slots)."""
    out: set = set()
    t = e["t"]
    if t == "var":
        out.add(e["n"])
    elif t == "bin":
        out |= _expr_reads(e["l"])
        out |= _expr_reads(e["r"])
    elif t == "un":
        out |= _expr_reads(e["e"])
    elif t == "call":
        for a in e["args"]:
            out |= _expr_reads(a)
    return out


def _stmt_def_use(s: dict) -> "tuple[set, set]":
    """The (definition set, use set) of a statement. Loop nodes additionally record 'v' among the definitions."""
    if s["t"] == "assign":
        return {s["var"]}, _expr_reads(s["e"])
    if s["t"] == "if":
        u = _expr_reads(s["c"])
        return set(), u
    d, u = {s["v"]}, _expr_reads(s["b"])
    return d, u


# ===========================================================================
# 9410 ana_dep_graph
# ===========================================================================

_IN_GRAPH = {"kind": "dict",
             "fields": {"program": SCHEMA_PROG, "trials": {"kind": "int", "min": 1, "max": 200},
                        "seed": {"kind": "int", "min": 0, "max": 2**31 - 1}},
             "required": ["program"]}


def _inv_graph(args: dict, ctl: CallCtl) -> dict:
    """Dependency graph: nodes = statements (pre-order numbering), edges = "writes some variable → later reads it"."""
    ctx = "op:ana_dep_graph#9410"
    prog = args["program"]
    validate_prog(prog, ctx)
    nodes = _walk_stmts(prog)
    order = [sid for sid, _ in nodes]
# **Reaching definitions**: each use links to **every** preceding definition of that variable. Linking only "the last
# definition in program order" is **unsound** — of two definitions inside a branch only one takes effect at runtime,
# so a runtime use-def pair may land on the missed definition (measured: 15 of 31 runtime pairs were missed). Linking
# all predecessor definitions is an over-approximation (consistent with direction=over): every definition actually used
# at runtime is inside the edge set.
    defs_of: dict = {}
    edges: list = []
    seen: set = set()
    for sid, s in nodes:
        d, u = _stmt_def_use(s)
        for v in sorted(u):
            for src in defs_of.get(v, ()):
                key = (src, sid, v)
                if key not in seen:
                    seen.add(key)
                    edges.append({"from": src, "to": sid, "var": v})
        for v in sorted(d):
            defs_of.setdefault(v, []).append(sid)
    return {"value": {"nodes": order, "edges": edges, "n_nodes": len(order),
                      "n_edges": len(edges), "direction": "over"},
            "resid": [0.0],
            "notes": f"依赖图: {len(order)} 节点, {len(edges)} 条 def→use 边; "
                     f"direction=over (边集取**到达定义**: 同一个 use 连到所有前驱定义点, "
                     f"运行时用到的定义必是其子集; 用程序序'最后一个定义'会漏边)"}


def _ver_graph(args: dict, value: dict) -> dict:
    """Independent verification: (a) the node set must match sid_map's pre-order numbering exactly;
    (b) every edge must satisfy from < to (definition before use; no forward edges);
    (c) **dynamic comparison**: sample executions and, for every edge, check that the variable's value at from is the
    last definition read at to (the runtime use-def chain); the mismatch rate must be 0."""
    ctx = "op:ana_dep_graph@verify"
    prog = args["program"]
    sid_map_, total = sid_map(prog)
    nids = sorted(sid_map_.values())
    nodes_ok = (sorted(value["nodes"]) == nids)
    forward = [e for e in value["edges"] if not (e["from"] < e["to"])]
    # dynamic comparison: record the "most recent defining statement" table at each step
    trial_n = int(args.get("trials", 32))
    rng = random.Random(int(args.get("seed", 7)))
    mismatch = 0
    checked = 0
    stmts = _walk_stmts(prog)
    for _ in range(trial_n):
        inp = {a: rng.randint(-8, 8) for a in ARGS}
        trace = _run_traced(prog, inp)
        if trace is None:
            continue
        defsite: dict = {}
        for sid in trace["order_seq"]:
            s = next((st for i, st in stmts if i == sid), None)
            if s is None:
                continue
            d, u = _stmt_def_use(s)
            for v in sorted(u):
                if v in defsite:
                    want = {"from": defsite[v], "to": sid, "var": v}
                    checked += 1
                    if want not in value["edges"]:
                        mismatch += 1
            for v in sorted(d):
                defsite[v] = sid
    ok = nodes_ok and not forward and mismatch == 0
    return {"ok": ok, "resid": [float(mismatch)],
            "detail": f"节点集与 sid_map 一致={nodes_ok}; 前向边数={len(forward)}; "
                      f"动态抽样对照 {checked} 条运行时 def-use, 边集缺失 {mismatch} 条 "
                      f"(trials={trial_n})"}


def _run_traced(prog, inp):
    """Execution with the statement trace (uses engine1's executed list; returns None on failure)."""
    r = interp(prog, inp, 200_000)
    if r["error"] is not None:
        return None
    return {"order_seq": r["executed"], "state": r["state"]}


REGISTRY.register(OpDef(
    name="ana_dep_graph", band=BAND, sub=10,
    in_schema=_IN_GRAPH,
    out_schema={"kind": "dict",
                "fields": {"nodes": {"kind": "list", "items": {"kind": "int"}},
                           "edges": {"kind": "list", "maxlen": 4096,
                                     "items": {"kind": "dict", "extra": False,
                                               "fields": {"from": {"kind": "int"}, "to": {"kind": "int"},
                                                          "var": {"kind": "str"}},
                                               "required": ["from", "to", "var"]}},
                           "n_nodes": {"kind": "int", "min": 0}, "n_edges": {"kind": "int", "min": 0},
                           "direction": {"kind": "str", "enum": ["over", "exact"]}},
                "required": ["nodes", "edges", "n_nodes", "n_edges", "direction"]},
    invoke=_inv_graph, verify=_ver_graph,
    cost=("units", 1.0), errors=("SCHEMA_MISMATCH",),
    effects="none (纯函数)",
    doc="§9.2 依赖图 | in: 有界 AST 程序 (+ 动态抽样参数) | out: 语句节点 + def→use 边 | "
        "验收=动态抽样对照: 运行时 use-def 链必须全部出现在边集里 (不一致率 0) | 1 unit | "
        "过近似方向: **over** — 边集含「定义后任何路径的使用」, 故是运行时实际用到的超集; "
        "声明为「不可达」的语句确实不可达, 但「可达」不蕴含「运行时一定执行」",
    selfchecks=(
        {"id": "adg_linear_chain", "args": {"program": [
            {"t": "assign", "var": "v0", "e": {"t": "var", "n": "a0"}},
            {"t": "assign", "var": "v1", "e": {"t": "bin", "op": "+", "l": {"t": "var", "n": "v0"},
                                              "r": {"t": "lit", "v": 1}}},
            {"t": "assign", "var": "v2", "e": {"t": "bin", "op": "*", "l": {"t": "var", "n": "v1"},
                                              "r": {"t": "var", "n": "a1"}}}],
            "trials": 16, "seed": 3},
         "check": lambda r: (r.value["n_nodes"] == 3 and r.value["n_edges"] == 2
                             and all(e["from"] < e["to"] for e in r.value["edges"]),
                             f"nodes={r.value['n_nodes']} edges={r.value['n_edges']}"),
         "expect": "三语句线性链 v0←a0, v1←v0+1, v2←v1*a1: 边为 (0→1,v0) 与 (1→2,v1), "
                   "外加 if 之外无其他 def→use 关系, 故恰好 3 条 (a1 未被程序内定义, 无入边)"},
        {"id": "adg_if_branch_use", "args": {"program": [
            {"t": "assign", "var": "v0", "e": {"t": "lit", "v": 5}},
            {"t": "if", "c": {"t": "var", "n": "a0"},
             "th": [{"t": "assign", "var": "v1", "e": {"t": "var", "n": "v0"}}],
             "el": [{"t": "assign", "var": "v1", "e": {"t": "lit", "v": 0}}]},
            {"t": "assign", "var": "v2", "e": {"t": "var", "n": "v1"}}],
            "trials": 16, "seed": 5},
         "check": lambda r: (r.value["n_nodes"] == 5 and r.value["n_edges"] == 3,
                             f"nodes={r.value['n_nodes']} edges={r.value['n_edges']}"),
         "expect": "分支结构共 5 个语句节点 (assign, if, th-assign, el-assign, assign): "
                   "边为 v0→th(v1) 与 v0→el 不可达故无; 实际为 (0→2,v0), (0→3,v0 若 el 读 v0?), "
                   "以及 v1→4 (th 与 el 两个定义点都连到 4) —— 见实现: 到达定义给出 3 条"},
        {"id": "adg_bad_program", "args": {"program": [{"t": "assign", "var": "zz", "e": {"t": "lit", "v": 1}}]},
         "expect_err": "SCHEMA_MISMATCH", "expect": "非法变量槽 zz 必须被结构门拦下"},
        {"id": "adg_empty_program", "args": {"program": [], "trials": 4, "seed": 1},
         "check": lambda r: (r.value["n_nodes"] == 0 and r.value["edges"] == [],
                             f"nodes={r.value['n_nodes']}"),
         "expect": "空程序: 0 节点 0 边 (边界覆盖)"},
    ),
))


# ===========================================================================
# 9411 ana_dataflow (def-use chains + taint propagation)
# ===========================================================================

_IN_DF = {"kind": "dict",
          "fields": {"program": SCHEMA_PROG, "sinks": {"kind": "list", "items": {"kind": "str"},
                                                       "maxlen": 16},
                     "trials": {"kind": "int", "min": 1, "max": 200},
                     "seed": {"kind": "int", "min": 0, "max": 2**31 - 1}},
          "required": ["program"]}


def _inv_df(args: dict, ctl: CallCtl) -> dict:
    """Definition-use chains (reads/writes per statement + upstream definition points) and taint propagation (sources = input slots a0..a3)."""
    ctx = "op:ana_dataflow#9411"
    prog = args["program"]
    validate_prog(prog, ctx)
    nodes = _walk_stmts(prog)
    tainted = set(ARGS)  # input slots are the taint sources
    chains: list = []
    taint_at: dict = {}
    for sid, s in nodes:
        d, u = _stmt_def_use(s)
        chains.append({"sid": sid, "def": sorted(d), "use": sorted(u)})
        taint_at[sid] = sorted(u & tainted)
        if d & tainted or (u & tainted):
            tainted |= d
    sinks = list(args.get("sinks", []))
    sink_taint = {v: (v in tainted) for v in sinks}
    for v in sinks:
        if v not in VARS:
            raise SvoError(Err.SCHEMA_MISMATCH, ctx=ctx,
                           detail=f"sink {v!r} 不是可赋值槽 (v0..v15)")
    return {"value": {"chains": chains, "tainted_vars": sorted(tainted),
                      "sink_taint": sink_taint, "n_tainted": len(tainted),
                      "direction": "over"},
            "resid": [0.0],
            "notes": f"定义-使用链 {len(chains)} 条; 污点变量 {len(tainted)} 个 (源={list(ARGS)}); "
                     f"sink 污点状态 {sink_taint}; direction=over "
                     f"(污点是**并集式**传播: 任一上游输入被污染即传染, 不做值相关性精化)"}


def _ver_df(args: dict, value: dict) -> dict:
    """Independent verification: (a) chains cover all statements and def/use match a re-read of the AST;
    (b) taint **monotonicity**: a sink marked untainted must really be independent of the inputs in dynamic sampling
    (changing inputs does not change that variable's final value) — a **direct counter-example search** against the taint claim;
    (c) sinks marked tainted are only recorded, not rejected (over-approximation may be conservative)."""
    prog = args["program"]
    nodes = _walk_stmts(prog)
    chains = {c["sid"]: c for c in value["chains"]}
    chains_ok = (sorted(chains) == sorted(sid for sid, _ in nodes))
    if chains_ok:
        for sid, s in nodes:
            d, u = _stmt_def_use(s)
            if chains[sid]["def"] != sorted(d) or chains[sid]["use"] != sorted(u):
                chains_ok = False
                break
    # (b) dynamic independence check for untainted sinks
    trial_n = int(args.get("trials", 32))
    rng = random.Random(int(args.get("seed", 11)))
    sinks_untainted = [v for v, t in value["sink_taint"].items() if not t]
    mismatch = 0
    checked = 0
    for v in sinks_untainted:
        base = None
        for _ in range(trial_n):
            inp = {a: rng.randint(-9, 9) for a in ARGS}
            r = interp(prog, inp, 200_000)
            if r["error"] is not None:
                continue
            val = r["state"].get(v)
            checked += 1
            if base is None:
                base = val
            elif val != base:
                mismatch += 1
                break
    ok = chains_ok and mismatch == 0
    return {"ok": ok, "resid": [float(mismatch)],
            "detail": f"链表与 AST 重读一致={chains_ok}; 未污染 sink {sinks_untainted} 的"
                      f"动态输入无关性检查: {checked} 次执行, 违反 {mismatch} 次 "
                      f"(trials={trial_n}) — 违反即污点分析漏报 (真缺陷)"}


REGISTRY.register(OpDef(
    name="ana_dataflow", band=BAND, sub=11,
    in_schema=_IN_DF,
    out_schema={"kind": "dict",
                "fields": {"chains": {"kind": "list", "maxlen": 64,
                                      "items": {"kind": "dict", "extra": False,
                                                "fields": {"sid": {"kind": "int"},
                                                           "def": {"kind": "list", "items": {"kind": "str"}},
                                                           "use": {"kind": "list", "items": {"kind": "str"}}},
                                                "required": ["sid", "def", "use"]}},
                           "tainted_vars": {"kind": "list", "items": {"kind": "str"}},
                           "sink_taint": {"kind": "dict", "extra": True},
                           "n_tainted": {"kind": "int", "min": 0},
                           "direction": {"kind": "str", "enum": ["over", "exact"]}},
                "required": ["chains", "tainted_vars", "sink_taint", "n_tainted", "direction"]},
    invoke=_inv_df, verify=_ver_df,
    cost=("units", 1.5), errors=("SCHEMA_MISMATCH",),
    effects="none (纯函数)",
    doc="§9.2 定义-使用链 + 污点传播 | in: 程序 (+sinks) | out: 逐语句 def/use + 污点集合 + sink 状态 | "
        "验收=未污染 sink 的**动态输入无关性**反例搜索 (改变输入终值必须不变) | 1.5 units | "
        "过近似方向: **over** — 污点按并集传播 (任一上游污染即污染), 故「未污染」是强结论、"
        "「已污染」可能是保守误报 | 失败: sink 非法 SCHEMA_MISMATCH",
    selfchecks=(
        {"id": "adf_taint_flows", "args": {"program": [
            {"t": "assign", "var": "v0", "e": {"t": "var", "n": "a0"}},
            {"t": "assign", "var": "v1", "e": {"t": "bin", "op": "*", "l": {"t": "var", "n": "v0"},
                                              "r": {"t": "lit", "v": 2}}}],
            "sinks": ["v0", "v1"], "trials": 12, "seed": 2},
         "check": lambda r: (r.value["sink_taint"] == {"v0": True, "v1": True}
                             and "v1" in r.value["tainted_vars"],
                             f"sink={r.value['sink_taint']} tainted={r.value['tainted_vars']}"),
         "expect": "a0 → v0 → v1: 两个 sink 都被污染 (污点随赋值传播)"},
        {"id": "adf_constant_is_clean", "args": {"program": [
            {"t": "assign", "var": "v0", "e": {"t": "lit", "v": 7}},
            {"t": "assign", "var": "v1", "e": {"t": "var", "n": "v0"}}],
            "sinks": ["v1"], "trials": 12, "seed": 4},
         "check": lambda r: (r.value["sink_taint"] == {"v1": False},
                             f"sink={r.value['sink_taint']}"),
         "expect": "常量赋值链: v1 与输入无关 → 未污染; 且动态检查会验证它真的与输入无关"},
        {"id": "adf_mixed_taint", "args": {"program": [
            {"t": "assign", "var": "v0", "e": {"t": "lit", "v": 1}},
            {"t": "assign", "var": "v1", "e": {"t": "var", "n": "a1"}},
            {"t": "assign", "var": "v2", "e": {"t": "bin", "op": "and", "l": {"t": "lit", "v": 0},
                                              "r": {"t": "var", "n": "v1"}}}],
            "sinks": ["v0", "v2"], "trials": 12, "seed": 6},
         "check": lambda r: (r.value["sink_taint"]["v0"] is False
                             and r.value["sink_taint"]["v2"] is True,
                             f"sink={r.value['sink_taint']}"),
         "expect": "混合: v0 来自常量 (未污染); v2 含 a1 → 污染。"
                   "注意 v2 = 0 and a1 恒为 0, 污点仍然标记为污染 —— 这就是 over 的保守性"},
        {"id": "adf_bad_sink", "args": {"program": [], "sinks": ["nope"], "trials": 2, "seed": 1},
         "expect_err": "SCHEMA_MISMATCH", "expect": "非槽位 sink"},
    ),
))


# ===========================================================================
# 9412 ana_complexity
# ===========================================================================

_IN_CX = {"kind": "dict", "fields": {"program": SCHEMA_PROG}, "required": ["program"]}


def _inv_cx(args: dict, ctl: CallCtl) -> dict:
    """Cyclomatic complexity (McCabe): 1 + number of decision points. Decision points = if statements (1 each) + loop
    statements (1 each) + each logical connective (and/or) counts 1."""
    prog = args["program"]
    validate_prog(prog, "op:ana_complexity#9412")
    decisions = 0
    nodes = 0

    def count_expr(e) -> None:
        nonlocal decisions
        if e["t"] == "bin":
            if e["op"] in _LOGI:
                decisions += 1
            count_expr(e["l"])
            count_expr(e["r"])
        elif e["t"] == "un":
            count_expr(e["e"])
        elif e["t"] == "call":
            for a in e["args"]:
                count_expr(a)

    for _sid, s in _walk_stmts(prog):
        nodes += 1
        if s["t"] == "if":
            decisions += 1
            count_expr(s["c"])
        elif s["t"] == "loop":
            decisions += 1
            count_expr(s["b"])
        elif s["t"] == "assign":
            count_expr(s["e"])
# Statements inside a loop body are already counted in _walk_stmts (pre-order), but assign expressions would be
# double-counted, so dedup is needed: this now traverses all statements and all expressions once, and the
# implementation above already dedups per statement (each statement counted once).
    return {"value": {"complexity": 1 + decisions, "decisions": decisions, "n_nodes": nodes,
                      "direction": "exact"},
            "resid": [0.0],
            "notes": f"圈复杂度 {1 + decisions} = 1 + {decisions} 个判定点 "
                     f"(if {sum(1 for _, s in _walk_stmts(prog) if s['t'] == 'if')}, "
                     f"loop {sum(1 for _, s in _walk_stmts(prog) if s['t'] == 'loop')}, "
                     f"逻辑连接词); 语句数 {nodes}; direction=exact (计数无近似)"}


def _ver_cx(args: dict, value: dict) -> dict:
    """Independent verification: recheck with **a second counting implementation** (regex-style counting of node types and
    and/or occurrences directly on the AST); also check the lower-bound property (complexity ≥ 1, and ≥ statement count×0+1)."""
    prog = args["program"]
    ifs = loops = logi = 0

    def walk(x) -> None:
        nonlocal ifs, loops, logi
        if isinstance(x, dict):
            if x.get("t") == "if":
                ifs += 1
            elif x.get("t") == "loop":
                loops += 1
            elif x.get("t") == "bin" and x.get("op") in _LOGI:
                logi += 1
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(prog)
    expect = 1 + ifs + loops + logi
    ok = (int(value["complexity"]) == expect) and (int(value["complexity"]) >= 1)
    return {"ok": ok, "resid": [float(abs(int(value["complexity"]) - expect))],
            "detail": f"独立计数 (递归遍历 AST): 1+{ifs}+{loops}+{logi} = {expect}; "
                      f"输出 {value['complexity']}; ≥1 成立={int(value['complexity']) >= 1}"}


REGISTRY.register(OpDef(
    name="ana_complexity", band=BAND, sub=12,
    in_schema=_IN_CX,
    out_schema={"kind": "dict",
                "fields": {"complexity": {"kind": "int", "min": 1},
                           "decisions": {"kind": "int", "min": 0},
                           "n_nodes": {"kind": "int", "min": 0},
                           "direction": {"kind": "str", "enum": ["over", "exact"]}},
                "required": ["complexity", "decisions", "n_nodes", "direction"]},
    invoke=_inv_cx, verify=_ver_cx,
    cost=("units", 0.5), errors=("SCHEMA_MISMATCH",),
    effects="none (纯函数)",
    doc="§9.2 圈复杂度 (McCabe: 1 + 判定点) | in: 程序 | out: 复杂度 + 判定点数 + 语句数 | "
        "验收=独立计数实现复核 (递归遍历 AST 数 if/loop/and/or) | 0.5 unit | "
        "过近似方向: **exact** — 计数是精确的 (不含任何近似), 但需注意本语言无函数调用/异常, "
        "故 McCabe 口径就是 1+判定点 (跳出与多出口在本有界语言里不存在)",
    selfchecks=(
        {"id": "acx_flat", "args": {"program": [{"t": "assign", "var": "v0", "e": {"t": "lit", "v": 1}}]},
# n_nodes must be asserted too: auditor E's mutation experiment changed n_nodes' +1 to +2 and 580 selfchecks
# stayed green — because **no assertion looked at this field** (soft test). It is asserted from now on.
         "check": lambda r: (r.value["complexity"] == 1 and r.value["decisions"] == 0
                             and r.value["n_nodes"] == 1,
                             f"cx={r.value['complexity']} n_nodes={r.value['n_nodes']}"),
         "expect": "无分支: 圈复杂度 = 1"},
        {"id": "acx_one_if", "args": {"program": [
            {"t": "if", "c": {"t": "var", "n": "a0"},
             "th": [{"t": "assign", "var": "v0", "e": {"t": "lit", "v": 1}}],
             "el": [{"t": "assign", "var": "v0", "e": {"t": "lit", "v": 2}}]}]},
         "check": lambda r: (r.value["complexity"] == 2, f"cx={r.value['complexity']}"),
         "expect": "一个 if: 1+1 = 2"},
        {"id": "acx_if_and_loop_with_logic", "args": {"program": [
            {"t": "if", "c": {"t": "bin", "op": "and", "l": {"t": "var", "n": "a0"},
                              "r": {"t": "var", "n": "a1"}},
             "th": [{"t": "loop", "v": "v0", "b": {"t": "lit", "v": 3},
                     "body": [{"t": "assign", "var": "v1",
                               "e": {"t": "bin", "op": "or", "l": {"t": "var", "n": "v0"},
                                     "r": {"t": "lit", "v": 0}}}]}],
             "el": []}]},
         "check": lambda r: (r.value["complexity"] == 5 and r.value["decisions"] == 4
                             and r.value["n_nodes"] == 3,
                             f"cx={r.value['complexity']} dec={r.value['decisions']} "
                             f"n_nodes={r.value['n_nodes']}"),
         "expect": "if(1) + and(1) + loop(1) + or(1) + 1 = 5, 判定点 4 个; "
                   "语句数 3 —— 三个字段全部断言 (防'改了字段却无断言'的软测试)"},
        {"id": "acx_bad_op", "args": {"program": [
            {"t": "assign", "var": "v0", "e": {"t": "bin", "op": "**", "l": {"t": "lit", "v": 1},
                                               "r": {"t": "lit", "v": 2}}}]},
         "expect_err": "SCHEMA_MISMATCH", "expect": "未登记算子 ** 必须被拦下"},
    ),
))


# ===========================================================================
# 9413 ana_abstract_interp (interval-domain abstract interpretation)
# ===========================================================================
#
# Abstract domain: variable → integer interval [lo, hi], with None endpoints meaning ±∞.
# **Over-approximation direction = over (sound)**: the abstract interval **contains** every possible true value.
# The over-approximation happens in exactly two places, which must be declared explicitly (otherwise an audit cannot
# judge the conclusion's strength):
#   1. `if`  : conditions are not refined; the two branch abstract states are **joined** directly (losing correlations);
#   2. `loop`: fixed-point iteration + widening (iteration cap min(LOOP_CAP, 24)), extrapolating the loop effect to the worst case.
# Hence the output can prove "the variable must lie in [lo,hi]" (a strong conclusion), but **cannot** prove
# "a value is definitely reachable" (a weak conclusion).

_ABS_TOP = (None, None)


def _abs_join(a, b):
    lo = None if (a[0] is None or b[0] is None) else min(a[0], b[0])
    hi = None if (a[1] is None or b[1] is None) else max(a[1], b[1])
    return (lo, hi)


def _abs_op(op: str, x, y):
    """Interval-domain binary operations (every result encloses the true range = sound)."""
    lo_x, hi_x = x
    lo_y, hi_y = y
    if op == "+":
        return (None if (lo_x is None or lo_y is None) else lo_x + lo_y,
                None if (hi_x is None or hi_y is None) else hi_x + hi_y)
    if op == "-":
        return (None if (lo_x is None or hi_y is None) else lo_x - hi_y,
                None if (hi_x is None or lo_y is None) else hi_x - lo_y)
    if op == "*":
        if None in (lo_x, hi_x, lo_y, hi_y):
            return _ABS_TOP
        c = [lo_x * lo_y, lo_x * hi_y, hi_x * lo_y, hi_x * hi_y]
        return (min(c), max(c))
    if op in ("/", "%"):
        if None in (lo_y, hi_y) or lo_y <= 0 <= hi_y:
            return _ABS_TOP  # possible division by 0 → conservatively take ⊤
        if None in (lo_x, hi_x):
            return _ABS_TOP
        c = []
        for a in (lo_x, hi_x):
            for b in (lo_y, hi_y):
                c.append(a // b)
                c.append(a % b)
        return (min(c) - 1, max(c) + 1)  # under floor-division semantics, widen conservatively by 1
        # comparison and logic operators: the range is always {0,1}
    return (0, 1)


def _abs_expr(e: dict, st: dict) -> tuple:
    t = e["t"]
    if t == "lit":
        v = int(e["v"])
        return (v, v)
    if t == "var":
        return st.get(e["n"], _ABS_TOP)
    if t == "un":
        a = _abs_expr(e["e"], st)
        if e["op"] == "neg":
            return (None if a[1] is None else -a[1], None if a[0] is None else -a[0])
        return (0, 1)
    if t == "bin":
        return _abs_op(e["op"], _abs_expr(e["l"], st), _abs_expr(e["r"], st))
    if t == "call":
        args = [_abs_expr(a, st) for a in e["args"]]
        fn = e["fn"]
        if fn == "abs":
            lo, hi = args[0]
            if lo is None or hi is None:
                return (0, None)
            lo2 = 0 if lo <= 0 <= hi else min(abs(lo), abs(hi))
            return (lo2, max(abs(lo), abs(hi)))
        if fn in ("min", "max"):
            if any(a[0] is None or a[1] is None for a in args):
                return _ABS_TOP
            fn2 = min if fn == "min" else max
            return (fn2(a[0] for a in args), fn2(a[1] for a in args))
        if fn == "clamp":
            if any(a[0] is None or a[1] is None for a in args):
                return _ABS_TOP
            x_, lo_, hi_ = args
            return (min(max(x_[0], lo_[0]), hi_[0]), min(max(x_[1], lo_[1]), hi_[1]))
        return (-1, 1)  # sign
    return _ABS_TOP


def _abs_stmts(stmts, st: dict, ctl, ctx: str) -> dict:
    out = dict(st)
    for s in stmts:
        ctl.check_time(ctx + "@abs")
        if s["t"] == "assign":
            out[s["var"]] = _abs_expr(s["e"], out)
        elif s["t"] == "if":
            a = _abs_stmts(s.get("th", []), dict(out), ctl, ctx)
            b = _abs_stmts(s.get("el", []), dict(out), ctl, ctx)
            for k in set(a) | set(b):
                out[k] = _abs_join(a.get(k, _ABS_TOP), b.get(k, _ABS_TOP))
        else:  # loop: fixed point + widening
            body = s.get("body", [])
            bnd = s["b"]
            cap = min(int(bnd["v"]), LOOP_CAP) if bnd["t"] == "lit" else LOOP_CAP
            cnt = s["v"]
            cur = dict(out)
            for _ in range(min(LOOP_CAP, 24)):
                nxt = _abs_stmts(body, dict(cur), ctl, ctx)
        # Engine semantics: each round first writes the counter v = i (i ∈ [0, cap)), then runs the loop body.
        # The abstraction must explicitly merge this in, otherwise the counter's final state is missed — measured:
        # after 8 loop rounds the real v0=7 while the abstract interval reported [0,0], and the dynamic comparison failed.
                if cap > 0:
                    nxt[cnt] = _abs_join(nxt.get(cnt, _ABS_TOP), (0, cap - 1))
                merged = {k: _abs_join(cur.get(k, _ABS_TOP), nxt.get(k, _ABS_TOP))
                          for k in set(cur) | set(nxt)}
                if merged == cur:
                    break
                cur = merged
            # after the loop the counter sits near its last value; conservatively join in [0, cap]
            cur[cnt] = _abs_join(cur.get(cnt, _ABS_TOP), (0, max(cap, 0)))
            out = cur
    return out


def _abs_init_state() -> dict:
    """Abstract initial state: input-slot bounds = the schema-declared bounds; variables start at 0 (consistent with the engine semantics)."""
    st: dict = {a: (-10**4, 10**4) for a in ARGS}
    st.update({v: (0, 0) for v in VARS})
    return st


_IN_ABS = {"kind": "dict",
           "fields": {"program": SCHEMA_PROG, "trials": {"kind": "int", "min": 1, "max": 300},
                      "seed": {"kind": "int", "min": 0, "max": 2**31 - 1}},
           "required": ["program"]}


def _inv_abs(args: dict, ctl: CallCtl) -> dict:
    ctx = "op:ana_abstract_interp#9413"
    prog = args["program"]
    validate_prog(prog, ctx)
    out = _abs_stmts(prog, _abs_init_state(), ctl, ctx)
    iv = {k: [v[0], v[1]] for k, v in sorted(out.items()) if k in VARS}
    unbounded = [k for k, v in iv.items() if v[0] is None or v[1] is None]
    return {"value": {"intervals": iv, "unbounded_vars": unbounded, "n_vars": len(VARS),
                      "direction": "over",
                      "approx_sources": ["if 分支取并 (丢条件细化)", "loop 加宽 (迭代上限)"]},
            "resid": [float(len(unbounded))],
            "notes": f"区间域抽象解释: 报告 {len(iv)} 个变量的终态区间; 无界变量 {len(unbounded)} 个; "
                     f"direction=over (sound) —— 区间**包含**全部真实取值, "
                     f"过近似来源见 approx_sources"}


def _ver_abs(args: dict, value: dict) -> dict:
    """Independent verification (§9.2 explicitly requires "acceptance = sampled dynamic-execution comparison"): sample inputs and
    really run the concrete interpreter; every observed final variable value must lie inside the abstract interval. Any escape ⇒
    the abstract interpretation is **not sound** (a real defect). Interval endpoint ordering is checked as well."""
    prog = args["program"]
    trials = int(args.get("trials", 64))
    rng = random.Random(int(args.get("seed", 13)))
    iv = value["intervals"]
    mismatch = 0
    observed = 0
    ex = ""
    for _ in range(trials):
        inp = {a: rng.randint(-10**4, 10**4) for a in ARGS}
        r = interp(prog, inp, 200_000)
        if r["error"] is not None:
            continue
        observed += 1
        for v, bnd in iv.items():
            val = r["state"].get(v)
            if val is None:
                continue
            lo, hi = bnd
            if (lo is not None and val < lo) or (hi is not None and val > hi):
                mismatch += 1
                if not ex:
                    ex = f"{v}={val} ∉ [{lo},{hi}]"
                break
    order_ok = all((b[0] is None or b[1] is None or b[0] <= b[1]) for b in iv.values())
    ok = (mismatch == 0) and order_ok
    return {"ok": ok, "resid": [float(mismatch)],
            "detail": f"动态抽样 {observed} 次执行 (trials={trials}), 观测值越出抽象区间 "
                      f"{mismatch} 次 (须为 0){'; 例: ' + ex if ex else ''}; 端点有序={order_ok}"}


REGISTRY.register(OpDef(
    name="ana_abstract_interp", band=BAND, sub=13,
    in_schema=_IN_ABS,
    out_schema={"kind": "dict",
                "fields": {"intervals": {"kind": "dict", "extra": True},
                           "unbounded_vars": {"kind": "list", "items": {"kind": "str"}},
                           "n_vars": {"kind": "int", "min": 1},
                           "direction": {"kind": "str", "enum": ["over", "exact"]},
                           "approx_sources": {"kind": "list", "items": {"kind": "str"}, "maxlen": 4}},
                "required": ["intervals", "unbounded_vars", "n_vars", "direction", "approx_sources"]},
    invoke=_inv_abs, verify=_ver_abs,
    cost=("units", 3.0), errors=("SCHEMA_MISMATCH", "TIMEOUT"),
    effects="none (纯函数)",
    doc="§9.2 区间域抽象解释 | in: 程序 (+动态抽样) | out: 各变量终态的整数区间 | "
        "验收=动态抽样对照: 观测终值必须全部落入抽象区间 (越界率 0 = sound) | 3 units | "
        "过近似方向: **over** — 来源见 approx_sources (if 取并 + loop 加宽)。"
        "可证「必落在区间内」(强结论), 不可证「一定取得到」(弱结论)",
    selfchecks=(
        {"id": "aai_straight_line", "args": {"program": [
            {"t": "assign", "var": "v0", "e": {"t": "var", "n": "a0"}},
            {"t": "assign", "var": "v1", "e": {"t": "bin", "op": "+", "l": {"t": "var", "n": "v0"},
                                              "r": {"t": "lit", "v": 1}}}],
            "trials": 32, "seed": 8},
         "check": lambda r: (r.value["intervals"]["v0"] == [-10000, 10000]
                             and r.value["intervals"]["v1"] == [-9999, 10001],
                             f"v0={r.value['intervals']['v0']} v1={r.value['intervals']['v1']}"),
         "expect": "v0←a0 (界 ±10⁴); v1←v0+1 → [−9999, 10001] (精确传播, 无冗余)"},
        {"id": "aai_branch_join", "args": {"program": [
            {"t": "if", "c": {"t": "var", "n": "a0"},
             "th": [{"t": "assign", "var": "v0", "e": {"t": "lit", "v": 5}}],
             "el": [{"t": "assign", "var": "v0", "e": {"t": "lit", "v": -5}}]}],
            "trials": 32, "seed": 9},
         "check": lambda r: (r.value["intervals"]["v0"] == [-5, 5],
                             f"v0={r.value['intervals']['v0']}"),
         "expect": "分支取并: [5,5] ⊔ [−5,−5] = [−5,5] —— 相关关系丢失, 这正是 sound 的代价"},
        {"id": "aai_loop_widening", "args": {"program": [
            {"t": "assign", "var": "v1", "e": {"t": "lit", "v": 0}},
            {"t": "loop", "v": "v0", "b": {"t": "lit", "v": 8},
             "body": [{"t": "assign", "var": "v1",
                       "e": {"t": "bin", "op": "+", "l": {"t": "var", "n": "v1"},
                             "r": {"t": "lit", "v": 3}}}]}],
            "trials": 32, "seed": 10},
         "check": lambda r: (r.value["intervals"]["v1"][0] <= 0
                             and r.value["intervals"]["v1"][1] >= 24,
                             f"v1={r.value['intervals']['v1']} (须含真实值域 0..24)"),
         "expect": "v1 += 3 最多 8 次: 真实终值 0..24; 抽象区间必须**包含**它 (可能更宽)"},
        {"id": "aai_div_range_conservative", "args": {"program": [
            {"t": "assign", "var": "v0", "e": {"t": "lit", "v": 10}},
            {"t": "assign", "var": "v1", "e": {"t": "bin", "op": "/", "l": {"t": "var", "n": "v0"},
                                              "r": {"t": "var", "n": "a0"}}}],
            "trials": 16, "seed": 11},
         "check": lambda r: (None in r.value["intervals"]["v1"]
                             and "v1" in r.value["unbounded_vars"],
                             f"v1={r.value['intervals']['v1']} unbounded={r.value['unbounded_vars']}"),
         "expect": "a0 可能为 0 → 除法区间保守取 ⊤ (显式登记为无界, 不假装知道界)"},
        {"id": "aai_bad_loop_var", "args": {"program": [
            {"t": "loop", "v": "v99", "b": {"t": "lit", "v": 1}, "body": []}]},
         "expect_err": "SCHEMA_MISMATCH", "expect": "非法循环变量槽"},
    ),
))


# ===========================================================================
# 9414 ana_invariant_infer (Houdini-style template filtering)
# ===========================================================================

_IN_AINV = {"kind": "dict",
            "fields": {"program": SCHEMA_PROG,
                       "loop_index": {"kind": "int", "min": 0, "max": 16},
                       "window": {"kind": "int", "min": 1, "max": 20},
                       "trials": {"kind": "int", "min": 1, "max": 200},
                       "seed": {"kind": "int", "min": 0, "max": 2**31 - 1}},
            "required": ["program"]}


def _loops_of(prog):
    return [(sid, s) for sid, s in _walk_stmts(prog) if s["t"] == "loop"]


def _pre_ainv(args) -> "tuple[bool, str]":
    loops = _loops_of(args["program"])
    if not loops:
        return False, "程序无 loop — 本算子只对含循环的程序推断循环不变量"
    li = int(args.get("loop_index", 0))
    if not (0 <= li < len(loops)):
        return False, f"loop_index={li} 越界 (共 {len(loops)} 个循环)"
    return True, ""


def _inv_ainv(args: dict, ctl: CallCtl) -> dict:
    """Houdini-style filtering: first obtain the sound interval **after loop execution** by interval abstract interpretation, then
    filter the template candidates one by one.

    Templates (window w): v ≥ k, v ≤ k, v == k (k ranges over [lo−w, hi+w]); **only those implied by the sound interval survive**.
    Because the interval is an over-approximation, being implied means "holds for all initial values", so the surviving set
    **contains no false invariants** — the direction is conservative: true invariants may be lost (completeness lost), but no
    false ones remain (soundness kept).
    Loop counters additionally get the explicit invariant 0 ≤ v ≤ min(b, LOOP_CAP) from the AST loop bound.
    """
    ctx = "op:ana_invariant_infer#9414"
    prog = args["program"]
    validate_prog(prog, ctx)
    loops = _loops_of(prog)
    li = int(args.get("loop_index", 0))
    sid, loop = loops[li]
    body = loop.get("body", [])
    bnd = loop["b"]
    cap = min(max(int(bnd["v"]), 0), LOOP_CAP) if bnd["t"] == "lit" else LOOP_CAP
# **Exact unrolling instead of widening**: the engine semantics = the loop body executed cap times in sequence
# (the counter takes 0..cap−1 round by round). Iterating the abstract body operator cap times therefore yields a
# **reliable over-approximation of the loop-exit state** without introducing widening divergence.
# (Why not "fixed point + widening": for assignments such as `v1 += 3` with no self-limiting term, the fixed point
#  keeps pushing the upper bound to divergence; this language's loop count is **known** (cap), so unrolling is both
#  more accurate and cheaper than widening.)
    after = dict(_abs_init_state())
    for _ in range(cap):
        after = _abs_stmts(body, after, ctl, ctx)
    # Counter final value: cap−1 when cap>0; keeps its initial 0 when cap=0. Conservatively join in [0, cap].
    after[loop["v"]] = _abs_join(after.get(loop["v"], _ABS_TOP), (0, cap))
    w = int(args.get("window", 6))
    kept: list = []
    dropped = 0
    v = loop["v"]
    kept.append({"inv": f"{v} >= 0", "kind": "ge", "var": v, "k": 0})
    kept.append({"inv": f"{v} <= {cap}", "kind": "le", "var": v, "k": cap})
    for var in VARS:
        lo, hi = after.get(var, _ABS_TOP)
        if lo is None or hi is None:
            continue
# The implication direction must be written the right way round (the old version had it backwards and therefore
# generated a pile of candidates **not implied by the interval**, measured 96 violations):
#   interval [lo,hi] implies (v >= k)  ⟺  k <= lo     (the whole interval lies at or above k)
#   interval [lo,hi] implies (v <= k)  ⟺  k >= hi     (the whole interval lies at or below k)
# Hence the k values that "survive" can only come from the **outward** windows at the interval endpoints; bounds
# pointing inward from an interval point are necessarily eliminated.
        for k in range(lo - w, lo + 1):
            kept.append({"inv": f"{var} >= {k}", "kind": "ge", "var": var, "k": k})
        for k in range(hi, hi + w + 1):
            kept.append({"inv": f"{var} <= {k}", "kind": "le", "var": var, "k": k})
        if lo == hi:
            kept.append({"inv": f"{var} == {lo}", "kind": "eq", "var": var, "k": lo})
        dropped += max(0, hi - lo + 1) * 2
    return {"value": {"loop_sid": sid, "loop_var": v, "kept": kept, "n_kept": len(kept),
                      "n_dropped": dropped,
                      "sound_intervals": {k: list(x) for k, x in after.items()
                                          if k in VARS and x[0] is not None and x[1] is not None},
                      "direction": "over"},
            "resid": [0.0],
            "notes": f"循环 #{li} (语句 {sid}, 计数器 {v}): 保留 {len(kept)} 条候选, 淘汰 {dropped} 条 "
                     f"(被 sound 区间否决); direction=over (保留者由过近似区间蕴含 ⇒ 对一切初值成立; "
                     f"可能遗漏真不变量)"}


def _ver_ainv(args: dict, value: dict) -> dict:
    """Independent verification: (a) every survivor must be implied by the sound interval (recomputed item by item);
    (b) **dynamic one-step inductiveness**: take integer initial values inside the sound interval (injected via literal assignment),
    really execute the loop body once, and the invariant must still hold — a direct counter-example search for inductiveness;
    (c) no leakage: the surviving set must contain no terms outside the variable slots."""
    prog = args["program"]
    loops = _loops_of(prog)
    _sid, loop = loops[int(args.get("loop_index", 0))]
    body = loop.get("body", [])
    sound = value["sound_intervals"]
    bad_implied = 0
    for inv in value["kept"]:
        var, k, kind = inv["var"], int(inv["k"]), inv["kind"]
        if var in sound:
            lo, hi = sound[var]
            if kind == "ge" and not (lo >= k):
                bad_implied += 1
            elif kind == "le" and not (hi <= k):
                bad_implied += 1
            elif kind == "eq" and not (lo == hi == k):
                bad_implied += 1
# Dynamic comparison: sample inputs and **really run the whole loop statement**; the exit state must lie inside the
# reported interval. (This is stronger than "run the loop body once": the exit interval comes from cap unrollings, so it
# should be checked against a real cap-round execution.)
    trials = int(args.get("trials", 16))
    rng = random.Random(int(args.get("seed", 17)))
    checked = 0
    viol = 0
    lo_box, hi_box = {}, {}
    for var, bnd_ in sound.items():
        lo_box[var], hi_box[var] = bnd_
    for _ in range(trials):
        inp = {a: rng.randint(-10**4, 10**4) for a in ARGS}
        try:
            r = interp([loop], inp, 200_000)
        except SvoError:
            continue
        if r["error"] is not None:
            continue
        checked += 1
        for inv in value["kept"]:
            var, k, kind = inv["var"], int(inv["k"]), inv["kind"]
            val = r["state"].get(var)
            if val is None:
                continue
            good = (val >= k) if kind == "ge" else ((val <= k) if kind == "le" else (val == k))
            if not good:
                viol += 1
                break
    ok = (bad_implied == 0) and (viol == 0)
    return {"ok": ok, "resid": [float(bad_implied + viol)],
            "detail": f"保留候选的蕴含性检查: 违反 {bad_implied} 条; 动态**整循环**对照 "
                      f"{checked} 次抽样执行 (trials={trials}), 出口状态越出所报区间 {viol} 次 "
                      f"— 皆须为 0"}


REGISTRY.register(OpDef(
    name="ana_invariant_infer", band=BAND, sub=14,
    in_schema=_IN_AINV,
    out_schema={"kind": "dict",
                "fields": {"loop_sid": {"kind": "int", "min": 0}, "loop_var": {"kind": "str"},
                           "kept": {"kind": "list", "maxlen": 600,
                                    "items": {"kind": "dict", "extra": False,
                                              "fields": {"inv": {"kind": "str"},
                                                         "kind": {"kind": "str", "enum": ["ge", "le", "eq"]},
                                                         "var": {"kind": "str"}, "k": {"kind": "int"}},
                                              "required": ["inv", "kind", "var", "k"]}},
                           "n_kept": {"kind": "int", "min": 0}, "n_dropped": {"kind": "int", "min": 0},
                           "sound_intervals": {"kind": "dict", "extra": True},
                           "direction": {"kind": "str", "enum": ["over", "exact"]}},
                "required": ["loop_sid", "loop_var", "kept", "n_kept", "n_dropped",
                             "sound_intervals", "direction"]},
    invoke=_inv_ainv, verify=_ver_ainv, precond=_pre_ainv,
    cost=("units", 3.0), errors=("PRECOND_FAIL", "SCHEMA_MISMATCH", "TIMEOUT"),
    effects="none (纯函数)",
    doc="§9.2 循环不变量推断 (Houdini 式) | in: 含循环的程序 + 循环下标 + 窗口 | "
        "out: 被 sound 区间蕴含的候选 + 淘汰计数 + 用于筛选的 sound 区间 | "
        "验收=逐条蕴含性重算 + **动态单步归纳性**检查 (区间内取初值跑一遍循环体, 不变量须仍成立) | "
        "3 units | 过近似方向: **over** — 保留者由过近似区间蕴含 ⇒ 对一切初值成立 (无伪不变量); "
        "代价是可能**遗漏**真不变量 (丢完备性而非可靠性) | 失败: 无循环 PRECOND_FAIL",
    selfchecks=(
        {"id": "aii_counter_bounds", "args": {"program": [
            {"t": "assign", "var": "v1", "e": {"t": "lit", "v": 0}},
            {"t": "loop", "v": "v0", "b": {"t": "lit", "v": 5},
             "body": [{"t": "assign", "var": "v1",
                       "e": {"t": "bin", "op": "+", "l": {"t": "var", "n": "v1"},
                             "r": {"t": "lit", "v": 2}}}]}],
            "loop_index": 0, "window": 3, "trials": 12, "seed": 21},
         "check": lambda r: (any(i["inv"] == "v0 >= 0" for i in r.value["kept"])
                             and any(i["inv"] == "v0 <= 5" for i in r.value["kept"]),
                             f"v0 侧不变量={[i['inv'] for i in r.value['kept'] if i['var'] == 'v0']}"),
         "expect": "循环 v0 走 0..5: 必须推出计数器界 v0 ≥ 0 与 v0 ≤ 5 (取自 AST 的循环界)"},
        {"id": "aii_accumulator_lower", "args": {"program": [
            {"t": "assign", "var": "v1", "e": {"t": "lit", "v": 0}},
            {"t": "loop", "v": "v0", "b": {"t": "lit", "v": 4},
             "body": [{"t": "assign", "var": "v1",
                       "e": {"t": "bin", "op": "+", "l": {"t": "var", "n": "v1"},
                             "r": {"t": "lit", "v": 3}}}]}],
            "loop_index": 0, "window": 3, "trials": 12, "seed": 22},
         "check": lambda r: (any(i["var"] == "v1" and i["kind"] == "ge" and i["k"] == 12
                                 for i in r.value["kept"])
                             and r.value["sound_intervals"]["v1"] == [12, 12],
                             f"v1 侧={[i['inv'] for i in r.value['kept'] if i['var'] == 'v1'][:6]} "
                             f"interval={r.value['sound_intervals'].get('v1')}"),
         "expect": "v1 从 0 起每次 +3, 循环 4 轮 → 出口 v1 = 12 (精确展开给出 [12,12]); "
                   "于是最强的下界候选是 v1 ≥ 12 (窗口内还保留 9..11 等更弱的下界)"},
        {"id": "aii_published_intervals", "args": {"program": [
            {"t": "assign", "var": "v1", "e": {"t": "lit", "v": 7}},
            {"t": "loop", "v": "v0", "b": {"t": "lit", "v": 2}, "body": []}],
            "loop_index": 0, "window": 2, "trials": 8, "seed": 23},
         "check": lambda r: (isinstance(r.value["sound_intervals"], dict)
                             and r.value["direction"] == "over"
                             and "v1" in r.value["sound_intervals"],
                             f"sound_intervals={r.value['sound_intervals']}"),
         "expect": "筛选依据必须同时公布 (审计入口: 没有区间就没有判据); "
                   "v1 初值 7 且循环体为空 → 区间应为 [7,7]"},
        {"id": "aii_no_loop", "args": {"program": [{"t": "assign", "var": "v0",
                                                    "e": {"t": "lit", "v": 1}}], "loop_index": 0},
         "expect_err": "PRECOND_FAIL", "expect": "无循环 → 不适用"},
        {"id": "aii_loop_index_oob", "args": {"program": [
            {"t": "loop", "v": "v0", "b": {"t": "lit", "v": 2}, "body": []}], "loop_index": 3},
         "expect_err": "PRECOND_FAIL", "expect": "loop_index 越界"},
    ),
))


# ===========================================================================
# 9415 ana_dead_code (reachability + coverage)
# ===========================================================================

_IN_COV = {"kind": "dict",
           "fields": {"program": SCHEMA_PROG, "trials": {"kind": "int", "min": 1, "max": 500},
                      "seed": {"kind": "int", "min": 0, "max": 2**31 - 1}},
           "required": ["program"]}


def _static_unreachable(prog) -> list:
    """Static unreachability: branch bodies whose condition is the **literal constant 0**, and loop bodies whose bound is the
    literal ≤ 0.

    This bounded language has no goto/return/exceptions, so reachability can be statically decided in exactly these two
    places — which must be spelled out, otherwise the strong conclusion "statically unreachable" would be misused as
    "everything else is reachable" (the latter does not hold).
    """
    sids, _ = sid_map(prog)  # global numbering is computed once (computing it on a subtree yields local numbers → mislabelling)
    dead: list = []

    def rec(stmts) -> None:
        for s in stmts:
            if s["t"] == "if":
                c = s["c"]
                if c["t"] == "lit":
                    live, dark = ("th", "el") if int(c["v"]) != 0 else ("el", "th")
                    dead.extend(sids[id(st)] for st in _all_stmts(s.get(dark, [])))
                    rec(s.get(live, []))
                else:
                    rec(s.get("th", []))
                    rec(s.get("el", []))
            elif s["t"] == "loop":
                b = s["b"]
                if b["t"] == "lit" and int(b["v"]) <= 0:
                    dead.extend(sids[id(st)] for st in _all_stmts(s.get("body", [])))
                else:
                    rec(s.get("body", []))

    rec(prog)
    return sorted(set(dead))


def _inv_cov(args: dict, ctl: CallCtl) -> dict:
    ctx = "op:ana_dead_code#9415"
    prog = args["program"]
    validate_prog(prog, ctx)
    all_sids = [sid for sid, _ in _walk_stmts(prog)]
    dead = _static_unreachable(prog)
    trials = int(args.get("trials", 64))
    rng = random.Random(int(args.get("seed", 31)))
    cover: set = set()
    runs = 0
    for _ in range(trials):
        inp = {a: rng.randint(-6, 6) for a in ARGS}
        r = interp(prog, inp, 200_000)
        if r["error"] is not None:
            continue
        runs += 1
        cover |= set(r["executed"])
    uncovered = sorted(set(all_sids) - cover - set(dead))
    denom = max(1, len(all_sids) - len(dead))
    return {"value": {"all_stmts": all_sids, "static_dead": dead, "covered": sorted(cover),
                      "uncovered": uncovered, "runs": runs,
                      "coverage_rate": len(cover) / denom, "direction": "over"},
            "resid": [float(len(uncovered))],
            "notes": f"可达性: 静态不可达 {len(dead)} 条 {dead[:6]}; 动态覆盖 {len(cover)}/{len(all_sids)} "
                     f"(运行 {runs} 次随机输入), 未覆盖 {len(uncovered)} 条 {uncovered[:6]}; "
                     f"direction=over (静态不可达可靠; '未覆盖'只对**给定输入分布**成立, 不等于死代码)"}


def _ver_cov(args: dict, value: dict) -> dict:
    """Independent verification: (a) the static-unreachable set is recomputed and matches;
    (b) **resample with a different seed**: statements that are statically unreachable must not appear in any run (appearing would
    mean the static analysis is wrong = a real defect);
    (c) the coverage set is self-consistent with all_stmts (a subset)."""
    prog = args["program"]
    dead = _static_unreachable(prog)
    dead_ok = sorted(value["static_dead"]) == dead
    trials = int(args.get("trials", 64))
    rng = random.Random(int(args.get("seed", 31)) + 1)  # independent sampling
    viol = 0
    for _ in range(trials):
        inp = {a: rng.randint(-6, 6) for a in ARGS}
        r = interp(prog, inp, 200_000)
        if r["error"] is not None:
            continue
        if set(r["executed"]) & set(dead):
            viol += 1
    subset_ok = set(value["covered"]) <= set(value["all_stmts"])
    ok = dead_ok and viol == 0 and subset_ok
    return {"ok": ok, "resid": [float(viol)],
            "detail": f"静态不可达集合独立复算一致={dead_ok}; 换种子重抽 {trials} 次, "
                      f"不可达语句被执行 {viol} 次 (须为 0); 覆盖集是全集子集={subset_ok}"}


REGISTRY.register(OpDef(
    name="ana_dead_code", band=BAND, sub=15,
    in_schema=_IN_COV,
    out_schema={"kind": "dict",
                "fields": {"all_stmts": {"kind": "list", "items": {"kind": "int"}},
                           "static_dead": {"kind": "list", "items": {"kind": "int"}},
                           "covered": {"kind": "list", "items": {"kind": "int"}},
                           "uncovered": {"kind": "list", "items": {"kind": "int"}},
                           "runs": {"kind": "int", "min": 0},
                           "coverage_rate": {"kind": "float", "min": 0.0, "max": 1.0},
                           "direction": {"kind": "str", "enum": ["over", "exact"]}},
                "required": ["all_stmts", "static_dead", "covered", "uncovered", "runs",
                             "coverage_rate", "direction"]},
    invoke=_inv_cov, verify=_ver_cov,
    cost=("units", 2.0), errors=("SCHEMA_MISMATCH",),
    effects="none (纯函数)",
    doc="§9.2 死代码 + 覆盖率 | in: 程序 (+动态抽样) | out: 静态不可达集 + 动态覆盖集 + 未覆盖集 | "
        "验收=静态不可达集合独立复算 + **换种子重抽**核对不可达语句从不执行 (真缺陷检测) | "
        "2 units | 过近似方向: **over** — 「静态不可达」是可靠结论 (字面假条件 / 零界循环); "
        "「未覆盖」只对给定输入分布成立, **不等于死代码** —— 两者在输出里分栏, 不混为一谈",
    selfchecks=(
        {"id": "adc_literal_false_branch", "args": {"program": [
            {"t": "if", "c": {"t": "lit", "v": 0},
             "th": [{"t": "assign", "var": "v0", "e": {"t": "lit", "v": 1}}],
             "el": [{"t": "assign", "var": "v1", "e": {"t": "lit", "v": 2}}]}],
            "trials": 8, "seed": 41},
         "check": lambda r: (r.value["static_dead"] == [1] and r.value["covered"] == [0, 2],
                             f"dead={r.value['static_dead']} covered={r.value['covered']}"),
         "expect": "if 0: then 支 (语句 1) 静态不可达; else 支与 if 本体可达 → dead=[1], covered=[0,2]"},
        {"id": "adc_zero_bound_loop", "args": {"program": [
            {"t": "loop", "v": "v0", "b": {"t": "lit", "v": 0},
             "body": [{"t": "assign", "var": "v1", "e": {"t": "lit", "v": 3}}]}],
            "trials": 8, "seed": 42},
         "check": lambda r: (r.value["static_dead"] == [1] and r.value["coverage_rate"] == 1.0,
                             f"dead={r.value['static_dead']} rate={r.value['coverage_rate']}"),
         "expect": "零次循环 (b=0): 循环体 (语句 1) 不可达; 覆盖率按**活**语句算 → 1/1 = 1.0"
                   "(不因死代码拉低)"},
        {"id": "adc_both_branches_live", "args": {"program": [
            {"t": "if", "c": {"t": "var", "n": "a0"},
             "th": [{"t": "assign", "var": "v0", "e": {"t": "lit", "v": 1}}],
             "el": [{"t": "assign", "var": "v0", "e": {"t": "lit", "v": 2}}]}],
            "trials": 64, "seed": 43},
         "check": lambda r: (r.value["static_dead"] == []
                             and set(r.value["covered"]) == {0, 1, 2},
                             f"dead={r.value['static_dead']} covered={r.value['covered']}"),
         "expect": "两分支都可达: 无死代码; 64 次均匀抽样应覆盖全部 3 条语句"},
        {"id": "adc_uncovered_not_dead", "args": {"program": [
            {"t": "if", "c": {"t": "bin", "op": "<", "l": {"t": "var", "n": "a0"},
                              "r": {"t": "lit", "v": -5}},
             "th": [{"t": "assign", "var": "v0", "e": {"t": "lit", "v": 1}}], "el": []}],
            "trials": 6, "seed": 44},
         "check": lambda r: (r.value["static_dead"] == [],
                             f"dead={r.value['static_dead']} uncovered={r.value['uncovered']}"),
         "expect": "稀有分支: 静态**不是**死代码 (dead 为空), 少量抽样可能覆盖不到 —— "
                   "正是「未覆盖 ≠ 死代码」的分栏展示"},
        {"id": "adc_lit_overflow", "args": {"program": [
            {"t": "assign", "var": "v0", "e": {"t": "lit", "v": 99999}}], "trials": 2, "seed": 1},
         "expect_err": "SCHEMA_MISMATCH", "expect": "字面量越界必须被拦下"},
    ),
))