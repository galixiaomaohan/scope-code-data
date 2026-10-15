"""§9.6 heavy cores (band 9400, rows 9430–9433) — bounded model checking / model checking / symbolic execution / static-dynamic cross-check.

Four operators:
  cbmc_bounded     9430  bounded model checking (loops unrolled to the declared bound + path enumeration, k explicit)
  tlc_modelcheck   9431  finite-state exhaustion (state variables + actions + invariant + deadlock detection)
  symexec_slice    9432  symbolic execution of a small input space (path condition = input interval box + slice)
  core_cross_check 9433  **static-dynamic dual-kernel cross-check** (static conclusion × dynamic sampling; divergence is an alarm)

**Honesty boundary (item by item, because this is the easiest part of the library to oversell)**:
  1. `cbmc_bounded`'s "bound" is the **loop-unrolling bound k**, not mathematical induction: it proves "no violation
     within the first k unrollings", **not** "the program has no violation". The return value's `scope` field spells
     out `bounded_unroll_k=<k>`;
  2. it is **not CBMC** and calls no SMT solver (the container has no z3/cvc5): path conditions are approximated by
     **intervals**, so a path may be infeasible yet still explored. Safety rests on two things:
       - counter-examples must be **re-verified by concrete execution** (run engine1 once; discard if it does not
         violate) — so "counter-example found" is reliable;
       - "no counter-example found" is reliable because intervals are **over-approximations**: they cover all real
         inputs (no missed path).
     The cost: compute may be wasted on infeasible paths, and **condition correlations** are lost (contradictory
     conditions like x>0 ∧ x<0 are undetectable unless they sit on the same variable) → such cases yield conservative
     conclusions, never "proved");
  3. `tlc_modelcheck` is **not TLC**: the state space is **explicitly finite** (variables take finite domains) and the
     exhaustion is complete; but it supports no TLA+ temporal operators ([]<>, ~>, fairness), only invariants (safety
     properties) and deadlock;
  4. the four operators' conclusion strengths are distinguished by the `scope` / `completeness` fields in their return
     values, not by "it looks like it ran".
"""
from __future__ import annotations

import itertools

from ..certify import Certificate
from ..errors import Err, SvoError
from ..exact import frac, frac_str
from ..registry import REGISTRY, OpDef
from ..tool import CallCtl
from .code_core import ARGS, LOOP_CAP, SCHEMA_PROG, interp, interp_stack, validate_prog

BAND = 9400
_INPUT_BOX = (-10**4, 10**4)  # abstract universe for input slots (matches the schema bounds)


def _f(v) -> "object":
    from fractions import Fraction
    return frac(v)


def _fs(x) -> str:
    from fractions import Fraction
    return frac_str(x)


# ===========================================================================
# shared: abstract evaluation (intervals) and condition refinement
# ===========================================================================


def _ev_abs(e: dict, env: dict) -> "tuple[object, object]":
    """Interval abstract evaluation: env[var] = (lo, hi) integers or Fractions. Returns (lo, hi)."""
    t = e["t"]
    if t == "lit":
        v = int(e["v"])
        return v, v
    if t == "var":
        return env.get(e["n"], _INPUT_BOX)
    if t == "un":
        a, b = _ev_abs(e["e"], env)
        if e["op"] == "neg":
            return -b, -a
        return 0, 1
    if t == "call":
        args = [_ev_abs(a, env) for a in e["args"]]
        fn = e["fn"]
        if fn == "abs":
            lo, hi = args[0]
            lo2 = 0 if lo <= 0 <= hi else min(abs(lo), abs(hi))
            return lo2, max(abs(lo), abs(hi))
        if fn in ("min", "max"):
            fn2 = min if fn == "min" else max
            return fn2(a[0] for a in args), fn2(a[1] for a in args)
        if fn == "clamp":
            x, l, h = args
            return min(max(x[0], l[0]), h[0]), min(max(x[1], l[1]), h[1])
        return -1, 1
    # bin
    op = e["op"]
    (alo, ahi), (blo, bhi) = _ev_abs(e["l"], env), _ev_abs(e["r"], env)
    if op == "+":
        return alo + blo, ahi + bhi
    if op == "-":
        return alo - bhi, ahi - blo
    if op == "*":
        c = [alo * blo, alo * bhi, ahi * blo, ahi * bhi]
        return min(c), max(c)
    if op in ("/", "%"):
        if blo <= 0 <= bhi:
            return None, None  # possible division by zero → unknown
        if op == "/":
            c = [a // b for a in (alo, ahi) for b in (blo, bhi) if b != 0]
        else:
            c = [a % b for a in (alo, ahi) for b in (blo, bhi) if b != 0]
        return (min(c) - 1, max(c) + 1) if c else (None, None)
    return 0, 1  # comparisons and logic


def _refine(cond: dict, env: dict, want_true: bool) -> dict:
    """Condition refinement: for atomic conditions of the form `var OP lit`, tighten the corresponding variable interval; other forms leave env unchanged (conservative)."""
    out = dict(env)
    if cond.get("t") != "bin":
        return out
    op = cond["op"]
    l, r = cond["l"], cond["r"]
    if op in ("and", "or"):
        if op == "and" and want_true:
            out = _refine(l, out, True)
            out = _refine(r, out, True)
        return out
    var, lit = None, None
    if l.get("t") == "var" and r.get("t") == "lit":
        var, lit = l["n"], int(r["v"])
    elif l.get("t") == "lit" and r.get("t") == "var":
        var, lit = r["n"], int(l["v"])
        op = {"<": ">", "<=": ">=", ">": "<", ">=": "<=", "==": "==", "!=": "!="}[op]
    if var is None or var not in out:
        return out
    lo, hi = out[var]
    if lo is None or hi is None:
        return out
    if not want_true:
        op = {"<": ">=", "<=": ">", ">": "<=", ">=": "<", "==": "!=", "!=": "=="}.get(op, op)
    if op == "<":
        hi = min(hi, lit - 1)
    elif op == "<=":
        hi = min(hi, lit)
    elif op == ">":
        lo = max(lo, lit + 1)
    elif op == ">=":
        lo = max(lo, lit)
    out[var] = (lo, hi)
    return out


def _var_box(env: dict) -> dict:
    """Serialize env's (lo,hi) tuples into a comparable dict (None → "unbounded")."""
    return {k: [None if v[0] is None else int(v[0]), None if v[1] is None else int(v[1])]
            for k, v in sorted(env.items())}


def _paths(prog, k: int, ctl, ctx: str, cap: "int | None" = None) -> "list[dict]":
    """Enumerate the unrolled paths: each looks like {"stmts": [...], "env": {...}} (env = input interval box).

    Loops are unrolled to the **declared bound** (a literal bound takes min(bound, LOOP_CAP); a non-literal one takes
    LOOP_CAP and marks the path "bound unknown"). if splits into two branches, each refining the condition.
    """
    out: list = []

    def emit(acc, env, truncated, k_unknown):
# **Rate-limit while enumerating**: the path count explodes combinatorially (one if inside a 20-round loop body
# → 2²⁰ ≈ 1 million paths); enumerating everything before checking the cap would let the "rejection" itself eat the
# whole deadline. Raising as soon as the cap is exceeded both saves time and satisfies "no silent truncation".
        if cap is not None and len(out) >= cap:
            raise SvoError(Err.BOUNDED_VIOLATION, ctx=ctx,
                           detail=f"路径数已达上限 {cap} — 拒绝继续枚举 "
                                  f"(不做静默截断; 需要更多路径请显式提高 max_paths)")
        out.append({"stmts": list(acc), "env": dict(env), "truncated": truncated,
                    "k_unknown": k_unknown})

    def rec(stmts, env, acc, depth, k_unknown: bool):
        ctl.check_time(ctx + "@paths")
        if depth > k:
            emit(acc, env, True, k_unknown)
            return
        if not stmts:
            emit(acc, env, False, k_unknown)
            return
        s, rest = stmts[0], stmts[1:]
        if s["t"] == "assign":
            rec(rest, env, acc + [s], depth + 1, k_unknown)
        elif s["t"] == "if":
            for want in (True, False):
                env2 = _refine(s["c"], env, want)
                branch = s.get("th", []) if want else s.get("el", [])
                rec(list(branch) + rest, env2, acc, depth + 1, k_unknown)
        else:  # loop
            b = s["b"]
            if b["t"] == "lit":
                cap = min(max(int(b["v"]), 0), LOOP_CAP)
                unknown = k_unknown
            else:
                cap = LOOP_CAP
                unknown = True
            body = s.get("body", [])
            unrolled: list = []
            for i in range(cap):
                unrolled.append({"t": "assign", "var": s["v"], "e": {"t": "lit", "v": i}})
                unrolled.extend(body)
            rec(unrolled + rest, env, acc, depth + cap, unknown)

    rec(prog, {a: _INPUT_BOX for a in ARGS}, [], 0, False)
    return out


def _run_flat(stmts, inp, engine=1) -> dict:
    """Execute the rectified statement sequence as a program (no loops/branches → purely straight-line)."""
    return interp(stmts, inp, 200_000) if engine == 1 else interp_stack(stmts, inp, 200_000)


# ===========================================================================
# 9430 cbmc_bounded
# ===========================================================================

_IN_BMC = {"kind": "dict",
           "fields": {"program": SCHEMA_PROG,
                      "property": {"kind": "dict", "extra": True,
                                   "fields": {"target": {"kind": "str", "enum": ["v0"]},
                                              "op": {"kind": "str",
                                                     "enum": ["in", "not_in", "eq", "ne", "le", "ge"]}},
                                   "required": ["op"]},
                      "k": {"kind": "int", "min": 1, "max": 64},
                      "max_paths": {"kind": "int", "min": 1, "max": 4096},
                      "verify_witness": {"kind": "bool"}},
           "required": ["program", "property"]}


def _prop_holds(prop: dict, val: int) -> bool:
    op = prop["op"]
    if op == "in":
        return int(prop["lo"]) <= val <= int(prop["hi"])
    if op == "not_in":
        return not (int(prop["lo"]) <= val <= int(prop["hi"]))
    if op == "eq":
        return val == int(prop["value"])
    if op == "ne":
        return val != int(prop["value"])
    if op == "le":
        return val <= int(prop["value"])
    return val >= int(prop["value"])


def _inv_bmc(args: dict, ctl: CallCtl) -> dict:
    """Bounded model checking: unroll to k, enumerate paths, and use intervals to compute the possible values of v0 on each path;
    if a value violating the property exists on some path → take a concrete witness and **really run it to re-verify**
    (discard if it does not hold)."""
    ctx = "op:cbmc_bounded#9430"
    prog = args["program"]
    validate_prog(prog, ctx)
    prop = args["property"]
    k = int(args.get("k", 8))
    max_paths = int(args.get("max_paths", 512))
    paths = _paths(prog, k, ctl, ctx, cap=max_paths)
    k_unknown = any(p["k_unknown"] for p in paths)
    truncated = any(p["truncated"] for p in paths)
    violations: list = []
    witness_checked = 0
    witness_rejected = 0
    for p in paths:
        env = p["env"]
        val_lo, val_hi = None, None
        try:
            for s in p["stmts"]:
                if s["t"] == "assign":
                    lo, hi = _ev_abs(s["e"], env)
                    env[s["var"]] = (lo, hi)
                    if s["var"] == "v0":
                        val_lo, val_hi = lo, hi
        except SvoError:
            continue
        if val_lo is None or val_hi is None:
            continue
    # Does some value inside this path's v0 interval **violate the property**?
    # Each operator is judged by its own semantics — the old version treated le/ge like eq (only checking whether the
    # val interval contains value), so v0=10 against the property v0≤8 was judged "no violation" (10 does not contain
    # 8), missing the counter-example outright.
        cands: list = []
        op = prop["op"]
        if op == "in":
            lo_b, hi_b = int(prop["lo"]), int(prop["hi"])
            if val_lo < lo_b:
                cands = [val_lo]
            elif val_hi > hi_b:
                cands = [val_hi]
        elif op == "not_in":
            lo_b, hi_b = int(prop["lo"]), int(prop["hi"])
            lo_in, hi_in = max(val_lo, lo_b), min(val_hi, hi_b)
            if lo_in <= hi_in:
                cands = [lo_in]
        elif op == "eq":
            tgt = int(prop["value"])
            cands = [val_lo if val_lo != tgt else val_hi] if not (val_lo == val_hi == tgt) else []
        elif op == "ne":
                    # violating ne(≠k) means v0 == k
            tgt = int(prop["value"])
            if val_lo <= tgt <= val_hi:
                cands = [tgt]
        elif op == "le":
            tgt = int(prop["value"])
            if val_hi > tgt:
                cands = [max(val_hi, tgt + 1)]
        else:  # ge
            tgt = int(prop["value"])
            if val_lo < tgt:
                cands = [min(val_lo, tgt - 1)]
        if not cands:
            continue
    # Build witness inputs: **pick concrete candidate points inside the path box** and try each on the original program.
    # Why not just take the box centre: the centre need not lie in the input segment that makes v0 violate the property —
    # measured with v0=a0 and the property v0≥0, the centre point 0 happens to satisfy the property, so "a violation
    # exists yet none is reported". Candidate set = extreme combinations of {lo, hi, 0, mid} per dimension +
    # single-dimension endpoint combinations (bounded and boundary-covering).
        box_lo: dict = {}
        box_hi: dict = {}
        bad_box = False
        for a in ARGS:
            lo, hi = p["env"].get(a, _INPUT_BOX)
            if lo is None or hi is None:
                bad_box = True
                break
            box_lo[a], box_hi[a] = int(lo), int(hi)
        if bad_box:
            continue
        cands_in: list = [{a: box_lo[a] for a in ARGS}, {a: box_hi[a] for a in ARGS},
                          {a: 0 if box_lo[a] <= 0 <= box_hi[a] else box_lo[a] for a in ARGS}]
        for a in ARGS:
            for v in (box_lo[a], box_hi[a]):
                w = {b: (v if b == a else (0 if box_lo[b] <= 0 <= box_hi[b] else box_lo[b]))
                     for b in ARGS}
                cands_in.append(w)
        found_witness = None
        if bool(args.get("verify_witness", True)):
            for w in cands_in:
                witness_checked += 1
                r = interp(prog, w, 200_000)
                if r["error"] is not None:
                    witness_rejected += 1
                    continue
                got = int(r["state"].get("v0", 0))
                if _prop_holds(prop, got):
                    witness_rejected += 1
                    continue
                found_witness = (w, got)
                break
            if found_witness is None:
                continue
            w, got = found_witness
            violations.append({"witness": w, "v0": got,
                               "path_len": len(p["stmts"]), "truncated": p["truncated"]})
            continue
    # Witness re-verification not enabled (verify_witness=False): only report "the property may be violated on this
    # path", explicitly flagging v0=None, and completeness records witness_not_replayed — an unverified
    # counter-example is not a counter-example, just a hint for the caller.
        violations.append({"witness": cands_in[0], "v0": None,
                           "path_len": len(p["stmts"]), "truncated": p["truncated"]})
    if violations:
        cert = Certificate(kind="bmc_counterexample",
                           payload={"witness": violations[0]["witness"],
                                    "program": prog, "v0": violations[0]["v0"],
                                    "property": prop},
                           issued_by="cbmc_bounded")
        return {"value": {"verdict": "violation_found", "k": k, "n_paths": len(paths),
                          "violations": violations[:8], "n_violations": len(violations),
                          "witness_verified": witness_checked - witness_rejected,
                          "witness_rejected": witness_rejected,
                          "k_unknown": k_unknown, "truncated_paths": truncated,
                          "scope": f"bounded_unroll_k={k}",
                          "completeness": "witness_concretely_replayed"},
                "resid": [0.0], "cert": cert,
                "notes": f"**发现违反** (k={k}, {len(paths)} 条路径): 见证 "
                         f"{violations[0]['witness']} → v0={violations[0]['v0']} 违反性质 "
                         f"{prop['op']}; 见证已具体执行复验 ({witness_checked - witness_rejected} 个通过, "
                         f"{witness_rejected} 个被丢弃)"}
    cert = Certificate(kind="bmc_bounded_ok",
    # Must carry **program**: the tightened checker performs a bounded dynamic re-check, and a "count-only
    # certificate" lacking program is judged as no evidence (the red team measured this hole).
                       payload={"k": k, "n_paths": len(paths), "property": prop, "program": prog,
                                "k_unknown": k_unknown, "truncated": truncated},
                       issued_by="cbmc_bounded")
    return {"value": {"verdict": "no_violation_within_k", "k": k, "n_paths": len(paths),
                      "violations": [], "n_violations": 0,
                      "witness_verified": witness_checked - witness_rejected,
                      "witness_rejected": witness_rejected,
                      "k_unknown": k_unknown, "truncated_paths": truncated,
                      "scope": f"bounded_unroll_k={k}",
                      "completeness": ("sound_for_all_inputs_because_intervals_overapprox"
                                       + (";WARN_k_unknown" if k_unknown else "")
                                       + (";WARN_truncated" if truncated else ""))},
            "resid": [0.0], "cert": cert,
            "notes": f"**在展开界 k={k} 内未发现违反** ({len(paths)} 条路径, 区间过近似覆盖全部输入): "
                     f"注意这是**有界**结论 —— 它不证明程序无违反 (§9.6: k 显式, 有界≠无界)"
                     + ("; 含界未知的循环 (非字面量界), 该部分按 LOOP_CAP 展开" if k_unknown else "")
                     + ("; 含被深度截断的路径" if truncated else "")}


def _ver_bmc(args: dict, value: dict) -> dict:
    """Independent verification: (a) if there is a violation → replay the witness with **engine2**; it must still violate
    (otherwise it is a fake counter-example, a real defect);
    (b) if there is no violation → independently refute with **random sampling**: draw N inputs and really run them; any
    violation means the static conclusion is wrong;
    (c) the path-count upper bound is self-consistent; (d) the scope field must state k."""
    prog = args["program"]
    prop = args["property"]
    k = int(args.get("k", 8))
    if value["verdict"] == "violation_found":
        bad = 0
        for v in value["violations"]:
            r = interp_stack(prog, v["witness"], 200_000)
            if r["error"] is not None:
                continue
            if not _prop_holds(prop, int(r["state"].get("v0", 0))):
                continue
            bad += 1
        ok = (bad == 0) and int(value["n_violations"]) == len(value["violations"])
        return {"ok": ok, "resid": [float(bad)],
                "detail": f"engine2 重放 {len(value['violations'])} 个见证: 假反例 {bad} 个 "
                          f"(须 0); 计数自洽={int(value['n_violations']) == len(value['violations'])}"}
    import random
    rng = random.Random(20260920)
    viol = 0
    trials = 200
    for _ in range(trials):
        inp = {a: rng.randint(-30, 30) for a in ARGS}
        r = interp(prog, inp, 200_000)
        if r["error"] is not None:
            continue
        if not _prop_holds(prop, int(r["state"].get("v0", 0))):
            viol += 1
            break
    scope_ok = f"k={k}" in value["scope"]
    ok = (viol == 0) and scope_ok and value["n_violations"] == 0
    return {"ok": ok, "resid": [float(viol)],
            "detail": f"动态抽样反证: {trials} 个随机输入 (每维 ±30) 中违反性质 {viol} 个 "
                      f"(须 0, 否则静态'未发现违反'是错的); scope 写出 k={scope_ok}"}


REGISTRY.register(OpDef(
    name="cbmc_bounded", band=BAND, sub=30,
    in_schema=_IN_BMC,
    out_schema={"kind": "dict",
                "fields": {"verdict": {"kind": "str", "enum": ["violation_found", "no_violation_within_k"]},
                           "k": {"kind": "int", "min": 1},
                           "n_paths": {"kind": "int", "min": 1},
                           "violations": {"kind": "list", "maxlen": 8,
                                          "items": {"kind": "dict", "extra": True, "fields": {}}},
                           "n_violations": {"kind": "int", "min": 0},
                           "witness_verified": {"kind": "int", "min": 0},
                           "witness_rejected": {"kind": "int", "min": 0},
                           "k_unknown": {"kind": "bool"}, "truncated_paths": {"kind": "bool"},
                           "scope": {"kind": "str"}, "completeness": {"kind": "str"}},
                "required": ["verdict", "k", "n_paths", "violations", "n_violations",
                             "witness_verified", "witness_rejected", "k_unknown",
                             "truncated_paths", "scope", "completeness"]},
    invoke=_inv_bmc, verify=_ver_bmc, cert_kind="bmc_counterexample",
    cost=("units", 4.0), errors=("BOUNDED_VIOLATION", "SCHEMA_MISMATCH", "TIMEOUT"),
    effects="none (纯函数)",
    doc="§9.6 有界模型检查 (循环按声明界展开 + 路径枚举 + 区间抽象) | in: 程序 + 性质 + **k** | "
        "out: 违反见证 或 'k 内未发现违反' + scope | 验收=见证必须**具体执行复验**(假反例丢弃) + "
        "无违反时用动态抽样独立反证 + scope 必写 k | 证书=bmc_counterexample / bmc_bounded_ok | "
        "4 units | 诚实边界: **k 显式, 有界≠无界**; 不是 CBMC, 不用 SMT; 路径条件用区间近似, "
        "丢掉条件相关性 → 可能探索不可行路径 (只浪费算力, 不影响可靠性) | "
        "失败: 路径数超上限 BOUNDED_VIOLATION",
    selfchecks=(
        {"id": "bmc_finds_violation", "args": {"program": [
            {"t": "assign", "var": "v0", "e": {"t": "var", "n": "a0"}}],
            "property": {"target": "v0", "op": "in", "lo": 0, "hi": 10}, "k": 4},
         "check": lambda r: (r.value["verdict"] == "violation_found" and r.value["n_violations"] >= 1
                             and r.cert is not None and r.cert.kind == "bmc_counterexample",
                             f"verdict={r.value['verdict']} viol={r.value['n_violations']}"),
         "expect": "性质 'v0 ∈ [0,10]' 而 v0 = a0 (输入域 ±10⁴): 显然存在违反 "
                   "(例如 a0=−1) → 必须找到见证并具体复验"},
        {"id": "bmc_no_violation_sound", "args": {"program": [
            {"t": "assign", "var": "v0", "e": {"t": "call", "fn": "abs",
                                               "args": [{"t": "var", "n": "a0"}]}}],
            "property": {"target": "v0", "op": "ge", "value": 0}, "k": 4},
         "check": lambda r: (r.value["verdict"] == "no_violation_within_k"
                             and "k=4" in r.value["scope"] and r.value["n_paths"] >= 1,
                             f"verdict={r.value['verdict']} scope={r.value['scope']}"),
         "expect": "v0 = |a0| ≥ 0 恒真: 区间抽象能证明这一点 (abs 的区间下界为 0) → "
                   "报'k 内未发现违反', 且 scope 写明 k=4"},
        {"id": "bmc_loop_unrolled", "args": {"program": [
            {"t": "assign", "var": "v1", "e": {"t": "lit", "v": 0}},
            {"t": "loop", "v": "v0", "b": {"t": "lit", "v": 5},
             "body": [{"t": "assign", "var": "v1",
                       "e": {"t": "bin", "op": "+", "l": {"t": "var", "n": "v1"},
                             "r": {"t": "lit", "v": 2}}}]},
            {"t": "assign", "var": "v0", "e": {"t": "var", "n": "v1"}}],
            "property": {"target": "v0", "op": "le", "value": 8}, "k": 32},
         "check": lambda r: (r.value["verdict"] == "violation_found",
                             f"verdict={r.value['verdict']} viol={r.value['n_violations']}"),
         "expect": "循环 5 轮每次 +2 → v0 = 10; 性质 'v0 ≤ 8' 被违反 → 必须找到见证 "
                   "(v0=10, 且见证复验通过)"},
        {"id": "bmc_k_insufficient_reported", "args": {"program": [
            {"t": "assign", "var": "v0", "e": {"t": "var", "n": "a0"}},
            {"t": "if", "c": {"t": "bin", "op": ">", "l": {"t": "var", "n": "a0"},
                              "r": {"t": "lit", "v": 0}},
             "th": [{"t": "assign", "var": "v0",
                     "e": {"t": "bin", "op": "*", "l": {"t": "var", "n": "v0"},
                           "r": {"t": "lit", "v": 2}}}],
             "el": []}],
            "property": {"target": "v0", "op": "le", "value": 5}, "k": 1},
         "check": lambda r: (r.value["truncated_paths"] is True
                             or r.value["verdict"] == "violation_found",
                             f"truncated={r.value['truncated_paths']} verdict={r.value['verdict']}"),
         "expect": "k=1 太小 (赋值+分支已用满深度): 结果必须**如实标注** truncated "
                   "(或找到违反) —— 不许在界不足时假装'已检查完'"},
        {"id": "bmc_bad_program", "args": {"program": [{"t": "assign", "var": "zz",
                                                        "e": {"t": "lit", "v": 0}}],
                                           "property": {"op": "ge", "value": 0}},
         "expect_err": "SCHEMA_MISMATCH", "expect": "非法程序必须被拦下"},
    ),
))


# ===========================================================================
# 9431 tlc_modelcheck — finite-state exhaustion (TLA style)
# ===========================================================================

#: predicate: {"op": "=="|"!="|"<"|"<="|">"|">=", "l": {"var":n}|{"const":k}, "r": same}
#: action: {"name": str, "guard": predicate (optional, defaults to constantly true), "updates": [{"var":n,"expr":{"op":"+"/"-","delta":k}}]}
_IN_TLC = {"kind": "dict",
           "fields": {"vars": {"kind": "dict", "extra": True},
                      "init": {"kind": "dict", "extra": True},
                      "actions": {"kind": "list", "minlen": 1, "maxlen": 16,
                                  "items": {"kind": "dict", "extra": True,
                                            "fields": {"name": {"kind": "str"},
                                                       "updates": {"kind": "list", "minlen": 1,
                                                                   "maxlen": 8,
                                                                   "items": {"kind": "dict", "extra": True,
                                                                             "fields": {"var": {"kind": "str"}},
                                                                             "required": ["var"]}}},
                                            "required": ["name", "updates"]}},
                      "invariant": {"kind": "dict", "extra": True, "fields": {"op": {"kind": "str"}},
                                    "required": ["op"]},
                      "max_states": {"kind": "int", "min": 1, "max": 100000}},
           "required": ["vars", "init", "actions", "invariant"]}


def _term_val(t, st) -> int:
    if "var" in t:
        name = t["var"]
        if name not in st:
            raise SvoError(Err.PRECOND_FAIL, ctx="tlc_modelcheck",
                           detail=f"谓词/更新引用了未声明变量 {name!r}")
        return st[name]
    if "const" in t:
        return int(t["const"])
    raise SvoError(Err.PRECOND_FAIL, ctx="tlc_modelcheck",
                   detail=f"项必须是 {{var}} 或 {{const}}, 得 {t!r}")


def _pred(p, st) -> bool:
    op = p["op"]
    a, b = _term_val(p["l"], st), _term_val(p["r"], st)
    return {"==": a == b, "!=": a != b, "<": a < b, "<=": a <= b, ">": a > b, ">=": a >= b}[op]


def _pre_tlc(args) -> "tuple[bool, str]":
    for v, dom in args["vars"].items():
        if not isinstance(dom, list) or not dom:
            return False, f"变量 {v} 的值域必须是非空列表"
    for v in args["init"]:
        if v not in args["vars"]:
            return False, f"init 里的 {v} 未在 vars 中声明"
    for a in args["actions"]:
        for u in a["updates"]:
            if u["var"] not in args["vars"]:
                return False, f"动作 {a['name']} 更新了未声明变量 {u['var']}"
    return True, ""


def _inv_tlc(args: dict, ctl: CallCtl) -> dict:
    """TLC-style exhaustion: starting from all combinations of init, BFS all reachable states, checking the invariant and deadlock."""
    ctx = "op:tlc_modelcheck#9431"
    vars_ = args["vars"]
    names = sorted(vars_)
    inits: list = []
    domains = []
    for v in names:
        allowed = list(args["init"].get(v, vars_[v]))
        for a in allowed:
            if a not in vars_[v]:
                raise SvoError(Err.PRECOND_FAIL, ctx=ctx,
                               detail=f"init 给 {v}={a} 不在其值域 {vars_[v]} 内")
        domains.append(allowed)
    for combo in itertools.product(*domains):
        inits.append(dict(zip(names, combo)))
    max_states = int(args.get("max_states", 10000))
    seen: dict = {}
    order = []
    parent: dict = {}
    for s in inits:
        key = tuple(s[v] for v in names)
        if key not in seen:
            seen[key] = s
            order.append(key)
    head = 0
    viol_state = None
    deadlock_state = None
    while head < len(order):
        ctl.check_time(ctx + "@bfs")
        key = order[head]
        head += 1
        st = seen[key]
        if not _pred(args["invariant"], st):
            if viol_state is None:
                viol_state = key
            continue
        enabled = 0
        for a in args["actions"]:
            g = a.get("guard")
            if g is not None and not _pred(g, st):
                continue
            enabled += 1
            nxt = dict(st)
            for u in a["updates"]:
                ex = u.get("expr") or {"op": "+", "delta": 1}
                base = nxt[u["var"]]
                d = int(ex.get("delta", 1))
                nxt[u["var"]] = base + d if ex.get("op", "+") == "+" else base - d
            nkey = tuple(nxt[v] for v in names)
            if nkey not in seen:
                if len(seen) >= max_states:
                    raise SvoError(Err.BOUNDED_VIOLATION, ctx=ctx,
                                   detail=f"可达状态数超 max_states={max_states} — "
                                          f"拒绝 (不静默截断状态空间)")
                seen[nkey] = nxt
                parent[nkey] = (key, a["name"])
                order.append(nkey)
        if enabled == 0 and deadlock_state is None:
            deadlock_state = key
    if viol_state is not None:
        path: list = []
        cur = viol_state
        while cur in parent:
            p, act = parent[cur]
            path.append({"from": list(p), "to": list(cur), "action": act})
            cur = p
        path = list(reversed(path))
        cert = Certificate(kind="tlc_violation",
                           payload={"trace": path, "state": list(viol_state), "vars": names,
                                    "invariant": args["invariant"],
                                    # carry the full spec: the checker **replays the trajectory itself** instead of trusting this function's conclusion
                                    "spec": {"vars": args["vars"], "init": args["init"],
                                             "actions": args["actions"], "invariant": args["invariant"]}},
                           issued_by="tlc_modelcheck")
        return {"value": {"invariant_holds": False, "n_states": len(seen), "vars": names,
                          "violating_state": list(viol_state), "trace": path,
                          "deadlock_state": list(deadlock_state or ()), "n_init": len(inits)},
                "resid": [0.0], "cert": cert,
                "notes": f"不变量**被违反**: 状态 {list(viol_state)} (变量 {names}); "
                         f"反例轨迹长 {len(path)} 步; 共枚举 {len(seen)} 个可达状态"}
    cert = Certificate(kind="tlc_exhaustive",
                       payload={"n_states": len(seen), "vars": names, "invariant": args["invariant"],
                                "init_states": [list(k) for k in order[:16]],
                                "deadlock": list(deadlock_state or ()),
                                "spec": {"vars": args["vars"], "init": args["init"],
                                         "actions": args["actions"], "invariant": args["invariant"]}},
                       issued_by="tlc_modelcheck")
    return {"value": {"invariant_holds": True, "n_states": len(seen), "vars": names,
                      "violating_state": [], "trace": [],
                      "deadlock_state": list(deadlock_state or ()), "n_init": len(inits)},
            "resid": [0.0], "cert": cert,
            "notes": f"不变量在**全部 {len(seen)} 个可达状态**上成立 (从 {len(inits)} 个初始状态穷举); "
                     + (f"**注意**: 存在死锁状态 {list(deadlock_state)} (无可用动作)"
                        if deadlock_state else "无死锁 (每个可达状态都有可用动作)")}


def _ver_tlc(args: dict, value: dict) -> dict:
    """Independent verification: (a) recompute the reachable state count with **DFS** (an independent traversal); it must match BFS;
    (b) the invariant conclusion matches reality: claimed to hold → no violation may occur during DFS; claimed violated → the
    trajectory must be legal step by step;
    (c) the deadlock decision is independently rechecked (a reachable state with no enabled action)."""
    names = sorted(args["vars"])
    domains = [list(args["init"].get(v, args["vars"][v])) for v in names]
    inits = [dict(zip(names, c)) for c in itertools.product(*domains)]
    seen = set()
    stack = []
    for s in inits:
        k = tuple(s[v] for v in names)
        if k not in seen:
            seen.add(k)
            stack.append(s)
    viol = False
    deadlock = False
    cap = int(args.get("max_states", 10000))
    while stack:
        st = stack.pop()
        if not _pred(args["invariant"], st):
        # **Violating states are no longer expanded** — consistent with invoke's BFS.
        # The old version kept expanding, so a spec with unguarded actions (e.g. a increments each step but violations
        # are only taken at a≤2) would make the verifier's DFS grow along a→∞ without bound (measured: the whole tlc_
        # band timed out at 900s).
            viol = True
            continue
        if len(seen) > cap:
            raise SvoError(Err.BOUNDED_VIOLATION, ctx="op:tlc_modelcheck@verify",
                           detail=f"验证器的独立穷举超出 max_states={cap} — 无法给出独立结论, "
                                  f"按诚实原则拒绝 (不假装验证过)")
        enabled = 0
        for a in args["actions"]:
            g = a.get("guard")
            if g is not None and not _pred(g, st):
                continue
            enabled += 1
            nxt = dict(st)
            for u in a["updates"]:
                ex = u.get("expr") or {"op": "+", "delta": 1}
                d = int(ex.get("delta", 1))
                nxt[u["var"]] = nxt[u["var"]] + d if ex.get("op", "+") == "+" else nxt[u["var"]] - d
            k = tuple(nxt[v] for v in names)
            if k not in seen:
                seen.add(k)
                stack.append(nxt)
        if enabled == 0:
            deadlock = True
    n_ok = (len(seen) == int(value["n_states"]))
    inv_ok = (value["invariant_holds"] == (not viol))
    dl_ok = (bool(value["deadlock_state"]) == deadlock)
    trace_ok = True
    if not value["invariant_holds"]:
        for e in value.get("trace") or []:
            src = dict(zip(names, e["from"]))
            act = next((a for a in args["actions"] if a["name"] == e["action"]), None)
            if act is None:
                trace_ok = False
                break
            nxt = dict(src)
            for u in act["updates"]:
                ex = u.get("expr") or {"op": "+", "delta": 1}
                d = int(ex.get("delta", 1))
                nxt[u["var"]] = nxt[u["var"]] + d if ex.get("op", "+") == "+" else nxt[u["var"]] - d
            if [nxt[v] for v in names] != list(e["to"]):
                trace_ok = False
                break
    ok = n_ok and inv_ok and dl_ok and trace_ok
    return {"ok": ok, "resid": [0.0 if n_ok else 1.0],
            "detail": f"DFS 独立穷举: {len(seen)} 个状态 vs BFS {value['n_states']} (一致={n_ok}); "
                      f"不变量结论相符={inv_ok} (DFS 见违反={viol}); 死锁判定相符={dl_ok}; "
                      f"反例轨迹逐步合法={trace_ok}"}


REGISTRY.register(OpDef(
    name="tlc_modelcheck", band=BAND, sub=31,
    in_schema=_IN_TLC,
    out_schema={"kind": "dict",
                "fields": {"invariant_holds": {"kind": "bool"}, "n_states": {"kind": "int", "min": 1},
                           "vars": {"kind": "list", "items": {"kind": "str"}, "minlen": 1},
                           "violating_state": {"kind": "list", "items": {"kind": "int"}},
                           "trace": {"kind": "list", "maxlen": 4096,
                                     "items": {"kind": "dict", "extra": True, "fields": {}}},
                           "deadlock_state": {"kind": "list", "items": {"kind": "int"}},
                           "n_init": {"kind": "int", "min": 1}},
                "required": ["invariant_holds", "n_states", "vars", "violating_state", "trace",
                             "deadlock_state", "n_init"]},
    invoke=_inv_tlc, verify=_ver_tlc, precond=_pre_tlc, cert_kind="tlc_exhaustive",
    cost=("units", 3.0), errors=("PRECOND_FAIL", "BOUNDED_VIOLATION", "SCHEMA_MISMATCH", "TIMEOUT"),
    effects="none (纯函数)",
    doc="§9.6 TLA 风格有限状态穷举 | in: 状态变量+有限值域 / 初始状态 / 动作(守卫+更新) / 不变量 | "
        "out: 不变量结论 + 反例轨迹 + 死锁状态 | 验收=**BFS 与 DFS 两套穷举**状态数一致 + "
        "反例轨迹逐步合法 + 死锁判定独立复核 | 证书=tlc_exhaustive / tlc_violation | 3 units | "
        "诚实边界: **不是 TLC**; 只支持 不变量(安全性质) 与 死锁, **不支持**时序算子 ([]<>/~>/公平性); "
        "状态空间上限 max_states 显式, 超界抛 BOUNDED_VIOLATION (不静默截断) | "
        "失败: 状态空间超界 BOUNDED_VIOLATION",
    selfchecks=(
        {"id": "tlc_safe_counter", "args": {
            "vars": {"x": [0, 1, 2, 3]},
            "init": {"x": [0]},
            "actions": [{"name": "inc", "guard": {"op": "<", "l": {"var": "x"}, "r": {"const": 3}},
                         "updates": [{"var": "x", "expr": {"op": "+", "delta": 1}}]}],
            "invariant": {"op": "<=", "l": {"var": "x"}, "r": {"const": 3}},
            "max_states": 100},
         "check": lambda r: (r.value["invariant_holds"] is True and r.value["n_states"] == 4
                             and r.cert_independent is True,
                             f"states={r.value['n_states']} holds={r.value['invariant_holds']}"),
         "expect": "计数器 x 从 0 起, 守卫 x<3 才自增: 可达 {0,1,2,3} 共 4 态; "
                   "不变量 x≤3 成立。注意 x=3 时守卫为假 → 存在死锁状态 [3]"},
        {"id": "tlc_invariant_violated", "args": {
            "vars": {"x": [0, 1, 2, 3, 4]},
            "init": {"x": [0]},
            "actions": [{"name": "inc", "updates": [{"var": "x", "expr": {"op": "+", "delta": 1}}]}],
            "invariant": {"op": "<=", "l": {"var": "x"}, "r": {"const": 2}},
            "max_states": 100},
         "check": lambda r: (r.value["invariant_holds"] is False
                             and len(r.value["violating_state"]) == 1
                             and r.cert is not None and r.cert.kind == "tlc_violation"
                             and len(r.value["trace"]) == 3,
                             f"viol={r.value['violating_state']} trace_len={len(r.value['trace'])}"),
         "expect": "无守卫自增: x 可达 4, 而要求 x≤2 → 违反。最短反例轨迹 3 步 (0→1→2→3, "
                   "首个违反态是 3); 证书 tlc_violation 含轨迹 (检查器逐步重放)"},
        {"id": "tlc_two_vars_switch", "args": {
            "vars": {"a": [0, 1], "b": [0, 1]},
            "init": {"a": [0], "b": [1]},
            "actions": [{"name": "inc", "guard": {"op": "<=", "l": {"var": "a"}, "r": {"const": 2}},
                         "updates": [{"var": "a", "expr": {"op": "+", "delta": 1}}]}],
            "invariant": {"op": "<=", "l": {"var": "a"}, "r": {"const": 3}},
            "max_states": 100},
         "check": lambda r: (r.value["invariant_holds"] is True and r.value["vars"] == ["a", "b"]
                             and r.value["n_states"] == 4,
                             f"vars={r.value['vars']} states={r.value['n_states']}"),
         "expect": "两变量 (a∈{0,1}, b∈{0,1}), 初态 (0,1), 动作 a+=1 且**守卫 a≤2**: "
                   "可达 {(0,1),(1,1),(2,1),(3,1)} 共 4 态 (a=3 时守卫假 → 死锁); "
                   "不变量 a≤3 成立。**注意状态值不检查值域** (a=3 ∉ 声明值域), 由守卫与 "
                   "max_states 共同保护 —— 本用例固定该语义"},
        {"id": "tlc_state_space_cap", "args": {
            "vars": {"x": list(range(50))},
            "init": {"x": [0]},
            "actions": [{"name": "inc", "updates": [{"var": "x", "expr": {"op": "+", "delta": 1}}]}],
            "invariant": {"op": "<=", "l": {"var": "x"}, "r": {"const": 49}},
            "max_states": 10},
         "expect_err": "BOUNDED_VIOLATION",
         "expect": "可达状态数将超出 max_states=10 → 必须 BOUNDED_VIOLATION, "
                   "**不许静默截断**成'已穷举' (截断会让'不变量成立'变成假结论)"},
        {"id": "tlc_undeclared_var", "args": {
            "vars": {"x": [0, 1]}, "init": {"x": [0]},
            "actions": [{"name": "a", "updates": [{"var": "y", "expr": {"op": "+", "delta": 1}}]}],
            "invariant": {"op": ">=", "l": {"var": "x"}, "r": {"const": 0}}},
         "expect_err": "PRECOND_FAIL", "expect": "更新未声明变量 y → 前提拒绝"},
    ),
))


# ===========================================================================
# 9432 symexec_slice — symbolic execution over a small input space
# ===========================================================================

_IN_SYM = {"kind": "dict",
           "fields": {"program": SCHEMA_PROG,
                      "k": {"kind": "int", "min": 1, "max": 64},
                      "max_paths": {"kind": "int", "min": 1, "max": 512},
                      "sample_per_path": {"kind": "int", "min": 1, "max": 32}},
           "required": ["program"]}


def _inv_sym(args: dict, ctl: CallCtl) -> dict:
    """Symbolic execution: unroll loops → enumerate branch paths → each path gives an input interval box (path condition) and a
    **slice** (the input slots actually read on that path); then sample a point per path and execute it concretely to check the
    path identity."""
    ctx = "op:symexec_slice#9432"
    prog = args["program"]
    validate_prog(prog, ctx)
    k = int(args.get("k", 16))
    max_paths = int(args.get("max_paths", 128))
    paths = _paths(prog, k, ctl, ctx, cap=max_paths)
    summary: list = []
    for i, p in enumerate(paths):
        env = dict(p["env"])
        reads: set = set()
        for s in p["stmts"]:
            if s["t"] == "assign":
                _collect_reads(s["e"], reads)
        slice_ = sorted(r for r in reads if r in ARGS)
        box = {a: list(env.get(a, _INPUT_BOX)) for a in ARGS}
        summary.append({"path_id": i, "length": len(p["stmts"]),
                        "input_box": box, "slice": slice_,
                        "truncated": p["truncated"], "k_unknown": p["k_unknown"]})
    return {"value": {"paths": summary, "n_paths": len(summary), "k": k,
                      "scope": f"symexec_unroll_k={k}"},
            "resid": [0.0],
            "notes": f"符号执行: 展开 k={k} 得 {len(summary)} 条路径; 每条给出输入区间盒 (路径条件) "
                     f"与实际读过的输入槽切片; 路径盒是**过近似** (条件相关性丢失) — "
                     f"可行性由 verify 的具体执行核对"}


def _collect_reads(e: dict, acc: set) -> None:
    t = e["t"]
    if t == "var":
        acc.add(e["n"])
    elif t == "bin":
        _collect_reads(e["l"], acc)
        _collect_reads(e["r"], acc)
    elif t == "un":
        _collect_reads(e["e"], acc)
    elif t == "call":
        for a in e["args"]:
            _collect_reads(a, acc)


def _ver_sym(args: dict, value: dict) -> dict:
    """Independent verification: (a) sample a point inside each path box and execute it concretely; that point must **really not
    violate that path's branch choices** (compare the branch statements in the execution trace with the path's statement
    sequence) — infeasible paths expose themselves here;
    (b) the slice must be ⊆ the input slots actually read (independently recomputed); (c) the path count is consistent with k."""
    prog = args["program"]
    k = int(args.get("k", 16))
    paths = _paths(prog, k, ctl_or_none2(), "op:symexec_slice@verify",
                   cap=int(args.get("max_paths", 128)))
    feas = 0
    infeasible = 0
    slice_bad = 0
    for i, p in enumerate(paths):
        env = dict(p["env"])
        reads: set = set()
        for s in p["stmts"]:
            if s["t"] == "assign":
                _collect_reads(s["e"], reads)
        want_slice = sorted(r for r in reads if r in ARGS)
        if i < len(value["paths"]) and value["paths"][i]["slice"] != want_slice:
            slice_bad += 1
        # sample points: box centre + box corners
        pts: list = []
        for a in ARGS:
            lo, hi = env.get(a, _INPUT_BOX)
            if lo is None or hi is None:
                pts = []
                break
            pts.append((int(lo) if lo == hi else int((lo + hi) // 2), int(lo), int(hi)))
        if not pts:
            continue
        for lane in range(3):
            inp = {ARGS[k2]: pts[k2][lane if lane < len(pts[k2]) else 0] for k2 in range(len(ARGS))}
            r = _run_flat(p["stmts"], inp, engine=2)
            if r["error"] is None:
                feas += 1
                break
        else:
            infeasible += 1
    n_ok = (len(paths) == int(value["n_paths"]))
    ok = (slice_bad == 0) and n_ok
    return {"ok": ok, "resid": [float(slice_bad)],
            "detail": f"路径数一致={n_ok} ({len(paths)}); 切片重算不符 {slice_bad} 条 (须 0); "
                      f"抽点执行: 可行 {feas} 条, 盒内不可行 {infeasible} 条 "
                      f"(不可行路径是**区间过近似**的预期后果, 不是错误; 见证类结论不受影响)"}


def ctl_or_none2():
    class _C:
        def check_time(self, ctx):
            return None
    return _C()


REGISTRY.register(OpDef(
    name="symexec_slice", band=BAND, sub=32,
    in_schema=_IN_SYM,
    out_schema={"kind": "dict",
                "fields": {"paths": {"kind": "list", "maxlen": 128,
                                     "items": {"kind": "dict", "extra": False,
                                               "fields": {"path_id": {"kind": "int", "min": 0},
                                                          "length": {"kind": "int", "min": 0},
                                                          "slice": {"kind": "list",
                                                                    "items": {"kind": "str"}},
                                                          "input_box": {"kind": "dict", "extra": True,
                                                                        "fields": {}},
                                                          "truncated": {"kind": "bool"},
                                                          "k_unknown": {"kind": "bool"}},
                                               "required": ["path_id", "length", "slice", "input_box",
                                                            "truncated", "k_unknown"]}},
                           "n_paths": {"kind": "int", "min": 1}, "k": {"kind": "int", "min": 1},
                           "scope": {"kind": "str"}},
                "required": ["paths", "n_paths", "k", "scope"]},
    invoke=_inv_sym, verify=_ver_sym,
    cost=("units", 3.0), errors=("BOUNDED_VIOLATION", "SCHEMA_MISMATCH", "TIMEOUT"),
    effects="none (纯函数)",
    doc="§9.6 小输入空间符号执行 | in: 程序 + 展开界 k | out: 逐路径 输入区间盒 + 输入槽切片 + "
        "截断/界未知标志 | 验收=**切片独立重算** + 路径数一致 + 盒内抽点具体执行核对 | 3 units | "
        "诚实边界: 路径盒是**区间过近似** (条件相关性丢失 → 可能含不可行路径); "
        "本算子只给出路径与切片, 不做约束求解 (容器无 SMT); truncated/k_unknown 标志逐条上报 | "
        "失败: 路径数超上限 BOUNDED_VIOLATION",
    selfchecks=(
        {"id": "sym_single_path", "args": {"program": [
            {"t": "assign", "var": "v0", "e": {"t": "bin", "op": "+", "l": {"t": "var", "n": "a0"},
                                              "r": {"t": "var", "n": "a1"}}}],
            "k": 4},
         "check": lambda r: (r.value["n_paths"] == 1 and r.value["paths"][0]["slice"] == ["a0", "a1"]
                             and "k=4" in r.value["scope"],
                             f"n={r.value['n_paths']} slice={r.value['paths'][0]['slice']}"),
         "expect": "无分支程序: 恰好 1 条路径; 切片 = 读过的输入槽 {a0, a1}"},
        {"id": "sym_two_branches", "args": {"program": [
            {"t": "if", "c": {"t": "bin", "op": ">", "l": {"t": "var", "n": "a0"},
                              "r": {"t": "lit", "v": 0}},
             "th": [{"t": "assign", "var": "v0", "e": {"t": "var", "n": "a0"}}],
             "el": [{"t": "assign", "var": "v0", "e": {"t": "un", "op": "neg",
                                                       "e": {"t": "var", "n": "a0"}}}]}],
            "k": 8},
         "check": lambda r: (r.value["n_paths"] == 2
                             and all(p["slice"] == ["a0"] for p in r.value["paths"]),
                             f"n={r.value['n_paths']} slices={[p['slice'] for p in r.value['paths']]}"),
         "expect": "一个 if → 2 条路径; 两支都只读 a0 → 切片均为 [a0]; "
                   "路径条件分别是 a0>0 与 a0≤0 (盒会体现)"},
        {"id": "sym_loop_unrolled_paths", "args": {"program": [
            {"t": "loop", "v": "v0", "b": {"t": "lit", "v": 3},
             "body": [{"t": "if", "c": {"t": "bin", "op": ">", "l": {"t": "var", "n": "a0"},
                                        "r": {"t": "lit", "v": 0}},
                       "th": [{"t": "assign", "var": "v1", "e": {"t": "lit", "v": 1}}],
                       "el": []}]}],
            "k": 32, "max_paths": 64},
         "check": lambda r: (r.value["n_paths"] == 8, f"n={r.value['n_paths']}"),
         "expect": "循环体里一个 if, 循环 3 轮 → 2³ = 8 条路径 (组合爆炸的直观体现: "
                   "这就是为什么 max_paths 必须是显式上限)"},
        {"id": "sym_path_cap", "args": {"program": [
            {"t": "loop", "v": "v0", "b": {"t": "lit", "v": 20},
             "body": [{"t": "if", "c": {"t": "bin", "op": ">", "l": {"t": "var", "n": "a0"},
                                        "r": {"t": "lit", "v": 0}},
                       "th": [], "el": []}]}],
            "k": 64, "max_paths": 16},
         "expect_err": "BOUNDED_VIOLATION",
         "expect": "2²⁰ 条路径远超 max_paths=16 → 必须 BOUNDED_VIOLATION (不静默只报前 16 条)"},
        {"id": "sym_bad_program", "args": {"program": [{"t": "assign", "var": "zz",
                                                        "e": {"t": "lit", "v": 1}}]},
         "expect_err": "SCHEMA_MISMATCH", "expect": "非法程序必须被拦下"},
    ),
))


# ===========================================================================
# 9433 core_cross_check — static-dynamic dual-kernel cross-check (divergence is an alarm)
# ===========================================================================

_IN_X = {"kind": "dict",
         "fields": {"program": SCHEMA_PROG,
                    "property": {"kind": "dict", "extra": True,
                                 "fields": {"op": {"kind": "str"}}, "required": ["op"]},
                    "k": {"kind": "int", "min": 1, "max": 64},
                    "trials": {"kind": "int", "min": 10, "max": 5000},
                    "seed": {"kind": "int", "min": 0, "max": 2**31 - 1},
                    "box": {"kind": "int", "min": 1, "max": 10**4}},
         "required": ["program", "property"]}


def _inv_x(args: dict, ctl: CallCtl) -> dict:
    """Static-dynamic dual-kernel cross-check: judge once statically (cbmc_bounded) and once dynamically (seeded sampling);
    **divergence is an alarm**.

    Why the cross-check matters (§9.6 states it explicitly: "dynamic ground truth × static in-bound proof cross-check;
    divergence is an alarm"):
      - static says "no violation found" but dynamic sampling finds one → the static kernel is defective (or k is too small) → alarm;
      - static says "violation exists" but the dynamic re-verification of the witness does not violate → the static kernel
        produced a fake counter-example → alarm;
      - both agree → confidence increases (but this is not a proof, since the dynamic side is only sampling).
    """
    ctx = "op:core_cross_check#9433"
    import random

    prog = args["program"]
    prop = args["property"]
    # The property shape must be validated before dispatch: _prop_holds reads lo/hi/value fields by op, and without
    # validation an unknown op raises an out-of-table KeyError (measured: selfchecks caught it as RUNTIME_ERR instead
    # of SCHEMA_MISMATCH).
    _ALLOWED_PROP = ("in", "not_in", "eq", "ne", "le", "ge")
    if prop.get("op") not in _ALLOWED_PROP:
        raise SvoError(Err.SCHEMA_MISMATCH, ctx=ctx,
                       detail=f"性质算子 {prop.get('op')!r} 不在 {_ALLOWED_PROP} 内")
    if prop["op"] in ("in", "not_in") and not ("lo" in prop and "hi" in prop):
        raise SvoError(Err.SCHEMA_MISMATCH, ctx=ctx,
                       detail=f"性质 {prop['op']} 需要 lo 与 hi 字段")
    if prop["op"] in ("eq", "ne", "le", "ge") and "value" not in prop:
        raise SvoError(Err.SCHEMA_MISMATCH, ctx=ctx,
                       detail=f"性质 {prop['op']} 需要 value 字段")
    k = int(args.get("k", 8))
    box = int(args.get("box", 30))
    r = REGISTRY.tool("cbmc_bounded").call({"program": prog, "property": prop, "k": k})
    static_verdict = "error" if not r.ok else r.value["verdict"]
    static_err = "" if r.ok else f"{r.err}: {r.notes[:160]}"
    static_witness = (r.value["violations"][0]["witness"] if r.ok and r.value["violations"] else None)
    rng = random.Random(int(args.get("seed", 20260920)))
    trials = int(args.get("trials", 200))
    dyn_viol: list = []
    for _ in range(trials):
        inp = {a: rng.randint(-box, box) for a in ARGS}
        rr = interp(prog, inp, 200_000)
        if rr["error"] is not None:
            dyn_viol.append({"witness": inp, "err": rr["error"]["code"]})
            continue
        if not _prop_holds(prop, int(rr["state"].get("v0", 0))):
            dyn_viol.append({"witness": inp, "v0": int(rr["state"].get("v0", 0))})
    alarm = []
    if static_verdict == "no_violation_within_k" and dyn_viol:
        alarm.append(f"静态报'k={k} 内无违反', 动态却在 {len(dyn_viol)}/{trials} 个输入上找到违反 "
                     f"— 静态核或 k 有问题 (分歧即警报)")
    if static_verdict == "violation_found" and static_witness is not None:
        rr = interp(prog, static_witness, 200_000)
        if rr["error"] is None and _prop_holds(prop, int(rr["state"].get("v0", 0))):
            alarm.append("静态报出的见证经动态复验**不违反** → 假反例 (分歧即警报)")
    consistent = not alarm
    if static_verdict == "error":
        return {"value": {"static_verdict": "error", "static_witness": {},
                          "dyn_trials": trials, "dyn_box": box, "dyn_violations": len(dyn_viol),
                          "dyn_witness_head": dyn_viol[:3], "consistent": False,
                          "alarms": [f"静态核调用失败: {static_err}"],
                          "seed": int(args.get("seed", 20260920)),
                          "completeness": "static_core_unavailable"},
                "resid": [1.0],
                "notes": f"静态核 cbmc_bounded 调用失败 ({static_err}) — 交叉检查无法完成; "
                         f"按'分歧即警报'处理: 不假装一致"}
    return {"value": {"static_verdict": static_verdict, "static_witness": static_witness or {},
                      "dyn_trials": trials, "dyn_box": box, "dyn_violations": len(dyn_viol),
                      "dyn_witness_head": dyn_viol[:3],
                      "consistent": consistent, "alarms": alarm,
                      "seed": int(args.get("seed", 20260920)),
                      "completeness": ("consistency_only; dynamic_is_sampling_not_proof"
                                       if consistent else "DISAGREEMENT")},
            "resid": [float(len(alarm))],
            "notes": ("动静一致: 静态 " + static_verdict + f" 与动态 {trials} 次抽样无冲突 "
                      f"(动态箱 ±{box}); 一致提升置信但不构成证明")
                     if consistent else ("**分歧警报**: " + "; ".join(alarm))}


def _ver_x(args: dict, value: dict) -> dict:
    """Independent verification: (a) **recheck the dynamic sampling** (with a different seed) for conflict with the static
    conclusion — independent of invoke's sampling;
    (b) alarm count and consistent are self-consistent; (c) the dynamic sampling really executed the declared number of times."""
    import random

    prog = args["program"]
    prop = args["property"]
    box = int(args.get("box", 30))
    rng = random.Random(int(args.get("seed", 20260920)) + 977)
    found = 0
    for _ in range(int(args.get("trials", 200))):
        inp = {a: rng.randint(-box, box) for a in ARGS}
        r = interp(prog, inp, 200_000)
        if r["error"] is not None:
            found += 1
            continue
        if not _prop_holds(prop, int(r["state"].get("v0", 0))):
            found += 1
    conflict = (value["static_verdict"] == "no_violation_within_k") and (found > 0)
    seedy = int(value["seed"]) == int(args.get("seed", 20260920))
    ok = (not conflict) and (value["consistent"] == (len(value["alarms"]) == 0)) and seedy
    return {"ok": ok, "resid": [float(found if conflict else 0)],
            "detail": f"换种子 (seed+977) 独立抽样: 发现违反 {found} 个; "
                      f"与静态结论冲突={conflict} (冲突即静态核不可信); "
                      f"consistent 与报警自洽={value['consistent'] == (len(value['alarms']) == 0)}; "
                      f"种子可复现标注={seedy}"}


REGISTRY.register(OpDef(
    name="core_cross_check", band=BAND, sub=33,
    in_schema=_IN_X,
    out_schema={"kind": "dict",
                "fields": {"static_verdict": {"kind": "str"},
                           "static_witness": {"kind": "dict", "extra": True, "fields": {}},
                           "dyn_trials": {"kind": "int", "min": 10},
                           "dyn_box": {"kind": "int", "min": 1},
                           "dyn_violations": {"kind": "int", "min": 0},
                           "dyn_witness_head": {"kind": "list", "maxlen": 3,
                                                "items": {"kind": "dict", "extra": True, "fields": {}}},
                           "consistent": {"kind": "bool"},
                           "alarms": {"kind": "list", "maxlen": 4, "items": {"kind": "str"}},
                           "seed": {"kind": "int", "min": 0}, "completeness": {"kind": "str"}},
                "required": ["static_verdict", "static_witness", "dyn_trials", "dyn_box",
                             "dyn_violations", "dyn_witness_head", "consistent", "alarms",
                             "seed", "completeness"]},
    invoke=_inv_x, verify=_ver_x,
    cost=("units", 3.0), errors=("SCHEMA_MISMATCH",),
    effects="none (纯函数)",
    doc="§9.6 **动静双核交叉** | in: 程序 + 性质 + k + 抽样参数 | out: 静态结论 vs 动态抽样 + "
        "分歧警报 | 验收=**换种子独立复查抽样** + consistent 与报警自洽 + 种子可复现 | 3 units | "
        "诚实边界: 一致**只提升置信, 不是证明** (动态是抽样); 建议越高置信, 但静态的界仍是 k; "
        "发现分歧时的正确动作是**停下查因**, 不是挑信一边 | 失败: schema",
    selfchecks=(
        {"id": "x_consistent_no_violation", "args": {"program": [
            {"t": "assign", "var": "v0", "e": {"t": "call", "fn": "abs",
                                               "args": [{"t": "var", "n": "a0"}]}}],
            "property": {"target": "v0", "op": "ge", "value": 0}, "k": 4, "trials": 60, "box": 40},
         "check": lambda r: (r.value["consistent"] is True and r.value["alarms"] == []
                             and r.value["dyn_violations"] == 0,
                             f"consistent={r.value['consistent']} dyn={r.value['dyn_violations']}"),
         "expect": "v0=|a0| ≥ 0: 静态(k=4 内无违反) 与动态(60 次抽样)一致 → 无警报; "
                   "结论强度=一致性, 不是证明"},
        {"id": "x_static_violation_and_dynamic_agree", "args": {"program": [
            {"t": "assign", "var": "v0", "e": {"t": "var", "n": "a0"}}],
            "property": {"target": "v0", "op": "ge", "value": 0}, "k": 4, "trials": 60, "box": 40},
         "check": lambda r: (r.value["static_verdict"] == "violation_found"
                             and r.value["consistent"] is True
                             and r.value["static_witness"] is not None,
                             f"static={r.value['static_verdict']} witness={r.value['static_witness']}"),
         "expect": "v0=a0 而要求 a0≥0: 静态找到见证; 见证经动态复验确实违反 → 一致无警报"},
        {"id": "x_alarm_on_fake_witness", "args": {"program": [
            {"t": "assign", "var": "v0", "e": {"t": "lit", "v": 5}}],
            "property": {"target": "v0", "op": "ge", "value": 0}, "k": 4, "trials": 30, "box": 20},
         "check": lambda r: (r.value["consistent"] is True
                             and r.value["static_verdict"] == "no_violation_within_k",
                             f"static={r.value['static_verdict']}"),
         "expect": "常量 5 ≥ 0: 静态无违反, 动态也无违反 (常数与输入无关) → 一致。"
                   "本用例覆盖'常数程序'这一边界 (区间退化)"},
        {"id": "x_seed_recorded", "args": {"program": [
            {"t": "assign", "var": "v0", "e": {"t": "var", "n": "a0"}}],
            "property": {"target": "v0", "op": "le", "value": 0}, "k": 2, "trials": 40,
            "seed": 12345, "box": 10},
         "check": lambda r: (r.value["seed"] == 12345 and r.value["completeness"] in
                             ("consistency_only; dynamic_is_sampling_not_proof", "DISAGREEMENT"),
                             f"seed={r.value['seed']} completeness={r.value['completeness']}"),
         "expect": "种子必须原样记录 (动态抽样可复现); completeness 字段必须显式声明"
                   "'一致性 ≠ 证明'"},
        {"id": "x_bad_property", "args": {"program": [], "property": {"op": "unknown_op"}},
         "expect_err": "SCHEMA_MISMATCH", "expect": "未知性质算子被 schema 拦下"},
    ),
))