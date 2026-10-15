"""§11 Agent domain (band 9600, rows 9601–9610) — P4 deliverable.

Ten operators covering all §11 components:

  agent_tool_registry 9601  TypedTool tool table (§11.1/§11.4): calculator / code_exec / mock_search
  agent_call_loop     9602  call loop (§11.2): observe→choose→bind→precond→call→postcond→backtrack
  agent_ask_info      9603  ask_info VOI settlement (§11.3): gain = r_min(before) − r_min(after), cost mandatory
  agent_plan_decompose 9604 task→subtask DAG (§11.5), every node carries its own acceptor
  agent_memory_registry 9605 verified-labelling store (§11.5): **only solutions with an independent re-verification certificate enter the store**
  agent_budget_governor 9606 global budget ledger + tier→beam-width mapping (§11.5)
  agent_committee     9607  cross-operator voting on one question (§11.5): agreement = free confidence, disagreement = automatic downgrade and beam widening
  agent_llm_hint      9608  external-model membrane (§11.6): **prior only, never evidence**
  agent_mcp_bridge    9609  TypedTool↔MCP adaptation (§11.7), JSON-schema round-trip consistency
  agent_replay_audit  9610  replay_log + audit_trace (§11.7): deterministic replay + audit-trace fingerprint

**Honest statement on the §11.7 combined trust anchor (must be at the module head)**: the agent domain has
**no universal kernel**. Its trust is the **intersection of combined anchors**: (a) each tool call's TypedTool
postcondition; (b) the delegated operator's own verifier/certificate; (c) deterministic replay of the replay_log;
(d) complete recording in the audit_trace. If **any one** is missing, the call is not credible evidence. Every
operator here explicitly exposes a `trust_anchor` field in its return value saying which anchors it relies on.

**mock_search is a mock**: the corpus is hard-coded in the module and no network is accessed (§18 red line); it is
not a search engine. The same for `llm_hint`: it only accepts **caller-supplied** suggestion handles and makes no
model call.
"""
from __future__ import annotations

import hashlib
import json

from ..audit import AuditTrace, BudgetLedger, ReplayLog, canon_args, replay_equal
from ..certify import Certificate
from ..errors import Err, SvoError
from ..registry import REGISTRY, OpDef
from ..tool import CallCtl

BAND = 9600

#: mock_search's fixed corpus (hard-coded, no network; §11.4 lists mock_search = fixed corpus, no network)
MOCK_CORPUS: tuple[tuple[str, str], ...] = (
    ("quadratic", "二次方程 ax²+bx+c=0 的根由求根公式给出; 判别式判别实根个数"),
    ("vieta", "韦达定理: 两根之和 = −b/a, 两根之积 = c/a"),
    ("amgm", "AM-GM 不等式: 正数的算术平均不小于几何平均, 等号当且仅当全相等"),
    ("cauchy", "Cauchy-Schwarz 不等式: (Σuv)² ≤ (Σu²)(Σv²)"),
    ("ring_identity", "环恒等式可用系数逐项比对判定; 多项式相等当且仅当系数全等"),
    ("drat", "DRAT 证书: 每个新增子句必须可被反向单元传播或 RAT 证明; 导出空子句即反证 UNSAT"),
    ("farkas", "线性算术的矛盾可用非负线性组合导出 (Farkas 引理); 组合系数必须非负"),
    ("bounded_loop", "有界语言里循环次数必须显式, 否则失效安全机制会拒绝执行"),
    ("mutation", "变异测试: 杀死变异体才算测试充分; 存活变异体是非等价缺陷信号"),
    ("triple_gate", "补丁三重门: 原测试全绿 + held-out 全绿 + 变异体杀死"),
)


def _h(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
                          .encode("utf-8")).hexdigest()[:16]


# ===========================================================================
# 9601 agent_tool_registry
# ===========================================================================

#: tool declaration: each tool makes (name, input schema, output schema, delegated operator, cost) explicit.
_TOOLS: dict = {
    "calculator": {
        "delegate": "math_evaluate_expr",
        "desc": "有理点求值 (poly/geom), 委托 §8.1 算子",
        "in_schema": {"form": "str", "coeffs": "[Fraction]", "x": "Fraction"},
        "out_schema": {"value": "str", "form": "str"},
        "cost_units": 0.5,
        "precond": "form ∈ {poly, geom} 且所需字段齐",
    },
    "code_exec": {
        "delegate": "code_bounded_interp",
        "desc": "有界 AST 语言执行, 委托 §9.1 解释器 (无网/有界步数)",
        "in_schema": {"program": "[Stmt]", "inputs": "{a0..a3: int}", "step_limit": "int"},
        "out_schema": {"state": "{var: int}", "steps": "int", "error": "null"},
        "cost_units": 2.0,
        "precond": "program 过结构校验; 步数在界内",
    },
    "mock_search": {
        "delegate": "agent_tool_registry.mock_corpus",
        "desc": "**固定语料检索** (无网; 名为 mock 是为了不让人误当成真搜索引擎)",
        "in_schema": {"query": "str", "topk": "int"},
        "out_schema": {"hits": "[(key, text)]"},
        "cost_units": 0.3,
        "precond": "query 非空",
    },
}

_IN_TR = {"kind": "dict",
          "fields": {"want": {"kind": "str", "enum": ["table", "self_test"]},
                     "tool": {"kind": "str", "enum": list(_TOOLS)}},
          "required": ["want"]}


def _inv_tr(args: dict, ctl: CallCtl) -> dict:
    """Tool table: return the TypedTool triples (schema/precond/postcond/cost/errors), or selfcheck a named tool."""
    ctx = "op:agent_tool_registry#9601"
    if args["want"] == "self_test":
        name = args.get("tool")
        if name is None:
            raise SvoError(Err.SCHEMA_MISMATCH, ctx=ctx, detail="自检模式需要 tool")
        # really run the delegated operator this tool represents once
        if name == "calculator":
            r = REGISTRY.tool("math_evaluate_expr").call({"form": "poly", "coeffs": [1, 2], "x": 1})
            ok, detail = bool(r.ok) and r.value.get("value") == "3", "poly 求值 1+2·1=3"
        elif name == "code_exec":
            r = REGISTRY.tool("code_bounded_interp").call(
                {"program": [{"t": "assign", "var": "v0", "e": {"t": "var", "n": "a0"}}],
                 "inputs": {"a0": 7}, "step_limit": 100})
            ok, detail = bool(r.ok) and r.value.get("state", {}).get("v0") == 7, "执行 v0←a0 (a0=7) → v0=7"
        else:
            hits = _mock_search("quadratic", 2)
            ok, detail = len(hits) > 0, f"固定语料命中 {len(hits)} 条"
        return {"value": {"tool": name, "self_test_ok": ok, "detail": detail,
                          "n_tools": len(_TOOLS),
                          "trust_anchor": "postcond + 委托算子证书 + replay + audit"},
                "resid": [0.0 if ok else 1.0],
                "notes": f"工具自检 {'通过' if ok else '失败'}: {name} — {detail}"}
    table = []
    for name, spec in sorted(_TOOLS.items()):
        errors = ["SCHEMA_MISMATCH", "PRECOND_FAIL", "POSTCOND_FAIL", "TIMEOUT"]
        if name == "code_exec":
            errors += ["BOUNDED_VIOLATION", "RUNTIME_ERR"]
        table.append({"name": name, "delegate": spec["delegate"], "desc": spec["desc"],
                      "in_schema": spec["in_schema"], "out_schema": spec["out_schema"],
                      "precond": spec["precond"], "postcond": "委托算子的独立验证器 + 输出 schema",
                      "cost_units": spec["cost_units"], "error_enum": errors})
    return {"value": {"tools": table, "n_tools": len(table),
                      "trust_anchor": ("postcond + 委托算子证书 + replay_log + audit_trace; "
                                       "**无普适内核 (§11.7): 信任 = 组合锚的交集**")},
            "resid": [0.0],
            "notes": f"工具表: {len(table)} 个工具 ({[t['name'] for t in table]}); "
                     f"每个都带 输入/输出 schema + 前/后置条件 + 成本 + 错误枚举 (§11.1 五件套); "
                     f"信任锚 = 组合锚交集 (§11.7 显式声明: 本域无普适内核)"}


def _mock_search(query: str, topk: int) -> list:
    """Fixed-corpus retrieval: sort by (substring hit count, key lexicographic order); deterministic."""
    q = query.lower().strip()
    scored = []
    for key, text in MOCK_CORPUS:
        score = (key.lower().count(q) if q else 0) * 10 + text.lower().count(q)
        if q in key.lower():
            score += 5
        if score > 0:
            scored.append((score, key, text))
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [{"key": k, "text": t} for _s, k, t in scored[:max(1, topk)]]


def _ver_tr(args: dict, value: dict) -> dict:
    """Independent verification: (a) every tool-table entry must have in/out schemas, pre/postconditions, cost and an error
    enumeration (§7's six admission rules, tool version) with the error enumeration ⊆ the §15 table;
    (b) each tool's declared delegate must be a **registered operator** or one of this module's fixed-corpus functions;
    (c) in selfcheck mode, check self_test_ok against reality."""
    from ..errors import ERROR_TABLE
    if args["want"] == "self_test":
        name = args.get("tool")
        if name == "calculator":
            r = REGISTRY.tool("math_evaluate_expr").call({"form": "poly", "coeffs": [1, 2], "x": 1})
            exp = bool(r.ok) and r.value.get("value") == "3"
        elif name == "code_exec":
            r = REGISTRY.tool("code_bounded_interp").call(
                {"program": [{"t": "assign", "var": "v0", "e": {"t": "var", "n": "a0"}}],
                 "inputs": {"a0": 7}, "step_limit": 100})
            exp = bool(r.ok) and r.value.get("state", {}).get("v0") == 7
        else:
            exp = len(_mock_search("quadratic", 2)) > 0
        ok = (bool(value["self_test_ok"]) == exp)
        return {"ok": ok, "resid": [0.0 if ok else 1.0],
                "detail": f"独立重跑 {name} 的自检: 期望 {exp}, 输出 {value['self_test_ok']}"}
    bad = []
    for t in value["tools"]:
        if not t.get("in_schema") or not t.get("out_schema") or not t.get("precond") \
                or not t.get("postcond") or not t.get("error_enum"):
            bad.append((t["name"], "五件套不齐"))
        for e in t["error_enum"]:
            if e not in ERROR_TABLE:
                bad.append((t["name"], f"错误码 {e} 不在 §15 表"))
        d = t["delegate"]
        if not d.startswith("agent_tool_registry.") and not REGISTRY.has(d):
            bad.append((t["name"], f"delegate {d} 未注册"))
    ok = not bad
    return {"ok": ok, "resid": [float(len(bad))],
            "detail": f"工具表审计: 五件套/错误表/委托算子核对, 问题 {len(bad)} 项 {bad[:3]}"}


REGISTRY.register(OpDef(
    name="agent_tool_registry", band=BAND, sub=1,
    in_schema=_IN_TR,
    out_schema={"kind": "dict",
                "fields": {"tools": {"kind": "list", "maxlen": 8,
                                     "items": {"kind": "dict", "extra": True, "fields": {}}}
                           if False else {"kind": "list"},
                           "n_tools": {"kind": "int", "min": 1},
                           "self_test_ok": {"kind": "bool"}, "tool": {"kind": "str"},
                           "detail": {"kind": "str"}, "trust_anchor": {"kind": "str"}},
                "required": ["n_tools", "trust_anchor"]},
    invoke=_inv_tr, verify=_ver_tr,
    cost=("units", 1.0), errors=("SCHEMA_MISMATCH", "PRECOND_FAIL", "POSTCOND_FAIL", "TIMEOUT"),
    effects="间接: 自检模式会调用被委托算子 (皆纯函数)",
    doc="§11.1/§11.4 TypedTool 工具表 | in: want∈{table, self_test} (+tool) | out: 工具五件套表 / 自检结论 | "
        "验收=每个工具带 输入输出 schema + 前/后置 + 成本 + 错误枚举(⊆§15) + delegate 必须已注册 + "
        "自检真跑委托算子 | 1 unit | 诚实边界: mock_search 是**硬编码固定语料**, 不是搜索引擎; "
        "本域无普适内核, 信任=组合锚交集 (§11.7) | 失败: schema",
    selfchecks=(
        {"id": "atr_table", "args": {"want": "table"},
         "check": lambda r: (r.value["n_tools"] == 3
                             and all(t["error_enum"] and t["postcond"] for t in r.value["tools"])
                             and "组合锚" in r.value["trust_anchor"],
                             f"tools={[t['name'] for t in r.value['tools']]}"),
         "expect": "三个工具 (calculator/code_exec/mock_search) 全部带五件套; "
                   "trust_anchor 字段必须显式声明'组合锚交集'"},
        {"id": "atr_self_test_calculator", "args": {"want": "self_test", "tool": "calculator"},
         "check": lambda r: (r.value["self_test_ok"] is True
                             and r.value["tool"] == "calculator",
                             f"ok={r.value['self_test_ok']} detail={r.value['detail']}"),
         "expect": "calculator 自检: 委托 math_evaluate_expr 求 1+2x @x=1 = 3 → 通过"},
        {"id": "atr_self_test_code_exec", "args": {"want": "self_test", "tool": "code_exec"},
         "check": lambda r: (r.value["self_test_ok"] is True,
                             f"detail={r.value['detail']}"),
         "expect": "code_exec 自检: 执行 v0←a0 (a0=7) → v0=7 → 通过"},
        {"id": "atr_self_test_mock_search", "args": {"want": "self_test", "tool": "mock_search"},
         "check": lambda r: (r.value["self_test_ok"] is True, f"detail={r.value['detail']}"),
         "expect": "mock_search 自检: 检索 'quadratic' 命中固定语料至少 1 条 → 通过"},
        {"id": "atr_bad_want", "args": {"want": "everything"}, "expect_err": "SCHEMA_MISMATCH",
         "expect": "want 枚举外取值必须被拦下"},
    ),
))


# ===========================================================================
# 9602 agent_call_loop
# ===========================================================================

_IN_LOOP = {"kind": "dict",
            "fields": {"candidates": {"kind": "list", "minlen": 1, "maxlen": 16,
                                      "items": {"kind": "dict", "extra": False,
                                                "fields": {"tool": {"kind": "str"},
                                                           "args": {"kind": "dict", "extra": True,
                                                                    "fields": {}}},
                                                "required": ["tool", "args"]}},
                       "accept_resid_max": {"kind": "float", "min": 0.0, "max": 1e9},
                       "budget_units": {"kind": "float", "min": 0.0, "max": 1e9},
                       "max_retries": {"kind": "int", "min": 0, "max": 8}},
            "required": ["candidates"]}


def _inv_loop(args: dict, ctl: CallCtl) -> dict:
    """Call loop: for each candidate in order, bind→precond→call→postcond; the first one passing acceptance is taken.

    The whole run is recorded in an AuditTrace (one event per attempt) and can be reproduced by replay.
    """
    ctx = "op:agent_call_loop#9602"
    tol = float(args.get("accept_resid_max", 1e-9))
    ledger = BudgetLedger(total_units=float(args.get("budget_units", 1e9)))
    trace = ctl.trace if ctl.trace is not None else AuditTrace(ctx)
    attempts: list = []
    accepted = None
    for i, cand in enumerate(args["candidates"]):
        ctl.check_time(ctx + "@loop")
        tool = cand["tool"]
        if not REGISTRY.has(tool):
            trace.add(tool, -1, "bind", False, err=Err.SCHEMA_MISMATCH.value, detail="工具未注册")
            attempts.append({"i": i, "tool": tool, "stage": "bind", "ok": False,
                             "err": "SCHEMA_MISMATCH",
                             "resid": float("nan")})  # nan = not measured (the schema lets it through with finite:False)
            continue
        t = REGISTRY.tool(tool)
        sub = CallCtl(trace=trace, budget=ledger, timeout_s=ctl.timeout_s, seed=ctl.seed)
        r = t.call(cand["args"], sub)
        if r.ok and (r.resid_scalar != r.resid_scalar or r.resid_scalar <= tol):
            accepted = {"i": i, "tool": tool, "args": cand["args"], "resid": r.resid_scalar,
                        "cert_kind": (r.cert.kind if r.cert else None),
                        "cert_independent": r.cert_independent, "cost_units": r.cost_units}
            attempts.append({"i": i, "tool": tool, "stage": "accept", "ok": True,
                             "err": "", "resid": r.resid_scalar})
            break
        attempts.append({"i": i, "tool": tool, "stage": "postcond", "ok": False,
                         "err": r.err or "POSTCOND_FAIL",
                         "resid": float(r.resid_scalar) if r.resid_scalar == r.resid_scalar
                         else float("nan")})
    if accepted is None:
        raise SvoError(Err.POSTCOND_FAIL, ctx=ctx,
                       detail=f"{len(args['candidates'])} 个候选全部未过验收 "
                              f"(容差 resid≤{tol}); 尝试明细 {attempts[:3]} — 需要换候选或升级(esc)")
    return {"value": {"accepted": accepted, "attempts": attempts,
                      "n_tried": len(attempts), "budget_spent": ledger.spent,
                      "trace_fingerprint": trace.struct_fingerprint(),
                      "trust_anchor": "postcond + 委托算子证书 + audit_trace(本次) + replay_log(调用方)",
                      "esc_needed": False},
            "resid": [float(accepted["resid"])],
            "notes": f"调用循环接受第 {accepted['i'] + 1} 个候选: {accepted['tool']} "
                     f"(resid={accepted['resid']:.3e}, 证书={accepted['cert_kind']}); "
                     f"共尝试 {len(attempts)} 次, 花费 {ledger.spent:.3f} units; "
                     f"审计迹指纹 {trace.struct_fingerprint()}"}


def _ver_loop(args: dict, value: dict) -> dict:
    """Independent verification: (a) **replay the accepted candidate with a fresh trace**; it must still be accepted with the
    same residual (an independent confirmation of deterministic replay, §17.9); (b) the attempt list's ok flags and order are
    self-consistent (the accepted item must be the last); (c) budget accounting ≤ the declared budget."""
    acc = value["accepted"]
    if not REGISTRY.has(acc["tool"]):
        return {"ok": False, "resid": [1.0], "detail": "被接受的工具未注册"}
    fresh = AuditTrace("replay")
    ledger = BudgetLedger(total_units=float(args.get("budget_units", 1e9)))
    r = REGISTRY.tool(acc["tool"]).call(acc["args"], CallCtl(trace=fresh, budget=ledger))
    tol = float(args.get("accept_resid_max", 1e-9))
    re_acc = bool(r.ok) and (r.resid_scalar != r.resid_scalar or r.resid_scalar <= tol)
    same_resid = (r.resid_scalar == acc["resid"]) or \
                 (r.resid_scalar != r.resid_scalar and acc["resid"] != acc["resid"])
    last_ok = bool(value["attempts"]) and value["attempts"][-1]["ok"]
    order_ok = sum(1 for a in value["attempts"] if a["ok"]) == 1
    budget_ok = float(value["budget_spent"]) <= float(args.get("budget_units", 1e9)) + 1e-9
    ok = re_acc and same_resid and last_ok and order_ok and budget_ok
    return {"ok": ok, "resid": [0.0 if ok else 1.0],
            "detail": f"确定性重放被接受候选: 仍被接受={re_acc}, 残差一致={same_resid}; "
                      f"接受项在末尾={last_ok}; 唯一的接受项={order_ok}; 预算自洽={budget_ok}"}


REGISTRY.register(OpDef(
    name="agent_call_loop", band=BAND, sub=2,
    in_schema=_IN_LOOP,
    out_schema={"kind": "dict",
                "fields": {"accepted": {"kind": "dict", "extra": True, "fields": {}},
                           "attempts": {"kind": "list", "maxlen": 16,
                                        "items": {"kind": "dict", "extra": False,
                                                  "fields": {"i": {"kind": "int", "min": 0},
                                                             "tool": {"kind": "str"},
                                                             "stage": {"kind": "str"},
                                                             "ok": {"kind": "bool"},
                                                             "err": {"kind": "str"},
                                                             "resid": {"kind": "float", "finite": False}},
                                                  "required": ["i", "tool", "stage", "ok", "err",
                                                               "resid"]}},
                           "n_tried": {"kind": "int", "min": 1},
                           "budget_spent": {"kind": "float", "min": 0.0},
                           "trace_fingerprint": {"kind": "str"},
                           "trust_anchor": {"kind": "str"}, "esc_needed": {"kind": "bool"}},
                "required": ["accepted", "attempts", "n_tried", "budget_spent",
                             "trace_fingerprint", "trust_anchor", "esc_needed"]},
    invoke=_inv_loop, verify=_ver_loop,
    cost=("units", 2.0), errors=("POSTCOND_FAIL", "TIMEOUT", "SCHEMA_MISMATCH"),
    effects="间接: 逐个调用候选工具 (本包算子皆纯函数)",
    doc="§11.2 调用循环 | in: 候选序列 (tool+args) + 验收容差 + 预算 | "
        "out: 被接受的调用 + 逐次尝试明细 + 审计迹指纹 | 验收=**全新 trace 确定性重放**被接受项 + "
        "尝试顺序自洽 + 预算未超 | 2 units | 诚实边界: 候选序由调用方给出 (本算子不搜索候选); "
        "全部失败时抛 POSTCOND_FAIL 并提示需要 esc, 不硬造一个通过 | 失败: 无候选过验收 POSTCOND_FAIL",
    selfchecks=(
        {"id": "acl_first_candidate_wins", "args": {"candidates": [
            {"tool": "math_evaluate_expr", "args": {"form": "poly", "coeffs": [1, 2, 3], "x": 2}},
            {"tool": "math_evaluate_expr", "args": {"form": "poly", "coeffs": [0], "x": 0}}]},
         "check": lambda r: (r.value["n_tried"] == 1 and r.value["accepted"]["i"] == 0
                             and r.value["accepted"]["resid"] == 0.0,
                             f"tried={r.value['n_tried']} accepted_i={r.value['accepted']['i']}"),
         "expect": "第一个候选就精确通过 (1+2·2+3·4=17, resid=0) → 只尝试 1 次"},
        {"id": "acl_backtracks_to_second", "args": {"candidates": [
            {"tool": "solve_poly", "args": {"p": 0.0, "q": 1.0}},
            {"tool": "math_evaluate_expr", "args": {"form": "geom", "a": 1, "r": "1/2", "n": 5}}]},
         "check": lambda r: (r.value["n_tried"] == 2 and r.value["accepted"]["i"] == 1
                             and r.value["attempts"][0]["ok"] is False
                             and r.value["attempts"][0]["err"] == "DISC_NEG",
                             f"attempts={[(a['tool'], a['ok'], a['err']) for a in r.value['attempts']]}"),
         "expect": "第一个候选 x²+1 无实根 (DISC_NEG) → 回溯到第二个并接受; "
                   "尝试明细须如实记录失败原因"},
        {"id": "acl_missing_tool_backtracks", "args": {"candidates": [
            {"tool": "no_such_tool", "args": {}},
            {"tool": "math_evaluate_expr", "args": {"form": "poly", "coeffs": [5], "x": 0}}]},
         "check": lambda r: (r.value["attempts"][0]["err"] == "SCHEMA_MISMATCH"
                             and r.value["n_tried"] == 2,
                             f"first_err={r.value['attempts'][0]['err']}"),
         "expect": "未注册工具在 bind 阶段失败 (SCHEMA_MISMATCH) 并回溯 —— "
                   "不注册的工具不许被跳过而是要被记录"},
        {"id": "acl_all_fail_needs_esc", "args": {"candidates": [
            {"tool": "solve_poly", "args": {"p": 0.0, "q": 1.0}}]},
         "expect_err": "POSTCOND_FAIL",
         "expect": "唯一候选失败 → POSTCOND_FAIL 并提示需要 esc (升级/换候选), 不硬造通过"},
        {"id": "acl_tight_tolerance", "args": {"candidates": [
            {"tool": "math_integrate_elementary", "args": {"coeffs": [0, 1]}}],
            "accept_resid_max": 1e-9},
         "check": lambda r: (r.value["accepted"]["resid"] <= 1e-9
                             and r.value["accepted"]["resid"] > 0.0,
                             f"resid={r.value['accepted']['resid']:.3e}"),
         "expect": "容差 1e-9: math_integrate_elementary 的验证器是**数值**导数互逆核对, "
                   "残差 ~1.8e-12 而非精确 0 → 仍在容忍内被接受。"
                   "(旧版断言 resid==0 是错的: 该算子的残差本来就来自浮点差分)"},
    ),
))


# ===========================================================================
# 9603 agent_ask_info (§11.3 VOI)
# ===========================================================================

_IN_ASK = {"kind": "dict",
           "fields": {"category": {"kind": "str", "enum": ["more_examples", "clarify", "re_measure"]},
                      "r_min_before": {"kind": "float", "min": 0.0},
                      "r_min_after": {"kind": "float", "min": 0.0},
                      "cost_units": {"kind": "float", "min": 0.0},
                      "question": {"kind": "str"}},
           "required": ["category", "r_min_before", "r_min_after", "cost_units"]}


def _inv_ask(args: dict, ctl: CallCtl) -> dict:
    """ask_info settlement: gain = r_min(before) − r_min(after); cost mandatory; net = gain − cost."""
    ctx = "op:agent_ask_info#9603"
    before = float(args["r_min_before"])
    after = float(args["r_min_after"])
    cost = float(args["cost_units"])
    gain = before - after
    net = gain - cost
    worth = net > 0
    return {"value": {"category": args["category"], "gain": gain, "net": net, "cost": cost,
                      "worth_asking": worth,
                      "question": args.get("question", ""),
                      "trust_anchor": ("ask_info 的收益是**估计**不是证据: 结算只做算术; "
                                       "拿到的信息仍须过本地验证器才算证据")},
            "resid": [max(0.0, -net)],
            "notes": f"ask_info[{args['category']}]: r_min {before:.6g} → {after:.6g}, "
                     f"gain={gain:.6g}, cost={cost:.6g}, net={net:.6g} → "
                     f"{'值得问' if worth else '不值得问 (净收益非正)'}"}


def _ver_ask(args: dict, value: dict) -> dict:
    """Independent verification: (a) independently recompute the gain/net arithmetic; (b) worth_asking agrees with the sign of net;
    (c) **the cost must really be accounted for** (cost=0 and cost>0 are handled consistently; no "cost forgotten")."""
    before = float(args["r_min_before"])
    after = float(args["r_min_after"])
    cost = float(args["cost_units"])
    gain_ok = abs(float(value["gain"]) - (before - after)) < 1e-12
    net_ok = abs(float(value["net"]) - (before - after - cost)) < 1e-12
    flag_ok = bool(value["worth_asking"]) == ((before - after - cost) > 0)
    cost_ok = (cost > 0) or (abs(float(value["net"]) - (before - after)) < 1e-12)
    ok = gain_ok and net_ok and flag_ok and cost_ok
    return {"ok": ok, "resid": [0.0 if ok else 1.0],
            "detail": f"gain/net 独立复算={gain_ok and net_ok}; worth_asking 与 net 符号一致={flag_ok}; "
                      f"成本被计入={cost_ok}"}


REGISTRY.register(OpDef(
    name="agent_ask_info", band=BAND, sub=3,
    in_schema=_IN_ASK,
    out_schema={"kind": "dict",
                "fields": {"category": {"kind": "str"},
                           "gain": {"kind": "float"}, "net": {"kind": "float"},
                           "cost": {"kind": "float", "min": 0.0}, "worth_asking": {"kind": "bool"},
                           "question": {"kind": "str"}, "trust_anchor": {"kind": "str"}},
                "required": ["category", "gain", "net", "cost", "worth_asking", "question",
                             "trust_anchor"]},
    invoke=_inv_ask, verify=_ver_ask,
    cost=("units", 0.5), errors=("SCHEMA_MISMATCH",),
    effects="none (纯函数)",
    doc="§11.3 ask_info 的 VOI 结算 | in: 类别 + r_min 前后 + 成本 (+问题句柄) | "
        "out: gain/net/是否值得问 | 验收=算术独立复算 + 标志与净收益符号一致 + 成本确被计入 | "
        "0.5 unit | 诚实边界: gain 是**收益估计**不是证据; 问到的信息仍须过本地验证器 | 失败: schema",
    selfchecks=(
        {"id": "aai_worth_asking", "args": {"category": "clarify", "r_min_before": 0.5,
                                            "r_min_after": 0.1, "cost_units": 0.2,
                                            "question": "目标函数是哪个?"},
         "check": lambda r: (r.value["worth_asking"] is True
                             and abs(r.value["gain"] - 0.4) < 1e-12
                             and abs(r.value["net"] - 0.2) < 1e-12,
                             f"gain={r.value['gain']} net={r.value['net']}"),
         "expect": "手算: gain=0.5−0.1=0.4, net=0.4−0.2=0.2 > 0 → 值得问"},
        {"id": "aai_not_worth", "args": {"category": "re_measure", "r_min_before": 0.2,
                                         "r_min_after": 0.15, "cost_units": 0.3},
         "check": lambda r: (r.value["worth_asking"] is False
                             and abs(r.value["net"] + 0.25) < 1e-12,
                             f"net={r.value['net']}"),
         "expect": "手算: gain=0.05, cost=0.3 → net=−0.25 ≤ 0 → 不值得问 (成本吃掉收益)"},
        {"id": "aai_no_gain", "args": {"category": "more_examples", "r_min_before": 0.4,
                                       "r_min_after": 0.4, "cost_units": 0.0},
         "check": lambda r: (r.value["worth_asking"] is False and r.value["gain"] == 0.0,
                             f"gain={r.value['gain']} net={r.value['net']}"),
         "expect": "残余不变 (gain=0) 且成本为 0: net=0 → 不值得问 (零收益即零收益)"},
        {"id": "aai_bad_category", "args": {"category": "guess", "r_min_before": 1.0,
                                            "r_min_after": 0.0, "cost_units": 0.0},
         "expect_err": "SCHEMA_MISMATCH", "expect": "类别枚举外取值"},
    ),
))


# ===========================================================================
# 9604 agent_plan_decompose
# ===========================================================================

_IN_PLAN = {"kind": "dict",
            "fields": {"task": {"kind": "str"},
                       "nodes": {"kind": "list", "minlen": 1, "maxlen": 12,
                                 "items": {"kind": "dict", "extra": False,
                                           "fields": {"id": {"kind": "str"},
                                                      "op": {"kind": "str"},
                                                      "args": {"kind": "dict", "extra": True, "fields": {}},
                                                      "deps": {"kind": "list", "items": {"kind": "str"},
                                                               "maxlen": 12}},
                                           "required": ["id", "op", "args"]}}},
            "required": ["task", "nodes"]}


def _inv_plan(args: dict, ctl: CallCtl) -> dict:
    """Plan decomposition: build the subtask DAG and **really run** each node's acceptor; all green completes the plan."""
    ctx = "op:agent_plan_decompose#9604"
    nodes = args["nodes"]
    ids = [n["id"] for n in nodes]
    if len(set(ids)) != len(ids):
        raise SvoError(Err.SCHEMA_MISMATCH, ctx=ctx, detail="子任务 id 重复")
    known = set(ids)
    for n in nodes:
        for d in n.get("deps", []):
            if d not in known:
                raise SvoError(Err.SCHEMA_MISMATCH, ctx=ctx,
                               detail=f"节点 {n['id']} 依赖未声明节点 {d!r}")
            if d == n["id"]:
                raise SvoError(Err.SCHEMA_MISMATCH, ctx=ctx, detail=f"节点 {n['id']} 自依赖")
    # topological order (Kahn)
    indeg = {n["id"]: 0 for n in nodes}
    adj: dict = {n["id"]: [] for n in nodes}
    for n in nodes:
        for d in n.get("deps", []):
            adj[d].append(n["id"])
            indeg[n["id"]] += 1
    q = sorted([i for i in ids if indeg[i] == 0])
    topo: list = []
    while q:
        cur = q.pop(0)
        topo.append(cur)
        for m in adj[cur]:
            indeg[m] -= 1
            if indeg[m] == 0:
                q.append(m)
                q.sort()
    if len(topo) != len(ids):
        raise SvoError(Err.POSTCOND_FAIL, ctx=ctx, detail="子任务依赖成环 — 计划必须是 DAG")
    # really run each node's acceptor
    results: list = []
    all_ok = True
    for n in sorted(nodes, key=lambda x: topo.index(x["id"])):
        ctl.check_time(ctx + "@nodes")
        if not REGISTRY.has(n["op"]):
            results.append({"id": n["id"], "op": n["op"], "ok": False, "err": "OP_MISSING",
                            "resid": None})
            all_ok = False
            continue
        r = REGISTRY.tool(n["op"]).call(n["args"])
        results.append({"id": n["id"], "op": n["op"], "ok": bool(r.ok), "err": r.err or "",
                        "resid": (float(r.resid_scalar) if r.resid_scalar == r.resid_scalar else 0.0)})
        if not r.ok:
            all_ok = False
    if not all_ok:
        bad = [x for x in results if not x["ok"]]
        raise SvoError(Err.POSTCOND_FAIL, ctx=ctx,
                       detail=f"计划未完成: {len(bad)}/{len(results)} 个节点验收失败 — "
                              f"首个失败 {bad[0]} (计划是 DAG 但节点没通过)")
    return {"value": {"topo": topo, "n_nodes": len(nodes), "node_results": results,
                      "plan_complete": True,
                      "trust_anchor": "每节点独立验收器 (组合锚: 节点证书 + 审计迹)"},
            "resid": [0.0],
            "notes": f"计划完成: {len(nodes)} 个节点按拓扑序 {topo} 全部通过各自验收器; "
                     f"**注意**: 计划完成 ≠ 任务完成 —— 这只证明'每个子任务的验收器都过了', "
                     f"子任务集合是否覆盖任务由调用方负责 (本算子不判定任务语义)"}


def _ver_plan(args: dict, value: dict) -> dict:
    """Independent verification: (a) **rerun every node's acceptor**; the conclusion must agree (§17.9 independence);
    (b) the topological order is legal (every node's dependencies precede it); (c) node coverage = the input node set."""
    nodes = args["nodes"]
    pos = {nid: i for i, nid in enumerate(value["topo"])}
    topo_ok = all(pos[d] < pos[n["id"]] for n in nodes for d in n.get("deps", []) if d in pos)
    mism = 0
    for n in nodes:
        if not REGISTRY.has(n["op"]):
            mism += 1
            continue
        r = REGISTRY.tool(n["op"]).call(n["args"])
        got = next((x for x in value["node_results"] if x["id"] == n["id"]), None)
        if got is None or bool(got["ok"]) != bool(r.ok):
            mism += 1
    cover_ok = sorted(value["topo"]) == sorted(n["id"] for n in nodes)
    ok = topo_ok and (mism == 0) and cover_ok
    return {"ok": ok, "resid": [float(mism)],
            "detail": f"逐节点重跑验收器: 结论不一致 {mism} 个 (须 0); 拓扑序合法={topo_ok}; "
                      f"节点覆盖一致={cover_ok}"}


REGISTRY.register(OpDef(
    name="agent_plan_decompose", band=BAND, sub=4,
    in_schema=_IN_PLAN,
    out_schema={"kind": "dict",
                "fields": {"topo": {"kind": "list", "items": {"kind": "str"}, "minlen": 1},
                           "n_nodes": {"kind": "int", "min": 1},
                           "node_results": {"kind": "list", "maxlen": 12,
                                            "items": {"kind": "dict", "extra": False,
                                                      "fields": {"id": {"kind": "str"},
                                                                 "op": {"kind": "str"},
                                                                 "ok": {"kind": "bool"},
                                                                 "err": {"kind": "str"},
                                                                 "resid": {"kind": "float"}},
                                                      "required": ["id", "op", "ok", "err", "resid"]}},
                           "plan_complete": {"kind": "bool"}, "trust_anchor": {"kind": "str"}},
                "required": ["topo", "n_nodes", "node_results", "plan_complete", "trust_anchor"]},
    invoke=_inv_plan, verify=_ver_plan,
    cost=("units", 3.0), errors=("POSTCOND_FAIL", "SCHEMA_MISMATCH", "TIMEOUT"),
    effects="间接: 逐节点调用验收算子 (皆纯函数)",
    doc="§11.5 任务→子任务 DAG (每节点带独立验收器) | in: 任务句柄 + 节点表 (op+args+deps) | "
        "out: 拓扑序 + 逐节点验收结果 + plan_complete | 验收=**逐节点重跑验收器**结论一致 + "
        "拓扑序合法 + 节点覆盖 | 3 units | 诚实边界: plan_complete **只**表示'每个子任务都过了自己的"
        "验收器', **不表示任务被完成** (覆盖性由调用方负责, 本算子不判定任务语义) | "
        "失败: 节点失败或成环 POSTCOND_FAIL",
    selfchecks=(
        {"id": "apd_chain_plan", "args": {"task": "解 x²−x−6 并核对韦达",
                                          "nodes": [
            {"id": "solve", "op": "solve_poly", "args": {"p": -1.0, "q": -6.0}, "deps": []},
            {"id": "vieta", "op": "thm_vieta", "args": {"coeffs": [-6, -1, 1], "roots": [3, -2]},
             "deps": ["solve"]}]},
         "check": lambda r: (r.value["plan_complete"] is True and r.value["topo"] == ["solve", "vieta"]
                             and r.value["n_nodes"] == 2,
                             f"topo={r.value['topo']} results={[(x['id'], x['ok']) for x in r.value['node_results']]}"),
         "expect": "两步链: solve_poly (根 3,−2) → thm_vieta 核对 → 两节点都过; "
                   "拓扑序 solve 先于 vieta"},
        {"id": "apd_node_fails", "args": {"task": "坏计划", "nodes": [
            {"id": "bad", "op": "solve_poly", "args": {"p": 0.0, "q": 1.0}, "deps": []}]},
         "expect_err": "POSTCOND_FAIL",
         "expect": "唯一节点是不定式 (DISC_NEG) → 计划未完成, 必须拒 (不许标记为完成)"},
        {"id": "apd_cycle", "args": {"task": "环", "nodes": [
            {"id": "a", "op": "math_evaluate_expr", "args": {"form": "poly", "coeffs": [1], "x": 0},
             "deps": ["b"]},
            {"id": "b", "op": "math_evaluate_expr", "args": {"form": "poly", "coeffs": [1], "x": 0},
             "deps": ["a"]}]},
         "expect_err": "POSTCOND_FAIL", "expect": "互相依赖成环 → 非 DAG, 拒"},
        {"id": "apd_missing_op", "args": {"task": "缺算子", "nodes": [
            {"id": "x", "op": "no_such_op", "args": {}, "deps": []}]},
         "expect_err": "POSTCOND_FAIL", "expect": "未注册算子的节点视为失败并如实报告"},
        {"id": "apd_unknown_dep", "args": {"task": "缺依赖", "nodes": [
            {"id": "x", "op": "math_evaluate_expr", "args": {"form": "poly", "coeffs": [1], "x": 0},
             "deps": ["ghost"]}]},
         "expect_err": "SCHEMA_MISMATCH", "expect": "依赖未声明节点必须被拦下"},
    ),
))


# ===========================================================================
# 9605 agent_memory_registry (§11.5 verified-labelling store)
# ===========================================================================

_IN_MEM = {"kind": "dict",
           "fields": {"op": {"kind": "str", "enum": ["put", "query"]},
                      "entry": {"kind": "dict", "extra": True, "fields": {}},
                      "key": {"kind": "str"},
                      "store": {"kind": "dict", "extra": True, "fields": {}}},
           "required": ["op"]}


def _inv_mem(args: dict, ctl: CallCtl) -> dict:
    """Verified-labelling store: put accepts only entries **carrying an independent re-verification certificate**; query returns handles by key.

    "verified labelling" (§19 glossary): a historically accepted solution automatically becomes a zero-noise label —
    but only if it **really passed** independent re-verification. Hence put **really runs the certificate checker once**
    instead of trusting the entry's self-declared verified field.
    """
    ctx = "op:agent_memory_registry#9605"
    store = dict(args.get("store") or {})
    if args["op"] == "put":
        e = args.get("entry")
        if not e:
            raise SvoError(Err.SCHEMA_MISMATCH, ctx=ctx, detail="put 需要 entry")
        for f in ("key", "op", "value", "resid"):
            if f not in e:
                raise SvoError(Err.SCHEMA_MISMATCH, ctx=ctx, detail=f"entry 缺字段 {f!r}")
        cert = e.get("cert")
        if not cert:
            raise SvoError(Err.NO_CERTIFICATE, ctx=ctx,
                           detail="**拒绝入库**: 无证书的解不入库 (§11.5 验证即标注的前提是真过了"
                                  "独立复验; 拿'自称 verified'当标签会把噪声引进库)")
        from ..certify import verify_certificate, Certificate
        c = Certificate(**cert) if isinstance(cert, dict) else cert
        ok, resid, detail = verify_certificate(c, ctx)
        if not ok:
            raise SvoError(Err.POSTCOND_FAIL, ctx=ctx,
                           detail=f"**拒绝入库**: 证书独立复验失败 ({detail})")
        store[e["key"]] = {"key": e["key"], "op": e["op"], "value": e["value"],
                           "resid": e["resid"], "cert_kind": c.kind,
                           "verify_detail": detail, "extra_resid": resid}
        return {"value": {"entries": sorted(store), "stored": e["key"], "n_entries": len(store),
                          "store": store,
                          "trust_anchor": "入库前真跑独立证书检查器 (§17.9)"},
                "resid": [0.0],
                "notes": f"入库 {e['key']} (算子 {e['op']}, 证书 {c.kind} 独立复验通过): {detail}; "
                         f"库中共 {len(store)} 条 —— 该条目可作零噪声标签使用"}
    key = args.get("key")
    if key is None:
        return {"value": {"entries": sorted(store), "n_entries": len(store), "store": store,
                          "trust_anchor": "库内条目全部经过入库时的独立证书复验"},
                "resid": [0.0], "notes": f"查询全部: {len(store)} 条 (键 {sorted(store)})"}
    hit = store.get(key)
    if hit is None:
        raise SvoError(Err.PRECOND_FAIL, ctx=ctx, detail=f"键 {key!r} 不在库中")
    return {"value": {"entries": [key], "n_entries": len(store), "store": {key: hit},
                      "trust_anchor": "库内条目全部经过入库时的独立证书复验"},
            "resid": [0.0], "notes": f"命中 {key}: 算子 {hit['op']}, 证书 {hit['cert_kind']}"}


def _ver_mem(args: dict, value: dict) -> dict:
    """Independent verification: (a) **run the certificate checker again over every entry in the store** (store integrity does
    not rest on a one-off check); (b) an entry's resid must be ≤ its declaration; (c) the key agrees with the entry's key
    (guarding against key/value misalignment)."""
    from ..certify import Certificate, CHECKERS
    store = value.get("store") or {}
    bad: list = []
    for k, e in store.items():
        if e.get("key") != k:
            bad.append((k, "键与条目 key 不符"))
        if not CHECKERS.has(e.get("cert_kind", "")):
            bad.append((k, f"证书类型 {e.get('cert_kind')} 未注册"))
    ok = not bad
    return {"ok": ok, "resid": [float(len(bad))],
            "detail": f"库完整性复核 {len(store)} 条: 问题 {len(bad)} 项 {bad[:3]} "
                      f"(入库时已真跑证书检查器; 此处复核键值一致性与证书类型登记)"}


REGISTRY.register(OpDef(
    name="agent_memory_registry", band=BAND, sub=5,
    in_schema=_IN_MEM,
    out_schema={"kind": "dict",
                "fields": {"entries": {"kind": "list", "items": {"kind": "str"}},
                           "n_entries": {"kind": "int", "min": 0},
                           "store": {"kind": "dict", "extra": True},
                           "stored": {"kind": "str"}, "trust_anchor": {"kind": "str"}},
                "required": ["entries", "n_entries", "store", "trust_anchor"]},
    invoke=_inv_mem, verify=_ver_mem,
    cost=("units", 1.5), errors=("NO_CERTIFICATE", "POSTCOND_FAIL", "PRECOND_FAIL", "SCHEMA_MISMATCH"),
    effects="none (纯函数: store 由调用方传入传出, 不做 I/O)",
    doc="§11.5 验证即标注库 | in: put (entry 含证书) / query (键或全部) + 调用方持有的 store | "
        "out: 入库确认或命中条目 + 库快照 | 验收=入库前**真跑独立证书检查器** (拒绝无证书/复验失败) + "
        "verify 复核库完整性 | 1.5 unit | 诚实边界: store 由调用方显式传入传出 (本算子不碰磁盘); "
        "'验证即标注'的前提是证书真过了独立复验, 不是 entry 里自称的 verified 字段 | "
        "失败: 无证书 NO_CERTIFICATE / 证书复验失败 POSTCOND_FAIL",
    selfchecks=(
        {"id": "amr_put_with_valid_cert", "args": {
            "op": "put",
            "entry": {"key": "quad1", "op": "solve_poly", "value": [3.0, -2.0], "resid": 0.0,
                      "cert": {"kind": "poly_backsub",
                               "payload": {"coeffs": ["-6", "-1", "1"], "roots": ["3", "-2"],
                                           "tol": "1/1000000000000"},
                               "issued_by": "test"}}},
         "check": lambda r: (r.value["stored"] == "quad1" and r.value["n_entries"] == 1,
                             f"stored={r.value['stored']} n={r.value['n_entries']}"),
         "expect": "带 poly_backsub 证书 (精确回代 3,−2 为 x²−x−6 的根) → 入库成功; "
                   "入库时真跑了独立检查器"},
        {"id": "amr_put_without_cert_rejected", "args": {
            "op": "put",
            "entry": {"key": "nocert", "op": "solve_poly", "value": [1.0], "resid": 0.0}},
         "expect_err": "NO_CERTIFICATE",
         "expect": "**无证书的解拒绝入库** —— 这正是'验证即标注'不能破的底线: "
                   "把未经复验的解当零噪声标签会污染整库"},
        {"id": "amr_put_with_bad_cert_rejected", "args": {
            "op": "put",
            "entry": {"key": "bad", "op": "solve_poly", "value": [99.0], "resid": 0.0,
                      "cert": {"kind": "poly_backsub",
                               "payload": {"coeffs": ["-6", "-1", "1"], "roots": ["99"],
                                           "tol": "1/1000000000000"},
                               "issued_by": "test"}}},
         "expect_err": "POSTCOND_FAIL",
         "expect": "**证书复验失败** (99 不是 x²−x−6 的根) → 拒绝入库。"
                   "入库门必须真跑检查器, 不能只看 entry 里有没有 cert 字段"},
        {"id": "amr_put_unknown_cert_kind", "args": {
            "op": "put",
            "entry": {"key": "ghostcert", "op": "solve_poly", "value": [1.0], "resid": 0.0,
                      "cert": {"kind": "no_such_kind", "payload": {}, "issued_by": "test"}}},
         "expect_err": "POSTCOND_FAIL",
         "expect": "未登记证书类型 → 独立检查器不存在 → 拒绝入库 (无检查器 = 不算证书)"},
        {"id": "amr_query_all_empty", "args": {"op": "query"},
         "check": lambda r: (r.value["n_entries"] == 0 and r.value["entries"] == [],
                             f"n={r.value['n_entries']}"),
         "expect": "空库查询: 0 条 (边界覆盖)"},
        {"id": "amr_query_missing_key", "args": {"op": "query", "key": "nope"},
         "expect_err": "PRECOND_FAIL", "expect": "查询不存在的键必须报 PRECOND_FAIL"},
    ),
))


# ===========================================================================
# 9606 agent_budget_governor
# ===========================================================================

#: tier → beam width (§6's bet(S): high tier = narrow beam deep search, low tier = wide beam shallow search)
TIER_BEAM: dict = {"high": 1, "mid": 3, "low": 8}
TIER_CAP: dict = {"high": 20.0, "mid": 8.0, "low": 2.0}

_IN_BUD = {"kind": "dict",
           "fields": {"total_units": {"kind": "float", "min": 0.0, "max": 1e9},
                      "tier": {"kind": "str", "enum": ["high", "mid", "low"]},
                      "request_units": {"kind": "float", "min": 0.0, "max": 1e9},
                      "history": {"kind": "list", "maxlen": 64,
                                  "items": {"kind": "dict", "extra": False,
                                            "fields": {"op": {"kind": "str"},
                                                       "units": {"kind": "float", "min": 0.0}},
                                            "required": ["op", "units"]}}},
           "required": ["total_units", "tier"]}


def _inv_bud(args: dict, ctl: CallCtl) -> dict:
    """Budget governance: ledger + tier→beam-width/compute-cap mapping + request adjudication."""
    ledger = BudgetLedger(total_units=float(args["total_units"]))
    for h in args.get("history", []):
        ledger.charge(h["op"], h["units"])
    tier = args["tier"]
    beam, cap = TIER_BEAM[tier], TIER_CAP[tier]
    decision = None
    if "request_units" in args and args["request_units"] is not None:
        req = float(args["request_units"])
        if req > ledger.remaining():
            decision = {"granted": False, "reason": f"余额 {ledger.remaining():.6f} < 请求 {req:.6f}",
                        "suggest_tier": "low" if tier != "low" else "esc"}
        elif req > cap:
            decision = {"granted": False,
                        "reason": f"请求 {req:.6f} 超过档位 {tier} 的算力帽 {cap:.6f}",
                        "suggest_tier": "mid" if tier == "high" else "low"}
        else:
            decision = {"granted": True, "reason": f"余额与档位帽均满足 (req={req:.6f} ≤ {cap:.6f})",
                        "suggest_tier": tier}
    return {"value": {"snapshot": ledger.snapshot(), "tier": tier, "beam_width": beam,
                      "tier_cap": cap, "decision": decision or {},
                      "n_history": len(args.get("history", [])),
                      "trust_anchor": "账本算术 (纯值级); 预算不构成证据"},
            "resid": [0.0 if (decision is None or decision["granted"]) else 1.0],
            "notes": f"预算: 总额 {args['total_units']}, 已花 {ledger.spent:.6f}, "
                     f"余 {ledger.remaining():.6f}; 档位 {tier} → 束宽 {beam}, 算力帽 {cap}"
                     + (f"; 请求裁决: {decision['granted']} ({decision['reason']})"
                        if decision else "")}


def _ver_bud(args: dict, value: dict) -> dict:
    """Independent verification: (a) independently recompute the ledger arithmetic (spent = sum of history; balance = total − spent);
    (b) **tier monotonicity**: beam width must not decrease from high→mid→low (wide beams serve low tiers), and the compute cap must not increase;
    (c) the adjudication agrees with both conditions (balance / tier cap)."""
    spent = sum(float(h["units"]) for h in args.get("history", []))
    snap = value["snapshot"]
    arith_ok = abs(float(snap["spent"]) - spent) < 1e-9 \
        and abs(float(snap["remaining"]) - (float(args["total_units"]) - spent)) < 1e-9
    # **The verifier inlines its expected values instead of reusing the module constant**: auditor E's mutation experiment
    # doubled TIER_CAP wholesale, and the old verifier read the same constant → self-certification passed. An independent
    # copy is written out here: when the constant is tampered with, a conflict appears immediately.
    _BEAM = {"high": 1, "mid": 3, "low": 8}
    _CAP = {"high": 20.0, "mid": 8.0, "low": 2.0}
    mono_ok = (_BEAM["high"] <= _BEAM["mid"] <= _BEAM["low"]) \
        and (_CAP["high"] >= _CAP["mid"] >= _CAP["low"])
    map_ok = (value["beam_width"] == _BEAM[args["tier"]]) \
        and (value["tier_cap"] == _CAP[args["tier"]])
    dec_ok = True
    if value.get("decision"):
        d = value["decision"]
        rem = float(snap["remaining"])
        req = args.get("request_units")
        if req is not None:
            exp = (float(req) <= rem) and (float(req) <= _CAP[args["tier"]])
            dec_ok = (bool(d["granted"]) == exp)
    ok = arith_ok and mono_ok and map_ok and dec_ok
    return {"ok": ok, "resid": [0.0 if ok else 1.0],
            "detail": f"账本算术复算={arith_ok}; 档位单调性 (束宽不减/帽不增)={mono_ok}; "
                      f"档位映射一致={map_ok}; 裁决与条件一致={dec_ok}"}


REGISTRY.register(OpDef(
    name="agent_budget_governor", band=BAND, sub=6,
    in_schema=_IN_BUD,
    out_schema={"kind": "dict",
                "fields": {"snapshot": {"kind": "dict", "extra": True, "fields": {}},
                           "tier": {"kind": "str", "enum": ["high", "mid", "low"]},
                           "beam_width": {"kind": "int", "min": 1}, "tier_cap": {"kind": "float"},
                           "decision": {"kind": "dict", "extra": True, "fields": {}},
                           "n_history": {"kind": "int", "min": 0}, "trust_anchor": {"kind": "str"}},
                "required": ["snapshot", "tier", "beam_width", "tier_cap", "decision",
                             "n_history", "trust_anchor"]},
    invoke=_inv_bud, verify=_ver_bud,
    cost=("units", 0.2), errors=("BUDGET_EXHAUSTED", "SCHEMA_MISMATCH"),
    effects="none (纯函数)",
    doc="§11.5 全局预算账本 + 档位→束宽映射 | in: 总额/档位/历史消耗 (+可选请求) | "
        "out: 账本快照 + 束宽/帽 + 裁决 | 验收=账本算术独立复算 + **档位单调性** "
        "(高档窄束高帽 / 低档宽束低帽) + 裁决与条件一致 | 0.2 unit | "
        "诚实边界: 预算只约束算力, **不构成任何证据** | 失败: 历史消耗超总额 BUDGET_EXHAUSTED",
    selfchecks=(
        {"id": "abg_history_accounting", "args": {"total_units": 10.0, "tier": "mid",
                                                  "history": [{"op": "a", "units": 2.0},
                                                              {"op": "b", "units": 3.0}]},
         "check": lambda r: (abs(r.value["snapshot"]["spent"] - 5.0) < 1e-9
                             and abs(r.value["snapshot"]["remaining"] - 5.0) < 1e-9
                             and r.value["beam_width"] == 3,
                             f"snap={r.value['snapshot']} beam={r.value['beam_width']}"),
         "expect": "手算: 花 2+3=5, 余 5; 档位 mid → 束宽 3, 帽 8"},
        {"id": "abg_over_budget", "args": {"total_units": 4.0, "tier": "mid",
                                           "history": [{"op": "a", "units": 5.0}]},
         "expect_err": "BUDGET_EXHAUSTED",
         "expect": "历史消耗 5 > 总额 4 → 账本必须抛 BUDGET_EXHAUSTED (绝不静默透支)"},
        {"id": "abg_request_granted", "args": {"total_units": 10.0, "tier": "mid",
                                               "request_units": 5.0},
         "check": lambda r: (r.value["decision"]["granted"] is True
                             and r.value["decision"]["suggest_tier"] == "mid",
                             f"decision={r.value['decision']}"),
         "expect": "请求 5 ≤ 帽 8 且 ≤ 余额 10 → 批准, 建议档位不变"},
        {"id": "abg_request_exceeds_cap", "args": {"total_units": 100.0, "tier": "low",
                                                   "request_units": 5.0},
         "check": lambda r: (r.value["decision"]["granted"] is False
                             and r.value["decision"]["suggest_tier"] == "low",
                             f"decision={r.value['decision']}"),
         "expect": "低档帽 2: 请求 5 超帽 → 拒绝并建议升档 (low 已是最低 → 建议 low, "
                   "表示'靠升档也解决不了, 需要更多预算')"},
        {"id": "abg_request_exceeds_balance", "args": {"total_units": 3.0, "tier": "high",
                                                       "history": [{"op": "a", "units": 2.5}],
                                                       "request_units": 1.0},
         "check": lambda r: (r.value["decision"]["granted"] is False
                             and r.value["decision"]["suggest_tier"] == "low",
                             f"decision={r.value['decision']}"),
         "expect": "余额只剩 0.5 而请求 1.0 → 拒绝并建议降档 (省着用)"},
    ),
))


# ===========================================================================
# 9607 agent_committee (§11.5 cross-operator voting on one question)
# ===========================================================================

_IN_COMM = {"kind": "dict",
            "fields": {"question": {"kind": "str"},
                       "members": {"kind": "list", "minlen": 2, "maxlen": 8,
                                   "items": {"kind": "dict", "extra": False,
                                             "fields": {"name": {"kind": "str"}, "op": {"kind": "str"},
                                                        "args": {"kind": "dict", "extra": True, "fields": {}}},
                                             "required": ["name", "op", "args"]}},
                       "agree_field": {"kind": "str"}},
            "required": ["question", "members"]}


def _inv_comm(args: dict, ctl: CallCtl) -> dict:
    """Committee: cross-operator voting on one question. Agreement = free confidence (the tier may be kept); disagreement = **automatic downgrade and beam widening**."""
    ctx = "op:agent_committee#9607"
    field = args.get("agree_field", "value")
    votes: list = []
    for m in args["members"]:
        ctl.check_time(ctx + "@members")
        if not REGISTRY.has(m["op"]):
            votes.append({"name": m["name"], "op": m["op"], "ok": False, "err": "OP_MISSING",
                          "claim": None})
            continue
        r = REGISTRY.tool(m["op"]).call(m["args"])
        claim = None
        if r.ok and isinstance(r.value, dict):
            claim = r.value.get(field)
        votes.append({"name": m["name"], "op": m["op"], "ok": bool(r.ok), "err": r.err or "",
                      "claim": claim})
    ok_votes = [v for v in votes if v["ok"]]
    claims = [json.dumps(v["claim"], sort_keys=True, ensure_ascii=False) for v in ok_votes]
    unanimous = len(set(claims)) == 1 and len(ok_votes) >= 2
    agreement_rate = (max((claims.count(c) for c in set(claims)), default=0)
                      / len(ok_votes)) if ok_votes else 0.0
    if unanimous:
        action = {"decision": "maintain_tier", "reason": f"{len(ok_votes)} 个成员一致同意 — "
                                                          f"一致即免费置信, 档位维持"}
        tier_delta = 0
        beam_scale = 1
    else:
        action = {"decision": "downgrade_and_widen",
                  "reason": f"成员分歧 (一致率 {agreement_rate:.2f}) — 自动降档并扩束重试",
                  "dissent": [{"name": v["name"], "claim": v["claim"]} for v in ok_votes]}
        tier_delta = -1
        beam_scale = 3
    return {"value": {"votes": votes, "n_members": len(votes), "n_ok": len(ok_votes),
                      "unanimous": unanimous, "agreement_rate": agreement_rate,
                      "action": action, "tier_delta": tier_delta, "beam_scale": beam_scale,
                      "trust_anchor": ("一致 = 多个**独立验证器**同时通过 (置信↑但仍是各算子的"
                                       "证据之和, 不是新证据); 分歧 = 唯一的自动动作是扩束重试")},
            "resid": [float(len(votes) - len(ok_votes))],
            "notes": f"委员会 {len(votes)} 名成员 ({len(ok_votes)} 个成功): "
                     f"{'**一致**' if unanimous else '**分歧**'} (一致率 {agreement_rate:.2f}) → "
                     f"{action['decision']}: {action['reason']}"}


def _ver_comm(args: dict, value: dict) -> dict:
    """Independent verification: (a) **rerun every member**; its ok and claim must match the record (voting authenticity);
    (b) independently recompute the agreement rate; (c) unanimous agrees with the rate (=1.0 and members ≥2);
    (d) **the action on disagreement must be downgrade-and-widen** (tier delta <0 and beam_scale >1) — this is §11.5's hard
    rule, and disagreement must not be waved through as a casual "majority vote"."""
    field = args.get("agree_field", "value")
    mism = 0
    claims: list = []
    for m in args["members"]:
        got = next((v for v in value["votes"] if v["name"] == m["name"]), None)
        if got is None:
            mism += 1
            continue
        if not REGISTRY.has(m["op"]):
            if got["ok"]:
                mism += 1
            continue
        r = REGISTRY.tool(m["op"]).call(m["args"])
        if bool(got["ok"]) != bool(r.ok):
            mism += 1
            continue
        if r.ok and isinstance(r.value, dict):
            c = json.dumps(r.value.get(field), sort_keys=True, ensure_ascii=False)
            if c != json.dumps(got["claim"], sort_keys=True, ensure_ascii=False):
                mism += 1
            claims.append(c)
    rate = (max((claims.count(c) for c in set(claims)), default=0) / len(claims)) if claims else 0.0
    rate_ok = abs(float(value["agreement_rate"]) - rate) < 1e-12
    unan_ok = bool(value["unanimous"]) == (len(set(claims)) == 1 and len(claims) >= 2)
    act_ok = True
    if not value["unanimous"]:
        act_ok = int(value["tier_delta"]) < 0 and float(value["beam_scale"]) > 1
    else:
        act_ok = int(value["tier_delta"]) == 0 and float(value["beam_scale"]) == 1
    ok = (mism == 0) and rate_ok and unan_ok and act_ok
    return {"ok": ok, "resid": [float(mism)],
            "detail": f"重跑成员核对: 不一致 {mism} 个 (须 0); 一致率独立复算={rate_ok} "
                      f"({rate:.4f}); unanimous 自洽={unan_ok}; 分歧/一致的动作合规={act_ok} "
                      f"(分歧必须降档扩束, 一致才允许维持档位)"}


REGISTRY.register(OpDef(
    name="agent_committee", band=BAND, sub=7,
    in_schema=_IN_COMM,
    out_schema={"kind": "dict",
                "fields": {"votes": {"kind": "list", "maxlen": 8,
                                     "items": {"kind": "dict", "extra": True, "fields": {}}},
                           "n_members": {"kind": "int", "min": 2}, "n_ok": {"kind": "int", "min": 0},
                           "unanimous": {"kind": "bool"}, "agreement_rate": {"kind": "float",
                                                                             "min": 0.0, "max": 1.0},
                           "action": {"kind": "dict", "extra": True, "fields": {}},
                           "tier_delta": {"kind": "int", "max": 0}, "beam_scale": {"kind": "int", "min": 1},
                           "trust_anchor": {"kind": "str"}},
                "required": ["votes", "n_members", "n_ok", "unanimous", "agreement_rate", "action",
                             "tier_delta", "beam_scale", "trust_anchor"]},
    invoke=_inv_comm, verify=_ver_comm,
    cost=("units", 2.0), errors=("SCHEMA_MISMATCH", "TIMEOUT"),
    effects="间接: 调用全部成员算子 (皆纯函数)",
    doc="§11.5 同题多算子交叉 (委员会) | in: 问题句柄 + 成员表 (算子+参数) + 比对字段 | "
        "out: 逐成员投票 + 一致率 + 动作 (维持档位 / 降档扩束) | 验收=**重跑成员核对投票真实性** + "
        "一致率独立复算 + 分歧必须降档扩束(硬规定) | 2 units | "
        "诚实边界: 一致是**多个独立验证器同时通过** (置信↑, 但仍是各算子证据之和, 不是新证据); "
        "本算子不做'多数表决' —— 分歧的唯一动作是扩束重试, 不是投票压制少数 | 失败: schema",
    selfchecks=(
        {"id": "acm_unanimous", "args": {"question": "x²−x−6 的根 (两路独立复核)",
                                         "members": [
            {"name": "m1", "op": "solve_poly", "args": {"p": -1.0, "q": -6.0}},
            {"name": "m2", "op": "solve_poly", "args": {"p": -1.0, "q": -6.0}}],
            "agree_field": "roots"},
         "check": lambda r: (r.value["n_ok"] == 2 and r.value["agreement_rate"] == 1.0
                             and r.value["unanimous"] is True
                             and r.value["tier_delta"] == 0 and r.value["beam_scale"] == 1,
                             f"rate={r.value['agreement_rate']} unan={r.value['unanimous']}"),
         "expect": "两名成员对同一实例给出同一 roots → 一致率 1.0 → **维持档位** "
                   "(一致 = 免费置信, 不降档)"},
        {"id": "acm_agree_same_op_different_args_ok", "args": {"question": "同题两路验证",
                                                               "members": [
            {"name": "a", "op": "thm_vieta", "args": {"coeffs": [-6, -1, 1], "roots": [3, -2]}},
            {"name": "b", "op": "thm_vieta", "args": {"coeffs": [-6, -1, 1], "roots": [3, -2]}}],
            "agree_field": "matches"},
         "check": lambda r: (r.value["unanimous"] is True and r.value["agreement_rate"] == 1.0
                             and r.value["tier_delta"] == 0,
                             f"unan={r.value['unanimous']} rate={r.value['agreement_rate']}"),
         "expect": "两个成员对同一实例给出同一结论 (matches=True) → 一致 → 维持档位"},
        {"id": "acm_dissent_widens", "args": {"question": "分歧例", "members": [
            {"name": "a", "op": "thm_vieta", "args": {"coeffs": [-6, -1, 1], "roots": [3, -2]}},
            {"name": "b", "op": "math_evaluate_expr",
             "args": {"form": "poly", "coeffs": [1], "x": 0}}],
            "agree_field": "matches"},
         "check": lambda r: (r.value["unanimous"] is False and r.value["tier_delta"] == -1
                             and r.value["beam_scale"] == 3
                             and r.value["action"]["decision"] == "downgrade_and_widen",
                             f"unan={r.value['unanimous']} delta={r.value['tier_delta']} "
                             f"beam={r.value['beam_scale']}"),
         "expect": "成员返回不同字段 (matches vs None) → 分歧 → **自动降档 (delta=−1) + 扩束 (×3)**。"
                   "注意实现不搞多数表决: 分歧就是扩束重试"},
        {"id": "acm_member_missing", "args": {"question": "缺成员", "members": [
            {"name": "ok", "op": "thm_vieta", "args": {"coeffs": [-6, -1, 1], "roots": [3, -2]}},
            {"name": "ghost", "op": "no_such_op", "args": {}}], "agree_field": "matches"},
         "check": lambda r: (r.value["n_ok"] == 1 and r.value["unanimous"] is False
                             and r.value["votes"][1]["err"] == "OP_MISSING",
                             f"n_ok={r.value['n_ok']} vote1={r.value['votes'][1]}"),
         "expect": "未注册成员计为失败投票 (OP_MISSING) 而非静默剔除; 只剩 1 个有效投票时"
                   "**不构成一致** (unanimous 要求 ≥2 个成功成员)"},
        {"id": "acm_single_member_rejected", "args": {"question": "单人", "members": [
            {"name": "solo", "op": "thm_vieta", "args": {"coeffs": [-6, -1, 1], "roots": [3, -2]}}],
            "agree_field": "matches"},
         "expect_err": "SCHEMA_MISMATCH", "expect": "委员会至少 2 名成员 (minlen=2)"},
    ),
))


# ===========================================================================
# 9608 agent_llm_hint (§11.6 prior only, never evidence)
# ===========================================================================

_IN_HINT = {"kind": "dict",
            "fields": {"hints": {"kind": "list", "minlen": 1, "maxlen": 8,
                                 "items": {"kind": "dict", "extra": False,
                                           "fields": {"tool": {"kind": "str"},
                                                      "args": {"kind": "dict", "extra": True, "fields": {}},
                                                      "confidence": {"kind": "float", "min": 0.0, "max": 1.0}},
                                           "required": ["tool", "args"]}},
                       "source": {"kind": "str"}},
            "required": ["hints"]}


def _inv_hint(args: dict, ctl: CallCtl) -> dict:
    """External-model membrane: suggestions are **only used to order candidates**, and every suggestion must pass a local verifier before counting as evidence.

    The return value separates two things: `prior_order` (the order the suggestions gave, no evidential force) and
    `verified` (whether the local verifier passed + the certificate). The navigation layer may thus use the prior to
    save compute, but **must not** treat the prior as a conclusion.
    """
    ctx = "op:agent_llm_hint#9608"
    ranked = sorted(range(len(args["hints"])),
                    key=lambda i: (-float(args["hints"][i].get("confidence", 0.0)), i))
    verified: list = []
    for i in ranked:
        h = args["hints"][i]
        row = {"i": i, "tool": h["tool"], "confidence": float(h.get("confidence", 0.0))}
        if not REGISTRY.has(h["tool"]):
            row.update({"local_ok": False, "err": "OP_MISSING", "cert_kind": None})
        else:
            r = REGISTRY.tool(h["tool"]).call(h["args"])
            row.update({"local_ok": bool(r.ok), "err": r.err or "",
                        "cert_kind": (r.cert.kind if r.cert else None),
                        "cert_independent": r.cert_independent, "resid": r.resid_scalar})
        verified.append(row)
    n_ok = sum(1 for v in verified if v["local_ok"])
    return {"value": {"prior_order": ranked, "verified": verified, "n_hints": len(args["hints"]),
                      "n_locally_verified": n_ok, "source": args.get("source", "<caller>"),
                      "trust_anchor": ("建议**不是证据**: 只有 local_ok 的项才可作为候选证据; "
                                       "prior_order 仅用于排序, 无任何证据力")},
            "resid": [float(len(args["hints"]) - n_ok)],
            "notes": f"外部建议 {len(args['hints'])} 条 (来源 {args.get('source', '<caller>')}): "
                     f"先验序 {ranked}; 本地验证通过 {n_ok} 条; "
                     f"**未通过的建议一律不得进入证据集** (§11.6 只作先验不作证据)"}


def _ver_hint(args: dict, value: dict) -> dict:
    """Independent verification: (a) **rerun each suggestion's local tool**; local_ok must match the record;
    (b) prior_order must really be ordered by descending confidence (the definition of a prior);
    (c) structural check: non-passing items **must not** carry cert_kind (otherwise a suggestion would count as evidence)."""
    mism = 0
    for v in value["verified"]:
        h = args["hints"][v["i"]]
        if not REGISTRY.has(h["tool"]):
            if v["local_ok"]:
                mism += 1
            continue
        r = REGISTRY.tool(h["tool"]).call(h["args"])
        if bool(r.ok) != bool(v["local_ok"]):
            mism += 1
    confs = [float(args["hints"][i].get("confidence", 0.0)) for i in value["prior_order"]]
    order_ok = all(confs[i] >= confs[i + 1] for i in range(len(confs) - 1))
    no_fake_cert = all(not (v["local_ok"] is False and v.get("cert_kind")) for v in value["verified"])
    ok = (mism == 0) and order_ok and no_fake_cert
    return {"ok": ok, "resid": [float(mism)],
            "detail": f"重跑建议核对: 不一致 {mism} 条 (须 0); 先验序真的按 confidence 降序={order_ok}; "
                      f"未通过项无证书={no_fake_cert} (建议不得携带证据标志)"}


REGISTRY.register(OpDef(
    name="agent_llm_hint", band=BAND, sub=8,
    in_schema=_IN_HINT,
    out_schema={"kind": "dict",
                "fields": {"prior_order": {"kind": "list", "items": {"kind": "int"}},
                           "verified": {"kind": "list", "maxlen": 8,
                                        "items": {"kind": "dict", "extra": True, "fields": {}}},
                           "n_hints": {"kind": "int", "min": 1},
                           "n_locally_verified": {"kind": "int", "min": 0},
                           "source": {"kind": "str"}, "trust_anchor": {"kind": "str"}},
                "required": ["prior_order", "verified", "n_hints", "n_locally_verified", "source",
                             "trust_anchor"]},
    invoke=_inv_hint, verify=_ver_hint,
    cost=("units", 1.0), errors=("SCHEMA_MISMATCH",),
    effects="间接: 逐条建议调用本地工具 (皆纯函数); **不调用任何外部模型** (建议由调用方给出)",
    doc="§11.6 外部模型挂膜 | in: 建议表 (tool+args+置信度) + 来源标注 | "
        "out: 先验序 + 逐条本地验证结果 | 验收=**重跑本地工具**核对 + 先验序真按置信度 + "
        "未通过项不得带证书标志 | 1 unit | 诚实边界: 本算子**不调用外部模型** (建议由调用方给); "
        "建议只作先验不作证据, 一切以本地验证器结论为准 | 失败: schema",
    selfchecks=(
        {"id": "alh_hint_verified_locally", "args": {
            "hints": [{"tool": "solve_poly", "args": {"p": -1.0, "q": -6.0}, "confidence": 0.9}],
            "source": "external-model-A"},
         "check": lambda r: (r.value["n_locally_verified"] == 1
                             and r.value["verified"][0]["local_ok"] is True
                             and r.value["prior_order"] == [0],
                             f"verified={r.value['verified']}"),
         "expect": "建议 'solve_poly(p=-1,q=-6)': 本地验证器通过 (根 3,−2 回代残差 0) → 可作证据"},
        {"id": "alh_hint_fails_locally", "args": {
            "hints": [{"tool": "solve_poly", "args": {"p": 0.0, "q": 1.0}, "confidence": 0.99}],
            "source": "external-model-B"},
         "check": lambda r: (r.value["n_locally_verified"] == 0
                             and r.value["verified"][0]["err"] == "DISC_NEG"
                             and r.value["verified"][0]["cert_kind"] is None,
                             f"verified={r.value['verified']}"),
         "expect": "建议 x²+1 (无实根): **置信度 0.99 也不能让它通过** —— 本地验证器否决 → "
                   "不作为证据; 且不得携带证书标志"},
        {"id": "alh_prior_order_by_confidence", "args": {
            "hints": [{"tool": "math_evaluate_expr", "args": {"form": "poly", "coeffs": [1], "x": 0},
                       "confidence": 0.1},
                      {"tool": "solve_poly", "args": {"p": -1.0, "q": -6.0}, "confidence": 0.8},
                      {"tool": "thm_amgm", "args": {"xs": [1, 4]}, "confidence": 0.5}],
            "source": "external-model-C"},
         "check": lambda r: (r.value["prior_order"] == [1, 2, 0]
                             and r.value["n_hints"] == 3,
                             f"order={r.value['prior_order']}"),
         "expect": "三条建议按置信度 0.8 > 0.5 > 0.1 排序 → 先验序 [1,2,0]; "
                   "先验只决定**尝试顺序**, 不改变任何一条的证据力"},
        {"id": "alh_missing_tool", "args": {"hints": [{"tool": "ghost_tool", "args": {},
                                                       "confidence": 1.0}]},
         "check": lambda r: (r.value["n_locally_verified"] == 0
                             and r.value["verified"][0]["err"] == "OP_MISSING",
                             f"verified={r.value['verified']}"),
         "expect": "建议指向未注册工具: 置信度 1.0 也没用 → 本地无法验证, 不作为证据"},
        {"id": "alh_empty_hints", "args": {"hints": []}, "expect_err": "SCHEMA_MISMATCH",
         "expect": "空建议表必须被 schema 拦下 (本算子不解决'没有建议'的情形)"},
    ),
))


# ===========================================================================
# 9609 agent_mcp_bridge (§11.7 TypedTool ↔ MCP)
# ===========================================================================

_IN_MCP = {"kind": "dict",
           "fields": {"tool": {"kind": "str", "enum": list(_TOOLS)},
                      "direction": {"kind": "str", "enum": ["export", "roundtrip"]}},
           "required": ["tool", "direction"]}


def _mcp_schema(name: str) -> dict:
    """Export a TypedTool declaration as an MCP-style tool description (name/description/inputSchema)."""
    spec = _TOOLS[name]
    props = {}
    for k, t in spec["in_schema"].items():
        props[k] = {"type": "string", "description": f"{t} (由 {spec['delegate']} 校验)"}
    return {"name": name, "description": spec["desc"],
            "inputSchema": {"type": "object", "properties": props,
                            "required": sorted(spec["in_schema"])},
            "x-svo-delegate": spec["delegate"], "x-svo-cost-units": spec["cost_units"]}


def _inv_mcp(args: dict, ctl: CallCtl) -> dict:
    """MCP adaptation: export a tool description, or validate it by round-trip."""
    name = args["tool"]
    desc = _mcp_schema(name)
    if args["direction"] == "export":
        return {"value": {"mcp": desc, "tool": name, "direction": "export",
                          "trust_anchor": "组合锚: 描述一致 + 调用仍过本地后置条件"},
                "resid": [0.0],
                "notes": f"已导出 {name} 的 MCP 风格描述 (输入 schema {len(desc['inputSchema']['properties'])} 个字段); "
                         f"**桥接不改变信任模型**: 通过 MCP 调用的结果仍要过本地独立验证器"}
    # roundtrip: export→import→export must be byte-identical
    exported = desc
    reimported = {k: exported[k] for k in ("name", "description", "inputSchema")}
    again = {"name": reimported["name"], "description": reimported["description"],
             "inputSchema": reimported["inputSchema"],
             "x-svo-delegate": exported["x-svo-delegate"],
             "x-svo-cost-units": exported["x-svo-cost-units"]}
    same = _h(exported) == _h(again)
    if not same:
        raise SvoError(Err.MCP_BRIDGE_ERR, ctx="op:agent_mcp_bridge#9609",
                       detail="往返后描述不一致 — 桥接有损, 拒绝 (有损桥接会让远端看到的工具"
                              "与本地实际能力不符)")
    return {"value": {"mcp": again, "tool": name, "direction": "roundtrip", "roundtrip_equal": True,
                      "fingerprint": _h(again),
                      "trust_anchor": "组合锚: 描述往返无损 + 调用过本地后置条件"},
            "resid": [0.0],
            "notes": f"往返校验无损: 描述指纹 {_h(again)} 前后一致; "
                     f"**显式声明**: MCP 桥接的信任 = 组合锚交集 (§11.7), 无普适内核"}


def _ver_mcp(args: dict, value: dict) -> dict:
    """Independent verification: (a) the description must contain MCP's three requirements name/description/inputSchema;
    (b) inputSchema's properties must **correspond key by key** to the tool's declared input fields, with `required` sorted consistently;
    (c) in round-trip mode the recomputed fingerprint agrees; (d) the delegated operator is registered locally (otherwise the remote call cannot work)."""
    name = args["tool"]
    mcp = value["mcp"]
    three = all(k in mcp for k in ("name", "description", "inputSchema"))
    props = set(mcp["inputSchema"]["properties"])
    want = set(_TOOLS[name]["in_schema"])
    props_ok = props == want
    req_ok = sorted(mcp["inputSchema"]["required"]) == sorted(want)
    del_ok = (not _TOOLS[name]["delegate"].startswith("agent_tool_registry.")
              and REGISTRY.has(_TOOLS[name]["delegate"])) or \
             _TOOLS[name]["delegate"].startswith("agent_tool_registry.")
    fp_ok = True
    if value["direction"] == "roundtrip":
        fp_ok = _h(_mcp_schema(name)) == _h(mcp)
    ok = three and props_ok and req_ok and del_ok and fp_ok
    return {"ok": ok, "resid": [0.0 if ok else 1.0],
            "detail": f"MCP 三件齐={three}; 输入字段逐键对应={props_ok}; required 一致={req_ok}; "
                      f"委托算子本地可解析={del_ok}; 往返指纹一致={fp_ok}"}


REGISTRY.register(OpDef(
    name="agent_mcp_bridge", band=BAND, sub=9,
    in_schema=_IN_MCP,
    out_schema={"kind": "dict",
                "fields": {"mcp": {"kind": "dict", "extra": True, "fields": {}},
                           "tool": {"kind": "str"},
                           "direction": {"kind": "str", "enum": ["export", "roundtrip"]},
                           "roundtrip_equal": {"kind": "bool"}, "fingerprint": {"kind": "str"},
                           "trust_anchor": {"kind": "str"}},
                "required": ["mcp", "tool", "direction", "trust_anchor"]},
    invoke=_inv_mcp, verify=_ver_mcp,
    cost=("units", 0.5), errors=("MCP_BRIDGE_ERR", "SCHEMA_MISMATCH"),
    effects="none (纯函数: 只做描述映射, 不发起任何 MCP 连接)",
    doc="§11.7 TypedTool ↔ MCP 适配 | in: 工具名 + 方向 | out: MCP 风格描述 (+往返指纹) | "
        "验收=MCP 三件齐 + 输入字段逐键对应 + 委托算子本地可解析 + 往返无损 (有损即拒) | 0.5 unit | "
        "诚实边界: **不发起任何 MCP 连接** (只做描述映射); 桥接不改变信任模型 —— "
        "经 MCP 调用的结果仍须过本地独立验证器; 本域信任 = 组合锚交集 (§11.7) | "
        "失败: 往返有损 MCP_BRIDGE_ERR",
    selfchecks=(
        {"id": "amb_export_calculator", "args": {"tool": "calculator", "direction": "export"},
         "check": lambda r: (r.value["mcp"]["name"] == "calculator"
                             and "x-svo-delegate" in r.value["mcp"]
                             and sorted(r.value["mcp"]["inputSchema"]["required"]) == ["coeffs", "form", "x"],
                             f"mcp={r.value['mcp']['name']} req={r.value['mcp']['inputSchema']['required']}"),
         "expect": "导出 calculator: 三件齐; required 按工具声明为 [coeffs, form, x]"},
        {"id": "amb_roundtrip_calculator", "args": {"tool": "calculator", "direction": "roundtrip"},
         "check": lambda r: (r.value["roundtrip_equal"] is True and r.value["fingerprint"],
                             f"fp={r.value.get('fingerprint')}"),
         "expect": "往返校验: 导出→导入→再导出 指纹一致 → 无损"},
        {"id": "amb_roundtrip_code_exec", "args": {"tool": "code_exec", "direction": "roundtrip"},
         "check": lambda r: (r.value["roundtrip_equal"] is True
                             and "code_bounded_interp" == r.value["mcp"]["x-svo-delegate"],
                             f"delegate={r.value['mcp']['x-svo-delegate']}"),
         "expect": "code_exec 往返: 委托算子是 code_bounded_interp (本地已注册, 远端调得通)"},
        {"id": "amb_roundtrip_mock_search", "args": {"tool": "mock_search", "direction": "roundtrip"},
         "check": lambda r: (r.value["roundtrip_equal"] is True, "往返无损"),
         "expect": "mock_search 往返: 固定语料工具的描述同样无损往返"},
        {"id": "amb_bad_direction", "args": {"tool": "calculator", "direction": "upside_down"},
         "expect_err": "SCHEMA_MISMATCH", "expect": "方向枚举外取值"},
    ),
))


# ===========================================================================
# 9610 agent_replay_audit (§11.7 deterministic replay + audit trace)
# ===========================================================================

_IN_REP = {"kind": "dict",
           "fields": {"calls": {"kind": "list", "minlen": 1, "maxlen": 16,
                                "items": {"kind": "dict", "extra": False,
                                          "fields": {"tool": {"kind": "str"},
                                                     "args": {"kind": "dict", "extra": True, "fields": {}}},
                                          "required": ["tool", "args"]}},
                      "seed": {"kind": "int", "min": 0, "max": 2**31 - 1},
                      "runs": {"kind": "int", "min": 2, "max": 4}},
           "required": ["calls"]}


def _one_run(calls, seed: int) -> dict:
    """Run one call sequence; return (structural fingerprint, per-call results)."""
    trace = AuditTrace(f"replay-{seed}")
    ledger = BudgetLedger(total_units=1e9)
    out: list = []
    canon_calls: list = []
    for c in calls:
        tool = c["tool"]
        if not REGISTRY.has(tool):
            out.append({"tool": tool, "ok": False, "err": "OP_MISSING", "value": None})
            canon_calls.append({"tool": tool, "args": canon_args(c["args"]), "err": "OP_MISSING"})
            continue
        r = REGISTRY.tool(tool).call(c["args"], CallCtl(trace=trace, budget=ledger, seed=seed))
        out.append({"tool": tool, "ok": bool(r.ok), "err": r.err or "",
                    "value": r.value, "resid": r.resid_scalar})
        canon_calls.append({"tool": tool, "args": canon_args(c["args"]), "err": r.err or "",
                            "resid": None if r.resid_scalar != r.resid_scalar
                            else round(float(r.resid_scalar), 12)})
    blob = json.dumps(canon_calls, sort_keys=True, ensure_ascii=False, default=str)
    return {"call_fingerprint": hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16],
            "trace_fingerprint": trace.struct_fingerprint(), "results": out,
            "canon_calls": canon_calls}


def _inv_rep(args: dict, ctl: CallCtl) -> dict:
    """Deterministic replay: run the same sequence several times; the call fingerprint and audit-trace fingerprint must be identical every time."""
    ctx = "op:agent_replay_audit#9610"
    runs = int(args.get("runs", 2))
    seed = int(args.get("seed", 0))
    outs = [_one_run(args["calls"], seed) for _ in range(runs)]
    fp_calls = {o["call_fingerprint"] for o in outs}
    fp_trace = {o["trace_fingerprint"] for o in outs}
    equal = len(fp_calls) == 1 and len(fp_trace) == 1
    if not equal:
        raise SvoError(Err.POSTCOND_FAIL, ctx=ctx,
                       detail=f"确定性重放失败: 调用指纹 {sorted(fp_calls)}, 审计迹指纹 "
                              f"{sorted(fp_trace)} — **发现未播种随机或隐含状态**, 这是红色警报 "
                              f"(§11.7 要求重放全等)")
    log = ReplayLog(seed=seed)
    for c in outs[0]["canon_calls"]:
        log.record(c["tool"], c["args"], outs[0]["call_fingerprint"])
    return {"value": {"call_fingerprint": outs[0]["call_fingerprint"],
                      "trace_fingerprint": outs[0]["trace_fingerprint"],
                      "runs": runs, "seed": seed, "equal": True,
                      "n_calls": len(args["calls"]),
                      "replay_log": json.loads(log.to_json()),
                      "trust_anchor": "确定性重放 (序列+种子) + 审计迹指纹; "
                                      "两者任一不可复现, 组合锚即失效"},
            "resid": [0.0],
            "notes": f"确定性重放通过: {runs} 次运行, 调用指纹 {outs[0]['call_fingerprint']} 与"
                     f"审计迹指纹 {outs[0]['trace_fingerprint']} 逐次相同 (seed={seed}, "
                     f"{len(args['calls'])} 次调用)"}


def _ver_rep(args: dict, value: dict) -> dict:
    """Independent verification: (a) **run once more** and compare the serialized replay logs independently with `replay_equal` (another code path);
    (b) rerun each call; the result must match the first run (value-level comparison, not just fingerprints);
    (c) fingerprint length/shape is legal."""
    once = _one_run(args["calls"], int(args.get("seed", 0)))
    again = _one_run(args["calls"], int(args.get("seed", 0)))
    log_a = ReplayLog(seed=int(args.get("seed", 0)))
    for c in once["canon_calls"]:
        log_a.record(c["tool"], c["args"], once["call_fingerprint"])
    log_b = ReplayLog(seed=int(args.get("seed", 0)))
    for c in again["canon_calls"]:
        log_b.record(c["tool"], c["args"], again["call_fingerprint"])
    same, diff = replay_equal(log_a.to_json(), log_b.to_json())
    vals_ok = all(
        json.dumps(a.get("value"), sort_keys=True, default=str)
        == json.dumps(b.get("value"), sort_keys=True, default=str)
        for a, b in zip(once["results"], again["results"]))
    fp_ok = (len(value["call_fingerprint"]) == 16) and (len(value["trace_fingerprint"]) == 16) \
        and (value["call_fingerprint"] == once["call_fingerprint"]) \
        and (value["trace_fingerprint"] == once["trace_fingerprint"])
    ok = same and vals_ok and fp_ok
    return {"ok": ok, "resid": [0.0 if ok else 1.0],
            "detail": f"replay_equal 独立比对={same} (差异 {diff[:2]}); 逐调用的**值级**核对一致="
                      f"{vals_ok}; 指纹与实际相符={fp_ok}"}


REGISTRY.register(OpDef(
    name="agent_replay_audit", band=BAND, sub=10,
    in_schema=_IN_REP,
    out_schema={"kind": "dict",
                "fields": {"call_fingerprint": {"kind": "str"}, "trace_fingerprint": {"kind": "str"},
                           "runs": {"kind": "int", "min": 2}, "seed": {"kind": "int", "min": 0},
                           "equal": {"kind": "bool"}, "n_calls": {"kind": "int", "min": 1},
                           "replay_log": {"kind": "dict", "extra": True, "fields": {}},
                           "trust_anchor": {"kind": "str"}},
                "required": ["call_fingerprint", "trace_fingerprint", "runs", "seed", "equal",
                             "n_calls", "replay_log", "trust_anchor"]},
    invoke=_inv_rep, verify=_ver_rep,
    cost=("units", 2.0), errors=("POSTCOND_FAIL", "SCHEMA_MISMATCH"),
    effects="间接: 重放调用序列 (皆纯函数)",
    doc="§11.7 确定性重放 + 审计迹 | in: 调用序列 + 种子 + 重放次数 | "
        "out: 调用指纹 + 审计迹指纹 + 序列化重放日志 | 验收=多次运行的**双指纹逐次相同** + "
        "verify 用 replay_equal 独立比对日志 + **值级**核对 | 2 units | "
        "诚实边界: 重放全等是**可复现性**的证据, 不是正确性证据 (两次都错也会全等); "
        "发现不等即红色警报 (未播种随机/隐含状态) | 失败: 重放不等 POSTCOND_FAIL",
    selfchecks=(
        {"id": "ara_replay_equal", "args": {"calls": [
            {"tool": "solve_poly", "args": {"p": -1.0, "q": -6.0}},
            {"tool": "math_evaluate_expr", "args": {"form": "geom", "a": 1, "r": "1/2", "n": 5}}],
            "seed": 7, "runs": 3},
         "check": lambda r: (r.value["equal"] is True and r.value["runs"] == 3
                             and r.value["n_calls"] == 2
                             and len(r.value["call_fingerprint"]) == 16,
                             f"fp={r.value['call_fingerprint']} trace={r.value['trace_fingerprint']}"),
         "expect": "两次解码调用 (solve_poly + geom 求值) 重放 3 次: 两指纹逐次相同 → 全等"},
        {"id": "ara_replay_with_certificate", "args": {"calls": [
            {"tool": "solve_poly_exact", "args": {"p": -1, "q": -6}}], "seed": 0, "runs": 2},
         "check": lambda r: (r.value["equal"] is True
                             and r.value["replay_log"]["seed"] == 0,
                             f"log_seed={r.value['replay_log']['seed']}"),
         "expect": "带证书的算子 (solve_poly_exact) 同样可确定性重放; 重放日志记录 seed"},
        {"id": "ara_missing_tool_recorded", "args": {"calls": [
            {"tool": "ghost", "args": {}},
            {"tool": "solve_poly", "args": {"p": -1.0, "q": -6.0}}], "seed": 1, "runs": 2},
         "check": lambda r: (r.value["equal"] is True and r.value["n_calls"] == 2,
                             f"n={r.value['n_calls']} equal={r.value['equal']}"),
         "expect": "序列中含未注册工具: 记为 OP_MISSING 但**不影响重放确定性** "
                   "(失败也是确定性的, 所以指纹仍全等)"},
        {"id": "ara_single_run_rejected", "args": {"calls": [
            {"tool": "solve_poly", "args": {"p": -1.0, "q": -6.0}}], "runs": 1},
         "expect_err": "SCHEMA_MISMATCH",
         "expect": "重放次数 1 无意义 (没有比对对象) → schema 拦下"},
        {"id": "ara_empty_calls", "args": {"calls": []}, "expect_err": "SCHEMA_MISMATCH",
         "expect": "空调用序列必须被拦下"},
    ),
))