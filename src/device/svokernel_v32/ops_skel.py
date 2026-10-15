# -*- coding: utf-8 -*-
"""V32 proof-skeleton meta-operators (band 10000, rows 10001..10006) -- the scope is the
**goal-hypothesis context stack**, not numbers.

Six operators (design doc §5):
  skel_case_split           9951  split the focused goal into **exhaustive** branches by the predicate (complete residue system)
  skel_induction_scheme     9952  produce the base-case + step (difference form) subgoal pair
  skel_subgoal_focus        9953  subgoal stack push / focus switch / hand back to the parent
  skel_leaf_close           9954  leaf-goal closure (complete-residue decision; **reports non-closure**)
  skel_contradiction_close  9955  negated goal: assume a solution -> enumerate residues -> contradiction closure
  skel_qed_check            9956  terminal audit (all closed + induction base/step paired) => QED

Three disciplines that differ from the v31 numeric operators:
1. **The input is goal text + small integer codes**, the output is **context state**
   (stack/branches/focus), not a number -- this is the ontological difference between "proof
   structure" and "compute an expression";
2. `closed=false` / `qed=false` are **legitimate return values**, not errors: "not proven" must
   never be packaged as "proven"; the independent checker likewise re-verifies locally that it
   really did not close;
3. `effects` is always none: the stack is a **value**, not global state -- no second call path can
   modify it.
"""
from __future__ import annotations

from .v32registry import SKEL_CERT_KINDS, V32OpDef, V32REGISTRY

from svokernel.certify import Certificate
from svokernel.errors import Err, SvoError
from svokernel.tool import CallCtl

from . import skelmath as sm

BAND_SKEL = 10000
LAYER_SKEL = 4
_RULE = {"kind": "str"}                       # free-form output-side string (not part of the value vocabulary)


def _sdict(fields, req):
    return {"kind": "dict", "fields": fields, "required": req}


def _int(lo=None, hi=None):
    d = {"kind": "int"}
    if lo is not None:
        d["min"] = lo
    if hi is not None:
        d["max"] = hi
    return d


def _ilist(n=16):
    return {"kind": "list", "items": {"kind": "int"}, "minlen": 0, "maxlen": n}


def _slist(n=16):
    return {"kind": "list", "items": {"kind": "str"}, "minlen": 0, "maxlen": n}


def _reg(**kw):
    kw.setdefault("status", "verified")
    kw.setdefault("layer", LAYER_SKEL)
    kw.setdefault("skeleton", True)
    kw.setdefault("requires_cert", True)
    kw.setdefault("effects", "none (纯函数)")
    return V32REGISTRY.register(V32OpDef(**kw))


def _fail(code, op, detail):
    raise SvoError(code, ctx="op:%s" % op, detail=detail)


def _chk(name, cond, detail):
    if not cond:
        _fail(Err.PRECOND_FAIL, name, detail)


# ===========================================================================
# 9951  skel_case_split -- case analysis (complete residue system)
# ===========================================================================

_CS_IN = _sdict({
    "goal": {"kind": "str"}, "m": _int(2, 12), "vid": _int(0, 4), "gid": _int(0, 16),
    "c0": _int(0, 16),                      # first child node id (children = c0 .. c0+m-1, declared by the caller)
}, ["goal", "m", "vid", "gid", "c0"])
_CS_OUT = _sdict({
    "gid": {"kind": "int"}, "m": {"kind": "int"}, "vid": {"kind": "int"},
    "ncase": {"kind": "int"}, "cover": _ilist(), "cases_res": _ilist(),
    "cases_goal": _slist(), "cases_poly": _slist(), "exhaustive": {"kind": "bool"},
    "focus_gid": {"kind": "int"}, "rule": _RULE,
}, ["gid", "m", "vid", "ncase", "cover", "cases_res", "cases_goal", "cases_poly",
    "exhaustive", "focus_gid", "rule"])


def _cs_precond(args):
    m = args["m"]
    if m < 2:
        return False, "m ≥ 2 (m=1 是平凡划分, 拒: 它不产生任何信息)"
    if m > 12:
        return False, "m ≤ 12 (划分枝数上界, 与 24 槽位/25 帧预算一致)"
    if args["vid"] not in range(5):
        return False, "vid ∈ 0..4"
    return True, ""


def _invoke_cs(args, ctl):
    goal, m, vid, gid = args["goal"], int(args["m"]), int(args["vid"]), int(args["gid"])
    c0 = int(args["c0"])
    try:
        g = sm.parse_goal(goal)
    except sm.ExprError as e:
        _fail(Err.SCHEMA_MISMATCH, "skel_case_split#9951", "目标文本不可解析: %s" % (e,))
    if g.form != "U":
        _fail(Err.PRECOND_FAIL, "skel_case_split#9951",
              "形态 %r 不是全称目标 (R4 只作用于 U 形态)" % (g.form,))
    if g.m != m:
        _fail(Err.PRECOND_FAIL, "skel_case_split#9951",
              "目标模数 %d 与 args.m=%d 不符" % (g.m, m))
    if sm.VAR_ID.get(g.var) != vid:
        _fail(Err.PRECOND_FAIL, "skel_case_split#9951",
              "目标变量 %r 的 vid=%r 与 args.vid=%d 不符" % (g.var, sm.VAR_ID.get(g.var), vid))
    kname = sm.new_var(0, avoid=(g.var,))
    cover, cres, cgoal, cpoly = [], [], [], []
    for r in range(m):
        sub = sm.subst_text(g.factored, g.var, r, m, kname)
        c = sm.subst_affine(g.coeffs, r, m)
        cover.append(r)
        cres.append(r)
        cgoal.append(sm.leaf_goal_text(m, sub, kname))
        cpoly.append(sm.poly_str(c, kname))
    cert = Certificate(kind="case_exhaustive",
                       payload={"goal": goal, "m": m, "var": g.var,
                                "cases": [{"res": r, "goal": cg} for r, cg in zip(cres, cgoal)],
                                "cover": cover},
                       issued_by="skel_case_split")
    return {"value": {"gid": gid, "m": m, "vid": vid, "ncase": m, "cover": cover,
                      "cases_res": cres, "cases_goal": cgoal, "cases_poly": cpoly,
                      "exhaustive": True, "focus_gid": c0, "rule": "R4"},
            "resid": [0.0], "cert": cert, "cost": 2.0,
            "notes": "按 v mod %d 的完备剩余系拆 %d 枝 (替换参数 %s); 子节点 g%d..g%d, 焦点 g%d"
                     % (m, m, kname, c0, c0 + m - 1, c0)}


def _verify_cs(args, value):
    """Independent verifier: re-parse the goal, recompute the substituted polynomials and the
    residue set branch by branch (a different path from invoke: consumes args/value only)."""
    bad = []
    try:
        g = sm.parse_goal(args["goal"])
    except sm.ExprError as e:
        return {"ok": False, "resid": [1.0], "detail": "目标重解析失败: %s" % (e,)}
    m = int(args["m"])
    if value["ncase"] != m or sorted(value["cover"]) != list(range(m)):
        bad.append("枝数/覆盖集不符")
    if len(value["cases_goal"]) != m or len(value["cases_poly"]) != m:
        bad.append("cases 长度不符")
    else:
        kname = None
        for r in range(m):
            try:
                sg = sm.parse_goal(value["cases_goal"][r])
            except sm.ExprError as e:
                bad.append("子目标 %d 重解析失败: %s" % (r, e))
                continue
            kname = kname or sg.var
            if tuple(sm.subst_affine(g.coeffs, r, m)) != tuple(sg.coeffs):
                bad.append("枝 %d 替换多项式不符" % r)
        if kname is not None and not sm.divides_all(sm.parse_poly(value["cases_poly"][0], kname), m):
            bad.append("首枝并不恒被 m 整除 (该描述本应给出可闭合的枝)")
    return {"ok": not bad, "resid": [float(len(bad))],
            "detail": "逐枝复算一致 (m=%d)" % m if not bad else "; ".join(bad)}


_reg(
    name="skel_case_split", band=BAND_SKEL, sub=1,
    in_schema=_CS_IN, out_schema=_CS_OUT, invoke=_invoke_cs, verify=_verify_cs,
    precond=_cs_precond, cert_kind="case_exhaustive",
    cost=("units", 2.0), errors=("SCHEMA_MISMATCH", "PRECOND_FAIL", "POSTCOND_FAIL"),
    doc=("Proof skeleton: case analysis. Split the focused universal goal U(m, f(v)) into m "
         "branches by the **complete residue system** of v mod m; each branch's subgoal is the "
         "substituted leaf goal U(m, f(r + m*k)). Certificate case_exhaustive is re-verified by "
         "the independent checker: cover is exactly {0..m-1} and every branch's substituted "
         "polynomial agrees coefficient by coefficient with the original goal. Lemma "
         "(completeness): ∀v∈ℤ. m|p(v) ⟺ ∀r∈{0..m-1}. m|p(r) -- so m branches are exhaustive, not "
         "sampling."),
    stmt="按谓词把当前目标分拆为穷尽分支子目标", thm="T-V32-CS",
    selfchecks=(
        {"id": "cs_two_branch", "args": {"goal": "for every integer n, 2 divides n*(n+1)",
                                         "m": 2, "vid": 0, "gid": 0, "c0": 1},
         "check": lambda r: (r.value["ncase"] == 2 and r.value["cover"] == [0, 1]
                             and r.value["exhaustive"] and r.value["focus_gid"] == 1
                             and len(r.value["cases_goal"]) == 2,
                             "2 枝, cover=%r, 焦点 g%d" % (r.value["cover"], r.value["focus_gid"])),
         "expect": "两连积按奇偶拆 2 枝"},
        {"id": "cs_three_branch", "args": {"goal": "for every integer n, 3 divides n*(n+1)*(n+2)",
                                           "m": 3, "vid": 0, "gid": 0, "c0": 1},
         "check": lambda r: (r.value["ncase"] == 3 and r.value["cover"] == [0, 1, 2],
                             "3 枝, cover=%r" % (r.value["cover"],)),
         "expect": "三连积按 mod 3 拆 3 枝"},
        {"id": "cs_six_branch", "args": {"goal": "for every integer n, 6 divides n*(n+1)*(n+2)*(n+3)",
                                         "m": 6, "vid": 0, "gid": 0, "c0": 1},
         "check": lambda r: (r.value["ncase"] == 6 and r.value["cover"] == list(range(6)),
                             "6 枝, cover=%r" % (r.value["cover"],)),
         "expect": "四连积按 mod 6 拆 6 枝 (枝管理压力的来源)"},
        {"id": "cs_determinism", "args": {"goal": "for every integer t, 2 divides t^2 - t",
                                          "m": 2, "vid": 3, "gid": 0, "c0": 1},
         "check": lambda r: (r.value["cases_poly"][0] == "4*k^2 - 2*k",
                             "首枝展开 %r (规范形, 确定性)" % (r.value["cases_poly"][0],)),
         "expect": "同输入同输出: t^2-t 在 t=0+2k 下展开为 4*k^2 - 2*k"},
        {"id": "cs_m1_reject", "args": {"goal": "for every integer n, 1 divides n*(n+1)",
                                        "m": 1, "vid": 0, "gid": 0, "c0": 1},
         "expect_err": "SCHEMA_MISMATCH", "expect": "m=1 被 schema 门拒 (min=2)"},
        {"id": "cs_m13_reject", "args": {"goal": "for every integer n, 13 divides n*(n+1)",
                                         "m": 13, "vid": 0, "gid": 0, "c0": 1},
         "expect_err": "SCHEMA_MISMATCH", "expect": "m=13 被 schema 门拒 (max=12)"},
        {"id": "cs_neg_goal_reject", "args": {"goal": "there is no integer n with 4 dividing n^2 - 2",
                                              "m": 4, "vid": 0, "gid": 0, "c0": 1},
         "expect_err": "PRECOND_FAIL", "expect": "否定形态目标不许走 case_split (规则表 R3)"},
        {"id": "cs_modulus_mismatch", "args": {"goal": "for every integer n, 2 divides n*(n+1)",
                                               "m": 3, "vid": 0, "gid": 0, "c0": 1},
         "expect_err": "PRECOND_FAIL", "expect": "args.m 与目标模数不符 ⇒ 拒 (不许错配)"},
        {"id": "cs_vid_mismatch", "args": {"goal": "for every integer n, 2 divides n*(n+1)",
                                           "m": 2, "vid": 1, "gid": 0, "c0": 1},
         "expect_err": "PRECOND_FAIL", "expect": "vid 与目标变量不符 ⇒ 拒"},
    ),
)


# ===========================================================================
# 9952  skel_induction_scheme -- base case + step
# ===========================================================================

_IN_IN = _sdict({
    "goal": {"kind": "str"}, "m": _int(2, 12), "vid": _int(0, 4), "gid": _int(0, 16),
    "n0": _int(0, 1),
    "c0": _int(0, 16),                      # children: c0 = base case, c0+1 = step (declared by the caller)
}, ["goal", "m", "vid", "gid", "n0", "c0"])
_IN_OUT = _sdict({
    "gid": {"kind": "int"}, "m": {"kind": "int"}, "vid": {"kind": "int"},
    "n0": {"kind": "int"}, "base_goal": {"kind": "str"}, "base_value": {"kind": "int"},
    "step_goal": {"kind": "str"}, "ih": {"kind": "str"}, "diff": {"kind": "str"},
    "diff_gcd": {"kind": "int"}, "diff_ok": {"kind": "bool"}, "ncase": {"kind": "int"},
    "c_base": {"kind": "int"}, "c_step": {"kind": "int"}, "rule": _RULE,
}, ["gid", "m", "vid", "n0", "base_goal", "base_value", "step_goal", "ih", "diff",
    "diff_gcd", "diff_ok", "ncase", "c_base", "c_step", "rule"])


def _invoke_ind(args, ctl):
    goal, m, vid, gid, n0 = (args["goal"], int(args["m"]), int(args["vid"]),
                             int(args["gid"]), int(args["n0"]))
    c0 = int(args["c0"])
    try:
        g = sm.parse_goal(goal)
    except sm.ExprError as e:
        _fail(Err.SCHEMA_MISMATCH, "skel_induction_scheme#9952", "目标文本不可解析: %s" % (e,))
    if g.form != "U":
        _fail(Err.PRECOND_FAIL, "skel_induction_scheme#9952",
              "形态 %r 不是全称目标 (R5 只作用于 U 形态)" % (g.form,))
    if g.m != m:
        _fail(Err.PRECOND_FAIL, "skel_induction_scheme#9952",
              "目标模数 %d 与 args.m=%d 不符" % (g.m, m))
    kn = sm.new_var(0, avoid=(g.var,))
    diff = sm.poly_diff(g.coeffs)
    base_val = sm.poly_eval(g.coeffs, n0)
    diff_ok = sm.divides_all(diff, m)
    base_goal = sm.universal_goal_text(m, sm.subst_text(g.factored, g.var, n0, 1, "j"), "j")
    ih = "%d divides %s (induction hypothesis)" % (m, g.factored)
    # the step subgoal is given in **difference form** -- it is a goal this small world can close
    # directly with skel_leaf_close (the closure language).
    step_goal = sm.universal_goal_text(m, sm.poly_str(diff, kn), kn)
    step_raw = ("for every integer %s: %d divides %s, given %s"
                % (kn, m, sm.subst_text(g.factored, g.var, 1, 1, kn), ih))
    cert = Certificate(kind="induction_scheme",
                       payload={"goal": goal, "m": m, "var": g.var, "n0": n0,
                                "base_value": base_val, "diff_repr": sm.poly_str(diff, g.var),
                                "scheme": "induction",
                                "base_ok": bool(base_val % m == 0), "diff_ok": bool(diff_ok)},
                       issued_by="skel_induction_scheme")
    return {"value": {"gid": gid, "m": m, "vid": vid, "n0": n0, "base_goal": base_goal,
                      "base_value": base_val, "step_goal": step_goal, "ih": ih,
                      "diff": sm.poly_str(diff, g.var), "diff_gcd": sm.poly_gcd(diff),
                      "diff_ok": bool(diff_ok), "ncase": 2, "c_base": c0, "c_step": c0 + 1,
                      "rule": "R5"},
            "resid": [0.0 if (diff_ok and base_val % m == 0) else 1.0],
            "cert": cert, "cost": 2.0,
            "notes": "基例 f(%d)=%d (%s) ⇒ 子节点 g%d; 步进差分 %s (恒被 %d 整除=%r) ⇒ 子节点 g%d"
                     " | 原始步进形态: %s"
                     % (n0, base_val, "成立" if base_val % m == 0 else "不成立", c0,
                        sm.poly_str(diff, g.var), m, diff_ok, c0 + 1, step_raw)}


def _verify_ind(args, value):
    bad = []
    try:
        g = sm.parse_goal(args["goal"])
    except sm.ExprError as e:
        return {"ok": False, "resid": [1.0], "detail": "目标重解析失败: %s" % (e,)}
    m, n0 = int(args["m"]), int(args["n0"])
    want_diff = sm.poly_diff(g.coeffs)
    got_diff = sm.parse_poly(value["diff"], g.var)
    if tuple(want_diff) != tuple(got_diff):
        bad.append("差分多项式不符")
    if int(value["base_value"]) != sm.poly_eval(g.coeffs, n0):
        bad.append("基例值不符")
    if bool(value["diff_ok"]) != sm.divides_all(want_diff, m):
        bad.append("diff_ok 与独立判定不符")
    if int(value["diff_gcd"]) != sm.poly_gcd(want_diff):
        bad.append("差分系数 gcd 不符")
    applies = bool(value["diff_ok"]) and (int(value["base_value"]) % m == 0)
    return {"ok": not bad, "resid": [float(len(bad)) if bad else (0.0 if applies else 1.0)],
            "detail": ("差分/基例/gcd 独立复算一致" if not bad else "; ".join(bad))
                      + ("" if applies else " | **方案对当前目标不成立 (残差 1.0, 如实)**")}


_reg(
    name="skel_induction_scheme", band=BAND_SKEL, sub=2,
    in_schema=_IN_IN, out_schema=_IN_OUT, invoke=_invoke_ind, verify=_verify_ind,
    cert_kind="induction_scheme",
    cost=("units", 2.0), errors=("SCHEMA_MISMATCH", "PRECOND_FAIL", "POSTCOND_FAIL"),
    doc=("Proof skeleton: induction scheme. For a power-difference goal, produce the **base case** "
         "U(m, f(n0)) and the **step**: goal shape m | f(v+1) given IH m | f(v), together with the "
         "equivalent **difference form** ∀k. m | (f(k+1)-f(k)). Certificate induction_scheme is "
         "re-verified by the independent checker (it recomputes the difference / base value / "
         "structure). diff_ok=false is a **legitimate return value**: when the scheme does not hold "
         "for the current goal it is reported as measured, never faked as proven."),
    stmt="生成归纳的基例与步进子目标对", thm="T-V32-IND",
    selfchecks=(
        {"id": "ind_cube3", "args": {"goal": "for every integer n, 3 divides n^3 - n",
                                     "m": 3, "vid": 0, "gid": 0, "n0": 0, "c0": 1},
         "check": lambda r: (r.value["diff"] == "3*n^2 + 3*n" and r.value["diff_ok"]
                             and r.value["base_value"] == 0 and r.value["ncase"] == 2,
                             "差分 %r, 基例值 %d" % (r.value["diff"], r.value["base_value"])),
         "expect": "n^3-n: 差分 3n+3n^2 恒被 3 整除, 基例 0"},
        {"id": "ind_pow5", "args": {"goal": "for every integer n, 5 divides n^5 - n",
                                    "m": 5, "vid": 0, "gid": 0, "n0": 0, "c0": 1},
         "check": lambda r: (r.value["diff_ok"] and r.value["diff_gcd"] % 5 == 0,
                             "差分 gcd=%d" % r.value["diff_gcd"]),
         "expect": "n^5-n 的差分系数 gcd 被 5 整除"},
        {"id": "ind_weak_honest", "args": {"goal": "for every integer n, 4 divides n^3 - n",
                                           "m": 4, "vid": 0, "gid": 0, "n0": 0, "c0": 1},
         "check": lambda r: (not r.value["diff_ok"] and r.resid_scalar > 0,
                             "如实回报: diff_ok=%r (残差 %.1f) —— 不假装证成"
                             % (r.value["diff_ok"], r.resid_scalar)),
         "expect": "4 ∤ n^3-n 的差分 ⇒ diff_ok=false 且残差非零 (诚实回报)"},
        {"id": "ind_n0_shift", "args": {"goal": "for every integer n, 2 divides n^2 - n",
                                        "m": 2, "vid": 0, "gid": 0, "n0": 1, "c0": 1},
         "check": lambda r: (r.value["base_value"] == 0 and r.value["n0"] == 1,
                             "基例取 n0=1: f(1)=%d" % r.value["base_value"]),
         "expect": "基例索引可移 (n0=1)"},
        {"id": "ind_n0_reject", "args": {"goal": "for every integer n, 2 divides n^2 - n",
                                         "m": 2, "vid": 0, "gid": 0, "n0": 2, "c0": 1},
         "expect_err": "SCHEMA_MISMATCH", "expect": "n0=2 被 schema 门拒 (max=1)"},
        {"id": "ind_prod_also_works", "args": {"goal": "for every integer n, 2 divides n*(n+1)",
                                               "m": 2, "vid": 0, "gid": 0, "n0": 0, "c0": 1},
         "check": lambda r: (r.value["diff_ok"] and r.value["diff"] == "2*n + 2"
                             and r.value["base_value"] == 0,
                             "两连积的差分 %r 亦恒被 2 整除 (归纳对它是**可行**策略, 只是规则表"
                             " R4 把它判给 case_split)" % (r.value["diff"],)),
         "expect": "同一目标可多策略: 算子如实报告归纳在此成立 (策略选择是模型的任务, 不是算子的)"},
    ),
)


# ===========================================================================
# 9953  skel_subgoal_focus -- subgoal stack push / focus switch
# ===========================================================================

_SF_IN = _sdict({
    "act": {"kind": "str", "enum": ["split", "apply", "rewrite", "noop"]},
    "gid": _int(0, 16), "vid": _int(0, 4), "stk": {"kind": "str"}, "tk": {"kind": "str"},
    "c1": _int(0, 16), "c2": _int(0, 16),   # the two child node ids for act="split" (declared by the caller)
}, ["act", "gid", "vid", "stk", "tk", "c1", "c2"])
_SF_OUT = _sdict({
    "act": {"kind": "str"}, "gid": {"kind": "int"}, "vid": {"kind": "int"},
    "focus": {"kind": "int"}, "depth": {"kind": "int"}, "n_open": {"kind": "int"},
    "stk_after": {"kind": "str"}, "focused_goal": {"kind": "str"}, "nodes": {"kind": "int"},
    "rule": _RULE,
}, ["act", "gid", "vid", "focus", "depth", "n_open", "stk_after", "focused_goal", "nodes",
    "rule"])


def _sf_precond(args):
    if args["act"] not in ("split", "apply", "rewrite", "noop"):
        return False, "act ∈ {split, apply, rewrite, noop}"
    if args["gid"] not in range(17):
        return False, "gid ∈ 0..16"
    return True, ""


def _invoke_sf(args, ctl):
    act, gid, vid = args["act"], int(args["gid"]), int(args["vid"])
    c1, c2 = int(args["c1"]), int(args["c2"])
    try:
        st = sm.parse_stack_str(args["stk"])
    except sm.ExprError as e:
        _fail(Err.SCHEMA_MISMATCH, "skel_subgoal_focus#9953", "栈状态串不可解析: %s" % (e,))
    nodes = [dict(x) for x in st["nodes"]]
    by = {x["gid"]: x for x in nodes}
    focus, depth = int(st["focus"]), int(st["depth"])
    nopen_before = sm.n_open_of(nodes)
    tk = args["tk"]
    cert = Certificate(kind="stack_machine",
                       payload={"before": {"nodes": [dict(x) for x in nodes], "focus": focus,
                                           "depth": depth},
                                "act": act, "gid": gid, "c1": c1, "c2": c2,
                                "n_open_in": nopen_before, "after": None},
                       issued_by="skel_subgoal_focus")
    if act == "split":
        if focus not in by:
            _fail(Err.PRECOND_FAIL, "skel_subgoal_focus#9953",
                  "split 的焦点 %d 不在栈里 (栈: %r)" % (focus, sorted(by)))
        if by[focus]["kind"] != "claim":
            _fail(Err.PRECOND_FAIL, "skel_subgoal_focus#9953",
                  "split 只作用于合取节点 (焦点 kind=%r)" % (by[focus]["kind"],))
        try:
            cg = sm.parse_goal(tk)
        except sm.ExprError as e:
            _fail(Err.SCHEMA_MISMATCH, "skel_subgoal_focus#9953", "tk 不可解析: %s" % (e,))
        if cg.form not in ("C", "NC") or len(cg.claims) != 2:
            _fail(Err.PRECOND_FAIL, "skel_subgoal_focus#9953",
                  "split 的目标必须是两条 claim 的合取 (form=%r, n=%d; "
                  "C=全称 claim 合取, NC=不存在性 claim 合取)"
                  % (cg.form, len(cg.claims)))
        for (mm, ff) in cg.claims:
            try:
                sm.parse_poly(ff, cg.var)
            except sm.ExprError as e:
                _fail(Err.SCHEMA_MISMATCH, "skel_subgoal_focus#9953",
                      "claim 表达式不可解析: %s" % (e,))
        if c1 in by or c2 in by or c1 == c2:
            _fail(Err.PRECOND_FAIL, "skel_subgoal_focus#9953",
                  "子节点 id 冲突 (c1=%d, c2=%d; 已在栈: %r) —— 编号由调用方声明, 不许撞号"
                  % (c1, c2, sorted(by)))
        nodes.append({"gid": c1, "kind": "claim", "status": "open", "parent": focus})
        nodes.append({"gid": c2, "kind": "claim", "status": "open", "parent": focus})
        by[focus]["status"] = "closed"
        focus, depth = c1, depth + 1
        focused_goal = (sm.negation_goal_text(cg.claims[0][0], cg.claims[0][1], cg.var)
                        if cg.form == "NC" else
                        sm.universal_goal_text(cg.claims[0][0], cg.claims[0][1], cg.var))
        note = ("split %s合取 ⇒ 压 2 条 claim (g%d, g%d), 焦点 g%d"
                % ("否定性 " if cg.form == "NC" else "", c1, c2, c1))
    elif act == "apply":
        if gid not in by:
            _fail(Err.PRECOND_FAIL, "skel_subgoal_focus#9953", "apply 的节点 %d 不在栈里" % gid)
        if by[gid]["status"] != "open":
            _fail(Err.PRECOND_FAIL, "skel_subgoal_focus#9953", "apply 的节点 %d 已闭合" % gid)
        focus = gid
        focused_goal = tk
        note = "焦点切到 g%d" % gid
    elif act == "rewrite":
        if focus not in by:
            _fail(Err.PRECOND_FAIL, "skel_subgoal_focus#9953", "rewrite 的焦点 %d 不在栈里" % focus)
        if by[focus]["status"] != "open":
            _fail(Err.PRECOND_FAIL, "skel_subgoal_focus#9953", "rewrite 的焦点 %d 已闭合" % focus)
        by[focus]["status"] = "closed"
        par = int(by[focus]["parent"])
        focus = par if (par in by and by[par]["status"] == "open") else -1
        focused_goal = by[focus]["goal"] if focus in by and "goal" in by[focus] else tk
        note = "闭合 g%d, 焦点交回父 g%d" % (gid, focus)
    else:
        focused_goal = tk
        note = "noop (状态不变)"
    nopen = sm.n_open_of(nodes)
    stk_after = sm.stack_str(nodes, focus, depth)
    cert.payload["after"] = {"nodes": [dict(x) for x in sorted(nodes, key=lambda x: x["gid"])],
                             "focus": focus, "depth": depth, "n_open": nopen}
    return {"value": {"act": act, "gid": gid, "vid": vid, "focus": focus, "depth": depth,
                      "n_open": nopen, "stk_after": stk_after, "focused_goal": focused_goal,
                      "nodes": len(nodes), "rule": "R2"},
            "resid": [0.0], "cert": cert, "cost": 1.5,
            "notes": "%s | n_open %d→%d" % (note, nopen_before, nopen)}


def _verify_sf(args, value):
    """Independent verifier: replay act from the before string and compare field by field with the
    recorded after (own implementation; never looks at invoke's intermediates)."""
    bad = []
    c1, c2 = int(args["c1"]), int(args["c2"])
    try:
        st = sm.parse_stack_str(args["stk"])
    except sm.ExprError as e:
        return {"ok": False, "resid": [1.0], "detail": "栈串重解析失败: %s" % (e,)}
    try:
        got = sm.parse_stack_str(value["stk_after"])
    except sm.ExprError as e:
        return {"ok": False, "resid": [1.0], "detail": "stk_after 不可解析: %s" % (e,)}
    nodes = {x["gid"]: dict(x) for x in st["nodes"]}
    act, gid = args["act"], int(args["gid"])
    if act == "split":
        nodes[c1] = {"gid": c1, "kind": "claim", "status": "open", "parent": int(st["focus"])}
        nodes[c2] = {"gid": c2, "kind": "claim", "status": "open", "parent": int(st["focus"])}
        nodes[int(st["focus"])]["status"] = "closed"
        want_focus, want_depth = c1, int(st["depth"]) + 1
    elif act == "apply":
        want_focus, want_depth = gid, int(st["depth"])
    elif act == "rewrite":
        nodes[int(st["focus"])]["status"] = "closed"
        par = int(nodes[int(st["focus"])]["parent"])
        want_focus = par if (par in nodes and nodes[par]["status"] == "open") else -1
        want_depth = int(st["depth"])
    else:
        want_focus, want_depth = int(st["focus"]), int(st["depth"])
    want_open = sum(1 for x in nodes.values() if x["status"] == "open")
    if int(value["focus"]) != want_focus:
        bad.append("focus 不符 (独立 %d vs 录得 %r)" % (want_focus, value["focus"]))
    if int(value["depth"]) != want_depth:
        bad.append("depth 不符")
    if int(value["n_open"]) != want_open:
        bad.append("n_open 不符 (独立 %d vs 录得 %r)" % (want_open, value["n_open"]))
    if int(value["nodes"]) != len(nodes):
        bad.append("节点数不符")
    gain = {x["gid"]: x for x in got["nodes"]}
    if set(gain) != set(nodes):
        bad.append("after 节点集合不符")
    else:
        for g in nodes:
            if (gain[g]["kind"], gain[g]["status"], gain[g]["parent"]) != \
               (nodes[g]["kind"], nodes[g]["status"], nodes[g]["parent"]):
                bad.append("节点 g%d 状态不符" % g)
    return {"ok": not bad, "resid": [float(len(bad))],
            "detail": "栈机独立重放一致 (act=%s, n_open=%d)" % (act, want_open)
                      if not bad else "; ".join(bad)}


_reg(
    name="skel_subgoal_focus", band=BAND_SKEL, sub=3,
    in_schema=_SF_IN, out_schema=_SF_OUT, invoke=_invoke_sf, verify=_verify_sf,
    precond=_sf_precond, cert_kind="stack_machine",
    cost=("units", 1.5), errors=("SCHEMA_MISMATCH", "PRECOND_FAIL", "POSTCOND_FAIL"),
    doc=("Proof skeleton: subgoal stack. Three actions -- `split` (decompose the focused "
         "conjunction goal into two claims, push them and focus the first), `apply` (focus a given "
         "node), `rewrite` (close the focused node and hand the focus back to its parent) -- plus "
         "`noop`. Stack state travels in and out as a canonical string (pure function, no global "
         "state); certificate stack_machine is re-verified by replaying it on an **independent "
         "stack machine**."),
    stmt="子目标栈: 压栈 / 切换聚焦 / 交回父节点", thm="T-V32-SF",
    selfchecks=(
        {"id": "sf_split_root", "args": {"act": "split", "gid": 0, "vid": 0,
                                         "stk": "g0|claim|open|-1@focus=0@depth=0", "c1": 1, "c2": 2,
                                         "tk": "for every integer n, (2 divides n*(n+1)) and "
                                               "(3 divides n*(n+1)*(n+2))"},
         "check": lambda r: (r.value["focus"] == 1 and r.value["n_open"] == 2
                             and r.value["depth"] == 1 and r.value["nodes"] == 3,
                             "focus=%d n_open=%d nodes=%d" % (r.value["focus"], r.value["n_open"],
                                                              r.value["nodes"])),
         "expect": "split: 父节点闭合, 两个 claim 打开, 焦点 g1"},
        {"id": "sf_apply", "args": {"act": "apply", "gid": 2, "vid": 0,
                                    "stk": "g0|claim|closed|-1;g1|claim|open|0;g2|claim|open|0"
                                           "@focus=1@depth=1", "c1": 0, "c2": 0,
                                    "tk": "for every integer n, 3 divides n*(n+1)*(n+2)"},
         "check": lambda r: (r.value["focus"] == 2 and r.value["n_open"] == 2,
                             "focus g%d, n_open=%d" % (r.value["focus"], r.value["n_open"])),
         "expect": "apply: 焦点切到 g2, 开启数不变"},
        {"id": "sf_rewrite", "args": {"act": "rewrite", "gid": 1, "vid": 0,
                                      "stk": "g0|root|closed|-1;g1|leaf|open|0"
                                             "@focus=1@depth=1", "c1": 0, "c2": 0,
                                      "tk": "closed"},
         "check": lambda r: (r.value["focus"] == -1 and r.value["n_open"] == 0,
                             "focus=%d, n_open=%d (栈已清空)" % (r.value["focus"],
                                                                r.value["n_open"])),
         "expect": "rewrite: 闭合最后一个开节点 ⇒ 焦点 -1, n_open=0"},
        {"id": "sf_split_neg_conj", "args": {
            "act": "split", "gid": 0, "vid": 0, "c1": 1, "c2": 2,
            "stk": "g0|c|O|-1@f=0@d=0",
            "tk": "for every integer n, (no 4 divides n^2 - 2) and (no 4 divides n^2 - 3)"},
         "check": lambda r: (r.value["focus"] == 1 and r.value["n_open"] == 2
                             and r.value["focused_goal"].startswith("there is no integer n"),
                             "焦点 g%d, 首 claim = %r" % (r.value["focus"],
                                                         r.value["focused_goal"][:40])),
         "expect": "否定合取 (NC): 压两条不存在性 claim, 首条 goal 是否定形态"},
        {"id": "sf_split_not_conj", "args": {"act": "split", "gid": 0, "vid": 0,
                                             "stk": "g0|claim|open|-1@focus=0@depth=0", "c1": 1, "c2": 2,
                                             "tk": "for every integer n, 2 divides n*(n+1)"},
         "expect_err": "PRECOND_FAIL", "expect": "split 作用于非合取目标 ⇒ 拒"},
        {"id": "sf_apply_closed", "args": {"act": "apply", "gid": 1, "vid": 0,
                                           "stk": "g0|root|closed|-1;g1|leaf|closed|0"
                                                  "@focus=-1@depth=1", "c1": 0, "c2": 0, "tk": "x"},
         "expect_err": "PRECOND_FAIL", "expect": "apply 到已闭合节点 ⇒ 拒"},
        {"id": "sf_missing_node", "args": {"act": "apply", "gid": 7, "vid": 0,
                                           "stk": "g0|claim|open|-1@focus=0@depth=0", "c1": 0, "c2": 0, "tk": "x"},
         "expect_err": "PRECOND_FAIL", "expect": "apply 到不存在的节点 ⇒ 拒"},
        {"id": "sf_bad_stk", "args": {"act": "apply", "gid": 0, "vid": 0, "stk": "garbage",
                                      "c1": 0, "c2": 0, "tk": "x"},
         "expect_err": "SCHEMA_MISMATCH", "expect": "栈串形态非法 ⇒ schema 门拒"},
        {"id": "sf_determinism", "args": {"act": "apply", "gid": 1, "vid": 0,
                                          "stk": "g0|claim|closed|-1;g1|claim|open|0"
                                                 "@focus=0@depth=1", "c1": 0, "c2": 0,
                                          "tk": "for every integer n, 2 divides n*(n+1)"},
         "check": lambda r: (r.value["stk_after"]
                             == "g0|c|C|-1;g1|c|O|0@f=1@d=1",
                             "规范串 %r (确定性)" % (r.value["stk_after"],)),
         "expect": "同输入同输出: 状态串逐字节可复现"},
    ),
)


# ===========================================================================
# 9954  skel_leaf_close -- leaf closure (complete-residue decision)
# ===========================================================================

_LC_IN = _sdict({
    "goal": {"kind": "str"}, "m": _int(2, 12), "vid": _int(0, 4), "gid": _int(0, 16),
}, ["goal", "m", "vid", "gid"])
_LC_OUT = _sdict({
    "gid": {"kind": "int"}, "m": {"kind": "int"}, "vid": {"kind": "int"},
    "poly": {"kind": "str"}, "factored": {"kind": "str"}, "residues": _ilist(),
    "coeff_gcd": {"kind": "int"}, "closed": {"kind": "bool"}, "ncase": {"kind": "int"},
    "rule": _RULE,
}, ["gid", "m", "vid", "poly", "factored", "residues", "coeff_gcd", "closed", "ncase", "rule"])


def _invoke_lc(args, ctl):
    goal, m, vid, gid = args["goal"], int(args["m"]), int(args["vid"]), int(args["gid"])
    try:
        g = sm.parse_goal(goal)
    except sm.ExprError as e:
        _fail(Err.SCHEMA_MISMATCH, "skel_leaf_close#9954", "目标文本不可解析: %s" % (e,))
    if g.form != "U":
        _fail(Err.PRECOND_FAIL, "skel_leaf_close#9954",
              "形态 %r 不是全称目标 (叶闭合只作用于 U 形态)" % (g.form,))
    if g.m != m:
        _fail(Err.PRECOND_FAIL, "skel_leaf_close#9954", "目标模数 %d ≠ args.m=%d" % (g.m, m))
    res = sm.residues(g.coeffs, m)
    cg = sm.poly_gcd(g.coeffs)
    closed = all(x == 0 for x in res)
    cert = Certificate(kind="leaf_residue_complete",
                       payload={"goal": goal, "m": m, "residues": res, "coeff_gcd": cg,
                                "closed": bool(closed)},
                       issued_by="skel_leaf_close")
    return {"value": {"gid": gid, "m": m, "vid": vid,
                      "poly": sm.poly_str(g.coeffs, g.var), "factored": g.factored,
                      "residues": res, "coeff_gcd": cg, "closed": bool(closed), "ncase": 1,
                      "rule": "R1"},
            "resid": [0.0 if closed else 1.0], "cert": cert, "cost": 1.0,
            "notes": ("叶闭合: %s 的完备剩余类 %r %s"
                      % (goal, res, "全零 ⇒ 闭合" if closed else "含非零 ⇒ **未闭合** (如实回报)"))}


def _verify_lc(args, value):
    bad = []
    try:
        g = sm.parse_goal(args["goal"])
    except sm.ExprError as e:
        return {"ok": False, "resid": [1.0], "detail": "目标重解析失败: %s" % (e,)}
    m = int(args["m"])
    want = sm.residues(g.coeffs, m)
    if [int(x) for x in value["residues"]] != want:
        bad.append("剩余类表不符 (独立 %r vs 录得 %r)" % (want, value["residues"]))
    if int(value["coeff_gcd"]) != sm.poly_gcd(g.coeffs):
        bad.append("系数 gcd 不符")
    if bool(value["closed"]) != all(x == 0 for x in want):
        bad.append("closed 与独立完备判定不符")
    if value["poly"] != sm.poly_str(g.coeffs, g.var):
        bad.append("展开式不符")
    return {"ok": not bad,
            "resid": [float(len(bad)) if bad else (0.0 if value["closed"] else 1.0)],
            "detail": ("剩余类/gcd/展开式独立复算一致 (closed=%r)" % (value["closed"],)
                       if not bad else "; ".join(bad))
                      + ("" if value["closed"] else " | **未闭合 (残差 1.0, 如实)**")}


_reg(
    name="skel_leaf_close", band=BAND_SKEL, sub=4,
    in_schema=_LC_IN, out_schema=_LC_OUT, invoke=_invoke_lc, verify=_verify_lc,
    cert_kind="leaf_residue_complete",
    cost=("units", 1.0), errors=("SCHEMA_MISMATCH", "PRECOND_FAIL", "POSTCOND_FAIL"),
    doc=("Proof skeleton: leaf-goal closure. For the substituted universal goal U(m, p(k)), close "
         "it with the **complete-residue decision** (∀v. m|p(v) ⟺ ∀r∈{0..m-1}. m|p(r), "
         "**necessary and sufficient**); the coefficient gcd serves as a separate **sufficient** "
         "witness (gcd divides m => always divisible; the converse fails: k^2-k has gcd=1 yet is "
         "always even). closed=false is a **legitimate return value**: when it cannot be decided, "
         "report non-closure (residual 1.0), and the independent checker likewise re-verifies locally "
         "that it really did not close."),
    stmt="叶目标闭合 (完备剩余类判定)", thm="T-V32-LC",
    selfchecks=(
        {"id": "lc_close_2k", "args": {"goal": "for every integer k, 2 divides (0 + 2*k)*((0 + 2*k) + 1)",
                                       "m": 2, "vid": 2, "gid": 1},
         "check": lambda r: (r.value["closed"] and r.value["residues"] == [0, 0]
                             and r.value["poly"] == "4*k^2 + 2*k",
                             "residues=%r poly=%r" % (r.value["residues"], r.value["poly"])),
         "expect": "2 | (2k)(2k+1) 完备判定闭合"},
        {"id": "lc_close_3k", "args": {"goal": "for every integer k, 3 divides (1 + 3*k)*((1 + 3*k) + 1)*((1 + 3*k) + 2)",
                                       "m": 3, "vid": 2, "gid": 1},
         "check": lambda r: (r.value["closed"] and r.value["coeff_gcd"] % 3 == 0,
                             "closed=%r gcd=%d" % (r.value["closed"], r.value["coeff_gcd"])),
         "expect": "3 | (3k+1)(3k+2)(3k+3) 闭合 (系数 gcd 也整除)"},
        {"id": "lc_not_closed_honest", "args": {"goal": "for every integer k, 2 divides k^2 + k + 1",
                                                "m": 2, "vid": 2, "gid": 1},
         "check": lambda r: ((not r.value["closed"]) and r.value["residues"] == [1, 1]
                             and r.resid_scalar > 0,
                             "如实回报未闭合: residues=%r resid=%.1f"
                             % (r.value["residues"], r.resid_scalar)),
         "expect": "恒奇多项式**不假装闭合** (residues 全 1, 残差 1.0)"},
        {"id": "lc_neg_reject", "args": {"goal": "there is no integer n with 4 dividing n^2 - 2",
                                         "m": 4, "vid": 0, "gid": 0},
         "expect_err": "PRECOND_FAIL", "expect": "否定形态不许走叶闭合 (规则表 R3)"},
        {"id": "lc_mod_mismatch", "args": {"goal": "for every integer k, 2 divides (0 + 2*k)*((0 + 2*k) + 1)",
                                           "m": 3, "vid": 2, "gid": 1},
         "expect_err": "PRECOND_FAIL", "expect": "args.m 与目标模数不符 ⇒ 拒"},
        {"id": "lc_bad_expr", "args": {"goal": "for every integer k, 2 divides k*z", "m": 2,
                                       "vid": 2, "gid": 1},
         "expect_err": "SCHEMA_MISMATCH", "expect": "表达式含未登记变量 ⇒ 拒绝解析"},
        {"id": "lc_determinism", "args": {"goal": "for every integer k, 2 divides k^2 - k",
                                          "m": 2, "vid": 2, "gid": 1},
         "check": lambda r: (r.value["poly"] == "k^2 - 1*k" and r.value["closed"]
                             and r.value["coeff_gcd"] == 1,
                             "规范展开 %r (系数 gcd=%d: **充分**条件不适用, 完备判定才闭合)"
                             % (r.value["poly"], r.value["coeff_gcd"])),
         "expect": "同输入同输出: k^2-k → 2*k + k^2"},
    ),
)


# ===========================================================================
# 9955  skel_contradiction_close -- contradiction closure (negated goal)
# ===========================================================================

_CC_IN = _sdict({
    "goal": {"kind": "str"}, "m": _int(2, 12), "vid": _int(0, 4), "gid": _int(0, 16),
}, ["goal", "m", "vid", "gid"])
_CC_OUT = _sdict({
    "gid": {"kind": "int"}, "m": {"kind": "int"}, "vid": {"kind": "int"},
    "assumption": {"kind": "str"}, "values": _ilist(), "closed": {"kind": "bool"},
    "ncase": {"kind": "int"}, "rule": _RULE,
}, ["gid", "m", "vid", "assumption", "values", "closed", "ncase", "rule"])


def _invoke_cc(args, ctl):
    goal, m, vid, gid = args["goal"], int(args["m"]), int(args["vid"]), int(args["gid"])
    try:
        g = sm.parse_goal(goal)
    except sm.ExprError as e:
        _fail(Err.SCHEMA_MISMATCH, "skel_contradiction_close#9955", "目标文本不可解析: %s" % (e,))
    if g.form != "N":
        _fail(Err.PRECOND_FAIL, "skel_contradiction_close#9955",
              "形态 %r 不是否定目标 (R3 只作用于 N 形态)" % (g.form,))
    if g.m != m:
        _fail(Err.PRECOND_FAIL, "skel_contradiction_close#9955",
              "目标模数 %d ≠ args.m=%d" % (g.m, m))
    vals = [sm.poly_eval(g.coeffs, r) % m for r in range(m)]
    no_sol = all(x != 0 for x in vals)
    assumption = "assume there is an integer %s with %d dividing %s" % (g.var, m, g.factored)
    cert = Certificate(kind="contradiction_witness",
                       payload={"goal": goal, "m": m, "values": vals,
                                "closed": bool(no_sol)},
                       issued_by="skel_contradiction_close")
    return {"value": {"gid": gid, "m": m, "vid": vid, "assumption": assumption,
                      "values": vals, "closed": bool(no_sol), "ncase": 1, "rule": "R3"},
            "resid": [0.0 if no_sol else 1.0], "cert": cert, "cost": 1.5,
            "notes": ("反证: 假设存在解 ⇒ 枚举 r=0..%d 得 %r %s"
                      % (m - 1, vals, "无一为 0 ⇒ 矛盾闭合" if no_sol else "有 0 ⇒ **未闭合** (如实回报)"))}


def _verify_cc(args, value):
    bad = []
    try:
        g = sm.parse_goal(args["goal"])
    except sm.ExprError as e:
        return {"ok": False, "resid": [1.0], "detail": "目标重解析失败: %s" % (e,)}
    m = int(args["m"])
    want = [sm.poly_eval(g.coeffs, r) % m for r in range(m)]
    if [int(x) for x in value["values"]] != want:
        bad.append("取值表不符 (独立 %r vs 录得 %r)" % (want, value["values"]))
    if bool(value["closed"]) != all(x != 0 for x in want):
        bad.append("closed 与独立判定不符")
    if not value["assumption"].startswith("assume there is an integer"):
        bad.append("假设串形态不符")
    return {"ok": not bad,
            "resid": [float(len(bad)) if bad else (0.0 if value["closed"] else 1.0)],
            "detail": ("取值表/闭合判定独立复算一致 (closed=%r)" % (value["closed"],)
                       if not bad else "; ".join(bad))
                      + ("" if value["closed"] else " | **反证不成立 (残差 1.0, 如实)**")}


_reg(
    name="skel_contradiction_close", band=BAND_SKEL, sub=5,
    in_schema=_CC_IN, out_schema=_CC_OUT, invoke=_invoke_cc, verify=_verify_cc,
    cert_kind="contradiction_witness",
    cost=("units", 1.5), errors=("SCHEMA_MISMATCH", "PRECOND_FAIL", "POSTCOND_FAIL"),
    doc=("Proof skeleton: contradiction closure. For the negated goal "
         "`there is no integer v with m dividing f(v)`: assume a solution exists and enumerate "
         "r=0..m-1 to show m ∤ f(r) => no integer solution (complete: f(v) mod m depends only on "
         "v mod m). closed=false is a **legitimate return value** (reported as measured when the goal "
         "actually has a solution)."),
    stmt="否定目标: 反证闭合", thm="T-V32-CC",
    selfchecks=(
        {"id": "cc_closed_mod4", "args": {"goal": "there is no integer n with 4 dividing n^2 - 2",
                                          "m": 4, "vid": 0, "gid": 0},
         "check": lambda r: (r.value["closed"] and r.value["values"] == [2, 3, 2, 3],
                             "取值 %r" % (r.value["values"],)),
         "expect": "n^2-2 在 mod 4 下取值 {2,3} ⇒ 无解, 闭合"},
        {"id": "cc_closed_mod7", "args": {"goal": "there is no integer n with 7 dividing n^3 - 2",
                                          "m": 7, "vid": 0, "gid": 0},
         "check": lambda r: (r.value["closed"] and 0 not in r.value["values"],
                             "取值 %r" % (r.value["values"],)),
         "expect": "立方 mod 7 不含 2 ⇒ 无解"},
        {"id": "cc_not_closed_honest", "args": {"goal": "there is no integer n with 2 dividing n^2 - n",
                                                "m": 2, "vid": 0, "gid": 0},
         "check": lambda r: ((not r.value["closed"]) and 0 in r.value["values"]
                             and r.resid_scalar > 0,
                             "如实回报: 取值 %r 含 0 ⇒ 有解, 反证不成立"
                             % (r.value["values"],)),
         "expect": "目标实际有解时不假装闭合 (残差 1.0)"},
        {"id": "cc_univ_reject", "args": {"goal": "for every integer n, 2 divides n*(n+1)",
                                          "m": 2, "vid": 0, "gid": 0},
         "expect_err": "PRECOND_FAIL", "expect": "全称形态不许走反证 (规则表 R4/R5)"},
        {"id": "cc_mod_mismatch", "args": {"goal": "there is no integer n with 4 dividing n^2 - 2",
                                           "m": 3, "vid": 0, "gid": 0},
         "expect_err": "PRECOND_FAIL", "expect": "args.m 与目标模数不符 ⇒ 拒"},
        {"id": "cc_determinism", "args": {"goal": "there is no integer t with 8 dividing t^4 - 2",
                                          "m": 8, "vid": 3, "gid": 0},
         "check": lambda r: (r.value["values"] == [6, 7, 6, 7, 6, 7, 6, 7],
                             "取值 %r (确定性)" % (r.value["values"],)),
         "expect": "同输入同输出: t^4-2 mod 8 交替 6/7"},
    ),
)


# ===========================================================================
# 9956  skel_qed_check -- terminal audit
# ===========================================================================

_QD_IN = _sdict({
    "stk": {"kind": "str"}, "ledger": {"kind": "str"}, "gid": _int(0, 16),
}, ["stk", "ledger", "gid"])
_QD_OUT = _sdict({
    "gid": {"kind": "int"}, "qed": {"kind": "bool"}, "n_nodes": {"kind": "int"},
    "n_closed": {"kind": "int"}, "n_open": {"kind": "int"}, "discharged": _slist(20),
    "reason": {"kind": "str"}, "rule": _RULE,
}, ["gid", "qed", "n_nodes", "n_closed", "n_open", "discharged", "reason", "rule"])


def _invoke_qd(args, ctl):
    gid = int(args["gid"])
    try:
        st = sm.parse_stack_str(args["stk"])
        led = sm.parse_ledger(args["ledger"])
    except sm.ExprError as e:
        _fail(Err.SCHEMA_MISMATCH, "skel_qed_check#9956", "栈串/台账不可解析: %s" % (e,))
    st_nodes = {(x["gid"], x["kind"], x["status"]) for x in st["nodes"]}
    led_nodes = {(g, k, s) for (g, k, s) in led}
    if not st["nodes"]:
        _fail(Err.PRECOND_FAIL, "skel_qed_check#9956",
              "空栈不许做 QED 审计 (没有根节点 ⇒ 不是证明)")
    mism = sorted(st_nodes ^ led_nodes)
    if mism:
        _fail(Err.PRECOND_FAIL, "skel_qed_check#9956",
              "栈串与台账不一致 (对称差 %r) —— 审计不许在自相矛盾的状态上出结论" % (mism[:5],))
    n_open = sm.n_open_of(st["nodes"])
    n_nodes = len(st["nodes"])
    n_closed = n_nodes - n_open
    kinds = {}
    for x in st["nodes"]:
        kinds[x["kind"]] = kinds.get(x["kind"], 0) + 1
    pairs_ok = kinds.get("ind-base", 0) == kinds.get("ind-step", 0)
    has_root = any(x["kind"] in ("root", "case", "claim", "neg") for x in st["nodes"])
    qed = bool(n_open == 0 and pairs_ok and has_root and n_nodes > 0)
    reason_bits = []
    if n_open:
        reason_bits.append("%d 个节点未闭合" % n_open)
    if not pairs_ok:
        reason_bits.append("归纳基例/步进配对不齐 (base=%d, step=%d)"
                           % (kinds.get("ind-base", 0), kinds.get("ind-step", 0)))
    if not has_root:
        reason_bits.append("无根节点")
    cert = Certificate(kind="skeleton_qed",
                       payload={"nodes": [{"gid": x["gid"], "kind": x["kind"],
                                           "status": x["status"]} for x in st["nodes"]],
                                "n_open": n_open, "qed": qed},
                       issued_by="skel_qed_check")
    return {"value": {"gid": gid, "qed": qed, "n_nodes": n_nodes, "n_closed": n_closed,
                      "n_open": n_open,
                      "discharged": ["g%d:%s" % (x["gid"], x["kind"]) for x in st["nodes"]],
                      "reason": "QED: 全部子目标闭合且归纳配对成立" if qed
                                else ("未 QED: " + "; ".join(reason_bits)), "rule": "R6"},
            "resid": [0.0 if qed else 1.0], "cert": cert, "cost": 1.0,
            "notes": "审计 %d 节点 (闭合 %d, 开 %d) ⇒ qed=%r" % (n_nodes, n_closed, n_open, qed)}


def _verify_qd(args, value):
    bad = []
    try:
        st = sm.parse_stack_str(args["stk"])
    except sm.ExprError as e:
        return {"ok": False, "resid": [1.0], "detail": "栈串重解析失败: %s" % (e,)}
    n_open = sm.n_open_of(st["nodes"])
    if int(value["n_open"]) != n_open:
        bad.append("n_open 不符 (独立 %d vs 录得 %r)" % (n_open, value["n_open"]))
    if int(value["n_nodes"]) != len(st["nodes"]):
        bad.append("节点数不符")
    if int(value["n_closed"]) != len(st["nodes"]) - n_open:
        bad.append("闭合数不符")
    kinds = {}
    for x in st["nodes"]:
        kinds[x["kind"]] = kinds.get(x["kind"], 0) + 1
    want_qed = bool(n_open == 0 and kinds.get("ind-base", 0) == kinds.get("ind-step", 0)
                    and len(st["nodes"]) > 0)
    if bool(value["qed"]) != want_qed:
        bad.append("qed 与独立审计不符 (独立 %r)" % (want_qed,))
    return {"ok": not bad,
            "resid": [float(len(bad)) if bad else (0.0 if value["qed"] else 1.0)],
            "detail": ("节点数/开启数/审计谓词独立复算一致 (qed=%r)" % (value["qed"],)
                       if not bad else "; ".join(bad))
                      + ("" if value["qed"] else " | **未 QED (残差 1.0, 如实)**")}


_reg(
    name="skel_qed_check", band=BAND_SKEL, sub=6,
    in_schema=_QD_IN, out_schema=_QD_OUT, invoke=_invoke_qd, verify=_verify_qd,
    cert_kind="skeleton_qed",
    cost=("units", 1.0), errors=("SCHEMA_MISMATCH", "PRECOND_FAIL", "POSTCOND_FAIL"),
    doc=("Proof skeleton: terminal audit (the counterpart of the END sentinel in the skeleton "
         "language). Checks: stack string and ledger agree, no open node, induction nodes have "
         "base/step paired, a root node exists => QED. qed=false is a legitimate return value (an "
         "unclosed proof is reported as measured)."),
    stmt="骨架终局审计 (全闭 + 归纳配对) ⇒ QED", thm="T-V32-QED",
    selfchecks=(
        {"id": "qd_case_qed", "args": {"stk": "g0|case|closed|-1;g1|leaf|closed|0;g2|leaf|closed|0"
                                              "@focus=-1@depth=1",
                                       "ledger": "g0:case:closed;g1:leaf:closed;g2:leaf:closed",
                                       "gid": 0},
         "check": lambda r: (r.value["qed"] and r.value["n_closed"] == 3
                             and r.value["n_open"] == 0,
                             "qed=%r 闭合 %d/%d" % (r.value["qed"], r.value["n_closed"],
                                                    r.value["n_nodes"])),
         "expect": "分情况证明收拢 ⇒ QED"},
        {"id": "qd_ind_pairs", "args": {"stk": "g0|case|closed|-1;g1|ind-base|closed|0;"
                                              "g2|ind-step|closed|0@focus=-1@depth=1",
                                       "ledger": "g0:case:closed;g1:ind-base:closed;"
                                                 "g2:ind-step:closed",
                                       "gid": 0},
         "check": lambda r: (r.value["qed"] and r.value["n_nodes"] == 3,
                             "归纳基例/步进配对 ⇒ QED (%d 节点)" % r.value["n_nodes"]),
         "expect": "归纳证明收拢 ⇒ QED"},
        {"id": "qd_open_honest", "args": {"stk": "g0|claim|closed|-1;g1|claim|open|0;g2|claim|open|0"
                                                "@focus=1@depth=1",
                                          "ledger": "g0:claim:closed;g1:claim:open;g2:claim:open",
                                          "gid": 0},
         "check": lambda r: ((not r.value["qed"]) and r.value["n_open"] == 2
                             and r.resid_scalar > 0,
                             "如实回报未 QED: %s" % (r.value["reason"],)),
         "expect": "还有 2 个子目标开 ⇒ qed=false (不假装证完)"},
        {"id": "qd_ind_pair_missing", "args": {"stk": "g0|case|closed|-1;g1|ind-base|closed|0"
                                                     "@focus=-1@depth=1",
                                               "ledger": "g0:case:closed;g1:ind-base:closed",
                                               "gid": 0},
         "check": lambda r: (not r.value["qed"] and "配对" in r.value["reason"],
                             "如实回报: %s" % (r.value["reason"],)),
         "expect": "只有基例没有步进 ⇒ 不许 QED"},
        {"id": "qd_ledger_mismatch", "args": {"stk": "g0|case|closed|-1;g1|leaf|closed|0"
                                                    "@focus=-1@depth=1",
                                              "ledger": "g0:case:closed;g9:leaf:closed", "gid": 0},
         "expect_err": "PRECOND_FAIL", "expect": "栈串与台账不一致 ⇒ 拒 (不许在矛盾状态上出结论)"},
        {"id": "qd_bad_stk", "args": {"stk": "nonsense", "ledger": "", "gid": 0},
         "expect_err": "SCHEMA_MISMATCH", "expect": "栈串非法 ⇒ schema 门拒"},
        {"id": "qd_empty_reject", "args": {"stk": "@focus=-1@depth=0", "ledger": "", "gid": 0},
         "expect_err": "PRECOND_FAIL", "expect": "空栈不许 QED (没有根节点)"},
        {"id": "qd_determinism", "args": {"stk": "g0|neg|closed|-1;g1|neg|closed|0@focus=-1@depth=0",
                                          "ledger": "g0:neg:closed;g1:neg:closed", "gid": 0},
         "check": lambda r: (r.value["discharged"] == ["g0:neg", "g1:neg"],
                             "discharged=%r (确定性序)" % (r.value["discharged"],)),
         "expect": "同输入同输出: discharged 逐项可复现"},
    ),
)

#: Post-registration re-check: the 6 operators of the band and their certificate kinds are all in
#: the skeleton list (audit_static_v32 has separate item-by-item criteria)
_SKEL_NAMES = tuple(n for n in V32REGISTRY.names() if n.startswith("skel_"))
if len(_SKEL_NAMES) != 6:
    from svokernel.errors import Err as _E, SvoError as _S
    raise _S(_E.SCHEMA_MISMATCH, ctx="v32:ops_skel",
             detail="骨架算子数 %d ≠ 6" % len(_SKEL_NAMES))
for _n in _SKEL_NAMES:
    _op = V32REGISTRY._ops[_n]
    if _op.cert_kind not in SKEL_CERT_KINDS:
        from svokernel.errors import Err as _E, SvoError as _S
        raise _S(_E.SCHEMA_MISMATCH, ctx="v32:ops_skel",
                 detail="%s 的证书 %r 不在骨架清单" % (_n, _op.cert_kind))
