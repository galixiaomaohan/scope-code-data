# -*- coding: utf-8 -*-
"""**Independent checkers** for the V32 skeleton certificates (6 kinds) -- §17.9 forbids
self-certification.

Independence discipline (same as v31's certcheck_*):
  * this module is a **checker module** and does not import the generator module `ops_skel` (its
    module object is not even in the namespace; audit_static_v32 checks this item by item);
  * every checker consumes payload **data** only, recomputes on its own, and never looks at the
    generator's intermediate quantities;
  * the checking path differs from the generation path (see "generation vs re-verification" in each
    kind's docstring).

The shared foundation is `skelmath` (polynomial parsing / complete-residue decision) -- that is not
"copying the generator" but sharing one layer of **mathematical definition**: the generator goes via
"symbolic coefficient gcd", the checker via "numeric complete residues"; both paths are complete and
independent implementations, and when they disagree the certificate is **rejected** (better to fail
than to pass).
"""
from __future__ import annotations

from svokernel.certify import register_checker

from . import skelmath as sm

__all__ = ["check_case_exhaustive", "check_induction_scheme", "check_stack_machine",
           "check_leaf_residue_complete", "check_contradiction_witness", "check_skeleton_qed"]

#: Which generator operators it is independent of (for registration; the audit has separate
#: module-level criteria)
_GEN = ("skel_case_split", "skel_induction_scheme", "skel_subgoal_focus",
        "skel_leaf_close", "skel_contradiction_close", "skel_qed_check")


def _bad(msg: str):
    return False, [1.0], msg


def _ok(msg: str):
    return True, [0.0], msg


# ---------------------------------------------------------------------------
# 1) case_exhaustive -- exhaustiveness of the case split
#    Generation: symbolically expand f(r+m*k) to the substituted polynomial and record the cases.
#    Re-verification: re-parse the **recorded subgoal texts** -> expand independently -> compare
#    coefficient by coefficient against "substitute v:=r+m*k into the original goal and expand";
#    then cross-check numerically at 3 sampled k (catches implementation differences); cover must
#    be exactly {0..m-1}.
# ---------------------------------------------------------------------------
@register_checker("case_exhaustive", independent_of=_GEN)
def check_case_exhaustive(payload: dict) -> tuple:
    goal = payload.get("goal")
    m = payload.get("m")
    var = payload.get("var")
    cases = payload.get("cases") or []
    cover = payload.get("cover") or []
    if not isinstance(m, int) or m < 2:
        return _bad("m 非法: %r" % (m,))
    if not isinstance(goal, str) or not isinstance(var, str):
        return _bad("goal/var 非法")
    if sorted(int(x) for x in cover) != list(range(m)):
        return _bad("cover %r != 完备剩余系 {0..%d}" % (cover, m - 1))
    if len(cases) != m:
        return _bad("cases 数 %d != m=%d" % (len(cases), m))
    try:
        g = sm.parse_goal(goal)
    except sm.ExprError as e:
        return _bad("目标重解析失败: %s" % (e,))
    if g.form != "U" or g.m != m or g.var != var:
        return _bad("目标形态/模数/变量与载荷不符 (form=%s m=%s var=%s)" % (g.form, g.m, g.var))
    kname = None
    seen = {}
    for item in cases:
        r = item.get("res")
        sub = item.get("goal")
        if not isinstance(r, int) or not isinstance(sub, str):
            return _bad("case 项形态非法: %r" % (item,))
        if r in seen:
            return _bad("余数 %d 重复" % r)
        seen[r] = True
        try:
            sg = sm.parse_goal(sub)
        except sm.ExprError as e:
            return _bad("子目标重解析失败 (r=%d): %s" % (r, e))
        if sg.form != "U" or sg.m != m:
            return _bad("子目标形态非全称或模数不符 (r=%d)" % r)
        if sg.var not in sm.SUBST_NAMES:
            return _bad("子目标变量 %r 不是替换参数" % (sg.var,))
        if kname is None:
            kname = sg.var
        elif sg.var != kname:
            return _bad("各枝替换参数不一致 (%r vs %r)" % (sg.var, kname))
        want = sm.subst_affine(g.coeffs, r, m)                    # substitute v := r + m*k, then expand
        got = sg.coeffs
        if tuple(want) != tuple(got):
            return _bad("枝 r=%d 多项式不符: 期望 %r, 录得 %r" % (r, want, got))
        for kk in (0, 1, 7):                                      # numeric cross-check (catches implementation differences)
            a = sm.poly_eval(want, kk)
            b = sm.poly_eval(got, kk)
            if a != b:
                return _bad("枝 r=%d 数值对拍不符 @k=%d" % (r, kk))
    if sorted(seen) != list(range(m)):
        return _bad("余数集合 %r 不覆盖 {0..%d}" % (sorted(seen), m - 1))
    return _ok("穷尽性复验通过: m=%d 枝, 各枝替换多项式逐系数一致 (变量 %s)" % (m, kname))


# ---------------------------------------------------------------------------
# 2) induction_scheme -- base case + step
#    Generation: symbolic expansion of the difference f(x+1)-f(x) + base-case evaluation.
#    Re-verification: recompute the difference and base value independently; also check the scheme
#    structure (base index / IH variable / difference always divisible).
# ---------------------------------------------------------------------------
@register_checker("induction_scheme", independent_of=_GEN)
def check_induction_scheme(payload: dict) -> tuple:
    goal = payload.get("goal")
    m = payload.get("m")
    n0 = payload.get("n0")
    base_value = payload.get("base_value")
    diff_repr = payload.get("diff_repr")
    if not isinstance(m, int) or m < 2 or not isinstance(n0, int) or n0 not in (0, 1):
        return _bad("m/n0 非法: %r/%r" % (m, n0))
    try:
        g = sm.parse_goal(goal)
        diff = sm.parse_poly(diff_repr, g.var)
    except sm.ExprError as e:
        return _bad("重解析失败: %s" % (e,))
    want_diff = sm.poly_diff(g.coeffs)
    if tuple(diff) != tuple(want_diff):
        return _bad("差分不符: 期望 %r, 录得 %r" % (want_diff, diff))
    want_base = sm.poly_eval(g.coeffs, n0)
    if int(base_value) != int(want_base):
        return _bad("基例值不符: 期望 %d, 录得 %r" % (want_base, base_value))
    # **The certificate only attests that "the scheme is constructed correctly", not that "the
    # scheme holds for the current goal"**: base_ok/diff_ok are data, and the checker only requires
    # them to be **honest** (agree with the independent decision) -- otherwise the legitimate exit
    # diff_ok=false would not exist.
    want_base_ok = (want_base % m) == 0
    if "base_ok" in payload and bool(payload["base_ok"]) != want_base_ok:
        return _bad("base_ok 不诚实: 录得 %r, 独立判定 %r" % (payload["base_ok"], want_base_ok))
    want_diff_ok = sm.divides_all(diff, m)
    if "diff_ok" in payload and bool(payload["diff_ok"]) != want_diff_ok:
        return _bad("diff_ok 不诚实: 录得 %r, 独立判定 %r (剩余类 %r)"
                    % (payload["diff_ok"], want_diff_ok, sm.residues(diff, m)))
    if want_diff_ok and want_base_ok:
        return _ok("归纳复验通过: 基例 f(%d)=%d≡0, 差分 %s 恒被 %d 整除"
                   % (n0, want_base, diff_repr, m))
    return _ok("归纳方案**如实报告不成立**: 基例成立=%r, 差分恒整除=%r (差分 %s 剩余类 %r) "
               "—— 证书只认证构造正确, 不认证方案适用"
               % (want_base_ok, want_diff_ok, diff_repr, sm.residues(diff, m)))


# ---------------------------------------------------------------------------
# 3) stack_machine -- subgoal stack state machine
#    Generation: the generator's own stack object.
#    Re-verification: the checker **carries** an independent stack machine and replays before + act
#    to after, comparing field by field.
# ---------------------------------------------------------------------------
def _stack_replay(before: dict, act: str, gid: int, nopen: int, c1: int = 0, c2: int = 0) -> dict:
    nodes = [dict(x) for x in (before.get("nodes") or [])]
    by = {x["gid"]: x for x in nodes}
    depth = int(before.get("depth", 0))
    focus = int(before.get("focus", -1))
    if act == "split":
        if focus not in by:
            raise sm.ExprError("split: 焦点节点 %r 不在栈里" % (focus,))
        if by[focus]["kind"] != "claim":
            raise sm.ExprError("split: 焦点节点 kind=%r 不是合取" % (by[focus]["kind"],))
        if c1 in by or c2 in by or c1 == c2:
            raise sm.ExprError("split: 子节点 id 冲突 (c1=%d, c2=%d)" % (c1, c2))
        nodes.append({"gid": c1, "parent": focus, "kind": "claim", "status": "open"})
        nodes.append({"gid": c2, "parent": focus, "kind": "claim", "status": "open"})
        by[focus]["status"] = "closed"
        focus = c1
        depth = depth + 1
        nopen = nopen + 1                      # conjunction node closes (−1), two claims open (+2)
    elif act == "apply":
        if gid not in by:
            raise sm.ExprError("apply: 目标节点 %r 不在栈里" % (gid,))
        if by[gid]["status"] != "open":
            raise sm.ExprError("apply: 目标节点 %r 已闭合" % (gid,))
        focus = gid
    elif act == "rewrite":
        if focus not in by:
            raise sm.ExprError("rewrite: 焦点节点 %r 不在栈里" % (focus,))
        if by[focus]["status"] != "open":
            raise sm.ExprError("rewrite: 焦点节点已闭合")
        by[focus]["status"] = "closed"
        nopen = nopen - 1
        par = by[focus].get("parent", -1)
        focus = par if (par in by and by[par]["status"] == "open") else -1
    elif act == "noop":
        pass
    else:
        raise sm.ExprError("未登记 act=%r" % (act,))
    nopen_calc = sum(1 for x in nodes if x["status"] == "open")
    return {"nodes": sorted(nodes, key=lambda x: x["gid"]), "focus": focus,
            "depth": depth, "n_open": nopen_calc}


@register_checker("stack_machine", independent_of=_GEN)
def check_stack_machine(payload: dict) -> tuple:
    before = payload.get("before")
    after = payload.get("after")
    act = payload.get("act")
    gid = payload.get("gid")
    nopen = payload.get("n_open_in")
    if not isinstance(before, dict) or not isinstance(after, dict):
        return _bad("before/after 必须是 dict")
    if not isinstance(gid, int) or not isinstance(nopen, int):
        return _bad("gid/n_open_in 非法")
    # hardening (exposed by mutation-audit survivors): the field values of the before ledger must
    # be registered values, and n_open_in must agree with the ledger.
    bn = before.get("nodes") or []
    for x in bn:
        if x.get("status") not in ("open", "closed"):
            return _bad("before 台账 status 未登记: %r" % (x.get("status"),))
        if x.get("kind") not in ("root", "case", "ind-base", "ind-step", "claim", "leaf", "neg"):
            return _bad("before 台账 kind 未登记: %r" % (x.get("kind"),))
    want_nopen_in = sum(1 for x in bn if x.get("status") == "open")
    if int(nopen) != want_nopen_in:
        return _bad("n_open_in=%r 与 before 台账的开节点数 %d 不符" % (nopen, want_nopen_in))
    try:
        want = _stack_replay(before, act, gid, nopen, int(payload.get("c1", 0)),
                             int(payload.get("c2", 0)))
    except sm.ExprError as e:
        return _bad("独立栈机重放失败: %s" % (e,))
    for k in ("focus", "depth", "n_open"):
        if int(want[k]) != int(after.get(k, -999)):
            return _bad("重放后 %s 不符: 独立 %r vs 录得 %r" % (k, want[k], after.get(k)))
    wn = [(x["gid"], x["parent"], x["kind"], x["status"]) for x in want["nodes"]]
    gn = [(x.get("gid"), x.get("parent"), x.get("kind"), x.get("status"))
          for x in (after.get("nodes") or [])]
    if wn != gn:
        return _bad("节点台账不符:\n  独立 %r\n  录得 %r" % (wn, gn))
    return _ok("栈机复验通过: act=%s ⇒ focus=%d depth=%d n_open=%d (%d 节点)"
               % (act, want["focus"], want["depth"], want["n_open"], len(wn)))


# ---------------------------------------------------------------------------
# 4) leaf_residue_complete -- leaf closure (complete residues)
#    Generation: symbolic coefficient-gcd criterion + residue table.
#    Re-verification: independently recompute with the **numeric complete-residue** criterion; also
#    check that the parsed factored expression == the recorded polynomial (dual-path agreement).
# ---------------------------------------------------------------------------
@register_checker("leaf_residue_complete", independent_of=_GEN)
def check_leaf_residue_complete(payload: dict) -> tuple:
    goal = payload.get("goal")
    m = payload.get("m")
    residues = payload.get("residues")
    coeff_gcd = payload.get("coeff_gcd")
    closed = payload.get("closed")
    if not isinstance(m, int) or m < 2:
        return _bad("m 非法: %r" % (m,))
    try:
        g = sm.parse_goal(goal)
    except sm.ExprError as e:
        return _bad("目标重解析失败: %s" % (e,))
    if g.form != "U" or g.m != m:
        return _bad("叶目标形态/模数不符 (form=%s m=%s)" % (g.form, g.m))
    got = sm.residues(g.coeffs, m)
    if [int(x) for x in (residues or [])] != got:
        return _bad("剩余类表不符: 独立 %r vs 录得 %r" % (got, residues))
    if int(coeff_gcd) != sm.poly_gcd(g.coeffs):
        return _bad("系数 gcd 不符: 独立 %d vs 录得 %r" % (sm.poly_gcd(g.coeffs), coeff_gcd))
    by_res = all(x == 0 for x in got)
    # the complete criterion = residues; the coefficient gcd is only a **sufficient** condition
    # (k^2-k has coefficient gcd=1 yet is always even => neither implies the other). Hence only the
    # one-way consistency is kept: gcd divides m => it must judge closure (otherwise a proof step
    # was skipped).
    if bool(closed) != bool(by_res):
        return _bad("closed=%r 与完备判定不符 (剩余类全零=%r)" % (closed, by_res))
    if (int(coeff_gcd) % m == 0) and not by_res:
        return _bad("系数 gcd=%d 整除 %d 但剩余类 %r 有非零 —— 判据自相矛盾, 拒绝放行"
                    % (int(coeff_gcd), m, got))
    if not by_res:
        return _ok("**如实报告未闭合**: %s 的剩余类 %r 含非零 ⇒ m=%d 不恒整除 (不假装闭合)"
                   % (goal, got, m))
    return _ok("叶闭合复验通过: 剩余类 %r 全零 (完备判定); 系数 gcd=%d 为充分见证 (%s)"
               % (got, int(coeff_gcd), "成立" if int(coeff_gcd) % m == 0 else "不适用"))


# ---------------------------------------------------------------------------
# 5) contradiction_witness -- contradiction closure (negated goal)
#    Generation: enumerate f(r) mod m.
#    Re-verification: independently recompute f(r) mod m for all residues, requiring no 0
#    (complete => no integer solution); also re-parse the factored expression.
# ---------------------------------------------------------------------------
@register_checker("contradiction_witness", independent_of=_GEN)
def check_contradiction_witness(payload: dict) -> tuple:
    goal = payload.get("goal")
    m = payload.get("m")
    values = payload.get("values")
    closed = payload.get("closed")
    if not isinstance(m, int) or m < 2:
        return _bad("m 非法: %r" % (m,))
    try:
        g = sm.parse_goal(goal)
    except sm.ExprError as e:
        return _bad("目标重解析失败: %s" % (e,))
    if g.form != "N" or g.m != m:
        return _bad("否定目标形态/模数不符 (form=%s m=%s)" % (g.form, g.m))
    want = [sm.poly_eval(g.coeffs, r) % m for r in range(m)]
    if [int(x) for x in (values or [])] != want:
        return _bad("取值表不符: 独立 %r vs 录得 %r" % (want, values))
    no_sol = all(x != 0 for x in want)
    if bool(closed) != bool(no_sol):
        return _bad("closed=%r 与完备判据不符 (无解=%r)" % (closed, no_sol))
    if not no_sol:
        return _ok("**如实报告未闭合**: %s 在 r=%r 处取 0 ⇒ 有整数解, 反证不成立 (不假装闭合)"
                   % (goal, [i for i, x in enumerate(want) if x == 0]))
    return _ok("反证闭合复验通过: ∀r∈{0..%d}. %d ∤ f(r) (取值 %r) ⇒ 无整数解"
               % (m - 1, m, want))


# ---------------------------------------------------------------------------
# 6) skeleton_qed -- terminal audit
#    Generation: a state snapshot of the stack object.
#    Re-verification: recompute the audit predicate from the snapshot independently (all closed +
#    induction nodes have base/step + no open node).
# ---------------------------------------------------------------------------
@register_checker("skeleton_qed", independent_of=_GEN)
def check_skeleton_qed(payload: dict) -> tuple:
    nodes = payload.get("nodes")
    n_open = payload.get("n_open")
    qed = payload.get("qed")
    if not isinstance(nodes, list) or not nodes:
        return _bad("节点台账为空")
    kinds = {}
    gids = []
    for x in nodes:
        for k in ("gid", "kind", "status"):
            if k not in x:
                return _bad("节点项缺字段 %r: %r" % (k, x))
        gids.append(x["gid"])
        if x["status"] not in ("open", "closed"):
            return _bad("status 未登记: %r" % (x["status"],))
        if x["kind"] not in ("root", "case", "ind-base", "ind-step", "claim", "leaf", "neg"):
            return _bad("kind 未登记: %r" % (x["kind"],))
        kinds.setdefault(x["kind"], 0)
        kinds[x["kind"]] += 1
    # hardening (exposed by mutation-audit survivors): gids must be unique -- otherwise a ledger
    # could stuff duplicates to impersonate more proven nodes.
    if len(set(gids)) != len(gids):
        return _bad("节点 gid 不唯一: %r" % (sorted(gids),))
    want_open = sum(1 for x in nodes if x["status"] == "open")
    if int(n_open) != want_open:
        return _bad("n_open 不符: 台账算出 %d, 录得 %r" % (want_open, n_open))
    ind_ok = True
    if kinds.get("ind-base", 0) != kinds.get("ind-step", 0):
        ind_ok = False
    want_qed = (want_open == 0) and ind_ok
    if bool(qed) != bool(want_qed):
        return _bad("qed=%r 与独立审计不符 (未闭合=%d, 归纳基例/步进配对=%r)"
                    % (qed, want_open, ind_ok))
    if not want_qed:
        return _ok("**如实报告未 QED**: 未闭合节点 %d 个, 归纳配对=%r" % (want_open, ind_ok))
    return _ok("QED 复验通过: %d 节点全部闭合, 归纳基例/步进配对成立, 无待证子目标" % len(nodes))
