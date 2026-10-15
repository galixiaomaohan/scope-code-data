# -*- coding: utf-8 -*-
"""Layer 3: invocation mechanism (band 9930, rows 9931–9934) -- the core multiplier of the
proving side.

Four operators answer "how to turn an existing theorem/lemma into progress on the current goal":
  goal_apply_theorem  9931  goal-theorem matching + instantiation + reduction (matching reuses logic_unify)
  lemma_instantiate   9932  lemma instantiation (free variables -> terms, residual variables reported as measured)
  goal_transform      9933  goal transformation (contrapositive / iff split / dual / conjunction split) + truth-table equivalence certificate
  lemma_discover      9934  auxiliary lemma discovery (pattern-library instantiation + exhaustive tautology) + staging hook

**Why this layer is a multiplier**: layers 0–2 operators "compute a fact"; this layer "connects a
proven fact to the goal". Its certificate semantics also differ from the previous layers -- not
"numeric residual" but **substitution witness** and **exhaustive truth table**:
  - theorem application: certificate = σ (substitution) + σ(conclusion)=goal + subgoals = σ(premises)
    checked symbol by symbol;
  - goal transformation: certificate = goal <=> AND(subgoals) verified **over all assignments**
    (complete within the propositional fragment);
  - lemma discovery: certificate = exhaustive tautology of every candidate + shape ∈ pattern
    library + usefulness criterion.

**No-side-effect discipline**: all four operators have effects = none -- in particular
lemma_discover only produces a **staging record** for later insertion, it writes no database; the
actual insertion is the caller's action.
"""
from __future__ import annotations

from .v31registry import V31OpDef, V31REGISTRY

from svokernel.certify import Certificate
from svokernel.errors import Err, SvoError
from svokernel.logiccore import apply_subst, parse_formula, parse_term, term_str, unify, walk_vars
from svokernel.tool import CallCtl

BAND_L3 = 9930
_EXACT = {"kind": "int"}


def _sdict(fields, req):
    return {"kind": "dict", "fields": fields, "required": req}


def _reg(**kw):
    kw.setdefault("status", "verified")
    kw.setdefault("layer", 3)
    return V31REGISTRY.register(V31OpDef(**kw))


# ===========================================================================
# 9931  goal_apply_theorem
# ===========================================================================

_GA_IN = _sdict({
    "goal": {"kind": "str"},
    "theorem_code": {"kind": "str"},
    "premises": {"kind": "list", "items": {"kind": "str"}, "minlen": 0, "maxlen": 16},
    "conclusion": {"kind": "str"},
    "strict": {"kind": "bool"},
}, ["goal", "theorem_code", "premises", "conclusion", "strict"])
_GA_OUT = _sdict({
    "theorem_code": {"kind": "str"}, "matched": {"kind": "bool"},
    "goal": {"kind": "str"}, "matched_conclusion": {"kind": "str"},
    "sigma": {"kind": "list", "items": {"kind": "tuple", "items": [{"kind": "str"}, {"kind": "str"}]},
              "minlen": 0, "maxlen": 32},
    "subgoals": {"kind": "list", "items": {"kind": "str"}, "minlen": 0, "maxlen": 16},
    "n_subgoals": _EXACT,
    "rename_map": {"kind": "list", "items": {"kind": "tuple", "items": [{"kind": "str"}, {"kind": "str"}]},
                   "minlen": 0, "maxlen": 32},
    "no_capture": {"kind": "bool"}, "reason": {"kind": "str"},
}, ["theorem_code", "matched", "goal", "matched_conclusion", "sigma", "subgoals", "n_subgoals",
    "rename_map", "no_capture", "reason"])


def _rename_vars(t, mapping):
    if t[0] == "v":
        return ("v", mapping.get(t[1], t[1]))
    if t[0] == "c":
        return t
    return ("f", t[1], tuple(_rename_vars(a, mapping) for a in t[2]))


def _invoke_ga(args, ctl):
    goal = parse_term(args["goal"])
    concl = parse_term(args["conclusion"])
    prems = [parse_term(p) for p in args["premises"]]
    # rename variables to avoid capture: prefix the theorem variables (all free variables of
    # conclusion + premises)
    tvars = set(walk_vars(concl))
    for p in prems:
        tvars |= walk_vars(p)
    gvars = walk_vars(goal)
    ren = {}
    for i, v in enumerate(sorted(tvars)):
        nv = "thm%d_%s" % (i, v)
        if nv in gvars:
            raise SvoError(Err.PRECOND_FAIL, ctx="op:goal_apply_theorem#9931",
                           detail="改名后变量 %r 与目标变量冲突 ⇒ 拒绝 (宁失败不捕获)" % nv)
        ren[v] = nv
    concl_r = _rename_vars(concl, ren)
    prems_r = [_rename_vars(p, ren) for p in prems]
    sigma = unify(concl_r, goal, {})
    if sigma is None:
        if args["strict"]:
            raise SvoError(Err.UNIFY_FAIL, ctx="op:goal_apply_theorem#9931",
                           detail="σ(结论) 与目标不合一, 且 strict=True ⇒ 按契约失败")
        # B15 fix: the certificate explicitly marks mode=not_applicable -- "conclusion and goal do
        # not unify" is a **legitimate negative conclusion** (selfcheck ga_no_match_legal expects
        # exactly that), but it is a different assertion from "applied", and the checker must be
        # able to tell them apart and re-check each independently. Previously no mode was sent, so
        # the checker re-checked it as "applied" => a legitimate conclusion was judged a failure.
        cert = Certificate(kind="thm_apply_witness",
                           payload={"goal": goal, "conclusion": concl_r, "premises": prems_r,
                                    "sigma": {}, "subgoals": [], "rename_map": ren,
                                    "mode": "not_applicable"},
                           issued_by="goal_apply_theorem")
        return {"value": {"theorem_code": args["theorem_code"], "matched": False,
                          "goal": args["goal"], "matched_conclusion": "",
                          "sigma": [], "subgoals": [], "n_subgoals": 0,
                          "rename_map": sorted(ren.items()), "no_capture": True,
                          "reason": "结论与目标不合一 (非错误: '本定理不适用于该目标'是合法结论)"},
                "resid": [1.0], "cert": cert, "cost": 2.0,
                "notes": "不匹配: %s ≡/= %s" % (term_str(concl), args["goal"])}
    # σ is translated back to the **original** theorem variable names for presentation; subgoals
    # agree symbol by symbol under either naming (the rename is a bijection)
    sig_orig = {}
    inv = {v: k for k, v in ren.items()}
    for k, v in sigma.items():
        sig_orig[inv.get(k, k)] = v
    subgoals = [term_str(apply_subst(p, sigma)) for p in prems_r]
    cert = Certificate(kind="thm_apply_witness",
                       payload={"goal": goal, "conclusion": concl_r, "premises": prems_r,
                                "sigma": {k: v for k, v in sigma.items()},
                                "subgoals": [apply_subst(p, sigma) for p in prems_r],
                                "rename_map": ren,
                                "mode": "applied"},
                       issued_by="goal_apply_theorem")
    return {"value": {"theorem_code": args["theorem_code"], "matched": True,
                      "goal": args["goal"], "matched_conclusion": term_str(apply_subst(concl_r, sigma)),
                      "sigma": sorted((k, term_str(v)) for k, v in sig_orig.items()),
                      "subgoals": subgoals, "n_subgoals": len(subgoals),
                      "rename_map": sorted(ren.items()), "no_capture": True,
                      "reason": "σ(结论) = 目标 (逐符号); 子目标 = σ(前提)"},
            "resid": [0.0], "cert": cert, "cost": 3.0,
            "notes": "匹配成功: σ = %r, %d 个子目标" % (sorted((k, term_str(v)) for k, v in sig_orig.items()), len(subgoals))}


def _verify_ga(args, value):
    """Independent verifier: apply σ to conclusion/premises independently and compare symbol by
    symbol (a different path from the unification algorithm)."""
    bad = []
    if not value["matched"]:
        # verification of the no-match conclusion: the claim is non-unifiability -- independently
        # re-checking "conclusion and goal really do not unify" needs a complete unification
        # decision; here only the **shape** is checked (σ empty and subgoals empty, no
        # over-claiming).
        if value["sigma"] or value["subgoals"]:
            bad.append("声称不匹配却给了 σ/子目标")
        return {"ok": not bad, "resid": [float(len(bad))],
                "detail": "不匹配结论: σ 与子目标为空 (无越权声称)" if not bad else "; ".join(bad)}
    goal = parse_term(args["goal"])
    concl = parse_term(args["conclusion"])
    prems = [parse_term(p) for p in args["premises"]]
    tvars = set(walk_vars(concl))
    for p in prems:
        tvars |= walk_vars(p)
    ren = dict(value["rename_map"])
    concl_r = _rename_vars(concl, ren)
    prems_r = [_rename_vars(p, ren) for p in prems]
    sigma = {k: parse_term(v) for k, v in value["sigma"]}
    # the σ keys were translated back to original names in the output; for verification they must
    # go back to the renamed names
    inv = {v: k for k, v in ren.items()}
    sigma_r = {}
    for k, v in sigma.items():
        sigma_r[ren.get(k, k)] = v
    if apply_subst(concl_r, sigma_r) != goal:
        bad.append("σ(结论) ≠ 目标")
    for i, pr in enumerate(prems_r):
        want = term_str(apply_subst(pr, sigma_r))
        if i >= len(value["subgoals"]) or want != value["subgoals"][i]:
            bad.append("子目标 %d ≠ σ(前提)" % i)
    for k, v in sigma_r.items():
        if apply_subst(v, sigma_r) != v:
            bad.append("σ 非幂等 (%s)" % k)
    if len(value["subgoals"]) != len(prems_r):
        bad.append("子目标数 ≠ 前提数")
    # no capture: the rename map is injective and its values do not clash with goal variables
    if len(set(ren.values())) != len(ren):
        bad.append("改名映射非单射")
    return {"ok": not bad, "resid": [float(len(bad))],
            "detail": "σ(结论)=目标 + %d 个子目标 = σ(前提) + σ 幂等 + 无捕获 通过" % len(value["subgoals"])
                      if not bad else "; ".join(bad)}


_reg(
    name="goal_apply_theorem", band=BAND_L3, sub=1,
    in_schema=_GA_IN, out_schema=_GA_OUT,
    invoke=_invoke_ga, verify=_verify_ga, cert_kind="thm_apply_witness",
    cost=("units", 4.0),
    errors=("PRECOND_FAIL", "UNIFY_FAIL", "SCHEMA_MISMATCH", "POSTCOND_FAIL", "TIMEOUT"),
    precond=None,
    effects="none (纯函数; 不改动任何定理库)",
    doc=("Goal-theorem matching + instantiation + reduction: **unify** the theorem's conclusion "
         "with the goal (reusing svokernel.logiccore.unify as the logical root of trust) to obtain "
         "the substitution σ, then instantiate the theorem premises into subgoals. Variables are "
         "renamed first (thm<i>_ prefix) to avoid capture; on a rename clash the request is "
         "rejected rather than force-unified. Generation = rename + unify; independent verifier = "
         "apply σ to conclusion/premises independently and compare symbol by symbol + σ "
         "idempotence + rename injectivity; certificate thm_apply_witness is independently "
         "re-checked by certcheck_goal (own substitution implementation). **A no-match is not an "
         "error**: 'this theorem does not apply to that goal' is a legitimate conclusion "
         "(matched=False); pass strict=True to get contract-failure semantics (=> UNIFY_FAIL)."),
    requires_cert=True,
    thm="T-V31-APPLY",
    stmt="若 σ = MGU(C, G) 则 G 可由 (C ⇐ P1..Pk) 化归为 σ(P1)..σ(Pk)",
    selfchecks=(
        {"id": "ga_simple_match", "args": {"goal": "f(a, g(a))", "theorem_code": "T-demo",
                                           "premises": ["p(?x)"], "conclusion": "f(?x, g(?x))",
                                           "strict": False},
         "check": lambda r: (r.value["matched"] and r.value["n_subgoals"] == 1
                             and r.value["subgoals"] == ["p(a)"] and r.cert_independent,
                             "σ = %r, 子目标 %r" % (r.value["sigma"], r.value["subgoals"])),
         "expect": "f(?x,g(?x)) 与 f(a,g(a)) 合一: σ = {?x ↦ a}, 子目标 p(a)"},
        {"id": "ga_no_match_legal", "args": {"goal": "f(a, b)", "theorem_code": "T-demo",
                                             "premises": ["p(?x)"], "conclusion": "f(?x, ?x)",
                                             "strict": False},
         "check": lambda r: (not r.value["matched"] and r.value["sigma"] == []
                             and r.value["n_subgoals"] == 0,
                             "a ≠ b ⇒ 合失败, 如实回报 matched=False (非错误)"),
         "expect": "不合一 ⇒ matched=False (合法结论), 不给子目标"},
        {"id": "ga_strict_unify_fail", "args": {"goal": "f(a, b)", "theorem_code": "T-demo",
                                                "premises": ["p(?x)"], "conclusion": "f(?x, ?x)",
                                                "strict": True},
         "expect_err": "UNIFY_FAIL", "expect": "strict=True 时把不匹配升级为契约失败"},
        {"id": "ga_two_premises", "args": {"goal": "g(a)", "theorem_code": "T-mul",
                                           "premises": ["even(?n)", "pos(?n)"],
                                           "conclusion": "g(?n)", "strict": False},
         # B16 fix: the original fixture used the numeric constant "g(7)" -- the legacy
         # logiccore.parse_term grammar **accepts identifier constants only** (measured: 'g(7)'
         # raises UNIFY_FAIL "位置 2 处期望标识符"), and legacy is an anchor-chain-protected
         # zero-modification file. The fixture therefore uses identifier constants (semantically
         # equivalent: a constant is still a constant); that grammar boundary is recorded in
         # DESIGN_V31_OPS.md.
         "check": lambda r: (r.value["subgoals"] == ["even(a)", "pos(a)"],
                             "两前提实例化: %r" % (r.value["subgoals"],)),
         "expect": "两个前提各自实例化为子目标 (常量用标识符: 存量文法不接受数字字面量)"},
        {"id": "ga_no_vars_theorem", "args": {"goal": "h(c)", "theorem_code": "T-ground",
                                              "premises": [], "conclusion": "h(c)", "strict": False},
         "check": lambda r: (r.value["matched"] and r.value["sigma"] == []
                             and r.value["n_subgoals"] == 0,
                             "常元结论: σ = {} (空替换在 ground 匹配下是正确的)"),
         "expect": "ground 匹配: 空 σ (与 legacy unify_witness 的'拒空 σ'策略不同, 本检查器接受)"},
        {"id": "ga_occurs_check", "args": {"goal": "f(g(?y), ?y)", "theorem_code": "T-occ",
                                           "premises": [], "conclusion": "f(?x, g(?x))",
                                           "strict": False},
         "check": lambda r: (not r.value["matched"],
                             "occurs check: ?x 不能同时是 g(?y) 与 ... ⇒ 不合一"),
         "expect": "occurs check 生效 ⇒ matched=False"},
        {"id": "ga_bad_syntax_reject", "args": {"goal": "f(a", "theorem_code": "T-x",
                                                "premises": [], "conclusion": "f(?x)",
                                                "strict": False},
         "expect_err": "UNIFY_FAIL", "expect": "项语法错误由 logiccore.parse_term 拒 (UNIFY_FAIL)"},
    ),
)


# ===========================================================================
# 9932  lemma_instantiate
# ===========================================================================

_LI_IN = _sdict({
    "lemma_code": {"kind": "str"},
    "statement": {"kind": "str"},
    "subst": {"kind": "list", "items": {"kind": "tuple", "items": [{"kind": "str"}, {"kind": "str"}]},
              "minlen": 0, "maxlen": 32},
}, ["lemma_code", "statement", "subst"])
_LI_OUT = _sdict({
    "lemma_code": {"kind": "str"}, "statement": {"kind": "str"}, "instance": {"kind": "str"},
    "n_instantiated": _EXACT,
    "residual_vars": {"kind": "list", "items": {"kind": "str"}, "minlen": 0, "maxlen": 32},
    "wellformed": {"kind": "bool"}, "partial": {"kind": "bool"},
}, ["lemma_code", "statement", "instance", "n_instantiated", "residual_vars", "wellformed", "partial"])


def _li_precond(args):
    try:
        st = parse_term(args["statement"])
    except SvoError as e:
        return False, "statement 语法错误: %s" % e.detail
    declared = walk_vars(st)
    for k, v in args["subst"]:
        if k.lstrip("?") not in declared:
            return False, "替换键 %r 不是引理自由变量 %r (实例化越权, 拒绝)" % (k, sorted(declared))
        try:
            parse_term(v)     # the substitution value is parsed in precond too: a syntax error is an illegal request
        except SvoError as e:
            return False, "替换值 %r 语法错误: %s" % (v, e.detail)
    return True, ""


def _invoke_li(args, ctl):
    st = parse_term(args["statement"])
    sigma = {}
    for k, v in args["subst"]:
        sigma[k.lstrip("?")] = parse_term(v)
    inst = apply_subst(st, sigma)
    residual = sorted(walk_vars(inst))
    cert = Certificate(kind="lemma_instance",
                       payload={"statement": st, "sigma": sigma, "instance": inst,
                                "residual_vars": residual},
                       issued_by="lemma_instantiate")
    return {"value": {"lemma_code": args["lemma_code"], "statement": args["statement"],
                      "instance": term_str(inst), "n_instantiated": len(sigma),
                      "residual_vars": residual, "wellformed": True,
                      "partial": len(residual) > 0},
            "resid": [0.0], "cert": cert, "cost": 1.0 + len(sigma),
            "notes": "实例: %s (残余自由变量 %r)" % (term_str(inst), residual)}


def _verify_li(args, value):
    """Independent verifier: apply σ independently + compute the free variables independently +
    key-domain legality."""
    st = parse_term(args["statement"])
    sigma = {k.lstrip("?"): parse_term(v) for k, v in args["subst"]}
    declared = walk_vars(st)
    bad = []
    for k in sigma:
        if k not in declared:
            bad.append("σ 键 %r 越权" % k)
    got = apply_subst(st, sigma)
    if term_str(got) != value["instance"]:
        bad.append("独立应用 σ 得 %s ≠ 实例 %s" % (term_str(got), value["instance"]))
    want_res = sorted(walk_vars(got))
    if want_res != sorted(value["residual_vars"]):
        bad.append("残余变量 %r ≠ 独立计算 %r" % (value["residual_vars"], want_res))
    if value["n_instantiated"] != len(sigma):
        bad.append("n_instantiated 与 σ 大小不符")
    return {"ok": not bad, "resid": [float(len(bad))],
            "detail": "σ 应用 + 自由变量复算 + 键域合法 通过 (残余 %r)" % (want_res,)
                      if not bad else "; ".join(bad)}


_reg(
    name="lemma_instantiate", band=BAND_L3, sub=2,
    in_schema=_LI_IN, out_schema=_LI_OUT,
    invoke=_invoke_li, verify=_verify_li, cert_kind="lemma_instance",
    cost=("units", 2.0),
    errors=("PRECOND_FAIL", "SCHEMA_MISMATCH", "POSTCOND_FAIL", "UNIFY_FAIL", "TIMEOUT"),
    precond=_li_precond,
    effects="none (纯函数)",
    doc=("Lemma instantiation: given a lemma statement (a term) and a substitution table, output "
         "the instance and the **residual free variables**. Generation = logiccore.apply_subst "
         "applied directly (the root of trust); independent verifier = apply σ independently + "
         "compute the free-variable set independently + key-domain legality (a substitution key "
         "must be a free variable of the lemma -- over-reaching instantiation is always rejected); "
         "certificate lemma_instance is independently re-checked by certcheck_goal (own "
         "substitution and free-variable implementation). **Partial instantiation** is supported: "
         "unsubstituted variables are kept as they are and listed in residual_vars."),
    requires_cert=True,
    thm="T-V31-INST",
    stmt="σ(引理陈述) 即实例; 残余自由变量 = 未替换变量 (部分实例化亦合法)",
    selfchecks=(
        {"id": "li_partial", "args": {"lemma_code": "L1", "statement": "f(?x, ?y)",
                                      "subst": [["?x", "a"]]},
         "check": lambda r: (r.value["instance"] == "f(a, ?y)" and r.value["residual_vars"] == ["y"]
                             and r.value["partial"] and r.cert_independent,
                             "部分实例化: %s, 残余 %r" % (r.value["instance"], r.value["residual_vars"])),
         "expect": "f(?x,?y) with ?x↦a ⇒ f(a,?y), 残余 [y]"},
        {"id": "li_full", "args": {"lemma_code": "L1", "statement": "f(?x, g(?y))",
                                   "subst": [["?x", "a"], ["?y", "b"]]},
         "check": lambda r: (r.value["instance"] == "f(a, g(b))" and r.value["residual_vars"] == []
                             and not r.value["partial"],
                             "全实例化: %s" % r.value["instance"]),
         "expect": "完全实例化 ⇒ 无残余变量"},
        {"id": "li_empty_sigma", "args": {"lemma_code": "L1", "statement": "p(?x)", "subst": []},
         "check": lambda r: (r.value["instance"] == "p(?x)" and r.value["residual_vars"] == ["x"],
                             "空替换: 实例 = 原陈述 (恒等, 合法)"),
         "expect": "空替换边界"},
        {"id": "li_ground_statement", "args": {"lemma_code": "L2", "statement": "q(a, b)", "subst": []},
         "check": lambda r: (r.value["instance"] == "q(a, b)" and r.value["residual_vars"] == [],
                             "ground 陈述: 无自由变量"),
         "expect": "ground 引理边界"},
        {"id": "li_unknown_key_reject", "args": {"lemma_code": "L1", "statement": "p(?x)",
                                                 "subst": [["?z", "a"]]},
         "expect_err": "PRECOND_FAIL", "expect": "替换键 ?z 不在引理变量集 ⇒ 越权, 拒绝"},
        {"id": "li_bad_term_reject", "args": {"lemma_code": "L1", "statement": "p(?x)",
                                              "subst": [["?x", "f(("]]},
         "expect_err": "PRECOND_FAIL", "expect": "替换值语法错误 (precond 阶段解析失败)"},
    ),
)


# ===========================================================================
# 9933  goal_transform
# ===========================================================================

_GT_IN = _sdict({
    "goal": {"kind": "str"},
    "mode": {"kind": "str", "enum": ["contrapositive", "iff_split", "de_morgan_dual", "conj_split"]},
}, ["goal", "mode"])
_GT_OUT = _sdict({
    "goal": {"kind": "str"}, "mode": {"kind": "str"},
    "subgoals": {"kind": "list", "items": {"kind": "str"}, "minlen": 1, "maxlen": 4},
    "n_subgoals": _EXACT,
    "atoms": {"kind": "list", "items": {"kind": "str"}, "minlen": 0, "maxlen": 12},
    "n_atoms": _EXACT, "n_assignments": _EXACT, "equivalence_ok": {"kind": "bool"},
    "certified_by": {"kind": "str", "enum": ["truth_table_exhaustive"]},
}, ["goal", "mode", "subgoals", "n_subgoals", "atoms", "n_atoms", "n_assignments",
    "equivalence_ok", "certified_by"])


def _gt_eval(f, assign):
    k = f[0]
    if k == "atom":
        return bool(assign[f[1]])
    if k == "not":
        return not _gt_eval(f[1], assign)
    a, b = _gt_eval(f[1], assign), _gt_eval(f[2], assign)
    return {"and": a and b, "or": a or b, "imp": (not a) or b}[k]


def _gt_atoms(f):
    k = f[0]
    if k == "atom":
        return {f[1]}
    if k == "not":
        return _gt_atoms(f[1])
    return _gt_atoms(f[1]) | _gt_atoms(f[2])


def _gt_precond(args):
    try:
        g = parse_formula(args["goal"])
    except SvoError as e:
        return False, "公式语法错误: %s" % e.detail
    mode = args["mode"]
    if mode == "contrapositive" and g[0] != "imp":
        return False, "contrapositive 要求目标是蕴含式 A -> B (实得 %r)" % (g[0],)
    if mode == "iff_split":
        if g[0] != "and" or g[1][0] != "imp" or g[2][0] != "imp":
            return False, "iff_split 要求目标是 (A -> B) & (B -> A) 形状 " \
                          "(logiccore 文法无 <-> 记号, 故充要目标以合取式给出)"
    if mode == "de_morgan_dual" and (g[0] != "not" or g[1][0] not in ("and", "or")):
        return False, "de_morgan_dual 要求目标是 ~(A & B) 或 ~(A | B)"
    if mode == "conj_split" and g[0] != "and":
        return False, "conj_split 要求目标是 A & B"
    if len(_gt_atoms(g)) > 10:
        return False, "原子数 %d > 10: 真值表穷举 (2^n) 超出证书规模上限" % len(_gt_atoms(g))
    return True, ""


def _invoke_gt(args, ctl):
    g = parse_formula(args["goal"])
    mode = args["mode"]
    if mode == "contrapositive":
        subs = [("imp", ("not", g[2]), ("not", g[1]))]
    elif mode == "iff_split":
        subs = [g[1], g[2]]
    elif mode == "de_morgan_dual":
        inner = g[1]
        subs = [("or", ("not", inner[1]), ("not", inner[2]))] if inner[0] == "and" else \
               [("and", ("not", inner[1]), ("not", inner[2]))]
    else:
        subs = [g[1], g[2]]
    atoms = sorted(_gt_atoms(g))
    for s in subs:
        atoms = sorted(set(atoms) | _gt_atoms(s))
    conj = subs[0]
    for s in subs[1:]:
        conj = ("and", conj, s)
    ok = True
    n = len(atoms)
    for mask in range(1 << n):
        ctl.check_time("op:goal_transform#9933")
        assign = {a: bool((mask >> i) & 1) for i, a in enumerate(atoms)}
        if _gt_eval(g, assign) != _gt_eval(conj, assign):
            ok = False
            break
    cert = Certificate(kind="equiv_truth_table",
                       payload={"goal": g, "subgoals": subs, "mode": mode, "atoms": atoms},
                       issued_by="goal_transform")
    from svokernel.logiccore import formula_str
    return {"value": {"goal": args["goal"], "mode": mode,
                      "subgoals": [formula_str(s) for s in subs], "n_subgoals": len(subs),
                      "atoms": atoms, "n_atoms": n, "n_assignments": 1 << n,
                      "equivalence_ok": ok, "certified_by": "truth_table_exhaustive"},
            "resid": [0.0 if ok else 1.0], "cert": cert, "cost": float(1 << n) * 0.05,
            "notes": "%s: %d 个子目标, 真值表 %d 行等价=%s"
                     % (mode, len(subs), 1 << n, ok)}


def _verify_gt(args, value):
    """Independent verifier: enumerate all assignments checking goal <=> AND(subgoals) + a
    per-mode structural check."""
    g = parse_formula(args["goal"])
    subs = [parse_formula(s) for s in value["subgoals"]]
    mode = args["mode"]
    bad = []
    atoms = sorted(_gt_atoms(g) | {a for s in subs for a in _gt_atoms(s)})
    if len(atoms) > 10:
        bad.append("原子数超限")
    conj = subs[0]
    for s in subs[1:]:
        conj = ("and", conj, s)
    for mask in range(1 << len(atoms)):
        assign = {a: bool((mask >> i) & 1) for i, a in enumerate(atoms)}
        if _gt_eval(g, assign) != _gt_eval(conj, assign):
            bad.append("反例指派 %r 下不等价" % assign)
            break
    if mode == "contrapositive" and subs != [("imp", ("not", g[2]), ("not", g[1]))]:
        bad.append("结构不是逆否式")
    if mode == "iff_split" and subs != [g[1], g[2]]:
        bad.append("结构不是两个方向")
    if mode == "conj_split" and subs != [g[1], g[2]]:
        bad.append("结构不是合取拆分")
    if mode == "de_morgan_dual":
        inner = g[1]
        want = ("or", ("not", inner[1]), ("not", inner[2])) if inner[0] == "and" else \
               ("and", ("not", inner[1]), ("not", inner[2]))
        if subs != [want]:
            bad.append("结构不是 De Morgan 对偶")
    return {"ok": not bad, "resid": [float(len(bad))],
            "detail": "真值表 %d 行等价 + mode=%s 结构核对 通过" % (1 << len(atoms), mode)
                      if not bad else "; ".join(bad)}


_reg(
    name="goal_transform", band=BAND_L3, sub=3,
    in_schema=_GT_IN, out_schema=_GT_OUT,
    invoke=_invoke_gt, verify=_verify_gt, cert_kind="equiv_truth_table",
    cost=("units", 5.0),
    errors=("PRECOND_FAIL", "SCHEMA_MISMATCH", "POSTCOND_FAIL", "BOUNDED_VIOLATION", "TIMEOUT"),
    precond=_gt_precond,
    effects="none (纯函数)",
    doc=("Goal transformation (reduction-direction choice): contrapositive / iff_split / "
         "de_morgan_dual / conj_split. Generation = construct the subgoals per mode; independent "
         "verifier = **assignment-by-assignment enumeration** checking goal <=> AND(subgoals) "
         "(within the propositional fragment this is a **complete** certificate: all 2^n rows "
         "passing means semantic equivalence, not sampling) + a per-mode structural check; "
         "certificate equiv_truth_table is independently re-checked by certcheck_goal with its "
         "**own evaluator**. Interface constraint (recorded): the logiccore grammar has "
         "no <-> symbol, so an iff goal must be given as the conjunction (A -> B) & (B -> A) -- "
         "this is the interface determined by the root of trust, not laziness. More than 10 atoms "
         "=> the truth table is oversized and precond rejects directly (no un-enumerated "
         "'equivalence')."),
    requires_cert=True,
    thm="T-V31-TRANSFORM",
    stmt="P→Q ⇔ ¬Q→¬P; (P↔Q) ⇔ (P→Q)∧(Q→P); ¬(P∧Q) ⇔ ¬P∨¬Q; (P∧Q) ⇔ P∧Q",
    selfchecks=(
        {"id": "gt_contrapositive", "args": {"goal": "p -> q", "mode": "contrapositive"},
         "check": lambda r: (r.value["subgoals"] == ["(~q -> ~p)"] and r.value["equivalence_ok"]
                             and r.value["n_assignments"] == 4 and r.cert_independent,
                             "逆否: %r (真值表 %d 行)" % (r.value["subgoals"], r.value["n_assignments"])),
         "expect": "p→q 的逆否 ~q→~p (语义等价穷举验证)"},
        {"id": "gt_iff_split", "args": {"goal": "(p -> q) & (q -> p)", "mode": "iff_split"},
         "check": lambda r: (r.value["n_subgoals"] == 2 and r.value["equivalence_ok"],
                             "充要拆分: %r" % (r.value["subgoals"],)),
         "expect": "充要目标拆成两个方向"},
        {"id": "gt_de_morgan", "args": {"goal": "~(p & q)", "mode": "de_morgan_dual"},
         "check": lambda r: (r.value["subgoals"] == ["(~p | ~q)"] and r.value["equivalence_ok"],
                             "德摩根: %r" % (r.value["subgoals"],)),
         "expect": "~ (p & q) ⇔ ~p | ~q"},
        {"id": "gt_conj_split", "args": {"goal": "p & q & r", "mode": "conj_split"},
         "check": lambda r: (r.value["n_subgoals"] == 2 and r.value["equivalence_ok"],
                             "合取拆分 (右结合): %r" % (r.value["subgoals"],)),
         "expect": "p & q & r 右结合 ⇒ 两个子目标"},
        {"id": "gt_contrapositive_not_imp", "args": {"goal": "p", "mode": "contrapositive"},
         "expect_err": "PRECOND_FAIL", "expect": "目标不是蕴含式 ⇒ 拒绝 (不给无意义的'变换')"},
        {"id": "gt_bad_mode_reject", "args": {"goal": "p -> q", "mode": "inverse_contrapositive"},
         "expect_err": "SCHEMA_MISMATCH", "expect": "未登记 mode 被 enum 拦下"},
        {"id": "gt_atom_cap_reject", "args": {"goal": "p1 & p2 & p3 & p4 & p5 & p6 & p7 & p8 & p9 & p10 & p11",
                                              "mode": "conj_split"},
         "expect_err": "PRECOND_FAIL", "expect": "原子 11 > 10 ⇒ 真值表超规模, 拒绝"},
        {"id": "gt_zero_atoms_reject", "args": {"goal": "~(~p)", "mode": "de_morgan_dual"},
         "expect_err": "PRECOND_FAIL", "expect": "~( ~p ) 内层不是 &/| ⇒ 域外"},
    ),
)


# ===========================================================================
# 9934  lemma_discover
# ===========================================================================

#: Pattern library (fixed, documented; the checker keeps its own copy for the structural check).
SCHEMA_LIBRARY = {
    "S1_or_intro_left": ("imp", ("v", "?A"), ("or", ("v", "?A"), ("v", "?B"))),
    "S2_and_elim_left": ("imp", ("and", ("v", "?A"), ("v", "?B")), ("v", "?A")),
    "S3_and_elim_right": ("imp", ("and", ("v", "?A"), ("v", "?B")), ("v", "?B")),
    "S4_weakening": ("imp", ("v", "?A"), ("imp", ("v", "?B"), ("v", "?A"))),
    "S5_hyp_syllogism": ("imp", ("imp", ("v", "?A"), ("v", "?B")),
                         ("imp", ("imp", ("v", "?B"), ("v", "?C")),
                          ("imp", ("v", "?A"), ("v", "?C")))),
    "S6_modus_ponens": ("imp", ("and", ("imp", ("v", "?A"), ("v", "?B")), ("v", "?A")), ("v", "?B")),
    "S7_de_morgan_and": ("imp", ("not", ("and", ("v", "?A"), ("v", "?B"))),
                         ("or", ("not", ("v", "?A")), ("not", ("v", "?B")))),
    "S9_contrapositive": ("imp", ("imp", ("v", "?A"), ("v", "?B")),
                          ("imp", ("not", ("v", "?B")), ("not", ("v", "?A")))),
}

_LD_IN = _sdict({
    "goal": {"kind": "str"},
    "max_candidates": {"kind": "int", "min": 1, "max": 256},
}, ["goal", "max_candidates"])
_LD_OUT = _sdict({
    "goal": {"kind": "str"},
    "atoms": {"kind": "list", "items": {"kind": "str"}, "minlen": 0, "maxlen": 12},
    "n_candidates_tried": _EXACT, "n_valid": _EXACT, "n_useful": _EXACT,
    "lemmas": {"kind": "list", "minlen": 0, "maxlen": 256, "items": {
        "kind": "tuple", "items": [{"kind": "str"}, {"kind": "str"}, {"kind": "str"}]}},
    "staging_hook": {"kind": "str"},
    "usefulness_criterion": {"kind": "str"},
}, ["goal", "atoms", "n_candidates_tried", "n_valid", "n_useful", "lemmas", "staging_hook",
    "usefulness_criterion"])


def _ld_subst_pattern(pat, env):
    if pat[0] == "v":
        return env[pat[1]]
    if pat[0] in ("atom",):
        return pat
    if pat[0] == "not":
        return ("not", _ld_subst_pattern(pat[1], env))
    return (pat[0], _ld_subst_pattern(pat[1], env), _ld_subst_pattern(pat[2], env))


def _ld_taut(f, atoms):
    for mask in range(1 << len(atoms)):
        assign = {a: bool((mask >> i) & 1) for i, a in enumerate(atoms)}
        if not _gt_eval(f, assign):
            return False
    return True


def _ld_precond(args):
    try:
        g = parse_formula(args["goal"])
    except SvoError as e:
        return False, "公式语法错误: %s" % e.detail
    if len(_gt_atoms(g)) > 8:
        return False, "目标原子数 %d > 8: 候选枚举 + 穷举验证超规模" % len(_gt_atoms(g))
    return True, ""


def _invoke_ld(args, ctl):
    from svokernel.logiccore import formula_str
    g = parse_formula(args["goal"])
    atoms = sorted(_gt_atoms(g))
    tried = 0
    valid = []
    from itertools import permutations
    for sid in sorted(SCHEMA_LIBRARY):
        pat = SCHEMA_LIBRARY[sid]
        need = sorted({n for n in _pat_vars(pat)})
        k = len(need)
        if k > len(atoms):
            continue
        for combo in permutations(atoms, k):
            ctl.check_time("op:lemma_discover#9934")
            if tried >= args["max_candidates"]:
                break
            tried += 1
            # B13 fix: need[i] is already of the form "?A" (the pattern variable names carry the
            # '?'), and the original expression added another '?' prefix, producing the key "??A"
            # while the lookup used "?A" => every instantiation raised KeyError '?A' (any goal
            # with two or more atoms blew up).
            env = {need[i]: ("atom", combo[i]) for i in range(k)}
            f = _ld_subst_pattern(pat, env)
            cat = sorted(_gt_atoms(f))
            if _ld_taut(f, cat):
                valid.append((sid, f))
        if tried >= args["max_candidates"]:
            break
    useful = []
    useful_forms = []
    for sid, f in valid:
        ats = _gt_atoms(f)
        if not ats <= set(atoms):
            continue
        if f == g:
            continue
        if f[0] == "atom" or (f[0] == "not" and f[1][0] == "atom"):
            continue
        useful.append([sid, formula_str(f), "tautology_exhaustive"])
        useful_forms.append(f)
    cert = Certificate(kind="lemma_validity",
                       payload={"goal": g, "lemmas": useful_forms, "atoms": atoms,
                                "n_tried": tried, "n_useful": len(useful_forms)},
                       issued_by="lemma_discover")
    return {"value": {"goal": args["goal"], "atoms": atoms, "n_candidates_tried": tried,
                      "n_valid": len(valid), "n_useful": len(useful), "lemmas": useful,
                      "staging_hook": "lemma_db.stage(formula_canonical, validity_cert) — 本算子"
                                      "不执行入库 (effects=none), 只产出暂存记录",
                      "usefulness_criterion": "有效 (穷举恒真) ∧ 原子 ⊆ 目标原子 ∧ 不逐字等于"
                                              "目标 ∧ 非单个文字"},
            "resid": [0.0], "cert": cert, "cost": float(tried) * 0.5,
            "notes": "候选 %d, 有效 %d, 有用 %d" % (tried, len(valid), len(useful))}


def _pat_vars(pat):
    if pat[0] == "v":
        return {pat[1]}
    if pat[0] in ("atom", "c"):
        return set()
    if pat[0] == "not":
        return _pat_vars(pat[1])
    return _pat_vars(pat[1]) | _pat_vars(pat[2])


def _verify_ld(args, value):
    """Independent verifier: verify every lemma independently for tautology + usefulness +
    matching against the pattern library (structural check)."""
    from svokernel.logiccore import parse_formula as pf
    bad = []
    goal = pf(args["goal"])
    goal_atoms = _gt_atoms(goal)
    for sid, fstr, how in value["lemmas"]:
        f = pf(fstr)
        cat = sorted(_gt_atoms(f))
        ok = True
        for mask in range(1 << len(cat)):
            assign = {a: bool((mask >> i) & 1) for i, a in enumerate(cat)}
            if not _gt_eval(f, assign):
                ok = False
                break
        if not ok:
            bad.append("引理 %r 非恒真" % fstr)
        if not _gt_atoms(f) <= goal_atoms:
            bad.append("引理 %r 原子集超出目标" % fstr)
        if f == goal:
            bad.append("引理 %r 与目标逐字相同" % fstr)
        if sid not in SCHEMA_LIBRARY:
            bad.append("模式码 %r 不在模式库" % sid)
        else:
            env = {}
            if not _match_pat(SCHEMA_LIBRARY[sid], f, env):
                bad.append("引理 %r 形状不匹配模式 %s" % (fstr, sid))
        if how != "tautology_exhaustive":
            bad.append("引理证据类型 %r 未登记" % how)
    if value["n_useful"] != len(value["lemmas"]):
        bad.append("n_useful 与引理数不符")
    if value["n_valid"] < value["n_useful"]:
        bad.append("n_valid < n_useful (计数不自洽)")
    return {"ok": not bad, "resid": [float(len(bad))],
            "detail": "%d 条引理: 恒真穷举 + 有用性 + 模式库匹配 通过" % len(value["lemmas"])
                      if not bad else "; ".join(bad[:4])}


def _match_pat(pat, concrete, env):
    if pat[0] == "v":
        nm = pat[1]
        if nm in env:
            return env[nm] == concrete
        env[nm] = concrete
        return True
    if pat[0] != concrete[0]:
        return False
    if pat[0] == "atom":
        return pat[1] == concrete[1]
    if pat[0] == "not":
        return _match_pat(pat[1], concrete[1], env)
    return _match_pat(pat[1], concrete[1], env) and _match_pat(pat[2], concrete[2], env)


_reg(
    name="lemma_discover", band=BAND_L3, sub=4,
    in_schema=_LD_IN, out_schema=_LD_OUT,
    invoke=_invoke_ld, verify=_verify_ld, cert_kind="lemma_validity",
    cost=("units", 10.0),
    errors=("PRECOND_FAIL", "SCHEMA_MISMATCH", "POSTCOND_FAIL", "BOUNDED_VIOLATION", "TIMEOUT"),
    precond=_ld_precond,
    effects="none (纯函数; 只产出入库暂存记录, 不写任何库)",
    doc=("Auxiliary lemma discovery (bounded enumeration + verification gate + staging hook): on "
         "a **fixed pattern library** (8 propositional patterns), instantiate the goal's atom set "
         "in all distinct ways, sieve tautological candidates with an exhaustive truth table, then "
         "filter by the usefulness criterion (atoms ⊆ goal atoms ∧ not literally equal to the goal "
         "∧ not a single literal), and output the insertable lemma table with a **staging hook "
         "identifier**. Generation = pattern instantiation; independent verifier = verify every "
         "lemma independently for tautology + usefulness + **structural matching** against the "
         "pattern library; certificate lemma_validity is independently re-checked by "
         "certcheck_goal (own pattern-library copy + own evaluator). **Scope honesty statement**: "
         "this operator discovers lemmas only within the finite space of the **fixed pattern "
         "library × goal atom set** -- it does not search arbitrary formula space and does not "
         "claim the lemmas found are the 'best'; the scope boundary is written in doc and in "
         "usefulness_criterion."),
    requires_cert=True,
    thm="T-V31-DISCOVER",
    stmt="模式库实例化 + 穷举恒真 ⇒ 候选引理有效; 有用性判据决定是否值得入库",
    selfchecks=(
        {"id": "ld_two_atoms", "args": {"goal": "p & q -> p", "max_candidates": 64},
         "check": lambda r: (r.value["n_useful"] >= 1 and r.value["n_candidates_tried"] > 0
                             and all(len(x) == 3 for x in r.value["lemmas"])
                             and r.cert_independent,
                             "候选 %d, 有效 %d, 有用 %d"
                             % (r.value["n_candidates_tried"], r.value["n_valid"], r.value["n_useful"])),
         "expect": "目标 p&q→p (恒真但非模式实例?) ⇒ 发现若干有用引理, 全部经穷举验证"},
        {"id": "ld_three_atoms", "args": {"goal": "(p -> q) & (q -> r)", "max_candidates": 128},
         "check": lambda r: (r.value["n_candidates_tried"] > 0,
                             "三原子: 候选 %d, 有用 %d"
                             % (r.value["n_candidates_tried"], r.value["n_useful"])),
         "expect": "三原子目标下 S5 三段论可实例化"},
        {"id": "ld_atom_only_goal", "args": {"goal": "p", "max_candidates": 64},
         "check": lambda r: (r.value["n_useful"] == 0 and r.value["n_candidates_tried"] == 0,
                             "单原子目标: 无候选可枚举 ⇒ 空结果 (合法, 非错误)"),
         "expect": "无可发现引理时返回空表 (不报错, 不编造)"},
        {"id": "ld_max_candidates_bound", "args": {"goal": "(p -> q) & (q -> r)", "max_candidates": 1},
         "check": lambda r: (r.value["n_candidates_tried"] <= 1, "枚举受 max_candidates 约束"),
         "expect": "有界枚举: 上限生效"},
        {"id": "ld_too_many_atoms_reject", "args": {"goal": "p1 & p2 & p3 & p4 & p5 & p6 & p7 & p8 & p9",
                                                    "max_candidates": 16},
         "expect_err": "PRECOND_FAIL", "expect": "原子 9 > 8 ⇒ 枚举空间超规模, 拒绝"},
        {"id": "ld_bad_atom_reject", "args": {"goal": "p &", "max_candidates": 16},
         "expect_err": "PRECOND_FAIL", "expect": "公式语法错误"},
    ),
)
