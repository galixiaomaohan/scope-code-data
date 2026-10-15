# -*- coding: utf-8 -*-
# ccp_taxonomy.py -- CCP 15-class problem taxonomy (advance team gen12, constitution change 58 step 1)
#
# Hard firewall: the taxonomy is built **only from miniF2F Valid(256) + public math corpora /
#   existing registry semantics**; this module **never reads Test statements**. The Test coverage
#   mapping consumes only the **class-label counts** frozen at Phase 0
#   (scratch/ccp/TEST_COVERAGE.json, produced mechanically by ccp_phase0.py --annotate); it adds no
#   new reads of Test.
# Artifacts: scratch/ccp/CCP_TAXONOMY.json (machine-readable) + CCP_TAXONOMY.md (human-readable).
import collections
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ccp_lean as L

RES = os.environ.get("SCOPE_RESEARCH_ROOT", ".")
CCP = os.path.join(RES, "scratch", "ccp")
MINIF2F = os.environ.get("SCOPE_MINIF2F", os.path.join(RES, "lean", "miniF2F-main"))

# ---------------------------------------------------------------- the 15 class definitions (proof-technique axis)
CLASSES = [
    ("subst_compute", "代入计算链", "沿假设等式逐条代入并闭式求值 (等式链/闭式代入)",
     ["intro", "rw [h_i]", "norm_num", "ring", "simp only [h_i]"], 1, "低-中"),
    ("linear_solve", "线性/联立求解", "多元一次方程组消元、线性不等式组求解",
     ["linarith", "omega", "ring_nf", "nlinarith"], 1, "低-中"),
    ("eq_identity", "代数恒等变形", "多项式/有理式恒等 (展开、因式分解、配方恒等)",
     ["ring", "ring_nf", "field_simp", "norm_num", "rfl"], 1, "低"),
    ("sos_bound", "配方法/平方和非负", "结论为序关系, 证法 = 写成平方和 (含余项非负)",
     ["nlinarith [sq_nonneg (...)]", "positivity", "linarith", "ring_nf"], 2, "低"),
    ("amgm_cauchy", "均值/柯西/三角放缩", "AM-GM、柯西-施瓦茨、三角不等式的模板化应用",
     ["nlinarith", "gcongr", "linarith"], 3, "低"),
    ("bound_chain", "单调夹逼/放缩链", "多步序关系传递 (le_trans/lt_of_le_of_lt/单调性)",
     ["gcongr", "linarith", "linarith [h_i]", "exact le_trans .."], 2, "低"),
    ("induction_rec", "归纳递推(闭式)", "递推关系 → 闭式; 归纳法 + 环归一",
     ["induction n with", "rw [hstep, ih]", "pow_succ", "ring", "omega"], 2, "低"),
    ("induction_sum", "归纳求和", "∑ 恒等式的归纳证明 (基例 + sum_range_succ)",
     ["induction n with", "Finset.sum_range_succ", "push_cast", "ring_nf", "nlinarith"], 3, "低"),
    ("sum_manip", "求和变换", "裂项/换元/平移/重排等求和恒等式变形",
     ["Finset.sum_congr", "Finset.sum_range_succ", "Nat.sub_add", "ring_nf", "norm_num"], 3, "低"),
    ("comb_binomial", "组合计数/二项式", "choose/binomial/Pascal/Vandermonde/计数恒等",
     ["Nat.choose_succ_succ", "Finset.card", "ring_nf", "Omega"], 3, "低-中"),
    ("div_closure", "整除闭包", "由若干假设 m∣x_i 推出组合式/乘积被 m^k 整除 (见证构造)",
     ["obtain ⟨w, rfl⟩ := h", "exact ⟨witness, by ring⟩", "Dvd.dvd", "ring"], 2, "低"),
    ("mod_arith", "模算术/同余", "同余链、模幂、剩余类计算",
     ["omega", "Nat.ModEq", "Nat.pow_mod", "ZMod", "decide", "norm_num"], 2, "中"),
    ("prime_factor", "素因数/因数结构", "Prime/gcd/lcm/因数个数/算术基本定理应用",
     ["Nat.Prime", "Nat.factorization", "Nat.gcd_eq_left", "omega", "norm_num"], 3, "低-中"),
    ("count_pigeonhole", "计数/鸽巢/容斥/存在", "存在性与计数界 (鸽巢、容斥、双射)",
     ["Finset.card", "by_contra", "push_neg", "omega", "Nat.card_le_card_of_injOn"], 3, "低-中"),
    ("analysis_eval", "分析/函数求值", "导数/积分/极限/级数/三角与指数的符号求值",
     ["norm_num", "ring_nf", "Real.exp", "deriv", "Real.sin_cos", "nlinarith"], 3, "低-中"),
]


def _feat(binders, concl, name):
    t = (binders or "") + " || " + (concl or "")
    f = {}
    f["sum"] = bool(re.search(r"∑|Finset\.sum|\.range|Icc|Ico", t))
    f["dvd"] = "∣" in t
    f["mod"] = bool(re.search(r"%|≡|ModEq|ZMod", t))
    f["prime"] = bool(re.search(r"\bPrime\b|\bgcd\b|\blcm\b|factorization|\bfactorial\b|Coprime", t))
    f["choose"] = bool(re.search(r"choose|binomial|Pascal", t))
    f["card"] = bool(re.search(r"\bcard\b|Pigeonhole|injection|surjection|bipartite", t))
    f["ineq"] = bool(re.search(r"≤|≥|<|>", concl or ""))
    f["eq"] = bool(re.search(r"=(?!=)", concl or ""))
    f["analysis"] = bool(re.search(r"deriv|∫|Integral|limit|Tendsto|Continuous|Real\.exp|Real\.log|"
                                   r"Real\.sin|Real\.cos|Real\.sqrt|Complex|convex", t))
    f["powvar"] = bool(re.search(r"\^\s*\(?[nkmijt]\b|\^\s*\(?[a-z]\s*[+\-*]", t))
    f["forall"] = bool(re.search(r"∀\s*\(?[a-z]", concl or ""))
    f["exists"] = bool(re.search(r"∃", concl or ""))
    f["hypchain"] = len(re.findall(r"\(h[₀-₉0-9]*\s*:", binders or "")) >= 1
    f["sq"] = bool(re.search(r"\^\s*2\b", t))
    return f


def classify(binders, concl, name):
    """Frozen decision column (a single priority chain; the order is the priority)."""
    f = _feat(binders, concl, name)
    if f["analysis"]:
        return "analysis_eval"
    if re.search(r"amgm|cauchy|jensen|sqrt|gm_|p2msqrt", name + " " + (concl or "")):
        return "amgm_cauchy"
    if f["sum"] and f["powvar"]:
        return "induction_sum"
    if f["sum"] and f["choose"]:
        return "comb_binomial"
    if f["sum"]:
        return "sum_manip"
    if f["choose"]:
        return "comb_binomial"
    if f["card"] or f["exists"]:
        return "count_pigeonhole"
    if f["dvd"]:
        return "div_closure"
    if f["prime"]:
        return "prime_factor"
    if f["mod"]:
        return "mod_arith"
    if f["forall"] and f["hypchain"]:
        return "induction_rec"
    if f["ineq"] and f["sq"]:
        return "sos_bound"
    if f["ineq"]:
        return "bound_chain"
    if f["forall"] and f["powvar"]:
        return "induction_rec"
    if f["hypchain"] and f["eq"] and not f["powvar"]:
        return "linear_solve"
    if f["hypchain"] and f["eq"]:
        return "subst_compute"
    if f["eq"]:
        return "eq_identity"
    return "subst_compute"


def load_valid():
    import importlib.machinery
    import importlib.util
    ld = importlib.machinery.SourceFileLoader(
        "le_ccpt", os.path.join(RES, "probes", "lean_exam.py.txt"))
    sp = importlib.util.spec_from_loader("le_ccpt", ld)
    le = importlib.util.module_from_spec(sp)
    ld.exec_module(le)
    return [p for p in le.read_minif2f(MINIF2F) if p.split == "Valid"]


# frozen Phase-0 labels -> **coarse projection** onto the 15 classes
# (consumes label counts only; the projection error is disclosed in the report)
PROJ = {
    ("inequality", "prove_inequality"): ["sos_bound", "bound_chain"],
    ("inequality", "solve_value"): ["bound_chain"],
    ("numbertheory", "prove_divisibility"): ["div_closure"],
    ("numbertheory", "solve_value"): ["mod_arith"],
    ("numbertheory", "prove_inequality"): ["bound_chain"],
    ("combinatorics", "solve_value"): ["comb_binomial"],
    ("combinatorics", "prove_identity"): ["induction_sum", "sum_manip"],
    ("combinatorics", "prove_inequality"): ["amgm_cauchy"],
    ("analysis", "solve_value"): ["analysis_eval"],
    ("algebra", "solve_value"): ["subst_compute"],
    ("algebra", "prove_identity"): ["eq_identity"],
    ("algebra", "count_or_exist"): ["count_pigeonhole"],
    ("algebra", "other"): ["eq_identity"],
}


def project_test_labels():
    p = os.path.join(CCP, "TEST_COVERAGE.json")
    d = json.load(open(p, encoding="utf-8"))
    cnt = collections.Counter()
    unmapped = 0
    layers = collections.Counter()
    for name, rec in d["per_problem"].items():
        c = rec["cls"]
        key = (c["domain"], c["struct"])
        tgt = PROJ.get(key)
        if not tgt:
            unmapped += 1
            tgt = ["subst_compute"]
        share = 1.0 / len(tgt)
        for t in tgt:
            cnt[t] += share
        layers[rec["layer"]] += 1
    return {"source": "scratch/ccp/TEST_COVERAGE.json (Phase 0 冻结类别标签; 未新增 Test 读取)",
            "n_test": d["layer_counts"] and sum(d["layer_counts"].values()),
            "layer_counts": d["layer_counts"],
            "class_counts_fractional": {k: round(v, 2) for k, v in cnt.most_common()},
            "unmapped_label_pairs": unmapped,
            "projection_rule": {"%s|%s" % k: v for k, v in PROJ.items()}}


def main():
    valid = load_valid()
    assert len(valid) == 256, len(valid)
    cnt = collections.Counter()
    ex = collections.defaultdict(list)
    for p in valid:
        c = classify(p.binders, p.concl, p.name)
        cnt[c] += 1
        if len(ex[c]) < 3:
            ex[c].append(p.name)
    rep = {"schema": "ccp-taxonomy/1", "split_used_for_design": "Valid(256) only",
           "firewall": "设计段未读 Test 题面; Test 仅用 Phase 0 冻结类别标签计数做覆盖映射",
           "n_classes": len(CLASSES),
           "classes": [{"id": cid, "zh": zh, "definition": dfn, "lean_tactics": tactics,
                        "difficulty": diff, "floor_est": flo,
                        "valid_count": cnt.get(cid, 0), "valid_examples": ex.get(cid, [])}
                       for (cid, zh, dfn, tactics, diff, flo) in CLASSES],
           "valid_total": sum(cnt.values()),
           "unmatched": [p.name for p in valid if classify(p.binders, p.concl, p.name) not in
                         {c[0] for c in CLASSES}][:5],
           "test_coverage_by_label": project_test_labels(),
           "operator_interface_status": {
               "note": "registry 名义算子 → 接口实现状态 (Phase 0 CLASS_SPECS 结论: 只有代入计算一种范式接了接口)",
               "pilot_implemented": {
                   "induction_rec": ["ccp_ind_recurrence (带证书)", "ccp_ring_identity (带证书)"],
                   "sos_bound": ["ccp_ineq_sos (带证书)", "ccp_ineq_close (带证书)"],
                   "div_closure": ["ccp_nt_dvd_witness (带证书)", "ccp_nt_dvd_combine (带证书)"]},
               "not_yet_wired": ["amgm_cauchy", "bound_chain", "induction_sum", "sum_manip",
                                 "comb_binomial", "mod_arith", "prime_factor", "analysis_eval",
                                 "count_pigeonhole", "eq_identity", "linear_solve", "subst_compute"]}}
    js = os.path.join(CCP, "CCP_TAXONOMY.json")
    md5 = L.save_json(js, rep)
    print("[tax] Valid=%d | class distribution %s" % (rep["valid_total"], dict(cnt.most_common())))
    print("[tax] → %s md5=%s" % (js, md5))
    return rep, md5


if __name__ == "__main__":
    main()
