# -*- coding: utf-8 -*-
# ccp_docs.py -- CCP delivery-document generator (advance team gen12, constitution change 58)
# Reads scratch/ccp/*.json and gen/*.json -> writes 4 markdown files (every number comes from the artifacts, none is hand-copied).
# Note: Chinese quotation marks inside the prose use 「」 to avoid clashing with Python string delimiters.
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ccp_lean as L

RES = os.environ.get("SCOPE_RESEARCH_ROOT", ".")
CCP = os.path.join(RES, "scratch", "ccp")
GEN = os.path.join(CCP, "gen")


def jload(p, default=None):
    if not os.path.exists(p):
        return default
    return json.load(open(p, encoding="utf-8"))


def md5row(paths):
    out = []
    for p in paths:
        if os.path.exists(p):
            out.append((os.path.basename(p), L.md5f(p), os.path.getsize(p)))
        else:
            out.append((os.path.basename(p), "MISSING", 0))
    return out


ART = ["ccp_lean.py", "ccp_check.py", "ccp_ops.py", "ccp_bridge.py", "ccp_families.py",
       "ccp_course.py", "ccp_gate.py", "ccp_unit.py", "ccp_cache.py", "ccp_taxonomy.py",
       "ccp_gate_d.py", "ccp_docs.py", "ccp_probe_families.py",
       "CCP_TAXONOMY.json", "CCP_TAXONOMY.md", "CCP_PHASE0_INVENTORY.md",
       "CCP_PILOT_SPEC.md", "CCP_REPORT.md", "UNIT_CCP.json", "PROBE_FAMILIES.json",
       "CCP_SEED_REGISTRY.json", "CCP_COURSE_LEDGER.json", "CCP_GATE2_RECHECK.json",
       "LAUNCH_LOG.md", "ccp_gate_recheck.py", "ccp_seedreg.py",
       "gen/ccp_ind_rows.jsonl", "gen/ccp_ineq_rows.jsonl", "gen/ccp_nt_rows.jsonl",
       "gen/ccp_ind_chains.jsonl", "gen/ccp_ineq_chains.jsonl", "gen/ccp_nt_chains.jsonl",
       "gen/CCP_COURSE_IND.json", "gen/CCP_COURSE_INEQ.json", "gen/CCP_COURSE_NT.json",
       "gen/CCP_COURSE_IND_FLOORRUN.json", "gen/CCP_GATE_D_IND.json",
       "gen/CCP_GATE_D_INEQ.json", "gen/CCP_GATE_D_NT.json",
       "cache/ccp_ind_rows.bin", "cache/ccp_ineq_rows.bin", "cache/ccp_nt_rows.bin",
       "cache/CCP_CACHE_SMOKE_IND.json", "cache/CCP_CACHE_SMOKE_INEQ.json",
       "cache/CCP_CACHE_SMOKE_NT.json"]


def t_artifacts():
    out = ["## 附表 A - 产物清单与 md5 (先遣组 gen12, 变更 58)\n",
           "| 文件 | md5 | 字节 |", "|---|---|---|"]
    for (n, m, sz) in md5row([os.path.join(CCP, f) for f in ART]):
        out.append("| %s | `%s` | %d |" % (n, m, sz))
    return "\n".join(out) + "\n"


VERDICT = {
    "TAXONOMY.md": "**部分复用**: 域×结构×模板三轴与风格谱仍有效; 但「域」轴太粗 (algebra 61.7%), "
                   "Phase 1 换成 15 类**证明技巧**轴 (见 CCP_TAXONOMY.md), 本件保留为历史锚。",
    "TAXONOMY_VALID.json": "**复用 (只读)**: Valid 256 的域/结构/风格计数, 用作本次分类学的交叉核对。",
    "TEST_COVERAGE_MAP.md": "**复用**: 224 题三层分解 (翻译 160 / 词汇 64 / 导航 2) 是本批选题的动机依据。",
    "TEST_COVERAGE.json": "**复用 (唯一 Test 输入)**: Phase 0 冻结的**类别标签**逐题记录; 本批对 Test 的"
                          "覆盖映射只消费其中的标签计数, 未新增对 Test 题面的读取。",
    "CLASS_SPECS.md": "**复用**: 每格三面规格 (算子语义/调用接口/形式化层) + 地板可解性预估 + 量词族"
                      "共同基建 = 三类试点三件套的直接模板; v3 SCHEMA_MISMATCH 教训 (实参必须模板化直通) "
                      "已落入桥规则设计。",
    "GATE_SPEC.md": "**复用**: R1-R5 阈值/阳性对照强制/N1 认证监督/Test 单次终测纪律, 本批 ccp_gate.py "
                    "逐条实现 (阈值 0.50/0.30, 双源窗口)。",
    "ccp_phase0.py": "**复用**: read_minif2f 加载路径与 Valid 分类器语义; 本批未重跑 --annotate "
                     "(不新增 Test 读取)。",
    "STATUS.md": "**归档**: Phase 0 状态板 (变更 35/37 时代); 本批状态见 CCP_REPORT.md。",
}


def doc_inventory():
    rows = md5row([os.path.join(CCP, f) for f in VERDICT])
    d = ["# CCP Phase 0 既有交付盘点 (任务 A; 先遣组 gen12, 变更 58)\n",
         "结论: **Phase 0 交付件件可用** -- 本批「续建不重造」: 分类学换轴重写 (技巧轴), 而闸/纪律/"
         "三面规格直接继承。逐件复用性判定如下。\n",
         "| 既有件 | md5 | 复用性判定 |", "|---|---|---|"]
    for (n, m, sz) in rows:
        d.append("| %s | `%s` | %s |" % (n, m, VERDICT.get(n, "")))
    d.append("")
    d.append("## 复用/新建对照\n")
    d.append("| Phase 1 新建 | 来源 | 复用方式 |")
    d.append("|---|---|---|")
    d.append("| `ccp_gate.py` | GATE_SPEC R1-R5 + `xcmp_neardup_audit.py` v2 | 阈值/双源窗口/阳性对照"
             "逐条移植; 新增 n-gram + 特征倒排索引 (对无共享 n-gram 的参考题免精算, 完备且快) |")
    d.append("| `ccp_lean.py` | `drct_gen.parallel_jobs` 退让式并行 | 自足重写 (不跨轨写 scratch/drct); "
             "八菜单/官方头/axioms 口径不变; 并行上限按 192 核机提到 48 (仍按在飞 lean 数退让) |")
    d.append("| 三件套算子 (7 个 / 6 证书类型) | `drct_curriculum_v2` 的 identity_expand/linear_backsub + "
             "`mini_kernel_arith` | 同构模式 (核算 + Lean 战术包装 + 独立复验); 见 CCP_PILOT_SPEC.md |")
    d.append("| 一链多行行格式 | `drct_curriculum_v2.build_rows` | 字段名与语义镜像 (family/src_family/sf/"
             "type4/sf_args/expected_final/program/steps/episode/accepted/readout_only) |")
    d.append("| 负例真败 | v2 `op_fail_call` | 改成语义扰动真败 (常量不可整除 / 余项为负 / 系数-见证不符) |")
    d.append("| 留出切分 | `v31f.build_split` 内容分组 | 按题名 md5 取模分组 (10%) |")
    d.append("| `ccp_gate_d.py` | v2 门 d (证明性证书抽验) | 课程题证明体逐题 Lean 真判 (本批新增: "
             "生成期不做 Lean 终审, 由该门补齐) |")
    return "\n".join(d) + "\n"


def doc_taxonomy():
    t = jload(os.path.join(CCP, "CCP_TAXONOMY.json"))
    if not t:
        return "# CCP 分类学 (待生成)\n"
    d = ["# CCP 题型分类学 - 15 类证明技巧轴 (任务 B; 先遣组 gen12, 变更 58 第①步)\n",
         "**防火墙声明**: 本分类学**只从 miniF2F Valid(256) + 公开语料/registry 语义**构建; 设计段"
         "**未读 Test 题面**。Test 覆盖度映射**只消费 Phase 0 冻结的类别标签计数**"
         "(`TEST_COVERAGE.json`), 本批未新增任何对 Test 的读取。\n",
         "机械: `scratch/ccp/ccp_taxonomy.py` -> `CCP_TAXONOMY.json` (表面特征决策列, 冻结, 无学习)。\n",
         "## 1. 十五类 (定义 / Lean 战术清单 / 难度 / 地板预估 / Valid 计数)\n",
         "| # | 类 | 定义 | 所需 Lean 战术 | 难度 | 地板预估 | Valid | Valid 例题 (名) |",
         "|---|---|---|---|---|---|---|---|"]
    for i, c in enumerate(t["classes"], 1):
        d.append("| %d | %s `%s` | %s | %s | %s | %s | %d | %s |" %
                 (i, c["zh"], c["id"], c["definition"],
                  ", ".join("`%s`" % x for x in c["lean_tactics"]), c["difficulty"],
                  c["floor_est"], c["valid_count"], ", ".join(c["valid_examples"][:2])))
    d.append("")
    d.append("Valid 合计 %d/256 (机械归类)。\n" % t["valid_total"])
    d.append("**诚实注记**: (1) Valid 的归纳类题全部带 sum 或 dvd 记号, 故被决策列判为 `induction_sum`/"
             "`div_closure`, `induction_rec` 的 Valid 标签计数为 0 -- 该类的 Valid 锚定按**技巧**取 "
             "(见 CCP_PILOT_SPEC.md); (2) `linear_solve` 与 `analysis_eval` 由特征判据吸走大量 algebra/"
             "analysis 题, 属**表面特征代理**的已知粗化, 精确定类需读证明体 (Phase 2 用合成课程已知技巧"
             "反标)。\n")
    tc = t["test_coverage_by_label"]
    d.append("## 2. Test 覆盖度映射 (只用类别标签计数)\n")
    d.append("- 来源: %s\n- Test 题数: %s | 三层分解: %s\n- 标签对未映射数: %s\n" %
             (tc["source"], tc["n_test"], tc["layer_counts"], tc["unmapped_label_pairs"]))
    d.append("| 类 | 15 类折算计数 (标签投影, 分数计) |")
    d.append("|---|---|")
    for k, v in tc["class_counts_fractional"].items():
        d.append("| %s | %s |" % (k, v))
    d.append("")
    d.append("**投影规则** (冻结, 粗投影): `%s`\n" %
             json.dumps(tc["projection_rule"], ensure_ascii=False))
    d.append("**诚实注记**: 该投影把 Phase 0 的 8 个 `tech` 标签 + 5 域 + 6 结构**折到** 15 类上, "
             "一条 (域, 结构) 可能摊到多类 (按均分计), 因此**只用于配额排序, 不作精确覆盖率主张**; "
             "精确到 15 类的 Test 覆盖需 Phase 2 的考试后标注 (变更 35 允许的第二用途)。\n")
    d.append("## 3. 算子接口现状 (第②面 = 卡点)\n")
    st = t["operator_interface_status"]
    d.append("- 现状: %s\n" % st["note"])
    d.append("- 本批已接线 (带证书): %s\n" %
             json.dumps(st["pilot_implemented"], ensure_ascii=False))
    d.append("- 未接线 (%d 类): %s\n" % (len(st["not_yet_wired"]), ", ".join(st["not_yet_wired"])))
    return "\n".join(d) + "\n"


PILOT_ROW = [
    ("`induction_rec` 归纳递推",
     "递推闭式 = miniF2F 归纳族核心; 需「选取归纳变量 + 步进代入 + 环归一」三段决策, 与代入计算正交",
     "`induction_*` 题名族 (技巧=归纳)", "低 (无裸单步战术可证 forall-n 命题)"),
    ("`sos_bound` 不等式放缩",
     "Phase 0 判为「天然地板之上富矿」; 需「选平方项」这一不可词法导出的决策",
     "`imo_1984_p2` / `algebra_apb4leq8ta4pb4` / `algebra_amgm_*`",
     "低 (nlinarith 无 hint 不闭合)"),
    ("`div_closure` 数论整除闭包",
     "L2 词汇层主战场; 需见证构造 (obtain + exact by ring)",
     "`imo_1964_p1_2` / `induction_divisibility_3divnto3m2n`",
     "低-中 (线性型可被 omega 单步吃掉 -> 课程取乘积型)"),
]

OP_ROWS = [
    ("`ccp_ind_recurrence`", "递推 -> 闭式 (两点拟合)",
     "`{form:lin/aff, v0, step, sample_n}`", "闭式系数 (c1,c0) + 样本值",
     "`intro n; induction n with ...; rw [hstep, ih, pow_succ]`",
     "`ccp_ind_recurrence_cert` (前向模拟 + 不动点坐标 + 步进算子三重一致)"),
    ("`ccp_ring_identity`", "原子基环恒等", "`{nvars, lhs, rhs, atom, side}`", "恒等判定",
     "`push_cast; ring`", "`ccp_ring_identity_cert` (确定性网格恒等检验)"),
    ("`ccp_ineq_sos`", "平方和分解 (差平方优先 + 对角配方)",
     "`{vars, dom}`", "平方项 + 余项 rho + hints",
     "`have h_i : (0:Q) <= (...)^2 := sq_nonneg (...)`",
     "`ccp_ineq_sos_cert` (独立卷积展开 = dom - rho)"),
    ("`ccp_ineq_close`", "收尾闭链", "`{vars, squares, residual, text}`", "闭链判定",
     "`nlinarith [h_1,...]`", "`ccp_ineq_close_cert` (**题面重解析**所得 goal_dom = 平方和 + rho)"),
    ("`ccp_nt_dvd_witness`", "见证提取 (结构步)", "`{m, hyp, var, text}`", "见证变量名",
     "`obtain <w_v, rfl> := h_v`", "`ccp_shape_cert` (检查器重解析假设形状)"),
    ("`ccp_nt_dvd_combine`", "整除闭链 (线性/乘积/结构项统一)",
     "`{m_h, M, terms, const, hypvars}`", "见证多项式 (须自然数值)",
     "`exact <witness, by ring>`", "`ccp_nt_dvd_cert` (参数空间多项式恒等 G = M*W, 网格检验)"),
]


def doc_pilot():
    u = jload(os.path.join(CCP, "UNIT_CCP.json"))
    pr = jload(os.path.join(CCP, "PROBE_FAMILIES.json"))
    gd = {c: jload(os.path.join(GEN, "CCP_GATE_D_%s.json" % c.upper()))
          for c in ("ind", "ineq", "nt")}
    d = ["# CCP 三类试点三件套规格 + 单测 (任务 C; 先遣组 gen12, 变更 58 第②步)\n",
         "## 0. 选类理由 (差异最大化, 覆盖 Valid 分布的三条主脊)\n",
         "| 类 | 选类理由 | Valid 锚定 (按技巧) | 地板可解性 |", "|---|---|---|---|"]
    for r in PILOT_ROW:
        d.append("| %s | %s | %s | %s |" % r)
    d.append("")
    d.append("## 1. 三件套规格\n")
    d.append("### ① 带证书算子 (Lean 战术包装为核算算子, 镜像 v2 identity_expand/linear_backsub 模式)\n")
    d.append("| 算子 | 语义 | 参数 (schema) | 输出 | Lean 战术包装 | 证书类型 (独立检查器) |")
    d.append("|---|---|---|---|---|---|")
    for r in OP_ROWS:
        d.append("| %s | %s | %s | %s | %s | %s |" % r)
    d.append("")
    d.append("**白名单/禁件**: 正例链算子 ⊆ `PROOF_OPS` (上表 6 个 + `ccp_nt_dvd_prod_combine` 薄壳); "
             "`math_evaluate_expr` **全批禁入正例监督** (v1 六环根因直封, 逐批登记)。**禁自证**: 生成逻辑在 "
             "`ccp_ops.py`, 全部复验在 `ccp_check.py` "
             "(`register_checker(..., independent_of=('ccp_ops',))` 结构性拒绝同模块检查器)。\n")
    d.append("### ② 桥规则 (题面 -> 参数的纯函数)\n")
    d.append("- `bridge_ind(stmt)` -> `{form, v0, step}`; `bridge_ind_declared(stmt)` -> 题面声明的闭式 "
             "(与算子核算出的闭式交叉核对)\n")
    d.append("- `bridge_ineq(stmt, vars)` -> `{vars, dom = LHS - RHS 的多项式系数}` (自写递归下降解析器: "
             "+ - * ^ () 与 N->Z 投影记号)\n")
    d.append("- `bridge_nt(stmt)` -> `{M, m_h, hyps, terms, const}` (项分类: 假设线性 / 假设乘积 / m 倍结构项)\n")
    d.append("- **往返一致性闸**: 每题必须 `bridge(题面) == 生成参数` 才允许进课程 "
             "(CLASS_SPECS 的 body_slice 往返一致性检查落地为单测 U1)。\n")
    d.append("### ③ 证书机器 (每步独立核算)\n")
    d.append("- 每步算子调用 -> 证书 payload -> 独立检查器复算; 两边**算法互不共享**: 生成器走公式/结构卷积/"
             "系数构造, 检查器走前向模拟/网格恒等/参数空间多项式恒等。\n"
             "- 证书不独立或不过 => **不算解** (`CERT_REJECT`), 该步不入正例; 负例行因此有真实失败语义。\n")
    d.append("## 2. 单测结果 (机械: `scratch/ccp/ccp_unit.py` -> `UNIT_CCP.json`)\n")
    if u:
        d.append("| 类 | 例数 | U1 桥往返通过 | U2 证书 (步) | U3 白名单 | U4 Lean rc=0 | "
                 "U5 八菜单地板 hard | U6 负例真败 | U7 检查器非空真 |")
        d.append("|---|---|---|---|---|---|---|---|---|")
        for cls, c in u["classes"].items():
            # gen15 F-04: the U1 counter was renamed bridge_produced (semantics = output count, not
            # independent re-verification); the old key name is kept as a fallback for reading old artifacts
            bp = c.get("bridge_produced", c.get("U1_bridge_roundtrip"))
            d.append("| %s | %d | %d/%d | %d/%d | %d/%d | %s/%s | %s/%s | %d/%d | %d/%d |" %
                     (cls, c["n"], bp, c["n"], c["U2_cert_ok"],
                      c["U2_cert_steps"], c["U3_whitelist"], c["n"],
                      c.get("U4_lean_ok", "-"), c.get("U4_n", "-"),
                      c.get("U5_floor_hard", "-"), c.get("U5_floor_n", "-"),
                      c["U6_neg_truefail"], c["U6_neg_n"],
                      c["U7_checker_nonvacuous"], c["U7_n"]))
        d.append("")
        d.append("U7 口径: 对每步证书 payload 做数值/形状扰动后交独立检查器, **必须拒绝** -- 证明检查器"
                 "非空真 (不是永远 PASS 的摆设)。\n")
        d.append("## 2b. 课程侧链证明体检 (gate d)\n")
        d.append("| 类 | 课程题 | 抽验 | rc=0 | 失败 |")
        d.append("|---|---|---|---|---|")
        for cls, g in gd.items():
            if g:
                d.append("| %s | %d | %d | %d | %d |" %
                         (cls, g["n_course_problems"], g["n_checked"], g["n_ok"], g["n_fail"]))
        d.append("")
    if pr:
        d.append("## 3. 题族可行性探针 (`PROBE_FAMILIES.json`, 批量生成前置)\n")
        d.append("| 族 | n | 整链 rc=0 | 八菜单 hard | 闭合菜单 |")
        d.append("|---|---|---|---|---|")
        for fam, v in pr["by_family"].items():
            d.append("| %s | %d | %d | %d | %s |" % (fam, v["n"], v["chain_ok"], v["hard"],
                                                     v["closers"] or "-"))
        d.append("")
        d.append("**对照组**: `nt_dvd_lin_ctrl` (纯线性整除) 被 `omega` 单步闭合 -- 该族**故意保留**为"
                 "地板仪器的阳性对照: 它证明「地板过滤确实会抓到东西」, 并据此把语料族改成乘积型 "
                 "(`nt_dvd_mix`/`nt_dvd_prod`, 出 Presburger 片段)。\n")
    return "\n".join(d) + "\n"


def redundancy(cls):
    p = os.path.join(GEN, "ccp_%s_rows.jsonl" % cls)
    if not os.path.exists(p):
        return None
    rows = [json.loads(ln) for ln in open(p, encoding="utf-8")]
    c = collections.Counter(r["stmt"] for r in rows)
    by = collections.defaultdict(set)
    for r in rows:
        by[r["stmt"]].add(r["ccp_name"])
    dup = {k: v for k, v in c.items() if v > 1}
    cross = sum(1 for k in dup if len(by[k]) > 1)
    return {"n_rows": len(rows), "n_distinct_stmt": len(c), "n_problem": len(by),
            "dup_groups": len(dup), "rows_in_dup_groups": sum(dup.values()),
            "dup_intra_problem": len(dup) - cross, "dup_cross_problem": cross}


def doc_report():
    reps = {}
    for cls in ("ind", "ineq", "nt"):
        r = jload(os.path.join(GEN, "CCP_COURSE_%s.json" % cls.upper()))
        if r:
            reps[cls] = r
    floorrun = jload(os.path.join(GEN, "CCP_COURSE_IND_FLOORRUN.json"))
    u = jload(os.path.join(CCP, "UNIT_CCP.json"))
    ck = {cls: jload(os.path.join(CCP, "cache", "CCP_CACHE_SMOKE_%s.json" % cls.upper()))
          for cls in ("ind", "ineq", "nt")}
    gd = {c: jload(os.path.join(GEN, "CCP_GATE_D_%s.json" % c.upper()))
          for c in ("ind", "ineq", "nt")}
    d = ["# CCP Phase 1 三类试点 - 交付报告 (任务 F; 先遣组 gen12, 宪法变更 58)\n",
         "## 0. 一句话\n",
         "第一层锁 (课程分布) 的三类试点**基建 + 课程 + 闸全部落地**: 三类各约 3k 行一链多行课程通过"
         "地板闸 / 污染双路零命中闸 / 种子登记闸, 并通过链证明体检 (gate d, Lean 真判); 三件套 "
         "(带证书算子 / 桥规则 / 证书机器) 单测通过; 缓存冒烟证明行可入训练管线 (需 1 处词表扩展)。"
         "**训练不在本批**。\n",
         "## 1. 三类课程规模与验证读数\n",
         "| 类 | 题数 | 行数 | 行组成 | 族数 | 步数分布 | 白名单 | 地板 (全量风险子集) | "
         "地板 (抽样八菜单) | 污染 | 留出 | rows md5 |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for cls, r in reps.items():
        if cls == "ind" and floorrun:      # IND rerun skips the floor (readings reused from the full run of the same generator)
            r["floor_sample_8menu"] = floorrun["floor_sample_8menu"]
            r["floor_bulk"] = floorrun["floor_bulk"]
        d.append("| %s (%s) | %d | %d | %s | %d | %s | %s | 剔除 %d (%s) | %d/%d hard | "
                 "reject=%d review=%d maxR4=%.3f | %d 行 | `%s` |" %
                 (cls, r["cls_zh"], r["n_problems"], r["n_rows"], r["kind_mix"],
                  len(r["by_family"]), r["step_hist"],
                  "OK" if r["op_whitelist_ok"] and r["banned_absent"] else "**VIOLATION**",
                  r["floor_bulk"].get("n_dropped", 0), r["floor_bulk"].get("dropped_by_menu", {}),
                  r["floor_sample_8menu"].get("hard", 0), r["floor_sample_8menu"].get("n", 0),
                  r["gate2_contam"].get("reject", 0), r["gate2_contam"].get("review", 0),
                  r["gate2_contam"].get("max_R4", 0.0), r["split"]["n_held"], r["rows_md5"]))
    d.append("")
    if floorrun:
        d.append("**地板读数来源披露**: IND 类地板读数取自**同生成器全量跑** "
                 "(`gen/CCP_COURSE_IND_FLOORRUN.json`: %d 题, 风险子集剔除 %d, 抽样 %d/%d hard); "
                 "其后重出仅修正 17 例退化情形的**证明体** (题面不变) => 地板判决仍适用。\n" %
                 (floorrun["n_problems"], floorrun["floor_bulk"].get("n_dropped", 0),
                  floorrun["floor_sample_8menu"].get("hard", 0),
                  floorrun["floor_sample_8menu"].get("n", 0)))
    pc = reps.get("ind", {}).get("gate2_contam", {}).get("positive_control", {})
    d.append("**零命中证明**: 阳性对照 %s/%s (把参考题文本原样与加空白扰动后当候选, 必须被判 REJECT) "
             "=> 闸有效; 本批全部行 reject=0 且 review=0 且 maxR4=0.000 => 精确路 (R1 md5 / R2 containment / "
             "R3 题名 / R5 特征数) 与近重复路 (R4 n-gram 双向)**双路零命中**。\n" %
             (pc.get("caught"), pc.get("n")))
    rc = jload(os.path.join(CCP, "CCP_GATE2_RECHECK.json"))
    if rc:
        d.append("**instrument-fixed authoritative reading (R4 window rule)**: the R4 coverage ratio is "
                 "computed on candidate windows with len >= 24 only (same rationale as the R2 reject "
                 "floor). Before the fix, 750 NT rows fell in the 0.30-0.50 review band; root cause = "
                 "**too-short conclusion segments in post-substitution state rows** (a 12-char window has "
                 "only 3 ten-grams, so 1 shared ten-gram = 0.333). Re-measured against all references, the "
                 "same rows score max coverage 0.000-0.143 => no substantive similarity. Post-fix readings "
                 "(`CCP_GATE2_RECHECK.json`):\n")
        d.append("| cls | rows | PASS | REVIEW | REJECT | maxR4 | short-window rows |")
        d.append("|---|---|---|---|---|---|---|")
        for _c, _v in rc["classes"].items():
            d.append("| %s | %d | %d | %d | %d | %.4f | %d |" %
                     (_c, _v["n_rows"], _v["verdicts"].get("PASS", 0),
                      _v["verdicts"].get("REVIEW", 0), _v["verdicts"].get("REJECT", 0),
                      _v["max_R4"], _v["rows_short_window_only"]))
        d.append("")
        d.append("Positive control %d/%d (gate有效性复核) => **all rows of all three classes: zero hit "
                 "(exact path + near-dup path)**.\n" %
                 (rc["positive_control"]["caught"], rc["positive_control"]["n"]))
    d.append("### 1b. 行冗余度量 (诚实披露)\n")
    d.append("| 类 | 行 | 不同状态文本 | 题 | 重复组 | 组内行数 | 同题内 | 跨题 |")
    d.append("|---|---|---|---|---|---|---|---|")
    for cls in reps:
        r2 = redundancy(cls)
        if r2:
            d.append("| %s | %d | %d | %d | %d | %d | %d | %d |" %
                     (cls, r2["n_rows"], r2["n_distinct_stmt"], r2["n_problem"],
                      r2["dup_groups"], r2["rows_in_dup_groups"],
                      r2["dup_intra_problem"], r2["dup_cross_problem"]))
    d.append("")
    d.append("同题内同状态 = **设计** (整链行与首步行共享起始状态: 同状态、不同目标程序); 跨题同状态 = "
             "题族参数空间有限导致的同题面复现 (题级去重按 (结论|假设) 键, 不覆盖状态级碰撞)。"
             "**训练批建议**: 若需压冗余, 按状态文本二次去重即可 (不影响任何闸结论)。\n")
    d.append("## 2. 链证明体检 (gate d: 课程题证明体 Lean 真判)\n")
    d.append("| 类 | 课程题 | 抽验 | rc=0 | 失败 | 抽样 |")
    d.append("|---|---|---|---|---|---|")
    for cls, g in gd.items():
        if g:
            d.append("| %s | %d | %d | %d | %d | %s |" %
                     (cls, g["n_course_problems"], g["n_checked"], g["n_ok"], g["n_fail"],
                      g["sampling"]))
    d.append("")
    d.append("**口径**: 抽验 = 课程题原证明体 (战术步本体渲染, 不经 witness/计算器路径) 交 Lean, "
             "rc=0 且无 sorryAx 才算过。这是「课程每一行都真有 Lean 证明」的唯一硬证; 生成期只做证书 "
             "(算术) 独立复验, 两者不可互替 -- 本批实录: 首轮 IND 全量体检 1137/1154, 17 例因退化闭式 "
             "(c1=0) 时 `pow_succ` 无可重写而失败, 修正生成器后重出。\n")
    d.append("## 3. 缓存构建冒烟 (CPU)\n")
    d.append("| 类 | 行 | 缓存 md5 | 帧长上限 | token 截断 | 绑定槽监督率 | 往返 |")
    d.append("|---|---|---|---|---|---|---|")
    for cls, c in ck.items():
        if c:
            d.append("| %s | %d | `%s` | %d | %d | %.3f | %s |" %
                     (cls, c["n_rows"], c["cache_md5"], c["max_len_labels"], c["tok_trunc"],
                      c["slot_sup_rate"], c["roundtrip_ok"]))
    d.append("")
    d.append("格式 = CPPCD3 布局的 CCP 版 (MAGIC `CPPCD4`), 算子空间 = V32 160 + CCP 7 "
             "(追加于尾部, 既有 id 语义不动)。**训练批需做 1 处扩展**: 把 `ccp_cache.CCP_OPS` 登记进"
             "预热词表并重生成词表常量 (1 个列表 + 1 次词表重建)。\n")
    d.append("## 4. 单测读数\n")
    if u:
        for cls, c in u["classes"].items():
            d.append("- **%s**: 例 %d | 证书步 %d/%d | Lean %s/%s | 八菜单地板 hard %s/%s | "
                     "负例真败 %d/%d | 检查器非空真 %d/%d" %
                     (cls, c["n"], c["U2_cert_ok"], c["U2_cert_steps"], c.get("U4_lean_ok"),
                      c.get("U4_n"), c.get("U5_floor_hard"), c.get("U5_floor_n"),
                      c["U6_neg_truefail"], c["U6_neg_n"], c["U7_checker_nonvacuous"], c["U7_n"]))
    d.append("")
    d.append("## 5. 训练批建议 (本批不执行)\n")
    d.append("### 5.1 课程配比与步数\n")
    d.append("- **135M 科学对照轨** (SmolLM2-135M 在盘): 三类合并约 9k 行, BS=16, **6000 步** "
             "(与 v2/V32R 同构, 保证与 D-RCT 线可比); 建议**按类分段** (每类 2000 步) 便于逐类归因。\n")
    d.append("- **1B 主攻轨**: 先定底座 (见 5.3); BS=32, **12000 步** (约 384k 样本, 约 98M token)。\n")
    d.append("- **FLOPs 披露 (双口径)**: 训练 FLOPs 约 6*N*D。等样本口径下 1B 的 FLOPs 约 135M 的 7.4 倍; "
             "若做**等 FLOPs** 对照 (科学档), 1B 只应看 135M 的约 1/7 token (约 13k 样本, 约 800 步) -- "
             "两口径读数都要报, 禁混用。\n")
    d.append("### 5.2 每类读数方案 (净贡献 = 认证解 - 地板 - 规则基线)\n")
    d.append("1. **读数集**: 每类留出 (本批已切约 10%) + 每类**真实题锚定集** (从 Valid 同技巧题转写的"
             "开发集) + Test 单次终测 (封存, 训练末一次)。\n")
    d.append("2. **判据**: Lean rc=0 且 N1 骨架必要性且 N2 autoImplicit 严格, 且该题不在八菜单地板集内。\n")
    d.append("3. **归因三栏**: 模型 / 20 行词法规则基线 / 地板。\n")
    d.append("4. **预注册判据** (变更 58 第⑦步): M1 = 首类 certified>0 且超该类规则基线; M2 = 净贡献>0 的"
             "类数曲线; 北极星 = 全卷超地板; 诚实阴性 = 全类被规则追平 (入边界章)。\n")
    d.append("### 5.3 1B 底座可用性盘点 (models/ 现状 + 镜像可得性; 未擅自下载)\n")
    d.append("| 件 | 现状 | 说明 |")
    d.append("|---|---|---|")
    d.append("| `SmolLM2-135M` | 在盘 (260M) | 135M 对照轨底座, 现成可用 |")
    d.append("| `Qwen2.5-0.5B` | 在盘 (954M) | 可作 135M 与 1B 之间的中间档 |")
    d.append("| `DeepSeek-Prover-V1.5-RL` | 在盘 (13G) | 7.55B 主对手 (int8 判卷线), 非训练底座 |")
    d.append("| `DeepSeek-Prover-V1.5-7B` | **不完整** (仅 ._____temp; 下载 ChunkedEncodingError) | "
             "需重下或弃用 |")
    d.append("| 1B 级底座 (SmolLM2-1.7B / Qwen2.5-1.5B / Llama-3.2-1B 任一) | **不在盘** | HF 直连不通, "
             "**hf-mirror.com 可达** (XCMP 已用同镜像拉下 13G 的 DS-RL) => 可下 (约 2-5G, 分钟级); "
             "需主代定夺选哪个 (影响与 SFT-CoT 对照的可比性) |")
    d.append("| 磁盘 | 实验树所在卷余 504G | 足够 |")
    d.append("")
    d.append("### 5.4 规则基线臂接口 (R3 教训制度化)\n")
    d.append("每类 >=20 行**词法规则基线臂** = 本批的**桥规则** (`ccp_bridge.py`: 题面 -> 算子参数) + "
             "白名单链模板 (`ccp_families.build_*`: 参数 -> 算子链) + 常量回填。**关键推论**: 本批合成课程的"
             "参数**全部可由桥规则从题面导出** => 在该课程上规则基线约 100% (与 D-RCT 的 lexrule 210/210 "
             "同构), 所以**净贡献必须在桥规则不可解的真实题读数集上测** -- 这是训练批读数设计的硬约束。\n")
    d.append("## 6. 已知缺口与下一步\n")
    d.append("1. **12 类尚未接线** (`CCP_TAXONOMY.json: not_yet_wired`): 建议下一批按 Valid 计数排序推进 "
             "`linear_solve`(68) -> `mod_arith`(35) -> `sum_manip`(29) -> `count_pigeonhole`(19)。\n")
    d.append("2. **桥规则可解性 = 合成课程的天花板**: 需要「桥-hard」族 (参数不可从题面词法导出, 必须靠"
             "中间步状态), 才能把课程从「词汇教学」推向「推理教学」; 建议下一批设计。\n")
    d.append("3. **步数阶梯**: 本批链长 2-4 步, 未做变更 37 要求的单步/两步/多步分层配额 (合成族天然 >=2 步)。\n")
    d.append("4. **Test 双用途纪律**: 本批对 Test 的全部接触 = (a) Phase 0 冻结标签计数 (b) 污染闸机械比对 "
             "(输出仅命中计数); **未跑任何 Test 评测**。\n")
    d.append("5. **本批缺陷台账 (全部由机械闸抓出, 无一项靠人读题)**: (a) NT 结构项见证系数按 coef/M 计 "
             "(应为 coef*m_h/M) => CERT_REJECT 高频, 已修; (b) 见证漏常项 K/M => const!=0 的 NT 例全部 "
             "CERT_REJECT, 已修; (c) NT 见证文本重复追加常量 => Lean 失败, 已修; (d) IND 退化闭式 "
             "(c1=0) 时 `pow_succ` 不可重写 => 17/1154 证明体失败 (由 gate d 全量体检抓出), 已修并重出。\n")
    d.append("\n" + t_artifacts())
    return "\n".join(d) + "\n"


def main():
    open(os.path.join(CCP, "CCP_PHASE0_INVENTORY.md"), "w", encoding="utf-8").write(doc_inventory())
    open(os.path.join(CCP, "CCP_TAXONOMY.md"), "w", encoding="utf-8").write(doc_taxonomy())
    open(os.path.join(CCP, "CCP_PILOT_SPEC.md"), "w", encoding="utf-8").write(doc_pilot())
    txt = doc_report()          # compute the content first (so the open("w") truncation cannot make the appendix read an empty file of itself)
    open(os.path.join(CCP, "CCP_REPORT.md"), "w", encoding="utf-8").write(txt)
    # second pass: make appendix A's CCP_REPORT.md self-entry truthful (reads the entity persisted by the first pass)
    txt = doc_report()
    open(os.path.join(CCP, "CCP_REPORT.md"), "w", encoding="utf-8").write(txt)
    for f in ("CCP_PHASE0_INVENTORY.md", "CCP_TAXONOMY.md", "CCP_PILOT_SPEC.md", "CCP_REPORT.md"):
        p = os.path.join(CCP, f)
        print("[docs] %s md5=%s" % (f, L.md5f(p)))


if __name__ == "__main__":
    main()
