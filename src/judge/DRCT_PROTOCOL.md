# DRCT_PROTOCOL.md — 域内随机对照试验预注册协议（宪法变更 38 立项）

- **组**: D-RCT 专组 · 立项 2026-09-22 · 本文件为预注册件，**在任何臂运行之前锁定**（md5 见 `DRCT_PROTOCOL.lock.md5`）
- **沙箱**: 服务器只写 `$R/scratch/drct/**`；`probes/`、`scratch/xcmp|evid|repro|v34|e1/`、`docs/`、`cppcuda/` 只读
- **宪法同步**: `docs/EXPERIMENT_PLAN_P1P2.md` md5 `430a03780f8d9d5a4307c259dfeed5b8`；`docs/DEVIATIONS.md` md5 `96108f7f117284f943ca7a96a86ef581`（只读，禁编辑）
- **锁定纪律**: 测试集 `drct_problems.json` 的 md5 与本协议在任一臂跑之前锁定；锁后改任何口径须先落 `ESCALATION_时间戳.md` 请主智能体裁决

---

## 0. 一句话设计与动机

在**算子域内**构造"地板-hard"测试集（域内题但八菜单单战术无法闭合），使 **认证解 = 真解决**
（Lean 完整证明 + N1 非装饰 + N2 严格 + 无 sorryAx，金标准自动满足、地板=0 by construction），
与**同底座（SmolLM2-135M）同规模**的 CoT/PoT 文本推理对照臂比认证正确率与成本，
因果隔离"架构（潜空间算子推理 vs 文本自回归推理）"这一个变量。

旧域内考卷 v1（19 题）被八菜单 100% 饱和（`scratch/evid/ALP_FLOOR_SUMMARY.json`: subset19
floor_rate=1.0），解出不证明内容优势——本试验的直接动因。

## 1. 域定义（机械，预注册）

**算子域** = 形式化层可接受的 miniF2F 形态题，机械判据四连环（全部只读复用既有仪器）：

1. `le.screen(p).class ∈ {C_CLOSED, S_SUBST}`（`probes/lean_exam.py.txt`：算术文法
   `ARITH_TOK_RE` + 排除表 `EXCLUDE_PAT` + 假设等式消解闭包；域内关系符 = `= ∣ ≤ < ≥ >`，
   `≠` 被文法排除）
2. `lc30.ref_slice(p, sc)` 无 err（`probes/lean_chains_v30c.py.txt`：ArithParser 解析+精确
   Fraction 求值，两侧全 ground，关系为真）
3. `le.RefProposer(tier="mathlib", imports=["Mathlib"], use_svokernel=False).propose(p, sc)` 非 None
4. `sd.run_program(["math_evaluate_expr"], "synth", sf)` 可执行（`probes/sla_data.py.txt`）

**算子词表** = V32 动作空间 161 头（`skel32/v32lib.build_allow()`，0..153 legacy + 154 END +
155..160 骨架算子；本域考所需算子 `math_evaluate_expr` idx=15，零 OOV——与 R5/R6 同口径）。

**题面渲染头** = XCMP E3 官方头（`import MiniF2F.ProblemImports` +
`open scoped Real Nat Topology Polynomial`），与 XCMP 判卷/alp_floor 逐字节同口径。

## 2. 测试集构建规格（drct_problems.json，目标 ≥200 题）

### 2.1 生成机制

自写参数化生成器 `scratch/drct/gen/drct_gen.py`（参考训练数据生成机制
`probes/sla_data.py.txt` 的 synth_chain 桥接思想，但题面为 miniF2F 形态 Lean 定理）。
每题 = 有序假设链 `(h0 : v0 = <闭式>) (h1 : v1 = <含 v0 的式>) ...` + 结论
`<lhs> <rel> <机器算出的值>`；**值一律 ArithParser 机器计算，绝不手算**
（试点教训：手算错值会被 RefProposer 正确拒绝）。

**障碍族**（试点 2026-09-22 夜实测校准，`pilot/PILOT2_RESULTS.json` + `PILOT3_CERTS.json`）：

| 族 | 类型 | 机制（为何过八菜单） |
|---|---|---|
| vdq | ℚ 变量除子 | `c = a/(y+b)`：linarith 视 `a/(y+b)` 为不透明原子；norm_num 不用假设 |
| vdr | ℝ 变量除子 | 同上 |
| vdp | ℝ 变量积除子 | 除子含变量积 `a/(p*q-4)` |
| rpw | ℝ rpow | 指数为变量（整数值）；2^k 原子化 |
| mdv | ℤ 变量模 | `n % k` 模数含变量；omega 只折叠常量模 |
| cdr | ℝ 深除链 | 2-7 层依赖的复合除法链 |
| cdq | ℚ 深除链 | 同上 |
| ltv | ℝ < 关系+变量除子 | linarith 原子失配 |
| pwz | ℤ 幂族 | `b^4 - b^2`：nlinarith 乘积组合不足 |
| rcq | ℚ 积+互易 | `u*v + 1/(u*v)`：非线性+互易原子 |

淘汰族（试点实测被菜单折叠，不入集）：ℤ 闭幂+常量模（omega 折叠）、纯 ℚ/ℝ 线性代入
（linarith 闭合，留作 few-shot 示例池）。

### 2.2 难度分层（预注册，机械）

**参考算子链长** `ref_chain_len = 假设消解步数 + 1（终值求值）= len(hyps) + 1`。

| 档 | 假设数 | ref_chain_len | 配额 |
|---|---|---|---|
| L1（floor+1 步） | 1–2 | 2–3 | 70 |
| L2（2 步超出地板） | 3–4 | 4–5 | 70 |
| L3+（3+ 步） | 5–7 | 6–8 | 70 |

全集目标 **210 题**（硬下限 200：每档 ≥50 时总 ≥150 不足以满足 200，故配额定 70/档；
若某档过滤后不足 50，落 ESCALATION 说明并如实报实际数）。生成候选 ~900 题（种子见
§2.5），过滤后每档按种子序取满 70，溢出备选不入集（记录于候选池 json）。

### 2.3 双边过滤（逐题，全部 Lean 实测）

- **①域内性**：§1 四连环全过（真实仪器跑，非复述）。
- **②地板-hard**：八菜单 `simp / norm_num / decide / linarith / nlinarith / omega / ring / rfl`
  逐战术单步渲染实测（机械照抄 `scratch/evid/alp_floor.py`：单行战术、2 空格缩进、
  `nice -n 19 lake env lean`、cwd=miniF2F 根、timeout 600s、rc==0 ∧ 非 not-run ∧ 无
  sorryAx = 闭合）——**全部 8 个失败才收**。
- **③Lean 可判（题面 elaborate）**：官方头 + 题面 + `:= by sorry` 编译 rc==0 且含
  sorryAx（题面可 elaboration，类型正确）。
- **④真值证书（构造期过滤器，非判卷认证器）**：`simp only [全部假设] + norm_num`
  编译 rc==0 无 sorryAx（主证书）；`rw [逆依赖序假设] + norm_num`（备证书）。二者任一
  过即证题在 Lean 语义下为真。**注意：simp only [假设列表] 是双战术组合，不在八菜单
  单战术集合内，不与 ② 冲突。**
- **⑤污染闸**：对 miniF2F Test+Valid 全 489 题零命中——规范化 containment（结论/题面
  规范化文本不得为任何 miniF2F 题面子串）+ 双向 10-gram 覆盖 max(fwd,rev) < 0.5
  （阈值照抄 `scratch/xcmp/xcmp_neardup_audit.py` near_dup 0.80 / review 0.50 档，
  DRCT 取更严的 0.5 硬上限，任何 ≥0.5 即弃）。机械照抄该件 norm/ngrams 双源窗口法。

### 2.4 每题元数据

`{name, stratum, family, ty, hyps, concl, head, raw, md5, lv, rel, ref_chain_len,
floor: {tactic: {closed, rc, secs}}, elab: {...}, cert: {variant, rc, axioms},
contam: {max_fwd, max_rev, containment_hit}}` + 生成种子追溯字段。

### 2.5 种子与确定性

主种子 `20260922`；逐题由 `(master_seed, stratum, family, seq)` 决定（random.Random
实例级，无全局态）；生成器文件与题集 json 一并锁 md5。重跑生成器须逐字节复现题集
（确定性断言写入生成器自检）。

## 3. 臂清单与采样预算（Phase B，GPU 排队后）

| 臂 | 底座 | 运行器 | 采样 |
|---|---|---|---|
| A1 V32 关态 | SmolLM2-135M 潜空间 | `scratch/drct/drct_v32_run.py`（薄壳：只读 import `probes/lean_chains_v32.py.txt` 的 load_model/nav_v32/FrameBuilder32 与 lc30/lc31 同口径装配，题源 = 本测试集 Problem 对象，非 miniF2F 文件）--ckpt `cppcuda/out/v32_state_v32bAP_final.pt` | n_attempts=8，seed_k = seed + 1000π + k（R5 同式） |
| A2 V32 开态 | 同上 | 同上 + think 三闸参数（F=200/N=32/W=600，R6 同参数） | 同上 |
| A3 SmolLM2-CoT | SmolLM2-135M 文本 | `scratch/drct/drct_player.py`（仿 xcmp_player：fp16、greedy pass@1 + T=1.0 top_p=0.95 pass@8、max_new=min(2048, ctx)、种子 seed=20260922+1000π+k、RepetitionStop 同式） | 3 few-shot 示例取自**不相交的易闭合 S_SUBST 示例池**（不入集，池 md5 锁定） |
| A4 SmolLM2-PoT | 同上 | 同上 --arm pot（PoT 提示风格照抄 xcmp_prompts） | 同上 |
| A5 Qwen2.5-0.5B-CoT（可选参考臂，非主要比较） | Qwen2.5-0.5B | 同上 --model models/Qwen2.5-0.5B | 同上 |

**架构因果隔离**: A1/A2/A3/A4 底座与规模完全相同（SmolLM2-135M）；A1/A2 与 A3/A4
只差推理形态（潜空间算子链 vs 文本）。**主比较 = A2 vs A3**（我方最强形态 vs 文本
推理最强传统形态）；A1 为消融。

GPU 纪律（修正案 06）：一卡一作业；上卡前 nvidia-smi 查空闲 >4500MB；排在 XCMP 选手
与 E1 考试之后。判卷 CPU ≤32 路 + nice 19，动态退让（照抄 alp_floor 退让控制器）。

## 4. 判卷（全体臂同闸）

1. **Lean 真判**: 官方头 + 题面 + 证明体 → `lake env lean`（cwd=miniF2F，600s），
   accepted = rc==0 ∧ 非 not-run ∧ 无 sorryAx。
2. **N1 剥参装饰判废**: **e0a 版语义**（`scratch/v34/e0a_n1n2.py` md5
   `7a127bafd0f1839c24930c72b3d4f990`，对拍胜出——只剥良构参数段保持语法合法：
   `simp only [..]→simp`、`simpa only [..] using h→simpa`、`(field_simp|norm_num|
   ring_nf|linarith|nlinarith|simp_all|simp_arith|simp)[..]→裸头`、
   `(simpa|linarith|nlinarith) using X→裸头`；**不剥 exact/rw/apply 等必带参战术**）。
   剥后仍闭合 ⇒ 装饰 ⇒ 判废。⚠️ xcmp_common.certify 的 N1 是旧版（剥全部实参），
   DRCT 判卷**不得**使用。
3. **N2 autoImplicit 严格**: `set_option autoImplicit false` + `relaxedAutoImplicit false`
   复编译失败 ⇒ 判废。
4. **E2 健全性**: sorryAx 依赖 ⇒ 判废（含在 1）。
5. **认证解定义**: Lean 接受 ∧ N1 非装饰 ∧ N2 过。判卷器 = `scratch/drct/drct_judge.py`
   （A3-A5）与 `scratch/drct/drct_v32_judge.py`（A1/A2，链→witness 三闸→渲染→同闸），
   两器共享 `scratch/drct/drct_common.py`；V32 链 plan 装配照抄 lc30.build_plan
   （首成功算子必须 math_evaluate_expr）。
6. 去重：同臂同题逐字节相同证明只判一次（值复用）。

## 5. 结果定义

- **认证正确率**: certified pass@1（greedy/首尝试）与 certified pass@8（8 随机尝试
  任一认证）；分母 = 卷内题数（200+）。
- **每认证解成本**: Σ(生成 token) / certified_solves@8；另报 FLOPs 代理
  （2·参数量·token）、墙钟、Lean 调用次数、VRAM 峰值。
- **吞吐**: 题/小时（含判卷）。
- **分层读数**: L1/L2/L3+ 各档 certified@1/@8。
- **raw vs certified 差**: Lean 接受率 − 认证率（装饰率读数）。

## 6. 统计（预注册）

1. **J1 疗效（共同主要结果①）**: A2 vs A3 certified pass@8 同题配对——精确 McNemar
   （不一致对二项检验，双侧 α=0.05）+ 配对 bootstrap 95% CI（10⁴ 重抽，种子 20260922）。
2. **J2 效率（共同主要结果②）**: 每认证解 token 成本 A2 vs A3——bootstrap 95% CI
   （成本比 ratio）。J2 判"优" = A2 成本显著低于 A3（CI 上界 < 1）。
3. **双判据总判**: J1 显著优 ∧ J2 显著低 → "架构绝对优势成立"；仅 J1 → "疗效成立，
   效率未证"；J1 不显著 → 如实报，不得事后换指标。次要：A1 vs A2（think 消融）、
   A4 对比、分档读数、A5 参考描点。
4. **零膨胀规则**: 若两臂 certified@8 皆 0 → 主结果记"不可区分"，报 rule-of-three
   95% CI（<3/n），不做任何事后挑选。
5. **多种子稳健性**: 主统计只用预注册种子（s+0）；另跑 s+7、s+13 两种子全套四臂，
   只报方向一致性，不做推断。
6. 分层读数为描述性（n=50/档），不做事后显著性（避免多重比较膨胀；如报 p 值须
   Holm 校正并明示探索性）。

## 7. 风险预注册（诚实边界）

1. **我方模型从未测过地板-hard 域内题**——可能极弱甚至 0%；若 A1/A2 全零而 A3 非零，
   结论如实为负；若全零，按 §6.4 零膨胀规则报不可区分。**这是本试验的诚实目的：
   量出真实位置，不制造有利读数。**
2. CoT few-shot（rw/norm_num 范式）可能让文本臂在 L1-L2 拿分——如发生，属真实架构
   比较的一部分，如实报。
3. 障碍族是构造性的：过八菜单 ≠ 过所有单战术（Mathlib 战术空间更大）；认证解仍需
   完整 Lean 证明，金标准不受影响；但"域内无单战术捷径"的表述严格限定在八菜单口径。
4. 生成题为合成分布，与 miniF2F 自然分布有差距——本试验问的是**域内架构因果效应**，
   不外推全域结论。
5. 测试集与训练数据的关系：本生成题**不在任何训练池**（污染闸⑤只对 miniF2F 审计；
   对训练缓存 rows_v32_mixed_clean/v34_pool 的同法审计在题集锁定前补跑，命中即弃——
   预注册为收题硬条件，若来不及跑完则落 ESCALATION 声明后补）。

## 8. 交付物

- `scratch/drct/DRCT_PROTOCOL.md`（本件）+ `DRCT_PROTOCOL.lock.md5`
- `scratch/drct/gen/drct_gen.py` + `gen/CANDIDATES_*.json`（生成器+候选池）
- `scratch/drct/drct_problems.json` + `drct_problems.md5`（测试集）
- `scratch/drct/gen/FLOOR_LEDGER.jsonl`（八菜单逐题×战术台账，append-only）
- `scratch/drct/gen/CONTAM_AUDIT.json`（污染闸审计）
- `scratch/drct/STATUS.md`（状态落盘，主智能体每 20 分钟巡检）
- Phase B: `runs/**`、`drct_player.py`、`drct_judge.py`、`drct_v32_judge.py`、`drct_common.py`
- Phase C: `DRCT_RESULTS.md`（双判据结论/分层读数/测试集 md5/污染审计结论/诚实边界）

## 9. 时间线

- 今夜 24:00 前：协议+测试集草样（本件 §2 全机械落实）
- 明天 GPU 窗口（XCMP/E1 之后，预计 21:00 后）：四臂开考 + 判卷
- 判卷后 24h 内：DRCT_RESULTS.md

---

## 修正 A1（2026-09-22 18:5x CST，主智能体令——臂未开考前，赛前修订）

**修订动因**：用户质询"基模型没训练过而我方训练过，不是公平对照"。增补第 5 臂
SFT-CoT（训练匹配对照）。

**修订前版本**：md5 `67e90a2b447fbe51db258b6ad85e36c0`（2026-09-22 17:5x 锁）。
**修订时点核验**：武装守护 18:40 点火至本修订时零发射（LAUNCH_LOG.md 可查）——
所有臂均在修订后开考，属干净的赛前修订，非事后换口径。**§1-§9 其余条款全部不变，
四臂主判据 J1/J2 不变。**

### A6 SFT-CoT 臂（训练匹配对照；结果表作因果解释行，不入主判据）

1. **SFT 数据**（`gen/sft_data.jsonl`，md5 随件锁定）：由同 10 障碍族、同深度分布
   （2..7，镜像测试集 L1:L2:L3≈1:1:1）**独立种子流**（master seed 90212026，与测试
   集 20260922 流不相交）新生成 ~6000 题。逐题：机器算值 + 域内四连环（screen/
   ref_slice/RefProposer/run_program）+ **对 210 考题与 miniF2F 同规格污染闸**
   （规范化 containment + 双向 10-gram max<0.5 命中即弃 + 精确键去重）+ 证明文本 =
   `rw [逆依赖序假设]` + `norm_num`（考试 few-shot 同款、试点 210/210 全过变体）
   **逐题 Lean 编译验证**（rc=0 无 sorryAx）。注册决定：训练题**不逐题**做八菜单
   地板过滤，另抽 200 题子探针实测并披露地板-hard 率（`gen/SFT_FLOOR_PROBE.json`）。
2. **训练**（`drct_sft_train.py`，超参登记于 `runs/sft_model_train_meta.json`）：
   SmolLM2-135M **全参 SFT**（与 V32 全参训练形态对齐，非 LoRA）；completion-only
   损失（prompt 段 label=-100）；lr 2e-5、3 epochs、有效 batch 16、cosine、
   warmup 3%、fp16、seed 90212026；训练文本 = 考试同款 few-shot prompt + 证明 + eos。
3. **考试**：与四臂同卷（drct_problems.json md5 7bb077da…）同判分闸（Lean 真判 +
   e0a N1 + N2 + sorryAx）同采样预算（greedy pass@1 + T=1.0 top_p 0.95 pass@8，
   种子 20260922+1000·题序+k）；runner 复用 drct_player.py（--model 指向 SFT 产物）。
4. **排程**：不阻塞四臂；守护队列 = v32_off → v32_on → smollm_cot → smollm_pot →
   sft_train → sft_cot（依赖闸：sft_cot 待 sft_train 完成后发射）；一卡一作业照旧。
5. **诚实注记（预注册）**：我方 V32 课程比 SFT 数据**更宽**（mixed_clean 全域 vs
   本域窄分布）——"训练匹配"是**同域同级**匹配，非逐字节同课；SFT-CoT 若仍显著
   落后 V32，不能完全排除"课程广度"贡献，如实在 DRCT_RESULTS 披露。反向同理：
   SFT-CoT 若大幅超 CoT，则"训练与否"是主要混杂，四臂主比较须降格解读为
   "架构+训练"联合效应。

---

## 修正 A2（2026-09-22 19:0x CST，主智能体最高优先级令——臂仍未开考，干净修订）

**修订动因**：用户令"直接开始重新训练，我们的模型和基模型都开始训练，然后立刻
用来做严格对比实验"。D-RCT 从"测现有 V32"升级为"**重训双方→立刻对比**"
（双向训练匹配，消除"训练与否"混杂）。这是 CCP Phase 1 处方（地板过滤课程+
认证监督，变更 37）的首次实战落地。

**修订前版本**：v2 md5 `34c38e453a84d8357e179da3a127bf73`。**修订时点核验**：
守护零发射（LAUNCH_LOG.md 可查），所有臂在修订后开考。§1-§9 与修正 A1 其余
条款不变；A1 的 SFT-CoT 臂完整保留并入本修正的主对照。

### A2.1 我方重训 V32R（治疗臂）

1. **课程**（`drct_curriculum.py` → `gen/curr_rows.jsonl` + `cache/rows_drct_curriculum.bin`
   + `out/batch_idx_drctR.txt`）：同 10 障碍族独立种子流（92320260）新生成 **15000 题**
   （R1 缩档备 8000）；行 schema 与 mixed_clean 同源（`prep_cache_v31.build` 权威渲染器
   消费的 v31 synth 行；g1 交叉校验全过）；**链监督 = 内容承重步**：program=
   [math_evaluate_expr]，sf/sf_args 携带机器算真值，svokernel 实跑 8d resid——
   剥离该步即无解，N1 认证监督构造性满足（变更 37 处方）；10% END-fail 教学
   （END 时机）+ 10% 真败负例（BCE/readout）。
2. **四要求**：①地板-hard 族级保证 + 抽 250 题八菜单实测（报告披露率）；②见上
   （链监督=生成链）；③污染闸对 210 考题 + miniF2F 同规格零命中 + 精确键去重；
   ④真值证书抽验 300 题（rw+norm_num 编译）。
3. **训练**（`cppcuda/tools/v32b_train.py` 只读复用，argv 照 E1/v32bAP 先例）：
   `--arch v32 --init ckpt/sla_v31f.pt --cache rows_drct_curriculum.bin
   --genidx batch_idx_drctR.txt --prior-on 1 --prior-table v31g_prior_table.pt
   --det-train 1 --bs 16 --steps 6000 --snap-every 500 --eval-every 500
   --seed 20260922 --tag drctV32R`；产物 `runs/v32r_state_final.pt` +
   `runs/curve_drctV32R.json`（R3：it500 loss/eval 读数必看；快照单文件模式）。
   --skel-src 指向不存在件以禁用骨架声明口径 heldout（本课程无 skel 段，该读数
   不适用——诚实披露）。
4. **考试**：drct_v32_run.py --ckpt runs/v32r_state_final.pt，off/on 双态，
   同卷同判分闸同采样预算。

### A2.2 基模型重训 SFT-CoT：照修正 A1 原计划执行（~6k 全参 SFT）。

### A2.3 主对照与参考行（覆盖 A1 的主判据配置）

- **主对照 = V32R-on vs SFT-CoT**（双向训练匹配：双方都在本域数据上重训过，
  架构是唯一剩余变量）。V32R-off 为主对照内的消融（think 态）。
- **参考行**：原 V32 双态（v32bAP_final）+ 未训 SmolLM2 CoT/PoT——给"重训前后
  增量"叙事（v32_off vs v32r_off / smollm_cot vs sft_cot 增量读数）。
- 统计：主 McNemar+bootstrap 同 §6；成本比跨介质口径 = "每认证解底座前向数"
  （V32 帧 vs 文本 token，同为 135M 底座单步前向，如实披露近似性）。

### A2.4 判决门 G-D（结果叙事预注册，变更 42）

**V32R 认证正确率 McNemar 显著优 ∧ 每认证解成本不劣（CI 上界 ≤1）⇒ 架构主张
成立；任一不成立 ⇒ 诚实阴性 + 分层诊断直接喂 CCP Phase 1。两种结果都是有效交付。**

### A2.5 降级规则 R1-R5（变更 42，授权自主执行）

- R1 课程构建超时（>23:00 未齐）→ 缩至 ~8k 行（检验功效仍足）；
- R2 卡窗不足（DS 片占卡）→ V32R 优先上双卡、SFT 串行后补，考试顺延；
- R3 V32R 训练失稳 → it500 曲线读数必看（loss/eval）；失稳则停（精确 PID）、
  原 V32 升回主对照、V32R 降参考行（优雅降级，实验不废）；
- R4 判卷优先级：D-RCT 判卷全项目最高（48 路可用满，XCMP 让位；判卷器默认
  不再 nice）；
- R5 硬时限：9/23 晚前 DRCT_RESULTS v2 落盘（论文 9/24 早终审）。

### A2.6 排程

守护队列（首空卡即发，依赖闸通用化）：v32r_train →（并行机会）sft_train →
v32r_off → v32r_on → sft_cot → v32_off → v32_on → smollm_cot → smollm_pot →
判卷（48 路）→ drct_stats → DRCT_RESULTS v2。E1 考试与 XCMP p8 不动（用其放卡）。
