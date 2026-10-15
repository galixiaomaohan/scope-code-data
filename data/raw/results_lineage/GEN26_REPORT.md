# GEN26_REPORT.md — 潜空间混合主臂：设计 + 建造 + 100 步联合冒烟

- **批次**: gen26（gen27 审后修正令执行批）
- **法源**: 宪法变更 **74**（A 主臂 = 潜空间混合架构）· **75**（红线 R0：潜空间思考主线不可偏离）· 49（模型路由/装置计算）· 73（M1_A）· §7 发现 8（三病）
- **范围**: 设计（HYBRID_DESIGN.md，md5 `6fded8224c6bd870a61773c93e742456`）+ 建造（4 件）+ 100 步联合冒烟（135M 轨，GPU0 独占门 free=12059MiB）+ 三臂五列评测接口。**正式训练未开**（前置见 §5）。
- **生成时刻**: 2026-09-23T19:2x+0800

---

## 0. 【首位】CRIT-1 修复与错位敏感性单测（修正令要求放报告首位）

**病灶**（审核批坐实）：`gen26_hybrid.render_batch` 原式 `rtgt[np:len] = ids[np:len]` ⇒ 输出位置 p 的标签 = 位置 p 的**输入**（自预测），tie_word_embeddings 下成为零代价"抄输入"捷径；且位置 `np-1`（唯一可预测首个生成 token 的位置）恒为 −100 ⇒ 推理首步无监督。

**修法**：`rtgt[np-1 : len-1] = ids[np : len]`（位置 p 标签 = `ids[p+1]`）。

**敏感性证据**（`gen26_selftest.py`，随单测 28/28 PASS 落盘 `out/GEN26_SELFTEST.json`）：

| 测试 | 判据 | 实测 |
|---|---|---|
| MISALIGN1 | 位置 `np-1` 的标签 == `ids[np]` | pos=137 label=3725 == ids[np]=3725 **PASS** |
| MISALIGN2 | 修正后 `label==input` 位置数 < 全部 | **0 / 48**（全不等）**PASS** |
| MISALIGN3 | 旧式目标与正确目标**逐位**不同 | 不同位置数 = 237 **PASS** |
| **MISALIGN4**（模型无关判据） | "抄输入头"（one-hot(input)×40）在旧式目标上 CE≈0、在正确错位目标上 CE≫1 | **旧式 CE = 0.0000（捷径存在，被证明）· 正确错位 CE = 40.0000（捷径失效）** **PASS** |
| MISALIGN5 | 起点模型两目标损失互不等价（观测列） | 正确 5.1712 vs 旧式 10.7379（ntxt 237/237）**PASS** |

> MISALIGN4 是**不依赖模型训练状态**的机制判据：旧式目标可被恒等复制头零损失实现（平凡捷径**存在性证明**），修正后同一头损失 40.0（捷径**失效证明**）。MISALIGN5 另记起点模型观测：两目标不等价（注：起点处"抄输入"并非更易——因 tie 嵌入下 logits 近均匀，实测旧式 10.74 高于正确 5.17；这不影响 MISALIGN4 的机制结论）。

**重跑**：修正后 100 步联合冒烟已重跑（§3，曲线见 `out/HYBRID_CURVE.jsonl` md5 `72efc838de3b5429dc21bd76bfd83f1e`）。

---

## 1. 交付物（逐件 md5）

| 件 | 路径（$R/scratch/ccp/gen26/） | md5 | 说明 |
|---|---|---|---|
| 设计书 | `HYBRID_DESIGN.md` | `6fded8224c6bd870a61773c93e742456` | HIGH-3 补写；模型/训练/考试接口/对照/风险五节，与代码逐条对应 |
| 接口适配 | `gen26_hybuild.py` | `563ef0a67c171a0b0b68406be325d406` | 四条件过滤 + 精确 join + 混合缓存 + 渲染/考场侧车 + 算子空间覆盖诊断 |
| 模型+训练 | `gen26_hybrid.py` | `12e14ca8209373ff161d24d860e9eef0` | HybridModel（链头+LM 头同体）+ 联合目标 + 交错混喂 + 三曲线 + 覆盖列 |
| 考试接口 | `gen26_eval.py` | `6f35a85eaa2b7c784e5583babddfbca8` | 自由走链留痕 + 渲染 + 归因 + 三臂 + 四段 + 冻结判卷接口 |
| 单元自测 | `gen26_selftest.py` | `63d97fa2f9a2f6dec3233a320a5d93c3` | 28 项（含 CRIT-1 敏感性 5 项），中间落盘 |
| 链缓存 | `cache/hyb_a_all.bin` (+`.segidx.json`) | `55dc37bb1076ef21e48c80863fea8756` | CPPCD5 布局，1424 行（train 1188 / held 236） |
| 渲染侧车 | `out/A_RENDER_SIDECAR.jsonl` | `ff90b412de72a7c1e005963bb704a572` | 1424 行；token 合计 239168 / 行 max 218 |
| 考场侧车 | `out/A_EXAM_SIDECAR.jsonl` | `398943f090fa1e354c216423276551a1` | 主判据集 1083 + 真题 3（join miss 0） |
| 构建报告 | `out/A_HYBUILD.json` | `6f38fc8f5ee4f5b213981d7a85b46a3b` | 过滤/join/覆盖/回读校验全登记 |
| 训练快照 | `in/A_CLEAN_TRAIN_SNAPSHOT.jsonl` | `4127d29315ddb7000d70819bc353d725` | 1188 行（契约仍是四条件过滤器） |
| 批序 | `out/batch_idx_hybrid.txt` | `42d8a1a502aaa2ce7789023c7080f2ed` | md5(f3afc2e4…)，held 命中 0 |
| 曲线 | `out/HYBRID_CURVE.jsonl` | `72efc838de3b5429dc21bd76bfd83f1e` | 100 步三损失 + 覆盖列 |
| 冒烟汇总 | `out/HYBRID_SMOKE.json` | `d092e875735ca0579a6f8cf887a8f6c1` | 三曲线齐 + 样例 5 条 + 五列表 |
| 冒烟 ckpt | `ckpt/hybrid_smoke100.pt` | `2def2238c9a2a50f938dc31587ef7552` | **冒烟件（命名含 smoke，禁止作正式产物）** |
| 单测结果 | `out/GEN26_SELFTEST.json` | `762f5d35938aa1e6a35b6693dc72b9b1` | 28/28 PASS，rc=0 |
| 五列表（7 份） | `out/GEN26_EVAL_*.json` | 见台账 | hybrid(a_held bridge/gold, exam_v1, exam_zhenti) · naked(a_held) · rule(a_held, exam_v1) |

---

## 2. 数据接口（三条件齐备 + 100% join）

**四条件过滤器**（契约；`filter_corpus`）：C1 `chain_ops` 非空 ∧ C2 `proof_text` 非空 ∧ C3 `split=="train"` ∧ C4 非考题。
**下游硬门**（显式登记）：`prescreen.certified` ∧ `eligible_train` ∧ `layer!="L2"` ⇒ 训练流；`split=="synth_held"` ⇒ 体温计段。

实测（语料 md5 `15d58dffecb4b272afe9351e34c9e935`，EXAM roster md5 `35e0632496a0720e1347fcf7998ae92f`）：

| 阶段 | 行数 |
|---|---|
| 语料总行 | 4842 |
| 过四条件 | 2119 |
| → 训练流（下游硬门后） | **1188**（ind 665 / ineq 289 / nt 234） |
| → 体温计段 | **236** |
| 剔除登记 | not_eligible 931 · exam_excluded 1083 · exam_disclosure 1184 · l2_neg 220 |

**链侧金标 = 精确 join**（`Program==chain_ops ∧ stmt==stmt_state` 逐字符串）：**train 1188/1188 · held 236/236 · miss 0 · ambiguous 0**。前两会话实测两次（语料 18:50 与 19:07 两版），均 100% 命中 ⇒ **接口对 gen23 重切免疫**（契约是过滤器，不是快照）。
⚠ 陷阱已加硬断言：同 `name` 多行（全链行 + 前缀行）；按 name 取行会得到"单算子前缀行"= 静默错误金标。

**缓存回读校验**（roundtrip_ok=True，7 项全过）：`labels == feat_pos+1` · `bsl 长 = labels×24` · `labels < head` · `bsl < n_vals` · hdr 常量 (168/238/154) · ids 逐位一致。
**缓存读数**：L(max/p50/mean) = 93/88/76.8；labels_max = 4（2–3 算子 + END）；槽监督率 0.0788；UNK 目标占比 0.0536。

**三损失数据面**（每行）：链段 L≈88 + 渲染段 max 218（均值见侧车 168）⇒ 成本 ≈ 1.5–1.6× 纯 SFT（风险 R5 已登记）。

---

## 3. 100 步联合冒烟读数（重跑版，CRIT-1 修正后）

```
CUDA_VISIBLE_DEVICES=0 python scratch/ccp/gen26/gen26_hybrid.py --smoke 100 --bs 6 \
  --out scratch/ccp/gen26/ckpt/hybrid_smoke100.pt --args-mode bridge
GPU GATE dev=cuda:0 used=223MiB free=12059MiB (>4500MiB 通过)   wall=48.5s
```

| 项 | 读数 |
|---|---|
| 起点 → 末点 **总损失** | 14.6464 → **0.1272**（min 0.1168） |
| **链损失**（step CE） | 9.4307 → **0.0043** |
| **槽损失**（slot CE, /24 加权） | 3.9859 → **1.4031**（min 0.8393） |
| **渲染损失**（text CE, 错位修正后） | 5.0496 → **0.0645** |
| 三曲线齐 | `has_three_curves = true`；100/100 步全有 chain/slot/text 三列 |
| 批 | bs=6（ind2+ineq2+nt2 每步必现）；微批 100；L max 289–299 |
| 覆盖（MED-6） | 终值 seen = {ind 200, ineq 200, nt 200} / 池 {665, 289, 234} |
| 留出评测（MED-9 轮换，48 行） | **链 sel_top1 = 1.000**（144 帧）· 槽 top1 = 0.758（nonunk 0.761，244 槽）· **渲染 token acc = 0.785**（1246 token） |
| nan / 停机 | `stopped_nan = false`，无早停，100/100 步完成 |
| 批序 | md5 `f3afc2e4dea5d9301626037dbfd95469`，**held 命中 0** |

**5 条样例（留出段 a_held，`--args-mode bridge`）**：链**逐条精确命中**金标（首算子 5/5、整链 5/5、5/5 以 END 正常终止），链-文一致性 5/5 = 1.0，渲染 token 均值 47.4。样例 0（`ccp_ind_lin_000004`）渲染全文：

```
```lean4
intro n
induction n with
| zero => simp [h0]
| succ n ih =>
  rw [hstep, ih]
  push_cast
  ring
```
```

> **读法纪律**：本冒烟是**机械通路验证**（100 步 / 语料子集 / 2–3 帧链、6 算子空间），**不得**读作能力结论；链头数字偏高的机制解释 = 算子空间仅 6 个 + 起点头已由 v31f 预热 + 载荷帧含金标 res（代理，见 R2）。

---

## 4. 三臂 × 五列表（同语料 / 同考场 / 同判卷口径）

五列 = `certified | first_hit | exact_hit | slot_col | consistency`（`table_of`）。`certified` 需 `--judge 1`（冻结判卷链 + Lean；冒烟默认关，故本轮为 `null`，**不作结论**）。`slot_col` = 金标一致帧中 `args_src=bridge` 占比（真槽 top-1 在留出段由 `eval_hybrid` 出：0.758）。

| 臂 | 段 | n | certified | first_hit | exact_hit | slot_col | consistency | op 越界 | 终止 END |
|---|---|---|---|---|---|---|---|---|---|
| **hybrid**（bridge） | a_held | 5 | null | **1.00** | **1.00** | **0.50** | **1.00** | 0 | 5/5 |
| hybrid（gold，诊断） | a_held | 5 | null | 1.00 | 1.00 | 0.00（定义：全代理） | 1.00 | 0 | 5/5 |
| hybrid（bridge） | exam_v1 | 5 | null | 不可评 | 不可评 | 不可评 | 1.00 | **5** | 5/5 |
| hybrid（bridge） | exam_zhenti | 3 | null | 不可评 | 不可评 | 不可评 | 0.22 | 3 | 3/3 |
| **naked**（零训练底座） | a_held | 5 | null | — | — | — | 0.00 | 0 | — |
| **rule**（装置规则臂） | exam_v1 | 5 | null | — | — | — | 0.00 | 0 | — |
| rule | a_held | 5 | null | — | — | — | — | — | `n_err=5`（no_chain：规则臂库不覆盖 gen12 synth 名，**如实登记**） |

要点：
- `--args-mode` 开关有效且可辨：bridge 行 `slot_col=0.5`（2 帧链的第 1 帧由题面桥导参），gold 行 0.0（全代理）⇒ **HIGH-2 闭环**（正式评估缺省 bridge）。
- naked 臂渲染 160 token 的重复符号串（未训练底座的典型行为），一致性 0.0；rule 臂在 exam_v1 上产出 25 token 装置战术行（`n_err=0`）⇒ 三臂接口全部连通，**certified 列待 Lean 判卷**（须按并发≤5 nice19 排程）。
- hybrid 对 exam_v1 渲染 47 token 的 ind 型证明（对 bh 家族题**不适用**）——与 §5 算子空间阻塞同源，是"已训空间太小"的直接表现，不作能力结论。

---

## 5. 【阻塞】HIGH-4 实测：训练算子空间 ∩ 考卷算子空间 = ∅

考场入口已接通（`--seg exam_v1` / `exam_zhenti`，侧车 join miss 0），但实测：

| 集合 | 规模 | 内容 |
|---|---|---|
| 训练流算子空间（四条件后） | **6** | ccp_ind_recurrence · ccp_ring_identity · ccp_ineq_sos · ccp_ineq_close · ccp_nt_dvd_witness · ccp_nt_dvd_combine |
| EXAM_V1 主判据集算子空间 | **11** | bh_ineq_block · bh_ind_geo · bh_ind_close_const · bh_ineq_tail_square · bh_ineq_tail_other · bh_ind_sum_d3 · bh_ind_close_sq4 · bh_ind_sum_d1 · bh_ind_close_sq2 · bh_ineq_tail_zero · bh_ineq_cross |
| 交集 | **0** | `bh_*` 既不在 V32 161 索引空间，也不在 CCP 7 扩展空间（`v32_op_index_map()` 查无此名） |
| **可评链导航的考卷项** | **0 / 1083** | `scorable=False` 逐行标注 |

⇒ **链导航列与槽列在考卷上不可评**；`certified` 与一致性列不受阻塞（渲染走 LM 头，prompt 由 `head ⊢ concl` 构造）。
**解除条件（正式训练前置）**：算子空间扩展批 —— 四处同步（`step_head` 行 + `allow` 行 + `lat_tab` 行 + **训练行含该算子**）后重训。**在此之前，M1_A 的链导航列与槽列无法产出，certified 列可产出**。

---

## 6. gen27 修正令逐条闭环

| 编号 | 项 | 落地点 | 证据 |
|---|---|---|---|
| **CRIT-1** | 渲染段损失 off-by-one | `gen26_hybrid.render_batch` 错位修正 | §0 五项 MISALIGN 单测 + 冒烟重跑 |
| **HIGH-2** | 评估链金标代理 | `--args-mode {bridge,gold}`，正式缺省 bridge；`trace.args_src` 逐帧登记 + `bridge_fallback` 计数 + 桥失败原因披露 | 五列表 bridge/gold 两行 slot_col 0.50 vs 0.00 |
| **HIGH-3** | 设计书缺失 | `HYBRID_DESIGN.md` 落盘（六节，含预注册选型与风险表 R1–R8），与代码注释 § 引用逐条对应 | md5 `6fded822…` |
| **HIGH-4** | 考试入口断链 | `--seg exam_v1`（EXAM_V1 主判据集 1083）+ `exam_zhenti`（真题 3）走同一冻结判卷件；`exam_sidecar` 产出侧车（miss 0）；真题 head/concl 直连 | `A_EXAM_SIDECAR.jsonl` + 两张 EVAL 表 |
| **HIGH-5** | 对照臂与命名 | `--arm {hybrid,naked,rule}`（naked=零训练底座直出 / rule=装置规则臂）；产物命名硬闸（冒烟名必须含 `smoke`，正式名不得含，须 `hybrid_formal_*.pt`） | 7 份 EVAL 表 + 代码 `SystemExit` 硬闸 |
| MED-6 | 采样器覆盖偏差 | 登记 + 曲线列 `seen{}`/`batch_cls_counts{}` + summary `coverage_med6` | seen 终值 200/200/200 vs 池 665/289/234 |
| MED-7 | NVALS 239 死字段 | `gen26_eval.NVALS`/`MiniCache.NVALS`/allow reshape 统一为 **238**（`CCP_VOCAB.n_vals_ext`） | 单测 T5d |
| MED-8 | 走链上界口径 | `WALK_BUDGET = 7` 预注册常量（非按行金标）；超界记 `overrun` | 单测 T5a2 |
| MED-9 | held 固定 24 条 | `--eval-rows 48` + `--eval-rotate 1`（每评测点轮换切片，`eval_idx` 入曲线） | summary `eval_med9` |
| MED-11 | 1B 口径 | summary `track.dual_track_disclosure` 明示"1.7B 未接未跑，不作读数" | HYBRID_SMOKE.json |
| LOW-16 | 中间落盘 | 单测每个 check 后原子写盘（`.tmp`+`os.replace`），并设 `partial` 标志 | 上一轮超时实测已保留 25 项证据 |

---

## 7. 合规自证

- **写面**：仅 `scratch/ccp/gen26/**` + `scratch/ccp/HEARTBEAT.gen26` + `LAUNCH_LOG.md`。既有件**零改动**（gen13/gen23/gen24/gen12 全部 imports-only 或只读）。
- **判卷件冻结令**：`drct_common`（`7c31b58da9d7228cee31b9be581fe5dd`）本批仅 **import**，零改动；本批未跑 Lean 判卷（`--judge 0`），未占用 Lean 并发额度。
- **红线 R0/75**：主臂为潜空间架构（V32 链头 + 渲染），无直出 SFT 立项；对照臂为**零训练**（裸基 + 规则臂）。
- **GPU**：仅冒烟与评测（单卡独占门 free=12059MiB 通过，短时）；**零训练型长占**。
- **不扰在飞**：未读未写 gen23 工作区（只按四条件读语料行 + 登记 md5）；未触 gen24/gen25；`probes/` 只读；Test 243 未读未评未引用；无 `/tmp`、无 `sudo`。
- **在飞重切适配**：语料 md5 先后 `7bbfdfc9`（17:42，gen25 探针）→ `72a8ad07`（18:41 统计）→ **`15d58dff`（本批消费）**，三次均以四条件过滤器消费且 join 100% ⇒ 契约不依赖任一瞬间快照。

---

## 8. 遗留与下一步（正式训练前置清单）

1. **R1 算子空间扩展**（最高，阻塞链导航/槽列出分）：`bh_*` 11 算子入 step_head/allow/lat_tab + 训练行（需 gen24 扩料语料 + 相应课程行）→ 重跑 hybuild → 重训。
2. **R2 载荷代理替换**：非首步算子槽/结果帧接装置状态导参闭环（现为金标代理，占比可读）。
3. **判卷**：`certified` 列需在有 Lean 额度时按并发≤5 nice19 排程跑三臂（hybrid/naked/rule × a_held/exam_v1 主卷 + 真题列）。
4. **R4 空链消融**：正式评估加"空链 vs 走链"两版渲染对照列（验证链-文耦合真实性）。
5. **正式产物命名**：`ckpt/hybrid_formal_<语料md5前8>_<步数>.pt`（代码硬闸已禁 `smoke` 字样入正式名）。
6. **1.7B 轨**：另行立项（本批未接）。

---

## 9. 附录（19:4x 只读观察，登记）：gen28 `A_CORPUS_V2` 对 R1 的影响

本批**只读**观测到并行批 gen28 已产出 `scratch/ccp/gen28/A_CORPUS_V2.jsonl`（9409 行，全 `split=train`）。按本批同一四条件口径统计其算子空间：

| 集合 | 规模 | 内容 |
|---|---|---|
| V2 训练流算子空间 | **12** | 6 个 `ccp_*`（不变）+ 6 个 `g24_*`（gen24 扩料新族：`g24_ind_ratio` · `g24_ind_pf` · `g24_ind_ratio_close` · `g24_ind_pf_close` · `g24_ineq_ht` · `g24_ineq_ht2`） |
| EXAM_V1 主判据集算子空间 | 11 | 11 个 `bh_*`（不变） |
| **交集** | **0** | `bh_*` 仍不在任何已注册算子空间内 |

⇒ **R1 未因 V2 解除**（训练空间 6→12 翻倍，但与考卷仍零交）。V2 的价值在**训练侧扩容**（行数 1188→9409、族 3→9），不改变"考卷算子缺行"的事实。正式训练前的算子空间扩展批仍需独立立项：需为 **11 个 `bh_*`** 建 head 行 / allow 行 / lat_tab 行，并让训练行覆盖这些算子（源 = gen14 bh 课程行或 gen24 新族行，二者与考卷族是否对齐须由主代裁定）。
（登记依据：本批仅读取该文件，未改动；其 md5 见 `GEN26_LEDGER.jsonl` 末行。）
