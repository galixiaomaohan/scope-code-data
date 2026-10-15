# HYBRID_DESIGN.md — A 主臂：潜空间混合架构（设计书 v1）

- **批次**: gen26（潜空间混合主臂设计与建造批）
- **法源**: 宪法变更 **74**（A 主臂回归潜空间思考架构；混合臂升主、直出臂降对照）· 变更 **75**（**红线 R0**：潜空间思考主线不可偏离；纯文本直出 SFT 禁作主臂；对照由裸基模型+规则臂零训练承担）· 变更 **49**（模型只路由、算子计算、Lean 终审）· 变更 **73**（M1_A 三栏读数）· §7 **发现 8**（能力面三病）
- **本批边界**: 设计 + 建造 + 100 步联合冒烟（135M 轨）。**正式训练不在本批**（等 gen23 泄漏修正版语料 + gen24 扩料入库，且需先解除下方 §5 R1 阻塞项）。
- **代码对应**: `gen26_hybuild.py`（接口适配/缓存+侧车）· `gen26_hybrid.py`（模型+联合训练）· `gen26_eval.py`（考试接口/三臂）· `gen26_selftest.py`（单测）。
- **审后修正**: 本节所有条目已按 gen27 审核批修正令（1 CRIT + 4 HIGH + 6 MED + 5 LOW）逐条落地，落地点写在各条「代码位置」中。

---

## 1. 模型：单体双头（V32 协议链头 + LM 渲染头）

### 1.1 体

| 项 | 135M 轨（本批实跑） | 1.7B 轨 |
|---|---|---|
| 底座 | `models/SmolLM2-135M`（HF） | `models/SmolLM2-1.7B` |
| 体精度 | fp32 | bf16（`cast_body`） |
| 头输入维 | 继承层原样（HID=576, DMODEL=1152） | 按 config 重建（HID=2048 ⇒ DMODEL=4096） |
| 起点 | `ckpt/sla_v31f.pt`（v31 步进头 155 行 + 值头 151 列**逐位**搬入；CCP 7 行 ← 父行；新列 87） | `--init none` ⇒ `init_fresh_heads`（头无预热可搬） |
| 状态 | **已接已跑**（冒烟） | **未接未跑**（代码路径存在 = `CCPModel(dyn=True)`；本批不作任何 1.7B 读数） |

体与头同体的三个通道（**这是"混合"的机制要点**）：
1. **帧通道**：题面 token 帧与渲染段 token 帧都走 `lm.model.embed_tokens`（同一嵌入矩阵）；
2. **协议通道**：算子帧走 `P_op(lat_tab[op])`、结果帧走 `P_res(res)`、状态帧走 `P_st(onehot)`、槽位走 `slot_emb`；
3. **输出通道**：链头（`step_head`/`value_head`）与 LM 头（`lm.lm_head`）读同一隐态序列的不同位置。

> **登记（tie）**：SmolLM2 的 `tie_word_embeddings=True` ⇒ `lm.lm_head.weight` 与 `lm.model.embed_tokens.weight` 是**同一张量**（单测 T3b 已证 tied=True）。后果有二：(a) 渲染损失同时更新输入嵌入，链段 token 帧的嵌入也被带动；(b) 这也正是 **CRIT-1** 必须修的原因——不修则"抄输入"是零代价捷径。

### 1.2 头（**沿用 gen13 尺寸与语义，逐位不改**）

| 头 | 尺寸 | 语义 | 来源（imports-only） |
|---|---|---|---|
| `step_head` | `Linear(2·HID → 168)` | 算子空间 = V32 161 索引（0..160，154=END）+ CCP 7（161..167） | `gen13/CCP_VOCAB.json` `head=168` |
| `value_head` | `Linear(2·HID+DMODEL → 238)` | 值空间 = V32 151 值 + CCP 追加 **87** ⇒ 238 | `CCP_VOCAB.n_vals_ext=238` |
| `slot_emb` | `Embedding(24, DMODEL)` | 24 槽（NSLOT）逐槽预测 | `v32lib.NSLOT` |
| `codebook.lat_tab` | `(168, 64)` | 潜态表：前 155 行搬 v31，CCP 7 行 ← 父行 | `lat_tab_v31.bin` + `init_map` |
| allow 表 | `(168, 24, 238) uint8` | 逐算子逐槽合法值掩码 | `gen13/ccp_allow_ext.bin` |
| `lm.lm_head` | `(vocab, HID)` | 渲染证明全文（tied） | HF 底座自带 |

> **MED-7 清死字段**：代码注释与历史报告里出现的"239 列"是**死值**；运行时真值以 `CCP_VOCAB.n_vals_ext = 238` 为准（`gen26_eval.NVALS`、`MiniCache.NVALS`、`allow` reshape 三处已统一为 238，单测 T5d 断言）。

### 1.3 单序列布局（帧级）

```
链段 (gen13 帧布局逐位不变):
  [K_EMB M_TASK] [K_EMB stmt_token]*  [K_EMB M_TYPE] [K_WIN type4]
  for k in 0..K-1:  [K_EMB M_OP] [K_OP op_k] [K_EMB M_RES] [K_RES res_k | K_PSZ] [K_EMB M_TAGFR+k] [K_PST state_k]
  [K_EMB M_END] [K_PST end]
渲染段 (本批新增; 复用 M_TASK 文本流语义, 不引入新帧 kind):
  [K_EMB M_TASK] [K_EMB render_prompt_token]* [K_EMB proof_text_token]*
```
- 链段决策点：`bc.idxH = pos_task_end`（题面池化位）、`bc.idxC = 上一步帧`；`heads()` 在 `H[:, :Lc, :]` 上算——**切片即等价**（位置编码与掩码在整段上算，因果性保证链段不受渲染段影响）。
- 渲染段计损：**仅 target 段**（`[M_TASK]+prompt` 位置 = −100）。
- 掩码：`lenB_full = 链长 + 渲染真实长`（含填充位屏蔽）；位置编码按**绝对位置**续接（链段末位之后继续编号，登记为设计选择）。
- `prompt_mask`/`prompt_span` 只认**第一处** `(M_TYPE, K_WIN)` ⇒ 渲染段的 M_TASK 不会污染题面池化位（已核 v31f_model.prompt_span 语义）。

---

## 2. 训练：联合目标（**预注册**）

### 2.1 目标式（预注册，不改）

```
L = L_step  +  (lam_bind / NSLOT) · L_slot  +  lam_txt · L_text
    └ 链损失（逐帧算子 CE，治发现 8 病二）   └ 槽损失（逐槽 CE，治病三）   └ 渲染损失（target token CE）
```
- `L_step`/`L_slot` **逐式沿用 gen13** `v32ccp_train.CCPModel.loss_f`（未改一字；含 UNK 掩码与 allow 掩码语义）。
- `L_text` = 渲染段 target 段 CE，**自回归错位**：位置 p 的标签 = `ids[p+1]`（CRIT-1 修正；见 §2.4）。
- 预注册默认：`lam_bind = 1.0`、`lam_txt = 1.0`、`NSLOT = 24`。**均权**（不引入未登记的加权搜索）。
- 诊断权重口径：曲线三列（chain/slot/text）分别为**按有效位置数加权**的微批均值（`nv`/`nbf`/`ntxt`），总损失列同口径。

### 2.2 两种方案与**预注册选型**

| 方案 | 做法 | 优点 | 缺点 | 判据 |
|---|---|---|---|---|
| **A 联合单阶段（选定）** | 一条序列 + 一次前向，链损失与渲染损失同步反传 | ① 共享体只跑一次前向（成本 ≈ 1 次，非 2 次）；② 链状态与渲染条件**同域**（渲染段 attend 的正是当步链隐态，无表征漂移）；③ 单一超参面，预注册简单 | 双头梯度在共享体上叠加 ⇒ 潜在干扰（风险 R3） | 冒烟：三损失同时下降且无 nan；链指标不低于 gen13 同起点读数（若低于，见 R3 处置） |
| B 两阶段（先链后文） | 先只训链头（体+链头），冻链头后只训渲染 | 隔离干净、各自可早停 | ① 前向两次（成本 ≈ 2×）；② 渲染段看不到"当步链隐态"，只能吃冻结链的只读投影 ⇒ 链-文脱节风险**更**大；③ 两套学习率/调度需各自登记 | 仅当 A 出现"链升文降"或"文升链降"**双向不稳**时启用，且须另开变更行 |

**选型结论（预注册）**：采 **A 联合单阶段**。判据三条：① 成本（本批目标是用最小成本验证联合可训）；② 机制（渲染必须**条件于当步链隐态**，两阶段会切断该条件，等于把"潜空间思考→渲染"退化为"先算一个潜态摘要再独立写文"，与变更 49/74 的接口语义不符）；③ 可判性不受影响（判卷只看渲染文本，链只留痕）。

### 2.3 数据接口（消费过滤器 = 契约，非文件快照）

**四条件过滤器**（变更 74 登记口径；唯一入口，代码 `gen26_hybuild.filter_corpus`）：

| 条件 | 判据 |
|---|---|
| C1 | 行含 `chain_ops`（非空金标算子序） |
| C2 | 行含 `proof_text`（非空参考证明全文） |
| C3 | `split == "train"` |
| C4 | 非考题：`name ∉ EXAM roster` ∧ `split ∉ {exam_excluded, exam_disclosure}` |

**下游硬门**（显式登记，不静默；与 `gen23_sft.load_corpus` 同口径）：`prescreen.certified` ∧ `eligible_train` ∧ `layer != "L2"` ⇒ 训练流；`split == "synth_held"` ⇒ **体温计段**（非训练）。

**实测（19:07 快照，语料 md5 = `15d58dffecb4b272afe9351e34c9e935`）**：4842 行 → 四条件 2119 → 训练流 **1188**（ind 665 / ineq 289 / nt 234；链长 2: 954、3: 234）；体温计 **236**。

**链侧金标 = 精确 join**（代码 `exact_join`，硬断言）：
```
A 语料行.chain_ops == gen12 全链行.program  ∧  A 语料行.stmt_state == gen12 全链行.stmt   （逐字符串）
```
- 实测命中 **1188/1188 + 236/236，miss 0，ambiguous 0**。
- ⚠ **陷阱登记**：gen12 行文件里同一 `name` 有多行（**全链行 + 各级前缀行**，比例约 3:1）。按 name 取首/末行会拿到"单算子前缀行" ⇒ **静默错误金标**。已加断言：`[step.op for step in steps] == chain_ops` 且 `len(steps)==len(sf_args)==len(chain_ops)`。

**渲染段侧车**（`A_RENDER_SIDECAR.jsonl`）：`ids = [M_TASK] + tok(prompt) + tok(target)`，`np = 1+len(prompt)`；附链侧元数据（`stmt/head/concl/type4/ops/sfa/res`）使 eval 与判卷**自足**（不必回读任何在飞批次文件）。

**语料版本纪律**：契约是四条件过滤器；每次消费登记（a）语料 md5、（b）EXAM roster md5、（c）本批一次性快照 md5。gen23 重切语料 ⇒ 重跑 `gen26_hybuild.py` 即得新缓存与新 md5（不冻结旧快照作依赖）。

### 2.4 CRIT-1（渲染段损失错位）——修正与敏感性证据

- **病灶（审核批坐实）**：原式 `rtgt[np:len] = ids[np:len]` ⇒ 位置 p 的标签 = 位置 p 的**输入**（自预测），tie 下成为零代价捷径；且位置 `np-1`（唯一能预测首个生成 token 的位置）恒为 −100 ⇒ 推理首步无监督。
- **修法**：`rtgt[np-1 : len-1] = ids[np : len]`（位置 p 标签 = `ids[p+1]`），见 `gen26_hybrid.render_batch`。
- **敏感性单测**（`gen26_selftest.py`，报告首位）：MISALIGN1 标签逐位等于 `ids[np]`；MISALIGN2 修正后 `label==input` 位置数 = 0/N；MISALIGN3 旧式目标与正确目标逐位不同；**MISALIGN4**（模型无关判据）用"抄输入头"（one-hot(input)×40）算 CE：旧式目标 CE ≈ 0（平凡捷径**证明存在**），正确错位目标 CE ≫ 1（捷径**失效**）；MISALIGN5 起点模型观测列（两目标互不等价）。

### 2.5 采样与批序

- **交错混喂**（发现 8 病一药方）：每步批内三类必现（`3 类 × bs/3 = 2 题`），类内无放回洗牌，逐 epoch 重洗。实现于 gen26（口径 = `gen23_sft.interleaved_batches`，**新写**实现以避免对在飞件瞬时状态强依赖；provenance 已登记）。
- **MED-6 覆盖偏差（登记 + 列）**：三类池不等（665/289/234）⇒ 每 epoch 批数受最小类截断，单 epoch 只覆盖各类 `per × 批数` 行（bs=6 时 per=2、117 批 ⇒ 各类 234 行/epoch，占总池 1188 的 39.3%）。跨 epoch 重洗逐步扩大覆盖：曲线每步写 `seen{cls}` 与批内类计数 `batch_cls_counts`，summary 写 `coverage_med6`（终值 seen/pool）。**不修采样器**（改动会动发现 8 药方口径），以登记+列披露。
- **批序预注册**：`build_plan` 生成，落盘 `out/batch_idx_hybrid.txt`（首行 `步数 bs`）+ md5 写 summary；硬断言 0 留出行命中。
- **MED-9 留出读数**：默认 `--eval-rows 48`（池 236），`--eval-rotate 1` ⇒ 每个评测点轮换切片（切片索引写进曲线 `eval_idx`），避免固定 24 条的温度计偏差。

---

## 3. 考试接口（两段式）

### 3.1 段一：潜空间思考链（**留痕，不判卷**）

```
for k in 0..BUDGET-1:
    前缀批 = build(帧前缀(已走算子序))            # 帧构造复用 gen13 ccp_cachefull.build
    logits = step_head(H[idx 决策点])             # 合法集掩码 = CCP 7 ∪ {END}
    op_k   = argmax(logits)                      # 模型自路由
    if op_k == END: 停
    载荷 (res/pst 帧) = 装置侧给出                  # 见下"载荷来源"
    追加帧; 记留痕: {k, op, 金标算子, 命中, p_max, margin, args_src}
```
- **MED-8 上界**：`WALK_BUDGET = 7` = **预注册常量**（2×最长金标链 3 + 1），**非按行金标**；超界记 `overrun`，不因金标长度提前终止（自由走链真跑）。
- **载荷来源（HIGH-2）**：
  - `--args-mode bridge`（**正式评估缺省**）：首算子参数由 `ccp_bridge` **从题面导参**（变更 49 装置导参闭环）；后续算子的槽属**装置状态**导出，本批以金标 args 作代理 ⇒ `trace.args_src` 逐帧登记 `bridge` / `device_state_gold_proxy`，`bridge_fallback` 计数。
  - `--args-mode gold`：全金标载荷，**仅诊断列披露**，不作正式读数。
- 归因列（对拍金标）：`first_hit`（首算子）、`prefix_hit`（共前缀）、`exact_hit`（整链）、`terminated_end`、`scorable`（金标算子是否在已训算子空间内）。
- **留痕证据**：`row["chain"]["trace"]` 逐帧落 JSON（不判卷）。

### 3.2 段二：渲染 → 冻结判卷链（**判卷**）

```
渲染段 = 链帧(走到的链; 潜空间, 不外显文本) + [M_TASK] + prompt → LM 头贪心解码
判卷   = drct_common + gen23_prescreen_batch.judge_items_batch(items={name,head,concl,proof_text})
         （冻结算件, 7c31b58d; 与 GT 侧同门; 冒烟默认关, 因 Lean 并发纪律）
```

### 3.3 链-文一致性诊断列（抽检，非判据）

登记表 `CONSISTENCY_KEYWORDS`：算子 → 渲染文本应出现的战术关键词（如 `ccp_ind_recurrence → {induction, hstep, ih}`；`ccp_ring_identity → {ring, push_cast, norm_num, linarith}`；`ccp_nt_dvd_witness → {dvd, ⟨, obtain, use }`）。逐算子命中率 = 一致性列；仅作**诊断披露**，不进 M1 判据。

### 3.4 段与臂（CLI）

| 开关 | 取值 | 说明 |
|---|---|---|
| `--seg` | `a_held`（体温计）/ `a_train` / `exam_v1`（EXAM_V1 主判据集 1083 项）/ `exam_zhenti`（真题 3 道参考列） | **HIGH-4**：考场入口不再只有体温计段 |
| `--arm` | `hybrid` / `naked`（零训练底座直出）/ `rule`（装置规则臂自动答题） | **HIGH-5**：三臂同卷同门 |
| `--args-mode` | `bridge`（缺省）/ `gold`（诊断） | **HIGH-2** |
| `--judge` | 0/1 | 1 = 走冻结判卷链（Lean；须守并发≤5 nice19） |

### 3.5 ⚠ **算子空间覆盖（HIGH-4 实测阻塞项，正式训练前置条件）**

| 集合 | 内容 | 规模 |
|---|---|---|
| 训练流算子空间（四条件过滤后） | `ccp_ind_recurrence` · `ccp_ring_identity` · `ccp_ineq_sos` · `ccp_ineq_close` · `ccp_nt_dvd_witness` · `ccp_nt_dvd_combine` | **6** |
| EXAM_V1 主判据集算子空间 | `bh_ineq_block` · `bh_ind_geo` · `bh_ind_close_const` · `bh_ineq_tail_square` · `bh_ineq_tail_other` · `bh_ind_sum_d3` · `bh_ind_close_sq4` · `bh_ind_sum_d1` · `bh_ind_close_sq2` · `bh_ineq_tail_zero` · `bh_ineq_cross` | **11** |
| **交集** | ∅ | **0** |
| 可评链导航的考卷项 | — | **0 / 1083** |

⇒ **链导航列在考卷上不可评**（不是"模型差"，是"头里没有这些算子"）。`bh_*` 既不在 V32 161 索引空间、也不在 CCP 7 扩展空间（`v32_op_index_map()` 查无此名）。**解除条件**：算子空间扩展批（四处同步：`step_head` 行 + `allow` 行 + `lat_tab` 行 + 训练行含该算子），随后重训。渲染列与 certified 列**不受此阻塞**（渲染走 LM 头，prompt 由 `head ⊢ concl` 构造）。
考试侧车已把主判据集 1083 项 + 真题 3 项全部接入（`exam_sidecar` 产出 `A_EXAM_SIDECAR.jsonl`，join miss 0），`scorable=False` 逐行标注。

---

## 4. 对照设计（同语料 / 同考场 / 同判卷）

**变更 75 后**：gen25 直出 SFT 对照臂**撤编**（LAUNCH_LOG 已记 CANCELLED），对照需求由**零训练臂**承担（确需训练型对照须用户明示批准）。

| 臂 | 定义 | 训练量 |
|---|---|---|
| **hybrid**（主臂） | 潜空间链头 + LM 渲染头（本设计） | 有（gen26 建造，正式训练待解除 R1） |
| **naked** | 裸基模型零训练直出（同 prompt、贪心） | 0 |
| **rule** | 装置规则臂自动答题（gen23 `bh_rule` 程序 → 装置执行取回战术行） | 0 |

**臂间对比表口径（五列）** — 代码 `gen26_eval.table_of`：

| # | 列 | 定义 | 备注 |
|---|---|---|---|
| 1 | `certified` | 冻结判卷链 certified 计数（模型侧文本过 N1/N2 同门） | 主判据列；0 vs 0 禁读作能力结论（EXAM_V1 criterion） |
| 2 | `first_hit` | 首算子命中率（对拍金标 `chain_ops[0]`） | 仅 `scorable` 行计入 |
| 3 | `exact_hit` | 整链命中率（算子序逐位相等） | 同时报 `prefix_hit`（共前缀） |
| 4 | `slot_col` | 槽列（代理口径）：金标一致帧中 `args_src=bridge` 占比 | 真槽准确率在留出段由 `eval_hybrid` 出（gen13 口径）；考卷段因 R1 不可评 ⇒ `None` + 披露 |
| 5 | `consistency` | 渲染一致性（算子→战术关键词命中率） | 诊断列 |

另附披露列：`op_out_of_space`（金标算子越界项数）、`terminated_end`、`render_tok_mean`、`n_err`、`args_mode`（必须随表登记，防 gold/bridge 混读）。

---

## 5. 风险登记

| # | 风险 | 现状/证据 | 处置（预注册） |
|---|---|---|---|
| **R1** | **算子空间不交（最高）**：训练 6 算子 vs 考卷 11 算子，交集 ∅ ⇒ 链导航与槽列在考卷不可评 | 本批实测 0/1083 可评（§3.5） | 正式训练**前置**：算子空间扩展批（step_head/allow/lat_tab/训练行四处同步）→ 重训。未解除前，考卷只出 certified 与一致性列 |
| **R2** | **载荷代理**：非首步算子的槽/结果帧用金标代理（装置状态导参未闭环） | `trace.args_src` 逐帧登记；`device_state_gold_proxy` 占比可读 | 接装置状态导参（`bh_ops`/`ccp_ops` 逐步真算）替换代理；替换前后读数分别登记 |
| **R3** | **双头干扰**：链损失与渲染损失在共享体上叠加，可能链升文降/反之 | 冒烟三曲线同报；单测 T3b 证三组参数梯度均非零 | 观测曲线三列相关性；若出现系统性互斥 ⇒ 切方案 B（§2.2，须新变更行） |
| **R4** | **链-文脱节**：渲染不看链（把链当无信息前缀） | 一致性列 + 消融（同 prompt、空链 vs 走链，两版渲染对比） | 冒烟先出空链样本对照；正式评估加"空链消融"列 |
| **R5** | **训练成本倍增**：链帧 + 渲染段同序列 ⇒ 序列变长 | 实测：链 L(max/p50/mean) = 93/88/76.8，渲染 token(max/mean) = 218/168 ⇒ 每行序列 ≈ 链长 + 渲染长 | 微批按 `B×L` 切（`--micro-token-cap`）；正式训练规模与成本另算入批次预算 |
| **R6** | **与 gen23 重切语料的接口适配** | 契约 = 四条件过滤器 + md5 登记（已实测两次重切均命中 100%） | 每次消费重跑 hybuild；禁用"冻结快照作依赖" |
| **R7** | **1.7B 轨未接**（MED-11） | 本批只跑 135M；1.7B 代码路径存在但未跑 | 任何 1.7B 读数须另开批次并显式登记；本批 summary 内 `dual_track_disclosure` 已如实写"未接" |
| **R8** | **死字段/命名**（MED-7/HIGH-5） | 239 死值已清；冒烟/正式产物名已隔离（`hybrid_smoke100.pt` vs `hybrid_formal_*.pt`，代码硬闸） | 保留硬闸：冒烟名必须含 `smoke`，正式名不得含 |

---

## 6. 复现入口

```bash
R=/home/data/ZHC/tcaboost/research; P=/home/data/ZHC/tcaboost/venv/bin/python
export PYTHONPATH=/home/data/ZHC/tcaboost/superoperator; cd $R

# 1) 接口适配 (四条件过滤 -> 链缓存 + 渲染侧车 + 考场侧车 + 算子空间覆盖)
CUDA_VISIBLE_DEVICES="" $P scratch/ccp/gen26/gen26_hybuild.py

# 2) 单元自测 (CRIT-1 错位敏感性 + 契约 + 联合前向 + 走链/渲染)
CUDA_VISIBLE_DEVICES=0 $P scratch/ccp/gen26/gen26_selftest.py

# 3) 100 步联合冒烟
CUDA_VISIBLE_DEVICES=0 $P scratch/ccp/gen26/gen26_hybrid.py --smoke 100 --bs 6 \
   --out scratch/ccp/gen26/ckpt/hybrid_smoke100.pt --args-mode bridge

# 4) 三臂评测 (五列表)
$P scratch/ccp/gen26/gen26_eval.py --arm hybrid --seg a_held --ckpt <ckpt> --args-mode bridge
$P scratch/ccp/gen26/gen26_eval.py --arm hybrid --seg exam_v1 --args-mode bridge
$P scratch/ccp/gen26/gen26_eval.py --arm naked --seg a_held
$P scratch/ccp/gen26/gen26_eval.py --arm rule  --seg a_held
```
