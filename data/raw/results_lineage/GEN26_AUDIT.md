# GEN26_AUDIT.md — gen26（潜空间混合主臂批）完工**对抗审核**

- **审核员**: gen26r（独立 · 只读 · 对抗性 —— 任务是推翻 gen26 结论，推翻不了才算通过）
- **审核时刻**: 2026-09-23T19:3x–19:5x+0800
- **被审件**（开工记全量 md5，收工复算自证零改动）: GEN26_REPORT.md `a6d32ff3123221bbbe1ec775a202edd1` · HYBRID_DESIGN.md `6fded8224c6bd870a61773c93e742456` · gen26_hybrid.py `12e14ca8209373ff161d24d860e9eef0` · GEN26_LEDGER.jsonl `a4ce8c1b12905e6e862abc0937e6cd9c` · GEN26_REPORT_ADDENDUM.md `19b7a5874f533af8da6a6a2f70961103` · gen26_hybuild.py `563ef0a67c171a0b0b68406be325d406` · gen26_eval.py `6f35a85eaa2b7c784e5583babddfbca8` · gen26_selftest.py `63d97fa2f9a2f6dec3233a320a5d93c3` · cache/hyb_a_all.bin `55dc37bb1076ef21e48c80863fea8756` · A_RENDER_SIDECAR.jsonl `ff90b412de72a7c1e005963bb704a572` · HYBRID_CURVE.jsonl `72efc838de3b5429dc21bd76bfd83f1e` · ckpt/hybrid_smoke100.pt `2def2238c9a2a50f938dc31587ef7552`
- **零改动自证**: 全目录 38 件聚合 md5 开工 = `146fc65011165c2093af429c190b384d`，收工复算 = **同值**（audit_r/ 新件除外；__pycache__ 亦无新 .pyc，全程 PYTHONDONTWRITEBYTECODE=1）。
- **铁律**: 零 GPU 训练（仅 CPU 前向/单测重放，--dev cpu）· 无 sudo · 无 /tmp · Test 243 未读未评 · drct_common.py md5 复算 = `7c31b58da9d7228cee31b9be581fe5dd`（**未变**）· 产出仅 scratch/ccp/gen26/audit_r/** + 本件 + LAUNCH_LOG append。

## 总裁决：**有条件通过**（GEN26_AUDIT_RC=0）

**推翻尝试的结果**：CRIT-1 的修复位、错位不变量（1424/1424 行）、接口四条件/精确 join、R1 算子空间不交、冒烟曲线读数 **全部经独立复算成立，未推翻**；28/28 单测在 CPU 上**整件重放成功（rc=0）**。但发现 **1 项 HIGH（读数解释缺口/未披露，见 AUD-1）** 与 5 项 MED 级证据口径缺陷（AUD-2/3/4/5/6），均不否定机制与接口，需以**勘误登记**闭环 ⇒ 有条件通过。

---

## 1. CRIT-1 修复位核（对抗点 1）— PASS（含 2 条口径注）

**亲读**（gen26_hybrid.py）：
```
169  # CRIT-1 修正 (gen27 审): 自回归错位 —— 位置 p 的标签必须是**下一个** token ids[p+1]
173  rtgt[b, np_ - 1:len(ids) - 1] = torch.tensor(ids[np_:], dtype=torch.long)
```
切片核算：[np-1, n-1) 长 = n-np；右值 ids[np:] 长 = n-np ⇒ 长度相容；位置 p 标签 = ids[p+1] ✓ 与设计书 §2.4「位置 p 标签 = ids[p+1]」逐字一致。

**全量不变量复算（独立实现，覆盖全部 1424 行 × 全部位置）**：label[p]==ids[p+1]（p∈[np-1,n-1)）∧ rtgt[n-1]==-100 ∧ rtgt[0:np-1]==-100 ⇒ **违规 0 行**。

**第二处残留搜索（全文件 + 全批）**：rtgt 在 gen26 全批只有 2 处写：① 训练路径 render_batch:163/173（唯一生产写）② 单测 gen26_selftest.py:176 的 bug 复现（测试专用）。gen26_eval.py **无任何 target 构造**（渲染是贪心生成，无 target）；留出读数 gen26_hybrid.eval_hybrid:576-577 用 tpk[r,p] == rtgt[r,p]（即「位置 p 预测 ids[p+1]」）⇒ **与修正后口径自洽，无第二种同位赋值**。**结论：无第二处残留。**

**tie 机理（如报告所述）**：models/SmolLM2-135M/config.json 实读 "tie_word_embeddings": true；单测 T3b 实跑打印 tied=True 且 lm.lm_head.weight 与 lm.model.embed_tokens.weight 梯度值**逐位相同**（8.831396 / 8.831396）⇒ 同一张量（非仅 config 声明）✓。故旧式 rtgt[p]=ids[p] 下「输出 = 输入 token」是同一参数化内可实现解 ⇒ 报告的捷径机理成立。
> 口径注①：严格说 tie 下抄输入是**极廉价**而非字面「零代价」（需隐态 H_p 与 e_{ids[p]} 对齐）。口径注②：MISALIGN4 的抄输入头是**构造的头**（one-hot×40），属「函数空间见证」，不是「训练后实测」——作为存在性/失效证明有效，但「零代价」措辞应从宽理解。

## 2. MISALIGN4 判据方法论独立性（对抗点 2）— 数字复现，判据成立；MISALIGN3 口径标签有误（AUD-4）

我**不用**被审件代码，独立按 CE = 40·(1-[label==input]) + log(1+(K-1)e^-40)（K=49152 ⇒ 尾项 2.1e-13）解析复算，并用真值张量核算：

| 判据 | 报告 | 我的独立复算 | 结论 |
|---|---|---|---|
| MISALIGN1 | pos=137 label=3725==ids[np] | 第 1188 行 np=138 ⇒ pos=137，label=ids[138]=3725 | ✓ 复现 |
| MISALIGN2 | label==input 0/48 | 0/48（该行 np-1=137，Lr=185 ⇒ 48 位）；**更强：全库 0/67448** | ✓ 复现并加强 |
| MISALIGN3 | 逐位差 237 | **selftest 重构造函数 vs 修正式 = 237**；**gen27 所引旧式（同位复制 [np,n)）vs 修正式 = 242** | ⚠ 口径标签不符 |
| MISALIGN4 | 旧 0.0000 / 新 40.0000 | 旧（同位复制，任一范围）CE=0.0000 精确；新 CE=40.0000 精确（因全库**无相邻同 token**，237=ntxt 即全部位不等） | ✓ 复现 |
| MISALIGN5 | 5.1712 vs 10.7379 | CPU 重放逐位相同（5.1712/10.7379，ntxt 237/237） | ✓ 复现 |

**对「能否证明」的直接回答**：(a) **旧式目标下捷径可用** —— CE_old = 0.0000 是「存在一个常量映射（抄输入）在旧目标上零损失」的存在性证明，且该映射落在 tie 参数化可实现族内 ⇒ 捷径**存在**成立。(b) **修复后捷径失效** —— 同一头在新目标上 CE = 40.0000（= 40 logit 间隔下「目标≠输入」位置的满额 CE），且全库 label==input 位置数 = **0/67448** ⇒ 抄输入在新目标上**不再是无损解**，捷径**失效**成立。判据为**模型无关**（不依赖训练状态），方法论独立、可重放 ⇒ **该判据成立，未推翻**。

**AUD-4（MED，口径标签不实）**：单测的 bug 目标是 rtgt_bug[p]=ids[p] 于 **[np-1, n-1)**（我称 recon），而 gen27 所引旧式是 rtgt[p]=ids[p] 于 **[np, n)**。二者差 2 处/行：recon 在 np-1 有标签而旧式无；旧式在 n-1 有标签而 recon 无。故：报告 §0 表格「MISALIGN3 旧式目标与正确目标逐位不同 = 237」实为 **recon-vs-修正 237**；**真旧式-vs-修正 = 242**（= 237 + 5 行 × n-1 位，我已逐位复算证实）。结论方向不变、MISALIGN4 不受影响（两种旧式变体 CE 均为 0.0000），但**数字与「旧式」标签不匹配**，应勘误或重算。

## 3. 28/28 单测重放（对抗点 3）— PASS（整件重放 rc=0），但 3/28 为空真（AUD-6）

**真跑**（非读码）：CUDA_VISIBLE_DEVICES="" python scratch/ccp/gen26/gen26_selftest.py --dev cpu --out audit_r/SELFTEST_REPLAY.json --log audit_r/selftest_replay.log
⇒ GEN26_SELFTEST n=28 pass=28 rc=0，stdout 36 行，证据件 audit_r/SELFTEST_REPLAY.json（CPU 前向/反传，零 GPU）。关键值与 GPU 版**逐位一致**（MISALIGN1/2/3/4/5、T3a chain=9.1775、T5d 238、tied=True；仅梯度末位有别 37.70179 vs 37.70187 = CPU/GPU fp32 实现差异）。**28/28 与 rc=0 的申报属实。**

**AUD-6（MED，3 项空真 PASS）**：GPU 原跑与我的 CPU 重放**都**出现：
- T5a_walk_legal_ops_only：walked=[] ⇒ all(...) over **空集 = 恒真**；
- T5a3_args_mode_switch：args_src={} fallback=0 ⇒ 无帧可验，空真；
- T5c_consistency_column：n=0 rate=None，断言 cons[n]==len([]) ⇒ 空真。
即「T5 接口（走链+渲染+一致性）」在自测里**未被真实执行**（起点头在合法集内直接选 END）。走链/args-mode 的真实证据在评测侧（§4 的 5/5 命中、asrc bridge:1），故接口本身有证据；但**单测这 3 项不构成证据**。处置：加「走链非空」断言或改用已训 ckpt 的单测列。

## 4. 冒烟真实性（对抗点 4）— 读数逐值复现；0.1272 的可信度判断见 AUD-1

HYBRID_CURVE.jsonl 与报告逐值对照（我直接读曲线）：step1 loss 14.64643 / chain 9.43071 / slot 3.98585 / text 5.04964、step100 0.12723 / 0.00426 / 1.40309 / 0.0645、min 0.11675 / 0.00426 / 0.83925 / 0.0645、secs 34.03（wall 48.48 ⇒ 报告 48.5s）· 100 行 · 三曲线齐 · stopped_nan=false · 留出 48 行 sel_top1 1.0(144 帧) / slot_top1 0.7582(nonunk 0.7613, 244 槽) / render_tok_acc 0.7849(1246 tok) ⇒ **报告 §3 表逐值一致，无夸大**。

**批序**：我按 build_plan 的口径（行内空格连接 → 换行 join）从 out/batch_idx_hybrid.txt 重算 md5 = **f3afc2e4dea5d9301626037dbfd95469**（与申报同值）；held 命中 **0**；每批恰 3 类、批内无重复行；步 1–100 各类 seen = **200/200/200**（= 最小类 234 ÷ 2 × 100 步），与 MED-6 披露一致。

**「是否过拟单批/泄漏标签」的独立判断**：**不是单批过拟**——100 步内每类 200 行**互不重复**（批序无回放，见上）。真正原因是**任务本身近零信噪**（本审核最高价值事实，见 AUD-1）：① 金标链**按类恒定**（1424 行只有 3 种链：ind=[recurrence,ring_identity]、ineq=[sos,close]、nt=[witness,witness,combine]）⇒ 链 top-1 等价于 3 类分类；② 渲染目标高度模板化：ind 类 763 行只有 **3 种**不同目标（众数占 59.2%），ineq/nt 逐位置众数基线即可达 0.75/0.75；③ 帧内载荷是金标 res 代理（R2 已登记）。**结论：0.1272 / 0.0043 / 0.0645 作为「机械通路连通」可信，作为任何能力信号不可用；报告的方向警告正确但机制清单不完备。**

**ckpt 命名**：ckpt/ 下**仅** hybrid_smoke100.pt；全批无 hybrid_formal_*；命名硬闸在 gen26_hybrid.py:323-326（SystemExit，双向：冒烟名必须含 smoke / 正式名不得含）✓ 未被当正式产物引用。

## 5. 设计-实现一致（对抗点 5）— PASS（逐项核对）

| 设计书条目 | 代码落地 | 判定 |
|---|---|---|
| L = L_step + (lam_bind/NSLOT)·L_slot + lam_txt·L_text | gen13 loss_f 返 ls + (lam_bind/NSLOT)*lb（v32ccp_train.py:421，NSLOT=24 实读）；forward_joint:134 total = loss + lam_txt*lt | ✓ |
| 权重 1.0/1.0 | argparse 默认 + summary.objective（无加权搜索） | ✓ |
| 联合单阶段（一次前向双头） | forward_joint：seq = cat([链段, 渲染段])，一次 forward_body，链头读 H[:, :Lc]、LM 头读 H[:, Lc:] | ✓ |
| 三损失诊断口径（按有效位置加权） | 曲线列 acc/nv、/nbf、/ntxt | ✓ |
| 考试两段式 + WALK_BUDGET=7 预注册常量 + 合法集=CCP7∪END | gen26_eval.py:59；mask[legal_idx]=0; mask[END31]=0；_legal_ops() 取 VOC ops_ccp 7 名 | ✓ |
| --args-mode **缺省 bridge** | add_argument default="bridge" | ✓ |
| 三对照臂零训练 | --arm{hybrid,naked,rule}；naked=HF 底座直出、rule=gen23_exam.arm_text 装置规则，二者均无训练步 | ✓ |
| 仅 target 计损；链段切片即等价 | rtgt 非 target 位 -100；链头读 H[:, :Lc]（因果性保证链段不被渲染段污染） | ✓ |
| MED-7 238 / MED-8 预算 / MED-9 轮换 | EV.NVALS=238 且 allow reshape (168,24,238)；WALK_BUDGET=7；--eval-rows 48 --eval-rotate 1（eval_idx 入曲线） | ✓ |

## 6. 接口 join 复算（对抗点 6）— 全链 100% 复现；「重切免疫」证据过度（AUD-2）

**我独立重实现四条件过滤器**（不 import 被审件），实读语料 md5 15d58dffecb4b272afe9351e34c9e935、roster md5 35e0632496a0720e1347fcf7998ae92f：
```
4842 → C1/C2/C3/C4 后 2119（drop_not_eligible 931 · exam_excluded 1083 · exam_disclosure 1184 · l2_neg 220）
     → 训练流 1188（ind 665 / ineq 289 / nt 234；链长 2:954 / 3:234）· 体温计 236
     ops_hist = {recurrence 665, ring_identity 665, ineq_sos 289, ineq_close 289, nt_dvd_witness 468, nt_dvd_combine 234}
```
与 A_HYBUILD.json（filter_stats / by_cls / ops_hist / n_ops_per_chain）**逐值相同**。**独立精确 join**（program==chain_ops ∧ stmt==stmt_state，gen12 三行文件 md5 8457f2a5 / 41279826 / 6296cca4 均与申报一致）：**train 1188/1188 · held 236/236 · miss 0 · ambiguous 0** ✓。副作用核对：filter_stats 的 drop_split_l2_neg 键成立（语料确有 split=l2_neg 220 行），非死键。

**AUD-2（MED，证据过度）**：gen26_hybuild.log 两次构建（18:50:39、19:07:05）**读到的 corpus md5 同为 15d58dff** ⇒ gen26 的 join 只在**一个语料版本**上实测过。报告 §2「前两会话实测两次（语料 18:50 与 19:07 **两版**）」与 §7「三次均以四条件过滤器消费且 join 100%」**不成立**：7bbfdfc9 出现于 LAUNCH_LOG:119 的 **gen25 探针**，72a8ad07 属 18:41 统计（gen26 18:45 才 LAUNCH）。契约=过滤器这一**架构**判断成立（我本人即在当前唯一快照上重放成功且结果与批一致），但**经验证据只有一次**。

**内容级泄漏面（我加的独立门，报告未做）**：训练流 vs 考卷侧车 —— **stmt 重叠 0 · head 重叠 0 · prompt-token 重叠 0 · name 重叠 0**（1188 vs 1086）⇒ 当前 C4 排除在**内容层**有效，是本批的**实质加分项**。

## 7. R1 复算与扩展批提案评估（对抗点 7）— ∅ 确认（未推翻）；提案比「四处同步」更大

**独立复算**：训练流快照（1188 行）算子集 = 6 个 ccp_*；EXAM_V1.json 主判据集 1083 项的 chain_ops 聚合 = 11 个 bh_*（多集与 A_HYBUILD.op_space_coverage 逐值一致）；**交集 = ∅**；全部金标算子都在训练空间内的考卷项 = **0/1083** ✓ 报告 §5 成立。旁证：gen28 A_CORPUS_V2.jsonl（9409 行 train、12 算子）**零 bh_***⇒「R1 未因 V2 解除」成立；v32_op_index_map 确有定义（svokernel_v32）故报告引名不虚。

**提案技术评估（只评估不裁决）**：报告称解除条件 =「step_head/allow/lat_tab/训练行**四处同步**」。我实测该清单**不充分**，至少还有两处 + 一道闸：
1. **索引空间已满**：CCP_VOCAB.head=168 = V32 161 + CCP 7（161..167，其中 167 未入训练流），**无空行** ⇒ 加 11 个 bh_* ⇒ head 168→179，step_head 输出维、lat_tab(168,64)、allow(168,24,238) 三处**同时变宽**；ckpt/sla_v31f.pt 的 155 行搬入只对前 168 行可复用，新 11 行须按父行初始化（init_map 需扩）。
2. **值空间/槽模式**：n_vals_ext=238 = 151 base + 87 extra，而 87 个 extra 值的来源实读为「**ccp_{ind,ineq,nt} train 行**的算子→槽→值扫描」（extra_value_source 字段）。bh_* 的槽模式与取值**未注册在任何值空间**；若 bh 算子的 args 不在 238 列内，则 value_head 宽度、allow 第三维、MiniCache/NVALS、桥接件须**一并扩** ⇒「四处」变「六处+」。
3. **装置导参桥缺口（我实测）**：gen26_eval._bridge_args 对 bh_ind_geo_* 实跑 **返回 None 并记 ValueError(步进形式未匹配) / ValueError(hstep 未匹配)** ⇒ ccp_bridge **无 bh 分支**；且考卷侧车 sfa=None（**无金标 args 可取**）⇒ 即便扩了头，考试链导航的载荷仍是「gold 代理」（正是 HIGH-2 要关的捷径）或直接不可走。**解除条件必须显式加第五处：bh 装置导参桥。**

**泄漏闸（提案风险）**：bh 训练行须来自与 1083 主判据集**题面不相交**的池；我实测 gen23 的 exam_excluded 行**自带 proof_text + chain_ops**（即与训练行同质料，另带 in_train_set / exam_role 旗标）⇒「从考卷池取训练行」是真实可达的错误路径。建议扩展批预注册四路零重叠门（name ∪ stmt ∪ head ∪ prompt-token），并保留现有口径（我实测当前为 0/0/0/0）。

**审核意见（不裁决，裁决权在用户/主代）**：甲（certified-only）**立即可做**，不动冻结头/词表，风险最低，但链导航/槽列**仍不可产**；乙（扩算子+桥接）是**唯一能让链导航列出分**的路线，但范围等同于「六处 + 桥 + 泄漏闸 + 全量重训 + Lean 判卷（并发≤5 nice19）」，且新头与既有 ckpt/词表**形状不兼容**（不能热启，需重训）；丙（熟算子分层）绕开 bh 算子，但产出的是「近似算子链」，与 M1_A 三栏口径的可比性须另立登记。

## 8. 自审 CRIT（build_plan 类池空无限循环）（对抗点 8）— 代码位 PASS；缺触发测试（AUD-5）

gen26_hybrid.build_plan 实读：类池空 ⇒ raise SystemExit(...「类池为空 %s (cls 映射未设/错) 池键=%s」)；另有 assert len(plan)==steps。触发路径：_cls_of 未 set_cls_map 时返回 "?" ⇒ 三类全缺 ⇒ 立即 fail-loud（不再无限 while len(plan)<steps）。selftest 侧 set_cls_map 在 T6 之前（gen26_selftest.py:100），训练侧在 build_plan 之前（gen26_hybrid.py:355-356）⇒ 正常路径不受影响 ✓。
**AUD-5（MED，守卫未测）**：单测 T6a/T6b/T6c 只覆盖**非空池**路径，**没有任何 check 断言空池 SystemExit**；台账 BUGFIX-build_plan 所称「单测超时实证」指**旧代码**的超时，不是新守卫的验证 ⇒ 修复在代码层成立，但**失效路径无测试证据**（回归风险）。

## 9. 合规面（对抗点 9）— PASS

- **写面**：gen26 自身产出全部落在 scratch/ccp/gen26/** + scratch/ccp/HEARTBEAT.gen26（0 字节，19:31）+ LAUNCH_LOG.md（LAUNCH 18:45 / DONE 19:35 / NOTE 19:42 / CORRECTION 19:46，append-only）✓。scratch/ccp/gen（gen12 课程行）、gen12、gen13、gen25 中**新于 18:45 的文件数 = 0** ⇒ imports-only 零改动成立 ✓（gen23/gen24/gen28 有新文件，属并行批 gen23r/gen24r/gen28 自身活动，不可归因 gen26）。allow_bin / CCP_VOCAB.json / lat_tab_v31.bin / sla_v31f.pt mtime 均 ≤ 09-23 11:47 ⇒ 未被动过 ✓。
- **判卷件冻结**：drct_common.py md5 = 7c31b58da9d7228cee31b9be581fe5dd **未变**；实测 judge_texts 只在 --judge 1 时调用，7 份 EVAL 的 inputs.judge 全为 false，台账零 judge 行 ⇒ 本批**连 import 都没发生**（比「仅 import」更强）✓；ps 无 Lean/lake 进程，无并发占用 ✓。
- **Test 243**：gen26 全部源码/日志/JSON 中无 Test 243 路径或题面引用（仅「未读未评」声明），未评未算 ✓。
- **GPU**：gpu_gate 实读日志 19:24:28 free=12059MiB（>4500MiB 门通过）；现况 0/1/2 = 223/11/11 MiB、util 0% ⇒ **全空闲、无孤儿进程** ✓；本批仅冒烟+评测短时占用，零训练型长占 ✓。
- **AUD-3（MED，时间线/标签）**：gen26_eval.py mtime **19:27**（LEDGER FIX-HIGH2 记 19:26，自认「ccp_bridge 未在 sys.path 造成静默回退」），而冒烟**在 19:24:28–19:25:17** ⇒ 冒烟的内嵌样例跑的是**修前** eval 件。证据：HYBRID_SMOKE.json 的 5 条样例 args_src_hist **全为 {device_state_gold_proxy: 2}、bridge_fallback=1、sample_table.slot_col=0.0**，而报告 §3 把它们标为「--args-mode bridge」；§4 的 bridge 行（slot_col 0.50）另行来自**修后**的 D1 文件（GEN26_EVAL_hybrid_a_held.json，asrc bridge:1 / fb=0，我复算其日志 19:27:49-58 落盘）。**数值影响 = 零**：我实跑 _bridge_args 在这 5 行**成功**且返回 {form,v0,step} 与金标 args **逐键相等**，三份件（冒烟 / D1-bridge / D2-gold）的**渲染文本逐字节相同**、走链序列相同 ⇒ 仅**标签与 provenance** 不准（§4 的 bridge/gold 对照真实有效）。
- **AUD-8 / AUD-9（LOW）**：① 台账末行自称「终版 md5 = 5cfe3389…（41 行）」，我复算：5cfe3389 = **文件 41 行前缀**的 md5，实际文件 43 行、md5 = a4ce8c1b（与 LAUNCH_LOG:179 CORRECTION 外锚一致）⇒ 链内自指值滞后 2 行（已被批自认并外锚闭合，属可接受）。② 台账 D1–D7 记 ts=19:33:08，而 7 份 EVAL 实际落盘 19:26:49–19:28:40（差 4–6 分钟，登记时刻晚于产物）。

## 10. EVAL 7 份抽查（对抗点 10）— PASS

- **产物命名闸**：7 份文件名与代码自动命名式 GEN26_EVAL_{arm}_{seg}[_gold].json **逐一相符**（抽核 hybrid_a_held / rule_exam_v1 / hybrid_exam_zhenti 三份），arm / seg / args_mode 字段与文件名自洽 ✓。
- **表-行内部一致性复算**（我按 table_of 口径从各行**重算五列**）：3 份抽查表**与所载 table 逐值相等**（first_hit / exact_hit / slot_col / consistency / op_out_of_space / terminated_end / render_tok_mean / n_err）✓。
- **侧车 join**：A_EXAM_SIDECAR.jsonl = 1083 exam_main + 3 exam_zhenti；name 集合与 EXAM_V1.json 的 main_criteria / zhenti_reference **完全相等**（缺 0 / 多 0）；ops 多集 = 报告 11 算子逐值一致；每行 ids[0]=730(M_TASK)、np=n 且 has_target=False（考点无 target 属正确布局，render_greedy 的 ids[1:np] 即整段 prompt）⇒ **无错位**。
- **AUD-10（LOW，空列口径）**：naked/rule 臂 chain=None ⇒ 各行 consistency.rate=None，table_of 以 sum(rate or 0)/n 得 **0.0** ⇒ 表里的 consistency 0.00 是**定义使然**，而非「渲染文本无关键词」的测量；报告对 naked 有解释、对 rule 未说明其为空列。

## 11. 逐条结论汇总

| 编号 | 级别 | 事项 | 证据位置 | 处置（条件） |
|---|---|---|---|---|
| — | **CRIT** | **无**（CRIT-1 修复经全量复算成立，未推翻） | §1 | — |
| AUD-1 | **HIGH** | 冒烟读数的**近零信息量未披露**：金标链按类恒定（1424 行仅 3 种链 ⇒ 链 top-1 = 3 类分类）；渲染目标模板化，**零参数「逐类逐位置众数」基线在同一个 48 行 / 1246 token 留出切片上得 0.7673**，模型 0.7849（**仅高 0.018 ≈ 22 token**）⇒ sel 1.000 / acc 0.785 / text 0.0645 不可作能力或增益读数 | 我复算（§4）· 报告 §3/§4 | 勘误：§3 读法纪律补两条机制 + 基线数字；正式训练后必须报同基线 |
| AUD-2 | MED | 「语料三次重切均 100% 命中」不成立：两次 hybuild（18:50/19:07）**同 md5 15d58dff**；7bbfdfc9 属 gen25 探针、72a8ad07 早于 gen26 LAUNCH | logs/gen26_hybuild.log 两行 · LAUNCH_LOG:119 | 勘误：改为「单一版本实测 100%，契约=过滤器」 |
| AUD-3 | MED | §3 五条样例标「bridge」实为**修前件静默回退**（asrc 全 gold proxy / fb=1 / slot_col 0.0）；数值零影响（bridge≡gold args、文本逐字节同） | HYBRID_SMOKE.json samples · eval 件 mtime 19:27 vs 冒烟 19:25 | 重跑样例块或改标「修前 bridge 回退≡gold」 |
| AUD-4 | MED | MISALIGN3 的「旧式」是重构造变体（[np-1,n-1) 同位复制）：申报 237，**真旧式应为 242** | 我逐位复算（§2） | 勘误数字或补真旧式一列 |
| AUD-5 | MED | build_plan fail-loud 守卫**无测试**（T6 只覆盖非空池） | gen26_selftest.py:254-261 | 加空池 SystemExit 断言 |
| AUD-6 | MED | 28 项中 3 项**空真 PASS**（T5a / T5a3 / T5c：走链为空集） | 原日志 + 我 CPU 重放 | 加非空断言 |
| AUD-7 | LOW | §4 五列表 hybrid 行未注明来自 ckpt/hybrid_smoke100.pt（4 份 EVAL 的 model.path 均为此） | EVAL JSON model.path | 表注补 ckpt |
| AUD-8 | LOW | 台账末行自指 md5（5cfe3389 = 41 行前缀）≠ 文件 md5（a4ce8c1b，43 行）；已由 LAUNCH_LOG:179 外锚闭合 | 我复算前缀 | 可接受（保持外锚） |
| AUD-9 | LOW | 台账 D1–D7 ts=19:33:08 vs 产物 mtime 19:26:49–19:28:40 | ls --time-style | 登记修正 |
| AUD-10 | LOW | naked / rule 的 consistency 0.00 为空列口径 | table_of | 表注说明 |

**未推翻项（经独立复算全部成立）**：CRIT-1 修复位与错位不变量（1424/1424）· tie 机理（config + 梯度同址）· MISALIGN1/2/4/5 数字 · 28/28 rc=0（CPU 整件重放）· 曲线 / 批序 md5 / 覆盖 / 留出三读数 · 四条件 1188+236 与精确 join 100% · 考卷 1086 侧车与 11 算子 · R1 交集 ∅ 与 0/1083 · 三臂五列表表-行自洽 · 命名闸 · 零判卷 / 零 Lean / GPU 空闲 / 写面合规 / drct 冻结。

## 12. 可重放命令（关键复算）

```bash
R=/home/data/ZHC/tcaboost/research; P=/home/data/ZHC/tcaboost/venv/bin/python; export PYTHONPATH=/home/data/ZHC/tcaboost/superoperator; cd $R
# ① 证据件零改动自证（收工应得聚合 md5 146fc65011165c2093af429c190b384d）
cd scratch/ccp/gen26 && find . -type f -not -path "./audit_r/*" -not -path "./__pycache__/*" | sort | while read f; do md5sum "$f"; done | md5sum
# ② 28/28 单测 CPU 整件重放（零 GPU；输出只落 audit_r，不碰原证据件）
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" $P scratch/ccp/gen26/gen26_selftest.py --dev cpu --out scratch/ccp/gen26/audit_r/SELFTEST_REPLAY.json --log scratch/ccp/gen26/audit_r/selftest_replay.log
# ③ 批序 md5 复算：batch_idx_hybrid.txt 去首行 → 行内空格连接 → 换行 join → md5 = f3afc2e4dea5d9301626037dbfd95469
# ④ 错位不变量：对 A_RENDER_SIDECAR.jsonl 全 1424 行逐位核 label[p]==ids[p+1]（应 0 违规）；旧式两变体（[np,n) 与 [np-1,n-1) 同位复制）vs 修正式 = 242 / 237（5 行子集）
# ⑤ 抄输入头 CE（解析式）：CE = 40·(1-[label==input])（旧式两变体均 0.0000；修正式 40.0000）
```

## 13. 审计痕迹

- audit_r/SELFTEST_REPLAY.json（28/28 rc=0 的 CPU 重放件）+ audit_r/selftest_replay.stdout（36 行）
- 本件 md5 见 LAUNCH_LOG 追加行；审核期间**未**新增/修改被审件任何文件（38 件聚合 md5 同值，__pycache__ 无新件）。

## 14. AUDIT_RC

**GEN26_AUDIT_RC=0** —— 总裁决 **有条件通过**：无 CRIT 级推翻（修复真、接口真、R1 真、合规真），但 AUD-1（HIGH，读数近零信息量未披露）与 AUD-2..AUD-6（MED，证据口径 / 守卫 / 空真）需以**勘误登记**闭环后，方可进入正式训练决策（正式训练另受 §7 R1 提案裁决约束）。
