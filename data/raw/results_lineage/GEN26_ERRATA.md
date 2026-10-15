# GEN26_ERRATA.md — gen26 勘误闭环（依据 GEN26_AUDIT.md 815b8954）

- **批次**: gen26r（对抗审后勘误闭环批）
- **审源**: `GEN26_AUDIT.md`（`815b8954…`，对抗审结论 =**有条件通过**；CRIT-1 修复/接口/R1/合规四项复核成立）
- **纪律**: append-only；**既有件零改动**（本批只新增本勘误件 + 1 个复算脚本 + 1 份复算 JSON + 单测守卫项；被勘误的原报告/台账**原文保留**，以本件为准）
- **复算件**: `gen26r_baseline.py`（`d611cb6a4352027ec2308705e558cbef`）→ `out/GEN26R_BASELINE.json`（`388259e9c1c800be60eded499cf6a48b`）
- **时刻**: 2026-09-23T19:5x+0800

---

## AUD-1（HIGH）读数纪律：三读数降级为「机械通路连通」证据

**勘误内容**：gen26 冒烟的三项读数——留出链 `sel_top1 = 1.000`、渲染 `tok_acc = 0.785`、渲染损失 `text = 0.0645`——**一律定性为"机械通路连通"证据，不作能力读数**；正式训练与考试的读出**必须同报众数基线**（见下"制度化"）。

**两条机制（本批独立复算，`GEN26R_BASELINE.json`）**：

| 机制 | 复算读数 | 含义 |
|---|---|---|
| ① 链在金标上**按类恒定** | 全库 1424 行的链种类数 = **3**（同样按类去重仍 = 3）：ind = `ccp_ind_recurrence+ccp_ring_identity`、ineq = `ccp_ineq_sos+ccp_ineq_close`、nt = `ccp_nt_dvd_witness+ccp_nt_dvd_witness+ccp_nt_dvd_combine` | 逐帧算子 top-1 在本题集上**等价于 3 类分类**；`sel=1.000` 不构成导航能力证据 |
| ② 渲染目标**模板化** | 逐类去重（全库，非仅 ind）：**ind 763 行 / 目标种类 3 / 众数占比 59.24%**（众数目标长 48 token）；ineq 362 行/148 种/8.29%；nt 299 行/253 种/4.68% | 渲染 token 准确率受"目标可枚举"支配，必须对照众数基线读 |

**基线数字（零参数逐类逐位置众数基线）**：

| 项 | 读数 |
|---|---|
| 基线定义 | 用**训练段**同类的 target 逐位置众数作为预测（零参数）；在**模型读数所用同一留出切片**上评（`HYBRID_CURVE` step100 `eval_idx` = 全局行 1284..1331，48 行，1246 token） |
| 众数基线 token 准确率 | **0.767255**（956/1246） |
| 模型 | **0.784912**（= 原报 0.785） |
| **差值** | **+1.766 pp ≈ 22 token** |

⇒ 原报"渲染 token acc 0.785"相对零参数众数基线仅 **+22 token**。**系统主张不得据此差值立论**（发现 5 常量基线铁律同源）。

**制度化（写入训练报告模板）**：
1. 任何渲染读数（token acc / 文本损失）**必须**同报"零参数逐类逐位置众数基线"＋差值（pp 与 token 数），并在同一评测切片上计算；
2. 任何链导航读数**必须**同报"链种类数 / 逐类链是否恒定"；若逐类恒定 ⇒ 该读数**必须**标注"等价 k 类分类（k = 类数）"，不得作导航能力证据；
3. 正式训练报告（gen26 甲读数）§冒烟/§评测两节各设"基线对照"行，缺失即视为不合格交付。

---

## AUD-2 接口陈述更正

**原文（GEN26_REPORT §2 / LAUNCH_LOG DONE 行）**："语料三次重切（`7bbfdfc9`→`72a8ad07`→`15d58dff`）均 100% 命中"。

**更正为**：**只有单一版本 `15d58dff` 经本批实测 100% 命中**（1188/1188 + 236/236）；`7bbfdfc9`（17:42, gen25 探针值）与 `72a8ad07`（18:41 统计值）**本批未消费、未复算**，不得计入实测证据。正确表述 = **"重切免疫"是过滤器架构契约（消费入口 = 四条件 ∧ 精确 join），非对多版本均已实测**。

---

## AUD-3 样例口径更正（§3 五条样例）

**原文**：§3 五条样例标为 `--args-mode bridge`。

**更正为**：该五条样例产于 **CRIT-1 修正前的运行**，其时 `ccp_bridge` 未在 `sys.path` ⇒ 装置导参**静默回退**，实际载荷 ≡ `gold`（审核批复核：**bridge 与 gold 两版渲染文本逐字节相同**）。故 §3 样例应读作 **"修前静默回退（≡gold）口径"**；`slot_col=0.50` 之真 bridge 读数出自修后重跑（`out/GEN26_EVAL_hybrid_a_held.json` 的 `slot_col`）。
**制度含义已修**：`args_src` 逐帧登记 + `bridge_fallback` 计数 + `bridge_errors` 披露（修后 `bridge_errors=[]`）三者齐备，静默回退**不再可能**。

---

## AUD-4 MISALIGN3 数字更正

| 项 | 原报 | **更正** |
|---|---|---|
| 差异位置数（旧式 vs 修正） | 237 | **242**（真旧式口径；逐行 48/49/49/48/48，5 行合计） |
| 原 237 的口径 | — | = **recon 变体**（把 `np-1` 也填成 `ids[np-1]` 的"全行同位"重建）与修正版的差异数；**不是**原病灶行的差异数 |
| 真旧式定义 | — | 位置 `[np, len)` 标签 = `ids[np:len]`（同位自预测）**且** 位置 `np-1` 无标签（−100） |

复算入口：`gen26r_baseline.py` §B4（`B4_true_oldstyle_vs_fix.total_diff_pos = 242`）。
**结论不变**：CRIT-1 病灶与修法均成立（审核批已复核）；本项只更正**数字口径**。

---

## AUD-5 守卫测试（新增 T6d）

**新增**：`gen26_selftest.py` T6d `empty_pool_fail_loud` —— 空类池调用 `build_plan` 必须 **SystemExit fail-loud**（防 18:5x 实测到的"空池 ⇒ 无限循环"重演）。

**CPU 重跑（全自测）**：`CUDA_VISIBLE_DEVICES="" … gen26_selftest.py --dev cpu`

| 项 | 修前 | **修后** |
|---|---|---|
| 测试项数 | 28 | **32**（+T5e · T5e2 · T5e3 · T6d） |
| 通过数 | 28 | **32** |
| rc | 0 | **0** |
| 产物 | `out/GEN26_SELFTEST.json`（`762f5d35…`，GPU 版） | `out/GEN26_SELFTEST.json`（**`50fdbd2cfce2b3d6c8ec0011c5e030f0`**，CPU 版；旧版另存 `out/GEN26_SELFTEST.pregen26r.bak.json` md5 `762f5d35938aa1e6a35b6693dc72b9b1`） |
| 源件 | `63d97fa2f9a2f6dec3233a320a5d93c3` | **`d4b30103e5de2feba89f6a090158a122`** |
| T6d 明细 | — | `SystemExit=True msg=build_plan: 类池为空 ['ind','ineq','nt'] (cls 映射未设/错) 池键=[]` |

---

## AUD-6 known-empty PASS 标注 + 非空夹具

| 测试 | 修前性质 | 标注 | 非空补齐 |
|---|---|---|---|
| T5a `walk_legal_ops_only` | **known-empty**（起点模型即时 END ⇒ `walked=[]`，"合法性"平凡成立） | 明细已标 `[known-empty: 起点模型即时 END; 非空夹具见 T5e]` | **T5e**：夹具 = `step_head.bias[END] -= 1e4`（抑制 END 行）⇒ 走链 **7 帧非空**（`['ccp_nt_dvd_combine'×3, 'ccp_ineq_sos', 'ccp_nt_dvd_combine'×3]`），逐帧 `pred_idx ∈ 合法集`；夹具后**立即还原** bias |
| T5a3 `args_mode_switch` | **known-empty**（空链 ⇒ `args_src={}`） | 明细已标 `[known-empty: 空链 => 无帧; 非空见 T5e2]` | **T5e2**：`args_src = {"bridge": 1, "device_state_gold_proxy": 6}` ⇒ 装置导参**在自测内被真实调用**（首帧 bridge，其余代理） |
| T5c `consistency_column` | **known-empty**（`n=0` 平凡成立） | 明细已标 `[known-empty: n=0 平凡成立; 非空见 T5e3]` | **T5e3**：`n=7 rate=0.0`（非空且非平凡：夹具下渲染文本与算子关键词不匹配 ⇒ 一致性列能给出 0，证明该列非恒真） |

---

## AUD-10 表注更正

§4 五列表中 **naked 臂 `consistency = 0.00` 与 rule 臂 `consistency = 0.00` 是"空列口径"**（定义使然：两臂无链 ⇒ 无算子可比 ⇒ 一致性列无定义，代码以 0 记），**不得读作"渲染与链不符"**。正确读法：该两臂的链相关列（`first_hit`/`exact_hit`/`slot_col`/`consistency`）均为 **"不适用（无链）"**；naked 臂有效列 = `render_tok_mean = 160`（重复符号串）与 `certified`（待判卷）。

---

## 待命状态（主代放行令）

- **不开训**：在收到主代下一条含 **V2F md5** 的消息并经其验收前，**不启动 135M 正式训练（甲读数）**。此前本批处于**待命**，零 GPU 占用。
- **已预登记的正式开训配置**（待放行即按此执行，任何偏离须另开变更行）：
  - 语料：`A_CORPUS_V2F` 定稿 md5（主代给值）→ **按四条件过滤器**消费其 `eligible_train`；重跑 `gen26_hybuild.py` 生成新缓存/侧车并登记全部 md5；
  - 产物命名：`ckpt/hybrid_formal_<语料md5前8>_<步数>.pt`（禁含 `smoke`，代码硬闸在位）；
  - GPU：单卡独占（`nvidia-smi` free > 4500MiB 门）；交错混喂三类必现；中间落盘 + 心跳照旧；
  - 报告：**同报零参数逐类逐位置众数基线 + 链种类数/逐类恒定标注**（AUD-1 制度化第 1/2 条），并单列 `--args-mode bridge` 正式读数（gold 仅诊断列）。
- **R1 未解除**：`bh_*` 11 算子仍不在任何已注册算子空间（V2 的 12 算子 = 6 `ccp_*` + 6 `g24_*`，与考卷交集 0）⇒ 正式训练的**链导航/槽列**在 EXAM_V1 主判据集上仍不可评，需算子空间扩展批；**certified 列不受阻**。

---

## 本批新增/变更件 md5

| 件 | md5 | 备注 |
|---|---|---|
| `GEN26_ERRATA.md`（本件） | 见次日志外锚行 | 新件 |
| `gen26r_baseline.py` | `d611cb6a4352027ec2308705e558cbef` | 新件（AUD-1/AUD-4 复算入口） |
| `out/GEN26R_BASELINE.json` | `388259e9c1c800be60eded499cf6a48b` | 新件（复算读数） |
| `gen26_selftest.py` | `d4b30103e5de2feba89f6a090158a122` | **追加**（+T5e/T5e2/T5e3/T6d + known-empty 标注）；原 `63d97fa2…` 为历史版本，未再使用 |
| `out/GEN26_SELFTEST.json` | `50fdbd2cfce2b3d6c8ec0011c5e030f0` | **CPU 版 32/32 rc=0**；历史 GPU 版备份 `out/GEN26_SELFTEST.pregen26r.bak.json` = `762f5d35938aa1e6a35b6693dc72b9b1` |
| 其余 gen26 件 | **未改动**（md5 见 GEN26_LEDGER.jsonl 与 LAUNCH_LOG DONE 行） | 既有件零改动 |
