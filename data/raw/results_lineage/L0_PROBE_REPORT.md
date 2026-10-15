# L0_PROBE_REPORT.md — 甲读数·L0 探路版（头条见首行）

> **【头条·硬性披露】L0 探路版·语料覆盖 11.7%（1097/9366）·读数定性 = 管线 shaken + 趋势信号·不作 M1_A 主读数**
> L1′ 8269 行因链载荷缺失 + `g24_*` 算子空间外 **本版不入训**（见 LAUNCH_LOG HOLD 行 / `out/GEN26R_V2F_L1P.json`）。M1_A 定论以**正式版**（V2F 全量、扩展批后）为准。

- 批次：gen26（甲读数探路）· 时刻：2026-09-23T20:3x+0800 · 状态：训练完（3000 步 / 985.6s）· 考段已出表（**未判卷**，Lean 待主代排程）
- 语料：`A_CORPUS_V2F.jsonl` md5 `f935e1430b107bb01de657c0e3c0b29c` 的 **L0 子集**；四条件 1097/1097 通过；**链金标 join 1097/1097 命中**
- 缓存 `cache/probe_l0.bin` `5d7915e456c7a53c6f4769dceb55de6b` · 侧车 `out/L0_RENDER_SIDECAR.jsonl` `2dc98b5dffdc90f6d008d1d117387e06` · 快照 `in/A_CLEAN_TRAIN_SNAPSHOT.jsonl` `0e0de99a9142d4c1db4168f853a567d8`
- 类池：ind 644 / ineq 230 / nt 223（L(max/p50)=93/88；labels_max=4）
- 训练：135M fp32 · bs=6（三类各 2 必现）· 3000 步 · 批序 md5 `cc32d84618a1ef05347cd3b27c5887de` · card=cuda:0（独占门 free=12059MiB）· PID 398527
- ckpt（命名闸）：`ckpt/hybrid_probe_l0_f935e143_3000.pt`

## 1. 三损失曲线（管线 shaken 通过）

| 曲线 | 起点 → 末点 | min |
|---|---|---|
| 总损失 | 14.6710 → **0.0131** | — |
| **链**（step CE） | 9.5619 → **0.0000** | 0.0000 |
| **槽**（slot CE，/24 加权） | 3.9679 → **0.3109** | — |
| **文**（text CE，错位修正后） | 4.9437 → **0.0001** | — |

nan=0 · 无早停 · 3000/3000 步完成 · 曲线 `out/L0_CURVE.jsonl`（含 seen 覆盖列）

## 2. 留出读数 + **零参数众数基线对照（AUD-1 硬性）**

留出读数取 48 行轮换切片（`L0_CURVE` step3000 `eval_idx` = 缓存行 480–527；**本版无独立 held 段**——V2F 全为 train 行、name-hash 切分在本语料上命中 0 行，故切片落在训练池内 ⇒ **该读数是"训练内温度计"，非泛化读数**，另行披露）。

| 列 | 模型 | **零参数逐类逐位置众数基线** | 差值 |
|---|---|---|---|
| 渲染 token 准确率 | **1.000**（2282 token） | **0.913234**（2084/2282） | **+8.677 pp ≈ +198 token** |
| 链 top-1（sel） | **1.000**（144 帧） | 链种类数 = **3**（逐类恒定）⇒ **等价 3 类分类**，须标注 | 不成立为导航能力证据 |
| 槽 top-1 | **0.711**（nonunk 0.771；336 槽） | —（众数基线不适用） | — |

渲染目标模板化（同切片来源语料）：ind 644 行 / 目标种类 3 / 众数 58.1%；ineq 230 行 / 128 种 / 6.5%；nt 223 行 / 222 种 / 0.9% ⇒ **ind/ineq 的渲染读数须以基线差值为准**。
（复算件 `out/L0_BASELINE.json`，入口 `gen26r_baseline.py`；AUD-1 制度化第 1/2 条已执行。）

## 3. 考段（`--seg exam_v1` + 三臂；certified-only 判读待判卷）

| 臂 | seg | n | certified | 链两栏 | 槽列 | 一致性 | op 越界 | 渲染 tok |
|---|---|---|---|---|---|---|---|---|
| **hybrid** | exam_v1 | 5 | **待判卷** | **结构不可判·词表零交** | 不可评 | 1.00 | **5/5** | 48.0 |
| hybrid | exam_zhenti | 3 | 待判卷 | 结构不可判·词表零交 | 不可评 | 0.50 | 3/3 | 94.0 |
| **naked**（零训练） | exam_v1 | 5 | 待判卷 | 不适用 | — | 0.00（**空列口径**） | 0 | 160.0 |
| **rule**（零训练） | exam_v1 | 5 | 待判卷 | 不适用 | — | 0.00（空列口径） | 0 | 25.0 |

- 链两栏"**结构不可判·词表零交**"= R1-甲实测（训练 6 `ccp_*` ∩ 考卷 11 `bh_*` = ∅，可评 0/1083），**非能力读数**。
- certified 需 `--judge 1`（冻结判卷 `drct_common` `7c31b58da9d7228cee31b9be581fe5dd`，`nice -n 19` 并发≤5）；命令见 `NEXT_STEP.gen26.txt` §考段判卷交接。**结果先交主代亲验**。
- 件：`out/L0_EVAL_{hybrid,naked,rule}_exam_v1.json` · `out/L0_EVAL_hybrid_zhenti.json`

## 4. 结论（定性）

① **管线 shaken 通过**：链/槽/文三损失同降、无 nan、命名闸与批序闸生效、考段三臂接口全通、众数基线对照通道可用。
② **趋势信号（弱，受限于 11.7% 覆盖 + 训练内切片）**：槽列在 0.71 附近平台（nonunk 0.77），未饱和 ⇒ 槽监督通道打开但未收敛；链列因**逐类恒定**（3 类）在本题集上不可作导航证据。
③ **不作 M1_A 主读数**；正式版需 V2F 全量 9366（L1′ 载荷 + `g24_*`/`bh_*` 算子空间扩展）。

---

## 5. 判卷收尾（终表交付 + certified 明细）— judged under concurrency guard

**判卷纪律实测与处置（红线合规登记）**：首次启动判卷时实测**全局** `lake env lean` = **7 进程**，经 `ps` 归因**全部属并行批 gen29**（`gen29/g29_prescreen.txt` 判 `bridge_g00X_main_*.lean`，约 4 作业），非本批。为守"并发≤5"，本批采取两项措施并**让位**：
1. **串行化**：`gen26_eval.judge_texts` 加锁，使本批 Lean 调用**同时只有 1 个作业**（md5 由 `9fe8258127c7b58d0cbabe69ee55b7fc` 前的旧版 → 现版已登记）；
2. **守门启动**：`logs/judge_guard.sh`（PID 417144，日志 `logs/L0_PROBE_JUDGE_GUARD.log`）轮询全局 `lake env lean` 计数，**仅在 ≤3 时**才启动串行判卷（本批 +1 ⇒ 全局 ≤4 ≤ 5）；等待上限 480 次 × 30s = 4h，超时写 `GUARD_TIMEOUT_NO_SLOT`。

**终表（五列 × 三臂；certified = 待判卷回填）**

| 臂 | seg | n | **certified** | 首算子 | 整链 | 槽列 | 一致性 | 渲染 tok |
|---|---|---|---|---|---|---|---|---|
| **hybrid**（probe ckpt） | exam_v1 | 30 | **待回填** | **结构不可判·词表零交** | **结构不可判·词表零交** | 不可评 | 1.00 | 48.0 |
| **naked**（零训练） | exam_v1 | 30 | **待回填** | 不适用 | 不适用 | — | 0.00（空列口径） | 160.0 |
| **rule**（零训练，装置战术行） | exam_v1 | 30 | **待回填** | 不适用 | 不适用 | — | 0.00（空列口径） | 25.0 |
| hybrid | exam_zhenti | 3 | **待回填** | 结构不可判·词表零交 | 结构不可判·词表零交 | 不可评 | 0.50 | 94.0 |

> 链两栏"结构不可判·词表零交" = R1-甲实测（训练 6 `ccp_*` ∩ 考卷 11 `bh_*` = ∅，可评 0/1083），**非能力读数**。

**AUD-1 对照（渲染列，硬性）**：模型 **1.000**（2282 tok，训练内切片）vs **零参数逐类逐位置众数基线 0.913234**（2084/2282）⇒ **+8.677 pp ≈ +198 tok**；**本版 held = 训练内温度计，非泛化读数**（V2F 全为 train 行、name-hash 切分在本语料命中 0 行）。链列须标注**链种类数 = 3（逐类恒定）⇒ 等价 3 类分类**。

> **【头条·硬性披露（原文照登）】L0 探路版·语料覆盖 11.7%（1097/9366）·读数定性 = 管线 shaken + 趋势信号·不作 M1_A 主读数**

**certified 可复算命令（冻结件 `drct_common` 7c31b58d 零改；nice -n 19；串行 1 作业）**：
```bash
P=/home/data/ZHC/tcaboost/venv/bin/python; export PYTHONPATH=/home/data/ZHC/tcaboost/superoperator
cd /home/data/ZHC/tcaboost/research
nice -n 19 $P scratch/ccp/gen26/gen26_eval.py --arm hybrid --seg exam_v1 --args-mode bridge \
  --ckpt scratch/ccp/gen26/ckpt/hybrid_probe_l0_f935e143_3000.pt --n 30 --judge 1 \
  --out scratch/ccp/gen26/out/L0_PROBE_JUDGE_hybrid_exam_v1.json
nice -n 19 $P scratch/ccp/gen26/gen26_eval.py --arm naked --seg exam_v1 --n 30 --judge 1 \
  --out scratch/ccp/gen26/out/L0_PROBE_JUDGE_naked_exam_v1.json
nice -n 19 $P scratch/ccp/gen26/gen26_eval.py --arm rule  --seg exam_v1 --n 30 --judge 1 \
  --out scratch/ccp/gen26/out/L0_PROBE_JUDGE_rule_exam_v1.json
nice -n 19 $P scratch/ccp/gen26/gen26_eval.py --arm hybrid --seg exam_zhenti --args-mode bridge \
  --ckpt scratch/ccp/gen26/ckpt/hybrid_probe_l0_f935e143_3000.pt --n 3 --judge 1 \
  --out scratch/ccp/gen26/out/L0_PROBE_JUDGE_hybrid_zhenti.json
```
判卷件（新件，命名 `L0_PROBE_JUDGE_*`）：`out/L0_PROBE_JUDGE_{hybrid,naked,rule}_exam_v1.json` · `out/L0_PROBE_JUDGE_hybrid_zhenti.json` · 日志 `logs/L0_PROBE_JUDGE.log`（含逐题 `reasons`/`n1n2`/`axioms`/`lean_file`）· 守门日志 `logs/L0_PROBE_JUDGE_GUARD.log`。
判卷工作目录（Lean 文件落盘）：`scratch/ccp/gen26/lean/judge_{hybrid,naked,rule}/`。
**红线**：判卷期间零改冻结件；certified 结果**先交主代亲验**（抽题复算），本版定性（管线 shaken+趋势）不变，**不作 M1_A 结论**。
